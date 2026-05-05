/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;
/**
 * ZGEMV performs one of the matrix-vector operations
 * y := alpha*A*x + beta*y  or  y := alpha*A^T*x + beta*y  or
 * y := alpha*A^H*x + beta*y
 */
interface Zgemv {

    static int BLOCK_SIZE = 64;
    static int THRESHOLD = 32;
    static int PAIRED_ROW_MIN_N = 32;
    static int PAIRED_ROW_MAX_N = 64;
    static int TRANS_PAIRED_ROW_MIN_N = 16;

    /**
     * Performs one of the matrix-vector operations
     * y := alpha*A*x + beta*y  or  y := alpha*A^T*x + beta*y  or
     * y := alpha*A^H*x + beta*y
     *
     * @param trans  Specifies the form of op(A) to be used in the matrix-vector product:
     *               BLAS.Trans.NoTrans: op(A) = A
     *               BLAS.Trans.Trans: op(A) = A^T
     *               BLAS.Trans.Conj: op(A) = A^H (conjugate transpose)
     * @param m      Number of rows of the matrix A
     * @param n      Number of columns of the matrix A
     * @param alphaRe Real part of the scalar alpha
     * @param alphaIm Imaginary part of the scalar alpha
     * @param A      Complex matrix A stored in row-major order, interleaved format [re, im, re, im, ...]
     * @param aStart Starting index of A
     * @param lda    Leading dimension of the matrix A (must be at least max(1, m) for NoTrans, max(1, n) otherwise)
     * @param x      Complex vector x stored in interleaved format [re, im, re, im, ...]
     * @param xStart Starting index of x
     * @param incX   Increment for the elements of x
     * @param betaRe  Real part of the scalar beta
     * @param betaIm  Imaginary part of the scalar beta
     * @param y      Complex vector y stored in interleaved format [re, im, re, im, ...]
     * @param yStart Starting index of y
     * @param incY   Increment for the elements of y
     */
    public static void zgemv(BLAS.Trans trans, int m, int n, double alphaRe, double alphaIm,
                            double[] A, int aStart, int lda,
                            double[] x, int xStart, int incX,
                            double betaRe, double betaIm,
                            double[] y, int yStart, int incY) {
        if (m <= 0 || n <= 0) return;
        boolean noTrans = (trans == BLAS.Trans.NoTrans);
        boolean conjTrans = (trans == BLAS.Trans.Conj);
        int lenY = noTrans ? m : n;
        if (betaRe == 0.0 && betaIm == 0.0) {
            for (int i = 0; i < lenY; i++) {
                int yp = yStart + i * incY * 2;
                y[yp] = 0.0; y[yp + 1] = 0.0;
            }
        } else if (betaRe != 1.0 || betaIm != 0.0) {
            for (int i = 0; i < lenY; i++) {
                int yp = yStart + i * incY * 2;
                double yr = y[yp], yi = y[yp + 1];
                y[yp] = betaRe * yr - betaIm * yi;
                y[yp + 1] = betaRe * yi + betaIm * yr;
            }
        }
        if (alphaRe == 0.0 && alphaIm == 0.0) return;
        if (noTrans) {
            gemvNoTrans(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
        } else if (conjTrans) {
            gemvConjTransOnly(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
        } else {
            gemvTrans(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
        }
    }

    static void gemvNoTrans(int m, int n, double alphaRe, double alphaIm,
                            double[] A, int aStart, int lda,
                            double[] x, int xStart, int incX,
                            double[] y, int yStart, int incY) {
        if (n < BLOCK_SIZE && m < BLOCK_SIZE) {
            gemvNoTransDirect(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
        } else {
            gemvNoTransBlocked(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
        }
    }

    static void gemvNoTransDirect(int m, int n, double alphaRe, double alphaIm,
                                  double[] A, int aStart, int lda,
                                  double[] x, int xStart, int incX,
                                  double[] y, int yStart, int incY) {
        if (incX == 1 && m > 1 && n >= PAIRED_ROW_MIN_N && n <= PAIRED_ROW_MAX_N && SIMD.supportZgemv()) {
            if (ZgemvSIMD.gemvNoTransPairedRows(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, y, yStart, incY)) {
                return;
            }
        }

        if (incX == 1 && m > 1 && n >= PAIRED_ROW_MIN_N && n <= PAIRED_ROW_MAX_N) {
            gemvNoTransPairedRows(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, y, yStart, incY);
            return;
        }

        gemvNoTransDirectScalar(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
    }

    static void gemvNoTransDirectScalar(int m, int n, double alphaRe, double alphaIm,
                                        double[] A, int aStart, int lda,
                                        double[] x, int xStart, int incX,
                                        double[] y, int yStart, int incY) {
        int n2 = (n / 2) * 2;
        for (int i = 0; i < m; i++) {
            int rowOff = aStart + i * lda * 2;
            int yp = yStart + i * incY * 2;
            double s0r = 0.0, s0i = 0.0, s1r = 0.0, s1i = 0.0;
            int j = 0;
            for (; j < n2; j += 2) {
                int xp0 = xStart + j * incX * 2;
                int xp1 = xStart + (j + 1) * incX * 2;
                int ap0 = rowOff + j * 2;
                double a0r = A[ap0], a0i = A[ap0 + 1];
                double a1r = A[ap0 + 2], a1i = A[ap0 + 3];
                double x0r = x[xp0], x0i = x[xp0 + 1];
                double x1r = x[xp1], x1i = x[xp1 + 1];
                s0r = Math.fma(a0r, x0r, Math.fma(-a0i, x0i, s0r));
                s0i = Math.fma(a0r, x0i, Math.fma(a0i, x0r, s0i));
                s1r = Math.fma(a1r, x1r, Math.fma(-a1i, x1i, s1r));
                s1i = Math.fma(a1r, x1i, Math.fma(a1i, x1r, s1i));
            }
            double sumr = s0r + s1r;
            double sumi = s0i + s1i;
            for (; j < n; j++) {
                int ap = rowOff + j * 2;
                int xp = xStart + j * incX * 2;
                double ar = A[ap], ai = A[ap + 1];
                double xr = x[xp], xi = x[xp + 1];
                sumr = Math.fma(ar, xr, Math.fma(-ai, xi, sumr));
                sumi = Math.fma(ar, xi, Math.fma(ai, xr, sumi));
            }
            y[yp] += alphaRe * sumr - alphaIm * sumi;
            y[yp + 1] += alphaRe * sumi + alphaIm * sumr;
        }
    }

    static void gemvNoTransPairedRows(int m, int n, double alphaRe, double alphaIm,
                                      double[] A, int aStart, int lda,
                                      double[] x, int xStart,
                                      double[] y, int yStart, int incY) {
        int n2 = (n / 2) * 2;
        int i = 0;

        for (; i + 1 < m; i += 2) {
            int row0 = aStart + i * lda * 2;
            int row1 = row0 + lda * 2;
            double s0r = 0.0;
            double s0i = 0.0;
            double s1r = 0.0;
            double s1i = 0.0;

            int j = 0;
            for (; j < n2; j += 2) {
                int x0 = xStart + j * 2;
                double x0r = x[x0];
                double x0i = x[x0 + 1];
                double x1r = x[x0 + 2];
                double x1i = x[x0 + 3];

                int a0 = row0 + j * 2;
                double a00r = A[a0];
                double a00i = A[a0 + 1];
                double a01r = A[a0 + 2];
                double a01i = A[a0 + 3];

                int a1 = row1 + j * 2;
                double a10r = A[a1];
                double a10i = A[a1 + 1];
                double a11r = A[a1 + 2];
                double a11i = A[a1 + 3];

                s0r = Math.fma(a00r, x0r, Math.fma(-a00i, x0i, s0r));
                s0i = Math.fma(a00r, x0i, Math.fma(a00i, x0r, s0i));
                s0r = Math.fma(a01r, x1r, Math.fma(-a01i, x1i, s0r));
                s0i = Math.fma(a01r, x1i, Math.fma(a01i, x1r, s0i));

                s1r = Math.fma(a10r, x0r, Math.fma(-a10i, x0i, s1r));
                s1i = Math.fma(a10r, x0i, Math.fma(a10i, x0r, s1i));
                s1r = Math.fma(a11r, x1r, Math.fma(-a11i, x1i, s1r));
                s1i = Math.fma(a11r, x1i, Math.fma(a11i, x1r, s1i));
            }

            for (; j < n; j++) {
                int x0 = xStart + j * 2;
                double x0r = x[x0];
                double x0i = x[x0 + 1];

                int a0 = row0 + j * 2;
                double a00r = A[a0];
                double a00i = A[a0 + 1];
                int a1 = row1 + j * 2;
                double a10r = A[a1];
                double a10i = A[a1 + 1];

                s0r = Math.fma(a00r, x0r, Math.fma(-a00i, x0i, s0r));
                s0i = Math.fma(a00r, x0i, Math.fma(a00i, x0r, s0i));
                s1r = Math.fma(a10r, x0r, Math.fma(-a10i, x0i, s1r));
                s1i = Math.fma(a10r, x0i, Math.fma(a10i, x0r, s1i));
            }

            int y0 = yStart + i * incY * 2;
            int y1 = y0 + incY * 2;
            y[y0] = Math.fma(alphaRe, s0r, Math.fma(-alphaIm, s0i, y[y0]));
            y[y0 + 1] = Math.fma(alphaRe, s0i, Math.fma(alphaIm, s0r, y[y0 + 1]));
            y[y1] = Math.fma(alphaRe, s1r, Math.fma(-alphaIm, s1i, y[y1]));
            y[y1 + 1] = Math.fma(alphaRe, s1i, Math.fma(alphaIm, s1r, y[y1 + 1]));
        }

        if (i < m) {
            gemvNoTransDirectScalar(1, n, alphaRe, alphaIm, A, aStart + i * lda * 2, lda, x, xStart, 1, y, yStart + i * incY * 2, incY);
        }
    }

    static void gemvNoTransBlocked(int m, int n, double alphaRe, double alphaIm,
                                   double[] A, int aStart, int lda,
                                   double[] x, int xStart, int incX,
                                   double[] y, int yStart, int incY) {
        for (int ii = 0; ii < m; ii += BLOCK_SIZE) {
            int iEnd = Math.min(ii + BLOCK_SIZE, m);
            gemvNoTransDirect(iEnd - ii, n, alphaRe, alphaIm,
                    A, aStart + ii * lda * 2, lda, x, xStart, incX,
                    y, yStart + ii * incY * 2, incY);
        }
    }

    static void gemvConjTransOnly(int m, int n, double alphaRe, double alphaIm,
                              double[] A, int aStart, int lda,
                              double[] x, int xStart, int incX,
                              double[] y, int yStart, int incY) {
        if (m <= 0) return;
        if (m <= THRESHOLD) {
            if (incY == 1 && m > 1 && n >= TRANS_PAIRED_ROW_MIN_N) {
                gemvConjTransPairedRows(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart);
                return;
            }
            gemvConjTransScalarLeaf(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
            return;
        }
        int mid = m / 2;
        gemvConjTransOnly(mid, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
        gemvConjTransOnly(m - mid, n, alphaRe, alphaIm, A, aStart + mid * lda * 2, lda,
                          x, xStart + mid * incX * 2, incX, y, yStart, incY);
    }

    static void gemvConjTransScalarLeaf(int m, int n, double alphaRe, double alphaIm,
                                        double[] A, int aStart, int lda,
                                        double[] x, int xStart, int incX,
                                        double[] y, int yStart, int incY) {
        int n4 = (n / 4) * 4;
        for (int i = 0; i < m; i++) {
            int xPos = xStart + i * incX * 2;
            double axr = alphaRe * x[xPos] - alphaIm * x[xPos + 1];
            double axi = alphaRe * x[xPos + 1] + alphaIm * x[xPos];
            int rowOff = aStart + i * lda * 2;
            int j = 0;
            for (; j < n4; j += 4) {
                int ap = rowOff + j * 2;
                updateConjLeaf(y, yStart + j * incY * 2, A[ap], A[ap + 1], axr, axi);
                updateConjLeaf(y, yStart + (j + 1) * incY * 2, A[ap + 2], A[ap + 3], axr, axi);
                updateConjLeaf(y, yStart + (j + 2) * incY * 2, A[ap + 4], A[ap + 5], axr, axi);
                updateConjLeaf(y, yStart + (j + 3) * incY * 2, A[ap + 6], A[ap + 7], axr, axi);
            }
            for (; j < n; j++) {
                int yp = yStart + j * incY * 2;
                int ap = rowOff + j * 2;
                updateConjLeaf(y, yp, A[ap], A[ap + 1], axr, axi);
            }
        }
    }

    static void gemvConjTransPairedRows(int m, int n, double alphaRe, double alphaIm,
                                        double[] A, int aStart, int lda,
                                        double[] x, int xStart, int incX,
                                        double[] y, int yStart) {
        int n2 = (n / 2) * 2;
        int i = 0;

        for (; i + 1 < m; i += 2) {
            int x0 = xStart + i * incX * 2;
            int x1 = x0 + incX * 2;
            double ax0r = alphaRe * x[x0] - alphaIm * x[x0 + 1];
            double ax0i = alphaRe * x[x0 + 1] + alphaIm * x[x0];
            double ax1r = alphaRe * x[x1] - alphaIm * x[x1 + 1];
            double ax1i = alphaRe * x[x1 + 1] + alphaIm * x[x1];
            int row0 = aStart + i * lda * 2;
            int row1 = row0 + lda * 2;

            int j = 0;
            for (; j < n2; j += 2) {
                int yp = yStart + j * 2;
                double y0r = y[yp];
                double y0i = y[yp + 1];
                double y1r = y[yp + 2];
                double y1i = y[yp + 3];

                int ap0 = row0 + j * 2;
                double a00r = A[ap0], a00i = A[ap0 + 1];
                double a01r = A[ap0 + 2], a01i = A[ap0 + 3];
                int ap1 = row1 + j * 2;
                double a10r = A[ap1], a10i = A[ap1 + 1];
                double a11r = A[ap1 + 2], a11i = A[ap1 + 3];

                y0r = Math.fma(a00r, ax0r, Math.fma(a00i, ax0i, y0r));
                y0i = Math.fma(a00r, ax0i, Math.fma(-a00i, ax0r, y0i));
                y1r = Math.fma(a01r, ax0r, Math.fma(a01i, ax0i, y1r));
                y1i = Math.fma(a01r, ax0i, Math.fma(-a01i, ax0r, y1i));

                y0r = Math.fma(a10r, ax1r, Math.fma(a10i, ax1i, y0r));
                y0i = Math.fma(a10r, ax1i, Math.fma(-a10i, ax1r, y0i));
                y1r = Math.fma(a11r, ax1r, Math.fma(a11i, ax1i, y1r));
                y1i = Math.fma(a11r, ax1i, Math.fma(-a11i, ax1r, y1i));

                y[yp] = y0r;
                y[yp + 1] = y0i;
                y[yp + 2] = y1r;
                y[yp + 3] = y1i;
            }

            for (; j < n; j++) {
                int yp = yStart + j * 2;
                int ap0 = row0 + j * 2;
                double a00r = A[ap0], a00i = A[ap0 + 1];
                int ap1 = row1 + j * 2;
                double a10r = A[ap1], a10i = A[ap1 + 1];
                y[yp] = Math.fma(a10r, ax1r, Math.fma(a10i, ax1i, Math.fma(a00r, ax0r, Math.fma(a00i, ax0i, y[yp]))));
                y[yp + 1] = Math.fma(a10r, ax1i, Math.fma(-a10i, ax1r, Math.fma(a00r, ax0i, Math.fma(-a00i, ax0r, y[yp + 1]))));
            }
        }

        if (i < m) {
            gemvConjTransScalarLeaf(1, n, alphaRe, alphaIm, A, aStart + i * lda * 2, lda, x, xStart + i * incX * 2, incX, y, yStart, 1);
        }
    }

    static void gemvTrans(int m, int n, double alphaRe, double alphaIm,
                          double[] A, int aStart, int lda,
                          double[] x, int xStart, int incX,
                          double[] y, int yStart, int incY) {
        if (m <= 0) return;
        if (m <= THRESHOLD) {
            if (incY == 1 && m > 1 && n >= TRANS_PAIRED_ROW_MIN_N) {
                gemvTransPairedRows(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart);
                return;
            }
            gemvTransScalarLeaf(m, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
            return;
        }
        int mid = m / 2;
        gemvTrans(mid, n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
        gemvTrans(m - mid, n, alphaRe, alphaIm, A, aStart + mid * lda * 2, lda,
                  x, xStart + mid * incX * 2, incX, y, yStart, incY);
    }

    static void gemvTransScalarLeaf(int m, int n, double alphaRe, double alphaIm,
                                    double[] A, int aStart, int lda,
                                    double[] x, int xStart, int incX,
                                    double[] y, int yStart, int incY) {
        int n4 = (n / 4) * 4;
        for (int i = 0; i < m; i++) {
            int xPos = xStart + i * incX * 2;
            double axr = alphaRe * x[xPos] - alphaIm * x[xPos + 1];
            double axi = alphaRe * x[xPos + 1] + alphaIm * x[xPos];
            int rowOff = aStart + i * lda * 2;
            int j = 0;
            for (; j < n4; j += 4) {
                int ap = rowOff + j * 2;
                updateTransLeaf(y, yStart + j * incY * 2, A[ap], A[ap + 1], axr, axi);
                updateTransLeaf(y, yStart + (j + 1) * incY * 2, A[ap + 2], A[ap + 3], axr, axi);
                updateTransLeaf(y, yStart + (j + 2) * incY * 2, A[ap + 4], A[ap + 5], axr, axi);
                updateTransLeaf(y, yStart + (j + 3) * incY * 2, A[ap + 6], A[ap + 7], axr, axi);
            }
            for (; j < n; j++) {
                int yp = yStart + j * incY * 2;
                int ap = rowOff + j * 2;
                updateTransLeaf(y, yp, A[ap], A[ap + 1], axr, axi);
            }
        }
    }

    private static void updateConjLeaf(double[] y, int yp, double ar, double ai, double axr, double axi) {
        y[yp] = Math.fma(ar, axr, Math.fma(ai, axi, y[yp]));
        y[yp + 1] = Math.fma(ar, axi, Math.fma(-ai, axr, y[yp + 1]));
    }

    private static void updateTransLeaf(double[] y, int yp, double ar, double ai, double axr, double axi) {
        y[yp] = Math.fma(ar, axr, Math.fma(-ai, axi, y[yp]));
        y[yp + 1] = Math.fma(ar, axi, Math.fma(ai, axr, y[yp + 1]));
    }

    static void gemvTransPairedRows(int m, int n, double alphaRe, double alphaIm,
                                    double[] A, int aStart, int lda,
                                    double[] x, int xStart, int incX,
                                    double[] y, int yStart) {
        int n2 = (n / 2) * 2;
        int i = 0;

        for (; i + 1 < m; i += 2) {
            int x0 = xStart + i * incX * 2;
            int x1 = x0 + incX * 2;
            double ax0r = alphaRe * x[x0] - alphaIm * x[x0 + 1];
            double ax0i = alphaRe * x[x0 + 1] + alphaIm * x[x0];
            double ax1r = alphaRe * x[x1] - alphaIm * x[x1 + 1];
            double ax1i = alphaRe * x[x1 + 1] + alphaIm * x[x1];
            int row0 = aStart + i * lda * 2;
            int row1 = row0 + lda * 2;

            int j = 0;
            for (; j < n2; j += 2) {
                int yp = yStart + j * 2;
                double y0r = y[yp];
                double y0i = y[yp + 1];
                double y1r = y[yp + 2];
                double y1i = y[yp + 3];

                int ap0 = row0 + j * 2;
                double a00r = A[ap0], a00i = A[ap0 + 1];
                double a01r = A[ap0 + 2], a01i = A[ap0 + 3];
                int ap1 = row1 + j * 2;
                double a10r = A[ap1], a10i = A[ap1 + 1];
                double a11r = A[ap1 + 2], a11i = A[ap1 + 3];

                y0r = Math.fma(a00r, ax0r, Math.fma(-a00i, ax0i, y0r));
                y0i = Math.fma(a00r, ax0i, Math.fma(a00i, ax0r, y0i));
                y1r = Math.fma(a01r, ax0r, Math.fma(-a01i, ax0i, y1r));
                y1i = Math.fma(a01r, ax0i, Math.fma(a01i, ax0r, y1i));

                y0r = Math.fma(a10r, ax1r, Math.fma(-a10i, ax1i, y0r));
                y0i = Math.fma(a10r, ax1i, Math.fma(a10i, ax1r, y0i));
                y1r = Math.fma(a11r, ax1r, Math.fma(-a11i, ax1i, y1r));
                y1i = Math.fma(a11r, ax1i, Math.fma(a11i, ax1r, y1i));

                y[yp] = y0r;
                y[yp + 1] = y0i;
                y[yp + 2] = y1r;
                y[yp + 3] = y1i;
            }

            for (; j < n; j++) {
                int yp = yStart + j * 2;
                int ap0 = row0 + j * 2;
                double a00r = A[ap0], a00i = A[ap0 + 1];
                int ap1 = row1 + j * 2;
                double a10r = A[ap1], a10i = A[ap1 + 1];
                y[yp] = Math.fma(a10r, ax1r, Math.fma(-a10i, ax1i, Math.fma(a00r, ax0r, Math.fma(-a00i, ax0i, y[yp]))));
                y[yp + 1] = Math.fma(a10r, ax1i, Math.fma(a10i, ax1r, Math.fma(a00r, ax0i, Math.fma(a00i, ax0r, y[yp + 1]))));
            }
        }

        if (i < m) {
            gemvTransScalarLeaf(1, n, alphaRe, alphaIm, A, aStart + i * lda * 2, lda, x, xStart + i * incX * 2, incX, y, yStart, 1);
        }
    }
}

final class ZgemvSIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;
    private static final VectorShuffle<Double> SWAP_RE_IM = VectorShuffle.fromArray(SPECIES, buildSwapIndexes(), 0);
    private static final DoubleVector REAL_PAIR_SIGN = DoubleVector.fromArray(SPECIES, buildRealPairSign(), 0);

    private ZgemvSIMD() {
    }

    static boolean gemvNoTransPairedRows(int m, int n, double alphaRe, double alphaIm,
                                         double[] A, int aStart, int lda,
                                         double[] x, int xStart,
                                         double[] y, int yStart, int incY) {
        if (LANES <= 1) {
            return false;
        }

        int length = n * 2;
        int limit = SPECIES.loopBound(length);
        int row = 0;
        for (; row + 1 < m; row += 2) {
            int row0 = aStart + row * lda * 2;
            int row1 = row0 + lda * 2;

            DoubleVector reAcc0 = DoubleVector.zero(SPECIES);
            DoubleVector imAcc0 = DoubleVector.zero(SPECIES);
            DoubleVector reAcc1 = DoubleVector.zero(SPECIES);
            DoubleVector imAcc1 = DoubleVector.zero(SPECIES);

            int k = 0;
            for (; k < limit; k += LANES) {
                DoubleVector xVec = DoubleVector.fromArray(SPECIES, x, xStart + k);
                DoubleVector xSwap = xVec.rearrange(SWAP_RE_IM);
                DoubleVector row0Vec = DoubleVector.fromArray(SPECIES, A, row0 + k);
                DoubleVector row1Vec = DoubleVector.fromArray(SPECIES, A, row1 + k);
                reAcc0 = row0Vec.fma(xVec, reAcc0);
                imAcc0 = row0Vec.fma(xSwap, imAcc0);
                reAcc1 = row1Vec.fma(xVec, reAcc1);
                imAcc1 = row1Vec.fma(xSwap, imAcc1);
            }

            double sum0r = reAcc0.mul(REAL_PAIR_SIGN).reduceLanes(VectorOperators.ADD);
            double sum0i = imAcc0.reduceLanes(VectorOperators.ADD);
            double sum1r = reAcc1.mul(REAL_PAIR_SIGN).reduceLanes(VectorOperators.ADD);
            double sum1i = imAcc1.reduceLanes(VectorOperators.ADD);

            for (; k < length; k += 2) {
                int xp = xStart + k;
                double xr = x[xp];
                double xi = x[xp + 1];

                int ap0 = row0 + k;
                double a0r = A[ap0];
                double a0i = A[ap0 + 1];
                sum0r = Math.fma(a0r, xr, Math.fma(-a0i, xi, sum0r));
                sum0i = Math.fma(a0r, xi, Math.fma(a0i, xr, sum0i));

                int ap1 = row1 + k;
                double a1r = A[ap1];
                double a1i = A[ap1 + 1];
                sum1r = Math.fma(a1r, xr, Math.fma(-a1i, xi, sum1r));
                sum1i = Math.fma(a1r, xi, Math.fma(a1i, xr, sum1i));
            }

            int y0 = yStart + row * incY * 2;
            int y1 = y0 + incY * 2;
            y[y0] = Math.fma(alphaRe, sum0r, Math.fma(-alphaIm, sum0i, y[y0]));
            y[y0 + 1] = Math.fma(alphaRe, sum0i, Math.fma(alphaIm, sum0r, y[y0 + 1]));
            y[y1] = Math.fma(alphaRe, sum1r, Math.fma(-alphaIm, sum1i, y[y1]));
            y[y1 + 1] = Math.fma(alphaRe, sum1i, Math.fma(alphaIm, sum1r, y[y1 + 1]));
        }

        if (row < m) {
            Zgemv.gemvNoTransDirectScalar(1, n, alphaRe, alphaIm,
                A, aStart + row * lda * 2, lda,
                x, xStart, 1,
                y, yStart + row * incY * 2, incY);
        }
        return true;
    }

    static boolean probe() {
        return LANES > 1;
    }

    private static int[] buildSwapIndexes() {
        int[] indexes = new int[LANES];
        for (int i = 0; i < LANES; i += 2) {
            indexes[i] = i + 1;
            indexes[i + 1] = i;
        }
        return indexes;
    }

    private static double[] buildRealPairSign() {
        double[] signs = new double[LANES];
        for (int i = 0; i < LANES; i++) {
            signs[i] = (i & 1) == 0 ? 1.0 : -1.0;
        }
        return signs;
    }
}
