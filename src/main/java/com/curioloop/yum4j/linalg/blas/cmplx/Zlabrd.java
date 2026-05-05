/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;

interface Zlabrd {

    int SMALL_PANEL_DIM = 32;

    static void zlabrd(int m, int n, int k, double[] A, int aOff, int lda, double[] d, int dOff, double[] e, int eOff, double[] tauQ, int tauQOff, double[] tauP, int tauPOff, double[] X, int xOff, int ldx, double[] Y, int yOff, int ldy) {
        if (m <= 0 || n <= 0 || k <= 0) return;
        k = Math.min(k, Math.min(m, n));
        // Internal helper contract: A/X/Y offsets arrive in raw interleaved-array slots, so convert back
        // to complex-entry coordinates only for vector helpers that index in complex elements.
        int aOffC = aOff / 2;
        int xOffC = xOff / 2;
        int yOffC = yOff / 2;

        if (m >= n) {
            for (int i = 0; i < k; i++) {
                Zlacgv.zlacgv(i, Y, yOffC + i * ldy, 1);
                panelZgemvNoTrans(m - i, i, -1.0, A, aOff + (i * lda) * 2, lda, Y, yOff + (i * ldy) * 2, 1, 1.0, A, aOff + (i * lda + i) * 2, lda);
                Zlacgv.zlacgv(i, Y, yOffC + i * ldy, 1);
                panelZgemvNoTrans(m - i, i, -1.0, X, xOff + (i * ldx) * 2, ldx, A, aOff + (i) * 2, lda, 1.0, A, aOff + (i * lda + i) * 2, lda);

                Zlarfg.zlarfg(m - i, A, aOff + (i * lda + i) * 2, A, aOffC + (i + 1) * lda + i, lda, tauQ, tauQOff + i * 2);
                d[dOff + i] = A[aOff + (i * lda + i) * 2];

                if (i < n - 1 && i < k - 1) {
                    A[aOff + (i * lda + i) * 2] = 1.0;
                    A[aOff + (i * lda + i) * 2 + 1] = 0.0;

                    panelZgemvTrans(m - i, n - i - 1, 1.0, A, aOff + (i * lda + i + 1) * 2, lda, A, aOff + (i * lda + i) * 2, lda, 0.0, Y, yOff + ((i + 1) * ldy + i) * 2, ldy);
                    panelZgemvTrans(m - i, i, 1.0, A, aOff + (i * lda) * 2, lda, A, aOff + (i * lda + i) * 2, lda, 0.0, Y, yOff + (i) * 2, ldy);
                    panelZgemvNoTrans(n - i - 1, i, -1.0, Y, yOff + ((i + 1) * ldy) * 2, ldy, Y, yOff + (i) * 2, ldy, 1.0, Y, yOff + ((i + 1) * ldy + i) * 2, ldy);
                    panelZgemvTrans(m - i, i, 1.0, X, xOff + (i * ldx) * 2, ldx, A, aOff + (i * lda + i) * 2, lda, 0.0, Y, yOff + (i) * 2, ldy);
                    panelZgemvTrans(i, n - i - 1, -1.0, A, aOff + (i + 1) * 2, lda, Y, yOff + (i) * 2, ldy, 1.0, Y, yOff + ((i + 1) * ldy + i) * 2, ldy);
                    Zscal.zscal(n - i - 1, tauQ[tauQOff + i * 2], tauQ[tauQOff + i * 2 + 1], Y, yOff + ((i + 1) * ldy + i) * 2, ldy);

                    Zlacgv.zlacgv(n - i - 1, A, aOffC + i * lda + i + 1, 1);
                    Zlacgv.zlacgv(i + 1, A, aOffC + i * lda, 1);
                    panelZgemvNoTrans(n - i - 1, i + 1, -1.0, Y, yOff + ((i + 1) * ldy) * 2, ldy, A, aOff + (i * lda) * 2, 1, 1.0, A, aOff + (i * lda + i + 1) * 2, 1);
                    Zlacgv.zlacgv(i + 1, A, aOffC + i * lda, 1);
                    Zlacgv.zlacgv(i, X, xOffC + i * ldx, 1);
                    panelZgemvTrans(i, n - i - 1, -1.0, A, aOff + (i + 1) * 2, lda, X, xOff + (i * ldx) * 2, 1, 1.0, A, aOff + (i * lda + i + 1) * 2, 1);
                    Zlacgv.zlacgv(i, X, xOffC + i * ldx, 1);

                    Zlarfg.zlarfg(n - i - 1, A, aOff + (i * lda + i + 1) * 2, A, aOffC + i * lda + i + 2, 1, tauP, tauPOff + i * 2);
                    e[eOff + i] = A[aOff + (i * lda + i + 1) * 2];
                    A[aOff + (i * lda + i + 1) * 2] = 1.0;
                    A[aOff + (i * lda + i + 1) * 2 + 1] = 0.0;

                    panelZgemvNoTrans(m - i - 1, n - i - 1, 1.0, A, aOff + ((i + 1) * lda + i + 1) * 2, lda, A, aOff + (i * lda + i + 1) * 2, 1, 0.0, X, xOff + ((i + 1) * ldx + i) * 2, ldx);
                    panelZgemvTrans(n - i - 1, i + 1, 1.0, Y, yOff + ((i + 1) * ldy) * 2, ldy, A, aOff + (i * lda + i + 1) * 2, 1, 0.0, X, xOff + (i) * 2, ldx);
                    panelZgemvNoTrans(m - i - 1, i + 1, -1.0, A, aOff + ((i + 1) * lda) * 2, lda, X, xOff + (i) * 2, ldx, 1.0, X, xOff + ((i + 1) * ldx + i) * 2, ldx);
                    panelZgemvNoTrans(i, n - i - 1, 1.0, A, aOff + (i + 1) * 2, lda, A, aOff + (i * lda + i + 1) * 2, 1, 0.0, X, xOff + (i) * 2, ldx);
                    panelZgemvNoTrans(m - i - 1, i, -1.0, X, xOff + ((i + 1) * ldx) * 2, ldx, X, xOff + (i) * 2, ldx, 1.0, X, xOff + ((i + 1) * ldx + i) * 2, ldx);
                    Zscal.zscal(m - i - 1, tauP[tauPOff + i * 2], tauP[tauPOff + i * 2 + 1], X, xOff + ((i + 1) * ldx + i) * 2, ldx);
                    Zlacgv.zlacgv(n - i - 1, A, aOffC + i * lda + i + 1, 1);
                }
            }
        } else {
            for (int i = 0; i < Math.min(k, m); i++) {
                Zlacgv.zlacgv(n - i, A, aOffC + i * lda + i, 1);
                Zlacgv.zlacgv(i, A, aOffC + i * lda, 1);
                panelZgemvNoTrans(n - i, i, -1.0, Y, yOff + (i * ldy) * 2, ldy, A, aOff + (i * lda) * 2, 1, 1.0, A, aOff + (i * lda + i) * 2, 1);
                Zlacgv.zlacgv(i, A, aOffC + i * lda, 1);
                Zlacgv.zlacgv(i, X, xOffC + i * ldx, 1);
                panelZgemvTrans(i, n - i, -1.0, A, aOff + (i) * 2, lda, X, xOff + (i * ldx) * 2, 1, 1.0, A, aOff + (i * lda + i) * 2, 1);
                Zlacgv.zlacgv(i, X, xOffC + i * ldx, 1);

                Zlarfg.zlarfg(n - i, A, aOff + (i * lda + i) * 2, A, aOffC + i * lda + i + 1, 1, tauP, tauPOff + i * 2);
                d[dOff + i] = A[aOff + (i * lda + i) * 2];

                if (i < m - 1 && i < k - 1) {
                    A[aOff + (i * lda + i) * 2] = 1.0;
                    A[aOff + (i * lda + i) * 2 + 1] = 0.0;

                    panelZgemvNoTrans(m - i - 1, n - i, 1.0, A, aOff + ((i + 1) * lda + i) * 2, lda, A, aOff + (i * lda + i) * 2, 1, 0.0, X, xOff + ((i + 1) * ldx + i) * 2, ldx);
                    panelZgemvTrans(n - i, i, 1.0, Y, yOff + (i * ldy) * 2, ldy, A, aOff + (i * lda + i) * 2, 1, 0.0, X, xOff + (i) * 2, ldx);
                    panelZgemvNoTrans(m - i - 1, i, -1.0, A, aOff + ((i + 1) * lda) * 2, lda, X, xOff + (i) * 2, ldx, 1.0, X, xOff + ((i + 1) * ldx + i) * 2, ldx);
                    panelZgemvNoTrans(i, n - i, 1.0, A, aOff + i * 2, lda, A, aOff + (i * lda + i) * 2, 1, 0.0, X, xOff + (i) * 2, ldx);
                    panelZgemvNoTrans(m - i - 1, i, -1.0, X, xOff + ((i + 1) * ldx) * 2, ldx, X, xOff + (i) * 2, ldx, 1.0, X, xOff + ((i + 1) * ldx + i) * 2, ldx);
                    Zscal.zscal(m - i - 1, tauP[tauPOff + i * 2], tauP[tauPOff + i * 2 + 1], X, xOff + ((i + 1) * ldx + i) * 2, ldx);
                    Zlacgv.zlacgv(n - i, A, aOffC + i * lda + i, 1);

                    Zlacgv.zlacgv(i, Y, yOffC + i * ldy, 1);
                    panelZgemvNoTrans(m - i - 1, i, -1.0, A, aOff + ((i + 1) * lda) * 2, lda, Y, yOff + (i * ldy) * 2, 1, 1.0, A, aOff + ((i + 1) * lda + i) * 2, lda);
                    Zlacgv.zlacgv(i, Y, yOffC + i * ldy, 1);
                    panelZgemvNoTrans(m - i - 1, i + 1, -1.0, X, xOff + ((i + 1) * ldx) * 2, ldx, A, aOff + (i) * 2, lda, 1.0, A, aOff + ((i + 1) * lda + i) * 2, lda);

                    Zlarfg.zlarfg(m - i - 1, A, aOff + ((i + 1) * lda + i) * 2, A, aOffC + (i + 2) * lda + i, lda, tauQ, tauQOff + i * 2);
                    e[eOff + i] = A[aOff + ((i + 1) * lda + i) * 2];
                    A[aOff + ((i + 1) * lda + i) * 2] = 1.0;
                    A[aOff + ((i + 1) * lda + i) * 2 + 1] = 0.0;

                    panelZgemvTrans(m - i - 1, n - i - 1, 1.0, A, aOff + ((i + 1) * lda + i + 1) * 2, lda, A, aOff + ((i + 1) * lda + i) * 2, lda, 0.0, Y, yOff + ((i + 1) * ldy + i) * 2, ldy);
                    panelZgemvTrans(m - i - 1, i, 1.0, A, aOff + ((i + 1) * lda) * 2, lda, A, aOff + ((i + 1) * lda + i) * 2, lda, 0.0, Y, yOff + (i) * 2, ldy);
                    panelZgemvNoTrans(n - i - 1, i, -1.0, Y, yOff + ((i + 1) * ldy) * 2, ldy, Y, yOff + (i) * 2, ldy, 1.0, Y, yOff + ((i + 1) * ldy + i) * 2, ldy);
                    panelZgemvTrans(m - i - 1, i + 1, 1.0, X, xOff + ((i + 1) * ldx) * 2, ldx, A, aOff + ((i + 1) * lda + i) * 2, lda, 0.0, Y, yOff + (i) * 2, ldy);
                    panelZgemvTrans(i + 1, n - i - 1, -1.0, A, aOff + (i + 1) * 2, lda, Y, yOff + (i) * 2, ldy, 1.0, Y, yOff + ((i + 1) * ldy + i) * 2, ldy);
                    Zscal.zscal(n - i - 1, tauQ[tauQOff + i * 2], tauQ[tauQOff + i * 2 + 1], Y, yOff + ((i + 1) * ldy + i) * 2, ldy);
                } else {
                    Zlacgv.zlacgv(n - i, A, aOffC + i * lda + i, 1);
                }
            }
        }
    }

    static void panelZgemvNoTrans(int m, int n, double alpha,
                                  double[] matrix, int matrixOff, int lda,
                                  double[] x, int xOff, int incX,
                                  double beta,
                                  double[] y, int yOff, int incY) {
        if (m <= 0 || n <= 0) {
            return;
        }
        if (n > SMALL_PANEL_DIM) {
            Zgemv.zgemv(BLAS.Trans.NoTrans, m, n, alpha, 0.0, matrix, matrixOff, lda, x, xOff, incX, beta, 0.0, y, yOff, incY);
            return;
        }
        scaleVector(m, beta, y, yOff, incY);
        if (alpha == 0.0) {
            return;
        }
        int incX2 = incX * 2;
        int incY2 = incY * 2;
        for (int row = 0; row < m; row++) {
            int rowOff = matrixOff + row * lda * 2;
            int xPos = xOff;
            double sumRe = 0.0;
            double sumIm = 0.0;
            for (int col = 0; col < n; col++) {
                int matrixPos = rowOff + col * 2;
                double matrixRe = matrix[matrixPos];
                double matrixIm = matrix[matrixPos + 1];
                double xRe = x[xPos];
                double xIm = x[xPos + 1];
                sumRe = Math.fma(matrixRe, xRe, Math.fma(-matrixIm, xIm, sumRe));
                sumIm = Math.fma(matrixRe, xIm, Math.fma(matrixIm, xRe, sumIm));
                xPos += incX2;
            }
            int yPos = yOff + row * incY2;
            y[yPos] = Math.fma(alpha, sumRe, y[yPos]);
            y[yPos + 1] = Math.fma(alpha, sumIm, y[yPos + 1]);
        }
    }

    static void panelZgemvTrans(int m, int n, double alpha,
                                double[] matrix, int matrixOff, int lda,
                                double[] x, int xOff, int incX,
                                double beta,
                                double[] y, int yOff, int incY) {
        if (m <= 0 || n <= 0) {
            return;
        }
        if (m > SMALL_PANEL_DIM && n > SMALL_PANEL_DIM) {
            Zgemv.zgemv(BLAS.Trans.Trans, m, n, alpha, 0.0, matrix, matrixOff, lda, x, xOff, incX, beta, 0.0, y, yOff, incY);
            return;
        }
        scaleVector(n, beta, y, yOff, incY);
        if (alpha == 0.0) {
            return;
        }
        int incX2 = incX * 2;
        int incY2 = incY * 2;
        if (m <= n) {
            for (int row = 0; row < m; row++) {
                int xPos = xOff + row * incX2;
                double scaledXRe = alpha * x[xPos];
                double scaledXIm = alpha * x[xPos + 1];
                if (scaledXRe == 0.0 && scaledXIm == 0.0) {
                    continue;
                }
                int rowOff = matrixOff + row * lda * 2;
                int yPos = yOff;
                for (int col = 0; col < n; col++) {
                    int matrixPos = rowOff + col * 2;
                    double matrixRe = matrix[matrixPos];
                    double matrixIm = matrix[matrixPos + 1];
                    y[yPos] = Math.fma(scaledXRe, matrixRe, Math.fma(-scaledXIm, matrixIm, y[yPos]));
                    y[yPos + 1] = Math.fma(scaledXRe, matrixIm, Math.fma(scaledXIm, matrixRe, y[yPos + 1]));
                    yPos += incY2;
                }
            }
            return;
        }
        for (int col = 0; col < n; col++) {
            int matrixPos = matrixOff + col * 2;
            int xPos = xOff;
            double sumRe = 0.0;
            double sumIm = 0.0;
            for (int row = 0; row < m; row++) {
                double matrixRe = matrix[matrixPos];
                double matrixIm = matrix[matrixPos + 1];
                double xRe = x[xPos];
                double xIm = x[xPos + 1];
                sumRe = Math.fma(matrixRe, xRe, Math.fma(-matrixIm, xIm, sumRe));
                sumIm = Math.fma(matrixRe, xIm, Math.fma(matrixIm, xRe, sumIm));
                matrixPos += lda * 2;
                xPos += incX2;
            }
            int yPos = yOff + col * incY2;
            y[yPos] = Math.fma(alpha, sumRe, y[yPos]);
            y[yPos + 1] = Math.fma(alpha, sumIm, y[yPos + 1]);
        }
    }

    static void scaleVector(int n, double beta, double[] vector, int off, int inc) {
        int inc2 = inc * 2;
        if (beta == 0.0) {
            for (int i = 0; i < n; i++) {
                int pos = off + i * inc2;
                vector[pos] = 0.0;
                vector[pos + 1] = 0.0;
            }
        } else if (beta != 1.0) {
            for (int i = 0; i < n; i++) {
                int pos = off + i * inc2;
                vector[pos] *= beta;
                vector[pos + 1] *= beta;
            }
        }
    }
}
