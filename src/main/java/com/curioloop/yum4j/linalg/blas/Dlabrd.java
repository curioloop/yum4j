/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas;

import static java.lang.Math.*;

/**
 * DLABRD: Reduces the first NB rows and columns of a real general m×n matrix A
 * to upper or lower bidiagonal form by an orthogonal transformation Q**T * A * P.
 * Also returns the matrices X and Y needed to apply the transformation to the
 * unreduced part of A: A := A - V*Y**T - X*U**T.
 *
 */
interface Dlabrd {

    int SMALL_PANEL_DIM = 32;

    static void dlabrd(int m, int n, int nb, double[] A, int aOff, int lda,
                       double[] d, int dOff, double[] e, int eOff,
                       double[] tauQ, int tauQOff, double[] tauP, int tauPOff,
                       double[] x, int xOff, int ldx, double[] y, int yOff, int ldy) {
        if (m == 0 || n == 0 || nb == 0) {
            return;
        }

        if (m >= n) {
            for (int i = 0; i < nb; i++) {
                panelDgemvNoTrans(m - i, i, -1, A, aOff + i * lda, lda, y, yOff + i * ldy, 1, 1, A, aOff + i * lda + i, lda);
                panelDgemvNoTrans(m - i, i, -1, x, xOff + i * ldx, ldx, A, aOff + i, lda, 1, A, aOff + i * lda + i, lda);

                double aii = Dlarfg.dlarfg(m - i, A[aOff + i * lda + i], A, aOff + min(i + 1, m - 1) * lda + i, lda, tauQ, tauQOff + i);
                d[dOff + i] = aii;

                if (i < n - 1) {
                    A[aOff + i * lda + i] = 1;
                    panelDgemvTrans(m - i, n - i - 1, 1, A, aOff + i * lda + i + 1, lda, A, aOff + i * lda + i, lda, 0, y, yOff + (i + 1) * ldy + i, ldy);
                    panelDgemvTrans(m - i, i, 1, A, aOff + i * lda, lda, A, aOff + i * lda + i, lda, 0, y, yOff + i, ldy);
                    panelDgemvNoTrans(n - i - 1, i, -1, y, yOff + (i + 1) * ldy, ldy, y, yOff + i, ldy, 1, y, yOff + (i + 1) * ldy + i, ldy);
                    panelDgemvTrans(m - i, i, 1, x, xOff + i * ldx, ldx, A, aOff + i * lda + i, lda, 0, y, yOff + i, ldy);
                    panelDgemvTrans(i, n - i - 1, -1, A, aOff + i + 1, lda, y, yOff + i, ldy, 1, y, yOff + (i + 1) * ldy + i, ldy);
                    BLAS.dscal(n - i - 1, tauQ[tauQOff + i], y, yOff + (i + 1) * ldy + i, ldy);

                    panelDgemvNoTrans(n - i - 1, i + 1, -1, y, yOff + (i + 1) * ldy, ldy, A, aOff + i * lda, 1, 1, A, aOff + i * lda + i + 1, 1);
                    panelDgemvTrans(i, n - i - 1, -1, A, aOff + i + 1, lda, x, xOff + i * ldx, 1, 1, A, aOff + i * lda + i + 1, 1);

                    double aii1 = Dlarfg.dlarfg(n - i - 1, A[aOff + i * lda + i + 1], A, aOff + i * lda + min(i + 2, n - 1), 1, tauP, tauPOff + i);
                    e[eOff + i] = aii1;
                    A[aOff + i * lda + i + 1] = 1;

                    panelDgemvNoTrans(m - i - 1, n - i - 1, 1, A, aOff + (i + 1) * lda + i + 1, lda, A, aOff + i * lda + i + 1, 1, 0, x, xOff + (i + 1) * ldx + i, ldx);
                    panelDgemvTrans(n - i - 1, i + 1, 1, y, yOff + (i + 1) * ldy, ldy, A, aOff + i * lda + i + 1, 1, 0, x, xOff + i, ldx);
                    panelDgemvNoTrans(m - i - 1, i + 1, -1, A, aOff + (i + 1) * lda, lda, x, xOff + i, ldx, 1, x, xOff + (i + 1) * ldx + i, ldx);
                    panelDgemvNoTrans(i, n - i - 1, 1, A, aOff + i + 1, lda, A, aOff + i * lda + i + 1, 1, 0, x, xOff + i, ldx);
                    panelDgemvNoTrans(m - i - 1, i, -1, x, xOff + (i + 1) * ldx, ldx, x, xOff + i, ldx, 1, x, xOff + (i + 1) * ldx + i, ldx);
                    BLAS.dscal(m - i - 1, tauP[tauPOff + i], x, xOff + (i + 1) * ldx + i, ldx);
                }
            }
        } else {
            for (int i = 0; i < nb; i++) {
                panelDgemvNoTrans(n - i, i, -1, y, yOff + i * ldy, ldy, A, aOff + i * lda, 1, 1, A, aOff + i * lda + i, 1);
                panelDgemvTrans(i, n - i, -1, A, aOff + i, lda, x, xOff + i * ldx, 1, 1, A, aOff + i * lda + i, 1);

                double aii = Dlarfg.dlarfg(n - i, A[aOff + i * lda + i], A, aOff + i * lda + min(i + 1, n - 1), 1, tauP, tauPOff + i);
                d[dOff + i] = aii;

                if (i < m - 1) {
                    A[aOff + i * lda + i] = 1;
                    panelDgemvNoTrans(m - i - 1, n - i, 1, A, aOff + (i + 1) * lda + i, lda, A, aOff + i * lda + i, 1, 0, x, xOff + (i + 1) * ldx + i, ldx);
                    panelDgemvTrans(n - i, i, 1, y, yOff + i * ldy, ldy, A, aOff + i * lda + i, 1, 0, x, xOff + i, ldx);
                    panelDgemvNoTrans(m - i - 1, i, -1, A, aOff + (i + 1) * lda, lda, x, xOff + i, ldx, 1, x, xOff + (i + 1) * ldx + i, ldx);
                    panelDgemvNoTrans(i, n - i, 1, A, aOff + i, lda, A, aOff + i * lda + i, 1, 0, x, xOff + i, ldx);
                    panelDgemvNoTrans(m - i - 1, i, -1, x, xOff + (i + 1) * ldx, ldx, x, xOff + i, ldx, 1, x, xOff + (i + 1) * ldx + i, ldx);
                    BLAS.dscal(m - i - 1, tauP[tauPOff + i], x, xOff + (i + 1) * ldx + i, ldx);

                    panelDgemvNoTrans(m - i - 1, i, -1, A, aOff + (i + 1) * lda, lda, y, yOff + i * ldy, 1, 1, A, aOff + (i + 1) * lda + i, lda);
                    panelDgemvNoTrans(m - i - 1, i + 1, -1, x, xOff + (i + 1) * ldx, ldx, A, aOff + i, lda, 1, A, aOff + (i + 1) * lda + i, lda);

                    double aii1 = Dlarfg.dlarfg(m - i - 1, A[aOff + (i + 1) * lda + i], A, aOff + min(i + 2, m - 1) * lda + i, lda, tauQ, tauQOff + i);
                    e[eOff + i] = aii1;
                    A[aOff + (i + 1) * lda + i] = 1;

                    panelDgemvTrans(m - i - 1, n - i - 1, 1, A, aOff + (i + 1) * lda + i + 1, lda, A, aOff + (i + 1) * lda + i, lda, 0, y, yOff + (i + 1) * ldy + i, ldy);
                    panelDgemvTrans(m - i - 1, i, 1, A, aOff + (i + 1) * lda, lda, A, aOff + (i + 1) * lda + i, lda, 0, y, yOff + i, ldy);
                    panelDgemvNoTrans(n - i - 1, i, -1, y, yOff + (i + 1) * ldy, ldy, y, yOff + i, ldy, 1, y, yOff + (i + 1) * ldy + i, ldy);
                    panelDgemvTrans(m - i - 1, i + 1, 1, x, xOff + (i + 1) * ldx, ldx, A, aOff + (i + 1) * lda + i, lda, 0, y, yOff + i, ldy);
                    panelDgemvTrans(i + 1, n - i - 1, -1, A, aOff + i + 1, lda, y, yOff + i, ldy, 1, y, yOff + (i + 1) * ldy + i, ldy);
                    BLAS.dscal(n - i - 1, tauQ[tauQOff + i], y, yOff + (i + 1) * ldy + i, ldy);
                }
            }
        }
    }

    static void panelDgemvNoTrans(int m, int n, double alpha,
                                  double[] matrix, int matrixOff, int lda,
                                  double[] x, int xOff, int incX,
                                  double beta,
                                  double[] y, int yOff, int incY) {
        if (m <= 0 || n <= 0) {
            return;
        }
        if (n > SMALL_PANEL_DIM) {
            BLAS.dgemv(BLAS.Trans.NoTrans, m, n, alpha, matrix, matrixOff, lda, x, xOff, incX, beta, y, yOff, incY);
            return;
        }
        scaleVector(m, beta, y, yOff, incY);
        if (alpha == 0.0) {
            return;
        }
        for (int row = 0; row < m; row++) {
            int rowOff = matrixOff + row * lda;
            int xPos = xOff;
            double sum = 0.0;
            for (int col = 0; col < n; col++) {
                sum = Math.fma(matrix[rowOff + col], x[xPos], sum);
                xPos += incX;
            }
            int yPos = yOff + row * incY;
            y[yPos] = Math.fma(alpha, sum, y[yPos]);
        }
    }

    static void panelDgemvTrans(int m, int n, double alpha,
                                double[] matrix, int matrixOff, int lda,
                                double[] x, int xOff, int incX,
                                double beta,
                                double[] y, int yOff, int incY) {
        if (m <= 0 || n <= 0) {
            return;
        }
        if (m > SMALL_PANEL_DIM && n > SMALL_PANEL_DIM) {
            BLAS.dgemv(BLAS.Trans.Trans, m, n, alpha, matrix, matrixOff, lda, x, xOff, incX, beta, y, yOff, incY);
            return;
        }
        scaleVector(n, beta, y, yOff, incY);
        if (alpha == 0.0) {
            return;
        }
        if (m <= n) {
            for (int row = 0; row < m; row++) {
                double scaledX = alpha * x[xOff + row * incX];
                if (scaledX == 0.0) {
                    continue;
                }
                int rowOff = matrixOff + row * lda;
                int yPos = yOff;
                for (int col = 0; col < n; col++) {
                    y[yPos] = Math.fma(scaledX, matrix[rowOff + col], y[yPos]);
                    yPos += incY;
                }
            }
            return;
        }
        for (int col = 0; col < n; col++) {
            int matrixPos = matrixOff + col;
            int xPos = xOff;
            double sum = 0.0;
            for (int row = 0; row < m; row++) {
                sum = Math.fma(matrix[matrixPos], x[xPos], sum);
                matrixPos += lda;
                xPos += incX;
            }
            int yPos = yOff + col * incY;
            y[yPos] = Math.fma(alpha, sum, y[yPos]);
        }
    }

    static void scaleVector(int n, double beta, double[] vector, int off, int inc) {
        if (beta == 0.0) {
            for (int i = 0; i < n; i++) {
                vector[off + i * inc] = 0.0;
            }
        } else if (beta != 1.0) {
            for (int i = 0; i < n; i++) {
                int pos = off + i * inc;
                vector[pos] *= beta;
            }
        }
    }
}
