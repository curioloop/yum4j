/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

/**
 * ZGERU performs the rank 1 operation
 * A := alpha*x*y^T + A
 * where alpha is a scalar, x is an m element vector, y is an n element vector
 * and A is an m by n matrix.
 */
interface Zgeru {

    /**
     * Performs the rank 1 operation
     * A := alpha*x*y^T + A
     *
     * @param m      Number of rows of the matrix A
     * @param n      Number of columns of the matrix A
     * @param alphaRe Real part of the scalar alpha
     * @param alphaIm Imaginary part of the scalar alpha
     * @param x      Complex vector x stored in interleaved format [re, im, re, im, ...]
     * @param xStart Starting index of x
     * @param incX   Increment for the elements of x
     * @param y      Complex vector y stored in interleaved format [re, im, re, im, ...]
     * @param yStart Starting index of y
     * @param incY   Increment for the elements of y
     * @param A      Complex matrix A stored in row-major order, interleaved format [re, im, re, im, ...]
     * @param aStart Starting index of A
     * @param lda    Leading dimension of the matrix A
     */
    static void zgeru(int m, int n, double alphaRe, double alphaIm,
                      double[] x, int xStart, int incX,
                      double[] y, int yStart, int incY,
                      double[] A, int aStart, int lda) {
        if (m <= 0 || n <= 0 || (alphaRe == 0.0 && alphaIm == 0.0)) {
            return;
        }

        if (incX == 1 && incY == 1) {
            for (int i = 0; i < m; i++) {
                int xp = xStart + i * 2;
                double xr = x[xp];
                double xi = x[xp + 1];
                double axr = alphaRe * xr - alphaIm * xi;
                double axi = alphaRe * xi + alphaIm * xr;
                int rowOff = aStart + i * lda * 2;
                int j = 0;
                int n4 = (n / 4) * 4;
                for (; j < n4; j += 4) {
                    int yp = yStart + j * 2;
                    int ap = rowOff + j * 2;
                    double y0r = y[yp];
                    double y0i = y[yp + 1];
                    double y1r = y[yp + 2];
                    double y1i = y[yp + 3];
                    double y2r = y[yp + 4];
                    double y2i = y[yp + 5];
                    double y3r = y[yp + 6];
                    double y3i = y[yp + 7];
                    A[ap] = Math.fma(axr, y0r, Math.fma(-axi, y0i, A[ap]));
                    A[ap + 1] = Math.fma(axr, y0i, Math.fma(axi, y0r, A[ap + 1]));
                    A[ap + 2] = Math.fma(axr, y1r, Math.fma(-axi, y1i, A[ap + 2]));
                    A[ap + 3] = Math.fma(axr, y1i, Math.fma(axi, y1r, A[ap + 3]));
                    A[ap + 4] = Math.fma(axr, y2r, Math.fma(-axi, y2i, A[ap + 4]));
                    A[ap + 5] = Math.fma(axr, y2i, Math.fma(axi, y2r, A[ap + 5]));
                    A[ap + 6] = Math.fma(axr, y3r, Math.fma(-axi, y3i, A[ap + 6]));
                    A[ap + 7] = Math.fma(axr, y3i, Math.fma(axi, y3r, A[ap + 7]));
                }
                for (; j < n; j++) {
                    int yp = yStart + j * 2;
                    int ap = rowOff + j * 2;
                    double yr = y[yp];
                    double yi = y[yp + 1];
                    A[ap] = Math.fma(axr, yr, Math.fma(-axi, yi, A[ap]));
                    A[ap + 1] = Math.fma(axr, yi, Math.fma(axi, yr, A[ap + 1]));
                }
            }
            return;
        }

        for (int i = 0; i < m; i++) {
            int xp = xStart + i * incX * 2;
            double xr = x[xp];
            double xi = x[xp + 1];
            double axr = alphaRe * xr - alphaIm * xi;
            double axi = alphaRe * xi + alphaIm * xr;
            int rowOff = aStart + i * lda * 2;
            for (int j = 0; j < n; j++) {
                int yp = yStart + j * incY * 2;
                int ap = rowOff + j * 2;
                double yr = y[yp];
                double yi = y[yp + 1];
                A[ap] = Math.fma(axr, yr, Math.fma(-axi, yi, A[ap]));
                A[ap + 1] = Math.fma(axr, yi, Math.fma(axi, yr, A[ap + 1]));
            }
        }
    }
}