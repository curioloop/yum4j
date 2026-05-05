package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
/**
 * ZHEMV performs the matrix-vector operation
 * y := alpha*A*x + beta*y
 * where alpha and beta are scalars, x and y are n element vectors
 * and A is an n by n Hermitian matrix.
 */
interface Zhemv {

    /**
     * Performs the matrix-vector operation
     * y := alpha*A*x + beta*y
     *
     * @param uplo   Specifies whether the upper or lower triangular part of the Hermitian matrix A is stored:
     *               BLAS.Uplo.Upper: Upper triangular part is stored
     *               BLAS.Uplo.Lower: Lower triangular part is stored
     * @param n      Order of the matrix A
     * @param alphaRe Real part of the scalar alpha
     * @param alphaIm Imaginary part of the scalar alpha
     * @param A      Complex Hermitian matrix A stored in row-major order, interleaved format [re, im, re, im, ...]
     * @param aStart Starting index of A
     * @param lda    Leading dimension of the matrix A (must be at least max(1, n))
     * @param x      Complex vector x stored in interleaved format [re, im, re, im, ...]
     * @param xStart Starting index of x
     * @param incX   Increment for the elements of x
     * @param betaRe  Real part of the scalar beta
     * @param betaIm  Imaginary part of the scalar beta
     * @param y      Complex vector y stored in interleaved format [re, im, re, im, ...]
     * @param yStart Starting index of y
     * @param incY   Increment for the elements of y
     */
    public static void zhemv(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm,
                            double[] A, int aStart, int lda,
                            double[] x, int xStart, int incX,
                            double betaRe, double betaIm,
                            double[] y, int yStart, int incY) {
        if (n <= 0) return;

        boolean upper = (uplo == BLAS.Uplo.Upper);

        if (betaRe == 0.0 && betaIm == 0.0) {
            for (int i = 0; i < n; i++) {
                int yPos = yStart + i * incY * 2;
                y[yPos] = 0.0;
                y[yPos + 1] = 0.0;
            }
        } else if (betaRe != 1.0 || betaIm != 0.0) {
            for (int i = 0; i < n; i++) {
                int yPos = yStart + i * incY * 2;
                double yRe = y[yPos];
                double yIm = y[yPos + 1];
                y[yPos] = betaRe * yRe - betaIm * yIm;
                y[yPos + 1] = betaRe * yIm + betaIm * yRe;
            }
        }

        if (alphaRe == 0.0 && alphaIm == 0.0) return;

        if (n == 1) {
            double aRe = A[aStart];
            double xRe = x[xStart];
            double xIm = x[xStart + 1];
            double axRe = aRe * xRe;
            double axIm = aRe * xIm;
            y[yStart] = Math.fma(alphaRe, axRe, Math.fma(-alphaIm, axIm, y[yStart]));
            y[yStart + 1] = Math.fma(alphaRe, axIm, Math.fma(alphaIm, axRe, y[yStart + 1]));
            return;
        }

        if (upper) {
            if (incX == 1 && incY == 1) {
                zhemvUpperUnitStride(n, alphaRe, alphaIm, A, aStart, lda, x, xStart, y, yStart);
            } else {
                zhemvUpperStride(n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
            }
        } else {
            if (incX == 1 && incY == 1) {
                zhemvLowerUnitStride(n, alphaRe, alphaIm, A, aStart, lda, x, xStart, y, yStart);
            } else {
                zhemvLowerStride(n, alphaRe, alphaIm, A, aStart, lda, x, xStart, incX, y, yStart, incY);
            }
        }
    }

    static void zhemvUpperUnitStride(int n, double alphaRe, double alphaIm,
                                     double[] A, int aStart, int lda,
                                     double[] x, int xStart,
                                     double[] y, int yStart) {
        for (int i = 0; i < n; i++) {
            double xiRe = x[xStart + i * 2];
            double xiIm = x[xStart + i * 2 + 1];
            double temp1Re = alphaRe * xiRe - alphaIm * xiIm;
            double temp1Im = alphaRe * xiIm + alphaIm * xiRe;
            int rowOff = aStart + i * lda * 2;

            double diagRe = A[rowOff + i * 2];
            double sumRe = diagRe * xiRe;
            double sumIm = diagRe * xiIm;

            int j = i + 1;
            int j2 = i + 1 + ((n - i - 1) / 2) * 2;

            for (; j < j2; j += 2) {
                double a0Re = A[rowOff + j * 2];
                double a0Im = A[rowOff + j * 2 + 1];
                double a1Re = A[rowOff + (j + 1) * 2];
                double a1Im = A[rowOff + (j + 1) * 2 + 1];

                double x0Re = x[xStart + j * 2];
                double x0Im = x[xStart + j * 2 + 1];
                double x1Re = x[xStart + (j + 1) * 2];
                double x1Im = x[xStart + (j + 1) * 2 + 1];

                sumRe = Math.fma(a0Re, x0Re, Math.fma(-a0Im, x0Im, sumRe));
                sumIm = Math.fma(a0Re, x0Im, Math.fma(a0Im, x0Re, sumIm));
                sumRe = Math.fma(a1Re, x1Re, Math.fma(-a1Im, x1Im, sumRe));
                sumIm = Math.fma(a1Re, x1Im, Math.fma(a1Im, x1Re, sumIm));

                int y0 = yStart + j * 2;
                int y1 = yStart + (j + 1) * 2;
                y[y0]     = Math.fma(temp1Re, a0Re, Math.fma(temp1Im, a0Im, y[y0]));
                y[y0 + 1] = Math.fma(temp1Im, a0Re, Math.fma(-temp1Re, a0Im, y[y0 + 1]));
                y[y1]     = Math.fma(temp1Re, a1Re, Math.fma(temp1Im, a1Im, y[y1]));
                y[y1 + 1] = Math.fma(temp1Im, a1Re, Math.fma(-temp1Re, a1Im, y[y1 + 1]));
            }

            for (; j < n; j++) {
                double aRe = A[rowOff + j * 2];
                double aIm = A[rowOff + j * 2 + 1];
                double xjRe = x[xStart + j * 2];
                double xjIm = x[xStart + j * 2 + 1];
                sumRe = Math.fma(aRe, xjRe, Math.fma(-aIm, xjIm, sumRe));
                sumIm = Math.fma(aRe, xjIm, Math.fma(aIm, xjRe, sumIm));
                int yjOff = yStart + j * 2;
                y[yjOff]     = Math.fma(temp1Re, aRe, Math.fma(temp1Im, aIm, y[yjOff]));
                y[yjOff + 1] = Math.fma(temp1Im, aRe, Math.fma(-temp1Re, aIm, y[yjOff + 1]));
            }

            int yiOff = yStart + i * 2;
            y[yiOff]     = Math.fma(alphaRe, sumRe, Math.fma(-alphaIm, sumIm, y[yiOff]));
            y[yiOff + 1] = Math.fma(alphaRe, sumIm, Math.fma(alphaIm, sumRe, y[yiOff + 1]));
        }
    }

    static void zhemvLowerUnitStride(int n, double alphaRe, double alphaIm,
                                     double[] A, int aStart, int lda,
                                     double[] x, int xStart,
                                     double[] y, int yStart) {
        for (int i = 0; i < n; i++) {
            double xiRe = x[xStart + i * 2];
            double xiIm = x[xStart + i * 2 + 1];
            double temp1Re = alphaRe * xiRe - alphaIm * xiIm;
            double temp1Im = alphaRe * xiIm + alphaIm * xiRe;
            int rowOff = aStart + i * lda * 2;

            double sumRe = 0.0;
            double sumIm = 0.0;

            int j = 0;
            int j2 = (i / 2) * 2;

            for (; j < j2; j += 2) {
                double a0Re = A[rowOff + j * 2];
                double a0Im = A[rowOff + j * 2 + 1];
                double a1Re = A[rowOff + (j + 1) * 2];
                double a1Im = A[rowOff + (j + 1) * 2 + 1];

                double x0Re = x[xStart + j * 2];
                double x0Im = x[xStart + j * 2 + 1];
                double x1Re = x[xStart + (j + 1) * 2];
                double x1Im = x[xStart + (j + 1) * 2 + 1];

                sumRe = Math.fma(a0Re, x0Re, Math.fma(-a0Im, x0Im, sumRe));
                sumIm = Math.fma(a0Re, x0Im, Math.fma(a0Im, x0Re, sumIm));
                sumRe = Math.fma(a1Re, x1Re, Math.fma(-a1Im, x1Im, sumRe));
                sumIm = Math.fma(a1Re, x1Im, Math.fma(a1Im, x1Re, sumIm));

                int y0 = yStart + j * 2;
                int y1 = yStart + (j + 1) * 2;
                y[y0]     = Math.fma(temp1Re, a0Re, Math.fma(temp1Im, a0Im, y[y0]));
                y[y0 + 1] = Math.fma(temp1Im, a0Re, Math.fma(-temp1Re, a0Im, y[y0 + 1]));
                y[y1]     = Math.fma(temp1Re, a1Re, Math.fma(temp1Im, a1Im, y[y1]));
                y[y1 + 1] = Math.fma(temp1Im, a1Re, Math.fma(-temp1Re, a1Im, y[y1 + 1]));
            }

            for (; j < i; j++) {
                double aRe = A[rowOff + j * 2];
                double aIm = A[rowOff + j * 2 + 1];
                double xjRe = x[xStart + j * 2];
                double xjIm = x[xStart + j * 2 + 1];
                sumRe = Math.fma(aRe, xjRe, Math.fma(-aIm, xjIm, sumRe));
                sumIm = Math.fma(aRe, xjIm, Math.fma(aIm, xjRe, sumIm));
                int yjOff = yStart + j * 2;
                y[yjOff]     = Math.fma(temp1Re, aRe, Math.fma(temp1Im, aIm, y[yjOff]));
                y[yjOff + 1] = Math.fma(temp1Im, aRe, Math.fma(-temp1Re, aIm, y[yjOff + 1]));
            }

            double diagRe = A[rowOff + i * 2];
            sumRe = Math.fma(diagRe, xiRe, sumRe);
            sumIm = Math.fma(diagRe, xiIm, sumIm);

            int yiOff = yStart + i * 2;
            y[yiOff]     = Math.fma(alphaRe, sumRe, Math.fma(-alphaIm, sumIm, y[yiOff]));
            y[yiOff + 1] = Math.fma(alphaRe, sumIm, Math.fma(alphaIm, sumRe, y[yiOff + 1]));
        }
    }

    static void zhemvUpperStride(int n, double alphaRe, double alphaIm,
                                 double[] A, int aStart, int lda,
                                 double[] x, int xStart, int incX,
                                 double[] y, int yStart, int incY) {
        int xStep = incX * 2;
        int yStep = incY * 2;
        for (int i = 0; i < n; i++) {
            int xiOff = xStart + i * xStep;
            double xiRe = x[xiOff];
            double xiIm = x[xiOff + 1];
            double temp1Re = alphaRe * xiRe - alphaIm * xiIm;
            double temp1Im = alphaRe * xiIm + alphaIm * xiRe;
            int rowOff = aStart + i * lda * 2;

            double diagRe = A[rowOff + i * 2];
            double s0Re = diagRe * xiRe;
            double s0Im = diagRe * xiIm;
            double s1Re = 0.0;
            double s1Im = 0.0;

            int j = i + 1;
            int j2 = i + 1 + ((n - i - 1) / 2) * 2;
            int xjOff = xStart + j * xStep;
            int yjOff = yStart + j * yStep;

            for (; j < j2; j += 2) {
                int aPos = rowOff + j * 2;
                double a0Re = A[aPos];
                double a0Im = A[aPos + 1];
                double a1Re = A[aPos + 2];
                double a1Im = A[aPos + 3];

                double x0Re = x[xjOff];
                double x0Im = x[xjOff + 1];
                double x1Re = x[xjOff + xStep];
                double x1Im = x[xjOff + xStep + 1];

                s0Re = Math.fma(a0Re, x0Re, Math.fma(-a0Im, x0Im, s0Re));
                s0Im = Math.fma(a0Re, x0Im, Math.fma(a0Im, x0Re, s0Im));
                s1Re = Math.fma(a1Re, x1Re, Math.fma(-a1Im, x1Im, s1Re));
                s1Im = Math.fma(a1Re, x1Im, Math.fma(a1Im, x1Re, s1Im));

                updateHemvStride(y, yjOff, temp1Re, temp1Im, a0Re, a0Im);
                updateHemvStride(y, yjOff + yStep, temp1Re, temp1Im, a1Re, a1Im);

                xjOff += xStep * 2;
                yjOff += yStep * 2;
            }

            double sumRe = s0Re + s1Re;
            double sumIm = s0Im + s1Im;

            for (; j < n; j++) {
                double aRe = A[rowOff + j * 2];
                double aIm = A[rowOff + j * 2 + 1];
                double xjRe = x[xjOff];
                double xjIm = x[xjOff + 1];
                sumRe = Math.fma(aRe, xjRe, Math.fma(-aIm, xjIm, sumRe));
                sumIm = Math.fma(aRe, xjIm, Math.fma(aIm, xjRe, sumIm));
                updateHemvStride(y, yjOff, temp1Re, temp1Im, aRe, aIm);
                xjOff += xStep;
                yjOff += yStep;
            }

            int yiOff = yStart + i * yStep;
            y[yiOff]     = Math.fma(alphaRe, sumRe, Math.fma(-alphaIm, sumIm, y[yiOff]));
            y[yiOff + 1] = Math.fma(alphaRe, sumIm, Math.fma(alphaIm, sumRe, y[yiOff + 1]));
        }
    }

    static void zhemvLowerStride(int n, double alphaRe, double alphaIm,
                                 double[] A, int aStart, int lda,
                                 double[] x, int xStart, int incX,
                                 double[] y, int yStart, int incY) {
        int xStep = incX * 2;
        int yStep = incY * 2;
        for (int i = 0; i < n; i++) {
            int xiOff = xStart + i * xStep;
            double xiRe = x[xiOff];
            double xiIm = x[xiOff + 1];
            double temp1Re = alphaRe * xiRe - alphaIm * xiIm;
            double temp1Im = alphaRe * xiIm + alphaIm * xiRe;
            int rowOff = aStart + i * lda * 2;

            double s0Re = 0.0;
            double s0Im = 0.0;
            double s1Re = 0.0;
            double s1Im = 0.0;

            int j = 0;
            int j2 = (i / 2) * 2;
            int xjOff = xStart;
            int yjOff = yStart;

            for (; j < j2; j += 2) {
                int aPos = rowOff + j * 2;
                double a0Re = A[aPos];
                double a0Im = A[aPos + 1];
                double a1Re = A[aPos + 2];
                double a1Im = A[aPos + 3];

                double x0Re = x[xjOff];
                double x0Im = x[xjOff + 1];
                double x1Re = x[xjOff + xStep];
                double x1Im = x[xjOff + xStep + 1];

                s0Re = Math.fma(a0Re, x0Re, Math.fma(-a0Im, x0Im, s0Re));
                s0Im = Math.fma(a0Re, x0Im, Math.fma(a0Im, x0Re, s0Im));
                s1Re = Math.fma(a1Re, x1Re, Math.fma(-a1Im, x1Im, s1Re));
                s1Im = Math.fma(a1Re, x1Im, Math.fma(a1Im, x1Re, s1Im));

                updateHemvStride(y, yjOff, temp1Re, temp1Im, a0Re, a0Im);
                updateHemvStride(y, yjOff + yStep, temp1Re, temp1Im, a1Re, a1Im);

                xjOff += xStep * 2;
                yjOff += yStep * 2;
            }

            double sumRe = s0Re + s1Re;
            double sumIm = s0Im + s1Im;

            for (; j < i; j++) {
                double aRe = A[rowOff + j * 2];
                double aIm = A[rowOff + j * 2 + 1];
                double xjRe = x[xjOff];
                double xjIm = x[xjOff + 1];
                sumRe = Math.fma(aRe, xjRe, Math.fma(-aIm, xjIm, sumRe));
                sumIm = Math.fma(aRe, xjIm, Math.fma(aIm, xjRe, sumIm));
                updateHemvStride(y, yjOff, temp1Re, temp1Im, aRe, aIm);
                xjOff += xStep;
                yjOff += yStep;
            }

            double diagRe = A[rowOff + i * 2];
            sumRe = Math.fma(diagRe, xiRe, sumRe);
            sumIm = Math.fma(diagRe, xiIm, sumIm);

            int yiOff = yStart + i * yStep;
            y[yiOff]     = Math.fma(alphaRe, sumRe, Math.fma(-alphaIm, sumIm, y[yiOff]));
            y[yiOff + 1] = Math.fma(alphaRe, sumIm, Math.fma(alphaIm, sumRe, y[yiOff + 1]));
        }
    }

    static void updateHemvStride(double[] y, int yOff, double temp1Re, double temp1Im, double aRe, double aIm) {
        y[yOff] = Math.fma(temp1Re, aRe, Math.fma(temp1Im, aIm, y[yOff]));
        y[yOff + 1] = Math.fma(temp1Im, aRe, Math.fma(-temp1Re, aIm, y[yOff + 1]));
    }
}
