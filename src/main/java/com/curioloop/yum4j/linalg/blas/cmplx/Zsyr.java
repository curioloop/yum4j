package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
/**
 * ZSYR performs the rank 1 operation
 * A := alpha*x*x^T + A
 * where alpha is a scalar, x is an n element vector
 * and A is an n by n symmetric matrix.
 */
interface Zsyr {

    /**
     * Performs the rank 1 operation
     * A := alpha*x*x^T + A
     *
     * @param uplo   Specifies whether the upper or lower triangular part of the symmetric matrix A is stored:
     *               BLAS.Uplo.Upper: Upper triangular part is stored
     *               BLAS.Uplo.Lower: Lower triangular part is stored
     * @param n      Order of the matrix A
     * @param alphaRe Real part of the scalar alpha
     * @param alphaIm Imaginary part of the scalar alpha
     * @param x      Complex vector x stored in interleaved format [re, im, re, im, ...]
     * @param xStart Starting index of x
     * @param incX   Increment for the elements of x
     * @param A      Complex symmetric matrix A stored in row-major order, interleaved format [re, im, re, im, ...]
     * @param aStart Starting index of A
     * @param lda    Leading dimension of the matrix A (must be at least max(1, n))
     */
    public static void zsyr(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm,
                           double[] x, int xStart, int incX,
                           double[] A, int aStart, int lda) {
        if (n <= 0) return;
        if (alphaRe == 0.0 && alphaIm == 0.0) return;

        boolean upper = (uplo == BLAS.Uplo.Upper);

        if (upper) {
            for (int i = 0; i < n; i++) {
                int xiOff = xStart + i * incX * 2;
                double xiRe = x[xiOff], xiIm = x[xiOff + 1];
                if (xiRe == 0.0 && xiIm == 0.0) continue;

                double axiRe = alphaRe * xiRe - alphaIm * xiIm;
                double axiIm = alphaRe * xiIm + alphaIm * xiRe;

                int aRow = aStart + i * lda * 2;

                int j = i;
                if (incX == 1) {
                    int j2 = i + ((n - i) / 2) * 2;
                    for (; j < j2; j += 2) {
                        double x0Re = x[xStart + j * 2], x0Im = x[xStart + j * 2 + 1];
                        double x1Re = x[xStart + (j + 1) * 2], x1Im = x[xStart + (j + 1) * 2 + 1];

                        int p0 = aRow + j * 2;
                        int p1 = aRow + (j + 1) * 2;
                        A[p0]     = Math.fma(axiRe, x0Re, Math.fma(-axiIm, x0Im, A[p0]));
                        A[p0 + 1] = Math.fma(axiRe, x0Im, Math.fma(axiIm, x0Re, A[p0 + 1]));
                        A[p1]     = Math.fma(axiRe, x1Re, Math.fma(-axiIm, x1Im, A[p1]));
                        A[p1 + 1] = Math.fma(axiRe, x1Im, Math.fma(axiIm, x1Re, A[p1 + 1]));
                    }
                }
                for (; j < n; j++) {
                    int xjOff = xStart + j * incX * 2;
                    double xjRe = x[xjOff], xjIm = x[xjOff + 1];
                    int aPos = aRow + j * 2;
                    A[aPos]     = Math.fma(axiRe, xjRe, Math.fma(-axiIm, xjIm, A[aPos]));
                    A[aPos + 1] = Math.fma(axiRe, xjIm, Math.fma(axiIm, xjRe, A[aPos + 1]));
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int xiOff = xStart + i * incX * 2;
                double xiRe = x[xiOff], xiIm = x[xiOff + 1];
                if (xiRe == 0.0 && xiIm == 0.0) continue;

                double axiRe = alphaRe * xiRe - alphaIm * xiIm;
                double axiIm = alphaRe * xiIm + alphaIm * xiRe;

                int aRow = aStart + i * lda * 2;

                int j = 0;
                if (incX == 1) {
                    int j2 = (i + 1) / 2 * 2;
                    for (; j < j2; j += 2) {
                        double x0Re = x[xStart + j * 2], x0Im = x[xStart + j * 2 + 1];
                        double x1Re = x[xStart + (j + 1) * 2], x1Im = x[xStart + (j + 1) * 2 + 1];

                        int p0 = aRow + j * 2;
                        int p1 = aRow + (j + 1) * 2;
                        A[p0]     = Math.fma(axiRe, x0Re, Math.fma(-axiIm, x0Im, A[p0]));
                        A[p0 + 1] = Math.fma(axiRe, x0Im, Math.fma(axiIm, x0Re, A[p0 + 1]));
                        A[p1]     = Math.fma(axiRe, x1Re, Math.fma(-axiIm, x1Im, A[p1]));
                        A[p1 + 1] = Math.fma(axiRe, x1Im, Math.fma(axiIm, x1Re, A[p1 + 1]));
                    }
                }
                for (; j <= i; j++) {
                    int xjOff = xStart + j * incX * 2;
                    double xjRe = x[xjOff], xjIm = x[xjOff + 1];
                    int aPos = aRow + j * 2;
                    A[aPos]     = Math.fma(axiRe, xjRe, Math.fma(-axiIm, xjIm, A[aPos]));
                    A[aPos + 1] = Math.fma(axiRe, xjIm, Math.fma(axiIm, xjRe, A[aPos + 1]));
                }
            }
        }
    }
}
