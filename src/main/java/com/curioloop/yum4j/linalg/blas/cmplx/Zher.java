package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
interface Zher {

    public static void zher(BLAS.Uplo uplo, int n, double alpha,
                           double[] x, int xStart, int incX,
                           double[] A, int aStart, int lda) {
        if (n <= 0) return;
        if (alpha == 0.0) return;

        boolean upper = (uplo == BLAS.Uplo.Upper);

        if (upper) {
            for (int i = 0; i < n; i++) {
                int xiOff = xStart + i * incX * 2;
                double xiRe = x[xiOff], xiIm = x[xiOff + 1];
                if (xiRe == 0.0 && xiIm == 0.0) continue;

                double axiRe = alpha * xiRe;
                double axiIm = alpha * xiIm;

                int aRow = aStart + i * lda * 2;

                int aPos = aRow + i * 2;
                A[aPos] = Math.fma(alpha, xiRe * xiRe + xiIm * xiIm, A[aPos]);

                int j = i + 1;
                if (incX == 1) {
                    int j2 = i + 1 + ((n - i - 1) / 2) * 2;
                    for (; j < j2; j += 2) {
                        double x0Re = x[xStart + j * 2], x0Im = x[xStart + j * 2 + 1];
                        double x1Re = x[xStart + (j + 1) * 2], x1Im = x[xStart + (j + 1) * 2 + 1];

                        int p0 = aRow + j * 2;
                        int p1 = aRow + (j + 1) * 2;
                        A[p0] = Math.fma(axiRe, x0Re, Math.fma(axiIm, x0Im, A[p0]));
                        A[p0 + 1] = Math.fma(axiIm, x0Re, Math.fma(-axiRe, x0Im, A[p0 + 1]));
                        A[p1] = Math.fma(axiRe, x1Re, Math.fma(axiIm, x1Im, A[p1]));
                        A[p1 + 1] = Math.fma(axiIm, x1Re, Math.fma(-axiRe, x1Im, A[p1 + 1]));
                    }
                }
                for (; j < n; j++) {
                    int xjOff = xStart + j * incX * 2;
                    double xjRe = x[xjOff], xjIm = x[xjOff + 1];
                    int aPos2 = aRow + j * 2;
                    A[aPos2] = Math.fma(axiRe, xjRe, Math.fma(axiIm, xjIm, A[aPos2]));
                    A[aPos2 + 1] = Math.fma(axiIm, xjRe, Math.fma(-axiRe, xjIm, A[aPos2 + 1]));
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int xiOff = xStart + i * incX * 2;
                double xiRe = x[xiOff], xiIm = x[xiOff + 1];
                if (xiRe == 0.0 && xiIm == 0.0) continue;

                double axiRe = alpha * xiRe;
                double axiIm = alpha * xiIm;

                int aRow = aStart + i * lda * 2;

                int j = 0;
                if (incX == 1) {
                    int j2 = ((i) / 2) * 2;
                    for (; j < j2; j += 2) {
                        double x0Re = x[xStart + j * 2], x0Im = x[xStart + j * 2 + 1];
                        double x1Re = x[xStart + (j + 1) * 2], x1Im = x[xStart + (j + 1) * 2 + 1];

                        int p0 = aRow + j * 2;
                        int p1 = aRow + (j + 1) * 2;
                        A[p0] = Math.fma(axiRe, x0Re, Math.fma(axiIm, x0Im, A[p0]));
                        A[p0 + 1] = Math.fma(axiIm, x0Re, Math.fma(-axiRe, x0Im, A[p0 + 1]));
                        A[p1] = Math.fma(axiRe, x1Re, Math.fma(axiIm, x1Im, A[p1]));
                        A[p1 + 1] = Math.fma(axiIm, x1Re, Math.fma(-axiRe, x1Im, A[p1 + 1]));
                    }
                }
                for (; j < i; j++) {
                    int xjOff = xStart + j * incX * 2;
                    double xjRe = x[xjOff], xjIm = x[xjOff + 1];
                    int aPos2 = aRow + j * 2;
                    A[aPos2] = Math.fma(axiRe, xjRe, Math.fma(axiIm, xjIm, A[aPos2]));
                    A[aPos2 + 1] = Math.fma(axiIm, xjRe, Math.fma(-axiRe, xjIm, A[aPos2 + 1]));
                }

                int aPos = aRow + i * 2;
                A[aPos] = Math.fma(alpha, xiRe * xiRe + xiIm * xiIm, A[aPos]);
            }
        }
    }
}
