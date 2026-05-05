package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
/**
 * ZSYR2 performs the rank 2 operation
 * A := alpha*x*y^T + alpha*y*x^T + A
 * where alpha is a scalar, x and y are n element vectors
 * and A is an n by n symmetric matrix.
 */
interface Zsyr2 {

    /**
     * Performs the rank 2 operation
     * A := alpha*x*y^T + alpha*y*x^T + A
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
     * @param y      Complex vector y stored in interleaved format [re, im, re, im, ...]
     * @param yStart Starting index of y
     * @param incY   Increment for the elements of y
     * @param A      Complex symmetric matrix A stored in row-major order, interleaved format [re, im, re, im, ...]
     * @param aStart Starting index of A
     * @param lda    Leading dimension of the matrix A (must be at least max(1, n))
     */
    public static void zsyr2(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm,
                            double[] x, int xStart, int incX,
                            double[] y, int yStart, int incY,
                            double[] A, int aStart, int lda) {
        if (n <= 0) return;
        if (alphaRe == 0.0 && alphaIm == 0.0) return;

        boolean upper = (uplo == BLAS.Uplo.Upper);

        if (upper) {
            for (int i = 0; i < n; i++) {
                int xiOff = xStart + i * incX * 2;
                int yiOff = yStart + i * incY * 2;
                double xiRe = x[xiOff], xiIm = x[xiOff + 1];
                double yiRe = y[yiOff], yiIm = y[yiOff + 1];
                if (xiRe == 0.0 && xiIm == 0.0 && yiRe == 0.0 && yiIm == 0.0) continue;

                double axiRe = alphaRe * xiRe - alphaIm * xiIm;
                double axiIm = alphaRe * xiIm + alphaIm * xiRe;
                double ayiRe = alphaRe * yiRe - alphaIm * yiIm;
                double ayiIm = alphaRe * yiIm + alphaIm * yiRe;

                int aRow = aStart + i * lda * 2;

                int j = i;
                if (incX == 1 && incY == 1) {
                    int j4 = i + ((n - i) / 4) * 4;
                    for (; j < j4; j += 4) {
                        double x0Re = x[xStart + j * 2], x0Im = x[xStart + j * 2 + 1];
                        double x1Re = x[xStart + (j + 1) * 2], x1Im = x[xStart + (j + 1) * 2 + 1];
                        double x2Re = x[xStart + (j + 2) * 2], x2Im = x[xStart + (j + 2) * 2 + 1];
                        double x3Re = x[xStart + (j + 3) * 2], x3Im = x[xStart + (j + 3) * 2 + 1];
                        double y0Re = y[yStart + j * 2], y0Im = y[yStart + j * 2 + 1];
                        double y1Re = y[yStart + (j + 1) * 2], y1Im = y[yStart + (j + 1) * 2 + 1];
                        double y2Re = y[yStart + (j + 2) * 2], y2Im = y[yStart + (j + 2) * 2 + 1];
                        double y3Re = y[yStart + (j + 3) * 2], y3Im = y[yStart + (j + 3) * 2 + 1];

                        int p0 = aRow + j * 2;
                        int p1 = aRow + (j + 1) * 2;
                        int p2 = aRow + (j + 2) * 2;
                        int p3 = aRow + (j + 3) * 2;

                        double t1Re = axiRe * y0Re - axiIm * y0Im;
                        double t1Im = axiRe * y0Im + axiIm * y0Re;
                        double t2Re = ayiRe * x0Re - ayiIm * x0Im;
                        double t2Im = ayiRe * x0Im + ayiIm * x0Re;
                        A[p0]     += t1Re + t2Re;
                        A[p0 + 1] += t1Im + t2Im;

                        t1Re = axiRe * y1Re - axiIm * y1Im;
                        t1Im = axiRe * y1Im + axiIm * y1Re;
                        t2Re = ayiRe * x1Re - ayiIm * x1Im;
                        t2Im = ayiRe * x1Im + ayiIm * x1Re;
                        A[p1]     += t1Re + t2Re;
                        A[p1 + 1] += t1Im + t2Im;

                        t1Re = axiRe * y2Re - axiIm * y2Im;
                        t1Im = axiRe * y2Im + axiIm * y2Re;
                        t2Re = ayiRe * x2Re - ayiIm * x2Im;
                        t2Im = ayiRe * x2Im + ayiIm * x2Re;
                        A[p2]     += t1Re + t2Re;
                        A[p2 + 1] += t1Im + t2Im;

                        t1Re = axiRe * y3Re - axiIm * y3Im;
                        t1Im = axiRe * y3Im + axiIm * y3Re;
                        t2Re = ayiRe * x3Re - ayiIm * x3Im;
                        t2Im = ayiRe * x3Im + ayiIm * x3Re;
                        A[p3]     += t1Re + t2Re;
                        A[p3 + 1] += t1Im + t2Im;
                    }
                }
                for (; j < n; j++) {
                    int xjOff = xStart + j * incX * 2;
                    int yjOff = yStart + j * incY * 2;
                    double xjRe = x[xjOff], xjIm = x[xjOff + 1];
                    double yjRe = y[yjOff], yjIm = y[yjOff + 1];
                    int aPos = aRow + j * 2;

                    double t1Re = axiRe * yjRe - axiIm * yjIm;
                    double t1Im = axiRe * yjIm + axiIm * yjRe;
                    double t2Re = ayiRe * xjRe - ayiIm * xjIm;
                    double t2Im = ayiRe * xjIm + ayiIm * xjRe;

                    A[aPos]     += t1Re + t2Re;
                    A[aPos + 1] += t1Im + t2Im;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int xiOff = xStart + i * incX * 2;
                int yiOff = yStart + i * incY * 2;
                double xiRe = x[xiOff], xiIm = x[xiOff + 1];
                double yiRe = y[yiOff], yiIm = y[yiOff + 1];
                if (xiRe == 0.0 && xiIm == 0.0 && yiRe == 0.0 && yiIm == 0.0) continue;

                double axiRe = alphaRe * xiRe - alphaIm * xiIm;
                double axiIm = alphaRe * xiIm + alphaIm * xiRe;
                double ayiRe = alphaRe * yiRe - alphaIm * yiIm;
                double ayiIm = alphaRe * yiIm + alphaIm * yiRe;

                int aRow = aStart + i * lda * 2;

                int j = 0;
                if (incX == 1 && incY == 1) {
                    int j4 = (i + 1) / 4 * 4;
                    for (; j < j4; j += 4) {
                        double x0Re = x[xStart + j * 2], x0Im = x[xStart + j * 2 + 1];
                        double x1Re = x[xStart + (j + 1) * 2], x1Im = x[xStart + (j + 1) * 2 + 1];
                        double x2Re = x[xStart + (j + 2) * 2], x2Im = x[xStart + (j + 2) * 2 + 1];
                        double x3Re = x[xStart + (j + 3) * 2], x3Im = x[xStart + (j + 3) * 2 + 1];
                        double y0Re = y[yStart + j * 2], y0Im = y[yStart + j * 2 + 1];
                        double y1Re = y[yStart + (j + 1) * 2], y1Im = y[yStart + (j + 1) * 2 + 1];
                        double y2Re = y[yStart + (j + 2) * 2], y2Im = y[yStart + (j + 2) * 2 + 1];
                        double y3Re = y[yStart + (j + 3) * 2], y3Im = y[yStart + (j + 3) * 2 + 1];

                        int p0 = aRow + j * 2;
                        int p1 = aRow + (j + 1) * 2;
                        int p2 = aRow + (j + 2) * 2;
                        int p3 = aRow + (j + 3) * 2;

                        double t1Re = axiRe * y0Re - axiIm * y0Im;
                        double t1Im = axiRe * y0Im + axiIm * y0Re;
                        double t2Re = ayiRe * x0Re - ayiIm * x0Im;
                        double t2Im = ayiRe * x0Im + ayiIm * x0Re;
                        A[p0]     += t1Re + t2Re;
                        A[p0 + 1] += t1Im + t2Im;

                        t1Re = axiRe * y1Re - axiIm * y1Im;
                        t1Im = axiRe * y1Im + axiIm * y1Re;
                        t2Re = ayiRe * x1Re - ayiIm * x1Im;
                        t2Im = ayiRe * x1Im + ayiIm * x1Re;
                        A[p1]     += t1Re + t2Re;
                        A[p1 + 1] += t1Im + t2Im;

                        t1Re = axiRe * y2Re - axiIm * y2Im;
                        t1Im = axiRe * y2Im + axiIm * y2Re;
                        t2Re = ayiRe * x2Re - ayiIm * x2Im;
                        t2Im = ayiRe * x2Im + ayiIm * x2Re;
                        A[p2]     += t1Re + t2Re;
                        A[p2 + 1] += t1Im + t2Im;

                        t1Re = axiRe * y3Re - axiIm * y3Im;
                        t1Im = axiRe * y3Im + axiIm * y3Re;
                        t2Re = ayiRe * x3Re - ayiIm * x3Im;
                        t2Im = ayiRe * x3Im + ayiIm * x3Re;
                        A[p3]     += t1Re + t2Re;
                        A[p3 + 1] += t1Im + t2Im;
                    }
                }
                for (; j <= i; j++) {
                    int xjOff = xStart + j * incX * 2;
                    int yjOff = yStart + j * incY * 2;
                    double xjRe = x[xjOff], xjIm = x[xjOff + 1];
                    double yjRe = y[yjOff], yjIm = y[yjOff + 1];
                    int aPos = aRow + j * 2;

                    double t1Re = axiRe * yjRe - axiIm * yjIm;
                    double t1Im = axiRe * yjIm + axiIm * yjRe;
                    double t2Re = ayiRe * xjRe - ayiIm * xjIm;
                    double t2Im = ayiRe * xjIm + ayiIm * xjRe;

                    A[aPos]     += t1Re + t2Re;
                    A[aPos + 1] += t1Im + t2Im;
                }
            }
        }
    }
}
