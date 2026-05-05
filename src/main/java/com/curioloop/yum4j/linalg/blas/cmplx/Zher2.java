package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
interface Zher2 {

    public static void zher2(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm,
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
                double cayiRe = alphaRe * yiRe + alphaIm * yiIm;
                double cayiIm = alphaRe * yiIm - alphaIm * yiRe;

                int aRow = aStart + i * lda * 2;

                int aPos = aRow + i * 2;
                double diagRe = (axiRe * yiRe + axiIm * yiIm) + (cayiRe * xiRe + cayiIm * xiIm);
                A[aPos] += diagRe;

                int j = i + 1;
                if (incX == 1 && incY == 1) {
                    int j4 = i + 1 + ((n - i - 1) / 4) * 4;
                    for (; j < j4; j += 4) {
                        double y0Re = y[yStart + j * 2], y0Im = y[yStart + j * 2 + 1];
                        double y1Re = y[yStart + (j + 1) * 2], y1Im = y[yStart + (j + 1) * 2 + 1];
                        double y2Re = y[yStart + (j + 2) * 2], y2Im = y[yStart + (j + 2) * 2 + 1];
                        double y3Re = y[yStart + (j + 3) * 2], y3Im = y[yStart + (j + 3) * 2 + 1];
                        double x0Re = x[xStart + j * 2], x0Im = x[xStart + j * 2 + 1];
                        double x1Re = x[xStart + (j + 1) * 2], x1Im = x[xStart + (j + 1) * 2 + 1];
                        double x2Re = x[xStart + (j + 2) * 2], x2Im = x[xStart + (j + 2) * 2 + 1];
                        double x3Re = x[xStart + (j + 3) * 2], x3Im = x[xStart + (j + 3) * 2 + 1];

                        int p0 = aRow + j * 2;
                        int p1 = aRow + (j + 1) * 2;
                        int p2 = aRow + (j + 2) * 2;
                        int p3 = aRow + (j + 3) * 2;

                        double t1Re0 = axiRe * y0Re + axiIm * y0Im;
                        double t1Im0 = axiIm * y0Re - axiRe * y0Im;
                        double t2Re0 = cayiRe * x0Re + cayiIm * x0Im;
                        double t2Im0 = cayiIm * x0Re - cayiRe * x0Im;
                        A[p0] += t1Re0 + t2Re0;
                        A[p0 + 1] += t1Im0 + t2Im0;

                        double t1Re1 = axiRe * y1Re + axiIm * y1Im;
                        double t1Im1 = axiIm * y1Re - axiRe * y1Im;
                        double t2Re1 = cayiRe * x1Re + cayiIm * x1Im;
                        double t2Im1 = cayiIm * x1Re - cayiRe * x1Im;
                        A[p1] += t1Re1 + t2Re1;
                        A[p1 + 1] += t1Im1 + t2Im1;

                        double t1Re2 = axiRe * y2Re + axiIm * y2Im;
                        double t1Im2 = axiIm * y2Re - axiRe * y2Im;
                        double t2Re2 = cayiRe * x2Re + cayiIm * x2Im;
                        double t2Im2 = cayiIm * x2Re - cayiRe * x2Im;
                        A[p2] += t1Re2 + t2Re2;
                        A[p2 + 1] += t1Im2 + t2Im2;

                        double t1Re3 = axiRe * y3Re + axiIm * y3Im;
                        double t1Im3 = axiIm * y3Re - axiRe * y3Im;
                        double t2Re3 = cayiRe * x3Re + cayiIm * x3Im;
                        double t2Im3 = cayiIm * x3Re - cayiRe * x3Im;
                        A[p3] += t1Re3 + t2Re3;
                        A[p3 + 1] += t1Im3 + t2Im3;
                    }
                }
                for (; j < n; j++) {
                    int xjOff = xStart + j * incX * 2;
                    int yjOff = yStart + j * incY * 2;
                    double xjRe = x[xjOff], xjIm = x[xjOff + 1];
                    double yjRe = y[yjOff], yjIm = y[yjOff + 1];
                    int aPos2 = aRow + j * 2;

                    double t1Re = axiRe * yjRe + axiIm * yjIm;
                    double t1Im = axiIm * yjRe - axiRe * yjIm;
                    double t2Re = cayiRe * xjRe + cayiIm * xjIm;
                    double t2Im = cayiIm * xjRe - cayiRe * xjIm;

                    A[aPos2] += t1Re + t2Re;
                    A[aPos2 + 1] += t1Im + t2Im;
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
                double cayiRe = alphaRe * yiRe + alphaIm * yiIm;
                double cayiIm = alphaRe * yiIm - alphaIm * yiRe;

                int aRow = aStart + i * lda * 2;

                int j = 0;
                if (incX == 1 && incY == 1) {
                    int j4 = (i / 4) * 4;
                    for (; j < j4; j += 4) {
                        double y0Re = y[yStart + j * 2], y0Im = y[yStart + j * 2 + 1];
                        double y1Re = y[yStart + (j + 1) * 2], y1Im = y[yStart + (j + 1) * 2 + 1];
                        double y2Re = y[yStart + (j + 2) * 2], y2Im = y[yStart + (j + 2) * 2 + 1];
                        double y3Re = y[yStart + (j + 3) * 2], y3Im = y[yStart + (j + 3) * 2 + 1];
                        double x0Re = x[xStart + j * 2], x0Im = x[xStart + j * 2 + 1];
                        double x1Re = x[xStart + (j + 1) * 2], x1Im = x[xStart + (j + 1) * 2 + 1];
                        double x2Re = x[xStart + (j + 2) * 2], x2Im = x[xStart + (j + 2) * 2 + 1];
                        double x3Re = x[xStart + (j + 3) * 2], x3Im = x[xStart + (j + 3) * 2 + 1];

                        int p0 = aRow + j * 2;
                        int p1 = aRow + (j + 1) * 2;
                        int p2 = aRow + (j + 2) * 2;
                        int p3 = aRow + (j + 3) * 2;

                        double t1Re0 = axiRe * y0Re + axiIm * y0Im;
                        double t1Im0 = axiIm * y0Re - axiRe * y0Im;
                        double t2Re0 = cayiRe * x0Re + cayiIm * x0Im;
                        double t2Im0 = cayiIm * x0Re - cayiRe * x0Im;
                        A[p0] += t1Re0 + t2Re0;
                        A[p0 + 1] += t1Im0 + t2Im0;

                        double t1Re1 = axiRe * y1Re + axiIm * y1Im;
                        double t1Im1 = axiIm * y1Re - axiRe * y1Im;
                        double t2Re1 = cayiRe * x1Re + cayiIm * x1Im;
                        double t2Im1 = cayiIm * x1Re - cayiRe * x1Im;
                        A[p1] += t1Re1 + t2Re1;
                        A[p1 + 1] += t1Im1 + t2Im1;

                        double t1Re2 = axiRe * y2Re + axiIm * y2Im;
                        double t1Im2 = axiIm * y2Re - axiRe * y2Im;
                        double t2Re2 = cayiRe * x2Re + cayiIm * x2Im;
                        double t2Im2 = cayiIm * x2Re - cayiRe * x2Im;
                        A[p2] += t1Re2 + t2Re2;
                        A[p2 + 1] += t1Im2 + t2Im2;

                        double t1Re3 = axiRe * y3Re + axiIm * y3Im;
                        double t1Im3 = axiIm * y3Re - axiRe * y3Im;
                        double t2Re3 = cayiRe * x3Re + cayiIm * x3Im;
                        double t2Im3 = cayiIm * x3Re - cayiRe * x3Im;
                        A[p3] += t1Re3 + t2Re3;
                        A[p3 + 1] += t1Im3 + t2Im3;
                    }
                }
                for (; j < i; j++) {
                    int xjOff = xStart + j * incX * 2;
                    int yjOff = yStart + j * incY * 2;
                    double xjRe = x[xjOff], xjIm = x[xjOff + 1];
                    double yjRe = y[yjOff], yjIm = y[yjOff + 1];
                    int aPos2 = aRow + j * 2;

                    double t1Re = axiRe * yjRe + axiIm * yjIm;
                    double t1Im = axiIm * yjRe - axiRe * yjIm;
                    double t2Re = cayiRe * xjRe + cayiIm * xjIm;
                    double t2Im = cayiIm * xjRe - cayiRe * xjIm;

                    A[aPos2] += t1Re + t2Re;
                    A[aPos2 + 1] += t1Im + t2Im;
                }

                int aPos = aRow + i * 2;
                double diagRe = (axiRe * yiRe + axiIm * yiIm) + (cayiRe * xiRe + cayiIm * xiIm);
                A[aPos] += diagRe;
            }
        }
    }
}
