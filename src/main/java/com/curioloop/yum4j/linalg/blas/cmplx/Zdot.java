/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

interface Zdot {

    static void zdotc(int n, double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY,
                      double[] out, int outOff) {
        if (n <= 0) {
            zero(out, outOff);
            return;
        }

        int xIdx = xOff * 2;
        int yIdx = yOff * 2;
        int strideX = incX * 2;
        int strideY = incY * 2;

        if (incX == 1) {
            if (incY == 1) {
                zdotcUnitStride(n, x, xIdx, y, yIdx, out, outOff);
            } else {
                zdotcContiguousXStridedY(n, x, xIdx, y, yIdx, strideY, out, outOff);
            }
            return;
        }

        double dotRe = 0.0;
        double dotIm = 0.0;
        for (int k = 0; k < n; k++) {
            double xRe = x[xIdx];
            double xIm = x[xIdx + 1];
            double yRe = y[yIdx];
            double yIm = y[yIdx + 1];
            dotRe = Math.fma(xRe, yRe, Math.fma(xIm, yIm, dotRe));
            dotIm = Math.fma(xRe, yIm, Math.fma(-xIm, yRe, dotIm));
            xIdx += strideX;
            yIdx += strideY;
        }
        out[outOff] = dotRe;
        out[outOff + 1] = dotIm;
    }

    static void zdotu(int n, double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY,
                      double[] out, int outOff) {
        if (n <= 0) {
            zero(out, outOff);
            return;
        }

        int xIdx = xOff * 2;
        int yIdx = yOff * 2;
        int strideX = incX * 2;
        int strideY = incY * 2;

        if (incX == 1) {
            if (incY == 1) {
                zdotuUnitStride(n, x, xIdx, y, yIdx, out, outOff);
            } else {
                zdotuContiguousXStridedY(n, x, xIdx, y, yIdx, strideY, out, outOff);
            }
            return;
        }

        double dotRe = 0.0;
        double dotIm = 0.0;
        for (int k = 0; k < n; k++) {
            double xRe = x[xIdx];
            double xIm = x[xIdx + 1];
            double yRe = y[yIdx];
            double yIm = y[yIdx + 1];
            dotRe = Math.fma(xRe, yRe, Math.fma(-xIm, yIm, dotRe));
            dotIm = Math.fma(xRe, yIm, Math.fma(xIm, yRe, dotIm));
            xIdx += strideX;
            yIdx += strideY;
        }
        out[outOff] = dotRe;
        out[outOff + 1] = dotIm;
    }

    private static void zdotcUnitStride(int n, double[] x, int xIdx,
                                        double[] y, int yIdx,
                                        double[] out, int outOff) {
        double s0Re = 0.0;
        double s0Im = 0.0;
        double s1Re = 0.0;
        double s1Im = 0.0;

        int k = 0;
        for (; k + 1 < n; k += 2) {
            double x0Re = x[xIdx];
            double x0Im = x[xIdx + 1];
            double x1Re = x[xIdx + 2];
            double x1Im = x[xIdx + 3];
            double y0Re = y[yIdx];
            double y0Im = y[yIdx + 1];
            double y1Re = y[yIdx + 2];
            double y1Im = y[yIdx + 3];
            s0Re = Math.fma(x0Re, y0Re, Math.fma(x0Im, y0Im, s0Re));
            s0Im = Math.fma(x0Re, y0Im, Math.fma(-x0Im, y0Re, s0Im));
            s1Re = Math.fma(x1Re, y1Re, Math.fma(x1Im, y1Im, s1Re));
            s1Im = Math.fma(x1Re, y1Im, Math.fma(-x1Im, y1Re, s1Im));
            xIdx += 4;
            yIdx += 4;
        }

        double dotRe = s0Re + s1Re;
        double dotIm = s0Im + s1Im;
        for (; k < n; k++) {
            double xRe = x[xIdx];
            double xIm = x[xIdx + 1];
            double yRe = y[yIdx];
            double yIm = y[yIdx + 1];
            dotRe = Math.fma(xRe, yRe, Math.fma(xIm, yIm, dotRe));
            dotIm = Math.fma(xRe, yIm, Math.fma(-xIm, yRe, dotIm));
            xIdx += 2;
            yIdx += 2;
        }

        out[outOff] = dotRe;
        out[outOff + 1] = dotIm;
    }

    private static void zdotcContiguousXStridedY(int n, double[] x, int xIdx,
                                                 double[] y, int yIdx, int strideY,
                                                 double[] out, int outOff) {
        double s0Re = 0.0;
        double s0Im = 0.0;
        double s1Re = 0.0;
        double s1Im = 0.0;
        double s2Re = 0.0;
        double s2Im = 0.0;
        double s3Re = 0.0;
        double s3Im = 0.0;

        int k = 0;
        for (; k + 3 < n; k += 4) {
            int y1Idx = yIdx + strideY;
            int y2Idx = y1Idx + strideY;
            int y3Idx = y2Idx + strideY;

            double x0Re = x[xIdx];
            double x0Im = x[xIdx + 1];
            double x1Re = x[xIdx + 2];
            double x1Im = x[xIdx + 3];
            double x2Re = x[xIdx + 4];
            double x2Im = x[xIdx + 5];
            double x3Re = x[xIdx + 6];
            double x3Im = x[xIdx + 7];

            double y0Re = y[yIdx];
            double y0Im = y[yIdx + 1];
            double y1Re = y[y1Idx];
            double y1Im = y[y1Idx + 1];
            double y2Re = y[y2Idx];
            double y2Im = y[y2Idx + 1];
            double y3Re = y[y3Idx];
            double y3Im = y[y3Idx + 1];

            s0Re = Math.fma(x0Re, y0Re, Math.fma(x0Im, y0Im, s0Re));
            s0Im = Math.fma(x0Re, y0Im, Math.fma(-x0Im, y0Re, s0Im));
            s1Re = Math.fma(x1Re, y1Re, Math.fma(x1Im, y1Im, s1Re));
            s1Im = Math.fma(x1Re, y1Im, Math.fma(-x1Im, y1Re, s1Im));
            s2Re = Math.fma(x2Re, y2Re, Math.fma(x2Im, y2Im, s2Re));
            s2Im = Math.fma(x2Re, y2Im, Math.fma(-x2Im, y2Re, s2Im));
            s3Re = Math.fma(x3Re, y3Re, Math.fma(x3Im, y3Im, s3Re));
            s3Im = Math.fma(x3Re, y3Im, Math.fma(-x3Im, y3Re, s3Im));

            xIdx += 8;
            yIdx = y3Idx + strideY;
        }

        double dotRe = (s0Re + s1Re) + (s2Re + s3Re);
        double dotIm = (s0Im + s1Im) + (s2Im + s3Im);
        for (; k < n; k++) {
            double xRe = x[xIdx];
            double xIm = x[xIdx + 1];
            double yRe = y[yIdx];
            double yIm = y[yIdx + 1];
            dotRe = Math.fma(xRe, yRe, Math.fma(xIm, yIm, dotRe));
            dotIm = Math.fma(xRe, yIm, Math.fma(-xIm, yRe, dotIm));
            xIdx += 2;
            yIdx += strideY;
        }

        out[outOff] = dotRe;
        out[outOff + 1] = dotIm;
    }

    private static void zdotuUnitStride(int n, double[] x, int xIdx,
                                        double[] y, int yIdx,
                                        double[] out, int outOff) {
        double s0Re = 0.0;
        double s0Im = 0.0;
        double s1Re = 0.0;
        double s1Im = 0.0;

        int k = 0;
        for (; k + 1 < n; k += 2) {
            double x0Re = x[xIdx];
            double x0Im = x[xIdx + 1];
            double x1Re = x[xIdx + 2];
            double x1Im = x[xIdx + 3];
            double y0Re = y[yIdx];
            double y0Im = y[yIdx + 1];
            double y1Re = y[yIdx + 2];
            double y1Im = y[yIdx + 3];
            s0Re = Math.fma(x0Re, y0Re, Math.fma(-x0Im, y0Im, s0Re));
            s0Im = Math.fma(x0Re, y0Im, Math.fma(x0Im, y0Re, s0Im));
            s1Re = Math.fma(x1Re, y1Re, Math.fma(-x1Im, y1Im, s1Re));
            s1Im = Math.fma(x1Re, y1Im, Math.fma(x1Im, y1Re, s1Im));
            xIdx += 4;
            yIdx += 4;
        }

        double dotRe = s0Re + s1Re;
        double dotIm = s0Im + s1Im;
        for (; k < n; k++) {
            double xRe = x[xIdx];
            double xIm = x[xIdx + 1];
            double yRe = y[yIdx];
            double yIm = y[yIdx + 1];
            dotRe = Math.fma(xRe, yRe, Math.fma(-xIm, yIm, dotRe));
            dotIm = Math.fma(xRe, yIm, Math.fma(xIm, yRe, dotIm));
            xIdx += 2;
            yIdx += 2;
        }

        out[outOff] = dotRe;
        out[outOff + 1] = dotIm;
    }

    private static void zdotuContiguousXStridedY(int n, double[] x, int xIdx,
                                                 double[] y, int yIdx, int strideY,
                                                 double[] out, int outOff) {
        double s0Re = 0.0;
        double s0Im = 0.0;
        double s1Re = 0.0;
        double s1Im = 0.0;
        double s2Re = 0.0;
        double s2Im = 0.0;
        double s3Re = 0.0;
        double s3Im = 0.0;

        int k = 0;
        for (; k + 3 < n; k += 4) {
            int y1Idx = yIdx + strideY;
            int y2Idx = y1Idx + strideY;
            int y3Idx = y2Idx + strideY;

            double x0Re = x[xIdx];
            double x0Im = x[xIdx + 1];
            double x1Re = x[xIdx + 2];
            double x1Im = x[xIdx + 3];
            double x2Re = x[xIdx + 4];
            double x2Im = x[xIdx + 5];
            double x3Re = x[xIdx + 6];
            double x3Im = x[xIdx + 7];

            double y0Re = y[yIdx];
            double y0Im = y[yIdx + 1];
            double y1Re = y[y1Idx];
            double y1Im = y[y1Idx + 1];
            double y2Re = y[y2Idx];
            double y2Im = y[y2Idx + 1];
            double y3Re = y[y3Idx];
            double y3Im = y[y3Idx + 1];

            s0Re = Math.fma(x0Re, y0Re, Math.fma(-x0Im, y0Im, s0Re));
            s0Im = Math.fma(x0Re, y0Im, Math.fma(x0Im, y0Re, s0Im));
            s1Re = Math.fma(x1Re, y1Re, Math.fma(-x1Im, y1Im, s1Re));
            s1Im = Math.fma(x1Re, y1Im, Math.fma(x1Im, y1Re, s1Im));
            s2Re = Math.fma(x2Re, y2Re, Math.fma(-x2Im, y2Im, s2Re));
            s2Im = Math.fma(x2Re, y2Im, Math.fma(x2Im, y2Re, s2Im));
            s3Re = Math.fma(x3Re, y3Re, Math.fma(-x3Im, y3Im, s3Re));
            s3Im = Math.fma(x3Re, y3Im, Math.fma(x3Im, y3Re, s3Im));

            xIdx += 8;
            yIdx = y3Idx + strideY;
        }

        double dotRe = (s0Re + s1Re) + (s2Re + s3Re);
        double dotIm = (s0Im + s1Im) + (s2Im + s3Im);
        for (; k < n; k++) {
            double xRe = x[xIdx];
            double xIm = x[xIdx + 1];
            double yRe = y[yIdx];
            double yIm = y[yIdx + 1];
            dotRe = Math.fma(xRe, yRe, Math.fma(-xIm, yIm, dotRe));
            dotIm = Math.fma(xRe, yIm, Math.fma(xIm, yRe, dotIm));
            xIdx += 2;
            yIdx += strideY;
        }

        out[outOff] = dotRe;
        out[outOff + 1] = dotIm;
    }

    private static void zero(double[] out, int outOff) {
        out[outOff] = 0.0;
        out[outOff + 1] = 0.0;
    }
}