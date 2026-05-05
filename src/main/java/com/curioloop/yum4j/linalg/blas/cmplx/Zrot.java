/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

interface Zrot {

    static void zrot(int n, double[] x, int xOff, int incx, double[] y, int yOff, int incy,
                     double c, double sRe, double sIm) {
        if (n <= 0) return;

        for (int i = 0; i < n; i++) {
            int offX = xOff + i * incx * 2;
            int offY = yOff + i * incy * 2;
            double xRe = x[offX];
            double xIm = x[offX + 1];
            double yRe = y[offY];
            double yIm = y[offY + 1];

            x[offX]     = c * xRe + sRe * yRe - sIm * yIm;
            x[offX + 1] = c * xIm + sRe * yIm + sIm * yRe;
            y[offY]     = c * yRe - sRe * xRe - sIm * xIm;
            y[offY + 1] = c * yIm - sRe * xIm + sIm * xRe;
        }
    }
}
