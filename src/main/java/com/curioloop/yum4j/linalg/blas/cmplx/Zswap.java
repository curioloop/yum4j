/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

/**
 * ZSWAP interchanges two complex vectors.
 * x[i] <-> y[i] for i = 0, 1, ..., n-1
 */
interface Zswap {

    /**
     * Interchanges two complex vectors.
     *
     * @param n      Number of elements in vectors x and y
     * @param x      Complex vector x stored in interleaved format [re, im, re, im, ...]
     * @param xOff Starting index of x
     * @param incX   Increment for x
     * @param y      Complex vector y stored in interleaved format [re, im, re, im, ...]
     * @param yOff Starting index of y
     * @param incY   Increment for y
     */
    static void zswap(int n, double[] x, int xOff, int incX, double[] y, int yOff, int incY) {
        if (n <= 0) return;
        if (incX == 1 && incY == 1) {
            int n4 = (n / 4) * 4;
            int i = 0;
            for (; i < n4 * 2; i += 8) {
                int xp = xOff + i, yp = yOff + i;
                double t0 = x[xp]; x[xp] = y[yp]; y[yp] = t0;
                double t1 = x[xp+1]; x[xp+1] = y[yp+1]; y[yp+1] = t1;
                double t2 = x[xp+2]; x[xp+2] = y[yp+2]; y[yp+2] = t2;
                double t3 = x[xp+3]; x[xp+3] = y[yp+3]; y[yp+3] = t3;
                double t4 = x[xp+4]; x[xp+4] = y[yp+4]; y[yp+4] = t4;
                double t5 = x[xp+5]; x[xp+5] = y[yp+5]; y[yp+5] = t5;
                double t6 = x[xp+6]; x[xp+6] = y[yp+6]; y[yp+6] = t6;
                double t7 = x[xp+7]; x[xp+7] = y[yp+7]; y[yp+7] = t7;
            }
            for (; i < n * 2; i += 2) {
                int xp = xOff + i, yp = yOff + i;
                double tr = x[xp]; x[xp] = y[yp]; y[yp] = tr;
                double ti = x[xp+1]; x[xp+1] = y[yp+1]; y[yp+1] = ti;
            }
        } else {
            for (int i = 0; i < n; i++) {
                int xp = xOff + i * incX * 2, yp = yOff + i * incY * 2;
                double tr = x[xp]; x[xp] = y[yp]; y[yp] = tr;
                double ti = x[xp+1]; x[xp+1] = y[yp+1]; y[yp+1] = ti;
            }
        }
    }
}
