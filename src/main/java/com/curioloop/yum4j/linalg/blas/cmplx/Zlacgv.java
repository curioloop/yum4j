/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

interface Zlacgv {

    static void zlacgv(int n, double[] x, int xOff, int incx) {
        if (n <= 0 || incx <= 0) return;

        for (int i = 0; i < n; i++) {
            int pos = xOff + i * incx;
            int off = pos * 2 + 1;
            if (off < x.length) {
                x[off] = -x[off];
            }
        }
    }
}
