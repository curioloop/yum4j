/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import static java.lang.Math.abs;

/**
 * DZASUM computes the sum of absolute values of a complex vector stored as
 * interleaved real and imaginary parts.
 */
interface Dzasum {

    /**
     * Computes Σ (|Re(xᵢ)| + |Im(xᵢ)|).
     *
     * @param n number of complex elements
     * @param x input vector in interleaved format [re, im, re, im, ...]
     * @param xOff offset in x, measured in raw double slots
     * @param incX storage spacing in complex elements
     * @return complex 1-norm accumulator used by BLAS DZASUM
     */
    static double dzasum(int n, double[] x, int xOff, int incX) {
        if (n <= 0) return 0.0;

        double result = 0.0;
        int step = incX * 2;
        int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-step));

        for (int i = 0; i < n; i++) {
            result += abs(x[ix]) + abs(x[ix + 1]);
            ix += step;
        }

        return result;
    }
}