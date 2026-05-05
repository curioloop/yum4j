/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.Dnrm2;

/**
 * DZNRM2 computes the Euclidean norm of a complex vector stored as interleaved
 * real and imaginary parts.
 */
interface Dznrm2 {

    /**
     * Computes the Euclidean norm of a complex vector using the same safe scaling
     * thresholds as DNRM2.
     *
     * @param n number of complex elements
     * @param x input vector in interleaved format [re, im, re, im, ...]
     * @param xOff offset in x, measured in raw double slots
     * @param incX storage spacing in complex elements
     * @return ‖x‖₂
     */
    static double dznrm2(int n, double[] x, int xOff, int incX) {
        if (n <= 0) return 0.0;
        if (n == 1) return Math.hypot(x[xOff], x[xOff + 1]);

        double asml = 0.0;
        double amed = 0.0;
        double abig = 0.0;
        boolean notbig = true;

        int step = incX * 2;
        int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-step));

        for (int i = 0; i < n; i++) {
            double ax = Math.abs(x[ix]);
            if (ax > Dnrm2.TBIG) {
                double scaled = ax * Dnrm2.SBIG;
                abig = Math.fma(scaled, scaled, abig);
                notbig = false;
            } else if (ax < Dnrm2.TSML) {
                if (notbig) {
                    double scaled = ax * Dnrm2.SSML;
                    asml = Math.fma(scaled, scaled, asml);
                }
            } else {
                amed = Math.fma(ax, ax, amed);
            }

            ax = Math.abs(x[ix + 1]);
            if (ax > Dnrm2.TBIG) {
                double scaled = ax * Dnrm2.SBIG;
                abig = Math.fma(scaled, scaled, abig);
                notbig = false;
            } else if (ax < Dnrm2.TSML) {
                if (notbig) {
                    double scaled = ax * Dnrm2.SSML;
                    asml = Math.fma(scaled, scaled, asml);
                }
            } else {
                amed = Math.fma(ax, ax, amed);
            }

            ix += step;
        }

        double scl;
        double sumsq;
        if (abig > 0.0) {
            if (amed > 0.0 || Double.isNaN(amed)) {
                abig += (amed * Dnrm2.SBIG) * Dnrm2.SBIG;
            }
            scl = 1.0 / Dnrm2.SBIG;
            sumsq = abig;
        } else if (asml > 0.0) {
            if (amed > 0.0 || Double.isNaN(amed)) {
                double ymin;
                double ymax;
                amed = Math.sqrt(amed);
                asml = Math.sqrt(asml) / Dnrm2.SSML;
                if (asml > amed) {
                    ymin = amed;
                    ymax = asml;
                } else {
                    ymin = asml;
                    ymax = amed;
                }
                scl = 1.0;
                double ratio = ymin / ymax;
                sumsq = ymax * ymax * (1.0 + ratio * ratio);
            } else {
                scl = 1.0 / Dnrm2.SSML;
                sumsq = asml;
            }
        } else {
            scl = 1.0;
            sumsq = amed;
        }

        return scl * Math.sqrt(sumsq);
    }
}