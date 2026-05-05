/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

/**
 * ZSCAL scales a complex vector by a complex scalar.
 * x[i] = alpha * x[i] for i = 0, 1, ..., n-1
 */
interface Zscal {

    int SIMD_MIN_N = 64;

    /**
     * Scales a complex vector by a complex scalar.
     *
     * @param n      Number of elements in vector x
     * @param daIm Real part of the complex scalar alpha
     * @param daRe Imaginary part of the complex scalar alpha
     * @param x      Complex vector x stored in interleaved format [re, im, re, im, ...]
     * @param xOff Starting index of x
     * @param incX   Increment for x
     */
    public static void zscal(int n, double daRe, double daIm, double[] x, int xOff, int incX) {
        if (n <= 0) return;
        if (daIm == 0.0) {
            Zdscal.zdscal(n, daRe, x, xOff, incX);
            return;
        }
        if (incX == 1 && n >= SIMD_MIN_N && SIMD.supportZscal()) {
            if (ZscalSIMD.zscal(n, daRe, daIm, x, xOff)) {
                return;
            }
        }
        if (incX == 1) {
            int n4 = (n / 4) * 4;
            int i = 0;
            for (; i < n4; i += 4) {
                int p = xOff + i * 2;
                double x0r = x[p], x0i = x[p+1];
                double x1r = x[p+2], x1i = x[p+3];
                double x2r = x[p+4], x2i = x[p+5];
                double x3r = x[p+6], x3i = x[p+7];
                x[p]   = daRe * x0r - daIm * x0i; x[p+1] = daRe * x0i + daIm * x0r;
                x[p+2] = daRe * x1r - daIm * x1i; x[p+3] = daRe * x1i + daIm * x1r;
                x[p+4] = daRe * x2r - daIm * x2i; x[p+5] = daRe * x2i + daIm * x2r;
                x[p+6] = daRe * x3r - daIm * x3i; x[p+7] = daRe * x3i + daIm * x3r;
            }
            for (; i < n; i++) {
                int p = xOff + i * 2;
                double xr = x[p], xi = x[p+1];
                x[p] = daRe * xr - daIm * xi; x[p+1] = daRe * xi + daIm * xr;
            }
        } else {
            for (int i = 0; i < n; i++) {
                int p = xOff + i * incX * 2;
                double xr = x[p], xi = x[p+1];
                x[p] = daRe * xr - daIm * xi; x[p+1] = daRe * xi + daIm * xr;
            }
        }
    }
}

final class ZscalSIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;
    private static final VectorShuffle<Double> SWAP_RE_IM = VectorShuffle.fromArray(SPECIES, buildSwapIndexes(), 0);
    private static final DoubleVector IMAG_SIGN = DoubleVector.fromArray(SPECIES, buildImagSign(), 0);

    private ZscalSIMD() {
    }

    static boolean zscal(int n, double alphaRe, double alphaIm, double[] x, int xOff) {
        if (LANES <= 1) {
            return false;
        }

        int length = n * 2;
        DoubleVector alphaReVec = DoubleVector.broadcast(SPECIES, alphaRe);
        DoubleVector alphaImVec = DoubleVector.broadcast(SPECIES, alphaIm).mul(IMAG_SIGN);
        int limit = SPECIES.loopBound(length);
        int k = 0;
        for (; k < limit; k += LANES) {
            DoubleVector xVec = DoubleVector.fromArray(SPECIES, x, xOff + k);
            alphaImVec.fma(xVec.rearrange(SWAP_RE_IM), alphaReVec.mul(xVec)).intoArray(x, xOff + k);
        }
        for (; k < length; k += 2) {
            int p = xOff + k;
            double xr = x[p];
            double xi = x[p + 1];
            x[p] = Math.fma(alphaRe, xr, -alphaIm * xi);
            x[p + 1] = Math.fma(alphaRe, xi, alphaIm * xr);
        }
        return true;
    }

    static boolean probe() {
        if (LANES <= 1) {
            return false;
        }
        int n = Math.max(1, LANES / 2);
        double[] x = new double[n * 2];
        return zscal(n, -0.75, 0.5, x, 0);
    }

    private static int[] buildSwapIndexes() {
        int[] indexes = new int[LANES];
        for (int i = 0; i < LANES; i += 2) {
            indexes[i] = i + 1;
            indexes[i + 1] = i;
        }
        return indexes;
    }

    private static double[] buildImagSign() {
        double[] signs = new double[LANES];
        for (int i = 0; i < LANES; i++) {
            signs[i] = (i & 1) == 0 ? -1.0 : 1.0;
        }
        return signs;
    }
}
