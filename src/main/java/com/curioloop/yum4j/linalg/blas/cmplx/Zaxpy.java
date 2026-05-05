/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

/**
 * ZAXPY computes y += α × x for complex vectors.
 * 
 * <p>Mathematical operation: y[yOff + 2i] += α_re × x[xOff + 2i] - α_im × x[xOff + 2i+1]</p>
 * <p>                        y[yOff + 2i+1] += α_re × x[xOff + 2i+1] + α_im × x[xOff + 2i]</p>
 * 
 * <p>Storage layout: Each complex number is stored as two consecutive doubles: [real, imag]</p>
 * <p>Reference: BLAS ZAXPY</p>
 */
interface Zaxpy {

    int SIMD_MIN_N = 48;

    /**
     * Performs complex vector update: y += alpha * x
     * 
     * @param n number of complex elements
     * @param alphaRe real part of alpha
     * @param alphaIm imaginary part of alpha
     * @param x complex vector (interleaved storage: [re0, im0, re1, im1, ...])
     * @param xOff offset into x (in complex elements, not bytes)
     * @param incX stride in x (in complex elements)
     * @param y complex vector (interleaved storage: [re0, im0, re1, im1, ...])
     * @param yOff offset into y (in complex elements, not bytes)
     * @param incY stride in y (in complex elements)
     */
    static void zaxpy(int n, double alphaRe, double alphaIm,
                      double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY) {
        if (n <= 0 || (alphaRe == 0.0 && alphaIm == 0.0)) return;

        if (incX == 1 && incY == 1 && n >= SIMD_MIN_N && SIMD.supportZaxpy()) {
            if (ZaxpySIMD.zaxpy(n, alphaRe, alphaIm, x, xOff, y, yOff)) {
                return;
            }
        }

        // Convert offsets to double array indices (each complex element takes 2 doubles)
        int xIdx = xOff * 2;
        int yIdx = yOff * 2;
        int strideX = incX * 2;
        int strideY = incY * 2;

        if (incX == 1 && incY == 1) {
            // Unit stride case - optimized for contiguous memory
            int k = 0;
            for (; k + 1 < n; k += 2) {
                double x0re = x[xIdx];
                double x0im = x[xIdx + 1];
                double x1re = x[xIdx + 2];
                double x1im = x[xIdx + 3];

                double y0re = y[yIdx];
                double y0im = y[yIdx + 1];
                double y1re = y[yIdx + 2];
                double y1im = y[yIdx + 3];

                y[yIdx]     = Math.fma(alphaRe, x0re, Math.fma(-alphaIm, x0im, y0re));
                y[yIdx + 1] = Math.fma(alphaRe, x0im, Math.fma(alphaIm, x0re, y0im));
                y[yIdx + 2] = Math.fma(alphaRe, x1re, Math.fma(-alphaIm, x1im, y1re));
                y[yIdx + 3] = Math.fma(alphaRe, x1im, Math.fma(alphaIm, x1re, y1im));

                xIdx += 4;
                yIdx += 4;
            }

            // Process remaining elements
            for (; k < n; k++) {
                double xre = x[xIdx];
                double xim = x[xIdx + 1];
                double yre = y[yIdx];
                double yim = y[yIdx + 1];

                y[yIdx]     = Math.fma(alphaRe, xre, Math.fma(-alphaIm, xim, yre));
                y[yIdx + 1] = Math.fma(alphaRe, xim, Math.fma(alphaIm, xre, yim));

                xIdx += 2;
                yIdx += 2;
            }
        } else {
            // Non-unit stride case
            for (int k = 0; k < n; k++) {
                double xre = x[xIdx];
                double xim = x[xIdx + 1];
                double yre = y[yIdx];
                double yim = y[yIdx + 1];

                y[yIdx]     = Math.fma(alphaRe, xre, Math.fma(-alphaIm, xim, yre));
                y[yIdx + 1] = Math.fma(alphaRe, xim, Math.fma(alphaIm, xre, yim));

                xIdx += strideX;
                yIdx += strideY;
            }
        }
    }

}

final class ZaxpySIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;
    private static final VectorShuffle<Double> SWAP_RE_IM = VectorShuffle.fromArray(SPECIES, buildSwapIndexes(), 0);
    private static final DoubleVector IMAG_SIGN = DoubleVector.fromArray(SPECIES, buildImagSign(), 0);

    private ZaxpySIMD() {
    }

    static boolean zaxpy(int n, double alphaRe, double alphaIm,
                         double[] x, int xOff,
                         double[] y, int yOff) {
        if (LANES <= 1) {
            return false;
        }

        int xIdx = xOff * 2;
        int yIdx = yOff * 2;
        int length = n * 2;
        DoubleVector alphaReVec = DoubleVector.broadcast(SPECIES, alphaRe);
        DoubleVector alphaImVec = DoubleVector.broadcast(SPECIES, alphaIm).mul(IMAG_SIGN);
        int limit = SPECIES.loopBound(length);
        int k = 0;
        for (; k < limit; k += LANES) {
            DoubleVector xVec = DoubleVector.fromArray(SPECIES, x, xIdx + k);
            DoubleVector yVec = DoubleVector.fromArray(SPECIES, y, yIdx + k);
            DoubleVector swapped = xVec.rearrange(SWAP_RE_IM);
            alphaImVec.fma(swapped, alphaReVec.fma(xVec, yVec)).intoArray(y, yIdx + k);
        }
        for (; k < length; k += 2) {
            int xp = xIdx + k;
            int yp = yIdx + k;
            double xre = x[xp];
            double xim = x[xp + 1];
            double yre = y[yp];
            double yim = y[yp + 1];
            y[yp] = Math.fma(alphaRe, xre, Math.fma(-alphaIm, xim, yre));
            y[yp + 1] = Math.fma(alphaRe, xim, Math.fma(alphaIm, xre, yim));
        }
        return true;
    }

    static boolean probe() {
        if (LANES <= 1) {
            return false;
        }
        int n = Math.max(1, LANES / 2);
        double[] x = new double[n * 2];
        double[] y = new double[n * 2];
        return zaxpy(n, 1.0, -0.5, x, 0, y, 0);
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