/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

interface Zdscal {

    int SIMD_MIN_N = 80;

    public static void zdscal(int n, double da, double[] x, int xOff, int incX) {
        if (n <= 0 || da == 1.0) return;
        if (incX == 1 && n >= SIMD_MIN_N && SIMD.supportZdscal()) {
            if (ZdscalSIMD.zdscal(n, da, x, xOff)) {
                return;
            }
        }
        if (incX == 1) {
            int n4 = (n / 4) * 4;
            int i = 0;
            for (; i < n4 * 2; i += 8) {
                int p = xOff + i;
                x[p]   *= da; x[p+1] *= da;
                x[p+2] *= da; x[p+3] *= da;
                x[p+4] *= da; x[p+5] *= da;
                x[p+6] *= da; x[p+7] *= da;
            }
            for (; i < n * 2; i += 2) {
                int p = xOff + i;
                x[p] *= da; x[p+1] *= da;
            }
        } else {
            for (int i = 0; i < n; i++) {
                int p = xOff + i * incX * 2;
                x[p] *= da; x[p+1] *= da;
            }
        }
    }
}

final class ZdscalSIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;

    private ZdscalSIMD() {
    }

    static boolean zdscal(int n, double alpha, double[] x, int xOff) {
        if (LANES <= 1) {
            return false;
        }

        int length = n * 2;
        DoubleVector alphaVec = DoubleVector.broadcast(SPECIES, alpha);
        int limit = SPECIES.loopBound(length);
        int k = 0;
        for (; k < limit; k += LANES) {
            alphaVec.mul(DoubleVector.fromArray(SPECIES, x, xOff + k)).intoArray(x, xOff + k);
        }
        for (; k < length; k++) {
            x[xOff + k] *= alpha;
        }
        return true;
    }

    static boolean probe() {
        if (LANES <= 1) {
            return false;
        }
        int n = Math.max(1, LANES);
        double[] x = new double[n * 2];
        return zdscal(n, 1.0, x, 0);
    }
}
