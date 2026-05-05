/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

import static java.lang.Math.max;

interface Zlarf {

    static int zlarf(BLAS.Side side, int m, int n, double[] v, int vOff, int incv, double[] tau, int tauOff, double[] c, int cOff, int ldc, double[] work, int workOff) {
        if (side == null) return -1;
        if (m < 0) return -2;
        if (n < 0) return -3;
        if (incv == 0) return -5;
        if (c == null) return -6;
        if (ldc < max(1, n)) return -8;

        if (m == 0 || n == 0) return 0;

        double tauRe = tau[tauOff];
        double tauIm = tau[tauOff + 1];
        if (tauRe == 0.0 && tauIm == 0.0) {
            return 0;
        }

        if (side == BLAS.Side.Left) {
            Zgemv.zgemv(BLAS.Trans.Conj, m, n, 1.0, 0.0, c, cOff, ldc, v, vOff, incv, 0.0, 0.0, work, workOff, 1);
            Zgerc.zgerc(m, n, -tauRe, -tauIm, v, vOff, incv, work, workOff, 1, c, cOff, ldc);
        } else {
            Zgemv.zgemv(BLAS.Trans.NoTrans, m, n, 1.0, 0.0, c, cOff, ldc, v, vOff, incv, 0.0, 0.0, work, workOff, 1);
            Zgerc.zgerc(m, n, -tauRe, -tauIm, work, workOff, 1, v, vOff, incv, c, cOff, ldc);
        }

        return 0;
    }
}
