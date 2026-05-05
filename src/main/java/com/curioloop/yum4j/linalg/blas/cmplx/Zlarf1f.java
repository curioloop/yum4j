/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

import static java.lang.Math.max;

interface Zlarf1f {

    static int zlarf1f(BLAS.Side side, int m, int n, double[] v, int vOff, int incv, double[] tau, int tauOff, double[] c, int cOff, int ldc, double[] work, int workOff) {
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
            if (m == 1) {
                Zscal.zscal(n, 1.0 - tauRe, -tauIm, c, cOff, 1);
            } else {
                Zgemv.zgemv(BLAS.Trans.Conj, m - 1, n, 1.0, 0.0, c, cOff + ldc * 2, ldc, v, vOff + incv * 2, incv, 0.0, 0.0, work, workOff, 1);
                for (int j = 0; j < n; j++) {
                    int cPos = cOff + j * 2;
                    int wPos = workOff + j * 2;
                    work[wPos] += c[cPos];
                    work[wPos + 1] -= c[cPos + 1];
                }
                for (int j = 0; j < n; j++) {
                    int cPos = cOff + j * 2;
                    int wPos = workOff + j * 2;
                    c[cPos] -= tauRe * work[wPos] + tauIm * work[wPos + 1];
                    c[cPos + 1] -= tauIm * work[wPos] - tauRe * work[wPos + 1];
                }
                Zgerc.zgerc(m - 1, n, -tauRe, -tauIm, v, vOff + incv * 2, incv, work, workOff, 1, c, cOff + ldc * 2, ldc);
            }
        } else {
            if (n == 1) {
                Zscal.zscal(m, 1.0 - tauRe, -tauIm, c, cOff, ldc);
            } else {
                Zgemv.zgemv(BLAS.Trans.NoTrans, m, n - 1, 1.0, 0.0, c, cOff + 2, ldc, v, vOff + incv * 2, incv, 0.0, 0.0, work, workOff, 1);
                for (int i = 0; i < m; i++) {
                    int cPos = cOff + i * ldc * 2;
                    int wPos = workOff + i * 2;
                    work[wPos] += c[cPos];
                    work[wPos + 1] += c[cPos + 1];
                }
                for (int i = 0; i < m; i++) {
                    int cPos = cOff + i * ldc * 2;
                    int wPos = workOff + i * 2;
                    c[cPos] -= tauRe * work[wPos] - tauIm * work[wPos + 1];
                    c[cPos + 1] -= tauRe * work[wPos + 1] + tauIm * work[wPos];
                }
                Zgerc.zgerc(m, n - 1, -tauRe, -tauIm, work, workOff, 1, v, vOff + incv * 2, incv, c, cOff + 2, ldc);
            }
        }

        return 0;
    }
}
