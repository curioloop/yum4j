/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
interface Zgebak {

    static void zgebak(char job, BLAS.Side side, int n, int ilo, int ihi,
                        double[] scale, int scaleOff, int m, double[] V, int vOff, int ldv) {
        if (n == 0 || m == 0) return;

        char jobUpper = Character.toUpperCase(job);

        if (jobUpper == 'N') return;

        if (ilo != ihi) {
            if (jobUpper == 'S' || jobUpper == 'B') {
                if (side == BLAS.Side.Right) {
                    for (int i = ilo; i <= ihi; i++) {
                        double s = scale[scaleOff + i];
                        if (s != 1.0) {
                            Zscal.zscal(m, s, 0.0, V, vOff + i * ldv * 2, 1);
                        }
                    }
                } else {
                    for (int i = ilo; i <= ihi; i++) {
                        double s = scale[scaleOff + i];
                        if (s != 1.0) {
                            Zscal.zscal(m, 1.0 / s, 0.0, V, vOff + i * ldv * 2, 1);
                        }
                    }
                }
            }
        }

        if (jobUpper == 'P' || jobUpper == 'B') {
            if (side == BLAS.Side.Right) {
                for (int ii = 0; ii < n; ii++) {
                    int i = ii;
                    if (i >= ilo && i <= ihi) continue;
                    if (i < ilo) i = ilo - 1 - ii;
                    int k = (int) scale[scaleOff + i];
                    if (k == i) continue;
                    Zswap.zswap(m, V, vOff + i * ldv * 2, 1, V, vOff + k * ldv * 2, 1);
                }
            } else {
                for (int ii = n - 1; ii >= 0; ii--) {
                    int i = ii;
                    if (i >= ilo && i <= ihi) continue;
                    if (i < ilo) i = ilo - 1 - ii;
                    int k = (int) scale[scaleOff + i];
                    if (k == i) continue;
                    Zswap.zswap(m, V, vOff + i * ldv * 2, 1, V, vOff + k * ldv * 2, 1);
                }
            }
        }
    }
}
