/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import static java.lang.Math.max;

interface Ztrsen {

    static boolean ztrsen(char job, char compq, boolean[] selects, int n,
                          double[] t, int tOff, int ldt,
                          double[] q, int qOff, int ldq,
                          double[] w, int wOff,
                          double[] work, int workOff, int lwork) {

        boolean wants = job == 'E' || job == 'B';
        boolean wantsp = job == 'V' || job == 'B';
        boolean wantq = compq == 'V';

        if (n == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return true;
        }

        int m = 0;
        for (int k = 0; k < n; k++) {
            if (selects[k]) m++;
        }

        int n1 = m;
        int n2 = n - m;
        int nn = n1 * n2;

        int lwmin = 1;
        if (wantsp) {
            lwmin = max(n, 2 * nn);
        } else if (job == 'N') {
            lwmin = max(1, n);
        } else if (job == 'E') {
            lwmin = max(1, nn);
        }

        if (lwork == -1) {
            work[workOff] = lwmin;
            return true;
        }

        if (m == n || m == 0) {
            for (int k = 0; k < n; k++) {
                w[wOff + k * 2] = t[tOff + k * ldt * 2 + k * 2];
                w[wOff + k * 2 + 1] = t[tOff + k * ldt * 2 + k * 2 + 1];
            }
            return true;
        }

        int ks = -1;
        for (int k = 0; k < n; k++) {
            if (selects[k]) {
                ks++;
                if (k != ks) {
                    if (!Ztrexc.ztrexc(compq, n, t, tOff, ldt, q, qOff, ldq, k, ks)) {
                        for (int kk = 0; kk < n; kk++) {
                            w[wOff + kk * 2] = t[tOff + kk * ldt * 2 + kk * 2];
                            w[wOff + kk * 2 + 1] = t[tOff + kk * ldt * 2 + kk * 2 + 1];
                        }
                        return false;
                    }
                }
            }
        }

        for (int k = 0; k < n; k++) {
            w[wOff + k * 2] = t[tOff + k * ldt * 2 + k * 2];
            w[wOff + k * 2 + 1] = t[tOff + k * ldt * 2 + k * 2 + 1];
        }
        return true;
    }
}
