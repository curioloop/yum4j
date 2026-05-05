/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import static java.lang.Math.max;

interface Zhseqr {

    static int zhseqrQuery(int n) {
        if (n == 0) return 1;
        return max(1, n) * 2;
    }

    static int zhseqr(char job, char compz, int n, int ilo, int ihi,
                      double[] H, int hOff, int ldh,
                      double[] w, int wOff,
                      double[] Z, int zOff, int ldz,
                      double[] work, int workOff, int lwork) {

        if (n == 0) {
            work[workOff] = 1;
            return 0;
        }

        int lwmin = max(1, n) * 2;

        if (lwork == -1) {
            work[workOff] = lwmin;
            return 0;
        }

        boolean wantt = Character.toUpperCase(job) == 'S';
        boolean wantz = Character.toUpperCase(compz) == 'V' || Character.toUpperCase(compz) == 'I';

        for (int i = 0; i < ilo; i++) {
            int p = hOff + i * ldh * 2 + i * 2;
            w[wOff + i * 2] = H[p];
            w[wOff + i * 2 + 1] = H[p + 1];
        }
        for (int i = ihi + 1; i < n; i++) {
            int p = hOff + i * ldh * 2 + i * 2;
            w[wOff + i * 2] = H[p];
            w[wOff + i * 2 + 1] = H[p + 1];
        }

        if (Character.toUpperCase(compz) == 'I') {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int p = zOff + i * ldz * 2 + j * 2;
                    Z[p] = (i == j) ? 1.0 : 0.0;
                    Z[p + 1] = 0.0;
                }
            }
        }

        if (ilo == ihi) {
            int p = hOff + ilo * ldh * 2 + ilo * 2;
            w[wOff + ilo * 2] = H[p];
            w[wOff + ilo * 2 + 1] = H[p + 1];
            work[workOff] = lwmin;
            return 0;
        }

        // Keep only the Hessenberg structure that the QR driver expects inside the active window.
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                int p = hOff + i * ldh * 2 + j * 2;
                H[p] = 0.0;
                H[p + 1] = 0.0;
            }
        }

        double[] tmpWr = new double[n];
        double[] tmpWi = new double[n];

        // ZLAHQR performs the actual QR sweeps; this wrapper only handles queries, cleanup and packing.
        int info = Zlahqr.zlahqr(wantt, wantz, n, ilo, ihi,
                H, hOff, ldh, tmpWr, tmpWi,
                0, n - 1, Z, zOff, ldz,
                work, workOff);

        for (int i = ilo; i <= ihi; i++) {
            w[wOff + i * 2] = tmpWr[i];
            w[wOff + i * 2 + 1] = tmpWi[i];
        }

        if (wantt || info > 0) {
            for (int i = 1; i < n; i++) {
                for (int j = 0; j < i - 1; j++) {
                    int p = hOff + i * ldh * 2 + j * 2;
                    H[p] = 0.0;
                    H[p + 1] = 0.0;
                }
            }
        }

        work[workOff] = lwmin;
        return info;
    }
}
