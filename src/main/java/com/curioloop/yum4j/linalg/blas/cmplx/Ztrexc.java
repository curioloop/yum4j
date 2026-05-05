/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

interface Ztrexc {

    static boolean ztrexc(char compq, int n,
                          double[] t, int tOff, int ldt,
                          double[] q, int qOff, int ldq,
                          int ifst, int ilst) {
        if (n <= 1 || ifst == ilst) return true;

        boolean wantq = compq == 'V';

        int m1, m2, m3;
        if (ifst < ilst) {
            m1 = 0;
            m2 = -1;
            m3 = 1;
        } else {
            m1 = -1;
            m2 = 0;
            m3 = -1;
        }

        double[] givens = new double[5];
        for (int k = ifst + m1; k != ilst + m2 + m3; k += m3) {
            int pKK = tOff + k * ldt * 2 + k * 2;
            int pK1K1 = tOff + (k + 1) * ldt * 2 + (k + 1) * 2;
            int pKK1 = tOff + k * ldt * 2 + (k + 1) * 2;

            double t11Re = t[pKK];
            double t11Im = t[pKK + 1];
            double t22Re = t[pK1K1];
            double t22Im = t[pK1K1 + 1];

            double fRe = t[pKK1];
            double fIm = t[pKK1 + 1];
            double gRe = t22Re - t11Re;
            double gIm = t22Im - t11Im;

            Zlartg.zlartg(fRe, fIm, gRe, gIm, givens, 0);
            double cs = givens[0];
            double snRe = givens[1];
            double snIm = givens[2];

            if (k + 2 <= n - 1) {
                Zrot.zrot(n - k - 2, t, tOff + k * ldt * 2 + (k + 2) * 2, 1,
                        t, tOff + (k + 1) * ldt * 2 + (k + 2) * 2, 1, cs, snRe, snIm);
            }

            Zrot.zrot(k, t, tOff + 0 * ldt * 2 + k * 2, ldt,
                    t, tOff + 0 * ldt * 2 + (k + 1) * 2, ldt, cs, snRe, -snIm);

            t[pKK] = t22Re;
            t[pKK + 1] = t22Im;
            t[pK1K1] = t11Re;
            t[pK1K1 + 1] = t11Im;

            if (wantq) {
                Zrot.zrot(n, q, qOff + 0 * ldq * 2 + k * 2, ldq,
                        q, qOff + 0 * ldq * 2 + (k + 1) * 2, ldq, cs, snRe, -snIm);
            }
        }

        return true;
    }
}
