/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;

import static java.lang.Math.max;
import static java.lang.Math.sqrt;

interface Zgees {

    static int zgees(char jobvs, char sort, Zselect select, int n,
                     double[] A, int aOff, int lda,
                     double[] w, int wOff,
                     double[] vs, int vsOff, int ldvs,
                     double[] work, int workOff, int lwork,
                     double[] rwork,
                     boolean[] bwork) {
        return zgees(jobvs, sort, select, n, A, aOff, lda, w, wOff, vs, vsOff, ldvs, work, workOff, lwork, rwork, 0, bwork);
    }

    static int zgees(char jobvs, char sort, Zselect select, int n,
                     double[] A, int aOff, int lda,
                     double[] w, int wOff,
                     double[] vs, int vsOff, int ldvs,
                     double[] work, int workOff, int lwork,
                     double[] rwork,
                     int rworkOff,
                     boolean[] bwork) {

        boolean wantvs = jobvs == 'V' || jobvs == 'v';
        boolean wantst = sort == 'S' || sort == 's';

        if (n < 0) return -3;
        if (lda < max(1, n)) return -6;
        if (ldvs < 1 || (wantvs && ldvs < n)) return -11;

        int minwrk = 1;
        if (n == 0) {
            if (lwork != -1) work[workOff] = 1;
            return 0;
        }

        minwrk = max(1, 2 * n);

        int nb = Ilaenv.ilaenv(1, "ZGEHRD", "", n, 1, n, 0);
        int maxwrk = 2 * n + 2 * n * nb;

        Zhseqr.zhseqr('S', wantvs ? 'V' : 'N', n, 0, n - 1,
                null, 0, n, null, 0, null, 0, n, work, workOff, -1);
        int hswork = work[workOff] > 0 ? (int) work[workOff] : 1;

        if (!wantvs) {
            maxwrk = max(maxwrk, 2 * hswork);
        } else {
            int nb2 = Ilaenv.ilaenv(1, "ZUNGHR", "", n, 1, n, -1);
            maxwrk = max(maxwrk, 2 * n + 2 * (n - 1) * nb2);
            maxwrk = max(maxwrk, 2 * hswork);
        }
        minwrk = max(minwrk, maxwrk);

        if (lwork == -1) {
            work[workOff] = minwrk;
            return 0;
        } else if (lwork < minwrk) {
            return -13;
        }

        int sdim = 0;

        double eps = BLAS.dlamch('P');
        double smlnum = BLAS.dlamch('S');
        double bignum = 1.0 / smlnum;
        smlnum = sqrt(smlnum) / eps;
        bignum = 1.0 / smlnum;

        // Internal Schur helpers still consume raw interleaved-slot offsets for matrix storage.
        int aOffRaw = aOff * 2;
        int vsOffRaw = vsOff * 2;

        double anrm = Zlange.zlange('M', n, n, A, aOffRaw, lda, rwork);
        boolean scalea = false;
        double cscale = 0;

        if (anrm > 0 && anrm < smlnum) {
            scalea = true;
            cscale = smlnum;
        } else if (anrm > bignum) {
            scalea = true;
            cscale = bignum;
        }

        if (scalea) {
            Zlascl.zlascl('G', 0, 0, anrm, cscale, n, n, A, aOffRaw, lda);
        }

        int ibal = rworkOff;
        // Zgebal writes ilo/ihi and permutation data that must be reused by both the Hessenberg
        // reduction pipeline and the final Zgebak call that undoes the balancing on Schur vectors.
        Zgebal.zgebal('P', n, A, aOffRaw, lda, rwork, ibal, work, workOff);
        int ilo = (int) work[workOff];
        int ihi = (int) work[workOff + 1];

        int itau = workOff;
        int iwrk = workOff + n * 2;

        Zgehrd.zgehrd(n, ilo, ihi, A, aOff, lda, work, itau, work, iwrk, lwork - (iwrk - workOff));

        if (wantvs) {
            // ZUNGHR consumes the Householder vectors from ZGEHRD, so VS needs its own copy.
            Zlacpy.zlacpy('L', n, n, A, aOff, lda, vs, vsOff, ldvs);
            Zlaset.zlaset('U', n, n, 0, 0, 1, 0, vs, vsOff, ldvs);
            Zgees.zorghr(n, ilo, ihi, vs, vsOff, ldvs, work, itau, work, iwrk, lwork - (iwrk - workOff));
        }

        sdim = 0;

        iwrk = itau;
        int ieval = Zhseqr.zhseqr('S', wantvs ? 'V' : 'N', n, ilo, ihi,
        A, aOffRaw, lda, w, wOff, vs, vsOffRaw, ldvs,
                work, iwrk, lwork - (iwrk - workOff));

        int info = 0;
        if (ieval > 0) {
            info = ieval;
        }

        if (wantst && info == 0) {
            if (scalea) {
                Zlascl.zlascl('G', 0, 0, cscale, anrm, n, 1, w, wOff, 1);
            }

            for (int i = 0; i < n; i++) {
                bwork[i] = select.select(w[wOff + i * 2], w[wOff + i * 2 + 1]);
            }

            boolean trsenOk = Ztrsen.ztrsen('N', wantvs ? 'V' : 'N', bwork, n,
            A, aOffRaw, lda, vs, vsOffRaw, ldvs,
                    w, wOff,
                    work, iwrk, lwork - iwrk);

            if (!trsenOk) {
                info = n + 1;
            }

            sdim = 0;
            for (int i = 0; i < n; i++) {
                if (bwork[i]) sdim++;
            }
        }

        if (wantvs) {
            Zgebak.zgebak('P', BLAS.Side.Right, n, ilo, ihi, rwork, ibal, n, vs, vsOffRaw, ldvs);
        }

        if (scalea) {
            Zlascl.zlascl('U', 0, 0, cscale, anrm, n, n, A, aOffRaw, lda);
            for (int i = 0; i < n; i++) {
                int diagPos = (aOff + i * lda + i) * 2;
                w[wOff + i * 2] = A[diagPos];
                w[wOff + i * 2 + 1] = A[diagPos + 1];
            }
        }

        work[workOff] = sdim;
        return info;
    }

    static int zorghr(int n, int ilo, int ihi, double[] A, int aOff, int lda,
                     double[] tau, int tOff, double[] work, int wOff, int lwork) {
        if (n == 0) {
            return 0;
        }

        int nh = ihi - ilo;
        int nb = Ilaenv.ilaenv(1, "ZUNGHR", "", n, ilo + 1, ihi + 1, -1);
        int lworkopt = Math.max(1, nh * nb * 2);

        if (lwork == -1) {
            work[wOff] = lworkopt;
            return 0;
        }

        if (ilo < 0 || ilo > Math.max(0, n - 1)) {
            return -2;
        }
        if (ihi < Math.min(ilo, n - 1) || ihi > n - 1) {
            return -3;
        }
        if (lda < Math.max(1, n)) {
            return -5;
        }
        if (lwork < Math.max(1, nh * 2)) {
            return -8;
        }

        for (int j = ihi; j >= ilo + 1; j--) {
            for (int i = 0; i < j; i++) {
                int pos = (aOff + i * lda + j) * 2;
                A[pos] = 0.0;
                A[pos + 1] = 0.0;
            }
            for (int i = j + 1; i <= ihi; i++) {
                int pos = (aOff + i * lda + j) * 2;
                int srcPos = (aOff + i * lda + j - 1) * 2;
                A[pos] = A[srcPos];
                A[pos + 1] = A[srcPos + 1];
            }
            for (int i = ihi + 1; i < n; i++) {
                int pos = (aOff + i * lda + j) * 2;
                A[pos] = 0.0;
                A[pos + 1] = 0.0;
            }
        }

        for (int j = 0; j <= ilo; j++) {
            for (int i = 0; i < n; i++) {
                int pos = (aOff + i * lda + j) * 2;
                A[pos] = 0.0;
                A[pos + 1] = 0.0;
            }
            int diagPos = (aOff + j * lda + j) * 2;
            A[diagPos] = 1.0;
            A[diagPos + 1] = 0.0;
        }

        for (int j = ihi + 1; j < n; j++) {
            for (int i = 0; i < n; i++) {
                int pos = (aOff + i * lda + j) * 2;
                A[pos] = 0.0;
                A[pos + 1] = 0.0;
            }
            int diagPos = (aOff + j * lda + j) * 2;
            A[diagPos] = 1.0;
            A[diagPos + 1] = 0.0;
        }

        if (nh > 0) {
            int subAOff = aOff + (ilo + 1) * lda + (ilo + 1);
            int subTauOff = tOff + ilo * 2;
            Zgeqr.zorgqr(nh, nh, nh, A, subAOff, lda, tau, subTauOff, work, wOff, lwork);
        }

        work[wOff] = lworkopt;
        return 0;
    }
}
