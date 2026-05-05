package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;

interface Zgeev {

    static int zgeev(char jobvl, char jobvr, int n, double[] A, int lda,
                     double[] wr, double[] wi,
                     double[] vl, int ldvl, double[] vr, int ldvr,
                     double[] work, int workOff, int lwork) {
        if (n == 0) {
            return 0;
        }

        boolean wantvl = Character.toUpperCase(jobvl) == 'V';
        boolean wantvr = Character.toUpperCase(jobvr) == 'V';

        char upperJobvl = Character.toUpperCase(jobvl);
        char upperJobvr = Character.toUpperCase(jobvr);
        if (upperJobvl != 'N' && upperJobvl != 'V') {
            return -1;
        }
        if (upperJobvr != 'N' && upperJobvr != 'V') {
            return -2;
        }
        if (n < 0) {
            return -3;
        }
        if (lda < Math.max(1, n)) {
            return -5;
        }
        if (wantvl && ldvl < Math.max(1, n)) {
            return -9;
        }
        if (wantvr && ldvr < Math.max(1, n)) {
            return -11;
        }

        int nb = Ilaenv.ilaenv(1, "ZGEHRD", "", n, 1, n, -1);
        int lworkopt = Math.max(1, (nb + 1) * n * 2);
        int minwrk = Math.max(1, 4 * n * 2);

        if (lwork == -1) {
            work[workOff] = Math.max(lworkopt, minwrk);
            return 0;
        }

        if (lwork < minwrk) {
            return -13;
        }

        if (n == 1) {
            wr[0] = A[0];
            wi[0] = A[1];
            work[workOff] = 8;
            if (wantvl) {
                vl[0] = 1.0;
                vl[1] = 0.0;
            }
            if (wantvr) {
                vr[0] = 1.0;
                vr[1] = 0.0;
            }
            return 0;
        }

        double safmin = BLAS.safmin();
        double eps = BLAS.eps();
        double smlnum = safmin / eps;
        double bignum = 1 / smlnum;

        double anrm = Zlange.zlange('M', n, n, A, 0, lda, work);
        boolean scaled = false;
        double sigma = 1.0;

        if (anrm > 0 && anrm < smlnum) {
            scaled = true;
            sigma = smlnum / anrm;
        } else if (anrm > bignum) {
            scaled = true;
            sigma = bignum / anrm;
        }

        if (scaled) {
            Zlascl.zlascl('G', 0, 0, 1.0, sigma, n, n, A, 0, lda);
        }

        char jobvs = (wantvl || wantvr) ? 'V' : 'N';
        double[] eigenvalues = new double[n * 2];
        double[] vs = null;
        int ldvs = n;
        if (wantvl || wantvr) {
            vs = new double[n * n * 2];
        }
        double[] rwork = new double[n];
        boolean[] bwork = null;

        int info = Zgees.zgees(jobvs, 'N', null, n, A, 0, lda,
                eigenvalues, 0, vs, 0, ldvs,
                work, workOff, lwork, rwork, bwork);

        for (int i = 0; i < n; i++) {
            wr[i] = eigenvalues[i * 2];
            wi[i] = eigenvalues[i * 2 + 1];
        }

        if (info != 0) {
            if (scaled) {
                for (int i = 0; i < n; i++) {
                    wr[i] /= sigma;
                    wi[i] /= sigma;
                }
            }
            return info;
        }

        if (scaled) {
            for (int i = 0; i < n; i++) {
                wr[i] /= sigma;
                wi[i] /= sigma;
            }
        }

        if (wantvl || wantvr) {
            double[] tmpWork = new double[n * 2];
            double[] vsSave = null;
            if (wantvl && wantvr) {
                // ZTREVC backtransforms in-place, so keep an untouched Schur basis for the second pass.
                vsSave = new double[n * n * 2];
                System.arraycopy(vs, 0, vsSave, 0, n * n * 2);
            }

            if (wantvr) {
                Ztrevc.ztrevc('R', 'B', n, A, 0, lda, null, 0, 1, vs, 0, ldvs, n, tmpWork, 0);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        int pos = i * ldvr * 2 + j * 2;
                        int zpos = i * ldvs * 2 + j * 2;
                        vr[pos] = vs[zpos];
                        vr[pos + 1] = vs[zpos + 1];
                    }
                }
            }

            if (wantvl) {
                double[] vsl = wantvr ? vsSave : vs;
                if (!wantvr) {
                    vsl = new double[n * n * 2];
                    System.arraycopy(vs, 0, vsl, 0, n * n * 2);
                }
                Ztrevc.ztrevc('L', 'B', n, A, 0, lda, vsl, 0, n, null, 0, 1, n, tmpWork, 0);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        int pos = i * ldvl * 2 + j * 2;
                        int zpos = i * n * 2 + j * 2;
                        vl[pos] = vsl[zpos];
                        vl[pos + 1] = vsl[zpos + 1];
                    }
                }
            }
        }

        work[workOff] = lworkopt;
        return 0;
    }
}
