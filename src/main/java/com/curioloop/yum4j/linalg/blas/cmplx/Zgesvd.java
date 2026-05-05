/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;

import static java.lang.Math.*;

interface Zgesvd {

    // 复数版本的辅助函数
    static void zlacpy(char uplo, int m, int n, double[] A, int aOff, int lda, double[] B, int bOff, int ldb) {
        Zlacpy.zlacpy(uplo, m, n, A, aOff, lda, B, bOff, ldb);
    }

    static void zlaset(char uplo, int m, int n, double alphaRe, double alphaIm, double betaRe, double betaIm, double[] A, int aOff, int lda) {
        Zlaset.zlaset(uplo, m, n, alphaRe, alphaIm, betaRe, betaIm, A, aOff, lda);
    }

    static int rawMatrixOff(int matrixOff) {
        return matrixOff * 2;
    }

    static void zgemm(BLAS.Trans transA, BLAS.Trans transB, int m, int n, int k, double alphaRe, double alphaIm, double[] A, int aOff, int lda, double[] B, int bOff, int ldb, double betaRe, double betaIm, double[] C, int cOff, int ldc) {
        Zgemm.zgemm(transA, transB, m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, betaRe, betaIm, C, cOff, ldc);
    }


    static int zgesvd(char jobU, char jobVT, int m, int n,
                      double[] A, int aOff, int lda,
                      double[] s, int sOff,
                      double[] u, int uOff, int ldu,
                      double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork) {
        boolean wantua = jobU == 'A' || jobU == 'a';
        boolean wantus = jobU == 'S' || jobU == 's';
        boolean wantuas = wantua || wantus;
        boolean wantuo = jobU == 'O' || jobU == 'o';
        boolean wantun = jobU == 'N' || jobU == 'n';

        boolean wantva = jobVT == 'A' || jobVT == 'a';
        boolean wantvs = jobVT == 'S' || jobVT == 's';
        boolean wantvas = wantva || wantvs;
        boolean wantvo = jobVT == 'O' || jobVT == 'o';
        boolean wantvn = jobVT == 'N' || jobVT == 'n';

        if (jobU != 'A' && jobU != 'a' && jobU != 'S' && jobU != 's' && jobU != 'O' && jobU != 'o' && jobU != 'N' && jobU != 'n') {
            return -1;
        }
        if (jobVT != 'A' && jobVT != 'a' && jobVT != 'S' && jobVT != 's' && jobVT != 'O' && jobVT != 'o' && jobVT != 'N' && jobVT != 'n') {
            return -2;
        }

        if (wantuo && wantvo) {
            return -1;
        }

        int minmn = min(m, n);
        int minwrk = 1;

        if (m < 0) return -3;
        if (n < 0) return -4;
        if (lda < max(1, n)) return -6;
        if (ldu < 1 || (wantua && ldu < m) || (wantus && ldu < minmn)) return -9;
        if (ldvt < 1 || (wantvas && ldvt < n)) return -11;
        if (minmn == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        int mnthr = Ilaenv.ilaenv(6, "ZGESVD", String.valueOf(jobU) + jobVT, m, n, 0, 0);

        int maxwrk = 1;
        int wrkbl = 0;
        int bdspac = 0;

        // Mirror LAPACK's tall-vs-wide split first, then pick the QR/LQ-to-bidiagonal path
        // that matches the requested singular vectors and the available workspace.
        if (m >= n) {
            bdspac = 5 * n;
            int lworkZgeqrf = queryZgeqrf(m, n, work, workOff);
            int lworkZorgqrN = queryZorgqr(m, n, n, work, workOff);
            int lworkZorgqrM = queryZorgqr(m, m, n, work, workOff);
            int lworkZgebrd = queryZgebrd(n, n, work, workOff);
            int lworkZorgbrP = queryZorgbr('P', n, n, n, work, workOff);
            int lworkZorgbrQ = queryZorgbr('Q', n, n, n, work, workOff);

            if (m >= mnthr) {
                // For a tall matrix, reduce through QR first so the blocked bidiagonal path stays efficient.
                if (wantun) {
                    maxwrk = n + lworkZgeqrf;
                    maxwrk = max(maxwrk, 3 * n + lworkZgebrd);
                    if (wantvo || wantvas) {
                        maxwrk = max(maxwrk, 3 * n + lworkZorgbrP);
                    }
                    maxwrk = max(maxwrk, bdspac);
                    minwrk = max(4 * n, bdspac);
                } else if (wantuo && wantvn) {
                    wrkbl = n + lworkZgeqrf;
                    wrkbl = max(wrkbl, n + lworkZorgqrN);
                    wrkbl = max(wrkbl, 3 * n + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = max(n * n + wrkbl, n * n + m * n + n);
                    minwrk = max(3 * n + m, bdspac);
                } else if (wantuo && wantvas) {
                    wrkbl = n + lworkZgeqrf;
                    wrkbl = max(wrkbl, n + lworkZorgqrN);
                    wrkbl = max(wrkbl, 3 * n + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrQ);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = max(n * n + wrkbl, n * n + m * n + n);
                    minwrk = max(3 * n + m, bdspac);
                } else if (wantus && wantvn) {
                    wrkbl = n + lworkZgeqrf;
                    wrkbl = max(wrkbl, n + lworkZorgqrN);
                    wrkbl = max(wrkbl, 3 * n + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = n * n + wrkbl;
                    minwrk = max(3 * n + m, bdspac);
                } else if (wantus && wantvo) {
                    wrkbl = n + lworkZgeqrf;
                    wrkbl = max(wrkbl, n + lworkZorgqrN);
                    wrkbl = max(wrkbl, 3 * n + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrQ);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = 2 * n * n + wrkbl;
                    minwrk = max(3 * n + m, bdspac);
                } else if (wantus && wantvas) {
                    wrkbl = n + lworkZgeqrf;
                    wrkbl = max(wrkbl, n + lworkZorgqrN);
                    wrkbl = max(wrkbl, 3 * n + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrQ);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = n * n + wrkbl;
                    minwrk = max(3 * n + m, bdspac);
                } else if (wantua && wantvn) {
                    wrkbl = n + lworkZgeqrf;
                    wrkbl = max(wrkbl, n + lworkZorgqrM);
                    wrkbl = max(wrkbl, 3 * n + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = n * n + wrkbl;
                    minwrk = max(3 * n + m, bdspac);
                } else if (wantua && wantvo) {
                    wrkbl = n + lworkZgeqrf;
                    wrkbl = max(wrkbl, n + lworkZorgqrM);
                    wrkbl = max(wrkbl, 3 * n + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrQ);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = 2 * n * n + wrkbl;
                    minwrk = max(3 * n + m, bdspac);
                } else if (wantua && wantvas) {
                    wrkbl = n + lworkZgeqrf;
                    wrkbl = max(wrkbl, n + lworkZorgqrM);
                    wrkbl = max(wrkbl, 3 * n + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrQ);
                    wrkbl = max(wrkbl, 3 * n + lworkZorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = n * n + wrkbl;
                    minwrk = max(3 * n + m, bdspac);
                }
            } else {
                maxwrk = 3 * n + lworkZgebrd;
                if (wantus || wantuo) {
                    maxwrk = max(maxwrk, 3 * n + lworkZorgbrQ);
                }
                if (wantua) {
                    maxwrk = max(maxwrk, 3 * n + lworkZorgbrQ);
                }
                if (!wantvn) {
                    maxwrk = max(maxwrk, 3 * n + lworkZorgbrP);
                }
                maxwrk = max(maxwrk, bdspac);
                minwrk = max(3 * n + m, bdspac);
            }
        } else {
            bdspac = 5 * m;
            int lworkZgelqf = queryZgelqf(m, n, work, workOff);
            int lworkZorglqN = queryZorglq(n, n, m, work, workOff);
            int lworkZorglqM = queryZorglq(m, n, m, work, workOff);
            int lworkZgebrd = queryZgebrd(m, m, work, workOff);
            int lworkZorgbrP = queryZorgbr('P', m, m, m, work, workOff);
            int lworkZorgbrQ = queryZorgbr('Q', m, m, m, work, workOff);

            if (n >= mnthr) {
                // For a wide matrix, reduce through LQ first for the symmetric set of LAPACK paths.
                if (wantvn) {
                    maxwrk = m + lworkZgelqf;
                    maxwrk = max(maxwrk, 3 * m + lworkZgebrd);
                    if (wantuo || wantuas) {
                        maxwrk = max(maxwrk, 3 * m + lworkZorgbrQ);
                    }
                    maxwrk = max(maxwrk, bdspac);
                    minwrk = max(4 * m, bdspac);
                } else if (wantvo && wantun) {
                    wrkbl = m + lworkZgelqf;
                    wrkbl = max(wrkbl, m + lworkZorglqM);
                    wrkbl = max(wrkbl, 3 * m + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = max(m * m + wrkbl, m * m + m * n + m);
                    minwrk = max(3 * m + n, bdspac);
                } else if (wantvo && wantuas) {
                    wrkbl = m + lworkZgelqf;
                    wrkbl = max(wrkbl, m + lworkZorglqM);
                    wrkbl = max(wrkbl, 3 * m + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrP);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = max(m * m + wrkbl, m * m + m * n + m);
                    minwrk = max(3 * m + n, bdspac);
                } else if (wantvs && wantun) {
                    wrkbl = m + lworkZgelqf;
                    wrkbl = max(wrkbl, m + lworkZorglqM);
                    wrkbl = max(wrkbl, 3 * m + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = m * m + wrkbl;
                    minwrk = max(3 * m + n, bdspac);
                } else if (wantvs && wantuo) {
                    wrkbl = m + lworkZgelqf;
                    wrkbl = max(wrkbl, m + lworkZorglqM);
                    wrkbl = max(wrkbl, 3 * m + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrP);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = 2 * m * m + wrkbl;
                    minwrk = max(3 * m + n, bdspac);
                } else if (wantvs && wantuas) {
                    wrkbl = m + lworkZgelqf;
                    wrkbl = max(wrkbl, m + lworkZorglqM);
                    wrkbl = max(wrkbl, 3 * m + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrP);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = m * m + wrkbl;
                    minwrk = max(3 * m + n, bdspac);
                } else if (wantva && wantun) {
                    wrkbl = m + lworkZgelqf;
                    wrkbl = max(wrkbl, m + lworkZorglqN);
                    wrkbl = max(wrkbl, 3 * m + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = m * m + wrkbl;
                    minwrk = max(3 * m + n, bdspac);
                } else if (wantva && wantuo) {
                    wrkbl = m + lworkZgelqf;
                    wrkbl = max(wrkbl, m + lworkZorglqN);
                    wrkbl = max(wrkbl, 3 * m + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrP);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = 2 * m * m + wrkbl;
                    minwrk = max(3 * m + n, bdspac);
                } else if (wantva && wantuas) {
                    wrkbl = m + lworkZgelqf;
                    wrkbl = max(wrkbl, m + lworkZorglqN);
                    wrkbl = max(wrkbl, 3 * m + lworkZgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrP);
                    wrkbl = max(wrkbl, 3 * m + lworkZorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = m * m + wrkbl;
                    minwrk = max(3 * m + n, bdspac);
                }
            } else {
                maxwrk = 3 * m + lworkZgebrd;
                if (wantvs || wantvo) {
                    maxwrk = max(maxwrk, 3 * m + lworkZorgbrP);
                }
                if (wantva) {
                    maxwrk = max(maxwrk, 3 * m + lworkZorgbrP);
                }
                if (!wantun) {
                    maxwrk = max(maxwrk, 3 * m + lworkZorgbrQ);
                }
                maxwrk = max(maxwrk, bdspac);
                minwrk = max(3 * m + n, bdspac);
            }
        }

        maxwrk = Math.max(maxwrk, minwrk);

        if (lwork == -1) {
            work[workOff] = maxwrk;
            return 0;
        }
        if (lwork < minwrk) return -13;

        double eps = BLAS.eps();
        double smlnum = sqrt(BLAS.safmin()) / eps;
        double bignum = 1.0 / smlnum;

        double anrm = Zlange.zlange('M', m, n, A, aOff, lda, work);
        boolean iscl = false;
        if (anrm > 0 && anrm < smlnum) {
            iscl = true;
            Zlascl.zlascl('G', 0, 0, anrm, smlnum, m, n, A, aOff, lda);
        } else if (anrm > bignum) {
            iscl = true;
            Zlascl.zlascl('G', 0, 0, anrm, bignum, m, n, A, aOff, lda);
        }

        boolean ok = true;
        int ie = 0;

        if (m >= n) {
            if (m >= mnthr) {
                if (wantun) {
                    ie = wantvo
                        ? path1(m, n, A, aOff, lda, s, sOff, vt, vtOff, ldvt,
                            work, workOff, lwork, true, false)
                            : path1(m, n, A, aOff, lda, s, sOff, vt, vtOff, ldvt,
                                    work, workOff, lwork, false, wantvas);
                    ok = ie >= 0;
                    if (ie < 0) ie = 0;
                } else if (wantuo) {
                    ie = path23(m, n, A, aOff, lda, s, sOff,
                            vt, vtOff, ldvt, work, workOff, lwork, wantvas);
                    ok = ie >= 0;
                    if (ie < 0) ie = 0;
                } else if (wantus) {
                    if (wantvo) {
                    ie = path58(m, n, A, aOff, lda, s, sOff,
                                u, uOff, ldu, work, workOff, lwork, 'S');
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantvn) {
                        ie = path4(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                   work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantvas) {
                        ie = path6(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                   work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    }
                } else if (wantua) {
                    if (wantvo) {
                    ie = path58(m, n, A, aOff, lda, s, sOff,
                                u, uOff, ldu, work, workOff, lwork, 'A');
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantvn) {
                        ie = path7(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                   work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantvas) {
                        ie = path9(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                   work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    }
                }
            } else {
                ie = path10(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                            work, workOff, lwork, wantuas, wantua, wantuo, wantvas, wantva, wantvo);
                ok = ie >= 0;
                if (ie < 0) ie = 0;
            }
        } else {
            if (n >= mnthr) {
                if (wantvn) {
                    ie = wantuo
                        ? path1t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu,
                            work, workOff, lwork, true, false)
                            : path1t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu,
                                    work, workOff, lwork, false, wantuas);
                    ok = ie >= 0;
                    if (ie < 0) ie = 0;
                } else if (wantvo) {
                    ie = path2t3t(m, n, A, aOff, lda, s, sOff,
                            u, uOff, ldu, work, workOff, lwork, wantuas);
                    ok = ie >= 0;
                    if (ie < 0) ie = 0;
                } else if (wantvs) {
                    if (wantuo) {
                    ie = path5t8t(m, n, A, aOff, lda, s, sOff,
                                vt, vtOff, ldvt, work, workOff, lwork, 'S');
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantun) {
                        ie = path4t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                    work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantuas) {
                        ie = path6t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                    work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    }
                } else if (wantva) {
                    if (wantuo) {
                    ie = path5t8t(m, n, A, aOff, lda, s, sOff,
                                vt, vtOff, ldvt, work, workOff, lwork, 'A');
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantun) {
                        ie = path7t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                    work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantuas) {
                        ie = path9t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                    work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    }
                }
            } else {
                ie = path10t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                             work, workOff, lwork, wantuas, wantuo, wantvas, wantva, wantvo);
                ok = ie >= 0;
                if (ie < 0) ie = 0;
            }
        }

        if (!ok) {
            if (ie > 1) {
                for (int i = 0; i < minmn - 1; i++) {
                    work[workOff + i + 1] = work[workOff + i + ie];
                }
            }
            if (ie < 1) {
                for (int i = minmn - 2; i >= 0; i--) {
                    work[workOff + i + 1] = work[workOff + i + ie];
                }
            }
        }

        if (iscl) {
            if (anrm > bignum) {
                Zlascl.zlascl('G', 0, 0, bignum, anrm, 1, minmn, s, sOff, 1);
            }
            if (!ok && anrm > bignum) {
                Zlascl.zlascl('G', 0, 0, bignum, anrm, 1, minmn - 1, work, workOff + 1, 1);
            }
            if (anrm < smlnum) {
                Zlascl.zlascl('G', 0, 0, smlnum, anrm, 1, minmn, s, sOff, 1);
            }
            if (!ok && anrm < smlnum) {
                Zlascl.zlascl('G', 0, 0, smlnum, anrm, 1, minmn - 1, work, workOff + 1, 1);
            }
        }

        work[workOff] = maxwrk;
        return ok ? 0 : 1;
    }

    // Query functions for workspace - directly compute using ILAENV, don't call actual LAPACK functions
    static int queryZgeqrf(int m, int n, double[] work, int workOff) {
        int nb = Ilaenv.ilaenv(1, "ZGEQRF", "", m, n, -1, -1);
        if (nb < 1) nb = 1;
        work[workOff] = max(1, n) * nb;
        return 0;
    }

    static int queryZorgqr(int m, int n, int k, double[] work, int workOff) {
        int nb = Ilaenv.ilaenv(1, "ZORGQR", "", m, n, k, -1);
        if (nb < 1) nb = 1;
        if (nb > k) nb = k;
        work[workOff] = nb * n;
        return 0;
    }

    static int queryZgebrd(int m, int n, double[] work, int workOff) {
        int nb = Ilaenv.ilaenv(1, "ZGEBRD", "", m, n, -1, -1);
        if (nb < 1) nb = 1;
        work[workOff] = (m + n) * nb;
        return 0;
    }

    static int queryZorgbr(char vect, int m, int n, int k, double[] work, int workOff) {
        int nb;
        if (vect == 'Q' || vect == 'q') {
            nb = Ilaenv.ilaenv(1, "ZUNGBR", "Q", m, n, k, -1);
        } else {
            nb = Ilaenv.ilaenv(1, "ZUNGBR", "P", m, n, k, -1);
        }
        if (nb < 1) nb = 1;
        if (nb > k) nb = k;
        work[workOff] = nb * n;
        return 0;
    }

    static int queryZgelqf(int m, int n, double[] work, int workOff) {
        int nb = Ilaenv.ilaenv(1, "ZGELQF", "", m, n, -1, -1);
        if (nb < 1) nb = 1;
        work[workOff] = max(1, m) * nb;
        return 0;
    }

    static int queryZorglq(int m, int n, int k, double[] work, int workOff) {
        int nb = Ilaenv.ilaenv(1, "ZORGLQ", "", m, n, k, -1);
        if (nb < 1) nb = 1;
        if (nb > k) nb = k;
        work[workOff] = nb * m;
        return 0;
    }

    // Row-major translation of Netlib ZGESVD paths 2/3.
    static int path23(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                                  double[] vt, int vtOff, int ldvt,
                                  double[] work, int workOff, int lwork, boolean wantVT) {
        int bdspac = 5 * n;
        if (lwork < n * n + max(4 * n, bdspac)) {
            if (!wantVT) {
            int ie = 0;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            int iwork = itaup + n * 2;

            Zgebrd.zgebrd(m, n, A, aOff, lda, s, sOff, work, workOff + ie,
                work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Zorgbr.zorgbr('Q', m, n, n, A, aOff, lda, work, workOff + itauq,
                work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, 0, m, 0, s, sOff, work, workOff + ie, work, workOff, 1, A, rawMatrixOff(aOff), lda, work, workOff, 1, work, workOff + iwork);
            return ieRet >= 0 ? ie : -ie;
            }

            int itau = 0;
            int iwork = itau + n * 2;

            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('U', n, n, A, aOff, lda, vt, vtOff, ldvt);
            if (n > 1) {
            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, vt, vtOff + ldvt, ldvt);
            }

            Zgeqr.zorgqr(m, n, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            iwork = itaup + n * 2;

            Zgebrd.zgebrd(n, n, vt, vtOff, ldvt, s, sOff, work, workOff + ie,
                work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Zormbr.zormbr('Q', 'R', 'N', m, n, n, vt, vtOff, ldvt, work, workOff + itauq,
                A, aOff, lda, work, workOff + iwork, lwork - iwork);
            Zorgbr.zorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, n, m, 0, s, sOff, work, workOff + ie,
                vt, rawMatrixOff(vtOff), ldvt, A, rawMatrixOff(aOff), lda, work, workOff, 1, work, workOff + iwork);
            return ieRet >= 0 ? ie : -ie;
        }

        int iu = 0;
        int itau = iu + n * n * 2;
        int iwork = itau + n * 2;

        Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

        zlacpy('U', n, n, A, aOff, lda, work, workOff + iu, n);
        if (n > 1) {
            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + iu + n, n);
        }

        Zgeqr.zorgqr(m, n, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

        int ie = itau;
        int itauq = ie + n;
        int itaup = itauq + n * 2;
        iwork = itaup + n * 2;

        Zgebrd.zgebrd(n, n, work, workOff + iu, n, s, sOff, work, workOff + ie,
            work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

        if (wantVT) {
            zlacpy('U', n, n, work, workOff + iu, n, vt, vtOff, ldvt);
            Zorgbr.zorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                work, workOff + iwork, lwork - iwork);
        }

        Zorgbr.zorgbr('Q', n, n, n, work, workOff + iu, n, work, workOff + itauq,
            work, workOff + iwork, lwork - iwork);

        iwork = ie + n;
        int ieRet = Zbdsqr.zbdsqr('U', n, wantVT ? n : 0, n, 0, s, sOff, work, workOff + ie,
            wantVT ? vt : work, wantVT ? rawMatrixOff(vtOff) : workOff, wantVT ? ldvt : 1,
            work, workOff + iu, n, work, workOff, 1, work, workOff + iwork);

        int ir = iwork;
        int chunk = max(1, (lwork - ir) / n);
        for (int row = 0; row < m; row += chunk) {
            int rows = min(chunk, m - row);
            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, rows, n, n, 1.0, 0.0, A, aOff + row * lda, lda, work, workOff + iu, n, 0.0, 0.0, work, workOff + ir, n);
            zlacpy('A', rows, n, work, workOff + ir, n, A, aOff + row * lda, lda);
        }

        return ieRet >= 0 ? ie : -ie;
    }

    // Row-major translation of Netlib ZGESVD paths 5/8.
    static int path58(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                                   double[] u, int uOff, int ldu,
                                   double[] work, int workOff, int lwork, char jobU) {
        int bdspac = 5 * n;
        if (lwork < 2 * n * n + max(max(n + m, 4 * n), bdspac)) {
            int itau = 0;
            int iwork = itau + n * 2;

            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);
            Zgeqr.zorgqr(m, jobU == 'A' || jobU == 'a' ? m : n, n, u, uOff, ldu, work, workOff + itau,
                work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            iwork = itaup + n * 2;

            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, A, aOff + lda, lda);
            Zgebrd.zgebrd(n, n, A, aOff, lda, s, sOff, work, workOff + ie,
                work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Zormbr.zormbr('Q', 'R', 'N', m, n, n, A, aOff, lda, work, workOff + itauq,
                u, uOff, ldu, work, workOff + iwork, lwork - iwork);
            Zorgbr.zorgbr('P', n, n, n, A, aOff, lda, work, workOff + itaup,
                work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, n, m, 0, s, sOff, work, workOff + ie,
                A, rawMatrixOff(aOff), lda, u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);
            return ieRet >= 0 ? ie : -ie;
        }

        int iu = 0;
        int ir = iu + n * n * 2;
        int itau = ir + n * n * 2;
        int iwork = itau + n * 2;

        if (jobU == 'A' || jobU == 'a') {
            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);
            Zgeqr.zorgqr(m, m, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);
        } else {
            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
        }

        zlacpy('U', n, n, A, aOff, lda, work, workOff + iu, n);
        if (n > 1) {
            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + iu + n, n);
        }

        if (jobU == 'S' || jobU == 's') {
            Zgeqr.zorgqr(m, n, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
        }

        int ie = itau;
        int itauq = ie + n;
        int itaup = itauq + n * 2;
        iwork = itaup + n * 2;

        Zgebrd.zgebrd(n, n, work, workOff + iu, n, s, sOff, work, workOff + ie,
                work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
        zlacpy('U', n, n, work, workOff + iu, n, work, workOff + ir, n);

        Zorgbr.zorgbr('Q', n, n, n, work, workOff + iu, n, work, workOff + itauq,
                work, workOff + iwork, lwork - iwork);
        Zorgbr.zorgbr('P', n, n, n, work, workOff + ir, n, work, workOff + itaup,
                work, workOff + iwork, lwork - iwork);
        iwork = ie + n;

        int ieRet = Zbdsqr.zbdsqr('U', n, n, n, 0, s, sOff, work, workOff + ie,
                work, workOff + ir, n, work, workOff + iu, n, work, workOff, 1, work, workOff + iwork);

        if (jobU == 'A' || jobU == 'a') {
            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0,
                    u, uOff, ldu, work, workOff + iu, n, 0.0, 0.0, A, aOff, lda);
            zlacpy('A', m, n, A, aOff, lda, u, uOff, ldu);
        } else {
            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0,
                    A, aOff, lda, work, workOff + iu, n, 0.0, 0.0, u, uOff, ldu);
        }
        zlacpy('A', n, n, work, workOff + ir, n, A, aOff, lda);

        return ieRet >= 0 ? ie : -ie;
    }

    // Row-major translation of Netlib ZGESVD paths 2t/3t.
    static int path2t3t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                                   double[] u, int uOff, int ldu,
                                   double[] work, int workOff, int lwork, boolean wantU) {
        int bdspac = 5 * m;
        if (lwork < m * m + max(4 * m, bdspac)) {
            if (!wantU) {
            int ie = 0;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            int iwork = itaup + m * 2;

            Zgebrd.zgebrd(m, n, A, aOff, lda, s, sOff, work, workOff + ie,
                work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Zorgbr.zorgbr('P', m, n, m, A, aOff, lda, work, workOff + itaup,
                work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('L', m, n, 0, 0, s, sOff, work, workOff + ie,
                A, rawMatrixOff(aOff), lda, work, workOff, 1, work, workOff, 1, work, workOff + iwork);
            return ieRet >= 0 ? ie : -ie;
            }

            int itau = 0;
            int iwork = itau + m * 2;

            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, m, A, aOff, lda, u, uOff, ldu);
            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, u, uOff + 1, ldu);
            Zgelq.zorglq(m, n, m, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            iwork = itaup + m * 2;

            Zgebrd.zgebrd(m, m, u, uOff, ldu, s, sOff, work, workOff + ie,
                work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Zormbr.zormbr('P', 'L', 'C', m, n, m, u, uOff, ldu, work, workOff + itaup,
                A, aOff, lda, work, workOff + iwork, lwork - iwork);
            Zorgbr.zorgbr('Q', m, m, m, u, uOff, ldu, work, workOff + itauq,
                work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('U', m, n, m, 0, s, sOff, work, workOff + ie,
                A, rawMatrixOff(aOff), lda, u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);
            return ieRet >= 0 ? ie : -ie;
        }

        int ir = 0;
        int itau = ir + m * m * 2;
        int iwork = itau + m * 2;

        Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

        if (wantU) {
            zlacpy('L', m, m, A, aOff, lda, u, uOff, ldu);
            if (m > 1) {
                zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, u, uOff + 1, ldu);
            }
        } else {
            zlacpy('L', m, m, A, aOff, lda, work, workOff + ir, m);
            if (m > 1) {
                zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + ir + 1, m);
            }
        }

        Zgelq.zorglq(m, n, m, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

        int ie = itau;
        int itauq = ie + m;
        int itaup = itauq + m * 2;
        iwork = itaup + m * 2;

        if (wantU) {
            Zgebrd.zgebrd(m, m, u, uOff, ldu, s, sOff, work, workOff + ie,
                    work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, m, u, uOff, ldu, work, workOff + ir, m);
            Zorgbr.zorgbr('P', m, m, m, work, workOff + ir, m, work, workOff + itaup,
                    work, workOff + iwork, lwork - iwork);
            Zorgbr.zorgbr('Q', m, m, m, u, uOff, ldu, work, workOff + itauq,
                    work, workOff + iwork, lwork - iwork);
        } else {
            Zgebrd.zgebrd(m, m, work, workOff + ir, m, s, sOff, work, workOff + ie,
                    work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Zorgbr.zorgbr('P', m, m, m, work, workOff + ir, m, work, workOff + itaup,
                    work, workOff + iwork, lwork - iwork);
        }

        iwork = ie + m;
        int ieRet = Zbdsqr.zbdsqr('U', m, m, wantU ? m : 0, 0, s, sOff, work, workOff + ie,
            work, workOff + ir, m, wantU ? u : work, wantU ? rawMatrixOff(uOff) : workOff, wantU ? ldu : 1,
                work, workOff, 1, work, workOff + iwork);

        int iu = iwork;
        int chunk = max(1, (lwork - iu) / m);
        for (int col = 0; col < n; col += chunk) {
            int cols = min(chunk, n - col);
            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, cols, m, 1.0, 0.0,
                    work, workOff + ir, m, A, aOff + col * 2, lda, 0.0, 0.0,
                    work, workOff + iu, cols);
            zlacpy('A', m, cols, work, workOff + iu, cols, A, aOff + col, lda);
        }

        return ieRet >= 0 ? ie : -ie;
    }

    // Row-major translation of Netlib ZGESVD paths 5t/8t.
    static int path5t8t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                                  double[] vt, int vtOff, int ldvt,
                                  double[] work, int workOff, int lwork, char jobVT) {
        int bdspac = 5 * m;
        if (lwork < 2 * m * m + max(max(m + n, 4 * m), bdspac)) {
            int itau = 0;
            int iwork = itau + m * 2;

            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);
            Zgelq.zorglq(jobVT == 'A' || jobVT == 'a' ? n : m, n, m, vt, vtOff, ldvt, work, workOff + itau,
                work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            iwork = itaup + m * 2;

            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, A, aOff + 1, lda);
            Zgebrd.zgebrd(m, m, A, aOff, lda, s, sOff, work, workOff + ie,
                work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Zormbr.zormbr('P', 'L', 'C', m, n, m, A, aOff, lda, work, workOff + itaup,
                vt, vtOff, ldvt, work, workOff + iwork, lwork - iwork);
            Zorgbr.zorgbr('Q', m, m, m, A, aOff, lda, work, workOff + itauq,
                work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('U', m, n, m, 0, s, sOff, work, workOff + ie,
                vt, rawMatrixOff(vtOff), ldvt, A, rawMatrixOff(aOff), lda, work, workOff, 1, work, workOff + iwork);
            return ieRet >= 0 ? ie : -ie;
        }

        int iu = 0;
        int ir = iu + m * m * 2;
        int itau = ir + m * m * 2;
        int iwork = itau + m * 2;

        if (jobVT == 'A' || jobVT == 'a') {
            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);
            Zgelq.zorglq(n, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);
        } else {
            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
        }

        zlacpy('L', m, m, A, aOff, lda, work, workOff + iu, m);
        if (m > 1) {
            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + iu + 1, m);
        }

        if (jobVT == 'S' || jobVT == 's') {
            Zgelq.zorglq(m, n, m, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
        }

        int ie = itau;
        int itauq = ie + m;
        int itaup = itauq + m * 2;
        iwork = itaup + m * 2;

        Zgebrd.zgebrd(m, m, work, workOff + iu, m, s, sOff, work, workOff + ie,
                work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
        zlacpy('L', m, m, work, workOff + iu, m, work, workOff + ir, m);

        Zorgbr.zorgbr('P', m, m, m, work, workOff + iu, m, work, workOff + itaup,
                work, workOff + iwork, lwork - iwork);
        Zorgbr.zorgbr('Q', m, m, m, work, workOff + ir, m, work, workOff + itauq,
                work, workOff + iwork, lwork - iwork);
        iwork = ie + m;

        int ieRet = Zbdsqr.zbdsqr('U', m, m, m, 0, s, sOff, work, workOff + ie,
                work, workOff + iu, m, work, workOff + ir, m, work, workOff, 1, work, workOff + iwork);

        if (jobVT == 'A' || jobVT == 'a') {
            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, m, 1.0, 0.0,
                    work, workOff + iu, m, vt, vtOff, ldvt, 0.0, 0.0, A, aOff, lda);
            zlacpy('A', m, n, A, aOff, lda, vt, vtOff, ldvt);
        } else {
            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, m, 1.0, 0.0,
                    work, workOff + iu, m, A, aOff, lda, 0.0, 0.0, vt, vtOff, ldvt);
        }
        zlacpy('A', m, m, work, workOff + ir, m, A, aOff, lda);

        return ieRet >= 0 ? ie : -ie;
    }

    static int path1(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                     double[] vt, int vtOff, int ldvt,
                     double[] work, int workOff, int lwork,
                     boolean wantvo, boolean wantvas) {
        int itau = 0;
        int iwork = itau + n * 2;

        Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

        if (n > 1) {
            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, A, aOff + lda, lda);
        }

        int ie = 0;
        int itauq = ie + n;
        int itaup = itauq + n * 2;
        iwork = itaup + n * 2;

        Zgebrd.zgebrd(n, n, A, aOff, lda, s, sOff, work, workOff + ie, work, workOff + itauq,
                      work, workOff + itaup, work, workOff + iwork, lwork - iwork);

        int ncvt = 0;
        if (wantvo || wantvas) {
            Zorgbr.zorgbr('P', n, n, n, A, aOff, lda, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            ncvt = n;
        }
        iwork = ie + n;

        int ieRet = Zbdsqr.zbdsqr('U', n, ncvt, 0, 0, s, sOff, work, workOff + ie, A, rawMatrixOff(aOff), lda,
                                   work, workOff, 1, work, workOff, 1, work, workOff + iwork);

        if (wantvas) {
            zlacpy('A', n, n, A, aOff, lda, vt, vtOff, ldvt);
        }

        return ieRet >= 0 ? ie : -ie;
    }

    static int path4(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                     double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                     double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworkr) {
        if (lwork >= n * n + max(max(4 * n, bdspac), wrkbl)) {
            int ir = 0;
            if (lwork >= wrkbl + ldworkr * n) {
                ldworkr = lda;
            } else {
                ldworkr = n;
            }
            int itau = ir + ldworkr * n * 2;
            int iwork = itau + n * 2;

            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            zlacpy('U', n, n, A, aOff, lda, work, workOff + ir, ldworkr);
            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + ir + ldworkr, ldworkr);

            Zgeqr.zorgqr(m, n, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            iwork = itaup + n * 2;

            Zgebrd.zgebrd(n, n, work, workOff + ir, ldworkr, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('Q', n, n, n, work, workOff + ir, ldworkr, work, workOff + itauq,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, 0, n, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                               work, workOff + ir, ldworkr, work, workOff, 1, work, workOff + iwork);

            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0, A, aOff, lda, work, workOff + ir, ldworkr, 0.0, 0.0, u, uOff, ldu);

            return ieRet >= 0 ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + n * 2;

            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Zgeqr.zorgqr(m, n, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            iwork = itaup + n * 2;

            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, A, aOff + lda, lda);

            Zgebrd.zgebrd(n, n, A, aOff, lda, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zormbr.zormbr('Q', 'R', 'N', m, n, n, A, aOff, lda, work, workOff + itauq,
                      u, uOff, ldu, work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, 0, m, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                               u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);

            return ieRet >= 0 ? ie : -ie;
        }
    }

    static int path6(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                     double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                     double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworku) {
        if (lwork >= n * n + max(max(4 * n, bdspac), wrkbl)) {
            int iu = 0;
            if (lwork >= wrkbl + ldworku * n) {
                ldworku = lda;
            } else {
                ldworku = n;
            }
            int itau = iu + ldworku * n * 2;
            int iwork = itau + n * 2;

            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            zlacpy('U', n, n, A, aOff, lda, work, workOff + iu, ldworku);
            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + iu + ldworku, ldworku);

            Zgeqr.zorgqr(m, n, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            iwork = itaup + n * 2;

            Zgebrd.zgebrd(n, n, work, workOff + iu, ldworku, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            zlacpy('U', n, n, work, workOff + iu, ldworku, vt, vtOff, ldvt);

            Zorgbr.zorgbr('Q', n, n, n, work, workOff + iu, ldworku, work, workOff + itauq,
                      work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, n, n, 0, s, sOff, work, workOff + ie, vt, rawMatrixOff(vtOff), ldvt,
                                       work, workOff + iu, ldworku, work, workOff, 1, work, workOff + iwork);

            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0, A, aOff, lda, work, workOff + iu, ldworku, 0.0, 0.0, u, uOff, ldu);

            return ieRet >= 0 ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + n * 2;

            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Zgeqr.zorgqr(m, n, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            zlacpy('U', n, n, A, aOff, lda, vt, vtOff, ldvt);
            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, vt, vtOff + ldvt, ldvt);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            iwork = itaup + n * 2;

            Zgebrd.zgebrd(n, n, vt, vtOff, ldvt, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zormbr.zormbr('Q', 'R', 'N', m, n, n, vt, vtOff, ldvt, work, workOff + itauq,
                      u, uOff, ldu, work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, n, m, 0, s, sOff, work, workOff + ie, vt, rawMatrixOff(vtOff), ldvt,
                                       u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);

            return ieRet >= 0 ? ie : -ie;
        }
    }

    static int path7(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                     double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                     double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworkr) {
        if (lwork >= n * n + max(max(n + m, 4 * n), max(bdspac, wrkbl))) {
            int ir = 0;
            if (lwork >= wrkbl + ldworkr * n) {
                ldworkr = lda;
            } else {
                ldworkr = n;
            }
            int itau = ir + ldworkr * n * 2;
            int iwork = itau + n * 2;

            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            zlacpy('U', n, n, A, aOff, lda, work, workOff + ir, ldworkr);
            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + ir + ldworkr, ldworkr);

            Zgeqr.zorgqr(m, m, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            iwork = itaup + n * 2;

            Zgebrd.zgebrd(n, n, work, workOff + ir, ldworkr, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('Q', n, n, n, work, workOff + ir, ldworkr, work, workOff + itauq,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, 0, n, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                                       work, workOff + ir, ldworkr, work, workOff, 1, work, workOff + iwork);

            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0, u, uOff, ldu, work, workOff + ir, ldworkr, 0.0, 0.0, A, aOff, lda);

            zlacpy('A', m, n, A, aOff, lda, u, uOff, ldu);

            return ieRet >= 0 ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + n * 2;

            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Zgeqr.zorgqr(m, m, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            iwork = itaup + n * 2;

            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, A, aOff + lda, lda);

            Zgebrd.zgebrd(n, n, A, aOff, lda, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zormbr.zormbr('Q', 'R', 'N', m, n, n, A, aOff, lda, work, workOff + itauq,
                      u, uOff, ldu, work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, 0, m, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                                       u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);

            return ieRet >= 0 ? ie : -ie;
        }
    }

    static int path9(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                     double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                     double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworku) {
        if (lwork >= n * n + max(max(n + m, 4 * n), max(bdspac, wrkbl))) {
            int iu = 0;
            if (lwork >= wrkbl + ldworku * n) {
                ldworku = lda;
            } else {
                ldworku = n;
            }
            int itau = iu + ldworku * n * 2;
            int iwork = itau + n * 2;

            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Zgeqr.zorgqr(m, m, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            zlacpy('U', n, n, A, aOff, lda, work, workOff + iu, ldworku);
            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + iu + ldworku, ldworku);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            iwork = itaup + n * 2;

            Zgebrd.zgebrd(n, n, work, workOff + iu, ldworku, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            zlacpy('U', n, n, work, workOff + iu, ldworku, vt, vtOff, ldvt);

            Zorgbr.zorgbr('Q', n, n, n, work, workOff + iu, ldworku, work, workOff + itauq,
                      work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, n, n, 0, s, sOff, work, workOff + ie, vt, rawMatrixOff(vtOff), ldvt,
                                       work, workOff + iu, ldworku, work, workOff, 1, work, workOff + iwork);

            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0, u, uOff, ldu, work, workOff + iu, ldworku, 0.0, 0.0, A, aOff, lda);

            zlacpy('A', m, n, A, aOff, lda, u, uOff, ldu);

            return ieRet >= 0 ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + n * 2;

            Zgeqr.zgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Zgeqr.zorgqr(m, m, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            zlacpy('U', n, n, A, aOff, lda, vt, vtOff, ldvt);
            zlaset('L', n - 1, n - 1, 0.0, 0.0, 0.0, 0.0, vt, vtOff + ldvt, ldvt);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n * 2;
            iwork = itaup + n * 2;

            Zgebrd.zgebrd(n, n, vt, vtOff, ldvt, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zormbr.zormbr('Q', 'R', 'N', m, n, n, vt, vtOff, ldvt, work, workOff + itauq,
                      u, uOff, ldu, work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            int ieRet = Zbdsqr.zbdsqr('U', n, n, m, 0, s, sOff, work, workOff + ie, vt, rawMatrixOff(vtOff), ldvt,
                                       u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);

            return ieRet >= 0 ? ie : -ie;
        }
    }

    static int path10(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork, boolean wantuas, boolean wantua, boolean wantuo, boolean wantvas, boolean wantva, boolean wantvo) {
        int ie = 0;
        int itauq = ie + n;
        int itaup = itauq + n * 2;
        int iwork = itaup + n * 2;

        Zgebrd.zgebrd(m, n, A, aOff, lda, s, sOff, work, workOff + ie, work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

        int ncu = n;
        if (wantuas) {
            zlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);
            if (wantua) ncu = m;
            Zorgbr.zorgbr('Q', m, ncu, n, u, uOff, ldu, work, workOff + itauq, work, workOff + iwork, lwork - iwork);
        }

        if (wantvas) {
            zlacpy('U', n, n, A, aOff, lda, vt, vtOff, ldvt);
            Zorgbr.zorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
        }

        if (wantuo) {
            Zorgbr.zorgbr('Q', m, n, n, A, aOff, lda, work, workOff + itauq, work, workOff + iwork, lwork - iwork);
        }

        if (wantvo) {
            Zorgbr.zorgbr('P', n, n, n, A, aOff, lda, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
        }

        int ncvt = 0, nru = 0;
        if (wantvas || wantvo) ncvt = n;
        if (wantuas || wantuo) nru = m;

        iwork = ie + n;
        int ieRet;
        if (!wantuo && !wantvo) {
            ieRet = Zbdsqr.zbdsqr('U', n, ncvt, nru, 0, s, sOff, work, workOff + ie, vt, rawMatrixOff(vtOff), ldvt, u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);
        } else if (!wantuo) {
            ieRet = Zbdsqr.zbdsqr('U', n, ncvt, nru, 0, s, sOff, work, workOff + ie, A, rawMatrixOff(aOff), lda, u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);
        } else {
            ieRet = Zbdsqr.zbdsqr('U', n, ncvt, nru, 0, s, sOff, work, workOff + ie, vt, rawMatrixOff(vtOff), ldvt, A, rawMatrixOff(aOff), lda, work, workOff, 1, work, workOff + iwork);
        }

        return ieRet >= 0 ? 0 : -1;
    }

    static int path1t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu,
                      double[] work, int workOff, int lwork, boolean wantuo, boolean wantuas) {
        int itau = 0;
        int iwork = itau + m * 2;

        Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

        if (m > 1) {
            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, A, aOff + 1, lda);
        }

        int ie = 0;
        int itauq = ie + m;
        int itaup = itauq + m * 2;
        iwork = itaup + m * 2;

        Zgebrd.zgebrd(m, m, A, aOff, lda, s, sOff, work, workOff + ie, work, workOff + itauq,
                      work, workOff + itaup, work, workOff + iwork, lwork - iwork);

        int ncvu = 0;
        if (wantuo || wantuas) {
            Zorgbr.zorgbr('Q', m, m, m, A, aOff, lda, work, workOff + itauq, work, workOff + iwork, lwork - iwork);
            ncvu = m;
        }
        iwork = ie + m;

        int ieRet = Zbdsqr.zbdsqr('U', m, 0, ncvu, 0, s, sOff, work, workOff + ie, A, rawMatrixOff(aOff), lda,
                                   work, workOff, 1, work, workOff, 1, work, workOff + iwork);

        if (wantuas) {
            zlacpy('A', m, m, A, aOff, lda, u, uOff, ldu);
        }

        return ieRet >= 0 ? ie : -ie;
    }

    static int path4t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworkr) {
        if (lwork >= m * m + max(max(4 * m, bdspac), wrkbl)) {
            int ir = 0;
            if (lwork >= wrkbl + ldworkr * m) {
                ldworkr = lda;
            } else {
                ldworkr = m;
            }
            int itau = ir + ldworkr * m * 2;
            int iwork = itau + m * 2;

            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            zlacpy('L', m, m, A, aOff, lda, work, workOff + ir, ldworkr);
            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + ir + 1, ldworkr);

            Zgelq.zorglq(m, n, m, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            iwork = itaup + m * 2;

            Zgebrd.zgebrd(m, m, work, workOff + ir, ldworkr, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('P', m, m, m, work, workOff + ir, ldworkr, work, workOff + itaup,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('U', m, 0, m, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                                       work, workOff + ir, ldworkr, work, workOff, 1, work, workOff + iwork);

            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, m, 1.0, 0.0, work, workOff + ir, ldworkr, A, aOff, lda, 0.0, 0.0, vt, vtOff, ldvt);

            return ieRet >= 0 ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + m * 2;

            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Zgelq.zorglq(m, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            iwork = itaup + m * 2;

            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, A, aOff + 1, lda);

            Zgebrd.zgebrd(m, m, A, aOff, lda, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zormbr.zormbr('P', 'L', 'C', m, n, m, A, aOff, lda, work, workOff + itaup,
                      vt, vtOff, ldvt, work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('U', m, 0, m, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                                       work, workOff, 1, vt, rawMatrixOff(vtOff), ldvt, work, workOff + iwork);

            return ieRet >= 0 ? ie : -ie;
        }
    }

    static int path6t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworkv) {
        if (lwork >= m * m + max(max(4 * m, bdspac), wrkbl)) {
            int iv = 0;
            if (lwork >= wrkbl + ldworkv * m) {
                ldworkv = lda;
            } else {
                ldworkv = m;
            }
            int itau = iv + ldworkv * m * 2;
            int iwork = itau + m * 2;

            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            zlacpy('L', m, m, A, aOff, lda, work, workOff + iv, ldworkv);
            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + iv + 1, ldworkv);

            Zgelq.zorglq(m, n, m, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            iwork = itaup + m * 2;

            Zgebrd.zgebrd(m, m, work, workOff + iv, ldworkv, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, m, work, workOff + iv, ldworkv, u, uOff, ldu);

            Zorgbr.zorgbr('P', m, m, m, work, workOff + iv, ldworkv, work, workOff + itaup,
                      work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('Q', m, m, m, u, uOff, ldu, work, workOff + itauq,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('U', m, m, m, 0, s, sOff, work, workOff + ie, work, workOff + iv, ldworkv, u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);

            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, m, 1.0, 0.0, work, workOff + iv, ldworkv, A, aOff, lda, 0.0, 0.0, vt, vtOff, ldvt);

            return ieRet >= 0 ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + m * 2;

            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Zgelq.zorglq(m, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            zlacpy('L', m, m, A, aOff, lda, u, uOff, ldu);
            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, u, uOff + 1, ldu);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            iwork = itaup + m * 2;

            Zgebrd.zgebrd(m, m, u, uOff, ldu, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zormbr.zormbr('P', 'L', 'C', m, n, m, u, uOff, ldu, work, workOff + itaup,
                      vt, vtOff, ldvt, work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('Q', m, m, m, u, uOff, ldu, work, workOff + itauq,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('U', m, m, m, 0, s, sOff, work, workOff + ie, u, rawMatrixOff(uOff), ldu,
                                       work, workOff, 1, vt, rawMatrixOff(vtOff), ldvt, work, workOff + iwork);

            return ieRet >= 0 ? ie : -ie;
        }
    }

    static int path7t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworkr) {
        if (lwork >= m * m + max(max(m + n, 4 * m), max(bdspac, wrkbl))) {
            int ir = 0;
            if (lwork >= wrkbl + ldworkr * m) {
                ldworkr = lda;
            } else {
                ldworkr = m;
            }
            int itau = ir + ldworkr * m * 2;
            int iwork = itau + m * 2;

            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            zlacpy('L', m, m, A, aOff, lda, work, workOff + ir, ldworkr);
            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + ir + 1, ldworkr);

            Zgelq.zorglq(n, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            iwork = itaup + m * 2;

            Zgebrd.zgebrd(m, m, work, workOff + ir, ldworkr, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('P', m, m, m, work, workOff + ir, ldworkr, work, workOff + itaup,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('U', m, 0, m, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                                       work, workOff + ir, ldworkr, work, workOff, 1, work, workOff + iwork);

            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, m, 1.0, 0.0, work, workOff + ir, ldworkr, vt, vtOff, ldvt, 0.0, 0.0, A, aOff, lda);

            zlacpy('A', m, n, A, aOff, lda, vt, vtOff, ldvt);

            return ieRet >= 0 ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + m * 2;

            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Zgelq.zorglq(n, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            iwork = itaup + m * 2;

            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, A, aOff + 1, lda);

            Zgebrd.zgebrd(m, m, A, aOff, lda, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zormbr.zormbr('P', 'L', 'C', m, n, m, A, aOff, lda, work, workOff + itaup,
                      vt, vtOff, ldvt, work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('U', m, 0, m, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                                       work, workOff, 1, vt, rawMatrixOff(vtOff), ldvt, work, workOff + iwork);

            return ieRet >= 0 ? ie : -ie;
        }
    }

    static int path9t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworkv) {
        if (lwork >= m * m + max(max(m + n, 4 * m), max(bdspac, wrkbl))) {
            int iv = 0;
            if (lwork >= wrkbl + ldworkv * m) {
                ldworkv = lda;
            } else {
                ldworkv = m;
            }
            int itau = iv + ldworkv * m * 2;
            int iwork = itau + m * 2;

            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Zgelq.zorglq(n, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            zlacpy('L', m, m, A, aOff, lda, work, workOff + iv, ldworkv);
            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, work, workOff + iv + 1, ldworkv);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            iwork = itaup + m * 2;

            Zgebrd.zgebrd(m, m, work, workOff + iv, ldworkv, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            zlacpy('L', m, m, work, workOff + iv, ldworkv, u, uOff, ldu);

            Zorgbr.zorgbr('P', m, m, m, work, workOff + iv, ldworkv, work, workOff + itaup,
                      work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('Q', m, m, m, u, uOff, ldu, work, workOff + itauq,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('U', m, m, m, 0, s, sOff, work, workOff + ie, work, workOff + iv, ldworkv, u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);

            zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, m, 1.0, 0.0, work, workOff + iv, ldworkv, vt, vtOff, ldvt, 0.0, 0.0, A, aOff, lda);

            zlacpy('A', m, n, A, aOff, lda, vt, vtOff, ldvt);

            return ieRet >= 0 ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + m * 2;

            Zgelq.zgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            zlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Zgelq.zorglq(n, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            zlacpy('L', m, m, A, aOff, lda, u, uOff, ldu);
            zlaset('U', m - 1, m - 1, 0.0, 0.0, 0.0, 0.0, u, uOff + 1, ldu);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m * 2;
            iwork = itaup + m * 2;

            Zgebrd.zgebrd(m, m, u, uOff, ldu, s, sOff, work, workOff + ie,
                      work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Zormbr.zormbr('P', 'L', 'C', m, n, m, u, uOff, ldu, work, workOff + itaup,
                      vt, vtOff, ldvt, work, workOff + iwork, lwork - iwork);

            Zorgbr.zorgbr('Q', m, m, m, u, uOff, ldu, work, workOff + itauq,
                      work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            int ieRet = Zbdsqr.zbdsqr('U', m, m, m, 0, s, sOff, work, workOff + ie, u, rawMatrixOff(uOff), ldu,
                                       work, workOff, 1, vt, rawMatrixOff(vtOff), ldvt, work, workOff + iwork);

            return ieRet >= 0 ? ie : -ie;
        }
    }

    static int path10t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                       double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                       double[] work, int workOff, int lwork, boolean wantuas, boolean wantuo, boolean wantvas, boolean wantva, boolean wantvo) {
        int ie = 0;
        int itauq = ie + m;
        int itaup = itauq + m * 2;
        int iwork = itaup + m * 2;

        Zgebrd.zgebrd(m, n, A, aOff, lda, s, sOff, work, workOff + ie, work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

        if (wantvas) {
            zlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);
            int nrvt = wantva ? n : m;
            Zorgbr.zorgbr('P', nrvt, n, m, vt, vtOff, ldvt, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
        }

        if (wantuas) {
            zlacpy('L', m, m, A, aOff, lda, u, uOff, ldu);
            Zorgbr.zorgbr('Q', m, m, n, u, uOff, ldu, work, workOff + itauq, work, workOff + iwork, lwork - iwork);
        }

        if (wantvo) {
            Zorgbr.zorgbr('P', m, n, m, A, aOff, lda, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
        }

        if (wantuo) {
            Zorgbr.zorgbr('Q', m, m, n, A, aOff, lda, work, workOff + itauq, work, workOff + iwork, lwork - iwork);
        }

        int ncvt = 0, nru = 0;
        if (wantvas || wantvo) ncvt = n;
        if (wantuas || wantuo) nru = m;

        iwork = ie + m;
        int ieRet;
        if (!wantuo && !wantvo) {
            ieRet = Zbdsqr.zbdsqr('L', m, ncvt, nru, 0, s, sOff, work, workOff + ie, vt, rawMatrixOff(vtOff), ldvt, u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);
        } else if (!wantuo) {
            ieRet = Zbdsqr.zbdsqr('L', m, ncvt, nru, 0, s, sOff, work, workOff + ie, A, rawMatrixOff(aOff), lda, u, rawMatrixOff(uOff), ldu, work, workOff, 1, work, workOff + iwork);
        } else {
            ieRet = Zbdsqr.zbdsqr('L', m, ncvt, nru, 0, s, sOff, work, workOff + ie, vt, rawMatrixOff(vtOff), ldvt, A, rawMatrixOff(aOff), lda, work, workOff, 1, work, workOff + iwork);
        }

        return ieRet >= 0 ? 0 : -1;
    }
}