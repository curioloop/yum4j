/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;

interface Zgebrd {

    static int zgebrd(int m, int n, double[] A, int aOff, int lda, double[] d, int dOff, double[] e, int eOff, double[] tauQ, int tauQOff, double[] tauP, int tauPOff, double[] work, int workOff, int lwork) {
        int info = 0;
        int minmn = min(m, n);

        boolean lquery = (lwork == -1);
        if (m < 0) {
            info = -1;
        } else if (n < 0) {
            info = -2;
        } else if (lda < max(1, n)) {
            info = -5;
        }

        int lwkmin;
        int lwkopt;
        if (minmn == 0) {
            lwkmin = 1;
            lwkopt = 1;
        } else {
            lwkmin = max(m, n);
            int nb = max(1, Ilaenv.ilaenv(1, "ZGEBRD", "", m, n, -1, -1));
            lwkopt = (m + n) * nb;
        }

        if (info < 0) {
            return info;
        }

        if (minmn == 0) {
            if (work != null && workOff >= 0 && workOff < work.length && lwork >= 1) {
                work[workOff] = lwkopt;
            }
            return 0;
        }

        if (lwork < lwkmin && !lquery) {
            return -10;
        }

        if (lquery) {
            if (work != null && workOff >= 0 && workOff < work.length) {
                work[workOff] = lwkopt;
            }
            return 0;
        }

        int ws = max(m, n);
        int ldwrkx = m;
        int ldwrky = n;
        int nx = 0;

        int nb = max(1, Ilaenv.ilaenv(1, "ZGEBRD", "", m, n, -1, -1));
        if (nb > 1 && nb < minmn) {
            nx = max(nb, Ilaenv.ilaenv(3, "ZGEBRD", "", m, n, -1, -1));
            if (nx < minmn) {
                ws = lwkopt;
                if (lwork < ws) {
                    int nbmin = Ilaenv.ilaenv(2, "ZGEBRD", "", m, n, -1, -1);
                    if (lwork >= (m + n) * nbmin) {
                        nb = lwork / (m + n);
                    } else {
                        nb = 1;
                        nx = minmn;
                    }
                }
            }
        } else {
            nx = minmn;
        }

        int i = 0;
        for (i = 0; i < minmn - nx; i += nb) {
            int jb = min(nb, minmn - i);
            // Public matrix offsets are in complex entries; the unblocked helpers consume raw interleaved-array offsets.
            Zlabrd.zlabrd(m - i, n - i, jb, A, (aOff + i * lda + i) * 2, lda, d, dOff + i, e, eOff + i, tauQ, tauQOff + i * 2, tauP, tauPOff + i * 2, work, workOff, ldwrkx, work, workOff + ldwrkx * jb * 2, ldwrky);

            if (m - i - jb > 0 && n - i - jb > 0) {
                Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m - i - jb, n - i - jb, jb, -1.0, 0.0, A, (aOff + (i + jb) * lda + i) * 2, lda, work, workOff + ldwrkx * jb * 2 + jb * 2, ldwrky, 1.0, 0.0, A, (aOff + (i + jb) * lda + i + jb) * 2, lda);
                Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m - i - jb, n - i - jb, jb, -1.0, 0.0, work, workOff + jb * 2, ldwrkx, A, (aOff + i * lda + i + jb) * 2, lda, 1.0, 0.0, A, (aOff + (i + jb) * lda + i + jb) * 2, lda);
            }

            if (m >= n) {
                for (int j = i; j < i + jb; j++) {
                    A[(aOff + j * lda + j) * 2] = d[dOff + j];
                    A[(aOff + j * lda + j) * 2 + 1] = 0.0;
                    if (j < n - 1) {
                        A[(aOff + j * lda + j + 1) * 2] = e[eOff + j];
                        A[(aOff + j * lda + j + 1) * 2 + 1] = 0.0;
                    }
                }
            } else {
                for (int j = i; j < i + jb; j++) {
                    A[(aOff + j * lda + j) * 2] = d[dOff + j];
                    A[(aOff + j * lda + j) * 2 + 1] = 0.0;
                    if (j < m - 1) {
                        A[(aOff + (j + 1) * lda + j) * 2] = e[eOff + j];
                        A[(aOff + (j + 1) * lda + j) * 2 + 1] = 0.0;
                    }
                }
            }
        }

        Zgebrd.zgebd2(m - i, n - i, A, (aOff + i * lda + i) * 2, lda, d, dOff + i, e, eOff + i, tauQ, tauQOff + i * 2, tauP, tauPOff + i * 2, work, workOff);
        work[workOff] = ws;

        return info;
    }

    static int zgebd2(int m, int n, double[] A, int aOff, int lda, double[] d, int dOff, double[] e, int eOff, double[] tauQ, int tauQOff, double[] tauP, int tauPOff, double[] work, int workOff) {
        int info = 0;

        // Internal helper contract: aOff is a raw interleaved-array offset in doubles, not a complex-entry offset.
        // Callers like zgebrd pass submatrix starts as (aOff + row * lda + col) * 2.

        if (m < 0) {
            info = -1;
        } else if (n < 0) {
            info = -2;
        } else if (A == null) {
            info = -3;
        } else if (lda < max(1, n)) {
            info = -4;
        }

        if (info < 0) {
            return info;
        }

        if (m == 0 || n == 0) {
            return 0;
        }

        if (m >= n) {
            for (int i = 0; i < n; i++) {
                int alphaPos = (i * lda + i) * 2;

                Zlarfg.zlarfg(m - i, A, aOff + alphaPos, A, aOff / 2 + (i + 1) * lda + i, lda, tauQ, tauQOff + i * 2);
                d[dOff + i] = A[aOff + alphaPos];

                if (i < n - 1) {
                    A[aOff + alphaPos] = 1.0;
                    A[aOff + alphaPos + 1] = 0.0;
                    tauQ[tauQOff + i * 2 + 1] = -tauQ[tauQOff + i * 2 + 1];
                    Zlarf.zlarf(BLAS.Side.Left, m - i, n - i - 1, A, aOff + alphaPos, lda, tauQ, tauQOff + i * 2, A, aOff + (i * lda + i + 1) * 2, lda, work, workOff);
                    tauQ[tauQOff + i * 2 + 1] = -tauQ[tauQOff + i * 2 + 1];
                }

                A[aOff + alphaPos] = d[dOff + i];
                A[aOff + alphaPos + 1] = 0.0;

                if (i < n - 1) {
                    Zlacgv.zlacgv(n - i - 1, A, aOff / 2 + i * lda + i + 1, 1);
                    int alphaBetaPos = (i * lda + i + 1) * 2;
                    Zlarfg.zlarfg(n - i - 1, A, aOff + alphaBetaPos, A, aOff / 2 + i * lda + i + 2, 1, tauP, tauPOff + i * 2);
                    e[eOff + i] = A[aOff + alphaBetaPos];
                    A[aOff + alphaBetaPos] = 1.0;
                    A[aOff + alphaBetaPos + 1] = 0.0;
                    Zlarf.zlarf(BLAS.Side.Right, m - i - 1, n - i - 1, A, aOff + alphaBetaPos, 1, tauP, tauPOff + i * 2, A, aOff + ((i + 1) * lda + i + 1) * 2, lda, work, workOff);
                    Zlacgv.zlacgv(n - i - 1, A, aOff / 2 + i * lda + i + 1, 1);
                    A[aOff + alphaBetaPos] = e[eOff + i];
                    A[aOff + alphaBetaPos + 1] = 0.0;
                } else {
                    tauP[tauPOff + i * 2] = 0.0;
                    tauP[tauPOff + i * 2 + 1] = 0.0;
                }
            }
        } else {
            for (int i = 0; i < m; i++) {
                Zlacgv.zlacgv(n - i, A, aOff / 2 + i * lda + i, 1);
                int alphaPos = (i * lda + i) * 2;
                Zlarfg.zlarfg(n - i, A, aOff + alphaPos, A, aOff / 2 + i * lda + i + 1, 1, tauP, tauPOff + i * 2);
                d[dOff + i] = A[aOff + alphaPos];

                if (i < m - 1) {
                    A[aOff + alphaPos] = 1.0;
                    A[aOff + alphaPos + 1] = 0.0;
                    Zlarf.zlarf(BLAS.Side.Right, m - i - 1, n - i, A, aOff + alphaPos, 1, tauP, tauPOff + i * 2, A, aOff + ((i + 1) * lda + i) * 2, lda, work, workOff);
                }

                Zlacgv.zlacgv(n - i, A, aOff / 2 + i * lda + i, 1);
                A[aOff + alphaPos] = d[dOff + i];
                A[aOff + alphaPos + 1] = 0.0;

                if (i < m - 1) {
                    int betaPos = ((i + 1) * lda + i) * 2;
                    Zlarfg.zlarfg(m - i - 1, A, aOff + betaPos, A, aOff / 2 + (i + 2) * lda + i, lda, tauQ, tauQOff + i * 2);
                    e[eOff + i] = A[aOff + betaPos];

                    A[aOff + betaPos] = 1.0;
                    A[aOff + betaPos + 1] = 0.0;
                    tauQ[tauQOff + i * 2 + 1] = -tauQ[tauQOff + i * 2 + 1];
                    Zlarf.zlarf(BLAS.Side.Left, m - i - 1, n - i - 1, A, aOff + betaPos, lda, tauQ, tauQOff + i * 2, A, aOff + ((i + 1) * lda + i + 1) * 2, lda, work, workOff);
                    tauQ[tauQOff + i * 2 + 1] = -tauQ[tauQOff + i * 2 + 1];
                    A[aOff + betaPos] = e[eOff + i];
                    A[aOff + betaPos + 1] = 0.0;
                } else {
                    tauQ[tauQOff + i * 2] = 0.0;
                    tauQ[tauQOff + i * 2 + 1] = 0.0;
                }
            }
        }

        return info;
    }

    static int max(int a, int b) {
        return a > b ? a : b;
    }

    static int min(int a, int b) {
        return a < b ? a : b;
    }
}