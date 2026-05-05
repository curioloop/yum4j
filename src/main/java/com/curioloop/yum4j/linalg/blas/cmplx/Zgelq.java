/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;

interface Zgelq {

    static int zgelq2(int m, int n, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] work, int workOff) {
        if (m < 0) return -1;
        if (n < 0) return -2;
        if (lda < max(1, n)) return -4;

        int k = min(m, n);

        for (int i = 0; i < k; i++) {
            Zlacgv.zlacgv(n - i, A, aOff + i * lda + i, 1);

            int alphaPos = (aOff + i * lda + i) * 2;
            // Zlarfg takes the scalar head in raw interleaved-array coordinates but the row tail in complex-entry units.
            Zlarfg.zlarfg(n - i, A, alphaPos, A, aOff + i * lda + Math.min(i + 1, n - 1), 1, tau, tauOff + i * 2);

            if (i < m - 1) {
                double saveRe = A[alphaPos];
                double saveIm = A[alphaPos + 1];
                A[alphaPos] = 1.0;
                A[alphaPos + 1] = 0.0;
                Zlarf.zlarf(BLAS.Side.Right, m - i - 1, n - i, A, alphaPos, 1, tau, tauOff + i * 2, A, (aOff + (i + 1) * lda + i) * 2, lda, work, workOff);
                A[alphaPos] = saveRe;
                A[alphaPos + 1] = saveIm;
            }

            Zlacgv.zlacgv(n - i, A, aOff + i * lda + i, 1);
        }

        return 0;
    }

    static int zgelqf(int m, int n, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (m < 0) return -1;
        if (n < 0) return -2;
        if (A == null) return -3;
        if (aOff < 0) return -4;
        if (lda < max(1, n)) return -5;
        if (tau == null) return -6;
        if (tauOff < 0) return -7;
        if (work == null) return -8;
        if (workOff < 0) return -9;
        if (lwork < -1) return -10;

        int k = min(m, n);
        if (k == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        int minWork = Math.max(1, m);
        if (lwork < minWork && lwork != -1) return -10;

        if (lwork == -1) {
            work[workOff] = minWork;
            return 0;
        }

        return zgelq2(m, n, A, aOff, lda, tau, tauOff, work, workOff);
    }

    static int zorg2l(int m, int n, int k, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] work, int workOff) {
        if (m < 0 || m > n) return -1;
        if (n < 0) return -2;
        if (k < 0 || k > m) return -3;
        if (lda < max(1, n)) return -5;

        for (int i = k; i < m; i++) {
            for (int l = 0; l < n; l++) {
                A[(aOff + i * lda + l) * 2] = 0.0;
                A[(aOff + i * lda + l) * 2 + 1] = 0.0;
            }
            if (i < n) {
                A[(aOff + i * lda + i) * 2] = 1.0;
                A[(aOff + i * lda + i) * 2 + 1] = 0.0;
            }
        }

        for (int i = k - 1; i >= 0; i--) {
            double tauIRe = tau[tauOff + i * 2];
            double tauIIm = tau[tauOff + i * 2 + 1];
            double tauIConjRe = tauIRe;
            double tauIConjIm = -tauIIm;

            if (i < n - 1) {
                Zlacgv.zlacgv(n - i - 1, A, aOff + i * lda + i + 1, 1);
            }

            if (i < m - 1) {
                int diagPos = (aOff + i * lda + i) * 2;
                A[diagPos] = 1.0;
                A[diagPos + 1] = 0.0;
                tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                Zlarf.zlarf(BLAS.Side.Right, m - i - 1, n - i, A, diagPos, 1, tau, tauOff + i * 2, A, (aOff + (i + 1) * lda + i) * 2, lda, work, workOff);
                tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
            }

            A[(aOff + i * lda + i) * 2] = 1.0 - tauIConjRe;
            A[(aOff + i * lda + i) * 2 + 1] = -tauIConjIm;

            if (i < n - 1) {
                Zscal.zscal(n - i - 1, -tauIRe, -tauIIm, A, (aOff + i * lda + i + 1) * 2, 1);
                Zlacgv.zlacgv(n - i - 1, A, aOff + i * lda + i + 1, 1);
            }

            for (int l = 0; l < i; l++) {
                A[(aOff + i * lda + l) * 2] = 0.0;
                A[(aOff + i * lda + l) * 2 + 1] = 0.0;
            }
        }

        return 0;
    }

    static int zorglq(int m, int n, int k, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (m < 0) return -1;
        if (n < 0) return -2;
        if (m > n) return -1;
        if (k < 0 || k > m) return -3;
        if (A == null) return -4;
        if (lda < max(1, n)) return -5;
        if (aOff < 0) return -6;
        if (tau == null) return -7;
        if (tauOff < 0) return -8;
        if (work == null) return -9;
        if (workOff < 0) return -10;
        if (lwork < -1) return -11;

        if (m == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        int nb = Ilaenv.ilaenv(1, "ZORGLQ", "", m, n, k, -1);
        int lworkopt = nb * m;
        int minWork = max(1, m);

        if (lwork == -1) {
            work[workOff] = lworkopt;
            return 0;
        }

        if (lwork < minWork) return -11;

        int info = zorg2l(m, n, k, A, aOff, lda, tau, tauOff, work, workOff);
        work[workOff] = lworkopt;
        return info;
    }

    static void zorm2l(char cside, char ctrans, int m, int n, int k,
                       double[] A, int aOff, int lda, double[] tau, int tauOff,
                       double[] C, int cOff, int ldc, double[] work, int workOff) {
        boolean left = (cside == 'L');
        boolean notran = (ctrans == 'N');

        if (left) {
            if (notran) {
                for (int i = 0; i < k; i++) {
                    int aiiPos = (aOff + i * lda + i) * 2;
                    double savedRe = A[aiiPos]; double savedIm = A[aiiPos + 1];
                    A[aiiPos] = 1.0; A[aiiPos + 1] = 0.0;
                    tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                    Zlarf.zlarf(BLAS.Side.Left, m - i, n, A, aiiPos, 1, tau, tauOff + i * 2, C, (cOff + i * ldc) * 2, ldc, work, workOff);
                    tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                    A[aiiPos] = savedRe; A[aiiPos + 1] = savedIm;
                }
            } else {
                for (int i = k - 1; i >= 0; i--) {
                    int aiiPos = (aOff + i * lda + i) * 2;
                    double savedRe = A[aiiPos]; double savedIm = A[aiiPos + 1];
                    A[aiiPos] = 1.0; A[aiiPos + 1] = 0.0;
                    Zlarf.zlarf(BLAS.Side.Left, m - i, n, A, aiiPos, 1, tau, tauOff + i * 2, C, (cOff + i * ldc) * 2, ldc, work, workOff);
                    A[aiiPos] = savedRe; A[aiiPos + 1] = savedIm;
                }
            }
        } else {
            if (notran) {
                for (int i = k - 1; i >= 0; i--) {
                    int aiiPos = (aOff + i * lda + i) * 2;
                    double savedRe = A[aiiPos]; double savedIm = A[aiiPos + 1];
                    A[aiiPos] = 1.0; A[aiiPos + 1] = 0.0;
                    tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                    Zlarf.zlarf(BLAS.Side.Right, m, n - i, A, aiiPos, 1, tau, tauOff + i * 2, C, (cOff + i) * 2, ldc, work, workOff);
                    tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                    A[aiiPos] = savedRe; A[aiiPos + 1] = savedIm;
                }
            } else {
                for (int i = 0; i < k; i++) {
                    int aiiPos = (aOff + i * lda + i) * 2;
                    double savedRe = A[aiiPos]; double savedIm = A[aiiPos + 1];
                    A[aiiPos] = 1.0; A[aiiPos + 1] = 0.0;
                    Zlarf.zlarf(BLAS.Side.Right, m, n - i, A, aiiPos, 1, tau, tauOff + i * 2, C, (cOff + i) * 2, ldc, work, workOff);
                    A[aiiPos] = savedRe; A[aiiPos + 1] = savedIm;
                }
            }
        }
    }

    static int zormlq(char side, char trans, int m, int n, int k, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] C, int cOff, int ldc, double[] work, int workOff, int lwork) {
        char cside = Character.toUpperCase(side);
        char ctrans = Character.toUpperCase(trans);

        if (cside != 'L' && cside != 'R') {
            return -1;
        }
        if (ctrans != 'N' && ctrans != 'T' && ctrans != 'C') {
            return -2;
        }
        if (m < 0) return -3;
        if (n < 0) return -4;
        if (k < 0) return -5;
        boolean left = (cside == 'L');
        boolean notrans = (ctrans == 'N');
        int nw = left ? n : m;
        if (left && k > m) return -5;
        if (!left && k > n) return -5;
        if (lda < max(1, left ? m : n)) return -7;
        if (ldc < max(1, n)) return -10;
        if (lwork < max(1, nw) && lwork != -1) return -11;

        if (m == 0 || n == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        int nbmax = 64;
        int ldt = nbmax;
        int tsize = nbmax * ldt;

        String opts = String.valueOf(cside) + ctrans;
        int nb = min(nbmax, Ilaenv.ilaenv(1, "ZORMLQ", opts, m, n, k, -1));
        if (nb < 1) nb = 1;
        int lworkopt = max(1, nw) * nb + tsize;

        if (lwork == -1) {
            work[workOff] = lworkopt;
            return 0;
        }

        int nbmin = 2;
        if (1 < nb && nb < k) {
            int iws = nw * nb + tsize;
            if (lwork < iws) {
                nb = (lwork - tsize) / nw;
                nbmin = max(2, Ilaenv.ilaenv(2, "ZORMLQ", opts, m, n, k, -1));
            }
        }

        if (nb < nbmin || k <= nb) {
            zorm2l(cside, ctrans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff);
            work[workOff] = lworkopt;
            return 0;
        }

        int ldwork = nb;
        int tOff = workOff / 2;
        int wrkOff = tOff + tsize;
        char transBlock = notrans ? 'C' : 'N';

        if (left && notrans) {
            for (int i = 0; i < k; i += nb) {
                int ib = min(nb, k - i);
                zlarftRowWise(m - i, ib, A, aOff + i * lda + i, lda, tau, tauOff + i * 2, work, tOff, ldt);
            Zlarfb.zlarfb('L', transBlock, 'F', 'R', m - i, n, ib,
                A, aOff + i * lda + i, lda,
                work, tOff, ldt,
                C, cOff + i * ldc, ldc,
                work, wrkOff, ldwork);
            }
        } else if (left) {
            for (int i = ((k - 1) / nb) * nb; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                zlarftRowWise(m - i, ib, A, aOff + i * lda + i, lda, tau, tauOff + i * 2, work, tOff, ldt);
            Zlarfb.zlarfb('L', transBlock, 'F', 'R', m - i, n, ib,
                A, aOff + i * lda + i, lda,
                work, tOff, ldt,
                C, cOff + i * ldc, ldc,
                work, wrkOff, ldwork);
            }
        } else if (notrans) {
            for (int i = ((k - 1) / nb) * nb; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                zlarftRowWise(n - i, ib, A, aOff + i * lda + i, lda, tau, tauOff + i * 2, work, tOff, ldt);
            Zlarfb.zlarfb('R', transBlock, 'F', 'R', m, n - i, ib,
                A, aOff + i * lda + i, lda,
                work, tOff, ldt,
                C, cOff + i, ldc,
                work, wrkOff, ldwork);
            }
        } else {
            for (int i = 0; i < k; i += nb) {
                int ib = min(nb, k - i);
                zlarftRowWise(n - i, ib, A, aOff + i * lda + i, lda, tau, tauOff + i * 2, work, tOff, ldt);
            Zlarfb.zlarfb('R', transBlock, 'F', 'R', m, n - i, ib,
                A, aOff + i * lda + i, lda,
                work, tOff, ldt,
                C, cOff + i, ldc,
                work, wrkOff, ldwork);
            }
        }

        work[workOff] = lworkopt;
        return 0;
    }

    static void zlarftRowWise(int n, int k, double[] V, int vOff, int ldv,
                              double[] tau, int tauOff, double[] T, int tOff, int ldt) {
        if (k <= 0) return;

        int prevlastv = n - 1;
        for (int i = 0; i < k; i++) {
            prevlastv = max(i, prevlastv);
            double tauRe = tau[tauOff + i * 2];
            double tauIm = tau[tauOff + i * 2 + 1];
            if (tauRe == 0.0 && tauIm == 0.0) {
                for (int j = 0; j <= i; j++) {
                    T[(tOff + j * ldt + i) * 2] = 0;
                    T[(tOff + j * ldt + i) * 2 + 1] = 0;
                }
                continue;
            }

            int lastv;
            for (lastv = n - 1; lastv >= i + 1; lastv--) {
                double re = V[(vOff + i * ldv + lastv) * 2];
                double im = V[(vOff + i * ldv + lastv) * 2 + 1];
                if (re != 0.0 || im != 0.0) break;
            }

            for (int j = 0; j < i; j++) {
                double vRe = V[(vOff + j * ldv + i) * 2];
                double vIm = V[(vOff + j * ldv + i) * 2 + 1];
                T[(tOff + j * ldt + i) * 2] = -(tauRe * vRe - tauIm * vIm);
                T[(tOff + j * ldt + i) * 2 + 1] = -(tauRe * vIm + tauIm * vRe);
            }

            int j = min(lastv, prevlastv);
            if (j > i) {
                for (int col = 0; col < i; col++) {
                    double sumRe = T[(tOff + col * ldt + i) * 2];
                    double sumIm = T[(tOff + col * ldt + i) * 2 + 1];
                    for (int row = i + 1; row <= j; row++) {
                        double v1Re = V[(vOff + col * ldv + row) * 2];
                        double v1Im = V[(vOff + col * ldv + row) * 2 + 1];
                        double v2Re = V[(vOff + i * ldv + row) * 2];
                        double v2Im = V[(vOff + i * ldv + row) * 2 + 1];
                        double prodRe = v1Re * v2Re + v1Im * v2Im;
                        double prodIm = v1Re * v2Im - v1Im * v2Re;
                        sumRe -= tauRe * prodRe - tauIm * prodIm;
                        sumIm -= tauRe * prodIm + tauIm * prodRe;
                    }
                    T[(tOff + col * ldt + i) * 2] = sumRe;
                    T[(tOff + col * ldt + i) * 2 + 1] = sumIm;
                }
            }

            if (i > 0) {
                for (int col = 0; col < i; col++) {
                    double sumRe = 0, sumIm = 0;
                    for (int p = col; p < i; p++) {
                        double t1Re = T[(tOff + col * ldt + p) * 2];
                        double t1Im = T[(tOff + col * ldt + p) * 2 + 1];
                        double t2Re = T[(tOff + p * ldt + i) * 2];
                        double t2Im = T[(tOff + p * ldt + i) * 2 + 1];
                        sumRe += t1Re * t2Re - t1Im * t2Im;
                        sumIm += t1Re * t2Im + t1Im * t2Re;
                    }
                    T[(tOff + col * ldt + i) * 2] = sumRe;
                    T[(tOff + col * ldt + i) * 2 + 1] = sumIm;
                }
            }

            T[(tOff + i * ldt + i) * 2] = tauRe;
            T[(tOff + i * ldt + i) * 2 + 1] = tauIm;
            if (i > 1) {
                prevlastv = max(prevlastv, lastv);
            } else {
                prevlastv = lastv;
            }
        }
    }

    static void zlarfbRowWiseLeft(BLAS.Side side, BLAS.Trans trans, int m, int n, int k,
                                   double[] V, int vOff, int ldv,
                                   double[] T, int tOff, int ldt,
                                   double[] C, int cOff, int ldc,
                                   double[] work, int wOff, int ldwork) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        BLAS.Trans transEnum = (trans == BLAS.Trans.NoTrans) ? BLAS.Trans.Conj : BLAS.Trans.NoTrans;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                work[wOff + (j * n + i) * 2] = C[(cOff + j * ldc + i) * 2];
                work[wOff + (j * n + i) * 2 + 1] = C[(cOff + j * ldc + i) * 2 + 1];
            }
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit,
                n, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff, n);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, k, m - k,
                    1.0, 0.0, C, cOff + k, ldc, V, vOff + k, ldv,
                    1.0, 0.0, work, wOff / 2, n);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, transEnum, BLAS.Diag.NonUnit,
                n, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff, n);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, m - k, n, k,
                    -1.0, 0.0, V, vOff + k, ldv, work, wOff / 2, n,
                    1.0, 0.0, C, cOff + k, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                n, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                C[(cOff + j * ldc + i) * 2] -= work[wOff + (j * n + i) * 2];
                C[(cOff + j * ldc + i) * 2 + 1] -= work[wOff + (j * n + i) * 2 + 1];
            }
        }
    }

    static void zlarfbRowWiseRight(BLAS.Side side, BLAS.Trans trans, int m, int n, int k,
                                    double[] V, int vOff, int ldv,
                                    double[] T, int tOff, int ldt,
                                    double[] C, int cOff, int ldc,
                                    double[] work, int wOff, int ldwork) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        BLAS.Trans transEnum = (trans == BLAS.Trans.NoTrans) ? BLAS.Trans.Conj : BLAS.Trans.NoTrans;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < m; i++) {
                work[wOff + (j * m + i) * 2] = C[(cOff + i * ldc + j) * 2];
                work[wOff + (j * m + i) * 2 + 1] = C[(cOff + i * ldc + j) * 2 + 1];
            }
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit,
                m, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff, m);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, k, n - k,
                    1.0, 0.0, C, cOff + k, ldc, V, vOff + k, ldv,
                    1.0, 0.0, work, wOff / 2, m);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, transEnum, BLAS.Diag.NonUnit,
                m, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff, m);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n - k, k,
                    -1.0, 0.0, work, wOff / 2, m, V, vOff + k, ldv,
                    1.0, 0.0, C, cOff + k, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                m, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff, m);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[(cOff + i * ldc + j) * 2] -= work[wOff + (j * m + i) * 2];
                C[(cOff + i * ldc + j) * 2 + 1] -= work[wOff + (j * m + i) * 2 + 1];
            }
        }
    }

    static int max(int a, int b) {
        return a > b ? a : b;
    }

    static int min(int a, int b) {
        return a < b ? a : b;
    }
}
