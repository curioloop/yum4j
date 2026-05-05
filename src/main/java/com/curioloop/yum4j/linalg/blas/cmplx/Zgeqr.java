/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;

import static java.lang.Math.max;
import static java.lang.Math.min;

interface Zgeqr {

    static int zgeqr2(int m, int n, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] work, int workOff) {
        if (m < 0) return -1;
        if (n < 0) return -2;
        if (A == null) return -3;
        if (aOff < 0) return -4;
        if (lda < max(1, n)) return -5;
        if (tau == null) return -6;
        if (tauOff < 0) return -7;
        if (work == null) return -8;
        if (workOff < 0) return -9;

        int k = min(m, n);

        for (int i = 0; i < k; i++) {
            // Matrix offsets are counted in complex entries; ZLARFG takes the leading scalar
            // as a raw double index and the reflector tail as a complex-entry offset.
            int pos = (aOff + i * lda + i) * 2;
            int xPos = aOff + (i + 1) * lda + i;
            if (i < m - 1) {
                Zlarfg.zlarfg(m - i, A, pos, A, xPos, lda, tau, tauOff + i * 2);
            } else {
                Zlarfg.zlarfg(1, A, pos, null, 0, 1, tau, tauOff + i * 2);
            }

            if (i < n - 1) {
                tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                Zlarf1f.zlarf1f(BLAS.Side.Left, m - i, n - i - 1, A, (aOff + i * lda + i) * 2, lda, tau, tauOff + i * 2, A, (aOff + i * lda + i + 1) * 2, lda, work, workOff);
                tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
            }
        }

        return 0;
    }

    static int zgeqrf(int m, int n, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] work, int workOff, int lwork) {
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

        int minWork = max(1, n - 1);
        if (lwork < minWork && lwork != -1) return -10;

        if (lwork == -1) {
            work[workOff] = n;
            return 0;
        }

        for (int i = 0; i < k; i++) {
            int pos = (aOff + i * lda + i) * 2;
            int xPos = aOff + (i + 1) * lda + i;
            if (i < m - 1) {
                Zlarfg.zlarfg(m - i, A, pos, A, xPos, lda, tau, tauOff + i * 2);
            } else {
                Zlarfg.zlarfg(1, A, pos, null, 0, 1, tau, tauOff + i * 2);
            }

            if (i < n - 1) {
                tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                Zlarf1f.zlarf1f(BLAS.Side.Left, m - i, n - i - 1, A, (aOff + i * lda + i) * 2, lda, tau, tauOff + i * 2, A, (aOff + i * lda + i + 1) * 2, lda, work, workOff);
                tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
            }
        }

        work[workOff] = 1;
        return 0;
    }

    static int zorg2r(int m, int n, int k, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] work, int workOff) {
        if (m < 0) return -1;
        if (n < 0) return -2;
        if (k < 0 || k > n) return -3;
        if (lda < max(1, n)) return -5;

        for (int j = k; j < n; j++) {
            for (int l = 0; l < m; l++) {
                A[(aOff + l * lda + j) * 2] = 0.0;
                A[(aOff + l * lda + j) * 2 + 1] = 0.0;
            }
            if (j < m) {
                A[(aOff + j * lda + j) * 2] = 1.0;
                A[(aOff + j * lda + j) * 2 + 1] = 0.0;
            }
        }

        for (int i = k - 1; i >= 0; i--) {
            if (i < n - 1) {
                Zlarf1f.zlarf1f(BLAS.Side.Left, m - i, n - i - 1, A, (aOff + i * lda + i) * 2, lda, tau, tauOff + i * 2, A, (aOff + i * lda + i + 1) * 2, lda, work, workOff);
            }
            if (i < m - 1) {
                double tauIRe = tau[tauOff + i * 2];
                double tauIIm = tau[tauOff + i * 2 + 1];
                Zscal.zscal(m - i - 1, -tauIRe, -tauIIm, A, (aOff + (i + 1) * lda + i) * 2, lda);
            }
            A[(aOff + i * lda + i) * 2] = 1.0 - tau[tauOff + i * 2];
            A[(aOff + i * lda + i) * 2 + 1] = -tau[tauOff + i * 2 + 1];
            for (int l = 0; l < i; l++) {
                A[(aOff + l * lda + i) * 2] = 0.0;
                A[(aOff + l * lda + i) * 2 + 1] = 0.0;
            }
        }

        return 0;
    }

    static int zorgqr(int m, int n, int k, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (m < 0) return -1;
        if (n < 0) return -2;
        if (k < 0 || k > n) return -3;
        if (A == null) return -4;
        if (lda < max(1, n)) return -5;
        if (aOff < 0) return -6;
        if (tau == null) return -7;
        if (tauOff < 0) return -8;
        if (work == null) return -9;
        if (workOff < 0) return -10;
        if (lwork < -1) return -11;

        if (n == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        int minWork = max(1, n - 1);
        if (lwork < minWork && lwork != -1) return -11;

        if (lwork == -1) {
            work[workOff] = n;
            return 0;
        }

        return zorg2r(m, n, k, A, aOff, lda, tau, tauOff, work, workOff);
    }

    static int zungqr(int m, int n, int k, double[] A, int aOff, int lda,
                     double[] tau, int tOff, double[] work, int wOff, int lwork) {
        if (m < 0) return -1;
        if (n < 0) return -2;
        if (k < 0 || k > n) return -3;
        if (A == null) return -4;
        if (lda < max(1, n)) return -5;
        if (aOff < 0) return -6;
        if (tau == null) return -7;
        if (tOff < 0) return -8;
        if (work == null) return -9;
        if (wOff < 0) return -10;
        if (lwork < -1) return -11;

        if (n == 0) {
            if (lwork >= 1) work[wOff] = 1;
            return 0;
        }

        int nb = Ilaenv.ilaenv(1, "ZUNGQR", "", m, n, k, -1);
        int lworkopt = nb * n;

        int minWork = max(1, n);
        if (lwork < minWork && lwork != -1) return -11;

        if (lwork == -1) {
            work[wOff] = lworkopt;
            return 0;
        }

        int info = zorg2r(m, n, k, A, aOff, lda, tau, tOff, work, wOff);
        work[wOff] = lworkopt;
        return info;
    }
}
