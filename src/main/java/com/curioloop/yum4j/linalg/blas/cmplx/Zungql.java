/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;

import static java.lang.Math.max;

interface Zungql {

    static int zungql(int m, int n, int k, double[] A, int aOff, int lda, double[] tau, int tOff, double[] work, int wOff, int lwork) {
        if (m < 0) return -1;
        if (n < 0 || n > m) return -2;
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
            work[wOff] = 1;
            return 0;
        }

        int nb = Ilaenv.ilaenv(1, "ZUNGQL", "", m, n, k, -1);
        int lworkopt = max(1, n) * nb;

        if (lwork == -1) {
            work[wOff] = lworkopt;
            return 0;
        }

        if (lwork < max(1, n)) {
            return -11;
        }

        zung2l(m, n, k, A, aOff, lda, tau, tOff, work, wOff);

        work[wOff] = lworkopt;
        return 0;
    }

    static void zung2l(int m, int n, int k, double[] A, int aOff, int lda, double[] tau, int tOff, double[] work, int wOff) {
        if (n == 0) return;

        for (int j = 0; j < n - k; j++) {
            for (int l = 0; l < m; l++) {
                int pos = (aOff + l * lda + j) * 2;
                A[pos] = 0.0;
                A[pos + 1] = 0.0;
            }
            int diagPos = (aOff + (m - n + j) * lda + j) * 2;
            A[diagPos] = 1.0;
            A[diagPos + 1] = 0.0;
        }

        for (int i = 0; i < k; i++) {
            int ii = n - k + i;

            int diagPos = (aOff + (m - n + ii) * lda + ii) * 2;
            A[diagPos] = 1.0;
            A[diagPos + 1] = 0.0;

            if (ii > 0) {
                Zlarf.zlarf(BLAS.Side.Left, m - n + ii + 1, ii,
                        A, (aOff + ii) * 2, lda, tau, (tOff + i) * 2,
                        A, aOff * 2, lda, work, wOff);
            }

            int colStart = (aOff + ii) * 2;
            int colLen = m - n + ii;
            double tauRe = tau[(tOff + i) * 2];
            double tauIm = tau[(tOff + i) * 2 + 1];
            Zscal.zscal(colLen, -tauRe, -tauIm, A, colStart, lda);

            diagPos = (aOff + (m - n + ii) * lda + ii) * 2;
            A[diagPos] = 1.0 - tauRe;
            A[diagPos + 1] = -tauIm;

            for (int l = m - n + ii + 1; l < m; l++) {
                int pos = (aOff + l * lda + ii) * 2;
                A[pos] = 0.0;
                A[pos + 1] = 0.0;
            }
        }
    }
}
