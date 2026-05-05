/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.Ilaenv;

interface Zorgbr {

    static int zorgbr(char vect, int m, int n, int k, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        boolean wantq = (vect == 'Q' || vect == 'q');

        if (m == 0 || n == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        int mn = wantq ? m : n;
        int nb = Ilaenv.ilaenv(1, wantq ? "ZUNGQR" : "ZUNGLQ", " ", m, n, k, -1);
        int lwkopt = max(1, mn) * nb;

        if (lwork == -1) {
            work[workOff] = lwkopt;
            return 0;
        }

        if (wantq) {
            if (m >= k) {
                Zgeqr.zorgqr(m, n, k, A, aOff, lda, tau, tauOff, work, workOff, lwork);
            } else {
                for (int j = m - 1; j >= 1; j--) {
                    A[(aOff + j) * 2] = 0;
                    A[(aOff + j) * 2 + 1] = 0;
                    for (int i = j + 1; i < m; i++) {
                        A[(aOff + i * lda + j) * 2] = A[(aOff + i * lda + j - 1) * 2];
                        A[(aOff + i * lda + j) * 2 + 1] = A[(aOff + i * lda + j - 1) * 2 + 1];
                    }
                }
                A[aOff * 2] = 1;
                A[aOff * 2 + 1] = 0;
                for (int i = 1; i < m; i++) {
                    A[(aOff + i * lda) * 2] = 0;
                    A[(aOff + i * lda) * 2 + 1] = 0;
                }
                if (m > 1) {
                    Zgeqr.zorgqr(m - 1, m - 1, m - 1, A, aOff + lda + 1, lda, tau, tauOff, work, workOff, lwork);
                }
            }
        } else {
            if (k < n) {
                Zgelq.zorglq(m, n, k, A, aOff, lda, tau, tauOff, work, workOff, lwork);
            } else {
                A[aOff * 2] = 1;
                A[aOff * 2 + 1] = 0;
                for (int i = 1; i < n; i++) {
                    A[(aOff + i * lda) * 2] = 0;
                    A[(aOff + i * lda) * 2 + 1] = 0;
                }
                for (int j = 1; j < n; j++) {
                    for (int i = j - 1; i >= 1; i--) {
                        A[(aOff + i * lda + j) * 2] = A[(aOff + (i - 1) * lda + j) * 2];
                        A[(aOff + i * lda + j) * 2 + 1] = A[(aOff + (i - 1) * lda + j) * 2 + 1];
                    }
                    A[(aOff + j) * 2] = 0;
                    A[(aOff + j) * 2 + 1] = 0;
                }
                if (n > 1) {
                    Zgelq.zorglq(n - 1, n - 1, n - 1, A, aOff + lda + 1, lda, tau, tauOff, work, workOff, lwork);
                }
            }
        }

        work[workOff] = lwkopt;
        return 0;
    }

    static int max(int a, int b) {
        return a > b ? a : b;
    }
}
