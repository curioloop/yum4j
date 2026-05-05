/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

interface Zormbr {

    static int zormbr(char vect, char side, char trans, int m, int n, int k, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] C, int cOff, int ldc, double[] work, int workOff, int lwork) {
        char cvect = Character.toUpperCase(vect);
        char cside = Character.toUpperCase(side);
        char ctrans = Character.toUpperCase(trans);

        if (cvect != 'Q' && cvect != 'P') return -1;
        if (cside != 'L' && cside != 'R') return -2;
        if (ctrans != 'N' && ctrans != 'T' && ctrans != 'C') return -3;
        if (m < 0) return -4;
        if (n < 0) return -5;
        if (k < 0) return -6;

        boolean applyq = (cvect == 'Q');
        boolean left = (cside == 'L');
        boolean notran = (ctrans == 'N');
        int nq = left ? m : n;

        if (applyq) {
            if (lda < max(1, nq)) return -8;
        } else {
            if (lda < max(1, min(nq, k))) return -8;
        }

        if (ldc < max(1, m)) return -11;

        if (m == 0 || n == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        if (applyq) {
            if (nq >= k) {
                return Zormqr.zormqr(side, trans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff, lwork);
            } else if (nq > 1) {
                int mi, ni, cRowOff, cColOff;
                if (left) {
                    mi = m - 1;
                    ni = n;
                    cRowOff = 1;
                    cColOff = 0;
                } else {
                    mi = m;
                    ni = n - 1;
                    cRowOff = 0;
                    cColOff = 1;
                }
                return Zormqr.zormqr(side, trans, mi, ni, nq - 1, A, aOff + lda, lda, tau, tauOff, C, cOff + cRowOff * ldc + cColOff, ldc, work, workOff, lwork);
            }
        } else {
            char transt = notran ? 'C' : 'N';
            if (nq > k) {
                return Zgelq.zormlq(side, transt, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff, lwork);
            } else if (nq > 1) {
                int mi, ni, cRowOff, cColOff;
                if (left) {
                    mi = m - 1;
                    ni = n;
                    cRowOff = 1;
                    cColOff = 0;
                } else {
                    mi = m;
                    ni = n - 1;
                    cRowOff = 0;
                    cColOff = 1;
                }
                return Zgelq.zormlq(side, transt, mi, ni, nq - 1, A, aOff + 1, lda, tau, tauOff, C, cOff + cRowOff * ldc + cColOff, ldc, work, workOff, lwork);
            }
        }

        if (lwork >= 1) work[workOff] = 1;
        return 0;
    }

    static int max(int a, int b) {
        return a > b ? a : b;
    }

    static int min(int a, int b) {
        return a < b ? a : b;
    }
}
