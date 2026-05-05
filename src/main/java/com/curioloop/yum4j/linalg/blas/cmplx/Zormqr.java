/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;
interface Zormqr {

    static int zormqr(char side, char trans, int m, int n, int k, double[] A, int aOff, int lda, double[] tau, int tauOff, double[] C, int cOff, int ldc, double[] work, int workOff, int lwork) {
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
        int nw = left ? n : m;
        if (left && k > m) return -5;
        if (!left && k > n) return -5;
        if (lda < max(1, k)) return -7;
        if (ldc < max(1, n)) return -10;

        if (m == 0 || n == 0 || k == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        if (lwork < max(1, nw) && lwork != -1) return -11;

        int nb = Ilaenv.ilaenv(1, "ZORMQR", "", m, n, k, -1);
        if (nb < 1) nb = 1;
        if (nb > k) nb = k;
        int lworkopt = max(1, nw) * nb;

        if (lwork == -1) {
            work[workOff] = lworkopt;
            return 0;
        }

        zunm2r(cside, ctrans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff);

        work[workOff] = lworkopt;
        return 0;
    }

    static void zunm2r(char cside, char ctrans, int m, int n, int k,
                       double[] A, int aOff, int lda, double[] tau, int tauOff,
                       double[] C, int cOff, int ldc, double[] work, int workOff) {
        boolean left = (cside == 'L');
        boolean notran = (ctrans == 'N');

        if (left) {
            if (notran) {
                for (int i = k - 1; i >= 0; i--) {
                    int aiiPos = (aOff + i * lda + i) * 2;
                    double savedRe = A[aiiPos]; double savedIm = A[aiiPos + 1];
                    A[aiiPos] = 1.0; A[aiiPos + 1] = 0.0;
                    Zlarf.zlarf(BLAS.Side.Left, m - i, n, A, aiiPos, lda, tau, tauOff + i * 2, C, (cOff + i * ldc) * 2, ldc, work, workOff);
                    A[aiiPos] = savedRe; A[aiiPos + 1] = savedIm;
                }
            } else {
                for (int i = 0; i < k; i++) {
                    int aiiPos = (aOff + i * lda + i) * 2;
                    double savedRe = A[aiiPos]; double savedIm = A[aiiPos + 1];
                    A[aiiPos] = 1.0; A[aiiPos + 1] = 0.0;
                    tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                    Zlarf.zlarf(BLAS.Side.Left, m - i, n, A, aiiPos, lda, tau, tauOff + i * 2, C, (cOff + i * ldc) * 2, ldc, work, workOff);
                    tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                    A[aiiPos] = savedRe; A[aiiPos + 1] = savedIm;
                }
            }
        } else {
            if (notran) {
                for (int i = 0; i < k; i++) {
                    int aiiPos = (aOff + i * lda + i) * 2;
                    double savedRe = A[aiiPos]; double savedIm = A[aiiPos + 1];
                    A[aiiPos] = 1.0; A[aiiPos + 1] = 0.0;
                    Zlarf.zlarf(BLAS.Side.Right, m, n - i, A, aiiPos, lda, tau, tauOff + i * 2, C, (cOff + i) * 2, ldc, work, workOff);
                    A[aiiPos] = savedRe; A[aiiPos + 1] = savedIm;
                }
            } else {
                for (int i = k - 1; i >= 0; i--) {
                    int aiiPos = (aOff + i * lda + i) * 2;
                    double savedRe = A[aiiPos]; double savedIm = A[aiiPos + 1];
                    A[aiiPos] = 1.0; A[aiiPos + 1] = 0.0;
                    tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                    Zlarf.zlarf(BLAS.Side.Right, m, n - i, A, aiiPos, lda, tau, tauOff + i * 2, C, (cOff + i) * 2, ldc, work, workOff);
                    tau[tauOff + i * 2 + 1] = -tau[tauOff + i * 2 + 1];
                    A[aiiPos] = savedRe; A[aiiPos + 1] = savedIm;
                }
            }
        }
    }

    static int max(int a, int b) { return a > b ? a : b; }
}