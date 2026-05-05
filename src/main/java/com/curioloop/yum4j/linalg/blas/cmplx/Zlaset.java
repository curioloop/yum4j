/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

interface Zlaset {

    static int zlaset(char uplo, int m, int n, double alphaRe, double alphaIm, double betaRe, double betaIm, double[] A, int aOff, int lda) {
        if (m < 0) return -1;
        if (n < 0) return -2;
        if (lda < max(1, n)) return -6;

        char cuplo = Character.toUpperCase(uplo);

        switch (cuplo) {
            case 'U':
                for (int j = 1; j < n; j++) {
                    for (int i = 0; i < Math.min(j, m); i++) {
                        int idx = (aOff + i * lda + j) * 2;
                        A[idx] = alphaRe;
                        A[idx + 1] = alphaIm;
                    }
                }
                for (int i = 0; i < Math.min(m, n); i++) {
                    int idx = (aOff + i * lda + i) * 2;
                    A[idx] = betaRe;
                    A[idx + 1] = betaIm;
                }
                break;
            case 'L':
                for (int j = 0; j < Math.min(n, m); j++) {
                    for (int i = j + 1; i < m; i++) {
                        int idx = (aOff + i * lda + j) * 2;
                        A[idx] = alphaRe;
                        A[idx + 1] = alphaIm;
                    }
                }
                for (int i = 0; i < Math.min(m, n); i++) {
                    int idx = (aOff + i * lda + i) * 2;
                    A[idx] = betaRe;
                    A[idx + 1] = betaIm;
                }
                break;
            default:
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        int idx = (aOff + i * lda + j) * 2;
                        A[idx] = alphaRe;
                        A[idx + 1] = alphaIm;
                    }
                }
                if (m <= n) {
                    for (int i = 0; i < m; i++) {
                        int idx = (aOff + i * lda + i) * 2;
                        A[idx] = betaRe;
                        A[idx + 1] = betaIm;
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        int idx = (aOff + i * lda + i) * 2;
                        A[idx] = betaRe;
                        A[idx + 1] = betaIm;
                    }
                }
                break;
        }

        return 0;
    }

    static int max(int a, int b) {
        return a > b ? a : b;
    }
}
