/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

interface Zlacpy {

    static int zlacpy(char uplo, int m, int n, double[] A, int aOff, int lda, double[] B, int bOff, int ldb) {
        if (m < 0) return -1;
        if (n < 0) return -2;
        if (lda < max(1, n)) return -5;
        if (ldb < max(1, n)) return -8;

        char cuplo = Character.toUpperCase(uplo);

        switch (cuplo) {
            case 'U':
                for (int i = 0; i < m; i++) {
                    for (int j = i; j < n; j++) {
                        int aIdx = (aOff + i * lda + j) * 2;
                        int bIdx = (bOff + i * ldb + j) * 2;
                        B[bIdx] = A[aIdx];
                        B[bIdx + 1] = A[aIdx + 1];
                    }
                }
                break;
            case 'L':
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j <= i && j < n; j++) {
                        int aIdx = (aOff + i * lda + j) * 2;
                        int bIdx = (bOff + i * ldb + j) * 2;
                        B[bIdx] = A[aIdx];
                        B[bIdx + 1] = A[aIdx + 1];
                    }
                }
                break;
            default:
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        int aIdx = (aOff + i * lda + j) * 2;
                        int bIdx = (bOff + i * ldb + j) * 2;
                        B[bIdx] = A[aIdx];
                        B[bIdx + 1] = A[aIdx + 1];
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
