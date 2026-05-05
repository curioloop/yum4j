/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

import static java.lang.Math.min;

interface Zlauum {

    int BLOCK_SIZE = 64;

    static void zlauum(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        if (n == 0) return;

        boolean lower = uplo == BLAS.Uplo.Lower;

        if (BLOCK_SIZE <= 1 || n <= BLOCK_SIZE) {
            zlauu2(uplo, n, A, aOff, lda);
            return;
        }

        if (lower) {
            for (int i = 0; i < n; i += BLOCK_SIZE) {
                int ib = min(BLOCK_SIZE, n - i);

                Ztrmm.ztrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                        ib, i, 1.0, 0.0,
                        A, (aOff + i * lda + i) * 2, lda,
                        A, (aOff + i * lda) * 2, lda);
                zlauu2(uplo, ib, A, aOff + i * lda + i, lda);

                if (n - i - ib > 0) {
                    Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ib, i, n - i - ib,
                            1.0, 0.0,
                            A, aOff + (i + ib) * lda + i, lda,
                            A, aOff + (i + ib) * lda, lda,
                            1.0, 0.0,
                            A, aOff + i * lda, lda);
                    Zherk.zherk(BLAS.Uplo.Lower, BLAS.Trans.Conj, ib, n - i - ib,
                            1.0,
                            A, aOff + (i + ib) * lda + i, lda,
                            1.0,
                            A, aOff + i * lda + i, lda);
                }
            }
        } else {
            for (int i = 0; i < n; i += BLOCK_SIZE) {
                int ib = min(BLOCK_SIZE, n - i);

                Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                        i, ib, 1.0, 0.0,
                        A, (aOff + i * lda + i) * 2, lda,
                        A, (aOff + i) * 2, lda);
                zlauu2(uplo, ib, A, aOff + i * lda + i, lda);

                if (n - i - ib > 0) {
                    Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, i, ib, n - i - ib,
                            1.0, 0.0,
                            A, aOff + i + ib, lda,
                            A, aOff + i * lda + i + ib, lda,
                            1.0, 0.0,
                            A, aOff + i, lda);
                    Zherk.zherk(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, ib, n - i - ib,
                            1.0,
                            A, aOff + i * lda + i + ib, lda,
                            1.0,
                            A, aOff + i * lda + i, lda);
                }
            }
        }
    }

    static void zlauu2(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        boolean lower = uplo == BLAS.Uplo.Lower;

        if (lower) {
            for (int i = 0; i < n; i++) {
                int rowI = aOff + i * lda;
                double aii = A[(rowI + i) * 2];
                if (i < n - 1) {
                    double sum = aii * aii;
                    for (int k = i + 1; k < n; k++) {
                        int p = (aOff + k * lda + i) * 2;
                        double re = A[p], im = A[p + 1];
                        sum += re * re + im * im;
                    }
                    A[(rowI + i) * 2] = sum;
                    A[(rowI + i) * 2 + 1] = 0;

                    for (int j = 0; j < i; j++) {
                        double dotRe = 0, dotIm = 0;
                        for (int k = i + 1; k < n; k++) {
                            int a1p = (aOff + k * lda + i) * 2;
                            int a2p = (aOff + k * lda + j) * 2;
                            double a1Re = A[a1p], a1Im = A[a1p + 1];
                            double a2Re = A[a2p], a2Im = A[a2p + 1];
                            dotRe += a1Re * a2Re + a1Im * a2Im;
                            dotIm += a1Im * a2Re - a1Re * a2Im;
                        }
                        int jp = (rowI + j) * 2;
                        double oldRe = A[jp], oldIm = A[jp + 1];
                        A[jp] = aii * oldRe + dotRe;
                        A[jp + 1] = aii * oldIm + dotIm;
                    }
                } else {
                    for (int j = 0; j <= i; j++) {
                        int jp = (rowI + j) * 2;
                        A[jp] = aii * A[jp];
                        A[jp + 1] = aii * A[jp + 1];
                    }
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int rowI = aOff + i * lda;
                double aii = A[(rowI + i) * 2];
                if (i < n - 1) {
                    double sum = aii * aii;
                    for (int k = i + 1; k < n; k++) {
                        int p = (rowI + k) * 2;
                        double re = A[p], im = A[p + 1];
                        sum += re * re + im * im;
                    }
                    A[(rowI + i) * 2] = sum;
                    A[(rowI + i) * 2 + 1] = 0;

                    for (int k = 0; k < i; k++) {
                        double dotRe = 0, dotIm = 0;
                        for (int j = i + 1; j < n; j++) {
                            int a1p = (aOff + k * lda + j) * 2;
                            int a2p = (rowI + j) * 2;
                            double a1Re = A[a1p], a1Im = A[a1p + 1];
                            double a2Re = A[a2p], a2Im = A[a2p + 1];
                            dotRe += a1Re * a2Re + a1Im * a2Im;
                            dotIm += a1Im * a2Re - a1Re * a2Im;
                        }
                        int kp = (aOff + k * lda + i) * 2;
                        double oldRe = A[kp], oldIm = A[kp + 1];
                        A[kp] = aii * oldRe + dotRe;
                        A[kp + 1] = aii * oldIm + dotIm;
                    }
                } else {
                    for (int k = 0; k <= i; k++) {
                        int kp = (aOff + k * lda + i) * 2;
                        A[kp] = aii * A[kp];
                        A[kp + 1] = aii * A[kp + 1];
                    }
                }
            }
        }
    }
}
