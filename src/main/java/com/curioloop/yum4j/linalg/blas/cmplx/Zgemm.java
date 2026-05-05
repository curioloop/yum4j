/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
/**
 * ZGEMM performs complex matrix-matrix multiplication.
 *
 * <p>BLAS Level-3 operation: C := alpha*op(A)*op(B) + beta*C</p>
 * <p>Storage layout: Each complex number is stored as two consecutive doubles: [real, imag]</p>
 * <p>Reference: BLAS ZGEMM</p>
 */
interface Zgemm {

    int BLOCK_M = 32;
    int BLOCK_N = 32;
    int BLOCK_K = 64;

    /**
     * Performs complex matrix-matrix multiplication.
     *
     * <p>Operation: C := alpha*op(A)*op(B) + beta*C</p>
     * 
     * @param transA 'N' for no transpose, 'T' for transpose, 'C' for conjugate transpose
     * @param transB 'N' for no transpose, 'T' for transpose, 'C' for conjugate transpose
     * @param m number of rows in op(A) and C
     * @param n number of columns in op(B) and C
     * @param k number of columns in op(A) and rows in op(B)
     * @param alphaRe real part of alpha
     * @param alphaIm imaginary part of alpha
     * @param A complex matrix (interleaved storage)
     * @param aOff offset into A (in complex elements)
     * @param lda leading dimension of A (in complex elements)
     * @param B complex matrix (interleaved storage)
     * @param bOff offset into B (in complex elements)
     * @param ldb leading dimension of B (in complex elements)
     * @param betaRe real part of beta
     * @param betaIm imaginary part of beta
     * @param C complex matrix (interleaved storage)
     * @param cOff offset into C (in complex elements)
     * @param ldc leading dimension of C (in complex elements)
     */
    static void zgemm(BLAS.Trans transA, BLAS.Trans transB, int m, int n, int k,
                      double alphaRe, double alphaIm,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb,
                      double betaRe, double betaIm,
                      double[] C, int cOff, int ldc) {

        boolean transAFlag = transA == BLAS.Trans.Trans || transA == BLAS.Trans.Conj;
        boolean conjAFlag = transA == BLAS.Trans.Conj;
        boolean transBFlag = transB == BLAS.Trans.Trans || transB == BLAS.Trans.Conj;
        boolean conjBFlag = transB == BLAS.Trans.Conj;

        if (m <= 0 || n <= 0 || ((alphaRe == 0.0 && alphaIm == 0.0) || k <= 0) && (betaRe == 1.0 && betaIm == 0.0)) {
            return;
        }

        scaleC(C, cOff, ldc, m, n, betaRe, betaIm);

        if (alphaRe == 0.0 && alphaIm == 0.0 || k <= 0) {
            return;
        }

        if (!transAFlag && !transBFlag) {
            zgemmNN(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        } else if (transAFlag && !transBFlag) {
            zgemmTN(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjAFlag);
        } else if (!transAFlag && transBFlag) {
            zgemmNT(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjBFlag);
        } else {
            zgemmTT(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjAFlag, conjBFlag);
        }
    }

    static void scaleC(double[] C, int cOff, int ldc, int m, int n, double betaRe, double betaIm) {
        if (betaRe == 0.0 && betaIm == 0.0) {
            for (int i = 0; i < m; i++) {
                int cIdx = (cOff + i * ldc) * 2;
                for (int j = 0; j < n; j++) {
                    C[cIdx + j * 2] = 0.0;
                    C[cIdx + j * 2 + 1] = 0.0;
                }
            }
        } else if (betaRe != 1.0 || betaIm != 0.0) {
            for (int i = 0; i < m; i++) {
                int cIdx = (cOff + i * ldc) * 2;
                for (int j = 0; j < n; j++) {
                    double cre = C[cIdx + j * 2];
                    double cim = C[cIdx + j * 2 + 1];
                    C[cIdx + j * 2] = betaRe * cre - betaIm * cim;
                    C[cIdx + j * 2 + 1] = betaRe * cim + betaIm * cre;
                }
            }
        }
    }

    // ======================== NN ========================

    static void zgemmNN(int m, int n, int k, double alphaRe, double alphaIm,
                        double[] A, int aOff, int lda,
                        double[] B, int bOff, int ldb,
                        double[] C, int cOff, int ldc) {
        int maxBlock = Math.max(BLOCK_M, Math.max(BLOCK_N, BLOCK_K));
        if (m < maxBlock && n < maxBlock && k < maxBlock) {
            zgemmNNDirect(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        } else {
            zgemmNNBlocked(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        }
    }

    static void zgemmNNDirect(int m, int n, int k, double alphaRe, double alphaIm,
                              double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc) {
        int m2 = (m / 2) * 2;
        int n2 = (n / 2) * 2;
        int strideA2 = lda * 2;
        int strideB2 = ldb * 2;
        int strideC2 = ldc * 2;
        int bBase = bOff * 2;

        for (int i = 0; i < m2; i += 2) {
            int aRow0 = (aOff + i * lda) * 2;
            int aRow1 = aRow0 + strideA2;
            int cRow0 = (cOff + i * ldc) * 2;
            int cRow1 = cRow0 + strideC2;

            for (int j = 0; j < n2; j += 2) {
                double c00r = 0, c00i = 0, c01r = 0, c01i = 0;
                double c10r = 0, c10i = 0, c11r = 0, c11i = 0;

                for (int l = 0; l < k; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int bRow = bBase + l * strideB2;
                    int j0 = j * 2, j1 = j0 + 2;
                    double b0r = B[bRow + j0], b0i = B[bRow + j0 + 1];
                    double b1r = B[bRow + j1], b1i = B[bRow + j1 + 1];
                    c00r = Math.fma(a0r, b0r, Math.fma(-a0i, b0i, c00r));
                    c00i = Math.fma(a0r, b0i, Math.fma(a0i, b0r, c00i));
                    c01r = Math.fma(a0r, b1r, Math.fma(-a0i, b1i, c01r));
                    c01i = Math.fma(a0r, b1i, Math.fma(a0i, b1r, c01i));
                    c10r = Math.fma(a1r, b0r, Math.fma(-a1i, b0i, c10r));
                    c10i = Math.fma(a1r, b0i, Math.fma(a1i, b0r, c10i));
                    c11r = Math.fma(a1r, b1r, Math.fma(-a1i, b1i, c11r));
                    c11i = Math.fma(a1r, b1i, Math.fma(a1i, b1r, c11i));
                }

                int j0 = j * 2, j1 = j0 + 2;
                C[cRow0 + j0] += alphaRe * c00r - alphaIm * c00i;
                C[cRow0 + j0 + 1] += alphaRe * c00i + alphaIm * c00r;
                C[cRow0 + j1] += alphaRe * c01r - alphaIm * c01i;
                C[cRow0 + j1 + 1] += alphaRe * c01i + alphaIm * c01r;
                C[cRow1 + j0] += alphaRe * c10r - alphaIm * c10i;
                C[cRow1 + j0 + 1] += alphaRe * c10i + alphaIm * c10r;
                C[cRow1 + j1] += alphaRe * c11r - alphaIm * c11i;
                C[cRow1 + j1 + 1] += alphaRe * c11i + alphaIm * c11r;
            }

            for (int j = n2; j < n; j++) {
                double s0r = 0, s0i = 0, s1r = 0, s1i = 0;
                for (int l = 0; l < k; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int bRow = bBase + l * strideB2;
                    int jj = j * 2;
                    double br = B[bRow + jj], bi = B[bRow + jj + 1];
                    s0r = Math.fma(a0r, br, Math.fma(-a0i, bi, s0r));
                    s0i = Math.fma(a0r, bi, Math.fma(a0i, br, s0i));
                    s1r = Math.fma(a1r, br, Math.fma(-a1i, bi, s1r));
                    s1i = Math.fma(a1r, bi, Math.fma(a1i, br, s1i));
                }
                int jj = j * 2;
                C[cRow0 + jj] += alphaRe * s0r - alphaIm * s0i;
                C[cRow0 + jj + 1] += alphaRe * s0i + alphaIm * s0r;
                C[cRow1 + jj] += alphaRe * s1r - alphaIm * s1i;
                C[cRow1 + jj + 1] += alphaRe * s1i + alphaIm * s1r;
            }
        }

        for (int i = m2; i < m; i++) {
            int aRow = (aOff + i * lda) * 2;
            int cRow = (cOff + i * ldc) * 2;
            for (int j = 0; j < n2; j += 2) {
                double c0r = 0, c0i = 0, c1r = 0, c1i = 0;
                for (int l = 0; l < k; l++) {
                    double ar = A[aRow + l * 2], ai = A[aRow + l * 2 + 1];
                    int bRow = bBase + l * strideB2;
                    int j0 = j * 2, j1 = j0 + 2;
                    double b0r = B[bRow + j0], b0i = B[bRow + j0 + 1];
                    double b1r = B[bRow + j1], b1i = B[bRow + j1 + 1];
                    c0r = Math.fma(ar, b0r, Math.fma(-ai, b0i, c0r));
                    c0i = Math.fma(ar, b0i, Math.fma(ai, b0r, c0i));
                    c1r = Math.fma(ar, b1r, Math.fma(-ai, b1i, c1r));
                    c1i = Math.fma(ar, b1i, Math.fma(ai, b1r, c1i));
                }
                int j0 = j * 2, j1 = j0 + 2;
                C[cRow + j0] += alphaRe * c0r - alphaIm * c0i;
                C[cRow + j0 + 1] += alphaRe * c0i + alphaIm * c0r;
                C[cRow + j1] += alphaRe * c1r - alphaIm * c1i;
                C[cRow + j1 + 1] += alphaRe * c1i + alphaIm * c1r;
            }
            for (int j = n2; j < n; j++) {
                double sr = 0, si = 0;
                for (int l = 0; l < k; l++) {
                    double ar = A[aRow + l * 2], ai = A[aRow + l * 2 + 1];
                    int bRow = bBase + l * strideB2;
                    double br = B[bRow + j * 2], bi = B[bRow + j * 2 + 1];
                    sr = Math.fma(ar, br, Math.fma(-ai, bi, sr));
                    si = Math.fma(ar, bi, Math.fma(ai, br, si));
                }
                C[cRow + j * 2] += alphaRe * sr - alphaIm * si;
                C[cRow + j * 2 + 1] += alphaRe * si + alphaIm * sr;
            }
        }
    }

    static void zgemmNNBlocked(int m, int n, int k, double alphaRe, double alphaIm,
                               double[] A, int aOff, int lda,
                               double[] B, int bOff, int ldb,
                               double[] C, int cOff, int ldc) {
        int outerM = BLOCK_M * 2;
        int outerN = BLOCK_N * 2;

        for (int ii = 0; ii < m; ii += outerM) {
            int iMax = Math.min(ii + outerM, m);
            for (int jj = 0; jj < n; jj += outerN) {
                int jMax = Math.min(jj + outerN, n);
                for (int ii2 = ii; ii2 < iMax; ii2 += BLOCK_M) {
                    int iMax2 = Math.min(ii2 + BLOCK_M, iMax);
                    for (int jj2 = jj; jj2 < jMax; jj2 += BLOCK_N) {
                        int jMax2 = Math.min(jj2 + BLOCK_N, jMax);
                        zgemmNNMicro(ii2, iMax2, jj2, jMax2, k, alphaRe, alphaIm,
                                A, aOff, lda, B, bOff, ldb, C, cOff, ldc, BLOCK_K);
                    }
                }
            }
        }
    }

    static void zgemmNNMicro(int iStart, int iEnd, int jStart, int jEnd, int kTotal,
                             double alphaRe, double alphaIm,
                             double[] A, int aOff, int lda,
                             double[] B, int bOff, int ldb,
                             double[] C, int cOff, int ldc, int kBlockSize) {
        int i2End = iStart + ((iEnd - iStart) / 2) * 2;
        int j2End = jStart + ((jEnd - jStart) / 2) * 2;
        int strideA2 = lda * 2;
        int strideB2 = ldb * 2;
        int strideC2 = ldc * 2;
        int bBase = bOff * 2;

        for (int i = iStart; i < i2End; i += 2) {
            int aRow0 = (aOff + i * lda) * 2;
            int aRow1 = aRow0 + strideA2;
            int cRow0 = (cOff + i * ldc) * 2;
            int cRow1 = cRow0 + strideC2;

            for (int j = jStart; j < j2End; j += 2) {
                double c00r = 0, c00i = 0, c01r = 0, c01i = 0;
                double c10r = 0, c10i = 0, c11r = 0, c11i = 0;

                for (int kk = 0; kk < kTotal; kk += kBlockSize) {
                    int kEnd = Math.min(kk + kBlockSize, kTotal);
                    for (int l = kk; l < kEnd; l++) {
                        int aIdx = l * 2;
                        double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                        double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                        int bRow = bBase + l * strideB2;
                        int j0 = j * 2, j1 = j0 + 2;
                        double b0r = B[bRow + j0], b0i = B[bRow + j0 + 1];
                        double b1r = B[bRow + j1], b1i = B[bRow + j1 + 1];
                        c00r = Math.fma(a0r, b0r, Math.fma(-a0i, b0i, c00r));
                        c00i = Math.fma(a0r, b0i, Math.fma(a0i, b0r, c00i));
                        c01r = Math.fma(a0r, b1r, Math.fma(-a0i, b1i, c01r));
                        c01i = Math.fma(a0r, b1i, Math.fma(a0i, b1r, c01i));
                        c10r = Math.fma(a1r, b0r, Math.fma(-a1i, b0i, c10r));
                        c10i = Math.fma(a1r, b0i, Math.fma(a1i, b0r, c10i));
                        c11r = Math.fma(a1r, b1r, Math.fma(-a1i, b1i, c11r));
                        c11i = Math.fma(a1r, b1i, Math.fma(a1i, b1r, c11i));
                    }
                }

                int j0 = j * 2, j1 = j0 + 2;
                C[cRow0 + j0] += alphaRe * c00r - alphaIm * c00i;
                C[cRow0 + j0 + 1] += alphaRe * c00i + alphaIm * c00r;
                C[cRow0 + j1] += alphaRe * c01r - alphaIm * c01i;
                C[cRow0 + j1 + 1] += alphaRe * c01i + alphaIm * c01r;
                C[cRow1 + j0] += alphaRe * c10r - alphaIm * c10i;
                C[cRow1 + j0 + 1] += alphaRe * c10i + alphaIm * c10r;
                C[cRow1 + j1] += alphaRe * c11r - alphaIm * c11i;
                C[cRow1 + j1 + 1] += alphaRe * c11i + alphaIm * c11r;
            }

            for (int j = j2End; j < jEnd; j++) {
                double s0r = 0, s0i = 0, s1r = 0, s1i = 0;
                for (int l = 0; l < kTotal; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int bRow = bBase + l * strideB2;
                    int jj = j * 2;
                    double br = B[bRow + jj], bi = B[bRow + jj + 1];
                    s0r = Math.fma(a0r, br, Math.fma(-a0i, bi, s0r));
                    s0i = Math.fma(a0r, bi, Math.fma(a0i, br, s0i));
                    s1r = Math.fma(a1r, br, Math.fma(-a1i, bi, s1r));
                    s1i = Math.fma(a1r, bi, Math.fma(a1i, br, s1i));
                }
                int jj = j * 2;
                C[cRow0 + jj] += alphaRe * s0r - alphaIm * s0i;
                C[cRow0 + jj + 1] += alphaRe * s0i + alphaIm * s0r;
                C[cRow1 + jj] += alphaRe * s1r - alphaIm * s1i;
                C[cRow1 + jj + 1] += alphaRe * s1i + alphaIm * s1r;
            }
        }

        for (int i = i2End; i < iEnd; i++) {
            int aRow = (aOff + i * lda) * 2;
            int cRow = (cOff + i * ldc) * 2;
            for (int j = jStart; j < j2End; j += 2) {
                double c0r = 0, c0i = 0, c1r = 0, c1i = 0;
                for (int l = 0; l < kTotal; l++) {
                    double ar = A[aRow + l * 2], ai = A[aRow + l * 2 + 1];
                    int bRow = bBase + l * strideB2;
                    int j0 = j * 2, j1 = j0 + 2;
                    double b0r = B[bRow + j0], b0i = B[bRow + j0 + 1];
                    double b1r = B[bRow + j1], b1i = B[bRow + j1 + 1];
                    c0r = Math.fma(ar, b0r, Math.fma(-ai, b0i, c0r));
                    c0i = Math.fma(ar, b0i, Math.fma(ai, b0r, c0i));
                    c1r = Math.fma(ar, b1r, Math.fma(-ai, b1i, c1r));
                    c1i = Math.fma(ar, b1i, Math.fma(ai, b1r, c1i));
                }
                int j0 = j * 2, j1 = j0 + 2;
                C[cRow + j0] += alphaRe * c0r - alphaIm * c0i;
                C[cRow + j0 + 1] += alphaRe * c0i + alphaIm * c0r;
                C[cRow + j1] += alphaRe * c1r - alphaIm * c1i;
                C[cRow + j1 + 1] += alphaRe * c1i + alphaIm * c1r;
            }
            for (int j = j2End; j < jEnd; j++) {
                double sr = 0, si = 0;
                for (int l = 0; l < kTotal; l++) {
                    double ar = A[aRow + l * 2], ai = A[aRow + l * 2 + 1];
                    int bRow = bBase + l * strideB2;
                    double br = B[bRow + j * 2], bi = B[bRow + j * 2 + 1];
                    sr = Math.fma(ar, br, Math.fma(-ai, bi, sr));
                    si = Math.fma(ar, bi, Math.fma(ai, br, si));
                }
                C[cRow + j * 2] += alphaRe * sr - alphaIm * si;
                C[cRow + j * 2 + 1] += alphaRe * si + alphaIm * sr;
            }
        }
    }

    // ======================== TN ========================

    static void zgemmTN(int m, int n, int k, double alphaRe, double alphaIm,
                        double[] A, int aOff, int lda,
                        double[] B, int bOff, int ldb,
                        double[] C, int cOff, int ldc, boolean conjA) {
        int maxBlock = Math.max(BLOCK_M, Math.max(BLOCK_N, BLOCK_K));
        if (m < maxBlock && n < maxBlock && k < maxBlock) {
            zgemmTNDirect(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjA);
        } else {
            zgemmTNBlocked(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjA);
        }
    }

    static void zgemmTNDirect(int m, int n, int k, double alphaRe, double alphaIm,
                              double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc, boolean conjA) {
        int m2 = (m / 2) * 2;
        int n2 = (n / 2) * 2;
        int strideB2 = ldb * 2;
        int strideC2 = ldc * 2;
        int bBase = bOff * 2;

        for (int i = 0; i < m2; i += 2) {
            int cRow0 = (cOff + i * ldc) * 2;
            int cRow1 = cRow0 + strideC2;

            for (int j = 0; j < n2; j += 2) {
                double c00r = 0, c00i = 0, c01r = 0, c01i = 0;
                double c10r = 0, c10i = 0, c11r = 0, c11i = 0;

                for (int l = 0; l < k; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double a0r = A[aIdx], a0i = A[aIdx + 1];
                    double a1r = A[aIdx + 2], a1i = A[aIdx + 3];
                    if (conjA) { a0i = -a0i; a1i = -a1i; }
                    int bRow = bBase + l * strideB2;
                    int j0 = j * 2, j1 = j0 + 2;
                    double b0r = B[bRow + j0], b0i = B[bRow + j0 + 1];
                    double b1r = B[bRow + j1], b1i = B[bRow + j1 + 1];
                    c00r = Math.fma(a0r, b0r, Math.fma(-a0i, b0i, c00r));
                    c00i = Math.fma(a0r, b0i, Math.fma(a0i, b0r, c00i));
                    c01r = Math.fma(a0r, b1r, Math.fma(-a0i, b1i, c01r));
                    c01i = Math.fma(a0r, b1i, Math.fma(a0i, b1r, c01i));
                    c10r = Math.fma(a1r, b0r, Math.fma(-a1i, b0i, c10r));
                    c10i = Math.fma(a1r, b0i, Math.fma(a1i, b0r, c10i));
                    c11r = Math.fma(a1r, b1r, Math.fma(-a1i, b1i, c11r));
                    c11i = Math.fma(a1r, b1i, Math.fma(a1i, b1r, c11i));
                }

                int j0 = j * 2, j1 = j0 + 2;
                C[cRow0 + j0] += alphaRe * c00r - alphaIm * c00i;
                C[cRow0 + j0 + 1] += alphaRe * c00i + alphaIm * c00r;
                C[cRow0 + j1] += alphaRe * c01r - alphaIm * c01i;
                C[cRow0 + j1 + 1] += alphaRe * c01i + alphaIm * c01r;
                C[cRow1 + j0] += alphaRe * c10r - alphaIm * c10i;
                C[cRow1 + j0 + 1] += alphaRe * c10i + alphaIm * c10r;
                C[cRow1 + j1] += alphaRe * c11r - alphaIm * c11i;
                C[cRow1 + j1 + 1] += alphaRe * c11i + alphaIm * c11r;
            }

            for (int j = n2; j < n; j++) {
                double s0r = 0, s0i = 0, s1r = 0, s1i = 0;
                for (int l = 0; l < k; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double a0r = A[aIdx], a0i = A[aIdx + 1];
                    double a1r = A[aIdx + 2], a1i = A[aIdx + 3];
                    if (conjA) { a0i = -a0i; a1i = -a1i; }
                    int bRow = bBase + l * strideB2;
                    int jj = j * 2;
                    double br = B[bRow + jj], bi = B[bRow + jj + 1];
                    s0r = Math.fma(a0r, br, Math.fma(-a0i, bi, s0r));
                    s0i = Math.fma(a0r, bi, Math.fma(a0i, br, s0i));
                    s1r = Math.fma(a1r, br, Math.fma(-a1i, bi, s1r));
                    s1i = Math.fma(a1r, bi, Math.fma(a1i, br, s1i));
                }
                int jj = j * 2;
                C[cRow0 + jj] += alphaRe * s0r - alphaIm * s0i;
                C[cRow0 + jj + 1] += alphaRe * s0i + alphaIm * s0r;
                C[cRow1 + jj] += alphaRe * s1r - alphaIm * s1i;
                C[cRow1 + jj + 1] += alphaRe * s1i + alphaIm * s1r;
            }
        }

        for (int i = m2; i < m; i++) {
            int cRow = (cOff + i * ldc) * 2;
            for (int j = 0; j < n; j++) {
                double sr = 0, si = 0;
                for (int l = 0; l < k; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double ar = A[aIdx], ai = A[aIdx + 1];
                    if (conjA) ai = -ai;
                    int bRow = bBase + l * strideB2;
                    double br = B[bRow + j * 2], bi = B[bRow + j * 2 + 1];
                    sr = Math.fma(ar, br, Math.fma(-ai, bi, sr));
                    si = Math.fma(ar, bi, Math.fma(ai, br, si));
                }
                C[cRow + j * 2] += alphaRe * sr - alphaIm * si;
                C[cRow + j * 2 + 1] += alphaRe * si + alphaIm * sr;
            }
        }
    }

    static void zgemmTNBlocked(int m, int n, int k, double alphaRe, double alphaIm,
                               double[] A, int aOff, int lda,
                               double[] B, int bOff, int ldb,
                               double[] C, int cOff, int ldc, boolean conjA) {
        int outerM = BLOCK_M * 2;
        int outerN = BLOCK_N * 2;

        for (int ii = 0; ii < m; ii += outerM) {
            int iMax = Math.min(ii + outerM, m);
            for (int jj = 0; jj < n; jj += outerN) {
                int jMax = Math.min(jj + outerN, n);
                for (int kk = 0; kk < k; kk += BLOCK_K) {
                    int kEnd = Math.min(kk + BLOCK_K, k);
                    for (int i = ii; i < iMax; i += BLOCK_M) {
                        int iEnd2 = Math.min(i + BLOCK_M, iMax);
                        for (int j = jj; j < jMax; j += BLOCK_N) {
                            int jEnd2 = Math.min(j + BLOCK_N, jMax);
                            zgemmTNBlock(i, iEnd2, j, jEnd2, kk, kEnd, alphaRe, alphaIm,
                                    A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjA);
                        }
                    }
                }
            }
        }
    }

    static void zgemmTNBlock(int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd,
                             double alphaRe, double alphaIm,
                             double[] A, int aOff, int lda,
                             double[] B, int bOff, int ldb,
                             double[] C, int cOff, int ldc, boolean conjA) {
        int i2End = iStart + ((iEnd - iStart) / 2) * 2;
        int j2End = jStart + ((jEnd - jStart) / 2) * 2;
        int strideB2 = ldb * 2;
        int strideC2 = ldc * 2;
        int bBase = bOff * 2;

        for (int i = iStart; i < i2End; i += 2) {
            int cRow0 = (cOff + i * ldc) * 2;
            int cRow1 = cRow0 + strideC2;

            for (int j = jStart; j < j2End; j += 2) {
                double c00r = 0, c00i = 0, c01r = 0, c01i = 0;
                double c10r = 0, c10i = 0, c11r = 0, c11i = 0;

                for (int l = kStart; l < kEnd; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double a0r = A[aIdx], a0i = A[aIdx + 1];
                    double a1r = A[aIdx + 2], a1i = A[aIdx + 3];
                    if (conjA) { a0i = -a0i; a1i = -a1i; }
                    int bRow = bBase + l * strideB2;
                    int j0 = j * 2, j1 = j0 + 2;
                    double b0r = B[bRow + j0], b0i = B[bRow + j0 + 1];
                    double b1r = B[bRow + j1], b1i = B[bRow + j1 + 1];
                    c00r = Math.fma(a0r, b0r, Math.fma(-a0i, b0i, c00r));
                    c00i = Math.fma(a0r, b0i, Math.fma(a0i, b0r, c00i));
                    c01r = Math.fma(a0r, b1r, Math.fma(-a0i, b1i, c01r));
                    c01i = Math.fma(a0r, b1i, Math.fma(a0i, b1r, c01i));
                    c10r = Math.fma(a1r, b0r, Math.fma(-a1i, b0i, c10r));
                    c10i = Math.fma(a1r, b0i, Math.fma(a1i, b0r, c10i));
                    c11r = Math.fma(a1r, b1r, Math.fma(-a1i, b1i, c11r));
                    c11i = Math.fma(a1r, b1i, Math.fma(a1i, b1r, c11i));
                }

                int j0 = j * 2, j1 = j0 + 2;
                C[cRow0 + j0] += alphaRe * c00r - alphaIm * c00i;
                C[cRow0 + j0 + 1] += alphaRe * c00i + alphaIm * c00r;
                C[cRow0 + j1] += alphaRe * c01r - alphaIm * c01i;
                C[cRow0 + j1 + 1] += alphaRe * c01i + alphaIm * c01r;
                C[cRow1 + j0] += alphaRe * c10r - alphaIm * c10i;
                C[cRow1 + j0 + 1] += alphaRe * c10i + alphaIm * c10r;
                C[cRow1 + j1] += alphaRe * c11r - alphaIm * c11i;
                C[cRow1 + j1 + 1] += alphaRe * c11i + alphaIm * c11r;
            }

            for (int j = j2End; j < jEnd; j++) {
                double s0r = 0, s0i = 0, s1r = 0, s1i = 0;
                for (int l = kStart; l < kEnd; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double a0r = A[aIdx], a0i = A[aIdx + 1];
                    double a1r = A[aIdx + 2], a1i = A[aIdx + 3];
                    if (conjA) { a0i = -a0i; a1i = -a1i; }
                    int bRow = bBase + l * strideB2;
                    int jj = j * 2;
                    double br = B[bRow + jj], bi = B[bRow + jj + 1];
                    s0r = Math.fma(a0r, br, Math.fma(-a0i, bi, s0r));
                    s0i = Math.fma(a0r, bi, Math.fma(a0i, br, s0i));
                    s1r = Math.fma(a1r, br, Math.fma(-a1i, bi, s1r));
                    s1i = Math.fma(a1r, bi, Math.fma(a1i, br, s1i));
                }
                int jj = j * 2;
                C[cRow0 + jj] += alphaRe * s0r - alphaIm * s0i;
                C[cRow0 + jj + 1] += alphaRe * s0i + alphaIm * s0r;
                C[cRow1 + jj] += alphaRe * s1r - alphaIm * s1i;
                C[cRow1 + jj + 1] += alphaRe * s1i + alphaIm * s1r;
            }
        }

        for (int i = i2End; i < iEnd; i++) {
            int cRow = (cOff + i * ldc) * 2;
            for (int j = jStart; j < jEnd; j++) {
                double sr = 0, si = 0;
                for (int l = kStart; l < kEnd; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double ar = A[aIdx], ai = A[aIdx + 1];
                    if (conjA) ai = -ai;
                    int bRow = bBase + l * strideB2;
                    double br = B[bRow + j * 2], bi = B[bRow + j * 2 + 1];
                    sr = Math.fma(ar, br, Math.fma(-ai, bi, sr));
                    si = Math.fma(ar, bi, Math.fma(ai, br, si));
                }
                C[cRow + j * 2] += alphaRe * sr - alphaIm * si;
                C[cRow + j * 2 + 1] += alphaRe * si + alphaIm * sr;
            }
        }
    }

    // ======================== NT ========================

    static void zgemmNT(int m, int n, int k, double alphaRe, double alphaIm,
                        double[] A, int aOff, int lda,
                        double[] B, int bOff, int ldb,
                        double[] C, int cOff, int ldc, boolean conjB) {
        int maxBlock = Math.max(BLOCK_M, Math.max(BLOCK_N, BLOCK_K));
        if (conjB) {
            if (m < maxBlock && n < maxBlock && k < maxBlock) {
                zgemmNTConjDirect(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
            } else {
                zgemmNTConjBlocked(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
            }
            return;
        }
        if (m < maxBlock && n < maxBlock && k < maxBlock) {
            zgemmNTDirect(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjB);
        } else {
            zgemmNTBlocked(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjB);
        }
    }

    static void zgemmNTConjDirect(int m, int n, int k, double alphaRe, double alphaIm,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  double[] C, int cOff, int ldc) {
        int m2 = (m / 2) * 2;
        int n2 = (n / 2) * 2;
        int strideA2 = lda * 2;
        int strideC2 = ldc * 2;

        for (int i = 0; i < m2; i += 2) {
            int aRow0 = (aOff + i * lda) * 2;
            int aRow1 = aRow0 + strideA2;
            int cRow0 = (cOff + i * ldc) * 2;
            int cRow1 = cRow0 + strideC2;

            for (int j = 0; j < n2; j += 2) {
                double c00r = 0, c00i = 0, c01r = 0, c01i = 0;
                double c10r = 0, c10i = 0, c11r = 0, c11i = 0;
                int bCol0 = (bOff + j * ldb) * 2;
                int bCol1 = (bOff + (j + 1) * ldb) * 2;

                for (int l = 0; l < k; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int l2 = l * 2;
                    double b0r = B[bCol0 + l2], b0i = B[bCol0 + l2 + 1];
                    double b1r = B[bCol1 + l2], b1i = B[bCol1 + l2 + 1];
                    c00r = Math.fma(a0r, b0r, Math.fma(a0i, b0i, c00r));
                    c00i = Math.fma(a0i, b0r, Math.fma(-a0r, b0i, c00i));
                    c01r = Math.fma(a0r, b1r, Math.fma(a0i, b1i, c01r));
                    c01i = Math.fma(a0i, b1r, Math.fma(-a0r, b1i, c01i));
                    c10r = Math.fma(a1r, b0r, Math.fma(a1i, b0i, c10r));
                    c10i = Math.fma(a1i, b0r, Math.fma(-a1r, b0i, c10i));
                    c11r = Math.fma(a1r, b1r, Math.fma(a1i, b1i, c11r));
                    c11i = Math.fma(a1i, b1r, Math.fma(-a1r, b1i, c11i));
                }

                int j0 = j * 2, j1 = j0 + 2;
                C[cRow0 + j0] += alphaRe * c00r - alphaIm * c00i;
                C[cRow0 + j0 + 1] += alphaRe * c00i + alphaIm * c00r;
                C[cRow0 + j1] += alphaRe * c01r - alphaIm * c01i;
                C[cRow0 + j1 + 1] += alphaRe * c01i + alphaIm * c01r;
                C[cRow1 + j0] += alphaRe * c10r - alphaIm * c10i;
                C[cRow1 + j0 + 1] += alphaRe * c10i + alphaIm * c10r;
                C[cRow1 + j1] += alphaRe * c11r - alphaIm * c11i;
                C[cRow1 + j1 + 1] += alphaRe * c11i + alphaIm * c11r;
            }

            for (int j = n2; j < n; j++) {
                double s0r = 0, s0i = 0, s1r = 0, s1i = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = 0; l < k; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    s0r = Math.fma(a0r, br, Math.fma(a0i, bi, s0r));
                    s0i = Math.fma(a0i, br, Math.fma(-a0r, bi, s0i));
                    s1r = Math.fma(a1r, br, Math.fma(a1i, bi, s1r));
                    s1i = Math.fma(a1i, br, Math.fma(-a1r, bi, s1i));
                }
                int jj = j * 2;
                C[cRow0 + jj] += alphaRe * s0r - alphaIm * s0i;
                C[cRow0 + jj + 1] += alphaRe * s0i + alphaIm * s0r;
                C[cRow1 + jj] += alphaRe * s1r - alphaIm * s1i;
                C[cRow1 + jj + 1] += alphaRe * s1i + alphaIm * s1r;
            }
        }

        for (int i = m2; i < m; i++) {
            int aRow = (aOff + i * lda) * 2;
            int cRow = (cOff + i * ldc) * 2;
            for (int j = 0; j < n; j++) {
                double sr = 0, si = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = 0; l < k; l++) {
                    double ar = A[aRow + l * 2], ai = A[aRow + l * 2 + 1];
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    sr = Math.fma(ar, br, Math.fma(ai, bi, sr));
                    si = Math.fma(ai, br, Math.fma(-ar, bi, si));
                }
                C[cRow + j * 2] += alphaRe * sr - alphaIm * si;
                C[cRow + j * 2 + 1] += alphaRe * si + alphaIm * sr;
            }
        }
    }

    static void zgemmNTDirect(int m, int n, int k, double alphaRe, double alphaIm,
                              double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc, boolean conjB) {
        int m2 = (m / 2) * 2;
        int n2 = (n / 2) * 2;
        int strideA2 = lda * 2;
        int strideC2 = ldc * 2;

        for (int i = 0; i < m2; i += 2) {
            int aRow0 = (aOff + i * lda) * 2;
            int aRow1 = aRow0 + strideA2;
            int cRow0 = (cOff + i * ldc) * 2;
            int cRow1 = cRow0 + strideC2;

            for (int j = 0; j < n2; j += 2) {
                double c00r = 0, c00i = 0, c01r = 0, c01i = 0;
                double c10r = 0, c10i = 0, c11r = 0, c11i = 0;
                int bCol0 = (bOff + j * ldb) * 2;
                int bCol1 = (bOff + (j + 1) * ldb) * 2;

                for (int l = 0; l < k; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int l2 = l * 2;
                    double b0r = B[bCol0 + l2], b0i = B[bCol0 + l2 + 1];
                    double b1r = B[bCol1 + l2], b1i = B[bCol1 + l2 + 1];
                    if (conjB) { b0i = -b0i; b1i = -b1i; }
                    c00r = Math.fma(a0r, b0r, Math.fma(-a0i, b0i, c00r));
                    c00i = Math.fma(a0r, b0i, Math.fma(a0i, b0r, c00i));
                    c01r = Math.fma(a0r, b1r, Math.fma(-a0i, b1i, c01r));
                    c01i = Math.fma(a0r, b1i, Math.fma(a0i, b1r, c01i));
                    c10r = Math.fma(a1r, b0r, Math.fma(-a1i, b0i, c10r));
                    c10i = Math.fma(a1r, b0i, Math.fma(a1i, b0r, c10i));
                    c11r = Math.fma(a1r, b1r, Math.fma(-a1i, b1i, c11r));
                    c11i = Math.fma(a1r, b1i, Math.fma(a1i, b1r, c11i));
                }

                int j0 = j * 2, j1 = j0 + 2;
                C[cRow0 + j0] += alphaRe * c00r - alphaIm * c00i;
                C[cRow0 + j0 + 1] += alphaRe * c00i + alphaIm * c00r;
                C[cRow0 + j1] += alphaRe * c01r - alphaIm * c01i;
                C[cRow0 + j1 + 1] += alphaRe * c01i + alphaIm * c01r;
                C[cRow1 + j0] += alphaRe * c10r - alphaIm * c10i;
                C[cRow1 + j0 + 1] += alphaRe * c10i + alphaIm * c10r;
                C[cRow1 + j1] += alphaRe * c11r - alphaIm * c11i;
                C[cRow1 + j1 + 1] += alphaRe * c11i + alphaIm * c11r;
            }

            for (int j = n2; j < n; j++) {
                double s0r = 0, s0i = 0, s1r = 0, s1i = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = 0; l < k; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    if (conjB) bi = -bi;
                    s0r = Math.fma(a0r, br, Math.fma(-a0i, bi, s0r));
                    s0i = Math.fma(a0r, bi, Math.fma(a0i, br, s0i));
                    s1r = Math.fma(a1r, br, Math.fma(-a1i, bi, s1r));
                    s1i = Math.fma(a1r, bi, Math.fma(a1i, br, s1i));
                }
                int jj = j * 2;
                C[cRow0 + jj] += alphaRe * s0r - alphaIm * s0i;
                C[cRow0 + jj + 1] += alphaRe * s0i + alphaIm * s0r;
                C[cRow1 + jj] += alphaRe * s1r - alphaIm * s1i;
                C[cRow1 + jj + 1] += alphaRe * s1i + alphaIm * s1r;
            }
        }

        for (int i = m2; i < m; i++) {
            int aRow = (aOff + i * lda) * 2;
            int cRow = (cOff + i * ldc) * 2;
            for (int j = 0; j < n; j++) {
                double sr = 0, si = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = 0; l < k; l++) {
                    double ar = A[aRow + l * 2], ai = A[aRow + l * 2 + 1];
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    if (conjB) bi = -bi;
                    sr = Math.fma(ar, br, Math.fma(-ai, bi, sr));
                    si = Math.fma(ar, bi, Math.fma(ai, br, si));
                }
                C[cRow + j * 2] += alphaRe * sr - alphaIm * si;
                C[cRow + j * 2 + 1] += alphaRe * si + alphaIm * sr;
            }
        }
    }

    static void zgemmNTBlocked(int m, int n, int k, double alphaRe, double alphaIm,
                               double[] A, int aOff, int lda,
                               double[] B, int bOff, int ldb,
                               double[] C, int cOff, int ldc, boolean conjB) {
        int outerM = BLOCK_M * 2;
        int outerN = BLOCK_N * 2;

        for (int ii = 0; ii < m; ii += outerM) {
            int iMax = Math.min(ii + outerM, m);
            for (int jj = 0; jj < n; jj += outerN) {
                int jMax = Math.min(jj + outerN, n);
                for (int kk = 0; kk < k; kk += BLOCK_K) {
                    int kEnd = Math.min(kk + BLOCK_K, k);
                    for (int i = ii; i < iMax; i += BLOCK_M) {
                        int iEnd2 = Math.min(i + BLOCK_M, iMax);
                        for (int j = jj; j < jMax; j += BLOCK_N) {
                            int jEnd2 = Math.min(j + BLOCK_N, jMax);
                            zgemmNTBlock(i, iEnd2, j, jEnd2, kk, kEnd, alphaRe, alphaIm,
                                    A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjB);
                        }
                    }
                }
            }
        }
    }

    static void zgemmNTConjBlocked(int m, int n, int k, double alphaRe, double alphaIm,
                                   double[] A, int aOff, int lda,
                                   double[] B, int bOff, int ldb,
                                   double[] C, int cOff, int ldc) {
        int outerM = BLOCK_M * 2;
        int outerN = BLOCK_N * 2;

        for (int ii = 0; ii < m; ii += outerM) {
            int iMax = Math.min(ii + outerM, m);
            for (int jj = 0; jj < n; jj += outerN) {
                int jMax = Math.min(jj + outerN, n);
                for (int kk = 0; kk < k; kk += BLOCK_K) {
                    int kEnd = Math.min(kk + BLOCK_K, k);
                    for (int i = ii; i < iMax; i += BLOCK_M) {
                        int iEnd2 = Math.min(i + BLOCK_M, iMax);
                        for (int j = jj; j < jMax; j += BLOCK_N) {
                            int jEnd2 = Math.min(j + BLOCK_N, jMax);
                            zgemmNTConjBlock(i, iEnd2, j, jEnd2, kk, kEnd, alphaRe, alphaIm,
                                A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
                        }
                    }
                }
            }
        }
    }

    static void zgemmNTConjBlock(int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd,
                                 double alphaRe, double alphaIm,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 double[] C, int cOff, int ldc) {
        int i2End = iStart + ((iEnd - iStart) / 2) * 2;
        int j2End = jStart + ((jEnd - jStart) / 2) * 2;
        int strideA2 = lda * 2;
        int strideC2 = ldc * 2;

        for (int i = iStart; i < i2End; i += 2) {
            int aRow0 = (aOff + i * lda) * 2;
            int aRow1 = aRow0 + strideA2;
            int cRow0 = (cOff + i * ldc) * 2;
            int cRow1 = cRow0 + strideC2;

            for (int j = jStart; j < j2End; j += 2) {
                double c00r = 0, c00i = 0, c01r = 0, c01i = 0;
                double c10r = 0, c10i = 0, c11r = 0, c11i = 0;
                int bCol0 = (bOff + j * ldb) * 2;
                int bCol1 = (bOff + (j + 1) * ldb) * 2;

                for (int l = kStart; l < kEnd; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int l2 = l * 2;
                    double b0r = B[bCol0 + l2], b0i = B[bCol0 + l2 + 1];
                    double b1r = B[bCol1 + l2], b1i = B[bCol1 + l2 + 1];
                    c00r = Math.fma(a0r, b0r, Math.fma(a0i, b0i, c00r));
                    c00i = Math.fma(a0i, b0r, Math.fma(-a0r, b0i, c00i));
                    c01r = Math.fma(a0r, b1r, Math.fma(a0i, b1i, c01r));
                    c01i = Math.fma(a0i, b1r, Math.fma(-a0r, b1i, c01i));
                    c10r = Math.fma(a1r, b0r, Math.fma(a1i, b0i, c10r));
                    c10i = Math.fma(a1i, b0r, Math.fma(-a1r, b0i, c10i));
                    c11r = Math.fma(a1r, b1r, Math.fma(a1i, b1i, c11r));
                    c11i = Math.fma(a1i, b1r, Math.fma(-a1r, b1i, c11i));
                }

                int j0 = j * 2, j1 = j0 + 2;
                C[cRow0 + j0] += alphaRe * c00r - alphaIm * c00i;
                C[cRow0 + j0 + 1] += alphaRe * c00i + alphaIm * c00r;
                C[cRow0 + j1] += alphaRe * c01r - alphaIm * c01i;
                C[cRow0 + j1 + 1] += alphaRe * c01i + alphaIm * c01r;
                C[cRow1 + j0] += alphaRe * c10r - alphaIm * c10i;
                C[cRow1 + j0 + 1] += alphaRe * c10i + alphaIm * c10r;
                C[cRow1 + j1] += alphaRe * c11r - alphaIm * c11i;
                C[cRow1 + j1 + 1] += alphaRe * c11i + alphaIm * c11r;
            }

            for (int j = j2End; j < jEnd; j++) {
                double s0r = 0, s0i = 0, s1r = 0, s1i = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = kStart; l < kEnd; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    s0r = Math.fma(a0r, br, Math.fma(a0i, bi, s0r));
                    s0i = Math.fma(a0i, br, Math.fma(-a0r, bi, s0i));
                    s1r = Math.fma(a1r, br, Math.fma(a1i, bi, s1r));
                    s1i = Math.fma(a1i, br, Math.fma(-a1r, bi, s1i));
                }
                int jj = j * 2;
                C[cRow0 + jj] += alphaRe * s0r - alphaIm * s0i;
                C[cRow0 + jj + 1] += alphaRe * s0i + alphaIm * s0r;
                C[cRow1 + jj] += alphaRe * s1r - alphaIm * s1i;
                C[cRow1 + jj + 1] += alphaRe * s1i + alphaIm * s1r;
            }
        }

        for (int i = i2End; i < iEnd; i++) {
            int aRow = (aOff + i * lda) * 2;
            int cRow = (cOff + i * ldc) * 2;
            for (int j = jStart; j < jEnd; j++) {
                double sr = 0, si = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = kStart; l < kEnd; l++) {
                    double ar = A[aRow + l * 2], ai = A[aRow + l * 2 + 1];
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    sr = Math.fma(ar, br, Math.fma(ai, bi, sr));
                    si = Math.fma(ai, br, Math.fma(-ar, bi, si));
                }
                int jj = j * 2;
                C[cRow + jj] += alphaRe * sr - alphaIm * si;
                C[cRow + jj + 1] += alphaRe * si + alphaIm * sr;
            }
        }
    }

    static void zgemmNTBlock(int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd,
                             double alphaRe, double alphaIm,
                             double[] A, int aOff, int lda,
                             double[] B, int bOff, int ldb,
                             double[] C, int cOff, int ldc, boolean conjB) {
        int i2End = iStart + ((iEnd - iStart) / 2) * 2;
        int j2End = jStart + ((jEnd - jStart) / 2) * 2;
        int strideA2 = lda * 2;
        int strideC2 = ldc * 2;

        for (int i = iStart; i < i2End; i += 2) {
            int aRow0 = (aOff + i * lda) * 2;
            int aRow1 = aRow0 + strideA2;
            int cRow0 = (cOff + i * ldc) * 2;
            int cRow1 = cRow0 + strideC2;

            for (int j = jStart; j < j2End; j += 2) {
                double c00r = 0, c00i = 0, c01r = 0, c01i = 0;
                double c10r = 0, c10i = 0, c11r = 0, c11i = 0;
                int bCol0 = (bOff + j * ldb) * 2;
                int bCol1 = (bOff + (j + 1) * ldb) * 2;

                for (int l = kStart; l < kEnd; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int l2 = l * 2;
                    double b0r = B[bCol0 + l2], b0i = B[bCol0 + l2 + 1];
                    double b1r = B[bCol1 + l2], b1i = B[bCol1 + l2 + 1];
                    if (conjB) { b0i = -b0i; b1i = -b1i; }
                    c00r = Math.fma(a0r, b0r, Math.fma(-a0i, b0i, c00r));
                    c00i = Math.fma(a0r, b0i, Math.fma(a0i, b0r, c00i));
                    c01r = Math.fma(a0r, b1r, Math.fma(-a0i, b1i, c01r));
                    c01i = Math.fma(a0r, b1i, Math.fma(a0i, b1r, c01i));
                    c10r = Math.fma(a1r, b0r, Math.fma(-a1i, b0i, c10r));
                    c10i = Math.fma(a1r, b0i, Math.fma(a1i, b0r, c10i));
                    c11r = Math.fma(a1r, b1r, Math.fma(-a1i, b1i, c11r));
                    c11i = Math.fma(a1r, b1i, Math.fma(a1i, b1r, c11i));
                }

                int j0 = j * 2, j1 = j0 + 2;
                C[cRow0 + j0] += alphaRe * c00r - alphaIm * c00i;
                C[cRow0 + j0 + 1] += alphaRe * c00i + alphaIm * c00r;
                C[cRow0 + j1] += alphaRe * c01r - alphaIm * c01i;
                C[cRow0 + j1 + 1] += alphaRe * c01i + alphaIm * c01r;
                C[cRow1 + j0] += alphaRe * c10r - alphaIm * c10i;
                C[cRow1 + j0 + 1] += alphaRe * c10i + alphaIm * c10r;
                C[cRow1 + j1] += alphaRe * c11r - alphaIm * c11i;
                C[cRow1 + j1 + 1] += alphaRe * c11i + alphaIm * c11r;
            }

            for (int j = j2End; j < jEnd; j++) {
                double s0r = 0, s0i = 0, s1r = 0, s1i = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = kStart; l < kEnd; l++) {
                    int aIdx = l * 2;
                    double a0r = A[aRow0 + aIdx], a0i = A[aRow0 + aIdx + 1];
                    double a1r = A[aRow1 + aIdx], a1i = A[aRow1 + aIdx + 1];
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    if (conjB) bi = -bi;
                    s0r = Math.fma(a0r, br, Math.fma(-a0i, bi, s0r));
                    s0i = Math.fma(a0r, bi, Math.fma(a0i, br, s0i));
                    s1r = Math.fma(a1r, br, Math.fma(-a1i, bi, s1r));
                    s1i = Math.fma(a1r, bi, Math.fma(a1i, br, s1i));
                }
                int jj = j * 2;
                C[cRow0 + jj] += alphaRe * s0r - alphaIm * s0i;
                C[cRow0 + jj + 1] += alphaRe * s0i + alphaIm * s0r;
                C[cRow1 + jj] += alphaRe * s1r - alphaIm * s1i;
                C[cRow1 + jj + 1] += alphaRe * s1i + alphaIm * s1r;
            }
        }

        for (int i = i2End; i < iEnd; i++) {
            int aRow = (aOff + i * lda) * 2;
            int cRow = (cOff + i * ldc) * 2;
            for (int j = jStart; j < jEnd; j++) {
                double sr = 0, si = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = kStart; l < kEnd; l++) {
                    double ar = A[aRow + l * 2], ai = A[aRow + l * 2 + 1];
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    if (conjB) bi = -bi;
                    sr = Math.fma(ar, br, Math.fma(-ai, bi, sr));
                    si = Math.fma(ar, bi, Math.fma(ai, br, si));
                }
                C[cRow + j * 2] += alphaRe * sr - alphaIm * si;
                C[cRow + j * 2 + 1] += alphaRe * si + alphaIm * sr;
            }
        }
    }

    // ======================== TT ========================

    static void zgemmTT(int m, int n, int k, double alphaRe, double alphaIm,
                        double[] A, int aOff, int lda,
                        double[] B, int bOff, int ldb,
                        double[] C, int cOff, int ldc, boolean conjA, boolean conjB) {
        int maxBlock = Math.max(BLOCK_M, Math.max(BLOCK_N, BLOCK_K));
        if (m < maxBlock && n < maxBlock && k < maxBlock) {
            zgemmTTDirect(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjA, conjB);
        } else {
            zgemmTTBlocked(m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjA, conjB);
        }
    }

    static void zgemmTTDirect(int m, int n, int k, double alphaRe, double alphaIm,
                              double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc, boolean conjA, boolean conjB) {
        int m2 = (m / 2) * 2;
        int n2 = (n / 2) * 2;
        int strideC2 = ldc * 2;

        for (int i = 0; i < m2; i += 2) {
            int cRow0 = (cOff + i * ldc) * 2;
            int cRow1 = cRow0 + strideC2;

            for (int j = 0; j < n2; j += 2) {
                double c00r = 0, c00i = 0, c01r = 0, c01i = 0;
                double c10r = 0, c10i = 0, c11r = 0, c11i = 0;
                int bCol0 = (bOff + j * ldb) * 2;
                int bCol1 = (bOff + (j + 1) * ldb) * 2;

                for (int l = 0; l < k; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double a0r = A[aIdx], a0i = A[aIdx + 1];
                    double a1r = A[aIdx + 2], a1i = A[aIdx + 3];
                    if (conjA) { a0i = -a0i; a1i = -a1i; }
                    int l2 = l * 2;
                    double b0r = B[bCol0 + l2], b0i = B[bCol0 + l2 + 1];
                    double b1r = B[bCol1 + l2], b1i = B[bCol1 + l2 + 1];
                    if (conjB) { b0i = -b0i; b1i = -b1i; }
                    c00r = Math.fma(a0r, b0r, Math.fma(-a0i, b0i, c00r));
                    c00i = Math.fma(a0r, b0i, Math.fma(a0i, b0r, c00i));
                    c01r = Math.fma(a0r, b1r, Math.fma(-a0i, b1i, c01r));
                    c01i = Math.fma(a0r, b1i, Math.fma(a0i, b1r, c01i));
                    c10r = Math.fma(a1r, b0r, Math.fma(-a1i, b0i, c10r));
                    c10i = Math.fma(a1r, b0i, Math.fma(a1i, b0r, c10i));
                    c11r = Math.fma(a1r, b1r, Math.fma(-a1i, b1i, c11r));
                    c11i = Math.fma(a1r, b1i, Math.fma(a1i, b1r, c11i));
                }

                int j0 = j * 2, j1 = j0 + 2;
                C[cRow0 + j0] += alphaRe * c00r - alphaIm * c00i;
                C[cRow0 + j0 + 1] += alphaRe * c00i + alphaIm * c00r;
                C[cRow0 + j1] += alphaRe * c01r - alphaIm * c01i;
                C[cRow0 + j1 + 1] += alphaRe * c01i + alphaIm * c01r;
                C[cRow1 + j0] += alphaRe * c10r - alphaIm * c10i;
                C[cRow1 + j0 + 1] += alphaRe * c10i + alphaIm * c10r;
                C[cRow1 + j1] += alphaRe * c11r - alphaIm * c11i;
                C[cRow1 + j1 + 1] += alphaRe * c11i + alphaIm * c11r;
            }

            for (int j = n2; j < n; j++) {
                double s0r = 0, s0i = 0, s1r = 0, s1i = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = 0; l < k; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double a0r = A[aIdx], a0i = A[aIdx + 1];
                    double a1r = A[aIdx + 2], a1i = A[aIdx + 3];
                    if (conjA) { a0i = -a0i; a1i = -a1i; }
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    if (conjB) bi = -bi;
                    s0r = Math.fma(a0r, br, Math.fma(-a0i, bi, s0r));
                    s0i = Math.fma(a0r, bi, Math.fma(a0i, br, s0i));
                    s1r = Math.fma(a1r, br, Math.fma(-a1i, bi, s1r));
                    s1i = Math.fma(a1r, bi, Math.fma(a1i, br, s1i));
                }
                int jj = j * 2;
                C[cRow0 + jj] += alphaRe * s0r - alphaIm * s0i;
                C[cRow0 + jj + 1] += alphaRe * s0i + alphaIm * s0r;
                C[cRow1 + jj] += alphaRe * s1r - alphaIm * s1i;
                C[cRow1 + jj + 1] += alphaRe * s1i + alphaIm * s1r;
            }
        }

        for (int i = m2; i < m; i++) {
            int cRow = (cOff + i * ldc) * 2;
            for (int j = 0; j < n; j++) {
                double sr = 0, si = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = 0; l < k; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double ar = A[aIdx], ai = A[aIdx + 1];
                    if (conjA) ai = -ai;
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    if (conjB) bi = -bi;
                    sr = Math.fma(ar, br, Math.fma(-ai, bi, sr));
                    si = Math.fma(ar, bi, Math.fma(ai, br, si));
                }
                C[cRow + j * 2] += alphaRe * sr - alphaIm * si;
                C[cRow + j * 2 + 1] += alphaRe * si + alphaIm * sr;
            }
        }
    }

    static void zgemmTTBlocked(int m, int n, int k, double alphaRe, double alphaIm,
                               double[] A, int aOff, int lda,
                               double[] B, int bOff, int ldb,
                               double[] C, int cOff, int ldc, boolean conjA, boolean conjB) {
        int outerM = BLOCK_M * 2;
        int outerN = BLOCK_N * 2;

        for (int ii = 0; ii < m; ii += outerM) {
            int iMax = Math.min(ii + outerM, m);
            for (int jj = 0; jj < n; jj += outerN) {
                int jMax = Math.min(jj + outerN, n);
                for (int kk = 0; kk < k; kk += BLOCK_K) {
                    int kEnd = Math.min(kk + BLOCK_K, k);
                    for (int i = ii; i < iMax; i += BLOCK_M) {
                        int iEnd2 = Math.min(i + BLOCK_M, iMax);
                        for (int j = jj; j < jMax; j += BLOCK_N) {
                            int jEnd2 = Math.min(j + BLOCK_N, jMax);
                            zgemmTTBlock(i, iEnd2, j, jEnd2, kk, kEnd, alphaRe, alphaIm,
                                    A, aOff, lda, B, bOff, ldb, C, cOff, ldc, conjA, conjB);
                        }
                    }
                }
            }
        }
    }

    static void zgemmTTBlock(int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd,
                             double alphaRe, double alphaIm,
                             double[] A, int aOff, int lda,
                             double[] B, int bOff, int ldb,
                             double[] C, int cOff, int ldc, boolean conjA, boolean conjB) {
        int i2End = iStart + ((iEnd - iStart) / 2) * 2;
        int j2End = jStart + ((jEnd - jStart) / 2) * 2;
        int strideC2 = ldc * 2;

        for (int i = iStart; i < i2End; i += 2) {
            int cRow0 = (cOff + i * ldc) * 2;
            int cRow1 = cRow0 + strideC2;

            for (int j = jStart; j < j2End; j += 2) {
                double c00r = 0, c00i = 0, c01r = 0, c01i = 0;
                double c10r = 0, c10i = 0, c11r = 0, c11i = 0;
                int bCol0 = (bOff + j * ldb) * 2;
                int bCol1 = (bOff + (j + 1) * ldb) * 2;

                for (int l = kStart; l < kEnd; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double a0r = A[aIdx], a0i = A[aIdx + 1];
                    double a1r = A[aIdx + 2], a1i = A[aIdx + 3];
                    if (conjA) { a0i = -a0i; a1i = -a1i; }
                    int l2 = l * 2;
                    double b0r = B[bCol0 + l2], b0i = B[bCol0 + l2 + 1];
                    double b1r = B[bCol1 + l2], b1i = B[bCol1 + l2 + 1];
                    if (conjB) { b0i = -b0i; b1i = -b1i; }
                    c00r = Math.fma(a0r, b0r, Math.fma(-a0i, b0i, c00r));
                    c00i = Math.fma(a0r, b0i, Math.fma(a0i, b0r, c00i));
                    c01r = Math.fma(a0r, b1r, Math.fma(-a0i, b1i, c01r));
                    c01i = Math.fma(a0r, b1i, Math.fma(a0i, b1r, c01i));
                    c10r = Math.fma(a1r, b0r, Math.fma(-a1i, b0i, c10r));
                    c10i = Math.fma(a1r, b0i, Math.fma(a1i, b0r, c10i));
                    c11r = Math.fma(a1r, b1r, Math.fma(-a1i, b1i, c11r));
                    c11i = Math.fma(a1r, b1i, Math.fma(a1i, b1r, c11i));
                }

                int j0 = j * 2, j1 = j0 + 2;
                C[cRow0 + j0] += alphaRe * c00r - alphaIm * c00i;
                C[cRow0 + j0 + 1] += alphaRe * c00i + alphaIm * c00r;
                C[cRow0 + j1] += alphaRe * c01r - alphaIm * c01i;
                C[cRow0 + j1 + 1] += alphaRe * c01i + alphaIm * c01r;
                C[cRow1 + j0] += alphaRe * c10r - alphaIm * c10i;
                C[cRow1 + j0 + 1] += alphaRe * c10i + alphaIm * c10r;
                C[cRow1 + j1] += alphaRe * c11r - alphaIm * c11i;
                C[cRow1 + j1 + 1] += alphaRe * c11i + alphaIm * c11r;
            }

            for (int j = j2End; j < jEnd; j++) {
                double s0r = 0, s0i = 0, s1r = 0, s1i = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = kStart; l < kEnd; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double a0r = A[aIdx], a0i = A[aIdx + 1];
                    double a1r = A[aIdx + 2], a1i = A[aIdx + 3];
                    if (conjA) { a0i = -a0i; a1i = -a1i; }
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    if (conjB) bi = -bi;
                    s0r = Math.fma(a0r, br, Math.fma(-a0i, bi, s0r));
                    s0i = Math.fma(a0r, bi, Math.fma(a0i, br, s0i));
                    s1r = Math.fma(a1r, br, Math.fma(-a1i, bi, s1r));
                    s1i = Math.fma(a1r, bi, Math.fma(a1i, br, s1i));
                }
                int jj = j * 2;
                C[cRow0 + jj] += alphaRe * s0r - alphaIm * s0i;
                C[cRow0 + jj + 1] += alphaRe * s0i + alphaIm * s0r;
                C[cRow1 + jj] += alphaRe * s1r - alphaIm * s1i;
                C[cRow1 + jj + 1] += alphaRe * s1i + alphaIm * s1r;
            }
        }

        for (int i = i2End; i < iEnd; i++) {
            int cRow = (cOff + i * ldc) * 2;
            for (int j = jStart; j < jEnd; j++) {
                double sr = 0, si = 0;
                int bCol = (bOff + j * ldb) * 2;
                for (int l = kStart; l < kEnd; l++) {
                    int aIdx = (aOff + l * lda + i) * 2;
                    double ar = A[aIdx], ai = A[aIdx + 1];
                    if (conjA) ai = -ai;
                    int l2 = l * 2;
                    double br = B[bCol + l2], bi = B[bCol + l2 + 1];
                    if (conjB) bi = -bi;
                    sr = Math.fma(ar, br, Math.fma(-ai, bi, sr));
                    si = Math.fma(ar, bi, Math.fma(ai, br, si));
                }
                C[cRow + j * 2] += alphaRe * sr - alphaIm * si;
                C[cRow + j * 2 + 1] += alphaRe * si + alphaIm * sr;
            }
        }
    }
}
