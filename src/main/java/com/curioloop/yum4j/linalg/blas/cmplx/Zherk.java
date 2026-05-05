/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

interface Zherk {

    int THRESHOLD = 64;

    static void zherk(BLAS.Uplo uplo, BLAS.Trans trans, int n, int k,
                      double alpha,
                      double[] A, int aOff, int lda,
                      double beta,
                      double[] C, int cOff, int ldc) {
        if (n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0)) return;

        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean noTrans = trans == BLAS.Trans.NoTrans;

        if (beta != 1.0) {
            scaleHermitian(C, cOff, ldc, n, beta, upper);
        }

        if (alpha == 0.0) return;

        if (noTrans) {
            herkNoTrans(n, k, alpha, A, aOff, lda, C, cOff, ldc, upper);
        } else {
            herkConjTrans(n, k, alpha, A, aOff, lda, C, cOff, ldc, upper);
        }
    }

    static void scaleHermitian(double[] C, int cOff, int ldc, int n, double beta, boolean upper) {
        if (beta == 0.0) {
            if (upper) {
                for (int i = 0; i < n; i++) {
                    int rowOff = (cOff + i * ldc) * 2;
                    for (int j = i; j < n; j++) {
                        C[rowOff + j * 2] = 0.0;
                        C[rowOff + j * 2 + 1] = 0.0;
                    }
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int rowOff = (cOff + i * ldc) * 2;
                    for (int j = 0; j <= i; j++) {
                        C[rowOff + j * 2] = 0.0;
                        C[rowOff + j * 2 + 1] = 0.0;
                    }
                }
            }
        } else {
            if (upper) {
                for (int i = 0; i < n; i++) {
                    int rowOff = (cOff + i * ldc) * 2;
                    for (int j = i; j < n; j++) {
                        if (i == j) {
                            C[rowOff + j * 2] *= beta;
                            C[rowOff + j * 2 + 1] = 0.0;
                        } else {
                            C[rowOff + j * 2] *= beta;
                            C[rowOff + j * 2 + 1] *= beta;
                        }
                    }
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int rowOff = (cOff + i * ldc) * 2;
                    for (int j = 0; j <= i; j++) {
                        if (i == j) {
                            C[rowOff + j * 2] *= beta;
                            C[rowOff + j * 2 + 1] = 0.0;
                        } else {
                            C[rowOff + j * 2] *= beta;
                            C[rowOff + j * 2 + 1] *= beta;
                        }
                    }
                }
            }
        }
    }

    static void herkNoTrans(int n, int k, double alpha,
                            double[] A, int aOff, int lda,
                            double[] C, int cOff, int ldc, boolean upper) {
        if (k <= 0) return;
        herkNoTransImpl(n, k, alpha, A, aOff, lda, C, cOff, ldc, upper, 0);
    }

    static void herkNoTransImpl(int n, int k, double alpha,
                                double[] A, int aOff, int lda,
                                double[] C, int cOff, int ldc, boolean upper, int kStart) {
        if (k <= THRESHOLD) {
            if (upper) {
                for (int i = 0; i < n; i++) {
                    int aiOff = (aOff + i * lda + kStart) * 2;
                    for (int j = i; j < n; j++) {
                        int ajOff = (aOff + j * lda + kStart) * 2;
                        double sumRe = 0.0, sumIm = 0.0;
                        int p = 0;
                        for (; p + 3 < k; p += 4) {
                            double a0r = A[aiOff + p * 2], a0i = A[aiOff + p * 2 + 1];
                            double b0r = A[ajOff + p * 2], b0i = A[ajOff + p * 2 + 1];
                            sumRe = Math.fma(a0r, b0r, Math.fma(a0i, b0i, sumRe));
                            sumIm = Math.fma(a0i, b0r, Math.fma(-a0r, b0i, sumIm));
                            double a1r = A[aiOff + (p + 1) * 2], a1i = A[aiOff + (p + 1) * 2 + 1];
                            double b1r = A[ajOff + (p + 1) * 2], b1i = A[ajOff + (p + 1) * 2 + 1];
                            sumRe = Math.fma(a1r, b1r, Math.fma(a1i, b1i, sumRe));
                            sumIm = Math.fma(a1i, b1r, Math.fma(-a1r, b1i, sumIm));
                            double a2r = A[aiOff + (p + 2) * 2], a2i = A[aiOff + (p + 2) * 2 + 1];
                            double b2r = A[ajOff + (p + 2) * 2], b2i = A[ajOff + (p + 2) * 2 + 1];
                            sumRe = Math.fma(a2r, b2r, Math.fma(a2i, b2i, sumRe));
                            sumIm = Math.fma(a2i, b2r, Math.fma(-a2r, b2i, sumIm));
                            double a3r = A[aiOff + (p + 3) * 2], a3i = A[aiOff + (p + 3) * 2 + 1];
                            double b3r = A[ajOff + (p + 3) * 2], b3i = A[ajOff + (p + 3) * 2 + 1];
                            sumRe = Math.fma(a3r, b3r, Math.fma(a3i, b3i, sumRe));
                            sumIm = Math.fma(a3i, b3r, Math.fma(-a3r, b3i, sumIm));
                        }
                        for (; p < k; p++) {
                            double ar = A[aiOff + p * 2], ai = A[aiOff + p * 2 + 1];
                            double br = A[ajOff + p * 2], bi = A[ajOff + p * 2 + 1];
                            sumRe = Math.fma(ar, br, Math.fma(ai, bi, sumRe));
                            sumIm = Math.fma(ai, br, Math.fma(-ar, bi, sumIm));
                        }
                        int cIdx = (cOff + i * ldc + j) * 2;
                        C[cIdx] += alpha * sumRe;
                        if (i != j) {
                            C[cIdx + 1] += alpha * sumIm;
                        }
                    }
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int aiOff = (aOff + i * lda + kStart) * 2;
                    for (int j = 0; j <= i; j++) {
                        int ajOff = (aOff + j * lda + kStart) * 2;
                        double sumRe = 0.0, sumIm = 0.0;
                        int p = 0;
                        for (; p + 3 < k; p += 4) {
                            double a0r = A[aiOff + p * 2], a0i = A[aiOff + p * 2 + 1];
                            double b0r = A[ajOff + p * 2], b0i = A[ajOff + p * 2 + 1];
                            sumRe = Math.fma(a0r, b0r, Math.fma(a0i, b0i, sumRe));
                            sumIm = Math.fma(a0i, b0r, Math.fma(-a0r, b0i, sumIm));
                            double a1r = A[aiOff + (p + 1) * 2], a1i = A[aiOff + (p + 1) * 2 + 1];
                            double b1r = A[ajOff + (p + 1) * 2], b1i = A[ajOff + (p + 1) * 2 + 1];
                            sumRe = Math.fma(a1r, b1r, Math.fma(a1i, b1i, sumRe));
                            sumIm = Math.fma(a1i, b1r, Math.fma(-a1r, b1i, sumIm));
                            double a2r = A[aiOff + (p + 2) * 2], a2i = A[aiOff + (p + 2) * 2 + 1];
                            double b2r = A[ajOff + (p + 2) * 2], b2i = A[ajOff + (p + 2) * 2 + 1];
                            sumRe = Math.fma(a2r, b2r, Math.fma(a2i, b2i, sumRe));
                            sumIm = Math.fma(a2i, b2r, Math.fma(-a2r, b2i, sumIm));
                            double a3r = A[aiOff + (p + 3) * 2], a3i = A[aiOff + (p + 3) * 2 + 1];
                            double b3r = A[ajOff + (p + 3) * 2], b3i = A[ajOff + (p + 3) * 2 + 1];
                            sumRe = Math.fma(a3r, b3r, Math.fma(a3i, b3i, sumRe));
                            sumIm = Math.fma(a3i, b3r, Math.fma(-a3r, b3i, sumIm));
                        }
                        for (; p < k; p++) {
                            double ar = A[aiOff + p * 2], ai = A[aiOff + p * 2 + 1];
                            double br = A[ajOff + p * 2], bi = A[ajOff + p * 2 + 1];
                            sumRe = Math.fma(ar, br, Math.fma(ai, bi, sumRe));
                            sumIm = Math.fma(ai, br, Math.fma(-ar, bi, sumIm));
                        }
                        int cIdx = (cOff + i * ldc + j) * 2;
                        C[cIdx] += alpha * sumRe;
                        if (i != j) {
                            C[cIdx + 1] += alpha * sumIm;
                        }
                    }
                }
            }
            return;
        }
        int mid = k / 2;
        herkNoTransImpl(n, mid, alpha, A, aOff, lda, C, cOff, ldc, upper, kStart);
        herkNoTransImpl(n, k - mid, alpha, A, aOff, lda, C, cOff, ldc, upper, kStart + mid);
    }

    static void herkConjTrans(int n, int k, double alpha,
                              double[] A, int aOff, int lda,
                              double[] C, int cOff, int ldc, boolean upper) {
        if (k <= 0) return;
        herkConjTransImpl(n, k, alpha, A, aOff, lda, C, cOff, ldc, upper);
    }

    static void herkConjTransImpl(int n, int k, double alpha,
                                  double[] A, int aOff, int lda,
                                  double[] C, int cOff, int ldc, boolean upper) {
        if (k <= THRESHOLD) {
            if (upper) {
                for (int p = 0; p < k; p++) {
                    int rowOff = (aOff + p * lda) * 2;
                    for (int i = 0; i < n; i++) {
                        double aRe = A[rowOff + i * 2];
                        double aIm = A[rowOff + i * 2 + 1];
                        double alphaConjRe = alpha * aRe;
                        double alphaConjIm = -alpha * aIm;
                        int cRowOff = (cOff + i * ldc) * 2;
                        for (int j = i; j < n; j++) {
                            double bRe = A[rowOff + j * 2];
                            double bIm = A[rowOff + j * 2 + 1];
                            C[cRowOff + j * 2] = Math.fma(alphaConjRe, bRe, Math.fma(-alphaConjIm, bIm, C[cRowOff + j * 2]));
                            if (i != j) {
                                C[cRowOff + j * 2 + 1] = Math.fma(alphaConjRe, bIm, Math.fma(alphaConjIm, bRe, C[cRowOff + j * 2 + 1]));
                            }
                        }
                    }
                }
            } else {
                for (int p = 0; p < k; p++) {
                    int rowOff = (aOff + p * lda) * 2;
                    for (int i = 0; i < n; i++) {
                        double aRe = A[rowOff + i * 2];
                        double aIm = A[rowOff + i * 2 + 1];
                        double alphaConjRe = alpha * aRe;
                        double alphaConjIm = -alpha * aIm;
                        int cRowOff = (cOff + i * ldc) * 2;
                        for (int j = 0; j <= i; j++) {
                            double bRe = A[rowOff + j * 2];
                            double bIm = A[rowOff + j * 2 + 1];
                            C[cRowOff + j * 2] = Math.fma(alphaConjRe, bRe, Math.fma(-alphaConjIm, bIm, C[cRowOff + j * 2]));
                            if (i != j) {
                                C[cRowOff + j * 2 + 1] = Math.fma(alphaConjRe, bIm, Math.fma(alphaConjIm, bRe, C[cRowOff + j * 2 + 1]));
                            }
                        }
                    }
                }
            }
            return;
        }
        int mid = k / 2;
        herkConjTransImpl(n, mid, alpha, A, aOff, lda, C, cOff, ldc, upper);
        herkConjTransImpl(n, k - mid, alpha, A, aOff + mid * lda, lda, C, cOff, ldc, upper);
    }
}
