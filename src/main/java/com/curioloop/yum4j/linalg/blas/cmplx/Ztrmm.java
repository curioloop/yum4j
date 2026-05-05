/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;

interface Ztrmm {

    int RIGHT_TRANS_PAIR_MIN_N = 32;

    static void ztrmm(BLAS.Side side, BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                      int m, int n,
                      double alphaRe, double alphaIm,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb) {
        if (m == 0 || n == 0) return;
        if (alphaRe == 0.0 && alphaIm == 0.0) {
            for (int i = 0; i < m; i++) {
                int bRow = bOff + i * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pos = bRow + j * 2;
                    B[pos] = 0.0;
                    B[pos + 1] = 0.0;
                }
            }
            return;
        }
        if (alphaRe != 1.0 || alphaIm != 0.0) {
            for (int i = 0; i < m; i++) {
                int bRow = bOff + i * ldb * 2;
                if (alphaIm == 0.0) {
                    Zdscal.zdscal(n, alphaRe, B, bRow, 1);
                } else {
                    Zscal.zscal(n, alphaRe, alphaIm, B, bRow, 1);
                }
            }
        }
        boolean unitDiag = diag == BLAS.Diag.Unit;
        if (side == BLAS.Side.Left) {
            if (uplo == BLAS.Uplo.Upper) {
                if (trans == BLAS.Trans.NoTrans) {
                    ztrmmLeftUpperNN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                } else if (trans == BLAS.Trans.Trans) {
                    ztrmmLeftUpperTN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                } else {
                    ztrmmLeftUpperCN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                }
            } else {
                if (trans == BLAS.Trans.NoTrans) {
                    ztrmmLeftLowerNN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                } else if (trans == BLAS.Trans.Trans) {
                    ztrmmLeftLowerTN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                } else {
                    ztrmmLeftLowerCN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                }
            }
        } else {
            if (uplo == BLAS.Uplo.Upper) {
                if (trans == BLAS.Trans.NoTrans) {
                    ztrmmRightUpperNN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                } else if (trans == BLAS.Trans.Trans) {
                    ztrmmRightUpperTN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                } else {
                    ztrmmRightUpperCN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                }
            } else {
                if (trans == BLAS.Trans.NoTrans) {
                    ztrmmRightLowerNN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                } else if (trans == BLAS.Trans.Trans) {
                    ztrmmRightLowerTN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                } else {
                    ztrmmRightLowerCN(m, n, unitDiag, A, aOff, lda, B, bOff, ldb);
                }
            }
        }
    }

    static void ztrmmLeftUpperNN(int m, int n, boolean unitDiag,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb) {
        for (int k = 0; k < m; k++) {
            int bRowK = bOff + k * ldb * 2;
            for (int i = 0; i < k; i++) {
                int aRowI = aOff + i * lda * 2;
                double aikRe = A[aRowI + k * 2], aikIm = A[aRowI + k * 2 + 1];
                int bRowI = bOff + i * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] += aikRe * bkRe - aikIm * bkIm;
                    B[pI + 1] += aikRe * bkIm + aikIm * bkRe;
                }
            }
            if (!unitDiag) {
                int aRowK = aOff + k * lda * 2;
                double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                for (int j = 0; j < n; j++) {
                    int pos = bRowK + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = akkRe * bjRe - akkIm * bjIm;
                    B[pos + 1] = akkRe * bjIm + akkIm * bjRe;
                }
            }
        }
    }

    static void ztrmmLeftLowerNN(int m, int n, boolean unitDiag,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb) {
        for (int k = m - 1; k >= 0; k--) {
            int bRowK = bOff + k * ldb * 2;
            for (int i = k + 1; i < m; i++) {
                int aRowI = aOff + i * lda * 2;
                double aikRe = A[aRowI + k * 2], aikIm = A[aRowI + k * 2 + 1];
                int bRowI = bOff + i * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] += aikRe * bkRe - aikIm * bkIm;
                    B[pI + 1] += aikRe * bkIm + aikIm * bkRe;
                }
            }
            if (!unitDiag) {
                int aRowK = aOff + k * lda * 2;
                double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                for (int j = 0; j < n; j++) {
                    int pos = bRowK + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = akkRe * bjRe - akkIm * bjIm;
                    B[pos + 1] = akkRe * bjIm + akkIm * bjRe;
                }
            }
        }
    }

    static void ztrmmLeftUpperTN(int m, int n, boolean unitDiag,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb) {
        for (int k = m - 1; k >= 0; k--) {
            int aRowK = aOff + k * lda * 2;
            int bRowK = bOff + k * ldb * 2;
            for (int i = k + 1; i < m; i++) {
                double akiRe = A[aRowK + i * 2], akiIm = A[aRowK + i * 2 + 1];
                int bRowI = bOff + i * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] += akiRe * bkRe - akiIm * bkIm;
                    B[pI + 1] += akiRe * bkIm + akiIm * bkRe;
                }
            }
            if (!unitDiag) {
                double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                for (int j = 0; j < n; j++) {
                    int pos = bRowK + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = akkRe * bjRe - akkIm * bjIm;
                    B[pos + 1] = akkRe * bjIm + akkIm * bjRe;
                }
            }
        }
    }

    static void ztrmmLeftLowerTN(int m, int n, boolean unitDiag,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb) {
        for (int k = 0; k < m; k++) {
            int aRowK = aOff + k * lda * 2;
            int bRowK = bOff + k * ldb * 2;
            for (int i = 0; i < k; i++) {
                double akiRe = A[aRowK + i * 2], akiIm = A[aRowK + i * 2 + 1];
                int bRowI = bOff + i * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] += akiRe * bkRe - akiIm * bkIm;
                    B[pI + 1] += akiRe * bkIm + akiIm * bkRe;
                }
            }
            if (!unitDiag) {
                double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                for (int j = 0; j < n; j++) {
                    int pos = bRowK + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = akkRe * bjRe - akkIm * bjIm;
                    B[pos + 1] = akkRe * bjIm + akkIm * bjRe;
                }
            }
        }
    }

    static void ztrmmLeftUpperCN(int m, int n, boolean unitDiag,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb) {
        for (int k = m - 1; k >= 0; k--) {
            int aRowK = aOff + k * lda * 2;
            int bRowK = bOff + k * ldb * 2;
            for (int i = k + 1; i < m; i++) {
                double akiRe = A[aRowK + i * 2], akiIm = A[aRowK + i * 2 + 1];
                int bRowI = bOff + i * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] += akiRe * bkRe + akiIm * bkIm;
                    B[pI + 1] += akiRe * bkIm - akiIm * bkRe;
                }
            }
            if (!unitDiag) {
                double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                for (int j = 0; j < n; j++) {
                    int pos = bRowK + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = akkRe * bjRe + akkIm * bjIm;
                    B[pos + 1] = akkRe * bjIm - akkIm * bjRe;
                }
            }
        }
    }

    static void ztrmmLeftLowerCN(int m, int n, boolean unitDiag,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb) {
        for (int k = 0; k < m; k++) {
            int aRowK = aOff + k * lda * 2;
            int bRowK = bOff + k * ldb * 2;
            for (int i = 0; i < k; i++) {
                double akiRe = A[aRowK + i * 2], akiIm = A[aRowK + i * 2 + 1];
                int bRowI = bOff + i * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] += akiRe * bkRe + akiIm * bkIm;
                    B[pI + 1] += akiRe * bkIm - akiIm * bkRe;
                }
            }
            if (!unitDiag) {
                double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                for (int j = 0; j < n; j++) {
                    int pos = bRowK + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = akkRe * bjRe + akkIm * bjIm;
                    B[pos + 1] = akkRe * bjIm - akkIm * bjRe;
                }
            }
        }
    }

    static void ztrmmRightUpperNN(int m, int n, boolean unitDiag,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb) {
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int k = n - 1; k >= 0; k--) {
                int posK = bRow + k * 2;
                double bkRe = B[posK], bkIm = B[posK + 1];
                int aRowK = aOff + k * lda * 2;
                for (int j = k + 1; j < n; j++) {
                    int posJ = bRow + j * 2;
                    double akjRe = A[aRowK + j * 2], akjIm = A[aRowK + j * 2 + 1];
                    B[posJ] += bkRe * akjRe - bkIm * akjIm;
                    B[posJ + 1] += bkRe * akjIm + bkIm * akjRe;
                }
                if (!unitDiag) {
                    double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                    B[posK] = akkRe * bkRe - akkIm * bkIm;
                    B[posK + 1] = akkRe * bkIm + akkIm * bkRe;
                }
            }
        }
    }

    static void ztrmmRightLowerNN(int m, int n, boolean unitDiag,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb) {
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int k = 0; k < n; k++) {
                int posK = bRow + k * 2;
                double bkRe = B[posK], bkIm = B[posK + 1];
                int aRowK = aOff + k * lda * 2;
                for (int j = 0; j < k; j++) {
                    int posJ = bRow + j * 2;
                    double akjRe = A[aRowK + j * 2], akjIm = A[aRowK + j * 2 + 1];
                    B[posJ] += bkRe * akjRe - bkIm * akjIm;
                    B[posJ + 1] += bkRe * akjIm + bkIm * akjRe;
                }
                if (!unitDiag) {
                    double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                    B[posK] = akkRe * bkRe - akkIm * bkIm;
                    B[posK + 1] = akkRe * bkIm + akkIm * bkRe;
                }
            }
        }
    }

    static void ztrmmRightUpperTN(int m, int n, boolean unitDiag,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb) {
        if (m >= 2 && n >= RIGHT_TRANS_PAIR_MIN_N) {
            ztrmmRightUpperTPairRows(m, n, unitDiag, false, A, aOff, lda, B, bOff, ldb);
            return;
        }
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int k = 0; k < n; k++) {
                int posK = bRow + k * 2;
                double bkRe = B[posK], bkIm = B[posK + 1];
                for (int j = 0; j < k; j++) {
                    int posJ = bRow + j * 2;
                    int aRowJ = aOff + j * lda * 2;
                    double ajkRe = A[aRowJ + k * 2], ajkIm = A[aRowJ + k * 2 + 1];
                    B[posJ] += bkRe * ajkRe - bkIm * ajkIm;
                    B[posJ + 1] += bkRe * ajkIm + bkIm * ajkRe;
                }
                if (!unitDiag) {
                    int aRowK = aOff + k * lda * 2;
                    double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                    B[posK] = akkRe * bkRe - akkIm * bkIm;
                    B[posK + 1] = akkRe * bkIm + akkIm * bkRe;
                }
            }
        }
    }

    static void ztrmmRightLowerTN(int m, int n, boolean unitDiag,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb) {
        if (m >= 2 && n >= RIGHT_TRANS_PAIR_MIN_N) {
            ztrmmRightLowerTPairRows(m, n, unitDiag, false, A, aOff, lda, B, bOff, ldb);
            return;
        }
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int k = n - 1; k >= 0; k--) {
                int posK = bRow + k * 2;
                double bkRe = B[posK], bkIm = B[posK + 1];
                for (int j = k + 1; j < n; j++) {
                    int posJ = bRow + j * 2;
                    int aRowJ = aOff + j * lda * 2;
                    double ajkRe = A[aRowJ + k * 2], ajkIm = A[aRowJ + k * 2 + 1];
                    B[posJ] += bkRe * ajkRe - bkIm * ajkIm;
                    B[posJ + 1] += bkRe * ajkIm + bkIm * ajkRe;
                }
                if (!unitDiag) {
                    int aRowK = aOff + k * lda * 2;
                    double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                    B[posK] = akkRe * bkRe - akkIm * bkIm;
                    B[posK + 1] = akkRe * bkIm + akkIm * bkRe;
                }
            }
        }
    }

    static void ztrmmRightUpperCN(int m, int n, boolean unitDiag,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb) {
        if (m >= 2 && n >= RIGHT_TRANS_PAIR_MIN_N) {
            ztrmmRightUpperTPairRows(m, n, unitDiag, true, A, aOff, lda, B, bOff, ldb);
            return;
        }
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int k = 0; k < n; k++) {
                int posK = bRow + k * 2;
                double bkRe = B[posK], bkIm = B[posK + 1];
                for (int j = 0; j < k; j++) {
                    int posJ = bRow + j * 2;
                    int aRowJ = aOff + j * lda * 2;
                    double ajkRe = A[aRowJ + k * 2], ajkIm = A[aRowJ + k * 2 + 1];
                    B[posJ] += bkRe * ajkRe + bkIm * ajkIm;
                    B[posJ + 1] += bkIm * ajkRe - bkRe * ajkIm;
                }
                if (!unitDiag) {
                    int aRowK = aOff + k * lda * 2;
                    double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                    B[posK] = akkRe * bkRe + akkIm * bkIm;
                    B[posK + 1] = akkRe * bkIm - akkIm * bkRe;
                }
            }
        }
    }

    static void ztrmmRightLowerCN(int m, int n, boolean unitDiag,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb) {
        if (m >= 2 && n >= RIGHT_TRANS_PAIR_MIN_N) {
            ztrmmRightLowerTPairRows(m, n, unitDiag, true, A, aOff, lda, B, bOff, ldb);
            return;
        }
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int k = n - 1; k >= 0; k--) {
                int posK = bRow + k * 2;
                double bkRe = B[posK], bkIm = B[posK + 1];
                for (int j = k + 1; j < n; j++) {
                    int posJ = bRow + j * 2;
                    int aRowJ = aOff + j * lda * 2;
                    double ajkRe = A[aRowJ + k * 2], ajkIm = A[aRowJ + k * 2 + 1];
                    B[posJ] += bkRe * ajkRe + bkIm * ajkIm;
                    B[posJ + 1] += bkIm * ajkRe - bkRe * ajkIm;
                }
                if (!unitDiag) {
                    int aRowK = aOff + k * lda * 2;
                    double akkRe = A[aRowK + k * 2], akkIm = A[aRowK + k * 2 + 1];
                    B[posK] = akkRe * bkRe + akkIm * bkIm;
                    B[posK + 1] = akkRe * bkIm - akkIm * bkRe;
                }
            }
        }
    }

    static void ztrmmRightUpperTPairRows(int m, int n, boolean unitDiag, boolean conjugate,
                                         double[] A, int aOff, int lda,
                                         double[] B, int bOff, int ldb) {
        int rowPairs = m & ~1;
        for (int j = 0; j < n; j++) {
            int diagPos = aOff + (j * lda + j) * 2;
            double diagRe = unitDiag ? 1.0 : A[diagPos];
            double diagIm = unitDiag ? 0.0 : A[diagPos + 1];
            if (conjugate) {
                diagIm = -diagIm;
            }

            for (int i = 0; i < rowPairs; i += 2) {
                int bRow0 = bOff + i * ldb * 2;
                int bRow1 = bOff + (i + 1) * ldb * 2;
                int bCol = j * 2;
                double b0jRe = B[bRow0 + bCol];
                double b0jIm = B[bRow0 + bCol + 1];
                double b1jRe = B[bRow1 + bCol];
                double b1jIm = B[bRow1 + bCol + 1];
                double sum0Re = diagRe * b0jRe - diagIm * b0jIm;
                double sum0Im = diagRe * b0jIm + diagIm * b0jRe;
                double sum1Re = diagRe * b1jRe - diagIm * b1jIm;
                double sum1Im = diagRe * b1jIm + diagIm * b1jRe;

                for (int k = j + 1; k < n; k++) {
                    int aPos = aOff + (j * lda + k) * 2;
                    double aRe = A[aPos];
                    double aIm = conjugate ? -A[aPos + 1] : A[aPos + 1];
                    int bK = k * 2;
                    double b0kRe = B[bRow0 + bK];
                    double b0kIm = B[bRow0 + bK + 1];
                    double b1kRe = B[bRow1 + bK];
                    double b1kIm = B[bRow1 + bK + 1];
                    sum0Re = Math.fma(b0kRe, aRe, Math.fma(-b0kIm, aIm, sum0Re));
                    sum0Im = Math.fma(b0kRe, aIm, Math.fma(b0kIm, aRe, sum0Im));
                    sum1Re = Math.fma(b1kRe, aRe, Math.fma(-b1kIm, aIm, sum1Re));
                    sum1Im = Math.fma(b1kRe, aIm, Math.fma(b1kIm, aRe, sum1Im));
                }

                B[bRow0 + bCol] = sum0Re;
                B[bRow0 + bCol + 1] = sum0Im;
                B[bRow1 + bCol] = sum1Re;
                B[bRow1 + bCol + 1] = sum1Im;
            }

            if ((m & 1) != 0) {
                int bRow = bOff + (m - 1) * ldb * 2;
                int bCol = j * 2;
                double bjRe = B[bRow + bCol];
                double bjIm = B[bRow + bCol + 1];
                double sumRe = diagRe * bjRe - diagIm * bjIm;
                double sumIm = diagRe * bjIm + diagIm * bjRe;
                for (int k = j + 1; k < n; k++) {
                    int aPos = aOff + (j * lda + k) * 2;
                    double aRe = A[aPos];
                    double aIm = conjugate ? -A[aPos + 1] : A[aPos + 1];
                    int bK = k * 2;
                    double bkRe = B[bRow + bK];
                    double bkIm = B[bRow + bK + 1];
                    sumRe = Math.fma(bkRe, aRe, Math.fma(-bkIm, aIm, sumRe));
                    sumIm = Math.fma(bkRe, aIm, Math.fma(bkIm, aRe, sumIm));
                }
                B[bRow + bCol] = sumRe;
                B[bRow + bCol + 1] = sumIm;
            }
        }
    }

    static void ztrmmRightLowerTPairRows(int m, int n, boolean unitDiag, boolean conjugate,
                                         double[] A, int aOff, int lda,
                                         double[] B, int bOff, int ldb) {
        int rowPairs = m & ~1;
        for (int j = n - 1; j >= 0; j--) {
            int diagPos = aOff + (j * lda + j) * 2;
            double diagRe = unitDiag ? 1.0 : A[diagPos];
            double diagIm = unitDiag ? 0.0 : A[diagPos + 1];
            if (conjugate) {
                diagIm = -diagIm;
            }

            for (int i = 0; i < rowPairs; i += 2) {
                int bRow0 = bOff + i * ldb * 2;
                int bRow1 = bOff + (i + 1) * ldb * 2;
                int bCol = j * 2;
                double b0jRe = B[bRow0 + bCol];
                double b0jIm = B[bRow0 + bCol + 1];
                double b1jRe = B[bRow1 + bCol];
                double b1jIm = B[bRow1 + bCol + 1];
                double sum0Re = diagRe * b0jRe - diagIm * b0jIm;
                double sum0Im = diagRe * b0jIm + diagIm * b0jRe;
                double sum1Re = diagRe * b1jRe - diagIm * b1jIm;
                double sum1Im = diagRe * b1jIm + diagIm * b1jRe;

                for (int k = 0; k < j; k++) {
                    int aPos = aOff + (j * lda + k) * 2;
                    double aRe = A[aPos];
                    double aIm = conjugate ? -A[aPos + 1] : A[aPos + 1];
                    int bK = k * 2;
                    double b0kRe = B[bRow0 + bK];
                    double b0kIm = B[bRow0 + bK + 1];
                    double b1kRe = B[bRow1 + bK];
                    double b1kIm = B[bRow1 + bK + 1];
                    sum0Re = Math.fma(b0kRe, aRe, Math.fma(-b0kIm, aIm, sum0Re));
                    sum0Im = Math.fma(b0kRe, aIm, Math.fma(b0kIm, aRe, sum0Im));
                    sum1Re = Math.fma(b1kRe, aRe, Math.fma(-b1kIm, aIm, sum1Re));
                    sum1Im = Math.fma(b1kRe, aIm, Math.fma(b1kIm, aRe, sum1Im));
                }

                B[bRow0 + bCol] = sum0Re;
                B[bRow0 + bCol + 1] = sum0Im;
                B[bRow1 + bCol] = sum1Re;
                B[bRow1 + bCol + 1] = sum1Im;
            }

            if ((m & 1) != 0) {
                int bRow = bOff + (m - 1) * ldb * 2;
                int bCol = j * 2;
                double bjRe = B[bRow + bCol];
                double bjIm = B[bRow + bCol + 1];
                double sumRe = diagRe * bjRe - diagIm * bjIm;
                double sumIm = diagRe * bjIm + diagIm * bjRe;
                for (int k = 0; k < j; k++) {
                    int aPos = aOff + (j * lda + k) * 2;
                    double aRe = A[aPos];
                    double aIm = conjugate ? -A[aPos + 1] : A[aPos + 1];
                    int bK = k * 2;
                    double bkRe = B[bRow + bK];
                    double bkIm = B[bRow + bK + 1];
                    sumRe = Math.fma(bkRe, aRe, Math.fma(-bkIm, aIm, sumRe));
                    sumIm = Math.fma(bkRe, aIm, Math.fma(bkIm, aRe, sumIm));
                }
                B[bRow + bCol] = sumRe;
                B[bRow + bCol + 1] = sumIm;
            }
        }
    }
}
