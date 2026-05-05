/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
interface Ztrsm {

    static void ztrsm(BLAS.Side side, BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                      int m, int n, double alphaRe, double alphaIm,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb) {

        boolean leftSide = side == BLAS.Side.Left;
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean noTrans = trans == BLAS.Trans.NoTrans;
        boolean conjTrans = trans == BLAS.Trans.Conj;
        boolean unitDiag = diag == BLAS.Diag.Unit;

        if (m <= 0 || n <= 0) return;

        if (alphaRe == 0.0 && alphaIm == 0.0) {
            for (int i = 0; i < m; i++) {
                int bRow = bOff + i * ldb * 2;
                for (int j = 0; j < n; j++) {
                    B[bRow + j * 2] = 0.0;
                    B[bRow + j * 2 + 1] = 0.0;
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

        if (leftSide) {
            if (upper) {
                if (noTrans) ztrsmLeftUpperNN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
                else if (conjTrans) ztrsmLeftUpperCN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
                else ztrsmLeftUpperTN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else {
                if (noTrans) ztrsmLeftLowerNN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
                else if (conjTrans) ztrsmLeftLowerCN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
                else ztrsmLeftLowerTN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            }
        } else {
            if (upper) {
                if (noTrans) ztrsmRightUpperNN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
                else if (conjTrans) ztrsmRightUpperCN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
                else ztrsmRightUpperTN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else {
                if (noTrans) ztrsmRightLowerNN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
                else if (conjTrans) ztrsmRightLowerCN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
                else ztrsmRightLowerTN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            }
        }
    }

    static void ztrsmLeftUpperNN(int m, int n,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        for (int i = m - 1; i >= 0; i--) {
            int aRow = aOff + i * lda * 2;
            int bRowI = bOff + i * ldb * 2;
            for (int k = i + 1; k < m; k++) {
                double aikRe = A[aRow + k * 2], aikIm = A[aRow + k * 2 + 1];
                int bRowK = bOff + k * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] -= aikRe * bkRe - aikIm * bkIm;
                    B[pI + 1] -= aikRe * bkIm + aikIm * bkRe;
                }
            }
            if (!unitDiag) {
                double aiiRe = A[aRow + i * 2], aiiIm = A[aRow + i * 2 + 1];
                double denom = aiiRe * aiiRe + aiiIm * aiiIm;
                for (int j = 0; j < n; j++) {
                    int pos = bRowI + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = (bjRe * aiiRe + bjIm * aiiIm) / denom;
                    B[pos + 1] = (bjIm * aiiRe - bjRe * aiiIm) / denom;
                }
            }
        }
    }

    static void ztrsmLeftLowerNN(int m, int n,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        for (int i = 0; i < m; i++) {
            int aRow = aOff + i * lda * 2;
            int bRowI = bOff + i * ldb * 2;
            for (int k = 0; k < i; k++) {
                double aikRe = A[aRow + k * 2], aikIm = A[aRow + k * 2 + 1];
                int bRowK = bOff + k * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] -= aikRe * bkRe - aikIm * bkIm;
                    B[pI + 1] -= aikRe * bkIm + aikIm * bkRe;
                }
            }
            if (!unitDiag) {
                double aiiRe = A[aRow + i * 2], aiiIm = A[aRow + i * 2 + 1];
                double denom = aiiRe * aiiRe + aiiIm * aiiIm;
                for (int j = 0; j < n; j++) {
                    int pos = bRowI + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = (bjRe * aiiRe + bjIm * aiiIm) / denom;
                    B[pos + 1] = (bjIm * aiiRe - bjRe * aiiIm) / denom;
                }
            }
        }
    }

    static void ztrsmLeftUpperTN(int m, int n,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        for (int i = 0; i < m; i++) {
            int bRowI = bOff + i * ldb * 2;
            for (int k = 0; k < i; k++) {
                double akiRe = A[aOff + k * lda * 2 + i * 2];
                double akiIm = A[aOff + k * lda * 2 + i * 2 + 1];
                int bRowK = bOff + k * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] -= akiRe * bkRe - akiIm * bkIm;
                    B[pI + 1] -= akiRe * bkIm + akiIm * bkRe;
                }
            }
            if (!unitDiag) {
                double aiiRe = A[aOff + i * lda * 2 + i * 2];
                double aiiIm = A[aOff + i * lda * 2 + i * 2 + 1];
                double denom = aiiRe * aiiRe + aiiIm * aiiIm;
                for (int j = 0; j < n; j++) {
                    int pos = bRowI + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = (bjRe * aiiRe + bjIm * aiiIm) / denom;
                    B[pos + 1] = (bjIm * aiiRe - bjRe * aiiIm) / denom;
                }
            }
        }
    }

    static void ztrsmLeftLowerTN(int m, int n,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        for (int i = m - 1; i >= 0; i--) {
            int bRowI = bOff + i * ldb * 2;
            for (int k = i + 1; k < m; k++) {
                double akiRe = A[aOff + k * lda * 2 + i * 2];
                double akiIm = A[aOff + k * lda * 2 + i * 2 + 1];
                int bRowK = bOff + k * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] -= akiRe * bkRe - akiIm * bkIm;
                    B[pI + 1] -= akiRe * bkIm + akiIm * bkRe;
                }
            }
            if (!unitDiag) {
                double aiiRe = A[aOff + i * lda * 2 + i * 2];
                double aiiIm = A[aOff + i * lda * 2 + i * 2 + 1];
                double denom = aiiRe * aiiRe + aiiIm * aiiIm;
                for (int j = 0; j < n; j++) {
                    int pos = bRowI + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = (bjRe * aiiRe + bjIm * aiiIm) / denom;
                    B[pos + 1] = (bjIm * aiiRe - bjRe * aiiIm) / denom;
                }
            }
        }
    }

    static void ztrsmLeftUpperCN(int m, int n,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        for (int i = 0; i < m; i++) {
            int bRowI = bOff + i * ldb * 2;
            for (int k = 0; k < i; k++) {
                double akiRe = A[aOff + k * lda * 2 + i * 2];
                double akiIm = A[aOff + k * lda * 2 + i * 2 + 1];
                int bRowK = bOff + k * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] -= akiRe * bkRe + akiIm * bkIm;
                    B[pI + 1] -= akiRe * bkIm - akiIm * bkRe;
                }
            }
            if (!unitDiag) {
                double aiiRe = A[aOff + i * lda * 2 + i * 2];
                double aiiIm = A[aOff + i * lda * 2 + i * 2 + 1];
                double denom = aiiRe * aiiRe + aiiIm * aiiIm;
                for (int j = 0; j < n; j++) {
                    int pos = bRowI + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = (bjRe * aiiRe - bjIm * aiiIm) / denom;
                    B[pos + 1] = (bjIm * aiiRe + bjRe * aiiIm) / denom;
                }
            }
        }
    }

    static void ztrsmLeftLowerCN(int m, int n,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        for (int i = m - 1; i >= 0; i--) {
            int bRowI = bOff + i * ldb * 2;
            for (int k = i + 1; k < m; k++) {
                double akiRe = A[aOff + k * lda * 2 + i * 2];
                double akiIm = A[aOff + k * lda * 2 + i * 2 + 1];
                int bRowK = bOff + k * ldb * 2;
                for (int j = 0; j < n; j++) {
                    int pI = bRowI + j * 2, pK = bRowK + j * 2;
                    double bkRe = B[pK], bkIm = B[pK + 1];
                    B[pI] -= akiRe * bkRe + akiIm * bkIm;
                    B[pI + 1] -= akiRe * bkIm - akiIm * bkRe;
                }
            }
            if (!unitDiag) {
                double aiiRe = A[aOff + i * lda * 2 + i * 2];
                double aiiIm = A[aOff + i * lda * 2 + i * 2 + 1];
                double denom = aiiRe * aiiRe + aiiIm * aiiIm;
                for (int j = 0; j < n; j++) {
                    int pos = bRowI + j * 2;
                    double bjRe = B[pos], bjIm = B[pos + 1];
                    B[pos] = (bjRe * aiiRe - bjIm * aiiIm) / denom;
                    B[pos + 1] = (bjIm * aiiRe + bjRe * aiiIm) / denom;
                }
            }
        }
    }

    static void ztrsmRightUpperNN(int m, int n,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        int i = 0;
        for (; i + 1 < m; i += 2) {
            int bRow0 = bOff + i * ldb * 2;
            int bRow1 = bRow0 + ldb * 2;
            for (int k = 0; k < n; k++) {
                int pos0k = bRow0 + k * 2;
                int pos1k = bRow1 + k * 2;
                double bk0Re = B[pos0k], bk0Im = B[pos0k + 1];
                double bk1Re = B[pos1k], bk1Im = B[pos1k + 1];
                if (!unitDiag) {
                    int aKK = aOff + k * lda * 2 + k * 2;
                    double akkRe = A[aKK], akkIm = A[aKK + 1];
                    double denom = akkRe * akkRe + akkIm * akkIm;
                    double next0Re = (bk0Re * akkRe + bk0Im * akkIm) / denom;
                    double next0Im = (bk0Im * akkRe - bk0Re * akkIm) / denom;
                    double next1Re = (bk1Re * akkRe + bk1Im * akkIm) / denom;
                    double next1Im = (bk1Im * akkRe - bk1Re * akkIm) / denom;
                    B[pos0k] = next0Re;
                    B[pos0k + 1] = next0Im;
                    B[pos1k] = next1Re;
                    B[pos1k + 1] = next1Im;
                    bk0Re = next0Re;
                    bk0Im = next0Im;
                    bk1Re = next1Re;
                    bk1Im = next1Im;
                }

                int aRowK = aOff + k * lda * 2;
                for (int j = k + 1; j < n; j++) {
                    int pos0j = bRow0 + j * 2;
                    int pos1j = bRow1 + j * 2;
                    double akjRe = A[aRowK + j * 2], akjIm = A[aRowK + j * 2 + 1];
                    B[pos0j] -= bk0Re * akjRe - bk0Im * akjIm;
                    B[pos0j + 1] -= bk0Re * akjIm + bk0Im * akjRe;
                    B[pos1j] -= bk1Re * akjRe - bk1Im * akjIm;
                    B[pos1j + 1] -= bk1Re * akjIm + bk1Im * akjRe;
                }
            }
        }

        for (; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int k = 0; k < n; k++) {
                int posK = bRow + k * 2;
                double bkRe = B[posK], bkIm = B[posK + 1];
                if (!unitDiag) {
                    int aKK = aOff + k * lda * 2 + k * 2;
                    double akkRe = A[aKK], akkIm = A[aKK + 1];
                    double denom = akkRe * akkRe + akkIm * akkIm;
                    double newBkRe = (bkRe * akkRe + bkIm * akkIm) / denom;
                    double newBkIm = (bkIm * akkRe - bkRe * akkIm) / denom;
                    B[posK] = newBkRe;
                    B[posK + 1] = newBkIm;
                    bkRe = newBkRe;
                    bkIm = newBkIm;
                }
                int aRowK = aOff + k * lda * 2;
                for (int j = k + 1; j < n; j++) {
                    int posJ = bRow + j * 2;
                    double akjRe = A[aRowK + j * 2], akjIm = A[aRowK + j * 2 + 1];
                    B[posJ] -= bkRe * akjRe - bkIm * akjIm;
                    B[posJ + 1] -= bkRe * akjIm + bkIm * akjRe;
                }
            }
        }
    }

    static void ztrsmRightLowerNN(int m, int n,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        int i = 0;
        for (; i + 1 < m; i += 2) {
            int bRow0 = bOff + i * ldb * 2;
            int bRow1 = bRow0 + ldb * 2;
            for (int k = n - 1; k >= 0; k--) {
                int pos0k = bRow0 + k * 2;
                int pos1k = bRow1 + k * 2;
                double bk0Re = B[pos0k], bk0Im = B[pos0k + 1];
                double bk1Re = B[pos1k], bk1Im = B[pos1k + 1];
                if (!unitDiag) {
                    int aKK = aOff + k * lda * 2 + k * 2;
                    double akkRe = A[aKK], akkIm = A[aKK + 1];
                    double denom = akkRe * akkRe + akkIm * akkIm;
                    double next0Re = (bk0Re * akkRe + bk0Im * akkIm) / denom;
                    double next0Im = (bk0Im * akkRe - bk0Re * akkIm) / denom;
                    double next1Re = (bk1Re * akkRe + bk1Im * akkIm) / denom;
                    double next1Im = (bk1Im * akkRe - bk1Re * akkIm) / denom;
                    B[pos0k] = next0Re;
                    B[pos0k + 1] = next0Im;
                    B[pos1k] = next1Re;
                    B[pos1k + 1] = next1Im;
                    bk0Re = next0Re;
                    bk0Im = next0Im;
                    bk1Re = next1Re;
                    bk1Im = next1Im;
                }

                int aRowK = aOff + k * lda * 2;
                for (int j = 0; j < k; j++) {
                    int pos0j = bRow0 + j * 2;
                    int pos1j = bRow1 + j * 2;
                    double akjRe = A[aRowK + j * 2], akjIm = A[aRowK + j * 2 + 1];
                    B[pos0j] -= bk0Re * akjRe - bk0Im * akjIm;
                    B[pos0j + 1] -= bk0Re * akjIm + bk0Im * akjRe;
                    B[pos1j] -= bk1Re * akjRe - bk1Im * akjIm;
                    B[pos1j + 1] -= bk1Re * akjIm + bk1Im * akjRe;
                }
            }
        }

        for (; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int k = n - 1; k >= 0; k--) {
                int posK = bRow + k * 2;
                double bkRe = B[posK], bkIm = B[posK + 1];
                if (!unitDiag) {
                    int aKK = aOff + k * lda * 2 + k * 2;
                    double akkRe = A[aKK], akkIm = A[aKK + 1];
                    double denom = akkRe * akkRe + akkIm * akkIm;
                    double newBkRe = (bkRe * akkRe + bkIm * akkIm) / denom;
                    double newBkIm = (bkIm * akkRe - bkRe * akkIm) / denom;
                    B[posK] = newBkRe;
                    B[posK + 1] = newBkIm;
                    bkRe = newBkRe;
                    bkIm = newBkIm;
                }
                int aRowK = aOff + k * lda * 2;
                for (int j = 0; j < k; j++) {
                    int posJ = bRow + j * 2;
                    double akjRe = A[aRowK + j * 2], akjIm = A[aRowK + j * 2 + 1];
                    B[posJ] -= bkRe * akjRe - bkIm * akjIm;
                    B[posJ + 1] -= bkRe * akjIm + bkIm * akjRe;
                }
            }
        }
    }

    static void ztrsmRightUpperTN(int m, int n,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int j = n - 1; j >= 0; j--) {
                int aRowJ = aOff + j * lda * 2;
                double sumRe = B[bRow + j * 2], sumIm = B[bRow + j * 2 + 1];
                for (int k = j + 1; k < n; k++) {
                    double ajkRe = A[aRowJ + k * 2], ajkIm = A[aRowJ + k * 2 + 1];
                    double bkRe = B[bRow + k * 2], bkIm = B[bRow + k * 2 + 1];
                    sumRe -= ajkRe * bkRe - ajkIm * bkIm;
                    sumIm -= ajkRe * bkIm + ajkIm * bkRe;
                }
                if (!unitDiag) {
                    double ajjRe = A[aRowJ + j * 2], ajjIm = A[aRowJ + j * 2 + 1];
                    double denom = ajjRe * ajjRe + ajjIm * ajjIm;
                    double newRe = (sumRe * ajjRe + sumIm * ajjIm) / denom;
                    double newIm = (sumIm * ajjRe - sumRe * ajjIm) / denom;
                    sumRe = newRe;
                    sumIm = newIm;
                }
                B[bRow + j * 2] = sumRe;
                B[bRow + j * 2 + 1] = sumIm;
            }
        }
    }

    static void ztrsmRightLowerTN(int m, int n,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int j = 0; j < n; j++) {
                int aRowJ = aOff + j * lda * 2;
                double sumRe = B[bRow + j * 2], sumIm = B[bRow + j * 2 + 1];
                for (int k = 0; k < j; k++) {
                    double ajkRe = A[aRowJ + k * 2], ajkIm = A[aRowJ + k * 2 + 1];
                    double bkRe = B[bRow + k * 2], bkIm = B[bRow + k * 2 + 1];
                    sumRe -= ajkRe * bkRe - ajkIm * bkIm;
                    sumIm -= ajkRe * bkIm + ajkIm * bkRe;
                }
                if (!unitDiag) {
                    double ajjRe = A[aRowJ + j * 2], ajjIm = A[aRowJ + j * 2 + 1];
                    double denom = ajjRe * ajjRe + ajjIm * ajjIm;
                    double newRe = (sumRe * ajjRe + sumIm * ajjIm) / denom;
                    double newIm = (sumIm * ajjRe - sumRe * ajjIm) / denom;
                    sumRe = newRe;
                    sumIm = newIm;
                }
                B[bRow + j * 2] = sumRe;
                B[bRow + j * 2 + 1] = sumIm;
            }
        }
    }

    static void ztrsmRightUpperCN(int m, int n,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int j = n - 1; j >= 0; j--) {
                int aRowJ = aOff + j * lda * 2;
                double sumRe = B[bRow + j * 2], sumIm = B[bRow + j * 2 + 1];
                for (int k = j + 1; k < n; k++) {
                    double ajkRe = A[aRowJ + k * 2], ajkIm = A[aRowJ + k * 2 + 1];
                    double bkRe = B[bRow + k * 2], bkIm = B[bRow + k * 2 + 1];
                    sumRe -= ajkRe * bkRe + ajkIm * bkIm;
                    sumIm -= ajkRe * bkIm - ajkIm * bkRe;
                }
                if (!unitDiag) {
                    double ajjRe = A[aRowJ + j * 2], ajjIm = A[aRowJ + j * 2 + 1];
                    double denom = ajjRe * ajjRe + ajjIm * ajjIm;
                    double newRe = (sumRe * ajjRe - sumIm * ajjIm) / denom;
                    double newIm = (sumIm * ajjRe + sumRe * ajjIm) / denom;
                    sumRe = newRe;
                    sumIm = newIm;
                }
                B[bRow + j * 2] = sumRe;
                B[bRow + j * 2 + 1] = sumIm;
            }
        }
    }

    static void ztrsmRightLowerCN(int m, int n,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        int i = 0;
        for (; i + 1 < m; i += 2) {
            int bRow0 = bOff + i * ldb * 2;
            int bRow1 = bRow0 + ldb * 2;
            for (int j = 0; j < n; j++) {
                int pos0j = bRow0 + j * 2;
                int pos1j = bRow1 + j * 2;
                double sum0Re = B[pos0j], sum0Im = B[pos0j + 1];
                double sum1Re = B[pos1j], sum1Im = B[pos1j + 1];
                int aRowJ = aOff + j * lda * 2;
                for (int k = 0; k < j; k++) {
                    double ajkRe = A[aRowJ + k * 2], ajkIm = A[aRowJ + k * 2 + 1];
                    int pos0k = bRow0 + k * 2;
                    int pos1k = bRow1 + k * 2;
                    double b0Re = B[pos0k], b0Im = B[pos0k + 1];
                    double b1Re = B[pos1k], b1Im = B[pos1k + 1];
                    sum0Re -= ajkRe * b0Re + ajkIm * b0Im;
                    sum0Im -= ajkRe * b0Im - ajkIm * b0Re;
                    sum1Re -= ajkRe * b1Re + ajkIm * b1Im;
                    sum1Im -= ajkRe * b1Im - ajkIm * b1Re;
                }
                if (!unitDiag) {
                    double ajjRe = A[aRowJ + j * 2], ajjIm = A[aRowJ + j * 2 + 1];
                    double denom = ajjRe * ajjRe + ajjIm * ajjIm;
                    double next0Re = (sum0Re * ajjRe - sum0Im * ajjIm) / denom;
                    double next0Im = (sum0Im * ajjRe + sum0Re * ajjIm) / denom;
                    double next1Re = (sum1Re * ajjRe - sum1Im * ajjIm) / denom;
                    double next1Im = (sum1Im * ajjRe + sum1Re * ajjIm) / denom;
                    sum0Re = next0Re;
                    sum0Im = next0Im;
                    sum1Re = next1Re;
                    sum1Im = next1Im;
                }
                B[pos0j] = sum0Re;
                B[pos0j + 1] = sum0Im;
                B[pos1j] = sum1Re;
                B[pos1j + 1] = sum1Im;
            }
        }

        for (; i < m; i++) {
            int bRow = bOff + i * ldb * 2;
            for (int j = 0; j < n; j++) {
                int aRowJ = aOff + j * lda * 2;
                double sumRe = B[bRow + j * 2], sumIm = B[bRow + j * 2 + 1];
                for (int k = 0; k < j; k++) {
                    double ajkRe = A[aRowJ + k * 2], ajkIm = A[aRowJ + k * 2 + 1];
                    double bkRe = B[bRow + k * 2], bkIm = B[bRow + k * 2 + 1];
                    sumRe -= ajkRe * bkRe + ajkIm * bkIm;
                    sumIm -= ajkRe * bkIm - ajkIm * bkRe;
                }
                if (!unitDiag) {
                    double ajjRe = A[aRowJ + j * 2], ajjIm = A[aRowJ + j * 2 + 1];
                    double denom = ajjRe * ajjRe + ajjIm * ajjIm;
                    double newRe = (sumRe * ajjRe - sumIm * ajjIm) / denom;
                    double newIm = (sumIm * ajjRe + sumRe * ajjIm) / denom;
                    sumRe = newRe;
                    sumIm = newIm;
                }
                B[bRow + j * 2] = sumRe;
                B[bRow + j * 2 + 1] = sumIm;
            }
        }
    }
}
