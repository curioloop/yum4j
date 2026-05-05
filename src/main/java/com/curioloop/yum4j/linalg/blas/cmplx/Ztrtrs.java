package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
/**
 * LAPACK ZTRTRS: Solves a triangular system of the form A * X = B or A^T * X = B or A^H * X = B.
 * 
 * Based on LAPACK.
 */
interface Ztrtrs {

    /**
     * ZTRTRS solves a triangular system of the form
     * A * X = B  or  A^T * X = B  or  A^H * X = B,
     * where A is a triangular matrix of order n, and B is an n-by-nrhs matrix.
     * A check is made to verify that A is nonsingular.
     *
     * @param uplo  'U' for upper triangular, 'L' for lower triangular
     * @param trans 'N' for no transpose, 'T' for transpose, 'C' for conjugate transpose
     * @param diag  'U' for unit triangular, 'N' for non-unit triangular
     * @param n     order of matrix A
     * @param nrhs  number of right-hand sides
     * @param A     triangular matrix (n x n, row-major, interleaved format)
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param B     right-hand side matrix (n x nrhs, row-major, interleaved format), overwritten with solution
     * @param bOff  offset into B
    * @param ldb   row stride of B in row-major storage (must be at least max(1, nrhs))
     * @return 0 on success, non-zero on failure (return value is the first index where A is singular)
     */
    static int ztrtrs(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, 
                      int n, int nrhs, double[] A, int aOff, int lda, 
                      double[] B, int bOff, int ldb) {
        // Check input parameters
        if (uplo == null) return -1;
        if (trans == null) return -2;
        if (diag == null) return -3;
        if (n < 0) return -4;
        if (nrhs < 0) return -5;
        if (A == null) return -6;
        if (aOff < 0) return -7;
        if (lda < Math.max(1, n)) return -8;
        if (B == null) return -9;
        if (bOff < 0) return -10;
        // Unlike LAPACK's column-major interface, the Java wrapper stores B row-major.
        if (ldb < Math.max(1, nrhs)) return -11;

        if (n == 0 || nrhs == 0) {
            return 0;
        }

        boolean upper = (uplo == BLAS.Uplo.Upper);
        boolean noTrans = (trans == BLAS.Trans.NoTrans);
        boolean conjTrans = (trans == BLAS.Trans.Conj);
        boolean unitDiag = (diag == BLAS.Diag.Unit);

        int info = 0;

        // Check for singularity
        if (!unitDiag) {
            for (int j = 0; j < n; j++) {
                double ajjRe = A[aOff + j * lda * 2 + j * 2];
                double ajjIm = A[aOff + j * lda * 2 + j * 2 + 1];
                if (ajjRe == 0.0 && ajjIm == 0.0) {
                    info = j + 1;
                    return info;
                }
            }
        }

        // Solve the system
        for (int k = 0; k < nrhs; k++) {
            if (noTrans) {
                if (upper) {
                    // Upper triangular, no transpose
                    for (int j = n - 1; j >= 0; j--) {
                        // First divide by diagonal element
                        if (!unitDiag) {
                            int aPos = aOff + (j * lda + j) * 2;
                            int bPos = bOff + (j * ldb + k) * 2;
                            double ajjRe = A[aPos];
                            double ajjIm = A[aPos + 1];
                            double bjRe = B[bPos];
                            double bjIm = B[bPos + 1];
                            // Compute B(j) /= A(j,j)
                            double denom = ajjRe * ajjRe + ajjIm * ajjIm;
                            double tempRe = (bjRe * ajjRe + bjIm * ajjIm) / denom;
                            double tempIm = (bjIm * ajjRe - bjRe * ajjIm) / denom;
                            B[bPos] = tempRe;
                            B[bPos + 1] = tempIm;
                        }
                        // Then subtract contributions from columns to the right
                        for (int i = 0; i < j; i++) {
                            int aPos = aOff + (i * lda + j) * 2;
                            int bPosI = bOff + (i * ldb + k) * 2;
                            int bPosJ = bOff + (j * ldb + k) * 2;
                            double aijRe = A[aPos];
                            double aijIm = A[aPos + 1];
                            double biRe = B[bPosI];
                            double biIm = B[bPosI + 1];
                            double bjRe = B[bPosJ];
                            double bjIm = B[bPosJ + 1];
                            // Compute B(i) -= A(i,j) * B(j)
                            double tempRe = aijRe * bjRe - aijIm * bjIm;
                            double tempIm = aijRe * bjIm + aijIm * bjRe;
                            B[bPosI] = biRe - tempRe;
                            B[bPosI + 1] = biIm - tempIm;
                        }
                    }
                } else {
                    // Lower triangular, no transpose
                    for (int j = 0; j < n; j++) {
                        // First divide by diagonal element
                        if (!unitDiag) {
                            int aPos = aOff + (j * lda + j) * 2;
                            int bPos = bOff + (j * ldb + k) * 2;
                            double ajjRe = A[aPos];
                            double ajjIm = A[aPos + 1];
                            double bjRe = B[bPos];
                            double bjIm = B[bPos + 1];
                            // Compute B(j) /= A(j,j)
                            double denom = ajjRe * ajjRe + ajjIm * ajjIm;
                            double tempRe = (bjRe * ajjRe + bjIm * ajjIm) / denom;
                            double tempIm = (bjIm * ajjRe - bjRe * ajjIm) / denom;
                            B[bPos] = tempRe;
                            B[bPos + 1] = tempIm;
                        }
                        // Then subtract contributions from rows above
                        for (int i = j + 1; i < n; i++) {
                            int aPos = aOff + (i * lda + j) * 2;
                            int bPosI = bOff + (i * ldb + k) * 2;
                            int bPosJ = bOff + (j * ldb + k) * 2;
                            double aijRe = A[aPos];
                            double aijIm = A[aPos + 1];
                            double biRe = B[bPosI];
                            double biIm = B[bPosI + 1];
                            double bjRe = B[bPosJ];
                            double bjIm = B[bPosJ + 1];
                            // Compute B(i) -= A(i,j) * B(j)
                            double tempRe = aijRe * bjRe - aijIm * bjIm;
                            double tempIm = aijRe * bjIm + aijIm * bjRe;
                            B[bPosI] = biRe - tempRe;
                            B[bPosI + 1] = biIm - tempIm;
                        }
                    }
                }
            } else {
                if (upper) {
                    // Upper triangular, transpose or conjugate transpose
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < j; i++) {
                            int aPos = aOff + (i * lda + j) * 2;
                            int bPosI = bOff + (i * ldb + k) * 2;
                            int bPosJ = bOff + (j * ldb + k) * 2;
                            double aijRe = A[aPos];
                            double aijIm = A[aPos + 1];
                            double biRe = B[bPosI];
                            double biIm = B[bPosI + 1];
                            double bjRe = B[bPosJ];
                            double bjIm = B[bPosJ + 1];
                            // Compute B(j) -= conj(A(i,j)) * B(i) if conjugate transpose
                            // or A(i,j) * B(i) if transpose
                            double tempRe, tempIm;
                            if (conjTrans) {
                                tempRe = aijRe * biRe + aijIm * biIm;
                                tempIm = aijRe * biIm - aijIm * biRe;
                            } else {
                                tempRe = aijRe * biRe - aijIm * biIm;
                                tempIm = aijRe * biIm + aijIm * biRe;
                            }
                            B[bPosJ] = bjRe - tempRe;
                            B[bPosJ + 1] = bjIm - tempIm;
                        }
                        if (!unitDiag) {
                            int aPos = aOff + (j * lda + j) * 2;
                            int bPos = bOff + (j * ldb + k) * 2;
                            double ajjRe = A[aPos];
                            double ajjIm = A[aPos + 1];
                            double bjRe = B[bPos];
                            double bjIm = B[bPos + 1];
                            // Compute B(j) /= A(j,j)
                            double denom = ajjRe * ajjRe + ajjIm * ajjIm;
                            double tempRe = (bjRe * ajjRe + bjIm * ajjIm) / denom;
                            double tempIm = (bjIm * ajjRe - bjRe * ajjIm) / denom;
                            B[bPos] = tempRe;
                            B[bPos + 1] = tempIm;
                        }
                    }
                } else {
                    // Lower triangular, transpose or conjugate transpose
                    for (int j = n - 1; j >= 0; j--) {
                        for (int i = j + 1; i < n; i++) {
                            int aPos = aOff + (i * lda + j) * 2;
                            int bPosI = bOff + (i * ldb + k) * 2;
                            int bPosJ = bOff + (j * ldb + k) * 2;
                            double aijRe = A[aPos];
                            double aijIm = A[aPos + 1];
                            double biRe = B[bPosI];
                            double biIm = B[bPosI + 1];
                            double bjRe = B[bPosJ];
                            double bjIm = B[bPosJ + 1];
                            // Compute B(j) -= conj(A(i,j)) * B(i) if conjugate transpose
                            // or A(i,j) * B(i) if transpose
                            double tempRe, tempIm;
                            if (conjTrans) {
                                tempRe = aijRe * biRe + aijIm * biIm;
                                tempIm = aijRe * biIm - aijIm * biRe;
                            } else {
                                tempRe = aijRe * biRe - aijIm * biIm;
                                tempIm = aijRe * biIm + aijIm * biRe;
                            }
                            B[bPosJ] = bjRe - tempRe;
                            B[bPosJ + 1] = bjIm - tempIm;
                        }
                        if (!unitDiag) {
                            int aPos = aOff + (j * lda + j) * 2;
                            int bPos = bOff + (j * ldb + k) * 2;
                            double ajjRe = A[aPos];
                            double ajjIm = A[aPos + 1];
                            double bjRe = B[bPos];
                            double bjIm = B[bPos + 1];
                            // Compute B(j) /= A(j,j)
                            double denom = ajjRe * ajjRe + ajjIm * ajjIm;
                            double tempRe = (bjRe * ajjRe + bjIm * ajjIm) / denom;
                            double tempIm = (bjIm * ajjRe - bjRe * ajjIm) / denom;
                            B[bPos] = tempRe;
                            B[bPos + 1] = tempIm;
                        }
                    }
                }
            }
        }

        return info;
    }
}
