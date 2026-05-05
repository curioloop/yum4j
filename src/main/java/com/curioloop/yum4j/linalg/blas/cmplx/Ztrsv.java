package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
/**
 * ZTRSV performs one of the matrix-vector operations
 * x := A^{-1}*x  or  x := A^{-T}*x  or  x := A^{-H}*x
 * where A is an n by n unit, or non-unit, upper or lower triangular matrix.
 */
interface Ztrsv {

    /**
     * Performs one of the matrix-vector operations
     * x := A^{-1}*x  or  x := A^{-T}*x  or  x := A^{-H}*x
     *
     * @param uplo   Specifies whether the matrix A is upper or lower triangular:
     *               BLAS.Uplo.Upper: A is upper triangular
     *               BLAS.Uplo.Lower: A is lower triangular
     * @param trans  Specifies the form of op(A) to be used in the matrix-vector product:
     *               BLAS.Trans.NoTrans: op(A) = A
     *               BLAS.Trans.Trans: op(A) = A^T
     *               BLAS.Trans.Conj: op(A) = A^H (conjugate transpose)
     * @param diag   Specifies whether or not A is unit triangular:
     *               BLAS.Diag.Unit: A is unit triangular
     *               BLAS.Diag.NonUnit: A is non-unit triangular
     * @param n      Order of the matrix A
     * @param A      Complex matrix A stored in row-major order, interleaved format [re, im, re, im, ...]
     * @param aStart Starting index of A
     * @param lda    Leading dimension of the matrix A (must be at least max(1, n))
     * @param x      Complex vector x stored in interleaved format [re, im, re, im, ...]
     * @param xStart Starting index of x
     * @param incX   Increment for the elements of x
     */
    public static void ztrsv(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, 
                            int n, double[] A, int aStart, int lda, 
                            double[] x, int xStart, int incX) {
        if (n <= 0) return;
        
        boolean upper = (uplo == BLAS.Uplo.Upper);
        boolean noTrans = (trans == BLAS.Trans.NoTrans);
        boolean conjTrans = (trans == BLAS.Trans.Conj);
        boolean unitDiag = (diag == BLAS.Diag.Unit);
        
        if (noTrans) {
            // x := A^{-1}*x
            if (upper) {
                for (int i = n - 1; i >= 0; i--) {
                    int xPos = xStart + i * incX * 2;
                    double xiRe = x[xPos];
                    double xiIm = x[xPos + 1];
                    
                    for (int j = i + 1; j < n; j++) {
                        int aPos = aStart + i * lda * 2 + j * 2;
                        int xjPos = xStart + j * incX * 2;
                        double aRe = A[aPos];
                        double aIm = A[aPos + 1];
                        double xjRe = x[xjPos];
                        double xjIm = x[xjPos + 1];
                        
                        xiRe -= aRe * xjRe - aIm * xjIm;
                        xiIm -= aRe * xjIm + aIm * xjRe;
                    }
                    
                    if (!unitDiag) {
                        int aPos = aStart + i * lda * 2 + i * 2;
                        double aRe = A[aPos];
                        double aIm = A[aPos + 1];
                        // Compute 1/(aRe + aIm*i) * (xiRe + xiIm*i)
                        double denom = aRe * aRe + aIm * aIm;
                        double tempRe = (aRe * xiRe + aIm * xiIm) / denom;
                        double tempIm = (aRe * xiIm - aIm * xiRe) / denom;
                        xiRe = tempRe;
                        xiIm = tempIm;
                    }
                    
                    x[xPos] = xiRe;
                    x[xPos + 1] = xiIm;
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int xPos = xStart + i * incX * 2;
                    double xiRe = x[xPos];
                    double xiIm = x[xPos + 1];
                    
                    for (int j = 0; j < i; j++) {
                        int aPos = aStart + i * lda * 2 + j * 2;
                        int xjPos = xStart + j * incX * 2;
                        double aRe = A[aPos];
                        double aIm = A[aPos + 1];
                        double xjRe = x[xjPos];
                        double xjIm = x[xjPos + 1];
                        
                        xiRe -= aRe * xjRe - aIm * xjIm;
                        xiIm -= aRe * xjIm + aIm * xjRe;
                    }
                    
                    if (!unitDiag) {
                        int aPos = aStart + i * lda * 2 + i * 2;
                        double aRe = A[aPos];
                        double aIm = A[aPos + 1];
                        // Compute 1/(aRe + aIm*i) * (xiRe + xiIm*i)
                        double denom = aRe * aRe + aIm * aIm;
                        double tempRe = (aRe * xiRe + aIm * xiIm) / denom;
                        double tempIm = (aRe * xiIm - aIm * xiRe) / denom;
                        xiRe = tempRe;
                        xiIm = tempIm;
                    }
                    
                    x[xPos] = xiRe;
                    x[xPos + 1] = xiIm;
                }
            }
        } else {
            // x := A^{-T}*x  or  x := A^{-H}*x
            if (upper) {
                for (int i = 0; i < n; i++) {
                    int xPos = xStart + i * incX * 2;
                    double xiRe = x[xPos];
                    double xiIm = x[xPos + 1];
                    
                    for (int j = 0; j < i; j++) {
                        int aPos = aStart + j * lda * 2 + i * 2;
                        int xjPos = xStart + j * incX * 2;
                        double aRe = A[aPos];
                        double aIm = A[aPos + 1];
                        double xjRe = x[xjPos];
                        double xjIm = x[xjPos + 1];
                        
                        if (conjTrans) {
                            xiRe -= aRe * xjRe + aIm * xjIm;
                            xiIm -= aRe * xjIm - aIm * xjRe;
                        } else {
                            xiRe -= aRe * xjRe - aIm * xjIm;
                            xiIm -= aRe * xjIm + aIm * xjRe;
                        }
                    }
                    
                    if (!unitDiag) {
                        int aPos = aStart + i * lda * 2 + i * 2;
                        double aRe = A[aPos];
                        double aIm = A[aPos + 1];
                        // Compute 1/(aRe + aIm*i) * (xiRe + xiIm*i) for transpose
                        // or 1/(aRe - aIm*i) * (xiRe + xiIm*i) for conjugate transpose
                        if (conjTrans) {
                            double denom = aRe * aRe + aIm * aIm;
                            double tempRe = (aRe * xiRe - aIm * xiIm) / denom;
                            double tempIm = (aRe * xiIm + aIm * xiRe) / denom;
                            xiRe = tempRe;
                            xiIm = tempIm;
                        } else {
                            double denom = aRe * aRe + aIm * aIm;
                            double tempRe = (aRe * xiRe + aIm * xiIm) / denom;
                            double tempIm = (aRe * xiIm - aIm * xiRe) / denom;
                            xiRe = tempRe;
                            xiIm = tempIm;
                        }
                    }
                    
                    x[xPos] = xiRe;
                    x[xPos + 1] = xiIm;
                }
            } else {
                for (int i = n - 1; i >= 0; i--) {
                    int xPos = xStart + i * incX * 2;
                    double xiRe = x[xPos];
                    double xiIm = x[xPos + 1];
                    
                    for (int j = i + 1; j < n; j++) {
                        int aPos = aStart + j * lda * 2 + i * 2;
                        int xjPos = xStart + j * incX * 2;
                        double aRe = A[aPos];
                        double aIm = A[aPos + 1];
                        double xjRe = x[xjPos];
                        double xjIm = x[xjPos + 1];
                        
                        if (conjTrans) {
                            xiRe -= aRe * xjRe + aIm * xjIm;
                            xiIm -= aRe * xjIm - aIm * xjRe;
                        } else {
                            xiRe -= aRe * xjRe - aIm * xjIm;
                            xiIm -= aRe * xjIm + aIm * xjRe;
                        }
                    }
                    
                    if (!unitDiag) {
                        int aPos = aStart + i * lda * 2 + i * 2;
                        double aRe = A[aPos];
                        double aIm = A[aPos + 1];
                        // Compute 1/(aRe + aIm*i) * (xiRe + xiIm*i) for transpose
                        // or 1/(aRe - aIm*i) * (xiRe + xiIm*i) for conjugate transpose
                        if (conjTrans) {
                            double denom = aRe * aRe + aIm * aIm;
                            double tempRe = (aRe * xiRe - aIm * xiIm) / denom;
                            double tempIm = (aRe * xiIm + aIm * xiRe) / denom;
                            xiRe = tempRe;
                            xiIm = tempIm;
                        } else {
                            double denom = aRe * aRe + aIm * aIm;
                            double tempRe = (aRe * xiRe + aIm * xiIm) / denom;
                            double tempIm = (aRe * xiIm - aIm * xiRe) / denom;
                            xiRe = tempRe;
                            xiIm = tempIm;
                        }
                    }
                    
                    x[xPos] = xiRe;
                    x[xPos + 1] = xiIm;
                }
            }
        }
    }
}
