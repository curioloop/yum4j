package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.*;
/**
 * Unified complex BLAS interface aggregating all available complex routines for external access.
 *
 * <p>All matrices use row-major layout. The leading dimension ({@code lda}, {@code ldb}, etc.)
 * equals the number of columns for a dense row-major matrix.
 * Offset parameters ({@code aOff}, {@code xOff}, etc.) allow operating on sub-arrays
 * without copying.
 *
 * <h2>Complex BLAS Level 1 — Vector Operations</h2>
 * <ul>
 *   <li>{@link #zcopy}   — y := x</li>
 *   <li>{@link #zscal}   — x := alpha·x</li>
 *   <li>{@link #zswap}   — x ↔ y</li>
 *   <li>{@link #zaxpy}   — y := α·x + y</li>
 *   <li>{@link #zdotc}   — dot product conj(x)·y</li>
 *   <li>{@link #zdotu}   — dot product xᵀ·y</li>
 *   <li>{@link #dnrm2}   — Euclidean norm ‖x‖₂</li>
 *   <li>{@link #dasum}   — L1 norm Σ|xᵢ|</li>
 *   <li>{@link #izamax}  — index of max |xᵢ|</li>
 * </ul>
 *
 * <h2>Complex BLAS Level 2 — Matrix-Vector Operations</h2>
 * <ul>
 *   <li>{@link #zgemv}   — y := α·op(A)·x + β·y  (general)</li>
 *   <li>{@link #zgerc}   — A := α·x·y^H + A  (rank-1 update)</li>
 *   <li>{@link #zgeru}   — A := α·x·y^T + A  (rank-1 update)</li>
 *   <li>{@link #ztrmv}   — x := op(A)·x  (triangular)</li>
 *   <li>{@link #ztrsv}   — x := A^{-1}·x  (triangular solve)</li>
 *   <li>{@link #zhemv}   — y := α·A·x + β·y  (Hermitian)</li>
 *   <li>{@link #zher}    — A := α·x·x^H + A  (Hermitian rank-1)</li>
 *   <li>{@link #zher2}   — A := α·x·y^H + conj(α)·y·x^H + A  (Hermitian rank-2)</li>
 *   <li>{@link #zsymv}   — y := α·A·x + β·y  (symmetric)</li>
 *   <li>{@link #zsyr}    — A := α·x·x^T + A  (symmetric rank-1)</li>
 *   <li>{@link #zsyr2}   — A := α·x·y^T + conj(α)·y·x^T + A  (symmetric rank-2)</li>
 * </ul>
 *
 * <h2>Complex BLAS Level 3 — Matrix-Matrix Operations</h2>
 * <ul>
 *   <li>{@link #zgemm}   — C := α·op(A)·op(B) + β·C  (general)</li>
 * </ul>
 */
public interface ZLAS {

    // ========== Complex BLAS Level 1: Vector-Vector Operations ==========

    /**
     * Copies a complex vector x to y: y := x (BLAS ZCOPY).
     *
     * @param n    number of elements
     * @param x    source vector (interleaved format [re, im, re, im, ...])
     * @param xOff offset into x
     * @param incX stride in x
     * @param y    destination vector (interleaved format [re, im, re, im, ...])
     * @param yOff offset into y
     * @param incY stride in y
     */
    static void zcopy(int n, double[] x, int xOff, int incX, double[] y, int yOff, int incY) {
        Zcopy.zcopy(n, x, xOff, incX, y, yOff, incY);
    }

    /**
     * Scales a complex vector by a complex constant: x := alpha*x (BLAS ZSCAL).
     *
     * @param n     number of elements
     * @param alphaRe real part of scalar multiplier
     * @param alphaIm imaginary part of scalar multiplier
     * @param x     vector (interleaved format [re, im, re, im, ...])
     * @param xOff  offset into x
     * @param incX  stride in x
     */
    static void zscal(int n, double alphaRe, double alphaIm, double[] x, int xOff, int incX) {
        Zscal.zscal(n, alphaRe, alphaIm, x, xOff, incX);
    }

    /**
     * Swaps the contents of two complex vectors: x ↔ y (BLAS ZSWAP).
     *
     * @param n    number of elements
     * @param x    first vector (interleaved format [re, im, re, im, ...])
     * @param xOff offset into x
     * @param incX stride in x
     * @param y    second vector (interleaved format [re, im, re, im, ...])
     * @param yOff offset into y
     * @param incY stride in y
     */
    static void zswap(int n, double[] x, int xOff, int incX, double[] y, int yOff, int incY) {
        Zswap.zswap(n, x, xOff, incX, y, yOff, incY);
    }

    /**
     * Returns the index of the complex element with maximum absolute value (BLAS IZAMAX).
     *
     * @param n    number of elements
     * @param x    input vector (interleaved format [re, im, re, im, ...])
     * @param xOff offset into x
     * @param incX stride in x
     * @return 0-based index of the element with maximum absolute value
     */
    static int izamax(int n, double[] x, int xOff, int incX) {
        return Izamax.izamax(n, x, xOff, incX);
    }

    /**
     * Computes the Euclidean norm of a complex vector: ||x||₂ (BLAS DNRM2).
     *
     * @param n    number of elements
     * @param x    input vector (interleaved format [re, im, re, im, ...])
     * @param xOff offset into x
     * @param incX stride in x
     * @return Euclidean norm
     */
    static double dnrm2(int n, double[] x, int xOff, int incX) {
        return Dnrm2.dnrm2(n, x, xOff, incX);
    }

    /**
     * Computes the L1 norm (sum of absolute values) of a complex vector: Σ|x[i]| (BLAS DASUM).
     *
     * @param n    number of elements
     * @param x    input vector (interleaved format [re, im, re, im, ...])
     * @param xOff offset into x
     * @param incX stride in x
     * @return sum of absolute values
     */
    static double dasum(int n, double[] x, int xOff, int incX) {
        return Dasum.dasum(n, x, xOff, incX);
    }

    /**
     * Computes the Euclidean norm of a complex vector: ||x||₂ (BLAS DZNRM2).
     *
     * @param n    number of elements
     * @param x    input vector (interleaved format [re, im, re, im, ...])
     * @param xOff offset into x
     * @param incX stride in x
     * @return Euclidean norm
     */
    static double dznrm2(int n, double[] x, int xOff, int incX) {
        return Dznrm2.dznrm2(n, x, xOff, incX);
    }

    /**
     * Computes the L1 norm (sum of absolute values) of a complex vector: Σ|x[i]| (BLAS DZASUM).
     *
     * @param n    number of elements
     * @param x    input vector (interleaved format [re, im, re, im, ...])
     * @param xOff offset into x
     * @param incX stride in x
     * @return sum of absolute values
     */
    static double dzasum(int n, double[] x, int xOff, int incX) {
        return Dzasum.dzasum(n, x, xOff, incX);
    }

    /**
     * Computes the dot product of two complex vectors with conjugate of x: sum(conj(x[i]) * y[i]) (BLAS ZDOTC).
     *
     * @param n    number of elements
     * @param x    first vector (interleaved format [re, im, re, im, ...])
     * @param xOff offset into x
     * @param incX stride in x
     * @param y    second vector (interleaved format [re, im, re, im, ...])
     * @param yOff offset into y
     * @param incY stride in y
     * @param result output array [real, imaginary]
     */
    static void zdotc(int n, double[] x, int xOff, int incX, double[] y, int yOff, int incY, double[] result) {
        Zdot.zdotc(n, x, xOff, incX, y, yOff, incY, result, 0);
    }

    /**
     * Computes the unconjugated dot product of two complex vectors: sum(x[i] * y[i]) (BLAS ZDOTU).
     *
     * @param n    number of elements
     * @param x    first vector (interleaved format [re, im, re, im, ...])
     * @param xOff offset into x
     * @param incX stride in x
     * @param y    second vector (interleaved format [re, im, re, im, ...])
     * @param yOff offset into y
     * @param incY stride in y
     * @param result output array [real, imaginary]
     */
    static void zdotu(int n, double[] x, int xOff, int incX, double[] y, int yOff, int incY, double[] result) {
        Zdot.zdotu(n, x, xOff, incX, y, yOff, incY, result, 0);
    }

    /**
     * Computes y := alpha*x + y for complex vectors (BLAS ZAXPY).
     *
     * @param n     number of elements
     * @param alphaRe real part of scalar multiplier
     * @param alphaIm imaginary part of scalar multiplier
     * @param x     input vector (interleaved format [re, im, re, im, ...])
     * @param xOff  offset into x
     * @param incX  stride in x
     * @param y     input/output vector (interleaved format [re, im, re, im, ...])
     * @param yOff  offset into y
     * @param incY  stride in y
     */
    static void zaxpy(int n, double alphaRe, double alphaIm, double[] x, int xOff, int incX, double[] y, int yOff, int incY) {
        Zaxpy.zaxpy(n, alphaRe, alphaIm, x, xOff, incX, y, yOff, incY);
    }

    // ========== Complex BLAS Level 2: Matrix-Vector Operations ==========

    /**
     * Complex matrix-vector product: y := alpha*op(A)*x + beta*y (BLAS ZGEMV).
     *
     * @param trans 'N' for no transpose, 'T' for transpose, 'C' for conjugate transpose
     * @param m number of rows in A
     * @param n number of columns in A
     * @param alphaRe real part of scalar alpha
     * @param alphaIm imaginary part of scalar alpha
     * @param A matrix (m × n, row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param x input vector (interleaved format)
     * @param xOff offset into x
     * @param incX stride in x
     * @param betaRe real part of scalar beta
     * @param betaIm imaginary part of scalar beta
     * @param y output vector (interleaved format)
     * @param yOff offset into y
     * @param incY stride in y
     */
    static void zgemv(BLAS.Trans trans, int m, int n, double alphaRe, double alphaIm, 
                      double[] A, int aOff, int lda, 
                      double[] x, int xOff, int incX, double betaRe, double betaIm, 
                      double[] y, int yOff, int incY) {
        Zgemv.zgemv(trans, m, n, alphaRe, alphaIm, A, aOff, lda, x, xOff, incX, betaRe, betaIm, y, yOff, incY);
    }

    /**
     * Performs complex rank-1 update: A := alpha * x * y^H + A (BLAS ZGERC).
     *
     * @param m number of rows in A
     * @param n number of columns in A
     * @param alphaRe real part of scalar alpha
     * @param alphaIm imaginary part of scalar alpha
     * @param x vector (length m, interleaved format)
     * @param xOff offset into x
     * @param incX stride in x
     * @param y vector (length n, interleaved format)
     * @param yOff offset into y
     * @param incY stride in y
     * @param A matrix (m × n, row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     */
    static void zgerc(int m, int n, double alphaRe, double alphaIm, double[] x, int xOff, int incX, 
                      double[] y, int yOff, int incY, double[] A, int aOff, int lda) {
        Zgerc.zgerc(m, n, alphaRe, alphaIm, x, xOff, incX, y, yOff, incY, A, aOff, lda);
    }

    /**
     * Performs complex rank-1 update without conjugation: A := alpha * x * y^T + A (BLAS ZGERU).
     *
     * @param m number of rows in A
     * @param n number of columns in A
     * @param alphaRe real part of scalar alpha
     * @param alphaIm imaginary part of scalar alpha
     * @param x vector (length m, interleaved format)
     * @param xOff offset into x
     * @param incX stride in x
     * @param y vector (length n, interleaved format)
     * @param yOff offset into y
     * @param incY stride in y
     * @param A matrix (m × n, row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     */
    static void zgeru(int m, int n, double alphaRe, double alphaIm, double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY, double[] A, int aOff, int lda) {
        Zgeru.zgeru(m, n, alphaRe, alphaIm, x, xOff, incX, y, yOff, incY, A, aOff, lda);
    }

    /**
     * Complex triangular matrix-vector multiplication: x := op(A)*x (BLAS ZTRMV).
     *
     * @param uplo 'U' for upper, 'L' for lower triangular
     * @param trans 'N' for no transpose, 'T' for transpose, 'C' for conjugate transpose
     * @param diag 'U' for unit diagonal, 'N' for non-unit
     * @param n dimension of A
     * @param A triangular matrix (row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param x vector (interleaved format)
     * @param xOff offset into x
     * @param incX stride of x
     */
    static void ztrmv(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, int n, 
                      double[] A, int aOff, int lda, 
                      double[] x, int xOff, int incX) {
        Ztrmv.ztrmv(uplo, trans, diag, n, A, aOff, lda, x, xOff, incX);
    }

    /**
     * Solves complex triangular system: x := A^{-1}*x or x := A^{-T}*x or x := A^{-H}*x (BLAS ZTRSV).
     *
     * @param uplo 'U' for upper, 'L' for lower triangular
     * @param trans 'N' for no transpose, 'T' for transpose, 'C' for conjugate transpose
     * @param diag 'U' for unit diagonal, 'N' for non-unit
     * @param n dimension of A
     * @param A triangular matrix (row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param x vector (interleaved format)
     * @param xOff offset into x
     * @param incX stride of x
     */
    static void ztrsv(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, int n, 
                      double[] A, int aOff, int lda, 
                      double[] x, int xOff, int incX) {
        Ztrsv.ztrsv(uplo, trans, diag, n, A, aOff, lda, x, xOff, incX);
    }

    /**
     * Complex Hermitian matrix-vector product: y := alpha*A*x + beta*y (BLAS ZHEMV).
     *
     * @param uplo 'U' for upper, 'L' for lower triangular part of A
     * @param n order of A
     * @param alphaRe real part of scalar alpha
     * @param alphaIm imaginary part of scalar alpha
     * @param A Hermitian matrix (row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param x input vector (interleaved format)
     * @param xOff offset into x
     * @param incX stride in x
     * @param betaRe real part of scalar beta
     * @param betaIm imaginary part of scalar beta
     * @param y output vector (interleaved format)
     * @param yOff offset into y
     * @param incY stride in y
     */
    static void zhemv(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm, 
                      double[] A, int aOff, int lda, 
                      double[] x, int xOff, int incX, double betaRe, double betaIm, 
                      double[] y, int yOff, int incY) {
        Zhemv.zhemv(uplo, n, alphaRe, alphaIm, A, aOff, lda, x, xOff, incX, betaRe, betaIm, y, yOff, incY);
    }

    /**
     * Complex Hermitian rank-1 update: A := alpha*x*x^H + A (BLAS ZHER).
     *
     * @param uplo 'U' for upper, 'L' for lower triangular part of A
     * @param n order of A
     * @param alpha real scalar alpha
     * @param x vector (interleaved format)
     * @param xOff offset into x
     * @param incX stride in x
     * @param A Hermitian matrix (row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     */
    static void zher(BLAS.Uplo uplo, int n, double alpha, 
                     double[] x, int xOff, int incX, 
                     double[] A, int aOff, int lda) {
        Zher.zher(uplo, n, alpha, x, xOff, incX, A, aOff, lda);
    }

    /**
     * Complex Hermitian rank-2 update: A := alpha*x*y^H + conj(alpha)*y*x^H + A (BLAS ZHER2).
     *
     * @param uplo 'U' for upper, 'L' for lower triangular part of A
     * @param n order of A
     * @param alphaRe real part of scalar alpha
     * @param alphaIm imaginary part of scalar alpha
     * @param x vector (interleaved format)
     * @param xOff offset into x
     * @param incX stride in x
     * @param y vector (interleaved format)
     * @param yOff offset into y
     * @param incY stride in y
     * @param A Hermitian matrix (row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     */
    static void zher2(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm, 
                      double[] x, int xOff, int incX, 
                      double[] y, int yOff, int incY, 
                      double[] A, int aOff, int lda) {
        Zher2.zher2(uplo, n, alphaRe, alphaIm, x, xOff, incX, y, yOff, incY, A, aOff, lda);
    }

    /**
     * Complex symmetric matrix-vector product: y := alpha*A*x + beta*y (BLAS ZSYMV).
     *
     * @param uplo 'U' for upper, 'L' for lower triangular part of A
     * @param n order of A
     * @param alphaRe real part of scalar alpha
     * @param alphaIm imaginary part of scalar alpha
     * @param A symmetric matrix (row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param x input vector (interleaved format)
     * @param xOff offset into x
     * @param incX stride in x
     * @param betaRe real part of scalar beta
     * @param betaIm imaginary part of scalar beta
     * @param y output vector (interleaved format)
     * @param yOff offset into y
     * @param incY stride in y
     */
    static void zsymv(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm, 
                      double[] A, int aOff, int lda, 
                      double[] x, int xOff, int incX, double betaRe, double betaIm, 
                      double[] y, int yOff, int incY) {
        Zsymv.zsymv(uplo, n, alphaRe, alphaIm, A, aOff, lda, x, xOff, incX, betaRe, betaIm, y, yOff, incY);
    }

    /**
     * Complex symmetric rank-1 update: A := alpha*x*x^T + A (BLAS ZSYR).
     *
     * @param uplo 'U' for upper, 'L' for lower triangular part of A
     * @param n order of A
     * @param alphaRe real part of scalar alpha
     * @param alphaIm imaginary part of scalar alpha
     * @param x vector (interleaved format)
     * @param xOff offset into x
     * @param incX stride in x
     * @param A symmetric matrix (row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     */
    static void zsyr(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm, 
                     double[] x, int xOff, int incX, 
                     double[] A, int aOff, int lda) {
        Zsyr.zsyr(uplo, n, alphaRe, alphaIm, x, xOff, incX, A, aOff, lda);
    }

    /**
     * Complex symmetric rank-2 update: A := alpha*x*y^T + conj(alpha)*y*x^T + A (BLAS ZSYR2).
     *
     * @param uplo 'U' for upper, 'L' for lower triangular part of A
     * @param n order of A
     * @param alphaRe real part of scalar alpha
     * @param alphaIm imaginary part of scalar alpha
     * @param x vector (interleaved format)
     * @param xOff offset into x
     * @param incX stride in x
     * @param y vector (interleaved format)
     * @param yOff offset into y
     * @param incY stride in y
     * @param A symmetric matrix (row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     */
    static void zsyr2(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm, 
                      double[] x, int xOff, int incX, 
                      double[] y, int yOff, int incY, 
                      double[] A, int aOff, int lda) {
        Zsyr2.zsyr2(uplo, n, alphaRe, alphaIm, x, xOff, incX, y, yOff, incY, A, aOff, lda);
    }

    // ========== Complex BLAS Level 3: Matrix-Matrix Operations ==========

    /**
     * Complex matrix-matrix multiplication: C := alpha*op(A)*op(B) + beta*C (BLAS ZGEMM).
     *
     * @param transA 'N' for A, 'T' for A^T, 'C' for A^H
     * @param transB 'N' for B, 'T' for B^T, 'C' for B^H
     * @param m rows in C and op(A)
     * @param n columns in C and op(B)
     * @param k columns in op(A) / rows in op(B)
     * @param alphaRe real part of scalar alpha
     * @param alphaIm imaginary part of scalar alpha
     * @param A matrix A (row-major, interleaved format)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param B matrix B (row-major, interleaved format)
     * @param bOff offset into B
     * @param ldb leading dimension of B
     * @param betaRe real part of scalar beta
     * @param betaIm imaginary part of scalar beta
     * @param C result matrix (row-major, interleaved format)
     * @param cOff offset into C
     * @param ldc leading dimension of C
     */
    static void zgemm(BLAS.Trans transA, BLAS.Trans transB, int m, int n, int k, 
                      double alphaRe, double alphaIm, double[] A, int aOff, int lda, 
                      double[] B, int bOff, int ldb, 
                      double betaRe, double betaIm, double[] C, int cOff, int ldc) {
        Zgemm.zgemm(transA, transB, m, n, k, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb, betaRe, betaIm, C, cOff, ldc);
    }

    // ========== Complex LAPACK Routines ==========

    /**
     * Computes all eigenvalues and optionally eigenvectors of a complex Hermitian matrix (LAPACK ZHEEV).
     *
     * @param jobz  'N' for eigenvalues only, 'V' for eigenvalues and eigenvectors
     * @param uplo  'U' for upper triangle, 'L' for lower triangle
     * @param n     order of matrix A
     * @param A     complex Hermitian matrix (n x n, row-major, interleaved format), overwritten on exit
     * @param lda   leading dimension of A
     * @param w     eigenvalues (output, length n)
     * @param wOff  offset into w
     * @param work  workspace (length lwork)
     * @param workOff offset into work
     * @param lwork length of workspace (>= 3*n-1, or -1 for query)
     * @return 0 on success, non-zero on failure
     */
    static int zheev(char jobz, char uplo, int n, double[] A, int lda, 
                     double[] w, int wOff, double[] work, int workOff, int lwork) {
        return Zheev.zheev(jobz, uplo, n, A, lda, w, wOff, work, workOff, lwork);
    }

    /**
     * Computes all eigenvalues and optionally eigenvectors of a complex general matrix (LAPACK ZGEEV).
     *
     * @param jobvl  'N' for no left eigenvectors, 'V' for left eigenvectors
     * @param jobvr  'N' for no right eigenvectors, 'V' for right eigenvectors
     * @param n      order of matrix A
     * @param A      complex general matrix (n x n, row-major, interleaved format), overwritten on exit
     * @param lda    leading dimension of A
     * @param wr     real parts of eigenvalues (output, length n)
     * @param wi     imaginary parts of eigenvalues (output, length n)
     * @param vl     left eigenvectors (output, n x n, row-major, interleaved format), null if jobvl = 'N'
     * @param ldvl   leading dimension of vl
     * @param vr     right eigenvectors (output, n x n, row-major, interleaved format), null if jobvr = 'N'
     * @param ldvr   leading dimension of vr
     * @param work   workspace (length lwork)
     * @param workOff offset into work
     * @param lwork  length of workspace (>= 4*n, or -1 for query)
     * @return 0 on success, non-zero on failure
     */
    static int zgeev(char jobvl, char jobvr, int n, double[] A, int lda, 
                     double[] wr, double[] wi, 
                     double[] vl, int ldvl, double[] vr, int ldvr, 
                     double[] work, int workOff, int lwork) {
        return Zgeev.zgeev(jobvl, jobvr, n, A, lda, wr, wi, vl, ldvl, vr, ldvr, work, workOff, lwork);
    }

    /**
     * Computes the Cholesky factorization of a complex Hermitian positive definite matrix (LAPACK ZPOTRF).
     *
     * @param uplo  'U' for upper triangle, 'L' for lower triangle
     * @param n     order of matrix A
     * @param A     complex Hermitian matrix (n x n, row-major, interleaved format), overwritten with factorization
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @return 0 on success, non-zero on failure (return value is the first index where factorization failed)
     */
    static int zpotrf(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        return Zpotr.zpotrf(uplo, n, A, aOff, lda);
    }

    /**
     * Computes the LU factorization of a complex general matrix (LAPACK ZGETRF).
     *
     * @param m     number of rows of matrix A
     * @param n     number of columns of matrix A
     * @param A     complex general matrix (m x n, row-major, interleaved format), overwritten with factorization
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param ipiv  pivot indices (output, length min(m,n))
     * @return 0 on success, non-zero on failure
     */
    static int zgetrf(int m, int n, double[] A, int aOff, int lda, int[] ipiv) {
        return Zgetr.zgetrf(m, n, A, aOff, lda, ipiv);
    }

    /**
     * Solves a triangular system of the form A * X = B or A^T * X = B or A^H * X = B (LAPACK ZTRTRS).
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
     * @param ldb   leading dimension of B
     * @return 0 on success, non-zero on failure (return value is the first index where A is singular)
     */
    static int ztrtrs(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, 
                      int n, int nrhs, double[] A, int aOff, int lda, 
                      double[] B, int bOff, int ldb) {
        return Ztrtrs.ztrtrs(uplo, trans, diag, n, nrhs, A, aOff, lda, B, bOff, ldb);
    }

    static int zgees(char jobvs, char sort, Zselect select, int n,
                     double[] A, int aOff, int lda,
                     double[] w, int wOff,
                     double[] vs, int vsOff, int ldvs,
                     double[] work, int workOff, int lwork,
                     double[] rwork,
                     boolean[] bwork) {
        return Zgees.zgees(jobvs, sort, select, n, A, aOff, lda,
                w, wOff, vs, vsOff, ldvs,
            work, workOff, lwork, rwork, 0, bwork);
        }

        static int zgees(char jobvs, char sort, Zselect select, int n,
                 double[] A, int aOff, int lda,
                 double[] w, int wOff,
                 double[] vs, int vsOff, int ldvs,
                 double[] work, int workOff, int lwork,
                 double[] rwork,
                 int rworkOff,
                 boolean[] bwork) {
        return Zgees.zgees(jobvs, sort, select, n, A, aOff, lda,
            w, wOff, vs, vsOff, ldvs,
            work, workOff, lwork, rwork, rworkOff, bwork);
    }

    static void ztrsm(BLAS.Side side, BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                      int m, int n, double alphaRe, double alphaIm,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb) {
        Ztrsm.ztrsm(side, uplo, trans, diag, m, n, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb);
    }

    static double ztrsyl(char trana, char tranb, int isgn, int m, int n,
                         double[] a, int aOff, int lda,
                         double[] b, int bOff, int ldb,
                         double[] c, int cOff, int ldc,
                         boolean[] okOut) {
        return Ztrsyl.ztrsyl(trana, tranb, isgn, m, n, a, aOff, lda, b, bOff, ldb, c, cOff, ldc, okOut);
    }

    static void zgetrs(BLAS.Trans trans, int n, int nrhs,
                       double[] A, int aOff, int lda, int[] ipiv,
                       double[] B, int bOff, int ldb) {
        Zgetr.zgetrs(trans, n, nrhs, A, aOff, lda, ipiv, B, bOff, ldb);
    }

    static int zgetf2(int m, int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff) {
        return Zgetr.zgetf2(m, n, A, aOff, lda, ipiv, ipivOff);
    }

    static boolean zgetri(int n, double[] A, int aOff, int lda, int[] ipiv,
                          double[] work, int workOff, int lwork) {
        return Zgetr.zgetri(n, A, aOff, lda, ipiv, work, workOff, lwork);
    }

    static double zgecon(char norm, int n, double[] A, int aOff, int lda, double anorm,
                         double[] work, int[] iwork) {
        return Zgetr.zgecon(norm, n, A, aOff, lda, anorm, work, iwork);
    }

    static boolean zpotf2(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        return Zpotr.zpotf2(uplo, n, A, aOff, lda);
    }

    static void zpotrs(BLAS.Uplo uplo, int n, int nrhs, double[] A, int aOff, int lda,
                       double[] B, int bOff, int ldb) {
        Zpotr.zpotrs(uplo, n, nrhs, A, aOff, lda, B, bOff, ldb);
    }

    static boolean zpotri(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        return Zpotr.zpotri(uplo, n, A, aOff, lda);
    }

    static double zpocon(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda, double anorm,
                         double[] work, int[] iwork) {
        return Zpotr.zpocon(uplo, n, A, aOff, lda, anorm, work, iwork);
    }

    static int ztrtri(BLAS.Uplo uplo, BLAS.Diag diag, int n, double[] A, int aOff, int lda) {
        return Ztrtri.ztrtri(uplo, diag, n, A, aOff, lda);
    }

    static double ztrcon(char norm, BLAS.Uplo uplo, BLAS.Diag diag, int n,
                         double[] A, int aOff, int lda,
                         double[] work, int[] iwork) {
        return Ztrtri.ztrcon(norm, uplo, diag, n, A, aOff, lda, work, iwork);
    }

    static void zlauum(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        Zlauum.zlauum(uplo, n, A, aOff, lda);
    }

    static void zherk(BLAS.Uplo uplo, BLAS.Trans trans, int n, int k,
                      double alpha, double[] A, int aOff, int lda,
                      double beta, double[] C, int cOff, int ldc) {
        Zherk.zherk(uplo, trans, n, k, alpha, A, aOff, lda, beta, C, cOff, ldc);
    }

    static void ztrmm(BLAS.Side side, BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                      int m, int n, double alphaRe, double alphaIm,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb) {
        Ztrmm.ztrmm(side, uplo, trans, diag, m, n, alphaRe, alphaIm, A, aOff, lda, B, bOff, ldb);
    }
}
