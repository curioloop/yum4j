/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZorgqrTest {

    private static final double TOL = 1e-10;

    @Test
    void testSquareMatrix() {
        // Test a 3x3 complex matrix
        // First, perform QR factorization
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 19.0, 20.0
        };
        int m = 3;
        int n = 3;
        int k = 3;
        int lda = 3;
        
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        double[] Q = new double[m * n * 2];
        System.arraycopy(A, 0, Q, 0, A.length);
        double[] work2 = new double[k * 2];
        int lwork2 = k;

        int info = Zgeqr.zorgqr(m, n, k, Q, 0, lda, tau, 0, work2, 0, lwork2);

        assertEquals(0, info);

        checkUnitaryMatrix(Q, m, n);
    }

    @Test
    void testRectangularMatrixMoreRows() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 25.0, 26.0
        };
        int m = 4;
        int n = 3;
        int k = 3;
        int lda = 3;
        
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        double[] Q = new double[m * n * 2];
        System.arraycopy(A, 0, Q, 0, A.length);
        double[] work2 = new double[k * 2];
        int lwork2 = k;

        int info = Zgeqr.zorgqr(m, n, k, Q, 0, lda, tau, 0, work2, 0, lwork2);

        assertEquals(0, info);

        checkUnitaryMatrix(Q, m, n);
    }

    @Test
    void testRectangularMatrixMoreColumns() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 25.0, 26.0
        };
        int m = 3;
        int n = 4;
        int k = 3;
        int lda = 4;
        
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        double[] Q = new double[m * n * 2];
        System.arraycopy(A, 0, Q, 0, A.length);
        double[] work2 = new double[k * 2];
        int lwork2 = k;

        int info = Zgeqr.zorgqr(m, n, k, Q, 0, lda, tau, 0, work2, 0, lwork2);

        assertEquals(0, info);

        checkUnitaryMatrix(Q, m, n);
    }

    @Test
    void testKLessThanMinMN() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 19.0, 20.0
        };
        int m = 3;
        int n = 3;
        int k = 2;
        int lda = 3;
        
        double[] tau = new double[Math.min(m, n) * 2];
        double[] work = new double[Math.min(m, n) * 2];
        int lwork = Math.min(m, n);

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        double[] Q = new double[m * n * 2];
        System.arraycopy(A, 0, Q, 0, A.length);
        double[] work2 = new double[k * 2];
        int lwork2 = k;

        int info = Zgeqr.zorgqr(m, n, k, Q, 0, lda, tau, 0, work2, 0, lwork2);

        assertEquals(0, info);

        checkUnitaryMatrix(Q, m, n);
    }

    @Test
    void testWorkspaceQuery() {
        // Test workspace query mode
        double[] A = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        };
        int m = 2;
        int n = 2;
        int k = 2;
        int lda = 2;
        
        double[] tau = new double[k * 2];
        double[] work = new double[1];
        int lwork = -1;

        int info = Zgeqr.zorgqr(m, n, k, A, 0, lda, tau, 0, work, 0, lwork);

        assertEquals(0, info);
        assertTrue(work[0] > 0);
    }

    @Test
    void testZeroMatrix() {
        // Test with zero matrix
        int m = 3;
        int n = 3;
        int k = 3;
        int lda = 3;
        
        double[] A = new double[m * n * 2]; // All zeros
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Now generate the orthogonal matrix Q
        double[] Q = new double[m * n * 2];
        System.arraycopy(A, 0, Q, 0, A.length);
        double[] work2 = new double[k * 2];
        int lwork2 = k;

        int info = Zgeqr.zorgqr(m, n, k, Q, 0, lda, tau, 0, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that Q is identity matrix
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    assertEquals(1.0, Q[i * lda * 2 + j * 2], TOL, "Q[" + i + "," + j + "] should be 1");
                    assertEquals(0.0, Q[i * lda * 2 + j * 2 + 1], TOL, "Q[" + i + "," + j + "] imaginary part should be 0");
                } else {
                    assertEquals(0.0, Q[i * lda * 2 + j * 2], TOL, "Q[" + i + "," + j + "] should be 0");
                    assertEquals(0.0, Q[i * lda * 2 + j * 2 + 1], TOL, "Q[" + i + "," + j + "] imaginary part should be 0");
                }
            }
        }
    }

    @Test
    void testEmptyMatrix() {
        // Test with zero-sized matrix
        double[] A = new double[0];
        double[] tau = new double[0];
        double[] work = new double[0];
        int lwork = 0;

        int info = Zgeqr.zorgqr(0, 0, 0, A, 0, 1, tau, 0, work, 0, lwork);

        assertEquals(0, info);
    }

    @Test
    void testSmallMatrix() {
        // Test a 1x1 matrix
        double[] A = {1.0, 2.0};
        int m = 1;
        int n = 1;
        int k = 1;
        int lda = 1;
        
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        // First, perform QR factorization
        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Now generate the orthogonal matrix Q
        double[] Q = new double[m * n * 2];
        System.arraycopy(A, 0, Q, 0, A.length);
        double[] work2 = new double[k * 2];
        int lwork2 = k;

        int info = Zgeqr.zorgqr(m, n, k, Q, 0, lda, tau, 0, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that Q is unitary
        checkUnitaryMatrix(Q, m, n);
    }

    @Test
    void testInvalidInputs() {
        // Test with invalid m
        double[] A = {1.0, 2.0, 3.0, 4.0};
        double[] tau = new double[2 * 2];
        double[] work = new double[2 * 2];
        int lwork = 2;

        int info = Zgeqr.zorgqr(-1, 2, 2, A, 0, 2, tau, 0, work, 0, lwork);
        assertEquals(-1, info);

        // Test with invalid n
        info = Zgeqr.zorgqr(2, -1, 2, A, 0, 2, tau, 0, work, 0, lwork);
        assertEquals(-2, info);

        // Test with invalid k
        info = Zgeqr.zorgqr(2, 2, -1, A, 0, 2, tau, 0, work, 0, lwork);
        assertEquals(-3, info);

        // Test with k > min(m, n)
        info = Zgeqr.zorgqr(2, 2, 3, A, 0, 2, tau, 0, work, 0, lwork);
        assertEquals(-3, info);

        // Test with invalid lda
        info = Zgeqr.zorgqr(2, 2, 2, A, 0, 1, tau, 0, work, 0, lwork);
        assertEquals(-5, info);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectQGeneration() {
        int m = 3;
        int n = 3;
        int k = 3;
        int lda = 5;
        int aOff = 6;

        double[] directFactorized = {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 19.0, 20.0
        };
        double[] tauDirect = new double[k * 2];
        double[] factorWork = new double[k * 2];
        Zgeqr.zgeqrf(m, n, directFactorized, 0, n, tauDirect, 0, factorWork, 0, k);

        double[] offsetFactorized = new double[(aOff + (m - 1) * lda + n + 1) * 2];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int src = (i * n + j) * 2;
                int dst = (aOff + i * lda + j) * 2;
                offsetFactorized[dst] = directFactorized[src];
                offsetFactorized[dst + 1] = directFactorized[src + 1];
            }
        }

        double[] directQ = directFactorized.clone();
        double[] offsetTau = tauDirect.clone();
        double[] workDirect = new double[k * 2];
        double[] workOffset = new double[k * 2];

        assertEquals(0, Zgeqr.zorgqr(m, n, k, directQ, 0, n, tauDirect, 0, workDirect, 0, k));
        assertEquals(0, Zgeqr.zorgqr(m, n, k, offsetFactorized, aOff, lda, offsetTau, 0, workOffset, 0, k));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int src = (i * n + j) * 2;
                int dst = (aOff + i * lda + j) * 2;
                assertEquals(directQ[src], offsetFactorized[dst], TOL);
                assertEquals(directQ[src + 1], offsetFactorized[dst + 1], TOL);
            }
        }
    }

    /**
     * Helper method to check if a matrix has orthonormal columns
     */
    private void checkUnitaryMatrix(double[] Q, int m, int n) {
        int minDim = Math.min(m, n);
        
        // Compute Q^H * Q
        double[] qhtimesq = new double[minDim * minDim * 2];
        Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, minDim, minDim, m, 1.0, 0.0, Q, 0, n, Q, 0, n, 0.0, 0.0, qhtimesq, 0, minDim);
        
        // Check that the result is identity matrix
        for (int i = 0; i < minDim; i++) {
            for (int j = 0; j < minDim; j++) {
                double real = qhtimesq[i * minDim * 2 + j * 2];
                double imag = qhtimesq[i * minDim * 2 + j * 2 + 1];
                
                if (i == j) {
                    assertEquals(1.0, real, TOL, "Q^H*Q[" + i + "," + j + "] real part should be 1");
                    assertEquals(0.0, imag, TOL, "Q^H*Q[" + i + "," + j + "] imaginary part should be 0");
                } else {
                    assertEquals(0.0, real, TOL, "Q^H*Q[" + i + "," + j + "] real part should be 0");
                    assertEquals(0.0, imag, TOL, "Q^H*Q[" + i + "," + j + "] imaginary part should be 0");
                }
            }
        }
    }

}
