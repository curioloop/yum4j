/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZgeqrfTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasicSquareMatrix() {
        // Test a 3x3 complex matrix
        // A = [[1+2i, 3+4i, 5+6i], [7+8i, 9+10i, 11+12i], [13+14i, 15+16i, 17+18i]]
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 19.0, 20.0
        };
        double[] tau = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgeqr.zgeqrf(3, 3, A, 0, 3, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        assertTrue(Math.abs(A[0 * 3 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[0 * 3 * 2 + 0 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[0 * 3 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[0 * 3 * 2 + 1 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[0 * 3 * 2 + 2 * 2 + 0]) > TOL || Math.abs(A[0 * 3 * 2 + 2 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[1 * 3 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[1 * 3 * 2 + 1 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[1 * 3 * 2 + 2 * 2 + 0]) > TOL || Math.abs(A[1 * 3 * 2 + 2 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[2 * 3 * 2 + 2 * 2 + 0]) > TOL || Math.abs(A[2 * 3 * 2 + 2 * 2 + 1]) > TOL);

        for (int i = 0; i < 3; i++) {
            assertTrue(Double.isFinite(tau[i * 2]) && Double.isFinite(tau[i * 2 + 1]));
        }
    }

    @Test
    void testRectangularMatrixMoreRows() {
        // Test a 3x2 complex matrix (more rows than columns)
        double[] A = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        };
        double[] tau = new double[2 * 2];
        double[] work = new double[2 * 2];
        int lwork = 2;

        int info = Zgeqr.zgeqrf(3, 2, A, 0, 2, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        // Check that the matrix is now in QR form
        assertTrue(Math.abs(A[0 * 2 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[0 * 2 * 2 + 0 * 2 + 1]) > TOL); // A[0][0]
        assertTrue(Math.abs(A[0 * 2 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[0 * 2 * 2 + 1 * 2 + 1]) > TOL); // A[0][1]
        assertTrue(Math.abs(A[1 * 2 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[1 * 2 * 2 + 1 * 2 + 1]) > TOL); // A[1][1]

        // Check that tau is non-zero
        for (int i = 0; i < 2; i++) {
            assertTrue(Math.abs(tau[i * 2]) > TOL || Math.abs(tau[i * 2 + 1]) > TOL);
        }
    }

    @Test
    void testRectangularMatrixMoreColumns() {
        // Test a 2x3 complex matrix (more columns than rows)
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        };
        double[] tau = new double[2 * 2];
        double[] work = new double[2 * 2];
        int lwork = 2;

        int info = Zgeqr.zgeqrf(2, 3, A, 0, 3, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        // Check that the matrix is now in QR form
        assertTrue(Math.abs(A[0 * 3 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[0 * 3 * 2 + 0 * 2 + 1]) > TOL); // A[0][0]
        assertTrue(Math.abs(A[0 * 3 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[0 * 3 * 2 + 1 * 2 + 1]) > TOL); // A[0][1]
        assertTrue(Math.abs(A[0 * 3 * 2 + 2 * 2 + 0]) > TOL || Math.abs(A[0 * 3 * 2 + 2 * 2 + 1]) > TOL); // A[0][2]
        assertTrue(Math.abs(A[1 * 3 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[1 * 3 * 2 + 1 * 2 + 1]) > TOL); // A[1][1]
        assertTrue(Math.abs(A[1 * 3 * 2 + 2 * 2 + 0]) > TOL || Math.abs(A[1 * 3 * 2 + 2 * 2 + 1]) > TOL); // A[1][2]

        for (int i = 0; i < 2; i++) {
            assertTrue(Double.isFinite(tau[i * 2]) && Double.isFinite(tau[i * 2 + 1]));
        }
    }

    @Test
    void testWorkspaceQuery() {
        // Test workspace query mode
        double[] A = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        };
        double[] tau = new double[2 * 2];
        double[] work = new double[1];
        int lwork = -1;

        int info = Zgeqr.zgeqrf(2, 2, A, 0, 2, tau, 0, work, 0, lwork);

        assertEquals(0, info);
        assertTrue(work[0] > 0);

        lwork = (int) work[0];
        work = new double[lwork * 2];
        info = Zgeqr.zgeqrf(2, 2, A, 0, 2, tau, 0, work, 0, lwork);
        assertEquals(0, info);
    }

    @Test
    void testZeroMatrix() {
        // Test with zero matrix
        double[] A = {
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        };
        double[] tau = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgeqr.zgeqrf(3, 3, A, 0, 3, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        // Check that all elements are still zero
        for (int i = 0; i < A.length; i++) {
            assertTrue(Math.abs(A[i]) < TOL);
        }

        // Check that tau is zero
        for (int i = 0; i < 3; i++) {
            assertTrue(Math.abs(tau[i * 2]) < TOL && Math.abs(tau[i * 2 + 1]) < TOL);
        }
    }

    @Test
    void testIdentityMatrix() {
        // Test with identity matrix
        double[] A = new double[3 * 3 * 2];
        // Set diagonal elements to 1.0 + 0.0i
        A[0 * 3 * 2 + 0 * 2 + 0] = 1.0; // A[0][0] real
        A[1 * 3 * 2 + 1 * 2 + 0] = 1.0; // A[1][1] real
        A[2 * 3 * 2 + 2 * 2 + 0] = 1.0; // A[2][2] real
        // All other elements are 0.0 + 0.0i
        double[] tau = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgeqr.zgeqrf(3, 3, A, 0, 3, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        // Check that the matrix is still diagonal
        assertTrue(Math.abs(A[0 * 3 * 2 + 0 * 2 + 0]) > TOL); // A[0][0]
        assertTrue(Math.abs(A[1 * 3 * 2 + 1 * 2 + 0]) > TOL); // A[1][1]
        assertTrue(Math.abs(A[2 * 3 * 2 + 2 * 2 + 0]) > TOL); // A[2][2]

        // Check that the other elements are zero
        assertTrue(Math.abs(A[0 * 3 * 2 + 1 * 2 + 0]) < TOL && Math.abs(A[0 * 3 * 2 + 1 * 2 + 1]) < TOL); // A[0][1]
        assertTrue(Math.abs(A[0 * 3 * 2 + 2 * 2 + 0]) < TOL && Math.abs(A[0 * 3 * 2 + 2 * 2 + 1]) < TOL); // A[0][2]
        assertTrue(Math.abs(A[1 * 3 * 2 + 0 * 2 + 0]) < TOL && Math.abs(A[1 * 3 * 2 + 0 * 2 + 1]) < TOL); // A[1][0]
        assertTrue(Math.abs(A[1 * 3 * 2 + 2 * 2 + 0]) < TOL && Math.abs(A[1 * 3 * 2 + 2 * 2 + 1]) < TOL); // A[1][2]
        assertTrue(Math.abs(A[2 * 3 * 2 + 0 * 2 + 0]) < TOL && Math.abs(A[2 * 3 * 2 + 0 * 2 + 1]) < TOL); // A[2][0]
        assertTrue(Math.abs(A[2 * 3 * 2 + 1 * 2 + 0]) < TOL && Math.abs(A[2 * 3 * 2 + 1 * 2 + 1]) < TOL); // A[2][1]
    }

    @Test
    void testEmpty() {
        // Test with zero-sized matrix
        double[] A = new double[0];
        double[] tau = new double[0];
        double[] work = new double[0];
        int lwork = 0;

        int info = Zgeqr.zgeqrf(0, 0, A, 0, 1, tau, 0, work, 0, lwork);

        assertEquals(0, info);
    }

    @Test
    void testInvalidInput() {
        // Test with invalid m
        double[] A = {1.0, 2.0, 3.0, 4.0};
        double[] tau = new double[2 * 2];
        double[] work = new double[2 * 2];
        int lwork = 2;

        int info = Zgeqr.zgeqrf(-1, 2, A, 0, 2, tau, 0, work, 0, lwork);
        assertEquals(-1, info);

        // Test with invalid n
        info = Zgeqr.zgeqrf(2, -1, A, 0, 2, tau, 0, work, 0, lwork);
        assertEquals(-2, info);

        // Test with invalid lda
        info = Zgeqr.zgeqrf(2, 2, A, 0, 1, tau, 0, work, 0, lwork);
        assertEquals(-5, info);

        // Test with invalid lwork
        info = Zgeqr.zgeqrf(2, 2, A, 0, 2, tau, 0, work, 0, -2);
        assertEquals(-10, info);
    }

    @Test
    void testWithSmallMatrix() {
        // Test a 1x1 matrix
        double[] A = {1.0, 2.0};
        double[] tau = new double[1 * 2];
        double[] work = new double[1 * 2];
        int lwork = 1;

        int info = Zgeqr.zgeqrf(1, 1, A, 0, 1, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        double expectedBeta = -Math.sqrt(1.0 * 1.0 + 2.0 * 2.0);
        assertEquals(expectedBeta, A[0], TOL);
        assertEquals(0.0, A[1], TOL);
        assertTrue(Math.abs(tau[0]) > TOL || Math.abs(tau[1]) > TOL);
    }

    @Test
    void testWithVector() {
        // Test a 3x1 vector
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        double[] tau = new double[1 * 2];
        double[] work = new double[1 * 2];
        int lwork = 1;

        int info = Zgeqr.zgeqrf(3, 1, A, 0, 1, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        // Check that tau is non-zero (since we're decomposing a non-zero vector)
        assertTrue(Math.abs(tau[0]) > TOL || Math.abs(tau[1]) > TOL);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectFactorization() {
        int m = 3;
        int n = 3;
        int lda = 5;
        int aOff = 6;
        int lwork = n;

        double[] direct = {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 19.0, 20.0
        };
        double[] padded = new double[(aOff + (m - 1) * lda + n + 1) * 2];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int src = (i * n + j) * 2;
                int dst = (aOff + i * lda + j) * 2;
                padded[dst] = direct[src];
                padded[dst + 1] = direct[src + 1];
            }
        }

        double[] tauDirect = new double[n * 2];
        double[] tauOffset = new double[n * 2];
        double[] workDirect = new double[n * 2];
        double[] workOffset = new double[n * 2];

        assertEquals(0, Zgeqr.zgeqrf(m, n, direct, 0, n, tauDirect, 0, workDirect, 0, lwork));
        assertEquals(0, Zgeqr.zgeqrf(m, n, padded, aOff, lda, tauOffset, 0, workOffset, 0, lwork));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int src = (i * n + j) * 2;
                int dst = (aOff + i * lda + j) * 2;
                assertEquals(direct[src], padded[dst], TOL);
                assertEquals(direct[src + 1], padded[dst + 1], TOL);
            }
        }

        for (int i = 0; i < n * 2; i++) {
            assertEquals(tauDirect[i], tauOffset[i], TOL);
        }
    }

}
