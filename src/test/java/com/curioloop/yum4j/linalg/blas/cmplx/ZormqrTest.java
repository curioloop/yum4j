/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class ZormqrTest {

    private static final double TOL = 1e-10;

    @Test
    void testLeftNoTransSquare() {
        // Test Q * C where Q is from QR factorization of square matrix
        int m = 3;
        int n = 3;
        int k = 3;
        int lda = 3;
        int ldc = 3;
        
        // First, perform QR factorization on A
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Create matrix C
        double[] C = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] work2 = new double[n * 2];
        int lwork2 = n;

        // Apply Q to C from the left with no transpose
        int info = Zormqr.zormqr('L', 'N', m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that C has been modified
        assertFalse(Math.abs(C[0] - 1.0) < TOL);
    }

    @Test
    void testLeftConjTransSquare() {
        // Test Q^H * C where Q is from QR factorization of square matrix
        int m = 3;
        int n = 3;
        int k = 3;
        int lda = 3;
        int ldc = 3;
        
        // First, perform QR factorization on A
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Create matrix C
        double[] C = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] work2 = new double[m * 2];
        int lwork2 = m;

        // Apply Q^H to C from the left
        int info = Zormqr.zormqr('L', 'C', m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that C has been modified
        assertFalse(Math.abs(C[0] - 1.0) < TOL);
    }

    @Test
    void testRightNoTransSquare() {
        // Test C * Q where Q is from QR factorization of square matrix
        int m = 3;
        int n = 3;
        int k = 3;
        int lda = 3;
        int ldc = 3;
        
        // First, perform QR factorization on A
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Create matrix C
        double[] C = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] work2 = new double[n * 2];
        int lwork2 = n;

        // Apply Q to C from the right with no transpose
        int info = Zormqr.zormqr('R', 'N', m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that C has been modified
        assertFalse(Math.abs(C[0] - 1.0) < TOL);
    }

    @Test
    void testRightConjTransSquare() {
        // Test C * Q^H where Q is from QR factorization of square matrix
        int m = 3;
        int n = 3;
        int k = 3;
        int lda = 3;
        int ldc = 3;
        
        // First, perform QR factorization on A
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Create matrix C
        double[] C = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] work2 = new double[n * 2];
        int lwork2 = n;

        // Apply Q^H to C from the right
        int info = Zormqr.zormqr('R', 'C', m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that C has been modified
        assertFalse(Math.abs(C[0] - 1.0) < TOL);
    }

    @Test
    void testRectangularMoreRows() {
        // Test with rectangular matrix (more rows than columns)
        int m = 4;
        int n = 3;
        int k = 3;
        int lda = 3;
        int ldc = 3;
        
        // First, perform QR factorization on A
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Create matrix C
        double[] C = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        double[] work2 = new double[m * 2];
        int lwork2 = m;

        // Apply Q to C from the left with no transpose
        int info = Zormqr.zormqr('L', 'N', m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that C has been modified
        assertFalse(Math.abs(C[0] - 1.0) < TOL);
    }

    @Test
    void testRectangularMoreColumns() {
        // Test with rectangular matrix (more columns than rows)
        int m = 3;
        int n = 4;
        int k = 3;
        int lda = 4;
        int ldc = 4;
        
        // First, perform QR factorization on A
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Create matrix C
        double[] C = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        double[] work2 = new double[n * 2];
        int lwork2 = n;

        // Apply Q to C from the left with no transpose
        int info = Zormqr.zormqr('L', 'N', m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that C has been modified
        assertFalse(Math.abs(C[0] - 1.0) < TOL);
    }

    @Test
    void testKLessThanMinMN() {
        // Test with k less than min(m, n)
        int m = 3;
        int n = 3;
        int k = 2;
        int lda = 3;
        int ldc = 3;
        
        // First, perform QR factorization on A
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] tau = new double[Math.min(m, n) * 2];
        double[] work = new double[Math.max(1, n - 1) * 2];
        int lwork = Math.max(1, n - 1);

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Create matrix C
        double[] C = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] work2 = new double[m * 2];
        int lwork2 = m;

        // Apply Q to C from the left with no transpose
        int info = Zormqr.zormqr('L', 'N', m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that C has been modified
        assertFalse(Math.abs(C[0] - 1.0) < TOL);
    }

    @Test
    void testWorkspaceQuery() {
        // Test workspace query mode
        int m = 2;
        int n = 2;
        int k = 2;
        int lda = 2;
        int ldc = 2;
        
        double[] A = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        };
        double[] tau = new double[k * 2];
        double[] C = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        };
        double[] work = new double[1];
        int lwork = -1;

        int info = Zormqr.zormqr('L', 'N', m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work, 0, lwork);

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
        int ldc = 3;
        
        // First, perform QR factorization on zero matrix
        double[] A = new double[m * n * 2]; // All zeros
        double[] tau = new double[k * 2];
        double[] work = new double[k * 2];
        int lwork = k;

        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Create matrix C
        double[] C = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] work2 = new double[m * 2];
        int lwork2 = m;

        // Apply Q to C from the left with no transpose
        int info = Zormqr.zormqr('L', 'N', m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that C is unchanged (Q should be identity matrix)
        assertEquals(1.0, C[0], TOL);
        assertEquals(2.0, C[1], TOL);
        assertEquals(3.0, C[2], TOL);
        assertEquals(4.0, C[3], TOL);
    }

    @Test
    void testEmptyMatrix() {
        // Test with zero-sized matrix
        double[] A = new double[0];
        double[] tau = new double[0];
        double[] C = new double[0];
        double[] work = new double[0];
        int lwork = 0;

        int info = Zormqr.zormqr('L', 'N', 0, 0, 0, A, 0, 1, tau, 0, C, 0, 1, work, 0, lwork);

        assertEquals(0, info);
    }

    @Test
    void testSmallMatrix() {
        // Test a 1x1 matrix
        int m = 1;
        int n = 1;
        int k = 1;
        int lda = 1;
        int ldc = 1;
        
        double[] A = {1.0, 2.0};
        double[] tau = new double[k * 2];
        double[] C = {3.0, 4.0};
        double[] work = new double[k * 2];
        int lwork = k;

        // First, perform QR factorization
        Zgeqr.zgeqrf(m, n, A, 0, lda, tau, 0, work, 0, lwork);

        // Apply Q to C from the left
        double[] work2 = new double[m * 2];
        int lwork2 = m;

        int info = Zormqr.zormqr('L', 'N', m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work2, 0, lwork2);

        assertEquals(0, info);

        // Check that C has been modified (Q is unitary, so C should be unchanged in magnitude)
        double normBefore = Math.sqrt(3.0 * 3.0 + 4.0 * 4.0);
        double normAfter = Math.sqrt(C[0] * C[0] + C[1] * C[1]);
        assertEquals(normBefore, normAfter, TOL);
    }

    @Test
    void testInvalidInputs() {
        // Test with invalid side
        double[] A = {1.0, 2.0, 3.0, 4.0};
        double[] tau = new double[2 * 2];
        double[] C = {1.0, 2.0, 3.0, 4.0};
        double[] work = new double[2 * 2];
        int lwork = 2;

        int info = Zormqr.zormqr('X', 'N', 2, 2, 2, A, 0, 2, tau, 0, C, 0, 2, work, 0, lwork);
        assertEquals(-1, info);

        // Test with invalid trans
        info = Zormqr.zormqr('L', 'X', 2, 2, 2, A, 0, 2, tau, 0, C, 0, 2, work, 0, lwork);
        assertEquals(-2, info);

        // Test with invalid m
        info = Zormqr.zormqr('L', 'N', -1, 2, 2, A, 0, 2, tau, 0, C, 0, 2, work, 0, lwork);
        assertEquals(-3, info);

        // Test with invalid n
        info = Zormqr.zormqr('L', 'N', 2, -1, 2, A, 0, 2, tau, 0, C, 0, 2, work, 0, lwork);
        assertEquals(-4, info);

        // Test with invalid k
        info = Zormqr.zormqr('L', 'N', 2, 2, -1, A, 0, 2, tau, 0, C, 0, 2, work, 0, lwork);
        assertEquals(-5, info);

        // Test with k > min(m, n)
        info = Zormqr.zormqr('L', 'N', 2, 2, 3, A, 0, 2, tau, 0, C, 0, 2, work, 0, lwork);
        assertEquals(-5, info);

        // Test with invalid lda
        info = Zormqr.zormqr('L', 'N', 2, 2, 2, A, 0, 1, tau, 0, C, 0, 2, work, 0, lwork);
        assertEquals(-7, info);

        // Test with invalid ldc
        info = Zormqr.zormqr('L', 'N', 2, 2, 2, A, 0, 2, tau, 0, C, 0, 1, work, 0, lwork);
        assertEquals(-10, info);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectApplication() {
        int m = 4;
        int n = 3;
        int k = 3;
        int lda = 5;
        int ldc = 6;
        int aOff = 5;
        int cOff = 7;

        double[] factorized = {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        double[] tau = new double[k * 2];
        double[] qrWork = new double[k * 2];
        assertEquals(0, Zgeqr.zgeqrf(m, n, factorized, 0, n, tau, 0, qrWork, 0, k));

        double[] directA = factorized.clone();
        double[] offsetA = new double[(aOff + (m - 1) * lda + n + 1) * 2];
        copyMatrixToOffset(factorized, m, n, n, offsetA, aOff, lda);
        double[] tauOffset = tau.clone();

        double[] directC = {
                2.0, 1.0, 4.0, 3.0, 6.0, 5.0,
                8.0, 7.0, 10.0, 9.0, 12.0, 11.0,
                14.0, 13.0, 16.0, 15.0, 18.0, 17.0,
                20.0, 19.0, 22.0, 21.0, 24.0, 23.0
        };
        double[] offsetC = new double[(cOff + (m - 1) * ldc + n + 1) * 2];
        copyMatrixToOffset(directC, m, n, n, offsetC, cOff, ldc);

        double[] directWork = new double[n * 2];
        double[] offsetWork = new double[n * 2];

        assertEquals(0, Zormqr.zormqr('L', 'N', m, n, k, directA, 0, n, tau, 0, directC, 0, n, directWork, 0, n));
        assertEquals(0, Zormqr.zormqr('L', 'N', m, n, k, offsetA, aOff, lda, tauOffset, 0, offsetC, cOff, ldc, offsetWork, 0, n));

        assertOffsetSubmatrixEquals(directC, m, n, n, offsetC, cOff, ldc);
    }

    @Test
    void testLeftModesMatchExplicitReferenceApplication() {
        assertMatchesExplicitApplication('L', 'N', 5, 3, 3, 20260425L);
        assertMatchesExplicitApplication('L', 'C', 5, 3, 3, 20260426L);
    }

    @Test
    void testRightModesMatchExplicitReferenceApplication() {
        assertMatchesExplicitApplication('R', 'N', 3, 5, 4, 20260427L);
        assertMatchesExplicitApplication('R', 'C', 3, 5, 4, 20260428L);
    }

    @Test
    void testLargeLeftWorkspaceQueryProvidesExecutableSize() {
        assertWorkspaceQueryExecutes('L', 'N', 96, 48, 80, 20260510L);
        assertWorkspaceQueryExecutes('L', 'C', 96, 48, 80, 20260511L);
    }

    @Test
    void testLargeRightWorkspaceQueryProvidesExecutableSize() {
        assertWorkspaceQueryExecutes('R', 'N', 48, 96, 80, 20260512L);
        assertWorkspaceQueryExecutes('R', 'C', 48, 96, 80, 20260513L);
    }

    private static void copyMatrixToOffset(double[] src, int rows, int cols, int srcLda, double[] dst, int aOff, int dstLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int srcPos = (i * srcLda + j) * 2;
                int dstPos = (aOff + i * dstLda + j) * 2;
                dst[dstPos] = src[srcPos];
                dst[dstPos + 1] = src[srcPos + 1];
            }
        }
    }

    private static void assertOffsetSubmatrixEquals(double[] expected, int rows, int cols, int expectedLda, double[] actual, int aOff, int actualLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int expectedPos = (i * expectedLda + j) * 2;
                int actualPos = (aOff + i * actualLda + j) * 2;
                assertEquals(expected[expectedPos], actual[actualPos], TOL);
                assertEquals(expected[expectedPos + 1], actual[actualPos + 1], TOL);
            }
        }
    }

    private static void assertMatchesExplicitApplication(char side, char trans, int m, int n, int k, long seed) {
        boolean left = side == 'L';
        int qRows = left ? m : n;
        int qCols = k;
        int lda = qCols;
        int ldc = n;

        Random random = new Random(seed);
        double[] factorizedA = new double[qRows * qCols * 2];
        for (int i = 0; i < factorizedA.length; i++) {
            factorizedA[i] = random.nextDouble() - 0.5;
        }
        double[] tau = new double[k * 2];
        double[] factorWork = new double[Math.max(1, qCols) * 2];
        assertEquals(0, Zgeqr.zgeqrf(qRows, qCols, factorizedA, 0, lda, tau, 0, factorWork, 0, Math.max(1, qCols)));

        double[] baseC = new double[m * n * 2];
        for (int i = 0; i < baseC.length; i++) {
            baseC[i] = random.nextDouble() - 0.5;
        }

        double[] expected = baseC.clone();
        double[] actual = baseC.clone();
        applyReferenceUnm2r(side, trans, m, n, k, factorizedA.clone(), lda, tau.clone(), expected, ldc);

        double[] query = new double[1];
        assertEquals(0, Zormqr.zormqr(side, trans, m, n, k, factorizedA.clone(), 0, lda, tau.clone(), 0, actual.clone(), 0, ldc, query, 0, -1));
        int lwork = Math.max(left ? n : m, (int) Math.ceil(query[0]));
        double[] work = new double[lwork * 2];
        int info = Zormqr.zormqr(side, trans, m, n, k, factorizedA.clone(), 0, lda, tau.clone(), 0, actual, 0, ldc, work, 0, lwork);
        assertEquals(0, info);
        ZBlasTestSupport.assertArrayClose("zormqr explicit " + side + trans, expected, actual, 1e-10);
    }

    private static void applyReferenceUnm2r(char side, char trans, int m, int n, int k,
                                            double[] a, int lda, double[] tau,
                                            double[] c, int ldc) {
        boolean left = side == 'L';
        boolean notrans = trans == 'N';
        boolean forward = (left && !notrans) || (!left && notrans);
        int start = forward ? 0 : k - 1;
        int end = forward ? k : -1;
        int step = forward ? 1 : -1;
        double[] tauValue = new double[2];
        double[] work = new double[(left ? n : m) * 2];

        for (int i = start; i != end; i += step) {
            int aiiPos = (i * lda + i) * 2;
            double savedRe = a[aiiPos];
            double savedIm = a[aiiPos + 1];
            a[aiiPos] = 1.0;
            a[aiiPos + 1] = 0.0;

            tauValue[0] = tau[i * 2];
            tauValue[1] = notrans ? tau[i * 2 + 1] : -tau[i * 2 + 1];

            if (left) {
                Zlarf.zlarf(BLAS.Side.Left, m - i, n, a, aiiPos, lda, tauValue, 0, c, i * ldc * 2, ldc, work, 0);
            } else {
                Zlarf.zlarf(BLAS.Side.Right, m, n - i, a, aiiPos, lda, tauValue, 0, c, i * 2, ldc, work, 0);
            }

            a[aiiPos] = savedRe;
            a[aiiPos + 1] = savedIm;
        }
    }

    private static void assertWorkspaceQueryExecutes(char side, char trans, int m, int n, int k, long seed) {
        boolean left = side == 'L';
        int qRows = left ? m : n;
        int qCols = k;
        int lda = qCols;
        int ldc = n;

        Random random = new Random(seed);
        double[] factorizedA = new double[qRows * qCols * 2];
        for (int i = 0; i < factorizedA.length; i++) {
            factorizedA[i] = random.nextDouble() - 0.5;
        }
        double[] tau = new double[k * 2];
        double[] factorWork = new double[Math.max(1, qCols) * 2];
        assertEquals(0, Zgeqr.zgeqrf(qRows, qCols, factorizedA, 0, lda, tau, 0, factorWork, 0, Math.max(1, qCols)));

        double[] c = new double[m * n * 2];
        for (int i = 0; i < c.length; i++) {
            c[i] = random.nextDouble() - 0.5;
        }

        double[] query = new double[1];
        assertEquals(0, Zormqr.zormqr(side, trans, m, n, k, factorizedA.clone(), 0, lda, tau.clone(), 0, c.clone(), 0, ldc, query, 0, -1));
        int lwork = Math.max(1, (int) Math.ceil(query[0]));
        double[] work = new double[lwork * 2];
        assertEquals(0, Zormqr.zormqr(side, trans, m, n, k, factorizedA.clone(), 0, lda, tau.clone(), 0, c.clone(), 0, ldc, work, 0, lwork));
    }

}
