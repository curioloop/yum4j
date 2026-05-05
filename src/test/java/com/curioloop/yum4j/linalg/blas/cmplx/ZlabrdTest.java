/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZlabrdTest {

    private static final double TOL = 1e-10;

    @Test
    void testSquareMatrix() {
        // Test a 3x3 complex matrix
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        int m = 3;
        int n = 3;
        int k = 3;
        int lda = 3;

        // D and E are REAL arrays (not complex), so size is k and k-1
        double[] d = new double[k];
        double[] e = new double[k - 1];
        double[] tauQ = new double[k * 2];
        double[] tauP = new double[k * 2];

        double[] X = new double[m * k * 2];
        double[] Y = new double[(k + 1) * n * 2];
        int ldx = k;
        int ldy = n;

        Zlabrd.zlabrd(m, n, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);

        // Check that d and e arrays are populated (D and E are REAL, not complex)
        for (int i = 0; i < k; i++) {
            assertTrue(Math.abs(d[i]) > TOL, "d[" + i + "] is zero");
        }
        for (int i = 0; i < k - 1; i++) {
            assertTrue(Math.abs(e[i]) > TOL, "e[" + i + "] is zero");
        }
    }

    @Test
    void testRectangularMatrixMoreRows() {
        // Test a 4x3 complex matrix (more rows than columns)
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        int m = 4;
        int n = 3;
        int k = 3;
        int lda = 3;

        // D and E are REAL arrays
        double[] d = new double[k];
        double[] e = new double[k - 1];
        double[] tauQ = new double[k * 2];
        double[] tauP = new double[k * 2];

        double[] X = new double[m * k * 2];
        double[] Y = new double[(k + 1) * n * 2];
        int ldx = k;
        int ldy = n;

        Zlabrd.zlabrd(m, n, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);

        // Check that d and e arrays are populated (D and E are REAL, not complex)
        for (int i = 0; i < k; i++) {
            assertTrue(Math.abs(d[i]) > TOL, "d[" + i + "] is zero");
        }
        for (int i = 0; i < k - 1; i++) {
            assertTrue(Math.abs(e[i]) > TOL, "e[" + i + "] is zero");
        }
    }

    @Test
    void testRectangularMatrixMoreColumns() {
        int m = 3;
        int n = 4;
        int k = 3;
        int lda = 6;
        int ldx = k;
        int ldy = n;
        int aOff = 10;
        int xOff = 6;
        int yOff = 12;

        double[] directA = {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        double[] offsetA = new double[aOff + m * lda * 2];
        copyMatrixToRawOffset(directA, m, n, n, offsetA, aOff, lda);

        double[] dDirect = new double[k];
        double[] dOffset = new double[k];
        double[] eDirect = new double[k - 1];
        double[] eOffset = new double[k - 1];
        double[] tauQDirect = new double[k * 2];
        double[] tauQOffset = new double[k * 2];
        double[] tauPDirect = new double[k * 2];
        double[] tauPOffset = new double[k * 2];
        double[] xDirect = new double[m * ldx * 2];
        double[] xOffset = new double[xOff + m * ldx * 2];
        double[] yDirect = new double[(k + 1) * ldy * 2];
        double[] yOffset = new double[yOff + (k + 1) * ldy * 2];

        Zlabrd.zlabrd(m, n, k, directA, 0, n, dDirect, 0, eDirect, 0, tauQDirect, 0, tauPDirect, 0, xDirect, 0, ldx, yDirect, 0, ldy);
        Zlabrd.zlabrd(m, n, k, offsetA, aOff, lda, dOffset, 0, eOffset, 0, tauQOffset, 0, tauPOffset, 0, xOffset, xOff, ldx, yOffset, yOff, ldy);

        assertRawSubmatrixEquals(directA, m, n, n, offsetA, aOff, lda);
        assertRawSubmatrixEquals(xDirect, m, ldx, ldx, xOffset, xOff, ldx);
        assertRawSubmatrixEquals(yDirect, k + 1, ldy, ldy, yOffset, yOff, ldy);
        assertVectorEquals(dDirect, dOffset);
        assertVectorEquals(eDirect, eOffset);
        assertVectorEquals(tauQDirect, tauQOffset);
        assertVectorEquals(tauPDirect, tauPOffset);
    }

    @Test
    void testKLessThanMinMN() {
        // Test with k less than min(m, n)
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        int m = 3;
        int n = 3;
        int k = 2;
        int lda = 3;

        // D and E are REAL arrays
        double[] d = new double[k];
        double[] e = new double[k - 1];
        double[] tauQ = new double[k * 2];
        double[] tauP = new double[k * 2];

        double[] X = new double[m * k * 2];
        double[] Y = new double[(k + 1) * n * 2];
        int ldx = k;
        int ldy = n;

        Zlabrd.zlabrd(m, n, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);

        // Check that d and e arrays are populated (D and E are REAL, not complex)
        for (int i = 0; i < k; i++) {
            assertTrue(Math.abs(d[i]) > TOL, "d[" + i + "] is zero");
        }
        if (k > 1) {
            for (int i = 0; i < k - 1; i++) {
                assertTrue(Math.abs(e[i]) > TOL, "e[" + i + "] is zero");
            }
        }
    }

    @Test
    void testZeroMatrix() {
        // Test with zero matrix
        int m = 3;
        int n = 3;
        int k = 3;
        int lda = 3;

        double[] A = new double[m * n * 2]; // All zeros
        // D and E are REAL arrays
        double[] d = new double[k];
        double[] e = new double[k - 1];
        double[] tauQ = new double[k * 2];
        double[] tauP = new double[k * 2];

        double[] X = new double[m * k * 2];
        double[] Y = new double[(k + 1) * n * 2];
        int ldx = k;
        int ldy = n;

        Zlabrd.zlabrd(m, n, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);

        // Check that d and e are zero (D and E are REAL, not complex)
        for (int i = 0; i < k; i++) {
            assertEquals(0.0, d[i], TOL, "d[" + i + "] should be zero");
        }
        for (int i = 0; i < k - 1; i++) {
            assertEquals(0.0, e[i], TOL, "e[" + i + "] should be zero");
        }
    }

    @Test
    void testIdentityMatrix() {
        // Test with identity matrix
        int m = 3;
        int n = 3;
        int k = 3;
        int lda = 3;

        double[] A = new double[m * n * 2];
        for (int i = 0; i < m; i++) {
            A[i * lda * 2 + i * 2] = 1.0; // Real part of diagonal
        }

        // D and E are REAL arrays
        double[] d = new double[k];
        double[] e = new double[k - 1];
        double[] tauQ = new double[k * 2];
        double[] tauP = new double[k * 2];

        double[] X = new double[m * k * 2];
        double[] Y = new double[(k + 1) * n * 2];
        int ldx = k;
        int ldy = n;

        Zlabrd.zlabrd(m, n, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);

        // Check that d is close to 1 or -1 (since Zlarfg can reflect to negative axis) and e is close to 0
        // D and E are REAL, not complex
        for (int i = 0; i < k; i++) {
            assertTrue(Math.abs(Math.abs(d[i]) - 1.0) < TOL, "d[" + i + "] should be 1 or -1, got " + d[i]);
        }
        for (int i = 0; i < k - 1; i++) {
            assertEquals(0.0, e[i], TOL, "e[" + i + "] should be 0, got " + e[i]);
        }
    }

    @Test
    void testEmptyMatrix() {
        // Test with zero-sized matrix
        int m = 0;
        int n = 0;
        int k = 0;
        int lda = 1;
        double[] A = new double[0];
        double[] d = new double[0];
        double[] e = new double[0];
        double[] tauQ = new double[0];
        double[] tauP = new double[0];
        double[] X = new double[0];
        double[] Y = new double[0];
        int ldx = 1;
        int ldy = 1;

        Zlabrd.zlabrd(m, n, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);
    }

    @Test
    void testSmallMatrix() {
        // Test a 1x1 matrix
        double[] A = {1.0, 2.0};
        int m = 1;
        int n = 1;
        int k = 1;
        int lda = 1;

        // D and E are REAL arrays (k=1 means e has 0 elements)
        double[] d = new double[k];
        double[] e = new double[0]; // Empty for k=1
        double[] tauQ = new double[k * 2];
        double[] tauP = new double[k * 2];

        double[] X = new double[m * k * 2];
        double[] Y = new double[(k + 1) * n * 2];
        int ldx = k;
        int ldy = n;

        Zlabrd.zlabrd(m, n, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);

        // Check that d is populated (D is REAL, not complex)
        assertTrue(Math.abs(d[0]) > TOL, "d[0] is zero");
    }

    @Test
    void testInvalidInputs() {
        // Test with invalid m
        int m = -1;
        int n = 2;
        int k = 2;
        int lda = 2;
        double[] A = {1.0, 2.0, 3.0, 4.0};
        double[] d = new double[10];
        double[] e = new double[10];
        double[] tauQ = new double[10 * 2];
        double[] tauP = new double[10 * 2];
        double[] X = new double[10]; // Dummy size for invalid input
        double[] Y = new double[10]; // Dummy size for invalid input
        int ldx = 1;
        int ldy = 1;

        Zlabrd.zlabrd(m, n, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);

        // Test with invalid n
        m = 2;
        n = -1;
        Zlabrd.zlabrd(m, n, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);

        // Test with k > min(m, n)
        k = 3;
        m = 2;
        n = 2;
        lda = 2;
        A = new double[m * n * 2]; // Ensure A is large enough for m x n matrix
        X = new double[m * k * 2];
        Y = new double[(k + 1) * n * 2];
        ldx = k;
        ldy = n;
        Zlabrd.zlabrd(m, n, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);

        // Test with invalid lda
        k = 1;
        lda = 1;
        A = new double[1 * 1 * 2]; // 1x1 matrix for lda=1
        Zlabrd.zlabrd(1, 1, k, A, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, X, 0, ldx, Y, 0, ldy);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectReductionTall() {
        int m = 4;
        int n = 3;
        int k = 2;
        int lda = 5;
        int ldx = k;
        int ldy = n;
        int aOff = 12;
        int xOff = 8;
        int yOff = 10;

        double[] directA = {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        double[] offsetA = new double[aOff + m * lda * 2];
        copyMatrixToRawOffset(directA, m, n, n, offsetA, aOff, lda);

        double[] dDirect = new double[k];
        double[] dOffset = new double[k];
        double[] eDirect = new double[k - 1];
        double[] eOffset = new double[k - 1];
        double[] tauQDirect = new double[k * 2];
        double[] tauQOffset = new double[k * 2];
        double[] tauPDirect = new double[k * 2];
        double[] tauPOffset = new double[k * 2];
        double[] xDirect = new double[m * ldx * 2];
        double[] xOffset = new double[xOff + m * ldx * 2];
        double[] yDirect = new double[ldy * (k + 1) * 2];
        double[] yOffset = new double[yOff + ldy * (k + 1) * 2];

        Zlabrd.zlabrd(m, n, k, directA, 0, n, dDirect, 0, eDirect, 0, tauQDirect, 0, tauPDirect, 0, xDirect, 0, ldx, yDirect, 0, ldy);
        Zlabrd.zlabrd(m, n, k, offsetA, aOff, lda, dOffset, 0, eOffset, 0, tauQOffset, 0, tauPOffset, 0, xOffset, xOff, ldx, yOffset, yOff, ldy);

        assertRawSubmatrixEquals(directA, m, n, n, offsetA, aOff, lda);
        assertRawSubmatrixEquals(xDirect, m, ldx, ldx, xOffset, xOff, ldx);
        assertRawSubmatrixEquals(yDirect, k + 1, ldy, ldy, yOffset, yOff, ldy);
        assertVectorEquals(dDirect, dOffset);
        assertVectorEquals(eDirect, eOffset);
        assertVectorEquals(tauQDirect, tauQOffset);
        assertVectorEquals(tauPDirect, tauPOffset);
    }

    private static void copyMatrixToRawOffset(double[] src, int m, int n, int srcLda, double[] dst, int aOff, int dstLda) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int srcPos = (i * srcLda + j) * 2;
                int dstPos = aOff + (i * dstLda + j) * 2;
                dst[dstPos] = src[srcPos];
                dst[dstPos + 1] = src[srcPos + 1];
            }
        }
    }

    private static void assertRawSubmatrixEquals(double[] expected, int m, int n, int expectedLda, double[] actual, int aOff, int actualLda) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int expectedPos = (i * expectedLda + j) * 2;
                int actualPos = aOff + (i * actualLda + j) * 2;
                assertEquals(expected[expectedPos], actual[actualPos], TOL);
                assertEquals(expected[expectedPos + 1], actual[actualPos + 1], TOL);
            }
        }
    }

    private static void assertVectorEquals(double[] expected, double[] actual) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], TOL);
        }
    }
}