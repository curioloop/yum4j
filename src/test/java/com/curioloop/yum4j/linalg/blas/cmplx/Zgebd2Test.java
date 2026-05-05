/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Zgebd2Test {

    private static final double TOL = 1e-10;

    @Test
    void testBasicSquareMatrix() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tauQ = new double[3 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[3 * 2];

        int info = Zgebrd.zgebd2(3, 3, A, 0, 3, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);

        for (int i = 0; i < 3; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
        for (int i = 0; i < 2; i++) {
            assertTrue(Double.isFinite(e[i]), "e[" + i + "] should be finite");
        }
        for (int i = 0; i < 3; i++) {
            assertTrue(Double.isFinite(tauQ[i * 2]), "tauQ[" + i + "] re should be finite");
            assertTrue(Double.isFinite(tauP[i * 2]), "tauP[" + i + "] re should be finite");
        }
    }

    @Test
    void testRectangularMatrixMoreRows() {
        int m = 4, n = 3;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            A[i * 2] = i + 1;
            A[i * 2 + 1] = i + 2;
        }
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tauQ = new double[3 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[4 * 2];

        int info = Zgebrd.zgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);

        for (int i = 0; i < 3; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }

    @Test
    void testRectangularMatrixMoreColumns() {
        int m = 3, n = 4;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            A[i * 2] = i + 1;
            A[i * 2 + 1] = i + 2;
        }
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tauQ = new double[3 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[4 * 2];

        int info = Zgebrd.zgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);

        for (int i = 0; i < 3; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }

    @Test
    void testZeroMatrix() {
        double[] A = new double[3 * 3 * 2];
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tauQ = new double[3 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[3 * 2];

        int info = Zgebrd.zgebd2(3, 3, A, 0, 3, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);

        for (int i = 0; i < 3; i++) {
            assertEquals(0.0, d[i], TOL);
        }
    }

    @Test
    void testIdentityMatrix() {
        int n = 3;
        double[] A = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            A[(i * n + i) * 2] = 1.0;
        }
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tauQ = new double[3 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[3 * 2];

        int info = Zgebrd.zgebd2(3, 3, A, 0, 3, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);

        for (int i = 0; i < 3; i++) {
            assertEquals(1.0, Math.abs(d[i]), TOL, "d[" + i + "] should be 1 or -1");
        }
        for (int i = 0; i < 2; i++) {
            assertEquals(0.0, e[i], TOL, "e[" + i + "] should be 0");
        }
    }

    @Test
    void testEmpty() {
        double[] A = new double[0];
        double[] d = new double[0];
        double[] e = new double[0];
        double[] tauQ = new double[0];
        double[] tauP = new double[0];
        double[] work = new double[0];

        int info = Zgebrd.zgebd2(0, 0, A, 0, 1, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);
    }

    @Test
    void testInvalidInput() {
        double[] A = {1.0, 2.0, 3.0, 4.0};
        double[] d = new double[2];
        double[] e = new double[1];
        double[] tauQ = new double[2 * 2];
        double[] tauP = new double[2 * 2];
        double[] work = new double[2 * 2];

        int info = Zgebrd.zgebd2(-1, 2, A, 0, 2, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);
        assertEquals(-1, info);

        info = Zgebrd.zgebd2(2, -1, A, 0, 2, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);
        assertEquals(-2, info);

        info = Zgebrd.zgebd2(2, 2, A, 0, 1, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);
        assertEquals(-4, info);
    }

    @Test
    void testWithSmallMatrix() {
        double[] A = {1.0, 2.0};
        double[] d = new double[1];
        double[] e = new double[0];
        double[] tauQ = new double[1 * 2];
        double[] tauP = new double[1 * 2];
        double[] work = new double[1 * 2];

        int info = Zgebrd.zgebd2(1, 1, A, 0, 1, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);

        double expectedBeta = -Math.sqrt(1.0 * 1.0 + 2.0 * 2.0);
        assertEquals(expectedBeta, d[0], TOL);
    }

    @Test
    void testWithVector() {
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        double[] d = new double[1];
        double[] e = new double[0];
        double[] tauQ = new double[1 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[1 * 2];

        int info = Zgebrd.zgebd2(1, 3, A, 0, 3, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);

        assertTrue(Double.isFinite(d[0]), "d[0] should be finite");
        assertEquals(0.0, tauQ[0], TOL, "tauQ should be 0 for m=1");
        assertEquals(0.0, tauQ[1], TOL);
    }

    @Test
    void testRandomMatrix() {
        int m = 6, n = 6;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            A[i * 2] = Math.random() * 10 - 5;
            A[i * 2 + 1] = Math.random() * 10 - 5;
        }
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn * 2];
        double[] tauP = new double[minmn * 2];
        double[] work = new double[Math.max(m, n) * 2];

        int info = Zgebrd.zgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);

        for (int i = 0; i < minmn; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
        for (int i = 0; i < minmn - 1; i++) {
            assertTrue(Double.isFinite(e[i]), "e[" + i + "] should be finite");
        }
    }

    @Test
    void testTallMatrix() {
        int m = 5, n = 3;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            A[i * 2] = Math.random() * 10 - 5;
            A[i * 2 + 1] = Math.random() * 10 - 5;
        }
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn * 2];
        double[] tauP = new double[minmn * 2];
        double[] work = new double[Math.max(m, n) * 2];

        int info = Zgebrd.zgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);

        for (int i = 0; i < minmn; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }

    @Test
    void testWideMatrix() {
        int m = 3, n = 5;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            A[i * 2] = Math.random() * 10 - 5;
            A[i * 2 + 1] = Math.random() * 10 - 5;
        }
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn * 2];
        double[] tauP = new double[minmn * 2];
        double[] work = new double[Math.max(m, n) * 2];

        int info = Zgebrd.zgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(0, info);

        for (int i = 0; i < minmn; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }

    @Test
    void testOffsetSubmatrixMatchesDirectReductionTall() {
        assertOffsetReductionMatchesDirect(4, 3, 5, 12);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectReductionWide() {
        assertOffsetReductionMatchesDirect(3, 4, 6, 10);
    }

    private static void assertOffsetReductionMatchesDirect(int m, int n, int lda, int aOff) {
        int minmn = Math.min(m, n);
        double[] direct = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            direct[i * 2] = i + 1;
            direct[i * 2 + 1] = i + 2;
        }

        double[] padded = new double[aOff + m * lda * 2];
        copyMatrixToRawOffset(direct, m, n, n, padded, aOff, lda);

        double[] dDirect = new double[minmn];
        double[] dOffset = new double[minmn];
        double[] eDirect = new double[Math.max(0, minmn - 1)];
        double[] eOffset = new double[Math.max(0, minmn - 1)];
        double[] tauQDirect = new double[minmn * 2];
        double[] tauQOffset = new double[minmn * 2];
        double[] tauPDirect = new double[minmn * 2];
        double[] tauPOffset = new double[minmn * 2];
        double[] workDirect = new double[Math.max(m, n) * 2];
        double[] workOffset = new double[Math.max(m, n) * 2];

        assertEquals(0, Zgebrd.zgebd2(m, n, direct, 0, n, dDirect, 0, eDirect, 0, tauQDirect, 0, tauPDirect, 0, workDirect, 0));
        assertEquals(0, Zgebrd.zgebd2(m, n, padded, aOff, lda, dOffset, 0, eOffset, 0, tauQOffset, 0, tauPOffset, 0, workOffset, 0));

        assertRawSubmatrixEquals(direct, m, n, n, padded, aOff, lda);
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
