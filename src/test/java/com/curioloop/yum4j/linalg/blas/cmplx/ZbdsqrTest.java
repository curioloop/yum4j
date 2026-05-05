/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZbdsqrTest {

    private static final double TOL = 1e-10;

    private static int workSize(int n) {
        return 4 * n + 16;
    }

    @Test
    void testBasicUpperBidiagonal() {
        int n = 2;
        double[] d = {2.0, 3.0};
        double[] e = {1.0};
        double[] work = new double[workSize(n)];

        int info = Zbdsqr.zbdsqr('U', n, 0, 0, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);

        assertEquals(0, info);
        assertTrue(d[0] > 0 && d[1] > 0);
        assertTrue(d[0] >= d[1]);
    }

    @Test
    void testBasicLowerBidiagonal() {
        int n = 2;
        double[] d = {2.0, 3.0};
        double[] e = {1.0};
        double[] work = new double[workSize(n)];

        int info = Zbdsqr.zbdsqr('L', n, 0, 0, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);

        assertEquals(0, info);
        assertTrue(d[0] > 0 && d[1] > 0);
        assertTrue(d[0] >= d[1]);
    }

    @Test
    void testOnlySingularValues() {
        int n = 2;
        double[] d = {2.0, 3.0};
        double[] e = {1.0};
        double[] work = new double[workSize(n)];

        int info = Zbdsqr.zbdsqr('U', n, 0, 0, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);

        assertEquals(0, info);
        assertTrue(d[0] > 0 && d[1] > 0);
        assertTrue(d[0] >= d[1]);
    }

    @Test
    void testWithRightHandSide() {
        int n = 2;
        double[] d = {2.0, 3.0};
        double[] e = {1.0};
        double[] c = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0};
        double[] work = new double[workSize(n)];

        int info = Zbdsqr.zbdsqr('U', n, 0, 0, n, d, 0, e, 0,
                null, 0, 1, null, 0, 1, c, 0, n, work, 0);

        assertEquals(0, info);
        assertTrue(d[0] > 0 && d[1] > 0);
        for (int i = 0; i < c.length; i++) {
            assertTrue(Double.isFinite(c[i]));
        }
    }

    @Test
    void testZeroMatrix() {
        int n = 2;
        double[] d = {0.0, 0.0};
        double[] e = {0.0};
        double[] work = new double[workSize(n)];

        int info = Zbdsqr.zbdsqr('U', n, 0, 0, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);

        assertEquals(0, info);
        assertTrue(d[0] < TOL && d[1] < TOL);
    }

    @Test
    void testIdentityMatrix() {
        int n = 2;
        double[] d = {1.0, 1.0};
        double[] e = {0.0};
        double[] work = new double[workSize(n)];

        int info = Zbdsqr.zbdsqr('U', n, 0, 0, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);

        assertEquals(0, info);
        assertEquals(1.0, d[0], TOL);
        assertEquals(1.0, d[1], TOL);
    }

    @Test
    void testLargerMatrix() {
        int n = 3;
        double[] d = {4.0, 5.0, 6.0};
        double[] e = {1.0, 2.0};
        double[] work = new double[workSize(n)];

        int info = Zbdsqr.zbdsqr('U', n, 0, 0, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);

        assertEquals(0, info);
        assertTrue(d[0] > 0 && d[1] > 0 && d[2] > 0);
        assertTrue(d[0] >= d[1] && d[1] >= d[2]);
    }

    @Test
    void testWithVTandU() {
        int n = 2;
        double[] d = {2.0, 3.0};
        double[] e = {1.0};
        double[] vt = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        double[] u = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        double[] work = new double[workSize(n)];

        int info = Zbdsqr.zbdsqr('U', n, n, n, 0, d, 0, e, 0,
                vt, 0, n, u, 0, n, null, 0, 1, work, 0);

        assertEquals(0, info);
        assertTrue(d[0] > 0 && d[1] > 0);
        assertTrue(d[0] >= d[1]);
        checkUnitary(vt, n);
        checkUnitary(u, n);
    }

        @Test
        void testRawOffsetMatricesMatchDirectOutputs() {
        int n = 3;
        int ldvt = 5;
        int ldu = 4;
        int ldc = 5;
        int vtOff = 8;
        int uOff = 10;
        int cOff = 12;

        double[] directD = {4.0, 2.5, 1.25};
        double[] directE = {0.75, 0.5};
        double[] offsetD = directD.clone();
        double[] offsetE = directE.clone();
        double[] directVt = identityMatrix(n);
        double[] directU = identityMatrix(n);
        double[] directC = {
            1.0, 0.0, 2.0, -1.0, 3.0, 0.5,
            4.0, 1.0, 5.0, 0.0, 6.0, -0.5,
            7.0, -1.0, 8.0, 0.5, 9.0, 0.0
        };

        double[] offsetVt = new double[vtOff + ldvt * n * 2];
        double[] offsetU = new double[uOff + ldu * n * 2];
        double[] offsetC = new double[cOff + ldc * n * 2];
        copyMatrixToRawOffset(directVt, n, n, n, offsetVt, vtOff, ldvt);
        copyMatrixToRawOffset(directU, n, n, n, offsetU, uOff, ldu);
        copyMatrixToRawOffset(directC, n, n, n, offsetC, cOff, ldc);

        double[] directWork = new double[workSize(n)];
        double[] offsetWork = new double[workSize(n)];

        int info = Zbdsqr.zbdsqr('U', n, n, n, n, directD, 0, directE, 0,
            directVt, 0, n, directU, 0, n, directC, 0, n, directWork, 0);
        assertEquals(0, info);

        info = Zbdsqr.zbdsqr('U', n, n, n, n, offsetD, 0, offsetE, 0,
            offsetVt, vtOff, ldvt, offsetU, uOff, ldu, offsetC, cOff, ldc, offsetWork, 0);
        assertEquals(0, info);

        assertRealVectorEquals(directD, offsetD);
        assertRealVectorEquals(directE, offsetE);
        assertRawSubmatrixEquals(directVt, n, n, n, offsetVt, vtOff, ldvt);
        assertRawSubmatrixEquals(directU, n, n, n, offsetU, uOff, ldu);
        assertRawSubmatrixEquals(directC, n, n, n, offsetC, cOff, ldc);
        }

    @Test
    void testEmpty() {
        double[] d = new double[0];
        double[] e = new double[0];
        double[] work = new double[0];

        int info = Zbdsqr.zbdsqr('U', 0, 0, 0, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);

        assertEquals(0, info);
    }

    @Test
    void testInvalidInput() {
        double[] d = {2.0, 3.0};
        double[] e = {1.0};
        double[] work = new double[workSize(2)];

        int info = Zbdsqr.zbdsqr('X', 2, 0, 0, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);
        assertEquals(-1, info);

        info = Zbdsqr.zbdsqr('U', -1, 0, 0, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);
        assertEquals(-2, info);

        info = Zbdsqr.zbdsqr('U', 2, -1, 0, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);
        assertEquals(-3, info);

        info = Zbdsqr.zbdsqr('U', 2, 0, -1, 0, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);
        assertEquals(-4, info);

        info = Zbdsqr.zbdsqr('U', 2, 0, 0, -1, d, 0, e, 0,
                null, 0, 1, null, 0, 1, null, 0, 1, work, 0);
        assertEquals(-5, info);
    }

    private static double[] identityMatrix(int n) {
        double[] identity = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            identity[(i * n + i) * 2] = 1.0;
        }
        return identity;
    }

    private static void copyMatrixToRawOffset(double[] src, int rows, int cols, int srcLda, double[] dst, int rawOff, int dstLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int srcPos = (i * srcLda + j) * 2;
                int dstPos = rawOff + (i * dstLda + j) * 2;
                dst[dstPos] = src[srcPos];
                dst[dstPos + 1] = src[srcPos + 1];
            }
        }
    }

    private static void assertRawSubmatrixEquals(double[] expected, int rows, int cols, int expectedLda, double[] actual, int rawOff, int actualLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int expectedPos = (i * expectedLda + j) * 2;
                int actualPos = rawOff + (i * actualLda + j) * 2;
                assertEquals(expected[expectedPos], actual[actualPos], TOL);
                assertEquals(expected[expectedPos + 1], actual[actualPos + 1], TOL);
            }
        }
    }

    private static void assertRealVectorEquals(double[] expected, double[] actual) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], TOL);
        }
    }

    private void checkUnitary(double[] matrix, int n) {
        double[] product = new double[n * n * 2];
        Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                matrix, 0, n, matrix, 0, n, 0.0, 0.0, product, 0, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int index = i * n * 2 + j * 2;
                if (i == j) {
                    assertEquals(1.0, product[index], TOL);
                    assertEquals(0.0, product[index + 1], TOL);
                } else {
                    assertEquals(0.0, product[index], TOL);
                    assertEquals(0.0, product[index + 1], TOL);
                }
            }
        }
    }
}
