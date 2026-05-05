package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZtrevcTest {

    private static final double TOL = 1e-10;

    @Test
    void testRawOffsetMatchesDirectBothEigenvectorSides() {
        int n = 4;
        double[] schur = ZgeesTest.GEES_A.clone();
        double[] w = new double[n * 2];
        double[] q = new double[n * n * 2];
        double[] rwork = new double[n];
        boolean[] bwork = new boolean[n];
        double[] query = new double[2];

        assertEquals(0, Zgees.zgees('V', 'N', null, n, schur, 0, n, w, 0, q, 0, n, query, 0, -1, rwork, bwork));
        int lwork = (int) query[0];
        double[] schurWork = new double[lwork];
        assertEquals(0, Zgees.zgees('V', 'N', null, n, schur, 0, n, w, 0, q, 0, n, schurWork, 0, lwork, rwork, bwork));

        double[] directT = schur.clone();
        double[] directVL = q.clone();
        double[] directVR = q.clone();
        double[] directWork = new double[n * 2];
        Ztrevc.ztrevc('B', 'B', n, directT, 0, n, directVL, 0, n, directVR, 0, n, n, directWork, 0);

        int ldt = 6;
        int ldvl = 6;
        int ldvr = 7;
        int tOff = 10;
        int vlOff = 8;
        int vrOff = 12;

        double[] offsetT = new double[tOff + ldt * n * 2];
        double[] offsetVL = new double[vlOff + ldvl * n * 2];
        double[] offsetVR = new double[vrOff + ldvr * n * 2];
        copyMatrixToRawOffset(directT, n, n, n, offsetT, tOff, ldt);
        copyMatrixToRawOffset(q, n, n, n, offsetVL, vlOff, ldvl);
        copyMatrixToRawOffset(q, n, n, n, offsetVR, vrOff, ldvr);

        double[] offsetWork = new double[n * 2];
        Ztrevc.ztrevc('B', 'B', n, offsetT, tOff, ldt, offsetVL, vlOff, ldvl, offsetVR, vrOff, ldvr, n, offsetWork, 0);

        assertRawSubmatrixEquals(directT, n, n, n, offsetT, tOff, ldt);
        assertRawSubmatrixEquals(directVL, n, n, n, offsetVL, vlOff, ldvl);
        assertRawSubmatrixEquals(directVR, n, n, n, offsetVR, vrOff, ldvr);
    }

    private static void copyMatrixToRawOffset(double[] src, int rows, int cols, int srcLda, double[] dst, int aOff, int dstLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int srcPos = (i * srcLda + j) * 2;
                int dstPos = aOff + (i * dstLda + j) * 2;
                dst[dstPos] = src[srcPos];
                dst[dstPos + 1] = src[srcPos + 1];
            }
        }
    }

    private static void assertRawSubmatrixEquals(double[] expected, int rows, int cols, int expectedLda, double[] actual, int aOff, int actualLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int expectedPos = (i * expectedLda + j) * 2;
                int actualPos = aOff + (i * actualLda + j) * 2;
                assertEquals(expected[expectedPos], actual[actualPos], TOL);
                assertEquals(expected[expectedPos + 1], actual[actualPos + 1], TOL);
            }
        }
    }
}