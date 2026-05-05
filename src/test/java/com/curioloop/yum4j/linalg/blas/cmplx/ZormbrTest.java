package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZormbrTest {

    private static final double TOL = 1e-10;

    @Test
    void testOffsetSubmatrixMatchesDirectQApplication() {
        int rowsA = 4;
        int colsA = 4;
        int m = 4;
        int n = 2;
        int k = 4;
        int lda = 6;
        int directLdc = 4;
        int ldc = 5;
        int aOff = 4;
        int cOff = 6;
        int minmn = Math.min(rowsA, colsA);

        double[] factorized = ZgesvdTest.GESVD_A_SQ.clone();
        double[] d = new double[minmn];
        double[] e = new double[minmn];
        double[] tauq = new double[minmn * 2];
        double[] taup = new double[minmn * 2];
        double[] gebrdWork = new double[4096];

        assertEquals(0, Zgebrd.zgebrd(rowsA, colsA, factorized, 0, colsA, d, 0, e, 0, tauq, 0, taup, 0, gebrdWork, 0, 4096));

        double[] directA = factorized.clone();
        double[] offsetA = new double[(aOff + (rowsA - 1) * lda + colsA + 1) * 2];
        copyMatrixToOffset(factorized, rowsA, colsA, colsA, offsetA, aOff, lda);

        double[] directBaseC = {
                2.0, 1.0, 4.0, 3.0,
                6.0, 5.0, 8.0, 7.0,
                10.0, 9.0, 12.0, 11.0,
                14.0, 13.0, 16.0, 15.0
        };
        double[] directC = new double[m * directLdc * 2];
        copyMatrixToOffset(directBaseC, m, n, n, directC, 0, directLdc);
        double[] offsetC = new double[(cOff + (m - 1) * ldc + n + 1) * 2];
        copyMatrixToOffset(directBaseC, m, n, n, offsetC, cOff, ldc);

        double[] directWork = new double[256];
        double[] offsetWork = new double[256];

        assertEquals(0, Zormbr.zormbr('Q', 'L', 'N', m, n, k, directA, 0, colsA, tauq, 0, directC, 0, directLdc, directWork, 0, 256));
        assertEquals(0, Zormbr.zormbr('Q', 'L', 'N', m, n, k, offsetA, aOff, lda, tauq, 0, offsetC, cOff, ldc, offsetWork, 0, 256));

        assertOffsetSubmatrixEquals(directC, m, n, directLdc, offsetC, cOff, ldc);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectPApplication() {
        int rowsA = 3;
        int colsA = 4;
        int m = 3;
        int n = 4;
        int k = 3;
        int lda = 6;
        int ldc = 6;
        int aOff = 5;
        int cOff = 7;
        int minmn = Math.min(rowsA, colsA);

        double[] factorized = {
                1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0,
                5.0, -5.0, 6.0, -6.0, 7.0, -7.0, 8.0, -8.0,
                9.0, -9.0, 10.0, -10.0, 11.0, -11.0, 12.0, -12.0
        };
        double[] d = new double[minmn];
        double[] e = new double[minmn];
        double[] tauq = new double[minmn * 2];
        double[] taup = new double[minmn * 2];
        double[] gebrdWork = new double[4096];

        assertEquals(0, Zgebrd.zgebrd(rowsA, colsA, factorized, 0, colsA, d, 0, e, 0, tauq, 0, taup, 0, gebrdWork, 0, 4096));

        double[] directA = factorized.clone();
        double[] offsetA = new double[(aOff + (rowsA - 1) * lda + colsA + 1) * 2];
        copyMatrixToOffset(factorized, rowsA, colsA, colsA, offsetA, aOff, lda);

        double[] directC = {
                2.0, 0.0, 4.0, 1.0, 6.0, 2.0, 8.0, 3.0,
                10.0, 4.0, 12.0, 5.0, 14.0, 6.0, 16.0, 7.0,
                18.0, 8.0, 20.0, 9.0, 22.0, 10.0, 24.0, 11.0
        };
        double[] offsetC = new double[(cOff + (m - 1) * ldc + n + 1) * 2];
        copyMatrixToOffset(directC, m, n, n, offsetC, cOff, ldc);

        double[] directWork = new double[256];
        double[] offsetWork = new double[256];

        assertEquals(0, Zormbr.zormbr('P', 'R', 'N', m, n, k, directA, 0, colsA, taup, 0, directC, 0, n, directWork, 0, 256));
        assertEquals(0, Zormbr.zormbr('P', 'R', 'N', m, n, k, offsetA, aOff, lda, taup, 0, offsetC, cOff, ldc, offsetWork, 0, 256));

        assertOffsetSubmatrixEquals(directC, m, n, n, offsetC, cOff, ldc);
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
}