package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZorgbrTest {

    private static final double TOL = 1e-10;

    @Test
    void testOffsetSubmatrixMatchesDirectQGenerator() {
        int m = 4;
        int n = 3;
        int k = 3;
        int lda = 5;
        int aOff = 4;
        int minmn = Math.min(m, n);

        double[] factorized = {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        double[] d = new double[minmn];
        double[] e = new double[minmn];
        double[] tauq = new double[minmn * 2];
        double[] taup = new double[minmn * 2];
        double[] gebrdWork = new double[4096];

        assertEquals(0, Zgebrd.zgebrd(m, n, factorized, 0, n, d, 0, e, 0, tauq, 0, taup, 0, gebrdWork, 0, 4096));

        double[] directQ = factorized.clone();
        double[] offsetQ = new double[(aOff + (m - 1) * lda + n + 1) * 2];
        copyMatrixToOffset(factorized, m, n, n, offsetQ, aOff, lda);
        double[] directWork = new double[4096];
        double[] offsetWork = new double[4096];

        assertEquals(0, Zorgbr.zorgbr('Q', m, n, k, directQ, 0, n, tauq, 0, directWork, 0, 4096));
        assertEquals(0, Zorgbr.zorgbr('Q', m, n, k, offsetQ, aOff, lda, tauq, 0, offsetWork, 0, 4096));

        assertOffsetSubmatrixEquals(directQ, m, n, n, offsetQ, aOff, lda);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectPGenerator() {
        int n = 4;
        int lda = 6;
        int aOff = 5;

        double[] factorized = ZgesvdTest.GESVD_A_SQ.clone();
        double[] d = new double[n];
        double[] e = new double[n];
        double[] tauq = new double[n * 2];
        double[] taup = new double[n * 2];
        double[] gebrdWork = new double[4096];

        assertEquals(0, Zgebrd.zgebrd(n, n, factorized, 0, n, d, 0, e, 0, tauq, 0, taup, 0, gebrdWork, 0, 4096));

        double[] directP = new double[n * n * 2];
        Zlacpy.zlacpy('U', n, n, factorized, 0, n, directP, 0, n);

        double[] offsetP = new double[(aOff + (n - 1) * lda + n + 1) * 2];
        copyMatrixToOffset(directP, n, n, n, offsetP, aOff, lda);
        double[] directWork = new double[4096];
        double[] offsetWork = new double[4096];

        assertEquals(0, Zorgbr.zorgbr('P', n, n, n, directP, 0, n, taup, 0, directWork, 0, 4096));
        assertEquals(0, Zorgbr.zorgbr('P', n, n, n, offsetP, aOff, lda, taup, 0, offsetWork, 0, 4096));

        assertOffsetSubmatrixEquals(directP, n, n, n, offsetP, aOff, lda);
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