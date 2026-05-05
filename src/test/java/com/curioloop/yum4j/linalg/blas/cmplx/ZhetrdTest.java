package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZhetrdTest {

    private static final double TOL = 1e-10;

    private static final double[] HERMITIAN_A = {
            1.0, 0.0, 2.0, 1.0, -1.0, 2.0, 0.5, -0.5,
            2.0, -1.0, 3.0, 0.0, 1.5, 0.25, -2.0, 1.0,
            -1.0, -2.0, 1.5, -0.25, 4.0, 0.0, 0.75, 1.5,
            0.5, 0.5, -2.0, -1.0, 0.75, -1.5, 5.0, 0.0
    };

    @Test
    void testOffsetSubmatrixMatchesDirectReductionUpper() {
        assertOffsetReductionMatchesDirect(BLAS.Uplo.Upper);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectReductionLower() {
        assertOffsetReductionMatchesDirect(BLAS.Uplo.Lower);
    }

    private static void assertOffsetReductionMatchesDirect(BLAS.Uplo uplo) {
        int n = 4;
        int lda = 6;
        int aOff = 7;

        double[] direct = HERMITIAN_A.clone();
        double[] padded = new double[(aOff + (n - 1) * lda + n + 1) * 2];
        copyMatrix(direct, n, padded, aOff, lda);

        double[] dDirect = new double[n];
        double[] dOffset = new double[n];
        double[] eDirect = new double[n - 1];
        double[] eOffset = new double[n - 1];
        double[] tauDirect = new double[(n - 1) * 2];
        double[] tauOffset = new double[(n - 1) * 2];

        double[] query = new double[1];
        assertEquals(0, Zhetrd.zhetrd(uplo, n, direct.clone(), 0, n, dDirect.clone(), 0, eDirect.clone(), 0, tauDirect.clone(), 0, query, 0, -1));
        int lwork = (int) query[0];
        double[] workDirect = new double[lwork];
        double[] workOffset = new double[lwork];

        assertEquals(0, Zhetrd.zhetrd(uplo, n, direct, 0, n, dDirect, 0, eDirect, 0, tauDirect, 0, workDirect, 0, lwork));
        assertEquals(0, Zhetrd.zhetrd(uplo, n, padded, aOff, lda, dOffset, 0, eOffset, 0, tauOffset, 0, workOffset, 0, lwork));

        assertSubmatrixEquals(direct, n, padded, aOff, lda, n, TOL);
        assertVectorEquals(dDirect, dOffset, TOL);
        assertVectorEquals(eDirect, eOffset, TOL);
        assertVectorEquals(tauDirect, tauOffset, TOL);
    }

    private static void copyMatrix(double[] src, int n, double[] dst, int aOff, int lda) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int srcPos = (i * n + j) * 2;
                int dstPos = (aOff + i * lda + j) * 2;
                dst[dstPos] = src[srcPos];
                dst[dstPos + 1] = src[srcPos + 1];
            }
        }
    }

    private static void assertSubmatrixEquals(double[] expected, int expectedLda, double[] actual, int aOff, int actualLda, int n, double tol) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int expectedPos = (i * expectedLda + j) * 2;
                int actualPos = (aOff + i * actualLda + j) * 2;
                assertEquals(expected[expectedPos], actual[actualPos], tol);
                assertEquals(expected[expectedPos + 1], actual[actualPos + 1], tol);
            }
        }
    }

    private static void assertVectorEquals(double[] expected, double[] actual, double tol) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tol);
        }
    }
}