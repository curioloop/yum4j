package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZorgtrTest {

    private static final double TOL = 1e-10;

    private static final double[] HERMITIAN_A = {
            1.0, 0.0, 2.0, 1.0, -1.0, 2.0, 0.5, -0.5,
            2.0, -1.0, 3.0, 0.0, 1.5, 0.25, -2.0, 1.0,
            -1.0, -2.0, 1.5, -0.25, 4.0, 0.0, 0.75, 1.5,
            0.5, 0.5, -2.0, -1.0, 0.75, -1.5, 5.0, 0.0
    };

    @Test
    void testOffsetSubmatrixMatchesDirectQGenerationUpper() {
        assertOffsetQMatchesDirect(BLAS.Uplo.Upper);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectQGenerationLower() {
        assertOffsetQMatchesDirect(BLAS.Uplo.Lower);
    }

    private static void assertOffsetQMatchesDirect(BLAS.Uplo uplo) {
        int n = 4;
        int lda = 6;
        int aOff = 7;

        double[] factorized = HERMITIAN_A.clone();
        double[] tau = new double[(n - 1) * 2];
        double[] query = new double[1];
        assertEquals(0, Zhetrd.zhetrd(uplo, n, factorized.clone(), 0, n, new double[n], 0, new double[n - 1], 0, tau.clone(), 0, query, 0, -1));
        int trdLwork = (int) query[0];
        assertEquals(0, Zhetrd.zhetrd(uplo, n, factorized, 0, n, new double[n], 0, new double[n - 1], 0, tau, 0, new double[trdLwork], 0, trdLwork));

        double[] padded = new double[(aOff + (n - 1) * lda + n + 1) * 2];
        copyMatrix(factorized, n, padded, aOff, lda);
        double[] tauOffset = tau.clone();

        query[0] = 0.0;
        assertEquals(0, Zorgtr.zorgtr(uplo, n, factorized.clone(), 0, n, tau.clone(), 0, query, 0, -1));
        int lwork = (int) query[0];
        double[] workDirect = new double[lwork];
        double[] workOffset = new double[lwork];

        assertEquals(0, Zorgtr.zorgtr(uplo, n, factorized, 0, n, tau, 0, workDirect, 0, lwork));
        assertEquals(0, Zorgtr.zorgtr(uplo, n, padded, aOff, lda, tauOffset, 0, workOffset, 0, lwork));

        assertSubmatrixEquals(factorized, n, padded, aOff, lda, n, TOL);
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
}