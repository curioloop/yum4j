/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import java.util.Random;

class ZtrmvTest {

    private static final double TOL = 1e-11;

    @Test
    void testUpperNoTransOffsetStrideMatchesReference() {
        assertMatchesReference(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
            9, 13, 6, 2, 101L, 103L);
    }

    @Test
    void testLowerTransUnitStrideMatchesReference() {
        assertMatchesReference(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit,
            10, 0, 4, 1, 107L, 109L);
    }

    @Test
    void testUpperConjTransOffsetStrideMatchesReference() {
        assertMatchesReference(BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
            11, 15, 8, 3, 113L, 127L);
    }

    @Test
    void testUpperNoTransAdjacentColumnViewMatchesReference() {
        int panel = 9;
        int lda = panel + 2;
        int vectorCol = 6;
        int n = vectorCol;

        double[] compactA = ZBlasTestSupport.randomTriangularMatrix(panel, BLAS.Uplo.Upper, false, 131L);
        double[] actualA = new double[(panel - 1) * lda * 2 + panel * 2 + 8];
        ZBlasTestSupport.embedMatrix(compactA, panel, panel, 0, panel, actualA, 0, lda);

        double[] expectedA = actualA.clone();
        double[] expectedX = extractColumn(expectedA, 0, lda, vectorCol, 0, n);

        ZBlasTestSupport.refZtrmv(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
            n, expectedA, 0, lda, expectedX, 0, 1);
        Ztrmv.ztrmv(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
            n, actualA, 0, lda, actualA, vectorCol * 2, lda);

        storeColumn(expectedA, 0, lda, vectorCol, 0, expectedX);
        ZBlasTestSupport.assertArrayClose("Ztrmv upper adjacent column view", expectedA, actualA, TOL);
    }

    @Test
    void testLowerNoTransAdjacentColumnViewMatchesReference() {
        int panel = 10;
        int lda = panel + 3;
        int j = 2;
        int n = panel - j - 1;

        double[] compactA = ZBlasTestSupport.randomTriangularMatrix(panel, BLAS.Uplo.Lower, false, 137L);
        double[] actualA = new double[(panel - 1) * lda * 2 + panel * 2 + 8];
        ZBlasTestSupport.embedMatrix(compactA, panel, panel, 0, panel, actualA, 0, lda);

        double[] expectedA = actualA.clone();
        double[] expectedX = extractColumn(expectedA, 0, lda, j, j + 1, n);

        int aStart = ((j + 1) * lda + j + 1) * 2;
        int xStart = ((j + 1) * lda + j) * 2;
        ZBlasTestSupport.refZtrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
            n, expectedA, aStart, lda, expectedX, 0, 1);
        Ztrmv.ztrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
            n, actualA, aStart, lda, actualA, xStart, lda);

        storeColumn(expectedA, 0, lda, j, j + 1, expectedX);
        ZBlasTestSupport.assertArrayClose("Ztrmv lower adjacent column view", expectedA, actualA, TOL);
    }

    private static void assertMatchesReference(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                                               int n, int aStart, int xStart, int incX,
                                               long matrixSeed, long vectorSeed) {
        int lda = n + 3;
        double[] compactA = ZBlasTestSupport.randomTriangularMatrix(n, uplo, diag == BLAS.Diag.Unit, matrixSeed);
        double[] expectedA = new double[aStart + (n - 1) * lda * 2 + n * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, n, n, 0, n, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, n, n, 0, n, actualA, aStart, lda);

        Random random = new Random(vectorSeed);
        double[] expectedX = new double[xStart + (n - 1) * incX * 2 + 8];
        double[] actualX = expectedX.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedX, xStart, incX, n);
        System.arraycopy(expectedX, 0, actualX, 0, expectedX.length);

        ZBlasTestSupport.refZtrmv(uplo, trans, diag, n, expectedA, aStart, lda, expectedX, xStart, incX);
        Ztrmv.ztrmv(uplo, trans, diag, n, actualA, aStart, lda, actualX, xStart, incX);

        ZBlasTestSupport.assertArrayClose("Ztrmv matrix unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Ztrmv result", expectedX, actualX, TOL);
    }

    private static double[] extractColumn(double[] matrix, int aStart, int lda, int column, int rowStart, int length) {
        double[] vector = new double[length * 2];
        for (int i = 0; i < length; i++) {
            int src = aStart + ((rowStart + i) * lda + column) * 2;
            vector[i * 2] = matrix[src];
            vector[i * 2 + 1] = matrix[src + 1];
        }
        return vector;
    }

    private static void storeColumn(double[] matrix, int aStart, int lda, int column, int rowStart, double[] vector) {
        int length = vector.length / 2;
        for (int i = 0; i < length; i++) {
            int dst = aStart + ((rowStart + i) * lda + column) * 2;
            matrix[dst] = vector[i * 2];
            matrix[dst + 1] = vector[i * 2 + 1];
        }
    }
}