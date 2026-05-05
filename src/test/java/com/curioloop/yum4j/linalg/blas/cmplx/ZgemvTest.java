/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import java.util.Random;

class ZgemvTest {

    private static final double TOL = 1e-11;

    @Test
    void testNoTransBlockedOffsetStrideMatchesReference() {
        int m = 72;
        int n = 40;
        int lda = 44;
        int aStart = 12;
        int xStart = 6;
        int yStart = 10;
        int incX = 2;
        int incY = 3;

        double[] compactA = ZBlasTestSupport.randomComplexMatrix(m, n, 61L);
        double[] expectedA = new double[aStart + (m - 1) * lda * 2 + n * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, actualA, aStart, lda);

        Random random = new Random(67L);
        double[] expectedX = new double[xStart + (n - 1) * incX * 2 + 8];
        double[] actualX = expectedX.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedX, xStart, incX, n);
        System.arraycopy(expectedX, 0, actualX, 0, expectedX.length);

        double[] expectedY = new double[yStart + (m - 1) * incY * 2 + 8];
        double[] actualY = expectedY.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedY, yStart, incY, m);
        System.arraycopy(expectedY, 0, actualY, 0, expectedY.length);

        ZBlasTestSupport.refZgemv(BLAS.Trans.NoTrans, m, n, 0.75, -0.375,
            expectedA, aStart, lda, expectedX, xStart, incX, -0.25, 0.5, expectedY, yStart, incY);
        Zgemv.zgemv(BLAS.Trans.NoTrans, m, n, 0.75, -0.375,
            actualA, aStart, lda, actualX, xStart, incX, -0.25, 0.5, actualY, yStart, incY);

        ZBlasTestSupport.assertArrayClose("Zgemv notrans matrix unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv notrans x unchanged", expectedX, actualX, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv notrans blocked", expectedY, actualY, TOL);
    }

    @Test
    void testTransRecursiveOffsetStrideMatchesReference() {
        int m = 48;
        int n = 37;
        int lda = 41;
        int aStart = 14;
        int xStart = 8;
        int yStart = 4;
        int incX = 2;
        int incY = 2;

        double[] compactA = ZBlasTestSupport.randomComplexMatrix(m, n, 71L);
        double[] expectedA = new double[aStart + (m - 1) * lda * 2 + n * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, actualA, aStart, lda);

        Random random = new Random(73L);
        double[] expectedX = new double[xStart + (m - 1) * incX * 2 + 8];
        double[] actualX = expectedX.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedX, xStart, incX, m);
        System.arraycopy(expectedX, 0, actualX, 0, expectedX.length);

        double[] expectedY = new double[yStart + (n - 1) * incY * 2 + 8];
        double[] actualY = expectedY.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedY, yStart, incY, n);
        System.arraycopy(expectedY, 0, actualY, 0, expectedY.length);

        ZBlasTestSupport.refZgemv(BLAS.Trans.Trans, m, n, -0.5, 0.875,
            expectedA, aStart, lda, expectedX, xStart, incX, 0.375, -0.125, expectedY, yStart, incY);
        Zgemv.zgemv(BLAS.Trans.Trans, m, n, -0.5, 0.875,
            actualA, aStart, lda, actualX, xStart, incX, 0.375, -0.125, actualY, yStart, incY);

        ZBlasTestSupport.assertArrayClose("Zgemv trans matrix unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv trans x unchanged", expectedX, actualX, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv trans recursive", expectedY, actualY, TOL);
    }

    @Test
    void testConjTransRecursiveOffsetStrideMatchesReference() {
        int m = 45;
        int n = 35;
        int lda = 39;
        int aStart = 10;
        int xStart = 6;
        int yStart = 8;
        int incX = 3;
        int incY = 2;

        double[] compactA = ZBlasTestSupport.randomComplexMatrix(m, n, 79L);
        double[] expectedA = new double[aStart + (m - 1) * lda * 2 + n * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, actualA, aStart, lda);

        Random random = new Random(83L);
        double[] expectedX = new double[xStart + (m - 1) * incX * 2 + 8];
        double[] actualX = expectedX.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedX, xStart, incX, m);
        System.arraycopy(expectedX, 0, actualX, 0, expectedX.length);

        double[] expectedY = new double[yStart + (n - 1) * incY * 2 + 8];
        double[] actualY = expectedY.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedY, yStart, incY, n);
        System.arraycopy(expectedY, 0, actualY, 0, expectedY.length);

        ZBlasTestSupport.refZgemv(BLAS.Trans.Conj, m, n, 0.625, 0.25,
            expectedA, aStart, lda, expectedX, xStart, incX, -0.75, 0.5, expectedY, yStart, incY);
        Zgemv.zgemv(BLAS.Trans.Conj, m, n, 0.625, 0.25,
            actualA, aStart, lda, actualX, xStart, incX, -0.75, 0.5, actualY, yStart, incY);

        ZBlasTestSupport.assertArrayClose("Zgemv conj matrix unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv conj x unchanged", expectedX, actualX, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv conj recursive", expectedY, actualY, TOL);
    }

    @Test
    void testNoTransContiguousUnitStrideMatchesReference() {
        int m = 35;
        int n = 48;
        int lda = 52;
        int aStart = 10;
        int xStart = 6;
        int yStart = 8;

        double[] compactA = ZBlasTestSupport.randomComplexMatrix(m, n, 137L);
        double[] expectedA = new double[aStart + (m - 1) * lda * 2 + n * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, actualA, aStart, lda);

        Random random = new Random(139L);
        double[] expectedX = new double[xStart + n * 2 + 8];
        double[] actualX = expectedX.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedX, xStart, 1, n);
        System.arraycopy(expectedX, 0, actualX, 0, expectedX.length);

        double[] expectedY = new double[yStart + m * 2 + 8];
        double[] actualY = expectedY.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedY, yStart, 1, m);
        System.arraycopy(expectedY, 0, actualY, 0, expectedY.length);

        ZBlasTestSupport.refZgemv(BLAS.Trans.NoTrans, m, n, 0.625, -0.25,
            expectedA, aStart, lda, expectedX, xStart, 1, -0.5, 0.375, expectedY, yStart, 1);
        Zgemv.zgemv(BLAS.Trans.NoTrans, m, n, 0.625, -0.25,
            actualA, aStart, lda, actualX, xStart, 1, -0.5, 0.375, actualY, yStart, 1);

        ZBlasTestSupport.assertArrayClose("Zgemv notrans contiguous matrix unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv notrans contiguous x unchanged", expectedX, actualX, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv notrans contiguous unit stride", expectedY, actualY, TOL);
    }

    @Test
    void testNoTransContiguousXPairedFallbackMatchesReference() {
        int m = 65;
        int n = 32;
        int lda = 37;
        int aStart = 8;
        int yStart = 6;
        int incY = 3;

        double[] compactA = ZBlasTestSupport.randomComplexMatrix(m, n, 89L);
        double[] expectedA = new double[aStart + (m - 1) * lda * 2 + n * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, actualA, aStart, lda);

        double[] expectedX = ZBlasTestSupport.randomComplexVector(n, 97L);
        double[] actualX = expectedX.clone();
        double[] expectedY = new double[yStart + (m - 1) * incY * 2 + 8];
        double[] actualY = expectedY.clone();
        Random random = new Random(101L);
        ZBlasTestSupport.fillUsedComplexVector(random, expectedY, yStart, incY, m);
        System.arraycopy(expectedY, 0, actualY, 0, expectedY.length);

        ZBlasTestSupport.refZgemv(BLAS.Trans.NoTrans, m, n, -0.375, 0.625,
            expectedA, aStart, lda, expectedX, 0, 1, 0.5, -0.75, expectedY, yStart, incY);
        Zgemv.zgemv(BLAS.Trans.NoTrans, m, n, -0.375, 0.625,
            actualA, aStart, lda, actualX, 0, 1, 0.5, -0.75, actualY, yStart, incY);

        ZBlasTestSupport.assertArrayClose("Zgemv paired notrans matrix unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv paired notrans x unchanged", expectedX, actualX, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv paired notrans", expectedY, actualY, TOL);
    }

    @Test
    void testTransContiguousYPairedLeafMatchesReference() {
        int m = 31;
        int n = 48;
        int lda = 53;
        int aStart = 10;
        int xStart = 6;
        int incX = 2;

        double[] compactA = ZBlasTestSupport.randomComplexMatrix(m, n, 103L);
        double[] expectedA = new double[aStart + (m - 1) * lda * 2 + n * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, actualA, aStart, lda);

        Random random = new Random(107L);
        double[] expectedX = new double[xStart + (m - 1) * incX * 2 + 8];
        double[] actualX = expectedX.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedX, xStart, incX, m);
        System.arraycopy(expectedX, 0, actualX, 0, expectedX.length);

        double[] expectedY = ZBlasTestSupport.randomComplexVector(n, 109L);
        double[] actualY = expectedY.clone();

        ZBlasTestSupport.refZgemv(BLAS.Trans.Trans, m, n, 0.625, 0.25,
            expectedA, aStart, lda, expectedX, xStart, incX, -0.75, 0.5, expectedY, 0, 1);
        Zgemv.zgemv(BLAS.Trans.Trans, m, n, 0.625, 0.25,
            actualA, aStart, lda, actualX, xStart, incX, -0.75, 0.5, actualY, 0, 1);

        ZBlasTestSupport.assertArrayClose("Zgemv trans paired matrix unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv trans paired x unchanged", expectedX, actualX, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv trans paired leaf", expectedY, actualY, TOL);
    }

    @Test
    void testConjContiguousYPairedLeafMatchesReference() {
        int m = 29;
        int n = 32;
        int lda = 36;
        int aStart = 12;
        int xStart = 4;
        int incX = 3;

        double[] compactA = ZBlasTestSupport.randomComplexMatrix(m, n, 113L);
        double[] expectedA = new double[aStart + (m - 1) * lda * 2 + n * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, m, n, 0, n, actualA, aStart, lda);

        Random random = new Random(127L);
        double[] expectedX = new double[xStart + (m - 1) * incX * 2 + 8];
        double[] actualX = expectedX.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedX, xStart, incX, m);
        System.arraycopy(expectedX, 0, actualX, 0, expectedX.length);

        double[] expectedY = ZBlasTestSupport.randomComplexVector(n, 131L);
        double[] actualY = expectedY.clone();

        ZBlasTestSupport.refZgemv(BLAS.Trans.Conj, m, n, -0.375, 0.625,
            expectedA, aStart, lda, expectedX, xStart, incX, 0.5, -0.75, expectedY, 0, 1);
        Zgemv.zgemv(BLAS.Trans.Conj, m, n, -0.375, 0.625,
            actualA, aStart, lda, actualX, xStart, incX, 0.5, -0.75, actualY, 0, 1);

        ZBlasTestSupport.assertArrayClose("Zgemv conj paired matrix unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv conj paired x unchanged", expectedX, actualX, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemv conj paired leaf", expectedY, actualY, TOL);
    }
}