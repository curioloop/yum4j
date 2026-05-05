/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import java.util.Random;

class ZsymvTest {

    private static final double TOL = 1e-11;

    @Test
    void testUpperUnitStrideMatchesReference() {
        int n = 8;
        double[] a = ZBlasTestSupport.randomSymmetricMatrix(n, 41L);
        double[] x = ZBlasTestSupport.randomComplexVector(n, 43L);
        double[] expected = ZBlasTestSupport.randomComplexVector(n, 47L);
        double[] actual = expected.clone();

        ZBlasTestSupport.refZsymv(BLAS.Uplo.Upper, n, 0.625, 0.375,
            a, 0, n, x, 0, 1, -0.25, 0.5, expected, 0, 1);
        Zsymv.zsymv(BLAS.Uplo.Upper, n, 0.625, 0.375,
            a, 0, n, x, 0, 1, -0.25, 0.5, actual, 0, 1);

        ZBlasTestSupport.assertArrayClose("Zsymv upper unit stride", expected, actual, TOL);
    }

    @Test
    void testLowerOffsetStrideMatchesReference() {
        int n = 7;
        int lda = 11;
        int aStart = 10;
        int xStart = 6;
        int yStart = 12;
        int incX = 2;
        int incY = 2;

        double[] compactA = ZBlasTestSupport.randomSymmetricMatrix(n, 53L);
        double[] expectedA = new double[aStart + (n - 1) * lda * 2 + n * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, n, n, 0, n, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, n, n, 0, n, actualA, aStart, lda);

        Random random = new Random(59L);
        double[] expectedX = new double[xStart + (n - 1) * incX * 2 + 6];
        double[] actualX = expectedX.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedX, xStart, incX, n);
        System.arraycopy(expectedX, 0, actualX, 0, expectedX.length);

        double[] expectedY = new double[yStart + (n - 1) * incY * 2 + 6];
        double[] actualY = expectedY.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedY, yStart, incY, n);
        System.arraycopy(expectedY, 0, actualY, 0, expectedY.length);

        ZBlasTestSupport.refZsymv(BLAS.Uplo.Lower, n, -0.5, 0.875,
            expectedA, aStart, lda, expectedX, xStart, incX, 0.75, -0.25, expectedY, yStart, incY);
        Zsymv.zsymv(BLAS.Uplo.Lower, n, -0.5, 0.875,
            actualA, aStart, lda, actualX, xStart, incX, 0.75, -0.25, actualY, yStart, incY);

        ZBlasTestSupport.assertArrayClose("Zsymv lower matrix unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zsymv lower x unchanged", expectedX, actualX, TOL);
        ZBlasTestSupport.assertArrayClose("Zsymv lower offset stride", expectedY, actualY, TOL);
    }
}