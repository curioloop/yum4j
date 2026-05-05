/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import java.util.Random;

class ZhemvTest {

    private static final double TOL = 1e-11;

    @Test
    void testUpperUnitStrideMatchesReference() {
        int n = 8;
        double[] a = ZBlasTestSupport.randomHermitianMatrix(n, 11L);
        double[] x = ZBlasTestSupport.randomComplexVector(n, 17L);
        double[] expected = ZBlasTestSupport.randomComplexVector(n, 23L);
        double[] actual = expected.clone();

        ZBlasTestSupport.refZhemv(BLAS.Uplo.Upper, n, 0.75, -0.25,
            a, 0, n, x, 0, 1, -0.5, 0.125, expected, 0, 1);
        Zhemv.zhemv(BLAS.Uplo.Upper, n, 0.75, -0.25,
            a, 0, n, x, 0, 1, -0.5, 0.125, actual, 0, 1);

        ZBlasTestSupport.assertArrayClose("Zhemv upper unit stride", expected, actual, TOL);
    }

    @Test
    void testLowerOffsetStrideMatchesReference() {
        int n = 7;
        int lda = 10;
        int aStart = 6;
        int xStart = 4;
        int yStart = 8;
        int incX = 2;
        int incY = 3;

        double[] compactA = ZBlasTestSupport.randomHermitianMatrix(n, 31L);
        double[] expectedA = new double[aStart + (n - 1) * lda * 2 + n * 2 + 4];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, n, n, 0, n, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, n, n, 0, n, actualA, aStart, lda);

        Random random = new Random(37L);
        double[] expectedX = new double[xStart + (n - 1) * incX * 2 + 6];
        double[] actualX = expectedX.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedX, xStart, incX, n);
        System.arraycopy(expectedX, 0, actualX, 0, expectedX.length);

        double[] expectedY = new double[yStart + (n - 1) * incY * 2 + 6];
        double[] actualY = expectedY.clone();
        ZBlasTestSupport.fillUsedComplexVector(random, expectedY, yStart, incY, n);
        System.arraycopy(expectedY, 0, actualY, 0, expectedY.length);

        ZBlasTestSupport.refZhemv(BLAS.Uplo.Lower, n, -0.375, 0.625,
            expectedA, aStart, lda, expectedX, xStart, incX, 0.5, -0.75, expectedY, yStart, incY);
        Zhemv.zhemv(BLAS.Uplo.Lower, n, -0.375, 0.625,
            actualA, aStart, lda, actualX, xStart, incX, 0.5, -0.75, actualY, yStart, incY);

        ZBlasTestSupport.assertArrayClose("Zhemv lower matrix unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zhemv lower x unchanged", expectedX, actualX, TOL);
        ZBlasTestSupport.assertArrayClose("Zhemv lower offset stride", expectedY, actualY, TOL);
    }
}