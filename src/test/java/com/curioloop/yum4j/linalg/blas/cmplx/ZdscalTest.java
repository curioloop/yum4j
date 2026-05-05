/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZdscalTest {

    private static final double TOL = 1e-14;

    @Test
    void testUnitStride() {
        double[] x = {1.0, 2.0, -3.0, 4.0, 5.0, -6.0};

        Zdscal.zdscal(3, 2.5, x, 0, 1);

        assertEquals(2.5, x[0], TOL);
        assertEquals(5.0, x[1], TOL);
        assertEquals(-7.5, x[2], TOL);
        assertEquals(10.0, x[3], TOL);
        assertEquals(12.5, x[4], TOL);
        assertEquals(-15.0, x[5], TOL);
    }

    @Test
    void testNonUnitStride() {
        double[] x = {1.0, 2.0, 0.0, 0.0, -3.0, 4.0, 0.0, 0.0, 5.0, -6.0};

        Zdscal.zdscal(3, -2.0, x, 0, 2);

        assertEquals(-2.0, x[0], TOL);
        assertEquals(-4.0, x[1], TOL);
        assertEquals(0.0, x[2], TOL);
        assertEquals(0.0, x[3], TOL);
        assertEquals(6.0, x[4], TOL);
        assertEquals(-8.0, x[5], TOL);
        assertEquals(0.0, x[6], TOL);
        assertEquals(0.0, x[7], TOL);
        assertEquals(-10.0, x[8], TOL);
        assertEquals(12.0, x[9], TOL);
    }

    @Test
    void testOffset() {
        double[] x = {9.0, 9.0, 1.0, -2.0, -3.0, 4.0, 7.0, 7.0};

        Zdscal.zdscal(2, 0.5, x, 2, 1);

        assertEquals(9.0, x[0], TOL);
        assertEquals(9.0, x[1], TOL);
        assertEquals(0.5, x[2], TOL);
        assertEquals(-1.0, x[3], TOL);
        assertEquals(-1.5, x[4], TOL);
        assertEquals(2.0, x[5], TOL);
        assertEquals(7.0, x[6], TOL);
        assertEquals(7.0, x[7], TOL);
    }

    @Test
    void testIdentityScaleNoOp() {
        double[] x = {1.0, -2.0, 3.0, -4.0};

        Zdscal.zdscal(2, 1.0, x, 0, 1);

        assertEquals(1.0, x[0], TOL);
        assertEquals(-2.0, x[1], TOL);
        assertEquals(3.0, x[2], TOL);
        assertEquals(-4.0, x[3], TOL);
    }

    @Test
    void testEmpty() {
        double[] x = {1.0, 2.0};

        Zdscal.zdscal(0, 3.0, x, 0, 1);

        assertEquals(1.0, x[0], TOL);
        assertEquals(2.0, x[1], TOL);
    }

    @Test
    void testLargeContiguousMatchesReference() {
        int n = 4096;
        double alpha = -0.875;
        double[] actual = ZBlasTestSupport.randomComplexVector(n, 2048L);
        double[] expected = Arrays.copyOf(actual, actual.length);

        zdscalReference(n, alpha, expected, 0, 1);
        Zdscal.zdscal(n, alpha, actual, 0, 1);

        ZBlasTestSupport.assertArrayClose("Zdscal large contiguous", expected, actual, TOL);
    }

    private void zdscalReference(int n, double alpha, double[] x, int xOff, int incX) {
        for (int i = 0; i < n; i++) {
            int index = xOff + i * incX * 2;
            x[index] *= alpha;
            x[index + 1] *= alpha;
        }
    }
}