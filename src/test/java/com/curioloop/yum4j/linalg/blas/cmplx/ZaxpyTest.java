/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZaxpyTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        // x = [1+2i, 3+4i, 5+6i]
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        // y = [7+8i, 9+10i, 11+12i]
        double[] y = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

        // alpha = 2.0 + 3.0i
        Zaxpy.zaxpy(3, 2.0, 3.0, x, 0, 1, y, 0, 1);

        // Expected: y = y + alpha * x
        // alpha * x[0] = (2+3i)(1+2i) = 2*1 - 3*2 + (2*2 + 3*1)i = -4 + 7i
        // y[0] = (7+8i) + (-4+7i) = 3 + 15i
        assertEquals(3.0, y[0], TOL);
        assertEquals(15.0, y[1], TOL);

        // alpha * x[1] = (2+3i)(3+4i) = 2*3 - 3*4 + (2*4 + 3*3)i = -6 + 17i
        // y[1] = (9+10i) + (-6+17i) = 3 + 27i
        assertEquals(3.0, y[2], TOL);
        assertEquals(27.0, y[3], TOL);

        // alpha * x[2] = (2+3i)(5+6i) = 2*5 - 3*6 + (2*6 + 3*5)i = -8 + 27i
        // y[2] = (11+12i) + (-8+27i) = 3 + 39i
        assertEquals(3.0, y[4], TOL);
        assertEquals(39.0, y[5], TOL);
    }

    @Test
    void testZeroAlpha() {
        // x = [1+2i, 3+4i]
        double[] x = {1.0, 2.0, 3.0, 4.0};
        // y = [5+6i, 7+8i]
        double[] y = {5.0, 6.0, 7.0, 8.0};

        // alpha = 0.0 + 0.0i
        Zaxpy.zaxpy(2, 0.0, 0.0, x, 0, 1, y, 0, 1);

        // y should remain unchanged
        assertEquals(5.0, y[0], TOL);
        assertEquals(6.0, y[1], TOL);
        assertEquals(7.0, y[2], TOL);
        assertEquals(8.0, y[3], TOL);
    }

    @Test
    void testSingleElement() {
        // x = [2+3i]
        double[] x = {2.0, 3.0};
        // y = [4+5i]
        double[] y = {4.0, 5.0};

        // alpha = 1.0 + 1.0i
        Zaxpy.zaxpy(1, 1.0, 1.0, x, 0, 1, y, 0, 1);

        // Expected: y = (4+5i) + (1+i)(2+3i) = (4+5i) + (-1+5i) = 3 + 10i
        assertEquals(3.0, y[0], TOL);
        assertEquals(10.0, y[1], TOL);
    }

    @Test
    void testEmpty() {
        Zaxpy.zaxpy(0, 2.0, 3.0, new double[0], 0, 1, new double[0], 0, 1);
    }

    @Test
    void testNonUnitStride() {
        // x = [1+2i, 0+0i, 3+4i, 0+0i, 5+6i]
        double[] x = {1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0};
        // y = [7+8i, 0+0i, 9+10i, 0+0i, 11+12i]
        double[] y = {7.0, 8.0, 0.0, 0.0, 9.0, 10.0, 0.0, 0.0, 11.0, 12.0};

        // alpha = 1.0 + 0.0i, stride = 2 (skip every other complex element)
        Zaxpy.zaxpy(3, 1.0, 0.0, x, 0, 2, y, 0, 2);

        // Expected: y = y + x (since alpha=1)
        assertEquals(8.0, y[0], TOL);  // 7+1
        assertEquals(10.0, y[1], TOL); // 8+2
        assertEquals(12.0, y[4], TOL); // 9+3
        assertEquals(14.0, y[5], TOL); // 10+4
        assertEquals(16.0, y[8], TOL); // 11+5
        assertEquals(18.0, y[9], TOL); // 12+6
    }

    @Test
    void testLargeContiguousMatchesReference() {
        int n = 4096;
        double alphaRe = -0.375;
        double alphaIm = 0.625;
        double[] x = ZBlasTestSupport.randomComplexVector(n, 4096L);
        double[] actual = ZBlasTestSupport.randomComplexVector(n, 6144L);
        double[] expected = Arrays.copyOf(actual, actual.length);

        zaxpyReference(n, alphaRe, alphaIm, x, 0, 1, expected, 0, 1);
        Zaxpy.zaxpy(n, alphaRe, alphaIm, x, 0, 1, actual, 0, 1);

        ZBlasTestSupport.assertArrayClose("Zaxpy large contiguous", expected, actual, TOL);
    }

    private void zaxpyReference(int n, double alphaRe, double alphaIm,
                                double[] x, int xOff, int incX,
                                double[] y, int yOff, int incY) {
        for (int i = 0; i < n; i++) {
            int xIndex = (xOff + i * incX) * 2;
            int yIndex = (yOff + i * incY) * 2;
            double xRe = x[xIndex];
            double xIm = x[xIndex + 1];
            double yRe = y[yIndex];
            double yIm = y[yIndex + 1];
            y[yIndex] = Math.fma(alphaRe, xRe, Math.fma(-alphaIm, xIm, yRe));
            y[yIndex + 1] = Math.fma(alphaRe, xIm, Math.fma(alphaIm, xRe, yIm));
        }
    }

}