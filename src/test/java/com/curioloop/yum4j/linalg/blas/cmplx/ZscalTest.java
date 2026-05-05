package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ZscalTest {

    private static final double TOL = 1e-14;

    @Test
    void testZscalUnitStride() {
        // Test with unit stride
        int n = 2;
        double[] x = {1.0, 2.0, 3.0, 4.0}; // [1+2i, 3+4i]
        double alphaRe = 2.0;
        double alphaIm = 1.0;

        ZLAS.zscal(n, alphaRe, alphaIm, x, 0, 1);

        // Expected: [2*1 - 1*2, 2*2 + 1*1, 2*3 - 1*4, 2*4 + 1*3] = [0, 5, 2, 11]
        assertEquals(0.0, x[0], TOL);
        assertEquals(5.0, x[1], TOL);
        assertEquals(2.0, x[2], TOL);
        assertEquals(11.0, x[3], TOL);
    }

    @Test
    void testZscalNonUnitStride() {
        // Test with non-unit stride
        int n = 2;
        double[] x = {1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0}; // [1+2i, 3+4i] with stride 2
        double alphaRe = 1.0;
        double alphaIm = -1.0;

        ZLAS.zscal(n, alphaRe, alphaIm, x, 0, 2);

        // Expected: [1*1 - (-1)*2, 1*2 + (-1)*1, 0, 0, 1*3 - (-1)*4, 1*4 + (-1)*3, 0, 0] = [3, 1, 0, 0, 7, 1, 0, 0]
        assertEquals(3.0, x[0], TOL);
        assertEquals(1.0, x[1], TOL);
        assertEquals(0.0, x[2], TOL);
        assertEquals(0.0, x[3], TOL);
        assertEquals(7.0, x[4], TOL);
        assertEquals(1.0, x[5], TOL);
        assertEquals(0.0, x[6], TOL);
        assertEquals(0.0, x[7], TOL);
    }

    @Test
    void testZscalZeroAlpha() {
        // Test with zero alpha
        int n = 2;
        double[] x = {1.0, 2.0, 3.0, 4.0};
        double alphaRe = 0.0;
        double alphaIm = 0.0;

        ZLAS.zscal(n, alphaRe, alphaIm, x, 0, 1);

        assertEquals(0.0, x[0], TOL);
        assertEquals(0.0, x[1], TOL);
        assertEquals(0.0, x[2], TOL);
        assertEquals(0.0, x[3], TOL);
    }

    @Test
    void testZscalZeroElements() {
        // Test with zero elements
        int n = 0;
        double[] x = {1.0, 2.0};
        double alphaRe = 2.0;
        double alphaIm = 1.0;

        ZLAS.zscal(n, alphaRe, alphaIm, x, 0, 1);

        assertEquals(1.0, x[0], TOL);
        assertEquals(2.0, x[1], TOL);
    }

    @Test
    void testZscalOffset() {
        // Test with offset
        int n = 2;
        double[] x = {0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0}; // [1+2i, 3+4i] starting at offset 2
        double alphaRe = 3.0;
        double alphaIm = 0.0;

        ZLAS.zscal(n, alphaRe, alphaIm, x, 2, 1);

        // Expected: [0, 0, 3*1, 3*2, 3*3, 3*4, 0, 0] = [0, 0, 3, 6, 9, 12, 0, 0]
        assertEquals(0.0, x[0], TOL);
        assertEquals(0.0, x[1], TOL);
        assertEquals(3.0, x[2], TOL);
        assertEquals(6.0, x[3], TOL);
        assertEquals(9.0, x[4], TOL);
        assertEquals(12.0, x[5], TOL);
        assertEquals(0.0, x[6], TOL);
        assertEquals(0.0, x[7], TOL);
    }

    @Test
    void testLargeContiguousMatchesReference() {
        int n = 4096;
        double alphaRe = -0.375;
        double alphaIm = 0.625;
        double[] actual = ZBlasTestSupport.randomComplexVector(n, 8192L);
        double[] expected = Arrays.copyOf(actual, actual.length);

        zscalReference(n, alphaRe, alphaIm, expected, 0, 1);
        ZLAS.zscal(n, alphaRe, alphaIm, actual, 0, 1);

        ZBlasTestSupport.assertArrayClose("Zscal large contiguous", expected, actual, TOL);
    }

    private void zscalReference(int n, double alphaRe, double alphaIm,
                                double[] x, int xOff, int incX) {
        for (int i = 0; i < n; i++) {
            int p = xOff + i * incX * 2;
            double xr = x[p];
            double xi = x[p + 1];
            x[p] = Math.fma(alphaRe, xr, -alphaIm * xi);
            x[p + 1] = Math.fma(alphaRe, xi, alphaIm * xr);
        }
    }
}
