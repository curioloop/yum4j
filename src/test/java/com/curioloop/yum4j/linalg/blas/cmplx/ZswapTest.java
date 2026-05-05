package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ZswapTest {

    private static final double TOL = 1e-14;

    @Test
    void testZswapUnitStride() {
        // Test with unit stride
        int n = 2;
        double[] x = {1.0, 2.0, 3.0, 4.0}; // [1+2i, 3+4i]
        double[] y = {5.0, 6.0, 7.0, 8.0}; // [5+6i, 7+8i]

        ZLAS.zswap(n, x, 0, 1, y, 0, 1);

        // After swap, x should be [5+6i, 7+8i], y should be [1+2i, 3+4i]
        assertEquals(5.0, x[0], TOL);
        assertEquals(6.0, x[1], TOL);
        assertEquals(7.0, x[2], TOL);
        assertEquals(8.0, x[3], TOL);

        assertEquals(1.0, y[0], TOL);
        assertEquals(2.0, y[1], TOL);
        assertEquals(3.0, y[2], TOL);
        assertEquals(4.0, y[3], TOL);
    }

    @Test
    void testZswapNonUnitStride() {
        // Test with non-unit stride
        int n = 2;
        double[] x = {1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0}; // [1+2i, 3+4i] with stride 2
        double[] y = {5.0, 6.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0}; // [5+6i, 7+8i] with stride 2

        ZLAS.zswap(n, x, 0, 2, y, 0, 2);

        // After swap
        assertEquals(5.0, x[0], TOL);
        assertEquals(6.0, x[1], TOL);
        assertEquals(0.0, x[2], TOL);
        assertEquals(0.0, x[3], TOL);
        assertEquals(7.0, x[4], TOL);
        assertEquals(8.0, x[5], TOL);
        assertEquals(0.0, x[6], TOL);
        assertEquals(0.0, x[7], TOL);

        assertEquals(1.0, y[0], TOL);
        assertEquals(2.0, y[1], TOL);
        assertEquals(0.0, y[2], TOL);
        assertEquals(0.0, y[3], TOL);
        assertEquals(3.0, y[4], TOL);
        assertEquals(4.0, y[5], TOL);
        assertEquals(0.0, y[6], TOL);
        assertEquals(0.0, y[7], TOL);
    }

    @Test
    void testZswapZeroElements() {
        // Test with zero elements
        int n = 0;
        double[] x = {1.0, 2.0};
        double[] y = {5.0, 6.0};

        ZLAS.zswap(n, x, 0, 1, y, 0, 1);

        // Should remain unchanged
        assertEquals(1.0, x[0], TOL);
        assertEquals(2.0, x[1], TOL);
        assertEquals(5.0, y[0], TOL);
        assertEquals(6.0, y[1], TOL);
    }

    @Test
    void testZswapOffset() {
        // Test with offset
        int n = 2;
        double[] x = {0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0}; // [1+2i, 3+4i] starting at offset 2
        double[] y = {0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0}; // [5+6i, 7+8i] starting at offset 2

        ZLAS.zswap(n, x, 2, 1, y, 2, 1);

        // After swap
        assertEquals(0.0, x[0], TOL);
        assertEquals(0.0, x[1], TOL);
        assertEquals(5.0, x[2], TOL);
        assertEquals(6.0, x[3], TOL);
        assertEquals(7.0, x[4], TOL);
        assertEquals(8.0, x[5], TOL);
        assertEquals(0.0, x[6], TOL);
        assertEquals(0.0, x[7], TOL);

        assertEquals(0.0, y[0], TOL);
        assertEquals(0.0, y[1], TOL);
        assertEquals(1.0, y[2], TOL);
        assertEquals(2.0, y[3], TOL);
        assertEquals(3.0, y[4], TOL);
        assertEquals(4.0, y[5], TOL);
        assertEquals(0.0, y[6], TOL);
        assertEquals(0.0, y[7], TOL);
    }
}
