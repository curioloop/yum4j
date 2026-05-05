package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ZcopyTest {

    private static final double TOL = 1e-14;

    @Test
    void testZcopyUnitStride() {
        // Test with unit stride
        int n = 3;
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // [1+2i, 3+4i, 5+6i]
        double[] y = new double[6];

        ZLAS.zcopy(n, x, 0, 1, y, 0, 1);

        assertEquals(1.0, y[0], TOL);
        assertEquals(2.0, y[1], TOL);
        assertEquals(3.0, y[2], TOL);
        assertEquals(4.0, y[3], TOL);
        assertEquals(5.0, y[4], TOL);
        assertEquals(6.0, y[5], TOL);
    }

    @Test
    void testZcopyNonUnitStride() {
        // Test with non-unit stride
        int n = 2;
        double[] x = {1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0}; // [1+2i, 3+4i] with stride 2
        double[] y = new double[8];

        ZLAS.zcopy(n, x, 0, 2, y, 2, 2);

        assertEquals(0.0, y[0], TOL);
        assertEquals(0.0, y[1], TOL);
        assertEquals(1.0, y[2], TOL);
        assertEquals(2.0, y[3], TOL);
        assertEquals(0.0, y[4], TOL);
        assertEquals(0.0, y[5], TOL);
        assertEquals(3.0, y[6], TOL);
        assertEquals(4.0, y[7], TOL);
    }

    @Test
    void testZcopyZeroElements() {
        // Test with zero elements
        int n = 0;
        double[] x = {1.0, 2.0};
        double[] y = new double[2];

        ZLAS.zcopy(n, x, 0, 1, y, 0, 1);

        assertEquals(0.0, y[0], TOL);
        assertEquals(0.0, y[1], TOL);
    }

    @Test
    void testZcopyOffset() {
        // Test with offset
        int n = 2;
        double[] x = {0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0}; // [1+2i, 3+4i] starting at offset 2
        double[] y = new double[6];

        ZLAS.zcopy(n, x, 2, 1, y, 1, 1);

        assertEquals(0.0, y[0], TOL);
        assertEquals(1.0, y[1], TOL);
        assertEquals(2.0, y[2], TOL);
        assertEquals(3.0, y[3], TOL);
        assertEquals(4.0, y[4], TOL);
        assertEquals(0.0, y[5], TOL);
    }

    @Test
    void testZcopyUnitStrideSameArrayForwardOverlap() {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

        ZLAS.zcopy(2, data, 0, 1, data, 2, 1);

        assertEquals(1.0, data[0], TOL);
        assertEquals(2.0, data[1], TOL);
        assertEquals(1.0, data[2], TOL);
        assertEquals(2.0, data[3], TOL);
        assertEquals(3.0, data[4], TOL);
        assertEquals(4.0, data[5], TOL);
        assertEquals(7.0, data[6], TOL);
        assertEquals(8.0, data[7], TOL);
    }

    @Test
    void testZcopyUnitStrideSameArrayBackwardOverlap() {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

        ZLAS.zcopy(2, data, 2, 1, data, 0, 1);

        assertEquals(3.0, data[0], TOL);
        assertEquals(4.0, data[1], TOL);
        assertEquals(5.0, data[2], TOL);
        assertEquals(6.0, data[3], TOL);
        assertEquals(5.0, data[4], TOL);
        assertEquals(6.0, data[5], TOL);
        assertEquals(7.0, data[6], TOL);
        assertEquals(8.0, data[7], TOL);
    }
}
