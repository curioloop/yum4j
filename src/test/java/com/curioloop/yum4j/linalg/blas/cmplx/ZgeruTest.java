/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ZgeruTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {1.0, 2.0, 3.0, 4.0};
        double[] y = {5.0, 6.0, 7.0, 8.0};
        double[] a = new double[8];

        ZLAS.zgeru(2, 2, 1.0, 0.0, x, 0, 1, y, 0, 1, a, 0, 2);

        assertArrayEquals(new double[]{-7.0, 16.0, -9.0, 22.0, -9.0, 38.0, -11.0, 52.0}, a, TOL);
    }

    @Test
    void testNonUnitStride() {
        double[] x = {1.0, 2.0, 0.0, 0.0, 3.0, 4.0};
        double[] y = {5.0, 6.0, 0.0, 0.0, 7.0, 8.0};
        double[] a = new double[8];

        ZLAS.zgeru(2, 2, 1.0, 0.0, x, 0, 2, y, 0, 2, a, 0, 2);

        assertArrayEquals(new double[]{-7.0, 16.0, -9.0, 22.0, -9.0, 38.0, -11.0, 52.0}, a, TOL);
    }
}