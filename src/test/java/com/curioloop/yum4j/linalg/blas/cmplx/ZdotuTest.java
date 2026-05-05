/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZdotuTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        double[] y = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

        double[] result = new double[2];
        ZLAS.zdotu(3, x, 0, 1, y, 0, 1, result);

        assertEquals(-39.0, result[0], TOL);
        assertEquals(214.0, result[1], TOL);
    }

    @Test
    void testSingleElement() {
        double[] x = {2.0, 3.0};
        double[] y = {4.0, 5.0};

        double[] result = new double[2];
        ZLAS.zdotu(1, x, 0, 1, y, 0, 1, result);

        assertEquals(-7.0, result[0], TOL);
        assertEquals(22.0, result[1], TOL);
    }

    @Test
    void testNonUnitStride() {
        double[] x = {1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0};
        double[] y = {7.0, 8.0, 0.0, 0.0, 9.0, 10.0, 0.0, 0.0, 11.0, 12.0};

        double[] result = new double[2];
        ZLAS.zdotu(3, x, 0, 2, y, 0, 2, result);

        assertEquals(-39.0, result[0], TOL);
        assertEquals(214.0, result[1], TOL);
    }
}