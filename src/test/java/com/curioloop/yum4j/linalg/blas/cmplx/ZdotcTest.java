/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZdotcTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        // x = [1+2i, 3+4i, 5+6i]
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        // y = [7+8i, 9+10i, 11+12i]
        double[] y = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

        double[] result = new double[2];
        Zdot.zdotc(3, x, 0, 1, y, 0, 1, result, 0);

        // Expected: conj(x) · y = (1-2i)(7+8i) + (3-4i)(9+10i) + (5-6i)(11+12i)
        // First term: (1)(7) + (2)(8) + (1)(8)i - (2)(7)i = 7 + 16 + (8 - 14)i = 23 - 6i
        // Second term: (3)(9) + (4)(10) + (3)(10)i - (4)(9)i = 27 + 40 + (30 - 36)i = 67 - 6i
        // Third term: (5)(11) + (6)(12) + (5)(12)i - (6)(11)i = 55 + 72 + (60 - 66)i = 127 - 6i
        // Total: 23+67+127 + (-6-6-6)i = 217 - 18i
        assertEquals(217.0, result[0], TOL);
        assertEquals(-18.0, result[1], TOL);
    }

    @Test
    void testSingleElement() {
        // x = [2+3i]
        double[] x = {2.0, 3.0};
        // y = [4+5i]
        double[] y = {4.0, 5.0};

        double[] result = new double[2];
        Zdot.zdotc(1, x, 0, 1, y, 0, 1, result, 0);

        // Expected: conj(x) · y = (2-3i)(4+5i) = 8 + 10i - 12i + 15 = 23 - 2i
        assertEquals(23.0, result[0], TOL);
        assertEquals(-2.0, result[1], TOL);
    }

    @Test
    void testEmpty() {
        double[] result = new double[2];
        Zdot.zdotc(0, new double[0], 0, 1, new double[0], 0, 1, result, 0);
        assertEquals(0.0, result[0], TOL);
        assertEquals(0.0, result[1], TOL);
    }

    @Test
    void testNonUnitStride() {
        // x = [1+2i, 0+0i, 3+4i, 0+0i, 5+6i]
        double[] x = {1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0};
        // y = [7+8i, 0+0i, 9+10i, 0+0i, 11+12i]
        double[] y = {7.0, 8.0, 0.0, 0.0, 9.0, 10.0, 0.0, 0.0, 11.0, 12.0};

        double[] result = new double[2];
        Zdot.zdotc(3, x, 0, 2, y, 0, 2, result, 0);

        // Expected: conj(x) · y = (1-2i)(7+8i) + (3-4i)(9+10i) + (5-6i)(11+12i) = 217 - 18i
        assertEquals(217.0, result[0], TOL);
        assertEquals(-18.0, result[1], TOL);
    }

    @Test
    void testResultArray() {
        // x = [1+2i, 3+4i]
        double[] x = {1.0, 2.0, 3.0, 4.0};
        // y = [5+6i, 7+8i]
        double[] y = {5.0, 6.0, 7.0, 8.0};

        double[] result = new double[2];
        Zdot.zdotc(2, x, 0, 1, y, 0, 1, result, 0);

        // Expected: conj(x) · y = (1-2i)(5+6i) + (3-4i)(7+8i)
        // First term: 5 + 6i - 10i + 12 = 17 - 4i
        // Second term: 21 + 24i - 28i + 32 = 53 - 4i
        // Total: 70 - 8i
        assertEquals(70.0, result[0], TOL);
        assertEquals(-8.0, result[1], TOL);
    }

}