package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class IzamaxTest {

    @Test
    void testIzamaxUnitStride() {
        int n = 3;
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        int result = ZLAS.izamax(n, x, 0, 1);
        assertEquals(2, result);
    }

    @Test
    void testIzamaxNonUnitStride() {
        int n = 2;
        double[] x = {1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0};
        int result = ZLAS.izamax(n, x, 0, 2);
        assertEquals(1, result);
    }

    @Test
    void testIzamaxZeroElements() {
        int n = 0;
        double[] x = {1.0, 2.0};
        int result = ZLAS.izamax(n, x, 0, 1);
        assertEquals(-1, result);
    }

    @Test
    void testIzamaxOffset() {
        int n = 2;
        double[] x = {0.0, 0.0, 3.0, 4.0, 1.0, 2.0, 0.0, 0.0};
        int result = ZLAS.izamax(n, x, 2, 1);
        assertEquals(0, result);
    }

    @Test
    void testIzamaxNegativeValues() {
        int n = 2;
        double[] x = {-3.0, 4.0, 5.0, -12.0};
        int result = ZLAS.izamax(n, x, 0, 1);
        assertEquals(1, result);
    }
}
