package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorOpsTest {

    @Test
    void axpyToMatchesScalarLoopForSlices() {
        for (int len : new int[]{0, 1, 7, 31, 32, 33, 127, 1024}) {
            int xOff = 3;
            int yOff = 5;
            int dOff = 7;
            double[] x = random(len + xOff + 4, 0xA11L + len);
            double[] y = random(len + yOff + 4, 0xB22L + len);
            double[] actual = new double[len + dOff + 4];
            double[] expected = new double[actual.length];

            VectorOps.axpyTo(actual, dOff, -1.75, x, xOff, y, yOff, len);
            for (int i = 0; i < len; i++) expected[dOff + i] = -1.75 * x[xOff + i] + y[yOff + i];

            for (int i = 0; i < actual.length; i++) {
                assertEquals(expected[i], actual[i], 1e-13, "len=" + len + ", i=" + i);
            }
        }
    }

    @Test
    void scalToHandlesCopyScaleAndOverlap() {
        double[] values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        VectorOps.scalTo(values, 2, 10.0, values, 1, 4);

        assertEquals(0.0, values[0], 0.0);
        assertEquals(1.0, values[1], 0.0);
        assertEquals(10.0, values[2], 0.0);
        assertEquals(20.0, values[3], 0.0);
        assertEquals(30.0, values[4], 0.0);
        assertEquals(40.0, values[5], 0.0);
        assertEquals(6.0, values[6], 0.0);
    }

    @Test
    void scalToCopiesOverlappingSlicesWhenFactorIsOne() {
        double[] values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        VectorOps.scalTo(values, 2, 1.0, values, 0, 4);

        assertEquals(0.0, values[0], 0.0);
        assertEquals(1.0, values[1], 0.0);
        assertEquals(0.0, values[2], 0.0);
        assertEquals(1.0, values[3], 0.0);
        assertEquals(2.0, values[4], 0.0);
        assertEquals(3.0, values[5], 0.0);
    }

    private static double[] random(int len, long seed) {
        Random random = new Random(seed);
        double[] values = new double[len];
        for (int i = 0; i < len; i++) values[i] = random.nextGaussian();
        return values;
    }
}