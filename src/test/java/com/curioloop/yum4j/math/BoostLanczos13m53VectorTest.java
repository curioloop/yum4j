package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BoostLanczos13m53VectorTest {

    private static final double[] SAMPLE_X = {
        0.125,
        0.5,
        0.75,
        1.0,
        1.5,
        2.5,
        4.0,
        10.0,
        100.0,
        1.0e10,
        1.0e20,
        1.0e26,
        1.0e40,
        1.0e100
    };

    @Test
    void lanczosSumVectorMatchesScalarPort() {
        for (double x : SAMPLE_X) {
            assertBitwiseEquals(
                Lanczos13m53.lanczosSumScalar(x),
                Lanczos13m53.lanczosSum(x)
            );
        }
    }

    @Test
    void lanczosSumExpGScaledVectorMatchesScalarPort() {
        for (double x : SAMPLE_X) {
            assertBitwiseEquals(
                Lanczos13m53.lanczosSumExpGScaledScalar(x),
                Lanczos13m53.lanczosSumExpGScaled(x)
            );
        }
    }

    @Test
    void boostGMatchesLanczos13m53Constant() {
        assertEquals(6.024680040776729583740234375, Lanczos13m53.g(), 0.0);
    }

    private static void assertBitwiseEquals(double expected, double actual) {
        assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual));
    }
}