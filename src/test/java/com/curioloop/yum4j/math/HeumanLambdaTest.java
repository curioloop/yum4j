package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HeumanLambdaTest {

    private static final double HALF_PI = 0.5 * Math.PI;

    @Test
    void specialModulusAndAmplitudeValuesMatchKnownLimits() {
        assertClose(Math.sin(0.8), HeumanLambda.lambda(0.0, 0.8), 5.0e-15);
        assertEquals(0.0, HeumanLambda.lambda(0.7, 0.0), 0.0);
    }

    @Test
    void heumanLambdaPreservesOddnessInAmplitude() {
        assertClose(-HeumanLambda.lambda(0.5, 0.8), HeumanLambda.lambda(0.5, -0.8), 5.0e-13);
        assertClose(-HeumanLambda.lambda(0.5, 2.0), HeumanLambda.lambda(0.5, -2.0), 5.0e-13);
    }

    @Test
    void heumanLambdaApproachesUnityAtHalfPiFromBothSides() {
        double below = HeumanLambda.lambda(0.7, HALF_PI - 1.0e-6);
        double above = HeumanLambda.lambda(0.7, HALF_PI + 1.0e-6);
        assertClose(1.0, below, 2.0e-4);
        assertClose(1.0, above, 5.0e-7);
    }

    @Test
    void heumanLambdaMatchesBoostBoundaryBehaviorAtUnityModulus() {
        IllegalArgumentException positive = assertThrows(IllegalArgumentException.class,
            () -> HeumanLambda.lambda(1.0, 0.0));
        assertTrue(positive.getMessage().contains("at most one argument may be zero"),
            "unexpected exception message: " + positive.getMessage());

        IllegalArgumentException negative = assertThrows(IllegalArgumentException.class,
            () -> HeumanLambda.lambda(-1.0, 0.0));
        assertTrue(negative.getMessage().contains("at most one argument may be zero"),
            "unexpected exception message: " + negative.getMessage());
    }

    @Test
    void heumanLambdaMatchesBoostBoundaryBehaviorAtHalfPiForNonZeroModulus() {
        IllegalArgumentException positive = assertThrows(IllegalArgumentException.class,
            () -> HeumanLambda.lambda(0.6, HALF_PI));
        assertTrue(positive.getMessage().contains("p must be non-zero"),
            "unexpected exception message: " + positive.getMessage());

        IllegalArgumentException negative = assertThrows(IllegalArgumentException.class,
            () -> HeumanLambda.lambda(0.6, -HALF_PI));
        assertTrue(negative.getMessage().contains("p must be non-zero"),
            "unexpected exception message: " + negative.getMessage());
    }

    @Test
    void heumanLambdaMatchesBoostZeroModulusHalfPiBehavior() {
        assertTrue(Double.isNaN(HeumanLambda.lambda(0.0, HALF_PI)));
        assertTrue(Double.isNaN(HeumanLambda.lambda(0.0, -HALF_PI)));
    }

    @Test
    void heumanLambdaMatchesBoostOverflowBehaviorAtUnityModulusBeyondHalfPi() {
        ArithmeticException positive = assertThrows(ArithmeticException.class,
            () -> HeumanLambda.lambda(1.0, 2.0));
        assertTrue(positive.getMessage().contains("overflow"),
            "unexpected exception message: " + positive.getMessage());

        ArithmeticException negative = assertThrows(ArithmeticException.class,
            () -> HeumanLambda.lambda(-1.0, -2.0));
        assertTrue(negative.getMessage().contains("overflow"),
            "unexpected exception message: " + negative.getMessage());
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> HeumanLambda.lambda(1.1, 0.5));
        assertThrows(IllegalArgumentException.class, () -> HeumanLambda.lambda(Double.NaN, 0.5));
        assertThrows(IllegalArgumentException.class, () -> HeumanLambda.lambda(0.5, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> HeumanLambda.lambda(0.0, 2.0));
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(expected - actual);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}