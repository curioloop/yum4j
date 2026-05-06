package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ContinuedFractionTest {

    @Test
    void continuedFractionAUsesLeadingNumeratorLikeBoost() {
        double expected = 2.0 / (1.0 + Math.sqrt(5.0));
        ContinuedFraction actual = ContinuedFraction.evaluateFractionA(
            index -> Double2.fraction(1.0, 1.0),
            1.0e-15,
            1000);

        assertTrue(actual.converged());
        assertClose(expected, actual.value(), 5.0e-15);
    }

    @Test
    void continuedFractionBIgnoresFirstNumeratorLikeBoost() {
        double expected = 0.5 * (1.0 + Math.sqrt(5.0));
        ContinuedFraction actual = ContinuedFraction.evaluateFractionB(
            index -> index == 0 ? Double2.fraction(12345.0, 1.0) : Double2.fraction(1.0, 1.0),
            1.0e-15,
            1000);

        assertTrue(actual.converged());
        assertClose(expected, actual.value(), 5.0e-15);
    }

    @Test
    void diagnosticResultReportsIterationsAndNonConvergence() {
        ContinuedFraction converged = ContinuedFraction.evaluateFractionA(
            index -> Double2.fraction(1.0, 1.0),
            1.0e-15,
            1000);

        assertTrue(converged.converged());
        assertTrue(converged.iterations() > 0);

        ContinuedFraction stalled = ContinuedFraction.evaluateFractionA(
            index -> Double2.fraction(1.0, 1.0),
            1.0e-30,
            1);

        assertFalse(stalled.converged());
        assertEquals(1, stalled.iterations());
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}