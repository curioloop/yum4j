package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import java.util.function.IntToDoubleFunction;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BesselZeroFunctionTest {

    @Test
    void jZerosMatchesScalarQueries() {
        double nu = 0.75;
        IntToDoubleFunction zeros = Bessel.jZeros(nu);

        assertClose(Bessel.jZero(nu, 1), zeros.applyAsDouble(1), 0.0);
        assertClose(Bessel.jZero(nu, 2), zeros.applyAsDouble(2), 0.0);
        assertClose(Bessel.jZero(nu, 3), zeros.applyAsDouble(3), 0.0);
    }

    @Test
    void jZerosPreservesRankZeroSemantics() {
        IntToDoubleFunction zeros = Bessel.jZeros(1.0);

        assertClose(Bessel.jZero(1.0, 0), zeros.applyAsDouble(0), 0.0);
        assertClose(Bessel.jZero(1.0, 1), zeros.applyAsDouble(1), 0.0);
        assertClose(Bessel.jZero(1.0, 2), zeros.applyAsDouble(2), 0.0);
    }

    @Test
    void yZerosPreservesRankZeroSemanticsWhereTheScalarSurfaceAllowsIt() {
        double nu = -0.5;
        IntToDoubleFunction zeros = Bessel.yZeros(nu);

        assertClose(Bessel.yZero(nu, 0), zeros.applyAsDouble(0), 0.0);
        assertClose(Bessel.yZero(nu, 1), zeros.applyAsDouble(1), 0.0);
        assertClose(Bessel.yZero(nu, 2), zeros.applyAsDouble(2), 0.0);
    }

    @Test
    void zeroFunctionsPreservePerRankDomainValidation() {
        IntToDoubleFunction jZeros = Bessel.jZeros(0.0);
        IntToDoubleFunction yZeros = Bessel.yZeros(0.75);

        assertThrows(IllegalArgumentException.class, () -> jZeros.applyAsDouble(0));
        assertThrows(IllegalArgumentException.class, () -> yZeros.applyAsDouble(0));
    }

    @Test
    void zeroFunctionsRejectNegativeRanks() {
        IntToDoubleFunction zeros = Bessel.jZeros(0.75);

        assertThrows(IllegalArgumentException.class, () -> zeros.applyAsDouble(-1));
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}