package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JacobiZetaTest {

    @Test
    void jacobiZetaMatchesBoostRjRepresentation() {
        double k = 0.5;
        double phi = 0.75;
        double sine = Math.sin(phi);
        double cosine = Math.cos(phi);
        double complementaryParameter = 1.0 - k * k;
        double oneMinusKSineSquared = complementaryParameter + cosine * cosine - complementaryParameter * cosine * cosine;
        double expected = k * k * sine * cosine * Math.sqrt(oneMinusKSineSquared)
            * CarlsonIntegrals.rj(0.0, complementaryParameter, 1.0, oneMinusKSineSquared)
            / (3.0 * EllipticIntegrals.completeFirstKind(k));
        assertClose(expected, JacobiZeta.zeta(k, phi), 5.0e-13);
    }

    @Test
    void jacobiZetaPreservesOddnessAndPiPeriodicity() {
        double k = 0.2;
        double phi = 1.1;
        assertClose(-JacobiZeta.zeta(k, phi), JacobiZeta.zeta(k, -phi), 5.0e-13);
        assertClose(JacobiZeta.zeta(k, phi), JacobiZeta.zeta(k, phi + Math.PI), 5.0e-13);
    }

    @Test
    void jacobiZetaIsEvenInModulus() {
        double phi = 0.8;
        assertClose(JacobiZeta.zeta(0.6, phi), JacobiZeta.zeta(-0.6, phi), 5.0e-13);
    }

    @Test
    void jacobiZetaHandlesBoundaryModuli() {
        assertEquals(0.0, JacobiZeta.zeta(0.0, 1.2), 0.0);
        double phi = 0.75;
        double expected = Math.sin(phi) * Math.signum(Math.cos(phi));
        assertClose(expected, JacobiZeta.zeta(1.0, phi), 5.0e-13);
    }

    @Test
    void jacobiZetaMatchesBoostBoundaryBehaviorAtNegativeUnity() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
            () -> JacobiZeta.zeta(-1.0, 0.75));
        assertTrue(exception.getMessage().contains("at most one of x, y, z may be zero"),
            "unexpected exception message: " + exception.getMessage());
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> JacobiZeta.zeta(1.1, 0.5));
        assertThrows(IllegalArgumentException.class, () -> JacobiZeta.zeta(Double.NaN, 0.5));
        assertThrows(IllegalArgumentException.class, () -> JacobiZeta.zeta(0.5, Double.NaN));
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}
