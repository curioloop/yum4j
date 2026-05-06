package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EllipticIntegralsTest {

    @Test
    void zeroModulusReducesFirstAndSecondKindsToElementaryIntegrals() {
        double phi = 1.1;
        assertEquals(0.5 * Math.PI, EllipticIntegrals.completeFirstKind(0.0), 1.0e-15);
        assertEquals(0.5 * Math.PI, EllipticIntegrals.completeSecondKind(0.0), 1.0e-15);
        assertEquals(phi, EllipticIntegrals.incompleteFirstKind(0.0, phi), 1.0e-13);
        assertEquals(phi, EllipticIntegrals.incompleteSecondKind(0.0, phi), 1.0e-13);
    }

    @Test
    void thirdKindReducesToFirstKindWhenCharacteristicIsZero() {
        double k = 0.65;
        double phi = 1.0;
        assertClose(EllipticIntegrals.completeFirstKind(k), EllipticIntegrals.completeThirdKind(k, 0.0), 1.0e-13);
        assertClose(EllipticIntegrals.incompleteFirstKind(k, phi), EllipticIntegrals.incompleteThirdKind(k, 0.0, phi), 1.0e-13);
    }

    @Test
    void completeValuesMatchHalfPiIncompleteValues() {
        double k = 0.55;
        double nu = 0.25;
        assertClose(EllipticIntegrals.completeFirstKind(k), EllipticIntegrals.incompleteFirstKind(k, 0.5 * Math.PI), 1.0e-13);
        assertClose(EllipticIntegrals.completeSecondKind(k), EllipticIntegrals.incompleteSecondKind(k, 0.5 * Math.PI), 1.0e-13);
        assertClose(EllipticIntegrals.completeThirdKind(k, nu), EllipticIntegrals.incompleteThirdKind(k, nu, 0.5 * Math.PI), 1.0e-13);
    }

    @Test
    void incompleteSurfacePreservesOddSymmetryAndPiShift() {
        double k = 0.7;
        double nu = -0.2;
        double phi = 0.8;

        assertClose(-EllipticIntegrals.incompleteFirstKind(k, phi), EllipticIntegrals.incompleteFirstKind(k, -phi), 1.0e-13);
        assertClose(-EllipticIntegrals.incompleteSecondKind(k, phi), EllipticIntegrals.incompleteSecondKind(k, -phi), 1.0e-13);
        assertClose(-EllipticIntegrals.incompleteThirdKind(k, nu, phi), EllipticIntegrals.incompleteThirdKind(k, nu, -phi), 1.0e-13);

        assertClose(
            2.0 * EllipticIntegrals.completeFirstKind(k) + EllipticIntegrals.incompleteFirstKind(k, phi),
            EllipticIntegrals.incompleteFirstKind(k, phi + Math.PI),
            1.0e-13
        );
        assertClose(
            2.0 * EllipticIntegrals.completeSecondKind(k) + EllipticIntegrals.incompleteSecondKind(k, phi),
            EllipticIntegrals.incompleteSecondKind(k, phi + Math.PI),
            1.0e-13
        );
        assertClose(
            2.0 * EllipticIntegrals.completeThirdKind(k, nu) + EllipticIntegrals.incompleteThirdKind(k, nu, phi),
            EllipticIntegrals.incompleteThirdKind(k, nu, phi + Math.PI),
            1.0e-13
        );
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> EllipticIntegrals.completeFirstKind(1.0));
        assertThrows(IllegalArgumentException.class, () -> EllipticIntegrals.incompleteFirstKind(0.5, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> EllipticIntegrals.completeThirdKind(0.5, 1.0));
        assertThrows(IllegalArgumentException.class, () -> EllipticIntegrals.incompleteThirdKind(0.5, 1.25, 1.2));
    }

    @Test
    void resultsRemainFiniteAcrossRepresentativeInteriorPoints() {
        assertTrue(Double.isFinite(EllipticIntegrals.completeFirstKind(0.8)));
        assertTrue(Double.isFinite(EllipticIntegrals.completeSecondKind(0.8)));
        assertTrue(Double.isFinite(EllipticIntegrals.completeThirdKind(0.8, -0.3)));
        assertTrue(Double.isFinite(EllipticIntegrals.incompleteThirdKind(0.8, 0.6, 0.7)));
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}