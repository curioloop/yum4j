package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CarlsonIntegralsTest {

    @Test
    void rcMatchesClosedFormBranches() {
        assertClose(0.25 * Math.PI, CarlsonIntegrals.rc(0.0, 4.0), 5.0e-15);
        assertClose(1.0 / 3.0, CarlsonIntegrals.rc(9.0, 9.0), 5.0e-15);
    }

    @Test
    void carlsonFormsReproduceLegendreCompleteIntegrals() {
        double modulus = 0.5;
        double complementary = 1.0 - modulus * modulus;

        assertClose(EllipticIntegrals.completeFirstKind(modulus),
            CarlsonIntegrals.rf(0.0, complementary, 1.0), 5.0e-13);
        assertClose(EllipticIntegrals.completeSecondKind(modulus),
            2.0 * CarlsonIntegrals.rg(0.0, complementary, 1.0), 5.0e-13);

        double characteristic = 0.2;
        double expectedThirdKind = EllipticIntegrals.completeThirdKind(modulus, characteristic);
        double actualThirdKind = CarlsonIntegrals.rf(0.0, complementary, 1.0)
            + characteristic * CarlsonIntegrals.rj(0.0, complementary, 1.0, 1.0 - characteristic) / 3.0;
        assertClose(expectedThirdKind, actualThirdKind, 5.0e-13);
    }

    @Test
    void carlsonFormsReproduceLegendreIncompleteIntegrals() {
        double modulus = 0.6;
        double characteristic = 0.15;
        double phi = 0.8;
        double sine = Math.sin(phi);
        double sineSquared = sine * sine;
        double cosineSquared = Math.cos(phi) * Math.cos(phi);
        double secondArgument = 1.0 - modulus * modulus * sineSquared;
        double fourthArgument = 1.0 - characteristic * sineSquared;

        double firstKind = sine * CarlsonIntegrals.rf(cosineSquared, secondArgument, 1.0);
        double secondKind = firstKind
            - modulus * modulus * sine * sineSquared * CarlsonIntegrals.rd(cosineSquared, secondArgument, 1.0) / 3.0;
        double thirdKind = firstKind
            + characteristic * sine * sineSquared * CarlsonIntegrals.rj(cosineSquared, secondArgument, 1.0, fourthArgument) / 3.0;

        assertClose(EllipticIntegrals.incompleteFirstKind(modulus, phi), firstKind, 5.0e-13);
        assertClose(EllipticIntegrals.incompleteSecondKind(modulus, phi), secondKind, 5.0e-13);
        assertClose(EllipticIntegrals.incompleteThirdKind(modulus, characteristic, phi), thirdKind, 5.0e-13);
    }

    @Test
    void specialReductionsHold() {
        assertClose(CarlsonIntegrals.rd(0.25, 1.5, 2.0),
            CarlsonIntegrals.rj(0.25, 1.5, 2.0, 2.0), 5.0e-13);
        assertClose(1.0 / (2.0 * Math.sqrt(2.0)), CarlsonIntegrals.rd(2.0, 2.0, 2.0), 5.0e-15);
        assertClose(Math.sqrt(2.0), CarlsonIntegrals.rg(2.0, 2.0, 2.0), 5.0e-15);
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> CarlsonIntegrals.rc(-1.0, 2.0));
        assertThrows(IllegalArgumentException.class, () -> CarlsonIntegrals.rf(0.0, 0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> CarlsonIntegrals.rd(0.0, 0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> CarlsonIntegrals.rj(0.0, 1.0, 1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> CarlsonIntegrals.rg(-1.0, 1.0, 1.0));
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}