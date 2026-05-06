package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BetaTest {

    @Test
    void completeAndIncompleteClosedFormsMatch() {
        assertEquals(Math.PI, Beta.beta(0.5, 0.5), 1.0e-15);
        assertEquals(1.0 / 12.0, Beta.beta(2.0, 3.0), 1.0e-15);

        assertEquals(0.21875, Beta.beta(1.0, 2.0, 0.25), 1.0e-15);
        assertEquals(0.28125, Beta.betac(1.0, 2.0, 0.25), 1.0e-15);
        assertEquals(0.5, Beta.beta(1.0, 2.0, 0.25) + Beta.betac(1.0, 2.0, 0.25), 1.0e-15);

        assertEquals(0.26171875, Beta.ibeta(2.0, 3.0, 0.25), 1.0e-14);
        assertEquals(0.73828125, Beta.ibetac(2.0, 3.0, 0.25), 1.0e-14);
        assertEquals(1.6875, Beta.ibetaDerivative(2.0, 3.0, 0.25), 1.0e-13);
    }

    @Test
    void normalizedBoundarySemanticsMatchBoostSpecialCases() {
        assertEquals(1.0, Beta.ibeta(0.0, 2.0, 0.5), 0.0);
        assertEquals(0.0, Beta.ibetac(0.0, 2.0, 0.5), 0.0);
        assertEquals(0.0, Beta.ibeta(3.0, 0.0, 0.5), 0.0);
        assertEquals(1.0, Beta.ibetac(3.0, 0.0, 0.5), 0.0);

        assertThrows(IllegalArgumentException.class, () -> Beta.ibeta(0.0, 0.0, 0.5));
        assertThrows(IllegalArgumentException.class, () -> Beta.beta(0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Beta.beta(1.0, 1.0, -0.1));
        assertThrows(IllegalArgumentException.class, () -> Beta.ibetaInv(1.0, 1.0, -0.1));
    }

    @Test
    void inverseOnXRoundTrips() {
        assertInverseRoundTrip(0.5, 0.5, 0.1);
        assertInverseRoundTrip(2.0, 3.0, 0.25);
        assertInverseRoundTrip(2.0, 3.0, 0.75);
        assertInverseRoundTrip(5.0, 1.5, 0.3);
        assertInverseRoundTrip(0.75, 2.5, 0.2);
    }

    @Test
    void inverseOnXRoundTripsAcrossBoostBoundaryCases() {
        assertInverseRoundTripAtX(3.0, 0.5, 1.0e-4, 5.0e-8);
        assertInverseRoundTripAtX(8.0, 2.0, 0.43868454635, 5.0e-10);
        assertInverseRoundTripAtX(8.0, 2.0, 0.5097693167348519, 5.0e-10);
        assertInverseRoundTripAtX(0.6, 7.5, 1.0e-6, 5.0e-8);
        assertInverseRoundTripAtX(11.0, 0.7, 0.999999, 5.0e-8);
    }

    @Test
    void inverseOnParametersRoundTrips() {
        double x = 0.25;
        double p = Beta.ibeta(2.0, 3.0, x);
        double q = Beta.ibetac(2.0, 3.0, x);

        assertEquals(2.0, Beta.ibetaInva(3.0, x, p), 1.0e-8);
        assertEquals(2.0, Beta.ibetacInva(3.0, x, q), 1.0e-8);
        assertEquals(3.0, Beta.ibetaInvb(2.0, x, p), 1.0e-8);
        assertEquals(3.0, Beta.ibetacInvb(2.0, x, q), 1.0e-8);
    }

    @Test
    void inverseOnParametersRoundTripAcrossBoostBoundaryCases() {
        assertParameterRoundTrip(22.0, 4.5, 0.94, 5.0e-5);
        assertParameterRoundTrip(0.85, 22.0, 0.06, 5.0e-5);
        assertParameterRoundTrip(3.0, 0.5, 1.0e-4, 5.0e-4);
        assertParameterRoundTrip(8.6, 6.25, 0.63, 5.0e-6);
    }

    @Test
    void inverseOnXCompanionPairApisMatchScalarAndPartition() {
        assertInverseRoundTripWithComplementPair(0.75, 2.5, 0.2, 5.0e-13);
        assertInverseRoundTripWithComplementPair(8.0, 2.0, 0.43868454635, 5.0e-10);
        assertInverseRoundTripWithComplementPair(11.0, 0.7, 0.999999, 5.0e-8);

        double p = 0.3;
        assertEquals(Beta.ibetaInv(2.0, 3.0, p), Beta.ibetaInvComplement(2.0, 3.0, p)._1(), 0.0);
        double q = 0.7;
        assertEquals(Beta.ibetacInv(2.0, 3.0, q), Beta.ibetacInvComplement(2.0, 3.0, q)._1(), 0.0);
    }

    @Test
    void inverseOnXCompanionPairApisHonorBoundaries() {
        Double2 fromPZero = Beta.ibetaInvComplement(2.0, 3.0, 0.0);
        assertEquals(0.0, fromPZero._1(), 0.0);
        assertEquals(1.0, fromPZero._2(), 0.0);

        Double2 fromPOne = Beta.ibetaInvComplement(2.0, 3.0, 1.0);
        assertEquals(1.0, fromPOne._1(), 0.0);
        assertEquals(0.0, fromPOne._2(), 0.0);

        Double2 fromQZero = Beta.ibetacInvComplement(2.0, 3.0, 0.0);
        assertEquals(1.0, fromQZero._1(), 0.0);
        assertEquals(0.0, fromQZero._2(), 0.0);

        Double2 fromQOne = Beta.ibetacInvComplement(2.0, 3.0, 1.0);
        assertEquals(0.0, fromQOne._1(), 0.0);
        assertEquals(1.0, fromQOne._2(), 0.0);
    }

    private static void assertInverseRoundTrip(double a, double b, double probability) {
        double x = Beta.ibetaInv(a, b, probability);
        double q = 1.0 - probability;

        assertEquals(probability, Beta.ibeta(a, b, x), 1.0e-10);
        assertEquals(q, Beta.ibetac(a, b, x), 1.0e-10);

        double xc = Beta.ibetacInv(a, b, q);
        assertEquals(probability, Beta.ibeta(a, b, xc), 1.0e-10);
        assertEquals(q, Beta.ibetac(a, b, xc), 1.0e-10);
    }

    private static void assertInverseRoundTripAtX(double a, double b, double x, double tolerance) {
        double p = Beta.ibeta(a, b, x);
        double q = Beta.ibetac(a, b, x);

        assertEquals(x, Beta.ibetaInv(a, b, p), tolerance);
        assertEquals(x, Beta.ibetacInv(a, b, q), tolerance);
    }

    private static void assertParameterRoundTrip(double a, double b, double x, double tolerance) {
        double p = Beta.ibeta(a, b, x);
        double q = Beta.ibetac(a, b, x);

        assertEquals(a, Beta.ibetaInva(b, x, p), tolerance);
        assertEquals(a, Beta.ibetacInva(b, x, q), tolerance);
        assertEquals(b, Beta.ibetaInvb(a, x, p), tolerance);
        assertEquals(b, Beta.ibetacInvb(a, x, q), tolerance);
    }

    private static void assertInverseRoundTripWithComplementPair(double a, double b, double x, double tolerance) {
        double p = Beta.ibeta(a, b, x);
        double q = Beta.ibetac(a, b, x);

        Double2 fromP = Beta.ibetaInvComplement(a, b, p);
        assertEquals(x, fromP._1(), tolerance);
        assertEquals(1.0 - x, fromP._2(), tolerance);
        assertTrue(Math.abs((fromP._1() + fromP._2()) - 1.0) <= 2.0 * tolerance);

        Double2 fromQ = Beta.ibetacInvComplement(a, b, q);
        assertEquals(x, fromQ._1(), tolerance);
        assertEquals(1.0 - x, fromQ._2(), tolerance);
        assertTrue(Math.abs((fromQ._1() + fromQ._2()) - 1.0) <= 2.0 * tolerance);
    }
}