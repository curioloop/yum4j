package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BesselTest {

    @Test
    void zeroBoundarySemanticsMatchTheRealSurface() {
        assertClose(1.0, Bessel.i(0.0, 0.0), 0.0);
        assertClose(1.0, Bessel.iExponentiallyWeighted(0.0, 0.0), 0.0);
        assertClose(0.0, Bessel.i(2.0, 0.0), 0.0);
        assertClose(0.0, Bessel.iExponentiallyWeighted(2.0, 0.0), 0.0);
        assertTrue(Double.isInfinite(Bessel.k(0.0, 0.0)));
        assertTrue(Double.isInfinite(Bessel.kExponentiallyWeighted(0.0, 0.0)));
    }

    @Test
    void halfIntegerClosedFormsHoldOnTheRealSurface() {
        double x = 2.0;
        assertClose(Math.sqrt(2.0 / (Math.PI * x)) * Math.sin(x), Bessel.j(0.5, x), 5.0e-13);
        assertClose(-Math.sqrt(2.0 / (Math.PI * x)) * Math.cos(x), Bessel.y(0.5, x), 5.0e-13);
        assertClose(Math.sqrt(2.0 / (Math.PI * x)) * Math.sinh(x), Bessel.i(0.5, x), 5.0e-13);
        assertClose(Math.sqrt(Math.PI / (2.0 * x)) * Math.exp(-x), Bessel.k(0.5, x), 5.0e-13);
    }

    @Test
    void weightedVariantsMatchTheDeclaredExponentialWeight() {
        double nu = 0.75;
        double x = 2.0;
        assertClose(Math.exp(-x) * Bessel.i(nu, x), Bessel.iExponentiallyWeighted(nu, x), 5.0e-13);
        assertClose(Math.exp(-x) * Bessel.k(nu, x), Bessel.kExponentiallyWeighted(nu, x), 5.0e-13);
    }

    @Test
    void integerOrdersReflectOnNegativeArgumentsForJAndI() {
        assertClose(Bessel.j(2.0, 3.0), Bessel.j(2.0, -3.0), 5.0e-13);
        assertClose(-Bessel.j(1.0, 3.0), Bessel.j(1.0, -3.0), 5.0e-13);
        assertClose(Bessel.i(2.0, 3.0), Bessel.i(2.0, -3.0), 5.0e-13);
        assertClose(-Bessel.i(1.0, 3.0), Bessel.i(1.0, -3.0), 5.0e-13);
    }

    @Test
    void integerOrderKEvennessRemainsFinite() {
        assertTrue(Double.isFinite(Bessel.k(0.0, 2.0)));
        assertTrue(Double.isFinite(Bessel.k(1.0, 2.0)));
        assertClose(Bessel.k(1.0, 2.0), Bessel.k(-1.0, 2.0), 0.0);
    }

    @Test
    void unsupportedDomainsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> Bessel.j(0.75, -1.0));
        assertThrows(IllegalArgumentException.class, () -> Bessel.i(0.75, -1.0));
        assertThrows(IllegalArgumentException.class, () -> Bessel.y(0.75, 0.0));
        assertThrows(IllegalArgumentException.class, () -> Bessel.k(0.75, 0.0));
        assertTrue(Double.isInfinite(Bessel.k(0.0, 0.0)));
    }

    @Test
    void cylindricalDerivativesFollowRecurrenceRelations() {
        double nu = 1.25;
        double x = 2.5;
        assertClose(0.5 * (Bessel.j(nu - 1.0, x) - Bessel.j(nu + 1.0, x)), Bessel.jPrime(nu, x), 5.0e-13);
        assertClose(0.5 * (Bessel.y(nu - 1.0, x) - Bessel.y(nu + 1.0, x)), Bessel.yPrime(nu, x), 5.0e-13);
        assertClose(0.5 * (Bessel.i(nu - 1.0, x) + Bessel.i(nu + 1.0, x)), Bessel.iPrime(nu, x), 5.0e-13);
        assertClose(-0.5 * (Bessel.k(nu - 1.0, x) + Bessel.k(nu + 1.0, x)), Bessel.kPrime(nu, x), 5.0e-13);
    }

    @Test
    void sphericalBesselClosedFormsHold() {
        double x = 2.0;
        assertClose(Math.sin(x) / x, Bessel.sphBessel(0, x), 5.0e-13);
        assertClose(Math.sin(x) / (x * x) - Math.cos(x) / x, Bessel.sphBessel(1, x), 5.0e-13);
        assertClose(-Math.cos(x) / x, Bessel.sphNeumann(0, x), 5.0e-13);
        assertClose(-Math.cos(x) / (x * x) - Math.sin(x) / x, Bessel.sphNeumann(1, x), 5.0e-13);
    }

    @Test
    void sphericalDerivativesFollowRecurrenceRelations() {
        int order = 2;
        double x = 3.0;
        assertClose((order / x) * Bessel.sphBessel(order, x) - Bessel.sphBessel(order + 1, x),
            Bessel.sphBesselPrime(order, x), 5.0e-13);
        assertClose((order / x) * Bessel.sphNeumann(order, x) - Bessel.sphNeumann(order + 1, x),
            Bessel.sphNeumannPrime(order, x), 5.0e-13);
    }

    @Test
    void halfIntegerZerosMatchClosedForms() {
        assertClose(Math.PI, Bessel.jZero(0.5, 1), 5.0e-13);
        assertClose(2.0 * Math.PI, Bessel.jZero(0.5, 2), 5.0e-13);
        assertClose(Math.PI / 2.0, Bessel.yZero(0.5, 1), 5.0e-13);
        assertClose(3.0 * Math.PI / 2.0, Bessel.yZero(0.5, 2), 5.0e-13);
        assertClose(0.0, Bessel.jZero(1.0, 0), 0.0);
        assertClose(0.0, Bessel.yZero(-0.5, 0), 0.0);
    }

    @Test
    void cylindricalZerosAreStrictlyIncreasingAndRooted() {
        double nu = 0.75;
        double j1 = Bessel.jZero(nu, 1);
        double j2 = Bessel.jZero(nu, 2);
        double y1 = Bessel.yZero(nu, 1);
        double y2 = Bessel.yZero(nu, 2);

        assertTrue(j1 > 0.0 && j2 > j1);
        assertTrue(y1 > 0.0 && y2 > y1);
        assertTrue(Math.abs(Bessel.j(nu, j1)) <= 1.0e-10);
        assertTrue(Math.abs(Bessel.j(nu, j2)) <= 1.0e-10);
        assertTrue(Math.abs(Bessel.y(nu, y1)) <= 1.0e-10);
        assertTrue(Math.abs(Bessel.y(nu, y2)) <= 1.0e-10);
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}