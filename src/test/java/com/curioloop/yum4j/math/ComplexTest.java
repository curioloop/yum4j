package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexTest {

    @Test
    void arithmeticOperatorsRemainSelfConsistent() {
        Complex left = new Complex(3.0, -4.0);
        Complex right = new Complex(-2.0, 5.0);

        assertComplexEquals(new Complex(1.0, 1.0), left.add(right), 1.0e-15);
        assertComplexEquals(new Complex(5.0, -9.0), left.sub(right), 1.0e-15);
        assertComplexEquals(new Complex(14.0, 23.0), left.mul(right), 1.0e-15);
        assertComplexEquals(left, left.div(right).mul(right), 1.0e-12);
        assertComplexEquals(Complex.ONE, left.mul(left.recip()), 1.0e-12);
    }

    @Test
    void expAndLogRoundTripAwayFromBranchCut() {
        Complex value = new Complex(1.25, -0.75);
        assertComplexEquals(value, value.log().exp(), 1.0e-12);
    }

    @Test
    void sqrtUsesPrincipalBranch() {
        Complex value = new Complex(-3.0, 4.0);
        Complex root = value.sqrt();

        assertTrue(root.real() >= 0.0);
        assertComplexEquals(value, root.mul(root), 1.0e-12);
    }

    @Test
    void absAndPowUsePrincipalBranch() {
        assertEquals(5.0, new Complex(3.0, -4.0).abs(), 1.0e-15);

        Complex squared = new Complex(1.0, 1.0).pow(2.0);
        assertComplexEquals(new Complex(0.0, 2.0), squared, 1.0e-12);

        Complex imaginaryPower = new Complex(-1.0, 0.0).pow(Complex.I);
        assertComplexEquals(new Complex(Math.exp(-Math.PI), 0.0), imaginaryPower, 1.0e-12);
    }

    @Test
    void scalarHelpersAndPolarFormRemainConsistent() {
        Complex value = new Complex(3.0, -4.0);

        assertComplexEquals(new Complex(6.0, -8.0), value.mul(2.0), 0.0);
        assertComplexEquals(new Complex(1.5, -2.0), value.div(2.0), 0.0);
        assertComplexEquals(new Complex(-3.0, 4.0), value.negate(), 0.0);
        assertComplexEquals(value, Complex.polar(value.abs(), value.arg()), 1.0e-12);

        assertEquals(25.0, value.norm(), 1.0e-15);
        assertEquals(Math.atan2(-4.0, 3.0), value.arg(), 1.0e-15);
        assertTrue(value.isFinite());
        assertFalse(Complex.NAN.isFinite());
        assertTrue(Complex.INF.isInfinite());
    }

    @Test
    void argMatchesAtan2AcrossFiniteQuadrants() {
        assertEquals(Math.atan2(-4.0, 3.0), new Complex(3.0, -4.0).arg(), 1.0e-15);
        assertEquals(Math.atan2(4.0, -3.0), new Complex(-3.0, 4.0).arg(), 1.0e-15);
        assertEquals(Math.atan2(-4.0, -3.0), new Complex(-3.0, -4.0).arg(), 1.0e-15);
        assertEquals(Math.atan2(3.0, 4.0), new Complex(4.0, 3.0).arg(), 1.0e-15);
        assertEquals(Math.atan2(2.0, 0.0), new Complex(0.0, 2.0).arg(), 0.0);
        assertEquals(Math.atan2(-2.0, 0.0), new Complex(0.0, -2.0).arg(), 0.0);
        assertEquals(Math.atan2(0.0, -2.0), new Complex(-2.0, 0.0).arg(), 0.0);
        assertEquals(Math.atan2(-0.0, -2.0), new Complex(-2.0, -0.0).arg(), 0.0);
    }

    @Test
    void directAndInverseFunctionsUsePrincipalBranchesConsistently() {
        Complex asinInput = new Complex(0.4, -0.3);
        Complex atanInput = new Complex(0.25, 0.35);
        Complex acoshInput = new Complex(2.0, 0.25);
        Complex atanhInput = new Complex(0.2, -0.3);

        assertComplexEquals(asinInput, asinInput.asin().sin(), 1.0e-12);
        assertComplexEquals(asinInput, asinInput.acos().cos(), 1.0e-12);
        assertComplexEquals(atanInput, atanInput.atan().tan(), 1.0e-12);
        assertComplexEquals(asinInput, asinInput.asinh().sinh(), 1.0e-12);
        assertComplexEquals(acoshInput, acoshInput.acosh().cosh(), 1.0e-12);
        assertComplexEquals(atanhInput, atanhInput.atanh().tanh(), 1.0e-12);
    }

    @Test
    void tanAndTanhStayFiniteForLargeFiniteArguments() {
        Complex tanValue = new Complex(0.75, 400.0).tan();
        assertTrue(tanValue.isFinite());
        assertTrue(Math.abs(tanValue.real()) < 1.0e-15);
        assertEquals(1.0, tanValue.imag(), 0.0);

        Complex tanhValue = new Complex(400.0, 0.75).tanh();
        assertTrue(tanhValue.isFinite());
        assertEquals(1.0, tanhValue.real(), 0.0);
        assertTrue(Math.abs(tanhValue.imag()) < 1.0e-15);
    }

    @Test
    void divideRemainsStableForLargeFiniteOperands() {
        Complex quotient = new Complex(1.0e300, -1.0e300).div(new Complex(1.0e300, 1.0e300));
        assertEquals(0.0, quotient.real(), 1.0e-15);
        assertEquals(-1.0, quotient.imag(), 1.0e-15);

        Complex reciprocal = new Complex(1.0e300, 1.0e300).recip();
        double tolerance = 8.0 * Math.ulp(5.0e-301);
        assertEquals(5.0e-301, reciprocal.real(), tolerance);
        assertEquals(-5.0e-301, reciprocal.imag(), tolerance);
    }

    @Test
    void powHandlesZeroBaseExplicitly() {
        assertComplexEquals(Complex.ONE, Complex.ZERO.pow(0.0), 0.0);
        assertComplexEquals(Complex.ZERO, Complex.ZERO.pow(2.0), 0.0);
        assertComplexEquals(Complex.ONE, Complex.ZERO.pow(Complex.ZERO), 0.0);
        assertComplexEquals(Complex.ZERO, Complex.ZERO.pow(new Complex(2.0, 0.0)), 0.0);

        assertThrows(ArithmeticException.class, () -> Complex.ZERO.pow(-1.0));
        assertThrows(ArithmeticException.class, () -> Complex.ZERO.pow(new Complex(-1.0, 0.0)));
        assertThrows(ArithmeticException.class, () -> Complex.ZERO.pow(new Complex(1.0, 1.0)));
    }

    @Test
    void conjugationAndSpecialValuesMirrorGoStyleHelpers() {
        Complex value = new Complex(3.0, -4.0);

        assertComplexEquals(new Complex(3.0, 4.0), value.conj(), 0.0);
        assertComplexEquals(value, value.conj().conj(), 0.0);
        assertTrue(Complex.ZERO.isZero());

        assertFalse(value.isNaN());
        assertTrue(Complex.NAN.isNaN());
        assertTrue(Double.isNaN(Complex.NAN.real()));
        assertTrue(Double.isNaN(Complex.NAN.imag()));

        Complex inf = Complex.INF;
        assertTrue(Double.isInfinite(inf.real()));
        assertTrue(Double.isInfinite(inf.imag()));
        assertTrue(inf.real() > 0.0);
        assertTrue(inf.imag() > 0.0);
    }

    private static void assertComplexEquals(Complex expected, Complex actual, double tolerance) {
        assertEquals(expected.real(), actual.real(), tolerance);
        assertEquals(expected.imag(), actual.imag(), tolerance);
    }
}