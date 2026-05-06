package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HankelTest {

    @Test
    void positiveRealAxisMatchesDeclaredBesselComposition() {
        double nu = 0.75;
        double x = 1.5;
        assertComplexClose(new Complex(Bessel.j(nu, x), Bessel.y(nu, x)), Hankel.cylHankel1(nu, x), 5.0e-13);
        assertComplexClose(new Complex(Bessel.j(nu, x), -Bessel.y(nu, x)), Hankel.cylHankel2(nu, x), 5.0e-13);
    }

    @Test
    void secondKindIsTheConjugateOfFirstKindOnThePositiveRealAxis() {
        Complex h1 = Hankel.cylHankel1(1.25, 2.5);
        Complex h2 = Hankel.cylHankel2(1.25, 2.5);
        assertComplexClose(h1.conj(), h2, 5.0e-13);
    }

    @Test
    void halfIntegerCylindricalClosedFormHoldsOnBothSidesOfTheRealAxis() {
        assertHalfIntegerCylindricalCase(2.0);
        assertHalfIntegerCylindricalCase(-2.0);
    }

    @Test
    void sphericalOrderZeroClosedFormHoldsOnBothSidesOfTheRealAxis() {
        assertSphericalOrderZeroCase(2.0);
        assertSphericalOrderZeroCase(-2.0);
    }

    @Test
    void sphericalSurfaceMatchesTheCylindricalReduction() {
        double nu = 1.25;
        double x = 3.0;
        Complex z = new Complex(x, 0.0);
        Complex expectedH1 = Hankel.cylHankel1(nu + 0.5, x)
            .mul(Math.sqrt(Math.PI / 2.0))
            .div(z.sqrt());
        Complex expectedH2 = Hankel.cylHankel2(nu + 0.5, x)
            .mul(Math.sqrt(Math.PI / 2.0))
            .div(z.sqrt());
        assertComplexClose(expectedH1, Hankel.sphHankel1(nu, x), 5.0e-13);
        assertComplexClose(expectedH2, Hankel.sphHankel2(nu, x), 5.0e-13);
    }

    @Test
    void singularAndNonFiniteInputsAreRejected() {
        assertThrows(ArithmeticException.class, () -> Hankel.cylHankel1(0.0, 0.0));
        assertThrows(ArithmeticException.class, () -> Hankel.cylHankel2(1.0, 0.0));
        assertThrows(ArithmeticException.class, () -> Hankel.sphHankel1(0.0, 0.0));
        assertThrows(ArithmeticException.class, () -> Hankel.sphHankel2(1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> Hankel.cylHankel1(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Hankel.cylHankel1(1.0, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Hankel.sphHankel1(Double.POSITIVE_INFINITY, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Hankel.sphHankel2(1.0, Double.NEGATIVE_INFINITY));
    }

    private static void assertHalfIntegerCylindricalCase(double x) {
        Complex z = new Complex(x, 0.0);
        Complex phasePositive = Complex.polar(1.0, x);
        Complex phaseNegative = Complex.polar(1.0, -x);
        Complex scaleH1 = new Complex(0.0, -Math.sqrt(2.0 / Math.PI)).div(z.sqrt());
        Complex scaleH2 = new Complex(0.0, Math.sqrt(2.0 / Math.PI)).div(z.sqrt());
        Complex expectedH1 = scaleH1.mul(phasePositive);
        Complex expectedH2 = scaleH2.mul(phaseNegative);
        assertComplexClose(expectedH1, Hankel.cylHankel1(0.5, x), 5.0e-13);
        assertComplexClose(expectedH2, Hankel.cylHankel2(0.5, x), 5.0e-13);
    }

    private static void assertSphericalOrderZeroCase(double x) {
        Complex z = new Complex(x, 0.0);
        Complex phasePositive = Complex.polar(1.0, x);
        Complex phaseNegative = Complex.polar(1.0, -x);
        Complex expectedH1 = new Complex(0.0, -1.0).mul(phasePositive).div(z);
        Complex expectedH2 = new Complex(0.0, 1.0).mul(phaseNegative).div(z);
        assertComplexClose(expectedH1, Hankel.sphHankel1(0.0, x), 5.0e-13);
        assertComplexClose(expectedH2, Hankel.sphHankel2(0.0, x), 5.0e-13);
    }

    private static void assertComplexClose(Complex expected, Complex actual, double tolerance) {
        assertClose(expected.real(), actual.real(), tolerance);
        assertClose(expected.imag(), actual.imag(), tolerance);
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}