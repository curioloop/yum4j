package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ComplexStdlibParityTest {

    @Test
    void sqrtAndAcosMatchStdComplexSignedZeroLips() {
        Complex sqrtUpper = new Complex(-4.0, 0.0).sqrt();
        assertPositiveZero(sqrtUpper.real());
        assertEquals(2.0, sqrtUpper.imag(), 1.0e-15);

        Complex sqrtLower = new Complex(-4.0, -0.0).sqrt();
        assertPositiveZero(sqrtLower.real());
        assertEquals(-2.0, sqrtLower.imag(), 1.0e-15);

        Complex acosUpper = new Complex(0.0, 0.0).acos();
        assertEquals(Math.PI / 2.0, acosUpper.real(), 1.0e-15);
        assertNegativeZero(acosUpper.imag());

        Complex acosLower = new Complex(0.0, -0.0).acos();
        assertEquals(Math.PI / 2.0, acosLower.real(), 1.0e-15);
        assertPositiveZero(acosLower.imag());
    }

    @Test
    void inverseHyperbolicBranchExamplesMatchStdComplexProbe() {
        assertComplexClose(
            new Complex(1.31695789692481657, -1.57079632679489656),
            new Complex(0.0, -2.0).asinh(),
            1.0e-15);

        assertComplexClose(
            new Complex(-1.31695789692481635, -1.57079632679489611),
            new Complex(-0.0, -2.0).asinh(),
            1.0e-15);

        assertComplexClose(
            new Complex(0.54930614433405489, 1.57079632679489656),
            new Complex(2.0, 0.0).atanh(),
            1.0e-15);

        assertComplexClose(
            new Complex(0.54930614433405489, -1.57079632679489656),
            new Complex(2.0, -0.0).atanh(),
            1.0e-15);

        assertComplexClose(
            new Complex(0.0, 1.04719755119659763),
            new Complex(0.5, 0.0).acosh(),
            1.0e-15);

        assertComplexClose(
            new Complex(0.0, -1.04719755119659763),
            new Complex(0.5, -0.0).acosh(),
            1.0e-15);
    }

    @Test
    void atanMatchesStdComplexImaginaryAxisLipSelection() {
        assertComplexClose(
            new Complex(1.57079632679489656, 0.54930614433405489),
            new Complex(0.0, 2.0).atan(),
            1.0e-15);

        assertComplexClose(
            new Complex(-1.57079632679489656, 0.54930614433405489),
            new Complex(-0.0, 2.0).atan(),
            1.0e-15);
    }

    @Test
    void logMatchesStdComplexExceptionalValues() {
        Complex logNegativeZero = new Complex(-0.0, 0.0).log();
        assertTrue(Double.isInfinite(logNegativeZero.real()));
        assertTrue(logNegativeZero.real() < 0.0);
        assertEquals(Math.PI, logNegativeZero.imag(), 0.0);

        Complex logLowerLipZero = new Complex(0.0, -0.0).log();
        assertTrue(Double.isInfinite(logLowerLipZero.real()));
        assertTrue(logLowerLipZero.real() < 0.0);
        assertNegativeZero(logLowerLipZero.imag());

        Complex logNegativeInfinity = new Complex(Double.NEGATIVE_INFINITY, 2.0).log();
        assertTrue(Double.isInfinite(logNegativeInfinity.real()));
        assertTrue(logNegativeInfinity.real() > 0.0);
        assertEquals(Math.PI, logNegativeInfinity.imag(), 0.0);

        Complex logPositiveInfinity = new Complex(Double.POSITIVE_INFINITY, 2.0).log();
        assertTrue(Double.isInfinite(logPositiveInfinity.real()));
        assertTrue(logPositiveInfinity.real() > 0.0);
        assertPositiveZero(logPositiveInfinity.imag());

        Complex logNaNInfinity = new Complex(Double.NaN, Double.POSITIVE_INFINITY).log();
        assertTrue(Double.isInfinite(logNaNInfinity.real()));
        assertTrue(logNaNInfinity.real() > 0.0);
        assertTrue(Double.isNaN(logNaNInfinity.imag()));
    }

    @Test
    void divideAndReciprocalMatchStdComplexExceptionalValues() {
        Complex divideByZero = new Complex(1.0, 2.0).div(Complex.ZERO);
        assertTrue(Double.isInfinite(divideByZero.real()));
        assertTrue(divideByZero.real() > 0.0);
        assertTrue(Double.isInfinite(divideByZero.imag()));
        assertTrue(divideByZero.imag() > 0.0);

        Complex divideByInfinitePair = new Complex(1.0, 2.0).div(new Complex(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY));
        assertPositiveZero(divideByInfinitePair.real());
        assertPositiveZero(divideByInfinitePair.imag());

        Complex infiniteNumerator = new Complex(Double.POSITIVE_INFINITY, 2.0).div(new Complex(1.0, 2.0));
        assertTrue(Double.isInfinite(infiniteNumerator.real()));
        assertTrue(infiniteNumerator.real() > 0.0);
        assertTrue(Double.isInfinite(infiniteNumerator.imag()));
        assertTrue(infiniteNumerator.imag() < 0.0);

        Complex divideByInfiniteReal = new Complex(1.0, 2.0).div(new Complex(Double.POSITIVE_INFINITY, 0.0));
        assertPositiveZero(divideByInfiniteReal.real());
        assertPositiveZero(divideByInfiniteReal.imag());

        Complex reciprocalZero = Complex.ONE.div(Complex.ZERO);
        assertTrue(Double.isInfinite(reciprocalZero.real()));
        assertTrue(reciprocalZero.real() > 0.0);
        assertTrue(Double.isNaN(reciprocalZero.imag()));
        assertComplexExceptionalEqual(reciprocalZero, Complex.ZERO.recip());

        Complex reciprocalNegativeZero = Complex.ONE.div(new Complex(-0.0, 0.0));
        assertTrue(Double.isInfinite(reciprocalNegativeZero.real()));
        assertTrue(reciprocalNegativeZero.real() < 0.0);
        assertTrue(Double.isNaN(reciprocalNegativeZero.imag()));
        assertComplexExceptionalEqual(reciprocalNegativeZero, new Complex(-0.0, 0.0).recip());

        Complex reciprocalInfinitePair = new Complex(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY).recip();
        assertPositiveZero(reciprocalInfinitePair.real());
        assertNegativeZero(reciprocalInfinitePair.imag());
    }

    @Test
    void inverseFunctionsMatchStdComplexExceptionalValues() {
        Complex asinhInfiniteReal = new Complex(Double.POSITIVE_INFINITY, 1.0).asinh();
        assertTrue(Double.isInfinite(asinhInfiniteReal.real()));
        assertTrue(asinhInfiniteReal.real() > 0.0);
        assertPositiveZero(asinhInfiniteReal.imag());

        Complex asinhInfiniteImag = new Complex(1.0, Double.POSITIVE_INFINITY).asinh();
        assertTrue(Double.isInfinite(asinhInfiniteImag.real()));
        assertTrue(asinhInfiniteImag.real() > 0.0);
        assertEquals(Math.PI / 2.0, asinhInfiniteImag.imag(), 0.0);

        Complex asinhInfinitePair = new Complex(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY).asinh();
        assertTrue(Double.isInfinite(asinhInfinitePair.real()));
        assertTrue(asinhInfinitePair.real() > 0.0);
        assertEquals(Math.PI / 4.0, asinhInfinitePair.imag(), 0.0);

        Complex asinhNaNZero = new Complex(Double.NaN, 0.0).asinh();
        assertTrue(Double.isNaN(asinhNaNZero.real()));
        assertPositiveZero(asinhNaNZero.imag());

        Complex acoshInfiniteReal = new Complex(Double.POSITIVE_INFINITY, 1.0).acosh();
        assertTrue(Double.isInfinite(acoshInfiniteReal.real()));
        assertTrue(acoshInfiniteReal.real() > 0.0);
        assertPositiveZero(acoshInfiniteReal.imag());

        Complex acoshNegativeInfiniteReal = new Complex(Double.NEGATIVE_INFINITY, 1.0).acosh();
        assertTrue(Double.isInfinite(acoshNegativeInfiniteReal.real()));
        assertTrue(acoshNegativeInfiniteReal.real() > 0.0);
        assertEquals(Math.PI, acoshNegativeInfiniteReal.imag(), 0.0);

        Complex acoshInfiniteImag = new Complex(1.0, Double.POSITIVE_INFINITY).acosh();
        assertTrue(Double.isInfinite(acoshInfiniteImag.real()));
        assertTrue(acoshInfiniteImag.real() > 0.0);
        assertEquals(Math.PI / 2.0, acoshInfiniteImag.imag(), 0.0);

        Complex acoshNaNInfinite = new Complex(Double.NaN, Double.POSITIVE_INFINITY).acosh();
        assertTrue(Double.isInfinite(acoshNaNInfinite.real()));
        assertTrue(acoshNaNInfinite.real() > 0.0);
        assertTrue(Double.isNaN(acoshNaNInfinite.imag()));

        Complex atanhUnit = new Complex(1.0, 0.0).atanh();
        assertTrue(Double.isInfinite(atanhUnit.real()));
        assertTrue(atanhUnit.real() > 0.0);
        assertPositiveZero(atanhUnit.imag());

        Complex atanhInfiniteReal = new Complex(Double.POSITIVE_INFINITY, 2.0).atanh();
        assertPositiveZero(atanhInfiniteReal.real());
        assertEquals(Math.PI / 2.0, atanhInfiniteReal.imag(), 0.0);

        Complex atanhZeroNaN = new Complex(0.0, Double.NaN).atanh();
        assertPositiveZero(atanhZeroNaN.real());
        assertTrue(Double.isNaN(atanhZeroNaN.imag()));

        Complex acosInfiniteReal = new Complex(Double.POSITIVE_INFINITY, 2.0).acos();
        assertPositiveZero(acosInfiniteReal.real());
        assertTrue(Double.isInfinite(acosInfiniteReal.imag()));
        assertTrue(acosInfiniteReal.imag() < 0.0);

        Complex acosNegativeInfiniteReal = new Complex(Double.NEGATIVE_INFINITY, 2.0).acos();
        assertEquals(Math.PI, acosNegativeInfiniteReal.real(), 0.0);
        assertTrue(Double.isInfinite(acosNegativeInfiniteReal.imag()));
        assertTrue(acosNegativeInfiniteReal.imag() < 0.0);

        Complex acosInfiniteImag = new Complex(1.0, Double.POSITIVE_INFINITY).acos();
        assertEquals(Math.PI / 2.0, acosInfiniteImag.real(), 0.0);
        assertTrue(Double.isInfinite(acosInfiniteImag.imag()));
        assertTrue(acosInfiniteImag.imag() < 0.0);

        Complex atanPositiveZeroInfinity = new Complex(0.0, Double.POSITIVE_INFINITY).atan();
        assertEquals(Math.PI / 2.0, atanPositiveZeroInfinity.real(), 0.0);
        assertPositiveZero(atanPositiveZeroInfinity.imag());

        Complex atanNegativeZeroInfinity = new Complex(-0.0, Double.POSITIVE_INFINITY).atan();
        assertEquals(-Math.PI / 2.0, atanNegativeZeroInfinity.real(), 0.0);
        assertPositiveZero(atanNegativeZeroInfinity.imag());
    }

    @Test
    void elementaryFunctionsMatchStdComplexExceptionalValues() {
        Complex expInfiniteNaN = new Complex(Double.POSITIVE_INFINITY, Double.NaN).exp();
        assertTrue(Double.isInfinite(expInfiniteNaN.real()));
        assertTrue(expInfiniteNaN.real() > 0.0);
        assertTrue(Double.isNaN(expInfiniteNaN.imag()));

        Complex expZeroInfinity = new Complex(0.0, Double.POSITIVE_INFINITY).exp();
        assertTrue(Double.isNaN(expZeroInfinity.real()));
        assertTrue(Double.isNaN(expZeroInfinity.imag()));

        Complex expNegativeInfinity = new Complex(Double.NEGATIVE_INFINITY, 2.0).exp();
        assertNegativeZero(expNegativeInfinity.real());
        assertPositiveZero(expNegativeInfinity.imag());

        Complex expNaNZero = new Complex(Double.NaN, 0.0).exp();
        assertTrue(Double.isNaN(expNaNZero.real()));
        assertPositiveZero(expNaNZero.imag());

        Complex sinhInfiniteNaN = new Complex(Double.POSITIVE_INFINITY, Double.NaN).sinh();
        assertTrue(Double.isInfinite(sinhInfiniteNaN.real()));
        assertTrue(sinhInfiniteNaN.real() > 0.0);
        assertTrue(Double.isNaN(sinhInfiniteNaN.imag()));

        Complex sinhZeroInfinity = new Complex(0.0, Double.POSITIVE_INFINITY).sinh();
        assertZero(sinhZeroInfinity.real());
        assertTrue(Double.isNaN(sinhZeroInfinity.imag()));

        Complex sinhInfiniteOne = new Complex(Double.POSITIVE_INFINITY, 1.0).sinh();
        assertTrue(Double.isInfinite(sinhInfiniteOne.real()));
        assertTrue(sinhInfiniteOne.real() > 0.0);
        assertTrue(Double.isInfinite(sinhInfiniteOne.imag()));
        assertTrue(sinhInfiniteOne.imag() > 0.0);

        Complex sinhNaNZero = new Complex(Double.NaN, 0.0).sinh();
        assertTrue(Double.isNaN(sinhNaNZero.real()));
        assertPositiveZero(sinhNaNZero.imag());

        Complex coshInfiniteNaN = new Complex(Double.POSITIVE_INFINITY, Double.NaN).cosh();
        assertTrue(Double.isInfinite(coshInfiniteNaN.real()));
        assertTrue(coshInfiniteNaN.real() > 0.0);
        assertTrue(Double.isNaN(coshInfiniteNaN.imag()));

        Complex coshZeroInfinity = new Complex(0.0, Double.POSITIVE_INFINITY).cosh();
        assertTrue(Double.isNaN(coshZeroInfinity.real()));
        assertZero(coshZeroInfinity.imag());

        Complex coshInfiniteZero = new Complex(Double.POSITIVE_INFINITY, 0.0).cosh();
        assertTrue(Double.isInfinite(coshInfiniteZero.real()));
        assertTrue(coshInfiniteZero.real() > 0.0);
        assertPositiveZero(coshInfiniteZero.imag());

        Complex coshNaNZero = new Complex(Double.NaN, 0.0).cosh();
        assertTrue(Double.isNaN(coshNaNZero.real()));
        assertPositiveZero(coshNaNZero.imag());

        Complex sinInfiniteReal = new Complex(Double.POSITIVE_INFINITY, 1.0).sin();
        assertTrue(Double.isNaN(sinInfiniteReal.real()));
        assertTrue(Double.isNaN(sinInfiniteReal.imag()));

        Complex sinInfiniteImag = new Complex(1.0, Double.POSITIVE_INFINITY).sin();
        assertTrue(Double.isInfinite(sinInfiniteImag.real()));
        assertTrue(sinInfiniteImag.real() > 0.0);
        assertTrue(Double.isInfinite(sinInfiniteImag.imag()));
        assertTrue(sinInfiniteImag.imag() > 0.0);

        Complex sinNaNZero = new Complex(Double.NaN, 0.0).sin();
        assertTrue(Double.isNaN(sinNaNZero.real()));
        assertPositiveZero(sinNaNZero.imag());

        Complex cosInfiniteReal = new Complex(Double.POSITIVE_INFINITY, 1.0).cos();
        assertTrue(Double.isNaN(cosInfiniteReal.real()));
        assertTrue(Double.isNaN(cosInfiniteReal.imag()));

        Complex cosInfiniteImag = new Complex(1.0, Double.POSITIVE_INFINITY).cos();
        assertTrue(Double.isInfinite(cosInfiniteImag.real()));
        assertTrue(cosInfiniteImag.real() > 0.0);
        assertTrue(Double.isInfinite(cosInfiniteImag.imag()));
        assertTrue(cosInfiniteImag.imag() < 0.0);

        Complex cosNaNZero = new Complex(Double.NaN, 0.0).cos();
        assertTrue(Double.isNaN(cosNaNZero.real()));
        assertNegativeZero(cosNaNZero.imag());
    }

    @Test
    void axisFastPathsPreserveSignedZeroSemantics() {
        Complex tanUpper = new Complex(0.75, 0.0).tan();
        assertEquals(Math.tan(0.75), tanUpper.real(), 0.0);
        assertPositiveZero(tanUpper.imag());

        Complex tanLower = new Complex(0.75, -0.0).tan();
        assertEquals(Math.tan(0.75), tanLower.real(), 0.0);
        assertNegativeZero(tanLower.imag());

        Complex tanhUpper = new Complex(0.75, 0.0).tanh();
        assertEquals(Math.tanh(0.75), tanhUpper.real(), 0.0);
        assertPositiveZero(tanhUpper.imag());

        Complex tanhLower = new Complex(0.75, -0.0).tanh();
        assertEquals(Math.tanh(0.75), tanhLower.real(), 0.0);
        assertNegativeZero(tanhLower.imag());

        assertComplexClose(
            new Complex(Math.cos(0.75), Math.sin(0.75)),
            new Complex(0.0, 0.75).exp(),
            1.0e-15);
    }

    private static void assertComplexClose(Complex expected, Complex actual, double tolerance) {
        assertEquals(expected.real(), actual.real(), tolerance);
        assertEquals(expected.imag(), actual.imag(), tolerance);
    }

    private static void assertComplexExceptionalEqual(Complex expected, Complex actual) {
        assertSameDouble(expected.real(), actual.real());
        assertSameDouble(expected.imag(), actual.imag());
    }

    private static void assertSameDouble(double expected, double actual) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual));
            return;
        }
        if (expected == 0.0 && actual == 0.0) {
            assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual));
            return;
        }
        assertEquals(expected, actual, 0.0);
    }

    private static void assertZero(double value) {
        assertTrue(value == 0.0);
    }

    private static void assertPositiveZero(double value) {
        assertTrue(value == 0.0);
        assertEquals(0L, Double.doubleToRawLongBits(value));
    }

    private static void assertNegativeZero(double value) {
        assertTrue(value == 0.0);
        assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(value));
    }
}