package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexBoostParityTest {

    @Test
    void inverseElementaryFunctionsMatchBoostReferencePoints() {
        assertComplexEquals(
            new Complex(0.91173829096848769, -1.0317185344477804),
            new Complex(1.25, -0.75).asin(),
            1.0e-12);

        assertComplexEquals(
            new Complex(0.65905803582640898, 1.0317185344477804),
            new Complex(1.25, -0.75).acos(),
            1.0e-12);

        assertComplexEquals(
            new Complex(-1.2767950250211129, 0.64123733936538418),
            new Complex(-0.5, 1.5).atan(),
            1.0e-12);

        assertComplexEquals(
            new Complex(1.1323936316053083, -0.45327617763879396),
            new Complex(1.25, -0.75).asinh(),
            1.0e-12);

        assertComplexEquals(
            new Complex(1.3287633703408077, 0.14287218227437073),
            new Complex(2.0, 0.25).acosh(),
            1.0e-12);

        assertComplexEquals(
            new Complex(0.20058661813123432, -0.4842544903299662),
            new Complex(0.25, -0.5).atanh(),
            1.0e-12);
    }

    @Test
    void acosMatchesBoostAcrossPrincipalBranchSamples() {
        assertComplexEquals(
            new Complex(Math.PI, -1.7627471740390860),
            new Complex(-3.0, 0.0).acos(),
            1.0e-12);

        assertComplexEquals(
            new Complex(2.0943951023931957, -0.0),
            new Complex(-0.5, 0.0).acos(),
            1.0e-12);

        assertComplexEquals(
            new Complex(1.0471975511965976, -0.0),
            new Complex(0.5, 0.0).acos(),
            1.0e-12);

        assertComplexEquals(
            new Complex(0.14287218227437073, -1.3287633703408077),
            new Complex(2.0, 0.25).acos(),
            1.0e-12);
    }

    private static void assertComplexEquals(Complex expected, Complex actual, double tolerance) {
        assertEquals(expected.real(), actual.real(), tolerance);
        assertEquals(expected.imag(), actual.imag(), tolerance);
    }
}