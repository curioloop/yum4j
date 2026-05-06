package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InverseHyperbolicTest {

    @Test
    void asinhMatchesBoostSpotValues() {
        assertRelativeEquals(0.0, InverseHyperbolic.asinh(0.0), 1.0e-15);
        assertRelativeEquals(3.8146827137560308199618503938547256175e-6,
            InverseHyperbolic.asinh(1.0 / 262145.0), 1.0e-15);
        assertRelativeEquals(-3.8146827137560308199618503938547256175e-6,
            InverseHyperbolic.asinh(-1.0 / 262145.0), 1.0e-15);
        assertRelativeEquals(13.169800245332588515868546082651117326,
            InverseHyperbolic.asinh(262145.0), 1.0e-15);
        assertRelativeEquals(-13.169800245332588515868546082651117326,
            InverseHyperbolic.asinh(-262145.0), 1.0e-15);
    }

    @Test
    void acoshMatchesBoostSpotValues() {
        assertRelativeEquals(0.0, InverseHyperbolic.acosh(1.0), 1.0e-15);
        assertRelativeEquals(4.3818703480400669869631326958660371708,
            InverseHyperbolic.acosh(40.0), 1.0e-15);
        assertRelativeEquals(13.169800245325312613765196252265982781,
            InverseHyperbolic.acosh(262145.0), 1.0e-15);
    }

    @Test
    void atanhMatchesBoostSpotValues() {
        assertRelativeEquals(0.0, InverseHyperbolic.atanh(0.0), 1.0e-15);
        assertRelativeEquals(Math.log(2.0), InverseHyperbolic.atanh(3.0 / 5.0), 1.0e-15);
        assertRelativeEquals(-Math.log(2.0), InverseHyperbolic.atanh(-3.0 / 5.0), 1.0e-15);
        assertRelativeEquals(3.8146827137837860779426484245661394028e-6,
            InverseHyperbolic.atanh(1.0 / 262145.0), 1.0e-15);
        assertRelativeEquals(-3.8146827137837860779426484245661394028e-6,
            InverseHyperbolic.atanh(-1.0 / 262145.0), 1.0e-15);
    }

    @Test
    void inverseRelationshipsHoldAcrossRepresentativeGrid() {
        double[] asinhInputs = {-1.0e8, -10.0, -0.5, -1.0e-6, 1.0e-6, 0.5, 10.0, 1.0e8};
        for (double x : asinhInputs) {
            assertRelativeEquals(x, Math.sinh(InverseHyperbolic.asinh(x)), 1.0e-12);
        }

        double[] acoshInputs = {1.0, 1.0 + 1.0e-12, 1.0001, 1.25, 2.0, 40.0, 1.0e8};
        for (double x : acoshInputs) {
            assertRelativeEquals(x, Math.cosh(InverseHyperbolic.acosh(x)), 1.0e-12);
        }

        double[] atanhInputs = {-0.99, -0.75, -0.5, -1.0e-6, 1.0e-6, 0.5, 0.75, 0.99};
        for (double x : atanhInputs) {
            assertRelativeEquals(x, Math.tanh(InverseHyperbolic.atanh(x)), 1.0e-12);
        }
    }

    @Test
    void preservesSignedZeroAndInfinityHandling() {
        assertEquals(Double.doubleToRawLongBits(-0.0),
            Double.doubleToRawLongBits(InverseHyperbolic.asinh(-0.0)));
        assertEquals(Double.doubleToRawLongBits(-0.0),
            Double.doubleToRawLongBits(InverseHyperbolic.atanh(-0.0)));

        assertEquals(Double.POSITIVE_INFINITY, InverseHyperbolic.asinh(Double.POSITIVE_INFINITY));
        assertEquals(Double.NEGATIVE_INFINITY, InverseHyperbolic.asinh(Double.NEGATIVE_INFINITY));
        assertEquals(Double.POSITIVE_INFINITY, InverseHyperbolic.acosh(Double.POSITIVE_INFINITY));
    }

    @Test
    void rejectsDomainErrorsLikeBoost() {
        assertThrows(IllegalArgumentException.class, () -> InverseHyperbolic.asinh(Double.NaN));

        assertThrows(IllegalArgumentException.class, () -> InverseHyperbolic.acosh(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> InverseHyperbolic.acosh(0.5));
        assertThrows(IllegalArgumentException.class, () -> InverseHyperbolic.acosh(-1.0));

        assertThrows(IllegalArgumentException.class, () -> InverseHyperbolic.atanh(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> InverseHyperbolic.atanh(-2.0));
        assertThrows(IllegalArgumentException.class, () -> InverseHyperbolic.atanh(2.0));
    }

    @Test
    void treatsAtanhEndpointsAsOverflow() {
        assertThrows(ArithmeticException.class, () -> InverseHyperbolic.atanh(-1.0));
        assertThrows(ArithmeticException.class, () -> InverseHyperbolic.atanh(1.0));
        assertThrows(ArithmeticException.class, () -> InverseHyperbolic.atanh(Math.nextUp(-1.0)));
        assertThrows(ArithmeticException.class, () -> InverseHyperbolic.atanh(Math.nextDown(1.0)));

        double safeNegative = Math.nextUp(Math.nextUp(-1.0));
        double safePositive = Math.nextDown(Math.nextDown(1.0));
        assertTrue(Double.isFinite(InverseHyperbolic.atanh(safeNegative)));
        assertTrue(Double.isFinite(InverseHyperbolic.atanh(safePositive)));
    }

    private static void assertRelativeEquals(double expected, double actual, double tolerance) {
        assertEquals(expected, actual, tolerance * Math.max(1.0, Math.abs(expected)));
    }
}