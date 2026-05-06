package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZetaTest {

    @Test
    void specialValuesMatchKnownClosedForms() {
        assertClose(Math.PI * Math.PI / 6.0, Zeta.zeta(2.0), 5.0e-15);
        assertClose(1.2020569031595942, Zeta.zeta(3.0), 5.0e-15);
        assertClose(-0.5, Zeta.zeta(0.0), 5.0e-15);
        assertClose(-1.0 / 12.0, Zeta.zeta(-1.0), 5.0e-15);
        assertClose(1.0 / 120.0, Zeta.zeta(-3.0), 5.0e-15);
    }

    @Test
    void trivialZerosAreReturnedExactly() {
        assertEquals(0.0, Zeta.zeta(-2.0), 0.0);
        assertEquals(0.0, Zeta.zeta(-4.0), 0.0);
        assertEquals(0.0, Zeta.zeta(-20.0), 0.0);
    }

    @Test
    void largeNegativeEvenIntegerReturnsPositiveZero() {
        double value = Zeta.zeta(-300.0);
        assertEquals(0.0, value, 0.0);
        assertEquals(0L, Double.doubleToRawLongBits(value));
    }

    @Test
    void largeNegativeArgumentsNearTrivialZeroOverflow() {
        double[] arguments = {
            -300.0 + Math.scalb(1.0, -40),
            -300.0 - Math.scalb(1.0, -40),
            -300.0 + Math.scalb(1.0, -20),
            -300.0 - Math.scalb(1.0, -20),
            -300.5,
            -301.0
        };

        for (double argument : arguments) {
            ArithmeticException exception = assertThrows(ArithmeticException.class, () -> Zeta.zeta(argument));
            assertTrue(exception.getMessage().contains("zeta overflow"),
                "unexpected message for s=" + argument + ": " + exception.getMessage());
        }
    }

    @Test
    void largePositiveArgumentsApproachOne() {
        double value = Zeta.zeta(40.0);
        assertTrue(value > 1.0);
        assertTrue(value < 1.0 + Math.pow(2.0, -39.0));
    }

    @Test
    void poleAndNonFiniteInputsAreRejected() {
        assertThrows(ArithmeticException.class, () -> Zeta.zeta(1.0));
        assertThrows(IllegalArgumentException.class, () -> Zeta.zeta(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Zeta.zeta(Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> Zeta.zeta(Double.NEGATIVE_INFINITY));
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(expected - actual);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}