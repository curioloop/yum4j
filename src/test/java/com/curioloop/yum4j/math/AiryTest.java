package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import java.util.function.IntToDoubleFunction;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AiryTest {

    @Test
    void spotValuesMatchBoostRepresentativePoints() {
        assertClose(0.35502805388781723926, Airy.ai(0.0), 5.0e-15);
        assertClose(0.61492662744600073515, Airy.bi(0.0), 5.0e-15);
        assertClose(-0.25881940379280679841, Airy.aiPrime(0.0), 5.0e-15);
        assertClose(0.44828835735382635791, Airy.biPrime(0.0), 5.0e-15);

        assertClose(0.034924130423274379135, Airy.ai(2.0), 5.0e-13);
        assertClose(3.2980949999782147103, Airy.bi(2.0), 5.0e-13);
        assertClose(-0.053090384433653631704, Airy.aiPrime(2.0), 5.0e-13);
        assertClose(4.1006820499328898894, Airy.biPrime(2.0), 5.0e-13);

        assertClose(-0.37553382314043191193, Airy.ai(-3.5), 5.0e-13);
        assertClose(0.16893983748105861184, Airy.bi(-3.5), 5.0e-13);
        assertClose(-0.34344343345404814629, Airy.aiPrime(-3.5), 5.0e-13);
        assertClose(-0.69311628490728880175, Airy.biPrime(-3.5), 5.0e-13);
    }

    @Test
    void wronskianMatchesOneOverPiAcrossRepresentativePoints() {
        double[] xs = {-30.25, -2.0, -0.5, 0.0, 2.0, 30.5};
        for (double x : xs) {
            double wronskian = Airy.ai(x) * Airy.biPrime(x) - Airy.aiPrime(x) * Airy.bi(x);
            assertClose(1.0 / Math.PI, wronskian, 5.0e-12);
        }
    }

    @Test
    void zeroLocatorsAreRootedAndMonotone() {
        double ai1 = Airy.aiZero(1);
        double ai2 = Airy.aiZero(2);
        double bi1 = Airy.biZero(1);
        double bi2 = Airy.biZero(2);

        assertTrue(ai1 < 0.0 && ai2 < ai1);
        assertTrue(bi1 < 0.0 && bi2 < bi1);
        assertTrue(Math.abs(Airy.ai(ai1)) <= 1.0e-11);
        assertTrue(Math.abs(Airy.ai(ai2)) <= 1.0e-11);
        assertTrue(Math.abs(Airy.bi(bi1)) <= 1.0e-11);
        assertTrue(Math.abs(Airy.bi(bi2)) <= 1.0e-11);
    }

    @Test
    void zeroSequenceFunctionsDelegateToScalarLocators() {
        IntToDoubleFunction aiZeros = Airy.aiZeros();
        IntToDoubleFunction biZeros = Airy.biZeros();

        assertClose(Airy.aiZero(1), aiZeros.applyAsDouble(1), 0.0);
        assertClose(Airy.aiZero(2), aiZeros.applyAsDouble(2), 0.0);
        assertClose(Airy.biZero(1), biZeros.applyAsDouble(1), 0.0);
        assertClose(Airy.biZero(2), biZeros.applyAsDouble(2), 0.0);
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> Airy.ai(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Airy.bi(Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> Airy.aiPrime(Double.NEGATIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> Airy.aiZero(0));
        assertThrows(IllegalArgumentException.class, () -> Airy.aiZero(-1));
        assertThrows(IllegalArgumentException.class, () -> Airy.biZero(0));
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(expected - actual);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}