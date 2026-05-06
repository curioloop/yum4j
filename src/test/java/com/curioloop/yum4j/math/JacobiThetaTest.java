package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JacobiThetaTest {

    @Test
    void smallQAsymptoticsMatchLeadingTerms() {
        double q = 1.0e-6;
        double z = 0.4;
        double qQuarter = Math.pow(q, 0.25);

        assertClose(2.0 * qQuarter * Math.sin(z), JacobiTheta.theta1(z, q), 1.0e-9);
        assertClose(2.0 * qQuarter * Math.cos(z), JacobiTheta.theta2(z, q), 1.0e-9);
        assertClose(2.0 * q * Math.cos(2.0 * z), JacobiTheta.theta3MinusOne(z, q), 1.0e-9);
        assertClose(-2.0 * q * Math.cos(2.0 * z), JacobiTheta.theta4MinusOne(z, q), 1.0e-9);
    }

    @Test
    void modularZeroRelationsHold() {
        double tau = 0.35;
        double inverseTau = 1.0 / tau;
        double scale = 1.0 / Math.sqrt(tau);

        assertClose(JacobiTheta.theta4Tau(0.0, inverseTau) * scale, JacobiTheta.theta2Tau(0.0, tau), 5.0e-13);
        assertClose(JacobiTheta.theta3Tau(0.0, inverseTau) * scale, JacobiTheta.theta3Tau(0.0, tau), 5.0e-13);
        assertClose(JacobiTheta.theta2Tau(0.0, inverseTau) * scale, JacobiTheta.theta4Tau(0.0, tau), 5.0e-13);
    }

    @Test
    void symmetryAndMinusOneIdentitiesHold() {
        double q = 0.2;
        double z = 0.5;

        assertClose(-JacobiTheta.theta1(z, q), JacobiTheta.theta1(-z, q), 5.0e-13);
        assertClose(JacobiTheta.theta2(z, q), JacobiTheta.theta2(-z, q), 5.0e-13);
        assertClose(JacobiTheta.theta3(z, q), JacobiTheta.theta3(-z, q), 5.0e-13);
        assertClose(JacobiTheta.theta4(z, q), JacobiTheta.theta4(-z, q), 5.0e-13);
        assertClose(JacobiTheta.theta3(z, q) - 1.0, JacobiTheta.theta3MinusOne(z, q), 5.0e-13);
        assertClose(JacobiTheta.theta4(z, q) - 1.0, JacobiTheta.theta4MinusOne(z, q), 5.0e-13);
    }

    @Test
    void qAndTauParameterizationsAgree() {
        double tau = 0.75;
        double q = Math.exp(-Math.PI * tau);
        double z = 0.4;

        assertClose(JacobiTheta.theta1(z, q), JacobiTheta.theta1Tau(z, tau), 5.0e-13);
        assertClose(JacobiTheta.theta2(z, q), JacobiTheta.theta2Tau(z, tau), 5.0e-13);
        assertClose(JacobiTheta.theta3(z, q), JacobiTheta.theta3Tau(z, tau), 5.0e-13);
        assertClose(JacobiTheta.theta4(z, q), JacobiTheta.theta4Tau(z, tau), 5.0e-13);
    }

    @Test
    void tauSplitRemainsContinuousAcrossUnity() {
        double z = 0.5;
        double below = 0.999999999999;
        double above = 1.000000000001;

        assertClose(JacobiTheta.theta1Tau(z, below), JacobiTheta.theta1Tau(z, above), 5.0e-12);
        assertClose(JacobiTheta.theta2Tau(z, below), JacobiTheta.theta2Tau(z, above), 5.0e-12);
        assertClose(JacobiTheta.theta3Tau(z, below), JacobiTheta.theta3Tau(z, above), 5.0e-12);
        assertClose(JacobiTheta.theta4Tau(z, below), JacobiTheta.theta4Tau(z, above), 5.0e-12);
    }

    @Test
    void largeTauArgumentsRespectBoostStylePeriodReductions() {
        double tau = 0.6;
        double base = 0.3;

        assertClose(JacobiTheta.theta1Tau(base, tau), JacobiTheta.theta1Tau(1000.0 * Math.PI + base, tau), 5.0e-12);
        assertClose(-JacobiTheta.theta1Tau(base, tau), JacobiTheta.theta1Tau(-1001.0 * Math.PI + base, tau), 5.0e-12);
        assertClose(JacobiTheta.theta2Tau(base, tau), JacobiTheta.theta2Tau(1000.0 * Math.PI + base, tau), 5.0e-12);
        assertClose(-JacobiTheta.theta2Tau(base, tau), JacobiTheta.theta2Tau(-1001.0 * Math.PI + base, tau), 5.0e-12);
        assertClose(JacobiTheta.theta3Tau(base, tau), JacobiTheta.theta3Tau(1000.0 * Math.PI + base, tau), 5.0e-12);
        assertClose(JacobiTheta.theta3Tau(base, tau), JacobiTheta.theta3Tau(-1001.0 * Math.PI + base, tau), 5.0e-12);
        assertClose(JacobiTheta.theta4Tau(base, tau), JacobiTheta.theta4Tau(1000.0 * Math.PI + base, tau), 5.0e-12);
        assertClose(JacobiTheta.theta4Tau(base, tau), JacobiTheta.theta4Tau(-1001.0 * Math.PI + base, tau), 5.0e-12);
    }

    @Test
    void invalidInputsAreRejected() {
        assertEquals(0.0, JacobiTheta.theta1(0.0, 0.2), 0.0);
        assertThrows(IllegalArgumentException.class, () -> JacobiTheta.theta1(0.2, 0.0));
        assertThrows(IllegalArgumentException.class, () -> JacobiTheta.theta2(0.2, 1.0));
        assertThrows(IllegalArgumentException.class, () -> JacobiTheta.theta3Tau(0.2, 0.0));
        assertThrows(IllegalArgumentException.class, () -> JacobiTheta.theta4Tau(Double.NaN, 0.5));
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}
