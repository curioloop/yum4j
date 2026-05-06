package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Beta;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NonCentralBetaDistributionTest {

    @Test
    void centralCaseReducesToClosedFormBetaStatistics() {
        NonCentralBetaDistribution distribution = new NonCentralBetaDistribution(2.5, 4.0, 0.0);
        double alphaPlusBeta = 6.5;

        assertEquals(2.5 / alphaPlusBeta, distribution.mean(), 1.0e-15);
        assertEquals(2.5 * 4.0 / (alphaPlusBeta * alphaPlusBeta * (alphaPlusBeta + 1.0)), distribution.variance(), 1.0e-15);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
    }

    @Test
    void staticKernelHandlesBoundariesAndCentralFallback() {
        double a = 2.0;
        double b = 5.0;
        double x = 0.3;
        double p = 0.4;
        double lambda = 5.0;

        assertEquals(0.0, NonCentralBetaDistribution.cdf(2.5, 3.5, lambda, 0.0), 0.0);
        assertEquals(1.0, NonCentralBetaDistribution.cdf(2.5, 3.5, lambda, 1.0), 0.0);
        assertEquals(1.0, NonCentralBetaDistribution.ccdf(2.5, 3.5, lambda, 0.0), 0.0);
        assertEquals(0.0, NonCentralBetaDistribution.ccdf(2.5, 3.5, lambda, 1.0), 0.0);
        assertEquals(0.0, NonCentralBetaDistribution.pdf(2.5, 3.5, lambda, 0.0), 0.0);
        assertEquals(0.0, NonCentralBetaDistribution.pdf(2.5, 3.5, lambda, 1.0), 0.0);
        assertEquals(0.0, NonCentralBetaDistribution.quantile(2.5, 3.5, lambda, 0.0), 0.0);
        assertEquals(1.0, NonCentralBetaDistribution.quantile(2.5, 3.5, lambda, 1.0), 0.0);
        assertEquals(0.0, NonCentralBetaDistribution.quantileUpperTail(2.5, 3.5, lambda, 1.0), 0.0);
        assertEquals(1.0, NonCentralBetaDistribution.quantileUpperTail(2.5, 3.5, lambda, 0.0), 0.0);

        assertClose(Beta.ibeta(a, b, x), NonCentralBetaDistribution.cdf(a, b, 0.0, x), 5.0e-15);
        assertClose(Beta.ibetac(a, b, x), NonCentralBetaDistribution.ccdf(a, b, 0.0, x), 5.0e-15);
        assertClose(Beta.ibetaDerivative(a, b, x), NonCentralBetaDistribution.pdf(a, b, 0.0, x), 5.0e-15);
        assertClose(Beta.ibetaInv(a, b, p), NonCentralBetaDistribution.quantile(a, b, 0.0, p), 5.0e-15);
        assertClose(Beta.ibetacInv(a, b, p), NonCentralBetaDistribution.quantileUpperTail(a, b, 0.0, p), 5.0e-15);
    }

    @Test
    void evaluationSurfaceDelegatesToMergedKernel() {
        NonCentralBetaDistribution distribution = new NonCentralBetaDistribution(2.5, 4.0, 3.0);
        double x = 0.45;

        assertEquals(NonCentralBetaDistribution.pdf(2.5, 4.0, 3.0, x), distribution.pdf(x), 1.0e-15);
        assertEquals(NonCentralBetaDistribution.cdf(2.5, 4.0, 3.0, x), distribution.cdf(x), 1.0e-15);
        assertEquals(NonCentralBetaDistribution.ccdf(2.5, 4.0, 3.0, x), distribution.ccdf(x), 1.0e-15);
        assertEquals(x, distribution.quantile(distribution.cdf(x)), 1.0e-8);
        assertEquals(x, distribution.quantileUpperTail(distribution.ccdf(x)), 1.0e-8);
    }

    @Test
    void staticKernelRoundTripsRepresentativeCases() {
        assertRoundTrip(2.5, 3.5, 5.0, 0.45147869946085711, 5.0e-11);
        assertRoundTrip(0.4, 0.35, 0.5, 0.521188628783254, 2.0e-10);
        assertRoundTrip(100.0, 3.0, 63.0, 0.94999999999999996, 5.0e-10);
    }

    @Test
    void modeStaysInsideUnitIntervalAndLocallyMaximizesDensity() {
        NonCentralBetaDistribution distribution = new NonCentralBetaDistribution(2.5, 4.0, 3.0);
        double mode = distribution.mode();
        double step = Math.max(1.0e-6, mode * 1.0e-6);

        assertTrue(mode >= 0.0 && mode <= 1.0);
        assertTrue(distribution.pdf(mode) >= distribution.pdf(Math.max(0.0, mode - step)));
        assertTrue(distribution.pdf(mode) >= distribution.pdf(Math.min(1.0, mode + step)));
    }

    @Test
    void invalidArgumentsAndUndefinedHigherMomentsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> NonCentralBetaDistribution.cdf(-1.0, 2.0, 0.5, 0.25));
        assertThrows(IllegalArgumentException.class, () -> NonCentralBetaDistribution.cdf(1.0, 0.0, 0.5, 0.25));
        assertThrows(IllegalArgumentException.class, () -> NonCentralBetaDistribution.cdf(1.0, 2.0, -0.5, 0.25));
        assertThrows(IllegalArgumentException.class, () -> NonCentralBetaDistribution.cdf(1.0, 2.0, 0.5, -0.1));
        assertThrows(IllegalArgumentException.class, () -> NonCentralBetaDistribution.quantile(1.0, 2.0, 0.5, -0.1));
        assertThrows(IllegalArgumentException.class, () -> NonCentralBetaDistribution.quantileUpperTail(1.0, 2.0, 0.5, 1.1));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralBetaDistribution(0.0, 1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralBetaDistribution(1.0, 0.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralBetaDistribution(1.0, 1.0, -1.0));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralBetaDistribution(1.0, 1.0, 0.0).cdf(-0.1));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralBetaDistribution(1.0, 1.0, 0.0).quantile(1.1));
        assertThrows(ArithmeticException.class, () -> new NonCentralBetaDistribution(1.0, 1.0, 0.0).skewness());
        assertThrows(ArithmeticException.class, () -> new NonCentralBetaDistribution(1.0, 1.0, 0.0).kurtosisExcess());
        assertThrows(ArithmeticException.class, () -> new NonCentralBetaDistribution(1.0, 1.0, 0.0).kurtosis());
    }

    private static void assertRoundTrip(double a, double b, double nonCentrality, double x, double tolerance) {
        double p = NonCentralBetaDistribution.cdf(a, b, nonCentrality, x);
        double q = NonCentralBetaDistribution.ccdf(a, b, nonCentrality, x);

        assertClose(1.0, p + q, 5.0e-13);
        assertClose(x, NonCentralBetaDistribution.quantile(a, b, nonCentrality, p), tolerance);
        assertClose(x, NonCentralBetaDistribution.quantileUpperTail(a, b, nonCentrality, q), tolerance);
        assertTrue(NonCentralBetaDistribution.pdf(a, b, nonCentrality, x) >= 0.0);
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}