package com.curioloop.yum4j.stats;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NonCentralFDistributionTest {

    @Test
    void accessorsMatchBoostRangeAndSupport() {
        NonCentralFDistribution distribution = new NonCentralFDistribution(6.0, 12.0, 4.0);

        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
    }

    @Test
    void evaluationSurfaceMatchesNonCentralBetaTransform() {
        NonCentralFDistribution distribution = new NonCentralFDistribution(6.0, 12.0, 4.0);
        double x = 1.1;
        double y = x * 6.0 / 12.0;
        double betaX = y / (1.0 + y);
        double derivative = (6.0 / 12.0) / Math.pow(1.0 + y, 2.0);

        assertEquals(NonCentralBetaDistribution.pdf(3.0, 6.0, 4.0, betaX) * derivative, distribution.pdf(x), 1.0e-15);
        assertEquals(NonCentralBetaDistribution.cdf(3.0, 6.0, 4.0, betaX), distribution.cdf(x), 1.0e-15);
        assertEquals(NonCentralBetaDistribution.ccdf(3.0, 6.0, 4.0, betaX), distribution.ccdf(x), 1.0e-15);
        assertEquals(x, distribution.quantile(distribution.cdf(x)), 1.0e-8);
        assertEquals(x, distribution.quantileUpperTail(distribution.ccdf(x)), 1.0e-8);
    }

    @Test
    void momentsMatchBoostClosedForms() {
        NonCentralFDistribution distribution = new NonCentralFDistribution(6.0, 12.0, 4.0);

        assertEquals(12.0 * 10.0 / (6.0 * 10.0), distribution.mean(), 1.0e-15);
        assertEquals(2.4, distribution.variance(), 1.0e-14);
        assertEquals(2.868876552746235, distribution.skewness(), 1.0e-12);
        assertEquals(27.444444444444443, distribution.kurtosis(), 1.0e-11);
        assertEquals(24.444444444444443, distribution.kurtosisExcess(), 1.0e-11);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
    }

    @Test
    void modeIsLocallyMaximalAndBoundarySemanticsHold() {
        NonCentralFDistribution distribution = new NonCentralFDistribution(6.0, 12.0, 4.0);
        double mode = distribution.mode();
        double step = Math.max(1.0e-6, mode * 1.0e-6);

        assertTrue(mode >= 0.0);
        assertTrue(distribution.pdf(mode) >= distribution.pdf(Math.max(0.0, mode - step)));
        assertTrue(distribution.pdf(mode) >= distribution.pdf(mode + step));
        assertEquals(0.0, distribution.cdf(0.0), 0.0);
        assertEquals(1.0, distribution.ccdf(0.0), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantile(1.0));
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantileUpperTail(0.0));
    }

    @Test
    void invalidArgumentsAndUndefinedMomentsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new NonCentralFDistribution(0.0, 1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralFDistribution(1.0, 0.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralFDistribution(1.0, 1.0, -1.0));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralFDistribution(1.0, 1.0, 0.0).cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralFDistribution(1.0, 1.0, 0.0).quantile(1.1));
        assertThrows(ArithmeticException.class, () -> new NonCentralFDistribution(6.0, 2.0, 4.0).mean());
        assertThrows(ArithmeticException.class, () -> new NonCentralFDistribution(6.0, 4.0, 4.0).variance());
        assertThrows(ArithmeticException.class, () -> new NonCentralFDistribution(6.0, 6.0, 4.0).skewness());
        assertThrows(ArithmeticException.class, () -> new NonCentralFDistribution(6.0, 8.0, 4.0).kurtosisExcess());
        assertThrows(ArithmeticException.class, () -> new NonCentralFDistribution(6.0, 8.0, 4.0).kurtosis());
    }
}