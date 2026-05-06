package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ExponentialDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        ExponentialDistribution distribution = new ExponentialDistribution(1.6);

        assertEquals(1.6, distribution.lambda(), 0.0);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper());
        assertEquals(Double.MIN_NORMAL, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper());
        assertEquals(0.0, distribution.mode(), 0.0);
        assertEquals(Math.log(2.0) / 1.6, distribution.median(), 1.0e-15);
        assertEquals(1.0 / 1.6, distribution.mean(), 1.0e-15);
        assertEquals(1.0 / (1.6 * 1.6), distribution.variance(), 1.0e-15);
        assertEquals(2.0, distribution.skewness(), 0.0);
        assertEquals(9.0, distribution.kurtosis(), 0.0);
        assertEquals(6.0, distribution.kurtosisExcess(), 0.0);
        assertEquals(1.0 / 1.6, Distributions.standardDeviation(distribution), 1.0e-15);
        assertEquals(1.0, Distributions.coefficientOfVariation(distribution), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostClosedForms() {
        ExponentialDistribution distribution = new ExponentialDistribution(1.6);
        double x = 0.75;

        assertEquals(1.6 * Math.exp(-1.6 * x), distribution.pdf(x), 1.0e-15);
        assertEquals(-Math.expm1(-1.6 * x), distribution.cdf(x), 1.0e-15);
        assertEquals(Math.exp(-1.6 * x), distribution.ccdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);
        assertEquals(distribution.quantileUpperTail(0.2),
            Distributions.quantile(Distributions.complement(distribution, 0.2)), 1.0e-15);
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
        assertEquals(0.0, distribution.pdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(1.0, distribution.cdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.ccdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(distribution.pdf(x) / distribution.ccdf(x), Distributions.hazard(distribution, x), 1.0e-15);
        assertEquals(-Math.log(distribution.ccdf(x)), Distributions.chf(distribution, x), 1.0e-15);
    }

    @Test
    void terminalQuantileBoundariesMatchBoostOverflowPolicy() {
        ExponentialDistribution distribution = new ExponentialDistribution(1.6);

        assertThrows(ArithmeticException.class, () -> distribution.quantile(1.0));
        assertThrows(ArithmeticException.class, () -> distribution.quantileUpperTail(0.0));
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new ExponentialDistribution(0.0));
        assertThrows(IllegalArgumentException.class, () -> new ExponentialDistribution(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new ExponentialDistribution().cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new ExponentialDistribution().quantile(Double.NaN));
    }
}