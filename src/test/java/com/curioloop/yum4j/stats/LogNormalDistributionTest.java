package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Normal;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LogNormalDistributionTest {

    @Test
    void accessorsMatchBoostRangeAndSupport() {
        LogNormalDistribution distribution = new LogNormalDistribution(0.3, 0.8);

        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
    }

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        LogNormalDistribution distribution = new LogNormalDistribution(0.3, 0.8);
        double sigmaSquared = 0.64;

        assertEquals(0.3, distribution.location(), 0.0);
        assertEquals(0.8, distribution.scale(), 0.0);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(Math.exp(0.3 - sigmaSquared), distribution.mode(), 1.0e-15);
        assertEquals(Math.exp(0.3), distribution.median(), 1.0e-15);
        assertEquals(Math.exp(0.3 + sigmaSquared / 2.0), distribution.mean(), 1.0e-15);
        assertEquals(Math.expm1(sigmaSquared) * Math.exp(0.6 + sigmaSquared), distribution.variance(), 1.0e-14);
        assertEquals((Math.exp(sigmaSquared) + 2.0) * Math.sqrt(Math.expm1(sigmaSquared)), distribution.skewness(), 1.0e-14);
        assertEquals(3.0 + distribution.kurtosisExcess(), distribution.kurtosis(), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostLogComposition() {
        LogNormalDistribution distribution = new LogNormalDistribution(0.3, 0.8);
        double x = 2.0;
        double z = (Math.log(x) - 0.3) / 0.8;

        assertEquals(Normal.pdf(z) / (0.8 * x), distribution.pdf(x), 1.0e-15);
        assertEquals(Normal.cdf(z), distribution.cdf(x), 1.0e-15);
        assertEquals(Normal.ccdf(z), distribution.ccdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);
        assertEquals(x, distribution.quantile(distribution.cdf(x)), 1.0e-10);
        assertEquals(x, distribution.quantileUpperTail(distribution.ccdf(x)), 1.0e-10);
        assertEquals(0.0, distribution.pdf(0.0), 0.0);
        assertEquals(0.0, distribution.cdf(0.0), 0.0);
        assertEquals(1.0, distribution.ccdf(0.0), 0.0);
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void terminalQuantileBoundariesMatchBoostOverflowPolicy() {
        LogNormalDistribution distribution = new LogNormalDistribution(0.3, 0.8);

        assertThrows(ArithmeticException.class, () -> distribution.quantile(1.0));
        assertThrows(ArithmeticException.class, () -> distribution.quantileUpperTail(0.0));
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new LogNormalDistribution(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new LogNormalDistribution(0.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new LogNormalDistribution().cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new LogNormalDistribution().quantile(Double.NaN));
    }
}