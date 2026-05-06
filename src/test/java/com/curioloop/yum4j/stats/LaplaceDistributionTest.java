package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LaplaceDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        LaplaceDistribution distribution = new LaplaceDistribution(1.25, 0.8);

        assertEquals(1.25, distribution.location(), 0.0);
        assertEquals(0.8, distribution.scale(), 0.0);
        assertEquals(Double.NEGATIVE_INFINITY, distribution.range().lower());
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper());
        assertEquals(Double.NEGATIVE_INFINITY, distribution.support().lower());
        assertEquals(Double.POSITIVE_INFINITY, distribution.support().upper());
        assertEquals(1.25, distribution.mode(), 0.0);
        assertEquals(1.25, distribution.median(), 0.0);
        assertEquals(1.25, distribution.mean(), 0.0);
        assertEquals(2.0 * 0.8 * 0.8, distribution.variance(), 1.0e-15);
        assertEquals(0.0, distribution.skewness(), 0.0);
        assertEquals(6.0, distribution.kurtosis(), 0.0);
        assertEquals(3.0, distribution.kurtosisExcess(), 0.0);
    }

    @Test
    void evaluationSurfaceMatchesBoostClosedForms() {
        LaplaceDistribution distribution = new LaplaceDistribution(1.25, 0.8);
        double left = 0.5;
        double right = 2.0;

        assertEquals(Math.exp(-(1.25 - left) / 0.8) / (2.0 * 0.8), distribution.pdf(left), 1.0e-15);
        assertEquals(0.5 * Math.exp((left - 1.25) / 0.8), distribution.cdf(left), 1.0e-15);
        assertEquals(1.0 - 0.5 * Math.exp((left - 1.25) / 0.8), distribution.ccdf(left), 1.0e-15);
        assertEquals(1.0 - 0.5 * Math.exp((1.25 - right) / 0.8), distribution.cdf(right), 1.0e-15);
        assertEquals(0.5 * Math.exp((1.25 - right) / 0.8), distribution.ccdf(right), 1.0e-15);
        assertEquals(distribution.ccdf(right), Distributions.cdf(Distributions.complement(distribution, right)), 1.0e-15);
        assertEquals(distribution.quantileUpperTail(0.2),
            Distributions.quantile(Distributions.complement(distribution, 0.2)), 1.0e-15);
        assertEquals(Double.NEGATIVE_INFINITY, distribution.quantile(0.0));
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantile(1.0));
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantileUpperTail(0.0));
        assertEquals(Double.NEGATIVE_INFINITY, distribution.quantileUpperTail(1.0));
        assertEquals(0.0, distribution.pdf(Double.NEGATIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.pdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.cdf(Double.NEGATIVE_INFINITY), 0.0);
        assertEquals(1.0, distribution.cdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(1.0, distribution.ccdf(Double.NEGATIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.ccdf(Double.POSITIVE_INFINITY), 0.0);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new LaplaceDistribution(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new LaplaceDistribution(0.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new LaplaceDistribution().cdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new LaplaceDistribution().quantile(-0.1));
    }
}