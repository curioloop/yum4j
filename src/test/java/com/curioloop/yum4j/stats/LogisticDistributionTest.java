package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LogisticDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        LogisticDistribution distribution = new LogisticDistribution(1.25, 0.8);

        assertEquals(1.25, distribution.location(), 0.0);
        assertEquals(0.8, distribution.scale(), 0.0);
        assertEquals(Double.NEGATIVE_INFINITY, distribution.range().lower());
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper());
        assertEquals(-Double.MAX_VALUE, distribution.support().lower());
        assertEquals(Double.MAX_VALUE, distribution.support().upper());
        assertEquals(1.25, distribution.mode(), 0.0);
        assertEquals(1.25, distribution.median(), 0.0);
        assertEquals(1.25, distribution.mean(), 0.0);
        assertEquals(Math.PI * Math.PI * 0.8 * 0.8 / 3.0, distribution.variance(), 1.0e-15);
        assertEquals(0.0, distribution.skewness(), 0.0);
        assertEquals(4.2, distribution.kurtosis(), 1.0e-15);
        assertEquals(1.2, distribution.kurtosisExcess(), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostClosedForms() {
        LogisticDistribution distribution = new LogisticDistribution(1.25, 0.8);
        double x = 2.0;
        double z = (x - 1.25) / 0.8;
        double cdf = 1.0 / (1.0 + Math.exp(-z));

        assertEquals(cdf * (1.0 - cdf) / 0.8, distribution.pdf(x), 1.0e-15);
        assertEquals(cdf, distribution.cdf(x), 1.0e-15);
        assertEquals(1.0 - cdf, distribution.ccdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);
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
        assertThrows(IllegalArgumentException.class, () -> new LogisticDistribution(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new LogisticDistribution(0.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new LogisticDistribution().cdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new LogisticDistribution().quantile(-0.1));
    }
}