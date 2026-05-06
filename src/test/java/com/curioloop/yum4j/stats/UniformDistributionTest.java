package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class UniformDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        UniformDistribution distribution = new UniformDistribution(-2.0, 4.0);

        assertEquals(-2.0, distribution.lower());
        assertEquals(4.0, distribution.upper());
        assertEquals(-Double.MAX_VALUE, distribution.range().lower());
        assertEquals(Double.MAX_VALUE, distribution.range().upper());
        assertEquals(-2.0, distribution.support().lower());
        assertEquals(4.0, distribution.support().upper());
        assertEquals(-2.0, distribution.mode(), 0.0);
        assertEquals(1.0, distribution.mean(), 1.0e-15);
        assertEquals(1.0, distribution.median(), 1.0e-15);
        assertEquals(3.0, distribution.variance(), 1.0e-15);
        assertEquals(0.0, distribution.skewness(), 1.0e-15);
        assertEquals(1.8, distribution.kurtosis(), 1.0e-15);
        assertEquals(-1.2, distribution.kurtosisExcess(), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesClosedForms() {
        UniformDistribution distribution = new UniformDistribution(-2.0, 4.0);

        assertEquals(0.0, distribution.pdf(-3.0), 0.0);
        assertEquals(1.0 / 6.0, distribution.pdf(0.0), 1.0e-15);
        assertEquals(0.0, distribution.pdf(5.0), 0.0);
        assertEquals(0.0, distribution.cdf(-3.0), 0.0);
        assertEquals(1.0 / 3.0, distribution.cdf(0.0), 1.0e-15);
        assertEquals(1.0, distribution.cdf(5.0), 0.0);
        assertEquals(1.0, distribution.ccdf(-3.0), 0.0);
        assertEquals(2.0 / 3.0, distribution.ccdf(0.0), 1.0e-15);
        assertEquals(0.0, distribution.ccdf(5.0), 0.0);
        assertEquals(2.0 / 3.0, Distributions.cdf(Distributions.complement(distribution, 0.0)), 1.0e-15);
        assertEquals(-2.0, distribution.quantile(0.0), 0.0);
        assertEquals(1.0, distribution.quantile(0.5), 1.0e-15);
        assertEquals(4.0, distribution.quantile(1.0), 0.0);
        assertEquals(4.0, distribution.quantileUpperTail(0.0), 0.0);
        assertEquals(1.0, distribution.quantileUpperTail(0.5), 1.0e-15);
        assertEquals(-2.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new UniformDistribution(1.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new UniformDistribution(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new UniformDistribution().cdf(Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> new UniformDistribution().quantile(-0.1));
    }
}