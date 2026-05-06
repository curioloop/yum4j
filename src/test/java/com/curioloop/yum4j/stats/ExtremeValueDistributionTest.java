package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ExtremeValueDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        ExtremeValueDistribution distribution = new ExtremeValueDistribution(1.25, 0.8);
        double mean = 1.25 + 0.5772156649015329 * 0.8;
        double variance = Math.PI * Math.PI * 0.8 * 0.8 / 6.0;

        assertEquals(1.25, distribution.location(), 0.0);
        assertEquals(0.8, distribution.scale(), 0.0);
        assertEquals(Double.NEGATIVE_INFINITY, distribution.range().lower());
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper());
        assertEquals(-Double.MAX_VALUE, distribution.support().lower());
        assertEquals(Double.MAX_VALUE, distribution.support().upper());
        assertEquals(1.25, distribution.mode(), 0.0);
        assertEquals(1.25 - 0.8 * Math.log(Math.log(2.0)), distribution.median(), 1.0e-15);
        assertEquals(mean, distribution.mean(), 1.0e-15);
        assertEquals(variance, distribution.variance(), 1.0e-15);
        assertEquals(1.1395470994046487, distribution.skewness(), 1.0e-15);
        assertEquals(5.4, distribution.kurtosis(), 1.0e-15);
        assertEquals(2.4, distribution.kurtosisExcess(), 1.0e-15);
        assertEquals(Math.sqrt(variance), Distributions.standardDeviation(distribution), 1.0e-15);
        assertEquals(Math.sqrt(variance) / mean, Distributions.coefficientOfVariation(distribution), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostClosedForms() {
        ExtremeValueDistribution distribution = new ExtremeValueDistribution(1.25, 0.8);
        double x = 0.75;
        double exponent = (1.25 - x) / 0.8;

        assertEquals(Math.exp(exponent - Math.exp(exponent)) / 0.8, distribution.pdf(x), 1.0e-15);
        assertEquals(Math.exp(-Math.exp(exponent)), distribution.cdf(x), 1.0e-15);
        assertEquals(-Math.expm1(-Math.exp(exponent)), distribution.ccdf(x), 1.0e-15);
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
        assertEquals(distribution.pdf(x) / distribution.ccdf(x), Distributions.hazard(distribution, x), 1.0e-15);
        assertEquals(-Math.log(distribution.ccdf(x)), Distributions.chf(distribution, x), 1.0e-15);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new ExtremeValueDistribution(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new ExtremeValueDistribution(0.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new ExtremeValueDistribution().cdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new ExtremeValueDistribution().quantile(-0.1));
    }
}