package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GeometricDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        GeometricDistribution distribution = new GeometricDistribution(0.25);

        assertEquals(0.25, distribution.successFraction(), 0.0);
        assertEquals(1.0, distribution.successes(), 0.0);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(0.0, distribution.mode(), 0.0);
        assertEquals(2.0, distribution.quantile(0.5), 0.0);
        assertEquals(Math.log1p(-0.5) / Math.log1p(-0.25) - 1.0, distribution.median(), 1.0e-15);
        assertEquals(3.0, distribution.mean(), 1.0e-15);
        assertEquals(12.0, distribution.variance(), 1.0e-15);
        assertEquals(1.75 / Math.sqrt(0.75), distribution.skewness(), 1.0e-15);
        assertEquals(3.0 + (0.0625 - 1.5 + 6.0) / 0.75, distribution.kurtosis(), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostDiscreteSemantics() {
        GeometricDistribution distribution = new GeometricDistribution(0.25);

        assertEquals(0.25, distribution.pdf(0.0), 0.0);
        assertEquals(0.25 * Math.pow(0.75, 3.0), distribution.pdf(3.0), 1.0e-15);
        assertEquals(0.0, distribution.pdf(1.5), 0.0);
        assertEquals(1.0 - Math.pow(0.75, 4.0), distribution.cdf(3.7), 1.0e-15);
        assertEquals(Math.pow(0.75, 4.0), distribution.ccdf(3.7), 1.0e-15);
        assertEquals(distribution.ccdf(3.7), Distributions.cdf(Distributions.complement(distribution, 3.7)), 1.0e-15);
        assertEquals(0.0, distribution.quantile(0.25), 0.0);
        assertEquals(1.0, distribution.quantile(0.3), 0.0);
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertThrows(ArithmeticException.class, () -> distribution.quantile(1.0));
        assertThrows(ArithmeticException.class, () -> distribution.quantileUpperTail(0.0));
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void degenerateAndInvalidCasesMatchBoostBoundaries() {
        GeometricDistribution certainSuccess = new GeometricDistribution(1.0);

        GeometricDistribution impossibleSuccess = new GeometricDistribution(0.0);
        assertEquals(1.0, certainSuccess.pdf(0.0), 0.0);
        assertEquals(1.0, certainSuccess.cdf(2.0), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, impossibleSuccess.median());
        assertEquals(Double.POSITIVE_INFINITY, impossibleSuccess.quantile(0.5), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, impossibleSuccess.quantileUpperTail(0.5), 0.0);
        assertThrows(ArithmeticException.class, () -> certainSuccess.quantile(1.0));
        assertThrows(ArithmeticException.class, () -> certainSuccess.quantileUpperTail(0.0));
        assertEquals(0.0, certainSuccess.mean(), 0.0);
        assertEquals(0.0, certainSuccess.variance(), 0.0);
        assertTrue(Double.isInfinite(certainSuccess.skewness()));
        assertTrue(Double.isInfinite(certainSuccess.kurtosisExcess()));
        assertTrue(Double.isInfinite(certainSuccess.kurtosis()));
        assertThrows(IllegalArgumentException.class, () -> new GeometricDistribution(-0.1));
        assertThrows(IllegalArgumentException.class, () -> new GeometricDistribution(0.5).quantile(Double.NaN));
    }
}