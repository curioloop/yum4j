package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TriangularDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        TriangularDistribution distribution = new TriangularDistribution(-1.0, 0.5, 3.0);
        double lower = -1.0;
        double mode = 0.5;
        double upper = 3.0;
        double variance = (lower * lower + upper * upper + mode * mode - lower * upper - lower * mode - upper * mode) / 18.0;
        double midpoint = 0.5 * (upper + lower);
        double median = mode >= midpoint
            ? lower + Math.sqrt((upper - lower) * (mode - lower)) / Math.sqrt(2.0)
            : upper - Math.sqrt((upper - lower) * (upper - mode)) / Math.sqrt(2.0);
        double skewness = Math.sqrt(2.0) * (lower + upper - 2.0 * mode)
            * (2.0 * lower - upper - mode)
            * (lower - 2.0 * upper + mode)
            / (5.0 * Math.pow(lower * lower + upper * upper + mode * mode - lower * upper - lower * mode - upper * mode, 1.5));

        assertEquals(-1.0, distribution.lower());
        assertEquals(0.5, distribution.mode(), 0.0);
        assertEquals(3.0, distribution.upper());
        assertEquals(-Double.MAX_VALUE, distribution.range().lower());
        assertEquals(Double.MAX_VALUE, distribution.range().upper());
        assertEquals(-1.0, distribution.support().lower());
        assertEquals(3.0, distribution.support().upper());
        assertEquals((lower + mode + upper) / 3.0, distribution.mean(), 1.0e-15);
        assertEquals(median, distribution.median(), 1.0e-15);
        assertEquals(variance, distribution.variance(), 1.0e-15);
        assertEquals(skewness, distribution.skewness(), 1.0e-15);
        assertEquals(2.4, distribution.kurtosis(), 1.0e-15);
        assertEquals(-0.6, distribution.kurtosisExcess(), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesClosedForms() {
        TriangularDistribution distribution = new TriangularDistribution(-1.0, 0.5, 3.0);
        double lower = -1.0;
        double mode = 0.5;
        double upper = 3.0;
        double pivot = (mode - lower) / (upper - lower);

        assertEquals(0.0, distribution.pdf(-2.0), 0.0);
        assertEquals(2.0 * (0.0 - lower) / ((upper - lower) * (mode - lower)), distribution.pdf(0.0), 1.0e-15);
        assertEquals(2.0 * (mode - lower) / ((upper - lower) * (mode - lower)), distribution.pdf(0.5), 1.0e-15);
        assertEquals(0.0, distribution.pdf(4.0), 0.0);
        assertEquals(0.0, distribution.cdf(-2.0), 0.0);
        assertEquals(((0.0 - lower) * (0.0 - lower)) / ((upper - lower) * (mode - lower)), distribution.cdf(0.0), 1.0e-15);
        assertEquals(1.0, distribution.cdf(4.0), 0.0);
        assertEquals(1.0, distribution.ccdf(-2.0), 0.0);
        assertEquals(1.0 - ((0.0 - lower) * (0.0 - lower)) / ((upper - lower) * (mode - lower)), distribution.ccdf(0.0), 1.0e-15);
        assertEquals(0.0, distribution.ccdf(4.0), 0.0);
        assertEquals(distribution.ccdf(0.0), Distributions.cdf(Distributions.complement(distribution, 0.0)), 1.0e-15);
        assertEquals(-1.0, distribution.quantile(0.0), 0.0);
        assertEquals(lower + Math.sqrt((upper - lower) * (mode - lower) * 0.25), distribution.quantile(0.25), 1.0e-15);
        assertEquals(3.0, distribution.quantile(1.0), 0.0);
        assertEquals(3.0, distribution.quantileUpperTail(0.0), 0.0);
        assertEquals(pivot > 0.25 ? lower + Math.sqrt((upper - lower) * (mode - lower) * 0.25) : mode, distribution.quantileUpperTail(0.75), 1.0e-15);
        assertEquals(-1.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void handlesEndpointModesAndRejectsInvalidInputs() {
        TriangularDistribution leftMode = new TriangularDistribution(-1.0, -1.0, 3.0);
        TriangularDistribution rightMode = new TriangularDistribution(-1.0, 3.0, 3.0);

        assertEquals(0.5, leftMode.pdf(-1.0), 1.0e-15);
        assertEquals(0.5, rightMode.pdf(3.0), 1.0e-15);
        assertTrue(Double.isFinite(leftMode.pdf(-1.0)));
        assertTrue(Double.isFinite(rightMode.pdf(3.0)));

        assertThrows(IllegalArgumentException.class, () -> new TriangularDistribution(1.0, 0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new TriangularDistribution(0.0, -1.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new TriangularDistribution().cdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new TriangularDistribution().quantile(1.1));
    }
}