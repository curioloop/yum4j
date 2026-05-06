package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Normal;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InverseGaussianDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        InverseGaussianDistribution distribution = new InverseGaussianDistribution(1.5, 4.0);
        double variance = Math.pow(1.5, 3.0) / 4.0;

        assertEquals(1.5, distribution.meanParameter(), 0.0);
        assertEquals(1.5, distribution.location(), 0.0);
        assertEquals(4.0, distribution.scale(), 0.0);
        assertEquals(4.0 / 1.5, distribution.shape(), 1.0e-15);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(1.5, distribution.mean(), 0.0);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
        assertEquals(variance, distribution.variance(), 1.0e-15);
        assertEquals(3.0 * Math.sqrt(1.5 / 4.0), distribution.skewness(), 1.0e-15);
        assertEquals(15.0 * 1.5 / 4.0 - 3.0, distribution.kurtosis(), 1.0e-15);
        assertEquals(15.0 * 1.5 / 4.0, distribution.kurtosisExcess(), 1.0e-15);
        assertEquals(Math.sqrt(variance), Distributions.standardDeviation(distribution), 1.0e-15);
        assertTrue(distribution.mode() > 0.0 && distribution.mode() < distribution.mean());
    }

    @Test
    void evaluationSurfaceMatchesBoostClosedFormsAndQuantilesRoundTrip() {
        InverseGaussianDistribution distribution = new InverseGaussianDistribution(1.5, 4.0);
        double x = 1.25;
        double root = Math.sqrt(4.0 / x);
        double first = root * (x / 1.5 - 1.0);
        double second = -root * (x / 1.5 + 1.0);

        assertEquals(Math.sqrt(4.0 / (2.0 * Math.PI * x * x * x))
            * Math.exp(-4.0 * (x - 1.5) * (x - 1.5) / (2.0 * x * 1.5 * 1.5)), distribution.pdf(x), 1.0e-15);
        assertEquals(Normal.cdf(first) + Math.exp(8.0 / 1.5) * Normal.cdf(second), distribution.cdf(x), 1.0e-13);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-13);

        double quantile = distribution.quantile(0.7);
        assertEquals(0.7, distribution.cdf(quantile), 1.0e-10);
        double upperTailQuantile = distribution.quantileUpperTail(0.25);
        assertEquals(0.25, distribution.ccdf(upperTailQuantile), 1.0e-10);
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.pdf(0.0), 0.0);
        assertEquals(0.0, distribution.cdf(0.0), 0.0);
        assertEquals(1.0, distribution.ccdf(0.0), 0.0);
    }

    @Test
    void terminalQuantileBoundariesMatchBoostPublicSurface() {
        InverseGaussianDistribution distribution = new InverseGaussianDistribution(1.5, 4.0);

        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertThrows(ArithmeticException.class, () -> distribution.quantile(1.0));
        assertEquals(0.0, distribution.quantileUpperTail(0.0), 0.0);
        assertThrows(ArithmeticException.class, () -> distribution.quantileUpperTail(1.0));
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new InverseGaussianDistribution(0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new InverseGaussianDistribution(1.0, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new InverseGaussianDistribution().cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new InverseGaussianDistribution().quantile(Double.NaN));
    }
}