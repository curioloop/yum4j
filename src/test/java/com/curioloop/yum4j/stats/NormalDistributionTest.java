package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class NormalDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        NormalDistribution distribution = new NormalDistribution(1.5, 2.0);

        assertEquals(1.5, distribution.mean());
        assertEquals(1.5, distribution.location());
        assertEquals(2.0, distribution.standardDeviation());
        assertEquals(2.0, distribution.scale());
        assertEquals(1.5, distribution.mode());
        assertEquals(1.5, distribution.median());
        assertEquals(4.0, distribution.variance());
        assertEquals(0.0, distribution.skewness());
        assertEquals(3.0, distribution.kurtosis());
        assertEquals(0.0, distribution.kurtosisExcess());
        assertEquals(Double.NEGATIVE_INFINITY, distribution.range().lower());
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper());
        assertEquals(Double.NEGATIVE_INFINITY, distribution.support().lower());
        assertEquals(Double.POSITIVE_INFINITY, distribution.support().upper());
        assertEquals(2.0, Distributions.standardDeviation(distribution));
        assertEquals(4.0 / 3.0, Distributions.coefficientOfVariation(distribution), 1.0e-15);
    }

    @Test
    void supportsInfiniteArgumentsAndProbabilityBoundaries() {
        NormalDistribution distribution = new NormalDistribution(1.5, 2.0);

        assertEquals(0.0, distribution.pdf(Double.NEGATIVE_INFINITY));
        assertEquals(0.0, distribution.pdf(Double.POSITIVE_INFINITY));
        assertEquals(0.0, distribution.cdf(Double.NEGATIVE_INFINITY));
        assertEquals(1.0, distribution.cdf(Double.POSITIVE_INFINITY));
        assertEquals(1.0, distribution.ccdf(Double.NEGATIVE_INFINITY));
        assertEquals(0.0, distribution.ccdf(Double.POSITIVE_INFINITY));

        assertEquals(Double.NEGATIVE_INFINITY, distribution.quantile(0.0));
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantile(1.0));
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantileUpperTail(0.0));
        assertEquals(Double.NEGATIVE_INFINITY, distribution.quantileUpperTail(1.0));

        assertEquals(0.0, Distributions.cdf(distribution, Double.NEGATIVE_INFINITY));
        assertEquals(1.0, Distributions.cdf(distribution, Double.POSITIVE_INFINITY));
        assertEquals(1.0, Distributions.cdf(Distributions.complement(distribution, Double.NEGATIVE_INFINITY)));
        assertEquals(0.0, Distributions.cdf(Distributions.complement(distribution, Double.POSITIVE_INFINITY)));
        assertEquals(Double.NEGATIVE_INFINITY, Distributions.quantile(distribution, 0.0));
        assertEquals(Double.POSITIVE_INFINITY, Distributions.quantile(distribution, 1.0));
        assertEquals(Double.POSITIVE_INFINITY, Distributions.quantile(Distributions.complement(distribution, 0.0)));
        assertEquals(Double.NEGATIVE_INFINITY, Distributions.quantile(Distributions.complement(distribution, 1.0)));
    }

    @Test
    void derivedAccessorsMatchBoostDefinitions() {
        NormalDistribution distribution = new NormalDistribution(1.5, 2.0);
        double x = 2.5;
        double expectedHazard = distribution.pdf(x) / distribution.ccdf(x);
        double expectedChf = -Math.log(distribution.ccdf(x));

        assertEquals(expectedHazard, Distributions.hazard(distribution, x), 1.0e-15);
        assertEquals(expectedChf, Distributions.chf(distribution, x), 1.0e-15);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new NormalDistribution(0.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new NormalDistribution(0.0, -1.0));
        assertThrows(IllegalArgumentException.class, () -> new NormalDistribution().quantile(-0.1));
        assertThrows(IllegalArgumentException.class, () -> new NormalDistribution().quantile(1.1));
        assertThrows(IllegalArgumentException.class, () -> new NormalDistribution().quantileUpperTail(Double.NaN));
        assertThrows(ArithmeticException.class, () -> Distributions.coefficientOfVariation(new NormalDistribution()));
        assertThrows(IllegalArgumentException.class, () -> Distributions.quantile(new NormalDistribution(), Double.NaN));
    }
}