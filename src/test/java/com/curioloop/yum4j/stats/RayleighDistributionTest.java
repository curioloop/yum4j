package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RayleighDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        RayleighDistribution distribution = new RayleighDistribution(1.6);
        double variance = (4.0 - Math.PI) * 1.6 * 1.6 / 2.0;

        assertEquals(1.6, distribution.sigma(), 0.0);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper());
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper());
        assertEquals(1.6, distribution.mode(), 0.0);
        assertEquals(1.6 * Math.sqrt(Math.log(4.0)), distribution.median(), 1.0e-15);
        assertEquals(1.6 * Math.sqrt(Math.PI / 2.0), distribution.mean(), 1.0e-15);
        assertEquals(variance, distribution.variance(), 1.0e-15);
        assertEquals(0.6311106578189371, distribution.skewness(), 1.0e-15);
        assertEquals(3.245089300687638, distribution.kurtosis(), 1.0e-15);
        assertEquals(0.24508930068763806, distribution.kurtosisExcess(), 1.0e-15);
        assertEquals(Math.sqrt(variance), Distributions.standardDeviation(distribution), 1.0e-15);
        assertEquals(Math.sqrt(variance) / distribution.mean(), Distributions.coefficientOfVariation(distribution), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostClosedForms() {
        RayleighDistribution distribution = new RayleighDistribution(1.6);
        double x = 1.25;
        double sigmaSquared = 1.6 * 1.6;
        double chf = x * x / (2.0 * sigmaSquared);

        assertEquals(x * Math.exp(-chf) / sigmaSquared, distribution.pdf(x), 1.0e-15);
        assertEquals(-Math.expm1(-chf), distribution.cdf(x), 1.0e-15);
        assertEquals(Math.exp(-chf), distribution.ccdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);
        assertEquals(distribution.quantileUpperTail(0.2),
            Distributions.quantile(Distributions.complement(distribution, 0.2)), 1.0e-15);
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantile(1.0));
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantileUpperTail(0.0));
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
        assertEquals(0.0, distribution.pdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(1.0, distribution.cdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.ccdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(x / sigmaSquared, Distributions.hazard(distribution, x), 1.0e-15);
        assertEquals(chf, Distributions.chf(distribution, x), 1.0e-15);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new RayleighDistribution(0.0));
        assertThrows(IllegalArgumentException.class, () -> new RayleighDistribution(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new RayleighDistribution().cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new RayleighDistribution().quantile(Double.NaN));
    }
}