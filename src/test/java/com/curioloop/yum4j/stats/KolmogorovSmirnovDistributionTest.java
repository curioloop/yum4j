package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.JacobiTheta;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KolmogorovSmirnovDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        KolmogorovSmirnovDistribution distribution = new KolmogorovSmirnovDistribution(25.0);
        double variance = ((Math.PI * Math.PI / 6.0) - Math.PI * Math.log(2.0) * Math.log(2.0)) / 50.0;

        assertEquals(25.0, distribution.numberOfObservations(), 0.0);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(Math.sqrt(Math.PI / 2.0) * Math.log(2.0) / 5.0, distribution.mean(), 1.0e-15);
        assertEquals(variance, distribution.variance(), 1.0e-15);
        assertTrue(distribution.mode() > 0.0);
        assertTrue(distribution.mode() < 1.0 / Math.sqrt(25.0));
    }

    @Test
    void evaluationSurfaceMatchesJacobiThetaDefinitionAndQuantilesRoundTrip() {
        KolmogorovSmirnovDistribution distribution = new KolmogorovSmirnovDistribution(25.0);
        double x = 0.2;
        double tau = 2.0 * x * x * 25.0 / Math.PI;

        assertEquals(JacobiTheta.theta4Tau(0.0, tau), distribution.cdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);
        assertTrue(distribution.pdf(x) > 0.0);
        assertEquals(0.0, distribution.cdf(0.0), 0.0);
        assertEquals(1.0, distribution.ccdf(0.0), 0.0);

        double quantile = distribution.quantile(0.7);
        assertEquals(0.7, distribution.cdf(quantile), 1.0e-9);
        double upperTailQuantile = distribution.quantileUpperTail(0.2);
        assertEquals(0.2, distribution.ccdf(upperTailQuantile), 1.0e-9);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new KolmogorovSmirnovDistribution(0.0));
        assertThrows(IllegalArgumentException.class, () -> new KolmogorovSmirnovDistribution(10.0).cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new KolmogorovSmirnovDistribution(10.0).quantile(Double.NaN));
    }
}