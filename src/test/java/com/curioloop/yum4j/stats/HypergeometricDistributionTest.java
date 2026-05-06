package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HypergeometricDistributionTest {

    @Test
    void accessorsAndClosedFormStatisticsMatchBoostDefinitions() {
        HypergeometricDistribution distribution = new HypergeometricDistribution(20, 15, 50);

        assertEquals(20, distribution.defective());
        assertEquals(15, distribution.sampleCount());
        assertEquals(50, distribution.total());
        assertEquals(0.0, distribution.range().lower());
        assertEquals(15.0, distribution.range().upper());
        assertEquals(6.0, distribution.mean(), 1.0e-15);
        assertEquals(2.5714285714285716, distribution.variance(), 1.0e-15);
        assertEquals(6.0, distribution.mode(), 1.0e-15);

        double skewness = (50.0 - 40.0) * Math.sqrt(49.0) * (50.0 - 30.0)
            / (Math.sqrt(15.0 * 20.0 * 30.0 * 35.0) * 48.0);
        assertEquals(skewness, distribution.skewness(), 1.0e-15);

        double kurtosisExcess = ((49.0) * 2500.0
            * ((3.0 * 20.0 * 30.0 * (225.0 * (-50.0) + 13.0 * 2500.0 + 6.0 * 15.0 * 35.0)) / 2500.0
            - 6.0 * 15.0 * 35.0 + 50.0 * 51.0))
            / (20.0 * 15.0 * 47.0 * 48.0 * 30.0 * 35.0);
        assertEquals(kurtosisExcess, distribution.kurtosisExcess(), 1.0e-12);
        assertEquals(kurtosisExcess + 3.0, distribution.kurtosis(), 1.0e-12);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-15);
    }

    @Test
    void smallIntegerCaseMatchesExactMassAndDiscreteSemantics() {
        HypergeometricDistribution distribution = new HypergeometricDistribution(3, 4, 10);

        assertEquals(0.0, distribution.support().lower());
        assertEquals(3.0, distribution.support().upper());
        assertEquals(1.0 / 6.0, distribution.pdf(0.0), 1.0e-15);
        assertEquals(0.5, distribution.pdf(1.0), 1.0e-12);
        assertEquals(0.3, distribution.pdf(2.0), 1.0e-12);
        assertEquals(1.0 / 30.0, distribution.pdf(3.0), 1.0e-12);
        assertEquals(0.0, distribution.pdf(1.5), 0.0);

        assertEquals(1.0 / 6.0, distribution.cdf(0.0), 1.0e-15);
        assertEquals(2.0 / 3.0, distribution.cdf(1.9), 1.0e-12);
        assertEquals(1.0, distribution.cdf(3.0), 0.0);
        assertEquals(5.0 / 6.0, distribution.ccdf(0.0), 1.0e-12);
        assertEquals(1.0 / 3.0, distribution.ccdf(1.2), 1.0e-12);
        assertEquals(0.0, distribution.ccdf(3.0), 0.0);

        assertEquals(0.0, distribution.quantile(1.0 / 6.0), 0.0);
        assertEquals(1.0, distribution.quantile(0.5), 0.0);
        assertEquals(2.0, distribution.quantile(0.9), 0.0);
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(3.0, distribution.quantile(1.0), 0.0);
        assertEquals(3.0, distribution.quantileUpperTail(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
        assertEquals(1.0, distribution.quantileUpperTail(1.0 / 3.0), 0.0);
        assertEquals(2.0, distribution.quantileUpperTail(1.0 / 30.0), 0.0);
    }

    @Test
    void degeneratePopulationAtZeroFollowsBoostRawFormulas() {
        HypergeometricDistribution distribution = new HypergeometricDistribution(0, 0, 0);

        assertEquals(1.0, distribution.pdf(0.0), 0.0);
        assertEquals(1.0, distribution.cdf(0.0), 0.0);
        assertEquals(0.0, distribution.ccdf(0.0), 0.0);
        assertTrue(Double.isNaN(distribution.mean()));
        assertTrue(Double.isNaN(distribution.variance()));
        assertTrue(Double.isNaN(distribution.skewness()));
        assertTrue(Double.isNaN(distribution.kurtosisExcess()));
        assertTrue(Double.isNaN(distribution.kurtosis()));
    }

    @Test
    void singletonPopulationDegeneracyAlsoFollowsBoostRawFormulas() {
        HypergeometricDistribution distribution = new HypergeometricDistribution(1, 1, 1);

        assertEquals(1.0, distribution.mean(), 0.0);
        assertTrue(Double.isNaN(distribution.variance()));
        assertTrue(Double.isNaN(distribution.skewness()));
        assertTrue(Double.isNaN(distribution.kurtosisExcess()));
        assertTrue(Double.isNaN(distribution.kurtosis()));
    }

    @Test
    void invalidArgumentsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new HypergeometricDistribution(-1, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> new HypergeometricDistribution(3, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> new HypergeometricDistribution(1, 3, 2));
        assertThrows(IllegalArgumentException.class, () -> new HypergeometricDistribution(1, 1, 2).pdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new HypergeometricDistribution(1, 1, 2).quantile(-0.1));
    }

    @Test
    void massFunctionSumsToOneAcrossSupport() {
        HypergeometricDistribution distribution = new HypergeometricDistribution(12, 8, 30);

        double totalMass = 0.0;
        for (long x = (long) distribution.support().lower(); x <= (long) distribution.support().upper(); x++) {
            totalMass += distribution.pdf(x);
        }
        assertEquals(1.0, totalMass, 1.0e-12);
        assertTrue(distribution.cdf(4.0) <= distribution.cdf(5.0));
    }
}