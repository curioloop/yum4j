package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class HyperexponentialDistributionTest {

    @Test
    void constructorNormalizesWeightsAndExposesBoostStyleAccessors() {
        HyperexponentialDistribution distribution = new HyperexponentialDistribution(
            new double[]{2.0, 1.0},
            new double[]{1.0, 3.0}
        );

        assertArrayEquals(new double[]{2.0 / 3.0, 1.0 / 3.0}, distribution.probabilities(), 1.0e-15);
        assertArrayEquals(new double[]{1.0, 3.0}, distribution.rates(), 0.0);
        assertEquals(2, distribution.numPhases());
        assertEquals(0.0, distribution.range().lower());
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper());
        assertEquals(Double.MIN_NORMAL, distribution.support().lower());
    }

    @Test
    void evaluationSurfaceMatchesMixtureOfExponentials() {
        HyperexponentialDistribution distribution = new HyperexponentialDistribution(
            new double[]{0.6, 0.4},
            new double[]{1.5, 0.4}
        );
        double x = 0.8;

        double expectedPdf = 0.6 * 1.5 * Math.exp(-1.5 * x) + 0.4 * 0.4 * Math.exp(-0.4 * x);
        double expectedCdf = 0.6 * -Math.expm1(-1.5 * x) + 0.4 * -Math.expm1(-0.4 * x);

        assertEquals(expectedPdf, distribution.pdf(x), 1.0e-15);
        assertEquals(expectedCdf, distribution.cdf(x), 1.0e-15);
        assertEquals(1.0 - expectedCdf, distribution.ccdf(x), 1.0e-15);

        double probability = distribution.cdf(x);
        assertEquals(x, distribution.quantile(probability), 1.0e-9);
        assertEquals(x, distribution.quantileUpperTail(distribution.ccdf(x)), 1.0e-9);
    }

    @Test
    void momentsReduceToExponentialCaseAndMatchMixtureFormulas() {
        HyperexponentialDistribution singlePhase = new HyperexponentialDistribution(new double[]{2.0});
        assertEquals(new ExponentialDistribution(2.0).mean(), singlePhase.mean(), 1.0e-15);
        assertEquals(new ExponentialDistribution(2.0).variance(), singlePhase.variance(), 1.0e-15);
        assertEquals(new ExponentialDistribution(2.0).quantile(0.7), singlePhase.quantile(0.7), 1.0e-15);

        HyperexponentialDistribution distribution = new HyperexponentialDistribution(
            new double[]{0.6, 0.4},
            new double[]{1.5, 0.4}
        );
        assertEquals(1.4, distribution.mean(), 1.0e-15);
        assertEquals(3.5733333333333324, distribution.variance(), 1.0e-14);
        assertEquals(0.0, distribution.mode(), 0.0);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
        assertEquals(3.081478659282253, distribution.skewness(), 1.0e-12);
        assertEquals(16.87079527734463, distribution.kurtosis(), 1.0e-11);
        assertEquals(13.87079527734463, distribution.kurtosisExcess(), 1.0e-11);
    }

    @Test
    void invalidArgumentsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new HyperexponentialDistribution(new double[]{}));
        assertThrows(IllegalArgumentException.class,
            () -> new HyperexponentialDistribution(new double[]{0.5}, new double[]{1.0, 2.0}));
        assertThrows(IllegalArgumentException.class,
            () -> new HyperexponentialDistribution(new double[]{-0.1}, new double[]{1.0}));
        assertThrows(IllegalArgumentException.class,
            () -> new HyperexponentialDistribution(new double[]{1.0}, new double[]{0.0}));
        assertThrows(IllegalArgumentException.class,
            () -> new HyperexponentialDistribution(new double[]{1.0}, new double[]{1.0}).cdf(-1.0));
        assertThrows(IllegalArgumentException.class,
            () -> new HyperexponentialDistribution(new double[]{1.0}, new double[]{1.0}).quantile(1.1));
    }
}