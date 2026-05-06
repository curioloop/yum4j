package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Gamma;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PoissonDistributionTest {

    @Test
    void accessorsMatchBoostRangeAndSupport() {
        PoissonDistribution distribution = new PoissonDistribution(2.0);

        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
    }

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        PoissonDistribution distribution = new PoissonDistribution(2.0);

        assertEquals(2.0, distribution.lambda());
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(2.0, distribution.mean(), 0.0);
        assertEquals(2.0, distribution.variance(), 0.0);
        assertEquals(2.0, distribution.mode(), 0.0);
        assertEquals(1.0 / Math.sqrt(2.0), distribution.skewness(), 1.0e-15);
        assertEquals(3.5, distribution.kurtosis(), 1.0e-15);
        assertEquals(0.5, distribution.kurtosisExcess(), 1.0e-15);
        assertEquals(distribution.quantile(0.5), distribution.median(), 0.0);
    }

    @Test
    void evaluationSurfaceMatchesGammaIdentities() {
        PoissonDistribution distribution = new PoissonDistribution(2.0);

        assertEquals(Math.exp(3.0 * Math.log(2.0) - Gamma.lgamma(4.0) - 2.0), distribution.pdf(3.0), 1.0e-15);
        assertEquals(Gamma.gammaQ(4.0, 2.0), distribution.cdf(3.0), 1.0e-15);
        assertEquals(Gamma.gammaP(4.0, 2.0), distribution.ccdf(3.0), 1.0e-15);
        assertEquals(0.0, distribution.pdf(3.5), 0.0);
        assertEquals(distribution.cdf(3.0), distribution.cdf(3.9), 0.0);
        assertEquals(distribution.ccdf(3.0), distribution.ccdf(3.9), 0.0);
    }

    @Test
    void quantileProvidesDiscreteLeftInverse() {
        PoissonDistribution distribution = new PoissonDistribution(1.0);

        double[] probabilities = {0.2, 0.5, 0.9, 0.98, 0.99, 0.999};
        double[] expected = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        for (int index = 0; index < probabilities.length; index++) {
            double quantile = distribution.quantile(probabilities[index]);
            assertEquals(expected[index], quantile, 0.0);
            assertTrue(distribution.cdf(quantile) >= probabilities[index]);
            if (quantile > 0.0) {
                assertTrue(distribution.cdf(quantile - 1.0) < probabilities[index]);
            }
            assertEquals(quantile, distribution.quantileUpperTail(distribution.ccdf(quantile)), 0.0);
        }
    }

    @Test
    void supportsBoundaryValues() {
        PoissonDistribution regular = new PoissonDistribution(2.0);

        assertEquals(0.0, regular.quantile(0.0), 0.0);
        assertEquals(0.0, regular.quantileUpperTail(1.0), 0.0);
        assertThrows(ArithmeticException.class, () -> regular.quantile(1.0));
        assertThrows(ArithmeticException.class, () -> regular.quantileUpperTail(0.0));
    }

    @Test
    void rejectsInvalidInputsAndBoostConstructorBoundary() {
        assertThrows(IllegalArgumentException.class, () -> new PoissonDistribution(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new PoissonDistribution(0.0));
        assertThrows(IllegalArgumentException.class, () -> new PoissonDistribution(1.0).pdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new PoissonDistribution(1.0).quantile(-0.1));
        assertThrows(IllegalArgumentException.class, () -> new PoissonDistribution(1.0).quantileUpperTail(1.1));
    }
}