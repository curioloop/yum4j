package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Beta;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BinomialDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        BinomialDistribution distribution = new BinomialDistribution(5, 0.2);
        BinomialDistribution medianBoundary = new BinomialDistribution(1, 0.7);

        assertEquals(5, distribution.trials());
        assertEquals(0.2, distribution.successProbability(), 0.0);
        assertEquals(1.0, distribution.mode(), 0.0);
        assertEquals(1.0, distribution.mean(), 1.0e-15);
        assertEquals(0.8, distribution.variance(), 1.0e-15);
        assertEquals((1.0 - 0.4) / Math.sqrt(0.8), distribution.skewness(), 1.0e-15);
        assertEquals(3.05, distribution.kurtosis(), 1.0e-15);
        assertEquals(0.05, distribution.kurtosisExcess(), 1.0e-15);
        assertEquals(1.0, distribution.median(), 0.0);
        assertEquals(0.0, medianBoundary.median(), 0.0);
        assertEquals(1.0, medianBoundary.quantile(0.5), 0.0);
    }

    @Test
    void evaluationSurfaceMatchesBetaIdentities() {
        BinomialDistribution distribution = new BinomialDistribution(5, 0.2);

        assertEquals(0.2048000000000001, distribution.pdf(2.0), 1.0e-15);
        assertEquals(Beta.ibetac(3.0, 3.0, 0.2), distribution.cdf(2.0), 1.0e-15);
        assertEquals(Beta.ibeta(3.0, 3.0, 0.2), distribution.ccdf(2.0), 1.0e-15);
        assertEquals(0.0, distribution.pdf(2.5), 0.0);
        assertEquals(distribution.cdf(2.0), distribution.cdf(2.9), 0.0);
        assertEquals(distribution.ccdf(2.0), distribution.ccdf(2.9), 0.0);
    }

    @Test
    void quantileProvidesDiscreteLeftInverse() {
        BinomialDistribution distribution = new BinomialDistribution(5, 0.2);

        double[] probabilities = {0.05, 0.4, 0.75, 0.95, 1.0};
        double[] expected = {0.0, 1.0, 2.0, 3.0, 5.0};
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
    void supportsBoundaryValuesAndDegenerateCases() {
        BinomialDistribution zeroProbability = new BinomialDistribution(5, 0.0);
        BinomialDistribution unitProbability = new BinomialDistribution(5, 1.0);
        BinomialDistribution regular = new BinomialDistribution(5, 0.2);

        assertEquals(1.0, zeroProbability.pdf(0.0), 0.0);
        assertEquals(0.0, zeroProbability.pdf(1.0), 0.0);
        assertEquals(1.0, zeroProbability.cdf(0.0), 0.0);
        assertEquals(0.0, zeroProbability.ccdf(0.0), 0.0);
        assertEquals(1.0, unitProbability.pdf(5.0), 0.0);
        assertEquals(0.0, unitProbability.cdf(4.0), 0.0);
        assertEquals(1.0, unitProbability.cdf(5.0), 0.0);
        assertEquals(0.0, zeroProbability.mode(), 0.0);
        assertEquals(6.0, unitProbability.mode(), 0.0);
        assertEquals(0.0, zeroProbability.median(), 0.0);
        assertEquals(5.0, unitProbability.median(), 0.0);

        assertEquals(0.0, regular.quantile(0.0), 0.0);
        assertEquals(5.0, regular.quantile(1.0), 0.0);
        assertEquals(5.0, regular.quantileUpperTail(0.0), 0.0);
        assertEquals(0.0, regular.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void rejectsInvalidInputsAndMatchesBoostDegenerateHighMoments() {
        assertThrows(IllegalArgumentException.class, () -> new BinomialDistribution(-1, 0.2));
        assertThrows(IllegalArgumentException.class, () -> new BinomialDistribution(5, -0.1));
        assertThrows(IllegalArgumentException.class, () -> new BinomialDistribution(5, 1.1));
        assertThrows(IllegalArgumentException.class, () -> new BinomialDistribution(5, 0.2).cdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new BinomialDistribution(5, 0.2).quantile(-0.1));

        BinomialDistribution zeroProbability = new BinomialDistribution(5, 0.0);
        BinomialDistribution unitProbability = new BinomialDistribution(5, 1.0);

        assertTrue(Double.isInfinite(zeroProbability.skewness()) && zeroProbability.skewness() > 0.0);
        assertTrue(Double.isInfinite(unitProbability.skewness()) && unitProbability.skewness() < 0.0);
        assertTrue(Double.isInfinite(zeroProbability.kurtosisExcess()) && zeroProbability.kurtosisExcess() > 0.0);
        assertTrue(Double.isInfinite(unitProbability.kurtosisExcess()) && unitProbability.kurtosisExcess() > 0.0);
        assertTrue(Double.isInfinite(zeroProbability.kurtosis()) && zeroProbability.kurtosis() > 0.0);
        assertTrue(Double.isInfinite(unitProbability.kurtosis()) && unitProbability.kurtosis() > 0.0);
    }
}