package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Gamma;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ChiSquareDistributionTest {

    @Test
    void accessorsMatchBoostRangeAndSupport() {
        ChiSquareDistribution distribution = new ChiSquareDistribution(6.0);

        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
    }

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        ChiSquareDistribution distribution = new ChiSquareDistribution(6.0);

        assertEquals(6.0, distribution.degreesOfFreedom());
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(4.0, distribution.mode(), 1.0e-15);
        assertEquals(6.0, distribution.mean(), 1.0e-15);
        assertEquals(12.0, distribution.variance(), 1.0e-15);
        assertEquals(Math.sqrt(8.0 / 6.0), distribution.skewness(), 1.0e-15);
        assertEquals(5.0, distribution.kurtosis(), 1.0e-15);
        assertEquals(2.0, distribution.kurtosisExcess(), 1.0e-15);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
    }

    @Test
    void evaluationSurfaceMatchesGammaIdentities() {
        ChiSquareDistribution distribution = new ChiSquareDistribution(7.0);
        double x = 4.0;

        assertEquals(0.5 * Gamma.gammaPDerivative(3.5, 2.0), distribution.pdf(x), 1.0e-15);
        assertEquals(Gamma.gammaP(3.5, 2.0), distribution.cdf(x), 1.0e-15);
        assertEquals(Gamma.gammaQ(3.5, 2.0), distribution.ccdf(x), 1.0e-15);

        double probability = distribution.cdf(x);
        assertEquals(x, distribution.quantile(probability), 1.0e-10);
        assertEquals(x, distribution.quantileUpperTail(distribution.ccdf(x)), 1.0e-10);
    }

    @Test
    void supportsBoundaryValues() {
        ChiSquareDistribution distribution = new ChiSquareDistribution(6.0);

        assertEquals(0.0, distribution.cdf(0.0), 0.0);
        assertEquals(1.0, distribution.ccdf(0.0), 0.0);
        assertEquals(1.0, distribution.cdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.ccdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.pdf(Double.POSITIVE_INFINITY), 0.0);

        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void terminalQuantileBoundariesMatchBoostOverflowPolicy() {
        ChiSquareDistribution distribution = new ChiSquareDistribution(6.0);

        assertThrows(ArithmeticException.class, () -> distribution.quantile(1.0));
        assertThrows(ArithmeticException.class, () -> distribution.quantileUpperTail(0.0));
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void rejectsInvalidInputsAndUndefinedMode() {
        assertThrows(IllegalArgumentException.class, () -> new ChiSquareDistribution(0.0));
        assertThrows(IllegalArgumentException.class, () -> new ChiSquareDistribution(2.0).cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new ChiSquareDistribution(2.0).quantile(1.1));
        assertThrows(ArithmeticException.class, () -> new ChiSquareDistribution(1.5).mode());
    }
}