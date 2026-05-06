package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Gamma;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GammaDistributionTest {

    @Test
    void accessorsMatchBoostRangeAndSupport() {
        GammaDistribution distribution = new GammaDistribution(3.5, 2.0);

        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper());
        assertEquals(Double.MIN_NORMAL, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper());
    }

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        GammaDistribution distribution = new GammaDistribution(3.5, 2.0);

        assertEquals(3.5, distribution.shape());
        assertEquals(2.0, distribution.scale());
        assertEquals(5.0, distribution.mode(), 1.0e-15);
        assertEquals(7.0, distribution.mean(), 1.0e-15);
        assertEquals(14.0, distribution.variance(), 1.0e-15);
        assertEquals(2.0 / Math.sqrt(3.5), distribution.skewness(), 1.0e-15);
        assertEquals(3.0 + 6.0 / 3.5, distribution.kurtosis(), 1.0e-15);
        assertEquals(6.0 / 3.5, distribution.kurtosisExcess(), 1.0e-15);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper());
        assertEquals(Double.MIN_NORMAL, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper());
        assertEquals(Math.sqrt(14.0), Distributions.standardDeviation(distribution), 1.0e-15);
        assertEquals(1.0 / Math.sqrt(3.5), Distributions.coefficientOfVariation(distribution), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesGammaIdentities() {
        GammaDistribution distribution = new GammaDistribution(3.5, 2.0);
        double x = 4.0;

        assertEquals(Gamma.gammaPDerivative(3.5, 2.0) / 2.0, distribution.pdf(x), 1.0e-15);
        assertEquals(Gamma.gammaP(3.5, 2.0), distribution.cdf(x), 1.0e-15);
        assertEquals(Gamma.gammaQ(3.5, 2.0), distribution.ccdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);

        double probability = distribution.cdf(x);
        assertEquals(x, distribution.quantile(probability), 1.0e-10);
        assertEquals(x, distribution.quantileUpperTail(distribution.ccdf(x)), 1.0e-10);
        assertEquals(x, Distributions.quantile(Distributions.complement(distribution, distribution.ccdf(x))), 1.0e-10);
        assertEquals(distribution.pdf(x) / distribution.ccdf(x), Distributions.hazard(distribution, x), 1.0e-15);
        assertEquals(-Math.log(distribution.ccdf(x)), Distributions.chf(distribution, x), 1.0e-15);
    }

    @Test
    void supportsBoundaryValues() {
        GammaDistribution distribution = new GammaDistribution(3.5, 2.0);

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
        GammaDistribution distribution = new GammaDistribution(3.5, 2.0);

        assertThrows(ArithmeticException.class, () -> distribution.quantile(1.0));
        assertThrows(ArithmeticException.class, () -> distribution.quantileUpperTail(0.0));
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void rejectsInvalidInputsAndUndefinedMode() {
        assertThrows(IllegalArgumentException.class, () -> new GammaDistribution(0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new GammaDistribution(1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new GammaDistribution(2.0, 1.0).cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new GammaDistribution(2.0, 1.0).quantile(Double.NaN));
        assertEquals(0.0, new GammaDistribution(1.0, 1.0).mode(), 0.0);
        assertThrows(ArithmeticException.class, () -> new GammaDistribution(0.75, 1.0).mode());
    }
}