package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Gamma;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class InverseGammaDistributionTest {

    @Test
    void accessorsMatchBoostRangeAndSupport() {
        InverseGammaDistribution distribution = new InverseGammaDistribution(5.0, 2.0);

        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
    }

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        InverseGammaDistribution distribution = new InverseGammaDistribution(5.0, 2.0);

        assertEquals(5.0, distribution.shape(), 0.0);
        assertEquals(2.0, distribution.scale(), 0.0);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(2.0 / 6.0, distribution.mode(), 1.0e-15);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
        assertEquals(0.5, distribution.mean(), 1.0e-15);
        assertEquals(1.0 / 12.0, distribution.variance(), 1.0e-15);
        assertEquals(4.0 * Math.sqrt(3.0) / 2.0, distribution.skewness(), 1.0e-15);
        assertEquals(45.0, distribution.kurtosis(), 1.0e-15);
        assertEquals(42.0, distribution.kurtosisExcess(), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostReciprocalGammaDefinitions() {
        InverseGammaDistribution distribution = new InverseGammaDistribution(5.0, 2.0);
        double x = 0.75;
        double scaled = 2.0 / x;

        assertEquals(Gamma.gammaPDerivative(5.0, scaled) * 2.0 / (x * x), distribution.pdf(x), 1.0e-15);
        assertEquals(Gamma.gammaQ(5.0, scaled), distribution.cdf(x), 1.0e-15);
        assertEquals(Gamma.gammaP(5.0, scaled), distribution.ccdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);
        assertEquals(x, distribution.quantile(distribution.cdf(x)), 1.0e-10);
        assertEquals(x, distribution.quantileUpperTail(distribution.ccdf(x)), 1.0e-10);
        assertEquals(x, Distributions.quantile(Distributions.complement(distribution, distribution.ccdf(x))), 1.0e-10);
        assertEquals(0.0, distribution.pdf(0.0), 0.0);
        assertEquals(0.0, distribution.pdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.cdf(0.0), 0.0);
        assertEquals(1.0, distribution.cdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(1.0, distribution.ccdf(0.0), 0.0);
        assertEquals(0.0, distribution.ccdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantile(1.0), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantileUpperTail(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void undefinedMomentsAndInvalidInputsMatchBoostBoundaries() {
        assertThrows(IllegalArgumentException.class, () -> new InverseGammaDistribution(0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new InverseGammaDistribution(1.0, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new InverseGammaDistribution().cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new InverseGammaDistribution().quantile(Double.NaN));
        assertThrows(ArithmeticException.class, () -> new InverseGammaDistribution(1.0, 2.0).mean());
        assertThrows(ArithmeticException.class, () -> new InverseGammaDistribution(2.0, 2.0).variance());
        assertThrows(ArithmeticException.class, () -> new InverseGammaDistribution(3.0, 2.0).skewness());
        assertThrows(ArithmeticException.class, () -> new InverseGammaDistribution(4.0, 2.0).kurtosisExcess());
    }
}