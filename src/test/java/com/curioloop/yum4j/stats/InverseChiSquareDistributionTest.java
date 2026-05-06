package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class InverseChiSquareDistributionTest {

    @Test
    void accessorsMatchBoostRangeAndSupport() {
        InverseChiSquareDistribution distribution = new InverseChiSquareDistribution(6.0, 1.5);

        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
    }

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        InverseChiSquareDistribution distribution = new InverseChiSquareDistribution(6.0, 1.5);

        assertEquals(6.0, distribution.degreesOfFreedom(), 0.0);
        assertEquals(1.5, distribution.scale(), 0.0);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(6.0 * 1.5 / 8.0, distribution.mode(), 1.0e-15);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
        assertEquals(6.0 * 1.5 / 4.0, distribution.mean(), 1.0e-15);
        assertEquals(2.0 * 36.0 * 1.5 * 1.5 / (16.0 * 2.0), distribution.variance(), 1.0e-15);
        assertEquals(1.0 / 6.0, new InverseChiSquareDistribution(6.0).scale(), 0.0);
    }

    @Test
    void evaluationSurfaceDelegatesToEquivalentInverseGamma() {
        InverseChiSquareDistribution distribution = new InverseChiSquareDistribution(6.0, 1.5);
        InverseGammaDistribution equivalent = new InverseGammaDistribution(3.0, 4.5);
        double x = 1.2;

        assertEquals(equivalent.pdf(x), distribution.pdf(x), 1.0e-15);
        assertEquals(equivalent.cdf(x), distribution.cdf(x), 1.0e-15);
        assertEquals(equivalent.ccdf(x), distribution.ccdf(x), 1.0e-15);
        assertEquals(equivalent.quantile(0.65), distribution.quantile(0.65), 1.0e-12);
        assertEquals(equivalent.quantileUpperTail(0.35), distribution.quantileUpperTail(0.35), 1.0e-12);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantile(1.0), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantileUpperTail(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void undefinedMomentsAndInvalidInputsMatchBoostBoundaries() {
        assertThrows(IllegalArgumentException.class, () -> new InverseChiSquareDistribution(0.0));
        assertThrows(IllegalArgumentException.class, () -> new InverseChiSquareDistribution(2.0, 0.0));
        assertThrows(ArithmeticException.class, () -> new InverseChiSquareDistribution(2.0, 1.5).mean());
        assertThrows(ArithmeticException.class, () -> new InverseChiSquareDistribution(4.0, 1.5).variance());
        assertThrows(ArithmeticException.class, () -> new InverseChiSquareDistribution(6.0, 1.5).skewness());
        assertThrows(ArithmeticException.class, () -> new InverseChiSquareDistribution(8.0, 1.5).kurtosisExcess());
    }
}