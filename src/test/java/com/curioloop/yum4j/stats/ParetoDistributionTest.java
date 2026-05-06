package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ParetoDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        ParetoDistribution distribution = new ParetoDistribution(2.0, 5.0);
        double variance = 2.0 * 2.0 * 5.0 / (4.0 * 4.0 * 3.0);

        assertEquals(2.0, distribution.scale(), 0.0);
        assertEquals(5.0, distribution.shape(), 0.0);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper());
        assertEquals(2.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper());
        assertEquals(2.0, distribution.mode(), 0.0);
        assertEquals(2.0 * Math.pow(2.0, 0.2), distribution.median(), 1.0e-15);
        assertEquals(2.5, distribution.mean(), 1.0e-15);
        assertEquals(variance, distribution.variance(), 1.0e-15);
        assertEquals(4.6475800154489, distribution.skewness(), 1.0e-13);
        assertEquals(73.8, distribution.kurtosis(), 1.0e-12);
        assertEquals(70.8, distribution.kurtosisExcess(), 1.0e-12);
        assertEquals(Math.sqrt(variance), Distributions.standardDeviation(distribution), 1.0e-15);
        assertEquals(Math.sqrt(variance) / distribution.mean(), Distributions.coefficientOfVariation(distribution), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostClosedForms() {
        ParetoDistribution distribution = new ParetoDistribution(2.0, 5.0);
        double x = 3.25;
        double exponent = 5.0 * Math.log(2.0 / x);

        assertEquals(5.0 * Math.exp(exponent) / x, distribution.pdf(x), 1.0e-15);
        assertEquals(-Math.expm1(exponent), distribution.cdf(x), 1.0e-15);
        assertEquals(Math.exp(exponent), distribution.ccdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);
        assertEquals(distribution.quantileUpperTail(0.2),
            Distributions.quantile(Distributions.complement(distribution, 0.2)), 1.0e-15);
        assertEquals(0.0, distribution.cdf(2.0), 0.0);
        assertEquals(1.0, distribution.ccdf(2.0), 0.0);
        assertEquals(0.0, distribution.pdf(1.5), 0.0);
        assertEquals(2.0, distribution.quantile(0.0), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantile(1.0));
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantileUpperTail(0.0));
        assertEquals(2.0, distribution.quantileUpperTail(1.0), 0.0);
        assertEquals(0.0, distribution.pdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(1.0, distribution.cdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.ccdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(5.0 / x, Distributions.hazard(distribution, x), 1.0e-15);
        assertEquals(-Math.log(distribution.ccdf(x)), Distributions.chf(distribution, x), 1.0e-15);
    }

    @Test
    void maxValueMeanAndUndefinedHigherMomentsMatchBoostBehavior() {
        ParetoDistribution maxValueMean = new ParetoDistribution(2.0, 1.0);
        ParetoDistribution undefinedVariance = new ParetoDistribution(2.0, 2.0);
        ParetoDistribution undefinedSkewness = new ParetoDistribution(2.0, 3.0);
        ParetoDistribution undefinedKurtosis = new ParetoDistribution(2.0, 4.0);

        assertEquals(Double.MAX_VALUE, maxValueMean.mean(), 0.0);
        assertThrows(ArithmeticException.class, undefinedVariance::variance);
        assertThrows(ArithmeticException.class, undefinedSkewness::skewness);
        assertThrows(ArithmeticException.class, undefinedKurtosis::kurtosisExcess);
        assertThrows(ArithmeticException.class, undefinedKurtosis::kurtosis);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new ParetoDistribution(0.0, 2.0));
        assertThrows(IllegalArgumentException.class, () -> new ParetoDistribution(2.0, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new ParetoDistribution(2.0, 2.0).cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new ParetoDistribution(2.0, 2.0).quantile(Double.NaN));
    }
}