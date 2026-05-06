package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Beta;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NegativeBinomialDistributionTest {

    @Test
    void accessorsMatchBoostRangeAndSupport() {
        NegativeBinomialDistribution distribution = new NegativeBinomialDistribution(3.5, 0.4);

        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
    }

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        NegativeBinomialDistribution distribution = new NegativeBinomialDistribution(3.5, 0.4);

        assertEquals(3.5, distribution.successes(), 0.0);
        assertEquals(0.4, distribution.successFraction(), 0.0);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(3.0, distribution.mode(), 0.0);
        assertEquals(distribution.quantile(0.5), distribution.median(), 0.0);
        assertEquals(5.25, distribution.mean(), 1.0e-15);
        assertEquals(13.125, distribution.variance(), 1.0e-14);
        assertEquals(1.6 / Math.sqrt(2.1), distribution.skewness(), 1.0e-15);
        assertEquals(3.0 + (6.0 - 0.4 * 5.6) / (3.5 * 0.6), distribution.kurtosis(), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostDiscreteSemantics() {
        NegativeBinomialDistribution distribution = new NegativeBinomialDistribution(3.5, 0.4);
        double x = 2.0;

        assertEquals((0.4 / 5.5) * Beta.ibetaDerivative(3.5, 3.0, 0.4), distribution.pdf(x), 1.0e-15);
        assertEquals(Beta.ibeta(3.5, 3.0, 0.4), distribution.cdf(2.2), 1.0e-15);
        assertEquals(Beta.ibetac(3.5, 3.0, 0.4), distribution.ccdf(2.2), 1.0e-15);
        assertEquals(distribution.ccdf(2.2), Distributions.cdf(Distributions.complement(distribution, 2.2)), 1.0e-15);
        assertEquals(0.0, distribution.pdf(2.5), 0.0);
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
        assertThrows(ArithmeticException.class, () -> distribution.quantile(1.0));
        assertThrows(ArithmeticException.class, () -> distribution.quantileUpperTail(0.0));
    }

    @Test
    void boundaryAndInvalidCasesMatchBoostDefinitions() {
        NegativeBinomialDistribution distribution = new NegativeBinomialDistribution(1.0, 0.35);
        GeometricDistribution geometric = new GeometricDistribution(0.35);
        NegativeBinomialDistribution impossibleSuccessHalf = new NegativeBinomialDistribution(0.5, 0.0);
        NegativeBinomialDistribution impossibleSuccessUnit = new NegativeBinomialDistribution(1.0, 0.0);
        NegativeBinomialDistribution impossibleSuccessMany = new NegativeBinomialDistribution(3.5, 0.0);
        NegativeBinomialDistribution certainSuccess = new NegativeBinomialDistribution(3.5, 1.0);

        assertEquals(geometric.pdf(3.0), distribution.pdf(3.0), 1.0e-15);
        assertEquals(geometric.cdf(3.0), distribution.cdf(3.0), 1.0e-15);
        assertEquals(0.0, distribution.mode(), 0.0);
        assertTrue(Double.isInfinite(impossibleSuccessHalf.mode()) && impossibleSuccessHalf.mode() < 0.0);
        assertTrue(Double.isNaN(impossibleSuccessUnit.mode()));
        assertTrue(Double.isInfinite(impossibleSuccessMany.mode()) && impossibleSuccessMany.mode() > 0.0);
        assertTrue(Double.isInfinite(impossibleSuccessHalf.mean()) && impossibleSuccessHalf.mean() > 0.0);
        assertTrue(Double.isInfinite(impossibleSuccessHalf.variance()) && impossibleSuccessHalf.variance() > 0.0);
        assertEquals(2.0 / Math.sqrt(0.5), impossibleSuccessHalf.skewness(), 1.0e-15);
        assertEquals(15.0, impossibleSuccessHalf.kurtosis(), 1.0e-15);
        assertEquals(12.0, impossibleSuccessHalf.kurtosisExcess(), 1.0e-15);
        assertThrows(ArithmeticException.class, () -> impossibleSuccessHalf.quantile(0.5));
        assertThrows(ArithmeticException.class, () -> impossibleSuccessHalf.quantileUpperTail(0.5));
        assertEquals(0.0, certainSuccess.mode(), 0.0);
        assertEquals(0.0, certainSuccess.mean(), 0.0);
        assertEquals(0.0, certainSuccess.variance(), 0.0);
        assertTrue(Double.isInfinite(certainSuccess.skewness()) && certainSuccess.skewness() > 0.0);
        assertTrue(Double.isInfinite(certainSuccess.kurtosisExcess()) && certainSuccess.kurtosisExcess() > 0.0);
        assertTrue(Double.isInfinite(certainSuccess.kurtosis()) && certainSuccess.kurtosis() > 0.0);
        assertEquals(0.0, certainSuccess.quantile(0.5), 0.0);
        assertEquals(0.0, certainSuccess.quantileUpperTail(0.5), 0.0);
        assertThrows(IllegalArgumentException.class, () -> new NegativeBinomialDistribution(0.0, 0.4));
        assertThrows(IllegalArgumentException.class, () -> new NegativeBinomialDistribution(2.0, Double.NaN));
    }
}