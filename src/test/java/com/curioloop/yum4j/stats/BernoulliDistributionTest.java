package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BernoulliDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        BernoulliDistribution distribution = new BernoulliDistribution(0.3);

        assertEquals(0.3, distribution.successProbability(), 0.0);
        assertEquals(0.0, distribution.range().lower());
        assertEquals(1.0, distribution.range().upper());
        assertEquals(0.0, distribution.support().lower());
        assertEquals(1.0, distribution.support().upper());
        assertEquals(0.0, distribution.mode(), 0.0);
        assertEquals(0.0, distribution.median(), 0.0);
        assertEquals(0.3, distribution.mean(), 1.0e-15);
        assertEquals(0.21, distribution.variance(), 1.0e-15);
        assertEquals((1.0 - 0.6) / Math.sqrt(0.21), distribution.skewness(), 1.0e-15);
        assertEquals(1.0 / 0.7 + 1.0 / 0.3 - 3.0, distribution.kurtosis(), 1.0e-15);
        assertEquals(1.0 / 0.7 + 1.0 / 0.3 - 6.0, distribution.kurtosisExcess(), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostDiscreteSemantics() {
        BernoulliDistribution distribution = new BernoulliDistribution(0.3);

        assertEquals(0.7, distribution.pdf(0.0), 0.0);
        assertEquals(0.3, distribution.pdf(1.0), 0.0);
        assertEquals(0.7, distribution.cdf(0.0), 0.0);
        assertEquals(1.0, distribution.cdf(1.0), 0.0);
        assertEquals(0.3, distribution.ccdf(0.0), 0.0);
        assertEquals(0.0, distribution.ccdf(1.0), 0.0);
        assertEquals(0.3, Distributions.cdf(Distributions.complement(distribution, 0.0)), 0.0);

        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantile(0.7), 0.0);
        assertEquals(1.0, distribution.quantile(0.7000000000000001), 0.0);
        assertEquals(1.0, distribution.quantile(1.0), 0.0);
        assertEquals(1.0, distribution.quantileUpperTail(0.0), 0.0);
        assertEquals(1.0, distribution.quantileUpperTail(0.7), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
        assertEquals(1.0, Distributions.quantile(Distributions.complement(distribution, 0.7)), 0.0);
    }

    @Test
    void rejectsInvalidInputsAndMatchesBoostDegenerateHighMomentSemantics() {
        assertThrows(IllegalArgumentException.class, () -> new BernoulliDistribution(-0.1));
        assertThrows(IllegalArgumentException.class, () -> new BernoulliDistribution(1.1));
        assertThrows(IllegalArgumentException.class, () -> new BernoulliDistribution().pdf(0.5));
        assertThrows(IllegalArgumentException.class, () -> new BernoulliDistribution().cdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new BernoulliDistribution().quantile(-0.1));

        BernoulliDistribution alwaysFailure = new BernoulliDistribution(0.0);
        BernoulliDistribution alwaysSuccess = new BernoulliDistribution(1.0);

        assertEquals(Double.POSITIVE_INFINITY, alwaysFailure.skewness());
        assertEquals(Double.NEGATIVE_INFINITY, alwaysSuccess.skewness());
        assertTrue(Double.isInfinite(alwaysFailure.kurtosisExcess()) && alwaysFailure.kurtosisExcess() > 0.0);
        assertTrue(Double.isInfinite(alwaysSuccess.kurtosisExcess()) && alwaysSuccess.kurtosisExcess() > 0.0);
        assertTrue(Double.isInfinite(alwaysFailure.kurtosis()) && alwaysFailure.kurtosis() > 0.0);
        assertTrue(Double.isInfinite(alwaysSuccess.kurtosis()) && alwaysSuccess.kurtosis() > 0.0);
    }
}