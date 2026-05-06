package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ArcsineDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        ArcsineDistribution distribution = new ArcsineDistribution(2.0, 5.0);

        assertEquals(2.0, distribution.xMin());
        assertEquals(5.0, distribution.xMax());
        assertEquals(2.0, distribution.range().lower());
        assertEquals(5.0, distribution.range().upper());
        assertEquals(2.0, distribution.support().lower());
        assertEquals(5.0, distribution.support().upper());
        assertEquals(3.5, distribution.mean(), 1.0e-15);
        assertEquals(3.5, distribution.median(), 1.0e-15);
        assertEquals(9.0 / 8.0, distribution.variance(), 1.0e-15);
        assertEquals(0.0, distribution.skewness(), 1.0e-15);
        assertEquals(1.5, distribution.kurtosis(), 1.0e-15);
        assertEquals(-1.5, distribution.kurtosisExcess(), 1.0e-15);
        assertEquals(Math.sqrt(9.0 / 8.0), Distributions.standardDeviation(distribution), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesClosedFormsAndBoostBoundaryBehavior() {
        ArcsineDistribution distribution = new ArcsineDistribution(2.0, 5.0);
        double x = 3.125;

        assertEquals(1.0 / (Math.PI * Math.sqrt((x - 2.0) * (5.0 - x))), distribution.pdf(x), 1.0e-15);
        assertEquals(2.0 * Math.asin(Math.sqrt((x - 2.0) / 3.0)) / Math.PI, distribution.cdf(x), 1.0e-15);
        assertEquals(2.0 * Math.acos(Math.sqrt((x - 2.0) / 3.0)) / Math.PI, distribution.ccdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);

        assertTrue(Double.isInfinite(distribution.pdf(2.0)));
        assertTrue(Double.isInfinite(distribution.pdf(5.0)));
        assertEquals(0.0, distribution.cdf(2.0), 0.0);
        assertEquals(1.0, distribution.cdf(5.0), 0.0);
        assertEquals(1.0, distribution.ccdf(2.0), 0.0);
        assertEquals(0.0, distribution.ccdf(5.0), 0.0);
        assertEquals(2.0, distribution.quantile(0.0), 0.0);
        assertEquals(5.0, distribution.quantile(1.0), 0.0);
        assertEquals(2.0, distribution.quantileUpperTail(1.0), 0.0);
        assertEquals(5.0, distribution.quantileUpperTail(0.0), 0.0);
    }

    @Test
    void rejectsInvalidInputsAndUndefinedMode() {
        assertThrows(IllegalArgumentException.class, () -> new ArcsineDistribution(1.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new ArcsineDistribution(Double.NEGATIVE_INFINITY, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new ArcsineDistribution(0.0, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new ArcsineDistribution().cdf(-0.1));
        assertThrows(IllegalArgumentException.class, () -> new ArcsineDistribution().ccdf(1.1));
        assertThrows(IllegalArgumentException.class, () -> new ArcsineDistribution().quantile(Double.NaN));
        assertThrows(ArithmeticException.class, () -> new ArcsineDistribution().mode());
        assertThrows(ArithmeticException.class, () -> Distributions.coefficientOfVariation(new ArcsineDistribution(-1.0, 1.0)));
    }
}