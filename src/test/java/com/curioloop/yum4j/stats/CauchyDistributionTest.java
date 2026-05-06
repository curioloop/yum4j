package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CauchyDistributionTest {

    @Test
    void accessorsAndLocationStatisticsMatchBoostDefinitions() {
        CauchyDistribution distribution = new CauchyDistribution(1.5, 2.0);

        assertEquals(1.5, distribution.location(), 0.0);
        assertEquals(2.0, distribution.scale(), 0.0);
        assertEquals(Double.NEGATIVE_INFINITY, distribution.range().lower());
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper());
        assertEquals(Double.NEGATIVE_INFINITY, distribution.support().lower());
        assertEquals(Double.POSITIVE_INFINITY, distribution.support().upper());
        assertEquals(1.5, distribution.mode(), 0.0);
        assertEquals(1.5, distribution.median(), 0.0);
    }

    @Test
    void evaluationSurfaceMatchesBoostClosedForms() {
        CauchyDistribution distribution = new CauchyDistribution(1.5, 2.0);
        double x = 0.75;
        double standardized = (x - 1.5) / 2.0;

        assertEquals(1.0 / (Math.PI * 2.0 * (1.0 + standardized * standardized)), distribution.pdf(x), 1.0e-15);
        assertEquals(Math.atan2(1.0, (1.5 - x) / 2.0) / Math.PI, distribution.cdf(x), 1.0e-15);
        assertEquals(Math.atan2(1.0, (x - 1.5) / 2.0) / Math.PI, distribution.ccdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);
        assertEquals(0.0, distribution.pdf(Double.NEGATIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.pdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.cdf(Double.NEGATIVE_INFINITY), 0.0);
        assertEquals(1.0, distribution.cdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(1.0, distribution.ccdf(Double.NEGATIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.ccdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(Double.NEGATIVE_INFINITY, distribution.quantile(0.0));
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantile(1.0));
        assertEquals(Double.POSITIVE_INFINITY, distribution.quantileUpperTail(0.0));
        assertEquals(Double.NEGATIVE_INFINITY, distribution.quantileUpperTail(1.0));
        assertEquals(1.5, distribution.quantile(0.5), 0.0);
        assertEquals(1.5, distribution.quantileUpperTail(0.5), 0.0);
        assertEquals(distribution.quantileUpperTail(0.2),
            Distributions.quantile(Distributions.complement(distribution, 0.2)), 1.0e-15);
    }

    @Test
    void undefinedMomentsRaise() {
        CauchyDistribution distribution = new CauchyDistribution();

        assertThrows(ArithmeticException.class, distribution::mean);
        assertThrows(ArithmeticException.class, distribution::variance);
        assertThrows(ArithmeticException.class, distribution::skewness);
        assertThrows(ArithmeticException.class, distribution::kurtosis);
        assertThrows(ArithmeticException.class, distribution::kurtosisExcess);
        assertThrows(ArithmeticException.class, () -> Distributions.standardDeviation(distribution));
        assertThrows(ArithmeticException.class, () -> Distributions.coefficientOfVariation(distribution));
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new CauchyDistribution(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new CauchyDistribution(0.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new CauchyDistribution().cdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new CauchyDistribution().quantile(-0.1));
    }
}