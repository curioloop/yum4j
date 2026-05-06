package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Beta;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BetaDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        BetaDistribution distribution = new BetaDistribution(2.5, 3.5);
        double sum = 6.0;

        assertEquals(2.5, distribution.alpha());
        assertEquals(3.5, distribution.beta());
        assertEquals(0.375, distribution.mode(), 1.0e-15);
        assertEquals(2.5 / sum, distribution.mean(), 1.0e-15);
        assertEquals(2.5 * 3.5 / (sum * sum * (sum + 1.0)), distribution.variance(), 1.0e-15);
        assertEquals(
            2.0 * (3.5 - 2.5) * Math.sqrt(sum + 1.0) / ((sum + 2.0) * Math.sqrt(2.5 * 3.5)),
            distribution.skewness(),
            1.0e-15
        );

        double excess = 6.0 * ((2.5 - 3.5) * (2.5 - 3.5) * (sum + 1.0) - 2.5 * 3.5 * (sum + 2.0))
            / (2.5 * 3.5 * (sum + 2.0) * (sum + 3.0));
        assertEquals(3.0 + excess, distribution.kurtosis(), 1.0e-15);
        assertEquals(excess, distribution.kurtosisExcess(), 1.0e-15);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
    }

    @Test
    void evaluationSurfaceMatchesBetaIdentities() {
        BetaDistribution distribution = new BetaDistribution(2.5, 3.5);
        double x = 0.4;

        assertEquals(Beta.ibetaDerivative(2.5, 3.5, x), distribution.pdf(x), 1.0e-15);
        assertEquals(Beta.ibeta(2.5, 3.5, x), distribution.cdf(x), 1.0e-15);
        assertEquals(Beta.ibetac(2.5, 3.5, x), distribution.ccdf(x), 1.0e-15);

        double probability = distribution.cdf(x);
        assertEquals(x, distribution.quantile(probability), 1.0e-10);
        assertEquals(x, distribution.quantileUpperTail(distribution.ccdf(x)), 1.0e-10);
    }

    @Test
    void supportsBoundaryValuesAndEndpointDensitySemantics() {
        BetaDistribution regular = new BetaDistribution(1.0, 2.0);
        BetaDistribution lowerSingular = new BetaDistribution(0.5, 2.0);
        BetaDistribution upperSingular = new BetaDistribution(2.0, 0.5);

        assertEquals(2.0, regular.pdf(0.0), 1.0e-15);
        assertEquals(0.0, regular.pdf(1.0), 0.0);
        assertTrue(Double.isInfinite(lowerSingular.pdf(0.0)));
        assertTrue(Double.isInfinite(upperSingular.pdf(1.0)));

        assertEquals(0.0, regular.cdf(0.0), 0.0);
        assertEquals(1.0, regular.ccdf(0.0), 0.0);
        assertEquals(1.0, regular.cdf(1.0), 0.0);
        assertEquals(0.0, regular.ccdf(1.0), 0.0);
        assertEquals(0.0, regular.quantile(0.0), 0.0);
        assertEquals(1.0, regular.quantile(1.0), 0.0);
        assertEquals(1.0, regular.quantileUpperTail(0.0), 0.0);
        assertEquals(0.0, regular.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void rejectsInvalidInputsAndUndefinedMode() {
        assertThrows(IllegalArgumentException.class, () -> new BetaDistribution(0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new BetaDistribution(1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new BetaDistribution(2.0, 3.0).cdf(-0.1));
        assertThrows(IllegalArgumentException.class, () -> new BetaDistribution(2.0, 3.0).pdf(1.1));
        assertThrows(IllegalArgumentException.class, () -> new BetaDistribution(2.0, 3.0).quantile(Double.NaN));
        assertThrows(ArithmeticException.class, () -> new BetaDistribution(1.0, 3.0).mode());
    }
}