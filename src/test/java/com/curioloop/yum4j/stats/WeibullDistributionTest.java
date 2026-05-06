package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Gamma;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class WeibullDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        WeibullDistribution distribution = new WeibullDistribution(2.5, 1.6);
        double gamma1 = Gamma.tgamma(1.0 + 1.0 / 2.5);
        double gamma2 = Gamma.tgamma(1.0 + 2.0 / 2.5);
        double variance = 1.6 * 1.6 * (gamma2 - gamma1 * gamma1);

        assertEquals(2.5, distribution.shape(), 0.0);
        assertEquals(1.6, distribution.scale(), 0.0);
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper());
        assertEquals(Double.MIN_NORMAL, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper());
        assertEquals(1.6 * Math.pow((2.5 - 1.0) / 2.5, 1.0 / 2.5), distribution.mode(), 1.0e-15);
        assertEquals(1.6 * Math.pow(Math.log(2.0), 1.0 / 2.5), distribution.median(), 1.0e-15);
        assertEquals(1.6 * gamma1, distribution.mean(), 1.0e-15);
        assertEquals(variance, distribution.variance(), 1.0e-15);
        assertEquals(Distributions.standardDeviation(distribution), Math.sqrt(variance), 1.0e-15);
        assertEquals(Math.sqrt(variance) / distribution.mean(), Distributions.coefficientOfVariation(distribution), 1.0e-15);
    }

    @Test
    void evaluationSurfaceMatchesBoostClosedForms() {
        WeibullDistribution distribution = new WeibullDistribution(2.5, 1.6);
        double x = 1.25;
        double power = Math.pow(x / 1.6, 2.5);

        assertEquals(Math.exp(-power) * Math.pow(x / 1.6, 1.5) * 2.5 / 1.6, distribution.pdf(x), 1.0e-15);
        assertEquals(-Math.expm1(-power), distribution.cdf(x), 1.0e-15);
        assertEquals(Math.exp(-power), distribution.ccdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);
        assertEquals(distribution.quantileUpperTail(0.2),
            Distributions.quantile(Distributions.complement(distribution, 0.2)), 1.0e-15);
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
        assertEquals(0.0, distribution.pdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(1.0, distribution.cdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, distribution.ccdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(distribution.pdf(x) / distribution.ccdf(x), Distributions.hazard(distribution, x), 1.0e-15);
        assertEquals(power, Distributions.chf(distribution, x), 1.0e-15);
    }

    @Test
    void terminalQuantileBoundariesMatchBoostOverflowPolicy() {
        WeibullDistribution distribution = new WeibullDistribution(2.5, 1.6);

        assertThrows(ArithmeticException.class, () -> distribution.quantile(1.0));
        assertThrows(ArithmeticException.class, () -> distribution.quantileUpperTail(0.0));
        assertEquals(0.0, distribution.quantile(0.0), 0.0);
        assertEquals(0.0, distribution.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void boundaryShapeCasesMatchBoostDefinitions() {
        WeibullDistribution exponentialCase = new WeibullDistribution(1.0, 1.6);
        WeibullDistribution singularCase = new WeibullDistribution(0.75, 1.6);
        WeibullDistribution interiorBoundaryMode = new WeibullDistribution(0.75, 1.6);

        assertEquals(1.0 / 1.6, exponentialCase.pdf(0.0), 1.0e-15);
        assertThrows(ArithmeticException.class, () -> singularCase.pdf(0.0));
        assertEquals(0.0, interiorBoundaryMode.mode(), 0.0);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new WeibullDistribution(0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new WeibullDistribution(1.0, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new WeibullDistribution(1.0, 1.0).cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new WeibullDistribution(1.0, 1.0).quantile(Double.NaN));
    }
}