package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Beta;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FisherFDistributionTest {

    @Test
    void accessorsMatchBoostRangeAndSupport() {
        FisherFDistribution distribution = new FisherFDistribution(8.0, 10.0);

        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
    }

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        FisherFDistribution distribution = new FisherFDistribution(8.0, 10.0);

        assertEquals(8.0, distribution.degreesOfFreedom1());
        assertEquals(10.0, distribution.degreesOfFreedom2());
        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(10.0 * 6.0 / (8.0 * 12.0), distribution.mode(), 1.0e-15);
        assertEquals(10.0 / 8.0, distribution.mean(), 1.0e-15);
        assertEquals(2.0 * 100.0 * 16.0 / (8.0 * 64.0 * 6.0), distribution.variance(), 1.0e-15);

        double skewness = (16.0 + 10.0 - 2.0) * Math.sqrt(8.0 * 6.0)
            / (4.0 * Math.sqrt(8.0 * 16.0));
        assertEquals(skewness, distribution.skewness(), 1.0e-15);

        double excess = 12.0 * (8.0 * (5.0 * 10.0 - 22.0) * 16.0 + 6.0 * 8.0 * 8.0)
            / (8.0 * 4.0 * 2.0 * 16.0);
        assertEquals(3.0 + excess, distribution.kurtosis(), 1.0e-15);
        assertEquals(excess, distribution.kurtosisExcess(), 1.0e-15);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
    }

    @Test
    void evaluationSurfaceMatchesBetaIdentities() {
        FisherFDistribution distribution = new FisherFDistribution(8.0, 10.0);
        double x = 1.5;
        double y = x / (x + 10.0 / 8.0);
        double derivative = (10.0 / 8.0) / Math.pow(x + 10.0 / 8.0, 2.0);

        assertEquals(Beta.ibetaDerivative(4.0, 5.0, y) * derivative, distribution.pdf(x), 1.0e-15);
        assertEquals(Beta.ibeta(4.0, 5.0, y), distribution.cdf(x), 1.0e-15);
        assertEquals(Beta.ibetac(4.0, 5.0, y), distribution.ccdf(x), 1.0e-15);

        double probability = distribution.cdf(x);
        assertEquals(x, distribution.quantile(probability), 1.0e-10);
        assertEquals(x, distribution.quantileUpperTail(distribution.ccdf(x)), 1.0e-10);
    }

    @Test
    void supportsBoundaryValuesAndEndpointDensitySemantics() {
        FisherFDistribution regular = new FisherFDistribution(8.0, 10.0);
        FisherFDistribution flatBoundary = new FisherFDistribution(2.0, 10.0);
        FisherFDistribution singularBoundary = new FisherFDistribution(1.5, 10.0);

        assertEquals(0.0, regular.pdf(0.0), 0.0);
        assertEquals(1.0, flatBoundary.pdf(0.0), 0.0);
        assertTrue(Double.isInfinite(singularBoundary.pdf(0.0)));
        assertEquals(0.0, regular.pdf(Double.POSITIVE_INFINITY), 0.0);

        assertEquals(0.0, regular.cdf(0.0), 0.0);
        assertEquals(1.0, regular.ccdf(0.0), 0.0);
        assertEquals(1.0, regular.cdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, regular.ccdf(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(0.0, regular.quantile(0.0), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, regular.quantile(1.0));
        assertEquals(Double.POSITIVE_INFINITY, regular.quantileUpperTail(0.0));
        assertEquals(0.0, regular.quantileUpperTail(1.0), 0.0);
    }

    @Test
    void rejectsInvalidInputsAndUndefinedMoments() {
        assertThrows(IllegalArgumentException.class, () -> new FisherFDistribution(0.0, 10.0));
        assertThrows(IllegalArgumentException.class, () -> new FisherFDistribution(8.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new FisherFDistribution(8.0, 10.0).cdf(-1.0));
        assertThrows(IllegalArgumentException.class, () -> new FisherFDistribution(8.0, 10.0).quantile(1.1));
        assertThrows(ArithmeticException.class, () -> new FisherFDistribution(2.0, 10.0).mode());
        assertThrows(ArithmeticException.class, () -> new FisherFDistribution(8.0, 2.0).mean());
        assertThrows(ArithmeticException.class, () -> new FisherFDistribution(8.0, 4.0).variance());
        assertThrows(ArithmeticException.class, () -> new FisherFDistribution(8.0, 6.0).skewness());
        assertThrows(ArithmeticException.class, () -> new FisherFDistribution(8.0, 8.0).kurtosisExcess());
        assertThrows(ArithmeticException.class, () -> new FisherFDistribution(8.0, 8.0).kurtosis());
    }
}