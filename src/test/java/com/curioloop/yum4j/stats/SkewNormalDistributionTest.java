package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Normal;
import com.curioloop.yum4j.math.OwensT;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SkewNormalDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        SkewNormalDistribution distribution = new SkewNormalDistribution(0.5, 1.2, 1.5);
        double delta = 1.5 / Math.sqrt(1.0 + 1.5 * 1.5);
        double variance = 1.2 * 1.2 * (1.0 - 2.0 * delta * delta / Math.PI);

        assertEquals(0.5, distribution.location(), 0.0);
        assertEquals(1.2, distribution.scale(), 0.0);
        assertEquals(1.5, distribution.shape(), 0.0);
        assertEquals(Double.NEGATIVE_INFINITY, distribution.range().lower(), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, distribution.range().upper(), 0.0);
        assertEquals(-Double.MAX_VALUE, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
        assertEquals(0.5 + 1.2 * delta * Math.sqrt(2.0 / Math.PI), distribution.mean(), 1.0e-15);
        assertEquals(variance, distribution.variance(), 1.0e-15);
        assertEquals(3.0 + distribution.kurtosisExcess(), distribution.kurtosis(), 1.0e-15);
        assertTrue(Double.isFinite(distribution.mode()));
    }

    @Test
    void evaluationSurfaceMatchesOwensTDefinitionAndNormalReduction() {
        SkewNormalDistribution distribution = new SkewNormalDistribution(0.5, 1.2, 1.5);
        double x = 0.25;
        double z = (x - 0.5) / 1.2;

        assertEquals(2.0 * Normal.pdf(z) * Normal.cdf(1.5 * z) / 1.2, distribution.pdf(x), 1.0e-15);
        assertEquals(Normal.cdf(z) - 2.0 * OwensT.owensT(z, 1.5), distribution.cdf(x), 1.0e-15);
        assertEquals(distribution.ccdf(x), Distributions.cdf(Distributions.complement(distribution, x)), 1.0e-15);

        SkewNormalDistribution normalCase = new SkewNormalDistribution(0.5, 1.2, 0.0);
        NormalDistribution normal = new NormalDistribution(0.5, 1.2);
        assertEquals(normal.pdf(x), normalCase.pdf(x), 1.0e-15);
        assertEquals(normal.cdf(x), normalCase.cdf(x), 1.0e-15);
        assertEquals(normal.quantile(0.75), normalCase.quantile(0.75), 1.0e-12);

        double quantile = distribution.quantile(0.7);
        assertEquals(0.7, distribution.cdf(quantile), 1.0e-10);
        double upperTailQuantile = distribution.quantileUpperTail(0.25);
        assertEquals(0.25, distribution.ccdf(upperTailQuantile), 1.0e-10);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> new SkewNormalDistribution(Double.NaN, 1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new SkewNormalDistribution(0.0, 0.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new SkewNormalDistribution().cdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new SkewNormalDistribution().quantile(-0.1));
    }
}