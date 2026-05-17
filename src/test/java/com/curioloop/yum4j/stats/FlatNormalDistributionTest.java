package com.curioloop.yum4j.stats;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class FlatNormalDistributionTest extends StatsDistributionFixtureTestBase {

    @Test
    void logPdfIsZeroForFinitePointsAndMinusInfinityForNonFinite() {
        JsonNode node = loadFixture("flatnormal_basic.json");
        double location = node.get("location").asDouble();
        double[] xs = asDoubleArray(node.get("xs"));
        double[] expected = asDoubleArray(node.get("logPdf"));

        FlatNormalDistribution d = new FlatNormalDistribution(location);

        for (int i = 0; i < xs.length; i++) {
            double got = d.logPdf(xs[i]);
            if (Double.isInfinite(expected[i])) {
                assertThat(Double.doubleToRawLongBits(got))
                    .as("x=%s", xs[i])
                    .isEqualTo(Double.doubleToRawLongBits(expected[i]));
            } else {
                assertThat(got).as("x=%s", xs[i]).isEqualTo(expected[i]);
            }
        }
    }

    @Test
    void logPdfBatchAgrees() {
        FlatNormalDistribution d = new FlatNormalDistribution(0.0);
        double[] xs = {-1.0, 0.0, 1.0, Double.NaN, Double.POSITIVE_INFINITY};
        double[] out = new double[xs.length];
        d.batch(Distribution.Metric.LOG_PDF, xs, 0, 1, xs.length, out, 0);
        assertThat(out[0]).isEqualTo(0.0);
        assertThat(out[1]).isEqualTo(0.0);
        assertThat(out[2]).isEqualTo(0.0);
        assertThat(out[3]).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(out[4]).isEqualTo(Double.NEGATIVE_INFINITY);
    }

    @Test
    void sampleThrowsUnsupported() {
        FlatNormalDistribution d = new FlatNormalDistribution(0.0);
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(1L);

        assertThatThrownBy(() -> d.sample(g))
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("missing observations")
            .hasMessageContaining("NormalDistribution");

        assertThatThrownBy(() -> d.sample(g, 10, new double[10], 0, 1))
            .isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void cdfAndQuantileThrow() {
        FlatNormalDistribution d = new FlatNormalDistribution(0.0);
        assertThatThrownBy(() -> d.cdf(0.0)).isInstanceOf(UnsupportedOperationException.class);
        assertThatThrownBy(() -> d.quantile(0.5)).isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void meanIsLocationVarianceIsInfinite() {
        FlatNormalDistribution d = new FlatNormalDistribution(7.25);
        assertThat(d.mean()).isEqualTo(7.25);
        assertThat(d.variance()).isEqualTo(Double.POSITIVE_INFINITY);
    }
}
