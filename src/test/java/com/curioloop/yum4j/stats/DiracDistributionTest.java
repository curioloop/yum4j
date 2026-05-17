package com.curioloop.yum4j.stats;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;

class DiracDistributionTest extends StatsDistributionFixtureTestBase {

    @Test
    void logPdfGridMatchesFixture() {
        JsonNode node = loadFixture("dirac_basic.json");
        double location = node.get("location").asDouble();
        double[] xs = asDoubleArray(node.get("xs"));
        double[] expected = asDoubleArray(node.get("logPdf"));

        DiracDistribution d = new DiracDistribution(location);
        for (int i = 0; i < xs.length; i++) {
            double got = d.logPdf(xs[i]);
            if (Double.isInfinite(expected[i])) {
                assertThat(Double.doubleToRawLongBits(got))
                    .as("x=%s expected=%s", xs[i], expected[i])
                    .isEqualTo(Double.doubleToRawLongBits(expected[i]));
            } else {
                assertThat(got).as("x=%s", xs[i]).isEqualTo(expected[i]);
            }
        }
    }

    @Test
    void pdfMatchesPointMassSemantics() {
        DiracDistribution d = new DiracDistribution(1.25);
        assertThat(d.pdf(1.25)).isEqualTo(Double.POSITIVE_INFINITY);
        assertThat(d.pdf(0.0)).isZero();
        assertThat(d.pdf(1.249999)).isZero();
    }

    @Test
    void cdfIsHeavisideStep() {
        DiracDistribution d = new DiracDistribution(2.0);
        assertThat(d.cdf(1.99)).isZero();
        assertThat(d.cdf(2.0)).isEqualTo(1.0);
        assertThat(d.cdf(100.0)).isEqualTo(1.0);
    }

    @Test
    void sampleAlwaysReturnsLocation() {
        DiracDistribution d = new DiracDistribution(-7.5);
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(0xBEEFL);
        for (int i = 0; i < 100; i++) {
            assertThat(d.sample(g)).isEqualTo(-7.5);
        }
    }

    @Test
    void batchSampleFillsStride1() {
        DiracDistribution d = new DiracDistribution(3.14);
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(0xFAFAL);
        double[] out = new double[10];
        d.sample(g, 10, out, 0, 1);
        for (double v : out) assertThat(v).isEqualTo(3.14);
    }

    @Test
    void batchSampleHonoursStride() {
        DiracDistribution d = new DiracDistribution(9.0);
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(1L);
        double[] out = new double[10];
        Arrays.fill(out, -1.0);
        d.sample(g, 5, out, 0, 2);
        for (int i = 0; i < 10; i++) {
            assertThat(out[i])
                .as("out[%d]", i)
                .isEqualTo(i % 2 == 0 ? 9.0 : -1.0);
        }
    }

    @Test
    void meanVarianceRangeSupport() {
        DiracDistribution d = new DiracDistribution(1.25);
        assertThat(d.mean()).isEqualTo(1.25);
        assertThat(d.variance()).isZero();
        assertThat(d.range()._1()).isEqualTo(1.25);
        assertThat(d.range()._2()).isEqualTo(1.25);
        assertThat(d.support()._1()).isEqualTo(1.25);
        assertThat(d.support()._2()).isEqualTo(1.25);
    }

    @Test
    void logPdfBatchHonoursStride() {
        DiracDistribution d = new DiracDistribution(0.5);
        double[] xs = {0.5, 0.0, 0.5, 0.5, 0.0};
        double[] out = new double[5];
        d.batch(Distribution.Metric.LOG_PDF, xs, 0, 2, 3, out, 0);
        // reads xs[0], xs[2], xs[4] = 0.5, 0.5, 0.0
        assertThat(out[0]).isEqualTo(0.0);
        assertThat(out[1]).isEqualTo(0.0);
        assertThat(out[2]).isEqualTo(Double.NEGATIVE_INFINITY);
    }
}
