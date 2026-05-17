package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.test.RecordedRandomGenerator;
import com.fasterxml.jackson.databind.JsonNode;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class StatsDistributionBehaviorTest extends StatsDistributionFixtureTestBase {

    @Test
    void categoricalAccessorsAreDefensive() {
        double[] raw = {2.0, 4.0, 4.0};
        CategoricalDistribution distribution = new CategoricalDistribution(raw);

        double[] probs = distribution.probs();
        probs[0] = -999.0;
        assertThat(distribution.probs()[0]).isCloseTo(0.2, Offset.offset(1e-15));

        double[] logProbs = distribution.logProbs();
        logProbs[0] = Double.NaN;
        assertThat(distribution.logProbs()[0]).isCloseTo(Math.log(0.2), Offset.offset(1e-14));
    }

    @Test
    void categoricalAliasTableMatchesReference() {
        JsonNode fixture = loadFixture("categorical.json", "sample");
        double[] probs = asDoubleArray(fixture.get("probs"));
        double[] expectedAliasProb = asDoubleArray(fixture.get("aliasProb"));
        int[] expectedAlias = asIntArray(fixture.get("alias"));

        CategoricalDistribution distribution = new CategoricalDistribution(probs);
        int categories = probs.length;
        for (int category = 0; category < categories; category++) {
            double aliasProbability = expectedAliasProb[category];
            int alias = expectedAlias[category];
            if (aliasProbability < 1.0 - 1e-12) {
                double belowThreshold = (category + aliasProbability * 0.5) / categories;
                double aboveThreshold = (category + (aliasProbability + 1.0) * 0.5) / categories;
                assertThat((int) distribution.sample(new RecordedRandomGenerator(new double[]{belowThreshold})))
                    .as("slot branch for category=%d", category)
                    .isEqualTo(category);
                assertThat((int) distribution.sample(new RecordedRandomGenerator(new double[]{aboveThreshold})))
                    .as("alias branch for category=%d -> %d", category, alias)
                    .isEqualTo(alias);
            } else {
                double midpoint = (category + 0.5) / categories;
                assertThat((int) distribution.sample(new RecordedRandomGenerator(new double[]{midpoint})))
                    .as("whole slot category=%d", category)
                    .isEqualTo(category);
            }
        }
    }

    @Test
    void exponentialLogPdfReturnsNegInfForNegativeInputs() {
        ExponentialDistribution distribution = new ExponentialDistribution(1.0);
        assertThat(distribution.logPdf(-1.0)).isEqualTo(Double.NEGATIVE_INFINITY);
        double[] x = {-0.5, 0.0, 1.0};
        double[] out = new double[3];
        distribution.batch(Distribution.Metric.LOG_PDF, x, 0, 1, 3, out, 0);
        assertThat(out[0]).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(out[1]).isCloseTo(distribution.logPdf(0.0), Offset.offset(0.0));
        assertThat(out[2]).isCloseTo(distribution.logPdf(1.0), Offset.offset(0.0));
    }

    @Test
    void gammaLogPdfReturnsNegInfAtNonPositiveInputs() {
        GammaDistribution distribution = new GammaDistribution(2.5, 1.3);
        assertThat(distribution.logPdf(0.0)).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(distribution.logPdf(-1.0)).isEqualTo(Double.NEGATIVE_INFINITY);

        double[] mixed = {-2.0, 0.0, 1.0};
        double[] out = new double[3];
        distribution.batch(Distribution.Metric.LOG_PDF, mixed, 0, 1, 3, out, 0);
        assertThat(out[0]).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(out[1]).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(out[2]).isFinite();
    }

    @Test
    void logNormalLogPdfReturnsNegInfBelowZero() {
        LogNormalDistribution distribution = new LogNormalDistribution(0.0, 1.0);
        assertThat(distribution.logPdf(0.0)).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(distribution.logPdf(-1.0)).isEqualTo(Double.NEGATIVE_INFINITY);

        double[] x = {0.0, -0.5, 1.0};
        double[] out = new double[3];
        distribution.batch(Distribution.Metric.LOG_PDF, x, 0, 1, 3, out, 0);
        assertThat(out[0]).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(out[1]).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(out[2]).isCloseTo(distribution.logPdf(1.0), Offset.offset(0.0));
    }

    @Test
    void uniformLogPdfReturnsNegInfOutsideSupport() {
        UniformDistribution distribution = new UniformDistribution(0.0, 1.0);
        assertThat(distribution.logPdf(-0.1)).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(distribution.logPdf(1.1)).isEqualTo(Double.NEGATIVE_INFINITY);
    }
}