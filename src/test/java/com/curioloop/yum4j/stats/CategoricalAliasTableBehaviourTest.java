package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class CategoricalAliasTableBehaviourTest {

    private static final int SAMPLES = 10_000;
    private static final double TOL = 2.0 / Math.sqrt(SAMPLES); // ~0.02

    /**
     * For a fixed seed on the standard L64X128MixRandom, the Walker
     * alias-table sampler should hit each category with empirical
     * frequency within a 2/sqrt(N) band (~2 SE for Bernoulli under the
     * null). We verify across K ∈ {2, 10, 1000}.
     */
    @Test
    void empiricalFrequenciesRecoverTrueProbsK2() {
        double[] probs = {0.25, 0.75};
        runFrequencyCheck(probs, 0xBEEFL);
    }

    @Test
    void empiricalFrequenciesRecoverTrueProbsK10() {
        double[] probs = new double[10];
        // arbitrary probabilities summing to 1
        double sum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            probs[i] = 1.0 + i * 0.3;
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) probs[i] /= sum;
        runFrequencyCheck(probs, 0xCAFEL);
    }

    @Test
    void empiricalFrequenciesRecoverTrueProbsK1000() {
        int K = 1000;
        double[] probs = new double[K];
        RandomGenerator aux = RandomGeneratorFactory.of("L64X128MixRandom").create(0xF00DCAFEL);
        double sum = 0.0;
        for (int i = 0; i < K; i++) {
            probs[i] = aux.nextDouble() + 1e-6;
            sum += probs[i];
        }
        for (int i = 0; i < K; i++) probs[i] /= sum;
        runFrequencyCheck(probs, 0xDEADBEEFL);
    }

    private static void runFrequencyCheck(double[] probs, long seed) {
        CategoricalDistribution d = new CategoricalDistribution(probs);
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(seed);
        double[] out = new double[SAMPLES];
        d.sample(g, SAMPLES, out, 0, 1);

        int K = probs.length;
        int[] counts = new int[K];
        for (int i = 0; i < SAMPLES; i++) {
            int k = (int) out[i];
            assertThat(k).as("sample[%d] in range", i).isBetween(0, K - 1);
            counts[k]++;
        }

        int maxDeviatingIndex = -1;
        double maxDeviation = 0.0;
        for (int k = 0; k < K; k++) {
            double empirical = counts[k] / (double) SAMPLES;
            double diff = Math.abs(empirical - probs[k]);
            if (diff > maxDeviation) {
                maxDeviation = diff;
                maxDeviatingIndex = k;
            }
        }
        assertThat(maxDeviation)
            .as("max |empirical - true| at k=%d for K=%d", maxDeviatingIndex, K)
            .isLessThanOrEqualTo(TOL);
    }

    @Test
    void zeroWeightCategoriesAreNeverSampled() {
        double[] probs = {0.0, 1.0, 0.0, 0.0};
        CategoricalDistribution d = new CategoricalDistribution(probs);
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(1L);
        for (int i = 0; i < 1000; i++) {
            assertThat((int) d.sample(g)).isEqualTo(1);
        }
    }

    @Test
    void rejectsEmptyOrAllZeroInput() {
        assertThatThrownBy(() -> new CategoricalDistribution(new double[0]))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new CategoricalDistribution(new double[]{0.0, 0.0}))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new CategoricalDistribution(new double[]{-0.1, 0.5}))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new CategoricalDistribution(new double[]{Double.NaN, 0.5}))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new CategoricalDistribution(new double[]{Double.POSITIVE_INFINITY, 0.5}))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void meanAndVarianceAreCorrect() {
        double[] probs = {0.2, 0.5, 0.3}; // values 0, 1, 2
        CategoricalDistribution d = new CategoricalDistribution(probs);
        double expectedMean = 0.0 * 0.2 + 1.0 * 0.5 + 2.0 * 0.3;
        double expectedVar = 0.2 * (0 - expectedMean) * (0 - expectedMean)
            + 0.5 * (1 - expectedMean) * (1 - expectedMean)
            + 0.3 * (2 - expectedMean) * (2 - expectedMean);
        assertThat(d.mean()).isCloseTo(expectedMean, org.assertj.core.data.Offset.offset(1e-15));
        assertThat(d.variance()).isCloseTo(expectedVar, org.assertj.core.data.Offset.offset(1e-15));
    }

    @Test
    void quantileAndCdfAreConsistent() {
        double[] probs = {0.1, 0.3, 0.2, 0.4};
        CategoricalDistribution d = new CategoricalDistribution(probs);
        // cumulative: 0.1, 0.4, 0.6, 1.0
        assertThat(d.cdf(-1.0)).isZero();
        assertThat(d.cdf(0.0)).isCloseTo(0.1, org.assertj.core.data.Offset.offset(1e-15));
        assertThat(d.cdf(1.0)).isCloseTo(0.4, org.assertj.core.data.Offset.offset(1e-15));
        assertThat(d.cdf(2.0)).isCloseTo(0.6, org.assertj.core.data.Offset.offset(1e-15));
        assertThat(d.cdf(3.0)).isEqualTo(1.0);
        assertThat(d.cdf(5.0)).isEqualTo(1.0);

        // quantile() returns smallest k with cumulative >= p
        assertThat(d.quantile(0.05)).isEqualTo(0.0);
        assertThat(d.quantile(0.1)).isEqualTo(0.0);
        assertThat(d.quantile(0.11)).isEqualTo(1.0);
        assertThat(d.quantile(0.4)).isEqualTo(1.0);
        assertThat(d.quantile(0.5)).isEqualTo(2.0);
        assertThat(d.quantile(0.99)).isEqualTo(3.0);
        assertThat(d.quantile(1.0)).isEqualTo(3.0);
    }
}
