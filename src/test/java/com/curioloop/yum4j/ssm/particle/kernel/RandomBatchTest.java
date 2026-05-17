package com.curioloop.yum4j.ssm.particle.kernel;

import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for {@link RandomBatch} implementations (both SIMD and scalar paths).
 *
 * <p>Validates statistical correctness (R4.8), determinism, and the scalar patch
 * that guards against {@code log(0) = -Infinity} in the Box–Muller transform.
 *
 * <p><b>Validates: Requirements R4.8</b>
 */
class RandomBatchTest {

    private static final long SEED = 0xDEAD_BEEF_CAFE_BABEL;
    private static final int N = 1_000_000;

    // ---- Gaussian statistical correctness ----------------------------

    @Test
    void nextGaussians_meanWithinBound() {
        RandomBatch rb = RandomBatch.of(SEED);
        double[] buf = new double[N];
        rb.nextGaussians(buf, 0, N);

        double mean = mean(buf, N);
        // R4.8: mean within 5σ/√N of 0, σ=1 for standard normal
        double bound = 5.0 / Math.sqrt(N);
        assertThat(mean).as("Sample mean of %d Gaussians", N)
                .isBetween(-bound, bound);
    }

    @Test
    void nextGaussians_varianceWithinBound() {
        RandomBatch rb = RandomBatch.of(SEED);
        double[] buf = new double[N];
        rb.nextGaussians(buf, 0, N);

        double mean = mean(buf, N);
        double variance = variance(buf, N, mean);
        // R4.8: variance within 10*√(2/N) of 1
        double bound = 10.0 * Math.sqrt(2.0 / N);
        assertThat(variance).as("Sample variance of %d Gaussians", N)
                .isBetween(1.0 - bound, 1.0 + bound);
    }

    // ---- Uniform statistical correctness -----------------------------

    @Test
    void nextUniforms_meanAndRange() {
        RandomBatch rb = RandomBatch.of(SEED);
        double[] buf = new double[N];
        rb.nextUniforms(buf, 0, N);

        double sum = 0;
        for (int i = 0; i < N; i++) {
            assertThat(buf[i]).as("Uniform draw %d", i)
                    .isGreaterThanOrEqualTo(0.0)
                    .isLessThan(1.0);
            sum += buf[i];
        }
        double mean = sum / N;
        // Mean of U[0,1) should be ≈ 0.5; use 5σ/√N where σ = 1/√12
        double sigma = 1.0 / Math.sqrt(12.0);
        double bound = 5.0 * sigma / Math.sqrt(N);
        assertThat(mean).as("Sample mean of %d uniforms", N)
                .isBetween(0.5 - bound, 0.5 + bound);
    }

    // ---- Exponential statistical correctness -------------------------

    @Test
    void nextExponentials_meanMatchesRate() {
        double lambda = 2.5;
        RandomBatch rb = RandomBatch.of(SEED);
        double[] buf = new double[N];
        rb.nextExponentials(buf, 0, N, lambda);

        double expectedMean = 1.0 / lambda;
        double mean = mean(buf, N);
        // σ of Exp(λ) = 1/λ, so bound = 5σ/√N = 5/(λ√N)
        double bound = 5.0 / (lambda * Math.sqrt(N));
        assertThat(mean).as("Sample mean of %d Exp(%.1f) draws", N, lambda)
                .isBetween(expectedMean - bound, expectedMean + bound);

        // All exponential draws must be positive
        for (int i = 0; i < N; i++) {
            assertThat(buf[i]).as("Exponential draw %d", i).isPositive();
        }
    }

    // ---- Determinism -------------------------------------------------

    @Test
    void determinism_sameSeedSameOutput() {
        int n = 1024;
        RandomBatch rb1 = RandomBatch.of(SEED);
        RandomBatch rb2 = RandomBatch.of(SEED);

        double[] gauss1 = new double[n];
        double[] gauss2 = new double[n];
        rb1.nextGaussians(gauss1, 0, n);
        rb2.nextGaussians(gauss2, 0, n);
        assertThat(gauss1).as("Gaussian streams from same seed")
                .containsExactly(gauss2);

        // Fresh instances for uniform check
        rb1 = RandomBatch.of(SEED);
        rb2 = RandomBatch.of(SEED);
        double[] uni1 = new double[n];
        double[] uni2 = new double[n];
        rb1.nextUniforms(uni1, 0, n);
        rb2.nextUniforms(uni2, 0, n);
        assertThat(uni1).as("Uniform streams from same seed")
                .containsExactly(uni2);
    }

    // ---- Scalar patch coverage (no -Infinity / NaN) ------------------

    @Test
    void scalarPatch_noNegativeInfinity() {
        // Generate a large batch and verify no -Infinity or NaN appears.
        // The scalar patch guards against log(0) in Box–Muller.
        RandomBatch rb = RandomBatch.of(SEED);
        double[] buf = new double[N];
        rb.nextGaussians(buf, 0, N);

        for (int i = 0; i < N; i++) {
            assertThat(buf[i]).as("Gaussian draw %d", i)
                    .isNotEqualTo(Double.NEGATIVE_INFINITY)
                    .isNotEqualTo(Double.POSITIVE_INFINITY);
            assertThat(Double.isNaN(buf[i])).as("Gaussian draw %d is NaN", i).isFalse();
        }
    }

    // ---- split determinism -------------------------------------------

    @Test
    void split_deterministic() {
        RandomBatch master1 = RandomBatch.of(SEED);
        RandomBatch master2 = RandomBatch.of(SEED);

        RandomBatch child1 = master1.split(99L);
        RandomBatch child2 = master2.split(99L);

        int n = 512;
        double[] out1 = new double[n];
        double[] out2 = new double[n];
        child1.nextGaussians(out1, 0, n);
        child2.nextGaussians(out2, 0, n);

        assertThat(out1).as("split(99) Gaussian streams from same master seed")
                .containsExactly(out2);
    }

    // ---- asRandomGenerator contract ----------------------------------

    @Test
    void asRandomGenerator_returnsNonNull() {
        RandomBatch rb = RandomBatch.of(SEED);
        RandomGenerator rg = rb.asRandomGenerator();
        assertThat(rg).isNotNull();
        // Basic sanity: can produce a double
        double d = rg.nextDouble();
        assertThat(d).isBetween(0.0, 1.0);
    }

    // ---- Helpers -----------------------------------------------------

    private static double mean(double[] arr, int n) {
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += arr[i];
        }
        return sum / n;
    }

    private static double variance(double[] arr, int n, double mean) {
        double sumSq = 0;
        for (int i = 0; i < n; i++) {
            double d = arr[i] - mean;
            sumSq += d * d;
        }
        return sumSq / (n - 1);
    }
}
