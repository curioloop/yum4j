package com.curioloop.yum4j.ssm.particle.kernel;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Unit tests for {@link ParticleOps} SIMD kernel primitives.
 *
 * <p><b>SIMD-vs-scalar tolerance.</b> Tests that compare the SIMD
 * kernels in {@link ParticleOps} against a hand-rolled scalar oracle
 * use a {@code 1e-12} relative tolerance. The SIMD kernels fuse
 * multiply-add pairs into single-rounding {@code FMA} ops and may
 * reorder lane-wise reductions, both of which produce 1-10 ULP drift
 * vs the scalar oracle's step-by-step rounding. The {@code 1e-12}
 * band matches the documented IEEE 754 / JIT noise floor used by
 * {@code particle-v2-perf} R16.3 and {@code particle-v2-mem} R9.2 for
 * the same family of comparisons; tightening to e.g. {@code 1e-14}
 * has been empirically observed to flake on Apple M-series + JDK 27 EA.
 */
class ParticleOpsTest {

    private static final double HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

    // ---------------------------------------------------------------
    // gaussianLogPdfInto — scalar mean
    // ---------------------------------------------------------------

    @Test
    void gaussianLogPdfInto_scalarMean_matchesReference() {
        int N = 100;
        double mu = 2.5;
        double sigma = 1.3;

        Random rnd = new Random(42);
        double[] x = new double[N];
        for (int i = 0; i < N; i++) {
            x[i] = rnd.nextGaussian() * 5.0 + mu;
        }

        double[] out = new double[N];
        ParticleOps.gaussianLogPdfInto(out, 0, N, x, 0, mu, sigma);

        for (int i = 0; i < N; i++) {
            double expected = -0.5 * Math.pow((x[i] - mu) / sigma, 2)
                              - Math.log(sigma) - HALF_LOG_2PI;
            assertRelativeOrAbsolute(out[i], expected, 1e-12);
        }
    }

    // ---------------------------------------------------------------
    // gaussianLogPdfInto — vector mean
    // ---------------------------------------------------------------

    @Test
    void gaussianLogPdfInto_vectorMean_matchesReference() {
        int N = 100;
        double sigma = 0.7;

        Random rnd = new Random(123);
        double[] x = new double[N];
        double[] mu = new double[N];
        for (int i = 0; i < N; i++) {
            mu[i] = rnd.nextGaussian() * 2.0;
            x[i] = mu[i] + rnd.nextGaussian() * sigma;
        }

        double[] out = new double[N];
        ParticleOps.gaussianLogPdfInto(out, 0, N, x, 0, mu, 0, sigma);

        for (int i = 0; i < N; i++) {
            double expected = -0.5 * Math.pow((x[i] - mu[i]) / sigma, 2)
                              - Math.log(sigma) - HALF_LOG_2PI;
            assertRelativeOrAbsolute(out[i], expected, 1e-12);
        }
    }

    // ---------------------------------------------------------------
    // arOneInto — matches scalar loop
    // ---------------------------------------------------------------

    @Test
    void arOneInto_matchesScalarLoop() {
        int N = 256;
        double c = 0.5;
        double rho = 0.9;
        double sigma = 0.3;
        long seed = 7777L;

        Random rnd = new Random(99);
        double[] Xprev = new double[N];
        for (int i = 0; i < N; i++) {
            Xprev[i] = rnd.nextGaussian() * 2.0;
        }

        // Generate the expected output using a separate RNG with the same seed
        // to capture the exact Gaussians that arOneInto will use.
        RandomBatch rng1 = RandomBatch.of(seed);
        double[] gaussians = new double[N];
        rng1.nextGaussians(gaussians, 0, N);

        double[] expected = new double[N];
        for (int n = 0; n < N; n++) {
            expected[n] = c + rho * Xprev[n] + sigma * gaussians[n];
        }

        // Now run arOneInto with a fresh RNG of the same seed
        RandomBatch rng2 = RandomBatch.of(seed);
        double[] out = new double[N];
        ParticleOps.arOneInto(out, 0, N, c, rho, Xprev, 0, sigma, rng2);

        for (int n = 0; n < N; n++) {
            assertRelativeOrAbsolute(out[n], expected[n], 1e-12);
        }
    }

    // ---------------------------------------------------------------
    // gaussianInto — scalar mean, moments correct
    // ---------------------------------------------------------------

    @Test
    void gaussianInto_scalarMean_momentsCorrect() {
        int N = 100_000;
        double mu = 3.0;
        double sigma = 2.0;
        long seed = 12345L;

        RandomBatch rng = RandomBatch.of(seed);
        double[] out = new double[N];
        ParticleOps.gaussianInto(out, 0, N, mu, sigma, rng);

        double sum = 0.0;
        for (double v : out) sum += v;
        double mean = sum / N;

        double sumSq = 0.0;
        for (double v : out) sumSq += (v - mean) * (v - mean);
        double variance = sumSq / (N - 1);

        // Mean within 5*sigma/sqrt(N) of mu
        double meanTol = 5.0 * sigma / Math.sqrt(N);
        assertThat(mean).isCloseTo(mu, within(meanTol));

        // Variance within 10*sqrt(2/N) * sigma^2 of sigma^2
        double varTol = 10.0 * Math.sqrt(2.0 / N) * sigma * sigma;
        assertThat(variance).isCloseTo(sigma * sigma, within(varTol));
    }

    // ---------------------------------------------------------------
    // gaussianInto — vector mean, per-particle mean correct
    // ---------------------------------------------------------------

    @Test
    void gaussianInto_vectorMean_perParticleMeanCorrect() {
        // Use multiple draws per particle to verify centering
        int particles = 50;
        int drawsPerParticle = 10_000;
        double sigma = 1.5;
        long baseSeed = 54321L;

        Random rnd = new Random(88);
        double[] means = new double[particles];
        for (int i = 0; i < particles; i++) {
            means[i] = rnd.nextGaussian() * 5.0;
        }

        // For each particle, draw many samples and check the empirical mean
        for (int p = 0; p < particles; p++) {
            double[] muArr = new double[drawsPerParticle];
            java.util.Arrays.fill(muArr, means[p]);

            RandomBatch rng = RandomBatch.of(baseSeed + p);
            double[] out = new double[drawsPerParticle];
            ParticleOps.gaussianInto(out, 0, drawsPerParticle, muArr, 0, sigma, rng);

            double sum = 0.0;
            for (double v : out) sum += v;
            double empiricalMean = sum / drawsPerParticle;

            double tol = 5.0 * sigma / Math.sqrt(drawsPerParticle);
            assertThat(empiricalMean)
                .as("Particle %d mean should be close to %.3f", p, means[p])
                .isCloseTo(means[p], within(tol));
        }
    }

    // ---------------------------------------------------------------
    // expInto — matches Math.exp
    // ---------------------------------------------------------------

    @Test
    void expInto_matchesMathExp() {
        int N = 100;
        Random rnd = new Random(2024);
        double[] x = new double[N];
        for (int i = 0; i < N; i++) {
            // Range that avoids overflow/underflow: [-500, 500]
            x[i] = rnd.nextDouble() * 1000.0 - 500.0;
        }

        double[] out = new double[N];
        ParticleOps.expInto(out, 0, N, x, 0);

        for (int i = 0; i < N; i++) {
            double expected = Math.exp(x[i]);
            assertRelativeOrAbsolute(out[i], expected, 1e-12);
        }
    }

    // ---------------------------------------------------------------
    // logInto — matches Math.log
    // ---------------------------------------------------------------

    @Test
    void logInto_matchesMathLog() {
        int N = 100;
        Random rnd = new Random(2025);
        double[] x = new double[N];
        for (int i = 0; i < N; i++) {
            // Positive values only: (0, 1000]
            x[i] = rnd.nextDouble() * 999.0 + 1e-10;
        }

        double[] out = new double[N];
        ParticleOps.logInto(out, 0, N, x, 0);

        for (int i = 0; i < N; i++) {
            double expected = Math.log(x[i]);
            assertRelativeOrAbsolute(out[i], expected, 1e-12);
        }
    }

    // ---------------------------------------------------------------
    // allMethods_noNaNOrInfOnValidInput
    // ---------------------------------------------------------------

    @Test
    void allMethods_noNaNOrInfOnValidInput() {
        int N = 64;
        long seed = 999L;
        RandomBatch rng = RandomBatch.of(seed);

        double mu = 1.0;
        double sigma = 2.0;
        double c = 0.1;
        double rho = 0.95;

        Random rnd = new Random(111);
        double[] x = new double[N];
        double[] muArr = new double[N];
        double[] Xprev = new double[N];
        for (int i = 0; i < N; i++) {
            x[i] = rnd.nextGaussian() * 3.0;
            muArr[i] = rnd.nextGaussian();
            Xprev[i] = rnd.nextGaussian() * 2.0;
        }

        // Positive values for exp/log input
        double[] positiveX = new double[N];
        for (int i = 0; i < N; i++) {
            positiveX[i] = Math.abs(x[i]) + 0.01;
        }

        double[] out = new double[N];

        // gaussianInto (scalar mean)
        ParticleOps.gaussianInto(out, 0, N, mu, sigma, rng);
        assertNoNaNOrInf(out, N, "gaussianInto(scalar)");

        // gaussianInto (vector mean)
        ParticleOps.gaussianInto(out, 0, N, muArr, 0, sigma, rng);
        assertNoNaNOrInf(out, N, "gaussianInto(vector)");

        // arOneInto
        ParticleOps.arOneInto(out, 0, N, c, rho, Xprev, 0, sigma, rng);
        assertNoNaNOrInf(out, N, "arOneInto");

        // gaussianLogPdfInto (scalar mean)
        ParticleOps.gaussianLogPdfInto(out, 0, N, x, 0, mu, sigma);
        assertNoNaNOrInf(out, N, "gaussianLogPdfInto(scalar)");

        // gaussianLogPdfInto (vector mean)
        ParticleOps.gaussianLogPdfInto(out, 0, N, x, 0, muArr, 0, sigma);
        assertNoNaNOrInf(out, N, "gaussianLogPdfInto(vector)");

        // expInto
        ParticleOps.expInto(out, 0, N, x, 0);
        assertNoNaNOrInf(out, N, "expInto");

        // logInto (positive inputs only)
        ParticleOps.logInto(out, 0, N, positiveX, 0);
        assertNoNaNOrInf(out, N, "logInto");
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    /**
     * Asserts relative tolerance when |expected| > 1e-15,
     * otherwise falls back to absolute tolerance.
     */
    private static void assertRelativeOrAbsolute(double actual, double expected, double relTol) {
        if (Math.abs(expected) > 1e-15) {
            double relError = Math.abs((actual - expected) / expected);
            assertThat(relError)
                .as("Expected %.16e but got %.16e (relError=%.2e)", expected, actual, relError)
                .isLessThanOrEqualTo(relTol);
        } else {
            assertThat(actual)
                .as("Expected ~0 but got %.16e", actual)
                .isCloseTo(expected, within(relTol));
        }
    }

    private static void assertNoNaNOrInf(double[] arr, int N, String method) {
        for (int i = 0; i < N; i++) {
            assertThat(Double.isNaN(arr[i]))
                .as("%s: output[%d] is NaN", method, i)
                .isFalse();
            assertThat(Double.isInfinite(arr[i]))
                .as("%s: output[%d] is Infinite", method, i)
                .isFalse();
        }
    }
}
