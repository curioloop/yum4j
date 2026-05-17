package com.curioloop.yum4j.ssm.particle.dist;

import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

/**
 * Mathematical correctness checks for {@link MixtureDistribution}.
 *
 * <p>Verifies the log-sum-exp definition, the degenerate single-
 * component case, and batch/scalar consistency. No external reference
 * fixtures.
 */
class MixtureCorrectnessTest {

    private static final double TOL = 1e-12;

    /**
     * {@code logPdf(x) = logSumExp_k (log w_k + comp_k.logPdf(x))} by
     * definition. This re-derives the expected value from the same
     * components and weights, so it traps any algebraic error in
     * Mixture's column-wise logsumexp implementation (overflow guard,
     * weight normalisation, etc.).
     */
    @Test
    void logPdfMatchesLogSumExpDefinition() {
        MvNormalDistribution comp0 = MvNormalDistribution.of(
            new double[]{-1.0, 0.0},
            new double[]{1.0, 0.0, 0.0, 1.0}
        );
        MvNormalDistribution comp1 = MvNormalDistribution.of(
            new double[]{1.0, 1.0},
            new double[]{2.0, 0.5, 0.5, 1.5}
        );
        // Unnormalised weights to exercise the normaliser path.
        double[] weights = {1.5, 0.5};
        MixtureDistribution mix = new MixtureDistribution(weights, comp0, comp1);

        // Recompute normalised log weights for the oracle.
        double sum = weights[0] + weights[1];
        double logW0 = Math.log(weights[0] / sum);
        double logW1 = Math.log(weights[1] / sum);

        int d = 2;
        int N = 16;
        double[] x = new double[d * N];
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(20251207L);
        comp0.sample(g, N, x, 0, null);

        double[] mixOut = new double[N];
        mix.logPdfBatch(x, 0, N, mixOut, 0, null);

        double[] log0 = new double[N];
        double[] log1 = new double[N];
        comp0.logPdfBatch(x, 0, N, log0, 0, null);
        comp1.logPdfBatch(x, 0, N, log1, 0, null);
        for (int n = 0; n < N; n++) {
            double a = log0[n] + logW0;
            double b = log1[n] + logW1;
            double max = Math.max(a, b);
            double expected = max + Math.log(Math.exp(a - max) + Math.exp(b - max));
            assertThat(mixOut[n])
                .as("mixture logPdf at particle %d", n)
                .isCloseTo(expected, offset(TOL));
        }
    }

    /**
     * A single-component mixture (weight 1) reduces to that component's
     * own logPdf. Catches any spurious extra log-weight or negative
     * sign in the normaliser.
     */
    @Test
    void singleComponentEqualsComponentLogPdf() {
        MvNormalDistribution comp = MvNormalDistribution.of(
            new double[]{0.5, -0.5, 1.0},
            new double[]{
                1.5, 0.2, 0.0,
                0.2, 1.0, 0.1,
                0.0, 0.1, 0.8
            }
        );
        MixtureDistribution mix = new MixtureDistribution(new double[]{3.7}, comp);

        int d = 3;
        int N = 8;
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(20251208L);
        double[] x = new double[d * N];
        comp.sample(g, N, x, 0, null);

        double[] mixOut = new double[N];
        double[] compOut = new double[N];
        mix.logPdfBatch(x, 0, N, mixOut, 0, null);
        comp.logPdfBatch(x, 0, N, compOut, 0, null);

        for (int n = 0; n < N; n++) {
            assertThat(mixOut[n])
                .as("mixture vs component at particle %d", n)
                .isCloseTo(compOut[n], offset(TOL));
        }
    }

    /**
     * Batch with caller scratch and batch without scratch must agree
     * bit-identically (Mixture allocates internally when scratch is
     * null; with a sufficiently large scratch the implementation reuses
     * caller-owned memory).
     */
    @Test
    void scratchSuppliedAndOmittedAgree() {
        MvNormalDistribution c0 = MvNormalDistribution.of(
            new double[]{0.0, 0.0}, new double[]{1.0, 0.0, 0.0, 1.0});
        MvNormalDistribution c1 = MvNormalDistribution.of(
            new double[]{2.0, -1.0}, new double[]{1.0, 0.3, 0.3, 1.0});
        MixtureDistribution mix = new MixtureDistribution(
            new double[]{0.4, 0.6}, c0, c1);

        int d = 2;
        int N = 12;
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(20251209L);
        double[] x = new double[d * N];
        c0.sample(g, N, x, 0, null);

        double[] outA = new double[N];
        mix.logPdfBatch(x, 0, N, outA, 0, null);

        // Scratch sized for K*N + d*N keeps the call on the no-allocation path.
        double[] scratch = new double[2 * N + d * N];
        double[] outB = new double[N];
        mix.logPdfBatch(x, 0, N, outB, 0, scratch);

        assertThat(outB).containsExactly(outA);
    }
}
