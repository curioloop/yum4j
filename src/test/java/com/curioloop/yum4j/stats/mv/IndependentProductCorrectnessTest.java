package com.curioloop.yum4j.ssm.particle.dist;

import com.curioloop.yum4j.stats.GammaDistribution;
import com.curioloop.yum4j.stats.NormalDistribution;
import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

/**
 * Mathematical correctness checks for
 * {@link IndependentProductDistribution}.
 *
 * <p>By definition, the joint logPdf of an independent product is the
 * sum of the marginals. The components used here (Normal, Gamma) have
 * upstream boost-parity coverage, so this test propagates that
 * verification to the multivariate wrapper without requiring a fresh
 * external reference.
 */
class IndependentProductCorrectnessTest {

    private static final double TOL = 1e-12;

    /**
     * For every particle in a {@code (d=2, N)} buffer, the joint
     * logPdf must equal {@code Normal.logPdf(x[0]) + Gamma.logPdf(x[1])}.
     */
    @Test
    void logPdfEqualsSumOfMarginalLogPdfs() {
        NormalDistribution normal = new NormalDistribution(0.5, 1.5);
        GammaDistribution gamma = new GammaDistribution(2.0, 1.5);
        IndependentProductDistribution product =
            new IndependentProductDistribution(normal, gamma);

        int N = 32;
        double[] x = new double[2 * N];
        // Row 0 = Normal samples, row 1 = strictly positive points for Gamma.
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(20251205L);
        for (int n = 0; n < N; n++) {
            x[n] = -2.0 + 4.0 * g.nextDouble();          // span the Normal
            x[N + n] = 0.05 + 5.0 * g.nextDouble();      // strictly > 0 for Gamma
        }

        double[] joint = new double[N];
        product.logPdfBatch(x, 0, N, joint, 0, null);

        for (int n = 0; n < N; n++) {
            double expected = normal.logPdf(x[n]) + gamma.logPdf(x[N + n]);
            assertThat(joint[n])
                .as("joint logPdf at particle %d", n)
                .isCloseTo(expected, offset(TOL));
        }
    }

    /**
     * Two consecutive {@code logPdfBatch} calls (one with caller
     * scratch, one without) must return identical answers.
     */
    @Test
    void scratchSuppliedAndOmittedAgree() {
        IndependentProductDistribution product = new IndependentProductDistribution(
            new NormalDistribution(0.0, 1.0),
            new NormalDistribution(0.0, 2.0),
            new NormalDistribution(0.0, 0.5)
        );

        int N = 16;
        double[] x = new double[3 * N];
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(20251206L);
        for (int i = 0; i < x.length; i++) x[i] = g.nextGaussian();

        double[] outNoScratch = new double[N];
        product.logPdfBatch(x, 0, N, outNoScratch, 0, null);

        double[] outWithScratch = new double[N];
        double[] scratch = new double[N];
        product.logPdfBatch(x, 0, N, outWithScratch, 0, scratch);

        assertThat(outWithScratch).containsExactly(outNoScratch);
    }
}
