package com.curioloop.yum4j.ssm.particle.sampler;

import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequence;
import com.curioloop.yum4j.ssm.particle.sampler.moves.AdaptiveMCMCSequence;
import com.curioloop.yum4j.ssm.particle.sampler.moves.ArrayRandomWalk;

import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.model.builtin.LinearGauss;
import com.curioloop.yum4j.stats.NormalDistribution;
import com.curioloop.yum4j.ssm.particle.dist.IndependentProductDistribution;
import com.curioloop.yum4j.ssm.particle.dist.MultivariateDistribution;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration test for {@link SMCSquared} on a univariate linear Gaussian
 * state-space model with known parameters.
 *
 * <p>Ported from the old SMCSquaredTest to use the new ParticleSSM-based API.
 */
class SMCSquaredTest {

    private static RandomGenerator rng(long seed) {
        return RandomGeneratorFactory.of("L64X128MixRandom").create(seed);
    }

    private static List<Double> generateData(double rho, double sigmaX, double sigmaY,
                                             int T, long seed) {
        RandomGenerator g = rng(seed);
        double x = sigmaX * g.nextGaussian();
        List<Double> data = new ArrayList<>(T);
        for (int t = 0; t < T; t++) {
            double y = x + sigmaY * g.nextGaussian();
            data.add(y);
            x = rho * x + sigmaX * g.nextGaussian();
        }
        return data;
    }

    @Test
    void linearGauss_logLikelihoodIsReasonable() {
        double rho = 0.9;
        double sigmaX = 1.0;
        double sigmaY = 0.2;
        int T = 20;

        List<Double> data = generateData(rho, sigmaX, sigmaY, T, 42L);

        MultivariateDistribution prior = new IndependentProductDistribution(
            new NormalDistribution(0.9, 0.2)
        );

        java.util.function.Function<double[], ParticleSSM<Double>> ssmFactory = theta -> {
            double r = theta[0];
            r = Math.max(-0.999, Math.min(0.999, r));
            return new LinearGauss(r, sigmaX, sigmaY, Double.NaN);
        };

        MCMCSequence move = new AdaptiveMCMCSequence(new ArrayRandomWalk(), 3);

        SMCSquared.Config config = new SMCSquared.Config(
            /* N */ 30,
            /* Nx */ 50,
            /* dimTheta */ 1,
            /* dimX */ 1,
            /* essRmin */ 0.5,
            /* essRminInner */ 0.5,
            move,
            /* seed */ 123L
        );

        SMCSquared.Result result = SMCSquared.run(ssmFactory, prior, data, config);

        assertThat(result.T()).isEqualTo(T);
        assertThat(result.logLt()).hasSize(T);
        assertThat(result.ESS()).hasSize(T);
        assertThat(result.rsFlag()).hasSize(T);

        for (int t = 0; t < T; t++) {
            assertThat(result.logLt()[t])
                .as("logLt[%d] should be finite", t)
                .isFinite();
        }

        double finalLogLt = result.logLt()[T - 1];
        assertThat(finalLogLt).isLessThan(0.0);
        assertThat(finalLogLt).isGreaterThan(-100.0);

        for (int t = 0; t < T; t++) {
            assertThat(result.ESS()[t])
                .as("ESS[%d] should be positive", t)
                .isGreaterThan(0.0);
        }

        assertThat(result.finalTheta()).isNotNull();
        assertThat(result.finalTheta().N).isEqualTo(30);
        assertThat(result.finalTheta().dim).isEqualTo(1);
    }

    @Test
    void linearGauss_resamplingOccurs() {
        double rho = 0.9;
        double sigmaX = 1.0;
        double sigmaY = 0.5;
        int T = 30;

        List<Double> data = generateData(rho, sigmaX, sigmaY, T, 99L);

        MultivariateDistribution prior = new IndependentProductDistribution(
            new NormalDistribution(0.9, 0.3)
        );

        java.util.function.Function<double[], ParticleSSM<Double>> ssmFactory = theta -> {
            double r = Math.max(-0.999, Math.min(0.999, theta[0]));
            return new LinearGauss(r, sigmaX, sigmaY, Double.NaN);
        };

        MCMCSequence move = new AdaptiveMCMCSequence(new ArrayRandomWalk(), 2);

        SMCSquared.Config config = new SMCSquared.Config(
            /* N */ 20,
            /* Nx */ 30,
            /* dimTheta */ 1,
            /* dimX */ 1,
            /* essRmin */ 0.5,
            /* essRminInner */ 0.5,
            move,
            /* seed */ 77L
        );

        SMCSquared.Result result = SMCSquared.run(ssmFactory, prior, data, config);

        boolean anyResampled = false;
        for (boolean rs : result.rsFlag()) {
            if (rs) { anyResampled = true; break; }
        }
        assertThat(anyResampled)
            .as("At least one θ-resampling should occur with N=20 over T=30 steps")
            .isTrue();
    }
}
