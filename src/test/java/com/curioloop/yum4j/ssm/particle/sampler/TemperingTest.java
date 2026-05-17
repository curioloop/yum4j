package com.curioloop.yum4j.ssm.particle.sampler;

import com.curioloop.yum4j.ssm.particle.dist.MvNormalDistribution;
import com.curioloop.yum4j.ssm.particle.engine.*;
import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.sampler.moves.AdaptiveMCMCSequence;
import com.curioloop.yum4j.ssm.particle.sampler.moves.ArrayRandomWalk;
import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequence;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import org.junit.jupiter.api.Test;

import java.util.Collections;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration test for the Tempering sampler using the new Engine API.
 * Verifies that the sampler produces finite logLt, positive ESS, and
 * that path-sampling log Z is finite.
 */
class TemperingTest {

    @Test
    void tempering_producesFiniteResults() {
        double sigmaPrior = 5.0;
        double sigmaObs = 1.0;
        int T = 10;
        int N = 50;
        int lenChain = 3;
        double essRmin = 0.5;

        // Generate synthetic data
        java.util.Random rng = new java.util.Random(42L);
        double[] dataArr = new double[T];
        for (int t = 0; t < T; t++) dataArr[t] = 1.0 + rng.nextGaussian();

        // Fixed exponent schedule
        int nBridges = 8;
        double[] exponents = new double[nBridges + 1];
        for (int i = 0; i <= nBridges; i++) exponents[i] = (double) i / nBridges;

        // Build static model
        StaticModel model = new StaticModel(
            MvNormalDistribution.of(new double[]{0.0}, new double[]{sigmaPrior * sigmaPrior}),
            T
        ) {
            @Override
            public void logpyt(double[] theta, int thetaOff, int N, int t,
                               double[] out, int outOff) {
                double yt = dataArr[t];
                double inv = 1.0 / sigmaObs;
                double logN = -0.5 * Math.log(2.0 * Math.PI) - Math.log(sigmaObs);
                for (int n = 0; n < N; n++) {
                    double z = (yt - theta[thetaOff + n]) * inv;
                    out[outOff + n] = logN - 0.5 * z * z;
                }
            }
        };

        // Build Tempering FK model
        MCMCSequence move = new AdaptiveMCMCSequence(new ArrayRandomWalk(), lenChain);
        Tempering fk = new Tempering(model, move, N, lenChain, essRmin, exponents);

        int horizon = exponents.length - 1;
        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(123L);
        RunState rs = RunState.allocate(horizon, essRmin, Scheme.SYSTEMATIC, Collections.emptyList(), 123L);

        // Sampler-style FK ignores observations; null-filled placeholder of
        // length `horizon` supplies the horizon to the engine.
        java.util.List<ThetaParticles> placeholder = Collections.nCopies(horizon, null);
        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, placeholder);

        // Verify results
        assertThat(rs.stepCount).isEqualTo(horizon);
        for (int t = 0; t < horizon; t++) {
            assertThat(rs.logLtSeries[t]).as("logLt[%d]", t).isFinite();
            assertThat(rs.essSeries[t]).as("ESS[%d]", t).isPositive();
        }

        // Path-sampling log Z should be finite
        assertThat(fk.pathSamplingLogZ).as("pathSamplingLogZ").isFinite();

        // At least one resampling should have occurred
        boolean anyResampled = false;
        for (int t = 0; t < horizon; t++) {
            if (rs.resampledFlags[t]) { anyResampled = true; break; }
        }
        assertThat(anyResampled).as("at least one resampling").isTrue();
    }
}
