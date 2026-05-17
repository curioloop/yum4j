package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.dist.IndependentProductDistribution;
import com.curioloop.yum4j.ssm.particle.dist.MultivariateDistribution;
import com.curioloop.yum4j.ssm.particle.mcmc.PMMHResult;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.model.builtin.LinearGauss;
import com.curioloop.yum4j.ssm.particle.sampler.SMC2Result;
import com.curioloop.yum4j.ssm.particle.sampler.moves.AdaptiveMCMCSequence;
import com.curioloop.yum4j.ssm.particle.sampler.moves.ArrayRandomWalk;
import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequence;
import com.curioloop.yum4j.stats.NormalDistribution;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * <ul>
 *   <li>{@link Particle#filter(ParticleSSM)} produces a stable
 *       {@link FilterResult} when invoked with and without a reusable
 *       {@link ParticleWorkspace};</li>
 *   <li>{@link Particle#online(ParticleSSM)} returns a stateful
 *       {@link ParticleOnline.Run} that processes observations
 *       incrementally and matches the batch-filter outputs after
 *       all observations have been fed in;</li>
 *   <li>{@link ParticleWorkspace} correctly reuses scratch across
 *       repeated {@code .run(workspace)} invocations without affecting
 *       results;</li>
 *   <li>{@link Particle#smooth(FilterResult)} consumes a completed
 *       filter result and produces a trajectory bundle;</li>
 *   <li>The model-path smoother
 *       ({@link Particle#smooth(ParticleSSM)})
 *       runs the forward filter with an automatically-chosen history
 *       mode;</li>
 *   <li>{@link Particle#sample(SamplingMethod)} runs the underlying
 *       PMCMC algorithm (PMMH / Particle Gibbs / CSMC) or the SMC
 *       sampler implementations (IBIS, SMC², tempering, nested) and
 *       returns their typed result records;</li>
 *   <li>The facade reports clear errors for missing required configuration.</li>
 * </ul>
 */
class ParticleApiUsabilityTest {

    private static final LinearGauss MODEL = new LinearGauss(0.9, 1.0, 0.5, 1.0);
    private static final int T = 50;
    private static final int N = 1024;
    private static final long DATA_SEED = 4242L;
    private static final long FILTER_SEED = 99L;

    private static List<Double> data() {
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(DATA_SEED);
        List<Double> out = new ArrayList<>(T);
        double x = MODEL.sigma0() * g.nextGaussian();
        out.add(x + MODEL.sigmaY() * g.nextGaussian());
        for (int t = 1; t < T; t++) {
            x = MODEL.rho() * x + MODEL.sigmaX() * g.nextGaussian();
            out.add(x + MODEL.sigmaY() * g.nextGaussian());
        }
        return out;
    }

    @Test
    void onlineFilterMatchesBatchFilter() {
        List<Double> obs = data();

        FilterResult<Double> batch = Particle.filter(MODEL)
                .particles(N)
                .observations(obs)
                .seed(FILTER_SEED)
                .run();

        try (ParticleWorkspace ws = Particle.workspace()) {
            ParticleOnline.Run<Double> online = Particle.online(MODEL)
                    .particles(N)
                    .seed(FILTER_SEED)
                    .start(ws);
            for (Double y : obs) {
                online.step(y);
            }
            FilterResult<Double> incremental = online.currentResult();

            assertThat(incremental.T()).isEqualTo(batch.T());
            assertSeriesClose(incremental.logLikelihoodSeries(),
                    batch.logLikelihoodSeries(), 1e-10);
            assertSeriesClose(incremental.essSeries(),
                    batch.essSeries(), 1e-10);
        }
    }

    @Test
    void onlineFilterAccessorsReflectStreamingState() {
        List<Double> obs = data();

        try (ParticleWorkspace ws = Particle.workspace()) {
            ParticleOnline.Run<Double> run = Particle.online(MODEL)
                    .particles(N)
                    .seed(FILTER_SEED)
                    .start(ws);

            assertThatThrownBy(run::logLikelihood)
                    .isInstanceOf(IllegalStateException.class);

            for (int t = 0; t < obs.size(); t++) {
                run.step(obs.get(t));
                assertThat(run.stepCount()).isEqualTo(t + 1);
                assertThat(Double.isFinite(run.logLikelihood())).isTrue();
                assertThat(run.ess()).isPositive();
            }
        }
    }

    @Test
    void onlineFilterResetReusesBuffers() {
        List<Double> obs = data();

        try (ParticleWorkspace ws = Particle.workspace()) {
            ParticleOnline.Run<Double> run = Particle.online(MODEL)
                    .particles(N)
                    .seed(FILTER_SEED)
                    .start(ws);
            for (Double y : obs) {
                run.step(y);
            }
            double firstLogLt = run.logLikelihood();

            run.reset();
            for (Double y : obs) {
                run.step(y);
            }
            double secondLogLt = run.logLikelihood();

            // Same seed, same observations ⇒ same trajectory; tolerance
            // matches R9.2 noise floor.
            double scale = Math.max(Math.abs(firstLogLt), Math.abs(secondLogLt));
            double tol = scale > 0 ? 1e-10 * scale : 1e-10;
            assertThat(Math.abs(firstLogLt - secondLogLt))
                    .as("reset(seed) should re-run deterministically")
                    .isLessThanOrEqualTo(tol);
        }
    }

    private static void assertSeriesClose(double[] actual, double[] expected, double rtol) {
        assertThat(actual).hasSameSizeAs(expected);
        for (int i = 0; i < actual.length; i++) {
            double a = actual[i];
            double e = expected[i];
            double scale = Math.max(Math.abs(a), Math.abs(e));
            double tol = scale > 0 ? rtol * scale : rtol;
            assertThat(Math.abs(a - e))
                    .as("element %d: actual=%s expected=%s rtol=%s", i, a, e, rtol)
                    .isLessThanOrEqualTo(tol);
        }
    }

    @Test
    void workspaceIsReusedAcrossRunsWithoutAffectingResults() {
        List<Double> obs = data();

        FilterResult<Double> baseline = Particle.filter(MODEL)
                .particles(N)
                .observations(obs)
                .seed(FILTER_SEED)
                .run();

        try (ParticleWorkspace ws = Particle.workspace()) {
            FilterResult<Double> first = Particle.filter(MODEL)
                    .particles(N)
                    .observations(obs)
                    .seed(FILTER_SEED)
                    .run(ws);
            FilterResult<Double> second = Particle.filter(MODEL)
                    .particles(N)
                    .observations(obs)
                    .seed(FILTER_SEED)
                    .run(ws);

            assertSeriesClose(first.logLikelihoodSeries(),
                    baseline.logLikelihoodSeries(), 1e-10);
            assertSeriesClose(second.logLikelihoodSeries(),
                    baseline.logLikelihoodSeries(), 1e-10);
        }
    }

        @Test
        void borrowedFilterResultAliasesWorkspaceUntilReuse() {
                List<Double> obs = data();

                try (ParticleWorkspace ws = Particle.workspace()) {
                        BorrowedFilterResult<Double> first = Particle.filter(MODEL)
                                        .particles(N)
                                        .observations(obs)
                                        .seed(FILTER_SEED)
                                        .runBorrowedUnsafe(ws);
                        double firstLastLogLikelihood = first.logLikelihoodAt(first.T() - 1);

                        BorrowedFilterResult<Double> second = Particle.filter(MODEL)
                                        .particles(N)
                                        .observations(obs)
                                        .seed(FILTER_SEED + 1)
                                        .runBorrowedUnsafe(ws);

                        assertThat(first.logLikelihoodSeries).isSameAs(second.logLikelihoodSeries);
                        assertThat(first.essSeries).isSameAs(second.essSeries);
                        assertThat(first.resampledFlags).isSameAs(second.resampledFlags);
                        assertThat(first.finalParticles).isSameAs(second.finalParticles);
                        assertThat(first.finalLogWeights).isSameAs(second.finalLogWeights);
                        assertThat(first.logLikelihoodAt(first.T() - 1)).isNotEqualTo(firstLastLogLikelihood);
                }
        }

        @Test
        void publicFilterResultKeepsSnapshotAfterWorkspaceReuse() {
                List<Double> obs = data();

                try (ParticleWorkspace ws = Particle.workspace()) {
                        FilterResult<Double> first = Particle.filter(MODEL)
                                        .particles(N)
                                        .observations(obs)
                                        .seed(FILTER_SEED)
                                        .run(ws);
                        double[] firstLogLikelihood = first.logLikelihoodSeries();

                        Particle.filter(MODEL)
                                        .particles(N)
                                        .observations(obs)
                                        .seed(FILTER_SEED + 1)
                                        .run(ws);

                        assertSeriesClose(first.logLikelihoodSeries(), firstLogLikelihood, 0.0);
                }
        }

    @Test
    void closeReleasesRetainedScratch() {
        List<Double> obs = data();
        ParticleWorkspace ws = Particle.workspace();
        Particle.filter(MODEL).particles(N).observations(obs).seed(FILTER_SEED).run(ws);
        assertThat(ws.filterSlot).isNotNull();
        assertThat(ws.filterSlot.workspace).isNotNull();
        assertThat(ws.filterSlot.runState).isNotNull();
        ws.close();
        assertThat(ws.filterSlot).isNull();
        assertThat(ws.smoothingSlot).isNull();
        assertThat(ws.mcmcSlot).isNull();
        assertThat(ws.samplerSlot).isNull();
    }

    @Test
    void slotsAreLazilyAllocated() {
        // A fresh workspace allocates no slot; only the inference family
        // that runs creates its slot. This covers the case where a
        // long-lived workspace is handed to e.g. only filter calls and
        // never pays for sampler / mcmc / smoothing scratch.
        try (ParticleWorkspace ws = Particle.workspace()) {
            assertThat(ws.filterSlot).isNull();
            assertThat(ws.smoothingSlot).isNull();
            assertThat(ws.mcmcSlot).isNull();
            assertThat(ws.samplerSlot).isNull();

            Particle.filter(MODEL)
                    .particles(N)
                    .observations(data())
                    .seed(FILTER_SEED)
                    .run(ws);

            assertThat(ws.filterSlot).isNotNull();
            assertThat(ws.smoothingSlot).isNull();
            assertThat(ws.mcmcSlot).isNull();
            assertThat(ws.samplerSlot).isNull();
        }
    }

    @Test
    void sampleFacadeReusesWorkspaceAcrossRunsWithoutAffectingResults() {
        List<Double> obs = data().subList(0, 8);
        MultivariateDistribution prior = new IndependentProductDistribution(
                new NormalDistribution(0.9, 0.2));
        Function<double[], ParticleSSM<Double>> modelFactory = theta ->
                new LinearGauss(clampRho(theta[0]), 1.0, 0.5, 1.0);

        PMMHResult baseline = Particle.sample(SamplingMethod.PMMH)
                .prior(prior)
                .model(modelFactory, obs)
                .particles(24)
                .iterations(5)
                .seed(17L)
                .run();

        try (ParticleWorkspace ws = Particle.workspace()) {
            PMMHResult first = Particle.sample(SamplingMethod.PMMH)
                    .prior(prior)
                    .model(modelFactory, obs)
                    .particles(24)
                    .iterations(5)
                    .seed(17L)
                    .run(ws);

            // Verify the mcmc-slot workspace was acquired and cached.
            assertThat(ws.mcmcSlot.workspace).isNotNull();
            assertThat(ws.mcmcSlot.runState).isNotNull();
            var cachedWs = ws.mcmcSlot.workspace;
            var cachedRs = ws.mcmcSlot.runState;

            PMMHResult second = Particle.sample(SamplingMethod.PMMH)
                    .prior(prior)
                    .model(modelFactory, obs)
                    .particles(24)
                    .iterations(5)
                    .seed(17L)
                    .run(ws);

            // Same shape ⇒ same buffer instance reused.
            assertThat(ws.mcmcSlot.workspace).isSameAs(cachedWs);
            assertThat(ws.mcmcSlot.runState).isSameAs(cachedRs);

            assertThat(first.chain()).containsExactly(baseline.chain());
            assertThat(second.chain()).containsExactly(baseline.chain());
            assertThat(first.accRate()).isEqualTo(baseline.accRate());
            assertThat(second.accRate()).isEqualTo(baseline.accRate());
        }
    }

    @Test
    void smootherFromFilterResultProducesTrajectoryBundle() {
        List<Double> obs = data();

        try (ParticleWorkspace ws = Particle.workspace()) {
            FilterResult<Double> result = Particle.filter(MODEL)
                    .particles(N)
                    .observations(obs)
                    .fullHistory()
                    .maxResampleRate(1.0)
                    .seed(FILTER_SEED)
                    .run(ws);

            TrajectoryBundle<Double> traj = Particle.smooth(result)
                    .ffbs()
                    .trajectories(16)
                    .seed(7L)
                    .run(ws);

            assertThat(traj.T()).isEqualTo(T);
            assertThat(traj.M()).isEqualTo(16);
        }
    }

    @Test
    void smootherFromModelRunsForwardFilterAutomatically() {
        List<Double> obs = data();

        TrajectoryBundle<Double> traj = Particle.smooth(MODEL)
                .ffbs()
                .particles(N)
                .observations(obs)
                .filterSeed(FILTER_SEED)
                .trajectories(8)
                .seed(11L)
                .run();

        assertThat(traj.T()).isEqualTo(T);
        assertThat(traj.M()).isEqualTo(8);
    }

    @Test
        void sampleFacadeRunsPmmhWithExistingResultType() {
        List<Double> obs = data().subList(0, 8);
        MultivariateDistribution prior = new IndependentProductDistribution(
                new NormalDistribution(0.9, 0.2));
        Function<double[], ParticleSSM<Double>> modelFactory = theta ->
                new LinearGauss(clampRho(theta[0]), 1.0, 0.5, 1.0);

        PMMHResult result = Particle.sample(SamplingMethod.PMMH)
                .prior(prior)
                .model(modelFactory, obs)
                .particles(24)
                .iterations(5)
                .seed(17L)
                .run();

        assertThat(result.dim()).isEqualTo(1);
        assertThat(result.niter()).isEqualTo(5);
        assertThat(result.chain()).hasSize(5);
        assertThat(result.logPost()).hasSize(5);
    }

    @Test
        void sampleFacadeRunsSmc2WithExistingResultType() {
        List<Double> obs = data().subList(0, 10);
        MultivariateDistribution prior = new IndependentProductDistribution(
                new NormalDistribution(0.9, 0.25));
        Function<double[], ParticleSSM<Double>> modelFactory = theta ->
                new LinearGauss(clampRho(theta[0]), 1.0, 0.5, 1.0);
        MCMCSequence move = new AdaptiveMCMCSequence(new ArrayRandomWalk(), 2);

        SMC2Result result = Particle.sample(SamplingMethod.SMC2)
                .prior(prior)
                .model(modelFactory, obs)
                .move(move)
                .particles(20)
                .innerParticles(30)
                .seed(23L)
                .run();

        assertThat(result.dimTheta()).isEqualTo(1);
        assertThat(result.N()).isEqualTo(20);
        assertThat(result.T()).isEqualTo(10);
        assertThat(result.logLt()).hasSize(10);
        assertThat(result.ess()).hasSize(10);
    }

    private static double clampRho(double rho) {
        return Math.max(-0.999, Math.min(0.999, rho));
    }

    @Test
    void smootherCachesSmoothingWorkspaceAcrossCalls() {
        List<Double> obs = data();

        try (ParticleWorkspace ws = Particle.workspace()) {
            FilterResult<Double> result = Particle.filter(MODEL)
                    .particles(N)
                    .observations(obs)
                    .fullHistory()
                    .maxResampleRate(1.0)
                    .seed(FILTER_SEED)
                    .run(ws);

            Particle.smooth(result).ffbs().trajectories(16).seed(7L).run(ws);
            var swAfterFirst = ws.smoothingSlot.smoothingWorkspace;
            assertThat(swAfterFirst).isNotNull();

            Particle.smooth(result).ffbs().trajectories(16).seed(7L).run(ws);
            // Same shape (M, N, dim, T) ⇒ same instance reused.
            assertThat(ws.smoothingSlot.smoothingWorkspace).isSameAs(swAfterFirst);
        }
    }

    @Test
    void filterFacadeReportsMissingConfiguration() {
        assertThatThrownBy(() -> Particle.filter(MODEL).run())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("particles");

        assertThatThrownBy(() -> Particle.filter(MODEL).particles(N).run())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("observations");

        assertThatThrownBy(() -> Particle.filter(null))
                .isInstanceOf(NullPointerException.class);
    }

    @Test
    void smootherRequiresHistoryMode() {
        List<Double> obs = data();

        FilterResult<Double> noHistory = Particle.filter(MODEL)
                .particles(N)
                .observations(obs)
                .history(HistoryMode.NONE)
                .seed(FILTER_SEED)
                .run();

        assertThatThrownBy(() -> Particle.smooth(noHistory).ffbs().trajectories(8).run())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("history");
    }
}
