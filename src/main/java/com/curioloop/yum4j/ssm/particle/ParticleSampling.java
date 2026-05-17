package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.dist.MultivariateDistribution;
import com.curioloop.yum4j.ssm.particle.engine.Engine;
import com.curioloop.yum4j.ssm.particle.engine.ParallelStrategy;
import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.mcmc.*;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import com.curioloop.yum4j.ssm.particle.sampler.FKSMCSampler;
import com.curioloop.yum4j.ssm.particle.sampler.IBISResult;
import com.curioloop.yum4j.ssm.particle.sampler.SMCSquared;
import com.curioloop.yum4j.ssm.particle.sampler.SMC2Result;
import com.curioloop.yum4j.ssm.particle.sampler.StaticModel;
import com.curioloop.yum4j.ssm.particle.sampler.TemperingResult;
import com.curioloop.yum4j.ssm.particle.sampler.ThetaParticles;
import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequence;
import com.curioloop.yum4j.ssm.particle.sampler.nested.NestedRwMoves;
import com.curioloop.yum4j.ssm.particle.sampler.nested.NestedSampling;
import com.curioloop.yum4j.ssm.particle.sampler.nested.NestedSamplingResult;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/** Configured particle sampling task whose result type is explicit in {@code R}. */
public interface ParticleSampling<R> extends ParticleInference<R, ParticleWorkspace> {

    SamplingMethod<?, R> method();

    private static double[] copyOrNull(double[] values) {
        return values == null ? null : Arrays.copyOf(values, values.length);
    }

    private static <T> T requireSet(T value, String name, String call) {
        if (value == null) {
            throw new IllegalStateException(
                    name + " must be set before run(); call " + call);
        }
        return value;
    }

    private static int requirePositive(int value, String name, String call) {
        if (value <= 0) {
            throw new IllegalStateException(
                    name + " must be set before run(); call " + call + " with a value > 0");
        }
        return value;
    }

    /**
     * Shared tempering driver used by {@link Tempering} (fixed schedule)
     * and {@link AdaptiveTempering} (data-driven schedule). The fork
     * point is the {@code adaptive} flag: it picks between the two
     * underlying {@link FKSMCSampler} subclasses and toggles the
     * early-termination loop in the engine.
     *
     * @param schedule        exponent schedule for fixed mode; {@code null} when {@code adaptive}
     * @param adaptive        {@code true} for adaptive tempering
     * @param adaptiveEssRmin ESS threshold for the adaptive exponent search; ignored when {@code !adaptive}
     * @param workspace       optional reusable workspace; supplies the
     *                        {@code samplerWorkspace} / {@code samplerRunState}
     *                        slots when non-null
     */
    private static TemperingResult runTempering(
            StaticModel model, MCMCSequence move,
            int particles, int lenChain, double essRmin, long seed,
            double[] schedule, boolean adaptive, double adaptiveEssRmin,
            ParticleWorkspace workspace) {

        FKSMCSampler sampler;
        if (adaptive) {
            sampler = new com.curioloop.yum4j.ssm.particle.sampler.AdaptiveTempering(
                    model, move, particles, lenChain, adaptiveEssRmin);
        } else {
            sampler = new com.curioloop.yum4j.ssm.particle.sampler.Tempering(
                    model, move, particles, lenChain, essRmin, schedule);
        }

        int horizon = sampler.horizon();
        int fkDim = sampler.dim();

        // Adaptive tempering terminates early; allocate a generous run-state
        // buffer rather than the algorithmic worst case.
        int rsHorizon = adaptive ? Integer.MAX_VALUE : horizon;
        int placeholderLen = Math.min(rsHorizon, 10_000);

        Workspace ws;
        RunState rs;
        if (workspace != null) {
            ws = workspace.acquireSamplerWorkspace(particles, fkDim);
            rs = workspace.acquireSamplerRunState(placeholderLen, essRmin, seed);
        } else {
            ws = Workspace.allocate(particles, fkDim, HistoryMode.NONE, 0);
            rs = RunState.allocate(
                    placeholderLen, essRmin, Scheme.SYSTEMATIC, List.of(), seed);
        }
        ws.rng = RandomBatch.of(seed);

        // Sampler-style FKs do not consume a real observation stream;
        // a null-filled placeholder of length placeholderLen drives the engine.
        List<ThetaParticles> placeholder = Collections.nCopies(placeholderLen, null);

        if (adaptive) {
            Engine.init(ws, rs, sampler, ParallelStrategy.SERIAL, placeholder);
            while (Engine.step(ws, rs, sampler, ParallelStrategy.SERIAL, placeholder)) {
                if (((com.curioloop.yum4j.ssm.particle.sampler.AdaptiveTempering) sampler).done(rs)) break;
            }
        } else {
            Engine.run(ws, rs, sampler, ParallelStrategy.SERIAL, placeholder);
        }

        int steps = rs.stepCount;
        double[] logLt = Arrays.copyOf(rs.logLtSeries, steps);
        double[] ess = Arrays.copyOf(rs.essSeries, steps);
        boolean[] resampled = Arrays.copyOf(rs.resampledFlags, steps);

        // Re-attach a ThetaParticles view over ws.X to retrieve the
        // exponents schedule and path-sampling logZ accumulated by the
        // FK sampler.
        ThetaParticles theta = ThetaParticles.allocate(model.dim(), particles);
        theta.attachArena(ws.X, 0);
        double[] exponents = theta.exponents != null
                ? Arrays.copyOf(theta.exponents, theta.exponentsLen)
                : new double[0];
        double pathLogZ = theta.pathSamplingLogZ;
        double[] finalTheta = Arrays.copyOf(ws.X, fkDim * particles);

        return new TemperingResult(finalTheta, model.dim(), particles,
                logLt, ess, resampled, exponents, pathLogZ, steps);
    }

    final class PMMH<Y> implements ParticleSampling<PMMHResult> {
        private MultivariateDistribution prior;
        private Function<double[], ParticleSSM<Y>> modelFactory;
        private List<Y> observations;
        private int particles = -1;
        private int iterations = -1;
        private boolean adaptive = true;
        private long seed = 0L;
        private double[] theta0;

        public PMMH() {
        }

        public PMMH<Y> prior(MultivariateDistribution prior) {
            this.prior = prior;
            return this;
        }

        public <Z> PMMH<Z> model(Function<double[], ParticleSSM<Z>> modelFactory) {
            PMMH<Z> next = copyForModel();
            next.modelFactory = modelFactory;
            next.observations = null;
            return next;
        }

        public <Z> PMMH<Z> model(Function<double[], ParticleSSM<Z>> modelFactory,
                                 List<Z> observations) {
            PMMH<Z> next = copyForModel();
            next.modelFactory = modelFactory;
            next.observations = copyListOrNull(observations);
            return next;
        }

        public PMMH<Y> observations(List<Y> observations) {
            this.observations = copyListOrNull(observations);
            return this;
        }

        public PMMH<Y> particles(int particles) {
            this.particles = particles;
            return this;
        }

        public PMMH<Y> iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        public PMMH<Y> adaptive(boolean adaptive) {
            this.adaptive = adaptive;
            return this;
        }

        public PMMH<Y> seed(long seed) {
            this.seed = seed;
            return this;
        }

        public PMMH<Y> theta0(double[] theta0) {
            this.theta0 = copyOrNull(theta0);
            return this;
        }

        public double[] theta0() {
            return copyOrNull(theta0);
        }

        @Override
        public SamplingMethod<?, PMMHResult> method() {
            return SamplingMethod.PMMH;
        }

        @Override
        public PMMHResult run(ParticleWorkspace workspace) {
            MultivariateDistribution prior = requireSet(this.prior, "prior", ".prior(prior)");
            Function<double[], ParticleSSM<Y>> modelFactory = requireSet(
                    this.modelFactory, "modelFactory", ".model(modelFactory, observations)");
            List<Y> observations = requireSet(this.observations, "observations", ".observations(data)");
            requirePositive(particles, "particles", ".particles(N)");
            requirePositive(iterations, "iterations", ".iterations(n)");

            int dim = prior.dim();
            // Probe SSM to discover the state dimension, mirroring the
            // logic that previously lived in PMCMC.PmmhBuilder#run.
            ParticleSSM<Y> probeSsm = modelFactory.apply(new double[dim]);
            int stateDim = probeSsm.dim();
            int T = observations.size();

            Function<double[], FeynmanKac<?>> fkFactory = theta -> {
                ParticleSSM<Y> ssm = modelFactory.apply(theta);
                return FeynmanKac.bootstrap(ssm);
            };

            // PMMH's inner filter never needs trajectory history; reuse
            // the mcmc slot when a workspace is supplied.
            Workspace ws = null;
            RunState rs = null;
            if (workspace != null) {
                ws = workspace.acquireMcmcWorkspace(particles, stateDim, HistoryMode.NONE, 0);
                rs = workspace.acquireMcmcRunState(T, 0L);
            }

            var pmmh = new com.curioloop.yum4j.ssm.particle.mcmc.PMMH(
                    prior, fkFactory, observations,
                    particles, stateDim, T, iterations, adaptive,
                    com.curioloop.yum4j.ssm.particle.mcmc.PMMH.DEFAULT_FILTER_SEED,
                    ws, rs);

            double[] initTheta;
            if (theta0 != null) {
                initTheta = Arrays.copyOf(theta0, dim);
            } else {
                initTheta = new double[dim];
                RandomGenerator g0 = RandomBatch.of(seed).asRandomGenerator();
                prior.sample(g0, 1, initTheta, 0, null);
            }

            RandomGenerator g = RandomBatch.of(seed + 1).asRandomGenerator();
            MCMCChain chain = pmmh.run(initTheta, g);

            double[] chainData = Arrays.copyOf(chain.arena, dim * iterations);
            double[] logPost = Arrays.copyOfRange(
                    chain.arena, chain.LPOST_OFF, chain.LPOST_OFF + iterations);

            return new PMMHResult(chainData, dim, iterations, logPost, chain.accRate);
        }

        private <Z> PMMH<Z> copyForModel() {
            PMMH<Z> next = new PMMH<>();
            next.prior = prior;
            next.particles = particles;
            next.iterations = iterations;
            next.adaptive = adaptive;
            next.seed = seed;
            next.theta0 = copyOrNull(theta0);
            return next;
        }
    }

    final class Gibbs implements ParticleSampling<GibbsResult> {
        private Function<double[], FeynmanKac<?>> feynmanKacFactory;
        private int parameterDimension = -1;
        private int stateDimension = -1;
        private int timeCount = -1;
        private List<?> observations;
        private ThetaUpdater thetaUpdater;
        private int particles = -1;
        private int iterations = -1;
        private long seed = 0L;
        private double[] theta0;

        public Gibbs() {
        }

        public Gibbs feynmanKac(Function<double[], FeynmanKac<?>> feynmanKacFactory) {
            this.feynmanKacFactory = feynmanKacFactory;
            return this;
        }

        public Gibbs parameterDimension(int parameterDimension) {
            this.parameterDimension = parameterDimension;
            return this;
        }

        public Gibbs stateDimension(int stateDimension) {
            this.stateDimension = stateDimension;
            return this;
        }

        public Gibbs timeCount(int timeCount) {
            this.timeCount = timeCount;
            return this;
        }

        public Gibbs observations(List<?> observations) {
            this.observations = copyListOrNull(observations);
            return this;
        }

        public Gibbs thetaUpdater(ThetaUpdater thetaUpdater) {
            this.thetaUpdater = thetaUpdater;
            return this;
        }

        public Gibbs particles(int particles) {
            this.particles = particles;
            return this;
        }

        public Gibbs iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        public Gibbs seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Gibbs theta0(double[] theta0) {
            this.theta0 = copyOrNull(theta0);
            return this;
        }

        public double[] theta0() {
            return copyOrNull(theta0);
        }

        @Override
        public SamplingMethod<?, GibbsResult> method() {
            return SamplingMethod.PARTICLE_GIBBS;
        }

        @Override
        public GibbsResult run(ParticleWorkspace workspace) {
            Function<double[], FeynmanKac<?>> feynmanKacFactory = requireSet(
                    this.feynmanKacFactory, "feynmanKacFactory", ".feynmanKac(factory)");
            int parameterDimension = requirePositive(
                    this.parameterDimension, "parameterDimension", ".parameterDimension(dim)");
            int stateDimension = requirePositive(this.stateDimension, "stateDimension", ".stateDimension(dim)");
            int timeCount = requirePositive(this.timeCount, "timeCount", ".timeCount(T)");
            List<?> observations = requireSet(this.observations, "observations", ".observations(data)");
            ThetaUpdater thetaUpdater = requireSet(
                    this.thetaUpdater, "thetaUpdater", ".thetaUpdater(updater)");
            requirePositive(particles, "particles", ".particles(N)");
            requirePositive(iterations, "iterations", ".iterations(n)");

            double[] initTheta;
            if (theta0 != null) {
                initTheta = Arrays.copyOf(theta0, parameterDimension);
            } else {
                initTheta = new double[parameterDimension];
            }

            ThetaUpdater updater = thetaUpdater;
            // Particle Gibbs needs FULL history to extract trajectories
            // from each unconditional/conditional SMC pass. When a
            // workspace is supplied we route its mcmc-slot scratch
            // through ParticleGibbs's external-scratch ctor; otherwise
            // pass null and let ParticleGibbs allocate internally.
            Workspace ws = workspace != null
                    ? workspace.acquireMcmcWorkspace(
                            particles, stateDimension, HistoryMode.FULL, timeCount)
                    : null;
            RunState rs = workspace != null
                    ? workspace.acquireMcmcRunState(timeCount, 0L)
                    : null;
            ParticleGibbs pg = new ParticleGibbs(
                    feynmanKacFactory, observations, particles, iterations,
                    parameterDimension, stateDimension, timeCount, false,
                    ws, rs) {
                @Override
                protected void updateTheta(MCMCChain chain, int iter,
                                           double[] trajectory, double[] thetaCurr) {
                    updater.update(chain, iter, trajectory, thetaCurr);
                }
            };

            RandomGenerator g = RandomBatch.of(seed).asRandomGenerator();
            MCMCChain chain = pg.run(initTheta, g);

            double[] chainData = Arrays.copyOf(chain.arena, parameterDimension * iterations);
            double[] logPost = Arrays.copyOfRange(
                    chain.arena, chain.LPOST_OFF, chain.LPOST_OFF + iterations);

            return new GibbsResult(chainData, parameterDimension, iterations, logPost, chain.accRate);
        }
    }

    final class CSMC implements ParticleSampling<CSMCResult> {
        private FeynmanKac<?> feynmanKac;
        private double[] referenceTrajectory;
        private List<?> observations;
        private int particles = -1;
        private long seed = 0L;

        public CSMC() {
        }

        public CSMC feynmanKac(FeynmanKac<?> feynmanKac) {
            this.feynmanKac = feynmanKac;
            return this;
        }

        public CSMC referenceTrajectory(double[] referenceTrajectory) {
            this.referenceTrajectory = copyOrNull(referenceTrajectory);
            return this;
        }

        public CSMC observations(List<?> observations) {
            this.observations = copyListOrNull(observations);
            return this;
        }

        public CSMC particles(int particles) {
            this.particles = particles;
            return this;
        }

        public CSMC seed(long seed) {
            this.seed = seed;
            return this;
        }

        public double[] referenceTrajectory() {
            return copyOrNull(referenceTrajectory);
        }

        @Override
        public SamplingMethod<?, CSMCResult> method() {
            return SamplingMethod.CONDITIONAL_SMC;
        }

        @Override
        public CSMCResult run(ParticleWorkspace workspace) {
            FeynmanKac<?> feynmanKac = requireSet(this.feynmanKac, "feynmanKac", ".feynmanKac(fk)");
            double[] referenceTrajectory = requireSet(
                    this.referenceTrajectory, "referenceTrajectory", ".referenceTrajectory(xstar)");
            List<?> observations = requireSet(this.observations, "observations", ".observations(data)");
            requirePositive(particles, "particles", ".particles(N)");

            int stateDim = feynmanKac.dim();
            int T = observations.size();

            Workspace ws = workspace != null
                    ? workspace.acquireMcmcWorkspace(particles, stateDim, HistoryMode.FULL, T)
                    : null;
            RunState rs = workspace != null
                    ? workspace.acquireMcmcRunState(T, 0L)
                    : null;

            com.curioloop.yum4j.ssm.particle.kernel.RandomBatch rng = RandomBatch.of(seed);
            @SuppressWarnings({"unchecked", "rawtypes"})
            com.curioloop.yum4j.ssm.particle.mcmc.CSMC.CsmcRun result =
                    com.curioloop.yum4j.ssm.particle.mcmc.CSMC.run(
                            (FeynmanKac) feynmanKac, particles, referenceTrajectory,
                            (List) observations, rng, ws, rs);

            double[] trajectory = com.curioloop.yum4j.ssm.particle.mcmc.CSMC.extractTrajectory(result);
            double logLik = result.logLikelihood();
            int extractedT = referenceTrajectory.length / stateDim;

            return new CSMCResult(
                    Arrays.copyOf(trajectory, trajectory.length),
                    logLik, extractedT, stateDim);
        }
    }

    final class IBIS implements ParticleSampling<IBISResult> {
        private StaticModel model;
        private MCMCSequence move;
        private int particles = -1;
        private int lenChain = 10;
        private double essRmin = 0.5;
        private long seed = 0L;

        public IBIS() {
        }

        public IBIS model(StaticModel model) {
            this.model = model;
            return this;
        }

        public IBIS move(MCMCSequence move) {
            this.move = move;
            return this;
        }

        public IBIS particles(int particles) {
            this.particles = particles;
            return this;
        }

        public IBIS lenChain(int lenChain) {
            this.lenChain = lenChain;
            return this;
        }

        public IBIS essRmin(double essRmin) {
            this.essRmin = essRmin;
            return this;
        }

        public IBIS seed(long seed) {
            this.seed = seed;
            return this;
        }

        @Override
        public SamplingMethod<?, IBISResult> method() {
            return SamplingMethod.IBIS;
        }

        @Override
        public IBISResult run(ParticleWorkspace workspace) {
            StaticModel model = requireSet(this.model, "model", ".model(model)");
            MCMCSequence move = requireSet(this.move, "move", ".move(move)");
            requirePositive(particles, "particles", ".particles(N)");

            com.curioloop.yum4j.ssm.particle.sampler.IBIS ibis =
                    new com.curioloop.yum4j.ssm.particle.sampler.IBIS(
                            model, move, particles, lenChain, essRmin);
            int T = model.T();
            int dim = model.dim();
            int fkDim = ibis.dim();

            Workspace ws;
            RunState rs;
            if (workspace != null) {
                ws = workspace.acquireSamplerWorkspace(particles, fkDim);
                rs = workspace.acquireSamplerRunState(T, essRmin, seed);
            } else {
                ws = Workspace.allocate(particles, fkDim, HistoryMode.NONE, 0);
                rs = RunState.allocate(T, essRmin, Scheme.SYSTEMATIC, List.of(), seed);
            }
            ws.rng = RandomBatch.of(seed);

            // Sampler-style FKs do not consume a real observation stream;
            // a null-filled placeholder of length T supplies the horizon.
            List<ThetaParticles> placeholder = Collections.nCopies(T, null);
            Engine.run(ws, rs, ibis, ParallelStrategy.SERIAL, placeholder);

            int steps = rs.stepCount;
            double[] logLt = Arrays.copyOf(rs.logLtSeries, steps);
            double[] ess = Arrays.copyOf(rs.essSeries, steps);
            boolean[] resampled = Arrays.copyOf(rs.resampledFlags, steps);
            double[] finalTheta = Arrays.copyOf(ws.X, fkDim * particles);

            return new IBISResult(finalTheta, dim, particles, logLt, ess, resampled, steps);
        }
    }

    final class SMC2<Y> implements ParticleSampling<SMC2Result> {
        private Function<double[], ParticleSSM<Y>> modelFactory;
        private MultivariateDistribution prior;
        private List<Y> observations;
        private MCMCSequence move;
        private int particles = -1;
        private int innerParticles = 100;
        private double essRmin = 0.5;
        private double essRminInner = 0.5;
        private long seed = 0L;

        public SMC2() {
        }

        public SMC2<Y> prior(MultivariateDistribution prior) {
            this.prior = prior;
            return this;
        }

        public <Z> SMC2<Z> model(Function<double[], ParticleSSM<Z>> modelFactory) {
            SMC2<Z> next = copyForModel();
            next.modelFactory = modelFactory;
            next.observations = null;
            return next;
        }

        public <Z> SMC2<Z> model(Function<double[], ParticleSSM<Z>> modelFactory,
                                 List<Z> observations) {
            SMC2<Z> next = copyForModel();
            next.modelFactory = modelFactory;
            next.observations = copyListOrNull(observations);
            return next;
        }

        public SMC2<Y> observations(List<Y> observations) {
            this.observations = copyListOrNull(observations);
            return this;
        }

        public SMC2<Y> move(MCMCSequence move) {
            this.move = move;
            return this;
        }

        public SMC2<Y> particles(int particles) {
            this.particles = particles;
            return this;
        }

        public SMC2<Y> innerParticles(int innerParticles) {
            this.innerParticles = innerParticles;
            return this;
        }

        public SMC2<Y> essRmin(double essRmin) {
            this.essRmin = essRmin;
            return this;
        }

        public SMC2<Y> essRminInner(double essRminInner) {
            this.essRminInner = essRminInner;
            return this;
        }

        public SMC2<Y> seed(long seed) {
            this.seed = seed;
            return this;
        }

        @Override
        public SamplingMethod<?, SMC2Result> method() {
            return SamplingMethod.SMC2;
        }

        /**
         * Runs SMC². The {@code workspace} argument is currently ignored:
         * SMC² owns a nested θ-particle × x-particle workspace that does
         * not fit the {@link ParticleWorkspace#acquireSamplerWorkspace}
         * or {@link ParticleWorkspace#acquireMcmcWorkspace} slots.
         */
        @Override
        public SMC2Result run(ParticleWorkspace workspace) {
            Function<double[], ParticleSSM<Y>> modelFactory = requireSet(
                    this.modelFactory, "modelFactory", ".model(modelFactory, observations)");
            MultivariateDistribution prior = requireSet(this.prior, "prior", ".prior(prior)");
            List<Y> observations = requireSet(this.observations, "observations", ".observations(data)");
            MCMCSequence move = requireSet(this.move, "move", ".move(move)");
            requirePositive(particles, "particles", ".particles(N)");

            int dimTheta = prior.dim();
            // Probe the SSM factory once to discover the state dimension.
            ParticleSSM<Y> probeSsm = modelFactory.apply(new double[dimTheta]);
            int dimX = probeSsm.dim();

            SMCSquared.Config config = new SMCSquared.Config(
                    particles, innerParticles, dimTheta, dimX,
                    essRmin, essRminInner, move, seed);
            SMCSquared.Result result = SMCSquared.run(modelFactory, prior, observations, config);

            double[] finalTheta = Arrays.copyOf(result.finalTheta().arena,
                    result.finalTheta().arenaLength());
            double[] logLt = Arrays.copyOf(result.logLt(), result.T());
            double[] ess = Arrays.copyOf(result.ESS(), result.T());
            boolean[] resampled = Arrays.copyOf(result.rsFlag(), result.T());

            return new SMC2Result(finalTheta, dimTheta, particles, logLt, ess, resampled, result.T());
        }

        private <Z> SMC2<Z> copyForModel() {
            SMC2<Z> next = new SMC2<>();
            next.prior = prior;
            next.move = move;
            next.particles = particles;
            next.innerParticles = innerParticles;
            next.essRmin = essRmin;
            next.essRminInner = essRminInner;
            next.seed = seed;
            return next;
        }
    }

    final class Tempering implements ParticleSampling<TemperingResult> {
        private StaticModel model;
        private MCMCSequence move;
        private double[] schedule;
        private int particles = -1;
        private int lenChain = 10;
        private double essRmin = 0.5;
        private long seed = 0L;

        public Tempering() {
        }

        public Tempering model(StaticModel model) {
            this.model = model;
            return this;
        }

        public Tempering move(MCMCSequence move) {
            this.move = move;
            return this;
        }

        public Tempering schedule(double[] schedule) {
            this.schedule = copyOrNull(schedule);
            return this;
        }

        public Tempering particles(int particles) {
            this.particles = particles;
            return this;
        }

        public Tempering lenChain(int lenChain) {
            this.lenChain = lenChain;
            return this;
        }

        public Tempering essRmin(double essRmin) {
            this.essRmin = essRmin;
            return this;
        }

        public Tempering seed(long seed) {
            this.seed = seed;
            return this;
        }

        public double[] schedule() {
            return copyOrNull(schedule);
        }

        @Override
        public SamplingMethod<?, TemperingResult> method() {
            return SamplingMethod.TEMPERING;
        }

        @Override
        public TemperingResult run(ParticleWorkspace workspace) {
            StaticModel model = requireSet(this.model, "model", ".model(model)");
            MCMCSequence move = requireSet(this.move, "move", ".move(move)");
            double[] schedule = requireSet(this.schedule, "schedule", ".schedule(schedule)");
            requirePositive(particles, "particles", ".particles(N)");
            return runTempering(model, move, particles, lenChain, essRmin, seed,
                    /* schedule */ schedule, /* adaptive */ false, /* adaptiveEssRmin */ 0.0,
                    workspace);
        }
    }

    final class AdaptiveTempering implements ParticleSampling<TemperingResult> {
        private StaticModel model;
        private MCMCSequence move;
        private double adaptiveEssRmin = 0.5;
        private int particles = -1;
        private int lenChain = 10;
        private double essRmin = 0.5;
        private long seed = 0L;

        public AdaptiveTempering() {
        }

        public AdaptiveTempering model(StaticModel model) {
            this.model = model;
            return this;
        }

        public AdaptiveTempering move(MCMCSequence move) {
            this.move = move;
            return this;
        }

        public AdaptiveTempering adaptiveEssRmin(double adaptiveEssRmin) {
            this.adaptiveEssRmin = adaptiveEssRmin;
            return this;
        }

        public AdaptiveTempering particles(int particles) {
            this.particles = particles;
            return this;
        }

        public AdaptiveTempering lenChain(int lenChain) {
            this.lenChain = lenChain;
            return this;
        }

        public AdaptiveTempering essRmin(double essRmin) {
            this.essRmin = essRmin;
            return this;
        }

        public AdaptiveTempering seed(long seed) {
            this.seed = seed;
            return this;
        }

        @Override
        public SamplingMethod<?, TemperingResult> method() {
            return SamplingMethod.ADAPTIVE_TEMPERING;
        }

        @Override
        public TemperingResult run(ParticleWorkspace workspace) {
            StaticModel model = requireSet(this.model, "model", ".model(model)");
            MCMCSequence move = requireSet(this.move, "move", ".move(move)");
            requirePositive(particles, "particles", ".particles(N)");
            return runTempering(model, move, particles, lenChain, essRmin, seed,
                    /* schedule */ null, /* adaptive */ true, adaptiveEssRmin,
                    workspace);
        }
    }

    final class Nested implements ParticleSampling<NestedSamplingResult> {
        private StaticModel model;
        private int livePoints = -1;
        private double eps = 1e-8;
        private int nsteps = 10;
        private double scale = 0.5;
        private long seed = 0L;

        public Nested() {
        }

        public Nested model(StaticModel model) {
            this.model = model;
            return this;
        }

        public Nested livePoints(int livePoints) {
            this.livePoints = livePoints;
            return this;
        }

        public Nested eps(double eps) {
            this.eps = eps;
            return this;
        }

        public Nested nsteps(int nsteps) {
            this.nsteps = nsteps;
            return this;
        }

        public Nested scale(double scale) {
            this.scale = scale;
            return this;
        }

        public Nested seed(long seed) {
            this.seed = seed;
            return this;
        }

        @Override
        public SamplingMethod<?, NestedSamplingResult> method() {
            return SamplingMethod.NESTED;
        }

        /**
         * Runs nested sampling. The {@code workspace} argument is
         * currently ignored: nested sampling drives a
         * live-points matrix that doesn't share shape with the filter,
         * mcmc, or sampler slots of {@link ParticleWorkspace}.
         */
        @Override
        public NestedSamplingResult run(ParticleWorkspace workspace) {
            StaticModel model = requireSet(this.model, "model", ".model(model)");
            requirePositive(livePoints, "livePoints", ".livePoints(N)");
            NestedSampling ns = new NestedRwMoves(model, livePoints, eps, nsteps, scale);
            RandomGenerator g = RandomBatch.of(seed).asRandomGenerator();
            return ns.run(g);
        }
    }

    private static <T> List<T> copyListOrNull(List<T> values) {
        return values == null ? null : List.copyOf(values);
    }
}
