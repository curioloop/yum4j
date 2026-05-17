package com.curioloop.yum4j.ssm.particle.sampler;
import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequence;

import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.engine.Engine;
import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.engine.ParallelStrategy;
import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.resample.Resample;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import com.curioloop.yum4j.ssm.particle.dist.MultivariateDistribution;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/**
 * SMC² (Sequential Monte Carlo squared) algorithm.
 *
 * <p>Combines an outer SMC sampler over θ-particles with inner particle
 * filters (one per θ-particle) to estimate the likelihood of a
 * state-space model. Each θ-particle maintains its own bootstrap
 * particle filter that provides an unbiased estimate of
 * {@code log p(y_{0:t} | θ)}.
 *
 * <p>Reference: Chopin, Jacob &amp; Papaspiliopoulos (2013),
 * "SMC² : an efficient algorithm for sequential analysis of state space
 * models".
 *
 * <p>All inner-filter state is stored in flat pre-allocated arrays to
 * avoid allocation in the main loop. Uses the new {@link Engine},
 * {@link Workspace}, and {@link RunState} infrastructure.
 *
 * @see ThetaParticles
 * @see MCMCSequence
 */
public final class SMCSquared {

    private SMCSquared() {}

    /**
     * Configuration for SMC².
     */
    public record Config(
        int N,
        int Nx,
        int dimTheta,
        int dimX,
        double essRmin,
        double essRminInner,
        MCMCSequence move,
        long seed
    ) {}

    /**
     * Result of an SMC² run.
     */
    public record Result(
        ThetaParticles finalTheta,
        double[] logLt,
        double[] ESS,
        boolean[] rsFlag,
        int T
    ) {}

    /**
     * Run SMC² on a state-space model.
     *
    * @param ssmFactory factory that creates a ParticleSSM from a theta vector
     * @param prior      prior distribution over θ
     * @param data       observation sequence
     * @param config     SMC² configuration
     * @param <Y>        observation type
     * @return SMC² result
     */
    public static <Y> Result run(
        Function<double[], ParticleSSM<Y>> ssmFactory,
        MultivariateDistribution prior,
        List<Y> data,
        Config config
    ) {
        final int N = config.N;
        final int Nx = config.Nx;
        final int dimTheta = config.dimTheta;
        final int dimX = config.dimX;
        final int T = data.size();
        final double essRmin = config.essRmin;
        final double essRminInner = config.essRminInner;

        RandomBatch masterRng = RandomBatch.of(config.seed);
        RandomGenerator rng = masterRng.asRandomGenerator();

        // --- θ-particles ---
        ThetaParticles theta = ThetaParticles.allocate(dimTheta, N);

        prior.sample(rng, N, theta.arena, 0, null);

        double[] priorScratch = new double[dimTheta * N];
        prior.logPdfBatch(theta.arena, 0, N, theta.arena, theta.lpriorOff(), priorScratch);

        // --- Inner filter state (flat packed arrays) ---
        double[] allInnerX = new double[N * Nx * dimX];
        double[] allInnerXprev = new double[N * Nx * dimX];
        double[] allInnerLw = new double[N * Nx];
        double[] innerLogLt = new double[N];
        double[] prevInnerLogLt = new double[N];

        // θ-particle log-weights
        double[] thetaLw = new double[N];

        // Per-step summaries
        double[] logLtArr = new double[T];
        double[] essArr = new double[T];
        boolean[] rsFlagArr = new boolean[T];

        // Scratch buffers (allocated once, reused across all steps).
        double[] innerLwSlice = new double[Nx];
        // innerResScratch must satisfy Resample's per-scheme contract
        // (R7.4). For SYSTEMATIC at M = Nx that's Nx + 1; size to 3*Nx
        // to match Workspace.scratch and cover every scheme.
        double[] innerResScratch = new double[3 * Nx];
        int[] innerAncestors = new int[Nx];
        double[] logGIncr = new double[Nx];
        // thetaResScratch likewise: SYSTEMATIC at M = N needs N + 1; use
        // 3*N for uniform coverage of any future scheme change.
        double[] thetaResScratch = new double[3 * N];
        int[] thetaAncestors = new int[N];

        // SSM instances per θ-particle — gather via index array (no clone).
        @SuppressWarnings("unchecked")
        ParticleSSM<Y>[] ssms = new ParticleSSM[N];
        @SuppressWarnings("unchecked")
        ParticleSSM<Y>[] ssmsTemp = new ParticleSSM[N];
        double[] thetaVec = new double[dimTheta];
        for (int n = 0; n < N; n++) {
            extractTheta(theta.arena, n, N, dimTheta, thetaVec);
            ssms[n] = ssmFactory.apply(thetaVec.clone());
        }

        // Pre-allocate fresh filter workspace/runstate for MCMC rejuvenation
        Workspace freshWs = (T > 0)
            ? Workspace.allocate(Nx, dimX, HistoryMode.NONE, 0)
            : null;
        RunState freshRs = (T > 0)
            ? RunState.allocate(T, essRminInner, Scheme.SYSTEMATIC, List.of(), 0L)
            : null;

        // Scratch for prior evaluation during MCMC (reused across iterations).
        double[] mcmcPriorOut = new double[1];
        double[] mcmcPriorScratch = new double[dimTheta];
        double[] thetaStar = new double[dimTheta];
        double[] thetaStarForSsm = new double[dimTheta];

        // Temp buffers for θ-resampling of inner state (reused).
        double[] tempInnerLw = new double[N * Nx];
        double[] tempInnerLogLt = new double[N];

        // Temp ThetaParticles for gather (avoids aliasing).
        ThetaParticles thetaTemp = ThetaParticles.allocate(dimTheta, N);

        // Uniform weights for MCMC calibration (reused).
        double[] uniformW = new double[N];
        Arrays.fill(uniformW, 1.0 / N);

        // Inner-filter RNG for resampling
        RandomBatch innerRng = RandomBatch.of(config.seed + 7777);

        // Pre-split a per-θ-particle RNG pool used by the MCMC rejuvenation
        // loop. mcmcRngBase splits once from the master at run start, then
        // each θ-particle index gets its own deterministic substream. This
        // replaces the per-MH-proposal RandomBatch.of(seedCounter++)
        // allocation with N + 1 allocations at startup and zero allocations
        // in the hot loop (R9.1).
        RandomBatch mcmcRngBase = masterRng.split(9999L);
        RandomBatch[] mcmcRngPool = new RandomBatch[N];
        for (int n = 0; n < N; n++) {
            mcmcRngPool[n] = mcmcRngBase.split(n);
        }

        // --- Main loop ---

        for (int t = 0; t < T; t++) {
            Y yt = data.get(t);

            // 1. Advance all inner filters by one step
            for (int n = 0; n < N; n++) {
                int xBase = n * Nx * dimX;
                int lwBase = n * Nx;

                prevInnerLogLt[n] = innerLogLt[n];

                if (t == 0) {
                    // Initial step: sample from prior and compute logG
                    sampleM0(ssms[n], allInnerX, xBase, Nx, rng);
                    logPY(ssms[n], 0, yt, null, 0, allInnerX, xBase, Nx, logGIncr, 0);
                    System.arraycopy(logGIncr, 0, allInnerLw, lwBase, Nx);
                    LogWeight.Triple ri = LogWeight.logSumEssMax(allInnerLw, lwBase, Nx);
                    innerLogLt[n] = ri.logSum() - Math.log(Nx);
                } else {
                    // Entry reduction: one pass for (logSumExp, ESS, max).
                    LogWeight.Triple ri = LogWeight.logSumEssMax(allInnerLw, lwBase, Nx);
                    double logSumCurr = ri.logSum();
                    double essInner = ri.ess();

                    double logSumBefore;
                    if (essInner < essRminInner * Nx) {
                        System.arraycopy(allInnerLw, lwBase, innerLwSlice, 0, Nx);
                        Resample.apply(Scheme.SYSTEMATIC, innerLwSlice, Nx, Nx,
                            innerAncestors, innerRng, innerResScratch);
                        gatherInnerParticles(allInnerX, xBase, allInnerXprev, xBase,
                            innerAncestors, dimX, Nx);
                        Arrays.fill(allInnerLw, lwBase, lwBase + Nx, 0.0);
                        logSumBefore = Math.log(Nx);
                    } else {
                        System.arraycopy(allInnerX, xBase, allInnerXprev, xBase, Nx * dimX);
                        logSumBefore = logSumCurr;
                    }

                    sampleM(ssms[n], t, allInnerXprev, xBase, allInnerX, xBase, Nx, rng);
                    logPY(ssms[n], t, yt, allInnerXprev, xBase, allInnerX, xBase, Nx, logGIncr, 0);

                    // Exit reduction fused with lw += logG.
                    LogWeight.Triple re = LogWeight.addIntoAndReduce(
                        allInnerLw, lwBase, logGIncr, 0, Nx);
                    innerLogLt[n] += re.logSum() - logSumBefore;
                }
            }

            // 2. Reweight θ-particles
            for (int n = 0; n < N; n++) {
                thetaLw[n] += innerLogLt[n] - prevInnerLogLt[n];
            }

            // 3. Record summaries — fused logSumExp/ESS over theta.
            LogWeight.Triple rt = LogWeight.logSumEssMax(thetaLw, 0, N);
            double logSumTheta = rt.logSum();
            double essTheta = rt.ess();
            essArr[t] = essTheta;
            logLtArr[t] = logSumTheta - Math.log(N);

            // 4. Resample θ-particles if ESS < threshold
            if (essTheta < essRmin * N) {
                rsFlagArr[t] = true;

                Resample.apply(Scheme.SYSTEMATIC, thetaLw, N, N,
                    thetaAncestors, masterRng, thetaResScratch);

                // Deep-gather inner filter states using allInnerXprev as a scratch destination.
                for (int n = 0; n < N; n++) {
                    int src = thetaAncestors[n];
                    System.arraycopy(allInnerX, src * Nx * dimX,
                        allInnerXprev, n * Nx * dimX, Nx * dimX);
                }
                System.arraycopy(allInnerXprev, 0, allInnerX, 0, N * Nx * dimX);

                // Gather inner log-weights
                for (int n = 0; n < N; n++) {
                    int src = thetaAncestors[n];
                    System.arraycopy(allInnerLw, src * Nx, tempInnerLw, n * Nx, Nx);
                }
                System.arraycopy(tempInnerLw, 0, allInnerLw, 0, N * Nx);

                // Gather innerLogLt
                for (int n = 0; n < N; n++) {
                    tempInnerLogLt[n] = innerLogLt[thetaAncestors[n]];
                }
                System.arraycopy(tempInnerLogLt, 0, innerLogLt, 0, N);

                // Gather θ-particles (arena includes lpost/llik/lprior)
                theta.gatherInto(thetaTemp, thetaAncestors, N);
                System.arraycopy(thetaTemp.arena, 0, theta.arena, 0, theta.arenaLength());

                // Gather SSM instances via ssmsTemp (no per-iteration clone()).
                for (int n = 0; n < N; n++) {
                    ssmsTemp[n] = ssms[thetaAncestors[n]];
                }
                System.arraycopy(ssmsTemp, 0, ssms, 0, N);

                Arrays.fill(thetaLw, 0.0);

                // 5. MCMC rejuvenation
                config.move.calibrate(uniformW, theta);

                for (int n = 0; n < N; n++) {
                    extractTheta(theta.arena, n, N, dimTheta, thetaVec);

                    for (int j = 0; j < dimTheta; j++) {
                        thetaStar[j] = thetaVec[j] + 0.1 * rng.nextGaussian();
                    }

                    prior.logPdfBatch(thetaStar, 0, 1, mcmcPriorOut, 0, mcmcPriorScratch);
                    double logPriorStar = mcmcPriorOut[0];

                    if (logPriorStar == Double.NEGATIVE_INFINITY) {
                        continue;
                    }

                    // Freshly-constructed SSMs for thetaStar — unavoidable allocation
                    // (the factory is user-supplied and may hold non-sharable state).
                    System.arraycopy(thetaStar, 0, thetaStarForSsm, 0, dimTheta);
                    ParticleSSM<Y> ssmStar = ssmFactory.apply(thetaStarForSsm.clone());

                    // Run a fresh inner filter using the new engine infrastructure.
                    // Reuses the pre-split RandomBatch from the pool — zero
                    // allocations per MH proposal (R9.1).
                    freshWs.rng = mcmcRngPool[n];
                    freshRs.reset(0L);
                    FeynmanKac<Y> fk = FeynmanKac.bootstrap(ssmStar);
                    Engine.run(freshWs, freshRs, fk, ParallelStrategy.SERIAL, data.subList(0, t + 1));
                    double logZStar = freshRs.logLtSeries[t];

                    double logPriorCurr = theta.arena[theta.lpriorOff() + n];
                    double logLikCurr = innerLogLt[n];
                    double lpostCurr = logPriorCurr + logLikCurr;

                    double lpostStar = logPriorStar + logZStar;

                    double logAlpha = lpostStar - lpostCurr;
                    if (Math.log(rng.nextDouble()) < logAlpha) {
                        for (int j = 0; j < dimTheta; j++) {
                            theta.arena[j * N + n] = thetaStar[j];
                        }
                        theta.arena[theta.lpriorOff() + n] = logPriorStar;
                        theta.arena[theta.llikOff() + n] = logZStar;
                        theta.arena[theta.lpostOff() + n] = lpostStar;

                        int xBase = n * Nx * dimX;
                        System.arraycopy(freshWs.X, 0, allInnerX, xBase, Nx * dimX);
                        System.arraycopy(freshWs.logW, 0, allInnerLw, n * Nx, Nx);
                        innerLogLt[n] = logZStar;

                        ssms[n] = ssmStar;
                    }
                }
            } else {
                rsFlagArr[t] = false;
            }

            // Update lpost and llik on theta particles
            for (int n = 0; n < N; n++) {
                theta.arena[theta.llikOff() + n] = innerLogLt[n];
                theta.arena[theta.lpostOff() + n] =
                    theta.arena[theta.lpriorOff() + n] + innerLogLt[n];
            }
        }

        return new Result(theta, logLtArr, essArr, rsFlagArr, T);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /**
     * Extract the theta vector for particle n from column-major arena.
     */
    private static void extractTheta(double[] arena, int n, int N, int dim, double[] out) {
        for (int j = 0; j < dim; j++) {
            out[j] = arena[j * N + n];
        }
    }

    /**
     * Gather inner particles according to ancestor indices.
     * Column-major (dimX, Nx) layout within each θ-particle's block.
     */
    private static void gatherInnerParticles(double[] src, int srcOff,
                                             double[] dst, int dstOff,
                                             int[] ancestors, int dimX, int Nx) {
        for (int j = 0; j < dimX; j++) {
            int rowOff = j * Nx;
            for (int i = 0; i < Nx; i++) {
                dst[dstOff + rowOff + i] = src[srcOff + rowOff + ancestors[i]];
            }
        }
    }

    /**
     * Sample initial state from the SSM's transition prior.
     * Delegates to the ParticleSSM's sampleM0 via a minimal StepContext.
     */
    private static <Y> void sampleM0(ParticleSSM<Y> ssm, double[] X, int xOff, int N, RandomGenerator g) {
        // Use a lightweight approach: call the transition's sampleM0 via StepContext
        RandomBatch rng = RandomBatch.of(g.nextLong());
        var ctx = new com.curioloop.yum4j.ssm.particle.engine.StepContext<Y>(
            0, null, null, 0, X, xOff, new double[N], 0, N, ssm.dim(), rng, new double[N], 0);
        ssm.sampleM0(ctx);
    }

    /**
     * Sample transition from the SSM.
     */
    private static <Y> void sampleM(ParticleSSM<Y> ssm, int t, double[] Xprev, int xpOff,
                                    double[] X, int xOff, int N, RandomGenerator g) {
        RandomBatch rng = RandomBatch.of(g.nextLong());
        var ctx = new com.curioloop.yum4j.ssm.particle.engine.StepContext<Y>(
            t, null, Xprev, xpOff, X, xOff, new double[N], 0, N, ssm.dim(), rng, new double[N], 0);
        ssm.sampleM(ctx);
    }

    /**
     * Compute log observation density from the SSM.
     */
    private static <Y> void logPY(ParticleSSM<Y> ssm, int t, Y obs,
                                  double[] Xprev, int xpOff,
                                  double[] X, int xOff, int N,
                                  double[] out, int outOff) {
        RandomBatch rng = RandomBatch.of(0L);
        var ctx = new com.curioloop.yum4j.ssm.particle.engine.StepContext<>(
            t, obs, Xprev != null ? Xprev : X, xpOff, X, xOff, out, outOff, N, ssm.dim(), rng, new double[N], 0);
        ssm.logG(ctx);
    }
}
