package com.curioloop.yum4j.ssm.particle.mcmc;

import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.engine.*;
import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import com.curioloop.yum4j.ssm.particle.dist.MultivariateDistribution;

import java.util.List;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/**
 * Particle Marginal Metropolis-Hastings (PMMH).
 *
 * <p>At each MCMC iteration, proposes θ* from a random walk, runs a
 * particle filter with N particles to estimate {@code log p(Y|θ*)},
 * and accepts/rejects via the MH ratio. The internal {@link Workspace}
 * and {@link RunState} are pre-allocated and reused across iterations
 * to avoid GC pressure.
 *
 * <p>Uses the new engine infrastructure: {@link Engine}, {@link Workspace},
 * {@link RunState}, {@link FeynmanKac}.
 *
 * <p>RNG ownership (R9.2): a single {@link RandomBatch} is constructed
 * once in the constructor from {@code filterSeed} and installed as
 * {@link Workspace#rng} for every {@code computeLogPost} call. The
 * shared stream advances monotonically across iterations; chain output
 * for a given {@code filterSeed} is stable but differs from the
 * pre-stage-F variant which allocated a fresh {@code RandomBatch.of(seedCounter++)}
 * per proposal. Baseline regenerated 2025-01 for stage F (R9.5):
 * PMMH now reuses a single RandomBatch across iterations.
 *
 * @see GenericRWHM
 */
public final class PMMH extends GenericRWHM {

    /** Default seed for the internal filter RNG when no override is supplied. */
    public static final long DEFAULT_FILTER_SEED = 42L;

    /** Factory that creates a FeynmanKac given a parameter vector. */
    private final Function<double[], FeynmanKac<?>> fkFactory;

    /** Observation sequence; supplies the horizon and observations to the engine. */
    private final List<?> data;

    /** Number of particles for the internal filter. */
    private final int Nx;

    /** Horizon (number of time steps). */
    private final int T;

    /** State dimension of the particle filter model. */
    private final int stateDim;

    /** Pre-allocated Workspace reused across iterations. */
    private Workspace reusableWs;

    /** Pre-allocated RunState reused across iterations. */
    private RunState reusableRs;

    /**
     * Single batched RNG owned by this PMMH instance and shared across every
     * {@code computeLogPost} call (R9.2). Constructed once from
     * {@code filterSeed}; never reallocated.
     */
    private final RandomBatch filterRng;

    /**
     * Create a PMMH sampler with an explicit filter-RNG seed.
     *
     * @param prior      prior distribution over θ
     * @param fkFactory  factory that creates a FeynmanKac from a parameter vector
     * @param data       observation sequence threaded into {@code Engine.run}
     * @param Nx         number of particles for the internal filter
     * @param stateDim   state dimension of the model
     * @param T          time horizon (number of observations); must equal {@code data.size()}
     * @param niter      number of MCMC iterations
     * @param adaptive   whether to adapt the proposal covariance
     * @param filterSeed seed for the shared internal-filter {@link RandomBatch}
     */
    public PMMH(MultivariateDistribution prior,
                Function<double[], FeynmanKac<?>> fkFactory,
                List<?> data,
                int Nx, int stateDim, int T,
                int niter, boolean adaptive,
                long filterSeed) {
        this(prior, fkFactory, data, Nx, stateDim, T, niter, adaptive, filterSeed,  null, null);
    }

    /**
     * Create a PMMH sampler that delegates inner-filter scratch to
     * caller-supplied {@link Workspace} / {@link RunState} instances.
     * The caller (typically a {@code ParticleWorkspace}) owns the
     * lifecycle of {@code reusableWs} / {@code reusableRs}; the
     * sampler only resets and uses them.
     *
     * @param prior         prior distribution over θ
     * @param fkFactory     factory creating a FeynmanKac from a parameter vector
     * @param data          observation sequence
     * @param Nx            number of particles for the internal filter
     * @param stateDim      state dimension of the model
     * @param T             time horizon
     * @param niter         number of MCMC iterations
     * @param adaptive      whether to adapt the proposal covariance
     * @param filterSeed    seed for the shared filter RNG
     * @param externalWs    pre-allocated workspace owned by the caller
     * @param externalRs    pre-allocated run state owned by the caller
     */
    public PMMH(MultivariateDistribution prior,
                Function<double[], FeynmanKac<?>> fkFactory,
                List<?> data,
                int Nx, int stateDim, int T,
                int niter, boolean adaptive,
                long filterSeed,
                Workspace externalWs, RunState externalRs) {
        super(prior, niter, adaptive, 0.6);
        this.fkFactory = fkFactory;
        this.data = data;
        this.Nx = Nx;
        this.T = T;
        this.stateDim = stateDim;
        this.reusableWs = externalWs != null ? externalWs : Workspace.allocate(Nx, stateDim, HistoryMode.NONE, 0);
        this.reusableRs = externalRs != null ? externalRs : RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, List.of(), 0L);
        this.filterRng = RandomBatch.of(filterSeed);
    }

    /**
     * Create a PMMH sampler with the default filter-RNG seed.
     */
    public PMMH(MultivariateDistribution prior,
                Function<double[], FeynmanKac<?>> fkFactory,
                List<?> data,
                int Nx, int stateDim, int T,
                int niter, boolean adaptive) {
        this(prior, fkFactory, data, Nx, stateDim, T, niter, adaptive, DEFAULT_FILTER_SEED);
    }

    /**
     * Create a PMMH sampler with adaptive proposals (default) and the default
     * filter-RNG seed.
     */
    public PMMH(MultivariateDistribution prior,
                Function<double[], FeynmanKac<?>> fkFactory,
                List<?> data,
                int Nx, int stateDim, int T, int niter) {
        this(prior, fkFactory, data, Nx, stateDim, T, niter, true, DEFAULT_FILTER_SEED);
    }

    @Override
    @SuppressWarnings("unchecked")
    protected double computeLogPost(double[] thetaProposed) {
        // Compute log-prior using parent's reusable buffers.
        if (priorOut == null || priorOut.length < 1) {
            priorOut = new double[1];
            priorScratch = new double[dim];
        }
        prior.logPdfBatch(thetaProposed, 0, 1, priorOut, 0, priorScratch);
        double lprior = priorOut[0];

        if (lprior == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        // Create FK model for proposed theta and run particle filter
        FeynmanKac fk = fkFactory.apply(thetaProposed);

        // Reset the run state and reuse the shared RandomBatch (R9.2).
        // The reference is installed once per call (no allocation); the
        // batched stream advances monotonically across iterations.
        reusableRs.reset(0L);
        reusableWs.rng = filterRng;
        reusableWs.resetBuffers();

        Engine.run(reusableWs, reusableRs, fk, ParallelStrategy.SERIAL, (List) data);

        // Extract log Z from the final step
        int lastStep = reusableRs.stepCount - 1;
        double logZ = lastStep >= 0 ? reusableRs.logLtSeries[lastStep] : 0.0;

        return lprior + logZ;
    }

    /**
     * Run PMMH with a custom initial theta and RNG.
     * Overrides the base class to skip the redundant prior check
     * (computeLogPost already checks prior).
     */
    @Override
    public MCMCChain run(double[] theta0, RandomGenerator g) {
        MCMCChain chain = new MCMCChain(dim, niter);

        if (thetaCurrBuf == null || thetaCurrBuf.length < dim) {
            thetaCurrBuf = new double[dim];
            thetaPropBuf = new double[dim];
            zBuf = new double[dim];
            priorOut = new double[1];
            priorScratch = new double[dim];
        }
        final double[] thetaCurr = thetaCurrBuf;
        final double[] thetaProp = thetaPropBuf;
        final double[] z = zBuf;

        System.arraycopy(theta0, 0, thetaCurr, 0, dim);

        // Compute initial log-posterior
        double lpostCurr = computeLogPost(thetaCurr);

        int accepted = 0;

        for (int iter = 0; iter < niter; iter++) {
            // Propose: thetaProp = thetaCurr + L * z
            for (int j = 0; j < dim; j++) {
                z[j] = g.nextGaussian();
            }
            for (int j = 0; j < dim; j++) {
                double shift = 0.0;
                int rowOff = j * dim;
                for (int k = 0; k <= j; k++) {
                    shift += L[rowOff + k] * z[k];
                }
                thetaProp[j] = thetaCurr[j] + shift;
            }

            // Check prior first (short-circuit via computeLogPost's own check).
            prior.logPdfBatch(thetaProp, 0, 1, priorOut, 0, priorScratch);

            double lpostProp;
            if (priorOut[0] == Double.NEGATIVE_INFINITY) {
                lpostProp = Double.NEGATIVE_INFINITY;
            } else {
                lpostProp = computeLogPost(thetaProp);
            }

            // Accept/reject
            double logAlpha = lpostProp - lpostCurr;
            if (Math.log(g.nextDouble()) < logAlpha) {
                System.arraycopy(thetaProp, 0, thetaCurr, 0, dim);
                lpostCurr = lpostProp;
                accepted++;
            }

            // Record
            chain.setTheta(iter, thetaCurr, 0);
            chain.arena[chain.LPOST_OFF + iter] = lpostCurr;

            // Adaptive covariance update
            if (adaptive) {
                covTracker.update(thetaCurr, 0);
                double[] trackerL = covTracker.choleskyL();
                for (int i = 0; i < dim * dim; i++) {
                    L[i] = scale * trackerL[i];
                }
            }
        }

        chain.accRate = (double) accepted / niter;
        return chain;
    }
}
