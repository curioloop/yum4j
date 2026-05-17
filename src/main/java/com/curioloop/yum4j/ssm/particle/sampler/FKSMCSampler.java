package com.curioloop.yum4j.ssm.particle.sampler;
import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequence;
import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequenceWF;

import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;
import com.curioloop.yum4j.math.VectorOps;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.random.RandomGenerator;

/**
 * Abstract base class for SMC samplers (IBIS, Tempering) that implements
 * {@link FeynmanKac} and integrates with the new {@code Engine}.
 *
 * <h3>Aliasing: {@code theta.arena} is {@code Workspace.X}</h3>
 *
 * <p>{@link FKSMCSampler} exposes {@link #dim()} as {@code model.dim() + 3}
 * so the engine's {@code Workspace.X} buffer is allocated with the exact
 * layout {@code [theta: dim*N | lpost: N | llik: N | lprior: N]} — the
 * same layout as {@link ThetaParticles}. We therefore {@link
 * ThetaParticles#attachArena alias} the side-car container's
 * {@code arena} directly to the engine's {@code X} (or {@code Xprev})
 * buffer on every entry point, eliminating the former round-trip
 * {@code arraycopy(X → theta.arena)} and {@code arraycopy(theta.arena → X)}.
 *
 * <p>The only genuine allocations that remain in the sampler are the
 * MCMC proposal buffer (inside {@code ArrayMCMC} implementations) and
 * growable metadata (tempering exponents, acceptance rates), all of
 * which are lazily grown and reused across steps.
 *
 * <p>In waste-free mode ({@code lenChain = P > 1} with waste-free sequence),
 * the effective number of particles is {@code N * P} at the Workspace level,
 * but resampling selects only {@code M = N} ancestors which are then expanded
 * back to {@code N * P} via the MCMC sequence.
 */
public abstract class FKSMCSampler implements FeynmanKac<ThetaParticles> {

    protected final StaticModel model;
    protected final MCMCSequence move;
    protected final boolean wasteFree;
    protected final int lenChain;
    protected final double essRmin;

    /**
     * Side-car container whose {@link ThetaParticles#arena} is aliased
     * to the current {@code Workspace.X} (or {@code Xprev}) buffer via
     * {@link ThetaParticles#attachArena}. The container is created
     * lazily on the first call to {@link #init}.
     */
    protected ThetaParticles theta;

    /** Whether resampling fired at the current step (set by logG). */
    protected boolean rsFlag;

    /** Current step index tracked internally. */
    protected int currentT;

    /** The effective N for resampling decisions (M in waste-free mode). */
    protected final int logicalN;

    /** Total particle count (N*P in waste-free, N otherwise). */
    protected final int totalN;

    /** Scratch buffer for log-prior evaluation during MCMC target eval. */
    protected double[] evalScratch;

    /** Uniform-weights buffer for MCMC calibration, grown lazily. */
    protected double[] uniformW;

    protected FKSMCSampler(StaticModel model, MCMCSequence move,
                           int N, int lenChain, double essRmin) {
        this.model = model;
        this.move = move;
        this.lenChain = lenChain;
        this.essRmin = essRmin;
        this.wasteFree = (move instanceof MCMCSequenceWF) && lenChain > 1;
        this.logicalN = N;
        this.totalN = wasteFree ? N * lenChain : N;
        this.currentT = -1;
    }

    /**
     * Returns the effective dimension seen by the SMC engine.
     *
     * <p>This is {@code model.dim() + 3} so that the engine's X buffer
     * includes the three per-particle scalars (lpost, llik, lprior) as
     * extra "rows" in the column-major layout. When the engine gathers
     * columns during resampling, the scalars are automatically propagated
     * alongside the theta values — eliminating the O(t) recomputation
     * that was previously needed after resampling.
     */
    @Override
    public int dim() {
        return model.dim() + 3;
    }

    /**
     * Number of tempering/IBIS/nested-sampling steps this sampler will execute.
     *
     * <p>No longer an {@code @Override}: the {@link FeynmanKac} interface
     * dropped {@code horizon()} in particle-v2-perf R12a.6 / R8.3 because the
     * engine now derives the horizon from the {@code data.size()} passed to
     * {@code Engine.run(..., data)}. For sampler-style Feynman-Kac models that
     * do not consume an external observation stream, callers in
     * {@code ParticleSampling} use this value to build a synthetic
     * {@code List<ThetaParticles>} of the right size (entries are null — the
     * observation is ignored by the sampler kernels).
     *
     * @return number of steps this sampler will execute
     */
    public int horizon() {
        return model.T();
    }

    /**
     * Adaptive termination hook. Subclasses (e.g. AdaptiveTempering)
     * override this to signal early termination.
     */
    public boolean done(RunState rs) {
        return false;
    }

    /**
     * Resampling decision. Returns true when ESS < essRmin * logicalN.
     * Uses the fused reduction to halve passes over {@code lw}.
     */
    protected boolean timeToResample(double[] lw, int N) {
        LogWeight.Triple r = LogWeight.logSumEssMax(lw, 0, N);
        return r.ess() < essRmin * logicalN;
    }

    /**
     * Ensure the side-car theta container exists and its arena is
     * aliased to the given external buffer.
     */
    protected final void aliasTheta(double[] externalArena, int N) {
        int d = model.dim();
        if (theta == null || theta.N != N || theta.dim != d) {
            theta = ThetaParticles.allocate(d, N);
        }
        theta.attachArena(externalArena, 0);
    }

    /** Allocate or grow the uniform-weights buffer. */
    protected final double[] uniformWeights(int N) {
        if (uniformW == null || uniformW.length < N) {
            uniformW = new double[N];
        }
        Arrays.fill(uniformW, 0, N, 1.0 / N);
        return uniformW;
    }

    @Override
    public void init(StepContext<ThetaParticles> ctx) {
        currentT = 0;
        int N = ctx.N();
        double[] X = ctx.X();
        double[] logW = ctx.logW();
        int lwOff = ctx.lwOff();
        RandomGenerator g = ctx.rng().asRandomGenerator();

        // Alias the side-car directly onto the engine's X buffer. No
        // round-trip copy is needed — reads/writes go through theta.arena
        // which IS X.
        aliasTheta(X, N);

        // Sample from prior into theta region of arena (== X).
        model.prior().sample(g, N, theta.arena, 0, null);

        // Evaluate initial log-prior and log-likelihood.
        initializeParticles(N);

        // Initial log-weights are zero (uniform)
        Arrays.fill(logW, lwOff, lwOff + N, 0.0);
    }

    /**
     * Initialize lpost/llik/lprior for the initial particles.
     * Subclasses may override for custom initialization.
     */
    protected void initializeParticles(int N) {
        int d = model.dim();
        int lpriorOff = theta.lpriorOff();
        int lpostOff = theta.lpostOff();
        ensureEvalScratch(d * N);
        // Compute log-prior into lprior region
        model.prior().logPdfBatch(theta.arena, 0, N, theta.arena, lpriorOff, evalScratch);
        System.arraycopy(theta.arena, lpriorOff, theta.arena, lpostOff, N);
    }

    @Override
    public void advance(StepContext<ThetaParticles> ctx) {
        int t = ctx.t();
        currentT = t;
        int N = ctx.N();
        double[] X = ctx.X();
        double[] Xprev = ctx.Xprev();
        double[] logW = ctx.logW();
        int lwOff = ctx.lwOff();
        RandomGenerator g = ctx.rng().asRandomGenerator();

        // Alias theta to the current X buffer. Because the engine's
        // gather already propagated lpost/llik/lprior alongside theta
        // (dim() = d+3), no recomputation is required.
        aliasTheta(X, N);

        // Perform the mutation step (MCMC moves if resampling happened)
        doMutation(t, Xprev, X, N, g);

        // Compute log-weight increment
        doLogG(t, Xprev, X, N, logW, lwOff);
    }

    /**
     * Perform the mutation step. Subclasses define the MCMC move logic.
     * Default implementation applies MCMC moves when resampling fired.
     */
    protected void doMutation(int t, double[] Xprev, double[] X, int N, RandomGenerator g) {
        if (rsFlag) {
            // Apply MCMC moves; target consumer writes into the
            // proposal buffer managed by the ArrayMCMC kernel.
            Consumer<ThetaParticles> target = this::evaluateTarget;
            ThetaParticles moved = move.apply(theta, target, g);

            if (wasteFree && moved.N != theta.N) {
                // Waste-free expanded — the moved buffer lives in the
                // MCMC sequence and is larger than X; the engine is not
                // expected to consume this directly in our current
                // architecture. Fall back to a container swap so that
                // downstream reads see the expanded buffer.
                theta = moved;
            } else if (moved != theta) {
                // Standard: the MCMC sequence wrote into its own buffer.
                // Copy the result back into our aliased arena (== X).
                System.arraycopy(moved.arena, 0, theta.arena, 0, theta.arenaLength());
                theta.copySharedFrom(moved);
            }
            // else: moved is theta itself — nothing to do.
        }
        // else: identity move (no resampling happened).
    }

    /**
     * Compute the log-weight increment. Subclasses must implement this.
     */
    protected abstract void doLogG(int t, double[] Xprev, double[] X, int N,
                                   double[] logW, int lwOff);

    /**
     * Evaluate the target density for the MCMC move.
     * Subclasses define what "target" means at the current step.
     */
    protected abstract void evaluateTarget(ThetaParticles x);

    /** Grow the shared eval scratch buffer if needed. */
    protected final void ensureEvalScratch(int size) {
        if (evalScratch == null || evalScratch.length < size) {
            evalScratch = new double[size];
        }
    }

    /**
     * Compute weighted mean and variance of theta particles into
     * caller-supplied output buffers.
     *
     * @param W    normalised weights (length N)
     * @param x    theta particles
     * @param mean output mean buffer (length ≥ x.dim)
     * @param var  output variance buffer (length ≥ x.dim)
     */
    public static void defaultMoments(double[] W, ThetaParticles x,
                                      double[] mean, double[] var) {
        int N = x.N;
        int d = x.dim;
        double[] arena = x.arena;

        for (int j = 0; j < d; j++) {
            int base = j * N;
            double m = VectorOps.dot(W, 0, arena, base, N);
            mean[j] = m;
            double v = 0.0;
            for (int n = 0; n < N; n++) {
                double diff = arena[base + n] - m;
                v += W[n] * diff * diff;
            }
            var[j] = v;
        }
    }
}
