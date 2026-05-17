package com.curioloop.yum4j.ssm.particle.engine;

import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.resample.Resample;

import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;

/**
 * Static engine driving one SMC (Sequential Monte Carlo) run from
 * {@code t = 0} through {@code t = T - 1}.
 *
 * <p>All methods are static; the class is a utility class with a private
 * constructor. The engine consumes a single {@link FeynmanKac} instance,
 * a pre-allocated {@link Workspace}, a {@link RunState} cursor, a
 * {@link ParallelStrategy} that governs slab partitioning, and the
 * observation sequence {@code List<Y> data} threaded explicitly into
 * every entry point. The horizon is derived from {@code data.size()}
 * (R8.3, R12.1).
 *
 * <p>The engine performs a fused advance: a single pass over the
 * {@code (N, dim)} particle buffer per time step. The reduction
 * (logSumExp + ESS + max) is computed once per step via
 * {@link LogWeight#logSumEssMax}. Resampling is triggered when the
 * cached ESS falls below {@code essRmin * N}.
 *
 * <p>Before dispatching {@code fk.init} / {@code fk.advance}, the engine
 * writes {@code ctx.observation = data.get(t)} so that user kernels read
 * the current observation via {@link StepContext#observation()}.
 *
 * @see FeynmanKac
 * @see Workspace
 * @see RunState
 * @see ParallelStrategy
 */
public final class Engine {

    private Engine() {}

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Initialises particles at time 0. After return, {@code rs.stepCount == 1}.
     *
     * <p>Calls {@code fk.init(ctx)} once over the full particle range
     * (or per slab under striped parallelism), computes the initial
     * reduction, fires collectors, and advances the cursor.
     *
     * @param ws   the pre-allocated workspace
     * @param rs   the run state (must have {@code stepCount == 0})
     * @param fk   the FeynmanKac instance
     * @param par  the parallel strategy
     * @param data the observation sequence; must be non-empty
     * @param <Y>  observation type
     */
    public static <Y> void init(Workspace ws, RunState rs, FeynmanKac<Y> fk,
                                ParallelStrategy par, List<Y> data) {
        if (ws.rng == null) {
            throw new IllegalStateException(
                    "Workspace.rng must be set before Engine.init " +
                    "(call ws.rng = RandomBatch.of(seed))");
        }
        assert rs.stepCount == 0 : "init requires rs.stepCount == 0, got " + rs.stepCount;

        // Attach collectors before the first step
        for (var collector : rs.collectors) {
            collector.attach(ws, rs);
        }

        Y obs = data.get(0);
        par.forRange(ws.N, (off, count) -> fk.init(makeCtx(ws, par, 0, off, count, obs)));

        // Fused reduction over initial log-weights
        LogWeight.Triple r = LogWeight.logSumEssMax(ws.logW, 0, ws.N);
        rs.publishReduction(0, r);

        // Initial logLt = logSumExp - log(N) (normalised)
        rs.logLtSeries[0] = r.logSum() - Math.log(ws.N);

        // Fire collectors
        rs.fireCollectors_afterReweight(ws, 0);
        rs.fireCollectors_afterMutation(ws, 0);

        // History save
        if (ws.history != null) {
            ws.history.save(0, ws, false);
        }

        rs.stepCount = 1;
    }

    /**
     * Advances the filter by one time step. Returns {@code true} if more
     * steps remain, {@code false} if the horizon (derived from {@code data.size()})
     * has been reached.
     *
     * <p>The step consists of:
     * <ol>
     *   <li>Resample decision based on ESS threshold</li>
     *   <li>Fused advance: single pass over {@code (N, dim)}</li>
     *   <li>Fused reduction over updated log-weights</li>
     *   <li>Log-likelihood accumulation and collector dispatch</li>
     * </ol>
     *
     * @param ws   the workspace
     * @param rs   the run state
     * @param fk   the FeynmanKac instance
     * @param par  the parallel strategy
     * @param data the observation sequence
     * @param <Y>  observation type
     * @return {@code true} if more steps remain after this one
     */
    public static <Y> boolean step(Workspace ws, RunState rs, FeynmanKac<Y> fk,
                                   ParallelStrategy par, List<Y> data) {
        int t = rs.stepCount;
        int horizon = data.size();
        if (t >= horizon) {
            return false;
        }

        double logSumBefore;
        boolean resampled = rs.essCache < rs.essRmin * ws.N;

        // (a) Resample decision — reuse cached ESS from previous step
        if (resampled) {
            // Resample: apply the resampling scheme
            Resample.apply(rs.scheme, ws.logW, ws.N, ws.N, ws.a, ws.rng, ws.scratch);

            // Gather particles by ancestor: Xprev[n] = X[a[n]]
            gather(ws.X, ws.Xprev, ws.a, ws.dim, ws.N);

            // Reset log-weights to uniform
            Arrays.fill(ws.logW, 0, ws.N, 0.0);

            rs.markResampled(t);
            logSumBefore = Math.log(ws.N);

            rs.fireCollectors_afterResample(ws, t);
        } else {
            // No resampling: identity ancestors, swap buffers
            System.arraycopy(ws.identity, 0, ws.a, 0, ws.N);
            ws.swapXandXprev();
            logSumBefore = rs.logSumCache;
            rs.markNotResampled(t);
        }

        // (b) Fused advance: write X and logW in one pass
        Y obs = data.get(t);
        par.forRange(ws.N, (off, count) -> fk.advance(makeCtx(ws, par, t, off, count, obs)));

        // (c) Fused reduction over logW
        LogWeight.Triple r = LogWeight.logSumEssMax(ws.logW, 0, ws.N);
        rs.publishReduction(t, r);

        // Accumulate log marginal likelihood
        rs.logLtSeries[t] = rs.logLtSeries[t - 1] + (r.logSum() - logSumBefore);

        // Fire collectors
        rs.fireCollectors_afterReweight(ws, t);
        rs.fireCollectors_afterMutation(ws, t);

        // History save
        if (ws.history != null) {
            ws.history.save(t, ws, resampled);
        }

        rs.stepCount = t + 1;
        return rs.stepCount < horizon;
    }

    /**
     * Runs the full filter from initialisation through all time steps.
     *
     * <p>Equivalent to calling {@link #init} followed by repeated
     * {@link #step} until the horizon (derived from {@code data.size()})
     * is reached.
     *
     * @param ws   the workspace
     * @param rs   the run state (must have {@code stepCount == 0})
     * @param fk   the FeynmanKac instance
     * @param par  the parallel strategy
     * @param data the observation sequence
     * @param <Y>  observation type
     */
    public static <Y> void run(Workspace ws, RunState rs, FeynmanKac<Y> fk,
                               ParallelStrategy par, List<Y> data) {
        init(ws, rs, fk, par, data);
        while (step(ws, rs, fk, par, data)) {
            // continue
        }
    }

    /**
     * Runs the filter until the user-supplied predicate returns {@code true}
     * or the horizon (derived from {@code data.size()}) is reached.
     *
     * <p>The predicate is evaluated after each step. If it returns
     * {@code true}, the run terminates early.
     *
     * @param ws   the workspace
     * @param rs   the run state (must have {@code stepCount == 0})
     * @param fk   the FeynmanKac instance
     * @param par  the parallel strategy
     * @param data the observation sequence
     * @param done predicate evaluated after each step; returns {@code true} to stop
     * @param <Y>  observation type
     */
    public static <Y> void runUntilDone(Workspace ws, RunState rs, FeynmanKac<Y> fk,
                                        ParallelStrategy par, List<Y> data,
                                        Predicate<RunState> done) {
        init(ws, rs, fk, par, data);
        while (step(ws, rs, fk, par, data)) {
            if (done.test(rs)) {
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /**
     * Builds a {@link StepContext} for a contiguous slab of particles.
     *
     * <p>Does not allocate on the steady-state hot path. The returned
     * {@link StepContext} is a pooled slot owned by {@link Workspace#slots}
     * whose fields are overwritten in place before return.
     *
     * <p>When the parallel strategy is {@link ParallelStrategy.Striped},
     * each slab receives a deterministic per-slab RNG derived via
     * {@code ws.rng.split(slabIndex)} and cached in
     * {@link Workspace#slabRngs}; subsequent dispatches to the same slab
     * reuse the cached {@link com.curioloop.yum4j.ssm.particle.kernel.RandomBatch}
     * instance. When the strategy is {@link ParallelStrategy.Serial},
     * the master RNG is used directly (no split) to preserve backward
     * compatibility with existing baselines.
     *
     * @param ws    the workspace
     * @param par   the parallel strategy (determines RNG derivation)
     * @param t     current time step
     * @param off   particle offset within the slab
     * @param count number of particles in the slab
     * @param obs   the observation at time {@code t}, written into the context
     * @param <Y>   observation type
     * @return the pooled StepContext for the slab, with fields overwritten
     */
    @SuppressWarnings("unchecked")
    private static <Y> StepContext<Y> makeCtx(Workspace ws,
                                              ParallelStrategy par,
                                              int t, int off, int count, Y obs) {
        int slabIndex;
        int chunks;
        RandomBatch slabRng;
        if (par instanceof ParallelStrategy.Striped striped) {
            chunks = striped.chunks();
            // Compute slab index from offset: slabIndex = off * chunks / N
            // Uses long arithmetic to avoid overflow.
            slabIndex = (int) ((long) off * chunks / ws.N);
            ws.ensureSlots(chunks);
            ws.ensureSlabRngs(chunks);
            slabRng = ws.slabRngs[slabIndex];
            if (slabRng == null) {
                slabRng = ws.rng.split(slabIndex);
                ws.slabRngs[slabIndex] = slabRng;
            }
        } else {
            chunks = 1;
            slabIndex = 0;
            ws.ensureSlots(1);
            slabRng = ws.rng;
        }

        StepContext<Y> slot = (StepContext<Y>) ws.slots[slabIndex];
        slot.t = t;
        slot.observation = obs;
        slot.Xprev = ws.Xprev;
        slot.xpOff = off * ws.dim;
        slot.X = ws.X;
        slot.xOff = off * ws.dim;
        slot.logW = ws.logW;
        slot.lwOff = off;
        slot.N = count;
        slot.dim = ws.dim;
        slot.rng = slabRng;
        slot.scratch = ws.scratch;
        slot.scratchOff = off;
        return slot;
    }

    /**
     * Gathers particles from {@code src} into {@code dst} according to
     * ancestor indices: for each {@code n}, copies
     * {@code src[a[n]*dim .. (a[n]+1)*dim)} into
     * {@code dst[n*dim .. (n+1)*dim)}.
     *
     * <p>This is the "scatter-by-ancestor" operation that reorders
     * particles after resampling. The source and destination must be
     * distinct arrays (no in-place gather).
     *
     * @param src source particle buffer (current X before resampling)
     * @param dst destination buffer (becomes Xprev for next advance)
     * @param a   ancestor indices, length N
     * @param dim state dimension per particle
     * @param N   number of particles
     */
    private static void gather(double[] src, double[] dst, int[] a, int dim, int N) {
        if (dim == 1) {
            // Fast path for scalar state (most common case)
            for (int n = 0; n < N; n++) {
                dst[n] = src[a[n]];
            }
        } else {
            for (int n = 0; n < N; n++) {
                int srcOff = a[n] * dim;
                int dstOff = n * dim;
                System.arraycopy(src, srcOff, dst, dstOff, dim);
            }
        }
    }
}
