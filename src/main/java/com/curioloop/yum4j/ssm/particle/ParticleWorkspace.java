package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.diag.Collector;
import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import com.curioloop.yum4j.ssm.particle.smooth.SmoothingWorkspace;

import java.util.List;

/**
 * Reusable scratch container for the {@link Particle} fluent facade.
 *
 * <p>Owns four independent slots, each encapsulating its own buffers,
 * cache key, and lifecycle. Slots are created lazily on first
 * {@code acquireXxx} call and torn down by {@link #close()}; an
 * inference family that never runs leaves its slot {@code null} and
 * pays no allocation. Slots never share state because the underlying
 * inference families operate on different shapes and mixing them
 * would thrash any shared cache:
 *
 * <ul>
 *   <li>{@link FilterSlot} — {@link Workspace} + {@link RunState} for
 *       {@link Particle#filter(com.curioloop.yum4j.ssm.particle.model.ParticleSSM)},
 *       {@link Particle#online(com.curioloop.yum4j.ssm.particle.model.ParticleSSM)},
 *       and the forward filter inside
 *       {@link Particle#smooth(com.curioloop.yum4j.ssm.particle.model.ParticleSSM)}.
 *       Cache key {@code (N, dim, historyMode, historyArg)} for the
 *       workspace and {@code (T, essRmin, scheme, collectors)} for the
 *       run state.</li>
 *   <li>{@link SmoothingSlot} — {@link SmoothingWorkspace} for
 *       {@link Particle#smooth(FilterResult)} backward passes. Cache
 *       key {@code (M, N, dim, T)}.</li>
 *   <li>{@link McmcSlot} — independent {@link Workspace} +
 *       {@link RunState} for the inner particle filter of PMMH /
 *       Particle Gibbs / Conditional SMC. Cache key
 *       {@code (Nx, stateDim, historyMode, T)}; the run state is
 *       always allocated with the canonical mcmc defaults
 *       ({@code essRmin = 0.5}, {@code Scheme.SYSTEMATIC},
 *       no collectors).</li>
 *   <li>{@link SamplerSlot} — independent {@link Workspace} +
 *       {@link RunState} for IBIS / Tempering / AdaptiveTempering.
 *       Sampler FKs always run with {@link HistoryMode#NONE}, so
 *       the cache key reduces to {@code (N, fkDim)}.</li>
 * </ul>
 *
 * <p>SMC² and Nested Sampling have algorithm-specific structures
 * (nested θ × x particle sets and a live-points matrix respectively)
 * that don't fit any of the above slots; they allocate their scratch
 * internally. {@link Particle#sample(SamplingMethod)} for those two
 * methods accepts a workspace but ignores it.
 *
 * <p>{@code AutoCloseable} so callers can use try-with-resources to
 * scope reuse. {@link #close()} drops every slot reference so a
 * long-lived workspace handed to multiple inferences gets its
 * scratch GC'd promptly when the resource block exits.
 *
 * <pre>{@code
 * try (ParticleWorkspace ws = Particle.workspace()) {
 *     for (int i = 0; i < 1000; i++) {
 *         FilterResult<Double> r = Particle.filter(model)
 *             .particles(10_000)
 *             .observations(dataset(i))
 *             .run(ws);
 *         consume(r);
 *     }
 * }
 * }</pre>
 *
 * <p>Not thread-safe: a single workspace must not be used from multiple
 * threads concurrently. For parallel inference, allocate one workspace
 * per worker.
 */
public final class ParticleWorkspace implements AutoCloseable {

    FilterSlot filterSlot;
    SmoothingSlot smoothingSlot;
    McmcSlot mcmcSlot;
    SamplerSlot samplerSlot;

    ParticleWorkspace() {
    }

    // ── Filter slot facade ─────────────────────────────────────────────

    Workspace acquireWorkspace(int N, int dim, HistoryMode mode, int historyArg) {
        if (filterSlot == null) filterSlot = new FilterSlot();
        return filterSlot.acquireWorkspace(N, dim, mode, historyArg);
    }

    RunState acquireRunState(int T, double essRmin, Scheme scheme,
                             List<Collector> collectors, long seed) {
        if (filterSlot == null) filterSlot = new FilterSlot();
        return filterSlot.acquireRunState(T, essRmin, scheme, collectors, seed);
    }

    // ── Smoothing slot facade ──────────────────────────────────────────

    SmoothingWorkspace acquireSmoothingWorkspace(int M, int N, int dim, int T) {
        if (smoothingSlot == null) smoothingSlot = new SmoothingSlot();
        return smoothingSlot.acquire(M, N, dim, T);
    }

    // ── MCMC slot facade ───────────────────────────────────────────────

    Workspace acquireMcmcWorkspace(int Nx, int stateDim, HistoryMode mode, int historyArg) {
        if (mcmcSlot == null) mcmcSlot = new McmcSlot();
        return mcmcSlot.acquireWorkspace(Nx, stateDim, mode, historyArg);
    }

    RunState acquireMcmcRunState(int T, long seed) {
        if (mcmcSlot == null) mcmcSlot = new McmcSlot();
        return mcmcSlot.acquireRunState(T, seed);
    }

    // ── Sampler slot facade ────────────────────────────────────────────

    Workspace acquireSamplerWorkspace(int N, int fkDim) {
        if (samplerSlot == null) samplerSlot = new SamplerSlot();
        return samplerSlot.acquireWorkspace(N, fkDim);
    }

    RunState acquireSamplerRunState(int T, double essRmin, long seed) {
        if (samplerSlot == null) samplerSlot = new SamplerSlot();
        return samplerSlot.acquireRunState(T, essRmin, seed);
    }

    // ── Lifecycle ──────────────────────────────────────────────────────

    /**
     * Releases all retained scratch so a long-lived workspace can shed
     * its memory between unrelated inference batches without being
     * discarded entirely. Drops every slot reference; subsequent
     * {@code acquireXxx} calls will lazily recreate the slot they need.
     * Called automatically by {@link #close()}.
     */
    public void releaseRetainedScratch() {
        filterSlot = null;
        smoothingSlot = null;
        mcmcSlot = null;
        samplerSlot = null;
    }

    @Override
    public void close() {
        releaseRetainedScratch();
    }

    // ───────────────────────────────────────────────────────────────────
    // Slots
    // ───────────────────────────────────────────────────────────────────

    /**
     * Backing storage for the filter facade
     * ({@link Particle#filter}, {@link Particle#online},
     * {@link Particle#smooth} forward pass).
     */
    static final class FilterSlot {
        Workspace workspace;
        RunState runState;

        // Workspace cache key.
        private int wsN = -1;
        private int wsDim = -1;
        private HistoryMode wsHistoryMode;
        private int wsHistoryArg = -1;

        // RunState cache key.
        private int rsCapacity = -1;
        private double rsEssRmin = Double.NaN;
        private Scheme rsScheme;
        private int rsCollectorCount = -1;

        Workspace acquireWorkspace(int N, int dim, HistoryMode mode, int historyArg) {
            if (workspace != null
                    && wsN == N
                    && wsDim == dim
                    && wsHistoryMode == mode
                    && wsHistoryArg == historyArg) {
                workspace.resetBuffers();
                return workspace;
            }
            workspace = Workspace.allocate(N, dim, mode, historyArg);
            wsN = N;
            wsDim = dim;
            wsHistoryMode = mode;
            wsHistoryArg = historyArg;
            return workspace;
        }

        RunState acquireRunState(int T, double essRmin, Scheme scheme,
                                 List<Collector> collectors, long seed) {
            int collectorCount = collectors == null ? 0 : collectors.size();
            if (runState != null
                    && rsCapacity >= T
                    && rsEssRmin == essRmin
                    && rsScheme == scheme
                    && rsCollectorCount == collectorCount) {
                runState.reset(seed);
                return runState;
            }
            runState = RunState.allocate(T, essRmin, scheme, collectors, seed);
            rsCapacity = T;
            rsEssRmin = essRmin;
            rsScheme = scheme;
            rsCollectorCount = collectorCount;
            return runState;
        }
    }

    /**
     * Backing storage for {@link Particle#smooth(FilterResult)} backward
     * passes. The smoother re-uses the same {@link SmoothingWorkspace}
     * across calls of identical shape; mismatches force a reallocation.
     */
    static final class SmoothingSlot {
        SmoothingWorkspace smoothingWorkspace;

        SmoothingWorkspace acquire(int M, int N, int dim, int T) {
            if (smoothingWorkspace == null
                    || smoothingWorkspace.M != M
                    || smoothingWorkspace.N != N
                    || smoothingWorkspace.dim != dim
                    || smoothingWorkspace.T != T) {
                smoothingWorkspace = SmoothingWorkspace.allocate(M, N, dim, T);
            }
            return smoothingWorkspace;
        }
    }

    /**
     * Backing storage for the inner particle filter of PMMH /
     * Particle Gibbs / Conditional SMC. The run state is always
     * allocated with the canonical mcmc defaults
     * ({@code essRmin = 0.5}, {@code Scheme.SYSTEMATIC}, no
     * collectors), so the cache key is just the time horizon.
     */
    static final class McmcSlot {
        Workspace workspace;
        RunState runState;

        private int wsN = -1;
        private int wsDim = -1;
        private HistoryMode wsHistoryMode;
        private int wsHistoryArg = -1;
        private int rsCapacity = -1;

        Workspace acquireWorkspace(int Nx, int stateDim, HistoryMode mode, int historyArg) {
            if (workspace != null
                    && wsN == Nx
                    && wsDim == stateDim
                    && wsHistoryMode == mode
                    && wsHistoryArg == historyArg) {
                workspace.resetBuffers();
                return workspace;
            }
            workspace = Workspace.allocate(Nx, stateDim, mode, historyArg);
            wsN = Nx;
            wsDim = stateDim;
            wsHistoryMode = mode;
            wsHistoryArg = historyArg;
            return workspace;
        }

        RunState acquireRunState(int T, long seed) {
            if (runState != null && rsCapacity >= T) {
                runState.reset(seed);
                return runState;
            }
            runState = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, List.of(), seed);
            rsCapacity = T;
            return runState;
        }
    }

    /**
     * Backing storage for IBIS / Tempering / AdaptiveTempering. Sampler
     * FKs operate on θ-particles without a trajectory history, so the
     * workspace is always {@link HistoryMode#NONE} and the cache key
     * reduces to {@code (N, fkDim)}.
     */
    static final class SamplerSlot {
        Workspace workspace;
        RunState runState;

        private int wsN = -1;
        private int wsDim = -1;
        private int rsCapacity = -1;

        Workspace acquireWorkspace(int N, int fkDim) {
            if (workspace != null
                    && wsN == N
                    && wsDim == fkDim) {
                workspace.resetBuffers();
                return workspace;
            }
            workspace = Workspace.allocate(N, fkDim, HistoryMode.NONE, 0);
            wsN = N;
            wsDim = fkDim;
            return workspace;
        }

        RunState acquireRunState(int T, double essRmin, long seed) {
            if (runState != null && rsCapacity >= T) {
                runState.reset(seed);
                return runState;
            }
            runState = RunState.allocate(T, essRmin, Scheme.SYSTEMATIC, List.of(), seed);
            rsCapacity = T;
            return runState;
        }
    }
}
