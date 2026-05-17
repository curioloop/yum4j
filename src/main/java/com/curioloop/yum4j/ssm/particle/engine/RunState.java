package com.curioloop.yum4j.ssm.particle.engine;

import com.curioloop.yum4j.ssm.particle.diag.Collector;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;

import java.util.Arrays;
import java.util.List;

/**
 * Per-run mutable state for the particle-filter engine.
 *
 * <p>Owns the time cursor, per-step log-likelihood series, ESS series,
 * resampled flags, and cached scalar summaries from the most recent
 * reduction. Separated from {@link Workspace} so that buffer shape
 * (immutable after allocation) is distinct from run progress (mutable).
 *
 * <p>No {@code _OFF} constants, no offset-encoded arrays. Series are
 * typed fields with direct indexing by time step {@code t}.
 *
 */
public final class RunState {

    // ------------------------------------------------------------------
    // Configuration — final; set at construction.
    // ------------------------------------------------------------------

    /** Total number of time steps (horizon). */
    public final int T;

    /** Resampling scheme used by the engine. */
    public final Scheme scheme;

    /** ESS threshold ratio: resample when {@code essCache < essRmin * N}. */
    public final double essRmin;

    /**
     * Diagnostic collectors dispatched after each engine phase.
     * Typed as {@code Collector[]} — the engine invokes lifecycle
     * methods directly without reflection.
     */
    public final Collector[] collectors;

    // ------------------------------------------------------------------
    // Typed per-step series.
    // ------------------------------------------------------------------

    /**
     * Running log marginal likelihood at each step. Length initially
     * {@code max(1, T)}; grown on demand by {@link #ensureCapacity(int)}
     * for open-ended runs driven by {@link com.curioloop.yum4j.ssm.particle.ParticleOnline}.
     */
    public double[] logLtSeries;

    /**
     * Effective sample size at each step. Same length as
     * {@link #logLtSeries}; grown together with it.
     */
    public double[] essSeries;

    /**
     * Whether resampling was triggered at each step. Same length as
     * {@link #logLtSeries}; grown together with it.
     */
    public boolean[] resampledFlags;

    // ------------------------------------------------------------------
    // Cursor + cached reduction from the last step.
    // ------------------------------------------------------------------

    /** Number of completed steps (0 before init, 1 after init, ...). */
    public int stepCount;

    /** Cached logSumExp from the most recent reduction. */
    public double logSumCache;

    /** Cached ESS from the most recent reduction. */
    public double essCache;

    /** Cached max log-weight from the most recent reduction. */
    public double maxLwCache;

    /** Whether the cached reduction values are valid. */
    public boolean fusedValid;

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    private RunState(int T, double essRmin, Scheme scheme, Collector[] collectors) {
        this.T = T;
        this.essRmin = essRmin;
        this.scheme = scheme;
        this.collectors = collectors;

        int len = Math.max(1, T);
        this.logLtSeries = new double[len];
        this.essSeries = new double[len];
        this.resampledFlags = new boolean[len];

        this.stepCount = 0;
        this.logSumCache = Double.NEGATIVE_INFINITY;
        this.essCache = 0.0;
        this.maxLwCache = Double.NEGATIVE_INFINITY;
        this.fusedValid = false;
    }

    /**
     * Allocates a new {@code RunState} for a filter run of {@code T} steps.
     *
     * @param T          number of time steps (horizon)
     * @param essRmin    ESS threshold ratio in {@code (0, 1]}
     * @param scheme     resampling scheme
     * @param collectors list of diagnostic collectors (may be empty)
     * @param seed       initial RNG seed (reserved for future per-run RNG state)
     * @return a freshly allocated {@code RunState}
     */
    public static RunState allocate(int T, double essRmin, Scheme scheme,
                                    List<Collector> collectors, long seed) {
        Collector[] colArr = collectors != null
            ? collectors.toArray(new Collector[0])
            : new Collector[0];
        return new RunState(T, essRmin, scheme, colArr);
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /**
     * Ensures the per-step series have capacity for at least
     * {@code minCapacity} steps. Preserves existing contents. Called by
     * {@link com.curioloop.yum4j.ssm.particle.ParticleOnline} before
     * each engine dispatch so the series can grow with the observation
     * buffer for unbounded online runs.
     *
     * @param minCapacity the minimum required array length
     */
    public void ensureCapacity(int minCapacity) {
        if (logLtSeries.length < minCapacity) {
            int newCap = Math.max(minCapacity, logLtSeries.length * 2);
            logLtSeries = Arrays.copyOf(logLtSeries, newCap);
            essSeries = Arrays.copyOf(essSeries, newCap);
            resampledFlags = Arrays.copyOf(resampledFlags, newCap);
        }
    }

    /**
     * Resets the cursor and all series for a fresh run with a new seed.
     *
     * @param seed new RNG seed (reserved for future per-run RNG state)
     */
    public void reset(long seed) {
        this.stepCount = 0;
        this.logSumCache = Double.NEGATIVE_INFINITY;
        this.essCache = 0.0;
        this.maxLwCache = Double.NEGATIVE_INFINITY;
        this.fusedValid = false;

        Arrays.fill(logLtSeries, 0.0);
        Arrays.fill(essSeries, 0.0);
        Arrays.fill(resampledFlags, false);
    }

    // ------------------------------------------------------------------
    // Reduction publishing
    // ------------------------------------------------------------------

    /**
     * Stores the reduction results from {@link LogWeight#logSumEssMax} into
     * the per-step series and updates the cached scalars.
     *
     * <p>Called by the engine after each fused reduction pass.
     *
     * @param t time step index
     * @param r the reduction triple (logSum, ess, max)
     */
    public void publishReduction(int t, LogWeight.Triple r) {
        essSeries[t] = r.ess();
        logSumCache = r.logSum();
        essCache = r.ess();
        maxLwCache = r.max();
        fusedValid = true;
    }

    // ------------------------------------------------------------------
    // Resample flags
    // ------------------------------------------------------------------

    /**
     * Marks that resampling was triggered at step {@code t}.
     *
     * @param t time step index
     */
    public void markResampled(int t) {
        resampledFlags[t] = true;
    }

    /**
     * Marks that resampling was NOT triggered at step {@code t}.
     *
     * @param t time step index
     */
    public void markNotResampled(int t) {
        resampledFlags[t] = false;
    }

    // ------------------------------------------------------------------
    // Collector dispatch
    // ------------------------------------------------------------------

    /**
     * Dispatches the {@code afterReweight} event to all registered collectors.
     *
     * @param ws the workspace (passed through to collectors)
     * @param t  current time step
     */
    public void fireCollectors_afterReweight(Workspace ws, int t) {
        for (Collector collector : collectors) {
            collector.afterReweight(ws, this, t);
        }
    }

    /**
     * Dispatches the {@code afterResample} event to all registered collectors.
     *
     * @param ws the workspace (passed through to collectors)
     * @param t  current time step
     */
    public void fireCollectors_afterResample(Workspace ws, int t) {
        for (Collector collector : collectors) {
            collector.afterResample(ws, this, t);
        }
    }

    /**
     * Dispatches the {@code afterMutation} event to all registered collectors.
     *
     * @param ws the workspace (passed through to collectors)
     * @param t  current time step
     */
    public void fireCollectors_afterMutation(Workspace ws, int t) {
        for (Collector collector : collectors) {
            collector.afterMutation(ws, this, t);
        }
    }
}
