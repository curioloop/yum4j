package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import com.curioloop.yum4j.ssm.particle.smooth.FFBS;
import com.curioloop.yum4j.ssm.particle.smooth.ParticleHistory;
import com.curioloop.yum4j.ssm.particle.smooth.SmoothingWorkspace;

import java.util.Arrays;
import java.util.Objects;

/**
 * Typed, immutable result of a completed particle-filter run.
 *
 * <p>The three per-step series ({@code logLikelihoodSeries},
 * {@code essSeries}, {@code resampledFlags}) are exposed through
 * {@code double[]} / {@code boolean[]} accessors that return defensive
 * copies. For O(1) single-element access without allocation, use
 * {@link #logLikelihoodAt(int)}, {@link #essAt(int)}, or
 * {@link #resampledAt(int)}. The final particle snapshot is wrapped in
 * a {@link ParticleDistribution} and never exposes its backing array.
 *
 * <p>The {@link #smooth(SmoothingMode, int)} method dispatches to the
 * appropriate backward-sampling smoother (FFBS, TwoFilter, FixedLag,
 * Paris) and returns a typed {@link TrajectoryBundle}.
 *
 * @param <Y> observation type
 */
public final class FilterResult<Y> {

    private final int N;
    private final int dim;
    private final int T;
    private final double[] logLikelihoodSeries;  // length T
    private final double[] essSeries;            // length T
    private final boolean[] resampledFlags;      // length T
    private final ParticleDistribution<Y> finalDistribution;
    private final ParticleHistory particleHistory; // null when HistoryMode.NONE
    private final TransitionDensity<Y> transitionDensity; // null if model doesn't provide it
    private final RandomBatch rng;

    /**
     * Cached smoothing workspace, lazily allocated on the first
     * {@link #smooth(SmoothingMode, int)} call and reused on subsequent
     * calls with the same shape (R5.2, R5.3). Not thread-safe:
     * {@link FilterResult} is an immutable snapshot of series/flags, but
     * concurrent {@code smooth(...)} calls on the same instance are a
     * programmer error because they would share this scratch buffer.
     */
    private SmoothingWorkspace cachedSmoothingWs;

    /**
     * Constructs a filter result from raw arrays (used internally by
     * {@link ParticleFilter}).
     *
     * <p>All array arguments are defensively copied at construction time.
     * The final particle snapshot is wrapped in a {@link ParticleDistribution}.
     *
     * @param N                    number of particles
     * @param dim                  state dimension per particle
     * @param T                    number of time steps
     * @param logLikelihoodSeries  running log marginal likelihood, length {@code T}
     * @param essSeries            effective sample size per step, length {@code T}
     * @param resampledFlags       whether resampling was triggered per step, length {@code T}
     * @param finalParticles       final-step particle states, length {@code N * dim}
     * @param finalLogWeights      final-step log-weights, length {@code N}
     * @param particleHistory      particle history (null when HistoryMode.NONE)
     */
    public FilterResult(int N, int dim, int T,
                        double[] logLikelihoodSeries,
                        double[] essSeries,
                        boolean[] resampledFlags,
                        double[] finalParticles,
                        double[] finalLogWeights,
                        ParticleHistory particleHistory) {
        this(N, dim, T, logLikelihoodSeries, essSeries, resampledFlags,
             finalParticles, finalLogWeights, particleHistory, null, null);
    }

    /**
     * Constructs a filter result with full smoothing support.
     *
     * <p>All array arguments are defensively copied at construction time.
     * The final particle snapshot is wrapped in a {@link ParticleDistribution}.
     *
     * @param N                    number of particles
     * @param dim                  state dimension per particle
     * @param T                    number of time steps
     * @param logLikelihoodSeries  running log marginal likelihood, length {@code T}
     * @param essSeries            effective sample size per step, length {@code T}
     * @param resampledFlags       whether resampling was triggered per step, length {@code T}
     * @param finalParticles       final-step particle states, length {@code N * dim}
     * @param finalLogWeights      final-step log-weights, length {@code N}
     * @param particleHistory      particle history (null when HistoryMode.NONE)
     * @param transitionDensity    transition density trait (null if unavailable)
     * @param rng                  random batch for smoothing operations (null if unavailable)
     */
    public FilterResult(int N, int dim, int T,
                        double[] logLikelihoodSeries,
                        double[] essSeries,
                        boolean[] resampledFlags,
                        double[] finalParticles,
                        double[] finalLogWeights,
                        ParticleHistory particleHistory,
                        TransitionDensity<Y> transitionDensity,
                        RandomBatch rng) {
        if (N <= 0) throw new IllegalArgumentException("N must be > 0: " + N);
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0: " + dim);
        if (T <= 0) throw new IllegalArgumentException("T must be > 0: " + T);
        Objects.requireNonNull(logLikelihoodSeries, "logLikelihoodSeries must not be null");
        Objects.requireNonNull(essSeries, "essSeries must not be null");
        Objects.requireNonNull(resampledFlags, "resampledFlags must not be null");
        Objects.requireNonNull(finalParticles, "finalParticles must not be null");
        Objects.requireNonNull(finalLogWeights, "finalLogWeights must not be null");
        if (logLikelihoodSeries.length < T) {
            throw new IllegalArgumentException(
                "logLikelihoodSeries.length must be >= T=" + T
                    + ", got " + logLikelihoodSeries.length);
        }
        if (essSeries.length < T) {
            throw new IllegalArgumentException(
                "essSeries.length must be >= T=" + T + ", got " + essSeries.length);
        }
        if (resampledFlags.length < T) {
            throw new IllegalArgumentException(
                "resampledFlags.length must be >= T=" + T
                    + ", got " + resampledFlags.length);
        }
        if (finalParticles.length != N * dim) {
            throw new IllegalArgumentException(
                "finalParticles.length must be N*dim=" + (N * dim)
                    + ", got " + finalParticles.length);
        }
        if (finalLogWeights.length != N) {
            throw new IllegalArgumentException(
                "finalLogWeights.length must be N=" + N
                    + ", got " + finalLogWeights.length);
        }

        this.N = N;
        this.dim = dim;
        this.T = T;
        this.logLikelihoodSeries = Arrays.copyOf(logLikelihoodSeries, T);
        this.essSeries = Arrays.copyOf(essSeries, T);
        this.resampledFlags = Arrays.copyOf(resampledFlags, T);
        this.finalDistribution = new ParticleDistribution<>(N, dim, finalParticles, finalLogWeights);
        this.particleHistory = particleHistory;
        this.transitionDensity = transitionDensity;
        this.rng = rng;
    }

    // ------------------------------------------------------------------
    // Immutable accessors
    // ------------------------------------------------------------------

    /** Number of particles. */
    public int N() { return N; }

    /** State dimension per particle. */
    public int dim() { return dim; }

    /** Number of time steps. */
    public int T() { return T; }

    /**
     * Returns a defensive copy of the running log marginal likelihood
     * series, length {@code T}. Element {@code t} is
     * {@code log p(y_0, ..., y_t)}.
     *
     * <p>For O(1) single-element access without allocating a new array,
     * use {@link #logLikelihoodAt(int)}.
     *
     * @return a new array with the log-likelihood series
     */
    public double[] logLikelihoodSeries() {
        return Arrays.copyOf(logLikelihoodSeries, T);
    }

    /**
     * Returns the running log marginal likelihood at step {@code t}
     * without allocating a copy.
     *
     * @param t time step in {@code [0, T)}
     * @return {@code log p(y_0, ..., y_t)}
     * @throws IndexOutOfBoundsException if {@code t} is out of range
     */
    public double logLikelihoodAt(int t) {
        if (t < 0 || t >= T) {
            throw new IndexOutOfBoundsException("t=" + t + " T=" + T);
        }
        return logLikelihoodSeries[t];
    }

    /**
     * Returns a defensive copy of the effective sample size series,
     * length {@code T}.
     *
     * <p>For O(1) single-element access without allocating a new array,
     * use {@link #essAt(int)}.
     *
     * @return a new array with the ESS series
     */
    public double[] essSeries() {
        return Arrays.copyOf(essSeries, T);
    }

    /**
     * Returns the effective sample size at step {@code t} without
     * allocating a copy.
     *
     * @param t time step in {@code [0, T)}
     * @return ESS at step {@code t}
     * @throws IndexOutOfBoundsException if {@code t} is out of range
     */
    public double essAt(int t) {
        if (t < 0 || t >= T) {
            throw new IndexOutOfBoundsException("t=" + t + " T=" + T);
        }
        return essSeries[t];
    }

    /**
     * Returns a defensive copy of the resampled flags, length {@code T}.
     *
     * <p>For O(1) single-element access without allocating a new array,
     * use {@link #resampledAt(int)}.
     *
     * @return a new array with the resampled flags
     */
    public boolean[] resampledFlags() {
        return Arrays.copyOf(resampledFlags, T);
    }

    /**
     * Returns whether resampling was triggered at step {@code t} without
     * allocating a copy.
     *
     * @param t time step in {@code [0, T)}
     * @return {@code true} if resampling fired at step {@code t}
     * @throws IndexOutOfBoundsException if {@code t} is out of range
     */
    public boolean resampledAt(int t) {
        if (t < 0 || t >= T) {
            throw new IndexOutOfBoundsException("t=" + t + " T=" + T);
        }
        return resampledFlags[t];
    }

    /**
     * Returns the final-step particle distribution (immutable snapshot
     * of the terminal particle cloud and log-weights).
     *
     * @return the final particle distribution
     */
    public ParticleDistribution<Y> finalDistribution() {
        return finalDistribution;
    }

    /**
     * Returns the particle history, or {@code null} if the filter was
     * run with {@link com.curioloop.yum4j.ssm.particle.HistoryMode#NONE}.
     *
     * @return the particle history, or null
     */
    public ParticleHistory particleHistory() {
        return particleHistory;
    }

    /**
     * Returns the final log marginal likelihood estimate
     * {@code log p(y_0, ..., y_{T-1})}.
     *
     * @return final log-likelihood
     */
    public double logLikelihood() {
        return logLikelihoodSeries[T - 1];
    }

    // ------------------------------------------------------------------
    // Smoothing dispatch
    // ------------------------------------------------------------------

    /**
     * Dispatches to the appropriate backward-sampling smoother and returns
     * a typed {@link TrajectoryBundle} containing the smoothed trajectory
     * indices.
     *
     * <p>The smoother is selected by the {@code mode} parameter:
     * <ul>
     *   <li>{@link SmoothingMode#FFBS} — Forward-Filtering Backward-Sampling</li>
     *   <li>{@link SmoothingMode#TWO_FILTER} — Two-filter smoother</li>
     *   <li>{@link SmoothingMode#FIXED_LAG} — Fixed-lag smoother</li>
     *   <li>{@link SmoothingMode#PARIS} — Paris-style online additive smoother</li>
     * </ul>
     *
     * @param mode         smoothing algorithm to use
     * @param trajectories number of smoothed trajectories to produce (M)
     * @return a trajectory bundle with the smoothed indices
     * @throws IllegalStateException    if the required history or transition
     *                                  density is not available
     * @throws IllegalArgumentException if trajectories &lt;= 0
     * @throws UnsupportedOperationException if the mode requires additional
     *         configuration not available from a single forward FilterResult
     */
    public TrajectoryBundle<Y> smooth(SmoothingMode mode, int trajectories) {
        Objects.requireNonNull(mode, "mode must not be null");
        if (trajectories <= 0) {
            throw new IllegalArgumentException("trajectories must be > 0: " + trajectories);
        }

        return switch (mode) {
            case FFBS -> smoothFFBS(trajectories);
            case TWO_FILTER -> smoothTwoFilter(trajectories);
            case FIXED_LAG -> smoothFixedLag(trajectories);
            case PARIS -> smoothParis(trajectories);
        };
    }

    private TrajectoryBundle<Y> smoothFFBS(int M) {
        if (particleHistory == null) {
            throw new IllegalStateException(
                "FFBS smoothing requires particle history (use HistoryMode.FULL)");
        }
        if (transitionDensity == null) {
            throw new IllegalStateException(
                "FFBS smoothing requires a TransitionDensity; "
                    + "ensure the model implements TransitionDensity<Y>");
        }
        if (rng == null) {
            throw new IllegalStateException(
                "FFBS smoothing requires a RandomBatch");
        }

        int[] indices = FFBS.sample(particleHistory, transitionDensity, M, rng, getOrAllocateWorkspace(M));
        return new TrajectoryBundle<>(T, M, indices, particleHistory);
    }

    /**
     * Returns a {@link SmoothingWorkspace} sized for a smoothing call
     * with {@code M} trajectories on this result's {@code (N, dim, T)}.
     * The workspace is cached across calls and reallocated if the cache
     * is too small for the requested shape (R5.2, R5.3).
     */
    private SmoothingWorkspace getOrAllocateWorkspace(int M) {
        if (cachedSmoothingWs == null
                || cachedSmoothingWs.M < M
                || cachedSmoothingWs.N < N
                || cachedSmoothingWs.dim < dim
                || cachedSmoothingWs.T < T) {
            cachedSmoothingWs = SmoothingWorkspace.allocate(M, N, dim, T);
        }
        return cachedSmoothingWs;
    }

    private TrajectoryBundle<Y> smoothTwoFilter(int M) {
        if (particleHistory == null) {
            throw new IllegalStateException(
                "Two-filter smoothing requires particle history "
                    + "(use HistoryMode.FULL)");
        }
        if (transitionDensity == null) {
            throw new IllegalStateException(
                "Two-filter smoothing requires a TransitionDensity; "
                    + "ensure the model implements TransitionDensity<Y>");
        }
        // Two-filter smoothing requires a backward information filter run,
        // which is not available from a single forward FilterResult.
        // This mode is intended to be used via SmoothingRun which manages
        // both forward and backward passes.
        throw new UnsupportedOperationException(
            "Two-filter smoothing requires a backward information filter; "
                + "use SmoothingRun for two-filter smoothing");
    }

    private TrajectoryBundle<Y> smoothFixedLag(int M) {
        if (particleHistory == null) {
            throw new IllegalStateException(
                "Fixed-lag smoothing requires particle history "
                    + "(use HistoryMode.ROLLING)");
        }
        // Fixed-lag smoothing produces per-step estimates rather than full
        // trajectory indices. It is intended to be used via the online
        // facade (Particle.online) which manages the rolling window.
        throw new UnsupportedOperationException(
            "Fixed-lag smoothing produces per-step estimates; "
                + "use Particle.online for fixed-lag smoothing");
    }

    private TrajectoryBundle<Y> smoothParis(int M) {
        if (particleHistory == null) {
            throw new IllegalStateException(
                "Paris smoothing requires particle history "
                    + "(use HistoryMode.FULL)");
        }
        if (transitionDensity == null) {
            throw new IllegalStateException(
                "Paris smoothing requires a TransitionDensity with "
                    + "upperBound; ensure the model implements "
                    + "TransitionDensity<Y>");
        }
        // Paris smoother produces additive functional estimates rather than
        // trajectory indices. For trajectory-based smoothing, use FFBS.
        // Paris is available through SmoothingRun for additive functionals.
        throw new UnsupportedOperationException(
            "Paris smoothing produces additive functional estimates; "
                + "use SmoothingRun for Paris-style smoothing, "
                + "or FFBS for trajectory sampling");
    }
}
