package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.diag.Collector;
import com.curioloop.yum4j.ssm.particle.engine.ParallelStrategy;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import com.curioloop.yum4j.ssm.particle.smooth.FFBS;
import com.curioloop.yum4j.ssm.particle.smooth.FixedLag;
import com.curioloop.yum4j.ssm.particle.smooth.Full;
import com.curioloop.yum4j.ssm.particle.smooth.ParticleHistory;
import com.curioloop.yum4j.ssm.particle.smooth.Paris;
import com.curioloop.yum4j.ssm.particle.smooth.Rolling;
import com.curioloop.yum4j.ssm.particle.smooth.SmoothingWorkspace;
import com.curioloop.yum4j.ssm.particle.smooth.TwoFilter;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Fluent configuration object for a particle smoother, produced by
 * {@link Particle#smooth(FilterResult)} or {@link Particle#smooth(ParticleSSM)}.
 *
 * <p>Two construction paths are supported:
 * <ul>
 *   <li><b>From a completed filter result</b> — the smoother consumes
 *       the result's {@link ParticleHistory} directly. The result must
 *       have been produced with a non-{@code NONE} history mode.</li>
 *   <li><b>From a model</b> — the smoother runs the forward filter
 *       internally with an automatic history mode appropriate to the
 *       selected smoothing algorithm.</li>
 * </ul>
 *
 * <p>{@link #run(ParticleWorkspace)} accepts a reusable workspace that
 * caches the {@link SmoothingWorkspace} between calls of identical shape.
 *
 * @param <Y> observation type
 */
public final class ParticleSmoothing<Y> implements ParticleInference<TrajectoryBundle<Y>, ParticleWorkspace> {

    private final FilterResult<Y> sourceResult;
    private final ParticleSSM<Y> sourceModel;

    // Smoother config
    private SmoothingMode mode = SmoothingMode.FFBS;
    private int trajectories = 10;
    private long seed = 0L;

    // Model-path filter config (only used when sourceModel != null)
    private int particles = -1;
    private List<Y> observations;
    private Scheme scheme = Scheme.SYSTEMATIC;
    private double essRmin = 0.5;
    private ParallelStrategy parallel = ParallelStrategy.SERIAL;
    private long filterSeed = 0L;
    private Weighting weighting = Weighting.BOOTSTRAP;
    private final List<Collector> collectors = new ArrayList<>();
    private int rollingLag = -1;
    /** Forward-filter ancestor-pool budget for model-path smoothing.
     *  {@code 1.0} so smoothers don't surprise users with overflow on
     *  high-noise workloads where every step resamples; advanced users
     *  can override via {@link #maxResampleRate(double)}. */
    private double forwardFilterMaxResampleRate = 1.0;

    // TwoFilter
    private Full infoHistory;
    private TwoFilter.LogGamma logGamma;

    // FixedLag
    private int fixedLagLag = -1;
    private FixedLag.Phi fixedLagPhi;

    // Paris
    private Paris.AdditiveFunction psi;
    private int maxTrials = 0;

    ParticleSmoothing(FilterResult<Y> result) {
        this.sourceResult = result;
        this.sourceModel = null;
    }

    ParticleSmoothing(ParticleSSM<Y> ssm) {
        this.sourceResult = null;
        this.sourceModel = ssm;
    }

    // ── Smoother selection ──────────────────────────────────────────────

    public ParticleSmoothing<Y> mode(SmoothingMode mode) {
        this.mode = Objects.requireNonNull(mode, "mode must not be null");
        return this;
    }

    public ParticleSmoothing<Y> ffbs() {
        return mode(SmoothingMode.FFBS);
    }

    public ParticleSmoothing<Y> twoFilter() {
        return mode(SmoothingMode.TWO_FILTER);
    }

    public ParticleSmoothing<Y> fixedLag() {
        return mode(SmoothingMode.FIXED_LAG);
    }

    public ParticleSmoothing<Y> paris() {
        return mode(SmoothingMode.PARIS);
    }

    public ParticleSmoothing<Y> trajectories(int M) {
        if (M <= 0) {
            throw new IllegalArgumentException("trajectories must be > 0: " + M);
        }
        this.trajectories = M;
        return this;
    }

    public ParticleSmoothing<Y> seed(long seed) {
        this.seed = seed;
        return this;
    }

    // ── Forward-filter config (model path only) ─────────────────────────

    public ParticleSmoothing<Y> particles(int N) {
        this.particles = N;
        return this;
    }

    public ParticleSmoothing<Y> observations(List<Y> data) {
        this.observations = Objects.requireNonNull(data, "observations must not be null");
        return this;
    }

    public ParticleSmoothing<Y> resampling(Scheme s) {
        this.scheme = Objects.requireNonNull(s, "scheme must not be null");
        return this;
    }

    public ParticleSmoothing<Y> essRmin(double r) {
        this.essRmin = r;
        return this;
    }

    public ParticleSmoothing<Y> parallel(ParallelStrategy strategy) {
        this.parallel = strategy != null ? strategy : ParallelStrategy.SERIAL;
        return this;
    }

    public ParticleSmoothing<Y> filterSeed(long seed) {
        this.filterSeed = seed;
        return this;
    }

    public ParticleSmoothing<Y> weighting(Weighting w) {
        this.weighting = Objects.requireNonNull(w, "weighting must not be null");
        return this;
    }

    public ParticleSmoothing<Y> collect(Collector... cs) {
        if (cs != null) {
            for (Collector c : cs) {
                if (c != null) {
                    this.collectors.add(c);
                }
            }
        }
        return this;
    }

    /**
     * Sets the rolling window length, used when {@link #fixedLag()} is
     * the selected mode and the smoother needs to allocate a
     * {@link HistoryMode#ROLLING} history. Ignored for FFBS / TwoFilter
     * / Paris which require {@link HistoryMode#FULL}.
     */
    public ParticleSmoothing<Y> rollingLag(int lag) {
        if (lag <= 0) {
            throw new IllegalArgumentException("rollingLag must be > 0: " + lag);
        }
        this.rollingLag = lag;
        return this;
    }

    /**
     * Overrides the forward-filter's {@code maxResampleRate} budget.
     * Defaults to {@code 1.0} on the model-path smoother (so
     * high-noise workloads do not trip the ancestor-pool overflow).
     * Lower values save memory when the empirical rate is known to be
     * smaller. Must satisfy {@code 0 < rate <= 1}.
     */
    public ParticleSmoothing<Y> maxResampleRate(double rate) {
        if (!(rate > 0.0 && rate <= 1.0)) {
            throw new IllegalArgumentException("maxResampleRate must be in (0, 1]: " + rate);
        }
        this.forwardFilterMaxResampleRate = rate;
        return this;
    }

    // ── TwoFilter ──────────────────────────────────────────────────────

    public ParticleSmoothing<Y> infoHistory(Full infoHistory) {
        this.infoHistory = infoHistory;
        return this;
    }

    public ParticleSmoothing<Y> logGamma(TwoFilter.LogGamma logGamma) {
        this.logGamma = logGamma;
        return this;
    }

    // ── FixedLag ───────────────────────────────────────────────────────

    public ParticleSmoothing<Y> lag(int lag) {
        this.fixedLagLag = lag;
        return this;
    }

    public ParticleSmoothing<Y> fixedLagPhi(FixedLag.Phi phi) {
        this.fixedLagPhi = phi;
        return this;
    }

    // ── Paris ──────────────────────────────────────────────────────────

    public ParticleSmoothing<Y> psi(Paris.AdditiveFunction psi) {
        this.psi = psi;
        return this;
    }

    public ParticleSmoothing<Y> maxTrials(int maxTrials) {
        this.maxTrials = maxTrials;
        return this;
    }

    // ── Run ────────────────────────────────────────────────────────────

    @Override
    public TrajectoryBundle<Y> run(ParticleWorkspace workspace) {
        FilterResult<Y> result = sourceResult != null
                ? sourceResult
                : runForwardFilter(workspace);
        return runSmoother(result, workspace);
    }

    /**
     * Runs Paris-style smoothing and returns the rich
     * {@link Paris.Result} (per-step estimates and final Phi values).
     * Only valid when {@link #mode(SmoothingMode)} is
     * {@link SmoothingMode#PARIS}.
     */
    public Paris.Result runParis(ParticleWorkspace workspace) {
        if (mode != SmoothingMode.PARIS) {
            throw new IllegalStateException(
                    "runParis() requires mode=PARIS; current mode=" + mode);
        }
        FilterResult<Y> result = sourceResult != null
                ? sourceResult
                : runForwardFilter(workspace);
        ParticleHistory history = result.particleHistory();
        if (!(history instanceof Full fullHist)) {
            throw new IllegalStateException(
                    "Paris smoothing requires Full particle history");
        }
        TransitionDensity<?> td = requireTransitionDensity(result);
        Paris.AdditiveFunction fn = psi != null ? psi : defaultPsi();
        RandomBatch rng = RandomBatch.of(seed);
        return Paris.smooth(fullHist, td, fn, rng, maxTrials);
    }

    /** Convenience overload of {@link #runParis(ParticleWorkspace)}. */
    public Paris.Result runParis() {
        return runParis(null);
    }

    // ── Internals ──────────────────────────────────────────────────────

    private FilterResult<Y> runForwardFilter(ParticleWorkspace workspace) {
        if (sourceModel == null) {
            throw new IllegalStateException(
                    "smoother source must be either a FilterResult or a model");
        }
        if (particles <= 0 || observations == null || observations.isEmpty()) {
            throw new IllegalStateException(
                    "model-path smoothing requires .particles(N) and .observations(data)");
        }
        HistoryMode mode = chooseHistoryModeForSmoother();
        int historyArg = (mode == HistoryMode.ROLLING)
                ? (rollingLag > 0 ? rollingLag
                                  : (fixedLagLag > 0 ? fixedLagLag : observations.size()))
                : 0;
        ParticleFiltering<Y> filter = Particle.filter(sourceModel)
                .particles(particles)
                .observations(observations)
                .resampling(scheme)
                .essRmin(essRmin)
                .parallel(parallel)
                .seed(filterSeed)
                .weighting(weighting)
                .history(mode, historyArg)
                .maxResampleRate(forwardFilterMaxResampleRate);
        for (Collector c : collectors) {
            filter.collect(c);
        }
        return filter.run(workspace);
    }

    private HistoryMode chooseHistoryModeForSmoother() {
        return switch (mode) {
            case FFBS, TWO_FILTER, PARIS -> HistoryMode.FULL;
            case FIXED_LAG -> HistoryMode.ROLLING;
        };
    }

    private TrajectoryBundle<Y> runSmoother(FilterResult<Y> result, ParticleWorkspace workspace) {
        RandomBatch rng = RandomBatch.of(seed);
        return switch (mode) {
            case FFBS -> runFFBS(result, rng, workspace);
            case TWO_FILTER -> runTwoFilter(result, rng);
            case FIXED_LAG -> runFixedLag(result);
            case PARIS -> runParisAsBundle(result, rng);
        };
    }

    private TrajectoryBundle<Y> runFFBS(FilterResult<Y> result, RandomBatch rng, ParticleWorkspace workspace) {
        ParticleHistory history = requireHistory(result, "FFBS", HistoryMode.FULL);
        TransitionDensity<?> td = requireTransitionDensity(result);
        SmoothingWorkspace sw = workspace == null
                ? null
                : workspace.acquireSmoothingWorkspace(trajectories, result.N(), result.dim(), result.T());
        int[] indices = (sw != null)
                ? FFBS.sample(history, td, trajectories, rng, sw)
                : FFBS.sample(history, td, trajectories, rng);
        return new TrajectoryBundle<>(result.T(), trajectories, indices, history);
    }

    private TrajectoryBundle<Y> runTwoFilter(FilterResult<Y> result, RandomBatch rng) {
        ParticleHistory history = requireHistory(result, "Two-filter", HistoryMode.FULL);
        if (!(history instanceof Full forward)) {
            throw new IllegalStateException("Two-filter smoothing requires Full particle history");
        }
        if (infoHistory == null) {
            throw new IllegalStateException(
                    "Two-filter smoothing requires an info-filter history; call .infoHistory(...)");
        }
        if (logGamma == null) {
            throw new IllegalStateException(
                    "Two-filter smoothing requires a logGamma function; call .logGamma(...)");
        }
        TransitionDensity<?> td = requireTransitionDensity(result);
        @SuppressWarnings("unchecked")
        TransitionDensity<Y> tdTyped = (TransitionDensity<Y>) td;
        int[] indices = TwoFilter.sample(forward, infoHistory, tdTyped, logGamma, trajectories, rng);
        return new TrajectoryBundle<>(result.T(), trajectories, indices, history);
    }

    private TrajectoryBundle<Y> runFixedLag(FilterResult<Y> result) {
        ParticleHistory history = requireHistory(result, "Fixed-lag", HistoryMode.ROLLING);
        if (!(history instanceof Rolling rolling)) {
            throw new IllegalStateException("Fixed-lag smoothing requires Rolling particle history");
        }
        int effectiveLag = fixedLagLag >= 0 ? fixedLagLag : 1;
        int newest = rolling.newest();
        int oldest = rolling.oldest();
        int T = newest - oldest + 1;
        int M = trajectories;
        int[] indices = new int[T * M];
        for (int t = oldest; t <= newest; t++) {
            int currentLag = Math.min(effectiveLag, newest - t);
            int[] ancestors = FixedLag.traceAncestors(rolling, currentLag);
            for (int m = 0; m < M; m++) {
                indices[(t - oldest) * M + m] = ancestors[m % rolling.N()];
            }
        }
        return new TrajectoryBundle<>(T, M, indices, history);
    }

    private TrajectoryBundle<Y> runParisAsBundle(FilterResult<Y> result, RandomBatch rng) {
        ParticleHistory history = requireHistory(result, "Paris", HistoryMode.FULL);
        if (!(history instanceof Full)) {
            throw new IllegalStateException("Paris smoothing requires Full particle history");
        }
        TransitionDensity<?> td = requireTransitionDensity(result);
        // Paris produces additive functional estimates, not trajectory indices.
        // The bundle path falls back to FFBS for trajectory indices; use
        // {@link #runParis(ParticleWorkspace)} for the rich Paris.Result.
        int[] indices = FFBS.sample(history, td, trajectories, rng);
        return new TrajectoryBundle<>(result.T(), trajectories, indices, history);
    }

    private static ParticleHistory requireHistory(FilterResult<?> result, String label, HistoryMode required) {
        ParticleHistory history = result.particleHistory();
        if (history == null) {
            throw new IllegalStateException(
                    label + " smoothing requires particle history (use HistoryMode." + required + ")");
        }
        return history;
    }

    private static TransitionDensity<?> requireTransitionDensity(FilterResult<?> result) {
        try {
            var field = FilterResult.class.getDeclaredField("transitionDensity");
            field.setAccessible(true);
            Object td = field.get(result);
            if (td instanceof TransitionDensity<?> density) return density;
        } catch (NoSuchFieldException | IllegalAccessException e) {
            // Fall through to throw below.
        }
        throw new IllegalStateException(
                "Smoothing requires a TransitionDensity; ensure the model implements TransitionDensity<Y>");
    }

    private static Paris.AdditiveFunction defaultPsi() {
        return (X, xOff, dim, t) -> X[xOff];
    }
}
