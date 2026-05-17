package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.diag.Collector;
import com.curioloop.yum4j.ssm.particle.engine.Engine;
import com.curioloop.yum4j.ssm.particle.engine.ParallelStrategy;
import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import com.curioloop.yum4j.ssm.particle.smooth.ParticleHistory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Fluent configuration object for a particle-filter run, produced by
 * {@link Particle#filter(ParticleSSM)}.
 *
 * <p>Implements {@link ParticleInference} so the same configuration can
 * be {@link #run() run() } for a fresh workspace or
 * {@link #run(ParticleWorkspace) run(workspace)} for reused scratch.
 *
 * @param <Y> observation type
 */
public final class ParticleFiltering<Y> implements ParticleInference<FilterResult<Y>, ParticleWorkspace> {

    private final ParticleSSM<Y> ssm;

    // Required
    private int particles = -1;
    private List<Y> observations;

    // Optional with defaults
    private Scheme scheme = Scheme.SYSTEMATIC;
    private double essRmin = 0.5;
    private HistoryMode historyMode = HistoryMode.NONE;
    private int historyArg = 0;
    private double maxResampleRate = Double.NaN;  // NaN = use variant default (0.6)
    private ParallelStrategy parallel = ParallelStrategy.SERIAL;
    private long seed = 0L;
    private Weighting weighting = Weighting.BOOTSTRAP;
    private final List<Collector> collectors = new ArrayList<>();

    ParticleFiltering(ParticleSSM<Y> ssm) {
        this.ssm = ssm;
    }

    // ── Required setters ────────────────────────────────────────────────

    public ParticleFiltering<Y> particles(int N) {
        this.particles = N;
        return this;
    }

    public ParticleFiltering<Y> observations(List<Y> data) {
        this.observations = Objects.requireNonNull(data, "observations must not be null");
        return this;
    }

    // ── Optional setters ────────────────────────────────────────────────

    public ParticleFiltering<Y> resampling(Scheme s) {
        this.scheme = Objects.requireNonNull(s, "scheme must not be null");
        return this;
    }

    public ParticleFiltering<Y> essRmin(double r) {
        this.essRmin = r;
        return this;
    }

    /**
     * Sets the history retention mode using the variant's default
     * {@code maxResampleRate} of {@code 0.6}. Override the rate via
     * {@link #maxResampleRate(double)} when the workload's empirical
     * resample rate is higher.
     */
    public ParticleFiltering<Y> history(HistoryMode mode) {
        this.historyMode = Objects.requireNonNull(mode, "historyMode must not be null");
        this.historyArg = 0;
        return this;
    }

    /**
     * Sets the history retention mode with a mode-specific argument
     * (e.g. window length for {@link HistoryMode#ROLLING}).
     */
    public ParticleFiltering<Y> history(HistoryMode mode, int windowOrLag) {
        this.historyMode = Objects.requireNonNull(mode, "historyMode must not be null");
        this.historyArg = windowOrLag;
        return this;
    }

    /**
     * Overrides the {@code maxResampleRate} budget used to size the
     * compressed ancestor pool ({@code particle-v2-mem} R5.5). Default
     * is {@code 0.6}; raise toward {@code 1.0} when the workload's
     * empirical resample rate is high (e.g. high observation noise or
     * a small ESS threshold). Must satisfy {@code 0 < rate <= 1}.
     */
    public ParticleFiltering<Y> maxResampleRate(double rate) {
        if (!(rate > 0.0 && rate <= 1.0)) {
            throw new IllegalArgumentException("maxResampleRate must be in (0, 1]: " + rate);
        }
        this.maxResampleRate = rate;
        return this;
    }

    public ParticleFiltering<Y> parallel(ParallelStrategy strategy) {
        this.parallel = strategy != null ? strategy : ParallelStrategy.SERIAL;
        return this;
    }

    public ParticleFiltering<Y> seed(long seed) {
        this.seed = seed;
        return this;
    }

    public ParticleFiltering<Y> weighting(Weighting w) {
        this.weighting = Objects.requireNonNull(w, "weighting must not be null");
        return this;
    }

    public ParticleFiltering<Y> collect(Collector... cs) {
        if (cs != null) {
            for (Collector c : cs) {
                if (c != null) {
                    this.collectors.add(c);
                }
            }
        }
        return this;
    }

    // ── Convenience surfaces ────────────────────────────────────────────

    /** Bootstrap weighting (proposal = transition kernel). */
    public ParticleFiltering<Y> bootstrap() {
        return weighting(Weighting.BOOTSTRAP);
    }

    /** Guided weighting (user-supplied proposal). */
    public ParticleFiltering<Y> guided() {
        return weighting(Weighting.GUIDED);
    }

    /** Auxiliary particle filter (Pitt–Shephard). */
    public ParticleFiltering<Y> auxiliary() {
        return weighting(Weighting.AUXILIARY);
    }

    /** Retain no per-step trajectory history. */
    public ParticleFiltering<Y> noHistory() {
        return history(HistoryMode.NONE);
    }

    /** Retain the full trajectory history (required for FFBS / TwoFilter / Paris). */
    public ParticleFiltering<Y> fullHistory() {
        return history(HistoryMode.FULL);
    }

    /** Retain a rolling window of the last {@code lag} steps (required for FixedLag). */
    public ParticleFiltering<Y> rollingHistory(int lag) {
        if (lag <= 0) {
            throw new IllegalArgumentException("rolling history lag must be > 0: " + lag);
        }
        return history(HistoryMode.ROLLING, lag);
    }

    /** Retain only ancestor lineages (Paris-style smoothers). */
    public ParticleFiltering<Y> partialHistory() {
        return history(HistoryMode.PARTIAL);
    }

    // ── Run ─────────────────────────────────────────────────────────────

    @Override
    public FilterResult<Y> run(ParticleWorkspace workspace) {
        RunOutput output = runCore(workspace);
        return buildResult(output);
        }

        BorrowedFilterResult<Y> runBorrowedUnsafe(ParticleWorkspace workspace) {
        Objects.requireNonNull(workspace, "workspace");
        RunOutput output = runCore(workspace);
        TransitionDensity<Y> transitionDensity = ssm instanceof TransitionDensity<?>
            ? cast(ssm) : null;
        return new BorrowedFilterResult<>(
            output.N(),
            output.dim(),
            output.T(),
            output.runState().logLtSeries,
            output.runState().essSeries,
            output.runState().resampledFlags,
            output.workspace().X,
            output.workspace().logW,
            output.workspace().history,
            transitionDensity,
            output.workspace().rng);
        }

        private RunOutput runCore(ParticleWorkspace workspace) {
        validate();
        int N = this.particles;
        int T = observations.size();
        int dim = ssm.dim();

        FeynmanKac<Y> fk = createFeynmanKac();

        Workspace ws;
        RunState rs;
        int historyArgEffective = historyArg > 0 ? historyArg : T;
        if (workspace != null) {
            ws = workspace.acquireWorkspace(N, dim, historyMode, historyArgEffective);
            rs = workspace.acquireRunState(T, essRmin, scheme, collectors, seed);
        } else {
            ws = Workspace.allocate(N, dim, historyMode, historyArgEffective);
            rs = RunState.allocate(T, essRmin, scheme, collectors, seed);
        }
        applyMaxResampleRate(ws, N, dim, T, historyArgEffective);
        ws.rng = RandomBatch.of(seed);

        Engine.run(ws, rs, fk, parallel, observations);

        return new RunOutput(ws, rs, N, dim, T);
    }

    /**
     * If the caller explicitly set a non-default {@link #maxResampleRate},
     * replace {@link Workspace#history} with a custom-rate variant
     * sized for the configured rate. This sidesteps the
     * {@link Workspace#allocate} factory's default-rate constructor
     * call while keeping the rest of the workspace unchanged.
     */
    private void applyMaxResampleRate(Workspace ws, int N, int dim, int T, int historyArgEffective) {
        if (Double.isNaN(maxResampleRate) || historyMode == HistoryMode.NONE) {
            return;
        }
        switch (historyMode) {
            case FULL -> ws.history = new com.curioloop.yum4j.ssm.particle.smooth.Full(
                    N, dim, T, maxResampleRate);
            case ROLLING -> ws.history = new com.curioloop.yum4j.ssm.particle.smooth.Rolling(
                    historyArgEffective, N, dim, maxResampleRate);
            case PARTIAL -> ws.history = new com.curioloop.yum4j.ssm.particle.smooth.Partial(
                    N, dim, T, maxResampleRate);
            case NONE -> { /* unreachable */ }
        }
    }

    // ── Internals ──────────────────────────────────────────────────────

    private void validate() {
        if (particles <= 0) {
            throw new IllegalStateException(
                    "particles must be set before calling run(); call .particles(N) with N > 0");
        }
        if (observations == null || observations.isEmpty()) {
            throw new IllegalStateException(
                    "observations must be set before calling run(); call .observations(data) with a non-empty list");
        }
    }

    private FeynmanKac<Y> createFeynmanKac() {
        return switch (weighting) {
            case BOOTSTRAP -> FeynmanKac.bootstrap(ssm);
            case GUIDED -> FeynmanKac.guided(ssm);
            case AUXILIARY -> FeynmanKac.auxiliary(ssm, observations);
        };
    }

        private FilterResult<Y> buildResult(RunOutput output) {
        Workspace ws = output.workspace();
        RunState rs = output.runState();
        int N = output.N();
        int dim = output.dim();
        int T = output.T();
        ParticleHistory history = ws.history;

        TransitionDensity<Y> td = ssm instanceof TransitionDensity<?>
                ? cast(ssm) : null;
        return new FilterResult<>(
                N, dim, T,
            rs.logLtSeries, rs.essSeries, rs.resampledFlags,
            ws.X, ws.logW,
                history,
                td,
                ws.rng);
    }

        private record RunOutput(Workspace workspace,
                                 RunState runState,
                                 int N,
                                 int dim,
                                 int T) {
        }

    @SuppressWarnings("unchecked")
    private TransitionDensity<Y> cast(ParticleSSM<Y> ssm) {
        return (TransitionDensity<Y>) ssm;
    }
}
