package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.diag.Collector;
import com.curioloop.yum4j.ssm.particle.engine.Engine;
import com.curioloop.yum4j.ssm.particle.engine.ParallelStrategy;
import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Fluent builder for a step-by-step online particle filter, produced by
 * {@link Particle#online(ParticleSSM)}.
 *
 * <p>Online filtering processes observations incrementally — one
 * {@link Run#step(Object)} call per observation — while reusing a
 * single {@link Workspace}, {@link RunState}, and {@link FeynmanKac}
 * across steps without per-step reallocation. The resulting
 * {@link Run} is stateful; its accessors ({@link Run#currentDistribution()},
 * {@link Run#logLikelihood()}, {@link Run#currentResult()}) reflect
 * everything observed so far.
 *
 * <p>Mirrors the configuration surface of {@link ParticleFiltering}
 * (resampling scheme, ESS threshold, history mode, parallel strategy,
 * weighting, seed, collectors, ancestor-pool budget). The difference is
 * the run shape: {@code .run(...)} on {@link ParticleFiltering} is
 * one-shot; {@link #start()} / {@link #start(ParticleWorkspace)} on
 * this class returns a long-lived {@link Run}.
 *
 * <pre>{@code
 * try (ParticleWorkspace ws = Particle.workspace()) {
 *     ParticleOnline.Run<Double> run = Particle.online(model)
 *         .particles(10_000)
 *         .seed(42L)
 *         .start(ws);
 *     for (Double obs : stream) {
 *         run.step(obs);
 *         consume(run.currentDistribution(), run.logLikelihood(), run.ess());
 *     }
 *     FilterResult<Double> finalResult = run.currentResult();
 * }
 * }</pre>
 *
 * @param <Y> observation type
 */
public final class ParticleOnline<Y> {

    private final ParticleSSM<Y> ssm;

    private int particles = -1;
    private Scheme scheme = Scheme.SYSTEMATIC;
    private double essRmin = 0.5;
    private HistoryMode historyMode = HistoryMode.NONE;
    private int historyArg = 0;
    private double maxResampleRate = Double.NaN;
    private ParallelStrategy parallel = ParallelStrategy.SERIAL;
    private long seed = 0L;
    private Weighting weighting = Weighting.BOOTSTRAP;
    private final List<Collector> collectors = new ArrayList<>();
    private int initialCapacity = 64;

    ParticleOnline(ParticleSSM<Y> ssm) {
        this.ssm = ssm;
    }

    // ── Required setters ────────────────────────────────────────────────

    public ParticleOnline<Y> particles(int N) {
        this.particles = N;
        return this;
    }

    // ── Optional setters ────────────────────────────────────────────────

    public ParticleOnline<Y> resampling(Scheme s) {
        this.scheme = Objects.requireNonNull(s, "scheme must not be null");
        return this;
    }

    public ParticleOnline<Y> essRmin(double r) {
        this.essRmin = r;
        return this;
    }

    public ParticleOnline<Y> history(HistoryMode mode) {
        this.historyMode = Objects.requireNonNull(mode, "historyMode must not be null");
        this.historyArg = 0;
        return this;
    }

    public ParticleOnline<Y> history(HistoryMode mode, int windowOrLag) {
        this.historyMode = Objects.requireNonNull(mode, "historyMode must not be null");
        this.historyArg = windowOrLag;
        return this;
    }

    /**
     * Overrides the ancestor-pool {@code maxResampleRate} budget.
     * See {@link ParticleFiltering#maxResampleRate(double)}.
     */
    public ParticleOnline<Y> maxResampleRate(double rate) {
        if (!(rate > 0.0 && rate <= 1.0)) {
            throw new IllegalArgumentException("maxResampleRate must be in (0, 1]: " + rate);
        }
        this.maxResampleRate = rate;
        return this;
    }

    public ParticleOnline<Y> parallel(ParallelStrategy strategy) {
        this.parallel = strategy != null ? strategy : ParallelStrategy.SERIAL;
        return this;
    }

    public ParticleOnline<Y> seed(long seed) {
        this.seed = seed;
        return this;
    }

    public ParticleOnline<Y> weighting(Weighting w) {
        this.weighting = Objects.requireNonNull(w, "weighting must not be null");
        return this;
    }

    public ParticleOnline<Y> bootstrap()  { return weighting(Weighting.BOOTSTRAP); }
    public ParticleOnline<Y> guided()     { return weighting(Weighting.GUIDED); }
    public ParticleOnline<Y> auxiliary()  { return weighting(Weighting.AUXILIARY); }

    public ParticleOnline<Y> noHistory()                   { return history(HistoryMode.NONE); }
    public ParticleOnline<Y> fullHistory()                 { return history(HistoryMode.FULL); }
    public ParticleOnline<Y> partialHistory()              { return history(HistoryMode.PARTIAL); }
    public ParticleOnline<Y> rollingHistory(int lag) {
        if (lag <= 0) {
            throw new IllegalArgumentException("rolling history lag must be > 0: " + lag);
        }
        return history(HistoryMode.ROLLING, lag);
    }

    public ParticleOnline<Y> collect(Collector... cs) {
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
     * Sets the initial capacity hint for the {@link RunState} per-step
     * series arrays. The arrays grow on demand via
     * {@link RunState#ensureCapacity(int)}; setting a value close to the
     * expected number of observations avoids early reallocation.
     */
    public ParticleOnline<Y> initialCapacity(int capacity) {
        this.initialCapacity = Math.max(1, capacity);
        return this;
    }

    // ── Build / start ──────────────────────────────────────────────────

    /** Equivalent to {@link #start(ParticleWorkspace) start(null)}. */
    public Run<Y> start() {
        return start(null);
    }

    /**
     * Allocates (or acquires from {@code workspace}) the engine state
     * and returns a {@link Run} ready for {@link Run#step(Object)} calls.
     *
     * @param workspace optional reusable workspace; pass {@code null} to
     *                  allocate a fresh, throw-away workspace
     */
    public Run<Y> start(ParticleWorkspace workspace) {
        if (particles <= 0) {
            throw new IllegalStateException(
                    "particles must be set before start(); call .particles(N) with N > 0");
        }
        int dim = ssm.dim();
        int histArg = historyArg > 0 ? historyArg : initialCapacity;

        Workspace ws;
        RunState rs;
        if (workspace != null) {
            ws = workspace.acquireWorkspace(particles, dim, historyMode, histArg);
            rs = workspace.acquireRunState(initialCapacity, essRmin, scheme, collectors, seed);
        } else {
            ws = Workspace.allocate(particles, dim, historyMode, histArg);
            rs = RunState.allocate(initialCapacity, essRmin, scheme, collectors, seed);
        }
        applyMaxResampleRate(ws, particles, dim, histArg);
        ws.rng = RandomBatch.of(seed);

        // Shared observation buffer: auxiliary FK reads y_{t+1} from it for
        // the one-step lookahead; bootstrap/guided read the current
        // observation via StepContext.observation (written by the engine).
        List<Y> observationBuffer = new ArrayList<>(initialCapacity);
        FeynmanKac<Y> fk = switch (weighting) {
            case BOOTSTRAP -> FeynmanKac.bootstrap(ssm);
            case GUIDED    -> FeynmanKac.guided(ssm);
            case AUXILIARY -> FeynmanKac.auxiliary(ssm, observationBuffer);
        };

        return new Run<>(ssm, ws, rs, fk, observationBuffer, parallel, seed);
    }

    private void applyMaxResampleRate(Workspace ws, int N, int dim, int historyArgEffective) {
        if (Double.isNaN(maxResampleRate) || historyMode == HistoryMode.NONE) {
            return;
        }
        switch (historyMode) {
            case FULL -> ws.history = new com.curioloop.yum4j.ssm.particle.smooth.Full(
                    N, dim, historyArgEffective, maxResampleRate);
            case ROLLING -> ws.history = new com.curioloop.yum4j.ssm.particle.smooth.Rolling(
                    historyArgEffective, N, dim, maxResampleRate);
            case PARTIAL -> ws.history = new com.curioloop.yum4j.ssm.particle.smooth.Partial(
                    N, dim, historyArgEffective, maxResampleRate);
            case NONE -> { /* unreachable */ }
        }
    }

    // ── Stateful run ───────────────────────────────────────────────────

    /**
     * A live online-filtering session. Stateful: every {@link #step(Object)}
     * call advances the engine by one observation, and accessor methods
     * report the cumulative state. Not thread-safe — drive it from a
     * single thread.
     *
     * <p>{@link #reset(long)} reuses all internal buffers for a fresh
     * observation stream. The engine's per-step series grow on demand
     * through {@link RunState#ensureCapacity(int)} so the run can absorb
     * an open-ended observation feed.
     *
     * @param <Y> observation type
     */
    public static final class Run<Y> {

        private final ParticleSSM<Y> ssm;
        private final Workspace ws;
        private final RunState rs;
        private final FeynmanKac<Y> fk;
        private final List<Y> observationBuffer;
        private final ParallelStrategy parallel;
        private final long initialSeed;

        private boolean initialized;

        Run(ParticleSSM<Y> ssm, Workspace ws, RunState rs, FeynmanKac<Y> fk,
            List<Y> observationBuffer, ParallelStrategy parallel, long initialSeed) {
            this.ssm = ssm;
            this.ws = ws;
            this.rs = rs;
            this.fk = fk;
            this.observationBuffer = observationBuffer;
            this.parallel = parallel;
            this.initialSeed = initialSeed;
            this.initialized = false;
        }

        /**
         * Processes a single observation, advancing the filter by one
         * time step. The first call initialises particles at time 0;
         * subsequent calls advance the filter.
         *
         * @throws NullPointerException if {@code observation} is null
         */
        public void step(Y observation) {
            Objects.requireNonNull(observation, "observation must not be null");
            observationBuffer.add(observation);
            rs.ensureCapacity(observationBuffer.size());

            if (!initialized) {
                Engine.init(ws, rs, fk, parallel, observationBuffer);
                initialized = true;
            } else {
                Engine.step(ws, rs, fk, parallel, observationBuffer);
            }
        }

        /** Defensive snapshot of the current particle distribution. */
        public ParticleDistribution<Y> currentDistribution() {
            requireInitialized();
            return new ParticleDistribution<>(ws.N, ws.dim,
                    Arrays.copyOf(ws.X, ws.N * ws.dim),
                    Arrays.copyOf(ws.logW, ws.N));
        }

        /**
         * Returns the current log marginal likelihood
         * {@code log p(y_0, ..., y_{t})}.
         */
        public double logLikelihood() {
            requireInitialized();
            return rs.logLtSeries[rs.stepCount - 1];
        }

        /** Returns the current effective sample size. */
        public double ess() {
            requireInitialized();
            return rs.essSeries[rs.stepCount - 1];
        }

        /** Number of observations processed so far. */
        public int stepCount() {
            return rs.stepCount;
        }

        /**
         * Returns a {@link FilterResult} snapshot covering everything
         * observed so far. Series arrays are defensively copied; the
         * snapshot is decoupled from subsequent {@link #step(Object)}
         * calls.
         */
        public FilterResult<Y> currentResult() {
            requireInitialized();
            int T = rs.stepCount;
            TransitionDensity<Y> td = ssm.transitionDensity();
            return new FilterResult<>(
                    ws.N, ws.dim, T,
                    Arrays.copyOf(rs.logLtSeries, T),
                    Arrays.copyOf(rs.essSeries, T),
                    Arrays.copyOf(rs.resampledFlags, T),
                    Arrays.copyOf(ws.X, ws.N * ws.dim),
                    Arrays.copyOf(ws.logW, ws.N),
                    ws.history,
                    td,
                    ws.rng);
        }

        /**
         * Resets the run to the initial uninitialised state and reseeds
         * the RNG. Internal buffers are reused without reallocation.
         */
        public void reset(long seed) {
            observationBuffer.clear();
            rs.reset(seed);
            initialized = false;
            ws.resetBuffers();
            ws.rng = RandomBatch.of(seed);
        }

        /** Resets with the seed originally configured on the builder. */
        public void reset() {
            reset(initialSeed);
        }

        private void requireInitialized() {
            if (!initialized) {
                throw new IllegalStateException(
                        "No observations processed yet; call step(obs) first");
            }
        }
    }
}
