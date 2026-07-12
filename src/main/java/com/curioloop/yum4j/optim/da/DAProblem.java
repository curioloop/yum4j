/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.da;

import com.curioloop.yum4j.optim.Bound;
import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;

import java.util.Objects;
import java.util.Random;

/**
 * Fluent configuration builder for dual annealing global optimization.
 *
 * <p>Dual annealing combines generalized simulated annealing with an optional local
 * minimization stage. The global phase uses a distorted Cauchy-Lorentz visiting
 * distribution controlled by {@link #visit(double)}, a generalized Metropolis acceptance
 * law controlled by {@link #accept(double)}, temperature decay, and periodic reannealing.
 * The local phase defaults to L-BFGS-B through {@link LocalSearch#lbfgsb()} and can be
 * replaced with a custom local search or disabled with {@code localSearch(null)}.</p>
 *
 * <h2>Quick Start</h2>
 * <pre>{@code
 * Optimization result = Minimizer.da()
 *     .objective((x, n) -> x[0] * x[0] + x[1] * x[1])
 *     .bounds(Bound.between(-5, 5), Bound.between(-5, 5))
 *     .random(new Random(42))
 *     .solve();
 * }</pre>
 *
 * <h2>Search Phases</h2>
 * <ul>
 *   <li>Annealing proposals wrap around finite bounds, matching SciPy's bounded search
 *       space behavior.</li>
 *   <li>Worse proposals may be accepted according to the generalized acceptance law, which
 *       allows escape from local minima while the temperature is high.</li>
 *   <li>When the temperature drops below {@code initialTemperature * restartTemperatureRatio},
 *       the chain is reinitialized from a random finite point.</li>
 *   <li>Unless the local minimizer is {@code null}, accepted minima can trigger a local
 *       minimizer using the remaining objective-evaluation budget.</li>
 * </ul>
 *
 * <h2>Workspace Reuse</h2>
 * <pre>{@code
 * DAProblem problem = Minimizer.da()
 *     .objective(fn)
 *     .bounds(bounds);
 * DAWorkspace ws = DAProblem.workspace();
 * Optimization r = problem.solve(ws);
 * }</pre>
 *
 * @see Minimizer#da()
 * @see DAWorkspace
 * @see LocalSearch
 */
public final class DAProblem
        extends Minimizer<Univariate.Objective, DAWorkspace, DAProblem> {

    int maxIterations = 1000;
    int maxEvaluations = 10_000_000;
    double initialTemperature = 5230.0;
    double restartTemperatureRatio = 2.0e-5;
    double visit = 2.62;
    double accept = -5.0;
    SearchHook hook;
    LocalSearch localSearch = LocalSearch.lbfgsb();
    private int localSearchMaxIterations = 0;
    private Random rng = new Random();

    public DAProblem() {}

    /** Creates a reusable workspace for repeated solves. */
    public static DAWorkspace workspace() {
        return new DAWorkspace();
    }

    /** Sets the derivative-free objective function. */
    public DAProblem objective(Univariate.Objective objective) {
        this.objective = Objects.requireNonNull(objective, "objective function must not be null");
        return this;
    }

    @Override
    public DAProblem bounds(Bound... bounds) {
        if (bounds == null || bounds.length == 0) {
            throw new IllegalArgumentException("bounds must not be null or empty");
        }
        if (initialPoint != null && bounds.length != initialPoint.length) {
            throw new IllegalArgumentException(
                    "bounds.length=" + bounds.length + " but dimension=" + initialPoint.length);
        }
        this.bounds = bounds;
        this.dimension = bounds.length;
        return this;
    }

    @Override
    public DAProblem initialPoint(double... x0) {
        if (x0 == null || x0.length == 0) {
            throw new IllegalArgumentException("initialPoint must not be null or empty");
        }
        if (bounds != null && bounds.length != x0.length) {
            throw new IllegalArgumentException("initialPoint.length=" + x0.length + " but bounds.length=" + bounds.length);
        }
        for (int i = 0; i < x0.length; i++) {
            if (!Double.isFinite(x0[i])) {
                throw new IllegalArgumentException("initialPoint[" + i + "] is not finite: " + x0[i]);
            }
        }
        this.initialPoint = x0;
        this.dimension = x0.length;
        return this;
    }

    /** Sets the maximum number of global annealing iterations. */
    public DAProblem maxIterations(int value) {
        if (value <= 0) throw new IllegalArgumentException("maxIterations must be positive, got " + value);
        this.maxIterations = value;
        return this;
    }

    /**
     * Sets the hard maximum number of objective evaluations.
     *
     * <p>Local search receives the remaining evaluation budget. If a custom local
     * search reports more evaluations than requested, the reported count is preserved
     * for diagnostics and the solve stops as budget-exhausted.</p>
     */
    public DAProblem maxEvaluations(int value) {
        if (value <= 0) throw new IllegalArgumentException("maxEvaluations must be positive, got " + value);
        this.maxEvaluations = value;
        return this;
    }

    /**
     * Sets the initial artificial temperature.
     *
     * <p>Higher temperatures produce wider visiting jumps and make early exploration more
     * global. The default matches SciPy's {@code initial_temp=5230.0}.</p>
     */
    public DAProblem initialTemperature(double value) {
        if (value <= 0 || !Double.isFinite(value)) {
            throw new IllegalArgumentException("initialTemperature must be positive and finite, got " + value);
        }
        this.initialTemperature = value;
        return this;
    }

    /**
     * Sets the ratio that triggers reannealing.
     *
     * <p>When the current temperature is below {@code initialTemperature * value}, the
     * solver restarts the Markov chain from a fresh finite point while preserving the best
     * solution found so far.</p>
     */
    public DAProblem restartTemperatureRatio(double value) {
        if (value <= 0.0 || value >= 1.0 || Double.isNaN(value)) {
            throw new IllegalArgumentException("restartTemperatureRatio must be in (0, 1), got " + value);
        }
        this.restartTemperatureRatio = value;
        return this;
    }

    /**
     * Sets the visiting-distribution parameter in {@code (1, 3]}.
     *
     * <p>Larger values create heavier tails and therefore more distant proposals. The value
     * {@code 3.0} is accepted for SciPy compatibility and evaluated as the closest lower
     * representable double to avoid the density singularity at exactly three.</p>
     */
    public DAProblem visit(double value) {
        if (value <= 1.0 || value > 3.0 || Double.isNaN(value)) {
            throw new IllegalArgumentException("visit must be in (1, 3], got " + value);
        }
        // The SciPy API accepts 3.0. The visiting-density formula is singular exactly at 3,
        // so the implementation evaluates that endpoint as the closest lower double.
        this.visit = value;
        return this;
    }

    /**
     * Sets the generalized acceptance parameter.
     *
     * <p>Lower values reduce the probability of accepting uphill moves. The accepted range
     * and default follow SciPy's {@code accept} parameter.</p>
     */
    public DAProblem accept(double value) {
        if (value <= -10_000.0 || value > -5.0 || Double.isNaN(value)) {
            throw new IllegalArgumentException("accept must be in (-10000, -5], got " + value);
        }
        this.accept = value;
        return this;
    }

    /** Sets the default evaluation/iteration budget supplied to each local-search run. */
    public DAProblem localSearchMaxIterations(int value) {
        if (value <= 0) throw new IllegalArgumentException("localSearchMaxIterations must be positive, got " + value);
        this.localSearchMaxIterations = value;
        return this;
    }

    /** Sets the local search implementation. Null disables the local minimization phase. */
    public DAProblem localSearch(LocalSearch value) {
        this.localSearch = value;
        return this;
    }

    /** Sets a no-copy hook invoked when a new global best is found. */
    public DAProblem hook(SearchHook value) {
        this.hook = value;
        return this;
    }

    /** Sets the random number generator used for initialization, visiting, and acceptance. */
    public DAProblem random(Random value) {
        this.rng = Objects.requireNonNull(value, "random must not be null");
        return this;
    }

    int localSearchMaxIterations() {
        if (localSearchMaxIterations > 0) return localSearchMaxIterations;
        return Math.clamp(dimension * 6, 100, 1000);
    }

    boolean localSearchEnabled() {
        return localSearch != null;
    }

    @Override
    public Optimization solve(DAWorkspace workspace) {
        validate();
        DAWorkspace ws = resolveWorkspace(workspace, DAWorkspace::new);
        return DACore.optimize(objective, bounds, initialPoint, ws, this, rng);
    }

    private void validate() {
        requireObjective();
        if (bounds == null || bounds.length == 0) {
            throw new IllegalStateException("finite bounds are required. Call .bounds(...) before .solve().");
        }
        dimension = bounds.length;
        if (initialPoint != null && initialPoint.length != dimension) {
            throw new IllegalArgumentException("initialPoint.length=" + initialPoint.length + " but bounds.length=" + dimension);
        }
        for (int i = 0; i < bounds.length; i++) {
            Bound bound = bounds[i];
            if (bound == null || !bound.hasBoth() || !Double.isFinite(bound.lower()) || !Double.isFinite(bound.upper())) {
                throw new IllegalArgumentException("dual annealing requires finite lower and upper bounds at index " + i);
            }
            if (!(bound.lower() < bound.upper())) {
                throw new IllegalArgumentException("bounds[" + i + "] must satisfy lower < upper");
            }
            if (initialPoint != null && (initialPoint[i] < bound.lower() || initialPoint[i] > bound.upper())) {
                throw new IllegalArgumentException("initialPoint[" + i + "] must lie within bounds " + bound);
            }
        }
    }
}