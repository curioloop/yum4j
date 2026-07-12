/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.de;

import com.curioloop.yum4j.optim.Bound;
import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;

import java.util.Objects;
import java.util.Random;

/**
 * Fluent configuration builder for differential evolution.
 *
 * <p>Differential evolution is a derivative-free, population-based global optimizer for
 * finite box-bounded continuous problems. Each generation builds a trial vector for every
 * population member by combining scaled population differences, crossover, bounds repair,
 * and greedy selection. The best member is kept at population slot 0 throughout the solve.</p>
 *
 * <p>This implementation follows SciPy's {@code differential_evolution} route: Storn-Price
 * mutation strategies, binomial/exponential crossover variants, optional deferred updating
 * for batch evaluation, Lampinen-style constrained selection, integrality projection, and a
 * final local-polish phase through {@link FinalPolisher}. By default the built-in polisher
 * uses L-BFGS-B for bound-only problems and SLSQP for constrained problems.</p>
 *
 * <h2>Quick Start</h2>
 * <pre>{@code
 * Optimization result = Minimizer.de()
 *     .objective((x, n) -> x[0] * x[0] + x[1] * x[1])
 *     .bounds(Bound.between(-5, 5), Bound.between(-5, 5))
 *     .random(new Random(42))
 *     .solve();
 * }</pre>
 *
 * <h2>Strategies</h2>
 * <p>The named strategies in {@link EvolutionStrategy} mirror SciPy's built-in
 * variants such as {@code best1bin}, {@code rand1exp}, and {@code currenttobest1bin}.
 * Alternatively, {@link #strategy(TrialStrategy)} accepts a custom
 * trial-vector generator that writes directly into solver-owned storage.</p>
 *
 * <h2>Updating and Evaluation</h2>
 * <ul>
 *   <li>Immediate updating is the default: improvements can affect later candidates in the
 *       same generation.</li>
 *   <li>Deferred updating evaluates a full trial population before greedy replacement and is
 *       required when a custom {@link DEEvaluator} is configured.</li>
 * </ul>
 *
 * <h2>Constraint Handling</h2>
 * <p>Bounds are mandatory and finite. Additional inequality constraints use the convention
 * {@code c(x) >= 0}; equality constraints use {@code h(x) = 0} with
 * {@link #equalityTolerance(double)}. Feasible trials dominate infeasible ones; if both are
 * infeasible, component-wise violation ordering decides acceptance.</p>
 *
 * <h2>Workspace Reuse</h2>
 * <pre>{@code
 * DEProblem problem = Minimizer.de()
 *     .objective(fn)
 *     .bounds(bounds);
 * DEWorkspace ws = DEProblem.workspace();
 * for (Random rng : rngs) {
 *     Optimization r = problem.random(rng).solve(ws);
 * }
 * }</pre>
 *
 * @see Minimizer#de()
 * @see EvolutionStrategy
 * @see PopulationInit
 * @see DEWorkspace
 */
public final class DEProblem
        extends Minimizer<Univariate.Objective, DEWorkspace, DEProblem> {

    EvolutionStrategy strategy = EvolutionStrategy.BEST1BIN;
    boolean deferredUpdating;
    PopulationInit initialization = PopulationInit.LATIN_HYPERCUBE;
    int maxIterations = 1000;
    int maxEvaluations = 0;
    int populationMultiplier = 15;
    int populationSize = 0;
    DEEvaluator evaluator = DEEvaluator.serial();
    boolean forceDeferred;
    double tolerance = 0.01;
    double absoluteTolerance = 0.0;
    Bound mutation = Bound.between(0.5, 1.0);
    double recombination = 0.7;
    FinalPolisher polisher = FinalPolisher.auto();
    double[][] initialPopulation;
    boolean[] integrality;
    Univariate.Objective[] equalityConstraints;
    Univariate.Objective[] inequalityConstraints;
    double equalityTolerance = 1.0e-8;
    TrialStrategy trialStrategy;
    GenerationHook hook;
    private int polishMaxEvaluations = 0;
    private Random rng = new Random();

    public DEProblem() {}

    /** Creates a reusable workspace for repeated solves. */
    public static DEWorkspace workspace() {
        return new DEWorkspace();
    }

    /** Sets the derivative-free objective function. */
    public DEProblem objective(Univariate.Objective objective) {
        this.objective = Objects.requireNonNull(objective, "objective function must not be null");
        return this;
    }

    @Override
    public DEProblem bounds(Bound... bounds) {
        if (bounds == null || bounds.length == 0) {
            throw new IllegalArgumentException("bounds must not be null or empty");
        }
        if (initialPoint != null && bounds.length != initialPoint.length) {
            throw new IllegalArgumentException(
                    "bounds.length=" + bounds.length + " but dimension=" + initialPoint.length);
        }
        if (initialPopulation != null && bounds.length != initialPopulation[0].length) {
            throw new IllegalArgumentException(
                "bounds.length=" + bounds.length + " but initialPopulation dimension=" + initialPopulation[0].length);
        }
        this.bounds = bounds;
        this.dimension = bounds.length;
        return this;
    }

    @Override
    public DEProblem initialPoint(double... x0) {
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

    /**
     * Sets the named built-in mutation/crossover strategy.
     *
     * <p>Calling this method clears any custom {@link TrialStrategy}.
     * The default is {@link EvolutionStrategy#BEST1BIN}.</p>
     */
    public DEProblem strategy(EvolutionStrategy value) {
        this.strategy = Objects.requireNonNull(value, "strategy must not be null");
        this.trialStrategy = null;
        return this;
    }

    /**
     * Sets a custom strategy that writes complete trial vectors in real parameter space.
     *
     * <p>The solver still performs bounds repair, integrality projection, objective evaluation,
     * and greedy selection. The custom strategy receives the live flat population and must fill
     * every coordinate of the requested trial vector with finite values.</p>
     */
    public DEProblem strategy(TrialStrategy value) {
        this.trialStrategy = Objects.requireNonNull(value, "trial strategy must not be null");
        return this;
    }

    /**
     * Sets whether best population member updates are deferred until the generation ends.
     *
     * <p>Deferred updating is more memory-intensive because it stages a full trial population,
     * but it enables batch and parallel evaluation. Immediate updating is usually faster for
     * cheap serial objectives because later candidates can use improvements found earlier in
     * the same generation.</p>
     */
    public DEProblem deferredUpdating(boolean value) {
        this.deferredUpdating = value;
        return this;
    }

    /**
     * Sets the population initialization method.
     *
     * <p>{@link PopulationInit#SOBOL} supports up to 10 dimensions and rounds
     * the effective population size up to the next power of two. {@link PopulationInit#HALTON}
     * supports up to 20 dimensions and keeps the requested population size.</p>
     */
    public DEProblem initialization(PopulationInit value) {
        this.initialization = Objects.requireNonNull(value, "initialization must not be null");
        return this;
    }

    /**
     * Sets an explicit initial population. Its row count becomes the effective population size.
     * Values outside finite bounds are clipped before evaluation.
     */
    public DEProblem initialPopulation(double[][] population) {
        if (population == null || population.length == 0) {
            throw new IllegalArgumentException("initialPopulation must not be null or empty");
        }
        if (population.length < 6) {
            throw new IllegalArgumentException("initialPopulation must contain at least 6 members");
        }
        if (population[0] == null || population[0].length == 0) {
            throw new IllegalArgumentException("initialPopulation rows must not be null or empty");
        }
        int rowDimension = population[0].length;
        if (bounds != null && bounds.length != rowDimension) {
            throw new IllegalArgumentException("initialPopulation dimension=" + rowDimension + " but bounds.length=" + bounds.length);
        }
        if (initialPoint != null && initialPoint.length != rowDimension) {
            throw new IllegalArgumentException("initialPopulation dimension=" + rowDimension + " but initialPoint.length=" + initialPoint.length);
        }
        this.initialPopulation = new double[population.length][rowDimension];
        for (int i = 0; i < population.length; i++) {
            if (population[i] == null || population[i].length != rowDimension) {
                throw new IllegalArgumentException("initialPopulation must be rectangular");
            }
            for (int j = 0; j < rowDimension; j++) {
                double value = population[i][j];
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException("initialPopulation[" + i + "][" + j + "] is not finite: " + value);
                }
                this.initialPopulation[i][j] = value;
            }
        }
        this.dimension = rowDimension;
        return this;
    }

    /** Marks variables that are constrained to integer values. */
    public DEProblem integrality(boolean... value) {
        if (value == null || value.length == 0) {
            throw new IllegalArgumentException("integrality must not be null or empty");
        }
        if (bounds != null && bounds.length != value.length) {
            throw new IllegalArgumentException("integrality.length=" + value.length + " but bounds.length=" + bounds.length);
        }
        if (initialPoint != null && initialPoint.length != value.length) {
            throw new IllegalArgumentException("integrality.length=" + value.length + " but initialPoint.length=" + initialPoint.length);
        }
        if (initialPopulation != null && initialPopulation[0].length != value.length) {
            throw new IllegalArgumentException("integrality.length=" + value.length + " but initialPopulation dimension=" + initialPopulation[0].length);
        }
        this.integrality = value.clone();
        this.dimension = Math.max(this.dimension, value.length);
        return this;
    }

    /**
     * Sets equality constraints {@code h(x) = 0}.
     *
     * <p>Constraint evaluations are not counted as objective evaluations. In constrained
     * selection, equality violation components are ordered after inequality components.</p>
     */
    @SafeVarargs
    public final DEProblem equalityConstraints(Univariate.Objective... constraints) {
        this.equalityConstraints = cloneConstraints("equalityConstraints", constraints);
        return this;
    }

    /**
     * Sets inequality constraints {@code c(x) >= 0}.
     *
     * <p>Constraint evaluations are not counted as objective evaluations. In constrained
     * selection, inequality violation components are ordered before equality components.</p>
     */
    @SafeVarargs
    public final DEProblem inequalityConstraints(Univariate.Objective... constraints) {
        this.inequalityConstraints = cloneConstraints("inequalityConstraints", constraints);
        return this;
    }

    /**
     * Sets the absolute tolerance used to convert equality constraints into violation components.
     *
     * <p>The equality component for {@code h(x)} is {@code max(0, abs(h(x)) - tolerance)}.</p>
     */
    public DEProblem equalityTolerance(double value) {
        if (value < 0.0 || Double.isNaN(value)) {
            throw new IllegalArgumentException("equalityTolerance must be non-negative, got " + value);
        }
        this.equalityTolerance = value;
        return this;
    }

    /** Sets the maximum number of generations. */
    public DEProblem maxIterations(int value) {
        if (value <= 0) throw new IllegalArgumentException("maxIterations must be positive, got " + value);
        this.maxIterations = value;
        return this;
    }

    /** Sets the absolute maximum number of objective evaluations. */
    public DEProblem maxEvaluations(int value) {
        if (value <= 0) throw new IllegalArgumentException("maxEvaluations must be positive, got " + value);
        this.maxEvaluations = value;
        return this;
    }

    /**
     * Sets the SciPy-style population multiplier.
     *
     * <p>The effective population size is {@code popsize * freeDimension}, clamped to at
     * least six members, unless an explicit population size or initial population is supplied.
     * Sobol initialization rounds that effective size up to the next power of two.</p>
     */
    public DEProblem popsize(int value) {
        if (value <= 0) throw new IllegalArgumentException("popsize must be positive, got " + value);
        this.populationMultiplier = value;
        this.populationSize = 0;
        return this;
    }

    /** Sets an explicit population size. */
    public DEProblem populationSize(int value) {
        if (value < 6) throw new IllegalArgumentException("populationSize must be at least 6, got " + value);
        this.populationSize = value;
        return this;
    }

    /**
     * Sets a custom batch evaluator.
     *
     * <p>Custom evaluators force deferred updating so the evaluator receives a stable flat
     * trial population. The caller owns any executor used by the evaluator.</p>
     */
    public DEProblem evaluator(DEEvaluator value) {
        this.evaluator = Objects.requireNonNull(value, "evaluator must not be null");
        this.forceDeferred = true;
        this.deferredUpdating = true;
        return this;
    }

    /** Sets the relative convergence tolerance. */
    public DEProblem tolerance(double value) {
        if (value < 0 || Double.isNaN(value)) throw new IllegalArgumentException("tolerance must be non-negative, got " + value);
        this.tolerance = value;
        return this;
    }

    /** Sets the absolute convergence tolerance. */
    public DEProblem absoluteTolerance(double value) {
        if (value < 0 || Double.isNaN(value)) throw new IllegalArgumentException("absoluteTolerance must be non-negative, got " + value);
        this.absoluteTolerance = value;
        return this;
    }

    /** Sets the differential weight F. Use {@link Bound#exactly(double)} for fixed F and {@link Bound#between(double, double)} for dithering. */
    public DEProblem mutation(Bound value) {
        validateMutation(value);
        this.mutation = value;
        return this;
    }

    /** Sets the crossover probability CR. */
    public DEProblem recombination(double value) {
        if (value < 0.0 || value > 1.0 || Double.isNaN(value)) {
            throw new IllegalArgumentException("recombination must be in [0, 1], got " + value);
        }
        this.recombination = value;
        return this;
    }

    /** Sets the final local polisher. Null restores the default L-BFGS-B/SLSQP selection. */
    public DEProblem polisher(FinalPolisher value) {
        this.polisher = value != null ? value : FinalPolisher.auto();
        return this;
    }

    /** Sets the default-polish evaluation budget used when {@link #maxEvaluations(int)} is not set. */
    public DEProblem polishMaxEvaluations(int value) {
        if (value <= 0) throw new IllegalArgumentException("polishMaxEvaluations must be positive, got " + value);
        this.polishMaxEvaluations = value;
        return this;
    }

    /** Sets a no-copy hook invoked after each completed generation. */
    public DEProblem hook(GenerationHook value) {
        this.hook = value;
        return this;
    }

    /** Sets the random number generator. */
    public DEProblem random(Random value) {
        this.rng = Objects.requireNonNull(value, "random must not be null");
        return this;
    }

    public int effectivePopulationSize() {
        if (initialPopulation != null) return initialPopulation.length;
        int size = populationSize > 0
                ? populationSize
                : Math.max(6, populationMultiplier * Math.max(1, freeDimension()));
        return initialization.populationSize(size);
    }

    private int freeDimension() {
        if (bounds == null) return dimension;
        int free = 0;
        for (Bound bound : bounds) {
            if (bound != null && bound.upper() > bound.lower()) free++;
        }
        return free;
    }

    public int effectiveMaxEvaluations() {
        if (maxEvaluations > 0) return maxEvaluations;
        int globalBudget = (maxIterations + 1) * effectivePopulationSize();
        return polisher == FinalPolisher.none() ? globalBudget : globalBudget + effectivePolishMaxEvaluations();
    }

    int effectivePolishMaxEvaluations() {
        if (polishMaxEvaluations > 0) return polishMaxEvaluations;
        return Math.clamp(dimension * 20, 100, 1000);
    }

    FinalPolisher.Context polishContext() {
        if (!hasConstraints()) return FinalPolisher.Context.empty();
        return new FinalPolisher.Context(equalityConstraints, inequalityConstraints, equalityTolerance);
    }

    double mutationScale(Random random) {
        if (mutation.isFixed()) return mutation.lower();
        return mutation.lower() + random.nextDouble() * (mutation.upper() - mutation.lower());
    }

    boolean hasIntegrality() {
        if (integrality == null) return false;
        for (boolean integral : integrality) {
            if (integral) return true;
        }
        return false;
    }

    boolean allIntegrality() {
        if (integrality == null) return false;
        for (boolean integral : integrality) {
            if (!integral) return false;
        }
        return true;
    }

    boolean hasConstraints() {
        return (equalityConstraints != null && equalityConstraints.length > 0)
                || (inequalityConstraints != null && inequalityConstraints.length > 0);
    }

    int constraintCount() {
        int count = 0;
        if (inequalityConstraints != null) count += inequalityConstraints.length;
        if (equalityConstraints != null) count += equalityConstraints.length;
        return count;
    }

    @Override
    public Optimization solve(DEWorkspace workspace) {
        validate();
        DEWorkspace ws = resolveWorkspace(workspace, DEWorkspace::new);
        return DECore.optimize(objective, bounds, initialPoint, ws, this, rng);
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
        if (integrality != null && integrality.length != dimension) {
            throw new IllegalArgumentException("integrality.length=" + integrality.length + " but bounds.length=" + dimension);
        }
        for (int i = 0; i < bounds.length; i++) {
            Bound bound = bounds[i];
            if (bound == null || !bound.hasBoth() || !Double.isFinite(bound.lower()) || !Double.isFinite(bound.upper())) {
                throw new IllegalArgumentException("differential evolution requires finite lower and upper bounds at index " + i);
            }
            if (bound.lower() > bound.upper()) {
                throw new IllegalArgumentException("bounds[" + i + "] must satisfy lower <= upper");
            }
            if (initialPoint != null && (initialPoint[i] < bound.lower() || initialPoint[i] > bound.upper())) {
                throw new IllegalArgumentException("initialPoint[" + i + "] must lie within bounds " + bound);
            }
            if (integrality != null && integrality[i] && Math.ceil(bound.lower()) > Math.floor(bound.upper())) {
                throw new IllegalArgumentException("integrality[" + i + "] has no integer value between bounds");
            }
        }
        if (initialPopulation == null) {
            initialization.validateDimension(dimension);
        }
        if (effectivePopulationSize() <= strategy.sampleCount) {
            throw new IllegalArgumentException("population is too small for strategy " + strategy);
        }
        if (maxEvaluations > 0 && maxEvaluations < effectivePopulationSize()) {
            throw new IllegalArgumentException(
                    "maxEvaluations must be at least the initial population size " + effectivePopulationSize());
        }
        if (forceDeferred && !deferredUpdating) {
            throw new IllegalArgumentException("custom evaluator requires deferred updating");
        }
    }

    private static void validateMutation(Bound value) {
        Objects.requireNonNull(value, "mutation must not be null");
        if (!value.hasBoth() || !Double.isFinite(value.lower()) || !Double.isFinite(value.upper())
                || value.lower() < 0.0 || value.upper() >= 2.0) {
            throw new IllegalArgumentException("mutation must be a finite bound within [0, 2), got " + value);
        }
    }

    private static Univariate.Objective[] cloneConstraints(String name, Univariate.Objective[] constraints) {
        if (constraints == null || constraints.length == 0) return null;
        Univariate.Objective[] copy = constraints.clone();
        for (int i = 0; i < copy.length; i++) {
            if (copy[i] == null) throw new IllegalArgumentException(name + "[" + i + "] must not be null");
        }
        return copy;
    }
}