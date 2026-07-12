/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.de;

import com.curioloop.yum4j.optim.Bound;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;

import java.util.Arrays;
import java.util.Random;

/**
 * Package-private execution engine for {@link DEProblem}.
 *
 * <p>The core owns the generation loop and keeps policy decisions out of the public builder:
 * population initialization, immediate/deferred evolution, constraint ranking, convergence,
 * callbacks, and final polishing. All mutable arrays are supplied by
 * {@link DEWorkspace}; the hot path does not allocate per candidate.</p>
 *
 * <h2>Solve flow</h2>
 * <pre>
 *   1. allocate/reset workspace buffers
 *   2. initialize and optionally overwrite population[0] with x0
 *   3. evaluate the initial population and promote the best member to slot 0
 *   4. repeat generations until budget, convergence, callback stop, or max iterations
 *   5. copy population[0] to best and optionally run the final polisher
 * </pre>
 *
 * <h2>Immediate vs deferred generations</h2>
 * <p>Immediate mode evaluates and accepts each trial as soon as it is produced, allowing a
 * newly improved best member to affect later candidates in the same generation. Deferred
 * mode first stages trials in {@code trialPopulation}, then evaluates and applies them in a
 * batch. Deferred mode is required for custom batch evaluators.</p>
 *
 * <h2>Constraint ordering</h2>
 * <p>Constrained selection follows SciPy/Lampinen semantics: feasible beats infeasible,
 * feasible trials compare by objective energy, and infeasible trials compare by violation
 * components. Inequality components are stored before equality components.</p>
 */
final class DECore {

    private DECore() {}

    static Optimization optimize(Univariate.Objective objective,
                                 Bound[] bounds,
                                 double[] x0,
                                 DEWorkspace workspace,
                                 DEProblem config,
                                 Random rng) {
        int n = bounds.length;
        int populationSize = config.effectivePopulationSize();
        int maxEvaluations = config.effectiveMaxEvaluations();
        int constraintCount = config.constraintCount();
        boolean constrained = constraintCount > 0;
        boolean deferred = config.forceDeferred || config.deferredUpdating;
        workspace.ensure(n, populationSize, constraintCount, deferred);
        initializePopulation(workspace, bounds, x0, config, rng);

        // Initial evaluation establishes the invariant that population slot 0 is best.
        int evaluations = constrained
            ? evaluatePopulationConstrained(objective, workspace, n, config)
            : evaluatePopulationUnconstrained(objective, workspace, n, config.evaluator);
        promoteBest(workspace, n, constrained);

        // If no usable initial member exists, return the best stored point as diagnostic data.
        if ((!constrained && !Double.isFinite(workspace.populationEnergies[0]))
            || (constrained && ((workspace.feasible[0] && !Double.isFinite(workspace.populationEnergies[0]))
            || (!workspace.feasible[0] && !hasFiniteViolation(workspace.constraintViolations, workspace.populationSize))))) {
            copyFromPopulation(workspace.population, 0, workspace.best, n);
            return new Optimization(Double.NaN, workspace.best.clone(), Double.POSITIVE_INFINITY,
                    Optimization.Status.INVALID_INPUT, 0, evaluations);
        }

        Optimization.Status status = Optimization.Status.MAX_ITERATIONS_REACHED;
        int iterations = 0;

        if (evaluations >= maxEvaluations) {
            status = Optimization.Status.MAX_EVALUATIONS_REACHED;
        } else if (converged(workspace.populationEnergies, populationSize,
                config.absoluteTolerance, config.tolerance)) {
            status = Optimization.Status.FUNCTION_TOLERANCE_REACHED;
        } else {
            for (int generation = 0; generation < config.maxIterations; generation++) {
                GenerationResult result = constrained
                        ? (deferred
                            ? evolveDeferredConstrained(objective, bounds, workspace, config, rng, maxEvaluations - evaluations)
                            : evolveImmediateConstrained(objective, bounds, workspace, config, rng, maxEvaluations - evaluations))
                        : (deferred
                            ? evolveDeferredUnconstrained(objective, bounds, workspace, config, rng, maxEvaluations - evaluations, config.evaluator)
                            : evolveImmediateUnconstrained(objective, bounds, workspace, config, rng, maxEvaluations - evaluations));
                evaluations += result.evaluations;
                if (result.evaluations > 0) iterations = generation + 1;
                if (result.maxEvaluationsReached) {
                    status = Optimization.Status.MAX_EVALUATIONS_REACHED;
                    break;
                }
                Optimization.Status hookStatus = invokeHook(config, workspace, iterations, evaluations);
                if (hookStatus != null) {
                    status = hookStatus;
                    break;
                }
                if (converged(workspace.populationEnergies, populationSize,
                        config.absoluteTolerance, config.tolerance)) {
                    status = Optimization.Status.FUNCTION_TOLERANCE_REACHED;
                    break;
                }
            }
        }

        copyFromPopulation(workspace.population, 0, workspace.best, n);
        double bestEnergy = workspace.populationEnergies[0];

        // Polishing is skipped only when every variable is integral. Mixed problems fix the
        // integral coordinates at the best integer values and polish the remaining coordinates.
        if (!config.allIntegrality() && evaluations < maxEvaluations) {
            int remaining = maxEvaluations - evaluations;
            if (remaining > 2 * n + 1) {
                Bound[] polishBounds = polishBounds(bounds, workspace.best, config, n);
                Optimization polished = config.polisher
                        .polish(objective, polishBounds, workspace.best, remaining, config.polishContext());
                evaluations += polished.evaluations();
                if (polished.solution() != null && Double.isFinite(polished.cost())
                        && polished.cost() < bestEnergy
                        && inBounds(polished.solution(), polishBounds, n)
                        && (!constrained || !polished.status().error())
                        && (!constrained || constraintViolation(polished.solution(), n, config) <= 0.0)) {
                    System.arraycopy(polished.solution(), 0, workspace.best, 0, n);
                    bestEnergy = polished.cost();
                    if (polished.status().converged()) status = polished.status();
                }
            }
        }

        return new Optimization(Double.NaN, workspace.best.clone(), bestEnergy,
                status, iterations, evaluations);
    }

    private static Optimization.Status invokeHook(DEProblem config,
                                                  DEWorkspace workspace,
                                                  int iterations,
                                                  int evaluations) {
        if (config.hook == null) return null;
        copyFromPopulation(workspace.population, 0, workspace.best, workspace.dimension);
        try {
            boolean stop = config.hook.onGeneration(iterations, evaluations, workspace.best,
                    workspace.populationEnergies[0], callbackConvergence(workspace.populationEnergies,
                            workspace.populationSize, config.tolerance), workspace.population,
                    workspace.populationEnergies, workspace.dimension, workspace.populationSize);
            return stop ? Optimization.Status.USER_REQUESTED_STOP : null;
        } catch (RuntimeException ex) {
            return Optimization.Status.CALLBACK_ERROR;
        }
    }

    private static GenerationResult evolveImmediateUnconstrained(Univariate.Objective objective,
                                                                 Bound[] bounds,
                                                                 DEWorkspace workspace,
                                                                 DEProblem config,
                                                                 Random rng,
                                                                 int remainingEvaluations) {
        int evaluations = 0;
        double scale = config.mutationScale(rng);
        for (int candidate = 0; candidate < workspace.populationSize; candidate++) {
            if (evaluations >= remainingEvaluations) return new GenerationResult(evaluations, true);
            if (config.trialStrategy != null) {
                createCustomTrialInto(candidate, workspace, config, rng, workspace.trial, 0);
            } else {
                createBuiltInTrialInto(candidate, scale, workspace, config, rng, workspace.trial, 0);
            }
            repairTrial(workspace, bounds, config, rng, workspace.dimension);

            double energy = evaluate(objective, workspace.trial, workspace.dimension);
            evaluations++;
            if (energy <= workspace.populationEnergies[candidate]) {
                copyToPopulation(workspace.trial, workspace.population, candidate, workspace.dimension);
                workspace.populationEnergies[candidate] = energy;
                if (energy <= workspace.populationEnergies[0]) {
                    swapMembers(workspace, 0, candidate, workspace.dimension);
                }
            }
        }
        return new GenerationResult(evaluations, false);
    }

    private static GenerationResult evolveImmediateConstrained(Univariate.Objective objective,
                                                               Bound[] bounds,
                                                               DEWorkspace workspace,
                                                               DEProblem config,
                                                               Random rng,
                                                               int remainingEvaluations) {
        int evaluations = 0;
        double scale = config.mutationScale(rng);
        for (int candidate = 0; candidate < workspace.populationSize; candidate++) {
            if (evaluations >= remainingEvaluations) return new GenerationResult(evaluations, true);
            if (config.trialStrategy != null) {
                createCustomTrialInto(candidate, workspace, config, rng, workspace.trial, 0);
            } else {
                createBuiltInTrialInto(candidate, scale, workspace, config, rng, workspace.trial, 0);
            }
            repairTrial(workspace, bounds, config, rng, workspace.dimension);

            double violation = constraintViolation(workspace.trial, workspace.dimension, config,
                    workspace.constraintViolationScratch, 0);
            boolean feasible = violation <= 0.0;
            double energy;
            if (feasible) {
                energy = evaluate(objective, workspace.trial, workspace.dimension);
                evaluations++;
            } else {
                energy = Double.POSITIVE_INFINITY;
            }
            if (acceptTrial(energy, feasible, violation, workspace.constraintViolationScratch, 0,
                    workspace.populationEnergies[candidate], workspace.feasible[candidate], workspace.constraintViolations[candidate],
                    workspace.constraintViolationComponents, candidate * workspace.constraintCount, workspace.constraintCount)) {
                copyComponents(workspace.constraintViolationScratch, 0,
                    workspace.constraintViolationComponents, candidate * workspace.constraintCount, workspace.constraintCount);
                copyToPopulation(workspace.trial, workspace.population, candidate, workspace.dimension);
                workspace.populationEnergies[candidate] = energy;
                workspace.feasible[candidate] = feasible;
                workspace.constraintViolations[candidate] = violation;
                if (acceptTrial(energy, feasible, violation, workspace.constraintViolationComponents,
                        candidate * workspace.constraintCount,
                        workspace.populationEnergies[0], workspace.feasible[0], workspace.constraintViolations[0],
                        workspace.constraintViolationComponents, 0, workspace.constraintCount)) {
                    swapMembers(workspace, 0, candidate, workspace.dimension);
                }
            }
        }
        return new GenerationResult(evaluations, false);
    }

    private static GenerationResult evolveDeferredUnconstrained(Univariate.Objective objective,
                                                                Bound[] bounds,
                                                                DEWorkspace workspace,
                                                                DEProblem config,
                                                                Random rng,
                                                                int remainingEvaluations,
                                                                DEEvaluator evaluator) {
        if (remainingEvaluations <= 0) return new GenerationResult(0, true);
        int trialCount = Math.min(workspace.populationSize, remainingEvaluations);
        generateDeferredTrials(bounds, workspace, config, rng, trialCount);
        evaluateFlatPopulation(objective, workspace, workspace.trialPopulation, workspace.trialEnergies,
                trialCount, workspace.dimension, evaluator);
        applyDeferredTrialsUnconstrained(workspace, trialCount);
        promoteBest(workspace, workspace.dimension, false);
        return new GenerationResult(trialCount, trialCount >= remainingEvaluations && trialCount < workspace.populationSize);
    }

    private static GenerationResult evolveDeferredConstrained(Univariate.Objective objective,
                                                              Bound[] bounds,
                                                              DEWorkspace workspace,
                                                              DEProblem config,
                                                              Random rng,
                                                              int remainingEvaluations) {
        if (remainingEvaluations <= 0) return new GenerationResult(0, true);
        int trialCount = workspace.populationSize;
        generateDeferredTrials(bounds, workspace, config, rng, trialCount);
        evaluateTrialConstraints(workspace, config, trialCount);
        int evaluations = evaluateTrialPopulationConstrained(objective, workspace, trialCount,
                remainingEvaluations, config.evaluator);
        applyDeferredTrialsConstrained(workspace, trialCount);
        promoteBest(workspace, workspace.dimension, true);
        return new GenerationResult(evaluations, evaluations >= remainingEvaluations);
    }

    private static void generateDeferredTrials(Bound[] bounds,
                                               DEWorkspace workspace,
                                               DEProblem config,
                                               Random rng,
                                               int trialCount) {
        double scale = config.mutationScale(rng);
        for (int candidate = 0; candidate < trialCount; candidate++) {
            int offset = candidate * workspace.dimension;
            if (config.trialStrategy != null) {
                createCustomTrialInto(candidate, workspace, config, rng, workspace.trialPopulation, offset);
                repairTrial(workspace.trialPopulation, offset, bounds, config, rng, workspace.dimension);
            } else {
                createBuiltInTrialInto(candidate, scale, workspace, config, rng, workspace.trialPopulation, offset);
                repairTrial(workspace.trialPopulation, offset, bounds, config, rng, workspace.dimension);
            }
        }
    }

    private static void applyDeferredTrialsUnconstrained(DEWorkspace workspace, int evaluatedTrials) {
        for (int candidate = 0; candidate < evaluatedTrials; candidate++) {
            double energy = workspace.trialEnergies[candidate];
            if (energy <= workspace.populationEnergies[candidate]) {
                System.arraycopy(workspace.trialPopulation, candidate * workspace.dimension,
                        workspace.population, candidate * workspace.dimension, workspace.dimension);
                workspace.populationEnergies[candidate] = energy;
            }
        }
    }

    private static void applyDeferredTrialsConstrained(DEWorkspace workspace, int evaluatedTrials) {
        for (int candidate = 0; candidate < workspace.populationSize; candidate++) {
            if (!workspace.trialEligible[candidate]) continue;
            double energy = workspace.trialEnergies[candidate];
            double violation = workspace.trialConstraintViolations[candidate];
            boolean feasible = violation <= 0.0;
            if (!feasible) energy = Double.POSITIVE_INFINITY;
            if (acceptTrial(energy, feasible, violation, workspace.trialConstraintViolationComponents,
                    candidate * workspace.constraintCount,
                    workspace.populationEnergies[candidate], workspace.feasible[candidate], workspace.constraintViolations[candidate],
                    workspace.constraintViolationComponents, candidate * workspace.constraintCount, workspace.constraintCount)) {
                System.arraycopy(workspace.trialPopulation, candidate * workspace.dimension,
                        workspace.population, candidate * workspace.dimension, workspace.dimension);
                workspace.populationEnergies[candidate] = energy;
                workspace.feasible[candidate] = feasible;
                workspace.constraintViolations[candidate] = violation;
                System.arraycopy(workspace.trialConstraintViolationComponents, candidate * workspace.constraintCount,
                        workspace.constraintViolationComponents, candidate * workspace.constraintCount, workspace.constraintCount);
            }
        }
        if (evaluatedTrials > 0) promoteBest(workspace, workspace.dimension, true);
    }

    private static void initializePopulation(DEWorkspace workspace,
                                             Bound[] bounds,
                                             double[] x0,
                                             DEProblem config,
                                             Random rng) {
        if (config.initialPopulation != null) {
            initializeFromArray(workspace, bounds, config.initialPopulation, config);
        } else {
            config.initialization.initialize(workspace.population, workspace.populationSize,
                    workspace.dimension, bounds, config, rng);
        }
        if (x0 != null) {
            for (int j = 0; j < workspace.dimension; j++) {
                workspace.population[j] = projectValue(x0[j], j, bounds, config);
            }
        }
    }

    private static void initializeFromArray(DEWorkspace workspace,
                                            Bound[] bounds,
                                            double[][] initialPopulation,
                                            DEProblem config) {
        for (int i = 0; i < workspace.populationSize; i++) {
            int off = i * workspace.dimension;
            for (int j = 0; j < workspace.dimension; j++) {
                workspace.population[off + j] = projectValue(initialPopulation[i][j], j, bounds, config);
            }
        }
    }

    private static int evaluatePopulationUnconstrained(Univariate.Objective objective,
                                                       DEWorkspace workspace,
                                                       int n,
                                                       DEEvaluator evaluator) {
        evaluateFlatPopulation(objective, workspace, workspace.population, workspace.populationEnergies,
                workspace.populationSize, n, evaluator);
        return workspace.populationSize;
    }

    private static int evaluatePopulationConstrained(Univariate.Objective objective,
                                                     DEWorkspace workspace,
                                                     int n,
                                                     DEProblem config) {
        for (int i = 0; i < workspace.populationSize; i++) {
            int off = i * n;
            System.arraycopy(workspace.population, off, workspace.trial, 0, n);
            double violation = constraintViolation(workspace.trial, n, config,
                    workspace.constraintViolationComponents, i * workspace.constraintCount);
            workspace.constraintViolations[i] = violation;
            workspace.feasible[i] = violation <= 0.0;
            workspace.populationEnergies[i] = Double.POSITIVE_INFINITY;
        }
        int evaluations = 0;
        for (int i = 0; i < workspace.populationSize; i++) {
            if (workspace.feasible[i]) {
                int off = i * n;
                System.arraycopy(workspace.population, off, workspace.trial, 0, n);
                workspace.populationEnergies[i] = evaluate(objective, workspace.trial, n);
                evaluations++;
            }
        }
        return evaluations;
    }

    private static void evaluateTrialConstraints(DEWorkspace workspace,
                                                 DEProblem config,
                                                 int trialCount) {
        Arrays.fill(workspace.trialConstraintViolations, 0.0);
        Arrays.fill(workspace.trialConstraintViolationComponents, 0.0);
        Arrays.fill(workspace.trialEligible, 0, trialCount, true);
        for (int i = 0; i < trialCount; i++) {
            int off = i * workspace.dimension;
            System.arraycopy(workspace.trialPopulation, off, workspace.trial, 0, workspace.dimension);
            workspace.trialConstraintViolations[i] = constraintViolation(workspace.trial, workspace.dimension, config,
                    workspace.trialConstraintViolationComponents, i * workspace.constraintCount);
        }
    }

    private static int evaluateTrialPopulationConstrained(Univariate.Objective objective,
                                                          DEWorkspace workspace,
                                                          int trialCount,
                                                          int remainingEvaluations,
                                                          DEEvaluator evaluator) {
        // Compact feasible trials so the evaluator never sees infeasible candidates and the
        // objective-evaluation budget counts objective calls rather than constraint calls.
        Arrays.fill(workspace.trialEnergies, Double.POSITIVE_INFINITY);
        Arrays.fill(workspace.feasibleTrialEnergies, Double.POSITIVE_INFINITY);
        int feasibleCount = 0;
        for (int i = 0; i < trialCount; i++) {
            if (workspace.trialConstraintViolations[i] <= 0.0) {
                if (feasibleCount >= remainingEvaluations) {
                    workspace.trialEligible[i] = false;
                    continue;
                }
                System.arraycopy(workspace.trialPopulation, i * workspace.dimension,
                        workspace.feasibleTrialPopulation, feasibleCount * workspace.dimension, workspace.dimension);
                workspace.feasibleTrialIndices[feasibleCount] = i;
                feasibleCount++;
            }
        }
        if (feasibleCount > 0) {
            evaluateFlatPopulation(objective, workspace, workspace.feasibleTrialPopulation,
                    workspace.feasibleTrialEnergies, feasibleCount, workspace.dimension, evaluator);
            for (int i = 0; i < feasibleCount; i++) {
                workspace.trialEnergies[workspace.feasibleTrialIndices[i]] = workspace.feasibleTrialEnergies[i];
            }
        }
        return feasibleCount;
    }

    private static void evaluateFlatPopulation(Univariate.Objective objective,
                                               DEWorkspace workspace,
                                               double[] population,
                                               double[] energies,
                                               int count,
                                               int n,
                                               DEEvaluator evaluator) {
        if (evaluator == DEEvaluator.serial()) {
            for (int i = 0; i < count; i++) {
                int off = i * n;
                System.arraycopy(population, off, workspace.trial, 0, n);
                energies[i] = evaluate(objective, workspace.trial, n);
            }
            return;
        }
        workspace.ensureEvaluationScratch(count, n);
        evaluator.evaluate(objective, population, n, 0, count, energies, workspace.evaluationScratch);
    }

    private static void promoteBest(DEWorkspace workspace, int n, boolean constrained) {
        // Keep the current best in population row 0 so built-in best-based strategies can read
        // population[0] directly without an extra best-index lookup in the hot loop.
        int best = 0;
        for (int i = 1; i < workspace.populationSize; i++) {
            boolean better;
            if (!constrained) {
                better = workspace.populationEnergies[i] < workspace.populationEnergies[best];
            } else if (workspace.feasible[i]) {
                better = !workspace.feasible[best] || workspace.populationEnergies[i] < workspace.populationEnergies[best];
            } else {
                better = !workspace.feasible[best] && workspace.constraintViolations[i] < workspace.constraintViolations[best];
            }
            if (better) {
                best = i;
            }
        }
        swapMembers(workspace, 0, best, n);
    }

    private static void mutate(int candidate,
                               double scale,
                               DEWorkspace workspace,
                               EvolutionStrategy strategy,
                               Random rng) {
        selectSamples(candidate, strategy.sampleCount, workspace, rng);
        strategy.mutate(workspace.population, workspace.dimension,
            candidate, workspace.samples, scale, workspace.mutant);
    }

    private static void createCustomTrialInto(int candidate,
                                              DEWorkspace workspace,
                                              DEProblem config,
                                              Random rng,
                                              double[] out,
                                              int offset) {
        // Fill with NaN first so custom strategies that forget a coordinate fail immediately.
        Arrays.fill(out, offset, offset + workspace.dimension, Double.NaN);
        config.trialStrategy.trial(candidate, workspace.population, workspace.dimension,
                workspace.populationSize, rng, out, offset);
        for (int i = 0; i < workspace.dimension; i++) {
            if (!Double.isFinite(out[offset + i])) {
                throw new IllegalStateException("trial strategy must fill every trial coordinate with a finite value");
            }
        }
    }

    private static void createBuiltInTrialInto(int candidate,
                                               double scale,
                                               DEWorkspace workspace,
                                               DEProblem config,
                                               Random rng,
                                               double[] out,
                                               int offset) {
        mutate(candidate, scale, workspace, config.strategy, rng);
        crossoverInto(candidate, workspace, config, rng, out, offset);
    }

    private static void selectSamples(int candidate,
                                      int sampleCount,
                                      DEWorkspace workspace,
                                      Random rng) {
        for (int i = 0; i < sampleCount; i++) {
            int value;
            do {
                value = rng.nextInt(workspace.populationSize);
            } while (value == candidate || contains(workspace.samples, i, value));
            workspace.samples[i] = value;
        }
    }

    private static boolean contains(int[] values, int length, int value) {
        for (int i = 0; i < length; i++) {
            if (values[i] == value) return true;
        }
        return false;
    }

    private static void crossoverInto(int candidate,
                                      DEWorkspace workspace,
                                      DEProblem config,
                                      Random rng,
                                      double[] out,
                                      int offset) {
        int n = workspace.dimension;
        int fillPoint = rng.nextInt(n);
        int populationOffset = candidate * n;
        // Fast paths preserve the guarantee that at least one coordinate comes from the mutant.
        if (config.recombination == 1.0) {
            System.arraycopy(workspace.mutant, 0, out, offset, n);
            return;
        }
        if (config.recombination == 0.0) {
            System.arraycopy(workspace.population, populationOffset, out, offset, n);
            out[offset + fillPoint] = workspace.mutant[fillPoint];
            return;
        }
        if (config.strategy.binomial) {
            for (int j = 0; j < n; j++) {
                out[offset + j] = (j == fillPoint || rng.nextDouble() < config.recombination)
                        ? workspace.mutant[j]
                        : workspace.population[populationOffset + j];
            }
        } else {
            System.arraycopy(workspace.population, populationOffset, out, offset, n);
            int j = fillPoint;
            int copied = 0;
            do {
                out[offset + j] = workspace.mutant[j];
                j = (j + 1) % n;
                copied++;
            } while (copied < n && rng.nextDouble() < config.recombination);
        }
    }

    private static void repairTrial(DEWorkspace workspace,
                                    Bound[] bounds,
                                    DEProblem config,
                                    Random rng,
                                    int n) {
        repairTrial(workspace.trial, 0, bounds, config, rng, n);
    }

    private static void repairTrial(double[] vector,
                                    int offset,
                                    Bound[] bounds,
                                    DEProblem config,
                                    Random rng,
                                    int n) {
        for (int j = 0; j < n; j++) {
            double v = vector[offset + j];
            if (!Double.isFinite(v) || v < bounds[j].lower() || v > bounds[j].upper()) {
                v = uniform(bounds[j].lower(), bounds[j].upper(), rng);
            }
            vector[offset + j] = projectValue(v, j, bounds, config);
        }
    }

    private static double evaluate(Univariate.Objective objective, double[] x, int n) {
        double energy = objective.evaluate(x, n);
        return Double.isFinite(energy) ? energy : Double.POSITIVE_INFINITY;
    }

    private static double constraintViolation(double[] x, int n, DEProblem config) {
        return constraintViolation(x, n, config, null, 0);
    }

    private static double constraintViolation(double[] x,
                                              int n,
                                              DEProblem config,
                                              double[] components,
                                              int componentOffset) {
        int component = componentOffset;
        int componentEnd = componentOffset + config.constraintCount();
        if (components != null) Arrays.fill(components, componentOffset, componentEnd, 0.0);
        double total = 0.0;
        if (config.inequalityConstraints != null) {
            for (Univariate.Objective constraint : config.inequalityConstraints) {
                double value = constraint.evaluate(x, n);
                double violation;
                if (!Double.isFinite(value)) {
                    violation = Double.POSITIVE_INFINITY;
                } else {
                    violation = value < 0.0 ? -value : 0.0;
                }
                if (components != null) components[component++] = violation;
                if (!Double.isFinite(violation)) return Double.POSITIVE_INFINITY;
                total += violation;
            }
        }
        if (config.equalityConstraints != null) {
            for (Univariate.Objective constraint : config.equalityConstraints) {
                double value = constraint.evaluate(x, n);
                double violation;
                if (!Double.isFinite(value)) {
                    violation = Double.POSITIVE_INFINITY;
                } else {
                    violation = Math.max(0.0, Math.abs(value) - config.equalityTolerance);
                }
                if (components != null) components[component++] = violation;
                if (!Double.isFinite(violation)) return Double.POSITIVE_INFINITY;
                total += violation;
            }
        }
        return total;
    }

    private static boolean acceptTrial(double trialEnergy,
                                       boolean trialFeasible,
                                       double trialViolation,
                                       double[] trialComponents,
                                       int trialComponentOffset,
                                       double currentEnergy,
                                       boolean currentFeasible,
                                       double currentViolation,
                                       double[] currentComponents,
                                       int currentComponentOffset,
                                       int constraintCount) {
        if (trialFeasible && currentFeasible) return trialEnergy <= currentEnergy;
        if (trialFeasible) return true;
        if (currentFeasible) return false;
        if (trialComponents != null && currentComponents != null && constraintCount > 0) {
            for (int i = 0; i < constraintCount; i++) {
                if (trialComponents[trialComponentOffset + i] > currentComponents[currentComponentOffset + i]) {
                    return false;
                }
            }
            return true;
        }
        return trialViolation <= currentViolation;
    }

    private static void copyComponents(double[] source, int sourceOffset, double[] target, int targetOffset, int count) {
        if (count > 0) System.arraycopy(source, sourceOffset, target, targetOffset, count);
    }

    private static boolean hasFiniteViolation(double[] violations, int length) {
        for (int i = 0; i < length; i++) {
            if (Double.isFinite(violations[i])) return true;
        }
        return false;
    }

    private static boolean converged(double[] energies, int length, double absoluteTolerance, double tolerance) {
        double mean = 0.0;
        for (int i = 0; i < length; i++) mean += energies[i];
        mean /= length;
        if (!Double.isFinite(mean)) return false;
        double variance = 0.0;
        for (int i = 0; i < length; i++) {
            double d = energies[i] - mean;
            variance += d * d;
        }
        double std = Math.sqrt(variance / length);
        return std <= absoluteTolerance + tolerance * Math.abs(mean);
    }

    private static double callbackConvergence(double[] energies, int length, double tolerance) {
        double mean = 0.0;
        for (int i = 0; i < length; i++) mean += energies[i];
        mean /= length;
        if (!Double.isFinite(mean)) return 0.0;
        double variance = 0.0;
        for (int i = 0; i < length; i++) {
            double d = energies[i] - mean;
            variance += d * d;
        }
        double std = Math.sqrt(variance / length);
        double eps = Math.ulp(1.0);
        return tolerance / (std / (Math.abs(mean) + eps) + eps);
    }

    private static boolean inBounds(double[] x, Bound[] bounds, int n) {
        for (int i = 0; i < n; i++) {
            if (!Double.isFinite(x[i]) || x[i] < bounds[i].lower() || x[i] > bounds[i].upper()) return false;
        }
        return true;
    }

    private static Bound[] polishBounds(Bound[] bounds, double[] best, DEProblem config, int n) {
        if (!config.hasIntegrality()) return bounds;
        Bound[] fixed = bounds.clone();
        for (int i = 0; i < n; i++) {
            if (config.integrality[i]) {
                fixed[i] = Bound.exactly(projectValue(best[i], i, bounds, config));
            }
        }
        return fixed;
    }

    private static void copyToPopulation(double[] source, double[] population, int member, int n) {
        System.arraycopy(source, 0, population, member * n, n);
    }

    private static void copyFromPopulation(double[] population, int member, double[] target, int n) {
        System.arraycopy(population, member * n, target, 0, n);
    }

    private static void swapMembers(DEWorkspace workspace, int a, int b, int n) {
        if (a == b) return;
        int ao = a * n;
        int bo = b * n;
        for (int j = 0; j < n; j++) {
            double tmp = workspace.population[ao + j];
            workspace.population[ao + j] = workspace.population[bo + j];
            workspace.population[bo + j] = tmp;
        }
        double tmpEnergy = workspace.populationEnergies[a];
        workspace.populationEnergies[a] = workspace.populationEnergies[b];
        workspace.populationEnergies[b] = tmpEnergy;
        if (workspace.feasible != null) {
            boolean tmpFeasible = workspace.feasible[a];
            workspace.feasible[a] = workspace.feasible[b];
            workspace.feasible[b] = tmpFeasible;
        }
        if (workspace.constraintViolations != null) {
            double tmpViolation = workspace.constraintViolations[a];
            workspace.constraintViolations[a] = workspace.constraintViolations[b];
            workspace.constraintViolations[b] = tmpViolation;
        }
        if (workspace.constraintViolationComponents != null) {
            int ac = a * workspace.constraintCount;
            int bc = b * workspace.constraintCount;
            for (int i = 0; i < workspace.constraintCount; i++) {
                double tmpComponent = workspace.constraintViolationComponents[ac + i];
                workspace.constraintViolationComponents[ac + i] = workspace.constraintViolationComponents[bc + i];
                workspace.constraintViolationComponents[bc + i] = tmpComponent;
            }
        }
    }

    private static double uniform(double lower, double upper, Random rng) {
        return lower + rng.nextDouble() * (upper - lower);
    }

    static double projectValue(double value,
                               int index,
                               Bound[] bounds,
                               DEProblem config) {
        Bound bound = bounds[index];
        double projected = Math.clamp(value, bound.lower(), bound.upper());
        if (config.integrality != null && config.integrality[index]) {
            projected = Math.rint(projected);
            projected = Math.clamp(projected, Math.ceil(bound.lower()), Math.floor(bound.upper()));
        }
        return projected;
    }

    private record GenerationResult(int evaluations, boolean maxEvaluationsReached) { }
}