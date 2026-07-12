/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.da;

import com.curioloop.yum4j.optim.Bound;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;

import java.util.Random;

/**
 * Package-private execution engine for {@link DAProblem}.
 *
 * <p>This class implements the SciPy/GenSA-style loop: initialize a finite point,
 * run generalized simulated annealing proposals at a decreasing temperature, reanneal
 * when the temperature falls below the configured restart threshold, and optionally
 * launch local searches from promising accepted points. Mutable vectors live in
 * {@link DAWorkspace} and are swapped rather than reallocated.</p>
 *
 * <h2>Main loop</h2>
 * <pre>
 *   reset current/best energy
 *   for each annealing temperature:
 *     generate 2*n visiting proposals
 *     accept improvements unconditionally
 *     accept uphill moves by generalized Metropolis probability
 *     notify hooks when the global best improves
 *     optionally run local search from the best or chain minimum
 * </pre>
 *
 * <h2>Temperature schedule</h2>
 * <p>The artificial temperature follows the Tsallis/Xiang generalized simulated annealing
 * schedule used by SciPy. {@link VisitingDistribution#temperatureScale(double)} transforms
 * that temperature into the scale used by the visiting distribution.</p>
 */
final class DACore {

    private static final int MAX_REINIT_COUNT = 1000;

    private DACore() {}

    static Optimization optimize(Univariate.Objective objective,
                                 Bound[] bounds,
                                 double[] x0,
                                 DAWorkspace workspace,
                                 DAProblem config,
                                 Random rng) {
        int n = bounds.length;
        boolean localSearch = config.localSearchEnabled();
        workspace.ensure(n, localSearch);
        VisitingDistribution visitDistribution = new VisitingDistribution(config.visit);

        // Establish a finite starting point. Invalid or infinite objective values are retried
        // with random bounded points up to the remaining evaluation budget.
        ResetResult reset = resetEnergy(objective, bounds, x0, workspace, rng, true, config.maxEvaluations);
        int evaluations = reset.evaluations;
        if (!reset.valid) {
            return new Optimization(Double.NaN, null, Double.POSITIVE_INFINITY,
                Optimization.Status.INVALID_INPUT, 0, evaluations);
        }
        if (evaluations >= config.maxEvaluations) {
            return new Optimization(Double.NaN, workspace.best.clone(), workspace.bestEnergy,
                Optimization.Status.MAX_EVALUATIONS_REACHED, 0, evaluations);
        }

        StrategyChainState chain = workspace.strategyChain;
        chain.reset(workspace.current, workspace.currentEnergy, localSearch);

        int iterations = 0;
        Optimization.Status status = Optimization.Status.MAX_ITERATIONS_REACHED;

        double restartTemperature = config.initialTemperature * config.restartTemperatureRatio;
        double t1 = Math.exp((config.visit - 1.0) * Math.log(2.0)) - 1.0;

        outer:
        while (true) {
            for (int i = 0; i < config.maxIterations; i++) {
                if (iterations >= config.maxIterations) {
                    status = Optimization.Status.MAX_ITERATIONS_REACHED;
                    break outer;
                }

                double s = i + 2.0;
                double t2 = Math.exp((config.visit - 1.0) * Math.log(s)) - 1.0;
                double temperature = config.initialTemperature * t1 / t2;
                if (temperature < restartTemperature) {
                    if (evaluations >= config.maxEvaluations) {
                        status = Optimization.Status.MAX_EVALUATIONS_REACHED;
                        break outer;
                    }
                        reset = resetEnergy(objective, bounds, null, workspace, rng, false,
                            config.maxEvaluations - evaluations);
                    evaluations += reset.evaluations;
                    if (!reset.valid) {
                        status = Optimization.Status.INVALID_INPUT;
                        break outer;
                    }
                    if (evaluations >= config.maxEvaluations) {
                        status = Optimization.Status.MAX_EVALUATIONS_REACHED;
                        break outer;
                    }
                    break;
                }

                chain.startRun(i, temperature);
                double temperatureScale = visitDistribution.temperatureScale(temperature);

                for (int j = 0; j < 2 * n; j++) {
                    if (evaluations >= config.maxEvaluations) {
                        status = Optimization.Status.MAX_EVALUATIONS_REACHED;
                        break outer;
                    }
                    visitDistribution.visiting(workspace.current, j, temperatureScale, bounds, workspace.visit, rng);
                    double energy = evaluate(objective, workspace.visit, n);
                    evaluations++;

                    if (energy < workspace.currentEnergy) {
                        acceptVisit(workspace, energy);
                        if (energy < workspace.bestEnergy) {
                            workspace.bestEnergy = energy;
                            System.arraycopy(workspace.current, 0, workspace.best, 0, n);
                            chain.bestImproved();
                            Optimization.Status hookStatus = invokeHook(config, workspace,
                                    iterations, evaluations, SearchType.ANNEALING);
                            if (hookStatus != null) {
                                status = hookStatus;
                                break outer;
                            }
                        }
                    } else if (accept(energy, workspace.currentEnergy, chain.temperatureStep, config.accept, rng)) {
                        acceptVisit(workspace, energy);
                        chain.accepted(workspace.current, workspace.currentEnergy);
                    }

                    chain.maybeRecordStagnationMinimum(j, workspace.current, workspace.currentEnergy);

                    if (evaluations >= config.maxEvaluations) {
                        status = Optimization.Status.MAX_EVALUATIONS_REACHED;
                        break outer;
                    }
                }

                if (localSearch) {
                    // SciPy's strategy chain first searches after a global improvement, then
                    // may search from a chain-local minimum after stagnation or a probability gate.
                    if (chain.energyStateImproved) {
                        workspace.localSearchRuns++;
                        LocalSearchResult local = runLocalSearch(objective, bounds,
                                workspace.best, workspace.bestEnergy, config, evaluations);
                        evaluations += local.evaluations;
                        if (local.improved) {
                            workspace.bestEnergy = local.energy;
                            System.arraycopy(local.x, 0, workspace.best, 0, n);
                            workspace.currentEnergy = local.energy;
                            System.arraycopy(local.x, 0, workspace.current, 0, n);
                            chain.notImproved = 0;
                            Optimization.Status hookStatus = invokeHook(config, workspace,
                                    iterations, evaluations, SearchType.LOCAL_SEARCH);
                            if (hookStatus != null) {
                                status = hookStatus;
                                break outer;
                            }
                        }
                        if (evaluations >= config.maxEvaluations) {
                            status = Optimization.Status.MAX_EVALUATIONS_REACHED;
                            break outer;
                        }
                    }

                    if (chain.shouldRunChainLocalSearch(workspace.bestEnergy, workspace.currentEnergy, rng)) {
                        workspace.localSearchRuns++;
                        LocalSearchResult local = runLocalSearch(objective, bounds,
                                chain.minimumLocation, chain.minimumEnergy, config, evaluations);
                        evaluations += local.evaluations;
                        if (local.improved) {
                            chain.localSearchImproved(local.x, local.energy);
                            if (local.energy < workspace.bestEnergy) {
                                workspace.bestEnergy = local.energy;
                                System.arraycopy(local.x, 0, workspace.best, 0, n);
                                workspace.currentEnergy = local.energy;
                                System.arraycopy(local.x, 0, workspace.current, 0, n);
                                Optimization.Status hookStatus = invokeHook(config, workspace,
                                        iterations, evaluations, SearchType.DUAL_ANNEALING);
                                if (hookStatus != null) {
                                    status = hookStatus;
                                    break outer;
                                }
                            }
                        }
                    }
                }

                iterations++;
            }
        }

        return new Optimization(Double.NaN, workspace.best.clone(), workspace.bestEnergy,
                status, iterations, evaluations);
    }

    private static void acceptVisit(DAWorkspace workspace, double energy) {
        // Swap buffers so the accepted proposal becomes current without copying n doubles.
        double[] previousCurrent = workspace.current;
        workspace.current = workspace.visit;
        workspace.visit = previousCurrent;
        workspace.currentEnergy = energy;
    }

    private static Optimization.Status invokeHook(DAProblem config,
                                                  DAWorkspace workspace,
                                                  int iterations,
                                                  int evaluations,
                                                  SearchType state) {
        if (config.hook == null) return null;
        try {
            boolean stop = config.hook.onMinimum(iterations, evaluations, workspace.best,
                    workspace.bestEnergy, state);
            return stop ? Optimization.Status.USER_REQUESTED_STOP : null;
        } catch (RuntimeException ex) {
            return Optimization.Status.CALLBACK_ERROR;
        }
    }

    private static ResetResult resetEnergy(Univariate.Objective objective,
                                           Bound[] bounds,
                                           double[] x0,
                                           DAWorkspace workspace,
                                           Random rng,
                                           boolean initializeBest,
                                           int maxEvaluations) {
        int evaluations = 0;
        int maxTries = Math.clamp(maxEvaluations, 0, MAX_REINIT_COUNT);
        for (int tries = 0; tries < maxTries; tries++) {
            if (x0 == null) {
                for (int i = 0; i < workspace.dimension; i++) {
                    workspace.current[i] = bounds[i].lower() + rng.nextDouble() * (bounds[i].upper() - bounds[i].lower());
                }
            } else {
                for (int i = 0; i < workspace.dimension; i++) {
                    workspace.current[i] = Math.clamp(x0[i], bounds[i].lower(), bounds[i].upper());
                }
            }
            double energy = evaluate(objective, workspace.current, workspace.dimension);
            evaluations++;
            if (Double.isFinite(energy)) {
                workspace.currentEnergy = energy;
                if (initializeBest || energy < workspace.bestEnergy) {
                    workspace.bestEnergy = energy;
                    System.arraycopy(workspace.current, 0, workspace.best, 0, workspace.dimension);
                }
                return new ResetResult(true, evaluations);
            }
            x0 = null;
        }
        return new ResetResult(false, evaluations);
    }

    private static boolean accept(double energy,
                                  double currentEnergy,
                                  double temperatureStep,
                                  double acceptanceParam,
                                  Random rng) {
        double probability = acceptanceProbability(energy, currentEnergy, temperatureStep, acceptanceParam);
        return probability > 0.0 && rng.nextDouble() <= probability;
    }

    static double acceptanceProbability(double energy,
                                        double currentEnergy,
                                        double temperatureStep,
                                        double acceptanceParam) {
        // Generalized Metropolis acceptance from generalized simulated annealing.
        double base = 1.0 - ((1.0 - acceptanceParam) * (energy - currentEnergy) / temperatureStep);
        if (base <= 0.0 || !Double.isFinite(base)) return 0.0;
        return Math.exp(Math.log(base) / (1.0 - acceptanceParam));
    }

    private static double evaluate(Univariate.Objective objective, double[] x, int n) {
        double value = objective.evaluate(x, n);
        return Double.isFinite(value) ? value : Double.POSITIVE_INFINITY;
    }

    private static LocalSearchResult runLocalSearch(Univariate.Objective objective,
                                                    Bound[] bounds,
                                                    double[] start,
                                                    double energy,
                                                    DAProblem config,
                                                    int evaluationsSoFar) {
        int remaining = config.maxEvaluations - evaluationsSoFar;
        int budget = Math.min(config.localSearchMaxIterations(), remaining);
        if (budget <= 2 * start.length + 1) return LocalSearchResult.notRun();
        Optimization result = config.localSearch.search(objective, bounds, start, budget);
        double[] solution = result.solution();
        if (solution != null && Double.isFinite(result.cost()) && result.cost() < energy
                && inBounds(solution, bounds)) {
            return new LocalSearchResult(true, solution, result.cost(), result.evaluations());
        }
        return new LocalSearchResult(false, null, energy, result.evaluations());
    }

    private static boolean inBounds(double[] x, Bound[] bounds) {
        for (int i = 0; i < x.length; i++) {
            if (!Double.isFinite(x[i]) || x[i] < bounds[i].lower() || x[i] > bounds[i].upper()) return false;
        }
        return true;
    }

    private record LocalSearchResult(boolean improved, double[] x, double energy, int evaluations) {
        static LocalSearchResult notRun() {
            return new LocalSearchResult(false, null, Double.POSITIVE_INFINITY, 0);
        }
    }

    private record ResetResult(boolean valid, int evaluations) { }
}