/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.de;

import com.curioloop.yum4j.optim.Bound;
import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.optim.Optimization;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class DEProblemTest {

    @ParameterizedTest
    @EnumSource(EvolutionStrategy.class)
    void mutationFormulaMatchesStrategy(EvolutionStrategy strategy) {
        double[] population = {
                1.0, 2.0,
                3.0, 5.0,
                7.0, 11.0,
                13.0, 17.0,
                19.0, 23.0,
                29.0, 31.0
        };
        int[] samples = {1, 2, 3, 4, 0};
        double[] mutant = new double[2];

        strategy.mutate(population, 2, 5, samples, 0.5, mutant);

        assertArrayEquals(expectedMutant(strategy), mutant, 1.0e-12);
    }

    @ParameterizedTest
    @EnumSource(EvolutionStrategy.class)
    void allStrategiesMinimizeShiftedQuadratic(EvolutionStrategy strategy) {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 0.75) + square(x[1] + 1.25))
                .bounds(Bound.between(-5.0, 5.0), Bound.between(-5.0, 5.0))
                .strategy(strategy)
                .populationSize(8)
                .maxIterations(6)
                .maxEvaluations(1000)
                .random(new Random(2026 + strategy.ordinal()))
                .solve();

        assertTrue(result.cost() < 1.0e-8, strategy + " failed: " + result.summary());
        assertEquals(0.75, result.solution()[0], 1.0e-4);
        assertEquals(-1.25, result.solution()[1], 1.0e-4);
    }

    @Test
    void deferredUpdatingMinimizesShiftedQuadratic() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 0.25) + square(x[1] + 0.75))
                .bounds(Bound.between(-4.0, 4.0), Bound.between(-4.0, 4.0))
                .deferredUpdating(true)
                .random(new Random(31415))
                .maxIterations(20)
                .popsize(6)
                .solve();

        assertTrue(result.cost() < 1.0e-8, result.summary());
        assertEquals(0.25, result.solution()[0], 1.0e-4);
        assertEquals(-0.75, result.solution()[1], 1.0e-4);
    }

    @Test
    void workersUseExecutorBackedDeferredEvaluation() {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Optimization result = Minimizer.de()
                    .objective((x, n) -> square(x[0] - 0.5) + square(x[1] + 0.25))
                    .bounds(Bound.between(-4.0, 4.0), Bound.between(-4.0, 4.0))
                    .evaluator(DEEvaluator.parallel(executor, 2))
                    .random(new Random(2718))
                    .maxIterations(12)
                    .maxEvaluations(500)
                    .populationSize(8)
                    .solve();

            assertTrue(result.cost() < 1.0e-8, result.summary());
            assertEquals(0.5, result.solution()[0], 1.0e-4);
            assertEquals(-0.25, result.solution()[1], 1.0e-4);
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void workersRejectsZeroWorkerCount() {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> DEEvaluator.parallel(executor, 0));
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void customEvaluatorIsUsedForBatchEvaluation() {
        AtomicInteger calls = new AtomicInteger();
        DEEvaluator evaluator = (objective, candidates, dimension, from, to, energies, scratch) -> {
            calls.incrementAndGet();
            DEEvaluator.serial().evaluate(objective, candidates, dimension, from, to, energies, scratch);
        };

        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 0.5) + square(x[1] + 0.25))
                .bounds(Bound.between(-4.0, 4.0), Bound.between(-4.0, 4.0))
                .evaluator(evaluator)
                .random(new Random(2719))
                .maxIterations(5)
                .maxEvaluations(200)
                .populationSize(8)
                .solve();

        assertTrue(calls.get() > 0);
        assertTrue(result.cost() < 1.0e-8, result.summary());
    }

    @Test
    void customEvaluatorRejectsImmediateUpdating() {
        DEProblem problem = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-1.0, 1.0))
                .evaluator(DEEvaluator.serial())
                .deferredUpdating(false);

        assertThrows(IllegalArgumentException.class,
                problem::solve);
    }

    @Test
    void fixedBoundsAreAllowedAndPreserved() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 2.0) + square(x[1] + 1.0))
                .bounds(Bound.exactly(2.0), Bound.between(-5.0, 5.0))
                .initialPopulation(new double[][] {
                        {2.0, -1.0}, {2.0, 0.0}, {2.0, 1.0},
                        {2.0, 2.0}, {2.0, -2.0}, {2.0, 3.0}
                })
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve();

        assertEquals(0.0, result.cost(), 0.0);
        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(-1.0, result.solution()[1], 0.0);
    }

    @Test
    void initialPopulationProvidesPopulationAndIsClippedToBounds() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 1.0) + square(x[1] + 2.0))
                .bounds(Bound.between(-3.0, 3.0), Bound.between(-3.0, 3.0))
                .initialPopulation(new double[][] {
                        {10.0, 10.0}, {1.0, -2.0}, {0.0, 0.0},
                        {-1.0, -1.0}, {2.0, 2.0}, {-2.0, 1.0}
                })
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve();

        assertEquals(6, result.evaluations());
        assertEquals(0.0, result.cost(), 0.0);
        assertEquals(1.0, result.solution()[0], 0.0);
        assertEquals(-2.0, result.solution()[1], 0.0);
    }

    @Test
    void rejectsInitialPointOutsideBounds() {
        DEProblem problem = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(0.0, 1.0))
                .initialPoint(1.1);

        assertThrows(IllegalArgumentException.class, problem::solve);
    }

    @Test
    void randomInitializationCanUseMinimumPopulation() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 0.25))
                .bounds(Bound.between(-1.0, 1.0))
                .initialization(PopulationInit.RANDOM)
                .populationSize(6)
                .maxIterations(8)
                .maxEvaluations(120)
                .random(new Random(20260712))
                .solve();

        assertTrue(result.solution()[0] >= -1.0 && result.solution()[0] <= 1.0);
        assertTrue(result.cost() < 1.0e-8, result.summary());
    }

    @Test
    void someInfiniteObjectiveValuesDoNotInvalidateFinitePopulation() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> x[0] < 0.0 ? Double.POSITIVE_INFINITY : square(x[0] - 0.25))
                .bounds(Bound.between(-1.0, 1.0))
                .initialPopulation(new double[][] {{-1.0}, {-0.5}, {0.25}, {0.5}, {0.75}, {1.0}})
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve();

        assertEquals(0.25, result.solution()[0], 0.0);
        assertEquals(0.0, result.cost(), 0.0);
        assertEquals(Optimization.Status.MAX_EVALUATIONS_REACHED, result.status());
    }

    @Test
    void deferredUpdatingStopsExactlyAtPartialEvaluationBudget() {
        AtomicInteger calls = new AtomicInteger();

        Optimization result = Minimizer.de()
                .objective((x, n) -> calls.incrementAndGet())
                .bounds(Bound.between(-1.0, 1.0))
                .populationSize(6)
                .deferredUpdating(true)
                .maxIterations(10)
                .maxEvaluations(9)
                .tolerance(0.0)
                .absoluteTolerance(0.0)
                .polisher(FinalPolisher.none())
                .random(new Random(8765))
                .solve();

        assertEquals(Optimization.Status.MAX_EVALUATIONS_REACHED, result.status());
        assertEquals(9, result.evaluations());
        assertEquals(9, calls.get());
        assertEquals(1, result.iterations());
    }

    @Test
    void reusedWorkspaceClearsConstraintStateForUnconstrainedSolve() {
        DEWorkspace workspace = DEProblem.workspace();

        Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(0.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] - 10.0)
                .initialPopulation(new double[][] {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}})
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve(workspace);

        assertNotNull(workspace.feasible);
        assertNotNull(workspace.constraintViolationComponents);

        Minimizer.de()
                .objective((x, n) -> square(x[0] - 1.0))
                .bounds(Bound.between(0.0, 5.0))
                .initialPopulation(new double[][] {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}})
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve(workspace);

        assertNull(workspace.feasible);
        assertNull(workspace.constraintViolations);
        assertNull(workspace.constraintViolationComponents);
    }

    @Test
    void ditheredMutationScaleStaysInsideConfiguredRange() {
        DEProblem problem = Minimizer.de()
                .mutation(Bound.between(0.25, 0.75));
        Random random = new Random(2468);

        for (int i = 0; i < 100; i++) {
            double scale = problem.mutationScale(random);
            assertTrue(scale >= 0.25, "scale below lower bound: " + scale);
            assertTrue(scale < 0.75, "scale should use an open upper random draw: " + scale);
        }
    }

    @Test
    void maxIterationsStopsAfterConfiguredGenerations() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> x[0])
                .bounds(Bound.between(0.0, 10.0))
                .initialPopulation(new double[][] {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}})
                .maxIterations(3)
                .maxEvaluations(24)
                .tolerance(0.0)
                .absoluteTolerance(0.0)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) ->
                    trialOut[trialOffset] = population[candidate * dimension])
                .solve();

        assertEquals(Optimization.Status.MAX_ITERATIONS_REACHED, result.status());
        assertEquals(3, result.iterations());
        assertEquals(24, result.evaluations());
        assertEquals(0.0, result.solution()[0], 0.0);
    }

    @Test
    void initialPopulationWithIdenticalEnergiesConvergesBeforeGeneration() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> 1.0)
                .bounds(Bound.between(-1.0, 1.0))
                .initialPopulation(new double[][] {{-1.0}, {-0.5}, {0.0}, {0.25}, {0.5}, {1.0}})
                .maxIterations(10)
                .maxEvaluations(100)
                .polisher(FinalPolisher.none())
                .solve();

        assertEquals(Optimization.Status.FUNCTION_TOLERANCE_REACHED, result.status());
        assertEquals(0, result.iterations());
        assertEquals(6, result.evaluations());
    }

    @Test
    void boundsRejectNaNAndInfiniteLimitsInAnyDimension() {
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.de().bounds(new Bound[0]));
        assertThrows(IllegalArgumentException.class,
                () -> Bound.between(1.0, 0.0));

        DEProblem nanLower = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, nanLower::solve);

        DEProblem infiniteUpper = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(0.0, Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, infiniteUpper::solve);
    }

    @Test
    void populationInitializationIsSeedReproducibleByMode() {
        double[] randomFirst = initializedPopulation(PopulationInit.RANDOM, 1802);
        double[] randomSecond = initializedPopulation(PopulationInit.RANDOM, 1802);
        double[] lhsFirst = initializedPopulation(PopulationInit.LATIN_HYPERCUBE, 1802);
        double[] lhsSecond = initializedPopulation(PopulationInit.LATIN_HYPERCUBE, 1802);

        assertArrayEquals(randomFirst, randomSecond, 0.0);
        assertArrayEquals(lhsFirst, lhsSecond, 0.0);
        assertFalse(Arrays.equals(randomFirst, lhsFirst));
    }

    @Test
    void sobolInitializationRoundsPopulationSizeToPowerOfTwo() {
        DEWorkspace workspace = DEProblem.workspace();

        Minimizer.de()
                .objective((x, n) -> 1.0)
                .bounds(Bound.between(0.0, 1.0), Bound.between(0.0, 1.0))
                .initialization(PopulationInit.SOBOL)
                .populationSize(6)
                .maxEvaluations(8)
                .polisher(FinalPolisher.none())
                .solve(workspace);

        assertEquals(8, workspace.populationSize);
    }

    @Test
    void haltonInitializationKeepsRequestedPopulationSize() {
        DEWorkspace workspace = DEProblem.workspace();

        Minimizer.de()
                .objective((x, n) -> 1.0)
                .bounds(Bound.between(0.0, 1.0), Bound.between(0.0, 1.0))
                .initialization(PopulationInit.HALTON)
                .populationSize(6)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve(workspace);

        assertEquals(6, workspace.populationSize);
    }

    @Test
    void sobolInitializationMatchesDirectRowMajorSequence() {
        DEWorkspace workspace = DEProblem.workspace();

        Minimizer.de()
                .objective((x, n) -> 1.0)
                .bounds(Bound.between(0.0, 1.0), Bound.between(10.0, 20.0))
                .initialization(PopulationInit.SOBOL)
                .populationSize(6)
                .maxEvaluations(8)
                .polisher(FinalPolisher.none())
                .solve(workspace);

        double eps = 1.0e-10;
        double[] expected = {
                eps, 10.0 + 10.0 * eps,
                0.5, 15.0,
                0.75, 12.5,
                0.25, 17.5,
                0.375, 13.75,
                0.875, 18.75,
                0.625, 11.25,
                0.125, 16.25
        };
        assertArrayEquals(expected, workspace.population, 1.0e-12);
    }

    @Test
    void haltonInitializationMatchesDirectRowMajorSequence() {
        DEWorkspace workspace = DEProblem.workspace();

        Minimizer.de()
                .objective((x, n) -> 1.0)
                .bounds(Bound.between(0.0, 1.0), Bound.between(0.0, 1.0))
                .initialization(PopulationInit.HALTON)
                .populationSize(6)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve(workspace);

        double[] expected = {
                0.5, 1.0 / 3.0,
                0.25, 2.0 / 3.0,
                0.75, 1.0 / 9.0,
                0.125, 4.0 / 9.0,
                0.625, 7.0 / 9.0,
                0.375, 2.0 / 9.0
        };
        assertArrayEquals(expected, workspace.population, 1.0e-12);
    }

    @Test
    void qmcInitializationHonorsIntegralityAndFixedBounds() {
        DEWorkspace workspace = DEProblem.workspace();

        Minimizer.de()
                .objective((x, n) -> 1.0)
                .bounds(Bound.exactly(2.0), Bound.between(0.0, 5.0))
                .integrality(false, true)
                .initialization(PopulationInit.SOBOL)
                .populationSize(6)
                .maxEvaluations(8)
                .polisher(FinalPolisher.none())
                .solve(workspace);

        for (int member = 0; member < workspace.populationSize; member++) {
            assertEquals(2.0, workspace.population[member * 2], 0.0);
            double integerValue = workspace.population[member * 2 + 1];
            assertEquals(Math.rint(integerValue), integerValue, 0.0);
        }
    }

    @Test
    void sobolRejectsUnsupportedDimension() {
        DEProblem problem = Minimizer.de()
                .objective((x, n) -> 1.0)
                .bounds(repeatedBounds(11))
                .initialization(PopulationInit.SOBOL);

        assertThrows(IllegalArgumentException.class, problem::solve);
    }

    @Test
    void haltonRejectsUnsupportedDimension() {
        DEProblem problem = Minimizer.de()
                .objective((x, n) -> 1.0)
                .bounds(repeatedBounds(21))
                .initialization(PopulationInit.HALTON);

        assertThrows(IllegalArgumentException.class, problem::solve);
    }

    @Test
    void initialPointAndBoundsSetterOrderIsEquivalent() {
        Optimization initialPointFirst = Minimizer.de()
                .objective((x, n) -> square(x[0] - 0.5))
                .initialPoint(0.5)
                .bounds(Bound.between(0.0, 2.0))
                .initialPopulation(new double[][] {{2.0}, {1.8}, {1.6}, {1.4}, {1.2}, {1.0}})
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve();

        Optimization boundsFirst = Minimizer.de()
                .objective((x, n) -> square(x[0] - 0.5))
                .bounds(Bound.between(0.0, 2.0))
                .initialPoint(0.5)
                .initialPopulation(new double[][] {{2.0}, {1.8}, {1.6}, {1.4}, {1.2}, {1.0}})
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve();

        assertArrayEquals(initialPointFirst.solution(), boundsFirst.solution(), 0.0);
        assertEquals(0.0, initialPointFirst.cost(), 0.0);
        assertEquals(0.0, boundsFirst.cost(), 0.0);
    }

    @Test
    void recombinationRejectsOutOfRangeAndNaN() {
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.de().recombination(-0.1));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.de().recombination(1.1));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.de().recombination(Double.NaN));

        assertDoesNotThrow(() -> Minimizer.de().recombination(0.0));
        assertDoesNotThrow(() -> Minimizer.de().recombination(1.0));
    }

    @Test
    void deferredPartialBudgetDoesNotAcceptUnevaluatedTrials() {
        DEWorkspace workspace = DEProblem.workspace();

        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0])
                .initialPopulation(new double[][] {{1.0}, {1.0}, {1.0}, {-5.0}, {-5.0}, {-5.0}})
                .deferredUpdating(true)
                .maxIterations(1)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) ->
                    trialOut[trialOffset] = candidate < 3 ? 2.0 : 0.0)
                .solve(workspace);

        assertEquals(Optimization.Status.MAX_EVALUATIONS_REACHED, result.status());
        assertEquals(6, result.evaluations());
        assertEquals(1.0, result.solution()[0], 0.0);
        for (int candidate = 3; candidate < 6; candidate++) {
            assertEquals(-5.0, workspace.population[candidate], 0.0);
        }
    }

    @Test
    void constrainedDeferredBudgetStopDoesNotInvokeHook() {
        AtomicInteger hookCalls = new AtomicInteger();

        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0])
                .initialPopulation(new double[][] {{1.0}, {1.0}, {1.0}, {-5.0}, {-5.0}, {-5.0}})
                .deferredUpdating(true)
                .maxIterations(1)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) ->
                    trialOut[trialOffset] = candidate < 3 ? 2.0 : 0.0)
                .hook((iteration, evaluations, best, cost, convergence,
                       population, populationEnergies, dimension, populationSize) -> {
                    hookCalls.incrementAndGet();
                    return false;
                })
                .solve();

        assertEquals(Optimization.Status.MAX_EVALUATIONS_REACHED, result.status());
        assertEquals(0, hookCalls.get());
    }

    @Test
    void minimizesShiftedQuadraticWithPolish() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 1.25) + square(x[1] + 2.0))
                .bounds(Bound.between(-5.0, 5.0), Bound.between(-5.0, 5.0))
                .random(new Random(1234))
                .maxIterations(60)
                .maxEvaluations(2000)
                .popsize(8)
                .mutation(Bound.exactly(0.7))
                .recombination(0.9)
                .solve();

        assertTrue(result.cost() < 1.0e-8, result.summary());
        assertEquals(1.25, result.solution()[0], 1.0e-4);
        assertEquals(-2.0, result.solution()[1], 1.0e-4);
    }

    @Test
    void usesBoundsAsDimensionWithoutInitialPoint() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]) + square(x[1]))
                .bounds(Bound.between(-2.0, 2.0), Bound.between(-3.0, 3.0))
                .random(new Random(7))
                .maxIterations(20)
                .maxEvaluations(1000)
                .popsize(5)
                .solve();

        assertEquals(2, result.solution().length);
        assertTrue(result.cost() < 1.0e-8, result.summary());
    }

    @Test
    void sameSeedIsReproducibleWithoutPolish() {
        Optimization first = reproducibleRun();
        Optimization second = reproducibleRun();

        assertEquals(first.cost(), second.cost(), 0.0);
        assertArrayEquals(first.solution(), second.solution(), 0.0);
        assertEquals(first.evaluations(), second.evaluations());
    }

    @Test
    void hookReceivesLiveGenerationState() {
        AtomicInteger calls = new AtomicInteger();
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 0.2) + square(x[1] + 0.4))
                .bounds(Bound.between(-2.0, 2.0), Bound.between(-2.0, 2.0))
                .random(new Random(27))
                .maxIterations(3)
                .maxEvaluations(100)
                .populationSize(8)
                .polisher(FinalPolisher.none())
                .tolerance(0.0)
                .absoluteTolerance(0.0)
                .hook((iteration, evaluations, best, cost, convergence,
                       population, populationEnergies, dimension, populationSize) -> {
                    calls.incrementAndGet();
                    assertEquals(iteration, calls.get());
                    assertEquals(2, dimension);
                    assertEquals(8, populationSize);
                    assertEquals(2, best.length);
                    assertEquals(16, population.length);
                    assertEquals(8, populationEnergies.length);
                    assertEquals(cost, populationEnergies[0], 0.0);
                    assertTrue(Double.isFinite(convergence));
                    return false;
                })
                .solve();

        assertEquals(3, calls.get());
        assertEquals(3, result.iterations());
    }

    @Test
    void hookCanStopDifferentialEvolution() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]) + square(x[1]))
                .bounds(Bound.between(-2.0, 2.0), Bound.between(-2.0, 2.0))
                .random(new Random(31))
                .maxIterations(10)
                .populationSize(8)
                .polisher(FinalPolisher.none())
                .hook((iteration, evaluations, best, cost, convergence,
                       population, populationEnergies, dimension, populationSize) -> true)
                .solve();

        assertEquals(Optimization.Status.USER_REQUESTED_STOP, result.status());
        assertEquals(1, result.iterations());
    }

    @Test
    void hookReceivesLivePopulationArray() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]) + square(x[1]))
                .bounds(Bound.between(-2.0, 2.0), Bound.between(-2.0, 2.0))
                .random(new Random(33))
                .maxIterations(10)
                .populationSize(8)
                .polisher(FinalPolisher.none())
                .hook((iteration, evaluations, best, cost, convergence,
                       population, populationEnergies, dimension, populationSize) -> {
                    population[0] = 1.234;
                    return true;
                })
                .solve();

        assertEquals(Optimization.Status.USER_REQUESTED_STOP, result.status());
        assertEquals(1.234, result.solution()[0], 0.0);
    }

    @Test
    void hookExceptionReturnsCallbackError() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]) + square(x[1]))
                .bounds(Bound.between(-2.0, 2.0), Bound.between(-2.0, 2.0))
                .random(new Random(32))
                .maxIterations(10)
                .populationSize(8)
                .polisher(FinalPolisher.none())
                .hook((iteration, evaluations, best, cost, convergence,
                       population, populationEnergies, dimension, populationSize) -> {
                    throw new IllegalStateException("boom");
                })
                .solve();

        assertEquals(Optimization.Status.CALLBACK_ERROR, result.status());
    }

    @Test
    void customTrialStrategyCanProvideTrialVector() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 1.0) + square(x[1] + 2.0))
                .bounds(Bound.between(-5.0, 5.0), Bound.between(-5.0, 5.0))
                .random(new Random(41))
                .populationSize(6)
                .maxIterations(1)
                .maxEvaluations(12)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) -> {
                    trialOut[trialOffset] = 1.0;
                    trialOut[trialOffset + 1] = -2.0;
                })
                .solve();

        assertEquals(0.0, result.cost(), 0.0);
        assertEquals(1.0, result.solution()[0], 0.0);
        assertEquals(-2.0, result.solution()[1], 0.0);
    }

    @Test
    void customTrialStrategyReceivesLivePopulation() {
        AtomicInteger calls = new AtomicInteger();
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .initialPopulation(new double[][] {{4.0}, {3.0}, {2.0}, {1.0}, {0.5}, {0.25}})
                .maxIterations(1)
                .maxEvaluations(12)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) -> {
                    calls.incrementAndGet();
                    assertEquals(1, dimension);
                    assertEquals(6, populationSize);
                    assertEquals(6, population.length);
                    trialOut[trialOffset] = population[0];
                })
                .solve();

        assertEquals(6, calls.get());
        assertEquals(0.0625, result.cost(), 0.0);
    }

    @Test
    void customTrialStrategyMustFillEveryCoordinate() {
        DEProblem problem = Minimizer.de()
                .objective((x, n) -> square(x[0]) + square(x[1]))
                .bounds(Bound.between(-5.0, 5.0), Bound.between(-5.0, 5.0))
                .random(new Random(42))
                .populationSize(6)
                .maxIterations(1)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) ->
                    trialOut[trialOffset] = 0.0);

        assertThrows(IllegalStateException.class, problem::solve);
    }

    @Test
    void integralityRoundsFinalSolution() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 2.2) + square(x[1] - 0.3))
                .bounds(Bound.between(0.0, 5.0), Bound.between(-1.0, 1.0))
                .integrality(true, false)
                .random(new Random(51))
                .maxIterations(40)
                .popsize(8)
                .solve();

        assertEquals(Math.rint(result.solution()[0]), result.solution()[0], 0.0);
        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(0.3, result.solution()[1], 5.0e-3);
    }

    @Test
    void integralityRejectsBoundsWithoutInteger() {
        DEProblem problem = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(0.1, 0.9))
                .integrality(true);

        assertThrows(IllegalArgumentException.class, problem::solve);
    }

    @Test
    void mutationRejectsInvalidBounds() {
        assertThrows(NullPointerException.class,
            () -> Minimizer.de().mutation(null));
        assertThrows(IllegalArgumentException.class,
            () -> Minimizer.de().mutation(Bound.atLeast(0.5)));
        assertThrows(IllegalArgumentException.class,
            () -> Minimizer.de().mutation(Bound.between(-0.1, 0.5)));
        assertThrows(IllegalArgumentException.class,
            () -> Minimizer.de().mutation(Bound.between(0.5, 2.0)));
    }

    @Test
    void inequalityConstraintMovesOptimumToFeasibleBoundary() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] - 1.0)
                .random(new Random(61))
                .maxIterations(80)
                .popsize(12)
                .polisher(FinalPolisher.none())
                .solve();

        assertTrue(result.solution()[0] >= 1.0 - 1.0e-3, result.summary());
        assertEquals(1.0, result.solution()[0], 5.0e-3);
        assertEquals(1.0, result.cost(), 1.0e-2);
    }

    @Test
    void infeasiblePopulationIsRankedByConstraintViolation() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(0.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] - 10.0)
                .initialPopulation(new double[][] {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}})
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve();

        assertEquals(Double.POSITIVE_INFINITY, result.cost());
        assertEquals(5.0, result.solution()[0], 0.0);
    }

    @Test
    void infeasibleTrialMustImproveEveryConstraintComponent() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]) + square(x[1]))
                .bounds(Bound.between(-2.0, 2.0), Bound.between(0.0, 2.0))
                .inequalityConstraints(
                        (x, n) -> x[0],
                        (x, n) -> x[1] - 10.0)
                .initialPopulation(new double[][] {
                        {0.0, 0.0}, {-1.1, 0.0}, {-1.2, 0.0},
                        {-1.3, 0.0}, {-1.4, 0.0}, {-1.5, 0.0}
                })
                .maxIterations(1)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) -> {
                    if (candidate == 0) {
                        trialOut[trialOffset] = -1.0;
                        trialOut[trialOffset + 1] = 1.0;
                    } else {
                        trialOut[trialOffset] = -2.0;
                        trialOut[trialOffset + 1] = 0.0;
                    }
                })
                .solve();

        assertEquals(Double.POSITIVE_INFINITY, result.cost());
        assertArrayEquals(new double[] {0.0, 0.0}, result.solution(), 0.0);
    }

    @Test
    void deferredInfeasibleTrialMustImproveEveryConstraintComponent() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]) + square(x[1]))
                .bounds(Bound.between(-2.0, 2.0), Bound.between(0.0, 2.0))
                .inequalityConstraints(
                        (x, n) -> x[0],
                        (x, n) -> x[1] - 10.0)
                .initialPopulation(new double[][] {
                        {0.0, 0.0}, {-1.1, 0.0}, {-1.2, 0.0},
                        {-1.3, 0.0}, {-1.4, 0.0}, {-1.5, 0.0}
                })
                .deferredUpdating(true)
                .maxIterations(1)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) -> {
                    if (candidate == 0) {
                        trialOut[trialOffset] = -1.0;
                        trialOut[trialOffset + 1] = 1.0;
                    } else {
                        trialOut[trialOffset] = -2.0;
                        trialOut[trialOffset + 1] = 0.0;
                    }
                })
                .solve();

        assertEquals(Double.POSITIVE_INFINITY, result.cost());
        assertArrayEquals(new double[] {0.0, 0.0}, result.solution(), 0.0);
    }

    @Test
    void mixedEqualityAndInequalityComponentsMustAllImprove() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-10.0, 1.0))
                .inequalityConstraints((x, n) -> x[0])
                .equalityConstraints((x, n) -> x[0] + 9.0)
                .initialPopulation(new double[][] {{0.0}, {-10.0}, {-10.0}, {-10.0}, {-10.0}, {-10.0}})
                .maxIterations(1)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) ->
                    trialOut[trialOffset] = candidate == 0 ? -1.0 : -10.0)
                .solve();

        assertEquals(Double.POSITIVE_INFINITY, result.cost());
        assertEquals(0.0, result.solution()[0], 0.0);
    }

    @Test
    void equalityToleranceTreatsNearZeroAsFeasible() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 0.05))
                .bounds(Bound.between(-1.0, 1.0))
                .equalityConstraints((x, n) -> x[0])
                .equalityTolerance(0.1)
                .initialPopulation(new double[][] {{0.05}, {-0.05}, {0.0}, {0.08}, {-0.08}, {0.1}})
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve();

        assertEquals(0.05, result.solution()[0], 0.0);
        assertEquals(0.0, result.cost(), 0.0);
        assertEquals(6, result.evaluations());
    }

    @Test
    void nonFiniteConstraintValueCannotWinSelection() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] < 0.0 ? Double.NaN : x[0] - 10.0)
                .initialPopulation(new double[][] {{-5.0}, {-4.0}, {0.0}, {1.0}, {2.0}, {5.0}})
                .maxIterations(1)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) ->
                    trialOut[trialOffset] = population[candidate * dimension])
                .solve();

        assertEquals(Double.POSITIVE_INFINITY, result.cost());
        assertEquals(5.0, result.solution()[0], 0.0);
    }

    @Test
    void constrainedDeferredUsesConfiguredEvaluatorForFeasibleTrials() {
        AtomicInteger calls = new AtomicInteger();
        AtomicInteger candidates = new AtomicInteger();
        DEEvaluator evaluator = (objective, trialPopulation, dimension, from, to, energies, scratch) -> {
            calls.incrementAndGet();
            candidates.addAndGet(to - from);
            DEEvaluator.serial().evaluate(objective, trialPopulation, dimension, from, to, energies, scratch);
        };

        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] - 1.0)
                .initialPopulation(new double[][] {{0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}})
                .evaluator(evaluator)
                .maxIterations(1)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) ->
                    trialOut[trialOffset] = 1.0)
                .solve();

        assertEquals(1, calls.get());
        assertEquals(6, candidates.get());
        assertEquals(6, result.evaluations());
        assertEquals(1.0, result.solution()[0], 0.0);
        assertEquals(1.0, result.cost(), 0.0);
    }

    @Test
    void deferredUpdatingHonorsConstraints() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] - 1.0)
                .deferredUpdating(true)
                .random(new Random(62))
                .maxIterations(80)
                .popsize(12)
                .polisher(FinalPolisher.none())
                .solve();

        assertTrue(result.solution()[0] >= 1.0 - 1.0e-3, result.summary());
        assertEquals(1.0, result.solution()[0], 5.0e-3);
    }

    @Test
    void constrainedPolishUsesSlsqp() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 2.0))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] - 1.0)
                .initialPopulation(new double[][] {{3.0}, {3.5}, {4.0}, {4.5}, {5.0}, {2.5}})
                .maxEvaluations(200)
                .tolerance(1.0e9)
                .solve();

        assertTrue(result.solution()[0] >= 1.0 - 1.0e-6, result.summary());
        assertEquals(2.0, result.solution()[0], 1.0e-4, result.summary());
        assertTrue(result.cost() < 1.0e-8, result.summary());
        assertTrue(result.evaluations() > 6);
    }

    @Test
    void customPolisherReceivesConstraintContextAndBudget() {
        AtomicInteger calls = new AtomicInteger();
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) -> {
            calls.incrementAndGet();
            assertEquals(14, maxEvaluations);
            assertTrue(context.hasConstraints());
            assertNull(context.equalityConstraints());
            assertEquals(1, context.inequalityConstraints().length);
            assertEquals(1.0e-8, context.equalityTolerance(), 0.0);
            assertTrue(start[0] >= 2.0);
            return new Optimization(Double.NaN, new double[] {1.0}, 1.0,
                    Optimization.Status.FUNCTION_TOLERANCE_REACHED, 0, 2);
        };

        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] - 1.0)
                .initialPopulation(new double[][] {{2.0}, {2.5}, {3.0}, {3.5}, {4.0}, {4.5}})
                .maxEvaluations(20)
                .tolerance(1.0e9)
                .polisher(polisher)
                .solve();

        assertEquals(1, calls.get());
        assertEquals(1.0, result.solution()[0], 0.0);
        assertEquals(1.0, result.cost(), 0.0);
        assertEquals(8, result.evaluations());
    }

    @Test
    void constrainedPolishRejectsErrorImprovement() {
        AtomicInteger calls = new AtomicInteger();
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) -> {
            calls.incrementAndGet();
            return new Optimization(Double.NaN, new double[] {1.0}, 1.0,
                    Optimization.Status.LINE_SEARCH_FAILED, 0, 2);
        };

        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] - 1.0)
                .initialPopulation(new double[][] {{2.0}, {2.5}, {3.0}, {3.5}, {4.0}, {4.5}})
                .maxEvaluations(20)
                .tolerance(1.0e9)
                .polisher(polisher)
                .solve();

        assertEquals(1, calls.get());
        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(4.0, result.cost(), 0.0);
        assertEquals(8, result.evaluations());
    }

    @Test
    void constrainedPolishAcceptsFeasibleLimitImprovement() {
        AtomicInteger calls = new AtomicInteger();
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) -> {
            calls.incrementAndGet();
            return new Optimization(Double.NaN, new double[] {1.0}, 1.0,
                    Optimization.Status.MAX_ITERATIONS_REACHED, 0, 2);
        };

        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] - 1.0)
                .initialPopulation(new double[][] {{2.0}, {2.5}, {3.0}, {3.5}, {4.0}, {4.5}})
                .maxEvaluations(20)
                .tolerance(1.0e9)
                .polisher(polisher)
                .solve();

        assertEquals(1, calls.get());
        assertEquals(1.0, result.solution()[0], 0.0);
        assertEquals(1.0, result.cost(), 0.0);
        assertEquals(8, result.evaluations());
    }

    @Test
    void constrainedPolishRejectsInfeasibleConvergedImprovement() {
        AtomicInteger calls = new AtomicInteger();
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) -> {
            calls.incrementAndGet();
            return new Optimization(Double.NaN, new double[] {0.5}, 0.25,
                    Optimization.Status.FUNCTION_TOLERANCE_REACHED, 0, 2);
        };

        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .inequalityConstraints((x, n) -> x[0] - 1.0)
                .initialPopulation(new double[][] {{2.0}, {2.5}, {3.0}, {3.5}, {4.0}, {4.5}})
                .maxEvaluations(20)
                .tolerance(1.0e9)
                .polisher(polisher)
                .solve();

        assertEquals(1, calls.get());
        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(4.0, result.cost(), 0.0);
        assertEquals(8, result.evaluations());
    }

    @Test
    void nonePolisherSkipsConfiguredCustomPolisher() {
        AtomicInteger calls = new AtomicInteger();
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) -> {
            calls.incrementAndGet();
            return new Optimization(Double.NaN, new double[] {1.0}, 1.0,
                    Optimization.Status.FUNCTION_TOLERANCE_REACHED, 0, 2);
        };

        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .initialPopulation(new double[][] {{2.0}, {2.5}, {3.0}, {3.5}, {4.0}, {4.5}})
                .maxEvaluations(20)
                .tolerance(1.0e9)
                .polisher(polisher)
                .polisher(FinalPolisher.none())
                .solve();

        assertEquals(0, calls.get());
        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(4.0, result.cost(), 0.0);
        assertEquals(6, result.evaluations());
    }

    @Test
    void nonePolisherCanBeReplacedByCustomPolisher() {
        AtomicInteger calls = new AtomicInteger();
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) -> {
            calls.incrementAndGet();
            return new Optimization(Double.NaN, new double[] {1.0}, 1.0,
                    Optimization.Status.FUNCTION_TOLERANCE_REACHED, 0, 2);
        };

        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-5.0, 5.0))
                .initialPopulation(new double[][] {{2.0}, {2.5}, {3.0}, {3.5}, {4.0}, {4.5}})
                .maxEvaluations(20)
                .tolerance(1.0e9)
                .polisher(FinalPolisher.none())
                .polisher(polisher)
                .solve();

        assertEquals(1, calls.get());
        assertEquals(1.0, result.solution()[0], 0.0);
        assertEquals(1.0, result.cost(), 0.0);
        assertEquals(8, result.evaluations());
    }

    @Test
    void integralityAppliesToInitialPopulationAndInitialPoint() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 3.0))
                .bounds(Bound.between(0.0, 5.0))
                .integrality(true)
                .initialPopulation(new double[][] {{0.2}, {1.2}, {2.2}, {3.2}, {4.2}, {4.8}})
                .initialPoint(2.7)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .solve();

        assertEquals(3.0, result.solution()[0], 0.0);
        assertEquals(0.0, result.cost(), 0.0);
    }

    @Test
    void customTrialStrategyOutputHonorsIntegrality() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 2.0))
                .bounds(Bound.between(0.0, 5.0))
                .integrality(true)
                .random(new Random(52))
                .populationSize(6)
                .maxIterations(1)
                .maxEvaluations(12)
                .polisher(FinalPolisher.none())
                .strategy((candidate, population, dimension, populationSize, rng, trialOut, trialOffset) ->
                    trialOut[trialOffset] = 1.7)
                .solve();

        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(0.0, result.cost(), 0.0);
    }

        @Test
        void partialIntegralityPolishesContinuousVariables() {
        Optimization result = Minimizer.de()
            .objective((x, n) -> square(x[0] - 2.0) + square(x[1] - 0.25))
            .bounds(Bound.between(0.0, 5.0), Bound.between(-5.0, 5.0))
            .integrality(true, false)
            .initialPopulation(new double[][] {{2.2, 4.0}, {3.2, 4.0}, {4.2, 4.0}, {1.2, 4.0}, {0.2, 4.0}, {2.8, 4.0}})
            .maxEvaluations(200)
            .tolerance(1.0e9)
            .solve();

        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(0.25, result.solution()[1], 1.0e-4);
        assertTrue(result.cost() < 1.0e-8, result.summary());
        }

        @Test
        void partialIntegralityPassesExactBoundsToCustomPolisher() {
        AtomicInteger calls = new AtomicInteger();
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) -> {
            calls.incrementAndGet();
            assertTrue(bounds[0].isFixed());
            assertEquals(2.0, bounds[0].lower(), 0.0);
            assertEquals(2.0, bounds[0].upper(), 0.0);
            assertEquals(-5.0, bounds[1].lower(), 0.0);
            assertEquals(5.0, bounds[1].upper(), 0.0);
            return new Optimization(Double.NaN, new double[] {2.0, 1.0}, 0.0,
                Optimization.Status.FUNCTION_TOLERANCE_REACHED, 0, 2);
        };

        Optimization result = Minimizer.de()
            .objective((x, n) -> square(x[0] - 2.0) + square(x[1] - 1.0))
            .bounds(Bound.between(0.0, 5.0), Bound.between(-5.0, 5.0))
            .integrality(true, false)
            .initialPopulation(new double[][] {{2.2, 4.0}, {3.2, 4.0}, {4.2, 4.0}, {1.2, 4.0}, {0.2, 4.0}, {2.8, 4.0}})
            .maxEvaluations(20)
            .tolerance(1.0e9)
            .polisher(polisher)
            .solve();

        assertEquals(1, calls.get());
        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(1.0, result.solution()[1], 0.0);
        assertEquals(0.0, result.cost(), 0.0);
        }

        @Test
        void allIntegralVariablesSkipPolish() {
        AtomicInteger calls = new AtomicInteger();
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) -> {
            calls.incrementAndGet();
            return new Optimization(Double.NaN, new double[] {2.0}, 0.0,
                Optimization.Status.FUNCTION_TOLERANCE_REACHED, 0, 2);
        };

        Optimization result = Minimizer.de()
            .objective((x, n) -> square(x[0] - 2.0))
            .bounds(Bound.between(0.0, 5.0))
            .integrality(true)
            .initialPopulation(new double[][] {{2.2}, {3.2}, {4.2}, {1.2}, {0.2}, {2.8}})
            .maxEvaluations(20)
            .tolerance(1.0e9)
            .polisher(polisher)
            .solve();

        assertEquals(0, calls.get());
        assertEquals(2.0, result.solution()[0], 0.0);
        }

        @Test
        void partialIntegralityRejectsPolisherMovingIntegralDimension() {
        AtomicInteger calls = new AtomicInteger();
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) -> {
            calls.incrementAndGet();
            return new Optimization(Double.NaN, new double[] {3.0, 1.0}, 0.0,
                Optimization.Status.FUNCTION_TOLERANCE_REACHED, 0, 2);
        };

        Optimization result = Minimizer.de()
            .objective((x, n) -> square(x[0] - 2.0) + square(x[1] - 1.0))
            .bounds(Bound.between(0.0, 5.0), Bound.between(-5.0, 5.0))
            .integrality(true, false)
            .initialPopulation(new double[][] {{2.2, 4.0}, {3.2, 4.0}, {4.2, 4.0}, {1.2, 4.0}, {0.2, 4.0}, {2.8, 4.0}})
            .maxEvaluations(20)
            .tolerance(1.0e9)
            .polisher(polisher)
            .solve();

        assertEquals(1, calls.get());
        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(4.0, result.solution()[1], 0.0);
        assertEquals(9.0, result.cost(), 0.0);
        }

        @Test
        void partialIntegralityWithConstraintsAcceptsFeasibleImprovement() {
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) ->
            new Optimization(Double.NaN, new double[] {2.0, 1.0}, 0.0,
                Optimization.Status.FUNCTION_TOLERANCE_REACHED, 0, 2);

        Optimization result = Minimizer.de()
            .objective((x, n) -> square(x[0] - 2.0) + square(x[1] - 1.0))
            .bounds(Bound.between(0.0, 5.0), Bound.between(-5.0, 5.0))
            .integrality(true, false)
            .inequalityConstraints((x, n) -> x[1] - 0.5)
            .initialPopulation(new double[][] {{2.2, 4.0}, {3.2, 4.0}, {4.2, 4.0}, {1.2, 4.0}, {0.2, 4.0}, {2.8, 4.0}})
            .maxEvaluations(20)
            .tolerance(1.0e9)
            .polisher(polisher)
            .solve();

        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(1.0, result.solution()[1], 0.0);
        assertEquals(0.0, result.cost(), 0.0);
        }

        @Test
        void partialIntegralityWithConstraintsRejectsInfeasibleImprovement() {
        FinalPolisher polisher = (objective, bounds, start, maxEvaluations, context) ->
            new Optimization(Double.NaN, new double[] {2.0, 0.0}, 0.0,
                Optimization.Status.FUNCTION_TOLERANCE_REACHED, 0, 2);

        Optimization result = Minimizer.de()
            .objective((x, n) -> square(x[0] - 2.0) + square(x[1]))
            .bounds(Bound.between(0.0, 5.0), Bound.between(-5.0, 5.0))
            .integrality(true, false)
            .inequalityConstraints((x, n) -> x[1] - 0.5)
            .initialPopulation(new double[][] {{2.2, 1.0}, {3.2, 1.0}, {4.2, 1.0}, {1.2, 1.0}, {0.2, 1.0}, {2.8, 1.0}})
            .maxEvaluations(20)
            .tolerance(1.0e9)
            .polisher(polisher)
            .solve();

        assertEquals(2.0, result.solution()[0], 0.0);
        assertEquals(1.0, result.solution()[1], 0.0);
        assertEquals(1.0, result.cost(), 0.0);
        }

    @Test
    void defaultBudgetLeavesRoomForPolish() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> square(x[0] - 1.0) + square(x[1] + 1.0))
                .bounds(Bound.between(-5.0, 5.0), Bound.between(-5.0, 5.0))
                .random(new Random(17))
                .maxIterations(1)
                .popsize(4)
                .solve();

        assertTrue(result.cost() < 1.0e-8, result.summary());
        assertTrue(result.evaluations() > (1 + 1) * 4 * 2);
    }

    @Test
    void rejectsExplicitBudgetSmallerThanInitialPopulation() {
        DEProblem problem = Minimizer.de()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-1.0, 1.0))
                .populationSize(6)
                .maxEvaluations(5);

        assertThrows(IllegalArgumentException.class, problem::solve);
    }

    @Test
    void returnsInvalidInputWhenNoFinitePopulationMemberExists() {
        Optimization result = Minimizer.de()
                .objective((x, n) -> Double.NaN)
                .bounds(Bound.between(-1.0, 1.0), Bound.between(-1.0, 1.0))
                .random(new Random(123))
                .maxIterations(3)
                .popsize(4)
                .polisher(FinalPolisher.none())
                .solve();

        assertEquals(Optimization.Status.INVALID_INPUT, result.status());
        assertEquals(Double.POSITIVE_INFINITY, result.cost());
    }

    @Test
    void rejectsUnboundedBounds() {
        DEProblem problem = Minimizer.de()
                .objective((x, n) -> x[0] * x[0])
                .bounds(Bound.atLeast(-1.0));

        assertThrows(IllegalArgumentException.class, problem::solve);
    }

    private static Optimization reproducibleRun() {
        return Minimizer.de()
                .objective((x, n) -> square(x[0] - 0.5) + square(x[1] + 0.25))
                .bounds(Bound.between(-2.0, 2.0), Bound.between(-2.0, 2.0))
                .random(new Random(99))
                .maxIterations(12)
                .popsize(4)
                .polisher(FinalPolisher.none())
                .solve();
    }

    private static double[] initializedPopulation(PopulationInit initialization, long seed) {
        DEWorkspace workspace = DEProblem.workspace();
        Minimizer.de()
                .objective((x, n) -> x[0] + 0.5 * x[1])
                .bounds(Bound.between(-1.0, 1.0), Bound.between(-2.0, 2.0))
                .initialization(initialization)
                .populationSize(6)
                .maxEvaluations(6)
                .polisher(FinalPolisher.none())
                .random(new Random(seed))
                .solve(workspace);
        return workspace.population.clone();
    }

    private static Bound[] repeatedBounds(int dimension) {
        Bound[] bounds = new Bound[dimension];
        Arrays.fill(bounds, Bound.between(0.0, 1.0));
        return bounds;
    }

    private static double square(double value) {
        return value * value;
    }

    private static double[] expectedMutant(EvolutionStrategy strategy) {
        return switch (strategy) {
            case BEST1BIN, BEST1EXP -> new double[] {-1.0, -1.0};
            case RAND1BIN, RAND1EXP -> new double[] {0.0, 2.0};
            case RAND2BIN, RAND2EXP -> new double[] {3.0, 6.5};
            case BEST2BIN, BEST2EXP -> new double[] {-10.0, -10.0};
            case RAND_TO_BEST1BIN, RAND_TO_BEST1EXP -> new double[] {-1.0, 0.5};
            case CURRENT_TO_BEST1BIN, CURRENT_TO_BEST1EXP -> new double[] {13.0, 13.5};
        };
    }
}