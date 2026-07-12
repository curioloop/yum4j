/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.da;

import com.curioloop.yum4j.optim.Bound;
import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.optim.Optimization;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.EnumSet;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class DAProblemTest {

    @Test
    void minimizesShiftedQuadraticWithLocalSearch() {
        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0] - 0.75) + square(x[1] + 1.5))
                .bounds(Bound.between(-5.0, 5.0), Bound.between(-5.0, 5.0))
                .initialPoint(4.0, 4.0)
                .random(new Random(42))
                .maxIterations(8)
                .maxEvaluations(2000)
                .localSearchMaxIterations(200)
                .solve();

        assertTrue(result.cost() < 1.0e-8, result.summary());
        assertEquals(0.75, result.solution()[0], 1.0e-4);
        assertEquals(-1.5, result.solution()[1], 1.0e-4);
    }

    @Test
    void noneLocalSearchHonorsBoundsAndBudget() {
        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0]) + square(x[1]))
                .bounds(Bound.between(-3.0, 3.0), Bound.between(-3.0, 3.0))
                .random(new Random(11))
                .maxIterations(5)
                .maxEvaluations(100)
                .localSearch(null)
                .solve();

        assertTrue(result.evaluations() <= 100);
        assertTrue(result.solution()[0] >= -3.0 && result.solution()[0] <= 3.0);
        assertTrue(result.solution()[1] >= -3.0 && result.solution()[1] <= 3.0);
        assertTrue(Double.isFinite(result.cost()));
    }

    @Test
    void sameSeedIsReproducibleWithoutLocalSearch() {
        Optimization first = reproducibleRun();
        Optimization second = reproducibleRun();

        assertEquals(first.cost(), second.cost(), 0.0);
        assertArrayEquals(first.solution(), second.solution(), 0.0);
        assertEquals(first.evaluations(), second.evaluations());
    }

    @Test
    void sameSeedIsReproducibleWithLocalSearch() {
        Optimization first = reproducibleRunWithLocalSearch();
        Optimization second = reproducibleRunWithLocalSearch();

        assertEquals(first.cost(), second.cost(), 0.0);
        assertArrayEquals(first.solution(), second.solution(), 0.0);
        assertEquals(first.evaluations(), second.evaluations());
    }

    @Test
    void functionCallCountMatchesEvaluationsWithoutLocalSearch() {
        AtomicInteger calls = new AtomicInteger();

        Optimization result = Minimizer.da()
                .objective((x, n) -> {
                    calls.incrementAndGet();
                    return square(x[0]) + square(x[1]);
                })
                .bounds(Bound.between(-1.0, 1.0), Bound.between(-1.0, 1.0))
                .random(new Random(1234))
                .maxIterations(3)
                .maxEvaluations(50)
                .localSearch(null)
                .solve();

        assertEquals(calls.get(), result.evaluations());
    }

    @ParameterizedTest
    @ValueSource(doubles = {1.1, 1.41, 2.0, 2.62, 2.9})
    void visitingDistributionChangesExpectedCoordinates(double visit) {
        VisitingDistribution distribution = new VisitingDistribution(visit);
        Bound[] bounds = {Bound.between(-5.12, 5.12), Bound.between(-5.12, 5.12)};
        double[] current = {0.0, 0.0};
        double[] out = new double[2];
        double temperatureScale = distribution.temperatureScale(5230.0);
        Random random = new Random(1234);

        distribution.visiting(current, 0, temperatureScale, bounds, out, random);
        assertTrue(out[0] != 0.0);
        assertTrue(out[1] != 0.0);
        assertInside(out, bounds);

        out[0] = 0.0;
        out[1] = 0.0;
        distribution.visiting(current, 2, temperatureScale, bounds, out, random);
        assertTrue(out[0] != 0.0);
        assertEquals(0.0, out[1], 0.0);
        assertInside(out, bounds);
    }

    @Test
    void hookReceivesAnnealingState() {
        AtomicInteger objectiveCalls = new AtomicInteger();
        EnumSet<SearchType> states = EnumSet.noneOf(SearchType.class);

        Optimization result = Minimizer.da()
                .objective((x, n) -> -objectiveCalls.incrementAndGet())
                .bounds(Bound.between(-1.0, 1.0), Bound.between(-1.0, 1.0))
                .random(new Random(91))
                .maxIterations(2)
                .maxEvaluations(20)
                .localSearch(null)
                .hook((iteration, evaluations, x, cost, state) -> {
                    states.add(state);
                    assertEquals(2, x.length);
                    assertTrue(evaluations > 0);
                    assertTrue(Double.isFinite(cost));
                    return false;
                })
                .solve();

        assertTrue(states.contains(SearchType.ANNEALING));
        assertTrue(Double.isFinite(result.cost()));
    }

    @Test
    void hookReceivesLocalSearchState() {
        EnumSet<SearchType> states = EnumSet.noneOf(SearchType.class);

        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0] - 0.5) + square(x[1] + 0.5))
                .bounds(Bound.between(-5.0, 5.0), Bound.between(-5.0, 5.0))
                .initialPoint(4.0, 4.0)
                .random(new Random(92))
                .maxIterations(1)
                .maxEvaluations(500)
                .localSearchMaxIterations(200)
                .hook((iteration, evaluations, x, cost, state) -> {
                    states.add(state);
                    return false;
                })
                .solve();

        assertTrue(states.contains(SearchType.LOCAL_SEARCH), result.summary());
        assertTrue(result.cost() < 1.0e-8, result.summary());
    }

    @Test
    void customLocalSearchCanImproveBestPoint() {
        AtomicInteger calls = new AtomicInteger();

        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0] - 0.25) + square(x[1] + 0.5))
                .bounds(Bound.between(-5.0, 5.0), Bound.between(-5.0, 5.0))
                .initialPoint(4.0, 4.0)
                .random(new Random(96))
                .maxIterations(1)
                .maxEvaluations(100)
                .localSearchMaxIterations(20)
                .localSearch((objective, bounds, start, maxEvaluations) -> {
                    calls.incrementAndGet();
                    double[] solution = {0.25, -0.5};
                    return new Optimization(Double.NaN, solution, objective.evaluate(solution, solution.length),
                            Optimization.Status.FUNCTION_TOLERANCE_REACHED, 1, 3);
                })
                .solve();

        assertTrue(calls.get() > 0);
        assertEquals(0.0, result.cost(), 0.0);
        assertEquals(0.25, result.solution()[0], 0.0);
        assertEquals(-0.5, result.solution()[1], 0.0);
    }

        @Test
        void disabledLocalSearchConsumesNoEvaluations() {
        DAWorkspace workspace = DAProblem.workspace();

        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0] - 0.25))
                .bounds(Bound.between(-1.0, 1.0))
                .initialPoint(0.75)
                .random(new Random(79))
                .maxIterations(1)
                .maxEvaluations(20)
                .localSearchMaxIterations(100)
                .localSearch(null)
                .solve(workspace);

        assertEquals(0, workspace.localSearchRuns);
        assertEquals(3, result.evaluations());
        assertTrue(Double.isFinite(result.cost()));
    }

    @Test
    void localSearchAcceptsFiniteImprovementRegardlessOfStatus() {
        AtomicInteger calls = new AtomicInteger();

        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0] - 0.125))
                .bounds(Bound.between(-1.0, 1.0))
                .initialPoint(0.75)
                .random(new Random(80))
                .maxIterations(1)
                .maxEvaluations(100)
                .localSearchMaxIterations(20)
                .localSearch((objective, bounds, start, maxEvaluations) -> {
                    calls.incrementAndGet();
                    double[] solution = {0.125};
                    return new Optimization(Double.NaN, solution, objective.evaluate(solution, solution.length),
                            Optimization.Status.INVALID_INPUT, 0, 1);
                })
                .solve();

        assertEquals(1, calls.get());
        assertEquals(0.0, result.cost(), 0.0);
        assertEquals(0.125, result.solution()[0], 0.0);
        assertEquals(Optimization.Status.MAX_ITERATIONS_REACHED, result.status());
    }

    @Test
    void hookCanStopDualAnnealing() {
        AtomicInteger objectiveCalls = new AtomicInteger();

        Optimization result = Minimizer.da()
                .objective((x, n) -> -objectiveCalls.incrementAndGet())
                .bounds(Bound.between(-1.0, 1.0), Bound.between(-1.0, 1.0))
                .random(new Random(93))
                .maxIterations(10)
                .maxEvaluations(100)
                .localSearch(null)
                .hook((iteration, evaluations, x, cost, state) -> true)
                .solve();

        assertEquals(Optimization.Status.USER_REQUESTED_STOP, result.status());
    }

    @Test
    void hookReceivesLiveBestArray() {
        AtomicInteger objectiveCalls = new AtomicInteger();

        Optimization result = Minimizer.da()
                .objective((x, n) -> -objectiveCalls.incrementAndGet())
                .bounds(Bound.between(-2.0, 2.0), Bound.between(-2.0, 2.0))
                .random(new Random(95))
                .maxIterations(10)
                .maxEvaluations(100)
                .localSearch(null)
                .hook((iteration, evaluations, x, cost, state) -> {
                    x[0] = 1.234;
                    return true;
                })
                .solve();

        assertEquals(Optimization.Status.USER_REQUESTED_STOP, result.status());
        assertEquals(1.234, result.solution()[0], 0.0);
    }

    @Test
    void hookExceptionReturnsCallbackError() {
        AtomicInteger objectiveCalls = new AtomicInteger();

        Optimization result = Minimizer.da()
                .objective((x, n) -> -objectiveCalls.incrementAndGet())
                .bounds(Bound.between(-1.0, 1.0), Bound.between(-1.0, 1.0))
                .random(new Random(94))
                .maxIterations(10)
                .maxEvaluations(100)
                .localSearch(null)
                .hook((iteration, evaluations, x, cost, state) -> {
                    throw new IllegalStateException("boom");
                })
                .solve();

        assertEquals(Optimization.Status.CALLBACK_ERROR, result.status());
    }

    @Test
    void localSearchDoesNotRunEveryIterationWithoutImprovement() {
        DAProblem problem = Minimizer.da()
                .objective((x, n) -> 1.0)
                .bounds(Bound.between(-1.0, 1.0), Bound.between(-1.0, 1.0))
                .random(new Random(123))
                .maxIterations(3)
                .maxEvaluations(1000)
                .localSearchMaxIterations(20);
        DAWorkspace workspace = DAProblem.workspace();

        problem.solve(workspace);

        assertTrue(workspace.localSearchRuns <= 1,
                "expected at most the first-iteration local search, got " + workspace.localSearchRuns);
    }

    @Test
    void strategyChainMinimumIsLazyWhenLocalSearchDisabled() {
        DAWorkspace workspace = DAProblem.workspace();

        Minimizer.da()
                .objective((x, n) -> square(x[0]) + square(x[1]))
                .bounds(Bound.between(-1.0, 1.0), Bound.between(-1.0, 1.0))
                .random(new Random(321))
                .maxIterations(1)
                .maxEvaluations(20)
                .localSearch(null)
                .solve(workspace);

        assertNotNull(workspace.strategyChain);
        assertNull(workspace.strategyChain.minimumLocation);
        assertTrue(Double.isFinite(workspace.strategyChain.minimumEnergy));
    }

    @Test
    void visitThreeUsesFiniteLowerEndpointForVisitingFormula() {
        VisitingDistribution distribution = new VisitingDistribution(3.0);
        assertEquals(Math.nextDown(3.0), distribution.visitingParam(), 0.0);

        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-1.0, 1.0))
                .visit(3.0)
                .random(new Random(9))
                .maxIterations(2)
                .maxEvaluations(30)
                .localSearch(null)
                .solve();

        assertTrue(Double.isFinite(result.cost()));
    }

    @Test
    void maxIterationsEnforcedWithoutLocalSearch() {
        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-1.0, 1.0))
                .random(new Random(75))
                .maxIterations(3)
                .maxEvaluations(1000)
                .localSearch(null)
                .solve();

        assertEquals(Optimization.Status.MAX_ITERATIONS_REACHED, result.status());
        assertEquals(3, result.iterations());
        assertTrue(result.evaluations() < 1000);
        assertInside(result.solution(), new Bound[] {Bound.between(-1.0, 1.0)});
    }

    @Test
    void reAnnealingPreservesGlobalBest() {
        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-1.0, 1.0))
                .initialPoint(0.0)
                .restartTemperatureRatio(0.99)
                .random(new Random(76))
                .maxIterations(3)
                .maxEvaluations(100)
                .localSearch(null)
                .solve();

        assertEquals(Optimization.Status.MAX_ITERATIONS_REACHED, result.status());
        assertEquals(0.0, result.cost(), 0.0);
        assertEquals(0.0, result.solution()[0], 0.0);
        assertTrue(result.evaluations() > 3);
    }

    @Test
    void initialPointInsideBoundsCanBeUsedAsInitialBest() {
        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0] - 0.25))
                .bounds(Bound.between(-1.0, 1.0))
                .initialPoint(0.25)
                .maxEvaluations(1)
                .localSearch(null)
                .solve();

        assertEquals(Optimization.Status.MAX_EVALUATIONS_REACHED, result.status());
        assertEquals(0.25, result.solution()[0], 0.0);
        assertEquals(0.0, result.cost(), 0.0);
    }

    @Test
    void rejectsInitialPointOutsideBounds() {
        DAProblem problem = Minimizer.da()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-1.0, 1.0))
                .initialPoint(1.1);

        assertThrows(IllegalArgumentException.class, problem::solve);
    }

    @Test
    void initialPointAndBoundsSetterOrderIsEquivalent() {
        Optimization initialPointFirst = Minimizer.da()
                .objective((x, n) -> square(x[0] - 0.25))
                .initialPoint(0.25)
                .bounds(Bound.between(-1.0, 1.0))
                .maxEvaluations(1)
                .localSearch(null)
                .solve();

        Optimization boundsFirst = Minimizer.da()
                .objective((x, n) -> square(x[0] - 0.25))
                .bounds(Bound.between(-1.0, 1.0))
                .initialPoint(0.25)
                .maxEvaluations(1)
                .localSearch(null)
                .solve();

        assertArrayEquals(initialPointFirst.solution(), boundsFirst.solution(), 0.0);
        assertEquals(initialPointFirst.cost(), boundsFirst.cost(), 0.0);
    }

    @Test
    void respectsSingleEvaluationBudget() {
        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0]) + square(x[1]))
                .bounds(Bound.between(-1.0, 1.0), Bound.between(-1.0, 1.0))
                .random(new Random(5))
                .maxIterations(10)
                .maxEvaluations(1)
                .localSearch(null)
                .solve();

        assertEquals(Optimization.Status.MAX_EVALUATIONS_REACHED, result.status());
        assertEquals(1, result.evaluations());
    }

    @Test
    void localSearchReceivesRemainingHardEvaluationBudget() {
        AtomicInteger budgetSeen = new AtomicInteger();

        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-1.0, 1.0))
                .initialPoint(0.75)
                .random(new Random(77))
                .maxIterations(1)
                .maxEvaluations(20)
                .localSearchMaxIterations(100)
                .localSearch((objective, bounds, start, maxEvaluations) -> {
                    budgetSeen.set(maxEvaluations);
                    return new Optimization(Double.NaN, start.clone(), objective.evaluate(start, start.length),
                            Optimization.Status.MAX_EVALUATIONS_REACHED, 0, maxEvaluations);
                })
                .solve();

        assertEquals(17, budgetSeen.get());
        assertEquals(Optimization.Status.MAX_EVALUATIONS_REACHED, result.status());
        assertEquals(20, result.evaluations());
    }

    @Test
    void customLocalSearchOverReportedEvaluationsArePreserved() {
        AtomicInteger budgetSeen = new AtomicInteger();

        Optimization result = Minimizer.da()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(-1.0, 1.0))
                .initialPoint(0.75)
                .random(new Random(78))
                .maxIterations(1)
                .maxEvaluations(20)
                .localSearchMaxIterations(100)
                .localSearch((objective, bounds, start, maxEvaluations) -> {
                    budgetSeen.set(maxEvaluations);
                    return new Optimization(Double.NaN, start.clone(), objective.evaluate(start, start.length),
                            Optimization.Status.MAX_EVALUATIONS_REACHED, 0, maxEvaluations + 5);
                })
                .solve();

        assertEquals(17, budgetSeen.get());
        assertEquals(Optimization.Status.MAX_EVALUATIONS_REACHED, result.status());
        assertEquals(25, result.evaluations());
    }

    @Test
    void returnsInvalidInputWhenNoFiniteStartingPointExists() {
        Optimization result = Minimizer.da()
                .objective((x, n) -> Double.NaN)
                .bounds(Bound.between(-1.0, 1.0), Bound.between(-1.0, 1.0))
                .random(new Random(5))
                .maxIterations(10)
                .maxEvaluations(20)
                .localSearch(null)
                .solve();

        assertEquals(Optimization.Status.INVALID_INPUT, result.status());
        assertEquals(20, result.evaluations());
        assertEquals(Double.POSITIVE_INFINITY, result.cost());
    }

    @Test
    void rejectsInvalidParameters() {
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().visit(1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().accept(-4.0));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().restartTemperatureRatio(1.0));
    }

    @Test
    void initialTemperatureRejectsNonPositiveNaNAndInfinity() {
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().initialTemperature(0.0));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().initialTemperature(-1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().initialTemperature(Double.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().initialTemperature(Double.POSITIVE_INFINITY));
    }

    @Test
    void restartTemperatureRatioRejectsBoundaryAndNaN() {
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().restartTemperatureRatio(0.0));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().restartTemperatureRatio(-0.1));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().restartTemperatureRatio(1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().restartTemperatureRatio(1.1));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().restartTemperatureRatio(Double.NaN));

        assertDoesNotThrow(() -> Minimizer.da().restartTemperatureRatio(0.5));
    }

    @Test
    void visitRejectsOutOfRangeAndAcceptsEndpoint() {
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().visit(1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().visit(0.9));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().visit(3.01));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().visit(Double.NaN));

        assertDoesNotThrow(() -> Minimizer.da().visit(3.0));
    }

    @Test
    void acceptRejectsOutOfRangeAndAcceptsBoundaries() {
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().accept(-10_000.0));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().accept(-10_001.0));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().accept(-4.999));
        assertThrows(IllegalArgumentException.class,
                () -> Minimizer.da().accept(Double.NaN));

        assertDoesNotThrow(() -> Minimizer.da().accept(-5.0));
        assertDoesNotThrow(() -> Minimizer.da().accept(-9_999.0));
    }

    @Test
    void rejectsInvalidBounds() {
        assertThrows(IllegalArgumentException.class,
                () -> Bound.between(1.0, 0.0));

        DAProblem infiniteLower = Minimizer.da()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(Double.NEGATIVE_INFINITY, 1.0));
        assertThrows(IllegalArgumentException.class, infiniteLower::solve);

        DAProblem nanUpper = Minimizer.da()
                .objective((x, n) -> square(x[0]))
                .bounds(Bound.between(0.0, Double.NaN));
        assertThrows(IllegalArgumentException.class, nanUpper::solve);
    }

    @Test
    void strategyChainLocalSearchGateUsesForcedNotImprovedThreshold() {
        StrategyChainState chain = new StrategyChainState();
        double[] current = {0.0, 0.0};

        chain.reset(current, 1.0, true);
        chain.temperatureStep = 1.0;
        chain.notImproved = chain.notImprovedMax;

        assertTrue(chain.shouldRunChainLocalSearch(0.5, 1.0, new Random(1)));
    }

    @Test
    void strategyChainLocalSearchImprovedResetsCounters() {
        StrategyChainState chain = new StrategyChainState();
        double[] current = {0.0, 0.0};
        double[] improved = {1.0, -2.0};

        chain.reset(current, 10.0, true);
        chain.notImproved = 7;
        chain.localSearchImproved(improved, 3.0);

        assertArrayEquals(improved, chain.minimumLocation, 0.0);
        assertEquals(3.0, chain.minimumEnergy, 0.0);
        assertEquals(0, chain.notImproved);
        assertEquals(2, chain.notImprovedMax);
    }

    @Test
    void acceptanceProbabilityFormulaMatchesSciPyReference() {
        assertEquals(1.0097587941791923,
                DACore.acceptanceProbability(0.0, 1.0, 100.0, -5.0), 1.0e-15);
        assertEquals(1.2599210498948732,
                DACore.acceptanceProbability(0.0, 1.0, 2.0, -5.0), 1.0e-15);
        assertEquals(0.8786035869128718,
                DACore.acceptanceProbability(10.0, 1.0, 100.0, -5.0), 1.0e-15);
        assertEquals(0.0,
                DACore.acceptanceProbability(10.0, 1.0, 1.0, -5.0), 0.0);
    }

    @Test
    void rejectsUnboundedBounds() {
        DAProblem problem = Minimizer.da()
                .objective((x, n) -> x[0] * x[0])
                .bounds(Bound.atMost(1.0));

        assertThrows(IllegalArgumentException.class, problem::solve);
    }

    private static Optimization reproducibleRun() {
        return Minimizer.da()
                .objective((x, n) -> square(x[0] - 0.25) + square(x[1] + 0.5))
                .bounds(Bound.between(-2.0, 2.0), Bound.between(-2.0, 2.0))
                .random(new Random(77))
                .maxIterations(6)
                .maxEvaluations(100)
                .localSearch(null)
                .solve();
    }

    private static Optimization reproducibleRunWithLocalSearch() {
        return Minimizer.da()
                .objective((x, n) -> square(x[0] - 0.25) + square(x[1] + 0.5))
                .bounds(Bound.between(-2.0, 2.0), Bound.between(-2.0, 2.0))
                .initialPoint(1.5, 1.5)
                .random(new Random(78))
                .maxIterations(2)
                .maxEvaluations(300)
                .localSearchMaxIterations(100)
                .solve();
    }

    private static void assertInside(double[] x, Bound[] bounds) {
        for (int i = 0; i < x.length; i++) {
            assertTrue(x[i] >= bounds[i].lower(), "coordinate below lower bound: " + x[i]);
            assertTrue(x[i] <= bounds[i].upper(), "coordinate above upper bound: " + x[i]);
        }
    }

    private static double square(double value) {
        return value * value;
    }
}