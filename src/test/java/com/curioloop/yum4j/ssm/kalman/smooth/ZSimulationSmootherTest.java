package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.filter.ZKalmanFilterTest;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZSimulationSmootherTest {

    private static final double TOL = 1e-10;

    private static double[] arangeTenths(int length) {
        double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = i / 10.0;
        }
        return values;
    }

    private static double[] embedReal(double[] values) {
        double[] embedded = new double[2 * values.length];
        for (int i = 0; i < values.length; i++) {
            embedded[2 * i] = values[i];
        }
        return embedded;
    }

    private static double[] activeSlice(double[] values, int base, int length) {
        return Arrays.copyOfRange(values, base, base + length);
    }

    private static void assertComplexMatchesReal(String label, double[] expectedReal, double[] actualComplex) {
        assertEquals(expectedReal.length * 2, actualComplex.length, label + " length mismatch");
        for (int i = 0; i < expectedReal.length; i++) {
            assertEquals(expectedReal[i], actualComplex[2 * i], TOL, label + " real mismatch at index " + i);
            assertEquals(0.0, actualComplex[2 * i + 1], TOL, label + " imag mismatch at index " + i);
        }
    }

    private static void assertRealEmbeddedMatches(SimulationSmootherResult expected, ZSimulationSmootherResult actual) {
        assertComplexMatchesReal("generatedMeasurementDisturbance",
            activeSlice(expected.generatedMeasurementDisturbance,
                expected.generatedMeasurementDisturbanceBase(),
                expected.generatedMeasurementDisturbanceLength()),
            activeSlice(actual.generatedMeasurementDisturbance,
                actual.generatedMeasurementDisturbanceBase(),
                actual.generatedMeasurementDisturbanceLength()));
        assertComplexMatchesReal("generatedStateDisturbance",
            activeSlice(expected.generatedStateDisturbance,
                expected.generatedStateDisturbanceBase(),
                expected.generatedStateDisturbanceLength()),
            activeSlice(actual.generatedStateDisturbance,
                actual.generatedStateDisturbanceBase(),
                actual.generatedStateDisturbanceLength()));
        assertComplexMatchesReal("generatedObs",
            activeSlice(expected.generatedObs, expected.generatedObsBase(), expected.generatedObsLength()),
            activeSlice(actual.generatedObs, actual.generatedObsBase(), actual.generatedObsLength()));
        assertComplexMatchesReal("generatedState",
            activeSlice(expected.generatedState, expected.generatedStateBase(), expected.generatedStateLength()),
            activeSlice(actual.generatedState, actual.generatedStateBase(), actual.generatedStateLength()));
        assertComplexMatchesReal("simulatedMeasurementDisturbance",
            activeSlice(expected.simulatedMeasurementDisturbance,
                expected.simulatedMeasurementDisturbanceBase(),
                expected.simulatedMeasurementDisturbanceLength()),
            activeSlice(actual.simulatedMeasurementDisturbance,
                actual.simulatedMeasurementDisturbanceBase(),
                actual.simulatedMeasurementDisturbanceLength()));
        assertComplexMatchesReal("simulatedStateDisturbance",
            activeSlice(expected.simulatedStateDisturbance,
                expected.simulatedStateDisturbanceBase(),
                expected.simulatedStateDisturbanceLength()),
            activeSlice(actual.simulatedStateDisturbance,
                actual.simulatedStateDisturbanceBase(),
                actual.simulatedStateDisturbanceLength()));
        assertComplexMatchesReal("simulatedState",
            activeSlice(expected.simulatedState, expected.simulatedStateBase(), expected.simulatedStateLength()),
            activeSlice(actual.simulatedState, actual.simulatedStateBase(), actual.simulatedStateLength()));
    }

    private static void assertSimulationResultMatches(ZSimulationSmootherResult expected,
                                                      ZSimulationSmootherResult actual) {
        assertArrayEquals(activeSlice(expected.generatedMeasurementDisturbance,
            expected.generatedMeasurementDisturbanceBase(), expected.generatedMeasurementDisturbanceLength()),
            activeSlice(actual.generatedMeasurementDisturbance,
                actual.generatedMeasurementDisturbanceBase(), actual.generatedMeasurementDisturbanceLength()), TOL);
        assertArrayEquals(activeSlice(expected.generatedStateDisturbance,
            expected.generatedStateDisturbanceBase(), expected.generatedStateDisturbanceLength()),
            activeSlice(actual.generatedStateDisturbance,
                actual.generatedStateDisturbanceBase(), actual.generatedStateDisturbanceLength()), TOL);
        assertArrayEquals(activeSlice(expected.generatedObs, expected.generatedObsBase(), expected.generatedObsLength()),
            activeSlice(actual.generatedObs, actual.generatedObsBase(), actual.generatedObsLength()), TOL);
        assertArrayEquals(activeSlice(expected.generatedState, expected.generatedStateBase(), expected.generatedStateLength()),
            activeSlice(actual.generatedState, actual.generatedStateBase(), actual.generatedStateLength()), TOL);
        assertArrayEquals(activeSlice(expected.simulatedMeasurementDisturbance,
            expected.simulatedMeasurementDisturbanceBase(), expected.simulatedMeasurementDisturbanceLength()),
            activeSlice(actual.simulatedMeasurementDisturbance,
                actual.simulatedMeasurementDisturbanceBase(), actual.simulatedMeasurementDisturbanceLength()), TOL);
        assertArrayEquals(activeSlice(expected.simulatedStateDisturbance,
            expected.simulatedStateDisturbanceBase(), expected.simulatedStateDisturbanceLength()),
            activeSlice(actual.simulatedStateDisturbance,
                actual.simulatedStateDisturbanceBase(), actual.simulatedStateDisturbanceLength()), TOL);
        assertArrayEquals(activeSlice(expected.simulatedState, expected.simulatedStateBase(), expected.simulatedStateLength()),
            activeSlice(actual.simulatedState, actual.simulatedStateBase(), actual.simulatedStateLength()), TOL);
    }

    private static void assertSuppressed(double[] values) {
        assertEquals(0, values.length);
    }

    private static void assertGeneratedSuppressed(ZSimulationSmootherResult result) {
        assertSuppressed(result.generatedMeasurementDisturbance);
        assertSuppressed(result.generatedStateDisturbance);
        assertSuppressed(result.generatedObs);
        assertSuppressed(result.generatedState);
    }

    private static long expectedReservedFilterResultDoubles(ModelFixture rep,
                                                            SimulationSmootherOptions spec) {
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();
        workspace.reserveComplexFilter(rep, spec.requiredFilterOptions());
        return workspace.retainedComplexFilterResultDoubleCount();
    }

    private static void assertAllZero(double[] values) {
        for (int i = 0; i < values.length; i++) {
            assertEquals(0.0, values[i], TOL, "non-zero at index " + i);
        }
    }

    private static ModelFixture buildZeroSystem(int nobs) {
        ModelFixture rep = ModelFixture.complex(1, 1, 1, nobs);
        double[] zero = {0.0, 0.0};
        double[] one = {1.0, 0.0};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(zero, t);
            rep.setObsIntercept(zero, t);
            rep.setObsCov(one, t);
            rep.setTransition(zero, t);
            rep.setStateIntercept(zero, t);
            rep.setSelection(one, t);
            rep.setStateCov(one, t);
            rep.setEndog(zero, t);
        }
        return rep;
    }

    private static ModelFixture buildRealZeroSystem(int nobs) {
        ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
        double[] zero = {0.0};
        double[] one = {1.0};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(zero, t);
            rep.setObsIntercept(zero, t);
            rep.setObsCov(one, t);
            rep.setTransition(zero, t);
            rep.setStateIntercept(zero, t);
            rep.setSelection(one, t);
            rep.setStateCov(one, t);
            rep.setEndog(zero, t);
        }
        return rep;
    }

    @Test
    void testComplexSimulationRealEmbeddingMatchesRealPath() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture realRep = ZKalmanFilterTest.buildExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ModelFixture complexRep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        double[] measurement = arangeTenths(realRep.nobs * realRep.kEndog);
        double[] state = arangeTenths(realRep.nobs * realRep.kPosdef);
        double[] initial = arangeTenths(realRep.kStates);
        SimulationSmootherOptions spec = SimulationSmootherOptions.defaults().withGeneratedOutputs();

        SimulationSmootherResult expected = SimulationSmootherEngine.simulate(realRep, InitialState.diffuse(), measurement, state, initial, spec);
        ZSimulationSmootherResult actual = SimulationSmootherEngine.simulateComplex(complexRep, InitialState.diffuse(), embedReal(measurement), embedReal(state), embedReal(initial), spec);

        assertRealEmbeddedMatches(expected, actual);
    }

    @Test
    void testComplexGeneratedSmoothWithPhasePoolFilterMatchesStandalone() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture complexRep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        double[] measurement = embedReal(arangeTenths(complexRep.nobs * complexRep.kEndog));
        double[] state = embedReal(arangeTenths(complexRep.nobs * complexRep.kPosdef));
        double[] initial = embedReal(arangeTenths(complexRep.kStates));
        SimulationSmootherOptions spec = SimulationSmootherOptions.defaults().withGeneratedOutputs();
        FilterOptions filterOptions = spec.requiredFilterOptions();
        SmootherOptions smootherOptions = spec.requiredSmootherOptions();

        ZSimulationSmootherResult complexSimulation = SimulationSmootherEngine.simulateComplex(complexRep, InitialState.diffuse(), measurement, state, initial, spec);
        ModelFixture complexGeneratedRep = ModelFixture.copyOf(complexRep);
        complexGeneratedRep.endog = complexSimulation.generatedObs;
        ZFilterResult standaloneGeneratedFilter = KalmanEngine.filterComplex(complexGeneratedRep, InitialState.diffuse(), filterOptions);
        ZSmootherResult standalone = SmootherEngine.smoothComplex(complexGeneratedRep, standaloneGeneratedFilter, smootherOptions);

        KalmanEngine.Workspace sharedFilterWorkspace = KalmanEngine.workspace();
        ZKalmanSmoother.Pool actualSmootherPool = new ZKalmanSmoother.Pool();
        ZFilterResult pooledActualFilter = KalmanEngine.filterComplexBorrowedUnsafe(complexRep, InitialState.diffuse(), sharedFilterWorkspace, filterOptions);
        SmootherEngine.smoothComplex(complexRep, pooledActualFilter, actualSmootherPool, smootherOptions);
        ZFilterResult pooledGeneratedFilter = KalmanEngine.filterComplexBorrowedUnsafe(complexGeneratedRep, InitialState.diffuse(), sharedFilterWorkspace, filterOptions);
        ZSmootherResult pooledFilterOnly = SmootherEngine.smoothComplex(complexGeneratedRep, pooledGeneratedFilter, smootherOptions);

        assertArrayEquals(
            activeSlice(standalone.smoothedObsDisturbance,
                standalone.smoothedObsDisturbanceBase(),
                standalone.smoothedObsDisturbanceLength()),
            activeSlice(pooledFilterOnly.smoothedObsDisturbance,
                pooledFilterOnly.smoothedObsDisturbanceBase(),
                pooledFilterOnly.smoothedObsDisturbanceLength()),
            TOL);
        assertArrayEquals(
            activeSlice(standalone.smoothedStateDisturbance,
                standalone.smoothedStateDisturbanceBase(),
                standalone.smoothedStateDisturbanceLength()),
            activeSlice(pooledFilterOnly.smoothedStateDisturbance,
                pooledFilterOnly.smoothedStateDisturbanceBase(),
                pooledFilterOnly.smoothedStateDisturbanceLength()),
            TOL);
    }

    @Test
    void testComplexSimulationRealEmbeddingHandlesMissingObservations() {
        double[] y = {10.2394, 1.0, 1.0, 1.0, 1.0, 1.0};
        ModelFixture realRep = ZKalmanFilterTest.buildExactDiffuseLocalLevel(y, 1.993, 8.253);
        ModelFixture complexRep = ZKalmanFilterTest.buildZExactDiffuseLocalLevel(y, 1.993, 8.253);
        realRep.setMissing(new boolean[]{true}, 2);
        complexRep.setMissing(new boolean[]{true}, 2);
        double[] measurement = arangeTenths(realRep.nobs * realRep.kEndog);
        double[] state = arangeTenths(realRep.nobs * realRep.kPosdef);
        double[] initial = arangeTenths(realRep.kStates);

        SimulationSmootherResult expected = SimulationSmootherEngine.simulate(realRep, InitialState.diffuse(), measurement, state, initial, SimulationSmootherOptions.defaults());
        ZSimulationSmootherResult actual = SimulationSmootherEngine.simulateComplex(complexRep, InitialState.diffuse(), embedReal(measurement), embedReal(state), embedReal(initial), SimulationSmootherOptions.defaults());

        assertGeneratedSuppressed(actual);
        assertComplexMatchesReal("simulatedMeasurementDisturbance", expected.simulatedMeasurementDisturbance, actual.simulatedMeasurementDisturbance);
        assertComplexMatchesReal("simulatedStateDisturbance", expected.simulatedStateDisturbance, actual.simulatedStateDisturbance);
        assertComplexMatchesReal("simulatedState", expected.simulatedState, actual.simulatedState);
    }

    @Test
    void testComplexSimulationRandomMatchesExplicitVariates() {
        int nobs = 10;
        ModelFixture rep = buildZeroSystem(nobs);
        InitialState init = InitialState.known(new double[]{0.0, 0.0}, new double[]{0.0, 0.0});

        Random expectedRng = new Random(1234L);
        double[] measurement = new double[2 * nobs * rep.kEndog];
        double[] state = new double[2 * nobs * rep.kPosdef];
        double[] initial = new double[2 * rep.kStates];
        for (int i = 0; i < measurement.length; i++) {
            measurement[i] = expectedRng.nextGaussian();
        }
        for (int i = 0; i < state.length; i++) {
            state[i] = expectedRng.nextGaussian();
        }
        for (int i = 0; i < initial.length; i++) {
            initial[i] = expectedRng.nextGaussian();
        }

        ZSimulationSmootherResult explicit = SimulationSmootherEngine.simulateComplex(rep, init, measurement, state, initial, SimulationSmootherOptions.defaults());
        ZSimulationSmootherResult fromRandom = SimulationSmootherEngine.simulateComplex(rep, init, new Random(1234L), SimulationSmootherOptions.defaults());

        assertGeneratedSuppressed(explicit);
        assertGeneratedSuppressed(fromRandom);
        assertSimulationResultMatches(explicit, fromRandom);
    }

    @Test
    void testComplexGeneratedInitialStateUsesCustomApproximateDiffuseVariance() {
        ModelFixture rep = buildZeroSystem(1);
        InitialState init = InitialState.approximateDiffuse(new double[]{1.5, 0.0}, 25.0);

        ZSimulationSmootherResult result = SimulationSmootherEngine.simulateComplex(
            rep,
            init,
            new double[]{0.0, 0.0},
            new double[]{0.0, 0.0},
            new double[]{2.0, 0.0},
            SimulationSmootherOptions.defaults().withGeneratedOutputs());

        assertEquals(11.5, result.generatedState[0], TOL);
        assertEquals(0.0, result.generatedState[1], TOL);
        assertEquals(0.0, result.generatedState[2], TOL);
        assertEquals(0.0, result.generatedState[3], TOL);
    }

    @Test
    void testComplexSimulationPoolMatchesStandalone() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        double[] measurement = embedReal(arangeTenths(rep.nobs * rep.kEndog));
        double[] state = embedReal(arangeTenths(rep.nobs * rep.kPosdef));
        double[] initial = embedReal(arangeTenths(rep.kStates));
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        ZSimulationSmootherResult expected = SimulationSmootherEngine.simulateComplex(rep, InitialState.diffuse(), measurement, state, initial, SimulationSmootherOptions.defaults());
        ZSimulationSmootherResult actual = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());

        assertSimulationResultMatches(expected, actual);
    }


    @Test
    void testEngineWorkspaceSharesSimulationResultArenaAcrossRealAndComplexRoutes() {
        ModelFixture realRep = buildRealZeroSystem(4);
        ModelFixture complexRep = buildZeroSystem(4);
        SimulationSmootherEngine.Workspace workspace = SimulationSmootherEngine.workspace();
        InitialState realInit = InitialState.known(new double[]{0.0}, new double[]{1.0});
        InitialState complexInit = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0});

        SimulationSmootherEngine.simulateBorrowedUnsafe(realRep, realInit, new Random(1), workspace,
            SimulationSmootherOptions.defaults());
        long realResultDoubles = workspace.retainedResultDoubleCount();
        assertTrue(realResultDoubles > 0L);
        assertEquals(0L, workspace.retainedComplexResultDoubleCount());

        workspace.releaseComplexRetainedResults();
        assertEquals(realResultDoubles, workspace.retainedResultDoubleCount());

        SimulationSmootherEngine.simulateComplexBorrowedUnsafe(complexRep, complexInit, new Random(1), workspace,
            SimulationSmootherOptions.defaults());
        assertEquals(0L, workspace.retainedResultDoubleCount());
        assertTrue(workspace.retainedComplexResultDoubleCount() > 0L);

        workspace.releaseRetainedResults();
        assertTrue(workspace.retainedComplexResultDoubleCount() > 0L);
        workspace.releaseComplexRetainedResults();
        assertEquals(0L, workspace.retainedResultDoubleCount());
        assertEquals(0L, workspace.retainedComplexResultDoubleCount());
    }

    @Test
    void testEngineWorkspaceInvalidatesExplicitVariateSimulationResultsAcrossRealAndComplexRoutes() {
        ModelFixture realRep = buildRealZeroSystem(4);
        ModelFixture complexRep = buildZeroSystem(4);
        SimulationSmootherEngine.Workspace workspace = SimulationSmootherEngine.workspace();
        InitialState realInit = InitialState.known(new double[]{0.0}, new double[]{1.0});
        InitialState complexInit = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0});
        double[] realMeasurement = new double[realRep.nobs * realRep.kEndog];
        double[] realState = new double[realRep.nobs * realRep.kPosdef];
        double[] realInitial = new double[realRep.kStates];
        double[] complexMeasurement = new double[2 * complexRep.nobs * complexRep.kEndog];
        double[] complexState = new double[2 * complexRep.nobs * complexRep.kPosdef];
        double[] complexInitial = new double[2 * complexRep.kStates];

        SimulationSmootherEngine.simulateBorrowedUnsafe(realRep, realInit,
            realMeasurement, realState, realInitial, workspace, SimulationSmootherOptions.defaults());
        assertTrue(workspace.retainedResultDoubleCount() > 0L);
        assertEquals(0L, workspace.retainedComplexResultDoubleCount());

        SimulationSmootherEngine.simulateComplexBorrowedUnsafe(complexRep, complexInit,
            complexMeasurement, complexState, complexInitial, workspace, SimulationSmootherOptions.defaults());
        assertEquals(0L, workspace.retainedResultDoubleCount());
        assertTrue(workspace.retainedComplexResultDoubleCount() > 0L);

        SimulationSmootherEngine.simulateBorrowedUnsafe(realRep, realInit,
            realMeasurement, realState, realInitial, workspace, SimulationSmootherOptions.defaults());
        assertTrue(workspace.retainedResultDoubleCount() > 0L);
        assertEquals(0L, workspace.retainedComplexResultDoubleCount());
    }
    @Test
    void testComplexSimulationPoolCopyPreservesPreviousResults() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        double[] measurement1 = embedReal(arangeTenths(rep.nobs * rep.kEndog));
        double[] state1 = embedReal(arangeTenths(rep.nobs * rep.kPosdef));
        double[] initial1 = embedReal(arangeTenths(rep.kStates));
        double[] measurement2 = new double[2 * rep.nobs * rep.kEndog];
        double[] state2 = new double[2 * rep.nobs * rep.kPosdef];
        double[] initial2 = new double[2 * rep.kStates];
        for (int i = 0; i < measurement2.length; i++) {
            measurement2[i] = -0.2 * (i + 1);
        }
        for (int i = 0; i < state2.length; i++) {
            state2[i] = 0.15 * (i + 1);
        }
        for (int i = 0; i < initial2.length; i++) {
            initial2[i] = -0.3 * (i + 1);
        }
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        ZSimulationSmootherResult expected1 = SimulationSmootherEngine.simulateComplex(rep, InitialState.diffuse(), measurement1, state1, initial1, SimulationSmootherOptions.defaults());
        ZSimulationSmootherResult actual1 = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), measurement1, state1, initial1, pool, SimulationSmootherOptions.defaults());
        ZSimulationSmootherResult snapshot1 = actual1.copy();

        ZSimulationSmootherResult expected2 = SimulationSmootherEngine.simulateComplex(rep, InitialState.diffuse(), measurement2, state2, initial2, SimulationSmootherOptions.defaults());
        ZSimulationSmootherResult actual2 = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), measurement2, state2, initial2, pool, SimulationSmootherOptions.defaults());

        assertSame(actual1, actual2);
        assertSimulationResultMatches(expected1, snapshot1);
        assertSimulationResultMatches(expected2, actual2);
    }

    @Test
    void testComplexSimulationPoolReuseReturnsBorrowedResultByDefault() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        double[] measurement = embedReal(arangeTenths(rep.nobs * rep.kEndog));
        double[] state = embedReal(arangeTenths(rep.nobs * rep.kPosdef));
        double[] initial = embedReal(arangeTenths(rep.kStates));
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        ZSimulationSmootherResult actual1 = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());
        ZSimulationSmootherResult actual2 = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());

        assertSame(actual1, actual2);
    }

    @Test
    void testComplexSimulationPoolReuseInvalidatesPreviousResult() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        double[] measurement1 = embedReal(arangeTenths(rep.nobs * rep.kEndog));
        double[] state1 = embedReal(arangeTenths(rep.nobs * rep.kPosdef));
        double[] initial1 = embedReal(arangeTenths(rep.kStates));
        double[] measurement2 = new double[2 * rep.nobs * rep.kEndog];
        double[] state2 = new double[2 * rep.nobs * rep.kPosdef];
        double[] initial2 = new double[2 * rep.kStates];
        for (int i = 0; i < measurement2.length; i++) {
            measurement2[i] = -0.2 * (i + 1);
        }
        for (int i = 0; i < state2.length; i++) {
            state2[i] = 0.15 * (i + 1);
        }
        for (int i = 0; i < initial2.length; i++) {
            initial2[i] = -0.3 * (i + 1);
        }
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        ZSimulationSmootherResult actual1 = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), measurement1, state1, initial1, pool, SimulationSmootherOptions.defaults());
        ZSimulationSmootherResult snapshot1 = actual1.copy();

        ZSimulationSmootherResult actual2 = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), measurement2, state2, initial2, pool, SimulationSmootherOptions.defaults());

        assertSame(actual1, actual2);
        assertFalse(Arrays.equals(
            activeSlice(snapshot1.simulatedState, snapshot1.simulatedStateBase(), snapshot1.simulatedStateLength()),
            activeSlice(actual1.simulatedState, actual1.simulatedStateBase(), actual1.simulatedStateLength())));
        assertArrayEquals(actual2.simulatedState(0, 0), actual1.simulatedState(0, 0), TOL);
    }

    @Test
    void testComplexSimulationPoolSupportsDiffuseInitialization() {
        double[] y = {10.2394, 1.0, 1.0, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLevel(y, 1.993, 8.253);
        rep.setMissing(new boolean[]{true}, 2);
        double[] measurement = embedReal(arangeTenths(rep.nobs * rep.kEndog));
        double[] state = embedReal(arangeTenths(rep.nobs * rep.kPosdef));
        double[] initial = embedReal(arangeTenths(rep.kStates));
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        ZSimulationSmootherResult expected = SimulationSmootherEngine.simulateComplex(rep, InitialState.diffuse(), measurement, state, initial, SimulationSmootherOptions.defaults());
        ZSimulationSmootherResult actual = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());

        assertSimulationResultMatches(expected, actual);
        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
    }

    @Test
    void testComplexRandomSimulationWarmPoolDoesNotGrowRetainedCountsAcrossRuns() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        ZSimulationSmootherResult first = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), new Random(123L), pool, SimulationSmootherOptions.defaults());

        long retainedResultDoubles = pool.retainedResultDoubleCount();
        long retainedWorkspaceDoubles = pool.retainedWorkspaceDoubleCount();
        long retainedVariateDoubles = pool.retainedVariateDoubleCount();
        long retainedNestedScratchDoubles = pool.retainedNestedScratchDoubleCount();
        long retainedNestedResultDoubles = pool.retainedNestedResultDoubleCount();
        long retainedTotalBytes = pool.retainedTotalByteCount();

        double[] generatedMeasurement = first.generatedMeasurementDisturbance;
        double[] generatedStateDisturbance = first.generatedStateDisturbance;
        double[] generatedObs = first.generatedObs;
        double[] generatedState = first.generatedState;
        double[] simulatedMeasurement = first.simulatedMeasurementDisturbance;
        double[] simulatedStateDisturbance = first.simulatedStateDisturbance;
        double[] simulatedState = first.simulatedState;

        assertEquals(0L, pool.retainedNestedScratchDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedNestedResultDoubleCount());

        ZSimulationSmootherResult second = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), new Random(456L), pool, SimulationSmootherOptions.defaults());

        assertSame(first, second);
        assertSame(generatedMeasurement, second.generatedMeasurementDisturbance);
        assertSame(generatedStateDisturbance, second.generatedStateDisturbance);
        assertSame(generatedObs, second.generatedObs);
        assertSame(generatedState, second.generatedState);
        assertSame(simulatedMeasurement, second.simulatedMeasurementDisturbance);
        assertSame(simulatedStateDisturbance, second.simulatedStateDisturbance);
        assertSame(simulatedState, second.simulatedState);
        assertEquals(retainedResultDoubles, pool.retainedResultDoubleCount());
        assertEquals(retainedWorkspaceDoubles, pool.retainedWorkspaceDoubleCount());
        assertEquals(retainedVariateDoubles, pool.retainedVariateDoubleCount());
        assertEquals(retainedNestedScratchDoubles, pool.retainedNestedScratchDoubleCount());
        assertEquals(retainedNestedResultDoubles, pool.retainedNestedResultDoubleCount());
        assertEquals(retainedTotalBytes, pool.retainedTotalByteCount());
    }

    @Test
    void testComplexSimulationPoolStateOnlySelectionShrinksRetainedResultBanks() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        double[] measurement = embedReal(arangeTenths(rep.nobs * rep.kEndog));
        double[] state = embedReal(arangeTenths(rep.nobs * rep.kPosdef));
        double[] initial = embedReal(arangeTenths(rep.kStates));
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        ZSimulationSmootherResult expectedStateOnly = SimulationSmootherEngine.simulateComplex(rep, InitialState.diffuse(), measurement, state, initial, SimulationSmootherOptions.stateOnly());
        ZSimulationSmootherResult full = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());
        long fullRetainedResultDoubles = pool.retainedResultDoubleCount();

        ZSimulationSmootherResult stateOnlyResult = SimulationSmootherEngine.simulateComplexBorrowed(rep, InitialState.diffuse(), measurement, state, initial, pool, SimulationSmootherOptions.stateOnly());

        assertSame(full, stateOnlyResult);
        assertEquals(fullRetainedResultDoubles, pool.retainedResultDoubleCount());
        assertArrayEquals(activeSlice(expectedStateOnly.simulatedState,
            expectedStateOnly.simulatedStateBase(), expectedStateOnly.simulatedStateLength()),
            activeSlice(stateOnlyResult.simulatedState,
                stateOnlyResult.simulatedStateBase(), stateOnlyResult.simulatedStateLength()), TOL);
        assertSuppressed(stateOnlyResult.simulatedMeasurementDisturbance);
        assertSuppressed(stateOnlyResult.simulatedStateDisturbance);
        assertGeneratedSuppressed(stateOnlyResult);
        assertTrue(stateOnlyResult.simulatedState.length > 0);
    }

    @Test
    void testComplexSimulationGeneratedOutputsAreOptIn() {
        ModelFixture rep = buildZeroSystem(8);
        InitialState init = InitialState.known(new double[]{0.0, 0.0}, new double[]{0.0, 0.0});
        double[] measurement = new double[2 * rep.nobs * rep.kEndog];
        double[] state = new double[2 * rep.nobs * rep.kPosdef];
        double[] initial = new double[2 * rep.kStates];

        ZSimulationSmootherResult lean = SimulationSmootherEngine.simulateComplex(rep, init, measurement, state, initial, SimulationSmootherOptions.defaults());
        ZSimulationSmootherResult generated = SimulationSmootherEngine.simulateComplex(rep, init, measurement, state, initial, SimulationSmootherOptions.defaults().withGeneratedOutputs());

        assertGeneratedSuppressed(lean);
        assertAllZero(generated.generatedMeasurementDisturbance);
        assertAllZero(generated.generatedStateDisturbance);
        assertAllZero(generated.generatedObs);
        assertAllZero(generated.generatedState);
        assertArrayEquals(lean.simulatedState, generated.simulatedState, TOL);
        assertArrayEquals(lean.simulatedMeasurementDisturbance, generated.simulatedMeasurementDisturbance, TOL);
        assertArrayEquals(lean.simulatedStateDisturbance, generated.simulatedStateDisturbance, TOL);
    }

    @Test
    void testComplexSimulationReserveOmitsFilteredFilterHistory() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        pool.reserve(rep, InitialState.diffuse(), SimulationSmootherOptions.defaults());

        assertEquals(expectedReservedFilterResultDoubles(rep, SimulationSmootherOptions.defaults()),
            pool.retainedNestedResultDoubleCount());
    }

    @Test
    void testComplexSimulationPoolSharesSmootherScratchAndOmitsReservedSmootherResultBanks() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();
        KalmanEngine.Workspace filterWorkspace = KalmanEngine.workspace();

        pool.reserve(rep, InitialState.diffuse(), SimulationSmootherOptions.defaults());
        filterWorkspace.reserveComplexFilter(rep, SimulationSmootherOptions.defaults().requiredFilterOptions());

        assertEquals(filterWorkspace.retainedComplexFilterResultDoubleCount(), pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());
        assertEquals(
            filterWorkspace.retainedComplexFilterScratchDoubleCount(),
            pool.retainedNestedScratchDoubleCount());
        assertTrue(pool.retainedTotalByteCount() > 0);
    }

    @Test
    void testComplexSimulationPoolReleaseRetainedScratchClearsSharedScratchAliases() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        pool.reserve(rep, InitialState.diffuse(), SimulationSmootherOptions.defaults());
        long retainedNestedResultDoubles = pool.retainedNestedResultDoubleCount();

        assertTrue(pool.retainedNestedScratchDoubleCount() > 0);

        pool.releaseRetainedScratch();

        assertEquals(0L, pool.retainedNestedScratchDoubleCount());
        assertEquals(retainedNestedResultDoubles, pool.retainedNestedResultDoubleCount());
    }

    @Test
    void testComplexSimulationRejectsUnivariateFilterProfile() {
        ModelFixture rep = buildZeroSystem(4);
        SimulationSmootherOptions spec = SimulationSmootherOptions.defaults().withUnivariateFilter();

        assertThrows(IllegalArgumentException.class,
            () -> SimulationSmootherEngine.simulateComplex(rep, InitialState.diffuse(),
                new double[2 * rep.nobs * rep.kEndog],
                new double[2 * rep.nobs * rep.kPosdef],
                new double[2 * rep.kStates],
                spec));
        assertThrows(IllegalArgumentException.class,
            () -> new ZSimulationSmoother.Pool().reserve(rep, InitialState.diffuse(), spec));
    }
}