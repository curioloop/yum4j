package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.kalman.filter.ZKalmanFilter;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.model.ModelFixture;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
import com.curioloop.yum4j.kalman.smooth.SimulationSmoother;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherSpec;
import com.curioloop.yum4j.kalman.smooth.ZKalmanSmoother;
import com.curioloop.yum4j.kalman.smooth.ZSmootherResult;
import com.curioloop.yum4j.kalman.smooth.ZSimulationSmoother;
import com.curioloop.yum4j.kalman.smooth.ZSimulationSmootherResult;
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
                                                            SimulationSmootherSpec spec) {
        ZKalmanFilter.Pool filterPool = new ZKalmanFilter.Pool();
        filterPool.reserve(rep, spec.requiredFilterSpec());
        return filterPool.retainedResultDoubleCount();
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

    @Test
    void testComplexSimulationRealEmbeddingMatchesRealPath() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture realRep = ZKalmanFilterTest.buildExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ModelFixture complexRep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        double[] measurement = arangeTenths(realRep.nobs * realRep.kEndog);
        double[] state = arangeTenths(realRep.nobs * realRep.kPosdef);
        double[] initial = arangeTenths(realRep.kStates);
        SimulationSmootherSpec spec = SimulationSmootherSpec.all().withGeneratedOutputs();

        SimulationSmootherResult expected = SimulationSmoother.simulate(realRep, StateInitialization.diffuse(), measurement, state, initial, spec);
        ZSimulationSmootherResult actual = ZSimulationSmoother.simulate(complexRep, StateInitialization.diffuse(), embedReal(measurement), embedReal(state), embedReal(initial), spec);

        assertRealEmbeddedMatches(expected, actual);
    }

    @Test
    void testComplexGeneratedSmoothWithPhasePoolFilterMatchesStandalone() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture complexRep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        double[] measurement = embedReal(arangeTenths(complexRep.nobs * complexRep.kEndog));
        double[] state = embedReal(arangeTenths(complexRep.nobs * complexRep.kPosdef));
        double[] initial = embedReal(arangeTenths(complexRep.kStates));
        SimulationSmootherSpec spec = SimulationSmootherSpec.all().withGeneratedOutputs();
        FilterSpec filterSpec = spec.requiredFilterSpec();
        SmootherSpec smootherSpec = spec.requiredSmootherSpec();

        ZSimulationSmootherResult complexSimulation = ZSimulationSmoother.simulate(complexRep, StateInitialization.diffuse(), measurement, state, initial, spec);
        ModelFixture complexGeneratedRep = ModelFixture.copyOf(complexRep);
        complexGeneratedRep.endog = complexSimulation.generatedObs;
        ZFilterResult standaloneGeneratedFilter = ZKalmanFilter.filter(complexGeneratedRep, StateInitialization.diffuse(), null, filterSpec);
        ZSmootherResult standalone = ZKalmanSmoother.smooth(complexGeneratedRep, standaloneGeneratedFilter, null, smootherSpec);

        ZKalmanFilter.Pool sharedFilterPool = new ZKalmanFilter.Pool();
        ZKalmanSmoother.Pool actualSmootherPool = new ZKalmanSmoother.Pool();
        ZFilterResult pooledActualFilter = ZKalmanFilter.filter(complexRep, StateInitialization.diffuse(), sharedFilterPool, filterSpec);
        ZKalmanSmoother.smooth(complexRep, pooledActualFilter, actualSmootherPool, smootherSpec);
        ZFilterResult pooledGeneratedFilter = ZKalmanFilter.filter(complexGeneratedRep, StateInitialization.diffuse(), sharedFilterPool, filterSpec);
        ZSmootherResult pooledFilterOnly = ZKalmanSmoother.smooth(complexGeneratedRep, pooledGeneratedFilter, null, smootherSpec);

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

        SimulationSmootherResult expected = SimulationSmoother.simulate(realRep, StateInitialization.diffuse(), measurement, state, initial);
        ZSimulationSmootherResult actual = ZSimulationSmoother.simulate(complexRep, StateInitialization.diffuse(), embedReal(measurement), embedReal(state), embedReal(initial));

        assertGeneratedSuppressed(actual);
        assertComplexMatchesReal("simulatedMeasurementDisturbance", expected.simulatedMeasurementDisturbance, actual.simulatedMeasurementDisturbance);
        assertComplexMatchesReal("simulatedStateDisturbance", expected.simulatedStateDisturbance, actual.simulatedStateDisturbance);
        assertComplexMatchesReal("simulatedState", expected.simulatedState, actual.simulatedState);
    }

    @Test
    void testComplexSimulationRandomMatchesExplicitVariates() {
        int nobs = 10;
        ModelFixture rep = buildZeroSystem(nobs);
        StateInitialization init = StateInitialization.known(new double[]{0.0, 0.0}, new double[]{0.0, 0.0});

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

        ZSimulationSmootherResult explicit = ZSimulationSmoother.simulate(rep, init, measurement, state, initial);
        ZSimulationSmootherResult fromRandom = ZSimulationSmoother.simulate(rep, init, new Random(1234L));

        assertGeneratedSuppressed(explicit);
        assertGeneratedSuppressed(fromRandom);
        assertSimulationResultMatches(explicit, fromRandom);
    }

    @Test
    void testComplexGeneratedInitialStateUsesCustomApproximateDiffuseVariance() {
        ModelFixture rep = buildZeroSystem(1);
        StateInitialization init = StateInitialization.approximateDiffuse(new double[]{1.5, 0.0}, 25.0);

        ZSimulationSmootherResult result = ZSimulationSmoother.simulate(
            rep,
            init,
            new double[]{0.0, 0.0},
            new double[]{0.0, 0.0},
            new double[]{2.0, 0.0},
            SimulationSmootherSpec.all().withGeneratedOutputs());

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

        ZSimulationSmootherResult expected = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, SimulationSmootherSpec.all());
        ZSimulationSmootherResult actual = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());

        assertSimulationResultMatches(expected, actual);
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

        ZSimulationSmootherResult expected1 = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement1, state1, initial1, SimulationSmootherSpec.all());
        ZSimulationSmootherResult actual1 = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement1, state1, initial1, pool, SimulationSmootherSpec.all());
        ZSimulationSmootherResult snapshot1 = actual1.copy();

        ZSimulationSmootherResult expected2 = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement2, state2, initial2, SimulationSmootherSpec.all());
        ZSimulationSmootherResult actual2 = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement2, state2, initial2, pool, SimulationSmootherSpec.all());

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

        ZSimulationSmootherResult actual1 = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());
        ZSimulationSmootherResult actual2 = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());

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

        ZSimulationSmootherResult actual1 = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement1, state1, initial1, pool, SimulationSmootherSpec.all());
        ZSimulationSmootherResult snapshot1 = actual1.copy();

        ZSimulationSmootherResult actual2 = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement2, state2, initial2, pool, SimulationSmootherSpec.all());

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

        ZSimulationSmootherResult expected = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, SimulationSmootherSpec.all());
        ZSimulationSmootherResult actual = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());

        assertSimulationResultMatches(expected, actual);
        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
    }

    @Test
    void testComplexRandomSimulationWarmPoolDoesNotGrowRetainedCountsAcrossRuns() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        ZSimulationSmootherResult first = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), new Random(123L), pool, SimulationSmootherSpec.all());

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

        assertTrue(pool.retainedNestedScratchDoubleCount() > 0L);
        assertEquals(0L, pool.retainedVariateDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedNestedResultDoubleCount());

        ZSimulationSmootherResult second = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), new Random(456L), pool, SimulationSmootherSpec.all());

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

        ZSimulationSmootherResult expectedStateOnly = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, SimulationSmootherSpec.stateOnly());
        ZSimulationSmootherResult full = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());
        long fullRetainedResultDoubles = pool.retainedResultDoubleCount();

        ZSimulationSmootherResult stateOnlyResult = ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, pool, SimulationSmootherSpec.stateOnly());

        assertSame(full, stateOnlyResult);
        assertTrue(pool.retainedResultDoubleCount() < fullRetainedResultDoubles);
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
        StateInitialization init = StateInitialization.known(new double[]{0.0, 0.0}, new double[]{0.0, 0.0});
        double[] measurement = new double[2 * rep.nobs * rep.kEndog];
        double[] state = new double[2 * rep.nobs * rep.kPosdef];
        double[] initial = new double[2 * rep.kStates];

        ZSimulationSmootherResult lean = ZSimulationSmoother.simulate(rep, init, measurement, state, initial, SimulationSmootherSpec.all());
        ZSimulationSmootherResult legacy = ZSimulationSmoother.simulate(rep, init, measurement, state, initial, SimulationSmootherSpec.all().withGeneratedOutputs());

        assertGeneratedSuppressed(lean);
        assertAllZero(legacy.generatedMeasurementDisturbance);
        assertAllZero(legacy.generatedStateDisturbance);
        assertAllZero(legacy.generatedObs);
        assertAllZero(legacy.generatedState);
        assertArrayEquals(lean.simulatedState, legacy.simulatedState, TOL);
        assertArrayEquals(lean.simulatedMeasurementDisturbance, legacy.simulatedMeasurementDisturbance, TOL);
        assertArrayEquals(lean.simulatedStateDisturbance, legacy.simulatedStateDisturbance, TOL);
    }

    @Test
    void testComplexSimulationReserveOmitsFilteredFilterHistory() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        pool.reserve(rep, StateInitialization.diffuse(), SimulationSmootherSpec.all());

        assertEquals(expectedReservedFilterResultDoubles(rep, SimulationSmootherSpec.all()),
            pool.retainedNestedResultDoubleCount());
    }

    @Test
    void testComplexSimulationPoolSharesSmootherScratchAndOmitsReservedSmootherResultBanks() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();
        ZKalmanFilter.Pool filterPool = new ZKalmanFilter.Pool();

        pool.reserve(rep, StateInitialization.diffuse(), SimulationSmootherSpec.all());
        filterPool.reserve(rep, SimulationSmootherSpec.all().requiredFilterSpec());

        assertEquals(filterPool.retainedResultDoubleCount(), pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());
        assertEquals(
            filterPool.retainedScratchDoubleCount(),
            pool.retainedNestedScratchDoubleCount());
        assertTrue(pool.retainedTotalByteCount() > 0);
    }

    @Test
    void testComplexSimulationPoolReleaseRetainedScratchClearsSharedScratchAliases() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();

        pool.reserve(rep, StateInitialization.diffuse(), SimulationSmootherSpec.all());
        long retainedNestedResultDoubles = pool.retainedNestedResultDoubleCount();

        assertTrue(pool.retainedNestedScratchDoubleCount() > 0);

        pool.releaseRetainedScratch();

        assertEquals(0L, pool.retainedNestedScratchDoubleCount());
        assertEquals(retainedNestedResultDoubles, pool.retainedNestedResultDoubleCount());
    }

    @Test
    void testComplexSimulationRejectsUnivariateFilterProfile() {
        ModelFixture rep = buildZeroSystem(4);
        SimulationSmootherSpec spec = SimulationSmootherSpec.all().withUnivariateFilter();

        assertThrows(IllegalArgumentException.class,
            () -> ZSimulationSmoother.simulate(rep, StateInitialization.diffuse(),
                new double[2 * rep.nobs * rep.kEndog],
                new double[2 * rep.nobs * rep.kPosdef],
                new double[2 * rep.kStates],
                spec));
        assertThrows(IllegalArgumentException.class,
            () -> new ZSimulationSmoother.Pool().reserve(rep, StateInitialization.diffuse(), spec));
    }
}