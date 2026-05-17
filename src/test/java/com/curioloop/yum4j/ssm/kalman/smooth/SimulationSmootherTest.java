package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.fixtures.StatsmodelsResources;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class SimulationSmootherTest {

    private static final double TOL = 1e-7;

    private static ModelFixture buildReferenceVar(double[] y) {
        int nobs = y.length / 3;
        ModelFixture rep = new ModelFixture(3, 3, 3, nobs);
        double[] Z = {
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
        };
        double[] H = {
                0.0000640649, 0.0, 0.0,
                0.0, 0.0000572802, 0.0,
                0.0, 0.0, 0.0017088585
        };
        double[] T = {
                -0.1119908792, 0.8441841604, 0.0238725303,
                0.2629347724, 0.4996718412, -0.0173023305,
                -3.2192369082, 4.1536028244, 0.4514379215
        };
        double[] R = {
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
        };
        double[] Q = {
                0.0000640649, 0.0000388496, 0.0002148769,
                0.0000388496, 0.0000572802, 0.000001555,
                0.0002148769, 0.000001555, 0.0017088585
        };
        double[] zero3 = {0.0, 0.0, 0.0};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(zero3, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(zero3, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t * 3], y[t * 3 + 1], y[t * 3 + 2]}, t);
        }
        return rep;
    }

    private static List<String[]> readReferenceCsv(String fileName) {
        return StatsmodelsResources.readStatespaceResultsCsv(fileName);
    }

    private static int csvColumnIndex(String[] header, String name) {
        for (int i = 0; i < header.length; i++) {
            if (header[i].equals(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Missing csv column: " + name);
    }

    private static double csvValue(String[] row, int index) {
        return Double.parseDouble(row[index]);
    }

    private static double[] reconstructObservedData() {
        List<String[]> rows = readReferenceCsv("results_simulation_smoothing0.csv");
        String[] header = rows.get(0);
        int signal1 = csvColumnIndex(header, "signal1");
        int signal2 = csvColumnIndex(header, "signal2");
        int signal3 = csvColumnIndex(header, "signal3");
        int eps1 = csvColumnIndex(header, "eps1");
        int eps2 = csvColumnIndex(header, "eps2");
        int eps3 = csvColumnIndex(header, "eps3");
        double[] y = new double[(rows.size() - 1) * 3];
        for (int t = 1; t < rows.size(); t++) {
            String[] row = rows.get(t);
            int off = (t - 1) * 3;
            y[off] = csvValue(row, signal1) + csvValue(row, eps1);
            y[off + 1] = csvValue(row, signal2) + csvValue(row, eps2);
            y[off + 2] = csvValue(row, signal3) + csvValue(row, eps3);
        }
        return y;
    }

    private static double[][] readMacrodata() {
        return StatsmodelsResources.readMacrodata();
    }

    private static double[] reconstructFullObservedData() {
        double[][] raw = readMacrodata();
        int nobs = raw.length - 1;
        double[] y = new double[nobs * 3];
        for (int t = 0; t < nobs; t++) {
            int off = t * 3;
            y[off] = Math.log(raw[t + 1][2]) - Math.log(raw[t][2]);
            y[off + 1] = Math.log(raw[t + 1][3]) - Math.log(raw[t][3]);
            y[off + 2] = Math.log(raw[t + 1][4]) - Math.log(raw[t][4]);
        }
        return y;
    }

    private static void applyMissingPattern(ModelFixture rep, int scenario) {
        boolean[] missing = new boolean[rep.kEndog];
        switch (scenario) {
            case 4 -> {
                for (int t = 0; t < 50; t++) {
                    rep.setMissing(new boolean[]{true, true, true}, t);
                }
            }
            case 5 -> {
                for (int t = 0; t < 50; t++) {
                    rep.setMissing(new boolean[]{true, false, false}, t);
                }
            }
            case 6 -> {
                for (int t = 0; t < rep.nobs; t++) {
                    missing[0] = t < 50 || (t >= 119 && t < 130) || t >= 192;
                    missing[1] = (t >= 19 && t < 70) || t >= 192;
                    missing[2] = (t >= 39 && t < 90) || (t >= 119 && t < 130) || t >= 192;
                    rep.setMissing(missing.clone(), t);
                }
            }
            default -> throw new IllegalArgumentException("Unsupported missing-data scenario: " + scenario);
        }
    }

    private static double[] readVariates(String fileName) {
        List<String[]> rows = readReferenceCsv(fileName);
        double[] values = new double[rows.size() - 1];
        for (int i = 1; i < rows.size(); i++) {
            values[i - 1] = csvValue(rows.get(i), 0);
        }
        return values;
    }

    private static SimulationSmootherResult simulateWithRecordedVariates(ModelFixture rep,
                                                                         InitialState init,
                                                                         SimulationSmootherOptions spec) {
        double[] variates = readVariates("results_simulation_smoothing3_variates.csv");
        int measurementCount = rep.nobs * rep.kEndog;
        int stateCount = rep.nobs * rep.kPosdef;
        int initialCount = rep.kStates;
        assertEquals(measurementCount + stateCount + initialCount, variates.length);

        double[] measurement = new double[measurementCount];
        double[] state = new double[stateCount];
        double[] initial = new double[initialCount];
        System.arraycopy(variates, 0, measurement, 0, measurementCount);
        System.arraycopy(variates, measurementCount, state, 0, stateCount);
        System.arraycopy(variates, measurementCount + stateCount, initial, 0, initialCount);
        return SimulationSmootherEngine.simulate(rep, init, measurement, state, initial, spec.withUnivariateFilter());
    }

    private static void assertArrayClose(double[] expected, double[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], TOL, "array mismatch at index " + i);
        }
    }

    private static void assertArrayClose(double[] expected, int expectedBase, int expectedLength,
                                         double[] actual, int actualBase, int actualLength) {
        assertEquals(expectedLength, actualLength);
        for (int i = 0; i < expectedLength; i++) {
            assertEquals(expected[expectedBase + i], actual[actualBase + i], TOL, "array mismatch at index " + i);
        }
    }

    private static double[] arangeTenths(int length) {
        double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = i / 10.0;
        }
        return values;
    }

    private static void assertRowMajorMatchesCsv(double[] values, int width,
                                                 String fileName, String... columns) {
        List<String[]> rows = readReferenceCsv(fileName);
        String[] header = rows.get(0);
        int[] indices = new int[columns.length];
        for (int i = 0; i < columns.length; i++) {
            indices[i] = csvColumnIndex(header, columns[i]);
        }
        for (int t = 1; t < rows.size(); t++) {
            String[] row = rows.get(t);
            int off = (t - 1) * width;
            for (int i = 0; i < columns.length; i++) {
                assertEquals(csvValue(row, indices[i]), values[off + i], TOL,
                        fileName + " mismatch at t=" + (t - 1) + " col=" + columns[i]);
            }
        }
    }

    private static void assertAllZero(double[] values) {
        for (int i = 0; i < values.length; i++) {
            assertEquals(0.0, values[i], TOL, "non-zero at index " + i);
        }
    }

    private static void assertSuppressed(double[] values) {
        assertEquals(0, values.length);
    }

    private static void assertGeneratedSuppressed(SimulationSmootherResult result) {
        assertSuppressed(result.generatedMeasurementDisturbance);
        assertSuppressed(result.generatedStateDisturbance);
        assertSuppressed(result.generatedObs);
        assertSuppressed(result.generatedState);
    }

    private static void assertSimulationSurfaceLengths(ModelFixture rep,
                                                       SimulationSmootherOptions options,
                                                       SimulationSmootherResult result) {
        boolean storeGenerated = options.storesGeneratedOutputs();
        boolean storeState = options.includes(SimulationSmootherOptions.Surface.STATE);
        boolean storeDisturbance = options.includes(SimulationSmootherOptions.Surface.DISTURBANCE);

        assertEquals(storeGenerated ? rep.kEndog * rep.nobs : 0,
            result.generatedMeasurementDisturbanceLength(), "generated measurement disturbance length");
        assertEquals(storeGenerated ? rep.kPosdef * rep.nobs : 0,
            result.generatedStateDisturbanceLength(), "generated state disturbance length");
        assertEquals(storeGenerated ? rep.kEndog * rep.nobs : 0,
            result.generatedObsLength(), "generated observation length");
        assertEquals(storeGenerated ? rep.kStates * (rep.nobs + 1) : 0,
            result.generatedStateLength(), "generated state length");
        assertEquals(storeDisturbance ? rep.kEndog * rep.nobs : 0,
            result.simulatedMeasurementDisturbanceLength(), "simulated measurement disturbance length");
        assertEquals(storeDisturbance ? rep.kPosdef * rep.nobs : 0,
            result.simulatedStateDisturbanceLength(), "simulated state disturbance length");
        assertEquals(storeState ? rep.kStates * rep.nobs : 0,
            result.simulatedStateLength(), "simulated state length");
    }

    private static long expectedReservedFilterResultDoubles(ModelFixture rep,
                                                            SimulationSmootherOptions spec) {
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();
        if (spec.usesUnivariateFilter()) {
            workspace.reserveUnivariateFilter(rep, spec.requiredFilterOptions());
            return workspace.retainedUnivariateResultDoubleCount();
        }
        workspace.reserveFilter(rep, spec.requiredFilterOptions());
        return workspace.retainedFilterResultDoubleCount();
    }

    private static void assertReservedSmootherResultBanksOmitted(ModelFixture rep,
                                                                 InitialState init,
                                                                 SimulationSmootherOptions spec) {
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();
        pool.reserve(rep, init, spec);

        assertEquals(expectedReservedFilterResultDoubles(rep, spec), pool.retainedNestedResultDoubleCount());
    }

    private static void assertSimulationResultMatches(SimulationSmootherResult expected,
                                                      SimulationSmootherResult actual) {
        assertArrayClose(expected.generatedMeasurementDisturbance,
            expected.generatedMeasurementDisturbanceBase(), expected.generatedMeasurementDisturbanceLength(),
            actual.generatedMeasurementDisturbance,
            actual.generatedMeasurementDisturbanceBase(), actual.generatedMeasurementDisturbanceLength());
        assertArrayClose(expected.generatedStateDisturbance,
            expected.generatedStateDisturbanceBase(), expected.generatedStateDisturbanceLength(),
            actual.generatedStateDisturbance,
            actual.generatedStateDisturbanceBase(), actual.generatedStateDisturbanceLength());
        assertArrayClose(expected.generatedObs,
            expected.generatedObsBase(), expected.generatedObsLength(),
            actual.generatedObs,
            actual.generatedObsBase(), actual.generatedObsLength());
        assertArrayClose(expected.generatedState,
            expected.generatedStateBase(), expected.generatedStateLength(),
            actual.generatedState,
            actual.generatedStateBase(), actual.generatedStateLength());
        assertArrayClose(expected.simulatedMeasurementDisturbance,
            expected.simulatedMeasurementDisturbanceBase(), expected.simulatedMeasurementDisturbanceLength(),
            actual.simulatedMeasurementDisturbance,
            actual.simulatedMeasurementDisturbanceBase(), actual.simulatedMeasurementDisturbanceLength());
        assertArrayClose(expected.simulatedStateDisturbance,
            expected.simulatedStateDisturbanceBase(), expected.simulatedStateDisturbanceLength(),
            actual.simulatedStateDisturbance,
            actual.simulatedStateDisturbanceBase(), actual.simulatedStateDisturbanceLength());
        assertArrayClose(expected.simulatedState,
            expected.simulatedStateBase(), expected.simulatedStateLength(),
            actual.simulatedState,
            actual.simulatedStateBase(), actual.simulatedStateLength());
    }

    private static void assertSimulationSnapshot(SimulationSmootherResult result, int t,
                                                 double[] expectedState,
                                                 double[] expectedMeasurementDisturbance,
                                                 double[] expectedStateDisturbance) {
        for (int i = 0; i < expectedState.length; i++) {
            assertEquals(expectedState[i], result.simulatedState(i, t), TOL,
                    "state mismatch at t=" + t + " i=" + i);
        }

        for (int i = 0; i < expectedMeasurementDisturbance.length; i++) {
            assertEquals(expectedMeasurementDisturbance[i],
                    result.simulatedMeasurementDisturbance(i, t), TOL,
                    "measurement disturbance mismatch at t=" + t + " i=" + i);
        }

        for (int i = 0; i < expectedStateDisturbance.length; i++) {
            assertEquals(expectedStateDisturbance[i],
                    result.simulatedStateDisturbance(i, t), TOL,
                    "state disturbance mismatch at t=" + t + " i=" + i);
        }
    }

    private static ModelFixture buildDiffuseInterceptModel(double intercept, boolean withMissing) {
        int nobs = 10;
        ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
        double[] Z = {1.0};
        double[] H = {1.0};
        double[] T = {0.0};
        double[] c = {intercept};
        double[] R = {1.0};
        double[] Q = {1.0};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(new double[]{0.0}, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{intercept}, t);
        }
        if (withMissing) {
            rep.setMissing(new boolean[]{true}, 5);
        }
        return rep;
    }

    private static ModelFixture buildZeroInitializationSystem(int nobs) {
        ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(new double[]{0.0}, t);
            rep.setObsIntercept(new double[]{0.0}, t);
            rep.setObsCov(new double[]{1.0}, t);
            rep.setTransition(new double[]{0.0}, t);
            rep.setStateIntercept(new double[]{0.0}, t);
            rep.setSelection(new double[]{1.0}, t);
            rep.setStateCov(new double[]{0.0}, t);
            rep.setEndog(new double[]{0.0}, t);
        }
        return rep;
    }

    @Test
    void testSimulationSmootherEngineOptionsMatchKernelOptions() {
        ModelFixture rep = buildDiffuseInterceptModel(0.75, false);
        InitialState init = InitialState.approximateDiffuse();
        double[] measurement = arangeTenths(rep.nobs * rep.kEndog);
        double[] state = arangeTenths(rep.nobs * rep.kPosdef);
        double[] initial = arangeTenths(rep.kStates);

        SimulationSmootherResult expected = SimulationSmootherEngine.simulate(rep,
            init,
            measurement,
            state,
            initial,
            SimulationSmootherOptions.stateOnly().withUnivariateFilter().withGeneratedOutputs());
        SimulationSmootherResult actual = SimulationSmootherEngine.simulate(rep,
            init,
            measurement,
            state,
            initial,
            SimulationSmootherOptions.builder()
                .univariateFilter(true)
                .retainOnly(SimulationSmootherOptions.Surface.STATE,
                    SimulationSmootherOptions.Surface.GENERATED_OUTPUTS)
                .build());

        assertSimulationResultMatches(expected, actual);
    }

    @Test
    void testCfaSimulationSmootherContractAndStateOnlyKernel() {
        ModelFixture rep = buildDiffuseInterceptModel(0.75, false);
        InitialState known = InitialState.known(new double[]{0.0}, new double[]{1.0});
        double[] measurement = arangeTenths(rep.nobs * rep.kEndog);
        double[] state = arangeTenths(rep.nobs * rep.kPosdef);
        double[] initial = arangeTenths(rep.kStates);
        SimulationSmootherOptions defaultCfa = SimulationSmootherOptions.defaults()
            .withMethod(SimulationSmootherOptions.Method.CFA);

        SimulationSmootherResult defaultResult = SimulationSmootherEngine.simulate(rep, known,
            measurement, state, initial, defaultCfa);
        assertEquals(rep.nobs * rep.kStates, defaultResult.simulatedStateLength());
        assertSuppressed(defaultResult.simulatedMeasurementDisturbance);
        assertSuppressed(defaultResult.simulatedStateDisturbance);
        assertGeneratedSuppressed(defaultResult);

        UnsupportedOperationException outputFailure = assertThrows(UnsupportedOperationException.class,
            () -> SimulationSmootherEngine.simulate(rep, known, measurement, state, initial,
                SimulationSmootherOptions.disturbanceOnly().withMethod(SimulationSmootherOptions.Method.CFA)));
        assertTrue(outputFailure.getMessage().contains("state-vector output only"));

        SimulationSmootherOptions stateOnlyCfa = SimulationSmootherOptions.stateOnly()
            .withMethod(SimulationSmootherOptions.Method.CFA);

        UnsupportedOperationException diffuseFailure = assertThrows(UnsupportedOperationException.class,
            () -> SimulationSmootherEngine.simulate(rep, InitialState.diffuse(1),
                measurement, state, initial, stateOnlyCfa));
        assertTrue(diffuseFailure.getMessage().contains("diffuse initialization"));

        ModelFixture singleObservation = buildZeroInitializationSystem(1);
        UnsupportedOperationException singleObservationFailure = assertThrows(UnsupportedOperationException.class,
            () -> SimulationSmootherEngine.simulate(singleObservation, known,
                arangeTenths(singleObservation.nobs * singleObservation.kEndog),
                arangeTenths(singleObservation.nobs * singleObservation.kPosdef),
                arangeTenths(singleObservation.kStates),
                stateOnlyCfa));
        assertTrue(singleObservationFailure.getMessage().contains("single observation"));

        SimulationSmootherResult cfa = SimulationSmootherEngine.simulate(rep, known,
            measurement, state, initial, stateOnlyCfa);
        SimulationSmootherResult repeated = SimulationSmootherEngine.simulate(rep, known,
            measurement, state, initial, stateOnlyCfa);

        assertEquals(rep.nobs * rep.kStates, cfa.simulatedStateLength());
        assertSuppressed(cfa.simulatedMeasurementDisturbance);
        assertSuppressed(cfa.simulatedStateDisturbance);
        assertGeneratedSuppressed(cfa);
        assertArrayClose(cfa.simulatedState, cfa.simulatedStateBase(), cfa.simulatedStateLength(),
            repeated.simulatedState, repeated.simulatedStateBase(), repeated.simulatedStateLength());
    }

    @Test
    void testCfaStateOnlyHandlesMoreEndogenousVariablesThanStates() {
        int nobs = 4;
        ModelFixture rep = new ModelFixture(2, 1, 1, nobs);
        double[] design = {1.0, -0.35};
        double[] obsCov = {
            0.7, 0.1,
            0.1, 0.9
        };
        double[] transition = {0.6};
        double[] stateIntercept = {0.05};
        double[] selection = {1.0};
        double[] stateCov = {0.4};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(design, t);
            rep.setObsIntercept(new double[]{0.0, 0.0}, t);
            rep.setObsCov(obsCov, t);
            rep.setTransition(transition, t);
            rep.setStateIntercept(stateIntercept, t);
            rep.setSelection(selection, t);
            rep.setStateCov(stateCov, t);
            rep.setEndog(new double[]{1.0 + 0.2 * t, -0.4 + 0.1 * t}, t);
        }
        InitialState known = InitialState.known(new double[]{0.2}, new double[]{1.1});
        SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep, known,
            new double[rep.nobs * rep.kEndog],
            new double[]{-0.2, 0.1, 0.3, -0.1},
            new double[rep.kStates],
            SimulationSmootherOptions.stateOnly().withMethod(SimulationSmootherOptions.Method.CFA));

        assertEquals(nobs, result.simulatedStateLength());
        for (int i = 0; i < result.simulatedStateLength(); i++) {
            assertTrue(Double.isFinite(result.simulatedState[result.simulatedStateBase() + i]));
        }
    }

    @Test
    void testComplexCfaSimulationSmootherContractAndStateOnlyKernel() {
        ModelFixture rep = ModelFixture.complex(1, 1, 1, 3);
        for (int t = 0; t < rep.nobs; t++) {
            rep.setDesign(new double[]{1.0, 0.0}, t);
            rep.setObsIntercept(new double[]{0.0, 0.0}, t);
            rep.setObsCov(new double[]{0.6, 0.0}, t);
            rep.setTransition(new double[]{0.8, 0.0}, t);
            rep.setStateIntercept(new double[]{0.05, 0.0}, t);
            rep.setSelection(new double[]{1.0, 0.0}, t);
            rep.setStateCov(new double[]{0.4, 0.0}, t);
            rep.setEndog(new double[]{1.0 + 0.2 * t, 0.1 - 0.05 * t}, t);
        }
        InitialState known = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0});
        SimulationSmootherOptions stateOnlyCfa = SimulationSmootherOptions.stateOnly()
            .withMethod(SimulationSmootherOptions.Method.CFA);
        double[] measurement = arangeTenths(2 * rep.nobs * rep.kEndog);
        double[] state = arangeTenths(2 * rep.nobs * rep.kStates);
        double[] initial = arangeTenths(2 * rep.kStates);

        ZSimulationSmootherResult cfa = SimulationSmootherEngine.simulateComplex(rep, known,
            measurement, state, initial, stateOnlyCfa);
        ZSimulationSmootherResult repeated = SimulationSmootherEngine.simulateComplex(rep, known,
            measurement, state, initial, stateOnlyCfa);

        assertEquals(2 * rep.nobs * rep.kStates, cfa.simulatedStateLength());
        assertSuppressed(cfa.simulatedMeasurementDisturbance);
        assertSuppressed(cfa.simulatedStateDisturbance);
        assertSuppressed(cfa.generatedMeasurementDisturbance);
        assertSuppressed(cfa.generatedStateDisturbance);
        assertSuppressed(cfa.generatedObs);
        assertSuppressed(cfa.generatedState);
        assertArrayClose(cfa.simulatedState, cfa.simulatedStateBase(), cfa.simulatedStateLength(),
            repeated.simulatedState, repeated.simulatedStateBase(), repeated.simulatedStateLength());

        UnsupportedOperationException outputFailure = assertThrows(UnsupportedOperationException.class,
            () -> SimulationSmootherEngine.simulateComplex(rep, known,
                measurement, state, initial,
                SimulationSmootherOptions.disturbanceOnly().withMethod(SimulationSmootherOptions.Method.CFA)));
        assertTrue(outputFailure.getMessage().contains("state-vector output only"));
    }

    @Test
    void testComplexCfaSimulationSmootherHandlesNonRealStateSelection() {
        ModelFixture rep = ModelFixture.complex(1, 1, 1, 3);
        for (int t = 0; t < rep.nobs; t++) {
            rep.setDesign(new double[]{0.9, -0.3}, t);
            rep.setObsIntercept(new double[]{0.0, 0.0}, t);
            rep.setObsCov(new double[]{0.6, 0.0}, t);
            rep.setTransition(new double[]{0.45, 0.2}, t);
            rep.setStateIntercept(new double[]{0.05, -0.02}, t);
            rep.setSelection(new double[]{1.0, 1.0}, t);
            rep.setStateCov(new double[]{0.4, 0.0}, t);
            rep.setEndog(new double[]{1.0 + 0.2 * t, 0.1 - 0.05 * t}, t);
        }
        InitialState known = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0});
        SimulationSmootherOptions stateOnlyCfa = SimulationSmootherOptions.stateOnly()
            .withMethod(SimulationSmootherOptions.Method.CFA);
        double[] variates = arangeTenths(2 * rep.nobs * rep.kStates);

        ZSimulationSmootherResult result = SimulationSmootherEngine.simulateComplex(rep, known,
            null, variates, null, stateOnlyCfa);

        assertEquals(2 * rep.nobs * rep.kStates, result.simulatedStateLength());
        for (int i = 0; i < result.simulatedStateLength(); i++) {
            assertTrue(Double.isFinite(result.simulatedState[result.simulatedStateBase() + i]));
        }
    }

    @Test
    void testCfaPosteriorMomentsMatchStatsmodelsScalarReference() {
        ModelFixture rep = buildDiffuseInterceptModel(0.75, false);
        InitialState known = InitialState.known(new double[]{0.0}, new double[]{1.0});

        CFASimulationSmoother.PosteriorMoments moments = CFASimulationSmoother.posteriorMoments(rep, known);

        assertEquals(rep.kStates, moments.kStates());
        assertEquals(rep.nobs, moments.nobs());
        assertEquals(rep.nobs * rep.kStates, moments.order());
        assertEquals(0.375, moments.mean(0, 0), TOL);
        for (int t = 1; t < rep.nobs; t++) {
            assertEquals(0.75, moments.mean(0, t), TOL, "posterior mean at t=" + t);
        }
        for (int t = 0; t < rep.nobs; t++) {
            assertEquals(0.5, moments.covariance(0, 0, t), TOL, "posterior variance at t=" + t);
        }
        for (int t = 0; t < rep.nobs; t++) {
            for (int s = 0; s < rep.nobs; s++) {
                if (t != s) {
                    assertEquals(0.0, moments.jointCovariance(0, t, 0, s), TOL,
                        "posterior cross covariance t=" + t + " s=" + s);
                }
            }
        }
    }

    @Test
    void testCfaPosteriorMomentsMatchKalmanSmootherMoments() {
        ModelFixture rep = new ModelFixture(2, 2, 2, 5);
        double[][] endog = {
            {1.20, -0.40},
            {0.70, 0.25},
            {0.0, -0.10},
            {1.10, 0.0},
            {0.30, 0.45}
        };
        boolean[][] missing = {
            {false, false},
            {false, false},
            {true, false},
            {false, true},
            {false, false}
        };
        for (int t = 0; t < rep.nobs; t++) {
            rep.setDesign(new double[]{1.0 + 0.05 * t, 0.25, -0.20, 0.80 - 0.03 * t}, t);
            rep.setTransition(new double[]{0.45, 0.10 + 0.02 * t, -0.05, 0.35}, t);
            rep.setObsCov(new double[]{0.80 + 0.04 * t, 0.10, 0.10, 0.65 + 0.03 * t}, t);
            rep.setStateCov(new double[]{0.70 + 0.02 * t, 0.06, 0.06, 0.90 + 0.01 * t}, t);
            rep.setSelection(new double[]{1.0, 0.0, 0.0, 1.0}, t);
            rep.setObsIntercept(new double[]{0.10 * t, -0.05 * t}, t);
            rep.setStateIntercept(new double[]{0.20, -0.10 + 0.03 * t}, t);
            rep.setEndog(endog[t], t);
            rep.setMissing(missing[t], t);
        }
        InitialState known = InitialState.known(new double[]{0.15, -0.25}, new double[]{1.2, 0.15, 0.15, 1.4});

        CFASimulationSmoother.PosteriorMoments moments = CFASimulationSmoother.posteriorMoments(rep, known);
        SmootherResult smoother = SmootherEngine.smooth(rep, known, SmootherOptions.conventional());

        for (int t = 0; t < rep.nobs; t++) {
            for (int i = 0; i < rep.kStates; i++) {
                assertEquals(smoother.smoothedState(i, t), moments.mean(i, t), TOL,
                    "posterior mean t=" + t + " i=" + i);
                for (int j = 0; j < rep.kStates; j++) {
                    assertEquals(smoother.smoothedStateCov(i, j, t), moments.covariance(i, j, t), TOL,
                        "posterior covariance t=" + t + " i=" + i + " j=" + j);
                }
            }
        }
    }

    @Test
    void testSimulationSmootherOutputMaskMatrixRetainsExpectedBanks() {
        ModelFixture rep = buildDiffuseInterceptModel(0.75, true);
        double[] measurement = arangeTenths(rep.nobs * rep.kEndog);
        double[] state = arangeTenths(rep.nobs * rep.kPosdef);
        double[] initial = arangeTenths(rep.kStates);
        SimulationSmootherOptions[] baseOptions = {
            SimulationSmootherOptions.stateOnly(),
            SimulationSmootherOptions.stateOnly().withGeneratedOutputs(),
            SimulationSmootherOptions.disturbanceOnly(),
            SimulationSmootherOptions.disturbanceOnly().withGeneratedOutputs(),
            SimulationSmootherOptions.defaults(),
            SimulationSmootherOptions.defaults().withGeneratedOutputs()
        };

        for (SimulationSmootherOptions base : baseOptions) {
            for (boolean univariate : new boolean[]{false, true}) {
                SimulationSmootherOptions options = univariate ? base.withUnivariateFilter() : base;
                SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep,
                    InitialState.approximateDiffuse(),
                    measurement,
                    state,
                    initial,
                    (SimulationSmoother.Pool) null,
                    options);

                assertSimulationSurfaceLengths(rep, options, result);
            }
        }
    }

    @Test
    void testSimulationSmoothingZeroVariatesMatchesStatsmodels() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;
        SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep,
                InitialState.approximateDiffuse(),
                new double[nobs * rep.kEndog],
                new double[nobs * rep.kPosdef],
                new double[rep.kStates], SimulationSmootherOptions.defaults());

        assertGeneratedSuppressed(result);

        assertRowMajorMatchesCsv(result.simulatedState, rep.kStates,
                "results_simulation_smoothing0.csv", "state1", "state2", "state3");
        assertRowMajorMatchesCsv(result.simulatedMeasurementDisturbance, rep.kEndog,
                "results_simulation_smoothing0.csv", "eps1", "eps2", "eps3");
        assertRowMajorMatchesCsv(result.simulatedStateDisturbance, rep.kPosdef,
                "results_simulation_smoothing0.csv", "eta1", "eta2", "eta3");
    }

    @Test
    void testSimulationSmoothingMeasurementVariatesMatchesStatsmodels() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;
        SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep,
                InitialState.approximateDiffuse(),
                arangeTenths(nobs * rep.kEndog),
                new double[nobs * rep.kPosdef],
                new double[rep.kStates], SimulationSmootherOptions.defaults());

        assertGeneratedSuppressed(result);

        assertRowMajorMatchesCsv(result.simulatedState, rep.kStates,
                "results_simulation_smoothing1.csv", "state1", "state2", "state3");
        assertRowMajorMatchesCsv(result.simulatedMeasurementDisturbance, rep.kEndog,
                "results_simulation_smoothing1.csv", "eps1", "eps2", "eps3");
        assertRowMajorMatchesCsv(result.simulatedStateDisturbance, rep.kPosdef,
                "results_simulation_smoothing1.csv", "eta1", "eta2", "eta3");
    }

    @Test
    void testSimulationSmoothingMeasurementAndStateVariatesMatchesStatsmodels() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;
        SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep,
                InitialState.approximateDiffuse(),
                arangeTenths(nobs * rep.kEndog),
                arangeTenths(nobs * rep.kPosdef),
                new double[rep.kStates], SimulationSmootherOptions.defaults());

        assertRowMajorMatchesCsv(result.simulatedState, rep.kStates,
                "results_simulation_smoothing2.csv", "state1", "state2", "state3");
        assertRowMajorMatchesCsv(result.simulatedMeasurementDisturbance, rep.kEndog,
                "results_simulation_smoothing2.csv", "eps1", "eps2", "eps3");
        assertRowMajorMatchesCsv(result.simulatedStateDisturbance, rep.kPosdef,
                "results_simulation_smoothing2.csv", "eta1", "eta2", "eta3");
    }

        @Test
        void testSimulationSmoothingRecordedVariatesMatchesStatsmodels() {
        ModelFixture rep = buildReferenceVar(reconstructFullObservedData());
        SimulationSmootherResult result = simulateWithRecordedVariates(
            rep,
            InitialState.approximateDiffuse(),
            SimulationSmootherOptions.defaults());

        assertRowMajorMatchesCsv(result.simulatedState, rep.kStates,
            "results_simulation_smoothing3.csv", "state1", "state2", "state3");
        assertRowMajorMatchesCsv(result.simulatedMeasurementDisturbance, rep.kEndog,
            "results_simulation_smoothing3.csv", "eps1", "eps2", "eps3");
        assertRowMajorMatchesCsv(result.simulatedStateDisturbance, rep.kPosdef,
            "results_simulation_smoothing3.csv", "eta1", "eta2", "eta3");
        }

        @Test
        void testSimulationSmoothingFullyMissingMatchesStatsmodels() {
        ModelFixture rep = buildReferenceVar(reconstructFullObservedData());
        applyMissingPattern(rep, 4);

        SimulationSmootherResult result = simulateWithRecordedVariates(
            rep,
            InitialState.approximateDiffuse(),
            SimulationSmootherOptions.defaults());

            assertSimulationSnapshot(result, 0,
                new double[]{-995.2670578592545, -149.44393474477295, 146.2227518105481},
                new double[]{-0.009661420914905084, 0.0020996902049820184, 0.04482900481760568},
                new double[]{0.008860125493147832, 0.013963736388595675, -0.02865161036453286});
            assertSimulationSnapshot(result, 49,
                new double[]{0.011479752331431729, 0.009607170676086963, 0.05149538322571562},
                new double[]{0.0022082696956163318, 0.0038316637518525128, 0.014367221938624522},
                new double[]{-0.0002893877312676337, 0.008081113586968444, -0.04811784578536755});
            assertSimulationSnapshot(result, 50,
                new double[]{0.007764527294633912, 0.015008977038438859, -0.02192257387260663},
                new double[]{-0.004989591227336612, 0.0014835207103853298, -0.009861303106968583},
                new double[]{0.004863047829176587, 0.0033641697862121878, 0.011064233879453345});
        }

        @Test
        void testSimulationSmoothingPartiallyMissingMatchesStatsmodels() {
        ModelFixture rep = buildReferenceVar(reconstructFullObservedData());
        applyMissingPattern(rep, 5);

        SimulationSmootherResult result = simulateWithRecordedVariates(
            rep,
            InitialState.approximateDiffuse(),
            SimulationSmootherOptions.defaults());

            assertSimulationSnapshot(result, 0,
                new double[]{0.019608982570787643, 0.006487116493841684, 0.031049006228266027},
                new double[]{-0.009661420914905084, 0.00879898684590763, 0.049163673904198585},
                new double[]{0.0007728509973893237, 0.0061739750325926385, -0.04921343190778884});
            assertSimulationSnapshot(result, 49,
                new double[]{0.005694500975222353, 0.006363721750635224, 0.021604671820217573},
                new double[]{0.0022082696956163318, 0.0015612318077454822, -0.008536528443810899},
                new double[]{0.0017227621575828093, 0.009829088005417471, -0.04248919159882597});
            assertSimulationSnapshot(result, 50,
                new double[]{0.006972934734801658, 0.014132330717746424, -0.024635605148522574},
                new double[]{-0.004197998667504358, 0.002360160645174277, -0.00714825802851924},
                new double[]{0.005366740852467464, 0.0037717324722771686, 0.012608298975427904});
        }

        @Test
        void testSimulationSmoothingMixedMissingMatchesStatsmodels() {
        ModelFixture rep = buildReferenceVar(reconstructFullObservedData());
        applyMissingPattern(rep, 6);

        SimulationSmootherResult result = simulateWithRecordedVariates(
            rep,
            InitialState.approximateDiffuse(),
            SimulationSmootherOptions.defaults());

            assertSimulationSnapshot(result, 0,
                new double[]{0.019608983017480777, 0.006487116483681369, 0.031049005810792593},
                new double[]{-0.009661420914905084, 0.008798987203587184, 0.04916367324169946},
                new double[]{0.0007728509901504476, 0.006173974410575615, -0.04921343426655491});
            assertSimulationSnapshot(result, 49,
                new double[]{0.011681498224071756, 0.011262783889228111, 0.05669437754471266},
                new double[]{0.0022082696956163318, 0.0038316637518525128, 0.014367221938624522},
                new double[]{0.003319416466976052, 0.00436744866649651, 0.011733548963279405});
            assertSimulationSnapshot(result, 50,
                new double[]{0.01287248612974157, 0.012085665953119248, 0.04650313668712897},
                new double[]{-0.0100975500877113, 0.0007388156634776367, 0.06774298564403552},
                new double[]{0.006570440531153048, 0.0010767841956384891, 0.03263951315364053});
            assertSimulationSnapshot(result, 119,
                new double[]{0.01981186689393063, 0.0032811775735434515, 0.11497385903377798},
                new double[]{-0.007028175241267462, 0.0003888278177837148, 0.08055656853418966},
                new double[]{-0.002500144810479577, -0.0002650647237858944, -0.017199987023360612});
            assertSimulationSnapshot(result, 129,
                new double[]{0.012607294866600954, 0.007084076129685547, 0.07584120917412348},
                new double[]{0.002527385979571712, -0.0032789086899823676, 0.00849790702596137},
                new double[]{0.00568550237284284, 0.0008650964421936495, 0.024474058014926398});
            assertSimulationSnapshot(result, rep.nobs - 1,
                new double[]{0.00408821426610572, 0.002218271066091684, 0.02667725450230516},
                new double[]{0.006280771151851087, 9.025381665555378E-5, -0.0074927930418541115},
                new double[]{0.0049558012706780255, -0.002991194783779755, 0.025045995020695622});
        }

        @Test
        void testSimulationSmoothingRandomMatchesExplicitVariates() {
            int nobs = 10;
            ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
            for (int t = 0; t < nobs; t++) {
                rep.setDesign(new double[]{0.0}, t);
                rep.setObsIntercept(new double[]{0.0}, t);
                rep.setObsCov(new double[]{1.0}, t);
                rep.setTransition(new double[]{0.0}, t);
                rep.setStateIntercept(new double[]{0.0}, t);
                rep.setSelection(new double[]{1.0}, t);
                rep.setStateCov(new double[]{1.0}, t);
                rep.setEndog(new double[]{0.0}, t);
            }

            Random expectedRng = new Random(1234L);
            double[] measurement = new double[nobs];
            double[] state = new double[nobs];
            double[] initial = new double[1];
            for (int i = 0; i < measurement.length; i++) {
                measurement[i] = expectedRng.nextGaussian();
            }
            for (int i = 0; i < state.length; i++) {
                state[i] = expectedRng.nextGaussian();
            }
            initial[0] = expectedRng.nextGaussian();

            SimulationSmootherResult explicit = SimulationSmootherEngine.simulate(rep, InitialState.known(new double[]{0.0}, new double[]{0.0}),
                    measurement, state, initial, SimulationSmootherOptions.defaults());
            SimulationSmootherResult fromRandom = SimulationSmootherEngine.simulate(rep, InitialState.known(new double[]{0.0}, new double[]{0.0}),
                    new Random(1234L), SimulationSmootherOptions.defaults());

                assertGeneratedSuppressed(explicit);
                assertGeneratedSuppressed(fromRandom);
            assertSimulationResultMatches(explicit, fromRandom);
        }

        @Test
        void testSimulationGeneratedInitialStateUsesCustomApproximateDiffuseVariance() {
            ModelFixture rep = buildZeroInitializationSystem(1);
            InitialState init = InitialState.approximateDiffuse(new double[]{1.5}, 25.0);

            SimulationSmootherResult result = SimulationSmootherEngine.simulate(
                rep,
                init,
                new double[]{0.0},
                new double[]{0.0},
                new double[]{2.0},
                SimulationSmootherOptions.defaults().withGeneratedOutputs());

            assertEquals(11.5, result.generatedState[0], TOL);
            assertEquals(0.0, result.generatedState[1], TOL);
        }

        @Test
        void testSimulationSmoothingRandomPoolMatchesExplicitVariates() {
            int nobs = 10;
            ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
            for (int t = 0; t < nobs; t++) {
                rep.setDesign(new double[]{0.0}, t);
                rep.setObsIntercept(new double[]{0.0}, t);
                rep.setObsCov(new double[]{1.0}, t);
                rep.setTransition(new double[]{0.0}, t);
                rep.setStateIntercept(new double[]{0.0}, t);
                rep.setSelection(new double[]{1.0}, t);
                rep.setStateCov(new double[]{1.0}, t);
                rep.setEndog(new double[]{0.0}, t);
            }

            Random expectedRng = new Random(4321L);
            double[] measurement = new double[nobs];
            double[] state = new double[nobs];
            double[] initial = new double[1];
            for (int i = 0; i < measurement.length; i++) {
                measurement[i] = expectedRng.nextGaussian();
            }
            for (int i = 0; i < state.length; i++) {
                state[i] = expectedRng.nextGaussian();
            }
            initial[0] = expectedRng.nextGaussian();

            SimulationSmootherResult explicit = SimulationSmootherEngine.simulate(rep, InitialState.known(new double[]{0.0}, new double[]{0.0}),
                    measurement, state, initial, SimulationSmootherOptions.defaults());
            SimulationSmootherResult fromRandom = SimulationSmootherEngine.simulate(rep, InitialState.known(new double[]{0.0}, new double[]{0.0}),
                    new Random(4321L), new SimulationSmoother.Pool(), SimulationSmootherOptions.defaults());

                assertGeneratedSuppressed(explicit);
                assertGeneratedSuppressed(fromRandom);
            assertSimulationResultMatches(explicit, fromRandom);
        }

            @Test
            void testSimulationSmootherPoolMatchesStandalone() {
                ModelFixture rep = buildReferenceVar(reconstructObservedData());
                int nobs = rep.nobs;
                double[] measurement = arangeTenths(nobs * rep.kEndog);
                double[] state = arangeTenths(nobs * rep.kPosdef);
                double[] initial = arangeTenths(rep.kStates);
                SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

                SimulationSmootherResult expected = SimulationSmootherEngine.simulate(rep, InitialState.approximateDiffuse(), measurement, state, initial, SimulationSmootherOptions.defaults());
                SimulationSmootherResult actual = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());

                assertSimulationResultMatches(expected, actual);
            }

            @Test
            void testSimulationSmootherPoolReuseReturnsBorrowedResultByDefault() {
                ModelFixture rep = buildReferenceVar(reconstructObservedData());
                int nobs = rep.nobs;
                double[] measurement = arangeTenths(nobs * rep.kEndog);
                double[] state = arangeTenths(nobs * rep.kPosdef);
                double[] initial = arangeTenths(rep.kStates);
                SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

                SimulationSmootherResult actual1 = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());
                SimulationSmootherResult actual2 = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());

                assertSame(actual1, actual2);
            }

            @Test
            void testSimulationSmootherPoolCopyPreservesPreviousResults() {
                ModelFixture rep = buildReferenceVar(reconstructObservedData());
                int nobs = rep.nobs;
                double[] measurement1 = arangeTenths(nobs * rep.kEndog);
                double[] state1 = arangeTenths(nobs * rep.kPosdef);
                double[] initial1 = arangeTenths(rep.kStates);
                double[] measurement2 = new double[nobs * rep.kEndog];
                double[] state2 = new double[nobs * rep.kPosdef];
                double[] initial2 = new double[rep.kStates];
                for (int i = 0; i < measurement2.length; i++) {
                measurement2[i] = -0.2 * (i + 1);
                }
                for (int i = 0; i < state2.length; i++) {
                state2[i] = 0.15 * (i + 1);
                }
                for (int i = 0; i < initial2.length; i++) {
                initial2[i] = -0.3 * (i + 1);
                }
                SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

                SimulationSmootherResult expected1 = SimulationSmootherEngine.simulate(rep, InitialState.approximateDiffuse(), measurement1, state1, initial1, SimulationSmootherOptions.defaults());
                SimulationSmootherResult actual1 = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), measurement1, state1, initial1, pool, SimulationSmootherOptions.defaults());
                SimulationSmootherResult snapshot1 = actual1.copy();

                SimulationSmootherResult expected2 = SimulationSmootherEngine.simulate(rep, InitialState.approximateDiffuse(), measurement2, state2, initial2, SimulationSmootherOptions.defaults());
                SimulationSmootherResult actual2 = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), measurement2, state2, initial2, pool, SimulationSmootherOptions.defaults());

                assertSame(actual1, actual2);
                assertSimulationResultMatches(expected1, snapshot1);
                assertSimulationResultMatches(expected2, actual2);
            }

            @Test
            void testSimulationSmootherPoolReuseInvalidatesPreviousResult() {
                ModelFixture rep = buildReferenceVar(reconstructObservedData());
                int nobs = rep.nobs;
                double[] measurement1 = arangeTenths(nobs * rep.kEndog);
                double[] state1 = arangeTenths(nobs * rep.kPosdef);
                double[] initial1 = arangeTenths(rep.kStates);
                double[] measurement2 = new double[nobs * rep.kEndog];
                double[] state2 = new double[nobs * rep.kPosdef];
                double[] initial2 = new double[rep.kStates];
                for (int i = 0; i < measurement2.length; i++) {
                    measurement2[i] = -0.2 * (i + 1);
                }
                for (int i = 0; i < state2.length; i++) {
                    state2[i] = 0.15 * (i + 1);
                }
                for (int i = 0; i < initial2.length; i++) {
                    initial2[i] = -0.3 * (i + 1);
                }
                SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

                SimulationSmootherResult actual1 = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), measurement1, state1, initial1, pool, SimulationSmootherOptions.defaults());
                double firstState = actual1.simulatedState(0, 0);

                SimulationSmootherResult actual2 = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), measurement2, state2, initial2, pool, SimulationSmootherOptions.defaults());

                assertSame(actual1, actual2);
                assertNotEquals(firstState, actual1.simulatedState(0, 0), TOL);
                assertEquals(actual2.simulatedState(0, 0), actual1.simulatedState(0, 0), TOL);
            }

            @Test
            void testSimulationSmootherPoolSupportsDiffuseInitialization() {
                ModelFixture rep = buildDiffuseInterceptModel(100.0, true);
                double[] measurement = arangeTenths(rep.nobs * rep.kEndog);
                double[] state = arangeTenths(rep.nobs * rep.kPosdef);
                double[] initial = arangeTenths(rep.kStates);
                SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

                SimulationSmootherResult expected = SimulationSmootherEngine.simulate(rep, InitialState.diffuse(), measurement, state, initial, SimulationSmootherOptions.defaults());
                SimulationSmootherResult actual = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.diffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());

                assertSimulationResultMatches(expected, actual);
                assertEquals(0L, pool.retainedNestedResultDoubleCount());
                assertEquals(0L, pool.retainedWorkspaceDoubleCount());
            }

            @Test
            void testSimulationSmootherPoolSupportsUnivariateFilter() {
                ModelFixture rep = buildReferenceVar(reconstructObservedData());
                int nobs = rep.nobs;
                double[] measurement = arangeTenths(nobs * rep.kEndog);
                double[] state = arangeTenths(nobs * rep.kPosdef);
                double[] initial = arangeTenths(rep.kStates);
                SimulationSmootherOptions spec = SimulationSmootherOptions.defaults().withUnivariateFilter();
                SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

                SimulationSmootherResult expected = SimulationSmootherEngine.simulate(rep, InitialState.approximateDiffuse(), measurement, state, initial, spec);
                SimulationSmootherResult actual = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), measurement, state, initial, pool, spec);

                assertSimulationResultMatches(expected, actual);
            }

            @Test
            void testSimulationSmootherPoolSupportsDifferentRepresentations() {
                ModelFixture rep1 = buildReferenceVar(reconstructObservedData());
                ModelFixture rep2 = buildReferenceVar(reconstructObservedData());
                rep2.design[0] += 0.25;
                rep2.transition[0] += 0.5;
                rep2.obsIntercept[0] -= 0.75;

                int nobs = rep1.nobs;
                double[] measurement = arangeTenths(nobs * rep1.kEndog);
                double[] state = arangeTenths(nobs * rep1.kPosdef);
                double[] initial = arangeTenths(rep1.kStates);
                SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

                SimulationSmootherResult expected1 = SimulationSmootherEngine.simulate(rep1, InitialState.approximateDiffuse(), measurement, state, initial, SimulationSmootherOptions.defaults());
                SimulationSmootherResult actual1 = SimulationSmootherEngine.simulateBorrowed(rep1, InitialState.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());
                SimulationSmootherResult snapshot1 = actual1.copy();

                SimulationSmootherResult expected2 = SimulationSmootherEngine.simulate(rep2, InitialState.approximateDiffuse(), measurement, state, initial, SimulationSmootherOptions.defaults());
                SimulationSmootherResult actual2 = SimulationSmootherEngine.simulateBorrowed(rep2, InitialState.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());

                assertSimulationResultMatches(expected1, snapshot1);
                assertSimulationResultMatches(expected2, actual2);
            }

            @Test
            void testRandomSimulationWarmPoolDoesNotGrowRetainedCountsAcrossRuns() {
            ModelFixture rep = buildReferenceVar(reconstructObservedData());
            SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

            SimulationSmootherResult first = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), new Random(123L), pool, SimulationSmootherOptions.defaults());

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

            SimulationSmootherResult second = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), new Random(456L), pool, SimulationSmootherOptions.defaults());

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
            void testSimulationPoolStateOnlySelectionShrinksRetainedResultBanks() {
            ModelFixture rep = buildReferenceVar(reconstructObservedData());
            int nobs = rep.nobs;
            double[] measurement = arangeTenths(nobs * rep.kEndog);
            double[] state = arangeTenths(nobs * rep.kPosdef);
            double[] initial = arangeTenths(rep.kStates);
            SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

            SimulationSmootherResult expectedStateOnly = SimulationSmootherEngine.simulate(rep, InitialState.approximateDiffuse(), measurement, state, initial, SimulationSmootherOptions.stateOnly());
            SimulationSmootherResult full = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherOptions.defaults());
            long fullRetainedResultDoubles = pool.retainedResultDoubleCount();

            SimulationSmootherResult stateOnlyResult = SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherOptions.stateOnly());

            assertSame(full, stateOnlyResult);
            assertEquals(fullRetainedResultDoubles, pool.retainedResultDoubleCount());
            assertArrayClose(expectedStateOnly.simulatedState,
                expectedStateOnly.simulatedStateBase(), expectedStateOnly.simulatedStateLength(),
                stateOnlyResult.simulatedState,
                stateOnlyResult.simulatedStateBase(), stateOnlyResult.simulatedStateLength());
            assertSuppressed(stateOnlyResult.simulatedMeasurementDisturbance);
            assertSuppressed(stateOnlyResult.simulatedStateDisturbance);
            assertGeneratedSuppressed(stateOnlyResult);
            assertTrue(stateOnlyResult.simulatedState.length > 0);
            }

        @Test
        void testSimulationSmoothingDiffuseStateIntercept() {
            ModelFixture rep = buildDiffuseInterceptModel(100.0, false);

            SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep,
                    InitialState.diffuse(),
                    new double[rep.nobs * rep.kEndog],
                    new double[rep.nobs * rep.kPosdef],
                    new double[rep.kStates], SimulationSmootherOptions.defaults());

            assertEquals(100.0, result.simulatedState(0, 0), TOL);
        }

        @Test
        void testSimulationSmoothingDiffuseStateInterceptWithMissing() {
            ModelFixture rep = buildDiffuseInterceptModel(100.0, true);

            SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep,
                    InitialState.diffuse(),
                    new double[rep.nobs * rep.kEndog],
                    new double[rep.nobs * rep.kPosdef],
                    new double[rep.kStates], SimulationSmootherOptions.defaults());

            assertEquals(100.0, result.simulatedState(0, 0), TOL);
        }

        @Test
        void testSimulationSmoothingObservationInterceptDoesNotLeakIntoStateDraw() {
            int nobs = 10;
            double intercept = 100.0;
            ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
            for (int t = 0; t < nobs; t++) {
                rep.setDesign(new double[]{1.0}, t);
                rep.setObsIntercept(new double[]{intercept}, t);
                rep.setObsCov(new double[]{1.0}, t);
                rep.setTransition(new double[]{1.0}, t);
                rep.setStateIntercept(new double[]{0.0}, t);
                rep.setSelection(new double[]{1.0}, t);
                rep.setStateCov(new double[]{1.0}, t);
                rep.setEndog(new double[]{intercept}, t);
            }

            SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep,
                InitialState.known(new double[]{0.0}, new double[]{0.0}),
                new double[nobs],
                new double[nobs],
                new double[1],
                SimulationSmootherOptions.stateOnly());

            for (int t = 0; t < nobs; t++) {
                assertEquals(0.0, result.simulatedState(0, t), TOL,
                    "state draw at t=" + t);
            }
        }

        @Test
        void testSimulationSmoothingKnownStateInterceptMatchesStatsmodelsSemantics() {
            double intercept = 100.0;
            ModelFixture rep = buildDiffuseInterceptModel(intercept, false);

            SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep,
                InitialState.known(new double[]{intercept}, new double[]{0.0}),
                new double[rep.nobs * rep.kEndog],
                new double[rep.nobs * rep.kPosdef],
                new double[rep.kStates],
                SimulationSmootherOptions.stateOnly());

            for (int t = 0; t < rep.nobs; t++) {
                assertEquals(intercept, result.simulatedState(0, t), TOL,
                    "state draw at t=" + t);
            }
        }

        @Test
        void testSimulationGeneratedOutputsFollowInterceptAwareStateSpaceRecursion() {
            int nobs = 4;
            ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
            for (int t = 0; t < nobs; t++) {
                rep.setDesign(new double[]{2.0}, t);
                rep.setObsIntercept(new double[]{5.0 + t}, t);
                rep.setObsCov(new double[]{1.0}, t);
                rep.setTransition(new double[]{0.5}, t);
                rep.setStateIntercept(new double[]{3.0}, t);
                rep.setSelection(new double[]{1.0}, t);
                rep.setStateCov(new double[]{4.0}, t);
                rep.setEndog(new double[]{0.0}, t);
            }

            SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep,
                InitialState.known(new double[]{2.0}, new double[]{9.0}),
                new double[nobs],
                new double[nobs],
                new double[]{0.0},
                SimulationSmootherOptions.defaults().withGeneratedOutputs());

            double state = 2.0;
            for (int t = 0; t < nobs; t++) {
                assertEquals(0.0, result.generatedMeasurementDisturbance(0, t), TOL,
                    "measurement disturbance at t=" + t);
                assertEquals(0.0, result.generatedStateDisturbance(0, t), TOL,
                    "state disturbance at t=" + t);
                assertEquals(state, result.generatedState(0, t), TOL,
                    "generated state at t=" + t);
                assertEquals(5.0 + t + 2.0 * state, result.generatedObs(0, t), TOL,
                    "generated observation at t=" + t);
                state = 3.0 + 0.5 * state;
            }
            assertEquals(state, result.generatedState(0, nobs), TOL,
                "terminal generated state");
        }

        @Test
        void testSimulationSmoothingInteriorMissingZeroVariatesMatchSmootherMean() {
            ModelFixture rep = new ModelFixture(1, 1, 1, 4);
            double[] endog = {1.0, 0.0, 0.0, 1.0};
            for (int t = 0; t < rep.nobs; t++) {
                rep.setDesign(new double[]{1.0}, t);
                rep.setObsIntercept(new double[]{0.0}, t);
                rep.setObsCov(new double[]{0.5}, t);
                rep.setTransition(new double[]{0.5}, t);
                rep.setStateIntercept(new double[]{0.0}, t);
                rep.setSelection(new double[]{1.0}, t);
                rep.setStateCov(new double[]{1.0}, t);
                rep.setEndog(new double[]{endog[t]}, t);
            }
            rep.setMissing(new boolean[]{true}, 1);
            rep.setMissing(new boolean[]{true}, 2);
            InitialState initialState = InitialState.known(new double[]{0.0}, new double[]{1.0});

            SmootherResult smoother = SmootherEngine.smooth(rep, initialState, SmootherOptions.univariate());
            SimulationSmootherResult result = SimulationSmootherEngine.simulate(rep,
                initialState,
                new double[rep.nobs],
                new double[rep.nobs],
                new double[rep.kStates],
                SimulationSmootherOptions.stateOnly().withUnivariateFilter());

            for (int t = 0; t < rep.nobs; t++) {
                assertTrue(Double.isFinite(result.simulatedState(0, t)));
                assertEquals(smoother.smoothedState(0, t), result.simulatedState(0, t), TOL,
                    "zero-variance missing-data draw at t=" + t);
            }
            assertTrue(smoother.smoothedStateCov(0, 0, 1) > 0.0);
            assertTrue(smoother.smoothedStateCov(0, 0, 2) > 0.0);
        }

        @Test
        void testSimulationSmootherStateOnlySelectionSuppressesDisturbances() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;
        double[] measurement = arangeTenths(nobs * rep.kEndog);
        double[] state = arangeTenths(nobs * rep.kPosdef);
        double[] initial = new double[rep.kStates];

        SimulationSmootherResult full = SimulationSmootherEngine.simulate(rep, InitialState.approximateDiffuse(), measurement, state, initial, SimulationSmootherOptions.defaults());
        SimulationSmootherResult stateOnly = SimulationSmootherEngine.simulate(rep, InitialState.approximateDiffuse(), measurement, state, initial, SimulationSmootherOptions.stateOnly());

        assertArrayClose(full.simulatedState, stateOnly.simulatedState);
        assertSuppressed(stateOnly.simulatedMeasurementDisturbance);
        assertSuppressed(stateOnly.simulatedStateDisturbance);
        assertGeneratedSuppressed(stateOnly);
        }

        @Test
        void testSimulationSmootherDisturbanceOnlySelectionSuppressesState() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;
        double[] measurement = arangeTenths(nobs * rep.kEndog);
        double[] state = arangeTenths(nobs * rep.kPosdef);
        double[] initial = new double[rep.kStates];

        SimulationSmootherResult full = SimulationSmootherEngine.simulate(rep, InitialState.approximateDiffuse(), measurement, state, initial, SimulationSmootherOptions.defaults());
        SimulationSmootherResult disturbanceOnly = SimulationSmootherEngine.simulate(rep, InitialState.approximateDiffuse(), measurement, state, initial, SimulationSmootherOptions.disturbanceOnly());

        assertSuppressed(disturbanceOnly.simulatedState);
        assertArrayClose(full.simulatedMeasurementDisturbance, disturbanceOnly.simulatedMeasurementDisturbance);
        assertArrayClose(full.simulatedStateDisturbance, disturbanceOnly.simulatedStateDisturbance);
        assertGeneratedSuppressed(disturbanceOnly);
        }

        @Test
        void testSimulationSmootherGeneratedOutputsAreOptIn() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;

        SimulationSmootherResult lean = SimulationSmootherEngine.simulate(rep,
            InitialState.approximateDiffuse(),
            new double[nobs * rep.kEndog],
            new double[nobs * rep.kPosdef],
            new double[rep.kStates],
            SimulationSmootherOptions.defaults());
        SimulationSmootherResult generated = SimulationSmootherEngine.simulate(rep,
            InitialState.approximateDiffuse(),
            new double[nobs * rep.kEndog],
            new double[nobs * rep.kPosdef],
            new double[rep.kStates],
            SimulationSmootherOptions.defaults().withGeneratedOutputs());

        assertGeneratedSuppressed(lean);
        assertAllZero(generated.generatedMeasurementDisturbance);
        assertAllZero(generated.generatedStateDisturbance);
        assertAllZero(generated.generatedObs);
        assertAllZero(generated.generatedState);
        assertArrayClose(lean.simulatedState, generated.simulatedState);
        assertArrayClose(lean.simulatedMeasurementDisturbance, generated.simulatedMeasurementDisturbance);
        assertArrayClose(lean.simulatedStateDisturbance, generated.simulatedStateDisturbance);
        }

        @Test
        void testLeanSimulationOptionsKeepWorkspaceCompactAndReduceRetention() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool leanPool = new SimulationSmoother.Pool();
        SimulationSmoother.Pool generatedPool = new SimulationSmoother.Pool();

        leanPool.reserve(rep, InitialState.approximateDiffuse(), SimulationSmootherOptions.defaults());
        generatedPool.reserve(rep, InitialState.approximateDiffuse(), SimulationSmootherOptions.defaults().withGeneratedOutputs());

        assertTrue(leanPool.retainedTotalByteCount() < generatedPool.retainedTotalByteCount());
        assertTrue(leanPool.retainedResultDoubleCount() < generatedPool.retainedResultDoubleCount());
        assertEquals(0L, leanPool.retainedWorkspaceDoubleCount());
        assertEquals(0L, generatedPool.retainedWorkspaceDoubleCount());
        }

        @Test
        void testSimulationSmootherReserveOmitsFilteredFilterHistory() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        pool.reserve(rep, InitialState.approximateDiffuse(), SimulationSmootherOptions.defaults());

        assertEquals(expectedReservedFilterResultDoubles(rep, SimulationSmootherOptions.defaults()),
            pool.retainedNestedResultDoubleCount());
        }

        @Test
        void testLeanUnivariateSimulationSpecKeepsWorkspaceCompactAndReducesTotalMemory() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());

        SimulationSmoother.Pool leanAllPool = new SimulationSmoother.Pool();
        SimulationSmoother.Pool generatedAllPool = new SimulationSmoother.Pool();
        SimulationSmoother.Pool leanStateOnlyPool = new SimulationSmoother.Pool();
        SimulationSmoother.Pool generatedStateOnlyPool = new SimulationSmoother.Pool();

        leanAllPool.reserve(rep, InitialState.approximateDiffuse(),
            SimulationSmootherOptions.defaults().withUnivariateFilter());
        generatedAllPool.reserve(rep, InitialState.approximateDiffuse(),
            SimulationSmootherOptions.defaults().withUnivariateFilter().withGeneratedOutputs());
        leanStateOnlyPool.reserve(rep, InitialState.approximateDiffuse(),
            SimulationSmootherOptions.stateOnly().withUnivariateFilter());
        generatedStateOnlyPool.reserve(rep, InitialState.approximateDiffuse(),
            SimulationSmootherOptions.stateOnly().withUnivariateFilter().withGeneratedOutputs());

        assertTrue(leanAllPool.retainedTotalByteCount() < generatedAllPool.retainedTotalByteCount());
        assertTrue(leanAllPool.retainedResultDoubleCount() < generatedAllPool.retainedResultDoubleCount());
        assertEquals(0L, leanAllPool.retainedWorkspaceDoubleCount());
        assertEquals(0L, generatedAllPool.retainedWorkspaceDoubleCount());
        assertTrue(leanStateOnlyPool.retainedTotalByteCount() < generatedStateOnlyPool.retainedTotalByteCount());
        assertTrue(leanStateOnlyPool.retainedResultDoubleCount() < generatedStateOnlyPool.retainedResultDoubleCount());
        assertEquals(0L, leanStateOnlyPool.retainedWorkspaceDoubleCount());
        assertEquals(0L, generatedStateOnlyPool.retainedWorkspaceDoubleCount());
        }

        @Test
    void testSimulationPoolSharesSmootherScratchAndOmitsReservedSmootherResultBanks() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();
        KalmanEngine.Workspace filterWorkspace = KalmanEngine.workspace();

        pool.reserve(rep, InitialState.approximateDiffuse(), SimulationSmootherOptions.defaults());
        filterWorkspace.reserveFilter(rep, SimulationSmootherOptions.defaults().requiredFilterOptions());

        assertEquals(filterWorkspace.retainedFilterResultDoubleCount(), pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(filterWorkspace.retainedFilterScratchDoubleCount(), pool.retainedNestedScratchDoubleCount());
        assertTrue(pool.retainedTotalByteCount() > 0);
        }

        @Test
        void testSimulationPoolReleaseRetainedScratchClearsSharedScratchAliases() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        pool.reserve(rep, InitialState.approximateDiffuse(), SimulationSmootherOptions.defaults());
        long retainedNestedResultDoubles = pool.retainedNestedResultDoubleCount();

        assertTrue(pool.retainedNestedScratchDoubleCount() > 0);

        pool.releaseRetainedScratch();

        assertEquals(0L, pool.retainedNestedScratchDoubleCount());
        assertEquals(retainedNestedResultDoubles, pool.retainedNestedResultDoubleCount());
        }

        @Test
        void testSimulationReserveSwitchingToUnivariateTrimsConventionalNestedFilterRetention() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();
        SimulationSmootherOptions conventionalSpec = SimulationSmootherOptions.defaults();
        SimulationSmootherOptions univariateSpec = SimulationSmootherOptions.disturbanceOnly().withUnivariateFilter();
        KalmanEngine.Workspace univariateWorkspace = KalmanEngine.workspace();

        pool.reserve(rep, InitialState.approximateDiffuse(), conventionalSpec);
        assertTrue(pool.retainedNestedResultDoubleCount() > 0L);

        pool.reserve(rep, InitialState.approximateDiffuse(), univariateSpec);
        univariateWorkspace.reserveUnivariateFilter(rep, univariateSpec.requiredFilterOptions());

        assertEquals(univariateWorkspace.retainedUnivariateResultDoubleCount(), pool.retainedNestedResultDoubleCount());
        assertEquals(univariateWorkspace.retainedUnivariateScratchDoubleCount(), pool.retainedNestedScratchDoubleCount());

        pool.releaseRetainedScratch();

        assertEquals(0L, pool.retainedNestedScratchDoubleCount());
        }

        @Test
        void testSimulationReserveDoesNotRetainVariateBanksForExplicitVariatesPath() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        pool.reserve(rep, InitialState.approximateDiffuse(), SimulationSmootherOptions.defaults());

        assertEquals(0L, pool.retainedVariateDoubleCount());
        }

        @Test
        void testRandomSimulationReleasesVariateBanksAfterPooledRun() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        SimulationSmootherEngine.simulateBorrowed(rep, InitialState.approximateDiffuse(), new Random(123), pool,
            SimulationSmootherOptions.defaults());

        assertEquals(0L, pool.retainedVariateDoubleCount());
        }

        @Test
        void testSimulationReleasesTransientNestedBanksAfterPooledRun() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        SimulationSmootherEngine.simulate(rep,
            InitialState.approximateDiffuse(),
            new double[nobs * rep.kEndog],
            new double[nobs * rep.kPosdef],
            new double[rep.kStates],
            pool,
            SimulationSmootherOptions.defaults());

        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());

        SimulationSmootherEngine.simulate(rep,
            InitialState.approximateDiffuse(),
            new double[nobs * rep.kEndog],
            new double[nobs * rep.kPosdef],
            new double[rep.kStates],
            pool,
            SimulationSmootherOptions.defaults());

        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        }

        @Test
        void testSimulationReserveRehydratesOnlyReserveScopedBanks() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        SimulationSmootherEngine.simulate(rep,
            InitialState.approximateDiffuse(),
            new Random(123),
            pool,
            SimulationSmootherOptions.defaults());

        long retainedResultDoubles = pool.retainedResultDoubleCount();

        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());

        pool.reserve(rep, InitialState.approximateDiffuse(), SimulationSmootherOptions.defaults());

        assertEquals(retainedResultDoubles, pool.retainedResultDoubleCount());
        assertEquals(expectedReservedFilterResultDoubles(rep, SimulationSmootherOptions.defaults()), pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());
        }

        @Test
    void testSimulationPoolOmitsReservedSmootherResultBanksAcrossRepresentativePaths() {
        ModelFixture referenceVar = buildReferenceVar(reconstructObservedData());
        ModelFixture scalarDiffuse = buildDiffuseInterceptModel(100.0, true);

        assertReservedSmootherResultBanksOmitted(referenceVar,
            InitialState.approximateDiffuse(),
            SimulationSmootherOptions.defaults());
        assertReservedSmootherResultBanksOmitted(referenceVar,
            InitialState.approximateDiffuse(),
            SimulationSmootherOptions.defaults().withGeneratedOutputs());
        assertReservedSmootherResultBanksOmitted(referenceVar,
            InitialState.approximateDiffuse(),
            SimulationSmootherOptions.disturbanceOnly().withUnivariateFilter());
        assertReservedSmootherResultBanksOmitted(scalarDiffuse,
            InitialState.diffuse(),
            SimulationSmootherOptions.defaults());
        assertReservedSmootherResultBanksOmitted(scalarDiffuse,
            InitialState.diffuse(),
            SimulationSmootherOptions.stateOnly());
        }
}