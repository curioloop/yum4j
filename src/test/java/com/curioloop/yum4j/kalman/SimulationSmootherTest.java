package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.fixtures.StatsmodelsResources;
import com.curioloop.yum4j.kalman.filter.KalmanFilter;
import com.curioloop.yum4j.kalman.filter.UnivariateFilter;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.model.ModelFixture;
import com.curioloop.yum4j.kalman.smooth.SimulationSmoother;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherSpec;
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
                                                                         StateInitialization init,
                                                                         SimulationSmootherSpec spec) {
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
        return SimulationSmoother.simulate(rep, init, measurement, state, initial, spec.withUnivariateFilter());
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

    private static long expectedReservedFilterResultDoubles(ModelFixture rep,
                                                            SimulationSmootherSpec spec) {
        if (spec.usesUnivariateFilter()) {
            return 0L;
        }
        KalmanFilter.Pool filterPool = new KalmanFilter.Pool();
        filterPool.reserve(rep, spec.requiredFilterSpec());
        return filterPool.retainedResultDoubleCount();
    }

    private static void assertReservedSmootherResultBanksOmitted(ModelFixture rep,
                                                                 StateInitialization init,
                                                                 SimulationSmootherSpec spec) {
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
    void testSimulationSmoothingZeroVariatesMatchesStatsmodels() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;
        SimulationSmootherResult result = SimulationSmoother.simulate(rep,
                StateInitialization.approximateDiffuse(),
                new double[nobs * rep.kEndog],
                new double[nobs * rep.kPosdef],
                new double[rep.kStates]);

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
        SimulationSmootherResult result = SimulationSmoother.simulate(rep,
                StateInitialization.approximateDiffuse(),
                arangeTenths(nobs * rep.kEndog),
                new double[nobs * rep.kPosdef],
                new double[rep.kStates]);

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
        SimulationSmootherResult result = SimulationSmoother.simulate(rep,
                StateInitialization.approximateDiffuse(),
                arangeTenths(nobs * rep.kEndog),
                arangeTenths(nobs * rep.kPosdef),
                new double[rep.kStates]);

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
            StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.all());

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
            StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.all());

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
            StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.all());

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
            StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.all());

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

            SimulationSmootherResult explicit = SimulationSmoother.simulate(rep, StateInitialization.known(new double[]{0.0}, new double[]{0.0}),
                    measurement, state, initial);
            SimulationSmootherResult fromRandom = SimulationSmoother.simulate(rep, StateInitialization.known(new double[]{0.0}, new double[]{0.0}),
                    new Random(1234L));

                assertGeneratedSuppressed(explicit);
                assertGeneratedSuppressed(fromRandom);
            assertSimulationResultMatches(explicit, fromRandom);
        }

        @Test
        void testSimulationGeneratedInitialStateUsesCustomApproximateDiffuseVariance() {
            ModelFixture rep = buildZeroInitializationSystem(1);
            StateInitialization init = StateInitialization.approximateDiffuse(new double[]{1.5}, 25.0);

            SimulationSmootherResult result = SimulationSmoother.simulate(
                rep,
                init,
                new double[]{0.0},
                new double[]{0.0},
                new double[]{2.0},
                SimulationSmootherSpec.all().withGeneratedOutputs());

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

            SimulationSmootherResult explicit = SimulationSmoother.simulate(rep, StateInitialization.known(new double[]{0.0}, new double[]{0.0}),
                    measurement, state, initial);
            SimulationSmootherResult fromRandom = SimulationSmoother.simulate(rep, StateInitialization.known(new double[]{0.0}, new double[]{0.0}),
                    new Random(4321L), new SimulationSmoother.Pool());

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

                SimulationSmootherResult expected = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, SimulationSmootherSpec.all());
                SimulationSmootherResult actual = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());

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

                SimulationSmootherResult actual1 = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());
                SimulationSmootherResult actual2 = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());

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

                SimulationSmootherResult expected1 = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement1, state1, initial1, SimulationSmootherSpec.all());
                SimulationSmootherResult actual1 = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement1, state1, initial1, pool, SimulationSmootherSpec.all());
                SimulationSmootherResult snapshot1 = actual1.copy();

                SimulationSmootherResult expected2 = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement2, state2, initial2, SimulationSmootherSpec.all());
                SimulationSmootherResult actual2 = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement2, state2, initial2, pool, SimulationSmootherSpec.all());

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

                SimulationSmootherResult actual1 = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement1, state1, initial1, pool, SimulationSmootherSpec.all());
                double firstState = actual1.simulatedState(0, 0);

                SimulationSmootherResult actual2 = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement2, state2, initial2, pool, SimulationSmootherSpec.all());

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

                SimulationSmootherResult expected = SimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, SimulationSmootherSpec.all());
                SimulationSmootherResult actual = SimulationSmoother.simulate(rep, StateInitialization.diffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());

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
                SimulationSmootherSpec spec = SimulationSmootherSpec.all().withUnivariateFilter();
                SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

                SimulationSmootherResult expected = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, spec);
                SimulationSmootherResult actual = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, pool, spec);

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

                SimulationSmootherResult expected1 = SimulationSmoother.simulate(rep1, StateInitialization.approximateDiffuse(), measurement, state, initial, SimulationSmootherSpec.all());
                SimulationSmootherResult actual1 = SimulationSmoother.simulate(rep1, StateInitialization.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());
                SimulationSmootherResult snapshot1 = actual1.copy();

                SimulationSmootherResult expected2 = SimulationSmoother.simulate(rep2, StateInitialization.approximateDiffuse(), measurement, state, initial, SimulationSmootherSpec.all());
                SimulationSmootherResult actual2 = SimulationSmoother.simulate(rep2, StateInitialization.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());

                assertSimulationResultMatches(expected1, snapshot1);
                assertSimulationResultMatches(expected2, actual2);
            }

            @Test
            void testRandomSimulationWarmPoolDoesNotGrowRetainedCountsAcrossRuns() {
            ModelFixture rep = buildReferenceVar(reconstructObservedData());
            SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

            SimulationSmootherResult first = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), new Random(123L), pool, SimulationSmootherSpec.all());

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

            SimulationSmootherResult second = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), new Random(456L), pool, SimulationSmootherSpec.all());

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

            SimulationSmootherResult expectedStateOnly = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, SimulationSmootherSpec.stateOnly());
            SimulationSmootherResult full = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherSpec.all());
            long fullRetainedResultDoubles = pool.retainedResultDoubleCount();

            SimulationSmootherResult stateOnlyResult = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, pool, SimulationSmootherSpec.stateOnly());

            assertSame(full, stateOnlyResult);
            assertTrue(pool.retainedResultDoubleCount() < fullRetainedResultDoubles);
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

            SimulationSmootherResult result = SimulationSmoother.simulate(rep,
                    StateInitialization.diffuse(),
                    new double[rep.nobs * rep.kEndog],
                    new double[rep.nobs * rep.kPosdef],
                    new double[rep.kStates]);

            assertEquals(100.0, result.simulatedState(0, 0), TOL);
        }

        @Test
        void testSimulationSmoothingDiffuseStateInterceptWithMissing() {
            ModelFixture rep = buildDiffuseInterceptModel(100.0, true);

            SimulationSmootherResult result = SimulationSmoother.simulate(rep,
                    StateInitialization.diffuse(),
                    new double[rep.nobs * rep.kEndog],
                    new double[rep.nobs * rep.kPosdef],
                    new double[rep.kStates]);

            assertEquals(100.0, result.simulatedState(0, 0), TOL);
        }

        @Test
        void testSimulationSmootherStateOnlySelectionSuppressesDisturbances() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;
        double[] measurement = arangeTenths(nobs * rep.kEndog);
        double[] state = arangeTenths(nobs * rep.kPosdef);
        double[] initial = new double[rep.kStates];

        SimulationSmootherResult full = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, SimulationSmootherSpec.all());
        SimulationSmootherResult stateOnly = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, SimulationSmootherSpec.stateOnly());

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

        SimulationSmootherResult full = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, SimulationSmootherSpec.all());
        SimulationSmootherResult disturbanceOnly = SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), measurement, state, initial, SimulationSmootherSpec.disturbanceOnly());

        assertSuppressed(disturbanceOnly.simulatedState);
        assertArrayClose(full.simulatedMeasurementDisturbance, disturbanceOnly.simulatedMeasurementDisturbance);
        assertArrayClose(full.simulatedStateDisturbance, disturbanceOnly.simulatedStateDisturbance);
        assertGeneratedSuppressed(disturbanceOnly);
        }

        @Test
        void testSimulationSmootherGeneratedOutputsAreOptIn() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;

        SimulationSmootherResult lean = SimulationSmoother.simulate(rep,
            StateInitialization.approximateDiffuse(),
            new double[nobs * rep.kEndog],
            new double[nobs * rep.kPosdef],
            new double[rep.kStates],
            SimulationSmootherSpec.all());
        SimulationSmootherResult legacy = SimulationSmoother.simulate(rep,
            StateInitialization.approximateDiffuse(),
            new double[nobs * rep.kEndog],
            new double[nobs * rep.kPosdef],
            new double[rep.kStates],
            SimulationSmootherSpec.all().withGeneratedOutputs());

        assertGeneratedSuppressed(lean);
        assertAllZero(legacy.generatedMeasurementDisturbance);
        assertAllZero(legacy.generatedStateDisturbance);
        assertAllZero(legacy.generatedObs);
        assertAllZero(legacy.generatedState);
        assertArrayClose(lean.simulatedState, legacy.simulatedState);
        assertArrayClose(lean.simulatedMeasurementDisturbance, legacy.simulatedMeasurementDisturbance);
        assertArrayClose(lean.simulatedStateDisturbance, legacy.simulatedStateDisturbance);
        }

        @Test
        void testLeanSimulationSpecKeepsWorkspaceCompactAndReducesRetainedMemory() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool leanPool = new SimulationSmoother.Pool();
        SimulationSmoother.Pool legacyPool = new SimulationSmoother.Pool();

        leanPool.reserve(rep, StateInitialization.approximateDiffuse(), SimulationSmootherSpec.all());
        legacyPool.reserve(rep, StateInitialization.approximateDiffuse(), SimulationSmootherSpec.all().withGeneratedOutputs());

        assertTrue(leanPool.retainedTotalByteCount() < legacyPool.retainedTotalByteCount());
        assertTrue(leanPool.retainedResultDoubleCount() < legacyPool.retainedResultDoubleCount());
        assertEquals(0L, leanPool.retainedWorkspaceDoubleCount());
        assertEquals(0L, legacyPool.retainedWorkspaceDoubleCount());
        }

        @Test
        void testSimulationSmootherReserveOmitsFilteredFilterHistory() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        pool.reserve(rep, StateInitialization.approximateDiffuse(), SimulationSmootherSpec.all());

        assertEquals(expectedReservedFilterResultDoubles(rep, SimulationSmootherSpec.all()),
            pool.retainedNestedResultDoubleCount());
        }

        @Test
        void testLeanUnivariateSimulationSpecKeepsWorkspaceCompactAndReducesTotalMemory() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());

        SimulationSmoother.Pool leanAllPool = new SimulationSmoother.Pool();
        SimulationSmoother.Pool legacyAllPool = new SimulationSmoother.Pool();
        SimulationSmoother.Pool leanStateOnlyPool = new SimulationSmoother.Pool();
        SimulationSmoother.Pool legacyStateOnlyPool = new SimulationSmoother.Pool();

        leanAllPool.reserve(rep, StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.all().withUnivariateFilter());
        legacyAllPool.reserve(rep, StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.all().withUnivariateFilter().withGeneratedOutputs());
        leanStateOnlyPool.reserve(rep, StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.stateOnly().withUnivariateFilter());
        legacyStateOnlyPool.reserve(rep, StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.stateOnly().withUnivariateFilter().withGeneratedOutputs());

        assertTrue(leanAllPool.retainedTotalByteCount() < legacyAllPool.retainedTotalByteCount());
        assertTrue(leanAllPool.retainedResultDoubleCount() < legacyAllPool.retainedResultDoubleCount());
        assertEquals(0L, leanAllPool.retainedWorkspaceDoubleCount());
        assertEquals(0L, legacyAllPool.retainedWorkspaceDoubleCount());
        assertTrue(leanStateOnlyPool.retainedTotalByteCount() < legacyStateOnlyPool.retainedTotalByteCount());
        assertTrue(leanStateOnlyPool.retainedResultDoubleCount() < legacyStateOnlyPool.retainedResultDoubleCount());
        assertEquals(0L, leanStateOnlyPool.retainedWorkspaceDoubleCount());
        assertEquals(0L, legacyStateOnlyPool.retainedWorkspaceDoubleCount());
        }

        @Test
    void testSimulationPoolSharesSmootherScratchAndOmitsReservedSmootherResultBanks() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();
        KalmanFilter.Pool filterPool = new KalmanFilter.Pool();

        pool.reserve(rep, StateInitialization.approximateDiffuse(), SimulationSmootherSpec.all());
        filterPool.reserve(rep, SimulationSmootherSpec.all().requiredFilterSpec());

        assertEquals(filterPool.retainedResultDoubleCount(), pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(filterPool.retainedScratchDoubleCount(), pool.retainedNestedScratchDoubleCount());
        assertTrue(pool.retainedTotalByteCount() > 0);
        }

        @Test
        void testSimulationPoolReleaseRetainedScratchClearsSharedScratchAliases() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        pool.reserve(rep, StateInitialization.approximateDiffuse(), SimulationSmootherSpec.all());
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
        SimulationSmootherSpec conventionalSpec = SimulationSmootherSpec.all();
        SimulationSmootherSpec univariateSpec = SimulationSmootherSpec.disturbanceOnly().withUnivariateFilter();
        UnivariateFilter.Pool univariatePool = new UnivariateFilter.Pool();

        pool.reserve(rep, StateInitialization.approximateDiffuse(), conventionalSpec);
        assertTrue(pool.retainedNestedResultDoubleCount() > 0L);

        pool.reserve(rep, StateInitialization.approximateDiffuse(), univariateSpec);
        univariatePool.reserve(rep, univariateSpec.requiredFilterSpec());

        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(univariatePool.retainedScratchDoubleCount(), pool.retainedNestedScratchDoubleCount());

        pool.releaseRetainedScratch();

        assertEquals(0L, pool.retainedNestedScratchDoubleCount());
        }

        @Test
        void testSimulationReserveDoesNotRetainVariateBanksForExplicitVariatesPath() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        pool.reserve(rep, StateInitialization.approximateDiffuse(), SimulationSmootherSpec.all());

        assertEquals(0L, pool.retainedVariateDoubleCount());
        }

        @Test
        void testRandomSimulationReleasesVariateBanksAfterPooledRun() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        SimulationSmoother.simulate(rep, StateInitialization.approximateDiffuse(), new Random(123), pool,
            SimulationSmootherSpec.all());

        assertEquals(0L, pool.retainedVariateDoubleCount());
        }

        @Test
        void testSimulationReleasesTransientNestedBanksAfterPooledRun() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        int nobs = rep.nobs;
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        SimulationSmoother.simulate(rep,
            StateInitialization.approximateDiffuse(),
            new double[nobs * rep.kEndog],
            new double[nobs * rep.kPosdef],
            new double[rep.kStates],
            pool,
            SimulationSmootherSpec.all());

        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());

        SimulationSmoother.simulate(rep,
            StateInitialization.approximateDiffuse(),
            new double[nobs * rep.kEndog],
            new double[nobs * rep.kPosdef],
            new double[rep.kStates],
            pool,
            SimulationSmootherSpec.all());

        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        }

        @Test
        void testSimulationReserveRehydratesOnlyReserveScopedBanks() {
        ModelFixture rep = buildReferenceVar(reconstructObservedData());
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();

        SimulationSmoother.simulate(rep,
            StateInitialization.approximateDiffuse(),
            new Random(123),
            pool,
            SimulationSmootherSpec.all());

        long retainedResultDoubles = pool.retainedResultDoubleCount();

        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());

        pool.reserve(rep, StateInitialization.approximateDiffuse(), SimulationSmootherSpec.all());

        assertEquals(retainedResultDoubles, pool.retainedResultDoubleCount());
        assertEquals(expectedReservedFilterResultDoubles(rep, SimulationSmootherSpec.all()), pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());
        }

        @Test
    void testSimulationPoolOmitsReservedSmootherResultBanksAcrossRepresentativePaths() {
        ModelFixture referenceVar = buildReferenceVar(reconstructObservedData());
        ModelFixture scalarDiffuse = buildDiffuseInterceptModel(100.0, true);

        assertReservedSmootherResultBanksOmitted(referenceVar,
            StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.all());
        assertReservedSmootherResultBanksOmitted(referenceVar,
            StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.all().withGeneratedOutputs());
        assertReservedSmootherResultBanksOmitted(referenceVar,
            StateInitialization.approximateDiffuse(),
            SimulationSmootherSpec.disturbanceOnly().withUnivariateFilter());
        assertReservedSmootherResultBanksOmitted(scalarDiffuse,
            StateInitialization.diffuse(),
            SimulationSmootherSpec.all());
        assertReservedSmootherResultBanksOmitted(scalarDiffuse,
            StateInitialization.diffuse(),
            SimulationSmootherSpec.stateOnly());
        }
}