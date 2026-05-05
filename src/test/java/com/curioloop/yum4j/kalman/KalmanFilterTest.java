package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.kalman.filter.*;
import com.curioloop.yum4j.kalman.init.*;
import com.curioloop.yum4j.kalman.model.*;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class KalmanFilterTest {

    private static final double TOL = 1e-6;

    private static int kEndog(StateSpaceModel rep) {
    return rep.observationDimension();
    }

    private static int kStates(StateSpaceModel rep) {
    return rep.stateCount();
    }

    private static int kPosdef(StateSpaceModel rep) {
    return rep.stateDisturbanceCount();
    }

    private static int nobs(StateSpaceModel rep) {
    return rep.observationCount();
    }

    private static long doubleCount(double[] values) {
    return values == null ? 0L : values.length;
    }

    private static double[] diagonal(int dim, double value) {
    double[] matrix = new double[dim * dim];
    for (int i = 0; i < dim; i++) {
        matrix[i * dim + i] = value;
    }
    return matrix;
    }

    private static long convergedSnapshotDoubleCount(KalmanFilter.Pool pool) {
    return doubleCount(pool.convergedForecastErrorCov)
        + doubleCount(pool.convergedFilteredStateCov)
        + doubleCount(pool.convergedPredictedStateCov)
        + doubleCount(pool.convergedKalmanGain);
    }

    private static long expectedConvergedSnapshotDoubleCount(StateSpaceModel rep) {
    return (long) kEndog(rep) * kEndog(rep)
        + 2L * kStates(rep) * kStates(rep)
        + (long) kStates(rep) * kEndog(rep);
    }

    private static boolean hasMissingObservations(StateSpaceModel rep) {
    for (int t = 0; t < nobs(rep); t++) {
        if (rep.missingCount(t) > 0) {
            return true;
        }
    }
    return false;
    }

    private static long expectedFilterBaseScratchDoubleCount(StateSpaceModel rep) {
    return 2L * kStates(rep) * kEndog(rep)
        + (long) kEndog(rep) * kEndog(rep)
        + (long) kStates(rep) * kStates(rep)
        + kEndog(rep)
        + (long) kStates(rep) * kPosdef(rep);
    }

    private static long expectedDiffuseDelegateScratchDoubleCount(StateSpaceModel rep) {
    return (long) kStates(rep) * kEndog(rep)
        + (long) kEndog(rep) * kEndog(rep)
        + 5L * kStates(rep) * kStates(rep)
        + 6L * kStates(rep);
    }

    private static long expectedFilterScratchDoubleCount(StateSpaceModel rep) {
    return expectedFilterScratchDoubleCount(rep, FilterSpec.defaults());
    }

    private static long expectedFilterScratchDoubleCount(StateSpaceModel rep, FilterSpec spec) {
    boolean storeForecast = spec.stores(FilterSpec.Storage.FORECAST);
    boolean storePredicted = spec.stores(FilterSpec.Storage.PREDICTED_STATE)
        || (storeForecast && hasMissingObservations(rep));
    boolean storeFiltered = spec.stores(FilterSpec.Storage.FILTERED_STATE);
    long total = expectedFilterBaseScratchDoubleCount(rep);
    if (!storePredicted) {
        total += 2L * kStates(rep) + 2L * kStates(rep) * kStates(rep);
    }
    if (!storeFiltered) {
        total += kStates(rep) + (long) kStates(rep) * kStates(rep);
    }
    return total;
    }

    private static long expectedDiffuseFilterScratchDoubleCount(StateSpaceModel rep, FilterSpec spec) {
    boolean storePredicted = spec.stores(FilterSpec.Storage.PREDICTED_STATE);
    boolean storeFiltered = spec.stores(FilterSpec.Storage.FILTERED_STATE);
    long total = expectedFilterBaseScratchDoubleCount(rep) + expectedDiffuseDelegateScratchDoubleCount(rep);
    if (!storePredicted) {
        total += 2L * kStates(rep) + 4L * kStates(rep) * kStates(rep);
    }
    if (!storeFiltered) {
        total += kStates(rep) + (long) kStates(rep) * kStates(rep);
    }
    return total;
    }

    private static long expectedFilterResultDoubleCount(StateSpaceModel rep) {
    return expectedFilterResultDoubleCount(rep, FilterSpec.defaults());
    }

    private static long expectedFilterResultDoubleCount(StateSpaceModel rep, FilterSpec spec) {
    long kEndog = kEndog(rep);
    long kStates = kStates(rep);
    long nobs = nobs(rep);
    boolean storeForecast = spec.stores(FilterSpec.Storage.FORECAST);
    boolean storePredicted = spec.stores(FilterSpec.Storage.PREDICTED_STATE)
        || (storeForecast && hasMissingObservations(rep));
    boolean storeFiltered = spec.stores(FilterSpec.Storage.FILTERED_STATE);
    boolean storeGain = spec.stores(FilterSpec.Storage.KALMAN_GAIN);
    boolean storeLikelihood = spec.stores(FilterSpec.Storage.LIKELIHOOD);
    long total = 0L;
    if (storePredicted) {
        total += kStates * (nobs + 1);
        total += kStates * kStates * (nobs + 1);
    }
    if (storeFiltered) {
        total += kStates * nobs;
        total += kStates * kStates * nobs;
    }
    if (storeForecast) {
        total += kEndog * nobs;
        total += kEndog * nobs;
        total += kEndog * kEndog * nobs;
    }
    if (storeGain) {
        total += kStates * kEndog * nobs;
    }
    if (storeLikelihood) {
        total += nobs;
    }
    return total;
    }

    private static long expectedDiffuseRegularFilterResultDoubleCount(StateSpaceModel rep, FilterSpec spec) {
    long kEndog = kEndog(rep);
    long kStates = kStates(rep);
    long nobs = nobs(rep);
    long total = 0L;
    if (spec.stores(FilterSpec.Storage.PREDICTED_STATE)) {
        total += kStates * (nobs + 1);
        total += kStates * kStates * (nobs + 1);
    }
    if (spec.stores(FilterSpec.Storage.FILTERED_STATE)) {
        total += kStates * nobs;
        total += kStates * kStates * nobs;
    }
    if (spec.stores(FilterSpec.Storage.FORECAST)) {
        total += kEndog * nobs;
        total += kEndog * nobs;
        total += kEndog * kEndog * nobs;
    }
    if (spec.stores(FilterSpec.Storage.KALMAN_GAIN)) {
        total += kStates * kEndog * nobs;
    }
    if (spec.stores(FilterSpec.Storage.LIKELIHOOD)) {
        total += nobs;
    }
    return total;
    }

    private static long expectedFilterDiffuseResultDoubleCount(StateSpaceModel rep) {
    return expectedFilterDiffuseResultDoubleCount(rep, FilterSpec.defaults());
    }

    private static long expectedFilterDiffuseResultDoubleCount(StateSpaceModel rep, FilterSpec spec) {
    long total = 0L;
    if (spec.stores(FilterSpec.Storage.PREDICTED_STATE)) {
        total += (long) kStates(rep) * kStates(rep) * (nobs(rep) + 1);
    }
    if (spec.stores(FilterSpec.Storage.FORECAST)) {
        total += (long) kEndog(rep) * kEndog(rep) * nobs(rep);
    }
    return total;
    }

    private static double[] activeSlice(double[] values, int base, int length) {
    return Arrays.copyOfRange(values, base, base + length);
    }

    private static double[] nullableActiveSlice(double[] values, int base, int length) {
    return values == null ? null : Arrays.copyOfRange(values, base, base + length);
    }

    private static void assertNullableArrayEquals(double[] expected, double[] actual) {
    assertEquals(expected == null, actual == null);
    if (expected != null) {
        assertArrayEquals(expected, actual, TOL);
    }
    }

    private static void assertFilterOutputsMatch(FilterResult expected, FilterResult actual) {
    assertArrayEquals(activeSlice(expected.predictedState, expected.predictedStateBase(), expected.predictedStateLength()),
        activeSlice(actual.predictedState, actual.predictedStateBase(), actual.predictedStateLength()), TOL);
    assertArrayEquals(activeSlice(expected.predictedStateCov, expected.predictedStateCovBase(), expected.predictedStateCovLength()),
        activeSlice(actual.predictedStateCov, actual.predictedStateCovBase(), actual.predictedStateCovLength()), TOL);
    assertArrayEquals(activeSlice(expected.filteredState, expected.filteredStateBase(), expected.filteredStateLength()),
        activeSlice(actual.filteredState, actual.filteredStateBase(), actual.filteredStateLength()), TOL);
    assertArrayEquals(activeSlice(expected.filteredStateCov, expected.filteredStateCovBase(), expected.filteredStateCovLength()),
        activeSlice(actual.filteredStateCov, actual.filteredStateCovBase(), actual.filteredStateCovLength()), TOL);
    assertArrayEquals(activeSlice(expected.forecast, expected.forecastBase(), expected.forecastLength()),
        activeSlice(actual.forecast, actual.forecastBase(), actual.forecastLength()), TOL);
    assertArrayEquals(activeSlice(expected.forecastError, expected.forecastErrorBase(), expected.forecastErrorLength()),
        activeSlice(actual.forecastError, actual.forecastErrorBase(), actual.forecastErrorLength()), TOL);
    assertArrayEquals(activeSlice(expected.forecastErrorCov, expected.forecastErrorCovBase(), expected.forecastErrorCovLength()),
        activeSlice(actual.forecastErrorCov, actual.forecastErrorCovBase(), actual.forecastErrorCovLength()), TOL);
    assertArrayEquals(activeSlice(expected.kalmanGain, expected.kalmanGainBase(), expected.kalmanGainLength()),
        activeSlice(actual.kalmanGain, actual.kalmanGainBase(), actual.kalmanGainLength()), TOL);
    assertArrayEquals(activeSlice(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
        activeSlice(actual.logLikelihoodObs, actual.logLikelihoodObsBase(), actual.logLikelihoodObsLength()), TOL);
    assertNullableArrayEquals(
        nullableActiveSlice(expected.predictedDiffuseStateCov, expected.predictedDiffuseStateCovBase(), expected.predictedDiffuseStateCovLength()),
        nullableActiveSlice(actual.predictedDiffuseStateCov, actual.predictedDiffuseStateCovBase(), actual.predictedDiffuseStateCovLength()));
    assertNullableArrayEquals(
        nullableActiveSlice(expected.forecastErrorDiffuseCov, expected.forecastErrorDiffuseCovBase(), expected.forecastErrorDiffuseCovLength()),
        nullableActiveSlice(actual.forecastErrorDiffuseCov, actual.forecastErrorDiffuseCovBase(), actual.forecastErrorDiffuseCovLength()));
    assertEquals(expected.nobsDiffuse, actual.nobsDiffuse);
    }

    private static ModelFixture buildAR1() {
        return KalmanModelFixtures.defaultScalarAr1();
    }

    private static ModelFixture buildAR1(double[] y) {
        return KalmanModelFixtures.scalarAr1(0.5, 0.5, 1.0, 1.0, y);
    }

    private static ModelFixture buildLocalLevel() {
        return KalmanModelFixtures.defaultLocalLevel();
    }

    private static ModelFixture buildLocalLevel(double[] y) {
        return KalmanModelFixtures.localLevel(1.0, 0.5, y);
    }

    private static StateSpaceModel buildConstantVarianceConvergingModel(int nobs) {
        return buildConstantVarianceConvergingModel(1, 1, nobs);
    }

    private static StateSpaceModel buildConstantVarianceConvergingModel(int kEndog, int kStates, int nobs) {
        int kPosdef = kStates;
        double[] Z = new double[kEndog * kStates];
        for (int i = 0; i < Math.min(kEndog, kStates); i++) {
            Z[i * kStates + i] = 1.0;
        }
        double[] d = new double[kEndog];
        double[] H = diagonal(kEndog, 1.0);
        double[] T = new double[kStates * kStates];
        double[] c = new double[kStates];
        double[] R = diagonal(kStates, 1.0);
        double[] Q = diagonal(kStates, 0.5);
        double[] y = new double[kEndog * nobs];
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                y[t * kEndog + i] = ((t & 1) == 0 ? 0.25 : -0.75) * (i + 1);
            }
        }
        return StateSpaceModel.builder(kEndog, kStates, kPosdef, nobs)
            .design(Z, false)
            .obsIntercept(d, false)
            .obsCov(H, false)
            .transition(T, false)
            .stateIntercept(c, false)
            .selection(R, false)
            .stateCovariance(Q, false)
            .endog(y)
            .allObserved()
            .build();
    }

    private static StateSpaceModel buildTimeVaryingSteadyStateTrapModel() {
        int nobs = 6;
        return StateSpaceModel.builder(1, 1, 1, nobs)
            .design(new double[]{1.0, 2.0, 1.0, 2.0, 1.0, 2.0}, true)
            .obsIntercept(new double[]{0.0}, false)
            .obsCov(new double[]{1.0}, false)
            .transition(new double[]{0.0}, false)
            .stateIntercept(new double[]{0.0}, false)
            .selection(new double[]{1.0}, false)
            .stateCovariance(new double[]{0.5}, false)
            .endog(new double[]{0.25, -0.75, 0.5, -0.25, 0.75, -0.5})
            .allObserved()
            .build();
    }

    private static StateSpaceModel buildMissingObservationConvergingModel() {
        StateSpaceModel rep = buildConstantVarianceConvergingModel(1, 1, 12);
        rep.setMissing(new boolean[]{true}, 1);
        return rep;
    }

    private static StateSpaceModel buildMissingForecastTailModel() {
        int nobs = 4;
        ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
        double[] Z = {1.0};
        double[] H = {0.0};
        double[] T = {0.0};
        double[] c = {0.0};
        double[] R = {1.0};
        double[] Q = {0.35};
        double[] y = {1.2, -0.7, 0.0, 0.0};
        double[] intercept = {0.8, -0.4, 1.2, -1.6};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(new double[]{intercept[t]}, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        rep.setMissing(new boolean[]{true}, 2);
        rep.setMissing(new boolean[]{true}, 3);
        return rep;
    }

    private static StateSpaceModel buildSingularInnovationModel() {
        int nobs = 3;
        ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
        double[] Z = {1.0};
        double[] H = {0.0};
        double[] T = {0.0};
        double[] c = {0.0};
        double[] R = {1.0};
        double[] Q = {0.35};
        double[] y = {1.2, -0.7, 0.4};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(new double[]{0.0}, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        return rep;
    }

    private static ModelFixture buildMultivariate() {
        return KalmanModelFixtures.defaultBivariate2State();
    }

    @Test
    void testAR1Model() {
        ModelFixture rep = buildAR1();
        FilterResult r = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        assertEquals(0.0, r.forecast(0, 0), TOL);
        assertEquals(1.0, r.forecastError(0, 0), TOL);
        assertEquals(250001.0, r.forecastErrorCov(0, 0, 0), TOL);

        double expectedFiltState0 = 500000.0 / 250001.0;
        assertEquals(expectedFiltState0, r.filteredState(0, 0), TOL);

        double expectedFiltStateCov0 = 1e6 - 250000000000.0 / 250001.0;
        assertEquals(expectedFiltStateCov0, r.filteredStateCov(0, 0, 0), TOL);

        double expectedPredState1 = 0.5 * expectedFiltState0;
        assertEquals(expectedPredState1, r.predictedState(0, 1), TOL);

        double expectedPredStateCov1 = 0.5 * expectedFiltStateCov0 * 0.5 + 1.0;
        assertEquals(expectedPredStateCov1, r.predictedStateCov(0, 0, 1), TOL);
    }

    @Test
    void testLocalLevelModel() {
        ModelFixture rep = buildLocalLevel();
        double[] a0 = {0.0};
        double[] P0 = {1.0};
        FilterResult r = KalmanFilter.filter(rep, StateInitialization.known(a0, P0));

        assertEquals(0.0, r.forecast(0, 0), TOL);
        assertEquals(1.0, r.forecastError(0, 0), TOL);
        assertEquals(2.0, r.forecastErrorCov(0, 0, 0), TOL);
        assertEquals(0.5, r.filteredState(0, 0), TOL);
        assertEquals(0.5, r.filteredStateCov(0, 0, 0), TOL);

        assertEquals(0.5, r.predictedState(0, 1), TOL);
        assertEquals(1.0, r.predictedStateCov(0, 0, 1), TOL);

        assertEquals(0.5, r.forecast(0, 1), TOL);
        assertEquals(1.5, r.forecastError(0, 1), TOL);
        assertEquals(2.0, r.forecastErrorCov(0, 0, 1), TOL);
        assertEquals(1.25, r.filteredState(0, 1), TOL);
        assertEquals(0.5, r.filteredStateCov(0, 0, 1), TOL);

        assertEquals(1.25, r.predictedState(0, 2), TOL);
        assertEquals(1.0, r.predictedStateCov(0, 0, 2), TOL);
    }

    @Test
    void testMultivariateModel() {
        ModelFixture rep = buildMultivariate();
        FilterResult r = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        assertEquals(0.0, r.forecast(0, 0), TOL);
        assertEquals(0.0, r.forecast(1, 0), TOL);
        assertEquals(1.0, r.forecastError(0, 0), TOL);
        assertEquals(0.5, r.forecastError(1, 0), TOL);
        assertEquals(1000001.0, r.forecastErrorCov(0, 0, 0), TOL);
        assertEquals(0.0, r.forecastErrorCov(0, 1, 0), TOL);
        assertEquals(0.0, r.forecastErrorCov(1, 0, 0), TOL);
        assertEquals(1000001.0, r.forecastErrorCov(1, 1, 0), TOL);

        double expectedFiltState0 = 1e6 / 1000001.0;
        double expectedFiltState1 = 0.5 * 1e6 / 1000001.0;
        assertEquals(expectedFiltState0, r.filteredState(0, 0), TOL);
        assertEquals(expectedFiltState1, r.filteredState(1, 0), TOL);

        double expectedFiltCov = 1e6 - 1e12 / 1000001.0;
        assertEquals(expectedFiltCov, r.filteredStateCov(0, 0, 0), TOL);
        assertEquals(0.0, r.filteredStateCov(0, 1, 0), TOL);
        assertEquals(0.0, r.filteredStateCov(1, 0, 0), TOL);
        assertEquals(expectedFiltCov, r.filteredStateCov(1, 1, 0), TOL);
    }

    @Test
    void testLogLikelihood() {
        ModelFixture rep = buildAR1();
        FilterResult r = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        double expectedLL0 = -0.5 * (Math.log(2 * Math.PI) + Math.log(250001.0) + 1.0 / 250001.0);
        assertEquals(expectedLL0, r.logLikelihoodObs[0], TOL);

        double totalLL = 0.0;
        for (int t = 0; t < r.nobs; t++) {
            totalLL += r.logLikelihoodObs[t];
        }
        assertEquals(totalLL, r.logLikelihood(), TOL);
    }

    @Test
    void testSymmetry() {
        ModelFixture rep = buildMultivariate();
        FilterResult r = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        for (int t = 0; t <= r.nobs; t++) {
            for (int i = 0; i < r.kStates; i++) {
                for (int j = i + 1; j < r.kStates; j++) {
                    assertEquals(r.predictedStateCov(i, j, t),
                            r.predictedStateCov(j, i, t), TOL);
                }
            }
        }
        for (int t = 0; t < r.nobs; t++) {
            for (int i = 0; i < r.kStates; i++) {
                for (int j = i + 1; j < r.kStates; j++) {
                    assertEquals(r.filteredStateCov(i, j, t),
                            r.filteredStateCov(j, i, t), TOL);
                }
            }
            for (int i = 0; i < r.kEndog; i++) {
                for (int j = i + 1; j < r.kEndog; j++) {
                    assertEquals(r.forecastErrorCov(i, j, t),
                            r.forecastErrorCov(j, i, t), TOL);
                }
            }
        }
    }

    @Test
    void testPoolReuse() {
        ModelFixture rep = buildAR1();
        KalmanFilter.Pool pool = new KalmanFilter.Pool();

        FilterResult r1 = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool);
        FilterResult r2 = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool);

        assertSame(r1, r2);
    }

    @Test
    void testApproximateDiffusePoolReuseResetsInitialStateAndCovariance() {
        StateSpaceModel rep = buildConstantVarianceConvergingModel(1, 2, 4);
        KalmanFilter.Pool pool = new KalmanFilter.Pool();

        KalmanFilter.filter(rep,
            StateInitialization.known(
                new double[]{2.0, -3.0},
                new double[]{4.0, 1.5, 1.5, 5.0}),
            pool);
        FilterResult reused = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool);

        int stateBase = reused.predictedStateBase();
        int covBase = reused.predictedStateCovBase();
        assertEquals(0.0, reused.predictedState[stateBase], TOL);
        assertEquals(0.0, reused.predictedState[stateBase + 1], TOL);
        assertEquals(1e6, reused.predictedStateCov[covBase], TOL);
        assertEquals(0.0, reused.predictedStateCov[covBase + 1], TOL);
        assertEquals(0.0, reused.predictedStateCov[covBase + 2], TOL);
        assertEquals(1e6, reused.predictedStateCov[covBase + 3], TOL);
    }

    @Test
    void testPoolReserveQuantifiesRetainedRealFilterMemory() {
        ModelFixture rep = buildAR1();
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        long expectedScratchDoubles = expectedFilterScratchDoubleCount(rep);
        long expectedResultDoubles = expectedFilterResultDoubleCount(rep);
        long expectedBaseScratchDoubles = expectedFilterBaseScratchDoubleCount(rep);

        pool.reserve(rep, FilterSpec.defaults());

        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedBaseScratchDoubles, (long) pool.scratchBacking.length);
        assertNull(pool.selectionBacking);
        assertEquals(expectedResultDoubles, pool.retainedResultDoubleCount());
        assertEquals(expectedResultDoubles, (long) pool.resultBacking.length);
        assertNull(pool.diffuseResultBacking);
        assertEquals((expectedScratchDoubles + expectedResultDoubles) * Double.BYTES,
            pool.retainedTotalByteCount());
    }

    @Test
    void testStationaryInitRetainsDedicatedWorkspaceAlongsideScratchBacking() {
        ModelFixture rep = buildAR1();
        KalmanFilter.Pool pool = new KalmanFilter.Pool();

        KalmanFilter.filter(rep, StateInitialization.stationary(rep.stateCount()), pool);

        long expectedWorkspaceDoubles = StateInitialization.requiredStationaryBackingLength(rep)
            + StateInitialization.requiredStationaryPivotLength(rep) * ((long) Integer.BYTES) / Double.BYTES;
        assertEquals(expectedFilterBaseScratchDoubleCount(rep) + expectedWorkspaceDoubles,
            pool.retainedScratchDoubleCount());
        assertEquals(expectedFilterBaseScratchDoubleCount(rep), (long) pool.scratchBacking.length);
    }

    @Test
    void testDiffusePoolQuantifiesAdditionalRealFilterBanks() {
        ModelFixture rep = buildLocalLevel();
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        long expectedScratchDoubles = expectedDiffuseFilterScratchDoubleCount(rep, FilterSpec.defaults());
        long expectedResultDoubles = expectedFilterResultDoubleCount(rep);
        long expectedDiffuseResultDoubles = expectedFilterDiffuseResultDoubleCount(rep);

        pool.reserve(rep, FilterSpec.defaults());

        FilterResult result = KalmanFilter.filter(rep, StateInitialization.diffuse(), pool);

        assertTrue(result.nobsDiffuse > 0);
        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedResultDoubles + expectedDiffuseResultDoubles,
            pool.retainedResultDoubleCount());
        assertEquals(expectedDiffuseResultDoubles, (long) pool.diffuseResultBacking.length);
        assertEquals((expectedScratchDoubles + expectedResultDoubles + expectedDiffuseResultDoubles) * Double.BYTES,
            pool.retainedTotalByteCount());
    }

    @Test
    void testReserveTrimsRealFilterOptionalRetainedBanksBackToBaseline() {
        ModelFixture rep = buildLocalLevel();
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        long expectedScratchDoubles = expectedFilterScratchDoubleCount(rep);
        long expectedResultDoubles = expectedFilterResultDoubleCount(rep);

        KalmanFilter.filter(rep, StateInitialization.diffuse(), pool);

        assertTrue(pool.retainedScratchDoubleCount() > expectedScratchDoubles);
        assertTrue(pool.retainedResultDoubleCount() > expectedResultDoubles);
        assertNotNull(pool.diffuseResultBacking);
        assertTrue(pool.diffusePool != null && pool.diffusePool.retainedScratchDoubleCount() > 0);

        pool.reserve(rep, FilterSpec.defaults());

        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedResultDoubles, pool.retainedResultDoubleCount());
        assertNull(pool.diffuseResultBacking);
        assertNotNull(pool.resultBacking);
        assertEquals(expectedResultDoubles, (long) pool.resultBacking.length);
        assertEquals(0L, pool.diffusePool == null ? 0L : pool.diffusePool.retainedScratchDoubleCount());
    }

    @Test
    void testPoolDefersConvergedSnapshotAllocationUntilNeeded() {
        StateSpaceModel rep = buildConstantVarianceConvergingModel(1);
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        long expectedSnapshotDoubles = expectedConvergedSnapshotDoubleCount(rep);

        pool.reserve(rep, FilterSpec.defaults());
        long baselineScratchDoubles = pool.retainedScratchDoubleCount();

        assertNull(pool.convergedForecastErrorCov);
        assertNull(pool.convergedFilteredStateCov);
        assertNull(pool.convergedPredictedStateCov);
        assertNull(pool.convergedKalmanGain);
        assertEquals(0L, convergedSnapshotDoubleCount(pool));
        assertEquals(4L, expectedSnapshotDoubles);

        KalmanFilter.filter(rep, StateInitialization.known(new double[]{0.0}, diagonal(1, 0.5)), pool);

        assertNull(pool.convergedForecastErrorCov);
        assertNull(pool.convergedFilteredStateCov);
        assertNull(pool.convergedPredictedStateCov);
        assertNull(pool.convergedKalmanGain);
        assertEquals(0L, convergedSnapshotDoubleCount(pool));
        assertEquals(baselineScratchDoubles, pool.retainedScratchDoubleCount());
    }

    @Test
    void testPoolAllocatesConvergedSnapshotsWhenFilterConverges() {
        StateSpaceModel rep = buildConstantVarianceConvergingModel(2, 2, 6);
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        long expectedSnapshotDoubles = expectedConvergedSnapshotDoubleCount(rep);

        pool.reserve(rep, FilterSpec.defaults());
        long baselineScratchDoubles = pool.retainedScratchDoubleCount();

        FilterResult result = KalmanFilter.filter(rep,
            StateInitialization.known(new double[rep.kStates], diagonal(rep.kStates, 0.5)), pool);

        assertTrue(result.converged);
        assertEquals(2, result.periodConverged);
        assertEquals(expectedSnapshotDoubles, convergedSnapshotDoubleCount(pool));
        assertEquals(baselineScratchDoubles + expectedSnapshotDoubles, pool.retainedScratchDoubleCount());
        assertNotNull(pool.convergedForecastErrorCov);
        assertNotNull(pool.convergedFilteredStateCov);
        assertNotNull(pool.convergedPredictedStateCov);
        assertNotNull(pool.convergedKalmanGain);
        assertEquals((long) rep.kEndog * rep.kEndog, doubleCount(pool.convergedForecastErrorCov));
        assertEquals((long) rep.kStates * rep.kStates, doubleCount(pool.convergedFilteredStateCov));
        assertEquals((long) rep.kStates * rep.kStates, doubleCount(pool.convergedPredictedStateCov));
        assertEquals((long) rep.kStates * rep.kEndog, doubleCount(pool.convergedKalmanGain));
        assertEquals(16L, expectedSnapshotDoubles);
    }

    @Test
    void testPoolReleaseRetainedScratchClearsOwnedBanks() {
        ModelFixture rep = buildLocalLevel();
        KalmanFilter.Pool pool = new KalmanFilter.Pool();

        KalmanFilter.filter(rep, StateInitialization.diffuse(), pool);

        assertTrue(pool.retainedScratchDoubleCount() > 0);

        pool.releaseRetainedScratch();

        assertEquals(0L, pool.retainedScratchDoubleCount());
        assertNull(pool.scratchBacking);
        assertNull(pool.selectionBacking);
        assertEquals(0L, pool.diffusePool == null ? 0L : pool.diffusePool.retainedScratchDoubleCount());
    }

    @Test
    void testTimeVaryingModelDoesNotReuseSteadyState() {
        StateSpaceModel rep = buildTimeVaryingSteadyStateTrapModel();
        StateInitialization init = StateInitialization.known(new double[]{0.0}, new double[]{0.5});

        FilterResult conventional = KalmanFilter.filter(rep, init);
        FilterResult reference = UnivariateFilter.filter(rep, init);

        assertFalse(rep.isTimeInvariant());
        assertFalse(conventional.converged);
        assertEquals(rep.observationCount(), conventional.periodConverged);
        for (int t = 0; t < rep.observationCount(); t++) {
            assertEquals(reference.filteredState(0, t), conventional.filteredState(0, t), TOL);
            assertEquals(reference.filteredStateCov(0, 0, t), conventional.filteredStateCov(0, 0, t), TOL);
            if (rep.missingCount(t) == 0) {
                assertEquals(reference.forecastErrorCov(0, 0, t), conventional.forecastErrorCov(0, 0, t), TOL);
            }
        }
    }

    @Test
    void testMissingObservationDefersSteadyStateReuse() {
        StateSpaceModel rep = buildMissingObservationConvergingModel();
        StateInitialization init = StateInitialization.known(new double[]{0.0}, new double[]{0.5});

        FilterResult conventional = KalmanFilter.filter(rep, init);
        FilterResult reference = UnivariateFilter.filter(rep, init);

        assertTrue(rep.isTimeInvariant());
        assertEquals(1, rep.missingCount(1));
        assertTrue(conventional.converged);
        assertTrue(conventional.periodConverged > 3,
            "steady-state reuse should not resume immediately after a missing period");
        for (int t = 0; t < rep.observationCount(); t++) {
            assertEquals(reference.filteredState(0, t), conventional.filteredState(0, t), TOL);
            assertEquals(reference.filteredStateCov(0, 0, t), conventional.filteredStateCov(0, 0, t), TOL);
            if (rep.missingCount(t) == 0) {
                assertEquals(reference.forecastErrorCov(0, 0, t), conventional.forecastErrorCov(0, 0, t), TOL);
            }
        }
    }

    @Test
    void testPooledFilterPreservesForecastMeanForFullyMissingTailObservations() {
        StateSpaceModel rep = buildMissingForecastTailModel();
        StateInitialization init = StateInitialization.known(new double[]{0.0}, new double[]{0.35});
        KalmanFilter.Pool pool = new KalmanFilter.Pool();

        FilterResult pooled = KalmanFilter.filter(rep, init, pool, FilterSpec.full());
        FilterResult reference = UnivariateFilter.filter(rep, init, null, FilterSpec.full());

        assertEquals(reference.forecast(0, 2), pooled.forecast(0, 2), TOL);
        assertEquals(reference.forecast(0, 3), pooled.forecast(0, 3), TOL);
        assertEquals(reference.forecastErrorCov(0, 0, 2), pooled.forecastErrorCov(0, 0, 2), TOL);
        assertEquals(reference.forecastErrorCov(0, 0, 3), pooled.forecastErrorCov(0, 0, 3), TOL);
        assertEquals(1.2, pooled.forecast(0, 2), TOL);
        assertEquals(-1.6, pooled.forecast(0, 3), TOL);
    }

    @Test
    void testScalarConventionalFilterSkipsSingularInnovationUpdate() {
        StateSpaceModel rep = buildSingularInnovationModel();
        StateInitialization init = StateInitialization.known(new double[]{0.0}, new double[]{0.0});

        FilterResult conventional = KalmanFilter.filter(rep, init);
        FilterResult reference = UnivariateFilter.filter(rep, init);

        for (int t = 0; t < rep.observationCount(); t++) {
            assertEquals(reference.forecast(0, t), conventional.forecast(0, t), TOL);
            assertEquals(reference.forecastError(0, t), conventional.forecastError(0, t), TOL);
            assertEquals(reference.forecastErrorCov(0, 0, t), conventional.forecastErrorCov(0, 0, t), TOL);
            assertEquals(reference.filteredState(0, t), conventional.filteredState(0, t), TOL);
            assertEquals(reference.filteredStateCov(0, 0, t), conventional.filteredStateCov(0, 0, t), TOL);
            assertFalse(Double.isNaN(conventional.filteredState(0, t)));
            assertFalse(Double.isNaN(conventional.filteredStateCov(0, 0, t)));
        }
        for (int t = 0; t <= rep.observationCount(); t++) {
            assertEquals(reference.predictedState(0, t), conventional.predictedState(0, t), TOL);
            assertEquals(reference.predictedStateCov(0, 0, t), conventional.predictedStateCov(0, 0, t), TOL);
            assertFalse(Double.isNaN(conventional.predictedState(0, t)));
            assertFalse(Double.isNaN(conventional.predictedStateCov(0, 0, t)));
        }
    }

    @Test
    void testClonePreservesBorrowedResultSnapshot() {
        ModelFixture rep1 = buildAR1(new double[]{1.0, 0.5, -0.3, 0.8, 0.2});
        ModelFixture rep2 = buildAR1(new double[]{-2.0, -1.0, 1.5, 0.4, -0.2});
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        FilterResult expected = KalmanFilter.filter(rep1, StateInitialization.approximateDiffuse());

        FilterResult borrowed = KalmanFilter.filter(rep1, StateInitialization.approximateDiffuse(), pool);
        FilterResult snapshot = borrowed.clone();

        KalmanFilter.filter(rep2, StateInitialization.approximateDiffuse(), pool);

        assertNotSame(snapshot, borrowed);

        for (int t = 0; t < rep1.nobs; t++) {
            assertEquals(expected.forecast(0, t), snapshot.forecast(0, t), TOL);
            assertEquals(expected.forecastError(0, t), snapshot.forecastError(0, t), TOL);
            assertEquals(expected.forecastErrorCov(0, 0, t), snapshot.forecastErrorCov(0, 0, t), TOL);
            assertEquals(expected.filteredState(0, t), snapshot.filteredState(0, t), TOL);
            assertEquals(expected.filteredStateCov(0, 0, t), snapshot.filteredStateCov(0, 0, t), TOL);
            assertEquals(expected.predictedState(0, t), snapshot.predictedState(0, t), TOL);
            assertEquals(expected.predictedStateCov(0, 0, t), snapshot.predictedStateCov(0, 0, t), TOL);
            assertEquals(expected.logLikelihoodObs[t], snapshot.logLikelihoodObs[t], TOL);
        }
    }

    @Test
    void testPoolReuseInvalidatesPreviousResult() {
        ModelFixture rep1 = buildAR1(new double[]{1.0, 0.5, -0.3, 0.8, 0.2});
        ModelFixture rep2 = buildAR1(new double[]{-2.0, -1.0, 1.5, 0.4, -0.2});
        KalmanFilter.Pool pool = new KalmanFilter.Pool();

        pool.reserve(rep1, FilterSpec.defaults());

        FilterResult r1 = KalmanFilter.filter(rep1, StateInitialization.approximateDiffuse(), pool, FilterSpec.defaults());
        double firstFilteredState = r1.filteredState(0, 0);

        FilterResult r2 = KalmanFilter.filter(rep2, StateInitialization.approximateDiffuse(), pool, FilterSpec.defaults());

        assertSame(r1, r2);
        assertNotEquals(firstFilteredState, r1.filteredState(0, 0), TOL);
        assertEquals(r2.filteredState(0, 0), r1.filteredState(0, 0), TOL);
    }

    @Test
    void testPoolCompactMaskWithMissingObservation() {
        ModelFixture rep = buildAR1(new double[]{1.0, Double.NaN, -0.3, 0.8, 0.2});
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        FilterSpec compactSpec = FilterSpec.defaults().without(
                FilterSpec.Storage.PREDICTED_STATE,
                FilterSpec.Storage.FILTERED_STATE,
                FilterSpec.Storage.KALMAN_GAIN,
                FilterSpec.Storage.LIKELIHOOD);
        long fullScratchDoubles = expectedFilterScratchDoubleCount(rep, FilterSpec.defaults());
        long fullResultDoubles = expectedFilterResultDoubleCount(rep, FilterSpec.defaults());
        long compactScratchDoubles = expectedFilterScratchDoubleCount(rep, compactSpec);
        long compactResultDoubles = expectedFilterResultDoubleCount(rep, compactSpec);

        pool.reserve(rep, compactSpec);

        FilterResult result = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool, compactSpec);

        assertTrue(compactResultDoubles < fullResultDoubles);
        assertEquals(compactScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(compactResultDoubles, pool.retainedResultDoubleCount());
        assertEquals(compactResultDoubles, (long) pool.resultBacking.length);
        assertTrue(pool.retainedTotalByteCount() < (fullScratchDoubles + fullResultDoubles) * Double.BYTES);
        assertEquals(0, result.predictedState.length);
        assertEquals(0, result.predictedStateCov.length);
        assertEquals(0, result.filteredState.length);
        assertEquals(0, result.filteredStateCov.length);
        assertEquals(0, result.kalmanGain.length);
        assertEquals(0, result.logLikelihoodObs.length);
        assertTrue(Double.isFinite(result.forecast(0, 1)));
        assertTrue(Double.isFinite(result.forecastErrorCov(0, 0, 1)));
    }

    @Test
    void testPoolSupportsDiffuseInitialization() {
        ModelFixture rep = buildLocalLevel();
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        FilterResult expected = KalmanFilter.filter(rep, StateInitialization.diffuse());

        FilterResult borrowed1 = KalmanFilter.filter(rep, StateInitialization.diffuse(), pool);
        FilterResult borrowed2 = KalmanFilter.filter(rep, StateInitialization.diffuse(), pool);

        assertSame(borrowed1, borrowed2);
        assertTrue(borrowed1.nobsDiffuse > 0);
        assertFilterOutputsMatch(expected, borrowed1);
    }

    @Test
    void testClonePreservesBorrowedDiffuseResultSnapshot() {
        ModelFixture rep1 = buildLocalLevel(new double[]{1.0, 2.0, 1.5, 3.0, 2.5});
        ModelFixture rep2 = buildLocalLevel(new double[]{-1.5, 0.25, 2.0, -0.75, 0.5});
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        FilterResult expected = KalmanFilter.filter(rep1, StateInitialization.diffuse());

        FilterResult borrowed = KalmanFilter.filter(rep1, StateInitialization.diffuse(), pool);
        FilterResult snapshot = borrowed.clone();

        KalmanFilter.filter(rep2, StateInitialization.diffuse(), pool);

        assertNotSame(snapshot, borrowed);
        assertFilterOutputsMatch(expected, snapshot);
    }

    @Test
    void testBorrowedDiffuseCompactResultUsesLogicalSuppression() {
        ModelFixture rep = buildLocalLevel();
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        FilterSpec compactSpec = FilterSpec.defaults().without(
                FilterSpec.Storage.PREDICTED_STATE,
                FilterSpec.Storage.FILTERED_STATE,
                FilterSpec.Storage.KALMAN_GAIN,
                FilterSpec.Storage.LIKELIHOOD);

        pool.reserve(rep, compactSpec);

        FilterResult result = KalmanFilter.filter(rep, StateInitialization.diffuse(), pool, compactSpec);

        assertTrue(result.nobsDiffuse > 0);
        assertEquals(0, result.predictedStateLength());
        assertEquals(0, result.predictedStateCovLength());
        assertEquals(0, result.predictedDiffuseStateCovLength());
        assertEquals(0, result.filteredStateLength());
        assertEquals(0, result.filteredStateCovLength());
        assertEquals(0, result.kalmanGainLength());
        assertEquals(0, result.logLikelihoodObsLength());
        assertTrue(result.forecastLength() > 0);
        assertTrue(result.forecastErrorLength() > 0);
        assertTrue(result.forecastErrorCovLength() > 0);
        assertTrue(result.forecastErrorDiffuseCovLength() > 0);
    }

    @Test
    void testBorrowedDiffuseCompactResultShrinksRetainedBanks() {
        ModelFixture rep = buildLocalLevel();
        KalmanFilter.Pool pool = new KalmanFilter.Pool();
        FilterSpec compactSpec = FilterSpec.defaults().without(
                FilterSpec.Storage.FORECAST,
                FilterSpec.Storage.PREDICTED_STATE,
                FilterSpec.Storage.FILTERED_STATE,
                FilterSpec.Storage.KALMAN_GAIN,
                FilterSpec.Storage.LIKELIHOOD);
        long fullScratchDoubles = expectedDiffuseFilterScratchDoubleCount(rep, FilterSpec.defaults());
        long fullRegularResultDoubles = expectedDiffuseRegularFilterResultDoubleCount(rep, FilterSpec.defaults());
        long fullDiffuseResultDoubles = expectedFilterDiffuseResultDoubleCount(rep, FilterSpec.defaults());
        long compactScratchDoubles = expectedDiffuseFilterScratchDoubleCount(rep, compactSpec);
        long compactRegularResultDoubles = expectedDiffuseRegularFilterResultDoubleCount(rep, compactSpec);
        long compactDiffuseResultDoubles = expectedFilterDiffuseResultDoubleCount(rep, compactSpec);

        pool.reserve(rep, compactSpec);

        FilterResult result = KalmanFilter.filter(rep, StateInitialization.diffuse(), pool, compactSpec);

        assertTrue(result.nobsDiffuse > 0);
        assertTrue(compactRegularResultDoubles < fullRegularResultDoubles);
        assertTrue(compactDiffuseResultDoubles < fullDiffuseResultDoubles);
        assertEquals(compactScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(compactRegularResultDoubles + compactDiffuseResultDoubles,
            pool.retainedResultDoubleCount());
        assertEquals(compactRegularResultDoubles, (long) pool.resultBacking.length);
        assertEquals(compactDiffuseResultDoubles, (long) pool.diffuseResultBacking.length);
        assertTrue(pool.retainedTotalByteCount()
            < (fullScratchDoubles + fullRegularResultDoubles + fullDiffuseResultDoubles) * Double.BYTES);
        assertEquals(0, result.predictedStateLength());
        assertEquals(0, result.predictedStateCovLength());
        assertEquals(0, result.predictedDiffuseStateCovLength());
        assertEquals(0, result.filteredStateLength());
        assertEquals(0, result.filteredStateCovLength());
        assertEquals(0, result.forecastLength());
        assertEquals(0, result.forecastErrorLength());
        assertEquals(0, result.forecastErrorCovLength());
        assertEquals(0, result.forecastErrorDiffuseCovLength());
        assertEquals(0, result.kalmanGainLength());
        assertEquals(0, result.logLikelihoodObsLength());
    }

    @Test
    void testStoreFilteredFalseUnivariate() {
        ModelFixture rep = buildAR1();

        FilterResult rFull = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        FilterResult rNoFilt = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), null,
            FilterSpec.defaults().without(FilterSpec.Storage.FILTERED_STATE));

        for (int t = 1; t <= rep.nobs; t++) {
            for (int i = 0; i < rep.kStates; i++) {
                assertEquals(rFull.predictedState(i, t), rNoFilt.predictedState(i, t), 1e-10,
                        "predictedState mismatch at t=" + t + " i=" + i);
            }
            for (int i = 0; i < rep.kStates; i++) {
                for (int j = 0; j < rep.kStates; j++) {
                    assertEquals(rFull.predictedStateCov(i, j, t), rNoFilt.predictedStateCov(i, j, t), 1e-10,
                            "predictedStateCov mismatch at t=" + t + " i=" + i + " j=" + j);
                }
            }
        }

        assertEquals(0, rNoFilt.filteredState.length);
        assertEquals(0, rNoFilt.filteredStateCov.length);
    }

    @Test
    void testStoreFilteredFalseMultivariate() {
        ModelFixture rep = buildMultivariate();

        FilterResult rFull = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        FilterResult rNoFilt = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), null,
            FilterSpec.defaults().without(FilterSpec.Storage.FILTERED_STATE));

        for (int t = 1; t <= rep.nobs; t++) {
            for (int i = 0; i < rep.kStates; i++) {
                assertEquals(rFull.predictedState(i, t), rNoFilt.predictedState(i, t), 1e-10,
                        "predictedState mismatch at t=" + t + " i=" + i);
            }
            for (int i = 0; i < rep.kStates; i++) {
                for (int j = 0; j < rep.kStates; j++) {
                    assertEquals(rFull.predictedStateCov(i, j, t), rNoFilt.predictedStateCov(i, j, t), 1e-10,
                            "predictedStateCov mismatch at t=" + t + " i=" + i + " j=" + j);
                }
            }
        }

        assertEquals(0, rNoFilt.filteredState.length);
        assertEquals(0, rNoFilt.filteredStateCov.length);
    }

    @Test
    void testStoreFilteredFalseLocalLevel() {
        ModelFixture rep = buildLocalLevel();
        double[] a0 = {0.0};
        double[] P0 = {1.0};

        FilterResult rFull = KalmanFilter.filter(rep, StateInitialization.known(a0, P0));
        FilterResult rNoFilt = KalmanFilter.filter(rep, StateInitialization.known(a0, P0), null,
            FilterSpec.defaults().without(FilterSpec.Storage.FILTERED_STATE));

        for (int t = 1; t <= rep.nobs; t++) {
            for (int i = 0; i < rep.kStates; i++) {
                assertEquals(rFull.predictedState(i, t), rNoFilt.predictedState(i, t), 1e-10,
                        "predictedState mismatch at t=" + t + " i=" + i);
            }
            for (int i = 0; i < rep.kStates; i++) {
                for (int j = 0; j < rep.kStates; j++) {
                    assertEquals(rFull.predictedStateCov(i, j, t), rNoFilt.predictedStateCov(i, j, t), 1e-10,
                            "predictedStateCov mismatch at t=" + t + " i=" + i + " j=" + j);
                }
            }
        }
    }

    @Test
    void testUnivariateForecastPerObs() {
        int kEndog = 2, kStates = 2, kPosdef = 1, nobs = 3;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1, 0, 0, 1};
        double[] d = {0, 0};
        double[] H = {1.0, 0, 0, 1.0};
        double[] T = {0.9, 0.1, 0.0, 0.9};
        double[] c = {0, 0};
        double[] R = {1, 0};
        double[] Q = {1.0};
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t * 2], y[t * 2 + 1]}, t);
        }
        FilterResult r = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse());

        assertEquals(0.0, r.forecast(0, 0), 1e-10);
        assertEquals(0.0, r.forecast(1, 0), 1e-6);
        assertEquals(1.0, r.forecastError(0, 0), 1e-10);
        assertEquals(0.5, r.forecastError(1, 0), 1e-10);
    }

    @Test
    void testUnivariateKalmanGainPerObs() {
        int kEndog = 2, kStates = 2, kPosdef = 1, nobs = 3;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1, 0, 0, 1};
        double[] d = {0, 0};
        double[] H = {1.0, 0, 0, 1.0};
        double[] T = {0.9, 0.1, 0.0, 0.9};
        double[] c = {0, 0};
        double[] R = {1, 0};
        double[] Q = {1.0};
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t * 2], y[t * 2 + 1]}, t);
        }
        FilterResult r = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse());

        double expectedK00 = 1e6 / 1000001.0;
        assertEquals(expectedK00, r.kalmanGain(0, 0, 0), 1e-10);
        assertEquals(0.0, r.kalmanGain(1, 0, 0), 1e-10);
        assertEquals(0.0, r.kalmanGain(0, 1, 0), 1e-10);
        assertEquals(expectedK00, r.kalmanGain(1, 1, 0), 1e-10);

        double convK00 = T[0] * expectedK00 + T[1] * 0.0;
        assertTrue(Math.abs(r.kalmanGain(0, 0, 0) - convK00) > 1e-6,
                "univariate kalmanGain should not include T matrix");
    }

    @Test
    void testUnivariateDiffuseForecast() {
        int kEndog = 1, kStates = 1, kPosdef = 1, nobs = 5;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0};
        double[] d = {0.0};
        double[] H = {1.0};
        double[] T = {0.9};
        double[] c = {0.0};
        double[] R = {1.0};
        double[] Q = {1.0};
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        FilterResult r = UnivariateFilter.filter(rep, StateInitialization.diffuse());

        assertEquals(0.0, r.forecast(0, 0), 1e-10);
        for (int t = 1; t < nobs; t++) {
            assertTrue(Math.abs(r.forecast(0, t)) > 1e-10,
                    "forecast should not be zero at t=" + t);
        }
    }
}
