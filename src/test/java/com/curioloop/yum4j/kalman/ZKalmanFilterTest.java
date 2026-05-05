package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.kalman.filter.*;
import com.curioloop.yum4j.kalman.init.*;
import com.curioloop.yum4j.kalman.model.*;
import com.curioloop.yum4j.kalman.smooth.*;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

public class ZKalmanFilterTest {

    private static final double TOL_REAL = 1e-10;
    private static final double TOL_CMPLX = 1e-10;

    private static long expectedZFilterScratchDoubleCount(ModelFixture rep) {
    return expectedZFilterScratchDoubleCount(rep, FilterSpec.defaults());
    }

    private static long expectedZFilterScratchDoubleCount(ModelFixture rep, FilterSpec spec) {
    long kEndog = rep.kEndog;
    long kStates = rep.kStates;
    long kPosdef = rep.kPosdef;
    long total = 4L * kStates * kEndog
        + 2L * kEndog * kEndog
        + 6L * kStates * kStates
        + 2L * kStates * kPosdef
        + 10L * kStates
        + 2L * kEndog;
    boolean storeForecast = spec.stores(FilterSpec.Storage.FORECAST);
    boolean storePredicted = spec.stores(FilterSpec.Storage.PREDICTED_STATE)
        || (storeForecast && hasMissingObservations(rep));
    boolean storeFiltered = spec.stores(FilterSpec.Storage.FILTERED_STATE);
    if (!storePredicted) {
        total += 4L * kStates + 4L * kStates * kStates;
    }
    if (!storeFiltered) {
        total += 2L * kStates + 2L * kStates * kStates;
    }
    return total;
    }

    private static long expectedZDiffuseFilterScratchDoubleCount(ModelFixture rep, FilterSpec spec) {
    long total = expectedZFilterScratchDoubleCount(rep, spec);
    if (!spec.stores(FilterSpec.Storage.PREDICTED_STATE)) {
        long kStates = rep.kStates;
        total += 4L * kStates * kStates;
    }
    return total;
    }

    private static long expectedZFilterResultDoubleCount(ModelFixture rep) {
    return expectedZFilterResultDoubleCount(rep, FilterSpec.defaults());
    }

    private static long expectedZFilterResultDoubleCount(ModelFixture rep, FilterSpec spec) {
    long kEndog = rep.kEndog;
    long kStates = rep.kStates;
    long nobs = rep.nobs;
    boolean storeForecast = spec.stores(FilterSpec.Storage.FORECAST);
    boolean storePredicted = spec.stores(FilterSpec.Storage.PREDICTED_STATE)
        || (storeForecast && hasMissingObservations(rep));
    boolean storeFiltered = spec.stores(FilterSpec.Storage.FILTERED_STATE);
    boolean storeGain = spec.stores(FilterSpec.Storage.KALMAN_GAIN);
    boolean storeLikelihood = spec.stores(FilterSpec.Storage.LIKELIHOOD);
    long total = 0L;
    if (storePredicted) {
        total += 2L * kStates * (nobs + 1);
        total += 2L * kStates * kStates * (nobs + 1);
    }
    if (storeFiltered) {
        total += 2L * kStates * nobs;
        total += 2L * kStates * kStates * nobs;
    }
    if (storeForecast) {
        total += 2L * kEndog * nobs;
        total += 2L * kEndog * nobs;
        total += 2L * kEndog * kEndog * nobs;
    }
    if (storeGain) {
        total += 2L * kStates * kEndog * nobs;
    }
    if (storeLikelihood) {
        total += nobs;
    }
    return total;
    }

    private static boolean hasMissingObservations(ModelFixture rep) {
    for (int t = 0; t < rep.nobs; t++) {
        if (rep.nmissing[t] > 0) {
            return true;
        }
    }
    return false;
    }

    private static long expectedZFilterDiffuseResultDoubleCount(ModelFixture rep) {
    return expectedZFilterDiffuseResultDoubleCount(rep, FilterSpec.defaults());
    }

    private static long expectedZFilterDiffuseResultDoubleCount(ModelFixture rep, FilterSpec spec) {
    long kEndog = rep.kEndog;
    long kStates = rep.kStates;
    long nobs = rep.nobs;
    long total = 0L;
    if (spec.stores(FilterSpec.Storage.PREDICTED_STATE)) {
        total += 2L * kStates * kStates * (nobs + 1);
    }
    if (spec.stores(FilterSpec.Storage.FORECAST)) {
        total += 2L * kEndog * kEndog * nobs;
    }
    return total;
    }

    private static long expectedZDiffuseRegularFilterResultDoubleCount(ModelFixture rep, FilterSpec spec) {
    long kEndog = rep.kEndog;
    long kStates = rep.kStates;
    long nobs = rep.nobs;
    long total = 0L;
    if (spec.stores(FilterSpec.Storage.PREDICTED_STATE)) {
        total += 2L * kStates * (nobs + 1)
            + 2L * kStates * kStates * (nobs + 1);
    }
    if (spec.stores(FilterSpec.Storage.FILTERED_STATE)) {
        total += 2L * kStates * nobs;
        total += 2L * kStates * kStates * nobs;
    }
    if (spec.stores(FilterSpec.Storage.FORECAST)) {
        total += 2L * kEndog * nobs;
        total += 2L * kEndog * nobs;
        total += 2L * kEndog * kEndog * nobs;
    }
    if (spec.stores(FilterSpec.Storage.KALMAN_GAIN)) {
        total += 2L * kStates * kEndog * nobs;
    }
    if (spec.stores(FilterSpec.Storage.LIKELIHOOD)) {
        total += nobs;
    }
    return total;
    }

    private static long doubleCount(double[] values) {
    return values == null ? 0L : values.length;
    }

    private static long convergedSnapshotDoubleCount(ZKalmanFilter.Pool pool) {
    return doubleCount(pool.convergedForecastErrorCov)
        + doubleCount(pool.convergedFilteredStateCov)
        + doubleCount(pool.convergedPredictedStateCov)
        + doubleCount(pool.convergedKalmanGain);
    }

    private static long expectedConvergedSnapshotDoubleCount(StateSpaceModel rep) {
    return 2L * rep.observationDimension() * rep.observationDimension()
        + 4L * rep.stateCount() * rep.stateCount()
        + 2L * rep.stateCount() * rep.observationDimension();
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
        assertArrayEquals(expected, actual, TOL_REAL);
    }
    }

    private static void assertFilterOutputsMatch(ZFilterResult expected, ZFilterResult actual) {
    assertArrayEquals(activeSlice(expected.predictedState, expected.predictedStateBase(), expected.predictedStateLength()),
        activeSlice(actual.predictedState, actual.predictedStateBase(), actual.predictedStateLength()), TOL_REAL);
    assertArrayEquals(activeSlice(expected.predictedStateCov, expected.predictedStateCovBase(), expected.predictedStateCovLength()),
        activeSlice(actual.predictedStateCov, actual.predictedStateCovBase(), actual.predictedStateCovLength()), TOL_REAL);
    assertArrayEquals(activeSlice(expected.filteredState, expected.filteredStateBase(), expected.filteredStateLength()),
        activeSlice(actual.filteredState, actual.filteredStateBase(), actual.filteredStateLength()), TOL_REAL);
    assertArrayEquals(activeSlice(expected.filteredStateCov, expected.filteredStateCovBase(), expected.filteredStateCovLength()),
        activeSlice(actual.filteredStateCov, actual.filteredStateCovBase(), actual.filteredStateCovLength()), TOL_REAL);
    assertArrayEquals(activeSlice(expected.forecast, expected.forecastBase(), expected.forecastLength()),
        activeSlice(actual.forecast, actual.forecastBase(), actual.forecastLength()), TOL_REAL);
    assertArrayEquals(activeSlice(expected.forecastError, expected.forecastErrorBase(), expected.forecastErrorLength()),
        activeSlice(actual.forecastError, actual.forecastErrorBase(), actual.forecastErrorLength()), TOL_REAL);
    assertArrayEquals(activeSlice(expected.forecastErrorCov, expected.forecastErrorCovBase(), expected.forecastErrorCovLength()),
        activeSlice(actual.forecastErrorCov, actual.forecastErrorCovBase(), actual.forecastErrorCovLength()), TOL_REAL);
    assertArrayEquals(activeSlice(expected.kalmanGain, expected.kalmanGainBase(), expected.kalmanGainLength()),
        activeSlice(actual.kalmanGain, actual.kalmanGainBase(), actual.kalmanGainLength()), TOL_REAL);
    assertArrayEquals(activeSlice(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
        activeSlice(actual.logLikelihoodObs, actual.logLikelihoodObsBase(), actual.logLikelihoodObsLength()), TOL_REAL);
    assertNullableArrayEquals(
        nullableActiveSlice(expected.predictedDiffuseStateCov, expected.predictedDiffuseStateCovBase(), expected.predictedDiffuseStateCovLength()),
        nullableActiveSlice(actual.predictedDiffuseStateCov, actual.predictedDiffuseStateCovBase(), actual.predictedDiffuseStateCovLength()));
    assertNullableArrayEquals(
        nullableActiveSlice(expected.forecastErrorDiffuseCov, expected.forecastErrorDiffuseCovBase(), expected.forecastErrorDiffuseCovLength()),
        nullableActiveSlice(actual.forecastErrorDiffuseCov, actual.forecastErrorDiffuseCovBase(), actual.forecastErrorDiffuseCovLength()));
    }

    static ModelFixture buildAR1(double phi, double sigma, double[] y) {
        return KalmanModelFixtures.signalOnlyAr1(phi, sigma, y);
    }

    static ModelFixture buildZAR1(double phiRe, double phiIm, double sigma, double[] y) {
        return KalmanModelFixtures.complexSignalOnlyAr1(phiRe, phiIm, sigma, y);
    }

    private static StateSpaceModel buildComplexApproximateDiffusePoolModel() {
        return StateSpaceModel.complexBuilder(1, 2, 2, 4)
            .design(new double[]{1.0, 0.0, 0.0, 0.0}, false)
            .obsIntercept(new double[]{0.0, 0.0}, false)
            .obsCov(new double[]{1.0, 0.0}, false)
            .transition(new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, false)
            .stateIntercept(new double[]{0.0, 0.0, 0.0, 0.0}, false)
            .selection(new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, false)
            .stateCovariance(new double[]{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0}, false)
            .endog(new double[]{0.25, 0.0, -0.75, 0.0, 0.5, 0.0, -0.25, 0.0})
            .allObserved()
            .build();
    }

    private static StateSpaceModel buildComplexTimeVaryingSteadyStateTrapModel() {
        return StateSpaceModel.complexBuilder(1, 1, 1, 6)
            .design(new double[]{1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0}, true)
            .obsIntercept(new double[]{0.0, 0.0}, false)
            .obsCov(new double[]{1.0, 0.0}, false)
            .transition(new double[]{0.0, 0.0}, false)
            .stateIntercept(new double[]{0.0, 0.0}, false)
            .selection(new double[]{1.0, 0.0}, false)
            .stateCovariance(new double[]{0.5, 0.0}, false)
            .endog(new double[]{0.25, 0.0, -0.75, 0.0, 0.5, 0.0, -0.25, 0.0, 0.75, 0.0, -0.5, 0.0})
            .allObserved()
            .build();
    }

    private static StateSpaceModel buildComplexMissingObservationConvergingModel() {
        StateSpaceModel rep = StateSpaceModel.complexBuilder(1, 1, 1, 12)
            .design(new double[]{1.0, 0.0}, false)
            .obsIntercept(new double[]{0.0, 0.0}, false)
            .obsCov(new double[]{1.0, 0.0}, false)
            .transition(new double[]{0.0, 0.0}, false)
            .stateIntercept(new double[]{0.0, 0.0}, false)
            .selection(new double[]{1.0, 0.0}, false)
            .stateCovariance(new double[]{0.5, 0.0}, false)
            .endog(new double[]{
                0.25, 0.0, -0.75, 0.0, 0.25, 0.0, -0.75, 0.0,
                0.25, 0.0, -0.75, 0.0, 0.25, 0.0, -0.75, 0.0,
                0.25, 0.0, -0.75, 0.0, 0.25, 0.0, -0.75, 0.0})
            .allObserved()
            .build();
        rep.setMissing(new boolean[]{true}, 1);
        return rep;
    }

    private static StateSpaceModel buildComplexConstantVarianceConvergingModel(int nobs) {
        return buildComplexConstantVarianceConvergingModel(1, 1, nobs);
    }

    private static StateSpaceModel buildComplexConstantVarianceConvergingModel(int kEndog, int kStates, int nobs) {
        int kPosdef = kStates;
        double[] Z = new double[2 * kEndog * kStates];
        for (int i = 0; i < Math.min(kEndog, kStates); i++) {
            Z[(i * kStates + i) * 2] = 1.0;
        }
        double[] d = new double[2 * kEndog];
        double[] H = new double[2 * kEndog * kEndog];
        for (int i = 0; i < kEndog; i++) {
            H[(i * kEndog + i) * 2] = 1.0;
        }
        double[] T = new double[2 * kStates * kStates];
        double[] c = new double[2 * kStates];
        double[] R = new double[2 * kStates * kStates];
        double[] Q = new double[2 * kStates * kStates];
        for (int i = 0; i < kStates; i++) {
            R[(i * kStates + i) * 2] = 1.0;
            Q[(i * kStates + i) * 2] = 0.5;
        }
        double[] y = new double[2 * kEndog * nobs];
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                y[(t * kEndog + i) * 2] = ((t & 1) == 0 ? 0.25 : -0.75) * (i + 1);
            }
        }
        return StateSpaceModel.complexBuilder(kEndog, kStates, kPosdef, nobs)
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

    private static StateSpaceModel buildRealMissingObservationConvergingModel() {
        StateSpaceModel rep = StateSpaceModel.builder(1, 1, 1, 12)
            .design(new double[]{1.0}, false)
            .obsIntercept(new double[]{0.0}, false)
            .obsCov(new double[]{1.0}, false)
            .transition(new double[]{0.0}, false)
            .stateIntercept(new double[]{0.0}, false)
            .selection(new double[]{1.0}, false)
            .stateCovariance(new double[]{0.5}, false)
            .endog(new double[]{0.25, -0.75, 0.25, -0.75, 0.25, -0.75, 0.25, -0.75, 0.25, -0.75, 0.25, -0.75})
            .allObserved()
            .build();
        rep.setMissing(new boolean[]{true}, 1);
        return rep;
    }

    static ModelFixture buildExactDiffuseLocalLevel(double[] y, double sigma2Y, double sigma2Mu) {
        ModelFixture rep = new ModelFixture(1, 1, 1, y.length);
        double[] Z = {1.0};
        double[] H = {sigma2Y};
        double[] T = {1.0};
        double[] c = {0.0};
        double[] R = {1.0};
        double[] Q = {sigma2Mu};
        for (int t = 0; t < y.length; t++) {
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

    static ModelFixture buildZExactDiffuseLocalLevel(double[] y, double sigma2Y, double sigma2Mu) {
        ModelFixture rep = ModelFixture.complex(1, 1, 1, y.length);
        double[] Z = {1.0, 0.0};
        double[] H = {sigma2Y, 0.0};
        double[] T = {1.0, 0.0};
        double[] c = {0.0, 0.0};
        double[] R = {1.0, 0.0};
        double[] Q = {sigma2Mu, 0.0};
        for (int t = 0; t < y.length; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(new double[]{0.0, 0.0}, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t], 0.0}, t);
        }
        return rep;
    }

    static ModelFixture buildExactDiffuseLocalLinearTrend(double[] y, double sigma2Y,
                                                            double sigma2Mu, double sigma2Beta) {
        ModelFixture rep = new ModelFixture(1, 2, 2, y.length);
        double[] Z = {1.0, 0.0};
        double[] H = {sigma2Y};
        double[] T = {1.0, 1.0, 0.0, 1.0};
        double[] c = {0.0, 0.0};
        double[] R = {1.0, 0.0, 0.0, 1.0};
        double[] Q = {sigma2Mu, 0.0, 0.0, sigma2Beta};
        for (int t = 0; t < y.length; t++) {
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

    static ModelFixture buildZExactDiffuseLocalLinearTrend(double[] y, double sigma2Y,
                                                              double sigma2Mu, double sigma2Beta) {
        ModelFixture rep = ModelFixture.complex(1, 2, 2, y.length);
        double[] Z = {1.0, 0.0, 0.0, 0.0};
        double[] H = {sigma2Y, 0.0};
        double[] T = {1.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0, 0.0};
        double[] c = {0.0, 0.0, 0.0, 0.0};
        double[] R = {1.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0};
        double[] Q = {sigma2Mu, 0.0, 0.0, 0.0,
                      0.0, 0.0, sigma2Beta, 0.0};
        for (int t = 0; t < y.length; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(new double[]{0.0, 0.0}, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t], 0.0}, t);
        }
        return rep;
    }

    @Test
    void testRealDegradationAR1() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7, -0.5, 0.3};

        ModelFixture realRep = buildAR1(phi, sigma, y);
        FilterResult realResult = KalmanFilter.filter(realRep, StateInitialization.approximateDiffuse());

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        ZFilterResult zResult = ZKalmanFilter.filter(zRep, StateInitialization.approximateDiffuse());

        int kStates = 1;
        int nobs = y.length;

        for (int t = 0; t < nobs; t++) {
            int zPredOff = t * 2 * kStates;
            int zFiltOff = t * 2 * kStates;
            int zForeOff = t * 2;
            int zForeErrOff = t * 2;

            assertEquals(realResult.predictedState[t * kStates],
                    zResult.predictedState[zPredOff], TOL_REAL,
                    "predictedState Re at t=" + t);
            assertEquals(0.0, zResult.predictedState[zPredOff + 1], TOL_REAL,
                    "predictedState Im at t=" + t);

            assertEquals(realResult.filteredState[t * kStates],
                    zResult.filteredState[zFiltOff], TOL_REAL,
                    "filteredState Re at t=" + t);
            assertEquals(0.0, zResult.filteredState[zFiltOff + 1], TOL_REAL,
                    "filteredState Im at t=" + t);

            assertEquals(realResult.forecast[t],
                    zResult.forecast[zForeOff], TOL_REAL,
                    "forecast Re at t=" + t);
            assertEquals(0.0, zResult.forecast[zForeOff + 1], TOL_REAL,
                    "forecast Im at t=" + t);

            assertEquals(realResult.forecastError[t],
                    zResult.forecastError[zForeErrOff], TOL_REAL,
                    "forecastError Re at t=" + t);
            assertEquals(0.0, zResult.forecastError[zForeErrOff + 1], TOL_REAL,
                    "forecastError Im at t=" + t);

            assertEquals(realResult.logLikelihoodObs[t],
                    zResult.logLikelihoodObs[t], TOL_REAL,
                    "logLikelihoodObs at t=" + t);
        }

        for (int t = 0; t <= nobs; t++) {
            int realCovOff = t * kStates * kStates;
            int zCovOff = t * kStates * 2 * kStates;
            assertEquals(realResult.predictedStateCov[realCovOff],
                    zResult.predictedStateCov[zCovOff], TOL_REAL,
                    "predictedStateCov Re at t=" + t);
            assertEquals(0.0, zResult.predictedStateCov[zCovOff + 1], TOL_REAL,
                    "predictedStateCov Im at t=" + t);
        }

        for (int t = 0; t < nobs; t++) {
            int realCovOff = t * kStates * kStates;
            int zCovOff = t * kStates * 2 * kStates;
            assertEquals(realResult.filteredStateCov[realCovOff],
                    zResult.filteredStateCov[zCovOff], TOL_REAL,
                    "filteredStateCov Re at t=" + t);
            assertEquals(0.0, zResult.filteredStateCov[zCovOff + 1], TOL_REAL,
                    "filteredStateCov Im at t=" + t);
        }
    }

    @Test
    void testComplexAR1Filter() {
        double phiRe = 0.5, phiIm = 0.3, sigma = 1.0;
        int nobs = 10;
        int kStates = 1;

        double[] yRe = new double[nobs];
        double[] yIm = new double[nobs];
        double[] eps = {1.0, 0.5, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, 0.1, -0.4};
        yRe[0] = eps[0];
        yIm[0] = 0.0;
        for (int t = 1; t < nobs; t++) {
            yRe[t] = phiRe * yRe[t - 1] - phiIm * yIm[t - 1] + eps[t];
            yIm[t] = phiRe * yIm[t - 1] + phiIm * yRe[t - 1];
        }

        ModelFixture rep = ModelFixture.complex(1, 1, 1, nobs);
        double[] Z = {1.0, 0.0};
        double[] d = {0.0, 0.0};
        double[] H = {0.0, 0.0};
        double[] T = {phiRe, phiIm};
        double[] c = {0.0, 0.0};
        double[] R = {1.0, 0.0};
        double[] Q = {sigma * sigma, 0.0};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{yRe[t], yIm[t]}, t);
        }

        ZFilterResult result = ZKalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        boolean hasNonZeroImagPred = false;
        boolean hasNonZeroImagFilt = false;
        for (int t = 0; t < nobs; t++) {
            int predOff = t * 2 * kStates;
            int filtOff = t * 2 * kStates;
            if (Math.abs(result.predictedState[predOff + 1]) > 1e-12) hasNonZeroImagPred = true;
            if (Math.abs(result.filteredState[filtOff + 1]) > 1e-12) hasNonZeroImagFilt = true;
        }
        assertTrue(hasNonZeroImagPred, "predictedState should have non-zero imaginary parts");
        assertTrue(hasNonZeroImagFilt, "filteredState should have non-zero imaginary parts");

        int ks2c = kStates * 2 * kStates;
        for (int t = 0; t <= nobs; t++) {
            int covOff = t * ks2c;
            assertHermitian(result.predictedStateCov, covOff, kStates,
                    "predictedStateCov at t=" + t);
        }
        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
            assertHermitian(result.filteredStateCov, covOff, kStates,
                    "filteredStateCov at t=" + t);
        }

        for (int t = 0; t < nobs; t++) {
            assertFalse(Double.isNaN(result.logLikelihoodObs[t]),
                    "logLikelihoodObs should be real (not NaN) at t=" + t);
            assertFalse(Double.isInfinite(result.logLikelihoodObs[t]),
                    "logLikelihoodObs should be finite at t=" + t);
        }
    }

    @Test
    void testHermitianSymmetry() {
        int kEndog = 1, kStates = 2, kPosdef = 2, nobs = 5;

        ModelFixture rep = ModelFixture.complex(kEndog, kStates, kPosdef, nobs);

        double[] Z = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};
        double[] d = {0.0, 0.0};
        double[] H = {0.0, 0.0};
        double[] T = {0.5, 0.1, 0.0, 0.0,
                       0.2, -0.05, 0.3, 0.1,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0};
        double[] c = {0.0, 0.0, 0.0, 0.0};
        double[] R = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};
        double[] Q = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};

        double[] y = {1.0, 0.0, 0.5, 0.0, -0.3, 0.0, 0.8, 0.0, 0.2, 0.0};

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

        ZFilterResult result = ZKalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        int ks2c = kStates * 2 * kStates;
        for (int t = 0; t <= nobs; t++) {
            int covOff = t * ks2c;
            assertHermitian(result.predictedStateCov, covOff, kStates,
                    "predictedStateCov at t=" + t);
        }
        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
            assertHermitian(result.filteredStateCov, covOff, kStates,
                    "filteredStateCov at t=" + t);
        }
    }

    @Test
    void testLogLikelihood() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7, -0.5, 0.3};

        ModelFixture realRep = buildAR1(phi, sigma, y);
        FilterResult realResult = KalmanFilter.filter(realRep, StateInitialization.approximateDiffuse());

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        ZFilterResult zResult = ZKalmanFilter.filter(zRep, StateInitialization.approximateDiffuse());

        assertEquals(realResult.logLikelihood(), zResult.logLikelihood(), TOL_REAL);
    }

    @Test
    void testDifferentInitializations() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7, -0.5, 0.3};
        int kStates = 1;

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);

        double[] initState = {0.0, 0.0};
        double[] initStateCov = {1.0, 0.0};
        ZFilterResult knownResult = ZKalmanFilter.filter(zRep, StateInitialization.known(initState, initStateCov));
        assertNotNull(knownResult);

        ZFilterResult diffuseResult = ZKalmanFilter.filter(zRep, StateInitialization.approximateDiffuse());
        assertNotNull(diffuseResult);

        StateInitialization stationaryInit = StateInitialization.stationary(zRep);
        assertNotNull(stationaryInit);

        ZFilterResult stationaryResult = ZKalmanFilter.filter(zRep, stationaryInit);
        assertNotNull(stationaryResult);

        double expectedP0 = sigma * sigma / (1 - phi * phi);
        assertEquals(expectedP0, stationaryInit.initialStateCov()[0], TOL_REAL,
                "Stationary initial covariance");
        assertEquals(0.0, stationaryInit.initialStateCov()[1], TOL_REAL,
                "Stationary initial covariance Im");

        ModelFixture realRep = buildAR1(phi, sigma, y);
        StateInitialization realStationaryInit = StateInitialization.stationary(realRep);
        FilterResult realStationaryResult = KalmanFilter.filter(realRep, realStationaryInit);

        for (int t = 0; t < y.length; t++) {
            int zPredOff = t * 2 * kStates;
            assertEquals(realStationaryResult.predictedState[t * kStates],
                    stationaryResult.predictedState[zPredOff], TOL_REAL,
                    "stationary predictedState Re at t=" + t);
            assertEquals(0.0, stationaryResult.predictedState[zPredOff + 1], TOL_REAL,
                    "stationary predictedState Im at t=" + t);
        }
    }

    @Test
    void testStationaryInitializationUsesMean() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        for (int t = 0; t < y.length; t++) {
            zRep.setStateIntercept(new double[]{1.0, 0.0}, t);
        }

        StateInitialization stationaryInit = StateInitialization.stationary(zRep);
        StateInitialization knownInit = StateInitialization.known(
                stationaryInit.initialState().clone(),
                stationaryInit.initialStateCov().clone());

        ZFilterResult stationaryResult = ZKalmanFilter.filter(zRep, stationaryInit);
        ZFilterResult knownResult = ZKalmanFilter.filter(zRep, knownInit);

        assertEquals(2.0, stationaryInit.initialState()[0], TOL_REAL);
        assertEquals(0.0, stationaryInit.initialState()[1], TOL_REAL);
        assertEquals(2.0, stationaryResult.predictedState[0], TOL_REAL);
        assertEquals(0.0, stationaryResult.predictedState[1], TOL_REAL);

        for (int t = 0; t <= y.length; t++) {
            int predOff = stationaryResult.predictedStateOffset(t);
            int covOff = stationaryResult.predictedStateCovOffset(t);
            assertEquals(knownResult.predictedState[predOff], stationaryResult.predictedState[predOff], TOL_REAL);
            assertEquals(knownResult.predictedState[predOff + 1], stationaryResult.predictedState[predOff + 1], TOL_REAL);
            assertEquals(knownResult.predictedStateCov[covOff], stationaryResult.predictedStateCov[covOff], TOL_REAL);
            assertEquals(knownResult.predictedStateCov[covOff + 1], stationaryResult.predictedStateCov[covOff + 1], TOL_REAL);
        }
        for (int t = 0; t < y.length; t++) {
            int filtOff = stationaryResult.filteredStateOffset(t);
            assertEquals(knownResult.filteredState[filtOff], stationaryResult.filteredState[filtOff], TOL_REAL);
            assertEquals(knownResult.filteredState[filtOff + 1], stationaryResult.filteredState[filtOff + 1], TOL_REAL);
        }
    }

        @Test
        void testExactDiffuseLocalLevelRealDegradation() {
        double[] y = {10.2394, 1.0, 1.0, 1.0, 1.0, 1.0};
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;

        ModelFixture realRep = buildExactDiffuseLocalLevel(y, sigma2Y, sigma2Mu);
        FilterResult realResult = KalmanFilter.filter(realRep, StateInitialization.diffuse());

        ModelFixture zRep = buildZExactDiffuseLocalLevel(y, sigma2Y, sigma2Mu);
        ZFilterResult zResult = ZKalmanFilter.filter(zRep, StateInitialization.diffuse());

        assertEquals(realResult.nobsDiffuse, zResult.nobsDiffuse);
        assertTrue(zResult.perObsKalmanGain);
        assertNotNull(zResult.predictedDiffuseStateCov);
        assertNotNull(zResult.forecastErrorDiffuseCov);

        for (int t = 0; t <= y.length; t++) {
            int realCovOff = t;
            int zCovOff = t * 2;
            assertEquals(realResult.predictedStateCov[realCovOff], zResult.predictedStateCov[zCovOff], TOL_REAL,
                "predictedStateCov Re at t=" + t);
            assertEquals(0.0, zResult.predictedStateCov[zCovOff + 1], TOL_REAL,
                "predictedStateCov Im at t=" + t);
            assertEquals(realResult.predictedDiffuseStateCov[realCovOff], zResult.predictedDiffuseStateCov[zCovOff], TOL_REAL,
                "predictedDiffuseStateCov Re at t=" + t);
            assertEquals(0.0, zResult.predictedDiffuseStateCov[zCovOff + 1], TOL_REAL,
                "predictedDiffuseStateCov Im at t=" + t);
        }

        for (int t = 0; t < y.length; t++) {
            int zOff = t * 2;
            assertEquals(realResult.predictedState[t], zResult.predictedState[zOff], TOL_REAL,
                "predictedState Re at t=" + t);
            assertEquals(0.0, zResult.predictedState[zOff + 1], TOL_REAL,
                "predictedState Im at t=" + t);
            assertEquals(realResult.filteredState[t], zResult.filteredState[zOff], TOL_REAL,
                "filteredState Re at t=" + t);
            assertEquals(0.0, zResult.filteredState[zOff + 1], TOL_REAL,
                "filteredState Im at t=" + t);
            assertEquals(realResult.forecast[t], zResult.forecast[zOff], TOL_REAL,
                "forecast Re at t=" + t);
            assertEquals(realResult.forecastError[t], zResult.forecastError[zOff], TOL_REAL,
                "forecastError Re at t=" + t);
            assertEquals(realResult.forecastErrorCov[t], zResult.forecastErrorCov[zOff], TOL_REAL,
                "forecastErrorCov Re at t=" + t);
            assertEquals(realResult.forecastErrorDiffuseCov[t], zResult.forecastErrorDiffuseCov[zOff], TOL_REAL,
                "forecastErrorDiffuseCov Re at t=" + t);
            assertEquals(realResult.logLikelihoodObs[t], zResult.logLikelihoodObs[t], TOL_REAL,
                "logLikelihoodObs at t=" + t);
        }
        }

        @Test
        void testExactDiffuseLocalLinearTrendMissingRealDegradation() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;
        double sigma2Beta = 2.334;

        ModelFixture realRep = buildExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        realRep.setMissing(new boolean[]{true}, 1);
        FilterResult realResult = KalmanFilter.filter(realRep, StateInitialization.diffuse());

        ModelFixture zRep = buildZExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        zRep.setMissing(new boolean[]{true}, 1);
        ZFilterResult zResult = ZKalmanFilter.filter(zRep, StateInitialization.diffuse());

        assertEquals(realResult.nobsDiffuse, zResult.nobsDiffuse);
        assertTrue(zResult.perObsKalmanGain);

        int kStates = 2;
        for (int t = 0; t <= y.length; t++) {
            int realCovOff = t * kStates * kStates;
            int zCovOff = t * kStates * 2 * kStates;
            for (int i = 0; i < kStates; i++) {
            for (int j = 0; j < kStates; j++) {
                int rIdx = realCovOff + i * kStates + j;
                int zIdx = zCovOff + i * 2 * kStates + j * 2;
                assertEquals(realResult.predictedStateCov[rIdx], zResult.predictedStateCov[zIdx], TOL_REAL,
                    "predictedStateCov Re at t=" + t + " [" + i + "," + j + "]");
                assertEquals(realResult.predictedDiffuseStateCov[rIdx], zResult.predictedDiffuseStateCov[zIdx], TOL_REAL,
                    "predictedDiffuseStateCov Re at t=" + t + " [" + i + "," + j + "]");
                assertEquals(0.0, zResult.predictedStateCov[zIdx + 1], TOL_REAL,
                    "predictedStateCov Im at t=" + t + " [" + i + "," + j + "]");
                assertEquals(0.0, zResult.predictedDiffuseStateCov[zIdx + 1], TOL_REAL,
                    "predictedDiffuseStateCov Im at t=" + t + " [" + i + "," + j + "]");
            }
            }
        }
        }

    @Test
    void testFilterSpecSuppressesStoredOutputs() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8};

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        FilterSpec spec = FilterSpec.defaults()
                .without(FilterSpec.Storage.FORECAST,
                        FilterSpec.Storage.PREDICTED_STATE,
                        FilterSpec.Storage.LIKELIHOOD,
                        FilterSpec.Storage.KALMAN_GAIN);

        ZFilterResult result = ZKalmanFilter.filter(zRep, StateInitialization.approximateDiffuse(), null, spec);

        assertEquals(0, result.predictedState.length);
        assertEquals(0, result.predictedStateCov.length);
        assertEquals(0, result.forecast.length);
        assertEquals(0, result.forecastError.length);
        assertEquals(0, result.forecastErrorCov.length);
        assertEquals(0, result.kalmanGain.length);
        assertEquals(0, result.logLikelihoodObs.length);
        assertNull(result.predictedDiffuseStateCov);
        assertNull(result.forecastErrorDiffuseCov);
        assertEquals(0, result.nobsDiffuse);
        assertFalse(result.perObsKalmanGain);
    }

    @Test
    void testMultivariateComplex() {
        int kEndog = 2, kStates = 2, kPosdef = 2, nobs = 5;

        ModelFixture rep = ModelFixture.complex(kEndog, kStates, kPosdef, nobs);

        double[] Z = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};
        double[] d = {0.0, 0.0, 0.0, 0.0};
        double[] H = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};
        double[] T = {0.5, 0.1, 0.0, 0.0,
                       0.2, -0.05, 0.3, 0.1,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0};
        double[] c = {0.0, 0.0, 0.0, 0.0};
        double[] R = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};
        double[] Q = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{1.0, 0.0, 0.5, 0.0}, t);
        }

        ZFilterResult result = ZKalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        assertNotNull(result);

        int ks2c = kStates * 2 * kStates;
        for (int t = 0; t <= nobs; t++) {
            int covOff = t * ks2c;
            assertHermitian(result.predictedStateCov, covOff, kStates,
                    "predictedStateCov at t=" + t);
        }
        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
            assertHermitian(result.filteredStateCov, covOff, kStates,
                    "filteredStateCov at t=" + t);
        }

        for (int t = 0; t < nobs; t++) {
            assertFalse(Double.isNaN(result.logLikelihoodObs[t]),
                    "logLikelihoodObs NaN at t=" + t);
            assertFalse(Double.isInfinite(result.logLikelihoodObs[t]),
                    "logLikelihoodObs Infinite at t=" + t);
        }
    }

    @Test
    void testPoolReuse() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7, -0.5, 0.3};

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        ZFilterResult expected = ZKalmanFilter.filter(zRep, StateInitialization.approximateDiffuse());

        ZFilterResult r1 = ZKalmanFilter.filter(zRep, StateInitialization.approximateDiffuse(), pool);
        ZFilterResult r2 = ZKalmanFilter.filter(zRep, StateInitialization.approximateDiffuse(), pool);

        assertSame(r1, r2);

        assertFilterOutputsMatch(expected, r1);
    }

    @Test
    void testApproximateDiffusePoolReuseResetsInitialComplexStateAndCovariance() {
        StateSpaceModel rep = buildComplexApproximateDiffusePoolModel();
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();

        ZKalmanFilter.filter(rep,
            StateInitialization.known(
                new double[]{2.0, 0.0, -3.0, 0.0},
                new double[]{4.0, 0.0, 1.5, 0.0, 1.5, 0.0, 5.0, 0.0}),
            pool);
        ZFilterResult reused = ZKalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool);

        int stateBase = reused.predictedStateBase();
        int covBase = reused.predictedStateCovBase();
        assertEquals(0.0, reused.predictedState[stateBase], TOL_REAL);
        assertEquals(0.0, reused.predictedState[stateBase + 1], TOL_REAL);
        assertEquals(0.0, reused.predictedState[stateBase + 2], TOL_REAL);
        assertEquals(0.0, reused.predictedState[stateBase + 3], TOL_REAL);
        assertEquals(1e6, reused.predictedStateCov[covBase], TOL_REAL);
        assertEquals(0.0, reused.predictedStateCov[covBase + 1], TOL_REAL);
        assertEquals(0.0, reused.predictedStateCov[covBase + 2], TOL_REAL);
        assertEquals(0.0, reused.predictedStateCov[covBase + 3], TOL_REAL);
        assertEquals(0.0, reused.predictedStateCov[covBase + 4], TOL_REAL);
        assertEquals(0.0, reused.predictedStateCov[covBase + 5], TOL_REAL);
        assertEquals(1e6, reused.predictedStateCov[covBase + 6], TOL_REAL);
        assertEquals(0.0, reused.predictedStateCov[covBase + 7], TOL_REAL);
    }

    @Test
    void testTimeVaryingComplexModelDoesNotReuseSteadyState() {
        StateSpaceModel realRep = StateSpaceModel.builder(1, 1, 1, 6)
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
        StateSpaceModel zRep = buildComplexTimeVaryingSteadyStateTrapModel();

        FilterResult reference = UnivariateFilter.filter(realRep, StateInitialization.known(new double[]{0.0}, new double[]{0.5}));
        ZFilterResult result = ZKalmanFilter.filter(zRep, StateInitialization.known(new double[]{0.0, 0.0}, new double[]{0.5, 0.0}));

        assertFalse(zRep.isTimeInvariant());
        assertFalse(result.converged);
        assertEquals(zRep.observationCount(), result.periodConverged);
        for (int t = 0; t < zRep.observationCount(); t++) {
            int filtOff = result.filteredStateOffset(t);
            int covOff = result.forecastErrorCovOffset(t);
            assertEquals(reference.filteredState(0, t), result.filteredState[filtOff], TOL_REAL);
            assertEquals(0.0, result.filteredState[filtOff + 1], TOL_REAL);
            if (zRep.missingCount(t) == 0) {
                assertEquals(reference.forecastErrorCov(0, 0, t), result.forecastErrorCov[covOff], TOL_REAL);
            }
            assertEquals(0.0, result.forecastErrorCov[covOff + 1], TOL_REAL);
        }
    }

    @Test
    void testMissingObservationDefersComplexSteadyStateReuse() {
        StateSpaceModel realRep = buildRealMissingObservationConvergingModel();
        StateSpaceModel zRep = buildComplexMissingObservationConvergingModel();

        FilterResult reference = UnivariateFilter.filter(realRep,
            StateInitialization.known(new double[]{0.0}, new double[]{0.5}));
        ZFilterResult result = ZKalmanFilter.filter(zRep,
            StateInitialization.known(new double[]{0.0, 0.0}, new double[]{0.5, 0.0}));

        assertTrue(zRep.isTimeInvariant());
        assertEquals(1, zRep.missingCount(1));
        assertTrue(result.converged);
        assertTrue(result.periodConverged > 3,
            "steady-state reuse should not resume immediately after a missing period");
        for (int t = 0; t < zRep.observationCount(); t++) {
            int filtOff = result.filteredStateOffset(t);
            int covOff = result.forecastErrorCovOffset(t);
            assertEquals(reference.filteredState(0, t), result.filteredState[filtOff], TOL_REAL);
            assertEquals(0.0, result.filteredState[filtOff + 1], TOL_REAL);
            if (zRep.missingCount(t) == 0) {
                assertEquals(reference.forecastErrorCov(0, 0, t), result.forecastErrorCov[covOff], TOL_REAL);
            }
            assertEquals(0.0, result.forecastErrorCov[covOff + 1], TOL_REAL);
        }
    }

    @Test
    void testPoolSupportsExactDiffuseInitialization() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        ZFilterResult expected = ZKalmanFilter.filter(rep, StateInitialization.diffuse());

        ZFilterResult borrowed1 = ZKalmanFilter.filter(rep, StateInitialization.diffuse(), pool);
        ZFilterResult borrowed2 = ZKalmanFilter.filter(rep, StateInitialization.diffuse(), pool);

        assertSame(borrowed1, borrowed2);
        assertTrue(borrowed1.nobsDiffuse > 0);
        assertFilterOutputsMatch(expected, borrowed1);
    }

    @Test
    void testPoolSupportsExactDiffuseLeanSimulationSpec() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        FilterSpec spec = FilterSpec.conventionalSmoothing();
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        ZFilterResult expected = ZKalmanFilter.filter(rep, StateInitialization.diffuse(), null, spec);

        ZFilterResult borrowed1 = ZKalmanFilter.filter(rep, StateInitialization.diffuse(), pool, spec);
        ZFilterResult borrowed2 = ZKalmanFilter.filter(rep, StateInitialization.diffuse(), pool, spec);

        assertSame(borrowed1, borrowed2);
        assertTrue(borrowed1.nobsDiffuse > 0);
        assertFilterOutputsMatch(expected, borrowed1);
    }

    @Test
    void testPoolReserveQuantifiesRetainedComplexFilterMemory() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long expectedScratchDoubles = expectedZFilterScratchDoubleCount(rep);
        long expectedResultDoubles = expectedZFilterResultDoubleCount(rep);

        pool.reserve(rep, FilterSpec.defaults());

        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedResultDoubles, pool.retainedResultDoubleCount());
        assertEquals(expectedResultDoubles, (long) pool.resultBacking.length);
        assertNull(pool.diffuseResultBacking);
        assertEquals((expectedScratchDoubles + expectedResultDoubles) * Double.BYTES,
            pool.retainedTotalByteCount());
    }

    @Test
    void testComplexStationaryInitRetainsDedicatedWorkspaceAlongsideScratchBacking() {
        ModelFixture rep = buildZAR1(0.5, 0.0, 1.0, new double[]{1.0, 0.5, -0.3, 0.2});
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();

        ZFilterResult result = ZKalmanFilter.filter(rep, StateInitialization.stationary(rep.stateCount()), pool);

        long expectedWorkspaceDoubles = StateInitialization.requiredStationaryBackingLength(rep)
            + StateInitialization.requiredStationaryPivotLength(rep) * ((long) Integer.BYTES) / Double.BYTES;
        long expectedSnapshotDoubles = expectedConvergedSnapshotDoubleCount(rep);
        assertTrue(result.converged);
        assertEquals(expectedZFilterScratchDoubleCount(rep) + expectedWorkspaceDoubles + expectedSnapshotDoubles,
            pool.retainedScratchDoubleCount());
        assertEquals(expectedZFilterScratchDoubleCount(rep), (long) pool.scratchBacking.length);
        assertEquals(expectedSnapshotDoubles, convergedSnapshotDoubleCount(pool));
    }

    @Test
    void testDiffusePoolQuantifiesAdditionalComplexFilterBanks() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long expectedScratchDoubles = expectedZFilterScratchDoubleCount(rep);
        long expectedResultDoubles = expectedZFilterResultDoubleCount(rep);
        long expectedDiffuseResultDoubles = expectedZFilterDiffuseResultDoubleCount(rep);

        pool.reserve(rep, FilterSpec.defaults());

        ZFilterResult result = ZKalmanFilter.filter(rep, StateInitialization.diffuse(), pool);

        assertTrue(result.nobsDiffuse > 0);
        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedResultDoubles + expectedDiffuseResultDoubles,
            pool.retainedResultDoubleCount());
        assertEquals(expectedDiffuseResultDoubles, (long) pool.diffuseResultBacking.length);
        assertEquals((expectedScratchDoubles + expectedResultDoubles + expectedDiffuseResultDoubles) * Double.BYTES,
            pool.retainedTotalByteCount());
    }

    @Test
    void testReserveTrimsComplexFilterOptionalRetainedBanksBackToBaseline() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long expectedScratchDoubles = expectedZFilterScratchDoubleCount(rep);
        long expectedResultDoubles = expectedZFilterResultDoubleCount(rep);

        ZKalmanFilter.filter(rep, StateInitialization.diffuse(), pool);

        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertTrue(pool.retainedResultDoubleCount() > expectedResultDoubles);
        assertNotNull(pool.diffuseResultBacking);

        pool.reserve(rep, FilterSpec.defaults());

        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedResultDoubles, pool.retainedResultDoubleCount());
        assertNull(pool.diffuseResultBacking);
        assertNotNull(pool.resultBacking);
        assertEquals(expectedResultDoubles, (long) pool.resultBacking.length);
    }

    @Test
    void testComplexPoolDefersConvergedSnapshotAllocationUntilNeeded() {
        StateSpaceModel rep = buildComplexConstantVarianceConvergingModel(1);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long expectedSnapshotDoubles = expectedConvergedSnapshotDoubleCount(rep);

        pool.reserve(rep, FilterSpec.defaults());
        long baselineScratchDoubles = pool.retainedScratchDoubleCount();

        assertNull(pool.convergedForecastErrorCov);
        assertNull(pool.convergedFilteredStateCov);
        assertNull(pool.convergedPredictedStateCov);
        assertNull(pool.convergedKalmanGain);
        assertEquals(0L, convergedSnapshotDoubleCount(pool));
        assertEquals(8L, expectedSnapshotDoubles);

        ZKalmanFilter.filter(rep, StateInitialization.known(new double[]{0.0, 0.0}, new double[]{0.5, 0.0}), pool);

        assertNull(pool.convergedForecastErrorCov);
        assertNull(pool.convergedFilteredStateCov);
        assertNull(pool.convergedPredictedStateCov);
        assertNull(pool.convergedKalmanGain);
        assertEquals(0L, convergedSnapshotDoubleCount(pool));
        assertEquals(baselineScratchDoubles, pool.retainedScratchDoubleCount());
    }

    @Test
    void testComplexPoolAllocatesConvergedSnapshotsWhenFilterConverges() {
        StateSpaceModel rep = buildComplexConstantVarianceConvergingModel(2, 2, 6);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long expectedSnapshotDoubles = expectedConvergedSnapshotDoubleCount(rep);

        pool.reserve(rep, FilterSpec.defaults());
        long baselineScratchDoubles = pool.retainedScratchDoubleCount();

        ZFilterResult result = ZKalmanFilter.filter(rep,
            StateInitialization.known(new double[2 * rep.stateCount()], new double[]{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0}), pool);

        assertTrue(result.converged);
        assertEquals(2, result.periodConverged);
        assertEquals(expectedSnapshotDoubles, convergedSnapshotDoubleCount(pool));
        assertEquals(baselineScratchDoubles + expectedSnapshotDoubles, pool.retainedScratchDoubleCount());
        assertNotNull(pool.convergedForecastErrorCov);
        assertNotNull(pool.convergedFilteredStateCov);
        assertNotNull(pool.convergedPredictedStateCov);
        assertNotNull(pool.convergedKalmanGain);
        assertEquals(2L * rep.observationDimension() * rep.observationDimension(), doubleCount(pool.convergedForecastErrorCov));
        assertEquals(2L * rep.stateCount() * rep.stateCount(), doubleCount(pool.convergedFilteredStateCov));
        assertEquals(2L * rep.stateCount() * rep.stateCount(), doubleCount(pool.convergedPredictedStateCov));
        assertEquals(2L * rep.stateCount() * rep.observationDimension(), doubleCount(pool.convergedKalmanGain));
        assertEquals(32L, expectedSnapshotDoubles);
    }

    @Test
    void testComplexPoolReleaseRetainedScratchClearsOwnedBanks() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();

        ZKalmanFilter.filter(rep, StateInitialization.diffuse(), pool);

        assertTrue(pool.retainedScratchDoubleCount() > 0);

        pool.releaseRetainedScratch();

        assertEquals(0L, pool.retainedScratchDoubleCount());
        assertNull(pool.scratchBacking);
        assertNull(pool.selectionBacking);
    }

    @Test
    void testBorrowedDiffuseCompactResultShrinksRetainedBanks() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        FilterSpec compactSpec = FilterSpec.defaults().without(
            FilterSpec.Storage.FORECAST,
            FilterSpec.Storage.PREDICTED_STATE,
            FilterSpec.Storage.FILTERED_STATE,
            FilterSpec.Storage.KALMAN_GAIN,
            FilterSpec.Storage.LIKELIHOOD);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long fullScratchDoubles = expectedZFilterScratchDoubleCount(rep, FilterSpec.defaults());
        long fullRegularResultDoubles = expectedZDiffuseRegularFilterResultDoubleCount(rep, FilterSpec.defaults());
        long fullDiffuseResultDoubles = expectedZFilterDiffuseResultDoubleCount(rep, FilterSpec.defaults());
        long compactScratchDoubles = expectedZDiffuseFilterScratchDoubleCount(rep, compactSpec);
        long compactRegularResultDoubles = expectedZDiffuseRegularFilterResultDoubleCount(rep, compactSpec);
        long compactDiffuseResultDoubles = expectedZFilterDiffuseResultDoubleCount(rep, compactSpec);

        pool.reserve(rep, compactSpec);

        ZFilterResult result = ZKalmanFilter.filter(rep, StateInitialization.diffuse(), pool, compactSpec);

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
        void testClonePreservesBorrowedResultSnapshot() {
        double phi = 0.5, sigma = 1.0;
        ModelFixture rep1 = buildZAR1(phi, 0.0, sigma, new double[]{1.0, 0.5, -0.3, 0.8, 0.2});
        ModelFixture rep2 = buildZAR1(phi, 0.0, sigma, new double[]{-1.0, 0.25, 0.75, -0.4, 0.1});
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        ZFilterResult expected = ZKalmanFilter.filter(rep1, StateInitialization.approximateDiffuse());

        ZFilterResult borrowed = ZKalmanFilter.filter(rep1, StateInitialization.approximateDiffuse(), pool);
        ZFilterResult snapshot = borrowed.clone();

        ZKalmanFilter.filter(rep2, StateInitialization.approximateDiffuse(), pool);

        assertNotSame(snapshot, borrowed);
        assertFilterOutputsMatch(expected, snapshot);
    }

    @Test
    void testBorrowedCompactResultUsesLogicalSuppression() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8};
        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        FilterSpec spec = FilterSpec.defaults()
            .without(FilterSpec.Storage.FORECAST,
                FilterSpec.Storage.PREDICTED_STATE,
                FilterSpec.Storage.LIKELIHOOD,
                FilterSpec.Storage.KALMAN_GAIN);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long fullScratchDoubles = expectedZFilterScratchDoubleCount(zRep, FilterSpec.defaults());
        long fullResultDoubles = expectedZFilterResultDoubleCount(zRep, FilterSpec.defaults());
        long compactScratchDoubles = expectedZFilterScratchDoubleCount(zRep, spec);
        long compactResultDoubles = expectedZFilterResultDoubleCount(zRep, spec);
        long expectedSnapshotDoubles = expectedConvergedSnapshotDoubleCount(zRep);

        ZFilterResult full = ZKalmanFilter.filter(zRep, StateInitialization.approximateDiffuse(), pool, FilterSpec.defaults());

        pool.reserve(zRep, spec);

        ZFilterResult result = ZKalmanFilter.filter(zRep, StateInitialization.approximateDiffuse(), pool, spec);

        assertSame(full, result);
    assertTrue(result.converged);
        assertEquals(fullScratchDoubles, expectedZFilterScratchDoubleCount(zRep, FilterSpec.defaults()));
        assertTrue(compactResultDoubles < fullResultDoubles);
    assertEquals(compactScratchDoubles + expectedSnapshotDoubles, pool.retainedScratchDoubleCount());
        assertEquals(compactResultDoubles, pool.retainedResultDoubleCount());
        assertEquals(compactResultDoubles, (long) pool.resultBacking.length);
        assertTrue(pool.retainedTotalByteCount() < (fullScratchDoubles + fullResultDoubles) * Double.BYTES);
        assertEquals(0, result.predictedStateLength());
        assertEquals(0, result.predictedStateCovLength());
        assertEquals(0, result.forecastLength());
        assertEquals(0, result.forecastErrorLength());
        assertEquals(0, result.forecastErrorCovLength());
        assertEquals(0, result.kalmanGainLength());
        assertEquals(0, result.logLikelihoodObsLength());
        assertTrue(result.filteredStateLength() > 0);
    }

    @Test
    void testKPosdefLessThanKStates() {
        int kEndog = 1, kStates = 2, kPosdef = 1, nobs = 10;

        ModelFixture realRep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] rZ = {1.0, 0.0};
        double[] rd = {0.0};
        double[] rH = {1.0};
        double[] rT = {0.5, 0.3, 0.0, 0.0};
        double[] rc = {0.0, 0.0};
        double[] rR = {1.0, 0.0};
        double[] rQ = {1.0};
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7, -0.5, 0.3};
        for (int t = 0; t < nobs; t++) {
            realRep.setDesign(rZ, t);
            realRep.setObsIntercept(rd, t);
            realRep.setObsCov(rH, t);
            realRep.setTransition(rT, t);
            realRep.setStateIntercept(rc, t);
            realRep.setSelection(rR, t);
            realRep.setStateCov(rQ, t);
            realRep.setEndog(new double[]{y[t]}, t);
        }
        FilterResult realFr = KalmanFilter.filter(realRep, StateInitialization.approximateDiffuse());
        SmootherResult realSr = KalmanSmoother.smooth(realRep, realFr);

        ModelFixture zRep = ModelFixture.complex(kEndog, kStates, kPosdef, nobs);
        double[] zZ = {1.0, 0.0, 0.0, 0.0};
        double[] zd = {0.0, 0.0};
        double[] zH = {1.0, 0.0};
        double[] zT = {0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0};
        double[] zc = {0.0, 0.0, 0.0, 0.0};
        double[] zR = {1.0, 0.0, 0.0, 0.0};
        double[] zQ = {1.0, 0.0};
        for (int t = 0; t < nobs; t++) {
            zRep.setDesign(zZ, t);
            zRep.setObsIntercept(zd, t);
            zRep.setObsCov(zH, t);
            zRep.setTransition(zT, t);
            zRep.setStateIntercept(zc, t);
            zRep.setSelection(zR, t);
            zRep.setStateCov(zQ, t);
            zRep.setEndog(new double[]{y[t], 0.0}, t);
        }
        ZFilterResult zFr = ZKalmanFilter.filter(zRep, StateInitialization.approximateDiffuse());
        ZSmootherResult zSr = ZKalmanSmoother.smooth(zRep, zFr);

        int ks2c = kStates * 2 * kStates;
        for (int t = 0; t <= nobs; t++) {
            int realCovOff = t * kStates * kStates;
            int zCovOff = t * ks2c;
            for (int i = 0; i < kStates; i++) {
                for (int j = 0; j < kStates; j++) {
                    int rIdx = realCovOff + i * kStates + j;
                    int zIdx = zCovOff + i * 2 * kStates + j * 2;
                    assertEquals(realFr.predictedStateCov[rIdx],
                            zFr.predictedStateCov[zIdx], TOL_REAL,
                            "predictedStateCov[" + i + "," + j + "] Re at t=" + t);
                    if (i != j) {
                        assertEquals(0.0, zFr.predictedStateCov[zIdx + 1], TOL_REAL,
                                "predictedStateCov[" + i + "," + j + "] Im at t=" + t);
                    }
                }
            }
        }

        for (int t = 0; t < nobs; t++) {
            int realOff = t * kStates;
            int zOff = t * 2 * kStates;
            for (int i = 0; i < kStates; i++) {
                assertEquals(realSr.smoothedState[realOff + i],
                        zSr.smoothedState[zOff + i * 2], TOL_REAL,
                        "smoothedState[" + i + "] Re at t=" + t);
                assertEquals(0.0, zSr.smoothedState[zOff + i * 2 + 1], TOL_REAL,
                        "smoothedState[" + i + "] Im at t=" + t);
            }
        }

        for (int t = 0; t < nobs; t++) {
            int realDistOff = t * kPosdef;
            int zDistOff = t * 2 * kPosdef;
            assertEquals(realSr.smoothedStateDisturbance[realDistOff],
                    zSr.smoothedStateDisturbance[zDistOff], TOL_REAL,
                    "smoothedStateDisturbance Re at t=" + t);
            assertEquals(0.0, zSr.smoothedStateDisturbance[zDistOff + 1], TOL_REAL,
                    "smoothedStateDisturbance Im at t=" + t);
        }
    }

    @Test
    void testPartialMissing() {
        int kEndog = 2, kStates = 2, kPosdef = 1, nobs = 5;
        ModelFixture rep = ModelFixture.complex(kEndog, kStates, kPosdef, nobs);

        double[] Z = {1, 0, 0, 0, 0, 1, 0, 0};
        double[] H = {1, 0, 0, 0, 0, 0, 1, 0};
        double[] T = {0.9, 0, 0.1, 0, 0, 0, 0.9, 0};
        double[] R = {1, 0, 0, 0};
        double[] Q = {1, 0};
        double[] d = {0, 0, 0, 0};
        double[] c = {0, 0, 0, 0};

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setObsIntercept(d, t);
            rep.setStateIntercept(c, t);
        }

        rep.setEndog(new double[]{1.0, 0, 0.5, 0}, 0);
        rep.setEndog(new double[]{0.3, 0, Double.NaN, Double.NaN}, 1);
        rep.setMissing(new boolean[]{false, true}, 1);
        rep.setEndog(new double[]{-0.2, 0, 0.8, 0}, 2);
        rep.setEndog(new double[]{Double.NaN, Double.NaN, -0.1, 0}, 3);
        rep.setMissing(new boolean[]{true, false}, 3);
        rep.setEndog(new double[]{0.4, 0, 0.6, 0}, 4);

        double[] a0 = {0, 0, 0, 0};
        double[] P0 = {1, 0, 0, 0, 0, 0, 1, 0};
        ZFilterResult r = ZKalmanFilter.filter(rep, StateInitialization.known(a0, P0));

        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                int off = t * 2 * kStates + i * 2;
                assertFalse(Double.isNaN(r.predictedState[off]), "predictedState Re NaN at t=" + t + " i=" + i);
                assertFalse(Double.isNaN(r.predictedState[off + 1]), "predictedState Im NaN at t=" + t + " i=" + i);
            }
        }

        int foreErrOff1 = 1 * 2 * kEndog;
        assertEquals(0.0, r.forecastError[foreErrOff1 + 1 * 2], TOL_REAL,
                "forecastError for missing obs at t=1, i=1 should be 0");
        assertEquals(0.0, r.forecastError[foreErrOff1 + 1 * 2 + 1], TOL_REAL,
                "forecastError Im for missing obs at t=1, i=1 should be 0");

        int foreErrOff3 = 3 * 2 * kEndog;
        assertEquals(0.0, r.forecastError[foreErrOff3 + 0 * 2], TOL_REAL,
                "forecastError for missing obs at t=3, i=0 should be 0");
        assertEquals(0.0, r.forecastError[foreErrOff3 + 0 * 2 + 1], TOL_REAL,
                "forecastError Im for missing obs at t=3, i=0 should be 0");

        int ks2c = kStates * 2 * kStates;
        for (int t = 0; t <= nobs; t++) {
            assertHermitian(r.predictedStateCov, t * ks2c, kStates,
                    "predictedStateCov at t=" + t);
        }
        for (int t = 0; t < nobs; t++) {
            assertHermitian(r.filteredStateCov, t * ks2c, kStates,
                    "filteredStateCov at t=" + t);
        }

        ZSmootherResult sr = ZKalmanSmoother.smooth(rep, r);
        assertNotNull(sr);

        for (int t = 0; t < nobs; t++) {
            assertHermitian(sr.smoothedStateCov, t * ks2c, kStates,
                    "smoothedStateCov at t=" + t);
        }
    }

    @Test
    void testMissingObservations() {
        int kEndog = 1, kStates = 1, kPosdef = 1, nobs = 5;

        ModelFixture rep = ModelFixture.complex(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0, 0.0};
        double[] d = {0.0, 0.0};
        double[] H = {1.0, 0.0};
        double[] T = {0.5, 0.0};
        double[] c = {0.0, 0.0};
        double[] R = {1.0, 0.0};
        double[] Q = {1.0, 0.0};
        double[][] y = {{1.0, 0.0}, {0.0, 0.0}, {-0.3, 0.0}, {0.0, 0.0}, {0.2, 0.0}};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(y[t], t);
        }
        rep.setMissing(new boolean[]{false}, 0);
        rep.setMissing(new boolean[]{true}, 1);
        rep.setMissing(new boolean[]{false}, 2);
        rep.setMissing(new boolean[]{true}, 3);
        rep.setMissing(new boolean[]{false}, 4);

        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        assertEquals(0.0, fr.logLikelihoodObs[1], TOL_REAL,
                "logLikelihoodObs should be 0 for missing observation");
        assertEquals(0.0, fr.logLikelihoodObs[3], TOL_REAL,
                "logLikelihoodObs should be 0 for missing observation");

        int ks2c = kStates * 2 * kStates;
        int filtCovOff1 = 1 * ks2c;
        int predCovOff1 = 1 * ks2c;
        assertEquals(fr.predictedStateCov[predCovOff1], fr.filteredStateCov[filtCovOff1], TOL_REAL,
                "For missing obs, filteredStateCov should equal predictedStateCov Re");

        ZSmootherResult sr = ZKalmanSmoother.smooth(rep, fr);
        assertNotNull(sr);

        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
            assertHermitian(sr.smoothedStateCov, covOff, kStates,
                    "smoothedStateCov at t=" + t);
        }
    }

    private static void assertHermitian(double[] cov, int offset, int kStates, String msg) {
        for (int i = 0; i < kStates; i++) {
            for (int j = i; j < kStates; j++) {
                int ij = offset + (i * 2 * kStates + j * 2);
                int ji = offset + (j * 2 * kStates + i * 2);
                assertEquals(cov[ij], cov[ji], TOL_CMPLX,
                        msg + " P[" + i + "," + j + "].Re vs P[" + j + "," + i + "].Re");
                assertEquals(cov[ij + 1], -cov[ji + 1], TOL_CMPLX,
                        msg + " P[" + i + "," + j + "].Im vs -P[" + j + "," + i + "].Im");
            }
        }
    }

}
