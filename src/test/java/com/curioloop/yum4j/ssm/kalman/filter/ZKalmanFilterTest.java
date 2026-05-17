package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.KalmanModelFixtures;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;

import com.curioloop.yum4j.ssm.kalman.init.*;
import com.curioloop.yum4j.ssm.kalman.model.*;
import com.curioloop.yum4j.ssm.kalman.smooth.*;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

public class ZKalmanFilterTest {

    private static final double TOL_REAL = 1e-10;
    private static final double TOL_CMPLX = 1e-10;

    private static long expectedZFilterScratchDoubleCount(ModelFixture rep) {
    return expectedZFilterScratchDoubleCount(rep, FilterOptions.defaults());
    }

    private static long expectedZFilterScratchDoubleCount(ModelFixture rep, FilterOptions spec) {
    long kEndog = rep.kEndog;
    long kStates = rep.kStates;
    long kPosdef = rep.kPosdef;
    long updateLength = 4L * kStates * kEndog
        + 2L * kEndog * kEndog
        + 2L * kEndog;
    long predictionLength = Math.max(2L * kStates * kStates, 2L * kStates * kPosdef);
    long total = Math.max(updateLength, predictionLength);
    boolean storeForecast = spec.stores(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);
    boolean storePredicted = spec.stores(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)
        || (storeForecast && hasMissingObservations(rep));
    boolean storeFiltered = spec.stores(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE);
    if (!storePredicted) {
        total += 4L * kStates + 4L * kStates * kStates;
    }
    if (!storeFiltered) {
        total += 2L * kStates + 2L * kStates * kStates;
    }
    return total;
    }

    private static long expectedZDiffuseFilterScratchDoubleCount(ModelFixture rep, FilterOptions spec) {
    long kStates = rep.kStates;
    long kPosdef = rep.kPosdef;
    long total = 2L * kStates
        + 4L * kStates * kStates
        + Math.max(8L * kStates, Math.max(2L * kStates * kStates, 2L * kStates * kPosdef));
    boolean storeForecast = spec.stores(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);
    boolean storePredicted = spec.stores(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)
        || (storeForecast && hasMissingObservations(rep));
    if (!storePredicted) {
        total += 4L * kStates + 8L * kStates * kStates;
    }
    return Math.max(expectedZFilterScratchDoubleCount(rep, spec), total);
    }

    private static long expectedZFilterResultDoubleCount(ModelFixture rep) {
    return expectedZFilterResultDoubleCount(rep, FilterOptions.defaults());
    }

    private static long expectedZFilterResultDoubleCount(ModelFixture rep, FilterOptions spec) {
    long kEndog = rep.kEndog;
    long kStates = rep.kStates;
    long nobs = rep.nobs;
    boolean storeForecast = spec.stores(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);
    boolean storePredicted = spec.stores(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)
        || (storeForecast && hasMissingObservations(rep));
    boolean storeFiltered = spec.stores(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE);
    boolean storeGain = spec.stores(FilterOptions.Surface.KALMAN_GAIN);
    boolean storeLikelihood = spec.stores(FilterOptions.Surface.LIKELIHOOD);
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
    return expectedZFilterDiffuseResultDoubleCount(rep, FilterOptions.defaults());
    }

    private static long expectedZFilterDiffuseResultDoubleCount(ModelFixture rep, FilterOptions spec) {
    long kEndog = rep.kEndog;
    long kStates = rep.kStates;
    long nobs = rep.nobs;
    long total = 0L;
    if (spec.stores(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)) {
        total += 2L * kStates * kStates * (nobs + 1);
    }
    if (spec.stores(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE)) {
        total += 2L * kEndog * kEndog * nobs;
    }
    return total;
    }

    private static long expectedZDiffuseRegularFilterResultDoubleCount(ModelFixture rep, FilterOptions spec) {
    long kEndog = rep.kEndog;
    long kStates = rep.kStates;
    long nobs = rep.nobs;
    long total = 0L;
    if (spec.stores(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)) {
        total += 2L * kStates * (nobs + 1)
            + 2L * kStates * kStates * (nobs + 1);
    }
    if (spec.stores(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE)) {
        total += 2L * kStates * nobs;
        total += 2L * kStates * kStates * nobs;
    }
    if (spec.stores(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE)) {
        total += 2L * kEndog * nobs;
        total += 2L * kEndog * nobs;
        total += 2L * kEndog * kEndog * nobs;
    }
    if (spec.stores(FilterOptions.Surface.KALMAN_GAIN)) {
        total += 2L * kStates * kEndog * nobs;
    }
    if (spec.stores(FilterOptions.Surface.LIKELIHOOD)) {
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

    private static long expectedConvergedSnapshotDoubleCount(KalmanSSM rep) {
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

    public static ModelFixture buildAR1(double phi, double sigma, double[] y) {
        return KalmanModelFixtures.signalOnlyAr1(phi, sigma, y);
    }

    public static ModelFixture buildZAR1(double phiRe, double phiIm, double sigma, double[] y) {
        return KalmanModelFixtures.complexSignalOnlyAr1(phiRe, phiIm, sigma, y);
    }

    private static KalmanSSM buildComplexApproximateDiffusePoolModel() {
        return KalmanSSM.complexBuilder(1, 2, 2, 4)
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

    private static KalmanSSM buildComplexTimeVaryingSteadyStateTrapModel() {
        return KalmanSSM.complexBuilder(1, 1, 1, 6)
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

    private static KalmanSSM buildComplexMissingObservationConvergingModel() {
        KalmanSSM rep = KalmanSSM.complexBuilder(1, 1, 1, 12)
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

    private static KalmanSSM buildComplexConstantVarianceConvergingModel(int nobs) {
        return buildComplexConstantVarianceConvergingModel(1, 1, nobs);
    }

    private static KalmanSSM buildComplexConstantVarianceConvergingModel(int kEndog, int kStates, int nobs) {
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
        return KalmanSSM.complexBuilder(kEndog, kStates, kPosdef, nobs)
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

    private static KalmanSSM buildRealMissingObservationConvergingModel() {
        KalmanSSM rep = KalmanSSM.builder(1, 1, 1, 12)
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

    public static ModelFixture buildExactDiffuseLocalLevel(double[] y, double sigma2Y, double sigma2Mu) {
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

    public static ModelFixture buildZExactDiffuseLocalLevel(double[] y, double sigma2Y, double sigma2Mu) {
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

    public static ModelFixture buildExactDiffuseLocalLinearTrend(double[] y, double sigma2Y,
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

    public static ModelFixture buildZExactDiffuseLocalLinearTrend(double[] y, double sigma2Y,
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
        FilterResult realResult = KalmanEngine.filter(realRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        ZFilterResult zResult = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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

        ZFilterResult result = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
                assertComplexSymmetric(result.predictedStateCov, covOff, kStates,
                    "predictedStateCov at t=" + t);
        }
        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
                assertComplexSymmetric(result.filteredStateCov, covOff, kStates,
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
    void testComplexSymmetricCovariances() {
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

        ZFilterResult result = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        int ks2c = kStates * 2 * kStates;
        for (int t = 0; t <= nobs; t++) {
            int covOff = t * ks2c;
                assertComplexSymmetric(result.predictedStateCov, covOff, kStates,
                    "predictedStateCov at t=" + t);
        }
        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
                assertComplexSymmetric(result.filteredStateCov, covOff, kStates,
                    "filteredStateCov at t=" + t);
        }
    }

    @Test
    void testLogLikelihood() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7, -0.5, 0.3};

        ModelFixture realRep = buildAR1(phi, sigma, y);
        FilterResult realResult = KalmanEngine.filter(realRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        ZFilterResult zResult = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(realResult.logLikelihood(), zResult.logLikelihood(), TOL_REAL);
    }

    @Test
    void testComplexInitialFilteredTimingMatchesPredictedInitialization() {
        double phiRe = 0.5;
        double phiIm = 0.25;
        double sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
        ModelFixture zRep = buildZAR1(phiRe, phiIm, sigma, y);

        double[] filteredState = {0.2, -0.1};
        double[] filteredStateCov = {1.3, 0.0};
        double expectedStateRe = phiRe * filteredState[0] - phiIm * filteredState[1];
        double expectedStateIm = phiRe * filteredState[1] + phiIm * filteredState[0];
        double expectedCov = (phiRe * phiRe + phiIm * phiIm) * filteredStateCov[0] + sigma * sigma;
        ZFilterResult expected = KalmanEngine.filterComplex(zRep,
            InitialState.known(new double[]{expectedStateRe, expectedStateIm}, new double[]{expectedCov, 0.0}),
            FilterOptions.defaults());

        FilterOptions initFiltered = FilterOptions.defaults().toBuilder()
            .timing(FilterOptions.Timing.INIT_FILTERED)
            .build();
        ZFilterResult actual = KalmanEngine.filterComplex(zRep,
            InitialState.known(filteredState, filteredStateCov), initFiltered);

        assertFilterOutputsMatch(expected, actual);
    }

    @Test
    void testComplexConcentratedLikelihoodMatchesRealEmbeddedModel() {
        double phi = 0.5;
        double sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
        FilterOptions spec = FilterOptions.defaults().toBuilder()
            .concentratedLikelihood(true)
            .concentratedLikelihoodBurn(1)
            .build();

        ModelFixture realRep = buildAR1(phi, sigma, y);
        FilterResult realResult = KalmanEngine.filter(realRep,
            InitialState.known(new double[]{0.0}, new double[]{0.5}), spec);

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        ZFilterResult zResult = KalmanEngine.filterComplex(zRep,
            InitialState.known(new double[]{0.0, 0.0}, new double[]{0.5, 0.0}), spec);

        assertTrue(zResult.concentratedLikelihood());
        assertEquals(realResult.scale(), zResult.scale(), TOL_REAL);
        assertEquals(realResult.scaleObservationCount(), zResult.scaleObservationCount());
        assertEquals(realResult.scaleObsLength(), zResult.scaleObsLength());
        assertEquals(realResult.scaleObsCountLength(), zResult.scaleObsCountLength());
        for (int t = 0; t < y.length; t++) {
            assertEquals(realResult.scaleObs(t), zResult.scaleObs(t), TOL_REAL,
                "scaleObs at t=" + t);
            assertEquals(realResult.scaleObsCount(t), zResult.scaleObsCount(t),
                "scaleObsCount at t=" + t);
            assertEquals(realResult.logLikelihoodObs[t], zResult.logLikelihoodObs[t], TOL_REAL,
                "logLikelihoodObs at t=" + t);
            int zCovOff = zResult.forecastErrorCovOffset(t);
            assertEquals(realResult.forecastErrorCov(0, 0, t), zResult.forecastErrorCov[zCovOff], TOL_REAL,
                "forecastErrorCov Re at t=" + t);
            assertEquals(0.0, zResult.forecastErrorCov[zCovOff + 1], TOL_REAL,
                "forecastErrorCov Im at t=" + t);
        }
        assertEquals(realResult.logLikelihood(), zResult.logLikelihood(), TOL_REAL);
    }

    @Test
    void testComplexExactDiffuseConcentratedLikelihoodMatchesRealEmbeddedModel() {
        double[] y = {10.2394, 1.0, 1.0, 1.0, 1.0, 1.0};
        FilterOptions spec = FilterOptions.defaults().toBuilder()
            .concentratedLikelihood(true)
            .build();

        ModelFixture realRep = buildExactDiffuseLocalLevel(y, 1.993, 8.253);
        FilterResult realResult = KalmanEngine.filter(realRep, InitialState.diffuse(), spec);

        ModelFixture zRep = buildZExactDiffuseLocalLevel(y, 1.993, 8.253);
        ZFilterResult zResult = KalmanEngine.filterComplex(zRep, InitialState.diffuse(), spec);

        assertTrue(zResult.concentratedLikelihood());
        assertEquals(realResult.nobsDiffuse, zResult.nobsDiffuse);
        assertEquals(realResult.scale(), zResult.scale(), TOL_REAL);
        assertEquals(realResult.scaleObservationCount(), zResult.scaleObservationCount());
        for (int t = 0; t < y.length; t++) {
            assertEquals(realResult.scaleObs(t), zResult.scaleObs(t), TOL_REAL,
                "scaleObs at t=" + t);
            assertEquals(realResult.scaleObsCount(t), zResult.scaleObsCount(t),
                "scaleObsCount at t=" + t);
            assertEquals(realResult.logLikelihoodObs[t], zResult.logLikelihoodObs[t], TOL_REAL,
                "logLikelihoodObs at t=" + t);
        }
        assertEquals(realResult.logLikelihood(), zResult.logLikelihood(), TOL_REAL);
    }

    @Test
    void testComplexUnivariateFilterUsesUnivariateHistory() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
        ModelFixture zRep = buildZAR1(0.5, 0.1, 1.0, y);
        InitialState init = InitialState.known(new double[]{0.0, 0.0}, new double[]{0.5, 0.0});
        FilterOptions univariate = FilterOptions.defaults().toBuilder()
            .method(FilterMethod.UNIVARIATE)
            .build();

        ZFilterResult actual = KalmanEngine.filterComplex(zRep, init, univariate);

        assertTrue(actual.perObsKalmanGain);
        for (int t = 0; t < zRep.nobs; t++) {
            assertTrue(Double.isFinite(actual.logLikelihoodObs[t]), "logLikelihoodObs at t=" + t);
        }
    }

    @Test
    void testDifferentInitializations() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7, -0.5, 0.3};
        int kStates = 1;

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);

        double[] initState = {0.0, 0.0};
        double[] initStateCov = {1.0, 0.0};
        ZFilterResult knownResult = KalmanEngine.filterComplex(zRep, InitialState.known(initState, initStateCov), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertNotNull(knownResult);

        ZFilterResult diffuseResult = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertNotNull(diffuseResult);

        InitialState stationaryInit = InitialState.stationary(zRep);
        assertNotNull(stationaryInit);

        ZFilterResult stationaryResult = KalmanEngine.filterComplex(zRep, stationaryInit, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertNotNull(stationaryResult);

        double expectedP0 = sigma * sigma / (1 - phi * phi);
        assertEquals(expectedP0, stationaryInit.initialStateCov()[0], TOL_REAL,
                "Stationary initial covariance");
        assertEquals(0.0, stationaryInit.initialStateCov()[1], TOL_REAL,
                "Stationary initial covariance Im");

        ModelFixture realRep = buildAR1(phi, sigma, y);
        InitialState realStationaryInit = InitialState.stationary(realRep);
        FilterResult realStationaryResult = KalmanEngine.filter(realRep, realStationaryInit, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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

        InitialState stationaryInit = InitialState.stationary(zRep);
        InitialState knownInit = InitialState.known(
                stationaryInit.initialState().clone(),
                stationaryInit.initialStateCov().clone());

        ZFilterResult stationaryResult = KalmanEngine.filterComplex(zRep, stationaryInit, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZFilterResult knownResult = KalmanEngine.filterComplex(zRep, knownInit, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
        FilterResult realResult = KalmanEngine.filter(realRep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        ModelFixture zRep = buildZExactDiffuseLocalLevel(y, sigma2Y, sigma2Mu);
        ZFilterResult zResult = KalmanEngine.filterComplex(zRep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
        FilterResult realResult = KalmanEngine.filter(realRep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        ModelFixture zRep = buildZExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        zRep.setMissing(new boolean[]{true}, 1);
        ZFilterResult zResult = KalmanEngine.filterComplex(zRep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
    void testFilterOptionsSuppressesStoredOutputs() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8};

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        FilterOptions spec = FilterOptions.defaults()
                .without(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE,
                        FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE,
                        FilterOptions.Surface.LIKELIHOOD,
                        FilterOptions.Surface.KALMAN_GAIN);

        ZFilterResult result = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), spec);

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

        ZFilterResult result = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertNotNull(result);

        int ks2c = kStates * 2 * kStates;
        for (int t = 0; t <= nobs; t++) {
            int covOff = t * ks2c;
                assertComplexSymmetric(result.predictedStateCov, covOff, kStates,
                    "predictedStateCov at t=" + t);
        }
        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
                assertComplexSymmetric(result.filteredStateCov, covOff, kStates,
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
        ZFilterResult expected = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        ZFilterResult r1 = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZFilterResult r2 = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertSame(r1, r2);

        assertFilterOutputsMatch(expected, r1);
    }

    @Test
    void testApproximateDiffusePoolReuseResetsInitialComplexStateAndCovariance() {
        KalmanSSM rep = buildComplexApproximateDiffusePoolModel();
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();

        KalmanEngine.filterComplex(rep,
            InitialState.known(
                new double[]{2.0, 0.0, -3.0, 0.0},
                new double[]{4.0, 0.0, 1.5, 0.0, 1.5, 0.0, 5.0, 0.0}),
            pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZFilterResult reused = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
        KalmanSSM realRep = KalmanSSM.builder(1, 1, 1, 6)
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
        KalmanSSM zRep = buildComplexTimeVaryingSteadyStateTrapModel();

        FilterResult reference = KalmanEngine.filterUnivariate(realRep, InitialState.known(new double[]{0.0}, new double[]{0.5}), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZFilterResult result = KalmanEngine.filterComplex(zRep, InitialState.known(new double[]{0.0, 0.0}, new double[]{0.5, 0.0}), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
        KalmanSSM realRep = buildRealMissingObservationConvergingModel();
        KalmanSSM zRep = buildComplexMissingObservationConvergingModel();

        FilterResult reference = KalmanEngine.filterUnivariate(realRep,
            InitialState.known(new double[]{0.0}, new double[]{0.5}), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZFilterResult result = KalmanEngine.filterComplex(zRep,
            InitialState.known(new double[]{0.0, 0.0}, new double[]{0.5, 0.0}), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
        ZFilterResult expected = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        ZFilterResult borrowed1 = KalmanEngine.filterComplex(rep, InitialState.diffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZFilterResult borrowed2 = KalmanEngine.filterComplex(rep, InitialState.diffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertSame(borrowed1, borrowed2);
        assertTrue(borrowed1.nobsDiffuse > 0);
        assertFilterOutputsMatch(expected, borrowed1);
    }

    @Test
    void testPoolSupportsExactDiffuseLeanSimulationSpec() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        FilterOptions spec = SmootherOptions.conventional().requiredFilterOptions();
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        ZFilterResult expected = KalmanEngine.filterComplex(rep, InitialState.diffuse(), spec);

        ZFilterResult borrowed1 = KalmanEngine.filterComplex(rep, InitialState.diffuse(), pool, spec);
        ZFilterResult borrowed2 = KalmanEngine.filterComplex(rep, InitialState.diffuse(), pool, spec);

        assertSame(borrowed1, borrowed2);
        assertTrue(borrowed1.nobsDiffuse > 0);
        assertFilterOutputsMatch(expected, borrowed1);
    }

    @Test
    void testPoolReserveQuantifiesRetainedComplexFilterSurfaces() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long expectedScratchDoubles = expectedZFilterScratchDoubleCount(rep);
        long expectedResultDoubles = expectedZFilterResultDoubleCount(rep);

        pool.reserve(rep, FilterOptions.defaults());

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

        ZFilterResult result = KalmanEngine.filterComplex(rep, InitialState.stationary(rep.stateCount()), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        long expectedWorkspaceDoubles = InitialState.requiredStationaryBackingLength(rep)
            + InitialState.requiredStationaryPivotLength(rep) * ((long) Integer.BYTES) / Double.BYTES;
        assertTrue(result.converged);
        assertEquals(expectedZFilterScratchDoubleCount(rep) + expectedWorkspaceDoubles,
            pool.retainedScratchDoubleCount());
        assertEquals(expectedZFilterScratchDoubleCount(rep), (long) pool.scratchBacking.length);
        assertEquals(0L, convergedSnapshotDoubleCount(pool));
    }

    @Test
    void testDiffusePoolQuantifiesAdditionalComplexFilterBanks() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long expectedScratchDoubles = expectedZDiffuseFilterScratchDoubleCount(rep, FilterOptions.defaults());
        long expectedResultDoubles = expectedZFilterResultDoubleCount(rep);
        long expectedDiffuseResultDoubles = expectedZFilterDiffuseResultDoubleCount(rep);

        pool.reserve(rep, FilterOptions.defaults());

        ZFilterResult result = KalmanEngine.filterComplex(rep, InitialState.diffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertTrue(result.nobsDiffuse > 0);
        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedResultDoubles + expectedDiffuseResultDoubles,
            pool.retainedResultDoubleCount());
        assertSame(pool.resultBacking, pool.diffuseResultBacking);
        assertEquals(expectedResultDoubles + expectedDiffuseResultDoubles, (long) pool.resultBacking.length);
        assertEquals((expectedScratchDoubles + expectedResultDoubles + expectedDiffuseResultDoubles) * Double.BYTES,
            pool.retainedTotalByteCount());
    }

    @Test
    void testReservePreservesComplexFilterRetainedBanksUntilExplicitRelease() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long expectedReservedScratchDoubles = expectedZFilterScratchDoubleCount(rep);
        long expectedDiffuseScratchDoubles = expectedZDiffuseFilterScratchDoubleCount(rep, FilterOptions.defaults());
        long expectedResultDoubles = expectedZFilterResultDoubleCount(rep);

        KalmanEngine.filterComplex(rep, InitialState.diffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(expectedDiffuseScratchDoubles, pool.retainedScratchDoubleCount());
        assertTrue(pool.retainedResultDoubleCount() > expectedResultDoubles);
        assertNotNull(pool.diffuseResultBacking);

        pool.reserve(rep, FilterOptions.defaults());

        assertEquals(expectedDiffuseScratchDoubles, pool.retainedScratchDoubleCount());
        assertTrue(pool.retainedResultDoubleCount() > expectedResultDoubles);
        assertNotNull(pool.diffuseResultBacking);

        pool.releaseRetainedScratch();
        pool.releaseRetainedResults();
        pool.reserve(rep, FilterOptions.defaults());

        assertEquals(expectedReservedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedResultDoubles, pool.retainedResultDoubleCount());
        assertNull(pool.diffuseResultBacking);
        assertNotNull(pool.resultBacking);
        assertEquals(expectedResultDoubles, (long) pool.resultBacking.length);
    }

    @Test
    void testComplexPoolDefersConvergedSnapshotAllocationUntilNeeded() {
        KalmanSSM rep = buildComplexConstantVarianceConvergingModel(1);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long expectedSnapshotDoubles = expectedConvergedSnapshotDoubleCount(rep);

        pool.reserve(rep, FilterOptions.defaults());
        long baselineScratchDoubles = pool.retainedScratchDoubleCount();

        assertNull(pool.convergedForecastErrorCov);
        assertNull(pool.convergedFilteredStateCov);
        assertNull(pool.convergedPredictedStateCov);
        assertNull(pool.convergedKalmanGain);
        assertEquals(0L, convergedSnapshotDoubleCount(pool));
        assertEquals(8L, expectedSnapshotDoubles);

        KalmanEngine.filterComplex(rep, InitialState.known(new double[]{0.0, 0.0}, new double[]{0.5, 0.0}), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertNull(pool.convergedForecastErrorCov);
        assertNull(pool.convergedFilteredStateCov);
        assertNull(pool.convergedPredictedStateCov);
        assertNull(pool.convergedKalmanGain);
        assertEquals(0L, convergedSnapshotDoubleCount(pool));
        assertEquals(baselineScratchDoubles, pool.retainedScratchDoubleCount());
    }

    @Test
    void testComplexPoolReleasesConvergedSnapshotsWhenFilterConverges() {
        KalmanSSM rep = buildComplexConstantVarianceConvergingModel(2, 2, 6);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long expectedSnapshotDoubles = expectedConvergedSnapshotDoubleCount(rep);

        pool.reserve(rep, FilterOptions.defaults());
        long baselineScratchDoubles = pool.retainedScratchDoubleCount();

        ZFilterResult result = KalmanEngine.filterComplex(rep,
            InitialState.known(new double[2 * rep.stateCount()], new double[]{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0}), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertTrue(result.converged);
        assertEquals(2, result.periodConverged);
        assertEquals(0L, convergedSnapshotDoubleCount(pool));
        assertEquals(baselineScratchDoubles, pool.retainedScratchDoubleCount());
        assertNull(pool.convergedForecastErrorCov);
        assertNull(pool.convergedFilteredStateCov);
        assertNull(pool.convergedPredictedStateCov);
        assertNull(pool.convergedKalmanGain);
        assertEquals(32L, expectedSnapshotDoubles);
    }

    @Test
    void testComplexPoolReleaseRetainedScratchClearsOwnedBanks() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        ModelFixture rep = buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();

        KalmanEngine.filterComplex(rep, InitialState.diffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
        FilterOptions compactSpec = FilterOptions.defaults().without(
            FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE,
            FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE,
            FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
            FilterOptions.Surface.KALMAN_GAIN,
            FilterOptions.Surface.LIKELIHOOD);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long fullScratchDoubles = expectedZFilterScratchDoubleCount(rep, FilterOptions.defaults());
        long fullRegularResultDoubles = expectedZDiffuseRegularFilterResultDoubleCount(rep, FilterOptions.defaults());
        long fullDiffuseResultDoubles = expectedZFilterDiffuseResultDoubleCount(rep, FilterOptions.defaults());
        long compactScratchDoubles = expectedZDiffuseFilterScratchDoubleCount(rep, compactSpec);
        long compactRegularResultDoubles = expectedZDiffuseRegularFilterResultDoubleCount(rep, compactSpec);
        long compactDiffuseResultDoubles = expectedZFilterDiffuseResultDoubleCount(rep, compactSpec);

        pool.reserve(rep, compactSpec);

        ZFilterResult result = KalmanEngine.filterComplex(rep, InitialState.diffuse(), pool, compactSpec);

        assertTrue(result.nobsDiffuse > 0);
        assertTrue(compactRegularResultDoubles < fullRegularResultDoubles);
        assertTrue(compactDiffuseResultDoubles < fullDiffuseResultDoubles);
        assertEquals(compactScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(compactRegularResultDoubles + compactDiffuseResultDoubles,
            pool.retainedResultDoubleCount());
        assertSame(pool.resultBacking, pool.diffuseResultBacking);
        assertEquals(compactRegularResultDoubles + compactDiffuseResultDoubles, (long) pool.resultBacking.length);
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
        ZFilterResult expected = KalmanEngine.filterComplex(rep1, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        ZFilterResult borrowed = KalmanEngine.filterComplex(rep1, InitialState.approximateDiffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZFilterResult snapshot = borrowed.clone();

        KalmanEngine.filterComplex(rep2, InitialState.approximateDiffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertNotSame(snapshot, borrowed);
        assertFilterOutputsMatch(expected, snapshot);
    }

    @Test
    void testBorrowedCompactResultUsesLogicalSuppression() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8};
        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        FilterOptions spec = FilterOptions.defaults()
            .without(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE,
                FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE,
                FilterOptions.Surface.LIKELIHOOD,
                FilterOptions.Surface.KALMAN_GAIN);
        ZKalmanFilter.Pool pool = new ZKalmanFilter.Pool();
        long fullScratchDoubles = expectedZFilterScratchDoubleCount(zRep, FilterOptions.defaults());
        long fullResultDoubles = expectedZFilterResultDoubleCount(zRep, FilterOptions.defaults());
        long compactScratchDoubles = expectedZFilterScratchDoubleCount(zRep, spec);
        long compactResultDoubles = expectedZFilterResultDoubleCount(zRep, spec);
        long expectedSnapshotDoubles = expectedConvergedSnapshotDoubleCount(zRep);

        ZFilterResult full = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), pool, FilterOptions.defaults());

        pool.reserve(zRep, spec);

        ZFilterResult result = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), pool, spec);

        assertSame(full, result);
        assertTrue(result.converged);
        assertEquals(fullScratchDoubles, expectedZFilterScratchDoubleCount(zRep, FilterOptions.defaults()));
        assertTrue(compactResultDoubles < fullResultDoubles);
        assertEquals(compactScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(fullResultDoubles, pool.retainedResultDoubleCount());
        assertEquals(fullResultDoubles, (long) pool.resultBacking.length);
        assertEquals((compactScratchDoubles + fullResultDoubles) * Double.BYTES,
            pool.retainedTotalByteCount());
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
        FilterResult realFr = KalmanEngine.filter(realRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult realSr = SmootherEngine.smooth(realRep, realFr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

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
        ZFilterResult zFr = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZSmootherResult zSr = SmootherEngine.smoothComplex(zRep, zFr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

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
        ZFilterResult r = KalmanEngine.filterComplex(rep, InitialState.known(a0, P0), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
            assertComplexSymmetric(r.predictedStateCov, t * ks2c, kStates,
                    "predictedStateCov at t=" + t);
        }
        for (int t = 0; t < nobs; t++) {
            assertComplexSymmetric(r.filteredStateCov, t * ks2c, kStates,
                    "filteredStateCov at t=" + t);
        }

        ZSmootherResult sr = SmootherEngine.smoothComplex(rep, r, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        assertNotNull(sr);

        for (int t = 0; t < nobs; t++) {
            assertComplexSymmetric(sr.smoothedStateCov, t * ks2c, kStates,
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

        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(0.0, fr.logLikelihoodObs[1], TOL_REAL,
                "logLikelihoodObs should be 0 for missing observation");
        assertEquals(0.0, fr.logLikelihoodObs[3], TOL_REAL,
                "logLikelihoodObs should be 0 for missing observation");

        int ks2c = kStates * 2 * kStates;
        int filtCovOff1 = 1 * ks2c;
        int predCovOff1 = 1 * ks2c;
        assertEquals(fr.predictedStateCov[predCovOff1], fr.filteredStateCov[filtCovOff1], TOL_REAL,
                "For missing obs, filteredStateCov should equal predictedStateCov Re");

        ZSmootherResult sr = SmootherEngine.smoothComplex(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        assertNotNull(sr);

        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
                assertComplexSymmetric(sr.smoothedStateCov, covOff, kStates,
                    "smoothedStateCov at t=" + t);
        }
    }

    private static void assertComplexSymmetric(double[] cov, int offset, int kStates, String msg) {
        for (int i = 0; i < kStates; i++) {
            for (int j = i; j < kStates; j++) {
                int ij = offset + (i * 2 * kStates + j * 2);
                int ji = offset + (j * 2 * kStates + i * 2);
                assertEquals(cov[ij], cov[ji], TOL_CMPLX,
                        msg + " P[" + i + "," + j + "].Re vs P[" + j + "," + i + "].Re");
                assertEquals(cov[ij + 1], cov[ji + 1], TOL_CMPLX,
                        msg + " P[" + i + "," + j + "].Im vs P[" + j + "," + i + "].Im");
            }
        }
    }

}
