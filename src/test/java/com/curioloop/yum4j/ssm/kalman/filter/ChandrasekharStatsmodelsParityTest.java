package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChandrasekharStatsmodelsParityTest {

    private static final double TOL = 1e-9;

    @Test
    void scalarChandrasekharMatchesConventionalFilterAndSmoother() {
        ModelFixture model = scalarStationaryModel(48);
        InitialState initialState = InitialState.stationary(model);

        FilterResult conventionalFilter = KalmanEngine.filter(model, initialState, filterOptions(FilterMethod.CONVENTIONAL,
            ForecastErrorStrategy.AUTO));
        FilterResult chandrasekharFilter = KalmanEngine.filter(model, initialState, filterOptions(FilterMethod.CHANDRASEKHAR,
            ForecastErrorStrategy.AUTO));
        assertFilterOutputsMatch(conventionalFilter, chandrasekharFilter, "scalar");

        SmootherOptions smootherOptions = SmootherOptions.builder().retainAll().build();
        SmootherResult conventionalSmoother = SmootherEngine.smooth(model, conventionalFilter, smootherOptions);
        SmootherResult chandrasekharSmoother = SmootherEngine.smooth(model, chandrasekharFilter, smootherOptions);
        assertSmootherOutputsMatch(conventionalSmoother, chandrasekharSmoother, "scalar");
    }

    @Test
    void multivariateChandrasekharMatchesConventionalForDenseObservationCovariance() {
        ModelFixture model = varLikeStationaryModel(48, true);
        InitialState initialState = InitialState.stationary(model);

        FilterResult conventional = KalmanEngine.filter(model, initialState, filterOptions(FilterMethod.CONVENTIONAL,
            ForecastErrorStrategy.AUTO));
        FilterResult chandrasekhar = KalmanEngine.filter(model, initialState, filterOptions(FilterMethod.CHANDRASEKHAR,
            ForecastErrorStrategy.AUTO));

        assertFilterOutputsMatch(conventional, chandrasekhar, "multivariate dense H");
    }

    @Test
    void multivariateChandrasekharMatchesConventionalForDiagonalObservationCovariance() {
        ModelFixture model = varLikeStationaryModel(48, false);
        InitialState initialState = InitialState.stationary(model);

        FilterResult conventional = KalmanEngine.filter(model, initialState, routeOptions(FilterMethod.CONVENTIONAL));
        FilterResult chandrasekhar = KalmanEngine.filter(model, initialState, filterOptions(FilterMethod.CHANDRASEKHAR,
            ForecastErrorStrategy.AUTO));

        assertFilterOutputsMatch(conventional, chandrasekhar, "multivariate diagonal H");
    }

    @Test
    void invalidChandrasekharProfilesRaiseStatsmodelsStyleErrors() {
        ModelFixture missing = varLikeStationaryModel(12, false);
        missing.setMissing(new boolean[]{true, false}, 2);
        assertUnsupported(missing, FilterOptions.builder().method(FilterMethod.CHANDRASEKHAR).build(),
            "missing observations");

        ModelFixture alternateTiming = varLikeStationaryModel(12, false);
        assertUnsupported(alternateTiming, FilterOptions.builder()
            .method(FilterMethod.CHANDRASEKHAR)
            .timing(FilterOptions.Timing.INIT_FILTERED)
            .build(), "INIT_FILTERED timing");

        ModelFixture timeVarying = varLikeStationaryModel(12, false);
        timeVarying.setObsCov(new double[]{0.8, 0.0, 0.0, 0.95}, 5);
        assertUnsupported(timeVarying, FilterOptions.builder().method(FilterMethod.CHANDRASEKHAR).build(),
            "time-varying system matrices");
    }

    private static ModelFixture scalarStationaryModel(int nobs) {
        ModelFixture model = new ModelFixture(1, 1, 1, nobs);
        for (int t = 0; t < nobs; t++) {
            model.setDesign(new double[]{1.0}, t);
            model.setObsCov(new double[]{0.4}, t);
            model.setTransition(new double[]{0.72}, t);
            model.setSelection(new double[]{1.0}, t);
            model.setStateCov(new double[]{0.8}, t);
            model.setEndog(new double[]{Math.sin(0.17 * t) + 0.1 * Math.cos(0.03 * t)}, t);
        }
        return model;
    }

    private static ModelFixture varLikeStationaryModel(int nobs, boolean denseObsCov) {
        ModelFixture model = new ModelFixture(2, 2, 2, nobs);
        double[] design = {1.0, 0.0, 0.0, 1.0};
        double[] obsCov = denseObsCov ? new double[]{0.7, 0.2, 0.2, 0.95} : new double[]{0.7, 0.0, 0.0, 0.95};
        double[] transition = {0.62, 0.11, -0.18, 0.54};
        double[] selection = {1.0, 0.0, 0.0, 1.0};
        double[] stateCov = {0.9, 0.12, 0.12, 0.6};
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{Math.sin(0.11 * t) + 0.03 * t, Math.cos(0.07 * t) - 0.02 * t}, t);
        }
        return model;
    }

    private static FilterOptions filterOptions(FilterMethod method, ForecastErrorStrategy strategy) {
        return FilterOptions.builder()
            .method(method)
            .forecastErrorStrategy(strategy)
            .retainAll()
            .build();
    }

    private static FilterOptions routeOptions(FilterMethod method) {
        return FilterOptions.builder()
            .method(method)
            .retainAll()
            .build();
    }

    private static void assertUnsupported(ModelFixture model, FilterOptions options, String messagePart) {
        UnsupportedOperationException exception = assertThrows(UnsupportedOperationException.class,
            () -> KalmanEngine.filter(model, InitialState.stationary(model), options));
        assertTrue(exception.getMessage().contains(messagePart), exception.getMessage());
    }

    private static void assertFilterOutputsMatch(FilterResult expected, FilterResult actual, String label) {
        assertArrayClose(activeSlice(expected.predictedState, expected.predictedStateBase(), expected.predictedStateLength()),
            activeSlice(actual.predictedState, actual.predictedStateBase(), actual.predictedStateLength()), label + " predictedState");
        assertArrayClose(activeSlice(expected.predictedStateCov, expected.predictedStateCovBase(), expected.predictedStateCovLength()),
            activeSlice(actual.predictedStateCov, actual.predictedStateCovBase(), actual.predictedStateCovLength()), label + " predictedStateCov");
        assertArrayClose(activeSlice(expected.filteredState, expected.filteredStateBase(), expected.filteredStateLength()),
            activeSlice(actual.filteredState, actual.filteredStateBase(), actual.filteredStateLength()), label + " filteredState");
        assertArrayClose(activeSlice(expected.filteredStateCov, expected.filteredStateCovBase(), expected.filteredStateCovLength()),
            activeSlice(actual.filteredStateCov, actual.filteredStateCovBase(), actual.filteredStateCovLength()), label + " filteredStateCov");
        assertArrayClose(activeSlice(expected.forecast, expected.forecastBase(), expected.forecastLength()),
            activeSlice(actual.forecast, actual.forecastBase(), actual.forecastLength()), label + " forecast");
        assertArrayClose(activeSlice(expected.forecastError, expected.forecastErrorBase(), expected.forecastErrorLength()),
            activeSlice(actual.forecastError, actual.forecastErrorBase(), actual.forecastErrorLength()), label + " forecastError");
        assertArrayClose(activeSlice(expected.forecastErrorCov, expected.forecastErrorCovBase(), expected.forecastErrorCovLength()),
            activeSlice(actual.forecastErrorCov, actual.forecastErrorCovBase(), actual.forecastErrorCovLength()), label + " forecastErrorCov");
        assertArrayClose(activeSlice(expected.standardizedForecastError, expected.standardizedForecastErrorBase(),
                expected.standardizedForecastErrorLength()),
            activeSlice(actual.standardizedForecastError, actual.standardizedForecastErrorBase(), actual.standardizedForecastErrorLength()),
            label + " standardizedForecastError");
        assertArrayClose(activeSlice(expected.kalmanGain, expected.kalmanGainBase(), expected.kalmanGainLength()),
            activeSlice(actual.kalmanGain, actual.kalmanGainBase(), actual.kalmanGainLength()), label + " kalmanGain");
        assertArrayClose(activeSlice(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            activeSlice(actual.logLikelihoodObs, actual.logLikelihoodObsBase(), actual.logLikelihoodObsLength()), label + " logLikelihoodObs");
        assertEquals(expected.nobsDiffuse, actual.nobsDiffuse, label + " nobsDiffuse");
    }

    private static void assertSmootherOutputsMatch(SmootherResult expected, SmootherResult actual, String label) {
        assertArrayClose(activeSlice(expected.smoothedState, expected.smoothedStateBase(), expected.smoothedStateLength()),
            activeSlice(actual.smoothedState, actual.smoothedStateBase(), actual.smoothedStateLength()), label + " smoothedState");
        assertArrayClose(activeSlice(expected.smoothedStateCov, expected.smoothedStateCovBase(), expected.smoothedStateCovLength()),
            activeSlice(actual.smoothedStateCov, actual.smoothedStateCovBase(), actual.smoothedStateCovLength()), label + " smoothedStateCov");
        assertArrayClose(activeSlice(expected.smoothedStateAutocovariance, expected.smoothedStateAutocovarianceBase(),
                expected.smoothedStateAutocovarianceLength()),
            activeSlice(actual.smoothedStateAutocovariance, actual.smoothedStateAutocovarianceBase(), actual.smoothedStateAutocovarianceLength()),
            label + " smoothedStateAutocovariance");
        assertArrayClose(activeSlice(expected.smoothedObsDisturbance, expected.smoothedObsDisturbanceBase(), expected.smoothedObsDisturbanceLength()),
            activeSlice(actual.smoothedObsDisturbance, actual.smoothedObsDisturbanceBase(), actual.smoothedObsDisturbanceLength()),
            label + " smoothedObsDisturbance");
        assertArrayClose(activeSlice(expected.smoothedStateDisturbance, expected.smoothedStateDisturbanceBase(),
                expected.smoothedStateDisturbanceLength()),
            activeSlice(actual.smoothedStateDisturbance, actual.smoothedStateDisturbanceBase(), actual.smoothedStateDisturbanceLength()),
            label + " smoothedStateDisturbance");
        assertArrayClose(activeSlice(expected.smoothedObsDisturbanceCov, expected.smoothedObsDisturbanceCovBase(),
                expected.smoothedObsDisturbanceCovLength()),
            activeSlice(actual.smoothedObsDisturbanceCov, actual.smoothedObsDisturbanceCovBase(), actual.smoothedObsDisturbanceCovLength()),
            label + " smoothedObsDisturbanceCov");
        assertArrayClose(activeSlice(expected.smoothedStateDisturbanceCov, expected.smoothedStateDisturbanceCovBase(),
                expected.smoothedStateDisturbanceCovLength()),
            activeSlice(actual.smoothedStateDisturbanceCov, actual.smoothedStateDisturbanceCovBase(), actual.smoothedStateDisturbanceCovLength()),
            label + " smoothedStateDisturbanceCov");
    }

    private static double[] activeSlice(double[] values, int base, int length) {
        return values == null ? null : Arrays.copyOfRange(values, base, base + length);
    }

    private static void assertArrayClose(double[] expected, double[] actual, String label) {
        if (expected == null || actual == null) {
            assertEquals(expected, actual, label);
            return;
        }
        assertEquals(expected.length, actual.length, label + " length");
        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual[i], label + "[" + i + "]");
        }
    }

    private static void assertClose(double expected, double actual, String label) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), label);
            return;
        }
        double scale = Math.max(1.0, Math.max(Math.abs(expected), Math.abs(actual)));
        assertEquals(expected, actual, TOL * scale, label);
    }
}