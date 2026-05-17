package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherMethod;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CollapsedStatsmodelsParityTest {

    private static final double TOL = 1e-8;

    @Test
    void trivariateCollapsedFilterMatchesConventionalWithPartialMissing() {
        assertCollapsedFilterMatchesConventional(trivariateModel(40, MissingPattern.PARTIAL), FilterOptions.Timing.INIT_PREDICTED);
    }

    @Test
    void trivariateCollapsedFilterMatchesConventionalWithAllMissingPeriods() {
        assertCollapsedFilterMatchesConventional(trivariateModel(40, MissingPattern.ALL), FilterOptions.Timing.INIT_PREDICTED);
    }

    @Test
    void trivariateCollapsedFilterMatchesConventionalWithAlternateTiming() {
        assertCollapsedFilterMatchesConventional(trivariateModel(40, MissingPattern.NONE), FilterOptions.Timing.INIT_FILTERED);
    }

    @Test
    void collapsedSmootherMatchesConventionalAcrossStatsmodelsMethods() {
        ModelFixture model = trivariateModel(40, MissingPattern.PARTIAL);
        for (SmootherMethod method : new SmootherMethod[]{SmootherMethod.CONVENTIONAL, SmootherMethod.CLASSICAL, SmootherMethod.ALTERNATIVE}) {
            SmootherOptions.Builder smootherBuilder = SmootherOptions.builder().method(method);
            if (method == SmootherMethod.CONVENTIONAL) {
                smootherBuilder.retainAll();
            }
            SmootherOptions smootherOptions = smootherBuilder.build();
            FilterOptions conventionalOptions = smootherOptions.requiredFilterOptions()
                .toBuilder()
                .method(FilterMethod.CONVENTIONAL)
                .build();
            FilterOptions collapsedOptions = smootherOptions.requiredFilterOptions()
                .toBuilder()
                .method(FilterMethod.COLLAPSED)
                .build();

            FilterResult conventionalFilter = KalmanEngine.filter(model, approximateDiffuse(), conventionalOptions);
            FilterResult collapsedFilter = KalmanEngine.filter(model, approximateDiffuse(), collapsedOptions);
            SmootherResult conventional = SmootherEngine.smooth(model, conventionalFilter, smootherOptions);
            SmootherResult collapsed = SmootherEngine.smooth(model, collapsedFilter, smootherOptions);

            assertSmootherOutputsMatch(conventional, collapsed, method.name());
        }
    }

    @Test
    void collapsedSingleFactorModelWithInitialMissingMatchesConventionalLikelihood() {
        ModelFixture model = singleFactorDfmMissingModel();
        FilterResult conventional = KalmanEngine.filter(model, InitialState.approximateDiffuse(1), filterOptions(FilterMethod.CONVENTIONAL,
            FilterOptions.Timing.INIT_PREDICTED));
        FilterResult collapsed = KalmanEngine.filter(model, InitialState.approximateDiffuse(1), filterOptions(FilterMethod.COLLAPSED,
            FilterOptions.Timing.INIT_PREDICTED));

        assertEquals(conventional.logLikelihood(), collapsed.logLikelihood(), TOL);
        assertFilterOutputsMatch(conventional, collapsed, "single factor DFM missing");
    }

    private static void assertCollapsedFilterMatchesConventional(ModelFixture model, FilterOptions.Timing timing) {
        FilterResult conventional = KalmanEngine.filter(model, approximateDiffuse(), filterOptions(FilterMethod.CONVENTIONAL, timing));
        FilterResult collapsed = KalmanEngine.filter(model, approximateDiffuse(), filterOptions(FilterMethod.COLLAPSED, timing));

        assertFilterOutputsMatch(conventional, collapsed, timing.name());
    }

    private static ModelFixture trivariateModel(int nobs, MissingPattern missingPattern) {
        ModelFixture model = new ModelFixture(3, 2, 2, nobs);
        double[] design = {0.5, 0.2, 0.0, 0.8, 1.0, -0.5};
        double[] obsCov = {0.2, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.5};
        double[] transition = {0.4, 0.5, 1.0, 0.0};
        double[] selection = {1.0, 0.0, 0.0, 1.0};
        double[] stateCov = {2.0, 0.0, 0.0, 1.0};
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{1.0 + 0.2 * t, -0.3 + 0.1 * t, 0.6 - 0.15 * t}, t);
            if (missingPattern == MissingPattern.PARTIAL && t >= 8 && t < 24) {
                model.setMissing(new boolean[]{true, true, false}, t);
            } else if (missingPattern == MissingPattern.ALL && t >= 8 && t < 24) {
                model.setMissing(new boolean[]{true, true, true}, t);
            }
        }
        return model;
    }

    private static ModelFixture singleFactorDfmMissingModel() {
        int nobs = 100;
        ModelFixture model = new ModelFixture(3, 1, 1, nobs);
        Random random = new Random(1234);
        double[] design = {0.5, -0.8, 1.2};
        double[] obsCov = {0.6, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.9};
        double[] transition = {0.72};
        double[] selection = {1.0};
        double[] stateCov = {1.0};
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{random.nextGaussian(), random.nextGaussian(), random.nextGaussian()}, t);
            if (t == 0) {
                model.setMissing(new boolean[]{true, false, false}, t);
            }
        }
        return model;
    }

    private static InitialState approximateDiffuse() {
        return InitialState.approximateDiffuse(2);
    }

    private static FilterOptions filterOptions(FilterMethod method, FilterOptions.Timing timing) {
        return FilterOptions.builder()
            .method(method)
            .timing(timing)
            .retainAll()
            .build();
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

    private enum MissingPattern {
        NONE,
        PARTIAL,
        ALL
    }
}