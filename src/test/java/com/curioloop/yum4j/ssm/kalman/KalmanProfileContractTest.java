package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherEngine;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.FilterMethod;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherMethod;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KalmanProfileContractTest {

    private static final double TOL = 1e-12;

    private static FilterOptions conventionalFilterOptions() {
        return SmootherOptions.conventional().requiredFilterOptions();
    }

    private static FilterOptions classicalFilterOptions() {
        return SmootherOptions.classical().requiredFilterOptions();
    }

    private static FilterOptions noForecastOptions() {
        return FilterOptions.builder()
            .drop(FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE,
                FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE)
            .build();
    }

    private static FilterOptions noPredictedOptions() {
        return FilterOptions.builder()
            .drop(FilterOptions.Surface.PREDICTED_STATE,
                FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
                FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)
            .build();
    }

    private static FilterOptions noFilteredOptions() {
        return FilterOptions.builder()
            .drop(FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE)
            .build();
    }

    private static ModelFixture buildScalarModel() {
        ModelFixture rep = new ModelFixture(1, 1, 1, 3);
        for (int t = 0; t < rep.nobs; t++) {
            rep.setDesign(new double[]{1.0}, t);
            rep.setObsIntercept(new double[]{0.0}, t);
            rep.setObsCov(new double[]{1.0}, t);
            rep.setTransition(new double[]{1.0}, t);
            rep.setStateIntercept(new double[]{0.0}, t);
            rep.setSelection(new double[]{1.0}, t);
            rep.setStateCov(new double[]{0.5}, t);
            rep.setEndog(new double[]{t + 1.0}, t);
        }
        return rep;
    }

    private static ModelFixture buildMissingScalarModel() {
        ModelFixture rep = buildScalarModel();
        rep.setMissing(new boolean[]{true}, 1);
        return rep;
    }

    private static SmootherOptions methodOptions(SmootherMethod method) {
        return switch (method) {
            case CONVENTIONAL -> SmootherOptions.conventional();
            case CLASSICAL -> SmootherOptions.classical();
            case ALTERNATIVE -> SmootherOptions.alternative();
            case UNIVARIATE -> SmootherOptions.univariate();
        };
    }

    private static SmootherOptions surfaceProfile(SmootherOptions base, int profile) {
        return switch (profile) {
            case 0 -> base.only(SmootherOptions.Surface.STATE);
            case 1 -> base.only(SmootherOptions.Surface.DISTURBANCE);
            case 2 -> base.withoutCovariances();
            case 3 -> base;
            default -> throw new IllegalArgumentException("Unknown smoother surface profile: " + profile);
        };
    }

    private static boolean includesAllCoreSmootherSurfaces(SmootherOptions options) {
        return options.includes(SmootherOptions.Surface.STATE)
            && options.includes(SmootherOptions.Surface.STATE_COVARIANCE)
            && options.includes(SmootherOptions.Surface.DISTURBANCE)
            && options.includes(SmootherOptions.Surface.DISTURBANCE_COVARIANCE);
    }

    private static void assertSmootherSurfaceLengths(ModelFixture rep,
                                                     SmootherOptions options,
                                                     SmootherResult result) {
        assertEquals(options.includes(SmootherOptions.Surface.STATE) ? rep.kStates * rep.nobs : 0,
            result.smoothedStateLength(), options.method() + " smoothed state length");
        assertEquals(options.includes(SmootherOptions.Surface.STATE_COVARIANCE) ? rep.kStates * rep.kStates * rep.nobs : 0,
            result.smoothedStateCovLength(), options.method() + " smoothed state covariance length");
        assertEquals(0, result.smoothedStateAutocovarianceLength(), options.method() + " state autocovariance length");
        assertEquals(options.includes(SmootherOptions.Surface.DISTURBANCE) ? rep.kEndog * rep.nobs : 0,
            result.smoothedObsDisturbanceLength(), options.method() + " observation disturbance length");
        assertEquals(options.includes(SmootherOptions.Surface.DISTURBANCE) ? rep.kPosdef * rep.nobs : 0,
            result.smoothedStateDisturbanceLength(), options.method() + " state disturbance length");
        assertEquals(options.includes(SmootherOptions.Surface.DISTURBANCE_COVARIANCE) ? rep.kEndog * rep.kEndog * rep.nobs : 0,
            result.smoothedObsDisturbanceCovLength(), options.method() + " observation disturbance covariance length");
        assertEquals(options.includes(SmootherOptions.Surface.DISTURBANCE_COVARIANCE) ? rep.kPosdef * rep.kPosdef * rep.nobs : 0,
            result.smoothedStateDisturbanceCovLength(), options.method() + " state disturbance covariance length");

        if (includesAllCoreSmootherSurfaces(options)) {
            assertTrue(result.scaledSmoothedEstimatorLength() > 0, options.method() + " scaled estimator length");
            assertTrue(result.scaledSmoothedEstimatorCovLength() > 0, options.method() + " scaled estimator covariance length");
            assertTrue(result.smoothingErrorLength() > 0, options.method() + " smoothing error length");
            assertTrue(result.innovationsTransitionLength() > 0, options.method() + " innovations transition length");
        } else {
            assertEquals(0, result.scaledSmoothedEstimatorLength(), options.method() + " scaled estimator length");
            assertEquals(0, result.scaledSmoothedEstimatorCovLength(), options.method() + " scaled estimator covariance length");
            assertEquals(0, result.smoothingErrorLength(), options.method() + " smoothing error length");
            assertEquals(0, result.innovationsTransitionLength(), options.method() + " innovations transition length");
        }
    }

    @Test
    void testNamedFilterProfilesMatchExpectedStorageContracts() {
        FilterOptions compact = FilterOptions.compact();
        FilterOptions conventional = conventionalFilterOptions();
        FilterOptions classical = classicalFilterOptions();
        FilterOptions storeAll = FilterOptions.storeAll();

        assertEquals(FilterOptions.builder().retainAll().build(), storeAll);
        assertTrue(storeAll.stores(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE));
        assertTrue(storeAll.stores(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE));
        assertTrue(storeAll.stores(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE));
        assertTrue(storeAll.stores(FilterOptions.Surface.KALMAN_GAIN));
        assertTrue(storeAll.stores(FilterOptions.Surface.LIKELIHOOD));

        assertFalse(compact.stores(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE));
        assertFalse(compact.stores(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE));
        assertFalse(compact.stores(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE));
        assertFalse(compact.stores(FilterOptions.Surface.KALMAN_GAIN));
        assertFalse(compact.stores(FilterOptions.Surface.LIKELIHOOD));

        assertFalse(noForecastOptions().stores(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE));
        assertFalse(noPredictedOptions().stores(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE));
        assertFalse(noFilteredOptions().stores(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE));
        assertFalse(FilterOptions.builder().drop(FilterOptions.Surface.KALMAN_GAIN).build().stores(FilterOptions.Surface.KALMAN_GAIN));
        assertFalse(FilterOptions.builder().drop(FilterOptions.Surface.LIKELIHOOD).build().stores(FilterOptions.Surface.LIKELIHOOD));
        assertFalse(FilterOptions.builder().drop(FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR).build().stores(FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR));

        assertTrue(conventional.stores(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE));
        assertTrue(conventional.stores(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE));
        assertFalse(conventional.stores(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE));
        assertTrue(conventional.stores(FilterOptions.Surface.KALMAN_GAIN));
        assertFalse(conventional.stores(FilterOptions.Surface.LIKELIHOOD));

        assertTrue(classical.stores(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE));
        assertTrue(classical.stores(FilterOptions.Surface.KALMAN_GAIN));
        assertFalse(classical.stores(FilterOptions.Surface.LIKELIHOOD));
    }

    @Test
    void testSmootherProfilesExposeRequiredFilterContracts() {
        assertEquals(conventionalFilterOptions(), SmootherOptions.conventional().requiredFilterOptions());
        assertEquals(classicalFilterOptions(), SmootherOptions.classical().requiredFilterOptions());
        assertEquals(classicalFilterOptions(), SmootherOptions.alternative().requiredFilterOptions());
        FilterOptions univariateFilter = SmootherOptions.univariate().requiredFilterOptions();
        assertEquals(FilterMethod.UNIVARIATE, univariateFilter.method());
        assertTrue(univariateFilter.stores(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE));
        assertTrue(univariateFilter.stores(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE));
        assertFalse(univariateFilter.stores(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE));
        assertTrue(univariateFilter.stores(FilterOptions.Surface.KALMAN_GAIN));
        assertFalse(SmootherOptions.conventional().requiresFilteredStateHistory());
        assertTrue(SmootherOptions.classical().requiresFilteredStateHistory());
        assertTrue(SmootherOptions.alternative().requiresFilteredStateHistory());
        assertFalse(SmootherOptions.univariate().requiresFilteredStateHistory());
        assertEquals(SmootherMethod.UNIVARIATE, SmootherOptions.univariate().method());
    }

    @Test
    void testUnivariateSmootherProfileRoundTripsThroughBuilder() {
        SmootherOptions stateOnly = SmootherOptions.univariate().only(SmootherOptions.Surface.STATE);
        SmootherOptions rebuilt = stateOnly.toBuilder().build();

        assertEquals(SmootherMethod.UNIVARIATE, stateOnly.method());
        assertEquals(stateOnly, rebuilt);
        assertEquals(stateOnly.hashCode(), rebuilt.hashCode());
        assertTrue(stateOnly.includes(SmootherOptions.Surface.STATE));
        assertFalse(stateOnly.includes(SmootherOptions.Surface.STATE_COVARIANCE));
        assertFalse(stateOnly.includes(SmootherOptions.Surface.STATE_AUTOCOVARIANCE));
        assertFalse(stateOnly.includes(SmootherOptions.Surface.DISTURBANCE));
        assertFalse(stateOnly.includes(SmootherOptions.Surface.DISTURBANCE_COVARIANCE));
    }

    @Test
    void testProfilesExposeRetainedStorageNeeds() {
        FilterOptions conventional = conventionalFilterOptions();
        assertTrue(conventional.stores(FilterOptions.Surface.FORECAST_MEAN));
        assertTrue(conventional.stores(FilterOptions.Surface.FORECAST_ERROR));
        assertTrue(conventional.stores(FilterOptions.Surface.FORECAST_COVARIANCE));
        assertTrue(conventional.stores(FilterOptions.Surface.PREDICTED_STATE));
        assertFalse(conventional.stores(FilterOptions.Surface.FILTERED_STATE));
        assertTrue(conventional.stores(FilterOptions.Surface.KALMAN_GAIN));
        assertFalse(conventional.stores(FilterOptions.Surface.LIKELIHOOD));

        FilterOptions forecastOnly = FilterOptions.defaults().without(
                FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE,
                FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
                FilterOptions.Surface.KALMAN_GAIN,
                FilterOptions.Surface.LIKELIHOOD);
        assertTrue(forecastOnly.stores(FilterOptions.Surface.FORECAST_MEAN));
        assertFalse(forecastOnly.stores(FilterOptions.Surface.PREDICTED_STATE));
        assertFalse(forecastOnly.stores(FilterOptions.Surface.FILTERED_STATE));
        assertFalse(forecastOnly.stores(FilterOptions.Surface.KALMAN_GAIN));
        assertFalse(forecastOnly.stores(FilterOptions.Surface.LIKELIHOOD));

        SmootherOptions conventionalOutput = SmootherOptions.conventional();
        assertTrue(conventionalOutput.includes(SmootherOptions.Surface.STATE));
        assertTrue(conventionalOutput.includes(SmootherOptions.Surface.STATE_COVARIANCE));
        assertFalse(conventionalOutput.includes(SmootherOptions.Surface.STATE_AUTOCOVARIANCE));
        assertTrue(conventionalOutput.includes(SmootherOptions.Surface.DISTURBANCE));
        assertTrue(conventionalOutput.includes(SmootherOptions.Surface.DISTURBANCE_COVARIANCE));

        SmootherOptions autocovOutput = conventionalOutput.with(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);
        assertTrue(autocovOutput.includes(SmootherOptions.Surface.STATE_AUTOCOVARIANCE));
        assertTrue(autocovOutput.toBuilder().retainAll().build().includes(SmootherOptions.Surface.STATE_AUTOCOVARIANCE));

        SmootherOptions leanOutput = SmootherOptions.conventional().withoutCovariances();
        assertTrue(leanOutput.includes(SmootherOptions.Surface.STATE));
        assertFalse(leanOutput.includes(SmootherOptions.Surface.STATE_COVARIANCE));
        assertFalse(leanOutput.includes(SmootherOptions.Surface.STATE_AUTOCOVARIANCE));
        assertTrue(leanOutput.includes(SmootherOptions.Surface.DISTURBANCE));
        assertFalse(leanOutput.includes(SmootherOptions.Surface.DISTURBANCE_COVARIANCE));
    }

    @Test
    void testSmootherMethodSurfaceMatrixRetainsExpectedBanks() {
        ModelFixture[] models = {buildScalarModel(), buildMissingScalarModel()};
        InitialState[] initialStates = {
            InitialState.known(new double[]{0.0}, new double[]{1.0}),
            InitialState.approximateDiffuse(1),
            InitialState.diffuse(1)
        };
        SmootherMethod[] methods = {
            SmootherMethod.CONVENTIONAL,
            SmootherMethod.CLASSICAL,
            SmootherMethod.ALTERNATIVE,
            SmootherMethod.UNIVARIATE
        };

        for (ModelFixture rep : models) {
            for (InitialState init : initialStates) {
                for (SmootherMethod method : methods) {
                    SmootherOptions base = methodOptions(method);
                    for (int profile = 0; profile < 4; profile++) {
                        SmootherOptions options = surfaceProfile(base, profile);
                        SmootherResult result = SmootherEngine.smooth(rep,
                            init,
                            options);

                        assertSmootherSurfaceLengths(rep, options, result);
                    }
                }
            }
        }
    }

    @Test
    void testStateAutocovarianceSurfaceIsOptInAndTrimmedIndependently() {
        ModelFixture rep = buildScalarModel();
        InitialState init = InitialState.known(new double[]{0.0}, new double[]{1.0});
        SmootherOptions options = SmootherOptions.conventional().only(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);
        FilterResult fr = KalmanEngine.filter(rep, init, options.requiredFilterOptions());
        SmootherResult result = SmootherEngine.smooth(rep, fr, options);

        assertEquals(0, result.smoothedStateLength());
        assertEquals(0, result.smoothedStateCovLength());
        assertEquals(rep.nobs, result.smoothedStateAutocovarianceLength());
        assertEquals(0, result.smoothedObsDisturbanceLength());
        assertEquals(0, result.scaledSmoothedEstimatorLength());
        assertEquals(0, result.innovationsTransitionLength());
        assertEquals(result.smoothedStateAutocovariance(0, 0, 1), result.copy().smoothedStateAutocovariance(0, 0, 1), TOL);
    }

    @Test
    void testStateAutocovarianceAdvancedRoutesAreGated() {
        ModelFixture rep = buildScalarModel();
        SmootherOptions classical = SmootherOptions.classical().with(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);
        FilterResult classicalFilter = KalmanEngine.filter(rep, InitialState.known(new double[]{0.0}, new double[]{1.0}),
            classical.requiredFilterOptions());
        UnsupportedOperationException classicalFailure = assertThrows(UnsupportedOperationException.class,
            () -> SmootherEngine.smooth(rep, classicalFilter, classical));
        assertTrue(classicalFailure.getMessage().contains("conventional and univariate real smoothing only"));

        SmootherOptions univariate = SmootherOptions.univariate().only(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);
        FilterResult univariateFilter = KalmanEngine.filterUnivariate(rep, InitialState.known(new double[]{0.0}, new double[]{1.0}),
            univariate.requiredFilterOptions());
        SmootherResult univariateResult = SmootherEngine.smooth(rep, univariateFilter, univariate);
        assertEquals(rep.nobs, univariateResult.smoothedStateAutocovarianceLength());
        assertEquals(0, univariateResult.smoothedStateLength());

        FilterResult diffuseUnivariateFilter = KalmanEngine.filterUnivariate(rep, InitialState.diffuse(),
            univariate.requiredFilterOptions());
        SmootherResult diffuseUnivariateResult = SmootherEngine.smooth(rep, diffuseUnivariateFilter, univariate);
        assertEquals(rep.nobs, diffuseUnivariateResult.smoothedStateAutocovarianceLength());

        SmootherOptions diffuse = SmootherOptions.conventional().only(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);
        FilterResult diffuseFilter = KalmanEngine.filter(rep, InitialState.diffuse(), diffuse.requiredFilterOptions());
        SmootherResult diffuseResult = SmootherEngine.smooth(rep, diffuseFilter, diffuse);
        assertEquals(rep.nobs, diffuseResult.smoothedStateAutocovarianceLength());
        assertArrayEquals(diffuseResult.smoothedStateAutocovariance,
            diffuseUnivariateResult.smoothedStateAutocovariance, TOL);
    }

    @Test
    void testSimulationProfilesMapToExplicitFilterAndSmootherContracts() {
        assertEquals(conventionalFilterOptions(), SimulationSmootherOptions.defaults().requiredFilterOptions());
        assertEquals(SmootherOptions.conventional().withoutCovariances(), SimulationSmootherOptions.defaults().requiredSmootherOptions());
        assertEquals(SimulationSmootherOptions.Method.KFS, SimulationSmootherOptions.defaults().method());
        assertEquals(SmootherOptions.conventional().only(SmootherOptions.Surface.STATE), SimulationSmootherOptions.stateOnly().requiredSmootherOptions());
        assertEquals(SmootherOptions.conventional().only(SmootherOptions.Surface.DISTURBANCE), SimulationSmootherOptions.disturbanceOnly().requiredSmootherOptions());
        assertFalse(SimulationSmootherOptions.defaults().storesGeneratedOutputs());
        assertTrue(SimulationSmootherOptions.defaults().withGeneratedOutputs().storesGeneratedOutputs());
        assertTrue(SimulationSmootherOptions.stateOnly().withGeneratedOutputs().storesGeneratedOutputs());
        assertTrue(SimulationSmootherOptions.disturbanceOnly().withGeneratedOutputs().storesGeneratedOutputs());
    }

    @Test
    void testExplicitFilterOptionssDriveRetainedStorage() {
        ModelFixture rep = buildScalarModel();
        InitialState init = InitialState.approximateDiffuse();

        FilterResult compact = KalmanEngine.filter(rep, init, FilterOptions.compact());
        FilterResult conventional = KalmanEngine.filter(rep, init, conventionalFilterOptions());

        assertEquals(0, compact.predictedStateLength());
        assertEquals(0, compact.forecastLength());
        assertEquals(0, compact.kalmanGainLength());
        assertTrue(conventional.predictedStateLength() > 0);
        assertTrue(conventional.forecastErrorLength() > 0);
        assertTrue(conventional.kalmanGainLength() > 0);
    }

    @Test
    void testFilterStateAndCovarianceSurfacesAreRetainedIndependently() {
        ModelFixture rep = buildScalarModel();
        InitialState init = InitialState.known(new double[]{0.0}, new double[]{1.0});
        FilterResult full = KalmanEngine.filter(rep, init, FilterOptions.defaults());

        FilterResult predictedStateOnly = KalmanEngine.filter(rep, init,
            FilterOptions.builder()
                .retainOnly(FilterOptions.Surface.PREDICTED_STATE)
                .build());
        FilterResult predictedStateOnlyCopy = predictedStateOnly.copy();
        assertTrue(predictedStateOnly.predictedStateLength() > 0);
        assertEquals(0, predictedStateOnly.predictedStateCovLength());
        assertEquals(full.predictedState(0, 0), predictedStateOnly.predictedState(0, 0), TOL);
        assertTrue(predictedStateOnlyCopy.predictedStateLength() > 0);
        assertEquals(0, predictedStateOnlyCopy.predictedStateCovLength());

        FilterResult predictedCovarianceOnly = KalmanEngine.filter(rep, init,
            FilterOptions.builder()
                .retainOnly(FilterOptions.Surface.PREDICTED_STATE_COVARIANCE)
                .build());
        assertEquals(0, predictedCovarianceOnly.predictedStateLength());
        assertTrue(predictedCovarianceOnly.predictedStateCovLength() > 0);
        assertEquals(full.predictedStateCov(0, 0, 0), predictedCovarianceOnly.predictedStateCov(0, 0, 0), TOL);

        FilterResult filteredStateOnly = KalmanEngine.filter(rep, init,
            FilterOptions.builder()
                .retainOnly(FilterOptions.Surface.FILTERED_STATE)
                .build());
        assertTrue(filteredStateOnly.filteredStateLength() > 0);
        assertEquals(0, filteredStateOnly.filteredStateCovLength());
        assertEquals(full.filteredState(0, 0), filteredStateOnly.filteredState(0, 0), TOL);

        FilterResult filteredCovarianceOnly = KalmanEngine.filter(rep, init,
            FilterOptions.builder()
                .retainOnly(FilterOptions.Surface.FILTERED_STATE_COVARIANCE)
                .build());
        assertEquals(0, filteredCovarianceOnly.filteredStateLength());
        assertTrue(filteredCovarianceOnly.filteredStateCovLength() > 0);
        assertEquals(full.filteredStateCov(0, 0, 0), filteredCovarianceOnly.filteredStateCov(0, 0, 0), TOL);
    }

    @Test
    void testExactDiffuseSurfaceRetentionIsIndependent() {
        ModelFixture rep = buildScalarModel();
        InitialState init = InitialState.diffuse();

        FilterResult full = KalmanEngine.filter(rep, init, FilterOptions.defaults());
        FilterResult forecastDiffuseOnly = KalmanEngine.filter(rep, init,
            FilterOptions.builder()
                .retainOnly(FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE)
                .build());
        FilterResult predictedDiffuseOnly = KalmanEngine.filter(rep, init,
            FilterOptions.builder()
                .retainOnly(FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)
                .build());
        FilterResult compact = KalmanEngine.filter(rep, init, FilterOptions.compact());

        assertTrue(full.nobsDiffuse() > 0);
        assertTrue(forecastDiffuseOnly.forecastErrorDiffuseCovLength() > 0);
        assertEquals(0, forecastDiffuseOnly.forecastLength());
        assertEquals(0, forecastDiffuseOnly.forecastErrorLength());
        assertEquals(0, forecastDiffuseOnly.forecastErrorCovLength());
        assertEquals(0, forecastDiffuseOnly.predictedDiffuseStateCovLength());
        assertEquals(full.forecastErrorDiffuseCov(0, 0, 0), forecastDiffuseOnly.forecastErrorDiffuseCov(0, 0, 0), TOL);

        assertTrue(predictedDiffuseOnly.predictedDiffuseStateCovLength() > 0);
    assertEquals(0, predictedDiffuseOnly.predictedStateLength());
    assertEquals(0, predictedDiffuseOnly.predictedStateCovLength());
        assertEquals(0, predictedDiffuseOnly.forecastErrorDiffuseCovLength());
        assertEquals(full.predictedDiffuseStateCov(0, 0, 0), predictedDiffuseOnly.predictedDiffuseStateCov(0, 0, 0), TOL);

        assertEquals(0, compact.predictedDiffuseStateCovLength());
        assertEquals(0, compact.forecastErrorDiffuseCovLength());
    }

    @Test
    void testExplicitSimulationSpecsControlGeneratedOutputs() {
        ModelFixture rep = buildScalarModel();
        InitialState init = InitialState.approximateDiffuse();
        double[] measurement = new double[rep.nobs * rep.kEndog];
        double[] state = new double[rep.nobs * rep.kPosdef];
        double[] initial = new double[rep.kStates];

        SimulationSmootherResult lean = SimulationSmootherEngine.simulate(rep, init,
                measurement, state, initial, SimulationSmootherOptions.defaults());
        SimulationSmootherResult generated = SimulationSmootherEngine.simulate(rep, init,
                measurement, state, initial, SimulationSmootherOptions.defaults().withGeneratedOutputs());

        assertEquals(0, lean.generatedObs.length);
        assertArrayEquals(lean.simulatedState, generated.simulatedState, TOL);
        assertArrayEquals(lean.simulatedMeasurementDisturbance, generated.simulatedMeasurementDisturbance, TOL);
        assertTrue(generated.generatedObs.length > 0);
        assertTrue(generated.generatedState.length > 0);
    }
}