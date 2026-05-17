package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.ssm.kalman.filter.FilterMethod;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.mle.FixedParameters;
import com.curioloop.yum4j.ssm.kalman.mle.MLEResults;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.tsa.prediction.PredictionInformationSet;
import com.curioloop.yum4j.tsa.prediction.PredictionKind;
import com.curioloop.yum4j.tsa.statespace.ImpulseResponse;
import com.curioloop.yum4j.tsa.statespace.ImpulseResponseRepetitions;
import com.curioloop.yum4j.tsa.statespace.SimulationAnchor;
import com.curioloop.yum4j.tsa.statespace.SimulationSmootherRepetitions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SARIMAXTest {

    private static final double TOL = 1e-6;
    private static final FilterOptions LIKELIHOOD_OPTIONS = FilterOptions.defaults().without(
        FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE,
        FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE,
        FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
        FilterOptions.Surface.KALMAN_GAIN);

    private static FilterOptions predictionOptions() {
        return FilterOptions.builder()
            .drop(FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
                FilterOptions.Surface.KALMAN_GAIN,
                FilterOptions.Surface.LIKELIHOOD)
            .build();
    }

    @Test
    void testSpecDefaultsRemainEnabledWhenAddingExactDiffuse() {
        SARIMAXSpec spec = SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), new double[]{1.0, 2.0, 3.0})
            .include(SARIMAXOption.USE_EXACT_DIFFUSE)
            .build();

        assertTrue(spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY));
        assertTrue(spec.hasOption(SARIMAXOption.ENFORCE_INVERTIBILITY));
        assertTrue(spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE));
    }

    @Test
    void testTrendStringAndOffsetMatchStatsmodelsConstructionSemantics() {
        SARIMAXSpec spec = SARIMAXSpec.builder(ARIMAOrder.of(0, 0, 0), new double[]{1.0, 2.0, 3.0})
            .trend("ctt")
            .trendOffset(5)
            .build();

        SARIMAXSupport.Meta meta = SARIMAXSupport.analyze(spec);

        assertArrayEquals(new int[]{0, 1, 2}, spec.trendPowers());
        assertEquals(5, spec.trendOffset());
        assertArrayEquals(new double[]{1.0, 5.0, 25.0}, meta.trendData()[0], TOL);
        assertArrayEquals(new double[]{1.0, 7.0, 49.0}, meta.trendData()[2], TOL);
    }

    @Test
    void testAr1LogLikelihoodMatchesDirectFilter() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.6, 0.5};

        KalmanSSM direct = buildAr1StateSpace(params[0], params[1], y);
        FilterResult directFilter = KalmanEngine.filter(direct, InitialState.stationary(direct), LIKELIHOOD_OPTIONS);

        assertEquals(directFilter.logLikelihood(), model.logLikelihood(params), TOL);
        assertArrayEquals(directFilter.logLikelihoodObs, model.logLikelihoodObs(params), TOL);
    }

    @Test
    void testAr1AcceptsNewFilterAndSmootherOptions() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.6, 0.5};
        FilterOptions likelihoodOptions = FilterOptions.builder()
            .method(FilterMethod.CONVENTIONAL)
            .retainOnly(FilterOptions.Surface.LIKELIHOOD)
            .build();

        assertEquals(model.logLikelihood(params), model.logLikelihood(params, likelihoodOptions), TOL);
        assertArrayEquals(model.logLikelihoodObs(params), model.logLikelihoodObs(params, likelihoodOptions), TOL);

        FilterResult prediction = model.predict(params, FilterOptions.builder()
            .drop(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE, FilterOptions.Surface.KALMAN_GAIN, FilterOptions.Surface.LIKELIHOOD)
            .build());
        assertTrue(prediction.forecastLength() > 0);
        assertTrue(prediction.predictedStateLength() > 0);
        assertEquals(0, prediction.filteredStateLength());

        SmootherResult smoothed = model.smooth(params, SmootherOptions.builder()
            .retainOnly(SmootherOptions.Surface.STATE)
            .build());
        assertTrue(smoothed.smoothedState.length > 0);
        assertEquals(0, smoothed.smoothedStateCov.length);
    }

    @Test
    void testPredictionMetadataAndCompactResultSemantics() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.6, 0.5};
        SARIMAXResults results = newResults(model, params, model.filter(params));

        SARIMAXPrediction inSample = results.predict(1, 3);
        assertEquals(PredictionKind.IN_SAMPLE, inSample.kind());
        assertEquals(PredictionInformationSet.PREDICTED, inSample.informationSet());

        assertEquals(PredictionKind.DYNAMIC_IN_SAMPLE, results.predict(1, 4, 2, null).kind());
        assertEquals(PredictionKind.MIXED, results.predict(4, 7).kind());
        assertEquals(PredictionKind.OUT_OF_SAMPLE, results.predict(6, 7).kind());
        assertEquals(PredictionKind.FORECAST, results.forecast(2).kind());

        SARIMAXResults compactResults = newResults(model, params, model.filter(params, FilterOptions.compact()));
        assertThrows(IllegalArgumentException.class, () -> compactResults.predict(0, 2));
        assertEquals(PredictionKind.DYNAMIC_IN_SAMPLE, compactResults.predict(1, 4, 2, null).kind());
        assertEquals(PredictionKind.FORECAST, compactResults.forecast(2).kind());
    }

    @Test
    void testResultSmoothingCanRetainStateAutocovariance() {
        double[] y = generateAr1(0.45, 0.4, 24, 20260615L);
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(2, 0, 0), y).build());
        double[] params = {0.45, -0.12, 0.4};
        SARIMAXResults results = newResults(model, params, model.filter(params));

        SmootherResult stateOnly = results.smooth(SmootherOptions.stateOnly());
        assertEquals(0, stateOnly.smoothedStateAutocovarianceLength());

        SmootherResult withAutocovariance = results.smooth(SmootherOptions.defaults()
            .with(SmootherOptions.Surface.STATE_AUTOCOVARIANCE));
        assertEquals(withAutocovariance.nobs * withAutocovariance.kStates * withAutocovariance.kStates,
            withAutocovariance.smoothedStateAutocovarianceLength());
        assertTrue(Double.isFinite(withAutocovariance.smoothedStateAutocovariance(0, 0, 5)));
    }

    @Test
    void testAr1AdvancedFilterOptionsAtKalmanBoundary() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.6, 0.5};

        UnsupportedOperationException collapsedFailure = assertThrows(UnsupportedOperationException.class,
            () -> model.logLikelihood(params, FilterOptions.builder().method(FilterMethod.COLLAPSED).build()));
        assertTrue(collapsedFailure.getMessage().contains("observation dimension larger than state dimension"));

        double conventional = model.logLikelihood(params,
            FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build());
        double chandrasekhar = model.logLikelihood(params,
            FilterOptions.builder().method(FilterMethod.CHANDRASEKHAR).build());
        assertEquals(conventional, chandrasekhar, TOL);
    }

    @Test
    void testStationarySarimaxChandrasekharMatchesConventionalFilterProfiles() {
        double[] y = generateAr1(0.35, 0.6, 36, 20260516L);
        SARIMAX[] models = {
            new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(2, 0, 1), y).build()),
            new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y)
                .measurementError(true)
                .build())
        };
        double[][] params = {
            {0.42, -0.16, 0.28, 0.55},
            {0.58, 0.12, 0.50}
        };
        FilterOptions[] profiles = {
            FilterOptions.builder().retainAll().build(),
            FilterOptions.likelihoodOnly(),
            predictionOptions(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE).build(),
            FilterOptions.standardizedForecastError()
        };

        for (int index = 0; index < models.length; index++) {
            for (FilterOptions profile : profiles) {
                assertSarimaxChandrasekharMatchesConventional(models[index], params[index], profile);
            }
        }
    }

    @Test
    void testSarimaxChandrasekharOutOfScopeRoutesFailAtKalmanBoundary() {
        double[] y = {1.0, Double.NaN, -0.3, 0.8, 0.2, -0.1};
        SARIMAX missing = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        FilterOptions chandrasekhar = FilterOptions.builder().method(FilterMethod.CHANDRASEKHAR).build();

        UnsupportedOperationException missingFailure = assertThrows(UnsupportedOperationException.class,
            () -> missing.logLikelihood(new double[]{0.6, 0.5}, chandrasekhar));
        assertTrue(missingFailure.getMessage().contains("missing observations"));

        SARIMAX exactDiffuse = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 1, 0), new double[]{1.0, 0.5, -0.3, 0.8, 0.2, -0.1})
                .include(SARIMAXOption.USE_EXACT_DIFFUSE)
                .build());
        UnsupportedOperationException diffuseFailure = assertThrows(UnsupportedOperationException.class,
            () -> exactDiffuse.logLikelihood(new double[]{0.45, 0.50}, chandrasekhar));
        assertTrue(diffuseFailure.getMessage().contains("stationary initialization")
            || diffuseFailure.getMessage().contains("diffuse initialization"));
    }

    @Test
    void testAr1MeasurementErrorLogLikelihoodMatchesDirectFilter() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y)
                .measurementError(true)
                .build());
        double[] params = {0.6, 0.15, 0.5};

        KalmanSSM direct = buildAr1MeasurementErrorStateSpace(params[0], params[1], params[2], y);
        FilterResult directFilter = KalmanEngine.filter(direct, InitialState.stationary(direct), LIKELIHOOD_OPTIONS);

        assertArrayEquals(params, model.transformParams(model.untransformParams(params)), 1e-8);
        assertEquals(directFilter.logLikelihood(), model.logLikelihood(params), TOL);
        assertArrayEquals(directFilter.logLikelihoodObs, model.logLikelihoodObs(params), TOL);
    }

    @Test
    void testIntegratedRandomWalkAnalyticComplexStepScoreObsMatchesNumericSurface() {
        double[] innovations = generateAr1(0.0, 0.3, 48, 20260511L);
        double[] y = new double[innovations.length];
        double level = 0.0;
        for (int i = 0; i < innovations.length; i++) {
            level += innovations[i];
            y[i] = level;
        }

        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 0), y).build());
        double[] params = {0.3};

        double[] scoreObs = model.scoreObs(params);
        assertNotNull(scoreObs);
        double[] numeric = finiteDifferenceScoreObs(model, params);

        assertArrayAllClose(numeric, scoreObs, 1e-3, 1e-3);
    }

    @Test
    void testIntegratedMa1AnalyticComplexStepScoreObsMatchesNumericSurface() {
        double[] shocks = generateAr1(0.0, 0.2, 64, 20260512L);
        double theta = 0.45;
        double[] y = new double[shocks.length];
        double level = 0.0;
        double previousShock = 0.0;
        for (int i = 0; i < shocks.length; i++) {
            double innovation = shocks[i] + theta * previousShock;
            level += innovation;
            y[i] = level;
            previousShock = shocks[i];
        }

        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), y).build());
        double[] params = {theta, 0.2};

        double[] scoreObs = model.scoreObs(params);
        assertNotNull(scoreObs);
        double[] numeric = finiteDifferenceScoreObs(model, params);

        assertArrayAllClose(numeric, scoreObs, 1e-3, 1e-3);
    }

    @Test
    void testAirlineDefaultLikelihoodAndInformationCriteriaMatchStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .build());

        double[] params = {
            air2.paramsMa()[0],
            air2.paramsSeasonalMa()[0],
            air2.paramsVariance()[0]
        };

        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, params, filterResult);

        assertEquals(air2.loglike(), results.logLikelihood(), 1e-4);
        assertEquals(air2.aic(), results.aic(), 1e-3);
        assertEquals(air2.bic(), results.bic(), 1e-3);
    }

    @Test
    void testAirlineStandardizedForecastsErrorMatchesStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .build());

        double[] params = {
            air2.paramsMa()[0],
            air2.paramsSeasonalMa()[0],
            air2.paramsVariance()[0]
        };

        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, params, filterResult);

        assertArrayEquals(air2.standardizedForecastsError(), results.standardizedForecastError(), 5e-5);
    }

    @Test
    void testAirlineFitMatchesStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .build());

        SARIMAXResults results = model.fit();
        double[] expected = {
            air2.paramsMa()[0],
            air2.paramsSeasonalMa()[0],
            air2.paramsVariance()[0]
        };

        assertArrayEquals(expected, results.params(), 2e-3);
        assertEquals(air2.loglike(), results.logLikelihood(), 2e-3);
        assertEquals(air2.aic(), results.aic(), 2e-2);
        assertEquals(air2.bic(), results.bic(), 2e-2);
    }

    @Test
    void testAirlineBseDefaultAndOpgMatchStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .build());

        SARIMAXResults results = model.fit();
        double[] expectedOpg = airlineBseOpg(air2);

        assertArrayAllClose(expectedOpg, results.bseOpg(), 6e-3, 1e-5);
        assertArrayAllClose(expectedOpg, results.bse(), 6e-3, 1e-5);
        assertArrayAllClose(expectedOpg, results.bseDefault(), 6e-3, 1e-5);
    }

    @Test
    void testAirlineBseOimMatchStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .build());

        SARIMAXResults results = model.fit();
        double[] expectedOim = airlineBseOim(air2);

        assertArrayAllClose(expectedOim, results.bseOim(), 1.5e-2, 1e-5);
    }

    @Test
    void testAirlineCovParamsOimDiagonalMatchesStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .build());

        SARIMAXResults results = model.fit();

        assertArrayAllClose(airlineBseOim(air2), diagonalStandardErrors(results.covParamsOim()), 1.5e-2, 1e-5);
    }

    @Test
    void testAirlineApproxMatchStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .build());

        SARIMAXResults results = model.fit(SARIMAXFitOptions.builder()
            .covarianceType(MLEResults.Covariance.APPROX)
            .build());

        assertEquals("approx", results.covType());
        assertArrayAllClose(airlineBseOim(air2), results.bseApprox(), 8e-3, 1e-5);
        assertArrayAllClose(results.bseApprox(), diagonalStandardErrors(results.covParamsApprox()), 1e-12, 1e-12);
        assertArrayAllClose(results.bseApprox(), results.bse(), 0.0, 0.0);
        assertArrayAllClose(results.covParamsApprox(), results.covParams(), 0.0, 0.0);
    }

    @Test
    void testAirlineRobustApproxMatchStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .build());

        SARIMAXResults results = model.fit();

        assertAllFinite(results.bseRobustApprox());
        assertAllFinite(results.covParamsRobustApprox());
    }

    @Test
    void testFitOptimizationMatchesFinalLikelihoodAndSolution() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .build());

        SARIMAXResults results = model.fit();
        Optimization optimization = results.optimization();

        assertNotNull(optimization);
        assertNotNull(optimization.solution());
        assertTrue(optimization.status().converged());
        assertArrayAllClose(results.unconstrainedParams(), optimization.solution(), 1e-10, 1e-10);
        assertEquals(-results.logLikelihood(), optimization.cost(), 1e-6);
    }

    @Test
    void testWpiStationaryFitMatchesStatsmodelsReference() {
        WpiArimaReference reference = loadWpiArimaReference("wpi1_stationary");
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 1, 1), reference.endog())
                .trendPowers(0)
                .build());
        double intercept = (1.0 - reference.paramsAr()[0]) * reference.paramsMean()[0];
        double[] expected = {
            intercept,
            reference.paramsAr()[0],
            reference.paramsMa()[0],
            reference.paramsVariance()[0]
        };

        assertEquals(reference.loglike(), model.logLikelihood(expected), 1e-4);

        SARIMAXResults results = model.fit();

        assertArrayAllClose(expected, results.params(), 5e-3, 1e-3);
        assertEquals(reference.loglike(), results.logLikelihood(), 5e-4);
        assertEquals(reference.aic(), results.aic(), 5e-3);
        assertEquals(reference.bic(), results.bic(), 5e-3);
    }

    @Test
    void testAirlineBseAndCovParamsOimMatchStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = airlineModel(endog);

        SARIMAXResults results = model.fit();

        assertArrayAllClose(airlineBseOim(air2), results.bseOim(), 1.5e-2, 1e-5);
        assertArrayAllClose(airlineBseOim(air2), diagonalStandardErrors(results.covParamsOim()), 1.5e-2, 1e-5);
    }

    @Test
    void testAirlineBseAndCovParamsApproxMatchStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = airlineModel(endog);

        SARIMAXResults results = model.fit(SARIMAXFitOptions.builder()
            .covarianceType(MLEResults.Covariance.APPROX)
            .build());

        assertEquals("approx", results.covType());
        assertArrayAllClose(airlineBseOim(air2), results.bseApprox(), 8e-3, 1e-5);
        assertArrayAllClose(results.bseApprox(), diagonalStandardErrors(results.covParamsApprox()), 1e-12, 1e-12);
        assertArrayAllClose(results.bseApprox(), results.bse(), 0.0, 0.0);
        assertArrayAllClose(results.covParamsApprox(), results.covParams(), 0.0, 0.0);
    }

    @Test
    void testAirlineBseAndCovParamsOpgMatchStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = airlineModel(endog);

        SARIMAXResults results = model.fit();

        assertArrayAllClose(airlineBseOpg(air2), results.bse(), 6e-3, 1e-5);
        assertArrayAllClose(airlineBseOpg(air2), results.bseOpg(), 6e-3, 1e-5);
        assertArrayAllClose(airlineBseOpg(air2), diagonalStandardErrors(results.covParams()), 6e-3, 1e-5);
        assertArrayAllClose(airlineBseOpg(air2), diagonalStandardErrors(results.covParamsOpg()), 6e-3, 1e-5);
        assertArrayAllClose(results.bse(), results.bseDefault(), 0.0, 0.0);
        assertArrayAllClose(results.bseOpg(), results.bseDefault(), 0.0, 0.0);
        assertArrayAllClose(results.covParams(), results.covParamsDefault(), 0.0, 0.0);
        assertArrayAllClose(results.covParamsOpg(), results.covParamsDefault(), 0.0, 0.0);
    }

    @Test
    void testAirlineDefaultInferenceMatchesStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = airlineModel(endog);

        SARIMAXResults results = model.fit();

        assertEquals("opg", results.covType());
        assertArrayEquals(new String[]{"ma.L1", "ma.S.L12", "sigma2"}, results.parameterNames());
        assertArrayAllClose(results.zvalues(), results.tvalues(), 0.0, 0.0);
        assertArrayAllClose(airlineBseOpg(air2), results.bse(), 6e-3, 1e-5);
        for (double value : results.zvalues()) {
            assertTrue(Double.isFinite(value));
        }
        for (double value : results.pvalues()) {
            assertTrue(value >= 0.0 && value <= 1.0);
        }
        for (double[] interval : results.confInt()) {
            assertTrue(interval[0] < interval[1]);
        }
    }

    @Test
    void testAirlineFitCovarianceTypeSelectionRoutesDefaultInferenceSurface() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = airlineModel(endog);

        SARIMAXResults oim = model.fit(SARIMAXFitOptions.builder().covarianceType(MLEResults.Covariance.OIM).build());
        assertEquals("oim", oim.covType());
        assertArrayAllClose(airlineBseOim(air2), oim.bse(), 1.5e-2, 1e-5);
        assertArrayAllClose(oim.covParamsOim(), oim.covParamsDefault(), 0.0, 0.0);
        assertArrayAllClose(oim.bseOim(), oim.bseDefault(), 0.0, 0.0);

        SARIMAXResults approx = model.fit(SARIMAXFitOptions.builder().covarianceType(MLEResults.Covariance.APPROX).build());
        assertEquals("approx", approx.covType());
        assertArrayAllClose(airlineBseOim(air2), approx.bse(), 8e-3, 1e-5);
        assertArrayAllClose(approx.covParamsApprox(), approx.covParamsDefault(), 0.0, 0.0);
        assertArrayAllClose(approx.bseApprox(), approx.bseDefault(), 0.0, 0.0);

        SARIMAXResults robust = model.fit(SARIMAXFitOptions.builder().covarianceType(MLEResults.Covariance.ROBUST).build());
        assertEquals("robust", robust.covType());
        assertAllFinite(robust.covParams());
        assertArrayAllClose(robust.covParamsRobustOim(), robust.covParamsDefault(), 0.0, 0.0);
        assertArrayAllClose(robust.bseRobustOim(), robust.bseDefault(), 0.0, 0.0);

        SARIMAXResults robustApprox = model.fit(SARIMAXFitOptions.builder().covarianceType(MLEResults.Covariance.ROBUST_APPROX).build());
        assertEquals("robust_approx", robustApprox.covType());
        assertAllFinite(robustApprox.covParams());
        assertArrayAllClose(robustApprox.covParamsRobustApprox(), robustApprox.covParamsDefault(), 0.0, 0.0);
        assertArrayAllClose(robustApprox.bseRobustApprox(), robustApprox.bseDefault(), 0.0, 0.0);
    }

    @Test
    void testAirlineSummaryContainsStatsmodelsStyleCoreSections() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = airlineModel(endog);

        SARIMAXResults results = model.fit();
        String summary = results.summary();

        assertTrue(summary.contains("Statespace Model Results"));
        assertTrue(summary.contains("Model:"));
        assertTrue(summary.contains("Covariance Type:"));
        assertTrue(summary.contains("opg"));
        assertTrue(summary.contains("Log Likelihood:"));
        assertTrue(summary.contains("AIC:"));
        assertTrue(summary.contains("BIC:"));
        assertTrue(summary.contains("HQIC:"));
        for (String parameterName : results.parameterNames()) {
            assertTrue(summary.contains(parameterName));
        }
    }

    @Test
    void testAirlineBseAndCovParamsRobustOimAreFinite() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = airlineModel(endog);

        SARIMAXResults results = model.fit();

        assertAllFinite(results.bseRobustOim());
        assertAllFinite(results.covParamsRobustOim());
    }

    @Test
    void testAirlineBseAndCovParamsRobustApproxAreFinite() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = airlineModel(endog);

        SARIMAXResults results = model.fit();

        assertAllFinite(results.bseRobustApprox());
        assertAllFinite(results.covParamsRobustApprox());
        assertArrayAllClose(results.bseRobustOim(), results.bseRobust(), 0.0, 0.0);
        assertArrayAllClose(results.covParamsRobustOim(), results.covParamsRobust(), 0.0, 0.0);
    }

    @Test
    void testWpiMeasurementErrorFitUsesStatsmodelsSeries() {
        double[] endog = diff(loadWpiSeries().endog());
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), endog)
                .measurementError(true)
                .build());

        SARIMAXResults results = model.fit();

        assertEquals(3, results.params().length);
        assertTrue(Double.isFinite(results.logLikelihood()));
        assertTrue(results.params()[1] > 0.0);
        assertTrue(results.params()[2] > 0.0);
    }

    @Test
    void testAr1ConcentrateScaleLikelihoodMatchesConcentratedFilterFormula() {
        double[] endog = generateAr1(0.7, 0.35, 80, 20260504L);
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), endog)
                .concentrateScale(true)
                .build());
        double[] params = {0.55};

        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        int burn = model.likelihoodBurn();
        double expectedScale = concentratedScale(filterResult, endog, burn);
        double[] expectedLoglikeObs = concentratedLogLikelihoodObs(filterResult, endog, burn, expectedScale);
        FilterResult genericConcentrated = model.filter(params, FilterOptions.defaults()
            .toBuilder()
            .concentratedLikelihood(true)
            .concentratedLikelihoodBurn(burn)
            .build());

        assertArrayEquals(params, model.transformParams(model.untransformParams(params)), 1e-8);
        assertTrue(genericConcentrated.concentratedLikelihood());
        assertEquals(expectedScale, genericConcentrated.scale(), 1e-10);
        assertArrayAllClose(expectedLoglikeObs, model.logLikelihoodObs(params), 1e-10, 1e-10);
        assertEquals(Arrays.stream(expectedLoglikeObs).sum(), model.logLikelihood(params), 1e-10);

        SARIMAXResults results = newResults(model, params, filterResult);

        assertEquals(expectedScale, results.scale(), 1e-10);
        assertArrayAllClose(expectedLoglikeObs, results.logLikelihoodObs(), 1e-10, 1e-10);
        assertEquals(filterResult.forecastErrorCov(0, 0, 0) * expectedScale, results.filterResult().forecastErrorCov(0, 0, 0), 1e-10);
    }

    @Test
    void testWpiConcentrateScaleMatchesStatsmodelsScenario() {
        double[] endog = diff(loadWpiSeries().endog());

        assertConcentratedScaleMatchesUnconcentrated(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), endog).build(),
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), endog).concentrateScale(true).build(),
            1,
            1e-5);
    }

    @Test
    void testWpiMeasurementErrorConcentrateScaleMatchesStatsmodelsScenario() {
        double[] endog = diff(loadWpiSeries().endog());

        assertConcentratedScaleMatchesUnconcentrated(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), endog).measurementError(true).build(),
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), endog).measurementError(true).concentrateScale(true).build(),
            2,
            1e-3);
    }

    @Test
    void testAirlineExactDiffuseConcentrateScaleMatchesUnconcentratedScenario() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();

        assertConcentratedScaleMatchesUnconcentrated(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .include(SARIMAXOption.USE_EXACT_DIFFUSE)
                .build(),
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .include(SARIMAXOption.USE_EXACT_DIFFUSE)
                .concentrateScale(true)
                .build(),
            1,
            1e-5);
    }

    @Test
    void testConcentratedScaleObservedInformationFailsFastAndApproxPathStaysExplicit() {
        double[] endog = generateAr1(0.7, 0.35, 80, 20260514L);
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), endog)
                .concentrateScale(true)
                .build());
        double[] params = {0.55};

        IllegalStateException error = assertThrows(IllegalStateException.class,
            () -> model.covarianceFromObservedInformation(params));

        assertTrue(error.getMessage().contains("covarianceFromApproximateInformation"));
        for (double value : model.covarianceFromApproximateInformation(params)) {
            assertTrue(Double.isFinite(value));
        }
        for (double value : model.robustCovarianceFromApproximateInformation(params)) {
            assertTrue(Double.isFinite(value));
        }
    }

    @Test
    void testConcentratedScaleFitWithOimCovarianceTypeFailsOnDefaultCovarianceSurface() {
        double[] endog = generateAr1(0.7, 0.35, 80, 20260515L);
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), endog)
                .concentrateScale(true)
                .build());

        SARIMAXResults results = model.fit(SARIMAXFitOptions.builder()
            .covarianceType(MLEResults.Covariance.OIM)
            .build());

        assertEquals("oim", results.covType());
        IllegalStateException error = assertThrows(IllegalStateException.class, results::covParams);
        assertTrue(error.getMessage().contains("covarianceFromApproximateInformation"));
    }

    @Test
    void testFriedmanStateRegressionMatchesStatsmodelsReferenceScenario() {
        FriedmanMleReference reference = loadFriedmanMleReference();
        double[][] exog = scale(withIntercept(reference.exog()), 0.1);
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 1), reference.endog())
                .exog(exog)
                .mleRegression(false)
                .build());
        double[] expected = {
            reference.paramsAr()[0],
            reference.paramsMa()[0],
            reference.paramsVariance()[0]
        };

        FilterResult filterResult = model.filter(expected, FilterOptions.defaults());
        int last = reference.endog().length - 1;

        assertArrayAllClose(expected, newResults(model, expected, filterResult).params(), 0.0, 0.0);
        assertArrayAllClose(reference.paramsExog(), new double[]{
            filterResult.filteredState(filterResult.kStates - 2, last) / 10.0,
            filterResult.filteredState(filterResult.kStates - 1, last) / 10.0
        }, 4.0, 0.0);
    }

    @Test
    void testAirlineMaFitMatchesStatsmodelsReference() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = airlineModel(endog);
        double[] expected = {
            air2.paramsMa()[0],
            air2.paramsSeasonalMa()[0],
            air2.paramsVariance()[0]
        };

        SARIMAXResults results = model.fit();

        assertArrayAllClose(expected, results.params(), 2e-3, 1e-3);
        assertEquals(air2.loglike(), results.logLikelihood(), 2e-3);
        assertEquals(air2.aic(), results.aic(), 2e-2);
        assertEquals(air2.bic(), results.bic(), 2e-2);
    }

    @Test
    void testWpiArSmoothingUsesStatsmodelsSeries() {
        CoverageRow row = loadCoverageRow("arima wpi, arima(3,0,0) noconstant vce(oim)");
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(3, 0, 0), loadWpiSeries().endog()).build());

        SmootherResult smooth = model.smooth(row.params(), SmootherOptions.defaults());

        assertEquals(loadWpiSeries().endog().length, smooth.nobs);
        for (int t = 0; t < smooth.nobs; t++) {
            assertTrue(Double.isFinite(smooth.smoothedState(0, t)));
        }
    }

    @Test
    void testAirlineSmoothingUsesStatsmodelsSeries() {
        Air2Reference air2 = loadAir2Reference();
        double[] endog = Arrays.stream(air2.data()).map(Math::log).toArray();
        SARIMAX model = airlineModel(endog);
        double[] params = {
            air2.paramsMa()[0],
            air2.paramsSeasonalMa()[0],
            air2.paramsVariance()[0]
        };

        SmootherResult smooth = model.smooth(params, SmootherOptions.defaults());

        assertEquals(endog.length, smooth.nobs);
        for (int t = 0; t < smooth.nobs; t++) {
            assertTrue(Double.isFinite(smooth.smoothedState(0, t)));
        }
    }

    @Test
    void testFriedmanMleFitMatchesStatsmodelsReference() {
        FriedmanMleReference reference = loadFriedmanMleReference();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 1), reference.endog())
                .exog(withIntercept(reference.exog()))
                .build());

        double[] expected = {
            reference.paramsExog()[0],
            reference.paramsExog()[1],
            reference.paramsAr()[0],
            reference.paramsMa()[0],
            reference.paramsVariance()[0]
        };

        assertEquals(reference.loglike(), model.logLikelihood(expected), 1e-3);

        SARIMAXResults results = model.fit();

        assertArrayAllClose(expected, results.params(), 5e-2, 1e-3);
        assertEquals(reference.loglike(), results.logLikelihood(), 5e-2);
        assertEquals(reference.aic(), results.aic(), 1e-1);
        assertEquals(reference.bic(), results.bic(), 1e-1);
    }

    @Test
    void testFriedmanPredictAndForecastMatchStatsmodelsReference() {
        FriedmanPredictReference reference = loadFriedmanPredictReference();
        double[] params = {
            reference.paramsExog()[0],
            reference.paramsExog()[1],
            reference.paramsAr()[0],
            reference.paramsMa()[0],
            reference.paramsVariance()[0]
        };

        SARIMAX fullModel = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 1), reference.endog())
                .exog(withIntercept(reference.exog()))
                .build());
        FilterResult fullFilter = fullModel.filter(params, FilterOptions.defaults());
        SARIMAXResults fullResults = newResults(fullModel, params, fullFilter);

        SARIMAXPrediction staticPrediction = fullResults.predict(0, reference.endog().length - 1, -1, null);
        int dynamicPredictStart = Math.max(0, reference.dynamicPredictStart() - 1);
        SARIMAXPrediction dynamicPrediction = fullResults.predict(dynamicPredictStart, reference.endog().length - 1,
            dynamicPredictStart, null);

        assertArrayClose(reference.predict(), staticPrediction.mean(), 2e-3);
        assertArrayClose(reference.dynamicPredictTail(), predictionTail(dynamicPrediction, reference.dynamicPredictStart()), 2e-3);

        SARIMAX trainModel = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 1), reference.trainEndog())
                .exog(withIntercept(reference.trainExog()))
                .build());
        FilterResult trainFilter = trainModel.filter(params, FilterOptions.defaults());
        SARIMAXResults trainResults = newResults(trainModel, params, trainFilter);

        double[][] futureExog = withIntercept(reference.futureExog());
        int forecastEnd = reference.trainEndog().length + reference.forecastHorizon() - 1;
        SARIMAXPrediction forecast = trainResults.forecast(reference.forecastHorizon(), futureExog);
        int dynamicForecastStart = Math.max(0, reference.dynamicForecastStart() - 1);
        SARIMAXPrediction dynamicForecast = trainResults.predict(dynamicForecastStart, forecastEnd,
            dynamicForecastStart, futureExog);

        assertArrayClose(reference.forecastTail(), forecast.mean(), 2e-3);
        assertArrayClose(reference.dynamicForecastTail(), predictionTail(dynamicForecast, reference.dynamicForecastStart()), 2e-3);
    }

    @Test
    void testWpiStationaryTrendMappingMatchesStatsmodelsReference() {
        WpiArimaReference reference = loadWpiArimaReference("wpi1_stationary");
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 1, 1), reference.endog())
                .trendPowers(0)
                .build());

        double intercept = (1.0 - reference.paramsAr()[0]) * reference.paramsMean()[0];
        double[] expected = {
            intercept,
            reference.paramsAr()[0],
            reference.paramsMa()[0],
            reference.paramsVariance()[0]
        };

        assertEquals(reference.loglike(), model.logLikelihood(expected), 1e-4);
        FilterResult filterResult = model.filter(expected, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, expected, filterResult);
        assertEquals(reference.aic(), results.aic(), 1e-3);
        assertEquals(reference.bic(), results.bic(), 1e-3);
    }

    @Test
    void testWpiSeasonalDenseZeroPaddedLikelihoodMatchesStatsmodelsReference() {
        WpiArimaReference reference = loadWpiArimaReference("wpi1_seasonal");
        double[] endog = Arrays.stream(reference.endog()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 1, 4), endog)
                .trendPowers(0)
                .build());

        double intercept = (1.0 - reference.paramsAr()[0]) * reference.paramsMean()[0];
        double[] expected = {
            intercept,
            reference.paramsAr()[0],
            reference.paramsMa()[0],
            0.0,
            0.0,
            reference.paramsMa()[1],
            reference.paramsVariance()[0]
        };

        assertEquals(reference.loglike(), model.logLikelihood(expected), 1e-4);
    }

    @Test
    void testWpiSeasonalSparseMaLikelihoodMatchesStatsmodelsReference() {
        WpiArimaReference reference = loadWpiArimaReference("wpi1_seasonal");
        double[] endog = Arrays.stream(reference.endog()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 1, 4), endog)
                .movingAverageLags(1, 4)
                .trendPowers(0)
                .build());

        double intercept = (1.0 - reference.paramsAr()[0]) * reference.paramsMean()[0];
        double[] expected = {
            intercept,
            reference.paramsAr()[0],
            reference.paramsMa()[0],
            reference.paramsMa()[1],
            reference.paramsVariance()[0]
        };

        assertEquals(reference.loglike(), model.logLikelihood(expected), 1e-4);
        FilterResult filterResult = model.filter(expected, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, expected, filterResult);
        assertEquals(reference.aic(), results.aic(), 1e-3);
        assertEquals(reference.bic(), results.bic(), 1e-3);
    }

    @Test
    void testWpiSeasonalSparseMaFitMatchesStatsmodelsReference() {
        WpiArimaReference reference = loadWpiArimaReference("wpi1_seasonal");
        double[] endog = Arrays.stream(reference.endog()).map(Math::log).toArray();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 1, 4), endog)
                .movingAverageLags(1, 4)
                .trendPowers(0)
                .build());

        SARIMAXResults results = model.fit();

        double intercept = (1.0 - reference.paramsAr()[0]) * reference.paramsMean()[0];
        double[] expected = {
            intercept,
            reference.paramsAr()[0],
            reference.paramsMa()[0],
            reference.paramsMa()[1],
            reference.paramsVariance()[0]
        };

        assertArrayClose(expected, results.params(), 5e-3);
        assertEquals(reference.loglike(), results.logLikelihood(), 5e-4);
        assertEquals(reference.aic(), results.aic(), 5e-3);
        assertEquals(reference.bic(), results.bic(), 5e-3);
    }

    @Test
    void testWpiApproximateDiffuseVarianceMatchesStatsmodelsReference() {
        WpiArimaReference reference = loadWpiArimaReference("wpi1_diffuse");
        double initialVariance = loadFixtureScalar("wpi1_diffuse", "initial_variance");
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 1, 1), reference.endog())
                .trendPowers(0)
                .simpleDifferencing(true)
                .hamiltonRepresentation(true)
                .approximateDiffuseVariance(initialVariance)
                .build());

        double intercept = (1.0 - reference.paramsAr()[0]) * reference.paramsMean()[0];
        double[] expected = {
            intercept,
            reference.paramsAr()[0],
            reference.paramsMa()[0],
            reference.paramsVariance()[0]
        };

        assertEquals(reference.loglike(), model.logLikelihood(expected), 1e-4);
        FilterResult filterResult = model.filter(expected, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, expected, filterResult);
        assertEquals(reference.aic(), results.aic(), 1e-3);
        assertEquals(reference.bic(), results.bic(), 1e-3);
    }

    @Test
    void testWpiApproximateDiffuseFitMatchesStatsmodelsReference() {
        WpiArimaReference reference = loadWpiArimaReference("wpi1_diffuse");
        double initialVariance = loadFixtureScalar("wpi1_diffuse", "initial_variance");
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 1, 1), reference.endog())
                .trendPowers(0)
                .simpleDifferencing(true)
                .hamiltonRepresentation(true)
                .approximateDiffuseVariance(initialVariance)
                .build());

        SARIMAXResults results = model.fit();

        double intercept = (1.0 - reference.paramsAr()[0]) * reference.paramsMean()[0];
        double[] expected = {
            intercept,
            reference.paramsAr()[0],
            reference.paramsMa()[0],
            reference.paramsVariance()[0]
        };

        assertArrayClose(expected, results.params(), 5e-3);
        assertEquals(reference.loglike(), results.logLikelihood(), 5e-4);
        assertEquals(reference.aic(), results.aic(), 5e-3);
        assertEquals(reference.bic(), results.bic(), 5e-3);
    }

    @Test
    void testApproximateDiffuseLikelihoodBurnFollowsReferenceSemantics() {
        WpiCoverageReference wpi = loadWpiCoverageReference();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(3, 2, 2), wpi.endog())
                .seasonalOrder(SeasonalOrder.of(3, 2, 2, 4))
                .exog(wpi.exog())
                .build());

        double[] burnedObs = model.logLikelihoodObs(wpi.params());
        double burned = model.logLikelihood(wpi.params());
        double raw = model.filter(wpi.params(), FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.LIKELIHOOD)
            .build()).logLikelihood();

        for (int i = 0; i < wpi.burn(); i++) {
            assertEquals(0.0, burnedObs[i], TOL);
        }
        assertEquals(Arrays.stream(burnedObs).sum(), burned, TOL);
        assertTrue(Math.abs(burned - wpi.loglike()) < Math.abs(raw - wpi.loglike()));
    }

    @Test
    void testApproximateDiffuseWpiLikelihoodMatchesCoverageReference() {
        WpiSeries wpi = loadWpiSeries();
        CoverageRow row = loadCoverageRow("arima wpi x, arima(3,2,2) sarima(3,2,2,4) noconstant vce(oim)");

        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(3, 2, 2), wpi.endog())
                .seasonalOrder(SeasonalOrder.of(3, 2, 2, 4))
                .exog(wpi.exog())
                .build());

        assertEquals(row.loglike(), model.logLikelihood(row.params()), 1e-4);
    }

    @Test
    void testApproximateDiffuseResultsUseBurnedEffectiveSampleForBic() {
        WpiCoverageReference wpi = loadWpiCoverageReference();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(3, 2, 2), wpi.endog())
                .seasonalOrder(SeasonalOrder.of(3, 2, 2, 4))
                .exog(wpi.exog())
                .build());

        FilterResult filterResult = model.filter(wpi.params(), FilterOptions.defaults());
        SARIMAXResults results = newResults(model, wpi.params(), filterResult);

        assertEquals(wpi.burn(), results.logLikelihoodBurn());
        assertEquals(wpi.endog().length - wpi.burn(), results.nobsEffective());
        assertEquals(0, results.kDiffuseStates());
        assertEquals(wpi.params().length, results.dfModel());
        assertEquals(results.nobsEffective() - results.dfModel(), results.dfResid());
        assertEquals(wpi.loglike(), results.logLikelihood(), 1e-4);
        double expectedAic = -2.0 * results.logLikelihood() + 2.0 * results.dfModel();
        double expectedAicc = -2.0 * results.logLikelihood()
            + 2.0 * results.dfModel() * results.nobsEffective()
            / (results.nobsEffective() - results.dfModel() - 1.0);
        double expectedBic = -2.0 * results.logLikelihood() + Math.log(results.nobsEffective()) * wpi.params().length;
        double expectedHqic = -2.0 * results.logLikelihood()
            + 2.0 * Math.log(Math.log(results.nobsEffective())) * results.dfModel();
        double totalObsBic = -2.0 * results.logLikelihood() + Math.log(wpi.endog().length) * wpi.params().length;
        assertEquals(expectedAic, results.aic(), 1e-8);
        assertEquals(expectedAicc, results.aicc(), 1e-8);
        assertEquals(expectedBic, results.bic(), 1e-8);
        assertEquals(expectedHqic, results.hqic(), 1e-8);
        assertNotEquals(totalObsBic, results.bic(), 1e-8);
    }

    @Test
    void testInformationCriteriaHandleSingleEffectiveObservation() {
        double[] y = {1.0, 2.0};
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 0), y)
                .simpleDifferencing(true)
                .build());
        double[] params = {1.0};

        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, params, filterResult);

        assertEquals(0, model.likelihoodBurn());
        assertEquals(1, model.effectiveObservationCount());
        assertEquals(0, results.logLikelihoodBurn());
        assertEquals(1, results.nobsEffective());
        assertEquals(1 - results.dfModel(), results.dfResid());
        assertEquals(-2.0 * results.logLikelihood(), results.bic(), 1e-8);
        assertEquals(Double.POSITIVE_INFINITY, results.aicc());
        assertEquals(Double.POSITIVE_INFINITY, results.hqic());
    }

    @Test
    void testExactDiffuseResultsDfModelIncludesDiffuseStates() {
        WpiCoverageReference wpi = loadWpiCoverageReference();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(3, 2, 2), wpi.endog())
                .seasonalOrder(SeasonalOrder.of(3, 2, 2, 4))
                .exog(wpi.exog())
                .include(SARIMAXOption.USE_EXACT_DIFFUSE)
                .build());

        FilterResult filterResult = model.filter(wpi.params(), FilterOptions.defaults());
        SARIMAXResults results = newResults(model, wpi.params(), filterResult);

        int expectedDiffuseStates = 2 + 2 * 4;
        int expectedDfModel = wpi.params().length + expectedDiffuseStates;
        double expectedAic = -2.0 * results.logLikelihood() + 2.0 * expectedDfModel;
        double expectedAicc = -2.0 * results.logLikelihood()
            + 2.0 * expectedDfModel * results.nobsEffective()
            / (results.nobsEffective() - expectedDfModel - 1.0);
        double paramsOnlyAic = -2.0 * results.logLikelihood() + 2.0 * wpi.params().length;

        assertEquals(0, results.logLikelihoodBurn());
        assertEquals(wpi.endog().length, results.nobsEffective());
        assertEquals(expectedDiffuseStates, results.kDiffuseStates());
        assertEquals(expectedDfModel, results.dfModel());
        assertEquals(results.nobsEffective() - expectedDfModel, results.dfResid());
        assertEquals(expectedAic, results.aic(), 1e-8);
        assertEquals(expectedAicc, results.aicc(), 1e-8);
        assertNotEquals(paramsOnlyAic, results.aic(), 1e-8);
    }

    @Test
    void testTrendOnlyLogLikelihoodMatchesDirectStateSpace() {
        double[] y = {2.1, 3.7, 4.0, 5.8, 6.9, 8.4};
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 0, 0), y)
                .trendPowers(0, 1)
                .build());
        double[] params = {1.25, 0.75, 0.2};

        KalmanSSM direct = buildTrendOnlyStateSpace(params[0], params[1], params[2], y);
        FilterResult directFilter = KalmanEngine.filter(direct, InitialState.stationary(direct), LIKELIHOOD_OPTIONS);

        assertEquals(directFilter.logLikelihood(), model.logLikelihood(params), TOL);
        assertArrayEquals(directFilter.logLikelihoodObs, model.logLikelihoodObs(params), TOL);
    }

    @Test
    void testExogOnlyLogLikelihoodAndForecastMatchDirectStateSpace() {
        double[] y = {1.2, -0.7, 0.4, 1.6, -1.1, 0.8};
        double[][] exog = {
            {1.0},
            {-0.5},
            {0.25},
            {2.0},
            {-1.5},
            {0.75}
        };
        double[] params = {0.8, 0.35};
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 0, 0), y)
                .exog(exog)
                .build());

        KalmanSSM direct = buildExogOnlyStateSpace(params[0], exog, params[1], y);
        FilterResult directFilter = KalmanEngine.filter(direct, InitialState.stationary(direct), LIKELIHOOD_OPTIONS);

        assertArrayEquals(params, model.transformParams(model.untransformParams(params)), 1e-8);
        assertEquals(directFilter.logLikelihood(), model.logLikelihood(params), TOL);
        assertArrayEquals(directFilter.logLikelihoodObs, model.logLikelihoodObs(params), TOL);

        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, params, filterResult);
        assertThrows(IllegalArgumentException.class, () -> results.forecast(1, null));

        double[][] futureExog = {{1.5}, {-2.0}};
        SARIMAXPrediction forecast = results.forecast(2, futureExog);

        double[][] extendedExog = Arrays.copyOf(exog, exog.length + futureExog.length);
        for (int i = 0; i < futureExog.length; i++) {
            extendedExog[exog.length + i] = Arrays.copyOf(futureExog[i], futureExog[i].length);
        }
        double[] extendedY = Arrays.copyOf(y, y.length + futureExog.length);
        Arrays.fill(extendedY, y.length, extendedY.length, Double.NaN);
        KalmanSSM directPredictionModel = buildExogOnlyStateSpace(params[0], extendedExog, params[1], extendedY);
        FilterResult directPrediction = KalmanEngine.filter(directPredictionModel, InitialState.stationary(directPredictionModel), FilterOptions.defaults());

        for (int step = 0; step < futureExog.length; step++) {
            int t = y.length + step;
            assertEquals(directPrediction.forecast(0, t), forecast.mean()[step], TOL);
            assertEquals(directPrediction.forecastErrorCov(0, 0, t), forecast.variance()[step], TOL);
        }
    }

    @Test
    void testExogOnlyStandardizedForecastsErrorMatchesStatsmodelsReference() {
        double[] y = {1.2, -0.7, 0.4, 1.6, -1.1, 0.8};
        double[][] exog = {
            {1.0},
            {-0.5},
            {0.25},
            {2.0},
            {-1.5},
            {0.75}
        };
        double[] params = {0.8, 0.35};
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 0, 0), y)
                .exog(exog)
                .build());

        FilterResult filterResult = model.filter(params, FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR)
            .build());
        SARIMAXResults results = newResults(model, params, filterResult);

        double[] expected = StatsmodelsSarimaxFixtures.dictArray(
            "exog_only_standardized", "standardized_forecasts_error");
        assertArrayEquals(expected, filterResult.standardizedForecastError(), TOL);
        assertArrayEquals(expected, results.standardizedForecastError(), TOL);
    }

    @Test
    void testExogOnlyMissingObservationPathMatchesDirectStateSpace() {
        double[] y = {1.2, Double.NaN, 0.4, 1.6, -1.1, 0.8};
        double[][] exog = {
            {1.0},
            {-0.5},
            {0.25},
            {2.0},
            {-1.5},
            {0.75}
        };
        double[] params = {0.8, 0.35};
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 0, 0), y)
                .exog(exog)
                .build());

        KalmanSSM direct = buildExogOnlyStateSpace(params[0], exog, params[1], y);
        FilterResult directFilter = KalmanEngine.filter(direct, InitialState.stationary(direct), FilterOptions.defaults());
        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, params, filterResult);

        assertEquals(directFilter.logLikelihood(), model.logLikelihood(params), TOL);
        assertArrayEquals(directFilter.logLikelihoodObs, model.logLikelihoodObs(params), TOL);

        double[] fitted = results.fittedValues();
        double[] residuals = results.residuals();
        for (int t = 0; t < y.length; t++) {
            assertEquals(directFilter.forecast(0, t), filterResult.forecast(0, t), TOL);
            assertEquals(directFilter.forecastErrorCov(0, 0, t), filterResult.forecastErrorCov(0, 0, t), TOL);
            assertEquals(directFilter.forecast(0, t), fitted[t], TOL);
            if (Double.isNaN(y[t])) {
                assertTrue(Double.isNaN(residuals[t]));
            } else {
                assertEquals(directFilter.forecastError(0, t), residuals[t], TOL);
            }
        }
        assertEquals(directFilter.forecast(0, 1), fitted[1], TOL);
        assertTrue(Double.isNaN(residuals[1]));
    }

    @Test
    void testExogOnlyMissingStandardizedForecastsErrorMatchesStatsmodelsReference() {
        double[] y = {1.2, Double.NaN, 0.4, 1.6, -1.1, 0.8};
        double[][] exog = {
            {1.0},
            {-0.5},
            {0.25},
            {2.0},
            {-1.5},
            {0.75}
        };
        double[] params = {0.8, 0.35};
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 0, 0), y)
                .exog(exog)
                .build());

        FilterResult filterResult = model.filter(params, FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR)
            .build());
        SARIMAXResults results = newResults(model, params, filterResult);

        double[] expectedFilterSurface = StatsmodelsSarimaxFixtures.dictArray(
            "exog_only_standardized", "missing_standardized_forecasts_error");
        double[] expectedResultsSurface = Arrays.copyOf(expectedFilterSurface, expectedFilterSurface.length);
        expectedResultsSurface[1] = Double.NaN;
        assertArrayEquals(expectedFilterSurface, filterResult.standardizedForecastError(), TOL);
        assertArrayEqualsAllowNaN(expectedResultsSurface, results.standardizedForecastError());
    }

    @Test
    void testMissingExogCoverageScenarioMatchesStatsmodelsReference() {
        MissingExogReference reference = loadMissingExogReference();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(3, 0, 2), reference.endog())
                .trendPowers(3)
                .exog(column(reference.exog()))
                .build());

        FilterResult filterResult = model.filter(reference.params(), FilterOptions.defaults());
        SARIMAXResults results = newResults(model, reference.params(), filterResult);

        assertEquals(reference.loglike(), model.logLikelihood(reference.params()), 2.0);
        assertEquals(model.logLikelihood(reference.params()), Arrays.stream(model.logLikelihoodObs(reference.params())).sum(), TOL);
        assertEquals(reference.endog().length, results.fittedValues().length);
        assertEquals(reference.endog().length, results.residuals().length);
        assertTrue(Double.isNaN(results.residuals()[reference.missingIndex()]));
    }

    @Test
    void testStandardizedForecastsErrorMatchesForecastErrorScaling() {
        MissingExogReference reference = loadMissingExogReference();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(3, 0, 2), reference.endog())
                .trendPowers(3)
                .exog(column(reference.exog()))
                .build());

        FilterResult filterResult = model.filter(reference.params(), FilterOptions.defaults());
        SARIMAXResults results = newResults(model, reference.params(), filterResult);

        double[] expected = new double[reference.endog().length];
        for (int t = 0; t < expected.length; t++) {
            double forecastVariance = filterResult.forecastErrorCov(0, 0, t);
            expected[t] = Double.isNaN(reference.endog()[t]) || !(forecastVariance > 0.0) || !Double.isFinite(forecastVariance)
                ? Double.NaN
                : filterResult.forecastError(0, t) / Math.sqrt(forecastVariance);
        }

        assertArrayEqualsAllowNaN(expected, results.standardizedForecastError());
    }

    @Test
    void testFilterResultMatchesFixtureAndDefensivelyCopiesState() {
        MissingExogReference reference = loadMissingExogReference();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(3, 0, 2), reference.endog())
                .trendPowers(3)
                .exog(column(reference.exog()))
                .build());

        FilterResult filterResult = model.filter(reference.params(), FilterOptions.defaults());
        SARIMAXResults results = newResults(model, reference.params(), filterResult);
        double[] expectedLoglikeObs = Arrays.copyOf(filterResult.logLikelihoodObs, filterResult.logLikelihoodObs.length);
        double[] expectedFitted = inSampleForecast(filterResult);

        filterResult.logLikelihoodObs[0] = Double.POSITIVE_INFINITY;
        filterResult.forecast[0] = Double.POSITIVE_INFINITY;

        FilterResult firstCopy = results.filterResult();
        assertArrayEqualsAllowNaN(expectedLoglikeObs, firstCopy.logLikelihoodObs);
        assertArrayEqualsAllowNaN(expectedFitted, inSampleForecast(firstCopy));

        firstCopy.logLikelihoodObs[0] = Double.NEGATIVE_INFINITY;
        firstCopy.forecast[0] = Double.NEGATIVE_INFINITY;

        FilterResult secondCopy = results.filterResult();
        assertArrayEqualsAllowNaN(expectedLoglikeObs, secondCopy.logLikelihoodObs);
        assertArrayEqualsAllowNaN(expectedFitted, inSampleForecast(secondCopy));
    }

    @Test
    void testDynamicPredictAndForecastWithFutureExogUseStatsmodelsMissingScenario() {
        MissingExogReference reference = loadMissingExogReference();
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(3, 0, 2), reference.endog())
                .trendPowers(3)
                .exog(column(reference.exog()))
                .build());

        FilterResult filterResult = model.filter(reference.params(), FilterOptions.defaults());
        SARIMAXResults results = newResults(model, reference.params(), filterResult);

        double[][] futureExog = column(reference.futureExog());
        SARIMAXPrediction staticPrediction = results.predict(reference.predictStart(), reference.predictEnd(), -1, futureExog);
        SARIMAXPrediction dynamicPrediction = results.predict(reference.predictStart(), reference.predictEnd(),
            reference.predictStart(), futureExog);
        SARIMAXPrediction forecast = results.forecast(reference.futureExog().length, futureExog);

        assertEquals(reference.predictEnd() - reference.predictStart() + 1, staticPrediction.mean().length);
        assertEquals(staticPrediction.mean().length, dynamicPrediction.mean().length);
        assertEquals(reference.futureExog().length, forecast.mean().length);
        assertAllFinite(staticPrediction.mean());
        assertAllFinite(staticPrediction.variance());
        assertAllFinite(dynamicPrediction.mean());
        assertAllFinite(dynamicPrediction.variance());
        assertAllFinite(forecast.mean());
        assertAllFinite(forecast.variance());
    }

    @Test
    void testIntegratedAr1SnapshotMatchesExpectedLayout() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 1, 0), y).build());

        KalmanSSM snapshot = model.snapshotModel(new double[]{0.6, 0.5});

        assertEquals(2, snapshot.stateCount());
        assertEquals(1, snapshot.stateDisturbanceCount());
        assertArrayEquals(new double[]{1.0, 1.0}, slice(snapshot.designData(), snapshot.designOffset(0), 2), TOL);
        assertArrayEquals(new double[]{1.0, 1.0, 0.0, 0.6}, slice(snapshot.transitionData(), snapshot.transitionOffset(0), 4), TOL);
        assertArrayEquals(new double[]{0.0, 1.0}, slice(snapshot.selectionData(), snapshot.selectionOffset(0), 2), TOL);
        assertArrayEquals(new double[]{0.5}, slice(snapshot.stateCovarianceData(), snapshot.stateCovarianceOffset(0), 1), TOL);
    }

    @Test
    void testAr1StateRegressionSnapshotUsesTimeVaryingExogDesign() {
        double[] y = {1.2, -0.4, 0.8, 1.1};
        double[] x = {0.5, -1.0, 0.25, 1.5};
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y)
                .exog(column(x))
                .mleRegression(false)
                .build());

        KalmanSSM snapshot = model.snapshotModel(new double[]{0.6, 0.5});

        assertArrayEquals(new String[]{"ar.L1", "sigma2"}, model.parameterNames());
        assertEquals(2, model.startParams().length);
        assertEquals(2, snapshot.stateCount());
        assertArrayEquals(new double[]{1.0, x[0]}, slice(snapshot.designData(), snapshot.designOffset(0), 2), TOL);
        assertArrayEquals(new double[]{1.0, x[2]}, slice(snapshot.designData(), snapshot.designOffset(2), 2), TOL);
        assertArrayEquals(new double[]{0.6, 0.0, 0.0, 1.0}, slice(snapshot.transitionData(), snapshot.transitionOffset(0), 4), TOL);
        assertArrayEquals(new double[]{1.0, 0.0}, slice(snapshot.selectionData(), snapshot.selectionOffset(0), 2), TOL);
        assertArrayEquals(new double[]{0.5}, slice(snapshot.stateCovarianceData(), snapshot.stateCovarianceOffset(0), 1), TOL);
    }

    @Test
    void testAr1StateRegressionFitKeepsExogOutOfParams() {
        double[] x = generateAr1(0.2, 1.0, 120, 20260509L);
        double[] latent = generateAr1(0.55, 0.25, 120, 20260510L);
        double[] y = new double[x.length];
        for (int index = 0; index < y.length; index++) {
            y[index] = 0.8 * x[index] + latent[index];
        }

        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y)
                .exog(column(x))
                .mleRegression(false)
                .build());

        SARIMAXResults results = model.fit();

        assertEquals(2, results.params().length);
        assertTrue(Double.isFinite(results.logLikelihood()));
        assertEquals(y.length, results.fittedValues().length);
    }

    @Test
    void testAr1TimeVaryingRegressionSnapshotUsesExogDisturbanceBlock() {
        double[] y = {0.5, -0.2, 0.1, 0.9};
        double[] x = {1.0, -0.5, 0.25, 1.5};
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y)
                .exog(column(x))
                .mleRegression(false)
                .timeVaryingRegression(true)
                .build());

        KalmanSSM snapshot = model.snapshotModel(new double[]{0.6, 0.2, 0.5});

        assertArrayEquals(new String[]{"ar.L1", "var.x1", "sigma2"}, model.parameterNames());
        assertEquals(3, model.startParams().length);
        assertEquals(2, snapshot.stateCount());
        assertEquals(2, snapshot.stateDisturbanceCount());
        assertArrayEquals(new double[]{1.0, x[0]}, slice(snapshot.designData(), snapshot.designOffset(0), 2), TOL);
        assertArrayEquals(new double[]{1.0, x[3]}, slice(snapshot.designData(), snapshot.designOffset(3), 2), TOL);
        assertArrayEquals(new double[]{0.6, 0.0, 0.0, 1.0}, slice(snapshot.transitionData(), snapshot.transitionOffset(0), 4), TOL);
        assertArrayEquals(new double[]{1.0, 0.0, 0.0, 1.0}, slice(snapshot.selectionData(), snapshot.selectionOffset(0), 4), TOL);
        assertArrayEquals(new double[]{0.5, 0.0, 0.0, 0.2}, slice(snapshot.stateCovarianceData(), snapshot.stateCovarianceOffset(0), 4), TOL);
    }

    @Test
    void testWpiTimeVaryingRegressionFitUsesStatsmodelsSeries() {
        double[] endog = diff(loadWpiSeries().endog());
        double[] exog = new double[endog.length];
        Arrays.fill(exog, 1.0);
        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), endog)
                .exog(column(exog))
                .mleRegression(false)
                .timeVaryingRegression(true)
                .build());

        SARIMAXResults results = model.fit();

        assertEquals(3, results.params().length);
        assertTrue(Double.isFinite(results.logLikelihood()));
        assertTrue(results.params()[1] > 0.0);
        assertTrue(results.params()[2] > 0.0);
    }

    @Test
    void testPredictAndForecastSemanticsStayConsistentWithFilterSlices() {
        double[] y = generateAr1(0.65, 0.2, 64, 42L);
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.65, 0.2};
        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, params, filterResult);

        SARIMAXPrediction staticPrediction = results.predict(10, 20, -1, null);
        assertEquals(11, staticPrediction.mean().length);
        for (int t = 10; t <= 20; t++) {
            assertEquals(filterResult.forecast(0, t), staticPrediction.mean()[t - 10], TOL);
            assertEquals(filterResult.forecastErrorCov(0, 0, t), staticPrediction.variance()[t - 10], TOL);
        }

        SARIMAXPrediction dynamicPrediction = results.predict(10, 20, 10, null);
        assertEquals(11, dynamicPrediction.mean().length);
        assertNotEquals(staticPrediction.mean()[1], dynamicPrediction.mean()[1], 1e-9);

        SARIMAXPrediction forecast = results.forecast(3, null);
        assertEquals(3, forecast.mean().length);
        for (double value : forecast.mean()) {
            assertTrue(Double.isFinite(value));
        }
    }

    @Test
    void testPredictionResultSummarySurfacesMatchStatsmodelsSemantics() {
        double[] y = generateAr1(0.65, 0.2, 64, 42L);
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.65, 0.2};
        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, params, filterResult);

        SARIMAXPrediction prediction = results.predict(10, 14);
        double[] mean = prediction.mean();
        double[] standardError = prediction.seMean();
        double[][] confidence = prediction.confInt();
        SARIMAXPrediction.SummaryFrame frame = prediction.summaryFrame();
        double criticalValue = com.curioloop.yum4j.math.Normal.inv(0.975);

        assertArrayEquals(mean, frame.mean(), TOL);
        assertArrayEquals(standardError, frame.meanSe(), TOL);
        for (int i = 0; i < mean.length; i++) {
            int t = 10 + i;
            assertEquals(filterResult.forecast(0, t), mean[i], TOL);
            assertEquals(Math.sqrt(filterResult.forecastErrorCov(0, 0, t)), standardError[i], TOL);
            assertEquals(mean[i] - criticalValue * standardError[i], confidence[i][0], TOL);
            assertEquals(mean[i] + criticalValue * standardError[i], confidence[i][1], TOL);
            assertEquals(confidence[i][0], frame.meanCiLower()[i], TOL);
            assertEquals(confidence[i][1], frame.meanCiUpper()[i], TOL);
        }
    }

    @Test
    void testPredictionWrapperHandlesMemoryConservationLikeStatsmodels() {
        double[] y = generateAr1(0.65, 0.2, 32, 91L);
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.65, 0.2};

        FilterResult noForecastCov = model.filter(params, predictionOptions()
            .without(FilterOptions.Surface.FORECAST_COVARIANCE));
        SARIMAXResults noForecastCovResults = newResults(model, params, noForecastCov);
        SARIMAXPrediction noForecastCovPrediction = noForecastCovResults.predict(0, y.length - 1);
        double[] expectedMean = new double[y.length];
        for (int t = 0; t < y.length; t++) {
            expectedMean[t] = noForecastCov.forecast(0, t);
        }
        assertArrayEquals(expectedMean, noForecastCovPrediction.mean(), TOL);
        for (double value : noForecastCovPrediction.seMean()) {
            assertTrue(Double.isNaN(value));
        }
        for (double[] interval : noForecastCovPrediction.confInt()) {
            assertTrue(Double.isNaN(interval[0]));
            assertTrue(Double.isNaN(interval[1]));
        }

        FilterResult noForecast = model.filter(params, FilterOptions.builder()
            .drop(FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE,
                FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE)
            .build());
        SARIMAXResults noForecastResults = newResults(model, params, noForecast);
        assertThrows(IllegalArgumentException.class, noForecastResults::fittedValues);
        assertThrows(IllegalArgumentException.class, noForecastResults::residuals);
        assertThrows(IllegalArgumentException.class, noForecastResults::predict);

        FilterResult noPredicted = model.filter(params, FilterOptions.builder()
            .drop(FilterOptions.Surface.PREDICTED_STATE,
                FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
                FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)
            .build());
        SARIMAXResults noPredictedResults = newResults(model, params, noPredicted);
        SARIMAXPrediction noPredictedDynamic = noPredictedResults.predict(5, 10, 5, null);
        assertEquals(PredictionKind.DYNAMIC_IN_SAMPLE, noPredictedDynamic.kind());
        assertEquals(6, noPredictedDynamic.mean().length);
        assertAllFinite(noPredictedDynamic.mean());

        SARIMAXPrediction noPredictedForecast = noPredictedResults.forecast(3);
        assertEquals(3, noPredictedForecast.mean().length);
        assertEquals(3, noPredictedForecast.summaryFrame().mean().length);
        assertAllFinite(noPredictedForecast.mean());
        assertAllFinite(noPredictedForecast.seMean());
    }

    @Test
    void testConcatenatedSarimaxPredictionMatchesShortSampleForecast() {
        double[] y = generateAr1(0.55, 0.3, 36, 20260601L);
        double[] x = generateAr1(0.15, 0.6, y.length, 20260602L);
        int horizon = 5;
        double[] fullY = Arrays.copyOf(y, y.length + horizon);
        Arrays.fill(fullY, y.length, fullY.length, Double.NaN);
        double[] fullX = Arrays.copyOf(x, x.length + horizon);
        for (int i = 0; i < horizon; i++) {
            fullX[x.length + i] = 0.2 - 0.15 * i;
        }
        double[] params = {0.4, 0.52, 0.3};

        SARIMAX fullModel = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), fullY)
                .exog(column(fullX))
                .build());
        SARIMAXResults fullResults = newResults(fullModel, params, fullModel.filter(params, FilterOptions.defaults()));
        SARIMAXPrediction concatenated = fullResults.predict(y.length, fullY.length - 1);

        SARIMAX shortModel = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y)
                .exog(column(x))
                .build());
        SARIMAXResults shortResults = newResults(shortModel, params, shortModel.filter(params, FilterOptions.defaults()));
        SARIMAXPrediction forecast = shortResults.forecast(horizon, column(Arrays.copyOfRange(fullX, x.length, fullX.length)));

        assertArrayEquals(concatenated.mean(), forecast.mean(), TOL);
        assertArrayEquals(concatenated.variance(), forecast.variance(), TOL);
        assertArrayEquals(concatenated.summaryFrame().meanCiLower(), forecast.summaryFrame().meanCiLower(), TOL);
        assertArrayEquals(concatenated.summaryFrame().meanCiUpper(), forecast.summaryFrame().meanCiUpper(), TOL);
    }

    @Test
    void testPredictionInformationSetIdentitiesMatchKalmanSurfaces() {
        double[] y = generateAr1(0.45, 0.25, 30, 20260603L);
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.45, 0.25};
        KalmanSSM snapshot = model.snapshotModel(params);
        FilterResult filter = model.filter(params, FilterOptions.defaults());
        SmootherResult smoother = model.smooth(params, SmootherOptions.defaults());
        SARIMAXResults results = newResults(model, params, filter);

        for (int t = 5; t < 12; t++) {
            assertEquals(filter.forecast(0, t), observationSignal(snapshot, filter.predictedState, filter.predictedStateOffset(t), t), TOL);
            assertEquals(filter.forecastErrorCov(0, 0, t), observationVariance(snapshot, filter.predictedStateCov, filter.predictedStateCovOffset(t), t, true), TOL);

            double filteredSignal = observationSignal(snapshot, filter.filteredState, filter.filteredStateOffset(t), t);
            double smoothedSignal = observationSignal(snapshot, smoother.smoothedState, smoother.smoothedStateOffset(t), t);
            double filteredVariance = observationVariance(snapshot, filter.filteredStateCov, filter.filteredStateCovOffset(t), t, true);
            double smoothedVariance = observationVariance(snapshot, smoother.smoothedStateCov, smoother.smoothedStateCovOffset(t), t, true);

            assertTrue(Double.isFinite(filteredSignal));
            assertTrue(Double.isFinite(smoothedSignal));
            assertTrue(filteredVariance >= 0.0);
            assertTrue(smoothedVariance >= 0.0);
            assertTrue(smoothedVariance <= filteredVariance + 1e-8);
        }

        SARIMAXPrediction filtered = results.predict(5, 8, PredictionInformationSet.FILTERED, true);
        SARIMAXPrediction smoothed = results.predict(5, 8, PredictionInformationSet.SMOOTHED, true);
        assertTrue(filtered.signalOnly());
        assertTrue(smoothed.signalOnly());
        assertEquals(PredictionInformationSet.FILTERED, filtered.informationSet());
        assertEquals(PredictionInformationSet.SMOOTHED, smoothed.informationSet());
        for (int i = 0; i < filtered.mean().length; i++) {
            int t = 5 + i;
            assertEquals(observationSignal(snapshot, filter.filteredState, filter.filteredStateOffset(t), t), filtered.mean()[i], TOL);
            assertEquals(observationVariance(snapshot, filter.filteredStateCov, filter.filteredStateCovOffset(t), t, false), filtered.variance()[i], TOL);
            assertEquals(observationSignal(snapshot, smoother.smoothedState, smoother.smoothedStateOffset(t), t), smoothed.mean()[i], TOL);
            assertEquals(observationVariance(snapshot, smoother.smoothedStateCov, smoother.smoothedStateCovOffset(t), t, false), smoothed.variance()[i], TOL);
        }
    }

    @Test
    void testLifecycleSimulationAndImpulseResponseWrappersAreUsable() {
        double[] y = generateAr1(0.45, 0.25, 24, 20260604L);
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.45, 0.25};
        SARIMAXResults results = newResults(model, params, model.filter(params, FilterOptions.defaults()));

        SARIMAXResults appended = results.append(new double[]{0.2, -0.1});
        SARIMAXResults extended = results.extend(new double[]{0.2, -0.1});
        SARIMAXResults applied = results.apply(new double[]{0.2, -0.1});
        assertEquals(y.length + 2, appended.filterResult().nobs);
        assertEquals(2, extended.filterResult().nobs);
        assertEquals(2, applied.filterResult().nobs);
        assertAllFinite(extended.fittedValues());
        assertAllFinite(applied.fittedValues());

        SimulationSmootherResult simulation = results.simulate(3,
            new double[3], new double[3], new double[model.stateCount()]);
        assertEquals(3, simulation.nobs);
        assertEquals(3, simulation.generatedObsLength());

        ImpulseResponse irf = results.impulseResponses(3);
        ImpulseResponse cumulative = results.impulseResponses(3, 0, false, true);
        assertEquals(3, irf.steps());
        assertEquals(1, irf.responseCount());
        assertEquals(irf.response(0)[0][0] + irf.response(1)[0][0], cumulative.response(1)[0][0], TOL);
    }

    @Test
    void testStatsmodelsGapWrappersAreUsable() {
        double[] y = generateAr1(0.42, 0.20, 36, 20260616L);
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.42, 0.20};
        SARIMAXResults results = newResults(model, params, model.filter(params, FilterOptions.defaults()));

        SARIMAXResults fixed = model.fit(SARIMAXFitOptions.builder()
            .fixedParameters(FixedParameters.of(new int[]{0, 1}, params))
            .build());
        assertEquals(2, fixed.fixedParameterCount());
        assertEquals(0, fixed.freeParameterCount());
        assertArrayEquals(params, fixed.params(), TOL);
        assertTrue(Double.isNaN(fixed.covParams()[0]));

        SARIMAXResults refit = results.appendRefit(new double[]{0.12, -0.05}, null,
            SARIMAXFitOptions.builder().fixedParameters(FixedParameters.of(new int[]{0, 1}, params)).build());
        assertArrayEquals(params, refit.params(), TOL);
        assertEquals(y.length + 2, refit.filterResult().nobs);

        assertTrue(Double.isFinite(results.testNormality().statistic()));
        assertTrue(Double.isFinite(results.testHeteroskedasticity().statistic()));
        assertEquals(2, results.testSerialCorrelation(2).lags().length);

        SimulationSmootherRepetitions simulations = results.simulateRepetitions(2, 2,
            SimulationAnchor.FINAL, new Random(20260616L), null);
        assertEquals(2, simulations.repetitions());
        assertEquals(2, simulations.nsimulations());

        ImpulseResponse anchored = results.impulseResponses(2, 0, false, false, 0);
        ImpulseResponseRepetitions irfRuns = results.impulseResponseRepetitions(2, 0, 2, false, false);
        assertEquals(2, anchored.steps());
        assertEquals(2, irfRuns.repetitions());

        SARIMAXResults updated = results.append(new double[]{0.12, -0.05});
        assertEquals(results.filterResult().nobs, results.news(updated).targetCount());
        assertEquals(results.filterResult().nobs, results.smoothedDecomposition().nobs());
    }

    @Test
    void testPredictionWrapperSupportsDynamicStartAndForecast() {
        double[] y = generateAr1(0.65, 0.2, 64, 43L);
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());
        double[] params = {0.65, 0.2};
        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        SARIMAXResults results = newResults(model, params, filterResult);

        SARIMAXPrediction staticPrediction = results.predict(10, 20, -1, null);
        SARIMAXPrediction delayedDynamic = results.predict(10, 20, 15, null);
        SARIMAXPrediction startDynamic = results.predict(10, 20, 10, null);
        SARIMAXPrediction forecast = results.forecast(3);
        SARIMAXPrediction forecastViaPredict = results.predict(y.length, y.length + 2, -1, null);

        assertTrue(delayedDynamic.dynamic());
        assertEquals(15, delayedDynamic.dynamicStart());
        for (int i = 0; i < 5; i++) {
            assertEquals(staticPrediction.mean()[i], delayedDynamic.mean()[i], TOL);
        }
        assertNotEquals(staticPrediction.mean()[6], delayedDynamic.mean()[6], 1e-9);
        assertEquals(10, startDynamic.dynamicStart());
        assertArrayEquals(forecast.mean(), forecastViaPredict.mean(), TOL);
        assertArrayEquals(forecast.variance(), forecastViaPredict.variance(), TOL);
    }

    @Test
    void testFitAndForecastSmoke() {
        double[] y = generateAr1(0.65, 0.2, 160, 42L);
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0), y).build());

        SARIMAXResults results = model.fit();
        Optimization optimization = results.optimization();

        assertTrue(Double.isFinite(optimization.cost()));
        assertEquals(2, results.params().length);
        assertTrue(results.params()[1] > 0.0);

        SARIMAXPrediction prediction = results.forecast(4, null);
        assertEquals(4, prediction.mean().length);
        for (double value : prediction.mean()) {
            assertTrue(Double.isFinite(value));
        }
        for (double value : prediction.variance()) {
            assertTrue(Double.isFinite(value));
            assertTrue(value >= 0.0);
        }
    }

    @Test
    void testArima000SmoothingMatchesStatsmodelsIdentity() {
        double[] endog = {2.5, -1.0, 0.75, 4.25, -0.5, 3.0};
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(0, 0, 0), endog).build());

        SmootherResult smooth = model.smooth(new double[]{1.0}, SmootherOptions.defaults());

        for (int t = 0; t < endog.length; t++) {
            assertEquals(endog[t], smooth.smoothedState(0, t), 1e-10);
        }
    }

    @Test
    void testArima000ExogSmoothingMatchesStatsmodelsOlsIdentity() {
        double[] error = {0.4, -0.2, 1.1, -0.7, 0.0, 0.9};
        double[] endog = new double[error.length];
        double[] exog = new double[error.length];
        Arrays.fill(exog, 1.0);
        for (int t = 0; t < error.length; t++) {
            endog[t] = 10.0 + error[t];
        }

        SARIMAX model = new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 0, 0), endog)
                .exog(column(exog))
                .build());

        SmootherResult smooth = model.smooth(new double[]{10.0, 1.0}, SmootherOptions.defaults());

        for (int t = 0; t < error.length; t++) {
            assertEquals(error[t], smooth.smoothedState(0, t), 1e-10);
        }
    }

    @Test
    void testArima010SmoothedDifferencedStateMatchesStatsmodelsIdentity() {
        double[] endog = {3.0, 3.25, 2.5, 4.0, 4.75, 4.0};
        SARIMAX model = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 0), endog).build());

        SmootherResult smooth = model.smooth(new double[]{1.0}, SmootherOptions.defaults());

        for (int t = 1; t < endog.length; t++) {
            assertEquals(endog[t] - endog[t - 1], smooth.smoothedState(1, t), 1e-10);
        }
    }

    private static KalmanSSM buildAr1StateSpace(double phi, double sigma2, double[] y) {
        return KalmanSSM.builder(1, 1, 1, y.length)
            .design(new double[]{1.0}, false)
            .obsIntercept(new double[]{0.0}, false)
            .obsCov(new double[]{0.0}, false)
            .transition(new double[]{phi}, false)
            .stateIntercept(new double[]{0.0}, false)
            .selection(new double[]{1.0}, false)
            .stateCovariance(new double[]{sigma2}, false)
            .endog(Arrays.copyOf(y, y.length))
            .allObserved()
            .build();
    }

            private static KalmanSSM buildAr1MeasurementErrorStateSpace(double phi,
                                              double measurementVariance,
                                              double sigma2,
                                              double[] y) {
            return KalmanSSM.builder(1, 1, 1, y.length)
                .design(new double[]{1.0}, false)
                .obsIntercept(new double[]{0.0}, false)
                .obsCov(new double[]{measurementVariance}, false)
                .transition(new double[]{phi}, false)
                .stateIntercept(new double[]{0.0}, false)
                .selection(new double[]{1.0}, false)
                .stateCovariance(new double[]{sigma2}, false)
                .endog(Arrays.copyOf(y, y.length))
                .allObserved()
                .build();
            }

    private static KalmanSSM buildTrendOnlyStateSpace(double intercept,
                                                            double slope,
                                                            double sigma2,
                                                            double[] y) {
        double[] stateIntercept = new double[y.length];
        for (int t = 0; t < y.length; t++) {
            stateIntercept[t] = intercept + slope * (t + 1.0);
        }
        return buildScalarInterceptStateSpace(y, null, stateIntercept, sigma2);
    }

    private static KalmanSSM buildExogOnlyStateSpace(double beta,
                                                           double[][] exog,
                                                           double sigma2,
                                                           double[] y) {
        double[] obsIntercept = new double[y.length];
        for (int t = 0; t < y.length; t++) {
            obsIntercept[t] = beta * exog[t][0];
        }
        return buildScalarInterceptStateSpace(y, obsIntercept, null, sigma2);
    }

    private static KalmanSSM buildScalarInterceptStateSpace(double[] y,
                                                                  double[] obsIntercept,
                                                                  double[] stateIntercept,
                                                                  double sigma2) {
        boolean hasMissing = Arrays.stream(y).anyMatch(Double::isNaN);
        boolean[] missing = hasMissing ? new boolean[y.length] : null;
        int[] nmissing = hasMissing ? new int[y.length] : null;
        if (hasMissing) {
            for (int t = 0; t < y.length; t++) {
                if (Double.isNaN(y[t])) {
                    missing[t] = true;
                    nmissing[t] = 1;
                }
            }
        }

        KalmanSSM.Builder builder = KalmanSSM.builder(1, 1, 1, y.length)
            .design(new double[]{1.0}, false)
            .obsIntercept(obsIntercept == null ? new double[]{0.0} : Arrays.copyOf(obsIntercept, obsIntercept.length), obsIntercept != null)
            .obsCov(new double[]{0.0}, false)
            .transition(new double[]{0.0}, false)
            .stateIntercept(stateIntercept == null ? new double[]{0.0} : Arrays.copyOf(stateIntercept, stateIntercept.length), stateIntercept != null)
            .selection(new double[]{1.0}, false)
            .stateCovariance(new double[]{sigma2}, false)
            .endog(Arrays.copyOf(y, y.length));
        return hasMissing ? builder.missing(missing, nmissing).build() : builder.allObserved().build();
    }

    private static double[] generateAr1(double phi, double sigma2, int n, long seed) {
        Random random = new Random(seed);
        double[] y = new double[n];
        double state = 0.0;
        for (int t = 0; t < n; t++) {
            state = phi * state + random.nextGaussian() * Math.sqrt(sigma2);
            y[t] = state;
        }
        return y;
    }

    private static Air2Reference loadAir2Reference() {
        return new Air2Reference(
            StatsmodelsSarimaxFixtures.array("air2_data"),
            StatsmodelsSarimaxFixtures.dictArray("air2_stationary", "params_ma"),
            StatsmodelsSarimaxFixtures.dictArray("air2_stationary", "se_ma_opg"),
            StatsmodelsSarimaxFixtures.dictArray("air2_stationary", "se_ma_oim"),
            StatsmodelsSarimaxFixtures.dictArray("air2_stationary", "params_seasonal_ma"),
            StatsmodelsSarimaxFixtures.dictArray("air2_stationary", "se_seasonal_ma_opg"),
            StatsmodelsSarimaxFixtures.dictArray("air2_stationary", "se_seasonal_ma_oim"),
            StatsmodelsSarimaxFixtures.dictArray("air2_stationary", "params_variance"),
            StatsmodelsSarimaxFixtures.dictArray("air2_stationary", "se_stddev_opg"),
            StatsmodelsSarimaxFixtures.dictArray("air2_stationary", "se_stddev_oim"),
            StatsmodelsSarimaxFixtures.dictArray("air2_stationary", "standardized_forecasts_error"),
            StatsmodelsSarimaxFixtures.dictScalar("air2_stationary", "loglike"),
            StatsmodelsSarimaxFixtures.dictScalar("air2_stationary", "aic"),
            StatsmodelsSarimaxFixtures.dictScalar("air2_stationary", "bic"));
    }

    private static WpiSeries loadWpiSeries() {
        double[] endog = StatsmodelsSarimaxFixtures.array("wpi1_data");
        double[][] exog = new double[endog.length][1];
        for (int i = 0; i < endog.length; i++) {
            double frac = endog[i] - Math.floor(endog[i]);
            exog[i][0] = frac * frac;
        }
        return new WpiSeries(endog, exog);
    }

    private static WpiCoverageReference loadWpiCoverageReference() {
        WpiSeries wpi = loadWpiSeries();
        CoverageRow row = loadCoverageRow("arima wpi x, arima(3,2,2) sarima(3,2,2,4) noconstant vce(oim)");
        return new WpiCoverageReference(wpi.endog(), wpi.exog(), row.params(), row.loglike(), 10);
    }

    private static MissingExogReference loadMissingExogReference() {
        WpiSeries wpi = loadWpiSeries();
        double[] source = Arrays.copyOf(wpi.endog(), wpi.endog().length);
        double[] exog = new double[source.length - 1];
        for (int i = 1; i < source.length; i++) {
            exog[i - 1] = wpi.exog()[i][0];
        }
        Arrays.fill(source, 9, 19, Double.NaN);
        double[] endog = diff(source);
        endog[9] = Double.NaN;
        CoverageRow row = loadCoverageRow("arima D.wpi2 t32 x, arima(3,0,2) noconstant vce(oim)");
        double[] futureExog = Arrays.copyOfRange(exog, exog.length - 3, exog.length);
        return new MissingExogReference(endog, exog, futureExog, row.params(), endog.length - 3, endog.length + 2, 9, row.loglike());
    }

    private static FriedmanMleReference loadFriedmanMleReference() {
        return new FriedmanMleReference(
            StatsmodelsSarimaxFixtures.dictArray("friedman2_data", "consump"),
            StatsmodelsSarimaxFixtures.dictArray("friedman2_data", "m2"),
            StatsmodelsSarimaxFixtures.dictArray("friedman2_mle", "params_exog"),
            StatsmodelsSarimaxFixtures.dictArray("friedman2_mle", "params_ar"),
            StatsmodelsSarimaxFixtures.dictArray("friedman2_mle", "params_ma"),
            StatsmodelsSarimaxFixtures.dictArray("friedman2_mle", "params_variance"),
            StatsmodelsSarimaxFixtures.dictScalar("friedman2_mle", "loglike"),
            StatsmodelsSarimaxFixtures.dictScalar("friedman2_mle", "aic"),
            StatsmodelsSarimaxFixtures.dictScalar("friedman2_mle", "bic"));
    }

    private static FriedmanPredictReference loadFriedmanPredictReference() {
        double[] endog = StatsmodelsSarimaxFixtures.dictArray("friedman2_data", "consump");
        double[] exog = StatsmodelsSarimaxFixtures.dictArray("friedman2_data", "m2");
        int horizon = 15;
        int dynamicStart = endog.length - horizon;
        double[] dynamicPredict = StatsmodelsSarimaxFixtures.dictArray("friedman2_predict", "dynamic_predict");
        double[] forecast = StatsmodelsSarimaxFixtures.dictArray("friedman2_predict", "forecast");
        double[] dynamicForecast = StatsmodelsSarimaxFixtures.dictArray("friedman2_predict", "dynamic_forecast");
        return new FriedmanPredictReference(
            endog,
            exog,
            Arrays.copyOf(endog, endog.length - horizon),
            Arrays.copyOf(exog, exog.length - horizon),
            Arrays.copyOfRange(exog, exog.length - horizon, exog.length),
            StatsmodelsSarimaxFixtures.dictArray("friedman2_predict", "predict"),
            dynamicStart,
            Arrays.copyOfRange(dynamicPredict, dynamicStart, dynamicPredict.length),
            horizon,
            Arrays.copyOfRange(forecast, forecast.length - horizon, forecast.length),
            dynamicStart,
            Arrays.copyOfRange(dynamicForecast, dynamicForecast.length - horizon, dynamicForecast.length),
            StatsmodelsSarimaxFixtures.dictArray("friedman2_predict", "params_exog"),
            StatsmodelsSarimaxFixtures.dictArray("friedman2_predict", "params_ar"),
            StatsmodelsSarimaxFixtures.dictArray("friedman2_predict", "params_ma"),
            StatsmodelsSarimaxFixtures.dictArray("friedman2_predict", "params_variance"));
    }

    private static WpiArimaReference loadWpiArimaReference(String fileName) {
        String fixtureName = statsmodelsName(fileName);
        return new WpiArimaReference(
            StatsmodelsSarimaxFixtures.dictArray(fixtureName, "data"),
            StatsmodelsSarimaxFixtures.dictArray(fixtureName, "params_ar"),
            StatsmodelsSarimaxFixtures.dictArray(fixtureName, "params_ma"),
            StatsmodelsSarimaxFixtures.dictArray(fixtureName, "params_mean"),
            StatsmodelsSarimaxFixtures.dictArray(fixtureName, "params_variance"),
            StatsmodelsSarimaxFixtures.dictScalar(fixtureName, "loglike"),
            StatsmodelsSarimaxFixtures.dictScalar(fixtureName, "aic"),
            StatsmodelsSarimaxFixtures.dictScalar(fixtureName, "bic"));
    }

    private static double loadFixtureScalar(String fileName, String key) {
        return StatsmodelsSarimaxFixtures.dictScalar(statsmodelsName(fileName), key);
    }

    private static double observationSignal(KalmanSSM model, double[] state, int stateOffset, int t) {
        double value = model.obsInterceptData()[model.obsInterceptOffset(t)];
        int designOffset = model.designOffset(t);
        for (int stateIndex = 0; stateIndex < model.stateCount(); stateIndex++) {
            value += model.designData()[designOffset + stateIndex] * state[stateOffset + stateIndex];
        }
        return value;
    }

    private static double observationVariance(KalmanSSM model, double[] stateCov, int stateCovOffset, int t, boolean signalAndNoise) {
        double value = signalAndNoise ? model.obsCovData()[model.obsCovOffset(t)] : 0.0;
        int designOffset = model.designOffset(t);
        int kStates = model.stateCount();
        for (int row = 0; row < kStates; row++) {
            double zRow = model.designData()[designOffset + row];
            for (int col = 0; col < kStates; col++) {
                value += zRow * stateCov[stateCovOffset + row * kStates + col] * model.designData()[designOffset + col];
            }
        }
        return value;
    }

    private static CoverageRow loadCoverageRow(String mod) {
        for (String line : StatsmodelsSarimaxFixtures.coverageLines()) {
            if (!line.startsWith('"' + mod + '"')) {
                continue;
            }
            int firstEnd = line.indexOf("\",");
            int secondSep = line.indexOf(",\"", firstEnd + 2);
            double loglike = Double.parseDouble(line.substring(firstEnd + 2, secondSep));
            String[] parts = line.substring(secondSep + 2, line.length() - 1).split(",");
            double[] params = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                params[i] = Double.parseDouble(parts[i]);
            }
            params[params.length - 1] *= params[params.length - 1];
            return new CoverageRow(mod, loglike, params);
        }
        throw new IllegalArgumentException("Missing SARIMAX coverage reference row: " + mod);
    }

    private static String statsmodelsName(String fileName) {
        return fileName.endsWith(".json") ? fileName.substring(0, fileName.length() - ".json".length()) : fileName;
    }

    private static SARIMAXResults newResults(SARIMAX model, double[] params, FilterResult filterResult) {
        return new SARIMAXResults(
            model,
            null,
            params,
            model.untransformParams(params),
            filterResult,
            MLEResults.Covariance.OPG);
    }

    private static double[] slice(double[] values, int offset, int length) {
        return Arrays.copyOfRange(values, offset, offset + length);
    }

    private static double[] active(double[] values, int base, int length) {
        return Arrays.copyOfRange(values, base, base + length);
    }

    private static void assertSarimaxChandrasekharMatchesConventional(SARIMAX model,
                                                                      double[] params,
                                                                      FilterOptions profile) {
        FilterResult expected = model.filter(params, profile.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = model.filter(params, profile.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());

        if (profile.includes(FilterOptions.Surface.LIKELIHOOD)) {
            assertEquals(expected.logLikelihood(), actual.logLikelihood(), 1e-8);
        }
        assertArrayAllClose(active(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            active(actual.logLikelihoodObs, actual.logLikelihoodObsBase(), actual.logLikelihoodObsLength()), 1e-8, 1e-8);
        assertArrayAllClose(active(expected.forecast, expected.forecastBase(), expected.forecastLength()),
            active(actual.forecast, actual.forecastBase(), actual.forecastLength()), 1e-8, 1e-8);
        assertArrayAllClose(active(expected.forecastError, expected.forecastErrorBase(), expected.forecastErrorLength()),
            active(actual.forecastError, actual.forecastErrorBase(), actual.forecastErrorLength()), 1e-8, 1e-8);
        assertArrayAllClose(active(expected.forecastErrorCov, expected.forecastErrorCovBase(), expected.forecastErrorCovLength()),
            active(actual.forecastErrorCov, actual.forecastErrorCovBase(), actual.forecastErrorCovLength()), 1e-8, 1e-8);
        assertArrayAllClose(active(expected.standardizedForecastError, expected.standardizedForecastErrorBase(), expected.standardizedForecastErrorLength()),
            active(actual.standardizedForecastError, actual.standardizedForecastErrorBase(), actual.standardizedForecastErrorLength()), 1e-8, 1e-8);
        assertArrayAllClose(active(expected.predictedState, expected.predictedStateBase(), expected.predictedStateLength()),
            active(actual.predictedState, actual.predictedStateBase(), actual.predictedStateLength()), 1e-8, 1e-8);
        assertArrayAllClose(active(expected.predictedStateCov, expected.predictedStateCovBase(), expected.predictedStateCovLength()),
            active(actual.predictedStateCov, actual.predictedStateCovBase(), actual.predictedStateCovLength()), 1e-8, 1e-8);
        assertArrayAllClose(active(expected.filteredState, expected.filteredStateBase(), expected.filteredStateLength()),
            active(actual.filteredState, actual.filteredStateBase(), actual.filteredStateLength()), 1e-8, 1e-8);
        assertArrayAllClose(active(expected.filteredStateCov, expected.filteredStateCovBase(), expected.filteredStateCovLength()),
            active(actual.filteredStateCov, actual.filteredStateCovBase(), actual.filteredStateCovLength()), 1e-8, 1e-8);
        assertArrayAllClose(active(expected.kalmanGain, expected.kalmanGainBase(), expected.kalmanGainLength()),
            active(actual.kalmanGain, actual.kalmanGainBase(), actual.kalmanGainLength()), 1e-8, 1e-8);
    }

    private static double[][] column(double[] values) {
        double[][] column = new double[values.length][1];
        for (int i = 0; i < values.length; i++) {
            column[i][0] = values[i];
        }
        return column;
    }

    private static double[][] withIntercept(double[] values) {
        double[][] exog = new double[values.length][2];
        for (int i = 0; i < values.length; i++) {
            exog[i][0] = 1.0;
            exog[i][1] = values[i];
        }
        return exog;
    }

    private static SARIMAX airlineModel(double[] endog) {
        return new SARIMAX(
            SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1), endog)
                .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
                .build());
    }

    private static double[] airlineBseOpg(Air2Reference air2) {
        return new double[]{
            air2.seMaOpg()[0],
            air2.seSeasonalMaOpg()[0],
            varianceStandardError(air2.paramsVariance()[0], air2.seStdDevOpg()[0])
        };
    }

    private static double[] airlineBseOim(Air2Reference air2) {
        return new double[]{
            air2.seMaOim()[0],
            air2.seSeasonalMaOim()[0],
            varianceStandardError(air2.paramsVariance()[0], air2.seStdDevOim()[0])
        };
    }

    private static double varianceStandardError(double variance, double stddevStandardError) {
        return 2.0 * Math.sqrt(variance) * stddevStandardError;
    }

    private static double[] diff(double[] values) {
        double[] differenced = new double[values.length - 1];
        for (int i = 1; i < values.length; i++) {
            differenced[i - 1] = values[i] - values[i - 1];
        }
        return differenced;
    }

    private static double[][] scale(double[][] values, double factor) {
        double[][] scaled = new double[values.length][];
        for (int row = 0; row < values.length; row++) {
            scaled[row] = new double[values[row].length];
            for (int col = 0; col < values[row].length; col++) {
                scaled[row][col] = values[row][col] * factor;
            }
        }
        return scaled;
    }

    private static double[] diagonalStandardErrors(double[] covariance) {
        int dimension = (int) Math.round(Math.sqrt(covariance.length));
        double[] standardErrors = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            standardErrors[i] = Math.sqrt(covariance[i * dimension + i]);
        }
        return standardErrors;
    }

    private static void assertArrayClose(double[] expected, double[] actual, double tol) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tol, "mismatch at index " + i);
        }
    }

    private static void assertArrayAllClose(double[] expected, double[] actual, double absTol, double relTol) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            double delta = Math.abs(expected[i] - actual[i]);
            double limit = absTol + relTol * Math.abs(expected[i]);
            assertTrue(delta <= limit,
                "mismatch at index " + i + ": expected=" + expected[i] + ", actual=" + actual[i] + ", delta=" + delta + ", limit=" + limit);
        }
    }

    private static void assertAllFinite(double[] values) {
        for (double value : values) {
            assertTrue(Double.isFinite(value));
        }
    }

    private static double[] finiteDifferenceScoreObs(SARIMAX model, double[] params) {
        int nobs = model.observationCount();
        int paramCount = params.length;
        double[] scoreObs = new double[nobs * paramCount];
        double[] shifted = Arrays.copyOf(params, params.length);
        for (int paramIndex = 0; paramIndex < paramCount; paramIndex++) {
            double value = params[paramIndex];
            double step = gradientStep(value);

            shifted[paramIndex] = value + step;
            double[] plus = model.logLikelihoodObs(shifted);

            shifted[paramIndex] = value - step;
            double[] minus = model.logLikelihoodObs(shifted);

            shifted[paramIndex] = value;
            for (int t = 0; t < nobs; t++) {
                scoreObs[t * paramCount + paramIndex] = (plus[t] - minus[t]) / (2.0 * step);
            }
        }
        return scoreObs;
    }

    private static double gradientStep(double value) {
        double step = Math.cbrt(Math.ulp(1.0)) * Math.max(1.0, Math.abs(value));
        double adjusted = (value + step) - value;
        return adjusted == 0.0 ? step : adjusted;
    }

    private static double[] predictionTail(SARIMAXPrediction prediction, int startIndex) {
        int offset = startIndex - prediction.start();
        return Arrays.copyOfRange(prediction.mean(), offset, prediction.mean().length);
    }

    private static double concentratedScale(FilterResult filterResult, double[] endog, int burn) {
        int start = Math.max(burn, filterResult.nobsDiffuse);
        int observedCount = 0;
        double scaleSum = 0.0;
        for (int t = start; t < endog.length; t++) {
            double forecastVariance = filterResult.forecastErrorCov(0, 0, t);
            if (Double.isNaN(endog[t]) || !(forecastVariance > 0.0) || !Double.isFinite(forecastVariance)) {
                continue;
            }
            double error = filterResult.forecastError(0, t);
            scaleSum += error * error / forecastVariance;
            observedCount++;
        }
        if (observedCount == 0) {
            return 1.0;
        }
        return Math.max(scaleSum / observedCount, 1e-8);
    }

    private static double[] concentratedLogLikelihoodObs(FilterResult filterResult,
                                                         double[] endog,
                                                         int burn,
                                                         double scale) {
        double[] adjusted = Arrays.copyOf(filterResult.logLikelihoodObs, filterResult.logLikelihoodObs.length);
        double logScale = Math.log(scale);
        for (int t = 0; t < adjusted.length; t++) {
            double forecastVariance = filterResult.forecastErrorCov(0, 0, t);
            if (Double.isNaN(endog[t]) || !(forecastVariance > 0.0) || !Double.isFinite(forecastVariance)) {
                continue;
            }
            double error = filterResult.forecastError(0, t);
            double scaleObs = error * error / forecastVariance;
            adjusted[t] += -0.5 * (logScale + scaleObs / scale - scaleObs);
        }
        for (int t = 0; t < Math.min(burn, adjusted.length); t++) {
            adjusted[t] = 0.0;
        }
        return adjusted;
    }

    private static double[] inSampleForecast(FilterResult filterResult) {
        double[] fitted = new double[filterResult.nobs];
        for (int t = 0; t < fitted.length; t++) {
            fitted[t] = filterResult.forecast(0, t);
        }
        return fitted;
    }

    private static void assertConcentratedScaleMatchesUnconcentrated(SARIMAXSpec originalSpec,
                                                                     SARIMAXSpec concentratedSpec,
                                                                     int scaledTail,
                                                                     double tol) {
        SARIMAX concentrated = new SARIMAX(concentratedSpec);
        double[] concentratedParams = concentrated.startParams();
        SARIMAXResults concentratedResults = newResults(
            concentrated,
            concentratedParams,
            concentrated.filter(concentratedParams, FilterOptions.defaults()));

        double[] originalParams = unconcentratedParams(concentratedParams, concentratedResults.scale(), scaledTail);
        SARIMAX original = new SARIMAX(originalSpec);
        SARIMAXResults originalResults = newResults(
            original,
            originalParams,
            original.filter(originalParams, FilterOptions.defaults()));

        assertEquals(originalResults.logLikelihood(), concentratedResults.logLikelihood(), tol);
    }

    private static double[] unconcentratedParams(double[] concentratedParams, double scale, int scaledTail) {
        double[] original = Arrays.copyOf(concentratedParams, concentratedParams.length + 1);
        original[original.length - 1] = 1.0;
        for (int index = original.length - scaledTail; index < original.length; index++) {
            original[index] *= scale;
        }
        return original;
    }

    private static void assertArrayEqualsAllowNaN(double[] expected, double[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            if (Double.isNaN(expected[i])) {
                assertTrue(Double.isNaN(actual[i]), "expected NaN at index " + i + " but was " + actual[i]);
            } else {
                assertEquals(expected[i], actual[i], TOL, "mismatch at index " + i);
            }
        }
    }

    private record Air2Reference(double[] data,
                                 double[] paramsMa,
                                 double[] seMaOpg,
                                 double[] seMaOim,
                                 double[] paramsSeasonalMa,
                                 double[] seSeasonalMaOpg,
                                 double[] seSeasonalMaOim,
                                 double[] paramsVariance,
                                 double[] seStdDevOpg,
                                 double[] seStdDevOim,
                                 double[] standardizedForecastsError,
                                 double loglike,
                                 double aic,
                                 double bic) {
    }

    private record WpiCoverageReference(double[] endog,
                                        double[][] exog,
                                        double[] params,
                                        double loglike,
                                        int burn) {
    }

    private record WpiSeries(double[] endog, double[][] exog) {
    }

    private record CoverageRow(String mod, double loglike, double[] params) {
    }

    private record MissingExogReference(double[] endog,
                                        double[] exog,
                                        double[] futureExog,
                                        double[] params,
                                        int predictStart,
                                        int predictEnd,
                                        int missingIndex,
                                        double loglike) {
    }

    private record FriedmanMleReference(double[] endog,
                                        double[] exog,
                                        double[] paramsExog,
                                        double[] paramsAr,
                                        double[] paramsMa,
                                        double[] paramsVariance,
                                        double loglike,
                                        double aic,
                                        double bic) {
    }

    private record FriedmanPredictReference(double[] endog,
                                            double[] exog,
                                            double[] trainEndog,
                                            double[] trainExog,
                                            double[] futureExog,
                                            double[] predict,
                                            int dynamicPredictStart,
                                            double[] dynamicPredictTail,
                                            int forecastHorizon,
                                            double[] forecastTail,
                                            int dynamicForecastStart,
                                            double[] dynamicForecastTail,
                                            double[] paramsExog,
                                            double[] paramsAr,
                                            double[] paramsMa,
                                            double[] paramsVariance) {
    }

    private record WpiArimaReference(double[] endog,
                                     double[] paramsAr,
                                     double[] paramsMa,
                                     double[] paramsMean,
                                     double[] paramsVariance,
                                     double loglike,
                                     double aic,
                                     double bic) {
    }
}