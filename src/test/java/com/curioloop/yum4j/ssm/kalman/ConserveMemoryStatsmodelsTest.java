package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.filter.FilterMethod;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ConserveMemoryStatsmodelsTest {

    private static final double TOL = 1e-10;

    @Test
    void noLikelihoodStorageRetainsAggregateLogLikelihood() {
        ModelFixture model = multivariateModelWithMissing();
        InitialState initialState = InitialState.known(new double[]{0.2, -0.1}, new double[]{1.1, 0.05, 0.05, 1.3});

        for (FilterMethod method : new FilterMethod[]{FilterMethod.CONVENTIONAL, FilterMethod.UNIVARIATE}) {
            for (boolean concentrated : new boolean[]{false, true}) {
                FilterOptions fullOptions = FilterOptions.builder()
                    .method(method)
                    .concentratedLikelihood(concentrated)
                    .build();
                FilterOptions noLikelihoodOptions = fullOptions.without(FilterOptions.Surface.LIKELIHOOD);

                FilterResult full = KalmanEngine.filter(model, initialState, fullOptions);
                FilterResult noLikelihood = KalmanEngine.filter(model, initialState, noLikelihoodOptions);

                assertEquals(model.nobs, full.logLikelihoodObsLength());
                assertEquals(0, noLikelihood.logLikelihoodObsLength());
                assertEquals(full.logLikelihood(), noLikelihood.logLikelihood(), TOL,
                    method + " concentrated=" + concentrated);
                assertEquals(full.scale(), noLikelihood.scale(), TOL,
                    method + " concentrated scale=" + concentrated);
                assertEquals(full.scaleObservationCount(), noLikelihood.scaleObservationCount(),
                    method + " concentrated count=" + concentrated);
            }
        }
    }

    @Test
    void compactConserveMemoryRetainsAggregateLogLikelihood() {
        ModelFixture model = multivariateModelWithMissing();
        InitialState initialState = InitialState.known(new double[]{0.2, -0.1}, new double[]{1.1, 0.05, 0.05, 1.3});

        for (FilterMethod method : new FilterMethod[]{FilterMethod.CONVENTIONAL, FilterMethod.UNIVARIATE}) {
            FilterOptions fullOptions = FilterOptions.builder().method(method).build();
            FilterOptions compactOptions = FilterOptions.compact().toBuilder().method(method).build();

            FilterResult full = KalmanEngine.filter(model, initialState, fullOptions);
            FilterResult compact = KalmanEngine.filter(model, initialState, compactOptions);

            assertEquals(0, compact.forecastLength());
            assertEquals(0, compact.forecastErrorLength());
            assertEquals(0, compact.forecastErrorCovLength());
            assertEquals(0, compact.predictedStateLength());
            assertEquals(0, compact.predictedStateCovLength());
            assertEquals(0, compact.filteredStateLength());
            assertEquals(0, compact.filteredStateCovLength());
            assertEquals(0, compact.kalmanGainLength());
            assertEquals(0, compact.logLikelihoodObsLength());
            assertEquals(full.logLikelihood(), compact.logLikelihood(), TOL, method.toString());
        }
    }

    @Test
    void suppressingLikelihoodAfterFilteringPreservesAggregateLogLikelihood() {
        ModelFixture model = multivariateModelWithMissing();
        InitialState initialState = InitialState.known(new double[]{0.2, -0.1}, new double[]{1.1, 0.05, 0.05, 1.3});
        FilterResult result = KalmanEngine.filter(model, initialState, FilterOptions.defaults());
        double expected = result.logLikelihood();

        result.suppressLikelihood();

        assertEquals(0, result.logLikelihoodObsLength());
        assertEquals(expected, result.logLikelihood(), TOL);
    }

    @Test
    void forecastSurfacesRemainAvailableWhenOnlyForecastCovarianceIsSuppressed() {
        ModelFixture model = multivariateModelWithMissing();
        InitialState initialState = InitialState.known(new double[]{0.2, -0.1}, new double[]{1.1, 0.05, 0.05, 1.3});
        FilterOptions noForecastCovariance = FilterOptions.defaults()
            .without(FilterOptions.Surface.FORECAST_COVARIANCE);

        FilterResult full = KalmanEngine.filter(model, initialState, FilterOptions.defaults());
        FilterResult compact = KalmanEngine.filter(model, initialState, noForecastCovariance);

        assertEquals(full.forecastLength(), compact.forecastLength());
        assertEquals(full.forecastErrorLength(), compact.forecastErrorLength());
        assertEquals(0, compact.forecastErrorCovLength());
        for (int t = 0; t < model.nobs; t++) {
            for (int i = 0; i < model.kEndog; i++) {
                assertEquals(full.forecast(i, t), compact.forecast(i, t), TOL,
                    "forecast t=" + t + " i=" + i);
                assertEquals(full.forecastError(i, t), compact.forecastError(i, t), TOL,
                    "forecast error t=" + t + " i=" + i);
            }
        }
    }

    private static ModelFixture multivariateModelWithMissing() {
        int nobs = 12;
        ModelFixture model = new ModelFixture(2, 2, 2, nobs);
        double[] design = {1.0, 0.0, 0.0, 1.0};
        double[] obsCov = {0.65, 0.0, 0.0, 0.85};
        double[] transition = {0.55, 0.12, -0.08, 0.42};
        double[] selection = {1.0, 0.0, 0.0, 1.0};
        double[] stateCov = {0.35, 0.04, 0.04, 0.45};
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsIntercept(new double[]{0.05 * t, -0.03 * t}, t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setStateIntercept(new double[]{0.02, -0.01}, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{Math.sin(0.2 * t), Math.cos(0.15 * t)}, t);
        }
        model.setMissing(new boolean[]{true, false}, 0);
        model.setMissing(new boolean[]{true, true}, 4);
        model.setMissing(new boolean[]{false, true}, 7);
        return model;
    }
}