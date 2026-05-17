package com.curioloop.yum4j.tsa.varmax;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.mle.FixedParameters;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.ssm.kalman.mle.MLEResults;
import com.curioloop.yum4j.tsa.prediction.PredictionInformationSet;
import com.curioloop.yum4j.tsa.prediction.PredictionKind;
import com.curioloop.yum4j.tsa.statespace.ForecastErrorVarianceDecomposition;
import com.curioloop.yum4j.tsa.statespace.ImpulseResponse;
import com.curioloop.yum4j.tsa.statespace.ImpulseResponseRepetitions;
import com.curioloop.yum4j.tsa.statespace.SimulationAnchor;
import com.curioloop.yum4j.tsa.statespace.SimulationSmootherRepetitions;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VARMAXTest {

    private static final double TOL = 1e-8;

    @Test
    void specValidatesMultivariateInputs() {
        assertThrows(IllegalArgumentException.class,
            () -> VARMAXSpec.builder(VARMAXOrder.of(1, 0), new double[][]{{1.0}, {2.0}}).build());
        assertThrows(IllegalArgumentException.class,
            () -> VARMAXSpec.builder(VARMAXOrder.of(0, 0), sampleEndog()).build());
        assertThrows(IllegalArgumentException.class,
            () -> VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
                .exog(new double[][]{{1.0}})
                .build());
        assertThrows(IllegalArgumentException.class,
            () -> VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
                .observationIntercept(1.0)
                .build());
        assertThrows(IllegalArgumentException.class,
            () -> VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
                .observationIntercept(1.0, Double.NaN)
                .build());
    }

    @Test
    void parameterNamesAndCountsFollowStatsmodelsOrder() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 1), sampleEndog())
            .trend("ct")
            .exog(sampleExog())
            .errorCovariance(VARMAXErrorCovariance.UNSTRUCTURED)
            .measurementError(true)
            .endogNames("inv", "inc")
            .exogNames("x")
            .build());

        assertEquals(19, model.paramCount());
        assertArrayEquals(new String[]{
            "intercept.inv", "drift.inv", "intercept.inc", "drift.inc",
            "L1.inv.inv", "L1.inc.inv", "L1.inv.inc", "L1.inc.inc",
            "L1.e(inv).inv", "L1.e(inc).inv", "L1.e(inv).inc", "L1.e(inc).inc",
            "beta.x.inv", "beta.x.inc",
            "sqrt.var.inv", "sqrt.cov.inv.inc", "sqrt.var.inc",
            "measurement_variance.inv", "measurement_variance.inc"
        }, model.parameterNames());
    }

    @Test
    void startParamsTransformAndFilterAreUsable() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());

        double[] start = model.startParams();
        assertEquals(model.paramCount(), start.length);
        for (double value : start) {
            assertTrue(Double.isFinite(value));
        }
        assertArrayEquals(start, model.transformParams(model.untransformParams(start)), 1e-6);
        assertTrue(Double.isFinite(model.logLikelihood(start)));
        assertEquals(sampleEndog().length, model.logLikelihoodObs(start).length);
    }

    @Test
    void diagonalVarianceTransformUsesStatsmodelsSquare() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .measurementError(true)
            .build());

        double[] transformed = model.transformParams(new double[]{
            0.0, 0.0, 0.0, 0.0,
            0.0, -2.0,
            0.0, 3.0
        });

        assertEquals(0.0, transformed[4], 0.0);
        assertEquals(4.0, transformed[5], TOL);
        assertEquals(0.0, transformed[6], 0.0);
        assertEquals(9.0, transformed[7], TOL);
        assertArrayEquals(new double[]{0.0, 2.0, 0.0, 3.0},
            new double[]{
                model.untransformParams(transformed)[4],
                model.untransformParams(transformed)[5],
                model.untransformParams(transformed)[6],
                model.untransformParams(transformed)[7]
            }, TOL);
    }

    @Test
    void observationInterceptIsFixedModelMetadata() {
        VARMAX baseline = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .observationIntercept(0.40, -0.20)
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.0, 0.0, 0.0, 0.0, 0.30, 0.20};

        assertEquals(baseline.paramCount(), model.paramCount());
        assertArrayEquals(baseline.parameterNames(), model.parameterNames());
        assertArrayEquals(new double[]{0.40, -0.20}, model.spec().observationIntercept(), TOL);

        KalmanSSM snapshot = model.snapshotModel(params);
    assertObservationIntercept(snapshot, 0.40, -0.20);
        assertEquals(0.0, snapshot.stateInterceptData()[0], TOL);
        assertEquals(0.0, snapshot.stateInterceptData()[1], TOL);

        VARMAXPrediction forecast = newResults(model, params, model.filter(params)).forecast(1);
        assertArrayEquals(new double[]{0.40, -0.20}, forecast.mean()[0], 1e-7);
    }

    @Test
    void exogModelUsesNanFinalStateInterceptWithoutFutureExog() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .exog(sampleExog())
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.35, 0.05, 0.10, 0.25, 0.20, -0.10, 0.25, 0.25};

        KalmanSSM snapshot = model.snapshotModel(params);
        int lastOffset = snapshot.stateInterceptOffset(snapshot.observationCount() - 1);
        assertTrue(Double.isNaN(snapshot.stateInterceptData()[lastOffset]));
        assertTrue(Double.isNaN(snapshot.stateInterceptData()[lastOffset + 1]));
    }

    @Test
    void constantOnlyTrendKeepsStateInterceptTimeInvariant() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("c")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.05, -0.02, 0.35, 0.05, -0.08, 0.25, 0.30, 0.20};

        KalmanSSM snapshot = model.snapshotModel(params);

        assertFalse(model.stateInterceptTimeVarying());
        assertEquals(0.05, snapshot.stateInterceptData()[0], TOL);
        assertEquals(-0.02, snapshot.stateInterceptData()[1], TOL);
    }

    @Test
    void filterSmoothAndPredictionReturnMultivariateSurfaces() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.45, 0.20, -0.10, 0.30, 0.30, 0.20};

        FilterResult filter = model.filter(params, FilterOptions.defaults());
        assertEquals(2, filter.kEndog);
        assertTrue(filter.forecastLength() > 0);
        assertTrue(filter.predictedStateLength() > 0);

        SmootherResult smoother = model.smooth(params, SmootherOptions.builder()
            .retainOnly(SmootherOptions.Surface.STATE, SmootherOptions.Surface.STATE_COVARIANCE)
            .build());
        assertTrue(smoother.smoothedStateLength() > 0);
        assertTrue(smoother.smoothedStateCovLength() > 0);

        VARMAXResults results = new VARMAXResults(model, null, params, model.untransformParams(params), filter, com.curioloop.yum4j.ssm.kalman.mle.MLEResults.Covariance.OPG);
        VARMAXPrediction prediction = results.predict(0, 3);
        assertEquals(2, prediction.observationDimension());
        assertEquals(4, prediction.mean().length);
        assertEquals(2, prediction.variance()[0].length);
    }

    @Test
    void predictionMetadataAndCompactResultSemantics() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.45, 0.20, -0.10, 0.30, 0.30, 0.20};
        VARMAXResults results = newResults(model, params, model.filter(params));

        VARMAXPrediction inSample = results.predict(1, 3);
        assertEquals(PredictionKind.IN_SAMPLE, inSample.kind());
        assertEquals(PredictionInformationSet.PREDICTED, inSample.informationSet());

        assertEquals(PredictionKind.DYNAMIC_IN_SAMPLE, results.predict(1, 4, 2, null).kind());
        assertEquals(PredictionKind.MIXED, results.predict(6, 9).kind());
        assertEquals(PredictionKind.OUT_OF_SAMPLE, results.predict(8, 9).kind());
        assertEquals(PredictionKind.FORECAST, results.forecast(2).kind());

        VARMAXResults compactResults = newResults(model, params, model.filter(params, FilterOptions.compact()));
        assertThrows(IllegalArgumentException.class, () -> compactResults.predict(0, 2));
        assertEquals(PredictionKind.DYNAMIC_IN_SAMPLE, compactResults.predict(1, 4, 2, null).kind());
        assertEquals(PredictionKind.FORECAST, compactResults.forecast(2).kind());
    }

    @Test
    void resultSmoothingCanRetainStateAutocovariance() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(2, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {
            0.35, 0.05, -0.02, 0.20,
            -0.08, 0.18, 0.04, 0.22,
            0.30, 0.20
        };
        VARMAXResults results = newResults(model, params, model.filter(params));

        SmootherResult stateOnly = results.smooth(SmootherOptions.stateOnly());
        assertEquals(0, stateOnly.smoothedStateAutocovarianceLength());

        SmootherResult withAutocovariance = results.smooth(SmootherOptions.defaults()
            .with(SmootherOptions.Surface.STATE_AUTOCOVARIANCE));
        assertEquals(withAutocovariance.nobs * withAutocovariance.kStates * withAutocovariance.kStates,
            withAutocovariance.smoothedStateAutocovarianceLength());
        assertTrue(Double.isFinite(withAutocovariance.smoothedStateAutocovariance(0, 0, 3)));
    }

    @Test
    void predictionInformationSetsUseFilteredAndSmoothedStateSurfaces() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.45, 0.20, -0.10, 0.30, 0.30, 0.20};
        KalmanSSM snapshot = model.snapshotModel(params);
        FilterResult filter = model.filter(params, FilterOptions.defaults());
        SmootherResult smoother = model.smooth(params, SmootherOptions.defaults());
        VARMAXResults results = newResults(model, params, filter);

        VARMAXPrediction filtered = results.predict(2, 4, PredictionInformationSet.FILTERED, true);
        VARMAXPrediction smoothed = results.predict(2, 4, PredictionInformationSet.SMOOTHED, true);

        assertTrue(filtered.signalOnly());
        assertTrue(smoothed.signalOnly());
        assertEquals(PredictionInformationSet.FILTERED, filtered.informationSet());
        assertEquals(PredictionInformationSet.SMOOTHED, smoothed.informationSet());
        for (int i = 0; i < filtered.mean().length; i++) {
            int t = 2 + i;
            assertArrayEquals(observationSignal(snapshot, filter.filteredState, filter.filteredStateOffset(t), t),
                filtered.mean()[i], TOL);
            assertMatrixEquals(observationVariance(snapshot, filter.filteredStateCov, filter.filteredStateCovOffset(t), t, false),
                filtered.variance()[i]);
            assertArrayEquals(observationSignal(snapshot, smoother.smoothedState, smoother.smoothedStateOffset(t), t),
                smoothed.mean()[i], TOL);
            assertMatrixEquals(observationVariance(snapshot, smoother.smoothedStateCov, smoother.smoothedStateCovOffset(t), t, false),
                smoothed.variance()[i]);
        }
    }

    @Test
    void filterResultIsDefensivelyCopied() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.45, 0.20, -0.10, 0.30, 0.30, 0.20};
        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        VARMAXResults results = newResults(model, params, filterResult);
        double[] expectedLoglikeObs = filterResult.logLikelihoodObs.clone();
        double[] expectedForecast = filterResult.forecast.clone();

        filterResult.logLikelihoodObs[0] = Double.POSITIVE_INFINITY;
        filterResult.forecast[0] = Double.POSITIVE_INFINITY;

        FilterResult firstCopy = results.filterResult();
        assertArrayEquals(expectedLoglikeObs, firstCopy.logLikelihoodObs, TOL);
        assertArrayEquals(expectedForecast, firstCopy.forecast, TOL);

        firstCopy.logLikelihoodObs[0] = Double.NEGATIVE_INFINITY;
        firstCopy.forecast[0] = Double.NEGATIVE_INFINITY;

        FilterResult secondCopy = results.filterResult();
        assertArrayEquals(expectedLoglikeObs, secondCopy.logLikelihoodObs, TOL);
        assertArrayEquals(expectedForecast, secondCopy.forecast, TOL);
    }

    @Test
    void lifecycleSimulationAndImpulseResponseWrappersAreUsable() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.45, 0.20, -0.10, 0.30, 0.30, 0.20};
        VARMAXResults results = newResults(model, params, model.filter(params, FilterOptions.defaults()));
        double[][] newRows = {{0.18, 0.09}, {0.12, 0.05}};

        VARMAXResults appended = results.append(newRows);
        VARMAXResults extended = results.extend(newRows);
        VARMAXResults applied = results.apply(newRows);
        assertEquals(sampleEndog().length + newRows.length, appended.filterResult().nobs);
        assertEquals(newRows.length, extended.filterResult().nobs);
        assertEquals(newRows.length, applied.filterResult().nobs);
        assertAllFinite(extended.fittedValues());
        assertAllFinite(applied.fittedValues());

        SimulationSmootherResult simulation = results.simulate(3,
            new double[3 * model.observationDimension()],
            new double[3 * model.snapshotModel(params).stateDisturbanceCount()],
            new double[model.stateCount()]);
        assertEquals(3, simulation.nobs);
        assertEquals(3 * model.observationDimension(), simulation.generatedObsLength());

        ImpulseResponse irf = results.impulseResponses(3, 0);
        ImpulseResponse cumulative = results.impulseResponses(3, 0, false, true);
        assertEquals(3, irf.steps());
        assertEquals(model.observationDimension(), irf.responseCount());
        assertEquals(irf.response(0)[0][0] + irf.response(1)[0][0], cumulative.response(1)[0][0], TOL);
    }

    @Test
    void statsmodelsGapWrappersAreUsable() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.45, 0.20, -0.10, 0.30, 0.30, 0.20};
        VARMAXResults results = newResults(model, params, model.filter(params, FilterOptions.defaults()));

        VARMAXResults fixed = model.fit(VARMAXFitOptions.builder()
            .fixedParameters(FixedParameters.of(new int[]{0, 1, 2, 3, 4, 5}, params))
            .build());
        assertEquals(params.length, fixed.fixedParameterCount());
        assertEquals(0, fixed.freeParameterCount());
        assertArrayEquals(params, fixed.params(), TOL);
        assertTrue(Double.isNaN(fixed.covParams()[0]));

        double[][] newRows = {{0.18, 0.09}, {0.12, 0.05}};
        VARMAXResults refit = results.appendRefit(newRows, null,
            VARMAXFitOptions.builder().fixedParameters(FixedParameters.of(new int[]{0, 1, 2, 3, 4, 5}, params)).build());
        assertArrayEquals(params, refit.params(), TOL);
        assertEquals(sampleEndog().length + newRows.length, refit.filterResult().nobs);

        assertEquals(model.observationDimension(), results.testNormality().length);
        assertEquals(model.observationDimension(), results.testHeteroskedasticity().length);
        assertEquals(2, results.testSerialCorrelation(2)[0].lags().length);

        SimulationSmootherRepetitions simulations = results.simulateRepetitions(2, 2,
            SimulationAnchor.FINAL, new Random(20260616L), null);
        assertEquals(2, simulations.repetitions());
        assertEquals(2, simulations.nsimulations());

        ImpulseResponse anchored = results.impulseResponses(2, 0, false, false, 0);
        ImpulseResponseRepetitions irfRuns = results.impulseResponseRepetitions(2, 0, 2, false, false);
        assertEquals(2, anchored.steps());
        assertEquals(2, irfRuns.repetitions());

        ForecastErrorVarianceDecomposition fevd = results.fevd(2);
        assertEquals(2, fevd.steps());
        assertEquals(model.observationDimension(), fevd.responseCount());
        assertTrue(Double.isFinite(fevd.share(1, 0, 0)) || Double.isNaN(fevd.share(1, 0, 0)));

        assertNotNull(results.testCausality(0, 1, 1).lag(1));
        assertEquals(1, results.testInstantaneousCausality(0, 1).degreesOfFreedom());

        VARMAXResults updated = results.append(newRows);
        assertEquals(results.filterResult().nobs, results.news(updated).targetCount());
        assertEquals(results.filterResult().nobs, results.smoothedDecomposition().nobs());
    }

    @Test
    void transitionMatrixIncludesCrossLagCoupling() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.40, 0.25, -0.15, 0.35, 0.30, 0.20};

        KalmanSSM snapshot = model.snapshotModel(params);
        double[] transition = snapshot.transitionData();
        assertEquals(0.40, transition[0], TOL);
        assertEquals(0.25, transition[1], TOL);
        assertEquals(-0.15, transition[snapshot.stateCount()], TOL);
        assertEquals(0.35, transition[snapshot.stateCount() + 1], TOL);
    }

    @Test
    void forecastExtendsWithFutureExog() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .exog(sampleExog())
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.35, 0.05, 0.10, 0.25, 0.20, -0.10, 0.25, 0.25};
        FilterResult filter = assertDoesNotThrow(() -> model.filter(params));
        VARMAXResults results = new VARMAXResults(model, null, params, model.untransformParams(params), filter, com.curioloop.yum4j.ssm.kalman.mle.MLEResults.Covariance.OPG);

        VARMAXPrediction forecast = results.forecast(2, new double[][]{{0.7}, {0.8}});
        assertEquals(2, forecast.mean().length);
        assertEquals(2, forecast.mean()[0].length);
        assertNotNull(forecast.confInt());
    }

    @Test
    void finalAnchorSimulationWithFutureExogIsFinite() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .exog(sampleExog())
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.35, 0.05, 0.10, 0.25, 0.20, -0.10, 0.25, 0.25};
        VARMAXResults results = newResults(model, params, model.filter(params));

        SimulationSmootherResult simulation = results.simulate(3,
            sampleEndog().length,
            new double[][]{{0.8}, {0.9}, {1.0}},
            new Random(20260516L),
            null);

        for (int t = 0; t < simulation.nobs; t++) {
            for (int i = 0; i < simulation.kEndog; i++) {
                assertTrue(Double.isFinite(simulation.generatedObs(i, t)), "generated obs t " + t + " index " + i);
            }
        }
    }

    @Test
    void forecastRejectsFutureExogShapeMismatch() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .exog(sampleExog())
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.35, 0.05, 0.10, 0.25, 0.20, -0.10, 0.25, 0.25};
        VARMAXResults results = newResults(model, params, model.filter(params));

        assertThrows(IllegalArgumentException.class, () -> results.forecast(2, new double[][]{{0.7}}));
        assertThrows(IllegalArgumentException.class, () -> results.forecast(2, new double[][]{{0.7}, {0.8}, {0.9}}));
        assertThrows(IllegalArgumentException.class, () -> results.forecast(2, new double[][]{{0.7, 1.0}, {0.8, 1.0}}));
        assertThrows(IllegalArgumentException.class, () -> results.predict(6, 9, -1, new double[][]{{0.7}}));
        assertThrows(IllegalArgumentException.class, () -> results.predict(6, 9, -1, new double[][]{{0.7}, {0.8}, {0.9}}));
        assertThrows(IllegalArgumentException.class, () -> results.predict(6, 9, -1, new double[][]{{0.7, 1.0}, {0.8, 1.0}}));
    }

    @Test
    void modelWithoutExogRejectsFutureExog() {
        VARMAX model = new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 0), sampleEndog())
            .trend("n")
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build());
        double[] params = {0.45, 0.20, -0.10, 0.30, 0.30, 0.20};
        VARMAXResults results = newResults(model, params, model.filter(params));

        assertThrows(IllegalArgumentException.class, () -> results.forecast(1, new double[][]{{1.0}}));
        assertThrows(IllegalArgumentException.class, () -> results.predict(6, 9, -1, new double[][]{{1.0}, {1.1}}));
    }

    @Test
    void varmaConstructionLogsIdentificationWarning() {
        Logger logger = Logger.getLogger(VARMAX.class.getName());
        CapturingHandler handler = new CapturingHandler();
        boolean useParentHandlers = logger.getUseParentHandlers();
        Level level = logger.getLevel();
        logger.setUseParentHandlers(false);
        logger.setLevel(Level.WARNING);
        logger.addHandler(handler);
        try {
            new VARMAX(VARMAXSpec.builder(VARMAXOrder.of(1, 1), sampleEndog())
                .trend("n")
                .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
                .build());

            assertTrue(handler.message().contains("identification issues"));
        } finally {
            logger.removeHandler(handler);
            logger.setLevel(level);
            logger.setUseParentHandlers(useParentHandlers);
        }
    }

    private static final class CapturingHandler extends Handler {
        private String message = "";

        @Override
        public void publish(LogRecord record) {
            message = record.getMessage();
        }

        @Override
        public void flush() {
        }

        @Override
        public void close() {
        }

        String message() {
            return message;
        }
    }

    private static double[][] sampleEndog() {
        return new double[][]{
            {1.00, 0.50},
            {0.80, 0.45},
            {0.63, 0.32},
            {0.52, 0.25},
            {0.44, 0.21},
            {0.35, 0.18},
            {0.30, 0.12},
            {0.22, 0.10}
        };
    }

    private static double[][] sampleExog() {
        return new double[][]{
            {0.0}, {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}
        };
    }

    private static VARMAXResults newResults(VARMAX model, double[] params, FilterResult filterResult) {
        return new VARMAXResults(model, null, params, model.untransformParams(params), filterResult, MLEResults.Covariance.OPG);
    }

    private static void assertObservationIntercept(KalmanSSM snapshot, double... expected) {
        for (int t = 0; t < snapshot.observationCount(); t++) {
            int offset = snapshot.obsInterceptOffset(t);
            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], snapshot.obsInterceptData()[offset + i], TOL, "t " + t + " index " + i);
            }
        }
    }

    private static double[] observationSignal(KalmanSSM model, double[] state, int stateOffset, int t) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] signal = new double[kEndog];
        int interceptOffset = model.obsInterceptOffset(t);
        int designOffset = model.designOffset(t);
        for (int row = 0; row < kEndog; row++) {
            double value = model.obsInterceptData()[interceptOffset + row];
            for (int col = 0; col < kStates; col++) {
                value += model.designData()[designOffset + row * kStates + col] * state[stateOffset + col];
            }
            signal[row] = value;
        }
        return signal;
    }

    private static double[][] observationVariance(KalmanSSM model, double[] stateCov, int stateCovOffset, int t, boolean signalAndNoise) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[][] variance = new double[kEndog][kEndog];
        int designOffset = model.designOffset(t);
        int obsCovOffset = model.obsCovOffset(t);
        for (int row = 0; row < kEndog; row++) {
            for (int col = 0; col < kEndog; col++) {
                double value = signalAndNoise ? model.obsCovData()[obsCovOffset + row * kEndog + col] : 0.0;
                for (int left = 0; left < kStates; left++) {
                    double zLeft = model.designData()[designOffset + row * kStates + left];
                    for (int right = 0; right < kStates; right++) {
                        value += zLeft * stateCov[stateCovOffset + left * kStates + right]
                            * model.designData()[designOffset + col * kStates + right];
                    }
                }
                variance[row][col] = value;
            }
        }
        return variance;
    }

    private static void assertMatrixEquals(double[][] expected, double[][] actual) {
        assertEquals(expected.length, actual.length);
        for (int row = 0; row < expected.length; row++) {
            assertArrayEquals(expected[row], actual[row], TOL);
        }
    }

    private static void assertAllFinite(double[][] values) {
        for (double[] row : values) {
            for (double value : row) {
                assertTrue(Double.isFinite(value));
            }
        }
    }
}
