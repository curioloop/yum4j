package com.curioloop.yum4j.tsa.varmax;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.mle.MLEResults;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.optim.Optimization;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StatsmodelsVARMAXAlignmentTest {

    private static final String FIXTURE_RESOURCE =
        "statsmodels/tsa/statespace/tests/results/varmax_public_alignment.json";
    private static final double TOL = 2e-8;
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
        .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
        .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    static Stream<CaseFixture> statsmodelsCases() {
        return FixtureRoot.load().cases().stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("statsmodelsCases")
    void publicVarmaxApiMatchesStatsmodelsFixtures(CaseFixture fixture) {
        VARMAX model = new VARMAX(spec(fixture));
        assertArrayEquals(fixture.paramNames(), model.parameterNames());
        assertArrayClose(fixture.transformConstrained(), model.transformParams(fixture.transformUnconstrained()), 5e-8);
        assertArrayClose(fixture.untransformConstrained(), model.untransformParams(fixture.transformConstrained()), 5e-8);

        FilterResult filter = model.filter(fixture.params(), FilterOptions.defaults());
        assertEquals(fixture.filter().llf(), filter.logLikelihood(), TOL);
        assertArrayClose(fixture.filter().llfObs(), filter.logLikelihoodObs, TOL);
        assertMatrixClose(fixture.filter().forecast(), surface(filter, Surface.FORECAST), TOL);
        assertMatrixClose(fixture.filter().forecastError(), surface(filter, Surface.FORECAST_ERROR), TOL);
        assertCubeClose(fixture.filter().forecastErrorCov(), covSurface(filter), TOL);
        assertMatrixClose(fixture.filter().predictedState(), predictedState(filter), TOL);
        assertMatrixClose(fixture.filter().filteredState(), filteredState(filter), TOL);
        assertNullableMatrixClose(fixture.filter().standardizedForecastError(), standardizedForecastError(filter), 5e-8);

        VARMAXResults results = results(model, fixture.params(), filter);
        assertMatrixClose(fixture.fittedValues(), results.fittedValues(), TOL);
        assertMatrixClose(fixture.residuals(), results.residuals(), TOL);
        assertNullableMatrixClose(fixture.filter().standardizedForecastError(), results.standardizedForecastError(), 5e-8);
        if (fixture.coefficientMatricesVar() != null) {
            assertMatrixClose(flattenMatrices(fixture.coefficientMatricesVar()), results.coefficientMatricesVAR(), TOL);
        }
        if (fixture.coefficientMatricesVma() != null) {
            assertMatrixClose(flattenMatrices(fixture.coefficientMatricesVma()), results.coefficientMatricesVMA(), TOL);
        }
        VARMAXPrediction forecast = results.forecast(fixture.prediction().forecastMean().length, fixture.futureExog());
        assertMatrixClose(fixture.prediction().forecastMean(), forecast.mean(), TOL);
        assertCubeClose(fixture.prediction().forecastVariance(), forecast.variance(), TOL);
        assertCubeClose(fixture.prediction().confInt(), forecast.confInt(), 5e-8);

        if (fixture.oosPrediction() != null) {
            int start = fixture.endog().length;
            int steps = fixture.oosPrediction().forecastMean().length;
            VARMAXPrediction oosPrediction = results.predict(start, start + steps - 1, -1, fixture.futureExog());
            assertMatrixClose(fixture.oosPrediction().forecastMean(), oosPrediction.mean(), TOL);
            assertCubeClose(fixture.oosPrediction().forecastVariance(), oosPrediction.variance(), TOL);
            assertCubeClose(fixture.oosPrediction().confInt(), oosPrediction.confInt(), 5e-8);
        }

        if (fixture.extend() != null) {
            VARMAXResults extended = results.extend(fixture.extend().endog(), fixture.extend().exog());
            assertMatrixClose(fixture.extend().filter().forecast(), surface(extended.filterResult(), Surface.FORECAST), TOL);
            assertMatrixClose(fixture.extend().filter().forecastError(), surface(extended.filterResult(), Surface.FORECAST_ERROR), TOL);
            assertCubeClose(fixture.extend().filter().forecastErrorCov(), covSurface(extended.filterResult()), TOL);
            assertMatrixClose(fixture.extend().filter().predictedState(), predictedState(extended.filterResult()), TOL);
            assertMatrixClose(fixture.extend().filter().filteredState(), filteredState(extended.filterResult()), TOL);
            assertNullableMatrixClose(fixture.extend().filter().standardizedForecastError(),
                standardizedForecastError(extended.filterResult()), 5e-8);
            assertMatrixClose(fixture.extend().fittedValues(), extended.fittedValues(), TOL);
            assertMatrixClose(fixture.extend().residuals(), extended.residuals(), TOL);
        }

        if (fixture.simulateFinal() != null) {
            SimulationFixture expected = fixture.simulateFinal();
            int nsimulations = expected.generatedObs().length;
            SimulationSmootherResult simulation = results.simulate(nsimulations,
                fixture.endog().length,
                expected.exog(),
                new double[nsimulations * model.observationDimension()],
                new double[nsimulations * model.snapshotModel(fixture.params()).stateDisturbanceCount()],
                new double[model.stateCount()],
                SimulationSmootherOptions.defaults().withGeneratedOutputs());
            assertMatrixClose(expected.generatedObs(), generatedObs(simulation), TOL);
            assertAllFinite(generatedObs(simulation));
        }

        DynamicPredictionFixture dynamic = fixture.dynamicPrediction();
        VARMAXPrediction dynamicPrediction = results.predict(
            dynamic.start(), dynamic.end(), dynamic.dynamicStart(), null);
        assertMatrixClose(dynamic.predictedMean(), dynamicPrediction.mean(), TOL);
        assertCubeClose(dynamic.forecastVariance(), dynamicPrediction.variance(), TOL);
        assertCubeClose(dynamic.confInt(), dynamicPrediction.confInt(), 5e-8);

        SmootherResult smoother = results.smooth(SmootherOptions.builder()
            .retainOnly(SmootherOptions.Surface.STATE,
                SmootherOptions.Surface.STATE_COVARIANCE,
                SmootherOptions.Surface.DISTURBANCE,
                SmootherOptions.Surface.DISTURBANCE_COVARIANCE)
            .build());
        assertMatrixClose(fixture.smoothing().smoothedState(), smoothedState(smoother), 5e-8);
        assertCubeClose(fixture.smoothing().smoothedStateCov(), smoothedStateCov(smoother), 5e-8);
        assertMatrixClose(fixture.smoothing().smoothedStateDisturbance(), smoothedStateDisturbance(smoother), 5e-8);
        assertCubeClose(fixture.smoothing().smoothedStateDisturbanceCov(), smoothedStateDisturbanceCov(smoother), 5e-8);
    }

    private static VARMAXSpec spec(CaseFixture fixture) {
        VARMAXSpec.Builder builder = VARMAXSpec.builder(
            VARMAXOrder.of(fixture.order()[0], fixture.order()[1]), fixture.endog())
            .trend(fixture.trend())
            .trendOffset(fixture.trendOffset())
            .errorCovariance(VARMAXErrorCovariance.valueOf(fixture.errorCovariance()))
            .measurementError(fixture.measurementError())
            .logLikelihoodBurn(fixture.logLikelihoodBurn());
        if (fixture.observationIntercept() != null) {
            builder.observationIntercept(fixture.observationIntercept());
        }
        if (fixture.exog() != null) {
            builder.exog(fixture.exog());
        }
        if (!fixture.enforceStationarity()) {
            builder.exclude(VARMAXOption.ENFORCE_STATIONARITY);
        }
        if (!fixture.enforceInvertibility()) {
            builder.exclude(VARMAXOption.ENFORCE_INVERTIBILITY);
        }
        if (fixture.useExactDiffuse()) {
            builder.include(VARMAXOption.USE_EXACT_DIFFUSE);
        }
        if (fixture.approximateDiffuseVariance() != null) {
            builder.approximateDiffuseVariance(fixture.approximateDiffuseVariance());
        }
        return builder.build();
    }

    private static VARMAXResults results(VARMAX model, double[] params, FilterResult filter) {
        Optimization optimization = new Optimization(Double.NaN,
            params.clone(),
            -filter.logLikelihood(),
            Optimization.Status.GRADIENT_TOLERANCE_REACHED,
            0,
            0);
        return new VARMAXResults(model,
            optimization,
            params.clone(),
            params.clone(),
            filter,
            MLEResults.Covariance.OPG);
    }

    private enum Surface {
        FORECAST,
        FORECAST_ERROR
    }

    private static double[][] surface(FilterResult filter, Surface surface) {
        int kEndog = filter.kEndog;
        double[][] values = new double[filter.nobs][kEndog];
        for (int t = 0; t < filter.nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                values[t][i] = surface == Surface.FORECAST
                    ? filter.forecast(i, t)
                    : filter.forecastError(i, t);
            }
        }
        return values;
    }

    private static double[][][] covSurface(FilterResult filter) {
        int kEndog = filter.kEndog;
        double[][][] values = new double[filter.nobs][kEndog][kEndog];
        for (int t = 0; t < filter.nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                for (int j = 0; j < kEndog; j++) {
                    values[t][i][j] = filter.forecastErrorCov(i, j, t);
                }
            }
        }
        return values;
    }

    private static double[][] predictedState(FilterResult filter) {
        double[][] values = new double[filter.nobs][filter.kStates];
        for (int t = 0; t < filter.nobs; t++) {
            int offset = filter.predictedStateOffset(t);
            System.arraycopy(filter.predictedState, offset, values[t], 0, filter.kStates);
        }
        return values;
    }

    private static double[][] filteredState(FilterResult filter) {
        double[][] values = new double[filter.nobs][filter.kStates];
        for (int t = 0; t < filter.nobs; t++) {
            int offset = filter.filteredStateOffset(t);
            System.arraycopy(filter.filteredState, offset, values[t], 0, filter.kStates);
        }
        return values;
    }

    private static double[][] standardizedForecastError(FilterResult filter) {
        double[] flat = filter.standardizedForecastError();
        double[][] values = new double[filter.nobs][filter.kEndog];
        for (int t = 0; t < filter.nobs; t++) {
            System.arraycopy(flat, t * filter.kEndog, values[t], 0, filter.kEndog);
        }
        return values;
    }

    private static double[][] generatedObs(SimulationSmootherResult simulation) {
        double[][] values = new double[simulation.nobs][simulation.kEndog];
        for (int t = 0; t < simulation.nobs; t++) {
            for (int i = 0; i < simulation.kEndog; i++) {
                values[t][i] = simulation.generatedObs(i, t);
            }
        }
        return values;
    }

    private static double[][] smoothedState(SmootherResult smoother) {
        double[][] values = new double[smoother.nobs][smoother.kStates];
        for (int t = 0; t < smoother.nobs; t++) {
            for (int i = 0; i < smoother.kStates; i++) {
                values[t][i] = smoother.smoothedState(i, t);
            }
        }
        return values;
    }

    private static double[][][] smoothedStateCov(SmootherResult smoother) {
        double[][][] values = new double[smoother.nobs][smoother.kStates][smoother.kStates];
        for (int t = 0; t < smoother.nobs; t++) {
            for (int i = 0; i < smoother.kStates; i++) {
                for (int j = 0; j < smoother.kStates; j++) {
                    values[t][i][j] = smoother.smoothedStateCov(i, j, t);
                }
            }
        }
        return values;
    }

    private static double[][] smoothedStateDisturbance(SmootherResult smoother) {
        double[][] values = new double[smoother.nobs][smoother.kPosdef];
        for (int t = 0; t < smoother.nobs; t++) {
            for (int i = 0; i < smoother.kPosdef; i++) {
                values[t][i] = smoother.smoothedStateDisturbance(i, t);
            }
        }
        return values;
    }

    private static double[][][] smoothedStateDisturbanceCov(SmootherResult smoother) {
        double[][][] values = new double[smoother.nobs][smoother.kPosdef][smoother.kPosdef];
        for (int t = 0; t < smoother.nobs; t++) {
            for (int i = 0; i < smoother.kPosdef; i++) {
                for (int j = 0; j < smoother.kPosdef; j++) {
                    values[t][i][j] = smoother.smoothedStateDisturbanceCov(i, j, t);
                }
            }
        }
        return values;
    }

    private static double[][] flattenMatrices(double[][][] matrices) {
        if (matrices == null || matrices.length == 0) {
            return new double[0][0];
        }
        double[][] flat = new double[matrices.length][matrices[0].length * matrices[0][0].length];
        for (int lag = 0; lag < matrices.length; lag++) {
            int offset = 0;
            for (double[] row : matrices[lag]) {
                System.arraycopy(row, 0, flat[lag], offset, row.length);
                offset += row.length;
            }
        }
        return flat;
    }

    private static void assertArrayClose(double[] expected, double[] actual, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tolerance, "index " + i);
        }
    }

    private static void assertMatrixClose(double[][] expected, double[][] actual, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int row = 0; row < expected.length; row++) {
            assertArrayClose(expected[row], actual[row], tolerance);
        }
    }

    private static void assertNullableMatrixClose(Double[][] expected, double[][] actual, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int row = 0; row < expected.length; row++) {
            assertEquals(expected[row].length, actual[row].length);
            for (int col = 0; col < expected[row].length; col++) {
                Double value = expected[row][col];
                if (value == null) {
                    assertEquals(Double.NaN, actual[row][col], "row " + row + " col " + col);
                } else {
                    assertEquals(value, actual[row][col], tolerance, "row " + row + " col " + col);
                }
            }
        }
    }

    private static void assertCubeClose(double[][][] expected, double[][][] actual, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int t = 0; t < expected.length; t++) {
            assertMatrixClose(expected[t], actual[t], tolerance);
        }
    }

    private static void assertAllFinite(double[][] values) {
        for (double[] row : values) {
            for (double value : row) {
                assertTrue(Double.isFinite(value));
            }
        }
    }

    private record FixtureRoot(List<CaseFixture> cases) {
        private static FixtureRoot load() {
            try (InputStream input = Thread.currentThread().getContextClassLoader().getResourceAsStream(FIXTURE_RESOURCE)) {
                assertNotNull(input, "missing fixture resource " + FIXTURE_RESOURCE);
                return OBJECT_MAPPER.readValue(input, FixtureRoot.class);
            } catch (IOException ex) {
                throw new UncheckedIOException(ex);
            }
        }
    }

    private record CaseFixture(String id,
                               int[] order,
                               String trend,
                               int trendOffset,
                               String errorCovariance,
                               boolean measurementError,
                               boolean enforceStationarity,
                               boolean enforceInvertibility,
                               boolean useExactDiffuse,
                               int logLikelihoodBurn,
                               Double approximateDiffuseVariance,
                               double[] observationIntercept,
                               double[][] endog,
                               double[][] exog,
                               double[][] futureExog,
                               double[] params,
                               String[] paramNames,
                               double[] transformUnconstrained,
                               double[] transformConstrained,
                               double[] untransformConstrained,
                               FilterFixture filter,
                               double[][] fittedValues,
                               double[][] residuals,
                               PredictionFixture prediction,
                               PredictionFixture oosPrediction,
                               ExtendFixture extend,
                               SimulationFixture simulateFinal,
                               DynamicPredictionFixture dynamicPrediction,
                               SmoothingFixture smoothing,
                               double[][][] coefficientMatricesVar,
                               double[][][] coefficientMatricesVma) {
        @Override
        public String toString() {
            return id;
        }
    }

    private record FilterFixture(double llf,
                                 double[] llfObs,
                                 double[][] forecast,
                                 double[][] forecastError,
                                 double[][][] forecastErrorCov,
                                 double[][] predictedState,
                                 double[][] filteredState,
                                 Double[][] standardizedForecastError) {
    }

    private record PredictionFixture(double[][] forecastMean,
                                     double[][][] forecastVariance,
                                     double[][][] confInt) {
    }

    private record ExtendFixture(double[][] endog,
                                 double[][] exog,
                                 FilterFixture filter,
                                 double[][] fittedValues,
                                 double[][] residuals) {
    }

    private record SimulationFixture(String anchor,
                                     double[][] exog,
                                     double[][] measurementShocks,
                                     double[][] stateShocks,
                                     double[] initialStateVariates,
                                     double[][] generatedObs) {
    }

    private record DynamicPredictionFixture(int start,
                                            int end,
                                            int dynamicStart,
                                            double[][] predictedMean,
                                            double[][][] forecastVariance,
                                            double[][][] confInt) {
    }

    private record SmoothingFixture(double[][] smoothedState,
                                    double[][][] smoothedStateCov,
                                    double[][] smoothedStateDisturbance,
                                    double[][][] smoothedStateDisturbanceCov) {
    }
}
