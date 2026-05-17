package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.mle.MLEResults;
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
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class StatsmodelsSARIMAXCoverageAlignmentTest {

    private static final String FIXTURE_RESOURCE =
        "statsmodels/tsa/statespace/tests/results/sarimax_coverage_alignment.json";
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
        .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
        .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    static Stream<CaseFixture> statsmodelsCoverageCases() {
        return FixtureRoot.load().cases().stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("statsmodelsCoverageCases")
    void coverageMatrixMatchesStatsmodelsFixtures(CaseFixture fixture) {
        assumeTrue(fixture.deferredReason() == null || fixture.deferredReason().isBlank(),
            () -> fixture.id() + " deferred: " + fixture.deferredReason());

        SARIMAX transformModel = new SARIMAX(spec(fixture,
            fixture.transformEnforceStationarity(), fixture.transformEnforceInvertibility()));
        assertArrayClose(fixture.transformConstrained(), transformModel.transformParams(fixture.transformUnconstrained()),
            5e-7, fixture.id() + " transformParams");
        assertArrayClose(fixture.untransformConstrained(), transformModel.untransformParams(fixture.transformConstrained()),
            5e-7, fixture.id() + " untransformParams");

        SARIMAX model = new SARIMAX(spec(fixture, fixture.enforceStationarity(), fixture.enforceInvertibility()));
        assertArrayEquals(fixture.paramNames(), model.parameterNames(), fixture.id() + " parameterNames");

        double tolerance = fixture.tolerance();
    double smoothingTolerance = Math.max(tolerance, fixture.smoothingTolerance());
        FilterResult filter = model.filter(fixture.params(), FilterOptions.defaults());
        SARIMAXResults results = results(model, fixture.params(), filter);
        FilterResult publicFilter = results.filterResult();

        assertClose(fixture.filter().llf(), model.logLikelihood(fixture.params()), tolerance, fixture.id() + " llf");
        assertNullableArrayClose(fixture.filter().llfObs(), results.logLikelihoodObs(), tolerance, fixture.id() + " llfObs");
        assertNullableArrayClose(fixture.filter().forecast(), surface(publicFilter, Surface.FORECAST), tolerance, fixture.id() + " forecast");
        if (fixture.comparesSurface("filter.forecast_error")) {
            assertNullableArrayClose(fixture.filter().forecastError(), surface(publicFilter, Surface.FORECAST_ERROR), tolerance, fixture.id() + " forecastError");
        }
        assertNullableArrayClose(fixture.filter().forecastErrorCov(), surface(publicFilter, Surface.FORECAST_ERROR_COV), tolerance,
            fixture.id() + " forecastErrorCov");
        assertNullableMatrixClose(fixture.filter().predictedState(), predictedState(publicFilter), tolerance, fixture.id() + " predictedState");
        assertNullableMatrixClose(fixture.filter().filteredState(), filteredState(publicFilter), tolerance, fixture.id() + " filteredState");
        if (fixture.comparesSurface("filter.standardized_forecast_error")) {
            assertNullableArrayClose(fixture.filter().standardizedForecastError(), publicFilter.standardizedForecastError(),
                Math.max(5e-7, tolerance), fixture.id() + " standardizedForecastError");
        }

        assertNullableArrayClose(fixture.fittedValues(), results.fittedValues(), tolerance, fixture.id() + " fittedValues");
        if (fixture.comparesSurface("residuals")) {
            assertNullableArrayClose(fixture.residuals(), results.residuals(), tolerance, fixture.id() + " residuals");
        }
        if (fixture.comparesSurface("result.standardized_forecast_error")) {
            assertNullableArrayClose(fixture.filter().standardizedForecastError(), results.standardizedForecastError(),
                Math.max(5e-7, tolerance), fixture.id() + " resultStandardizedForecastError");
        }

        SARIMAXPrediction forecast = results.forecast(fixture.prediction().forecastMean().length, matrix(fixture.futureExog()));
        assertNullableArrayClose(fixture.prediction().forecastMean(), forecast.mean(), tolerance, fixture.id() + " forecastMean");
        assertNullableArrayClose(fixture.prediction().forecastVariance(), forecast.variance(), tolerance, fixture.id() + " forecastVariance");
        assertNullableMatrixClose(fixture.prediction().confInt(), forecast.confInt(), Math.max(5e-7, tolerance),
            fixture.id() + " forecastConfInt");

        DynamicPredictionFixture dynamic = fixture.dynamicPrediction();
        SARIMAXPrediction dynamicPrediction = results.predict(
            dynamic.start(), dynamic.end(), dynamic.dynamicStart(), null);
        assertNullableArrayClose(dynamic.predictedMean(), dynamicPrediction.mean(), tolerance, fixture.id() + " dynamicMean");
        assertNullableArrayClose(dynamic.forecastVariance(), dynamicPrediction.variance(), tolerance, fixture.id() + " dynamicVariance");
        assertNullableMatrixClose(dynamic.confInt(), dynamicPrediction.confInt(), Math.max(5e-7, tolerance),
            fixture.id() + " dynamicConfInt");

        SmootherResult smoother = results.smooth(SmootherOptions.builder()
            .retainOnly(SmootherOptions.Surface.STATE, SmootherOptions.Surface.STATE_COVARIANCE)
            .build());
        assertNullableMatrixClose(fixture.smoothing().smoothedState(), smoothedState(smoother), Math.max(5e-7, smoothingTolerance),
            fixture.id() + " smoothedState");
        assertNullableCubeClose(fixture.smoothing().smoothedStateCov(), smoothedStateCov(smoother), Math.max(5e-7, smoothingTolerance),
            fixture.id() + " smoothedStateCov");
    }

    private static SARIMAXSpec spec(CaseFixture fixture, boolean enforceStationarity, boolean enforceInvertibility) {
        SARIMAXSpec.Builder builder = SARIMAXSpec.builder(
                ARIMAOrder.of(fixture.order()[0], fixture.order()[1], fixture.order()[2]), array(fixture.endog()))
            .seasonalOrder(SeasonalOrder.of(fixture.seasonalOrder()[0],
                fixture.seasonalOrder()[1], fixture.seasonalOrder()[2], fixture.seasonalOrder()[3]))
            .trendPowers(fixture.trendPowers())
            .measurementError(fixture.measurementError())
            .concentrateScale(fixture.concentrateScale())
            .mleRegression(fixture.mleRegression())
            .timeVaryingRegression(fixture.timeVaryingRegression())
            .simpleDifferencing(fixture.simpleDifferencing())
            .hamiltonRepresentation(fixture.hamiltonRepresentation());
        if (fixture.exog() != null) {
            builder.exog(matrix(fixture.exog()));
        }
        if (!enforceStationarity) {
            builder.exclude(SARIMAXOption.ENFORCE_STATIONARITY);
        }
        if (!enforceInvertibility) {
            builder.exclude(SARIMAXOption.ENFORCE_INVERTIBILITY);
        }
        if (fixture.useExactDiffuse()) {
            builder.include(SARIMAXOption.USE_EXACT_DIFFUSE);
        }
        if (fixture.approximateDiffuseVariance() != null) {
            builder.approximateDiffuseVariance(fixture.approximateDiffuseVariance());
        }
        builder.logLikelihoodBurn(fixture.logLikelihoodBurn());
        return builder.build();
    }

    private static SARIMAXResults results(SARIMAX model, double[] params, FilterResult filter) {
        Optimization optimization = new Optimization(Double.NaN,
            params.clone(),
            -filter.logLikelihood(),
            Optimization.Status.GRADIENT_TOLERANCE_REACHED,
            0,
            0);
        return new SARIMAXResults(model,
            optimization,
            params.clone(),
            model.untransformParams(params),
            filter,
            MLEResults.Covariance.OPG);
    }

    private enum Surface {
        FORECAST,
        FORECAST_ERROR,
        FORECAST_ERROR_COV
    }

    private static double[] surface(FilterResult filter, Surface surface) {
        double[] values = new double[filter.nobs];
        for (int t = 0; t < filter.nobs; t++) {
            values[t] = switch (surface) {
                case FORECAST -> filter.forecast(0, t);
                case FORECAST_ERROR -> filter.forecastError(0, t);
                case FORECAST_ERROR_COV -> filter.forecastErrorCov(0, 0, t);
            };
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

    private static double[] array(Double[] values) {
        double[] array = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            array[i] = values[i] == null ? Double.NaN : values[i];
        }
        return array;
    }

    private static double[][] matrix(Double[][] values) {
        if (values == null) {
            return null;
        }
        double[][] matrix = new double[values.length][];
        for (int row = 0; row < values.length; row++) {
            matrix[row] = array(values[row]);
        }
        return matrix;
    }

    private static void assertArrayClose(double[] expected, double[] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " length");
        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual[i], tolerance, label + " index " + i);
        }
    }

    private static void assertNullableArrayClose(Double[] expected, double[] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " length");
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] == null) {
                assertTrue(Double.isNaN(actual[i]), label + " index " + i + " expected NaN but was " + actual[i]);
            } else {
                assertClose(expected[i], actual[i], tolerance, label + " index " + i);
            }
        }
    }

    private static void assertNullableMatrixClose(Double[][] expected, double[][] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " rows");
        for (int row = 0; row < expected.length; row++) {
            assertNullableArrayClose(expected[row], actual[row], tolerance, label + " row " + row);
        }
    }

    private static void assertNullableCubeClose(Double[][][] expected, double[][][] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " slices");
        for (int t = 0; t < expected.length; t++) {
            assertNullableMatrixClose(expected[t], actual[t], tolerance, label + " slice " + t);
        }
    }

    private static void assertClose(double expected, double actual, double tolerance, String label) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), label + " expected NaN but was " + actual);
            return;
        }
        double allowed = tolerance * Math.max(1.0, Math.abs(expected));
        assertTrue(Math.abs(expected - actual) <= allowed,
            () -> label + " ==> expected: <" + expected + "> but was: <" + actual + "> tolerance: <" + allowed + ">");
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
                               String upstreamClass,
                               double tolerance,
                               double smoothingTolerance,
                               String deferredReason,
                               String[] deferredSurfaces,
                               int[] order,
                               int[] seasonalOrder,
                               int[] trendPowers,
                               boolean measurementError,
                               boolean concentrateScale,
                               boolean mleRegression,
                               boolean timeVaryingRegression,
                               boolean simpleDifferencing,
                               boolean hamiltonRepresentation,
                               boolean enforceStationarity,
                               boolean enforceInvertibility,
                               boolean transformEnforceStationarity,
                               boolean transformEnforceInvertibility,
                               boolean useExactDiffuse,
                               int logLikelihoodBurn,
                               Double approximateDiffuseVariance,
                               Double[] endog,
                               Double[][] exog,
                               Double[][] futureExog,
                               double[] params,
                               String[] paramNames,
                               double[] transformUnconstrained,
                               double[] transformConstrained,
                               double[] untransformConstrained,
                               FilterFixture filter,
                               Double[] fittedValues,
                               Double[] residuals,
                               PredictionFixture prediction,
                               DynamicPredictionFixture dynamicPrediction,
                               SmoothingFixture smoothing) {
        boolean comparesSurface(String surface) {
            return deferredSurfaces == null || Arrays.stream(deferredSurfaces).noneMatch(surface::equals);
        }

        @Override
        public String toString() {
            return id + " (" + upstreamClass + ")";
        }
    }

    private record FilterFixture(double llf,
                                 Double[] llfObs,
                                 Double[] forecast,
                                 Double[] forecastError,
                                 Double[] forecastErrorCov,
                                 Double[][] predictedState,
                                 Double[][] filteredState,
                                 Double[] standardizedForecastError) {
    }

    private record PredictionFixture(Double[] forecastMean,
                                     Double[] forecastVariance,
                                     Double[][] confInt) {
    }

    private record DynamicPredictionFixture(int start,
                                            int end,
                                            int dynamicStart,
                                            Double[] predictedMean,
                                            Double[] forecastVariance,
                                            Double[][] confInt) {
    }

    private record SmoothingFixture(Double[][] smoothedState,
                                    Double[][][] smoothedStateCov) {
    }
}