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
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StatsmodelsSARIMAXAlignmentTest {

    private static final String FIXTURE_RESOURCE =
        "statsmodels/tsa/statespace/tests/results/sarimax_public_alignment.json";
    private static final double TOL = 2e-8;
    private static final double APPROXIMATE_DIFFUSE_TOL = 5e-6;
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
        .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
        .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    static Stream<CaseFixture> statsmodelsCases() {
        return FixtureRoot.load().cases().stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("statsmodelsCases")
    void publicSarimaxApiMatchesStatsmodelsFixtures(CaseFixture fixture) {
        SARIMAX model = new SARIMAX(spec(fixture));
        assertArrayEquals(fixture.paramNames(), model.parameterNames());
        assertArrayClose(fixture.transformConstrained(), model.transformParams(fixture.transformUnconstrained()), 5e-8,
            fixture.id() + " transformParams");

        FilterResult filter = model.filter(fixture.params(), FilterOptions.defaults());
        SARIMAXResults results = results(model, fixture.params(), filter);
        FilterResult publicFilter = results.filterResult();
        double caseTol = results.logLikelihoodBurn() > 0 && !fixture.useExactDiffuse()
            ? APPROXIMATE_DIFFUSE_TOL
            : TOL;
        assertClose(fixture.filter().llf(), model.logLikelihood(fixture.params()), caseTol, fixture.id() + " llf");
        assertArrayClose(fixture.filter().llfObs(), results.logLikelihoodObs(), caseTol, fixture.id() + " llfObs");
        assertArrayClose(fixture.filter().forecast(), surface(publicFilter, Surface.FORECAST), caseTol, fixture.id() + " forecast");
        assertArrayClose(fixture.filter().forecastError(), surface(publicFilter, Surface.FORECAST_ERROR), caseTol, fixture.id() + " forecastError");
        assertArrayClose(fixture.filter().forecastErrorCov(), surface(publicFilter, Surface.FORECAST_ERROR_COV), caseTol, fixture.id() + " forecastErrorCov");
        assertMatrixClose(fixture.filter().predictedState(), predictedState(publicFilter), caseTol, fixture.id() + " predictedState");
        assertMatrixClose(fixture.filter().filteredState(), filteredState(publicFilter), caseTol, fixture.id() + " filteredState");
        assertArrayClose(fixture.filter().standardizedForecastError(), publicFilter.standardizedForecastError(), Math.max(5e-8, caseTol),
            fixture.id() + " standardizedForecastError");

        assertArrayClose(fixture.fittedValues(), results.fittedValues(), caseTol, fixture.id() + " fittedValues");
        assertArrayClose(fixture.residuals(), results.residuals(), caseTol, fixture.id() + " residuals");
        assertArrayClose(fixture.filter().standardizedForecastError(), results.standardizedForecastError(), Math.max(5e-8, caseTol),
            fixture.id() + " resultStandardizedForecastError");

        SARIMAXPrediction forecast = results.forecast(fixture.prediction().forecastMean().length, fixture.futureExog());
        assertArrayClose(fixture.prediction().forecastMean(), forecast.mean(), caseTol, fixture.id() + " forecastMean");
        assertArrayClose(fixture.prediction().forecastVariance(), forecast.variance(), caseTol, fixture.id() + " forecastVariance");
        assertMatrixClose(fixture.prediction().confInt(), forecast.confInt(), Math.max(5e-8, caseTol), fixture.id() + " forecastConfInt");

        DynamicPredictionFixture dynamic = fixture.dynamicPrediction();
        SARIMAXPrediction dynamicPrediction = results.predict(
            dynamic.start(), dynamic.end(), dynamic.dynamicStart(), null);
        assertArrayClose(dynamic.predictedMean(), dynamicPrediction.mean(), caseTol, fixture.id() + " dynamicMean");
        assertArrayClose(dynamic.forecastVariance(), dynamicPrediction.variance(), caseTol, fixture.id() + " dynamicVariance");
        assertMatrixClose(dynamic.confInt(), dynamicPrediction.confInt(), Math.max(5e-8, caseTol), fixture.id() + " dynamicConfInt");

        SmootherResult smoother = results.smooth(SmootherOptions.builder()
            .retainOnly(SmootherOptions.Surface.STATE, SmootherOptions.Surface.STATE_COVARIANCE)
            .build());
        assertMatrixClose(fixture.smoothing().smoothedState(), smoothedState(smoother), Math.max(5e-8, caseTol), fixture.id() + " smoothedState");
        assertCubeClose(fixture.smoothing().smoothedStateCov(), smoothedStateCov(smoother), Math.max(5e-8, caseTol), fixture.id() + " smoothedStateCov");
    }

    private static SARIMAXSpec spec(CaseFixture fixture) {
        SARIMAXSpec.Builder builder = SARIMAXSpec.builder(
                ARIMAOrder.of(fixture.order()[0], fixture.order()[1], fixture.order()[2]),
                fixture.endog())
            .seasonalOrder(SeasonalOrder.of(fixture.seasonalOrder()[0],
                fixture.seasonalOrder()[1], fixture.seasonalOrder()[2], fixture.seasonalOrder()[3]))
            .trendPowers(fixture.trendPowers())
            .measurementError(fixture.measurementError())
            .concentrateScale(fixture.concentrateScale())
            .mleRegression(fixture.mleRegression())
            .timeVaryingRegression(fixture.timeVaryingRegression())
            .simpleDifferencing(fixture.simpleDifferencing())
            .hamiltonRepresentation(fixture.hamiltonRepresentation());
        if (fixture.useExactDiffuse()) {
            builder.include(SARIMAXOption.USE_EXACT_DIFFUSE);
        }
        if (fixture.exog() != null) {
            builder.exog(fixture.exog());
        }
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

    private static void assertArrayClose(double[] expected, double[] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " length");
        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual[i], tolerance, label + " index " + i);
        }
    }

    private static void assertClose(double expected, double actual, double tolerance, String label) {
        double allowed = tolerance * Math.max(1.0, Math.abs(expected));
        assertTrue(Math.abs(expected - actual) <= allowed,
            () -> label + " ==> expected: <" + expected + "> but was: <" + actual + "> tolerance: <" + allowed + ">");
    }

    private static void assertMatrixClose(double[][] expected, double[][] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " rows");
        for (int row = 0; row < expected.length; row++) {
            assertArrayClose(expected[row], actual[row], tolerance, label + " row " + row);
        }
    }

    private static void assertCubeClose(double[][][] expected, double[][][] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " slices");
        for (int t = 0; t < expected.length; t++) {
            assertMatrixClose(expected[t], actual[t], tolerance, label + " slice " + t);
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
                               int[] seasonalOrder,
                               int[] trendPowers,
                               boolean measurementError,
                               boolean concentrateScale,
                               boolean mleRegression,
                               boolean timeVaryingRegression,
                               boolean simpleDifferencing,
                               boolean hamiltonRepresentation,
                               boolean useExactDiffuse,
                               double[] endog,
                               double[][] exog,
                               double[][] futureExog,
                               double[] params,
                               String[] paramNames,
                               double[] transformUnconstrained,
                               double[] transformConstrained,
                               FilterFixture filter,
                               double[] fittedValues,
                               double[] residuals,
                               PredictionFixture prediction,
                               DynamicPredictionFixture dynamicPrediction,
                               SmoothingFixture smoothing) {
        @Override
        public String toString() {
            return id;
        }
    }

    private record FilterFixture(double llf,
                                 double[] llfObs,
                                 double[] forecast,
                                 double[] forecastError,
                                 double[] forecastErrorCov,
                                 double[][] predictedState,
                                 double[][] filteredState,
                                 double[] standardizedForecastError) {
    }

    private record PredictionFixture(double[] forecastMean,
                                     double[] forecastVariance,
                                     double[][] confInt) {
    }

    private record DynamicPredictionFixture(int start,
                                            int end,
                                            int dynamicStart,
                                            double[] predictedMean,
                                            double[] forecastVariance,
                                            double[][] confInt) {
    }

    private record SmoothingFixture(double[][] smoothedState,
                                    double[][][] smoothedStateCov) {
    }
}