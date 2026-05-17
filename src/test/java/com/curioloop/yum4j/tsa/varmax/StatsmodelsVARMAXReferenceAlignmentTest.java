package com.curioloop.yum4j.tsa.varmax;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.mle.MLEResults;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
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

class StatsmodelsVARMAXReferenceAlignmentTest {

    private static final String FIXTURE_RESOURCE =
        "statsmodels/tsa/statespace/tests/results/varmax_reference_alignment.json";
    private static final double TOL = 2e-6;
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
        .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
        .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    static Stream<CaseFixture> statsmodelsReferenceCases() {
        return FixtureRoot.load().cases().stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("statsmodelsReferenceCases")
    void upstreamReferenceVarmaxCasesMatchStatsmodelsFixtures(CaseFixture fixture) {
        VARMAX model = new VARMAX(spec(fixture));
        assertArrayEquals(fixture.paramNames(), model.parameterNames(), fixture.id() + " parameterNames");
        assertArrayClose(fixture.transformConstrained(), model.transformParams(fixture.transformUnconstrained()), 5e-7,
            fixture.id() + " transformParams");
        assertArrayClose(fixture.untransformConstrained(), model.untransformParams(fixture.transformConstrained()), 5e-7,
            fixture.id() + " untransformParams");

        double tolerance = fixture.tolerance() == null
            ? (fixture.approximateDiffuseVariance() == null ? TOL : 2e-4)
            : fixture.tolerance();
        if (fixture.observationIntercept() != null) {
            assertObservationIntercept(fixture.observationIntercept(), model.snapshotModel(fixture.params()), 1e-12,
                fixture.id() + " observationIntercept");
        }
        FilterResult filter = model.filter(fixture.params(), FilterOptions.defaults());
        assertClose(fixture.filter().llf(), model.logLikelihood(fixture.params(), FilterOptions.defaults()), tolerance, fixture.id() + " llf");
        assertArrayClose(fixture.filter().llfObs(), filter.logLikelihoodObs, tolerance, fixture.id() + " llfObs");
        assertMatrixClose(fixture.filter().forecast(), surface(filter, Surface.FORECAST), tolerance, fixture.id() + " forecast");
        assertMatrixClose(fixture.filter().forecastError(), surface(filter, Surface.FORECAST_ERROR), tolerance, fixture.id() + " forecastError");
        assertCubeClose(fixture.filter().forecastErrorCov(), covSurface(filter), tolerance, fixture.id() + " forecastErrorCov");
        assertMatrixClose(fixture.filter().predictedState(), predictedState(filter), tolerance, fixture.id() + " predictedState");
        assertMatrixClose(fixture.filter().filteredState(), filteredState(filter), tolerance, fixture.id() + " filteredState");
        assertNullableMatrixClose(fixture.filter().standardizedForecastError(), standardizedForecastError(filter), Math.max(5e-7, tolerance),
            fixture.id() + " standardizedForecastError");

        VARMAXResults results = results(model, fixture.params(), filter);
        assertMatrixClose(fixture.fittedValues(), results.fittedValues(), tolerance, fixture.id() + " fittedValues");
        assertMatrixClose(fixture.residuals(), results.residuals(), tolerance, fixture.id() + " residuals");
        assertNullableMatrixClose(fixture.filter().standardizedForecastError(), results.standardizedForecastError(), Math.max(5e-7, tolerance),
            fixture.id() + " resultStandardizedForecastError");
        if (fixture.coefficientMatricesVar() != null) {
            assertMatrixClose(flattenMatrices(fixture.coefficientMatricesVar()), results.coefficientMatricesVAR(), tolerance,
                fixture.id() + " coefficientMatricesVAR");
        }
        if (fixture.coefficientMatricesVma() != null) {
            assertMatrixClose(flattenMatrices(fixture.coefficientMatricesVma()), results.coefficientMatricesVMA(), tolerance,
                fixture.id() + " coefficientMatricesVMA");
        }

        VARMAXPrediction forecast = results.forecast(fixture.prediction().forecastMean().length, fixture.futureExog());
        assertMatrixClose(fixture.prediction().forecastMean(), forecast.mean(), tolerance, fixture.id() + " forecastMean");
        assertCubeClose(fixture.prediction().forecastVariance(), forecast.variance(), Math.max(5e-7, tolerance),
            fixture.id() + " forecastVariance");
        assertCubeClose(fixture.prediction().confInt(), forecast.confInt(), Math.max(5e-7, tolerance), fixture.id() + " forecastConfInt");

        if (fixture.oosPrediction() != null) {
            int start = fixture.endog().length;
            int steps = fixture.oosPrediction().forecastMean().length;
            VARMAXPrediction oosPrediction = results.predict(start, start + steps - 1, -1, fixture.futureExog());
            assertMatrixClose(fixture.oosPrediction().forecastMean(), oosPrediction.mean(), tolerance,
                fixture.id() + " oosPredictionMean");
            assertCubeClose(fixture.oosPrediction().forecastVariance(), oosPrediction.variance(), Math.max(5e-7, tolerance),
                fixture.id() + " oosPredictionVariance");
            assertCubeClose(fixture.oosPrediction().confInt(), oosPrediction.confInt(), Math.max(5e-7, tolerance),
                fixture.id() + " oosPredictionConfInt");
        }

        if (fixture.extend() != null) {
            VARMAXResults extended = results.extend(fixture.extend().endog(), fixture.extend().exog());
            assertMatrixClose(fixture.extend().filter().forecast(), surface(extended.filterResult(), Surface.FORECAST), tolerance,
                fixture.id() + " extendForecast");
            assertMatrixClose(fixture.extend().filter().forecastError(), surface(extended.filterResult(), Surface.FORECAST_ERROR), tolerance,
                fixture.id() + " extendForecastError");
            assertCubeClose(fixture.extend().filter().forecastErrorCov(), covSurface(extended.filterResult()), Math.max(5e-7, tolerance),
                fixture.id() + " extendForecastErrorCov");
            assertMatrixClose(fixture.extend().filter().predictedState(), predictedState(extended.filterResult()), Math.max(5e-7, tolerance),
                fixture.id() + " extendPredictedState");
            assertMatrixClose(fixture.extend().filter().filteredState(), filteredState(extended.filterResult()), Math.max(5e-7, tolerance),
                fixture.id() + " extendFilteredState");
            assertNullableMatrixClose(fixture.extend().filter().standardizedForecastError(),
                standardizedForecastError(extended.filterResult()), Math.max(5e-7, tolerance),
                fixture.id() + " extendStandardizedForecastError");
            assertMatrixClose(fixture.extend().fittedValues(), extended.fittedValues(), tolerance, fixture.id() + " extendFittedValues");
            assertMatrixClose(fixture.extend().residuals(), extended.residuals(), tolerance, fixture.id() + " extendResiduals");
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
            assertMatrixClose(expected.generatedObs(), generatedObs(simulation), Math.max(5e-7, tolerance),
                fixture.id() + " simulateFinalGeneratedObs");
            assertAllFinite(generatedObs(simulation), fixture.id() + " simulateFinalGeneratedObs");
        }

        DynamicPredictionFixture dynamic = fixture.dynamicPrediction();
        VARMAXPrediction dynamicPrediction = results.predict(
            dynamic.start(), dynamic.end(), dynamic.dynamicStart(), null);
        assertMatrixClose(dynamic.predictedMean(), dynamicPrediction.mean(), tolerance, fixture.id() + " dynamicMean");
        assertCubeClose(dynamic.forecastVariance(), dynamicPrediction.variance(), Math.max(5e-7, tolerance),
            fixture.id() + " dynamicVariance");
        assertCubeClose(dynamic.confInt(), dynamicPrediction.confInt(), Math.max(5e-7, tolerance), fixture.id() + " dynamicConfInt");

        SmootherResult smoother = results.smooth(SmootherOptions.builder()
            .retainOnly(SmootherOptions.Surface.STATE,
                SmootherOptions.Surface.STATE_COVARIANCE,
                SmootherOptions.Surface.DISTURBANCE,
                SmootherOptions.Surface.DISTURBANCE_COVARIANCE)
            .build());
        assertMatrixClose(fixture.smoothing().smoothedState(), smoothedState(smoother), Math.max(5e-7, tolerance),
            fixture.id() + " smoothedState");
        assertCubeClose(fixture.smoothing().smoothedStateCov(), smoothedStateCov(smoother), Math.max(5e-7, tolerance),
            fixture.id() + " smoothedStateCov");
        assertMatrixClose(fixture.smoothing().smoothedStateDisturbance(), smoothedStateDisturbance(smoother), Math.max(5e-7, tolerance),
            fixture.id() + " smoothedStateDisturbance");
        assertCubeClose(fixture.smoothing().smoothedStateDisturbanceCov(), smoothedStateDisturbanceCov(smoother), Math.max(5e-7, tolerance),
            fixture.id() + " smoothedStateDisturbanceCov");
    }

    private static VARMAXSpec spec(CaseFixture fixture) {
        VARMAXSpec.Builder builder = VARMAXSpec.builder(
            VARMAXOrder.of(fixture.order()[0], fixture.order()[1]), fixture.endog())
            .trend(fixture.trend())
            .trendOffset(fixture.trendOffset())
            .errorCovariance(VARMAXErrorCovariance.valueOf(fixture.errorCovariance()))
            .measurementError(fixture.measurementError())
            .logLikelihoodBurn(fixture.logLikelihoodBurn())
            .endogNames(fixture.endogNames())
            .exogNames(fixture.exogNames());
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
        double[][] values = new double[filter.nobs][filter.kEndog];
        for (int t = 0; t < filter.nobs; t++) {
            for (int i = 0; i < filter.kEndog; i++) {
                values[t][i] = surface == Surface.FORECAST
                    ? filter.forecast(i, t)
                    : filter.forecastError(i, t);
            }
        }
        return values;
    }

    private static double[][][] covSurface(FilterResult filter) {
        double[][][] values = new double[filter.nobs][filter.kEndog][filter.kEndog];
        for (int t = 0; t < filter.nobs; t++) {
            for (int i = 0; i < filter.kEndog; i++) {
                for (int j = 0; j < filter.kEndog; j++) {
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

    private static void assertArrayClose(double[] expected, double[] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " length");
        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual[i], tolerance, label + " index " + i);
        }
    }

    private static void assertObservationIntercept(double[] expected, KalmanSSM snapshot, double tolerance, String label) {
        for (int t = 0; t < snapshot.observationCount(); t++) {
            int offset = snapshot.obsInterceptOffset(t);
            for (int i = 0; i < expected.length; i++) {
                assertClose(expected[i], snapshot.obsInterceptData()[offset + i], tolerance, label + " t " + t + " index " + i);
            }
        }
    }

    private static void assertMatrixClose(double[][] expected, double[][] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " rows");
        for (int row = 0; row < expected.length; row++) {
            assertArrayClose(expected[row], actual[row], tolerance, label + " row " + row);
        }
    }

    private static void assertNullableMatrixClose(Double[][] expected, double[][] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " rows");
        for (int row = 0; row < expected.length; row++) {
            assertEquals(expected[row].length, actual[row].length, label + " row " + row + " cols");
            for (int col = 0; col < expected[row].length; col++) {
                Double value = expected[row][col];
                if (value == null) {
                    assertTrue(Double.isNaN(actual[row][col]), label + " row " + row + " col " + col + " expected NaN");
                } else {
                    assertClose(value, actual[row][col], tolerance, label + " row " + row + " col " + col);
                }
            }
        }
    }

    private static void assertCubeClose(double[][][] expected, double[][][] actual, double tolerance, String label) {
        assertEquals(expected.length, actual.length, label + " slices");
        for (int t = 0; t < expected.length; t++) {
            assertMatrixClose(expected[t], actual[t], tolerance, label + " slice " + t);
        }
    }

    private static void assertAllFinite(double[][] values, String label) {
        for (int row = 0; row < values.length; row++) {
            for (int col = 0; col < values[row].length; col++) {
                assertTrue(Double.isFinite(values[row][col]), label + " row " + row + " col " + col);
            }
        }
    }

    private static void assertClose(double expected, double actual, double tolerance, String label) {
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
                               Double tolerance,
                               double[] observationIntercept,
                               String[] endogNames,
                               String[] exogNames,
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
            return id + " (" + upstreamClass + ")";
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