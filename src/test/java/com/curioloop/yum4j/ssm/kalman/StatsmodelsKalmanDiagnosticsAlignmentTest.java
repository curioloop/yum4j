package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.AlignmentCase;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.SurfacePoint;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherDiagnostics;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StatsmodelsKalmanDiagnosticsAlignmentTest {

    @ParameterizedTest(name = "{0}")
    @MethodSource("diagnosticsCases")
    void diagnosticsSurfacesMatchStatsmodelsReference(AlignmentCase fixture) {
        ModelFixture model = fixture.modelFixture();
        InitialState initialState = fixture.initialState(model);
        SmootherOptions options = fixture.smootherOptions();
        List<SurfacePoint> points = KalmanStatsmodelsFixtures.diagnosticsSurfaces(fixture);

        boolean needsWeights = points.stream().anyMatch(StatsmodelsKalmanDiagnosticsAlignmentTest::isWeightSurface);
        boolean needsDecomposition = points.stream().anyMatch(StatsmodelsKalmanDiagnosticsAlignmentTest::isDecompositionSurface);
        boolean needsNews = points.stream().anyMatch(point -> point.surface().startsWith("news_"));
        SmootherDiagnostics.SmoothedStateWeights weights = needsWeights
            ? SmootherDiagnostics.computeSmoothedStateWeights(model, initialState, options)
            : null;
        SmootherDiagnostics.SmoothedDecomposition decomposition = needsDecomposition
            ? SmootherDiagnostics.getSmoothedDecomposition(model, initialState, options)
            : null;
        SmootherDiagnostics.NewsImpact news = needsNews ? newsImpact(fixture, model, initialState, options) : null;

        for (SurfacePoint point : points) {
            double actual = actualValue(point, weights, decomposition, news);
            assertClose(point.value(), actual, fixture.tolerance(), point.caseId() + " " + point.surface()
                + " t=" + point.t() + " row=" + point.row() + " col=" + point.col());
        }
    }

    static Stream<AlignmentCase> diagnosticsCases() {
        return KalmanStatsmodelsFixtures.cases().stream()
            .filter(fixture -> !fixture.complex())
            .filter(AlignmentCase::hasDiagnosticsSurfaces);
    }

    private static SmootherDiagnostics.NewsImpact newsImpact(AlignmentCase fixture,
                                                             ModelFixture updatedModel,
                                                             InitialState updatedInitialState,
                                                             SmootherOptions options) {
        AlignmentCase previous = KalmanStatsmodelsFixtures.caseById(fixture.newsPreviousCaseId());
        ModelFixture previousModel = previous.modelFixture();
        return SmootherDiagnostics.news(previousModel, previous.initialState(previousModel),
            updatedModel, updatedInitialState, options);
    }

    private static double actualValue(SurfacePoint point,
                                      SmootherDiagnostics.SmoothedStateWeights weights,
                                      SmootherDiagnostics.SmoothedDecomposition decomposition,
                                      SmootherDiagnostics.NewsImpact news) {
        return switch (point.surface()) {
            case "observation_weight" -> weights.observationWeight(
                point.t(), point.col() / weights.kEndog(), point.row(), point.col() % weights.kEndog());
            case "state_intercept_weight" -> weights.stateInterceptWeight(
                point.t(), point.col() / weights.kStates(), point.row(), point.col() % weights.kStates());
            case "prior_weight" -> weights.priorWeight(point.t(), point.row(), point.col());
            case "data_contribution" -> decomposition.dataContribution()[observationOffset(
                point.t(), point.col(), point.row(), decomposition.kEndog(), decomposition.kStates(), decomposition.nobs())];
            case "observation_intercept_contribution" -> decomposition.observationInterceptContribution()[observationOffset(
                point.t(), point.col(), point.row(), decomposition.kEndog(), decomposition.kStates(), decomposition.nobs())];
            case "state_intercept_contribution" -> decomposition.stateInterceptContribution()[stateInterceptOffset(
                point.t(), point.col(), point.row(), decomposition.kStates(), decomposition.nobs())];
            case "prior_contribution" -> decomposition.priorContribution()[priorOffset(
                point.t(), point.row(), point.col(), decomposition.kStates())];
            case "news_smoothed_state_impact" -> news.smoothedStateImpact(point.row(), point.t());
            case "news_smoothed_signal_impact" -> news.smoothedSignalImpact(point.row(), point.t());
            case "news_revision_impact" -> news.revisionImpact(point.row(), point.t());
            case "news_new_observation_impact" -> news.newObservationImpact(point.row(), point.t());
            default -> throw new IllegalArgumentException("Unsupported diagnostics surface: " + point.surface());
        };
    }

    private static boolean isWeightSurface(SurfacePoint point) {
        return point.surface().equals("observation_weight")
            || point.surface().equals("state_intercept_weight")
            || point.surface().equals("prior_weight");
    }

    private static boolean isDecompositionSurface(SurfacePoint point) {
        return point.surface().equals("data_contribution")
            || point.surface().equals("observation_intercept_contribution")
            || point.surface().equals("state_intercept_contribution")
            || point.surface().equals("prior_contribution");
    }

    private static int observationOffset(int targetTime, int encodedSourceColumn, int stateIndex,
                                         int kEndog, int kStates, int nobs) {
        int sourceTime = encodedSourceColumn / kEndog;
        int observationIndex = encodedSourceColumn % kEndog;
        return (((targetTime * nobs + sourceTime) * kStates + stateIndex) * kEndog) + observationIndex;
    }

    private static int stateInterceptOffset(int targetTime, int encodedSourceColumn, int stateIndex,
                                            int kStates, int nobs) {
        int sourceTime = encodedSourceColumn / kStates;
        int stateFrom = encodedSourceColumn % kStates;
        return (((targetTime * nobs + sourceTime) * kStates + stateIndex) * kStates) + stateFrom;
    }

    private static int priorOffset(int targetTime, int stateIndex, int stateFrom, int kStates) {
        return ((targetTime * kStates + stateIndex) * kStates) + stateFrom;
    }

    private static void assertClose(double expected, double actual, double tolerance, String message) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), message);
            return;
        }
        double scale = Math.max(1.0, Math.max(Math.abs(expected), Math.abs(actual)));
        assertEquals(expected, actual, tolerance * scale, message);
    }
}
