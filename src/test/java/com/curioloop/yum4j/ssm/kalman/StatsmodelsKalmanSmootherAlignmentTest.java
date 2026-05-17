package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.AlignmentCase;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.SurfacePoint;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StatsmodelsKalmanSmootherAlignmentTest {

    @ParameterizedTest(name = "{0}")
    @MethodSource("smootherCases")
    void smootherSurfacesMatchStatsmodelsReference(AlignmentCase fixture) {
        ModelFixture model = fixture.modelFixture();
        InitialState initialState = fixture.initialState(model);
        List<SurfacePoint> points = KalmanStatsmodelsFixtures.smootherSurfaces(fixture);
        SmootherOptions smootherOptions = fixture.smootherOptions();
        if (points.stream().anyMatch(point -> point.surface().equals("smoothed_state_autocov"))) {
            smootherOptions = smootherOptions.with(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);
        }
        FilterResult filterResult = KalmanEngine.filter(model, initialState, smootherOptions.requiredFilterOptions());
        SmootherResult smootherResult = SmootherEngine.smooth(model, filterResult, smootherOptions);

        for (SurfacePoint point : points) {
            double actual = actualValue(smootherResult, point);
            assertClose(point.value(), actual, fixture.tolerance(), point.caseId() + " " + point.surface()
                + " t=" + point.t() + " row=" + point.row() + " col=" + point.col());
        }
    }

    static Stream<AlignmentCase> smootherCases() {
        return KalmanStatsmodelsFixtures.cases().stream()
            .filter(fixture -> !fixture.complex())
            .filter(AlignmentCase::hasSmootherSurfaces);
    }

    private static double actualValue(SmootherResult result, SurfacePoint point) {
        return switch (point.surface()) {
            case "smoothed_state" -> result.smoothedState(point.row(), point.t());
            case "smoothed_state_cov" -> result.smoothedStateCov(point.row(), point.col(), point.t());
            case "smoothed_state_autocov" -> result.smoothedStateAutocovariance(point.row(), point.col(), point.t());
            case "smoothed_measurement_disturbance" -> result.smoothedObsDisturbance(point.row(), point.t());
            case "smoothed_state_disturbance" -> result.smoothedStateDisturbance(point.row(), point.t());
            case "smoothed_measurement_disturbance_cov" -> result.smoothedObsDisturbanceCov(point.row(), point.col(), point.t());
            case "smoothed_state_disturbance_cov" -> result.smoothedStateDisturbanceCov(point.row(), point.col(), point.t());
            case "scaled_smoothed_estimator" -> result.scaledSmoothedEstimator[
                result.scaledSmoothedEstimatorOffset(point.t()) + point.row()];
            case "scaled_smoothed_estimator_cov" -> result.scaledSmoothedEstimatorCov[
                result.scaledSmoothedEstimatorCovOffset(point.t()) + point.row() * result.kStates + point.col()];
            default -> throw new IllegalArgumentException("Unsupported smoother surface: " + point.surface());
        };
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