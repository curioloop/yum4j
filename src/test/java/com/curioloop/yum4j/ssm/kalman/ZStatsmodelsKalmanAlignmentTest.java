package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.AlignmentCase;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.SurfacePoint;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.ZSmootherResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZStatsmodelsKalmanAlignmentTest {

    @ParameterizedTest(name = "{0}")
    @MethodSource("complexFilterCases")
    void complexFilterSurfacesMatchStatsmodelsReference(AlignmentCase fixture) {
        ModelFixture model = fixture.modelFixture();
        InitialState initialState = fixture.initialState(model);
        ZFilterResult result = KalmanEngine.filterComplex(model, initialState, fixture.filterOptions());

        assertEquals((int) fixture.scalar("nobsDiffuse"), result.nobsDiffuse(), fixture.id() + " nobsDiffuse");
        if (fixture.filter().concentratedLikelihood()) {
            assertClose(fixture.scalar("scale"), result.scale(), fixture.tolerance(), fixture.id() + " scale");
        }
        for (SurfacePoint point : KalmanStatsmodelsFixtures.surfaces(fixture)) {
            assertComplexClose(point.value(), point.valueImag(), actualFilterValue(result, point),
                fixture.tolerance(), point.caseId() + " " + point.surface()
                    + " t=" + point.t() + " row=" + point.row() + " col=" + point.col());
        }
    }

    @Test
    void complexSmootherSurfacesMatchStatsmodelsReference() {
        complexSmootherCases().forEach(this::assertComplexSmootherSurfacesMatchStatsmodelsReference);
    }

    private void assertComplexSmootherSurfacesMatchStatsmodelsReference(AlignmentCase fixture) {
        ModelFixture model = fixture.modelFixture();
        InitialState initialState = fixture.initialState(model);
        List<SurfacePoint> points = KalmanStatsmodelsFixtures.smootherSurfaces(fixture);
        SmootherOptions smootherOptions = fixture.smootherOptions();
        if (points.stream().anyMatch(point -> point.surface().equals("smoothed_state_autocov"))) {
            smootherOptions = smootherOptions.with(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);
        }
        ZFilterResult filterResult = KalmanEngine.filterComplex(model, initialState,
            smootherOptions.requiredFilterOptions());
        ZSmootherResult smootherResult = SmootherEngine.smoothComplex(model, filterResult, smootherOptions);

        for (SurfacePoint point : points) {
            assertComplexClose(point.value(), point.valueImag(), actualSmootherValue(smootherResult, point),
                fixture.tolerance(), point.caseId() + " " + point.surface()
                    + " t=" + point.t() + " row=" + point.row() + " col=" + point.col());
        }
    }

    static Stream<AlignmentCase> complexFilterCases() {
        return KalmanStatsmodelsFixtures.cases().stream()
            .filter(AlignmentCase::complex);
    }

    static Stream<AlignmentCase> complexSmootherCases() {
        return KalmanStatsmodelsFixtures.cases().stream()
            .filter(AlignmentCase::complex)
            .filter(AlignmentCase::hasSmootherSurfaces);
    }

    private static double[] actualFilterValue(ZFilterResult result, SurfacePoint point) {
        if (point.surface().equals("log_likelihood_obs")) {
            return new double[]{result.logLikelihoodObs[result.logLikelihoodObsOffset(point.t())], 0.0};
        }
        double[] surface = switch (point.surface()) {
            case "forecast" -> result.forecast;
            case "forecast_error" -> result.forecastError;
            case "forecast_error_cov" -> result.forecastErrorCov;
            case "predicted_state" -> result.predictedState;
            case "predicted_state_cov" -> result.predictedStateCov;
            case "filtered_state" -> result.filteredState;
            case "filtered_state_cov" -> result.filteredStateCov;
            case "kalman_gain" -> result.kalmanGain;
            case "standardized_forecast_error" -> result.standardizedForecastError;
            case "predicted_diffuse_state_cov" -> result.predictedDiffuseStateCov;
            case "forecast_error_diffuse_cov" -> result.forecastErrorDiffuseCov;
            default -> throw new IllegalArgumentException("Unsupported filter surface: " + point.surface());
        };
        int offset = switch (point.surface()) {
            case "forecast" -> result.forecastOffset(point.t()) + point.row() * 2;
            case "forecast_error" -> result.forecastErrorOffset(point.t()) + point.row() * 2;
            case "forecast_error_cov" -> result.forecastErrorCovOffset(point.t())
                + point.row() * 2 * result.kEndog + point.col() * 2;
            case "predicted_state" -> result.predictedStateOffset(point.t()) + point.row() * 2;
            case "predicted_state_cov" -> result.predictedStateCovOffset(point.t())
                + point.row() * 2 * result.kStates + point.col() * 2;
            case "filtered_state" -> result.filteredStateOffset(point.t()) + point.row() * 2;
            case "filtered_state_cov" -> result.filteredStateCovOffset(point.t())
                + point.row() * 2 * result.kStates + point.col() * 2;
            case "kalman_gain" -> result.kalmanGainOffset(point.t())
                + point.row() * 2 * result.kEndog + point.col() * 2;
            case "standardized_forecast_error" -> result.standardizedForecastErrorOffset(point.t()) + point.row() * 2;
            case "predicted_diffuse_state_cov" -> result.predictedDiffuseStateCovOffset(point.t())
                + point.row() * 2 * result.kStates + point.col() * 2;
            case "forecast_error_diffuse_cov" -> result.forecastErrorDiffuseCovOffset(point.t())
                + point.row() * 2 * result.kEndog + point.col() * 2;
            default -> throw new IllegalArgumentException("Unsupported filter surface: " + point.surface());
        };
        return new double[]{surface[offset], surface[offset + 1]};
    }

    private static double[] actualSmootherValue(ZSmootherResult result, SurfacePoint point) {
        double[] surface = switch (point.surface()) {
            case "smoothed_state" -> result.smoothedState;
            case "smoothed_state_cov" -> result.smoothedStateCov;
            case "smoothed_state_autocov" -> result.smoothedStateAutocovariance;
            case "smoothed_measurement_disturbance" -> result.smoothedObsDisturbance;
            case "smoothed_state_disturbance" -> result.smoothedStateDisturbance;
            case "smoothed_measurement_disturbance_cov" -> result.smoothedObsDisturbanceCov;
            case "smoothed_state_disturbance_cov" -> result.smoothedStateDisturbanceCov;
            case "scaled_smoothed_estimator" -> result.scaledSmoothedEstimator;
            case "scaled_smoothed_estimator_cov" -> result.scaledSmoothedEstimatorCov;
            default -> throw new IllegalArgumentException("Unsupported smoother surface: " + point.surface());
        };
        int offset = switch (point.surface()) {
            case "smoothed_state" -> result.smoothedStateOffset(point.t()) + point.row() * 2;
            case "smoothed_state_cov" -> result.smoothedStateCovOffset(point.t())
                + point.row() * 2 * result.kStates + point.col() * 2;
            case "smoothed_state_autocov" -> result.smoothedStateAutocovarianceOffset(point.t())
                + point.row() * 2 * result.kStates + point.col() * 2;
            case "smoothed_measurement_disturbance" -> result.smoothedObsDisturbanceOffset(point.t()) + point.row() * 2;
            case "smoothed_state_disturbance" -> result.smoothedStateDisturbanceOffset(point.t()) + point.row() * 2;
            case "smoothed_measurement_disturbance_cov" -> result.smoothedObsDisturbanceCovOffset(point.t())
                + point.row() * 2 * result.kEndog + point.col() * 2;
            case "smoothed_state_disturbance_cov" -> result.smoothedStateDisturbanceCovOffset(point.t())
                + point.row() * 2 * result.kPosdef + point.col() * 2;
            case "scaled_smoothed_estimator" -> result.scaledSmoothedEstimatorOffset(point.t()) + point.row() * 2;
            case "scaled_smoothed_estimator_cov" -> result.scaledSmoothedEstimatorCovOffset(point.t())
                + point.row() * 2 * result.kStates + point.col() * 2;
            default -> throw new IllegalArgumentException("Unsupported smoother surface: " + point.surface());
        };
        return new double[]{surface[offset], surface[offset + 1]};
    }

    private static void assertComplexClose(double expectedReal,
                                           double expectedImag,
                                           double[] actual,
                                           double tolerance,
                                           String message) {
        assertClose(expectedReal, actual[0], tolerance, message + " Re");
        assertClose(expectedImag, actual[1], tolerance, message + " Im");
    }

    private static void assertClose(double expected, double actual, double tolerance, String message) {
        double scale = Math.max(1.0, Math.max(Math.abs(expected), Math.abs(actual)));
        assertEquals(expected, actual, tolerance * scale, message);
    }
}