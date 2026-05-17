package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.AlignmentCase;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.SurfacePoint;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StatsmodelsKalmanAlignmentTest {

    @ParameterizedTest(name = "{0}")
    @MethodSource("filterCases")
    void filterSurfacesMatchStatsmodelsReference(AlignmentCase fixture) {
        ModelFixture model = fixture.modelFixture();
        InitialState initialState = fixture.initialState(model);
        FilterResult result = filterResult(fixture, model, initialState);

        if (retainsLikelihood(fixture)) {
            assertClose(fixture.scalar("llf"), result.logLikelihood(), fixture.tolerance(), fixture.id() + " llf");
        }
        if (fixture.scalars().containsKey("scale")) {
            assertClose(fixture.scalar("scale"), result.scale(), fixture.tolerance(), fixture.id() + " scale");
        }
        if (fixture.scalars().containsKey("llfBurn") && retainsLikelihood(fixture)) {
            int start = (int) fixture.scalar("logLikelihoodStart");
            assertClose(fixture.scalar("llfBurn"), logLikelihoodFrom(result, start),
                fixture.tolerance(), fixture.id() + " llfBurn");
        }
        assertEquals((int) fixture.scalar("nobsDiffuse"), result.nobsDiffuse(), fixture.id() + " nobsDiffuse");
        for (SurfacePoint point : KalmanStatsmodelsFixtures.surfaces(fixture)) {
            double actual = actualValue(result, point);
            assertClose(point.value(), actual, fixture.tolerance(), point.caseId() + " " + point.surface()
                + " t=" + point.t() + " row=" + point.row() + " col=" + point.col());
        }
    }

    static Stream<AlignmentCase> filterCases() {
        return KalmanStatsmodelsFixtures.cases().stream()
            .filter(fixture -> !fixture.complex());
    }

    private static FilterResult filterResult(AlignmentCase fixture, ModelFixture model, InitialState initialState) {
        if (!fixture.hasEndogOverride()) {
            return KalmanEngine.filter(model, initialState, fixture.filterOptions());
        }
        return KalmanEngine.filterBorrowedUnsafe(model, initialState,
            flatten(fixture.model().endogOverride()), 0,
            KalmanEngine.workspace(), fixture.filterOptions());
    }

    private static double[] flatten(double[][] rows) {
        int rowLength = rows.length == 0 ? 0 : rows[0].length;
        double[] values = new double[rows.length * rowLength];
        for (int t = 0; t < rows.length; t++) {
            System.arraycopy(rows[t], 0, values, t * rowLength, rowLength);
        }
        return values;
    }

    private static double actualValue(FilterResult result, SurfacePoint point) {
        return switch (point.surface()) {
            case "log_likelihood_obs" -> result.logLikelihoodObs[result.logLikelihoodObsOffset(point.t())];
            case "forecast" -> result.forecast(point.row(), point.t());
            case "forecast_error" -> result.forecastError(point.row(), point.t());
            case "forecast_error_cov" -> result.forecastErrorCov(point.row(), point.col(), point.t());
            case "predicted_state" -> result.predictedState(point.row(), point.t());
            case "predicted_state_cov" -> result.predictedStateCov(point.row(), point.col(), point.t());
            case "filtered_state" -> result.filteredState(point.row(), point.t());
            case "filtered_state_cov" -> result.filteredStateCov(point.row(), point.col(), point.t());
            case "kalman_gain" -> result.kalmanGain(point.row(), point.col(), point.t());
            case "standardized_forecast_error" -> result.standardizedForecastError(point.row(), point.t());
            case "predicted_diffuse_state_cov" -> result.predictedDiffuseStateCov(point.row(), point.col(), point.t());
            case "forecast_error_diffuse_cov" -> result.forecastErrorDiffuseCov(point.row(), point.col(), point.t());
            default -> throw new IllegalArgumentException("Unsupported surface: " + point.surface());
        };
    }

    private static double logLikelihoodFrom(FilterResult result, int start) {
        double sum = 0.0;
        for (int t = start; t < result.nobs; t++) {
            sum += result.logLikelihoodObs[result.logLikelihoodObsOffset(t)];
        }
        return sum;
    }

    private static boolean retainsLikelihood(AlignmentCase fixture) {
        String[] retainedSurfaces = fixture.filter().retainedSurfaces();
        if (retainedSurfaces == null) {
            return fixture.filter().retainAll();
        }
        for (String retainedSurface : retainedSurfaces) {
            if (retainedSurface.equals("LIKELIHOOD")) {
                return true;
            }
        }
        return false;
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