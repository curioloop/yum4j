package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.AlignmentCase;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.SurfacePoint;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CFASimulationSmootherStatsmodelsAlignmentTest {

    @ParameterizedTest(name = "{0}")
    @MethodSource("cfaCases")
    void cfaPosteriorSurfacesMatchStatsmodelsReference(AlignmentCase fixture) {
        ModelFixture model = fixture.modelFixture();
        InitialState initialState = fixture.initialState(model);
        CFASimulationSmoother.PosteriorMoments moments = CFASimulationSmoother.posteriorMoments(model, initialState);
        List<SurfacePoint> points = KalmanStatsmodelsFixtures.cfaSurfaces(fixture);

        for (SurfacePoint point : points) {
            double actual = actualValue(moments, point);
            assertClose(point.value(), actual, fixture.tolerance(), point.caseId() + " " + point.surface()
                + " t=" + point.t() + " row=" + point.row() + " col=" + point.col());
        }
    }

    static Stream<AlignmentCase> cfaCases() {
        return KalmanStatsmodelsFixtures.cases().stream()
            .filter(fixture -> !fixture.complex())
            .filter(AlignmentCase::hasCfaSurfaces);
    }

    private static double actualValue(CFASimulationSmoother.PosteriorMoments moments, SurfacePoint point) {
        return switch (point.surface()) {
            case "posterior_mean" -> moments.mean(point.row(), point.t());
            case "posterior_cov" -> moments.covariance(point.row(), point.col(), point.t());
            default -> throw new IllegalArgumentException("Unsupported CFA surface: " + point.surface());
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
