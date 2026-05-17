package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.AlignmentCase;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.SurfacePoint;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.ssm.kalman.filter.*;
import com.curioloop.yum4j.ssm.kalman.init.*;
import com.curioloop.yum4j.ssm.kalman.model.*;
import com.curioloop.yum4j.ssm.kalman.smooth.*;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class StatsmodelsComprehensiveTest {

    private static final double TOL_REF = 1e-4;
    private static final double TOL_STRUCT = 1e-8;
    private static final double TOL_LOOSE = 1e-2;
    private static final String CLARK1987_CASE_ID = "clark1987_gdp_known_conventional";
    private static final String CLARK1989_CASE_ID = "clark1989_gdp_unemployment_known_conventional";
    private static final String WPI_AR3_CASE_ID = "wpi_ar3_stationary_conventional";
    private static final String WPI_AR3_APPROX_DIFFUSE_CASE_ID = "wpi_ar3_approximate_diffuse_conventional";
    private static final String WPI_AR3_MISSING_CASE_ID = "wpi_ar3_missing_stationary_conventional";
    private static final String MACRODATA_MISSING_CASE_ID = "macrodata_diffuse_missing_conventional";
    private static final String MACRODATA_VAR_CASE_ID = "macrodata_var_diffuse_conventional";

    private static ModelFixture buildClark1987Model() {
        return alignmentCase(CLARK1987_CASE_ID).modelFixture();
    }

    private static double[][] clark1987Init() {
        AlignmentCase fixture = alignmentCase(CLARK1987_CASE_ID);
        return new double[][]{fixture.initialization().state(), fixture.initialization().cov()};
    }

    private static ModelFixture buildClark1989Model() {
        return alignmentCase(CLARK1989_CASE_ID).modelFixture();
    }

    private static double[][] clark1989Init() {
        AlignmentCase fixture = alignmentCase(CLARK1989_CASE_ID);
        return new double[][]{fixture.initialization().state(), fixture.initialization().cov()};
    }

    private static ModelFixture buildAR3Model() {
        return alignmentCase(WPI_AR3_CASE_ID).modelFixture();
    }

    private static double[][] ar3StationaryInit() {
        double phi1 = 0.5270715, phi2 = 0.0952613, phi3 = 0.2580355;
        double sigma2 = 0.5307459;
        double[] T = {phi1, phi2, phi3, 1, 0, 0, 0, 1, 0};
        double[] R = {1, 0, 0};
        int k = 3;
        double[] P = new double[k * k];
        double[] tmp = new double[k * k];
        double[] RQ = new double[k];
        for (int i = 0; i < k; i++) RQ[i] = R[i] * sigma2;
        for (int iter = 0; iter < 1000; iter++) {
            for (int i = 0; i < k; i++)
                for (int j = 0; j < k; j++) {
                    tmp[i * k + j] = 0;
                    for (int m = 0; m < k; m++)
                        for (int n = 0; n < k; n++)
                            tmp[i * k + j] += T[i * k + m] * P[m * k + n] * T[j * k + n];
                }
            for (int i = 0; i < k; i++)
                for (int j = 0; j < k; j++)
                    tmp[i * k + j] += RQ[i] * R[j];
            double maxDiff = 0;
            for (int i = 0; i < k * k; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(tmp[i] - P[i]));
                P[i] = tmp[i];
            }
            if (maxDiff < 1e-15) break;
        }
        double[] a0 = new double[k];
        return new double[][]{a0, P};
    }

    private static ModelFixture buildBivariateAR1(double[] y) {
        int nobs = y.length / 2;
        int kEndog = 2, kStates = 4, kPosdef = 2;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1, 0, 0, 0, 0, 1, 0, 0};
        double[] H = {0.1, 0, 0, 0.1};
        double[] T = {0.9, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0.6};
        double[] R = {1, 0, 0, 1, 0, 0, 0, 0};
        double[] Q = {1, 0, 0, 1};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t * 2], y[t * 2 + 1]}, t);
        }
        return rep;
    }

    private static double[] generateBivariate(int n, long seed) {
        Random rng = new Random(seed);
        double[] y = new double[n * 2];
        double s0 = 0, s1 = 0;
        for (int t = 0; t < n; t++) {
            s0 = 0.9 * s0 + rng.nextGaussian();
            s1 = 0.8 * s1 + rng.nextGaussian();
            y[t * 2] = s0 + rng.nextGaussian() * Math.sqrt(0.1);
            y[t * 2 + 1] = s1 + rng.nextGaussian() * Math.sqrt(0.1);
        }
        return y;
    }

    private static InitialState knownInitialization(double[][] init) {
        return InitialState.known(init[0], init[1]);
    }

    private static AlignmentCase alignmentCase(String id) {
        return KalmanStatsmodelsFixtures.cases().stream()
            .filter(fixture -> fixture.id().equals(id))
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException("Missing Kalman statsmodels fixture: " + id));
    }

    private static FilterResult filterAlignmentCase(AlignmentCase fixture) {
        ModelFixture model = fixture.modelFixture();
        return KalmanEngine.filter(model, fixture.initialState(model), fixture.filterOptions());
    }

    private static SmootherResult smoothAlignmentCase(AlignmentCase fixture) {
        ModelFixture model = fixture.modelFixture();
        SmootherOptions smootherOptions = fixture.smootherOptions();
        FilterResult filter = KalmanEngine.filter(model, fixture.initialState(model), smootherOptions.requiredFilterOptions());
        return SmootherEngine.smooth(model, filter, smootherOptions);
    }

    private static double logLikelihoodFrom(FilterResult result, int start) {
        double sum = 0.0;
        for (int t = start; t < result.nobs; t++) {
            sum += result.logLikelihoodObs[result.logLikelihoodObsOffset(t)];
        }
        return sum;
    }

    private static void assertFilterSurface(AlignmentCase fixture, FilterResult result, String surface) {
        for (SurfacePoint point : KalmanStatsmodelsFixtures.surfaces(fixture)) {
            if (!point.surface().equals(surface)) {
                continue;
            }
            assertFixtureClose(point.value(), filterSurfaceValue(result, point), fixture.tolerance(),
                point.caseId() + " " + surface + " t=" + point.t()
                    + " row=" + point.row() + " col=" + point.col());
        }
    }

    private static double filterSurfaceValue(FilterResult result, SurfacePoint point) {
        return switch (point.surface()) {
            case "forecast" -> result.forecast(point.row(), point.t());
            case "forecast_error" -> result.forecastError(point.row(), point.t());
            case "forecast_error_cov" -> result.forecastErrorCov(point.row(), point.col(), point.t());
            case "predicted_state" -> result.predictedState(point.row(), point.t());
            case "predicted_state_cov" -> result.predictedStateCov(point.row(), point.col(), point.t());
            case "filtered_state" -> result.filteredState(point.row(), point.t());
            case "filtered_state_cov" -> result.filteredStateCov(point.row(), point.col(), point.t());
            default -> throw new IllegalArgumentException("Unsupported filter surface: " + point.surface());
        };
    }

    private static void assertSmootherSurface(AlignmentCase fixture, SmootherResult result, String surface) {
        for (SurfacePoint point : KalmanStatsmodelsFixtures.smootherSurfaces(fixture)) {
            if (!point.surface().equals(surface)) {
                continue;
            }
            assertFixtureClose(point.value(), smootherSurfaceValue(result, point), fixture.tolerance(),
                point.caseId() + " " + surface + " t=" + point.t()
                    + " row=" + point.row() + " col=" + point.col());
        }
    }

    private static double smootherSurfaceValue(SmootherResult result, SurfacePoint point) {
        return switch (point.surface()) {
            case "smoothed_state" -> result.smoothedState(point.row(), point.t());
            case "smoothed_state_cov" -> result.smoothedStateCov(point.row(), point.col(), point.t());
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

    private static void assertFixtureClose(double expected, double actual, double tolerance, String message) {
        double scale = Math.max(1.0, Math.max(Math.abs(expected), Math.abs(actual)));
        assertEquals(expected, actual, tolerance * scale, message);
    }

    private static InitialState stationaryInitialization(double[][] init) {
        return InitialState.stationary(init[0], init[1]);
    }

    private static InitialState stationaryInitialization(double[] transition, int kStates,
                                                                double[] selection, int kPosdef,
                                                                double[] stateCov, double[] stateIntercept) {
        ModelFixture rep = new ModelFixture(1, kStates, kPosdef, 1);
        rep.setTransition(transition, 0);
        rep.setSelection(selection, 0);
        rep.setStateCov(stateCov, 0);
        rep.setStateIntercept(stateIntercept, 0);
        return InitialState.stationary(rep);
    }

    private static ModelFixture buildMultivariateMissingModel() {
        return alignmentCase(MACRODATA_MISSING_CASE_ID).modelFixture();
    }

    private static ModelFixture buildMultivariateVARModel() {
        return alignmentCase(MACRODATA_VAR_CASE_ID).modelFixture();
    }

    @Test
    void testClark1987Loglike() {
        AlignmentCase fixture = alignmentCase(CLARK1987_CASE_ID);
        FilterResult result = filterAlignmentCase(fixture);
        int start = (int) fixture.scalar("logLikelihoodStart");
        assertFixtureClose(fixture.scalar("llfBurn"), logLikelihoodFrom(result, start),
            fixture.tolerance(), "Clark1987 burned loglike");
    }

    @Test
    void testClark1987FilteredState() {
        AlignmentCase fixture = alignmentCase(CLARK1987_CASE_ID);
        assertFilterSurface(fixture, filterAlignmentCase(fixture), "filtered_state");
    }

    @Test
    void testClark1989Loglike() {
        AlignmentCase fixture = alignmentCase(CLARK1989_CASE_ID);
        assertFixtureClose(fixture.scalar("llf"), filterAlignmentCase(fixture).logLikelihood(),
            fixture.tolerance(), "Clark1989 loglike");
    }

    @Test
    void testClark1989FilteredState() {
        AlignmentCase fixture = alignmentCase(CLARK1989_CASE_ID);
        assertFilterSurface(fixture, filterAlignmentCase(fixture), "filtered_state");
    }

    @Test
    void testStationaryInit1D() {
        double phi = 0.5, sigma2 = 1.0;
        double[] T = {phi};
        double[] R = {1.0};
        double[] Q = {sigma2};
        double[] c = {0.0};
        InitialState init = stationaryInitialization(T, 1, R, 1, Q, c);
        assertEquals(0.0, init.initialState()[0], TOL_STRUCT);
        assertEquals(sigma2 / (1 - phi * phi), init.initialStateCov()[0], TOL_STRUCT);
    }

    @Test
    void testStationaryInit2D() {
        double[] T = {0.5, 0.1, 0, 0.5};
        double[] R = {1, 0, 0, 1};
        double[] Q = {1, 0, 0, 1};
        double[] c = {0, 0};
        InitialState init = stationaryInitialization(T, 2, R, 2, Q, c);
        assertEquals(0.0, init.initialState()[0], TOL_STRUCT);
        assertEquals(0.0, init.initialState()[1], TOL_STRUCT);
        assertTrue(init.initialStateCov()[0] > 0);
        assertTrue(init.initialStateCov()[3] > 0);
        assertEquals(init.initialStateCov()[1], init.initialStateCov()[2], TOL_STRUCT);
    }

    @Test
    void testStationaryInitWithIntercept() {
        double phi = 0.5;
        double[] T = {phi};
        double[] R = {1.0};
        double[] Q = {1.0};
        double[] c = {2.0};
        InitialState init = stationaryInitialization(T, 1, R, 1, Q, c);
        double expectedA = c[0] / (1 - phi);
        assertEquals(expectedA, init.initialState()[0], TOL_STRUCT);
    }

    @Test
    void testAR3SmoothedState() {
        AlignmentCase fixture = alignmentCase(WPI_AR3_APPROX_DIFFUSE_CASE_ID);
        assertSmootherSurface(fixture, smoothAlignmentCase(fixture), "smoothed_state");
    }

    @Test
    void testAR3SmoothedStateCovDet() {
        AlignmentCase fixture = alignmentCase(WPI_AR3_APPROX_DIFFUSE_CASE_ID);
        assertSmootherSurface(fixture, smoothAlignmentCase(fixture), "smoothed_state_cov");
    }

    @Test
    void testAR3SmoothedObsDisturbance() {
        AlignmentCase fixture = alignmentCase(WPI_AR3_APPROX_DIFFUSE_CASE_ID);
        assertSmootherSurface(fixture, smoothAlignmentCase(fixture), "smoothed_measurement_disturbance");
    }

    @Test
    void testAR3SmoothedStateDisturbanceVsStatsmodelsFixture() {
        AlignmentCase fixture = alignmentCase(WPI_AR3_CASE_ID);
        assertSmootherSurface(fixture, smoothAlignmentCase(fixture), "smoothed_state_disturbance");
    }

    @Test
    void testMultivariateMissingSmoothedState() {
        AlignmentCase fixture = alignmentCase(MACRODATA_MISSING_CASE_ID);
        assertSmootherSurface(fixture, smoothAlignmentCase(fixture), "smoothed_state");
    }

    @Test
    void testMultivariateMissingLoglike() {
        AlignmentCase fixture = alignmentCase(MACRODATA_MISSING_CASE_ID);
        assertFixtureClose(fixture.scalar("llf"), filterAlignmentCase(fixture).logLikelihood(),
            fixture.tolerance(), "MultivariateMissing loglike");
    }

    @Test
    void testMultivariateMissingConvVsUni() {
        ModelFixture rep = buildMultivariateMissingModel();
        FilterResult frConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(frUni.logLikelihood(), frConv.logLikelihood(), TOL_REF,
                "Conv vs Uni logLikelihood mismatch");

        SmootherResult srConv = SmootherEngine.smooth(rep, frConv, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        SmootherResult srUni = SmootherEngine.smooth(rep, frUni, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        int nobs = frConv.nobs;
        int kStates = 3;
        for (int t = 1; t < nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                assertEquals(srConv.smoothedState(i, t), srUni.smoothedState(i, t), TOL_REF,
                        "Conv vs Uni smoothed state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testMultivariateVARSmoothedState() {
        AlignmentCase fixture = alignmentCase(MACRODATA_VAR_CASE_ID);
        assertSmootherSurface(fixture, smoothAlignmentCase(fixture), "smoothed_state");
    }

    @Test
    void testMultivariateVARLoglike() {
        AlignmentCase fixture = alignmentCase(MACRODATA_VAR_CASE_ID);
        assertFixtureClose(fixture.scalar("llf"), filterAlignmentCase(fixture).logLikelihood(),
            fixture.tolerance(), "MultivariateVAR loglike");
    }

    @Test
    void testUnivariateClark1989Forecasts() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();

        FilterResult frConv = KalmanEngine.filter(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        int nobs = frConv.nobs;
        int kEndog = 2;
        for (int t = 4; t < nobs; t++) {
            assertEquals(frConv.forecast(0, t), frUni.forecast(0, t), TOL_REF,
                    "forecast mismatch at t=" + t + " i=0");
            for (int i = 0; i < kEndog; i++) {
                assertEquals(rep.endog[rep.endogOffset(t) + i],
                        frUni.forecast(i, t) + frUni.forecastError(i, t), TOL_REF,
                        "forecast + forecastError != y at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testUnivariateClark1989FilteredState() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();

        FilterResult frConv = KalmanEngine.filter(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        int nobs = frConv.nobs;
        int kStates = 6;
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                assertEquals(frConv.filteredState(i, t), frUni.filteredState(i, t), TOL_REF,
                        "filtered state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testUnivariateClark1989Loglike() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();

        FilterResult frConv = KalmanEngine.filter(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF);
    }

    @Test
    void testUnivariateClark1989SmoothedState() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();

        FilterResult frConv = KalmanEngine.filter(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult srConv = SmootherEngine.smooth(rep, frConv, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        FilterResult frUni = KalmanEngine.filterUnivariate(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult srUni = SmootherEngine.smooth(rep, frUni, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        int nobs = frConv.nobs;
        int kStates = 6;
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                assertEquals(srConv.smoothedState(i, t), srUni.smoothedState(i, t), TOL_REF,
                        "smoothed state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testUnivariateMissingNone() {
        double[] y = generateBivariate(20, 111);
        ModelFixture rep = buildBivariateAR1(y);
        FilterResult frConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF);
    }

    @Test
    void testUnivariateMissingAll() {
        double[] y = generateBivariate(20, 333);
        ModelFixture rep = buildBivariateAR1(y);
        rep.setMissing(new boolean[]{true, true}, 5);
        FilterResult frConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF);
    }

    @Test
    void testUnivariateMissingPartial() {
        double[] y = generateBivariate(20, 222);
        ModelFixture rep = buildBivariateAR1(y);
        rep.setMissing(new boolean[]{true, false}, 5);
        rep.setMissing(new boolean[]{false, true}, 10);
        FilterResult frConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF);
    }

    @Test
    void testUnivariateMissingMixed() {
        double[] y = generateBivariate(30, 444);
        ModelFixture rep = buildBivariateAR1(y);
        rep.setMissing(new boolean[]{true, true}, 5);
        rep.setMissing(new boolean[]{true, false}, 10);
        rep.setMissing(new boolean[]{false, true}, 15);
        rep.setMissing(new boolean[]{true, false}, 20);
        rep.setMissing(new boolean[]{true, true}, 25);
        FilterResult frConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF);
    }

    @Test
    void testDiffuseLocalLevelAnalytic() {
        int nobs = 10;
        ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
        double[] Z = {1}, H = {1.993}, T = {1}, R = {1}, Q = {8.253};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
        }
        double[] y = {26.3, 23.96, 18.93, 13.83, 19.13, 2.92, 2.6, 0.68, 16.7, 8.57};
        for (int t = 0; t < nobs; t++) rep.setEndog(new double[]{y[t]}, t);

        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertEquals(1e6, fr.predictedStateCov(0, 0, 0), TOL_STRUCT);
    }

    @Test
    void testDiffuseVsApproximate() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 50, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frDiffuse = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frApprox = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        int skip = 5;
        double llDiff = 0, llApprox = 0;
        for (int t = skip; t < 50; t++) {
            llDiff += frDiffuse.logLikelihoodObs[t];
            llApprox += frApprox.logLikelihoodObs[t];
        }
        assertEquals(llApprox, llDiff, 0.5);
    }

    private static ModelFixture buildAR1(double phi, double sigma2, double[] y) {
        return KalmanModelFixtures.statsmodelsAr1(phi, sigma2, y);
    }

    private static double[] generateAR1(double phi, double sigma2, int n, long seed) {
        Random rng = new Random(seed);
        double[] y = new double[n];
        double state = 0;
        for (int t = 0; t < n; t++) {
            state = phi * state + rng.nextGaussian() * Math.sqrt(sigma2);
            y[t] = state + rng.nextGaussian() * Math.sqrt(0.5);
        }
        return y;
    }

    @Test
    void testDiffuseLocalLinearTrend() {
        int nobs = 10;
        int kStates = 2, kPosdef = 2;
        ModelFixture rep = new ModelFixture(1, kStates, kPosdef, nobs);
        double[] Z = {1, 0};
        double[] H = {1.993};
        double[] T = {1, 1, 0, 1};
        double[] R = {1, 0, 0, 1};
        double[] Q = {8.253, 0, 0, 2.334};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
        }
        double[] y = {26.3, 23.96, 18.93, 13.83, 19.13, 2.92, 2.6, 0.68, 16.7, 8.57};
        for (int t = 0; t < nobs; t++) rep.setEndog(new double[]{y[t]}, t);

        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertEquals(1e6, fr.predictedStateCov(0, 0, 0), TOL_STRUCT);
        assertEquals(1e6, fr.predictedStateCov(1, 1, 0), TOL_STRUCT);
    }

    @Test
    void testMemoryNoLikelihood() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frNoLL = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
            FilterOptions.defaults().without(FilterOptions.Surface.LIKELIHOOD));
        assertEquals(0, frNoLL.logLikelihoodObs.length);
    }

    @Test
    void testMemoryNoForecast() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frAll = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frNoFC = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
            FilterOptions.builder()
                .drop(FilterOptions.Surface.FORECAST_MEAN,
                    FilterOptions.Surface.FORECAST_ERROR,
                    FilterOptions.Surface.FORECAST_COVARIANCE,
                    FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR,
                    FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE)
                .build());
        assertEquals(0, frNoFC.forecast.length);
        assertEquals(0, frNoFC.forecastError.length);
        assertEquals(0, frNoFC.forecastErrorCov.length);
        assertEquals(frAll.logLikelihood(), frNoFC.logLikelihood(), TOL_REF);
    }

    @Test
    void testMemoryNoPredicted() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frNoPred = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
            FilterOptions.builder()
                .drop(FilterOptions.Surface.PREDICTED_STATE,
                    FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
                    FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)
                .build());
        assertEquals(0, frNoPred.predictedState.length);
        assertEquals(0, frNoPred.predictedStateCov.length);
    }

    @Test
    void testMemoryNoFiltered() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frNoFilt = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
            FilterOptions.builder()
                .drop(FilterOptions.Surface.FILTERED_STATE,
                    FilterOptions.Surface.FILTERED_STATE_COVARIANCE)
                .build());
        assertEquals(0, frNoFilt.filteredState.length);
        assertEquals(0, frNoFilt.filteredStateCov.length);
    }

    @Test
    void testMemoryNoGain() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frAll = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frNoGain = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
            FilterOptions.builder()
                .drop(FilterOptions.Surface.KALMAN_GAIN)
                .build());
        assertEquals(0, frNoGain.kalmanGain.length);
        assertEquals(frAll.logLikelihood(), frNoGain.logLikelihood(), TOL_STRUCT);
    }

    @Test
    void testMemoryConserve() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
            FilterOptions.compact());
        assertEquals(0, fr.forecast.length);
        assertEquals(0, fr.forecastError.length);
        assertEquals(0, fr.forecastErrorCov.length);
        assertEquals(0, fr.filteredState.length);
        assertEquals(0, fr.filteredStateCov.length);
        assertEquals(0, fr.predictedState.length);
        assertEquals(0, fr.predictedStateCov.length);
        assertEquals(0, fr.kalmanGain.length);
        assertEquals(0, fr.logLikelihoodObs.length);
    }

    @Test
    void testSmootherMethodsIdentical() {
        double phi = 0.8;
        double[] y = generateAR1(phi, 1.0, 30, 12345);
        ModelFixture rep = buildAR1(phi, 0.3, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        SmootherResult conv = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        SmootherResult cls = SmootherEngine.smooth(rep, fr, SmootherOptions.classical());
        SmootherResult alt = SmootherEngine.smooth(rep, fr, SmootherOptions.alternative());

        int nobs = 30;
        for (int t = 0; t < nobs; t++) {
            assertEquals(conv.smoothedState(0, t), cls.smoothedState(0, t), TOL_REF,
                    "smoothedState conv vs cls t=" + t);
            assertEquals(conv.smoothedState(0, t), alt.smoothedState(0, t), TOL_REF,
                    "smoothedState conv vs alt t=" + t);
            assertEquals(conv.smoothedStateCov(0, 0, t), cls.smoothedStateCov(0, 0, t), TOL_REF,
                    "smoothedStateCov conv vs cls t=" + t);
            assertEquals(conv.smoothedStateCov(0, 0, t), alt.smoothedStateCov(0, 0, t), TOL_REF,
                    "smoothedStateCov conv vs alt t=" + t);
            assertEquals(conv.smoothedObsDisturbance(0, t), cls.smoothedObsDisturbance(0, t), TOL_REF,
                    "smoothedObsDisturbance conv vs cls t=" + t);
            assertEquals(conv.smoothedObsDisturbance(0, t), alt.smoothedObsDisturbance(0, t), TOL_REF,
                    "smoothedObsDisturbance conv vs alt t=" + t);
            assertEquals(conv.smoothedStateDisturbance(0, t), cls.smoothedStateDisturbance(0, t), TOL_REF,
                    "smoothedStateDisturbance conv vs cls t=" + t);
            assertEquals(conv.smoothedStateDisturbance(0, t), alt.smoothedStateDisturbance(0, t), TOL_REF,
                    "smoothedStateDisturbance conv vs alt t=" + t);
            assertEquals(conv.smoothedObsDisturbanceCov(0, 0, t), cls.smoothedObsDisturbanceCov(0, 0, t), TOL_REF,
                    "smoothedObsDisturbanceCov conv vs cls t=" + t);
            assertEquals(conv.smoothedObsDisturbanceCov(0, 0, t), alt.smoothedObsDisturbanceCov(0, 0, t), TOL_REF,
                    "smoothedObsDisturbanceCov conv vs alt t=" + t);
            assertEquals(conv.smoothedStateDisturbanceCov(0, 0, t), cls.smoothedStateDisturbanceCov(0, 0, t), TOL_REF,
                    "smoothedStateDisturbanceCov conv vs cls t=" + t);
            assertEquals(conv.smoothedStateDisturbanceCov(0, 0, t), alt.smoothedStateDisturbanceCov(0, 0, t), TOL_REF,
                    "smoothedStateDisturbanceCov conv vs alt t=" + t);
        }
    }

    @Test
    void testSmootherMethodsMultivariate() {
        int nobs = 10;
        int kEndog = 2, kStates = 4, kPosdef = 2;
        Random rng = new Random(99);
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1, 0.5, 0, 0, 0, 0, 1, 0};
        double[] H = {0.5, 0.1, 0.1, 0.7};
        double[] T = {0.9, 0, 0.1, 0, 0, 0.95, 0, 0.05, 0, 0, 0.9, 0, 0, 0, 0, 0.85};
        double[] R = {1, 0, 0, 1, 0, 0, 0, 0};
        double[] Q = {0.3, 0, 0, 0.2};

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{rng.nextGaussian(), rng.nextGaussian()}, t);
        }
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        SmootherResult conv = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        SmootherResult cls = SmootherEngine.smooth(rep, fr, SmootherOptions.classical());
        SmootherResult alt = SmootherEngine.smooth(rep, fr, SmootherOptions.alternative());

        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                assertEquals(conv.smoothedState(i, t), cls.smoothedState(i, t), TOL_REF,
                        "smoothedState[" + i + "] conv vs cls t=" + t);
                assertEquals(conv.smoothedState(i, t), alt.smoothedState(i, t), TOL_REF,
                        "smoothedState[" + i + "] conv vs alt t=" + t);
            }
            for (int i = 0; i < kStates; i++)
                for (int j = 0; j < kStates; j++) {
                    double covConv = conv.smoothedStateCov(i, j, t);
                    assertEquals(covConv, cls.smoothedStateCov(i, j, t), TOL_REF,
                            "smoothedStateCov[" + i + "," + j + "] conv vs cls t=" + t);
                    assertEquals(covConv, alt.smoothedStateCov(i, j, t),
                            Math.max(TOL_REF, Math.abs(covConv) * 1e-6),
                            "smoothedStateCov[" + i + "," + j + "] conv vs alt t=" + t);
                }
            for (int i = 0; i < kEndog; i++) {
                assertEquals(conv.smoothedObsDisturbance(i, t), cls.smoothedObsDisturbance(i, t), TOL_REF,
                        "smoothedObsDisturbance[" + i + "] conv vs cls t=" + t);
                assertEquals(conv.smoothedObsDisturbance(i, t), alt.smoothedObsDisturbance(i, t), TOL_REF,
                        "smoothedObsDisturbance[" + i + "] conv vs alt t=" + t);
            }
            for (int i = 0; i < kPosdef; i++) {
                assertEquals(conv.smoothedStateDisturbance(i, t), cls.smoothedStateDisturbance(i, t), TOL_REF,
                        "smoothedStateDisturbance[" + i + "] conv vs cls t=" + t);
                assertEquals(conv.smoothedStateDisturbance(i, t), alt.smoothedStateDisturbance(i, t), TOL_REF,
                        "smoothedStateDisturbance[" + i + "] conv vs alt t=" + t);
            }
        }
    }

    @Test
    void testSmootherMethodsMissingData() {
        double phi = 0.6;
        double[] y = generateAR1(phi, 1.0, 20, 77);
        ModelFixture rep = buildAR1(phi, 0.4, y);
        rep.setMissing(new boolean[]{false, false, true, false, false, true, true, false,
                false, false, true, false, false, false, true, false, false, false, false, false}, 0);
        for (int t = 0; t < 20; t++) {
            if (t == 2 || t == 5 || t == 6 || t == 10 || t == 14) {
                rep.setMissing(new boolean[]{true}, t);
            }
        }
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        SmootherResult conv = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        SmootherResult cls = SmootherEngine.smooth(rep, fr, SmootherOptions.classical());
        SmootherResult alt = SmootherEngine.smooth(rep, fr, SmootherOptions.alternative());

        int nobs = 20;
        for (int t = 0; t < nobs; t++) {
            assertEquals(conv.smoothedState(0, t), cls.smoothedState(0, t), TOL_REF,
                    "smoothedState conv vs cls t=" + t);
            assertEquals(conv.smoothedState(0, t), alt.smoothedState(0, t), TOL_REF,
                    "smoothedState conv vs alt t=" + t);
            assertEquals(conv.smoothedStateCov(0, 0, t), cls.smoothedStateCov(0, 0, t), TOL_REF,
                    "smoothedStateCov conv vs cls t=" + t);
            assertEquals(conv.smoothedStateCov(0, 0, t), alt.smoothedStateCov(0, 0, t), TOL_REF,
                    "smoothedStateCov conv vs alt t=" + t);
            assertEquals(conv.smoothedObsDisturbance(0, t), cls.smoothedObsDisturbance(0, t), TOL_REF,
                    "smoothedObsDisturbance conv vs cls t=" + t);
            assertEquals(conv.smoothedObsDisturbance(0, t), alt.smoothedObsDisturbance(0, t), TOL_REF,
                    "smoothedObsDisturbance conv vs alt t=" + t);
        }
    }

    @Test
    void testAR3PredictedStateVsStatsmodelsFixture() {
        AlignmentCase fixture = alignmentCase(WPI_AR3_CASE_ID);
        assertFilterSurface(fixture, filterAlignmentCase(fixture), "predicted_state");
    }

    @Test
    void testAR3FilteredStateVsStatsmodelsFixture() {
        AlignmentCase fixture = alignmentCase(WPI_AR3_CASE_ID);
        assertFilterSurface(fixture, filterAlignmentCase(fixture), "filtered_state");
    }

    @Test
    void testAR3SmoothedStateVsStatsmodelsFixture() {
        AlignmentCase fixture = alignmentCase(WPI_AR3_CASE_ID);
        assertSmootherSurface(fixture, smoothAlignmentCase(fixture), "smoothed_state");
    }

    @Test
    void testAR3MissingSmoothedStateVsStatsmodelsFixture() {
        AlignmentCase fixture = alignmentCase(WPI_AR3_MISSING_CASE_ID);
        assertSmootherSurface(fixture, smoothAlignmentCase(fixture), "smoothed_state");
    }

    @Test
    void testAR3MissingForecastVsStatsmodelsFixture() {
        AlignmentCase fixture = alignmentCase(WPI_AR3_MISSING_CASE_ID);
        assertFilterSurface(fixture, filterAlignmentCase(fixture), "forecast");
    }

    @Test
    void testMultivariateMissingSmootherR() {
        AlignmentCase fixture = alignmentCase(MACRODATA_MISSING_CASE_ID);
        assertSmootherSurface(fixture, smoothAlignmentCase(fixture), "scaled_smoothed_estimator");
    }

    @Test
    void testMultivariateMissingForecastVsR() {
        AlignmentCase fixture = alignmentCase(MACRODATA_MISSING_CASE_ID);
        assertFilterSurface(fixture, filterAlignmentCase(fixture), "forecast");
    }

    @Test
    void testMultivariateMissingSmoothedDisturbance() {
        AlignmentCase fixture = alignmentCase(MACRODATA_MISSING_CASE_ID);
        SmootherResult result = smoothAlignmentCase(fixture);
        assertSmootherSurface(fixture, result, "smoothed_state_disturbance");
        assertSmootherSurface(fixture, result, "smoothed_measurement_disturbance");
    }

    @Test
    void testMultivariateMissingFullTraceAgainstStatsmodelsFixture() {
        AlignmentCase fixture = alignmentCase(MACRODATA_MISSING_CASE_ID);
        FilterResult filterResult = filterAlignmentCase(fixture);
        SmootherResult smootherResult = smoothAlignmentCase(fixture);
        assertFilterSurface(fixture, filterResult, "forecast_error");
        assertFilterSurface(fixture, filterResult, "forecast_error_cov");
        assertFilterSurface(fixture, filterResult, "predicted_state");
        assertFilterSurface(fixture, filterResult, "predicted_state_cov");
        assertSmootherSurface(fixture, smootherResult, "scaled_smoothed_estimator_cov");
        assertSmootherSurface(fixture, smootherResult, "smoothed_state_cov");
        assertSmootherSurface(fixture, smootherResult, "smoothed_state_disturbance_cov");
        assertSmootherSurface(fixture, smootherResult, "smoothed_measurement_disturbance_cov");
    }

    @Test
    void testMultivariateVARSmootherR() {
        AlignmentCase fixture = alignmentCase(MACRODATA_VAR_CASE_ID);
        assertSmootherSurface(fixture, smoothAlignmentCase(fixture), "scaled_smoothed_estimator");
    }

    @Test
    void testMultivariateVARForecastVsR() {
        AlignmentCase fixture = alignmentCase(MACRODATA_VAR_CASE_ID);
        assertFilterSurface(fixture, filterAlignmentCase(fixture), "forecast");
    }

    @Test
    void testMultivariateVARSmoothedDisturbance() {
        AlignmentCase fixture = alignmentCase(MACRODATA_VAR_CASE_ID);
        SmootherResult result = smoothAlignmentCase(fixture);
        assertSmootherSurface(fixture, result, "smoothed_state_disturbance");
        assertSmootherSurface(fixture, result, "smoothed_measurement_disturbance");
    }

    @Test
    void testMultivariateVARFullTraceAgainstStatsmodelsFixture() {
        AlignmentCase fixture = alignmentCase(MACRODATA_VAR_CASE_ID);
        FilterResult filterResult = filterAlignmentCase(fixture);
        SmootherResult smootherResult = smoothAlignmentCase(fixture);
        assertFilterSurface(fixture, filterResult, "forecast_error");
        assertFilterSurface(fixture, filterResult, "forecast_error_cov");
        assertFilterSurface(fixture, filterResult, "predicted_state");
        assertFilterSurface(fixture, filterResult, "predicted_state_cov");
        assertSmootherSurface(fixture, smootherResult, "scaled_smoothed_estimator_cov");
        assertSmootherSurface(fixture, smootherResult, "smoothed_state_cov");
        assertSmootherSurface(fixture, smootherResult, "smoothed_state_disturbance_cov");
        assertSmootherSurface(fixture, smootherResult, "smoothed_measurement_disturbance_cov");
    }

    @Test
    void testUnivariateNonDiagonalObsCov() {
        int nobs = 20;
        int kEndog = 2, kStates = 2, kPosdef = 2;
        Random rng = new Random(555);
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1, 0, 0, 1};
        double[] H = {0.5, 0, 0, 0.8};
        double[] T = {0.9, 0, 0, 0.8};
        double[] R = {1, 0, 0, 1};
        double[] Q = {1, 0, 0, 1};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{rng.nextGaussian(), rng.nextGaussian()}, t);
        }

        FilterResult frConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_LOOSE,
                "Diagonal obs_cov: loglike mismatch");

        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                assertEquals(frConv.filteredState(i, t), frUni.filteredState(i, t), TOL_LOOSE,
                        "Diagonal obs_cov: filtered state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testUnivariateTimeVaryingT() {
        int nobs = 20;
        int kEndog = 2, kStates = 2, kPosdef = 2;
        Random rng = new Random(666);
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1, 0, 0, 1};
        double[] H = {0.1, 0, 0, 0.1};
        double[] R = {1, 0, 0, 1};
        double[] Q = {1, 0, 0, 1};

        for (int t = 0; t < nobs; t++) {
            double phi1 = 0.5 + 0.3 * Math.sin(t * 0.5);
            double phi2 = 0.3 + 0.2 * Math.cos(t * 0.3);
            double[] T = {phi1, 0, 0, phi2};
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{rng.nextGaussian(), rng.nextGaussian()}, t);
        }

        FilterResult frConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF,
                "Time-varying T: loglike mismatch");

        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                assertEquals(frConv.filteredState(i, t), frUni.filteredState(i, t), TOL_REF,
                        "Time-varying T: filtered state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testUnivariateClark1989ForecastErrorCov() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();

        FilterResult frConv = KalmanEngine.filter(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        int nobs = frConv.nobs;
        int kEndog = 2;
        for (int t = 4; t < nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                for (int j = 0; j < kEndog; j++) {
                    assertEquals(frConv.forecastErrorCov(i, j, t), frUni.forecastErrorCov(i, j, t), TOL_REF,
                            "forecast error cov mismatch at t=" + t + " i=" + i + " j=" + j);
                }
            }
        }
    }

    @Test
    void testUnivariateClark1989PredictedStateCov() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();

        FilterResult frConv = KalmanEngine.filter(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        int nobs = frConv.nobs;
        int kStates = 6;
        for (int t = 0; t <= nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                for (int j = i; j < kStates; j++) {
                    assertEquals(frConv.predictedStateCov(i, j, t), frUni.predictedStateCov(i, j, t), TOL_REF,
                            "predicted state cov mismatch at t=" + t + " i=" + i + " j=" + j);
                }
            }
        }
    }

    @Test
    void testUnivariateClark1989SmoothedDisturbance() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();

        FilterResult frConv = KalmanEngine.filter(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult srConv = SmootherEngine.smooth(rep, frConv, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        FilterResult frUni = KalmanEngine.filterUnivariate(rep, knownInitialization(init), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult srUni = SmootherEngine.smooth(rep, frUni, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        int nobs = frConv.nobs;
        int kEndog = 2, kPosdef = 6;
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                assertEquals(srConv.smoothedObsDisturbance(i, t), srUni.smoothedObsDisturbance(i, t), TOL_REF,
                        "smoothed obs disturbance mismatch at t=" + t + " i=" + i);
            }
        }
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kPosdef; i++) {
                assertEquals(srConv.smoothedStateDisturbance(i, t), srUni.smoothedStateDisturbance(i, t), TOL_REF,
                        "smoothed state disturbance mismatch at t=" + t + " i=" + i);
            }
        }
    }
}
