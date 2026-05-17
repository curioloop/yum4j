package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures;
import com.curioloop.yum4j.fixtures.StatsmodelsResources;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import com.curioloop.yum4j.ssm.kalman.filter.*;
import com.curioloop.yum4j.ssm.kalman.init.*;
import com.curioloop.yum4j.ssm.kalman.model.*;
import com.curioloop.yum4j.ssm.kalman.smooth.*;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class StatsmodelsTest {

    private static final double TOL_REF = 1e-6;
    private static final double TOL_STRUCT = 1e-8;

    private static ModelFixture buildAR1(double phi, double sigma2, double[] y) {
        return KalmanModelFixtures.statsmodelsAr1(phi, sigma2, y);
    }

    private static ModelFixture buildAR3(double phi1, double phi2, double phi3,
                                            double sigma2Obs, double sigma2State,
                                            double[] y) {
        return KalmanModelFixtures.statsmodelsAr3(phi1, phi2, phi3, sigma2Obs, sigma2State, y);
    }

    private static ModelFixture buildBivariateAR1(double[] y) {
        return KalmanModelFixtures.statsmodelsBivariateAr1(y);
    }

    private static ModelFixture buildCorrelatedBivariateStandardizedModel(double[] y) {
        int nobs = y.length / 2;
        ModelFixture rep = new ModelFixture(2, 4, 2, nobs);
        double[] Z = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
        double[] H = {0.1, 0.03, 0.03, 0.2};
        double[] T = {0.9, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.6};
        double[] R = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
        double[] Q = {1.0, 0.25, 0.25, 1.5};
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

    private static double[] generateAR3(double phi1, double phi2, double phi3,
                                         double sigma2State, double sigma2Obs,
                                         int n, long seed) {
        Random rng = new Random(seed);
        double[] y = new double[n];
        double s0 = 0, s1 = 0, s2 = 0;
        for (int t = 0; t < n; t++) {
            double ns0 = phi1 * s0 + phi2 * s1 + phi3 * s2 + rng.nextGaussian() * Math.sqrt(sigma2State);
            y[t] = ns0 + rng.nextGaussian() * Math.sqrt(sigma2Obs);
            s2 = s1;
            s1 = s0;
            s0 = ns0;
        }
        return y;
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

    private static List<String[]> readExactDiffuseReferenceCsv(String fileName) {
        return StatsmodelsResources.readStatespaceResultsCsv(fileName);
    }

    private static int csvColumnIndex(String[] header, String name) {
        for (int i = 0; i < header.length; i++) {
            if (header[i].equals(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Missing csv column: " + name);
    }

    private static double csvValue(String[] row, int index) {
        String value = row[index];
        if (value == null || value.isEmpty() || value.equals("NA")) {
            return Double.NaN;
        }
        return Double.parseDouble(value);
    }

    private static double sumMatrix(double[] values, int offset, int length) {
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            sum += values[offset + i];
        }
        return sum;
    }

    private static void assertSuppressed(double[] values) {
        assertEquals(0, values.length);
    }

    private static void assertStateOnlySmootherMatches(SmootherResult full, SmootherResult stateOnly, double tol) {
        assertEquals(full.kStates, stateOnly.kStates);
        assertEquals(full.kEndog, stateOnly.kEndog);
        assertEquals(full.kPosdef, stateOnly.kPosdef);
        assertEquals(full.nobs, stateOnly.nobs);

        for (int t = 0; t < full.nobs; t++) {
            for (int i = 0; i < full.kStates; i++) {
                assertEquals(full.smoothedState(i, t), stateOnly.smoothedState(i, t), tol,
                        "smoothedState mismatch at t=" + t + " i=" + i);
            }
        }
        assertSuppressed(stateOnly.smoothedStateCov);
        assertSuppressed(stateOnly.smoothedStateAutocovariance);
        assertSuppressed(stateOnly.smoothedObsDisturbance);
        assertSuppressed(stateOnly.smoothedStateDisturbance);
        assertSuppressed(stateOnly.smoothedObsDisturbanceCov);
        assertSuppressed(stateOnly.smoothedStateDisturbanceCov);
        assertSuppressed(stateOnly.scaledSmoothedEstimator);
        assertSuppressed(stateOnly.scaledSmoothedEstimatorCov);
        assertSuppressed(stateOnly.smoothingError);
        assertSuppressed(stateOnly.innovationsTransition);
    }

    private static ModelFixture buildExactDiffuseLocalLevel(double[] y, double sigma2Y, double sigma2Mu) {
        ModelFixture rep = new ModelFixture(1, 1, 1, y.length);
        double[] Z = {1.0};
        double[] H = {sigma2Y};
        double[] T = {1.0};
        double[] R = {1.0};
        double[] Q = {sigma2Mu};
        for (int t = 0; t < y.length; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        return rep;
    }

    private static ModelFixture buildExactDiffuseLocalLinearTrend(double[] y, double sigma2Y,
                                                                    double sigma2Mu, double sigma2Beta) {
        ModelFixture rep = new ModelFixture(1, 2, 2, y.length);
        double[] Z = {1.0, 0.0};
        double[] H = {sigma2Y};
        double[] T = {1.0, 1.0, 0.0, 1.0};
        double[] R = {1.0, 0.0, 0.0, 1.0};
        double[] Q = {sigma2Mu, 0.0, 0.0, sigma2Beta};
        for (int t = 0; t < y.length; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        return rep;
    }

    @Test
    void testStationaryInit1D() {
        double phi = 0.5;
        double sigma2 = 1.0;
        double[] T = {phi};
        double[] R = {1.0};
        double[] Q = {sigma2};
        double[] c = {0.0};
        InitialState init = stationaryInitialization(T, 1, R, 1, Q, c);
        assertNotNull(init);
        assertEquals(0.0, init.initialState()[0], TOL_STRUCT);
        double expectedP = sigma2 / (1 - phi * phi);
        assertEquals(expectedP, init.initialStateCov()[0], TOL_STRUCT);
    }

    @Test
    void testStationaryInit2D() {
        double[] T = {0.5, 0.1, 0, 0.5};
        double[] R = {1, 0, 0, 1};
        double[] Q = {1, 0, 0, 1};
        double[] c = {0, 0};
        InitialState init = stationaryInitialization(T, 2, R, 2, Q, c);
        assertNotNull(init);
        assertEquals(0.0, init.initialState()[0], TOL_STRUCT);
        assertEquals(0.0, init.initialState()[1], TOL_STRUCT);
        assertTrue(init.initialStateCov()[0] > 0);
        assertTrue(init.initialStateCov()[3] > 0);
        double tol = 1e-6;
        assertEquals(init.initialStateCov()[1], init.initialStateCov()[2], tol);
    }

    @Test
    void testStationaryInitWithIntercept() {
        double phi = 0.5;
        double[] T = {phi};
        double[] R = {1.0};
        double[] Q = {1.0};
        double[] c = {2.0};
        InitialState init = stationaryInitialization(T, 1, R, 1, Q, c);
        assertNotNull(init);
        double expectedA = c[0] / (1 - phi);
        assertEquals(expectedA, init.initialState()[0], TOL_STRUCT);
    }

    @Test
    void testAR1FilterLoglikeAgainstStatsmodelsFixture() {
        KalmanStatsmodelsFixtures.AlignmentCase fixture = KalmanStatsmodelsFixtures.cases().stream()
            .filter(item -> item.id().equals("scalar_ar1_known_conventional"))
            .findFirst()
            .orElseThrow();
        ModelFixture rep = fixture.modelFixture();
        InitialState init = fixture.initialState(rep);
        FilterResult fr = KalmanEngine.filter(rep, init, fixture.filterOptions());
        assertEquals(fixture.scalar("llf"), fr.logLikelihood(), fixture.tolerance());
    }

    @Test
    void testAR3SmootherStructural() {
        double[] y = generateAR3(0.5, 0.1, -0.2, 0.5, 0.5, 20, 123);
        ModelFixture rep = buildAR3(0.5, 0.1, -0.2, 0.5, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult sr = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        int kStates = 3;
        for (int t = 0; t < 20; t++) {
            for (int i = 0; i < kStates; i++) {
                for (int j = i; j < kStates; j++) {
                    double Vij = sr.smoothedStateCov(i, j, t);
                    double Vji = sr.smoothedStateCov(j, i, t);
                    assertEquals(Vij, Vji, TOL_STRUCT, "V not symmetric at t=" + t);
                }
            }
        }

        for (int t = 1; t < 20; t++) {
            int rOff = (t - 1) * kStates;
            for (int i = 0; i < kStates; i++) {
                double expected = fr.predictedState(i, t);
                for (int j = 0; j < kStates; j++) {
                    expected += fr.predictedStateCov(i, j, t) * sr.scaledSmoothedEstimator[j + rOff];
                }
                assertEquals(expected, sr.smoothedState(i, t), TOL_STRUCT,
                        "alpha != a + P*r at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testAR3SmootherDisturbances() {
        double[] y = generateAR3(0.5, 0.1, -0.2, 0.5, 0.5, 20, 456);
        ModelFixture rep = buildAR3(0.5, 0.1, -0.2, 0.5, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult sr = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        for (int t = 0; t < 20; t++) {
            assertFalse(Double.isNaN(sr.smoothedObsDisturbance[t]));
            assertFalse(Double.isNaN(sr.smoothedStateDisturbance[t]));
        }
    }

    @Test
    void testUnivariateVsConventional() {
        double[] y = generateBivariate(30, 789);
        ModelFixture rep = buildBivariateAR1(y);
        FilterResult frConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        int kStates = 4, nobs = 30;
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                assertEquals(frConv.filteredState(i, t), frUni.filteredState(i, t), TOL_REF,
                        "filtered state mismatch at t=" + t + " i=" + i);
            }
        }
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                for (int j = 0; j < kStates; j++) {
                    assertEquals(frConv.filteredStateCov(i, j, t), frUni.filteredStateCov(i, j, t), TOL_REF,
                            "filtered cov mismatch at t=" + t + " i=" + i + " j=" + j);
                }
            }
        }
        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF);
    }

    @Test
    void testUnivariateSmootherVsConventional() {
        double[] y = generateBivariate(30, 101);
        ModelFixture rep = buildBivariateAR1(y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult srConv = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        FilterResult frUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult srUni = SmootherEngine.smooth(rep, frUni, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        SmootherResult srEngine = SmootherEngine.smooth(rep,
                InitialState.approximateDiffuse(),
                SmootherOptions.univariate());

        int kStates = 4, nobs = 30;
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kStates; i++) {
                assertEquals(srConv.smoothedState(i, t), srUni.smoothedState(i, t), TOL_REF,
                        "smoothed state mismatch at t=" + t);
                assertEquals(srUni.smoothedState(i, t), srEngine.smoothedState(i, t), TOL_REF,
                        "engine univariate smoothed state mismatch at t=" + t);
            }
        }
    }

    @Test
    void testUnivariateSmootherEngineMissingObservationMatchesManualPath() {
        double[] y = generateBivariate(18, 2028);
        ModelFixture rep = buildBivariateAR1(y);
        rep.setMissing(new boolean[]{false, true}, 4);
        rep.setMissing(new boolean[]{true, false}, 9);
        InitialState init = InitialState.approximateDiffuse();
        SmootherOptions options = SmootherOptions.univariate().only(SmootherOptions.Surface.STATE);
        FilterResult manualFilter = KalmanEngine.filterUnivariate(rep, init, options.requiredFilterOptions());
        SmootherResult full = SmootherEngine.smooth(rep, manualFilter, SmootherOptions.univariate());

        SmootherResult actual = SmootherEngine.smooth(rep,
                init,
                options);

        assertTrue(manualFilter.perObsKalmanGain);
        assertStateOnlySmootherMatches(full, actual, TOL_REF);
    }

    @Test
    void testMissingDataAll() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 10, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        rep.setMissing(new boolean[]{true}, 5);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        double expectedForecast = rep.obsIntercept[rep.obsInterceptOffset(5)] +
                rep.design[rep.designOffset(5)] * fr.predictedState(0, 5);
        assertEquals(expectedForecast, fr.forecast(0, 5), TOL_STRUCT);
        assertEquals(0.0, fr.forecastError(0, 5), TOL_STRUCT);
        assertEquals(0.0, fr.logLikelihoodObs[5], TOL_STRUCT);
        assertEquals(fr.predictedState(0, 5), fr.filteredState(0, 5), TOL_STRUCT);
    }

    @Test
    void testMissingDataPartial() {
        double[] y = generateBivariate(20, 555);
        ModelFixture rep = buildBivariateAR1(y);
        rep.setMissing(new boolean[]{true, false}, 5);
        rep.setMissing(new boolean[]{false, true}, 10);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertFalse(Double.isNaN(fr.logLikelihood()));
    }

    @Test
    void testMemoryNoGain() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frAll = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frNoGain = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
            FilterOptions.defaults().without(FilterOptions.Surface.KALMAN_GAIN));
        assertSuppressed(frNoGain.kalmanGain);
        assertEquals(frAll.logLikelihood(), frNoGain.logLikelihood(), TOL_STRUCT);
    }

    @Test
    void testMemoryNoForecast() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frNoForecast = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
                FilterOptions.defaults().without(FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE));
        assertSuppressed(frNoForecast.forecast);
        assertSuppressed(frNoForecast.forecastError);
        assertSuppressed(frNoForecast.forecastErrorCov);
    }

    @Test
    void testMemoryNoLikelihood() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frNoLL = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
                FilterOptions.defaults().without(FilterOptions.Surface.LIKELIHOOD));
        assertSuppressed(frNoLL.logLikelihoodObs);
    }

    @Test
    void testMemoryNoPredicted() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frNoPred = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
                FilterOptions.defaults().without(FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE));
        assertSuppressed(frNoPred.predictedState);
        assertSuppressed(frNoPred.predictedStateCov);
    }

    @Test
    void testMemoryNoFiltered() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frNoFilt = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
                FilterOptions.defaults().without(FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE));
        assertSuppressed(frNoFilt.filteredState);
        assertSuppressed(frNoFilt.filteredStateCov);
    }

    @Test
    void testDiffuseInitLocalLevel() {
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

        FilterResult frDiffuse = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frKnown = KalmanEngine.filter(rep, InitialState.known(
            new double[]{0}, new double[]{1e6}), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        int skip = 3;
        double llDiffuse = 0, llKnown = 0;
        for (int t = skip; t < nobs; t++) {
            llDiffuse += frDiffuse.logLikelihoodObs[t];
            llKnown += frKnown.logLikelihoodObs[t];
        }
        assertEquals(llKnown, llDiffuse, 0.1);
    }

    @Test
    void testDiffuseInitLocalLinearTrend() {
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

        FilterResult frDiffuse = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frKnown = KalmanEngine.filter(rep, InitialState.known(
            new double[]{0, 0}, new double[]{1e6, 0, 0, 1e6}), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        int skip = 5;
        double llDiffuse = 0, llKnown = 0;
        for (int t = skip; t < nobs; t++) {
            llDiffuse += frDiffuse.logLikelihoodObs[t];
            llKnown += frKnown.logLikelihoodObs[t];
        }
        assertEquals(llKnown, llDiffuse, 0.5);
    }

    @Test
    void testExactDiffuseLocalLevelAnalytic() {
        double y1 = 10.2394;
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;
        double[] y = {y1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        ModelFixture rep = buildExactDiffuseLocalLevel(y, sigma2Y, sigma2Mu);

        FilterResult fr = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(0.0, fr.predictedStateCov(0, 0, 0), TOL_STRUCT);
        assertEquals(1.0, fr.predictedDiffuseStateCov(0, 0, 0), TOL_STRUCT);

        assertEquals(y1, fr.forecastError(0, 0), TOL_STRUCT);
        assertEquals(sigma2Y, fr.forecastErrorCov(0, 0, 0), TOL_STRUCT);
        assertEquals(1.0, fr.forecastErrorDiffuseCov(0, 0, 0), TOL_STRUCT);
        assertEquals(1.0, fr.kalmanGain(0, 0, 0), TOL_STRUCT);

        assertEquals(y1, fr.predictedState(0, 1), TOL_STRUCT);
        assertEquals(sigma2Y + sigma2Mu, fr.predictedStateCov(0, 0, 1), TOL_STRUCT);
        assertEquals(0.0, fr.predictedDiffuseStateCov(0, 0, 1), TOL_STRUCT);
        assertEquals(1, fr.nobsDiffuse);
    }

    @Test
    void testExactDiffuseLocalLinearTrendAnalytic() {
        double y1 = 10.2394;
        double y2 = 4.2039;
        double y3 = 6.123123;
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;
        double sigma2Beta = 2.334;
        double[] y = {y1, y2, y3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        ModelFixture rep = buildExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);

        FilterResult fr = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(0.0, fr.predictedStateCov(0, 0, 0), TOL_STRUCT);
        assertEquals(0.0, fr.predictedStateCov(0, 1, 0), TOL_STRUCT);
        assertEquals(0.0, fr.predictedStateCov(1, 0, 0), TOL_STRUCT);
        assertEquals(0.0, fr.predictedStateCov(1, 1, 0), TOL_STRUCT);
        assertEquals(1.0, fr.predictedDiffuseStateCov(0, 0, 0), TOL_STRUCT);
        assertEquals(0.0, fr.predictedDiffuseStateCov(0, 1, 0), TOL_STRUCT);
        assertEquals(0.0, fr.predictedDiffuseStateCov(1, 0, 0), TOL_STRUCT);
        assertEquals(1.0, fr.predictedDiffuseStateCov(1, 1, 0), TOL_STRUCT);

        assertEquals(y1, fr.forecastError(0, 0), TOL_STRUCT);
        assertEquals(1.0, fr.kalmanGain(0, 0, 0), TOL_STRUCT);
        assertEquals(0.0, fr.kalmanGain(1, 0, 0), TOL_STRUCT);
        assertEquals(y1, fr.predictedState(0, 1), TOL_STRUCT);
        assertEquals(0.0, fr.predictedState(1, 1), TOL_STRUCT);

        double qMu = sigma2Mu / sigma2Y;
        double qBeta = sigma2Beta / sigma2Y;
        double[] expectedP2 = {
                sigma2Y * (1 + qMu), 0.0,
                0.0, sigma2Y * qBeta
        };
        assertEquals(expectedP2[0], fr.predictedStateCov(0, 0, 1), TOL_STRUCT);
        assertEquals(expectedP2[1], fr.predictedStateCov(0, 1, 1), TOL_STRUCT);
        assertEquals(expectedP2[2], fr.predictedStateCov(1, 0, 1), TOL_STRUCT);
        assertEquals(expectedP2[3], fr.predictedStateCov(1, 1, 1), TOL_STRUCT);
        assertEquals(1.0, fr.predictedDiffuseStateCov(0, 0, 1), TOL_STRUCT);
        assertEquals(1.0, fr.predictedDiffuseStateCov(0, 1, 1), TOL_STRUCT);
        assertEquals(1.0, fr.predictedDiffuseStateCov(1, 0, 1), TOL_STRUCT);
        assertEquals(1.0, fr.predictedDiffuseStateCov(1, 1, 1), TOL_STRUCT);

        assertEquals(2 * y2 - y1, fr.predictedState(0, 2), 1e-6);
        assertEquals(y2 - y1, fr.predictedState(1, 2), 1e-6);

        double[] expectedP3 = {
                sigma2Y * (5 + 2 * qMu + qBeta), sigma2Y * (3 + qMu + qBeta),
                sigma2Y * (3 + qMu + qBeta), sigma2Y * (2 + qMu + 2 * qBeta)
        };
        assertEquals(expectedP3[0], fr.predictedStateCov(0, 0, 2), 1e-6);
        assertEquals(expectedP3[1], fr.predictedStateCov(0, 1, 2), 1e-6);
        assertEquals(expectedP3[2], fr.predictedStateCov(1, 0, 2), 1e-6);
        assertEquals(expectedP3[3], fr.predictedStateCov(1, 1, 2), 1e-6);
        assertEquals(0.0, fr.predictedDiffuseStateCov(0, 0, 2), TOL_STRUCT);
        assertEquals(0.0, fr.predictedDiffuseStateCov(0, 1, 2), TOL_STRUCT);
        assertEquals(0.0, fr.predictedDiffuseStateCov(1, 0, 2), TOL_STRUCT);
        assertEquals(0.0, fr.predictedDiffuseStateCov(1, 1, 2), TOL_STRUCT);
        assertEquals(2, fr.nobsDiffuse);
    }

    @Test
    void testExactDiffuseLocalLinearTrendAnalyticWithMissingObservation() {
        double y1 = 10.2394;
        double y3 = 6.123123;
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;
        double sigma2Beta = 2.334;
        double[] y = {y1, 0.0, y3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        ModelFixture rep = buildExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        rep.setMissing(new boolean[]{true}, 1);

        FilterResult fr = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        double qMu = sigma2Mu / sigma2Y;
        double qBeta = sigma2Beta / sigma2Y;
        double[] expectedA4 = {1.5 * y3 - 0.5 * y1, 0.5 * y3 - 0.5 * y1};
        assertEquals(expectedA4[0], fr.predictedState(0, 3), 1e-6);
        assertEquals(expectedA4[1], fr.predictedState(1, 3), 1e-6);

        double[] expectedP4 = {
                sigma2Y * (2.5 + 1.5 * qMu + 1.25 * qBeta),
                sigma2Y * (1 + 0.5 * qMu + 1.25 * qBeta),
                sigma2Y * (1 + 0.5 * qMu + 1.25 * qBeta),
                sigma2Y * (0.5 + 0.5 * qMu + 2.25 * qBeta)
        };
        assertEquals(expectedP4[0], fr.predictedStateCov(0, 0, 3), 1e-6);
        assertEquals(expectedP4[1], fr.predictedStateCov(0, 1, 3), 1e-6);
        assertEquals(expectedP4[2], fr.predictedStateCov(1, 0, 3), 1e-6);
        assertEquals(expectedP4[3], fr.predictedStateCov(1, 1, 3), 1e-6);
        assertEquals(3, fr.nobsDiffuse);
    }

    @Test
    void testInitKnown() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 10, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        double[] a0 = {1.0};
        double[] P0 = {2.0};
        FilterResult fr = KalmanEngine.filter(rep, InitialState.known(a0, P0), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertEquals(1.0, fr.predictedState(0, 0), TOL_STRUCT);
        assertEquals(2.0, fr.predictedStateCov(0, 0, 0), TOL_STRUCT);
    }

    @Test
    void testInitDiffuse() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 10, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertNotNull(fr.predictedDiffuseStateCov);
        assertEquals(1.0, fr.predictedDiffuseStateCov(0, 0, 0), TOL_STRUCT);
    }

    @Test
    void testInitApproximateDiffuse() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 10, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertEquals(1e6, fr.predictedStateCov(0, 0, 0), TOL_STRUCT);
    }

    @Test
    void testInitStationary() {
        double phi = 0.5;
        double sigma2 = 1.0;
        double[] y = generateAR1(phi, sigma2, 10, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep,
            InitialState.stationary(rep), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        double expectedP = sigma2 / (1 - phi * phi);
        assertEquals(expectedP, fr.predictedStateCov(0, 0, 0), TOL_STRUCT);
    }

    @Test
    void testInitStationaryRejectsUnitRoot() {
        ModelFixture rep = buildAR1(1.0, 0.5, new double[]{0.0, 0.0, 0.0, 0.0});

        assertThrows(IllegalArgumentException.class, () -> InitialState.stationary(rep));
    }

    @Test
    void testConvergenceDetection() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 200, 999);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertTrue(fr.converged);
        assertTrue(fr.periodConverged < 200);
        assertTrue(fr.periodConverged > 1);
    }

    @Test
    void testSmootherCovPSD() {
        double[] y = generateAR3(0.5, 0.1, -0.2, 0.5, 0.5, 20, 777);
        ModelFixture rep = buildAR3(0.5, 0.1, -0.2, 0.5, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult sr = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        int kStates = 3;

        for (int t = 0; t < 20; t++) {
            for (int i = 0; i < kStates; i++) {
                assertTrue(sr.smoothedStateCov(i, i, t) >= -TOL_STRUCT,
                        "V diagonal negative at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testSmootherStateCovFormula() {
        double[] y = generateAR3(0.5, 0.1, -0.2, 0.5, 0.5, 20, 321);
        ModelFixture rep = buildAR3(0.5, 0.1, -0.2, 0.5, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult sr = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        int kStates = 3, kStates2 = kStates * kStates;
        for (int t = 1; t < 20; t++) {
            int nOff = (t - 1) * kStates2;
            double[] expectedV = new double[kStates2];
            for (int i = 0; i < kStates; i++) {
                for (int j = 0; j < kStates; j++) {
                    double pnp = 0;
                    for (int k = 0; k < kStates; k++) {
                        for (int l = 0; l < kStates; l++) {
                            pnp += fr.predictedStateCov(i, k, t) *
                                    sr.scaledSmoothedEstimatorCov[k * kStates + l + nOff] *
                                    fr.predictedStateCov(l, j, t);
                        }
                    }
                    expectedV[i * kStates + j] = fr.predictedStateCov(i, j, t) - pnp;
                }
            }
            for (int i = 0; i < kStates2; i++) {
                assertEquals(expectedV[i], sr.smoothedStateCov[i + t * kStates2], TOL_STRUCT,
                        "V formula mismatch at t=" + t);
            }
        }
    }

    @Test
    void testKalmanGainFinite() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        for (int t = 0; t < 20; t++) {
            assertFalse(Double.isNaN(fr.kalmanGain(0, 0, t)));
            assertFalse(Double.isInfinite(fr.kalmanGain(0, 0, t)));
        }
    }

    @Test
    void testForecastErrorCovSymmetric() {
        double[] y = generateBivariate(20, 42);
        ModelFixture rep = buildBivariateAR1(y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        for (int t = 0; t < 20; t++) {
            assertEquals(fr.forecastErrorCov(0, 1, t), fr.forecastErrorCov(1, 0, t), TOL_STRUCT);
        }
    }

    @Test
    void testMultivariateStandardizedForecastErrorMatchesStatsmodelsReference() {
        List<String[]> rows = readExactDiffuseReferenceCsv("results_standardized_forecast_error_multivariate.csv");
        String[] header = rows.get(0);
        int y1Col = csvColumnIndex(header, "y1");
        int y2Col = csvColumnIndex(header, "y2");
        int standardized1Col = csvColumnIndex(header, "standardized_forecast_error1");
        int standardized2Col = csvColumnIndex(header, "standardized_forecast_error2");
        int error1Col = csvColumnIndex(header, "forecast_error1");
        int error2Col = csvColumnIndex(header, "forecast_error2");
        int cov11Col = csvColumnIndex(header, "forecast_error_cov11");
        int cov12Col = csvColumnIndex(header, "forecast_error_cov12");
        int cov21Col = csvColumnIndex(header, "forecast_error_cov21");
        int cov22Col = csvColumnIndex(header, "forecast_error_cov22");
        double[] y = new double[(rows.size() - 1) * 2];
        for (int row = 1; row < rows.size(); row++) {
            String[] values = rows.get(row);
            int offset = (row - 1) * 2;
            y[offset] = csvValue(values, y1Col);
            y[offset + 1] = csvValue(values, y2Col);
        }

        FilterResult result = KalmanEngine.filter(buildCorrelatedBivariateStandardizedModel(y),
            InitialState.approximateDiffuse(), FilterOptions.defaults());
        double[] standardized = result.standardizedForecastError();

        assertEquals(y.length, standardized.length);
        for (int row = 1; row < rows.size(); row++) {
            String[] values = rows.get(row);
            int t = row - 1;
            assertEquals(csvValue(values, error1Col), result.forecastError(0, t), TOL_REF);
            assertEquals(csvValue(values, error2Col), result.forecastError(1, t), TOL_REF);
            assertEquals(csvValue(values, cov11Col), result.forecastErrorCov(0, 0, t), TOL_REF);
            assertEquals(csvValue(values, cov12Col), result.forecastErrorCov(0, 1, t), TOL_REF);
            assertEquals(csvValue(values, cov21Col), result.forecastErrorCov(1, 0, t), TOL_REF);
            assertEquals(csvValue(values, cov22Col), result.forecastErrorCov(1, 1, t), TOL_REF);
            assertEquals(csvValue(values, standardized1Col), standardized[t * 2], TOL_REF);
            assertEquals(csvValue(values, standardized2Col), standardized[t * 2 + 1], TOL_REF);
        }
    }

    @Test
    void testPredictedStateCovSymmetric() {
        double[] y = generateBivariate(20, 42);
        ModelFixture rep = buildBivariateAR1(y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        for (int t = 0; t <= 20; t++) {
            for (int i = 0; i < 4; i++) {
                for (int j = i + 1; j < 4; j++) {
                    assertEquals(fr.predictedStateCov(i, j, t), fr.predictedStateCov(j, i, t), TOL_STRUCT);
                }
            }
        }
    }

    @Test
    void testClassicalSmootherOptionsWithoutOutputs() {
        double phi = 0.7;
        double[] y = generateAR1(phi, 1.0, 25, 2024);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        SmootherResult full = SmootherEngine.smooth(rep, fr, SmootherOptions.classical());
        SmootherResult stateOnly = SmootherEngine.smooth(rep, fr,
            SmootherOptions.classical().only(SmootherOptions.Surface.STATE));

        assertStateOnlySmootherMatches(full, stateOnly, TOL_STRUCT);
    }

    @Test
    void testConventionalSmootherOptionsWithoutOutputs() {
        double phi = 0.7;
        double[] y = generateAR1(phi, 1.0, 25, 2025);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        SmootherResult full = SmootherEngine.smooth(rep, fr, SmootherOptions.conventional());
        SmootherResult stateOnly = SmootherEngine.smooth(rep, fr,
            SmootherOptions.conventional().only(SmootherOptions.Surface.STATE));

        assertStateOnlySmootherMatches(full, stateOnly, TOL_STRUCT);
    }

    @Test
    void testAlternativeSmootherOptionsWithoutOutputs() {
        double[] y = generateAR3(0.5, 0.1, -0.2, 0.5, 0.5, 24, 2026);
        ModelFixture rep = buildAR3(0.5, 0.1, -0.2, 0.5, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        SmootherResult full = SmootherEngine.smooth(rep, fr, SmootherOptions.alternative());
        SmootherResult stateOnly = SmootherEngine.smooth(rep, fr,
            SmootherOptions.alternative().only(SmootherOptions.Surface.STATE));

        assertStateOnlySmootherMatches(full, stateOnly, TOL_REF);
    }

    @Test
    void testUnivariateSmootherOptionsWithoutOutputs() {
        double[] y = generateBivariate(18, 2027);
        ModelFixture rep = buildBivariateAR1(y);
        FilterResult fr = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertTrue(fr.perObsKalmanGain);

        SmootherResult full = SmootherEngine.smooth(rep, fr, SmootherOptions.conventional());
        SmootherResult stateOnly = SmootherEngine.smooth(rep, fr,
            SmootherOptions.conventional().only(SmootherOptions.Surface.STATE));

        assertStateOnlySmootherMatches(full, stateOnly, TOL_REF);
    }

    @Test
    void testDiffuseSmootherOptionsWithoutOutputs() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        double sigma2Y = 1.0;
        double sigma2Mu = 0.5;
        double sigma2Beta = 0.2;
        ModelFixture rep = buildExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertTrue(fr.nobsDiffuse > 0);

        SmootherResult full = SmootherEngine.smooth(rep, fr, SmootherOptions.conventional());
        SmootherResult stateOnly = SmootherEngine.smooth(rep, fr,
            SmootherOptions.conventional().only(SmootherOptions.Surface.STATE));

        assertStateOnlySmootherMatches(full, stateOnly, TOL_REF);
        assertSuppressed(stateOnly.scaledSmoothedDiffuseEstimator);
        assertSuppressed(stateOnly.scaledSmoothedDiffuse1EstimatorCov);
        assertSuppressed(stateOnly.scaledSmoothedDiffuse2EstimatorCov);
    }

            @Test
            void testSmootherMethodVariantsAgree() {
            double[] y = generateAR3(0.5, 0.1, -0.2, 0.5, 0.5, 24, 2025);
            ModelFixture rep = buildAR3(0.5, 0.1, -0.2, 0.5, 0.5, y);
            FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

            SmootherResult conventional = SmootherEngine.smooth(rep, fr, SmootherOptions.conventional());
            SmootherResult classical = SmootherEngine.smooth(rep, fr, SmootherOptions.classical());
            SmootherResult alternative = SmootherEngine.smooth(rep, fr, SmootherOptions.alternative());

            for (int t = 0; t < fr.nobs; t++) {
                for (int i = 0; i < rep.kStates; i++) {
                assertEquals(conventional.smoothedState(i, t), classical.smoothedState(i, t), TOL_REF,
                    "classical smoothedState mismatch at t=" + t + " i=" + i);
                assertEquals(conventional.smoothedState(i, t), alternative.smoothedState(i, t), TOL_REF,
                    "alternative smoothedState mismatch at t=" + t + " i=" + i);
                for (int j = 0; j < rep.kStates; j++) {
                    assertEquals(conventional.smoothedStateCov(i, j, t), classical.smoothedStateCov(i, j, t), TOL_REF,
                        "classical smoothedStateCov mismatch at t=" + t + " i=" + i + " j=" + j);
                    assertEquals(conventional.smoothedStateCov(i, j, t), alternative.smoothedStateCov(i, j, t), TOL_REF,
                        "alternative smoothedStateCov mismatch at t=" + t + " i=" + i + " j=" + j);
                }
                }
                for (int i = 0; i < rep.kEndog; i++) {
                assertEquals(conventional.smoothedObsDisturbance(i, t), classical.smoothedObsDisturbance(i, t), TOL_REF,
                    "classical smoothedObsDisturbance mismatch at t=" + t + " i=" + i);
                assertEquals(conventional.smoothedObsDisturbance(i, t), alternative.smoothedObsDisturbance(i, t), TOL_REF,
                    "alternative smoothedObsDisturbance mismatch at t=" + t + " i=" + i);
                for (int j = 0; j < rep.kEndog; j++) {
                    assertEquals(conventional.smoothedObsDisturbanceCov(i, j, t), classical.smoothedObsDisturbanceCov(i, j, t), TOL_REF,
                        "classical smoothedObsDisturbanceCov mismatch at t=" + t + " i=" + i + " j=" + j);
                    assertEquals(conventional.smoothedObsDisturbanceCov(i, j, t), alternative.smoothedObsDisturbanceCov(i, j, t), TOL_REF,
                        "alternative smoothedObsDisturbanceCov mismatch at t=" + t + " i=" + i + " j=" + j);
                }
                }
                for (int i = 0; i < rep.kPosdef; i++) {
                assertEquals(conventional.smoothedStateDisturbance(i, t), classical.smoothedStateDisturbance(i, t), TOL_REF,
                    "classical smoothedStateDisturbance mismatch at t=" + t + " i=" + i);
                assertEquals(conventional.smoothedStateDisturbance(i, t), alternative.smoothedStateDisturbance(i, t), TOL_REF,
                    "alternative smoothedStateDisturbance mismatch at t=" + t + " i=" + i);
                for (int j = 0; j < rep.kPosdef; j++) {
                    assertEquals(conventional.smoothedStateDisturbanceCov(i, j, t), classical.smoothedStateDisturbanceCov(i, j, t), TOL_REF,
                        "classical smoothedStateDisturbanceCov mismatch at t=" + t + " i=" + i + " j=" + j);
                    assertEquals(conventional.smoothedStateDisturbanceCov(i, j, t), alternative.smoothedStateDisturbanceCov(i, j, t), TOL_REF,
                        "alternative smoothedStateDisturbanceCov mismatch at t=" + t + " i=" + i + " j=" + j);
                }
                }
            }
            }

            @Test
            void testExactDiffuseLocalLevelSmootherAgainstKFAS() {
            double y1 = 10.2394;
            double sigma2Y = 1.993;
            double sigma2Mu = 8.253;
            double[] y = {y1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
            ModelFixture rep = buildExactDiffuseLocalLevel(y, sigma2Y, sigma2Mu);

            FilterResult fr = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
            SmootherResult sr = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
            List<String[]> csv = readExactDiffuseReferenceCsv("results_exact_initial_local_level_R.csv");
            String[] header = csv.get(0);
            int alphaCol = csvColumnIndex(header, "alphahat_1");
            int sumVCol = csvColumnIndex(header, "sumV");
            int epsCol = csvColumnIndex(header, "epshat_1");
            int vepsCol = csvColumnIndex(header, "Veps_1");
            for (int t = 0; t < fr.nobs; t++) {
                String[] row = csv.get(t + 1);
                assertEquals(csvValue(row, alphaCol), sr.smoothedState(0, t), 1e-8,
                    "exact diffuse local level smoothed state mismatch at t=" + t);
                assertEquals(csvValue(row, sumVCol), sr.smoothedStateCov(0, 0, t), 1e-8,
                    "exact diffuse local level smoothed cov mismatch at t=" + t);
                assertEquals(csvValue(row, epsCol), sr.smoothedObsDisturbance(0, t), 1e-8,
                    "exact diffuse local level obs disturbance mismatch at t=" + t);
                assertEquals(csvValue(row, vepsCol), sr.smoothedObsDisturbanceCov(0, 0, t), 1e-8,
                    "exact diffuse local level obs disturbance cov mismatch at t=" + t);
            }
            }

            @Test
            void testExactDiffuseLocalLinearTrendSmootherAgainstKFAS() {
            double y1 = 10.2394;
            double y2 = 4.2039;
            double y3 = 6.123123;
            double sigma2Y = 1.993;
            double sigma2Mu = 8.253;
            double sigma2Beta = 2.334;
            double[] y = {y1, y2, y3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
            ModelFixture rep = buildExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);

            FilterResult fr = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
            SmootherResult sr = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
            List<String[]> csv = readExactDiffuseReferenceCsv("results_exact_initial_local_linear_trend_R.csv");
            String[] header = csv.get(0);
            int alpha1Col = csvColumnIndex(header, "alphahat_1");
            int alpha2Col = csvColumnIndex(header, "alphahat_2");
            int sumVCol = csvColumnIndex(header, "sumV");
            int epsCol = csvColumnIndex(header, "epshat_1");
            int vepsCol = csvColumnIndex(header, "Veps_1");
            for (int t = 0; t < fr.nobs; t++) {
                String[] row = csv.get(t + 1);
                assertEquals(csvValue(row, alpha1Col), sr.smoothedState(0, t), 1e-7,
                    "exact diffuse lltrend smoothed state[0] mismatch at t=" + t);
                assertEquals(csvValue(row, alpha2Col), sr.smoothedState(1, t), 1e-7,
                    "exact diffuse lltrend smoothed state[1] mismatch at t=" + t);
                assertEquals(csvValue(row, sumVCol), sumMatrix(sr.smoothedStateCov, t * 4, 4), 1e-7,
                    "exact diffuse lltrend smoothed cov sum mismatch at t=" + t);
                assertEquals(csvValue(row, epsCol), sr.smoothedObsDisturbance(0, t), 1e-7,
                    "exact diffuse lltrend obs disturbance mismatch at t=" + t);
                assertEquals(csvValue(row, vepsCol), sr.smoothedObsDisturbanceCov(0, 0, t), 1e-7,
                    "exact diffuse lltrend obs disturbance cov mismatch at t=" + t);
            }
            }

            @Test
            void testExactDiffuseLocalLinearTrendMissingSmootherAgainstKFAS() {
            double y1 = 10.2394;
            double y3 = 6.123123;
            double sigma2Y = 1.993;
            double sigma2Mu = 8.253;
            double sigma2Beta = 2.334;
            double[] y = {y1, 0.0, y3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
            ModelFixture rep = buildExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
            rep.setMissing(new boolean[]{true}, 1);

            FilterResult fr = KalmanEngine.filter(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
            SmootherResult sr = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
            List<String[]> csv = readExactDiffuseReferenceCsv("results_exact_initial_local_linear_trend_missing_R.csv");
            String[] header = csv.get(0);
            int alpha1Col = csvColumnIndex(header, "alphahat_1");
            int alpha2Col = csvColumnIndex(header, "alphahat_2");
            int sumVCol = csvColumnIndex(header, "sumV");
            int epsCol = csvColumnIndex(header, "epshat_1");
            int vepsCol = csvColumnIndex(header, "Veps_1");
            for (int t = 0; t < fr.nobs; t++) {
                String[] row = csv.get(t + 1);
                double alpha1 = csvValue(row, alpha1Col);
                double alpha2 = csvValue(row, alpha2Col);
                double eps = csvValue(row, epsCol);

                if (!Double.isNaN(alpha1)) {
                assertEquals(alpha1, sr.smoothedState(0, t), 1e-7,
                    "exact diffuse missing lltrend smoothed state[0] mismatch at t=" + t);
                }
                if (!Double.isNaN(alpha2)) {
                assertEquals(alpha2, sr.smoothedState(1, t), 1e-7,
                    "exact diffuse missing lltrend smoothed state[1] mismatch at t=" + t);
                }
                if (!Double.isNaN(csvValue(row, sumVCol))) {
                assertEquals(csvValue(row, sumVCol), sumMatrix(sr.smoothedStateCov, t * 4, 4), 1e-7,
                    "exact diffuse missing lltrend smoothed cov sum mismatch at t=" + t);
                }
                if (!Double.isNaN(eps)) {
                assertEquals(eps, sr.smoothedObsDisturbance(0, t), 1e-7,
                    "exact diffuse missing lltrend obs disturbance mismatch at t=" + t);
                }
                if (!Double.isNaN(csvValue(row, vepsCol))) {
                assertEquals(csvValue(row, vepsCol), sr.smoothedObsDisturbanceCov(0, 0, t), 1e-7,
                    "exact diffuse missing lltrend obs disturbance cov mismatch at t=" + t);
                }
            }
            }

    @Test
    void testFilteredStateCovSymmetric() {
        double[] y = generateBivariate(20, 42);
        ModelFixture rep = buildBivariateAR1(y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        for (int t = 0; t < 20; t++) {
            for (int i = 0; i < 4; i++) {
                for (int j = i + 1; j < 4; j++) {
                    assertEquals(fr.filteredStateCov(i, j, t), fr.filteredStateCov(j, i, t), TOL_STRUCT);
                }
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
    void testUnivariateMissingAll() {
        double[] y = generateBivariate(20, 333);
        ModelFixture rep = buildBivariateAR1(y);
        rep.setMissing(new boolean[]{true, true}, 5);
        FilterResult frConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult frUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF);
    }

    @Test
    void testDiffuseVsApproximateConvergence() {
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

    @Test
    void testSmootherMissingData() {
        double[] y = generateAR3(0.5, 0.1, -0.2, 0.5, 0.5, 20, 444);
        ModelFixture rep = buildAR3(0.5, 0.1, -0.2, 0.5, 0.5, y);
        rep.setMissing(new boolean[]{true}, 5);
        rep.setMissing(new boolean[]{true}, 10);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult sr = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        for (int t = 0; t < 20; t++) {
            for (int i = 0; i < 3; i++) {
                assertFalse(Double.isNaN(sr.smoothedState(i, t)));
            }
        }
    }

    @Test
    void testMemoryConserve() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult fr = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
            FilterOptions.compact());
        assertSuppressed(fr.forecast);
        assertSuppressed(fr.forecastError);
        assertSuppressed(fr.forecastErrorCov);
        assertSuppressed(fr.filteredState);
        assertSuppressed(fr.filteredStateCov);
        assertSuppressed(fr.predictedState);
        assertSuppressed(fr.predictedStateCov);
        assertSuppressed(fr.kalmanGain);
        assertSuppressed(fr.logLikelihoodObs);
    }
}
