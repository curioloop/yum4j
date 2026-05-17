package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.init.*;
import com.curioloop.yum4j.ssm.kalman.model.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class UnivariateFilterTest {

    @Test
    void testNearSingularObservation() {
        int kEndog = 1, kStates = 1, kPosdef = 1, nobs = 3;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0};
        double[] d = {0.0};
        double[] H = {1e-15};
        double[] T = {0.9};
        double[] c = {0.0};
        double[] R = {1.0};
        double[] Q = {1.0};
        double[] y = {1.0, 0.5, -0.3};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        FilterResult r = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        for (int t = 0; t < nobs; t++) {
            assertFalse(Double.isNaN(r.logLikelihoodObs[t]), "logLik NaN at t=" + t);
            assertFalse(Double.isInfinite(r.logLikelihoodObs[t]), "logLik Inf at t=" + t);
            for (int i = 0; i < kStates; i++) {
                assertFalse(Double.isNaN(r.predictedState(i, t)), "predictedState NaN at t=" + t);
            }
        }
    }

    @Test
    void testUnivariateForecastPerObs() {
        int kEndog = 2, kStates = 2, kPosdef = 1, nobs = 3;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1, 0, 0, 1};
        double[] d = {0, 0};
        double[] H = {1.0, 0, 0, 1.0};
        double[] T = {0.9, 0.1, 0.0, 0.9};
        double[] c = {0, 0};
        double[] R = {1, 0};
        double[] Q = {1.0};
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t * 2], y[t * 2 + 1]}, t);
        }
        FilterResult r = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertTrue(r.perObsKalmanGain);
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                assertFalse(Double.isNaN(r.forecast(i, t)), "forecast NaN at t=" + t + " i=" + i);
                assertFalse(Double.isNaN(r.forecastError(i, t)), "forecastError NaN at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testUnivariateKalmanGainNoT() {
        int kEndog = 1, kStates = 1, kPosdef = 1, nobs = 3;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0};
        double[] H = {1.0};
        double[] T = {0.9};
        double[] R = {1.0};
        double[] Q = {1.0};
        double[] y = {1.0, 0.5, -0.3};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        FilterResult r = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        double P_pred = 1e6;
        double Fi = P_pred * 1.0 + 1.0;
        double expectedK = P_pred / Fi;
        assertEquals(expectedK, r.kalmanGain(0, 0, 0), 1e-4, "K should be P*Z'/F, not T*P*Z'/F");
    }

    @Test
    void testUnivariateDiffuseForecast() {
        int kEndog = 1, kStates = 1, kPosdef = 1, nobs = 5;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0};
        double[] H = {1.0};
        double[] T = {0.9};
        double[] R = {1.0};
        double[] Q = {1.0};
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        FilterResult r = KalmanEngine.filterUnivariate(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        boolean hasNonZeroForecast = false;
        for (int t = 0; t < nobs; t++) {
            assertFalse(Double.isNaN(r.forecast(0, t)), "forecast NaN at t=" + t);
            if (Math.abs(r.forecast(0, t)) > 1e-15) hasNonZeroForecast = true;
        }
        assertTrue(hasNonZeroForecast, "at least some forecast should be non-zero");
    }

    @Test
    void testUnivariateFilterOptionsSuppressesStoredOutputs() {
        int kEndog = 2, kStates = 2, kPosdef = 1, nobs = 4;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1, 0, 0, 1};
        double[] d = {0, 0};
        double[] H = {1.0, 0, 0, 1.0};
        double[] T = {0.9, 0.1, 0.0, 0.9};
        double[] c = {0, 0};
        double[] R = {1, 0};
        double[] Q = {1.0};
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t * 2], y[t * 2 + 1]}, t);
        }

        FilterResult result = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(),
                FilterOptions.defaults().without(
                        FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE,
                        FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE,
                        FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
                        FilterOptions.Surface.KALMAN_GAIN,
                        FilterOptions.Surface.LIKELIHOOD));

        assertEquals(0, result.forecast.length);
        assertEquals(0, result.forecastError.length);
        assertEquals(0, result.forecastErrorCov.length);
        assertEquals(0, result.predictedState.length);
        assertEquals(0, result.predictedStateCov.length);
        assertEquals(0, result.filteredState.length);
        assertEquals(0, result.filteredStateCov.length);
        assertEquals(0, result.kalmanGain.length);
        assertEquals(0, result.logLikelihoodObs.length);
        assertTrue(result.perObsKalmanGain);
    }

    @Test
    void testUnivariateDiffuseFilterOptionsSuppressesStoredOutputs() {
        int kEndog = 1, kStates = 1, kPosdef = 1, nobs = 5;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0};
        double[] H = {1.0};
        double[] T = {0.9};
        double[] R = {1.0};
        double[] Q = {1.0};
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }

        FilterResult result = KalmanEngine.filterUnivariate(rep, InitialState.diffuse(),
                FilterOptions.defaults().without(
                        FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE,
                        FilterOptions.Surface.PREDICTED_STATE, FilterOptions.Surface.PREDICTED_STATE_COVARIANCE, FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE,
                        FilterOptions.Surface.FILTERED_STATE, FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
                        FilterOptions.Surface.KALMAN_GAIN,
                        FilterOptions.Surface.LIKELIHOOD));

        assertEquals(0, result.forecast.length);
        assertEquals(0, result.forecastError.length);
        assertEquals(0, result.forecastErrorCov.length);
        assertEquals(0, result.forecastErrorDiffuseCov.length);
        assertEquals(0, result.predictedState.length);
        assertEquals(0, result.predictedStateCov.length);
        assertEquals(0, result.predictedDiffuseStateCov.length);
        assertEquals(0, result.filteredState.length);
        assertEquals(0, result.filteredStateCov.length);
        assertEquals(0, result.kalmanGain.length);
        assertEquals(0, result.logLikelihoodObs.length);
        assertTrue(result.nobsDiffuse > 0);
        assertTrue(result.perObsKalmanGain);
    }

    @Test
    void testUnivariatePoolUsesUnifiedScratchBacking() {
        int kEndog = 2, kStates = 3, kPosdef = 1, nobs = 4;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0, 0.0, 0.0,
                0.0, 1.0, 0.0};
        double[] d = {0.0, 0.0};
        double[] H = {1.0, 0.0,
                0.0, 1.0};
        double[] T = {0.9, 0.1, 0.0,
                0.0, 0.8, 0.2,
                0.0, 0.0, 0.7};
        double[] c = {0.0, 0.0, 0.0};
        double[] R = {1.0, 0.0, 0.0};
        double[] Q = {1.0};
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t * 2], y[t * 2 + 1]}, t);
        }

        UnivariateFilter.Pool pool = new UnivariateFilter.Pool();
        FilterResult result = KalmanEngine.filterUnivariate(rep, InitialState.diffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        long expectedScratchDoubles = expectedDiffuseUnivariateScratchDoubleCount(kStates, kPosdef);
        long expectedResultDoubles = expectedUnivariateResultDoubleCount(rep, FilterOptions.defaults())
            + expectedUnivariateDiffuseResultDoubleCount(rep, FilterOptions.defaults());

        assertTrue(result.nobsDiffuse > 0);
        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedScratchDoubles, (long) pool.scratchBacking.length);
        assertEquals(expectedResultDoubles, pool.retainedResultDoubleCount());
        assertEquals((expectedScratchDoubles + expectedResultDoubles) * Double.BYTES, pool.retainedTotalByteCount());
    }

    @Test
    void testUnivariatePoolReusesResultBackingAcrossBorrowedRuns() {
        int kEndog = 1, kStates = 1, kPosdef = 1, nobs = 4;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0};
        double[] H = {1.0};
        double[] T = {0.8};
        double[] R = {1.0};
        double[] Q = {0.5};
        double[] y = {1.0, 0.5, -0.3, 0.2};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        UnivariateFilter.Pool pool = new UnivariateFilter.Pool();

        FilterResult first = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), pool, FilterOptions.defaults());
        double[] predictedState = first.predictedState;
        double[] forecast = first.forecast;
        long retainedResultDoubles = pool.retainedResultDoubleCount();
        FilterResult second = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), pool, FilterOptions.defaults());

        assertSame(first, second);
        assertSame(predictedState, second.predictedState);
        assertSame(forecast, second.forecast);
        assertEquals(retainedResultDoubles, pool.retainedResultDoubleCount());
    }

    @Test
    void testNonDiagonalObservationTransformReusesWorkspaceWithoutStateSystemCopies() {
        int kEndog = 3, kStates = 2, kPosdef = 1, nobs = 4;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {
            1.0, 0.0,
            0.5, 1.0,
            0.0, 1.0
        };
        double[] d = {0.1, -0.2, 0.3};
        double[] H = {
            1.0, 0.2, 0.1,
            0.2, 1.2, 0.05,
            0.1, 0.05, 0.9
        };
        double[] T = {
            0.7, 0.1,
            0.0, 0.5
        };
        double[] c = {0.0, 0.0};
        double[] R = {1.0, 0.25};
        double[] Q = {0.3};
        double[] y = {
            1.0, 0.4, -0.3,
            0.8, 0.2, -0.1,
            0.6, -0.4, 0.5,
            0.3, 0.1, 0.7
        };
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t * kEndog], y[t * kEndog + 1], y[t * kEndog + 2]}, t);
        }
        UnivariateFilter.Pool pool = new UnivariateFilter.Pool();
        FilterResult expected = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), FilterOptions.defaults());

        FilterResult first = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), pool, FilterOptions.defaults());
        double[] resultBacking = pool.resultBacking;
        long retainedScratchDoubles = pool.retainedScratchDoubleCount();
        FilterResult second = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), pool, FilterOptions.defaults());

        assertSame(first, second);
        assertSame(resultBacking, pool.resultBacking);
        assertEquals(retainedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedRegularUnivariateScratchDoubleCount(kStates, kPosdef),
            pool.retainedScratchDoubleCount());
        assertEquals(0L, pool.retainedObservationTransformDoubleCount());
        assertEquals(expected.logLikelihood(), second.logLikelihood(), 1e-9);
        assertEquals(expected.filteredState(0, nobs - 1), second.filteredState(0, nobs - 1), 1e-9);
        assertEquals(expected.filteredState(1, nobs - 1), second.filteredState(1, nobs - 1), 1e-9);
    }

    @Test
    void testUnivariateStationaryInitAppendsWorkspaceToSharedScratchBacking() {
        int kEndog = 1, kStates = 1, kPosdef = 1, nobs = 4;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0};
        double[] H = {1.0};
        double[] T = {0.8};
        double[] R = {1.0};
        double[] Q = {0.5};
        double[] y = {1.0, 0.5, -0.3, 0.2};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }

        UnivariateFilter.Pool pool = new UnivariateFilter.Pool();
        KalmanEngine.filterUnivariate(rep, InitialState.stationary(rep.stateCount()), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        long expectedWorkspaceDoubles = InitialState.requiredStationaryBackingLength(rep)
            + InitialState.requiredStationaryPivotLength(rep) * ((long) Integer.BYTES) / Double.BYTES;
        assertEquals(expectedRegularUnivariateScratchDoubleCount(kStates, kPosdef) + expectedWorkspaceDoubles,
            pool.retainedScratchDoubleCount());
        assertEquals(expectedRegularUnivariateScratchDoubleCount(kStates, kPosdef), (long) pool.scratchBacking.length);
    }

    @Test
    void testUnivariateReservePreservesStationaryWorkspaceUntilExplicitRelease() {
        int kEndog = 1, kStates = 1, kPosdef = 1, nobs = 4;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0};
        double[] H = {1.0};
        double[] T = {0.8};
        double[] R = {1.0};
        double[] Q = {0.5};
        double[] y = {1.0, 0.5, -0.3, 0.2};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }

        UnivariateFilter.Pool pool = new UnivariateFilter.Pool();
        long expectedScratchDoubles = expectedRegularUnivariateScratchDoubleCount(kStates, kPosdef);

        KalmanEngine.filterUnivariate(rep, InitialState.stationary(rep.stateCount()), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertTrue(pool.retainedScratchDoubleCount() > expectedScratchDoubles);

        pool.reserve(rep);

        assertTrue(pool.retainedScratchDoubleCount() > expectedScratchDoubles);

        pool.releaseRetainedScratch();
        pool.reserve(rep);

        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedScratchDoubles, (long) pool.scratchBacking.length);
    }

    @Test
    void testUnivariatePoolReleaseRetainedScratchClearsOwnedBanks() {
        int kEndog = 1, kStates = 1, kPosdef = 1, nobs = 4;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1.0};
        double[] H = {1.0};
        double[] T = {1.0};
        double[] R = {1.0};
        double[] Q = {0.5};
        double[] y = {1.0, 0.5, -0.3, 0.2};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }

        UnivariateFilter.Pool pool = new UnivariateFilter.Pool();
        KalmanEngine.filterUnivariate(rep, InitialState.diffuse(), pool, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertTrue(pool.retainedScratchDoubleCount() > 0);

        pool.releaseRetainedScratch();

        assertEquals(0L, pool.retainedScratchDoubleCount());
        assertNull(pool.scratchBacking);
    }

    private static long expectedRegularUnivariateScratchDoubleCount(int kStates, int kPosdef) {
        return kStates + (long) kStates * kStates
            + Math.max(kStates, Math.max((long) kStates * kStates, (long) kStates * kPosdef));
    }

    private static long expectedDiffuseUnivariateScratchDoubleCount(int kStates, int kPosdef) {
        return kStates + 2L * kStates * kStates
            + Math.max(4L * kStates, Math.max((long) kStates * kStates, (long) kStates * kPosdef));
    }

    private static long expectedObservationTransformDoubleCount(int kEndog, int kStates, int nobs) {
        return (long) kEndog * kStates * nobs
            + (long) kEndog * nobs
            + (long) kEndog * kEndog * nobs
            + (long) kEndog * nobs
            + (long) kEndog * kEndog
            + 2L * kEndog;
    }

    private static long expectedUnivariateResultDoubleCount(KalmanSSM model, FilterOptions spec) {
        return spec.createResultLayout(
            1,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            hasMissingObservations(model)).totalLength();
    }

    private static long expectedUnivariateDiffuseResultDoubleCount(KalmanSSM model, FilterOptions spec) {
        return spec.createDiffuseLayout(
            1,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            hasMissingObservations(model)).totalLength();
    }

    private static boolean hasMissingObservations(KalmanSSM model) {
        for (int t = 0; t < model.observationCount(); t++) {
            if (model.missingCount(t) > 0) {
                return true;
            }
        }
        return false;
    }
}
