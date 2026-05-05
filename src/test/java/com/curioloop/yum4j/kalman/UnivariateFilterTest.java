package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.kalman.filter.*;
import com.curioloop.yum4j.kalman.init.*;
import com.curioloop.yum4j.kalman.model.*;
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
        FilterResult r = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse());
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
        FilterResult r = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse());
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
        FilterResult r = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse());
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
        FilterResult r = UnivariateFilter.filter(rep, StateInitialization.diffuse());
        boolean hasNonZeroForecast = false;
        for (int t = 0; t < nobs; t++) {
            assertFalse(Double.isNaN(r.forecast(0, t)), "forecast NaN at t=" + t);
            if (Math.abs(r.forecast(0, t)) > 1e-15) hasNonZeroForecast = true;
        }
        assertTrue(hasNonZeroForecast, "at least some forecast should be non-zero");
    }

    @Test
    void testUnivariateFilterSpecSuppressesStoredOutputs() {
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

        FilterResult result = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse(), null,
                FilterSpec.defaults().without(
                        FilterSpec.Storage.FORECAST,
                        FilterSpec.Storage.PREDICTED_STATE,
                        FilterSpec.Storage.FILTERED_STATE,
                        FilterSpec.Storage.KALMAN_GAIN,
                        FilterSpec.Storage.LIKELIHOOD));

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
    void testUnivariateDiffuseFilterSpecSuppressesStoredOutputs() {
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

        FilterResult result = UnivariateFilter.filter(rep, StateInitialization.diffuse(), null,
                FilterSpec.defaults().without(
                        FilterSpec.Storage.FORECAST,
                        FilterSpec.Storage.PREDICTED_STATE,
                        FilterSpec.Storage.FILTERED_STATE,
                        FilterSpec.Storage.KALMAN_GAIN,
                        FilterSpec.Storage.LIKELIHOOD));

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
        FilterResult result = UnivariateFilter.filter(rep, StateInitialization.diffuse(), pool);
        long expectedScratchDoubles = expectedUnivariateScratchDoubleCount(kEndog, kStates);

        assertTrue(result.nobsDiffuse > 0);
        assertEquals(expectedScratchDoubles, pool.retainedScratchDoubleCount());
        assertEquals(expectedScratchDoubles, (long) pool.scratchBacking.length);
        assertEquals(expectedScratchDoubles * Double.BYTES, pool.retainedTotalByteCount());
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
        UnivariateFilter.filter(rep, StateInitialization.stationary(rep.stateCount()), pool);

        long expectedWorkspaceDoubles = StateInitialization.requiredStationaryBackingLength(rep)
            + StateInitialization.requiredStationaryPivotLength(rep) * ((long) Integer.BYTES) / Double.BYTES;
        assertEquals(expectedUnivariateScratchDoubleCount(kEndog, kStates) + expectedWorkspaceDoubles,
            pool.retainedScratchDoubleCount());
        assertEquals(expectedUnivariateScratchDoubleCount(kEndog, kStates), (long) pool.scratchBacking.length);
    }

    @Test
    void testUnivariateReserveTrimsStationaryWorkspaceBackToBase() {
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
        long expectedScratchDoubles = expectedUnivariateScratchDoubleCount(kEndog, kStates);

        UnivariateFilter.filter(rep, StateInitialization.stationary(rep.stateCount()), pool);

        assertTrue(pool.retainedScratchDoubleCount() > expectedScratchDoubles);

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
        UnivariateFilter.filter(rep, StateInitialization.diffuse(), pool);

        assertTrue(pool.retainedScratchDoubleCount() > 0);

        pool.releaseRetainedScratch();

        assertEquals(0L, pool.retainedScratchDoubleCount());
        assertNull(pool.scratchBacking);
    }

    private static long expectedUnivariateScratchDoubleCount(int kEndog, int kStates) {
        return 5L * kStates * kStates + (long) kStates * kEndog + (long) kEndog * kEndog + 6L * kStates;
    }
}
