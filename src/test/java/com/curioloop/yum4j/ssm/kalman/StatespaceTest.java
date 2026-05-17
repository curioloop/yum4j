package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.ssm.kalman.filter.*;
import com.curioloop.yum4j.ssm.kalman.init.*;
import com.curioloop.yum4j.ssm.kalman.model.*;
import com.curioloop.yum4j.ssm.kalman.smooth.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class StatespaceTest {

    private static final double TOL = 1e-8;

    private static ModelFixture buildAR1() {
        return KalmanModelFixtures.defaultScalarAr1();
    }

    private static ModelFixture buildLocalLevel() {
        return KalmanModelFixtures.defaultLocalLevel();
    }

    private static ModelFixture buildMultivariate() {
        return KalmanModelFixtures.defaultBivariate2State();
    }

    private static KalmanSSM buildSharedAR1Default() {
        return KalmanSSM.builder(1, 1, 1, 5)
                .design(new double[]{0.5}, false)
                .obsIntercept(new double[]{0.0}, false)
                .obsCov(new double[]{1.0}, false)
                .transition(new double[]{0.5}, false)
                .stateIntercept(new double[]{0.0}, false)
                .selection(new double[]{1.0}, false)
                .stateCovariance(new double[]{1.0}, false)
                .endog(new double[]{1.0, 0.5, -0.3, 0.8, 0.2})
                .allObserved()
                .build();
    }

    @Test
    void testAR1Filter() {
        ModelFixture rep = buildAR1();
        FilterResult r = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(0.0, r.forecast(0, 0), TOL);
        assertEquals(1.0, r.forecastError(0, 0), TOL);
        assertEquals(250001.0, r.forecastErrorCov(0, 0, 0), TOL);

        double expectedFiltState0 = 500000.0 / 250001.0;
        assertEquals(expectedFiltState0, r.filteredState(0, 0), TOL);

        double expectedFiltStateCov0 = 1e6 - 250000000000.0 / 250001.0;
        assertEquals(expectedFiltStateCov0, r.filteredStateCov(0, 0, 0), TOL);

        double expectedPredState1 = 0.5 * expectedFiltState0;
        assertEquals(expectedPredState1, r.predictedState(0, 1), TOL);

        double expectedPredStateCov1 = 0.5 * expectedFiltStateCov0 * 0.5 + 1.0;
        assertEquals(expectedPredStateCov1, r.predictedStateCov(0, 0, 1), TOL);

        double expectedKG = 0.5 * 1e6 * 0.5 / 250001.0;
        assertEquals(expectedKG, r.kalmanGain(0, 0, 0), TOL);
    }

    @Test
    void testLocalLevelFilter() {
        ModelFixture rep = buildLocalLevel();
        double[] a0 = {0.0};
        double[] P0 = {1.0};
        FilterResult r = KalmanEngine.filter(rep, InitialState.known(a0, P0), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(0.0, r.forecast(0, 0), TOL);
        assertEquals(1.0, r.forecastError(0, 0), TOL);
        assertEquals(2.0, r.forecastErrorCov(0, 0, 0), TOL);
        assertEquals(0.5, r.filteredState(0, 0), TOL);
        assertEquals(0.5, r.filteredStateCov(0, 0, 0), TOL);

        assertEquals(0.5, r.predictedState(0, 1), TOL);
        assertEquals(1.0, r.predictedStateCov(0, 0, 1), TOL);

        assertEquals(0.5, r.forecast(0, 1), TOL);
        assertEquals(1.5, r.forecastError(0, 1), TOL);
        assertEquals(2.0, r.forecastErrorCov(0, 0, 1), TOL);
        assertEquals(1.25, r.filteredState(0, 1), TOL);
        assertEquals(0.5, r.filteredStateCov(0, 0, 1), TOL);

        assertEquals(1.25, r.predictedState(0, 2), TOL);
        assertEquals(1.0, r.predictedStateCov(0, 0, 2), TOL);
    }

    @Test
    void testMultivariateFilter() {
        ModelFixture rep = buildMultivariate();
        FilterResult r = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(0.0, r.forecast(0, 0), TOL);
        assertEquals(0.0, r.forecast(1, 0), TOL);
        assertEquals(1.0, r.forecastError(0, 0), TOL);
        assertEquals(0.5, r.forecastError(1, 0), TOL);
        assertEquals(1000001.0, r.forecastErrorCov(0, 0, 0), TOL);
        assertEquals(0.0, r.forecastErrorCov(0, 1, 0), TOL);
        assertEquals(0.0, r.forecastErrorCov(1, 0, 0), TOL);
        assertEquals(1000001.0, r.forecastErrorCov(1, 1, 0), TOL);

        double expectedFiltState0 = 1e6 / 1000001.0;
        double expectedFiltState1 = 0.5 * 1e6 / 1000001.0;
        assertEquals(expectedFiltState0, r.filteredState(0, 0), TOL);
        assertEquals(expectedFiltState1, r.filteredState(1, 0), TOL);

        double expectedFiltCov = 1e6 - 1e12 / 1000001.0;
        assertEquals(expectedFiltCov, r.filteredStateCov(0, 0, 0), TOL);
        assertEquals(0.0, r.filteredStateCov(0, 1, 0), TOL);
        assertEquals(0.0, r.filteredStateCov(1, 0, 0), TOL);
        assertEquals(expectedFiltCov, r.filteredStateCov(1, 1, 0), TOL);
    }

    @Test
    void testLogLikelihood() {
        ModelFixture rep = buildAR1();
        FilterResult r = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        double expectedLL0 = -0.5 * (Math.log(2 * Math.PI) + Math.log(250001.0) + 1.0 / 250001.0);
        assertEquals(expectedLL0, r.logLikelihoodObs[0], TOL);

        double totalLL = 0.0;
        for (int t = 0; t < r.nobs; t++) {
            totalLL += r.logLikelihoodObs[t];
        }
        assertEquals(totalLL, r.logLikelihood(), TOL);
    }

    @Test
    void testSymmetry() {
        ModelFixture rep = buildMultivariate();
        FilterResult r = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        for (int t = 0; t <= r.nobs; t++) {
            for (int i = 0; i < r.kStates; i++) {
                for (int j = i + 1; j < r.kStates; j++) {
                    assertEquals(r.predictedStateCov(i, j, t),
                            r.predictedStateCov(j, i, t), TOL);
                }
            }
        }
        for (int t = 0; t < r.nobs; t++) {
            for (int i = 0; i < r.kStates; i++) {
                for (int j = i + 1; j < r.kStates; j++) {
                    assertEquals(r.filteredStateCov(i, j, t),
                            r.filteredStateCov(j, i, t), TOL);
                }
            }
            for (int i = 0; i < r.kEndog; i++) {
                for (int j = i + 1; j < r.kEndog; j++) {
                    assertEquals(r.forecastErrorCov(i, j, t),
                            r.forecastErrorCov(j, i, t), TOL);
                }
            }
        }
    }

    @Test
    void testMissingDataAll() {
        ModelFixture rep = buildAR1();
        rep.setMissing(new boolean[]{true}, 2);

        FilterResult r = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        double expectedForecast = rep.obsIntercept[rep.obsInterceptOffset(2)] +
                rep.design[rep.designOffset(2)] * r.predictedState(0, 2);
        assertEquals(expectedForecast, r.forecast(0, 2), TOL);
        assertEquals(0.0, r.forecastError(0, 2), TOL);
        assertEquals(r.filteredState(0, 2), r.predictedState(0, 2), TOL);
        assertEquals(r.filteredStateCov(0, 0, 2), r.predictedStateCov(0, 0, 2), TOL);
        assertEquals(0.0, r.logLikelihoodObs[2], TOL);
    }

    @Test
    void testMissingDataPartial() {
        ModelFixture rep = buildMultivariate();
        rep.setMissing(new boolean[]{true, false}, 1);

        FilterResult r = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertNotNull(r);
        assertEquals(3, r.nobs);
        for (int t = 0; t < r.nobs; t++) {
            assertFalse(Double.isNaN(r.filteredState(0, t)));
            assertFalse(Double.isNaN(r.filteredState(1, t)));
        }
    }

    @Test
    void testConventionalSmoother() {
        ModelFixture rep = buildLocalLevel();
        double[] a0 = {0.0};
        double[] P0 = {1.0};
        FilterResult fr = KalmanEngine.filter(rep, InitialState.known(a0, P0), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult sr = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        for (int t = 0; t < fr.nobs; t++) {
            double pred = fr.predictedState(0, t);
            double filt = fr.filteredState(0, t);
            double smooth = sr.smoothedState(0, t);
            assertTrue(smooth >= Math.min(pred, filt) - TOL || smooth <= Math.max(pred, filt) + TOL);
        }

        for (int t = 0; t < fr.nobs; t++) {
            for (int i = 0; i < fr.kStates; i++) {
                for (int j = i + 1; j < fr.kStates; j++) {
                    assertEquals(sr.smoothedStateCov(i, j, t),
                            sr.smoothedStateCov(j, i, t), TOL);
                }
            }
        }

        for (int t = 1; t < fr.nobs; t++) {
            double r_prev = sr.scaledSmoothedEstimator[(t - 1) * fr.kStates];
            double expectedSmoothed = fr.predictedState(0, t) + fr.predictedStateCov(0, 0, t) * r_prev;
            assertEquals(expectedSmoothed, sr.smoothedState(0, t), TOL);
        }
    }

    @Test
    void testSmootherAR1() {
        ModelFixture rep = buildAR1();
        double[] a0 = {0.0};
        double[] P0 = {1.0};
        FilterResult fr = KalmanEngine.filter(rep, InitialState.known(a0, P0), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult sr = SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        for (int t = 1; t < fr.nobs; t++) {
            double r_prev = sr.scaledSmoothedEstimator[(t - 1) * fr.kStates];
            double expectedSmoothed = fr.predictedState(0, t) + fr.predictedStateCov(0, 0, t) * r_prev;
            assertEquals(expectedSmoothed, sr.smoothedState(0, t), TOL);
        }

        for (int t = 0; t < fr.nobs; t++) {
            assertFalse(Double.isNaN(sr.smoothedState(0, t)));
            assertFalse(Double.isInfinite(sr.smoothedState(0, t)));
            assertTrue(sr.smoothedStateCov(0, 0, t) > 0);
        }
    }

    @Test
    void testUnivariateFilter() {
        ModelFixture rep = buildAR1();
        FilterResult rConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult rUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        for (int t = 0; t < rConv.nobs; t++) {
            assertEquals(rConv.filteredState(0, t), rUni.filteredState(0, t), TOL);
            assertEquals(rConv.filteredStateCov(0, 0, t), rUni.filteredStateCov(0, 0, t), TOL);
        }
        assertEquals(rConv.logLikelihood(), rUni.logLikelihood(), TOL);
    }

    @Test
    void testUnivariateMultivariate() {
        int kEndog = 2, kStates = 2, kPosdef = 2, nobs = 3;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] Z = {1, 0, 0, 1};
        double[] d = {0, 0};
        double[] H = {1.0, 0, 0, 1.0};
        double[] T = {0.9, 0.1, 0.0, 0.9};
        double[] c = {0, 0};
        double[] R = {1, 0, 0, 1};
        double[] Q = {1.0, 0, 0, 1.0};
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

        FilterResult rConv = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult rUni = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        for (int t = 0; t < rConv.nobs; t++) {
            for (int i = 0; i < rConv.kStates; i++) {
                assertEquals(rConv.filteredState(i, t), rUni.filteredState(i, t), TOL);
                for (int j = 0; j < rConv.kStates; j++) {
                    assertEquals(rConv.filteredStateCov(i, j, t), rUni.filteredStateCov(i, j, t), TOL);
                }
            }
        }
        assertEquals(rConv.logLikelihood(), rUni.logLikelihood(), TOL);
    }

    @Test
    void testStationaryInit() {
        int kStates = 1, kPosdef = 1;
        double[] T = {0.5};
        double[] R = {1.0};
        double[] Q = {1.0};
        double[] c = {0.0};

        ModelFixture rep = new ModelFixture(1, kStates, kPosdef, 1);
        rep.setTransition(T, 0);
        rep.setSelection(R, 0);
        rep.setStateCov(Q, 0);
        rep.setStateIntercept(c, 0);

        InitialState result = InitialState.stationary(rep);
        double expectedP0 = 4.0 / 3.0;
        assertEquals(expectedP0, result.initialStateCov()[0], TOL);
        assertEquals(0.0, result.initialState()[0], TOL);

        InitialState repeated = InitialState.stationary(rep);
        assertEquals(result.initialState()[0], repeated.initialState()[0], TOL);
        assertEquals(result.initialStateCov()[0], repeated.initialStateCov()[0], TOL);
    }

    @Test
    void testStationaryInitUsesMeanInFilters() {
        ModelFixture rep = buildAR1();
        for (int t = 0; t < rep.nobs; t++) {
            rep.setStateIntercept(new double[]{1.0}, t);
        }

        InitialState stationaryInit = InitialState.stationary(rep);
        InitialState knownInit = InitialState.known(
            stationaryInit.initialState().clone(),
            stationaryInit.initialStateCov().clone());

        FilterResult kalmanStationary = KalmanEngine.filter(rep, stationaryInit, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult kalmanKnown = KalmanEngine.filter(rep, knownInit, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult univariateStationary = KalmanEngine.filterUnivariate(rep, stationaryInit, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult univariateKnown = KalmanEngine.filterUnivariate(rep, knownInit, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(2.0, stationaryInit.initialState()[0], TOL);
        assertEquals(2.0, kalmanStationary.predictedState(0, 0), TOL);
        assertEquals(2.0, univariateStationary.predictedState(0, 0), TOL);

        for (int t = 0; t <= rep.nobs; t++) {
            assertEquals(kalmanKnown.predictedState(0, t), kalmanStationary.predictedState(0, t), TOL);
            assertEquals(kalmanKnown.predictedStateCov(0, 0, t), kalmanStationary.predictedStateCov(0, 0, t), TOL);
            assertEquals(univariateKnown.predictedState(0, t), univariateStationary.predictedState(0, t), TOL);
            assertEquals(univariateKnown.predictedStateCov(0, 0, t), univariateStationary.predictedStateCov(0, 0, t), TOL);
        }
        for (int t = 0; t < rep.nobs; t++) {
            assertEquals(kalmanKnown.filteredState(0, t), kalmanStationary.filteredState(0, t), TOL);
            assertEquals(univariateKnown.filteredState(0, t), univariateStationary.filteredState(0, t), TOL);
        }
    }

    @Test
    void testKalmanGain() {
        ModelFixture rep = buildAR1();
        FilterResult r = KalmanEngine.filter(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        for (int t = 0; t < r.nobs; t++) {
            double kg = r.kalmanGain(0, 0, t);
            assertFalse(Double.isNaN(kg));
            assertFalse(Double.isInfinite(kg));
        }
    }

    @Test
    void testMemoryControl() {
        ModelFixture rep = buildAR1();
        FilterResult r = KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
            FilterOptions.defaults().without(FilterOptions.Surface.KALMAN_GAIN));

        assertEquals(0, r.kalmanGain.length);
    }

    @Test
    void testConcreteDefaultModelMatchesFixture() {
        KalmanSSM direct = buildSharedAR1Default();
        ModelFixture fixture = buildAR1();

        FilterResult directResult = KalmanEngine.filter(direct, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult fixtureResult = KalmanEngine.filter(fixture, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        assertEquals(fixtureResult.logLikelihood(), directResult.logLikelihood(), TOL);
        for (int t = 0; t < fixtureResult.nobs; t++) {
            assertEquals(fixtureResult.forecast(0, t), directResult.forecast(0, t), TOL);
            assertEquals(fixtureResult.forecastError(0, t), directResult.forecastError(0, t), TOL);
            assertEquals(fixtureResult.filteredState(0, t), directResult.filteredState(0, t), TOL);
            assertEquals(fixtureResult.filteredStateCov(0, 0, t), directResult.filteredStateCov(0, 0, t), TOL);
        }
    }

    @Test
    void testSharedBuilderReusesStaticBlocks() {
        KalmanSSM rep = buildSharedAR1Default();

        assertEquals(0, rep.designOffset(0));
        assertEquals(0, rep.designOffset(4));
        assertEquals(0, rep.transitionOffset(4));
        assertEquals(4, rep.endogOffset(4));
        assertEquals(12L, rep.retainedModelDoubleCount());
        assertEquals(0, rep.missingCount(0));
        assertFalse(rep.isMissing(0, 3));
    }

    @Test
    void testComplexDefaultModelLayout() {
        KalmanSSM rep = KalmanSSM.complex(1, 2, 1, 3);

        assertTrue(rep.complex());
        assertEquals(12, rep.design.length);
        assertEquals(8, rep.designOffset(2));
        assertEquals(6, rep.endog.length);
        assertEquals(4, rep.endogOffset(2));
    }
}
