package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.KalmanModelFixtures;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.ssm.kalman.filter.*;
import com.curioloop.yum4j.ssm.kalman.init.*;
import com.curioloop.yum4j.ssm.kalman.model.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ZKalmanSmootherTest {

    private static final double TOL = 1e-10;

    static ModelFixture buildAR1(double phi, double sigma, double[] y) {
        return KalmanModelFixtures.signalOnlyAr1(phi, sigma, y);
    }

    static ModelFixture buildZAR1(double phiRe, double phiIm, double sigma, double[] y) {
        return KalmanModelFixtures.complexSignalOnlyAr1(phiRe, phiIm, sigma, y);
    }

    @Test
    void testRealDegradationAR1() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7, -0.5, 0.3};
        int kStates = 1;
        int nobs = y.length;

        ModelFixture realRep = buildAR1(phi, sigma, y);
        FilterResult realFr = KalmanEngine.filter(realRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult realSr = SmootherEngine.smooth(realRep, realFr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        ZFilterResult zFr = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZSmootherResult zSr = SmootherEngine.smoothComplex(zRep, zFr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        int ks2c = kStates * 2 * kStates;

        for (int t = 0; t < nobs; t++) {
            int realOff = t * kStates;
            int zOff = t * 2 * kStates;
            assertEquals(realSr.smoothedState[realOff],
                    zSr.smoothedState[zOff], TOL,
                    "smoothedState Re at t=" + t);
            assertEquals(0.0, zSr.smoothedState[zOff + 1], TOL,
                    "smoothedState Im at t=" + t);
        }

        for (int t = 0; t < nobs; t++) {
            int realCovOff = t * kStates * kStates;
            int zCovOff = t * ks2c;
            assertEquals(realSr.smoothedStateCov[realCovOff],
                    zSr.smoothedStateCov[zCovOff], TOL,
                    "smoothedStateCov Re at t=" + t);
            assertEquals(0.0, zSr.smoothedStateCov[zCovOff + 1], TOL,
                    "smoothedStateCov Im at t=" + t);
        }

        for (int t = 0; t < nobs; t++) {
            int realOff = t * kStates;
            int zOff = t * 2 * kStates;
            assertEquals(realSr.scaledSmoothedEstimator[realOff],
                    zSr.scaledSmoothedEstimator[zOff], TOL,
                    "scaledSmoothedEstimator Re at t=" + t);
            assertEquals(0.0, zSr.scaledSmoothedEstimator[zOff + 1], TOL,
                    "scaledSmoothedEstimator Im at t=" + t);
        }
    }

    @Test
    void testComplexAR1Smoother() {
        double phiRe = 0.5, phiIm = 0.3, sigma = 1.0;
        int nobs = 10;
        int kStates = 1;

        double[] yRe = new double[nobs];
        double[] yIm = new double[nobs];
        double[] eps = {1.0, 0.5, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, 0.1, -0.4};
        yRe[0] = eps[0];
        yIm[0] = 0.0;
        for (int t = 1; t < nobs; t++) {
            yRe[t] = phiRe * yRe[t - 1] - phiIm * yIm[t - 1] + eps[t];
            yIm[t] = phiRe * yIm[t - 1] + phiIm * yRe[t - 1];
        }

        ModelFixture rep = ModelFixture.complex(1, 1, 1, nobs);
        double[] Z = {1.0, 0.0};
        double[] d = {0.0, 0.0};
        double[] H = {0.5, 0.0};
        double[] T = {phiRe, phiIm};
        double[] c = {0.0, 0.0};
        double[] R = {1.0, 0.0};
        double[] Q = {sigma * sigma, 0.0};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{yRe[t], yIm[t]}, t);
        }

        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZSmootherResult sr = SmootherEngine.smoothComplex(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        boolean hasNonZeroImag = false;
        for (int t = 0; t < nobs; t++) {
            int off = t * 2 * kStates;
            if (Math.abs(sr.smoothedState[off + 1]) > 1e-12) hasNonZeroImag = true;
        }
        assertTrue(hasNonZeroImag, "smoothedState should have non-zero imaginary parts");

        int ks2c = kStates * 2 * kStates;
        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
                assertComplexSymmetric(sr.smoothedStateCov, covOff, kStates,
                    "smoothedStateCov at t=" + t);
        }

        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
            double diagRe = sr.smoothedStateCov[covOff];
            assertTrue(diagRe > 0,
                    "smoothedStateCov diagonal Re should be positive at t=" + t);
        }
    }

    @Test
    void testComplexUnivariateSmootherUsesUnivariateFilterHistory() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
        ModelFixture rep = buildZAR1(0.5, 0.1, 1.0, y);
        ZFilterResult filterResult = KalmanEngine.filterComplex(rep,
            InitialState.known(new double[]{0.0, 0.0}, new double[]{0.5, 0.0}),
            SmootherOptions.univariate().requiredFilterOptions());

        ZSmootherResult actual = SmootherEngine.smoothComplex(rep, filterResult, SmootherOptions.univariate());

        assertTrue(filterResult.perObsKalmanGain);
        assertEquals(rep.nobs * rep.kStates * 2, actual.smoothedStateLength());
        assertEquals((rep.nobs + 1) * rep.kStates * 2, actual.scaledSmoothedEstimatorLength());
        for (int t = 0; t < rep.nobs; t++) {
            int stateOff = actual.smoothedStateOffset(t);
            assertTrue(Double.isFinite(actual.smoothedState[stateOff]), "smoothedState Re at t=" + t);
            assertTrue(Double.isFinite(actual.smoothedState[stateOff + 1]), "smoothedState Im at t=" + t);
        }
    }

    @Test
    void testComplexUnivariateSmootherRejectsNonDiagonalObservationCovariance() {
        ModelFixture rep = complexCorrelatedObservationModel();
        InitialState init = InitialState.known(
            new double[]{0.0, 0.0, 0.0, 0.0},
            new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});

        UnsupportedOperationException initialPathFailure = assertThrows(UnsupportedOperationException.class,
            () -> SmootherEngine.smoothComplex(rep, init, SmootherOptions.univariate()));
        assertTrue(initialPathFailure.getMessage().contains("UNIVARIATE smoothing requires diagonal observation covariance"));

        ZFilterResult filterResult = KalmanEngine.filterComplex(rep, init,
            SmootherOptions.univariate().requiredFilterOptions());
        UnsupportedOperationException filterPathFailure = assertThrows(UnsupportedOperationException.class,
            () -> SmootherEngine.smoothComplex(rep, filterResult, SmootherOptions.univariate()));
        assertTrue(filterPathFailure.getMessage().contains("UNIVARIATE smoothing requires diagonal observation covariance"));
    }

    private static ModelFixture complexCorrelatedObservationModel() {
        ModelFixture rep = ModelFixture.complex(2, 2, 2, 3);
        double[] design = {
            1.0, 0.0, 0.2, -0.1,
            0.1, 0.05, 1.0, 0.0
        };
        double[] obsCov = {
            0.6, 0.0, 0.12, 0.04,
            0.12, -0.04, 0.9, 0.0
        };
        double[] transition = {
            0.7, 0.0, 0.1, 0.05,
            0.0, 0.0, 0.5, 0.0
        };
        double[] selection = {
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        };
        double[] stateCov = {
            0.3, 0.0, 0.02, 0.0,
            0.02, 0.0, 0.4, 0.0
        };
        double[][] endog = {
            {1.0, 0.2, -0.3, 0.1},
            {0.5, -0.1, 0.1, 0.05},
            {-0.3, 0.4, 0.4, -0.2}
        };
        for (int t = 0; t < rep.nobs; t++) {
            rep.setDesign(design, t);
            rep.setObsCov(obsCov, t);
            rep.setTransition(transition, t);
            rep.setSelection(selection, t);
            rep.setStateCov(stateCov, t);
            rep.setEndog(endog[t], t);
        }
        return rep;
    }

    @Test
    void testComplexSymmetricCovariances() {
        int kEndog = 1, kStates = 2, kPosdef = 2, nobs = 5;

        ModelFixture rep = ModelFixture.complex(kEndog, kStates, kPosdef, nobs);

        double[] Z = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};
        double[] d = {0.0, 0.0};
        double[] H = {0.0, 0.0};
        double[] T = {0.5, 0.1, 0.0, 0.0,
                       0.2, -0.05, 0.3, 0.1,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0};
        double[] c = {0.0, 0.0, 0.0, 0.0};
        double[] R = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};
        double[] Q = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};

        double[] y = {1.0, 0.0, 0.5, 0.0, -0.3, 0.0, 0.8, 0.0, 0.2, 0.0};

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

        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZSmootherResult sr = SmootherEngine.smoothComplex(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        int ks2c = kStates * 2 * kStates;
        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
                assertComplexSymmetric(sr.smoothedStateCov, covOff, kStates,
                    "smoothedStateCov at t=" + t);
        }
    }

    @Test
    void testScaledSmoothedEstimatorIndex() {
        double phi = 0.5, sigma = 1.0;
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7, -0.5, 0.3};
        int kStates = 1;
        int nobs = y.length;

        ModelFixture zRep = buildZAR1(phi, 0.0, sigma, y);
        ZFilterResult zFr = KalmanEngine.filterComplex(zRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZSmootherResult zSr = SmootherEngine.smoothComplex(zRep, zFr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        for (int t = 1; t < nobs; t++) {
            int predOff = t * 2 * kStates;
            int predCovOff = t * kStates * 2 * kStates;
            int rPrevOff = (t - 1) * 2 * kStates;

            double aRe = zFr.predictedState[predOff];
            double aIm = zFr.predictedState[predOff + 1];
            double PRe = zFr.predictedStateCov[predCovOff];
            double PIm = zFr.predictedStateCov[predCovOff + 1];
            double rRe = zSr.scaledSmoothedEstimator[rPrevOff];
            double rIm = zSr.scaledSmoothedEstimator[rPrevOff + 1];

            double expectedRe = aRe + PRe * rRe - PIm * rIm;
            double expectedIm = aIm + PRe * rIm + PIm * rRe;

            int smoothOff = t * 2 * kStates;
            assertEquals(expectedRe, zSr.smoothedState[smoothOff], TOL,
                    "smoothedState Re via formula at t=" + t);
            assertEquals(expectedIm, zSr.smoothedState[smoothOff + 1], TOL,
                    "smoothedState Im via formula at t=" + t);
        }
    }

    @Test
    void testMultivariateComplexSmoother() {
        int kEndog = 2, kStates = 2, kPosdef = 2, nobs = 5;

        ModelFixture rep = ModelFixture.complex(kEndog, kStates, kPosdef, nobs);

        double[] Z = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};
        double[] d = {0.0, 0.0, 0.0, 0.0};
        double[] H = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};
        double[] T = {0.5, 0.1, 0.0, 0.0,
                       0.2, -0.05, 0.3, 0.1,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0};
        double[] c = {0.0, 0.0, 0.0, 0.0};
        double[] R = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};
        double[] Q = {1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0};

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{1.0, 0.0, 0.5, 0.0}, t);
        }

        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZSmootherResult sr = SmootherEngine.smoothComplex(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        int ks2c = kStates * 2 * kStates;
        for (int t = 0; t < nobs; t++) {
            int covOff = t * ks2c;
                assertComplexSymmetric(sr.smoothedStateCov, covOff, kStates,
                    "smoothedStateCov at t=" + t);
        }

        boolean hasNonZeroImag = false;
        for (int t = 0; t < nobs; t++) {
            int off = t * 2 * kStates;
            for (int i = 0; i < kStates; i++) {
                if (Math.abs(sr.smoothedState[off + i * 2 + 1]) > 1e-12) hasNonZeroImag = true;
            }
        }
        assertTrue(hasNonZeroImag, "smoothedState should have non-zero imaginary parts");
    }

    @Test
    void testDiffuseStorageAllocation() {
        ZSmootherResult result = new ZSmootherResult(2, 3, 1, 4);

        assertNull(result.scaledSmoothedDiffuseEstimator);
        assertNull(result.scaledSmoothedDiffuse1EstimatorCov);
        assertNull(result.scaledSmoothedDiffuse2EstimatorCov);

        result.allocateDiffuse();

        assertNotNull(result.scaledSmoothedDiffuseEstimator);
        assertNotNull(result.scaledSmoothedDiffuse1EstimatorCov);
        assertNotNull(result.scaledSmoothedDiffuse2EstimatorCov);
        assertEquals(2 * 3 * 5, result.scaledSmoothedDiffuseEstimator.length);
        assertEquals(3 * 2 * 3 * 5, result.scaledSmoothedDiffuse1EstimatorCov.length);
        assertEquals(3 * 2 * 3 * 5, result.scaledSmoothedDiffuse2EstimatorCov.length);
    }

        @Test
        void testExactDiffuseLocalLevelRealDegradation() {
        double[] y = {10.2394, 1.0, 1.0, 1.0, 1.0, 1.0};
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;

        ModelFixture realRep = ZKalmanFilterTest.buildExactDiffuseLocalLevel(y, sigma2Y, sigma2Mu);
        FilterResult realFr = KalmanEngine.filter(realRep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult realSr = SmootherEngine.smooth(realRep, realFr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        ModelFixture zRep = ZKalmanFilterTest.buildZExactDiffuseLocalLevel(y, sigma2Y, sigma2Mu);
        ZFilterResult zFr = KalmanEngine.filterComplex(zRep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZSmootherResult zSr = SmootherEngine.smoothComplex(zRep, zFr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        for (int t = 0; t < y.length; t++) {
            int zOff = t * 2;
            assertEquals(realSr.smoothedState[t], zSr.smoothedState[zOff], TOL,
                "smoothedState Re at t=" + t);
            assertEquals(0.0, zSr.smoothedState[zOff + 1], TOL,
                "smoothedState Im at t=" + t);
            assertEquals(realSr.smoothedStateCov[t], zSr.smoothedStateCov[zOff], TOL,
                "smoothedStateCov Re at t=" + t);
            assertEquals(realSr.smoothedObsDisturbance[t], zSr.smoothedObsDisturbance[zOff], TOL,
                "smoothedObsDisturbance Re at t=" + t);
            assertEquals(realSr.smoothedObsDisturbanceCov[t], zSr.smoothedObsDisturbanceCov[zOff], TOL,
                "smoothedObsDisturbanceCov Re at t=" + t);
        }

        for (int t = 0; t <= y.length; t++) {
            int zOff = t * 2;
            assertEquals(realSr.scaledSmoothedEstimator[t], zSr.scaledSmoothedEstimator[zOff], TOL,
                "scaledSmoothedEstimator Re at t=" + t);
            assertEquals(realSr.scaledSmoothedEstimatorCov[t], zSr.scaledSmoothedEstimatorCov[zOff], TOL,
                "scaledSmoothedEstimatorCov Re at t=" + t);
            assertEquals(realSr.scaledSmoothedDiffuseEstimator[t], zSr.scaledSmoothedDiffuseEstimator[zOff], TOL,
                "scaledSmoothedDiffuseEstimator Re at t=" + t);
            assertEquals(realSr.scaledSmoothedDiffuse1EstimatorCov[t], zSr.scaledSmoothedDiffuse1EstimatorCov[zOff], TOL,
                "scaledSmoothedDiffuse1EstimatorCov Re at t=" + t);
            assertEquals(realSr.scaledSmoothedDiffuse2EstimatorCov[t], zSr.scaledSmoothedDiffuse2EstimatorCov[zOff], TOL,
                "scaledSmoothedDiffuse2EstimatorCov Re at t=" + t);
        }
        }

        @Test
        void testExactDiffuseLocalLinearTrendMissingRealDegradation() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;
        double sigma2Beta = 2.334;

        ModelFixture realRep = ZKalmanFilterTest.buildExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        realRep.setMissing(new boolean[]{true}, 1);
        FilterResult realFr = KalmanEngine.filter(realRep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        SmootherResult realSr = SmootherEngine.smooth(realRep, realFr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        ModelFixture zRep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        zRep.setMissing(new boolean[]{true}, 1);
        ZFilterResult zFr = KalmanEngine.filterComplex(zRep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZSmootherResult zSr = SmootherEngine.smoothComplex(zRep, zFr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        int kStates = 2;
        for (int t = 0; t < y.length; t++) {
            int zStateOff = t * 2 * kStates;
            int zCovOff = t * kStates * 2 * kStates;
            for (int i = 0; i < kStates; i++) {
                assertEquals(realSr.smoothedState[t * kStates + i], zSr.smoothedState[zStateOff + i * 2], TOL,
                    "smoothedState Re at t=" + t + " i=" + i);
                assertEquals(0.0, zSr.smoothedState[zStateOff + i * 2 + 1], TOL,
                    "smoothedState Im at t=" + t + " i=" + i);
                for (int j = 0; j < kStates; j++) {
                    int rIdx = t * kStates * kStates + i * kStates + j;
                    int zIdx = zCovOff + i * 2 * kStates + j * 2;
                    assertEquals(realSr.smoothedStateCov[rIdx], zSr.smoothedStateCov[zIdx], TOL,
                        "smoothedStateCov Re at t=" + t + " [" + i + "," + j + "]");
                    assertEquals(0.0, zSr.smoothedStateCov[zIdx + 1], TOL,
                        "smoothedStateCov Im at t=" + t + " [" + i + "," + j + "]");
                }
            }
            assertEquals(realSr.smoothedObsDisturbance[t], zSr.smoothedObsDisturbance[t * 2], TOL,
                "smoothedObsDisturbance Re at t=" + t);
            assertEquals(realSr.smoothedObsDisturbanceCov[t], zSr.smoothedObsDisturbanceCov[t * 2], TOL,
                "smoothedObsDisturbanceCov Re at t=" + t);
        }
    }

    private static void assertComplexSymmetric(double[] cov, int offset, int kStates, String msg) {
        for (int i = 0; i < kStates; i++) {
            for (int j = i; j < kStates; j++) {
                int ij = offset + (i * 2 * kStates + j * 2);
                int ji = offset + (j * 2 * kStates + i * 2);
                assertEquals(cov[ij], cov[ji], TOL,
                        msg + " P[" + i + "," + j + "].Re vs P[" + j + "," + i + "].Re");
                assertEquals(cov[ij + 1], cov[ji + 1], TOL,
                        msg + " P[" + i + "," + j + "].Im vs P[" + j + "," + i + "].Im");
            }
        }
    }
}
