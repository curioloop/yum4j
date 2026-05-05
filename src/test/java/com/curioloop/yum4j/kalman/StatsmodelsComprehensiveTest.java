package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.fixtures.StatsmodelsResources;
import com.curioloop.yum4j.kalman.filter.*;
import com.curioloop.yum4j.kalman.init.*;
import com.curioloop.yum4j.kalman.model.*;
import com.curioloop.yum4j.kalman.smooth.*;
import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class StatsmodelsComprehensiveTest {

    private static final double TOL_REF = 1e-4;
    private static final double TOL_STRUCT = 1e-8;
    private static final double TOL_LOOSE = 1e-2;

    private static ModelFixture buildClark1987Model() {
        double sigmaV = 0.005539, sigmaE = 0.006164, sigmaW = 0.000184;
        double phi1 = 1.531659, phi2 = -0.585422;
        int kEndog = 1, kStates = 4, kPosdef = 4;
        int nobs = KalmanFilterRefData.Clark1987.NOBS;

        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);

        double[] Z = {1, 1, 0, 0};
        double[] H = {0};
        double[] T = {1, 0, 0, 1, 0, phi1, phi2, 0, 0, 1, 0, 0, 0, 0, 0, 1};
        double[] R = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        double[] Q = {sigmaV*sigmaV, 0, 0, 0, 0, sigmaE*sigmaE, 0, 0, 0, 0, 0, 0, 0, 0, 0, sigmaW*sigmaW};

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
        }

        double[] gdpData = KalmanFilterRefData.Clark1987.DATA;
        for (int t = 0; t < nobs; t++) {
            rep.setEndog(new double[]{Math.log(gdpData[t])}, t);
        }

        return rep;
    }

    private static double[][] clark1987Init() {
        int kStates = 4;
        double phi1 = 1.531659, phi2 = -0.585422;
        double[] T = {1, 0, 0, 1, 0, phi1, phi2, 0, 0, 1, 0, 0, 0, 0, 0, 1};
        double[] P0 = new double[kStates * kStates];
        for (int i = 0; i < kStates; i++) P0[i * kStates + i] = 100.0;

        double[] tmpTP = new double[kStates * kStates];
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, T, 0, kStates, P0, 0, kStates, 0.0, tmpTP, 0, kStates);
        double[] P0mod = new double[kStates * kStates];
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                1.0, tmpTP, 0, kStates, T, 0, kStates, 0.0, P0mod, 0, kStates);

        double[] a0 = new double[kStates];
        return new double[][]{a0, P0mod};
    }

    private static ModelFixture buildClark1989Model() {
        double sigmaV = 0.004863, sigmaE = 0.00668, sigmaW = 0.000295;
        double sigmaVl = 0.001518, sigmaEc = 0.000306;
        double phi1 = 1.43859, phi2 = -0.517385;
        double alpha1 = -0.336789, alpha2 = -0.163511, alpha3 = -0.072012;
        int kEndog = 2, kStates = 6, kPosdef = 6;
        int skip = 4;
        int nobs = KalmanFilterRefData.Clark1989.NOBS - skip;

        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);

        double[] Z = {1, 1, 0, 0, 0, 0, 0, alpha1, alpha2, alpha3, 0, 1};
        double[] H = {0, 0, 0, sigmaEc * sigmaEc};
        double[] T = {1, 0, 0, 0, 1, 0, 0, phi1, phi2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};
        double[] R = {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};
        double[] Q = new double[36];
        Q[0] = sigmaV * sigmaV; Q[7] = sigmaE * sigmaE; Q[28] = sigmaW * sigmaW; Q[35] = sigmaVl * sigmaVl;

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
        }

        double[] data = KalmanFilterRefData.Clark1989.DATA;
        for (int t = 0; t < nobs; t++) {
            int src = (t + skip) * 2;
            double gdp = Math.log(data[src]);
            double unemp = data[src + 1] / 100.0;
            rep.setEndog(new double[]{gdp, unemp}, t);
        }

        return rep;
    }

    private static double[][] clark1989Init() {
        int kStates = 6;
        double phi1 = 1.43859, phi2 = -0.517385;
        double[] T = {1, 0, 0, 0, 1, 0, 0, phi1, phi2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};
        double[] P0 = new double[kStates * kStates];
        for (int i = 0; i < kStates; i++) P0[i * kStates + i] = 100.0;

        double[] tmpTP = new double[kStates * kStates];
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, T, 0, kStates, P0, 0, kStates, 0.0, tmpTP, 0, kStates);
        double[] P0mod = new double[kStates * kStates];
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                1.0, tmpTP, 0, kStates, T, 0, kStates, 0.0, P0mod, 0, kStates);

        double[] a0 = new double[kStates];
        return new double[][]{a0, P0mod};
    }

    private static ModelFixture buildAR3Model() {
        double phi1 = 0.5270715, phi2 = 0.0952613, phi3 = 0.2580355;
        double sigma2 = 0.5307459;
        int kEndog = 1, kStates = 3, kPosdef = 1;
        double[] wpi = KalmanFilterRefData.Wpi1Ar3.WPI;
        int nobs = wpi.length - 1;

        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);

        double[] Z = {1, 0, 0};
        double[] H = {0};
        double[] T = {phi1, phi2, phi3, 1, 0, 0, 0, 1, 0};
        double[] R = {1, 0, 0};
        double[] Q = {sigma2};

        double[] y = new double[nobs];
        for (int t = 0; t < nobs; t++) {
            y[t] = wpi[t + 1] - wpi[t];
        }

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }

        return rep;
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

    private static StateInitialization knownInitialization(double[][] init) {
        return StateInitialization.known(init[0], init[1]);
    }

    private static StateInitialization stationaryInitialization(double[][] init) {
        return StateInitialization.stationary(init[0], init[1]);
    }

    private static StateInitialization stationaryInitialization(double[] transition, int kStates,
                                                                double[] selection, int kPosdef,
                                                                double[] stateCov, double[] stateIntercept) {
        ModelFixture rep = new ModelFixture(1, kStates, kPosdef, 1);
        rep.setTransition(transition, 0);
        rep.setSelection(selection, 0);
        rep.setStateCov(stateCov, 0);
        rep.setStateIntercept(stateIntercept, 0);
        return StateInitialization.stationary(rep);
    }

    private static double[][] readMacrodata() {
        return StatsmodelsResources.readMacrodata();
    }

    private static List<String[]> readResultsCsv(String fileName) {
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

    @FunctionalInterface
    private interface MatrixValueProvider {
        double get(int row, int col);
    }

    private static double determinant3(MatrixValueProvider values) {
        double a00 = values.get(0, 0);
        double a01 = values.get(0, 1);
        double a02 = values.get(0, 2);
        double a10 = values.get(1, 0);
        double a11 = values.get(1, 1);
        double a12 = values.get(1, 2);
        double a20 = values.get(2, 0);
        double a21 = values.get(2, 1);
        double a22 = values.get(2, 2);
        return a00 * (a11 * a22 - a12 * a21)
                - a01 * (a10 * a22 - a12 * a20)
                + a02 * (a10 * a21 - a11 * a20);
    }

    private static void assertRelativeEquals(double expected, double actual,
                                             double relTol, double absTol,
                                             String message) {
        double scale = Math.max(Math.max(Math.abs(expected), Math.abs(actual)), 1.0);
        double tolerance = Math.max(absTol, relTol * scale);
        assertEquals(expected, actual, tolerance, message);
    }

    private static void assertMultivariateSmoothingTraceAgainstCsv(FilterResult fr,
                                                                   SmootherResult sr,
                                                                   String fileName,
                                                                   int forecastErrorSeriesCount,
                                                                   boolean compareObsDisturbanceCov,
                                                                   boolean skipLastDetN) {
        List<String[]> csv = readResultsCsv(fileName);
        String[] header = csv.get(0);

        int[] rCols = {
                csvColumnIndex(header, "r1"),
                csvColumnIndex(header, "r2"),
                csvColumnIndex(header, "r3")
        };
        int detNCol = csvColumnIndex(header, "detN");
        int[] mCols = {
                csvColumnIndex(header, "m1"),
                csvColumnIndex(header, "m2"),
                csvColumnIndex(header, "m3")
        };
        int[] vCols = {
                csvColumnIndex(header, "v1"),
                csvColumnIndex(header, "v2"),
                csvColumnIndex(header, "v3")
        };
        int[] fCols = {
                csvColumnIndex(header, "F1"),
                csvColumnIndex(header, "F2"),
                csvColumnIndex(header, "F3")
        };
        int[] aCols = {
                csvColumnIndex(header, "a1"),
                csvColumnIndex(header, "a2"),
                csvColumnIndex(header, "a3")
        };
        int detPCol = csvColumnIndex(header, "detP");
        int[] alphaCols = {
                csvColumnIndex(header, "alphahat1"),
                csvColumnIndex(header, "alphahat2"),
                csvColumnIndex(header, "alphahat3")
        };
        int detVCol = csvColumnIndex(header, "detV");
        int[] muhatCols = {
                csvColumnIndex(header, "muhat1"),
                csvColumnIndex(header, "muhat2"),
                csvColumnIndex(header, "muhat3")
        };
        int[] etaCols = {
                csvColumnIndex(header, "etahat1"),
                csvColumnIndex(header, "etahat2"),
                csvColumnIndex(header, "etahat3")
        };
        int detVetaCol = csvColumnIndex(header, "detVeta");
        int[] epsCols = {
                csvColumnIndex(header, "epshat1"),
                csvColumnIndex(header, "epshat2"),
                csvColumnIndex(header, "epshat3")
        };
        int[] vepsCols = {
                csvColumnIndex(header, "Veps1"),
                csvColumnIndex(header, "Veps2"),
                csvColumnIndex(header, "Veps3")
        };

        assertEquals(fr.nobs + 1, csv.size(), "reference row count mismatch");
        for (int t = 0; t < fr.nobs; t++) {
            final int time = t;
            String[] row = csv.get(t + 1);
            for (int i = 0; i < 3; i++) {
                double expectedR = csvValue(row, rCols[i]);
                if (!Double.isNaN(expectedR)) {
                    assertEquals(expectedR, sr.scaledSmoothedEstimator[t * 3 + i], TOL_REF,
                            "scaled smoothed estimator mismatch at t=" + t + " i=" + i);
                }

                double expectedM = csvValue(row, mCols[i]);
                if (!Double.isNaN(expectedM)) {
                    assertEquals(expectedM, fr.forecast(i, t), TOL_REF,
                            "forecast mismatch at t=" + t + " i=" + i);
                }

                double expectedV = csvValue(row, vCols[i]);
                if (i < forecastErrorSeriesCount && !Double.isNaN(expectedV)) {
                    assertEquals(expectedV, fr.forecastError(i, t), TOL_REF,
                            "forecast error mismatch at t=" + t + " i=" + i);
                }

                double expectedF = csvValue(row, fCols[i]);
                if (i < forecastErrorSeriesCount && !Double.isNaN(expectedF)) {
                    assertEquals(expectedF, fr.forecastErrorCov(i, i, t), TOL_REF,
                            "forecast error covariance mismatch at t=" + t + " i=" + i);
                }

                double expectedA = csvValue(row, aCols[i]);
                if (!Double.isNaN(expectedA)) {
                    assertEquals(expectedA, fr.predictedState(i, t + 1), TOL_REF,
                            "predicted state mismatch at t=" + t + " i=" + i);
                }

                double expectedAlpha = csvValue(row, alphaCols[i]);
                if (!Double.isNaN(expectedAlpha)) {
                    assertEquals(expectedAlpha, sr.smoothedState(i, t), TOL_REF,
                            "smoothed state mismatch at t=" + t + " i=" + i);
                }

                // These reference models use Z = I and zero observation intercept, so signal == state.
                double expectedMuhat = csvValue(row, muhatCols[i]);
                if (!Double.isNaN(expectedMuhat)) {
                    assertEquals(expectedMuhat, sr.smoothedState(i, t), TOL_REF,
                            "smoothed signal mismatch at t=" + t + " i=" + i);
                }

                double expectedEta = csvValue(row, etaCols[i]);
                if (!Double.isNaN(expectedEta)) {
                    assertEquals(expectedEta, sr.smoothedStateDisturbance(i, t), TOL_REF,
                            "smoothed state disturbance mismatch at t=" + t + " i=" + i);
                }

                double expectedEps = csvValue(row, epsCols[i]);
                if (!Double.isNaN(expectedEps)) {
                    assertEquals(expectedEps, sr.smoothedObsDisturbance(i, t), TOL_REF,
                            "smoothed obs disturbance mismatch at t=" + t + " i=" + i);
                }

                double expectedVeps = csvValue(row, vepsCols[i]);
                if (compareObsDisturbanceCov && !Double.isNaN(expectedVeps)) {
                    assertEquals(expectedVeps, sr.smoothedObsDisturbanceCov(i, i, t), TOL_REF,
                            "smoothed obs disturbance covariance mismatch at t=" + t + " i=" + i);
                }
            }

            if (!(skipLastDetN && t == fr.nobs - 1)) {
                double expectedDetN = csvValue(row, detNCol);
                if (!Double.isNaN(expectedDetN)) {
                    final int off = sr.scaledSmoothedEstimatorCovOffset(time);
                    double actualDetN = determinant3((i, j) ->
                            sr.scaledSmoothedEstimatorCov[off + i * 3 + j]);
                    assertRelativeEquals(expectedDetN, actualDetN, TOL_REF, TOL_STRUCT,
                            "scaled smoothed estimator covariance determinant mismatch at t=" + t);
                }
            }

            double expectedDetP = csvValue(row, detPCol);
            if (!Double.isNaN(expectedDetP)) {
                double actualDetP = determinant3((i, j) -> fr.predictedStateCov(i, j, time + 1));
                assertRelativeEquals(expectedDetP, actualDetP, TOL_REF, TOL_STRUCT,
                        "predicted state covariance determinant mismatch at t=" + t);
            }

            double actualDetV = determinant3((i, j) -> sr.smoothedStateCov(i, j, time));
            double expectedDetV = csvValue(row, detVCol);
            if (!Double.isNaN(expectedDetV)) {
                assertRelativeEquals(expectedDetV, actualDetV, TOL_REF, TOL_STRUCT,
                        "smoothed state covariance determinant mismatch at t=" + t);
            }

            double expectedDetVeta = csvValue(row, detVetaCol);
            if (!Double.isNaN(expectedDetVeta)) {
                double actualDetVeta = determinant3((i, j) -> sr.smoothedStateDisturbanceCov(i, j, time));
                assertRelativeEquals(expectedDetVeta, actualDetVeta, TOL_REF, TOL_STRUCT,
                        "smoothed state disturbance covariance determinant mismatch at t=" + t);
            }
        }
    }

    private static ModelFixture buildMultivariateMissingModel() {
        return KalmanModelFixtures.statsmodelsMultivariateMissingModel();
    }

    private static ModelFixture buildMultivariateVARModel() {
        double[][] raw = readMacrodata();
        int nobs = 202;
        int kEndog = 3, kStates = 3, kPosdef = 3;

        double[] realgdp = new double[nobs];
        double[] realcons = new double[nobs];
        double[] realinv = new double[nobs];
        for (int t = 0; t < nobs; t++) {
            realgdp[t] = Math.log(raw[t + 1][2]) - Math.log(raw[t][2]);
            realcons[t] = Math.log(raw[t + 1][3]) - Math.log(raw[t][3]);
            realinv[t] = Math.log(raw[t + 1][4]) - Math.log(raw[t][4]);
        }

        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);

        double[] Z = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        double[] H = {0.0000640649, 0, 0, 0, 0.0000572802, 0, 0, 0, 0.0017088585};
        double[] T = {-0.1119908792, 0.8441841604, 0.0238725303, 0.2629347724, 0.4996718412, -0.0173023305, -3.2192369082, 4.1536028244, 0.4514379215};
        double[] R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        double[] Q = {0.0000640649, 0.0000388496, 0.0002148769, 0.0000388496, 0.0000572802, 0.000001555, 0.0002148769, 0.000001555, 0.0017088585};

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{realgdp[t], realcons[t], realinv[t]}, t);
        }

        return rep;
    }

    @Test
    void testClark1987Loglike() {
        ModelFixture rep = buildClark1987Model();
        double[][] init = clark1987Init();
        FilterResult fr = KalmanFilter.filter(rep, knownInitialization(init));
        int start = KalmanFilterRefData.Clark1987.START_YEAR - 1;
        double loglike = 0;
        for (int t = start; t < fr.nobs; t++) loglike += fr.logLikelihoodObs[t];
        assertEquals(KalmanFilterRefData.Clark1987.LOGLIKE, loglike, TOL_LOOSE);
    }

    @Test
    void testClark1987FilteredState() {
        ModelFixture rep = buildClark1987Model();
        double[][] init = clark1987Init();
        FilterResult fr = KalmanFilter.filter(rep, knownInitialization(init));
        int start = KalmanFilterRefData.Clark1987.START_YEAR - 1;
        double[] refStates = KalmanFilterRefData.Clark1987.STATES;
        int nRef = refStates.length / 3;
        int[] stateIdx = {0, 1, 3};
        for (int t = start; t < fr.nobs; t++) {
            int refT = t - start;
            if (refT >= nRef) break;
            for (int col = 0; col < 3; col++) {
                assertEquals(refStates[refT * 3 + col], fr.filteredState(stateIdx[col], t), TOL_REF,
                        "Clark1987 filtered state mismatch at t=" + t + " col=" + col);
            }
        }
    }

    @Test
    void testClark1989Loglike() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();
        FilterResult fr = KalmanFilter.filter(rep, knownInitialization(init));
        double loglike = fr.logLikelihood();
        assertEquals(KalmanFilterRefData.Clark1989.LOGLIKE, loglike, TOL_LOOSE);
    }

    @Test
    void testClark1989FilteredState() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();
        FilterResult fr = KalmanFilter.filter(rep, knownInitialization(init));
        int start = KalmanFilterRefData.Clark1989.START_YEAR - KalmanFilterRefData.Clark1989.START_QUARTER - 1;
        double[] refStates = KalmanFilterRefData.Clark1989.STATES;
        int nRef = refStates.length / 4;
        int[] stateIdx = {0, 1, 4, 5};
        for (int t = start; t < fr.nobs; t++) {
            int refT = t - start;
            if (refT >= nRef) break;
            for (int col = 0; col < 4; col++) {
                assertEquals(refStates[refT * 4 + col], fr.filteredState(stateIdx[col], t), TOL_REF,
                        "Clark1989 filtered state mismatch at t=" + t + " col=" + col);
            }
        }
    }

    @Test
    void testStationaryInit1D() {
        double phi = 0.5, sigma2 = 1.0;
        double[] T = {phi};
        double[] R = {1.0};
        double[] Q = {sigma2};
        double[] c = {0.0};
        StateInitialization init = stationaryInitialization(T, 1, R, 1, Q, c);
        assertEquals(0.0, init.initialState()[0], TOL_STRUCT);
        assertEquals(sigma2 / (1 - phi * phi), init.initialStateCov()[0], TOL_STRUCT);
    }

    @Test
    void testStationaryInit2D() {
        double[] T = {0.5, 0.1, 0, 0.5};
        double[] R = {1, 0, 0, 1};
        double[] Q = {1, 0, 0, 1};
        double[] c = {0, 0};
        StateInitialization init = stationaryInitialization(T, 2, R, 2, Q, c);
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
        StateInitialization init = stationaryInitialization(T, 1, R, 1, Q, c);
        double expectedA = c[0] / (1 - phi);
        assertEquals(expectedA, init.initialState()[0], TOL_STRUCT);
    }

    @Test
    void testAR3SmoothedState() {
        ModelFixture rep = buildAR3Model();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] refSmoothed = KalmanFilterRefData.Wpi1Ar3.STATA_SMOOTHED;
        int kStates = 3;
        int nobs = fr.nobs;

        for (int t = 0; t < nobs; t++) {
            double ref = refSmoothed[(t + 1) * kStates];
            if (Double.isNaN(ref)) continue;
            assertEquals(ref, sr.smoothedState(0, t), TOL_REF,
                    "AR3 smoothed state mismatch at t=" + t);
        }
    }

    @Test
    void testAR3SmoothedStateCovDet() {
        ModelFixture rep = buildAR3Model();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] refDet = KalmanFilterRefData.Wpi1Ar3.MATLAB_DET_SMOOTHED_STATE_COV;
        int kStates = 3;
        int nobs = fr.nobs;
        int nRef = refDet.length;

        for (int t = 0; t < Math.min(nobs, nRef); t++) {
            double det = 1.0;
            for (int i = 0; i < kStates; i++) {
                det *= sr.smoothedStateCov(i, i, t);
            }
            assertEquals(refDet[t], det, TOL_REF,
                    "AR3 smoothed state cov det mismatch at t=" + t);
        }
    }

    @Test
    void testAR3SmoothedObsDisturbance() {
        ModelFixture rep = buildAR3Model();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] ref = KalmanFilterRefData.Wpi1Ar3.MATLAB_SMOOTHED_OBS_DISTURBANCE;
        int nobs = fr.nobs;
        int nRef = ref.length;

        for (int t = 0; t < Math.min(nobs, nRef); t++) {
            assertEquals(ref[t], sr.smoothedObsDisturbance[t], TOL_REF,
                    "AR3 smoothed obs disturbance mismatch at t=" + t);
        }
    }

    @Test
    void testAR3SmoothedStateDisturbance() {
        ModelFixture rep = buildAR3Model();
        double[][] init = ar3StationaryInit();
        FilterResult fr = KalmanFilter.filter(rep, stationaryInitialization(init));
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] ref = KalmanFilterRefData.Wpi1Ar3.MATLAB_SMOOTHED_STATE_DISTURBANCE;
        int nobs = fr.nobs;
        int nRef = ref.length;
        int kPosdef = 1;

        for (int t = 1; t <= Math.min(nobs - 1, nRef); t++) {
            assertEquals(ref[t - 1], sr.smoothedStateDisturbance[t * kPosdef], TOL_REF,
                    "AR3 smoothed state disturbance mismatch at t=" + t);
        }
    }

    @Test
    void testMultivariateMissingSmoothedState() {
        ModelFixture rep = buildMultivariateMissingModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] ref = KalmanFilterRefData.MultivariateMissing.SMOOTHED_STATE;
        int kStates = 3;
        int nobs = fr.nobs;
        int nRef = ref.length / kStates;

        for (int t = 1; t < Math.min(nobs, nRef); t++) {
            for (int i = 0; i < kStates; i++) {
                if (Double.isNaN(ref[t * kStates + i])) continue;
                assertEquals(ref[t * kStates + i], sr.smoothedState(i, t), TOL_REF,
                        "MultivariateMissing smoothed state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testMultivariateMissingLoglike() {
        ModelFixture rep = buildMultivariateMissingModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        assertEquals(-205310.9767, fr.logLikelihood(), TOL_REF);
    }

    @Test
    void testMultivariateMissingConvVsUni() {
        ModelFixture rep = buildMultivariateMissingModel();
        FilterResult frConv = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        FilterResult frUni = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse());

        assertEquals(frUni.logLikelihood(), frConv.logLikelihood(), TOL_REF,
                "Conv vs Uni logLikelihood mismatch");

        SmootherResult srConv = KalmanSmoother.smooth(rep, frConv);
        SmootherResult srUni = KalmanSmoother.smooth(rep, frUni);

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
        ModelFixture rep = buildMultivariateVARModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] ref = KalmanFilterRefData.MultivariateVAR.SMOOTHED_STATE;
        int kStates = 3;
        int nobs = fr.nobs;
        int nRef = ref.length / kStates;

        for (int t = 1; t < Math.min(nobs, nRef); t++) {
            for (int i = 0; i < kStates; i++) {
                if (Double.isNaN(ref[t * kStates + i])) continue;
                assertEquals(ref[t * kStates + i], sr.smoothedState(i, t), TOL_REF,
                        "MultivariateVAR smoothed state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testMultivariateVARLoglike() {
        ModelFixture rep = buildMultivariateVARModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        assertEquals(1695.34872, fr.logLikelihood(), TOL_LOOSE);
    }

    @Test
    void testUnivariateClark1989Forecasts() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();

        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, knownInitialization(init), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, knownInitialization(init), pool2);

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

        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, knownInitialization(init), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, knownInitialization(init), pool2);

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

        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, knownInitialization(init), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, knownInitialization(init), pool2);

        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF);
    }

    @Test
    void testUnivariateClark1989SmoothedState() {
        ModelFixture rep = buildClark1989Model();
        double[][] init = clark1989Init();

        KalmanFilter.Pool fPool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, knownInitialization(init), fPool1);
        SmootherResult srConv = KalmanSmoother.smooth(rep, frConv);

        UnivariateFilter.Pool fPool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, knownInitialization(init), fPool2);
        SmootherResult srUni = KalmanSmoother.smooth(rep, frUni);

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
        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse(), pool2);
        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF);
    }

    @Test
    void testUnivariateMissingAll() {
        double[] y = generateBivariate(20, 333);
        ModelFixture rep = buildBivariateAR1(y);
        rep.setMissing(new boolean[]{true, true}, 5);
        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse(), pool2);
        assertEquals(frConv.logLikelihood(), frUni.logLikelihood(), TOL_REF);
    }

    @Test
    void testUnivariateMissingPartial() {
        double[] y = generateBivariate(20, 222);
        ModelFixture rep = buildBivariateAR1(y);
        rep.setMissing(new boolean[]{true, false}, 5);
        rep.setMissing(new boolean[]{false, true}, 10);
        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse(), pool2);
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
        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse(), pool2);
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

        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        assertEquals(1e6, fr.predictedStateCov(0, 0, 0), TOL_STRUCT);
    }

    @Test
    void testDiffuseVsApproximate() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 50, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frDiffuse = KalmanFilter.filter(rep, StateInitialization.diffuse());
        FilterResult frApprox = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
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

        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        assertEquals(1e6, fr.predictedStateCov(0, 0, 0), TOL_STRUCT);
        assertEquals(1e6, fr.predictedStateCov(1, 1, 0), TOL_STRUCT);
    }

    @Test
    void testMemoryNoLikelihood() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frNoLL = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), null,
            FilterSpec.defaults().without(FilterSpec.Storage.LIKELIHOOD));
        assertEquals(0, frNoLL.logLikelihoodObs.length);
    }

    @Test
    void testMemoryNoForecast() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frAll = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        FilterResult frNoFC = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), null,
            FilterSpec.defaults().without(FilterSpec.Storage.FORECAST));
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
        FilterResult frNoPred = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), null,
                FilterSpec.defaults().without(FilterSpec.Storage.PREDICTED_STATE));
        assertEquals(0, frNoPred.predictedState.length);
        assertEquals(0, frNoPred.predictedStateCov.length);
    }

    @Test
    void testMemoryNoFiltered() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frNoFilt = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), null,
                FilterSpec.defaults().without(FilterSpec.Storage.FILTERED_STATE));
        assertEquals(0, frNoFilt.filteredState.length);
        assertEquals(0, frNoFilt.filteredStateCov.length);
    }

    @Test
    void testMemoryNoGain() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult frAll = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        FilterResult frNoGain = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), null,
                FilterSpec.defaults().without(FilterSpec.Storage.KALMAN_GAIN));
        assertEquals(0, frNoGain.kalmanGain.length);
        assertEquals(frAll.logLikelihood(), frNoGain.logLikelihood(), TOL_STRUCT);
    }

    @Test
    void testMemoryConserve() {
        double phi = 0.5;
        double[] y = generateAR1(phi, 1.0, 20, 42);
        ModelFixture rep = buildAR1(phi, 0.5, y);
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), null,
            FilterSpec.defaults().without(
                FilterSpec.Storage.FORECAST,
                FilterSpec.Storage.PREDICTED_STATE,
                FilterSpec.Storage.FILTERED_STATE,
                FilterSpec.Storage.LIKELIHOOD,
                FilterSpec.Storage.KALMAN_GAIN));
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
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        SmootherResult conv = KalmanSmoother.smooth(rep, fr);
        SmootherResult cls = KalmanSmoother.smooth(rep, fr, null, SmootherSpec.classical());
        SmootherResult alt = KalmanSmoother.smooth(rep, fr, null, SmootherSpec.alternative());

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
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        SmootherResult conv = KalmanSmoother.smooth(rep, fr);
        SmootherResult cls = KalmanSmoother.smooth(rep, fr, null, SmootherSpec.classical());
        SmootherResult alt = KalmanSmoother.smooth(rep, fr, null, SmootherSpec.alternative());

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
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        SmootherResult conv = KalmanSmoother.smooth(rep, fr);
        SmootherResult cls = KalmanSmoother.smooth(rep, fr, null, SmootherSpec.classical());
        SmootherResult alt = KalmanSmoother.smooth(rep, fr, null, SmootherSpec.alternative());

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
    void testAR3PredictedStateVsStata() {
        ModelFixture rep = buildAR3Model();
        double[][] init = ar3StationaryInit();
        FilterResult fr = KalmanFilter.filter(rep, stationaryInitialization(init));
        double[] ref = KalmanFilterRefData.Wpi1Ar3.STATA_PREDICTED;
        int kStates = 3;
        int nobs = fr.nobs;
        int nRef = ref.length / kStates;
        for (int t = 0; t < Math.min(nobs, nRef - 1); t++) {
            for (int i = 0; i < kStates; i++) {
                if (Double.isNaN(ref[(t + 1) * kStates + i])) continue;
                assertEquals(ref[(t + 1) * kStates + i], fr.predictedState(i, t), TOL_REF,
                        "AR3 predicted state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testAR3FilteredStateVsStata() {
        ModelFixture rep = buildAR3Model();
        double[][] init = ar3StationaryInit();
        FilterResult fr = KalmanFilter.filter(rep, stationaryInitialization(init));
        double[] ref = KalmanFilterRefData.Wpi1Ar3.STATA_FILTERED;
        int kStates = 3;
        int nobs = fr.nobs;
        int nRef = ref.length / kStates;
        for (int t = 0; t < Math.min(nobs, nRef - 1); t++) {
            for (int i = 0; i < kStates; i++) {
                if (Double.isNaN(ref[(t + 1) * kStates + i])) continue;
                assertEquals(ref[(t + 1) * kStates + i], fr.filteredState(i, t), TOL_REF,
                        "AR3 filtered state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testAR3SmoothedStateVsMATLAB() {
        ModelFixture rep = buildAR3Model();
        double[][] init = ar3StationaryInit();
        FilterResult fr = KalmanFilter.filter(rep, stationaryInitialization(init));
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);
        double[] ref = KalmanFilterRefData.Wpi1Ar3.MATLAB_SMOOTHED_STATE;
        int kStates = 3;
        int nRef = ref.length / kStates;
        for (int t = 0; t < nRef; t++) {
            for (int i = 0; i < kStates; i++) {
                if (Double.isNaN(ref[t * kStates + i])) continue;
                assertEquals(ref[t * kStates + i], sr.smoothedState(i, t + 1), TOL_REF,
                        "AR3 MATLAB smoothed state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    private static ModelFixture buildAR3MissingModel() {
        double phi1 = 0.5270715, phi2 = 0.0952613, phi3 = 0.2580355;
        double sigma2 = 0.5307459;
        int kEndog = 1, kStates = 3, kPosdef = 1;
        double[] wpi = KalmanFilterRefData.Wpi1Ar3.WPI;
        int nobs = wpi.length - 1;

        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);

        double[] Z = {1, 0, 0};
        double[] H = {0};
        double[] T = {phi1, phi2, phi3, 1, 0, 0, 0, 1, 0};
        double[] R = {1, 0, 0};
        double[] Q = {sigma2};

        double[] y = new double[nobs];
        for (int t = 0; t < nobs; t++) {
            y[t] = wpi[t + 1] - wpi[t];
        }

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }

        for (int t = 9; t <= 19; t++) {
            rep.setMissing(new boolean[]{true}, t);
        }

        return rep;
    }

    @Test
    void testAR3MissingSmoothedStateVsR() {
        ModelFixture rep = buildAR3MissingModel();
        double[][] init = ar3StationaryInit();
        FilterResult fr = KalmanFilter.filter(rep, stationaryInitialization(init));
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] ref = KalmanFilterRefData.Wpi1Ar3MissingR.SMOOTHED_STATE;
        int kStates = 3;
        int nobs = fr.nobs;
        int nRef = ref.length / kStates;

        for (int t = 0; t < Math.min(nobs, nRef); t++) {
            for (int i = 0; i < kStates; i++) {
                if (Double.isNaN(ref[t * kStates + i])) continue;
                assertEquals(ref[t * kStates + i], sr.smoothedState(i, t), TOL_REF,
                        "AR3 missing smoothed state mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testAR3MissingForecastVsR() {
        ModelFixture rep = buildAR3MissingModel();
        double[][] init = ar3StationaryInit();
        FilterResult fr = KalmanFilter.filter(rep, stationaryInitialization(init));

        double[] ref = KalmanFilterRefData.Wpi1Ar3MissingR.FORECAST;
        int nobs = fr.nobs;
        int nRef = ref.length;

        for (int t = 0; t < Math.min(nobs, nRef); t++) {
            if (Double.isNaN(ref[t])) continue;
            assertEquals(ref[t], fr.forecast(0, t), TOL_REF,
                    "AR3 missing forecast mismatch at t=" + t);
        }
    }

    @Test
    void testMultivariateMissingSmootherR() {
        ModelFixture rep = buildMultivariateMissingModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] refR = KalmanFilterRefData.MultivariateMissing.SMOOTHER_R;
        int kStates = 3;
        int nobs = fr.nobs;
        int nRef = refR.length / kStates;
        int skip = 5;

        for (int t = skip; t < Math.min(nobs, nRef); t++) {
            for (int i = 0; i < kStates; i++) {
                if (Double.isNaN(refR[t * kStates + i])) continue;
                assertEquals(refR[t * kStates + i], sr.scaledSmoothedEstimator[t * kStates + i], TOL_REF,
                        "smoother r mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testMultivariateMissingForecastVsR() {
        ModelFixture rep = buildMultivariateMissingModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        double[] ref = KalmanFilterRefData.MultivariateMissing.FORECAST;
        int kEndog = 3;
        int nobs = fr.nobs;
        int nRef = ref.length / kEndog;
        int skip = 5;

        for (int t = skip; t < Math.min(nobs, nRef); t++) {
            for (int i = 0; i < kEndog; i++) {
                if (Double.isNaN(ref[t * kEndog + i])) continue;
                assertEquals(ref[t * kEndog + i], fr.forecast(i, t), TOL_REF,
                        "forecast mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testMultivariateMissingSmoothedDisturbance() {
        ModelFixture rep = buildMultivariateMissingModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] refStateDist = KalmanFilterRefData.MultivariateMissing.SMOOTHED_STATE_DISTURBANCE;
        double[] refObsDist = KalmanFilterRefData.MultivariateMissing.SMOOTHED_OBS_DISTURBANCE;
        int kEndog = 3, kPosdef = 3;
        int nobs = fr.nobs;
        int nRefState = refStateDist.length / kPosdef;
        int nRefObs = refObsDist.length / kEndog;
        int skip = 5;

        for (int t = skip; t < Math.min(nobs, nRefState); t++) {
            for (int i = 0; i < kPosdef; i++) {
                if (Double.isNaN(refStateDist[t * kPosdef + i])) continue;
                assertEquals(refStateDist[t * kPosdef + i], sr.smoothedStateDisturbance(i, t), TOL_REF,
                        "smoothed state disturbance mismatch at t=" + t + " i=" + i);
            }
        }
        for (int t = skip; t < Math.min(nobs, nRefObs); t++) {
            for (int i = 0; i < kEndog; i++) {
                if (Double.isNaN(refObsDist[t * kEndog + i])) continue;
                assertEquals(refObsDist[t * kEndog + i], sr.smoothedObsDisturbance(i, t), TOL_REF,
                        "smoothed obs disturbance mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testMultivariateMissingFullTraceAgainstKFASCsv() {
        ModelFixture rep = buildMultivariateMissingModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        assertMultivariateSmoothingTraceAgainstCsv(fr, sr, "results_smoothing_R.csv", 3, true, false);
    }

    @Test
    void testMultivariateVARSmootherR() {
        ModelFixture rep = buildMultivariateVARModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] refR = KalmanFilterRefData.MultivariateVAR.SMOOTHER_R;
        int kStates = 3;
        int nobs = fr.nobs;
        int nRef = refR.length / kStates;
        int skip = 5;

        for (int t = skip; t < Math.min(nobs, nRef); t++) {
            for (int i = 0; i < kStates; i++) {
                if (Double.isNaN(refR[t * kStates + i])) continue;
                assertEquals(refR[t * kStates + i], sr.scaledSmoothedEstimator[t * kStates + i], TOL_REF,
                        "VAR smoother r mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testMultivariateVARForecastVsR() {
        ModelFixture rep = buildMultivariateVARModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        double[] ref = KalmanFilterRefData.MultivariateVAR.FORECAST;
        int kEndog = 3;
        int nobs = fr.nobs;
        int nRef = ref.length / kEndog;
        int skip = 5;

        for (int t = skip; t < Math.min(nobs, nRef); t++) {
            for (int i = 0; i < kEndog; i++) {
                if (Double.isNaN(ref[t * kEndog + i])) continue;
                assertEquals(ref[t * kEndog + i], fr.forecast(i, t), TOL_REF,
                        "VAR forecast mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testMultivariateVARSmoothedDisturbance() {
        ModelFixture rep = buildMultivariateVARModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        double[] refStateDist = KalmanFilterRefData.MultivariateVAR.SMOOTHED_STATE_DISTURBANCE;
        double[] refObsDist = KalmanFilterRefData.MultivariateVAR.SMOOTHED_OBS_DISTURBANCE;
        int kEndog = 3, kPosdef = 3;
        int nobs = fr.nobs;
        int nRefState = refStateDist.length / kPosdef;
        int nRefObs = refObsDist.length / kEndog;
        int skip = 5;

        for (int t = skip; t < Math.min(nobs, nRefState); t++) {
            for (int i = 0; i < kPosdef; i++) {
                if (Double.isNaN(refStateDist[t * kPosdef + i])) continue;
                assertEquals(refStateDist[t * kPosdef + i], sr.smoothedStateDisturbance(i, t), TOL_REF,
                        "VAR smoothed state disturbance mismatch at t=" + t + " i=" + i);
            }
        }
        for (int t = skip; t < Math.min(nobs, nRefObs); t++) {
            for (int i = 0; i < kEndog; i++) {
                if (Double.isNaN(refObsDist[t * kEndog + i])) continue;
                assertEquals(refObsDist[t * kEndog + i], sr.smoothedObsDisturbance(i, t), TOL_REF,
                        "VAR smoothed obs disturbance mismatch at t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void testMultivariateVARFullTraceAgainstKFASCsv() {
        ModelFixture rep = buildMultivariateVARModel();
        FilterResult fr = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        SmootherResult sr = KalmanSmoother.smooth(rep, fr);

        assertMultivariateSmoothingTraceAgainstCsv(fr, sr, "results_smoothing2_R.csv", 1, true, true);
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

        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse(), pool2);

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

        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, StateInitialization.approximateDiffuse(), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, StateInitialization.approximateDiffuse(), pool2);

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

        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, knownInitialization(init), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, knownInitialization(init), pool2);

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

        KalmanFilter.Pool pool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, knownInitialization(init), pool1);
        UnivariateFilter.Pool pool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, knownInitialization(init), pool2);

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

        KalmanFilter.Pool fPool1 = new KalmanFilter.Pool();
        FilterResult frConv = KalmanFilter.filter(rep, knownInitialization(init), fPool1);
        SmootherResult srConv = KalmanSmoother.smooth(rep, frConv);

        UnivariateFilter.Pool fPool2 = new UnivariateFilter.Pool();
        FilterResult frUni = UnivariateFilter.filter(rep, knownInitialization(init), fPool2);
        SmootherResult srUni = KalmanSmoother.smooth(rep, frUni);

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
