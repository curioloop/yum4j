package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.fixtures.StatsmodelsResources;
import com.curioloop.yum4j.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.kalman.filter.ZKalmanFilter;
import com.curioloop.yum4j.kalman.filter.FilterResult;
import com.curioloop.yum4j.kalman.filter.KalmanFilter;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.model.ModelFixture;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
import com.curioloop.yum4j.kalman.smooth.ZKalmanSmoother;
import com.curioloop.yum4j.kalman.smooth.ZSmootherResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class ZKalmanStatsmodelsTest {

    private static final double TOL_REF = 1e-4;
    private static final double TOL_DIFFUSE = 1e-7;
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

    private static double[][] readMacrodata() {
        return StatsmodelsResources.readMacrodata();
    }

    private static double[] embedComplexVector(double... values) {
        double[] complex = new double[values.length * 2];
        for (int i = 0; i < values.length; i++) {
            complex[i * 2] = values[i];
        }
        return complex;
    }

    private static double[] embedComplexMatrix(double[] values, int rows, int cols) {
        double[] complex = new double[rows * 2 * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                complex[i * 2 * cols + j * 2] = values[i * cols + j];
            }
        }
        return complex;
    }

    private static int vectorOffset(int t, int i, int width) {
        return t * 2 * width + i * 2;
    }

    private static int matrixOffset(int t, int row, int col, int size) {
        return t * size * 2 * size + row * 2 * size + col * 2;
    }

    private static double sumRealSquareMatrix(double[] values, int t, int size) {
        int offset = t * size * 2 * size;
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                sum += values[offset + i * 2 * size + j * 2];
            }
        }
        return sum;
    }

    private static void assertRealComplexEquals(double expected, double[] values, int offset, double tol, String label) {
        assertEquals(expected, values[offset], tol, label + " Re");
        assertEquals(0.0, values[offset + 1], tol, label + " Im");
    }

    private static void assertSuppressedEmpty(double[] values, String label) {
        assertEquals(0, values.length, label);
    }

    private static void assertStateOnlySmootherMatches(ZSmootherResult full, ZSmootherResult stateOnly, double tol) {
        assertEquals(full.kStates, stateOnly.kStates);
        assertEquals(full.kEndog, stateOnly.kEndog);
        assertEquals(full.kPosdef, stateOnly.kPosdef);
        assertEquals(full.nobs, stateOnly.nobs);

        for (int i = 0; i < full.smoothedState.length; i++) {
            assertEquals(full.smoothedState[i], stateOnly.smoothedState[i], tol,
                    "smoothedState mismatch at flat index=" + i);
        }

        assertSuppressedEmpty(stateOnly.smoothedStateCov, "smoothedStateCov should be suppressed");
        assertSuppressedEmpty(stateOnly.smoothedObsDisturbance, "smoothedObsDisturbance should be suppressed");
        assertSuppressedEmpty(stateOnly.smoothedStateDisturbance, "smoothedStateDisturbance should be suppressed");
        assertSuppressedEmpty(stateOnly.smoothedObsDisturbanceCov, "smoothedObsDisturbanceCov should be suppressed");
        assertSuppressedEmpty(stateOnly.smoothedStateDisturbanceCov, "smoothedStateDisturbanceCov should be suppressed");
    }

    private static void assertSmootherOutputsEqual(ZSmootherResult lhs, ZSmootherResult rhs, double tol) {
        assertEquals(lhs.smoothedState.length, rhs.smoothedState.length);
        for (int i = 0; i < lhs.smoothedState.length; i++) {
            assertEquals(lhs.smoothedState[i], rhs.smoothedState[i], tol, "smoothedState flat mismatch at " + i);
        }
        for (int i = 0; i < lhs.smoothedStateCov.length; i++) {
            assertEquals(lhs.smoothedStateCov[i], rhs.smoothedStateCov[i], tol, "smoothedStateCov flat mismatch at " + i);
        }
        for (int i = 0; i < lhs.smoothedObsDisturbance.length; i++) {
            assertEquals(lhs.smoothedObsDisturbance[i], rhs.smoothedObsDisturbance[i], tol,
                    "smoothedObsDisturbance flat mismatch at " + i);
        }
        for (int i = 0; i < lhs.smoothedStateDisturbance.length; i++) {
            assertEquals(lhs.smoothedStateDisturbance[i], rhs.smoothedStateDisturbance[i], tol,
                    "smoothedStateDisturbance flat mismatch at " + i);
        }
        for (int i = 0; i < lhs.smoothedObsDisturbanceCov.length; i++) {
            assertEquals(lhs.smoothedObsDisturbanceCov[i], rhs.smoothedObsDisturbanceCov[i], tol,
                    "smoothedObsDisturbanceCov flat mismatch at " + i);
        }
        for (int i = 0; i < lhs.smoothedStateDisturbanceCov.length; i++) {
            assertEquals(lhs.smoothedStateDisturbanceCov[i], rhs.smoothedStateDisturbanceCov[i], tol,
                    "smoothedStateDisturbanceCov flat mismatch at " + i);
        }
    }

    private static ModelFixture buildZMultivariateMissingModel() {
        double[][] raw = readMacrodata();
        int nobs = 202;
        int kEndog = 3, kStates = 3, kPosdef = 3;

        double[] realgdp = new double[nobs];
        double[] realcons = new double[nobs];
        double[] realinv = new double[nobs];
        for (int t = 0; t < nobs; t++) {
            realgdp[t] = raw[t + 1][2] - raw[t][2];
            realcons[t] = raw[t + 1][3] - raw[t][3];
            realinv[t] = raw[t + 1][4] - raw[t][4];
        }

        ModelFixture rep = ModelFixture.complex(kEndog, kStates, kPosdef, nobs);

        double[] Z = embedComplexMatrix(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, kEndog, kStates);
        double[] H = embedComplexMatrix(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, kEndog, kEndog);
        double[] T = embedComplexMatrix(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, kStates, kStates);
        double[] R = embedComplexMatrix(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, kStates, kPosdef);
        double[] Q = embedComplexMatrix(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, kPosdef, kPosdef);

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(embedComplexVector(realgdp[t], realcons[t], realinv[t]), t);
        }

        boolean[][] miss = new boolean[nobs][kEndog];
        for (int t = 0; t < 50; t++) miss[t][0] = true;
        for (int t = 19; t < 70; t++) miss[t][1] = true;
        for (int t = 39; t < 90; t++) miss[t][2] = true;
        for (int t = 119; t < Math.min(130, nobs); t++) {
            miss[t][0] = true;
            miss[t][2] = true;
        }
        for (int t = 0; t < nobs; t++) {
            rep.setMissing(miss[t], t);
        }

        return rep;
    }

    private static ModelFixture buildRealMultivariateMissingModel() {
        return KalmanModelFixtures.statsmodelsMultivariateMissingModel();
    }

    private static ModelFixture buildZMultivariateVARModel() {
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

        ModelFixture rep = ModelFixture.complex(kEndog, kStates, kPosdef, nobs);

        double[] Z = embedComplexMatrix(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, kEndog, kStates);
        double[] H = embedComplexMatrix(new double[]{
                0.0000640649, 0, 0,
                0, 0.0000572802, 0,
                0, 0, 0.0017088585
        }, kEndog, kEndog);
        double[] T = embedComplexMatrix(new double[]{
                -0.1119908792, 0.8441841604, 0.0238725303,
                0.2629347724, 0.4996718412, -0.0173023305,
                -3.2192369082, 4.1536028244, 0.4514379215
        }, kStates, kStates);
        double[] R = embedComplexMatrix(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, kStates, kPosdef);
        double[] Q = embedComplexMatrix(new double[]{
                0.0000640649, 0.0000388496, 0.0002148769,
                0.0000388496, 0.0000572802, 0.000001555,
                0.0002148769, 0.000001555, 0.0017088585
        }, kPosdef, kPosdef);

        for (int t = 0; t < nobs; t++) {
            rep.setDesign(Z, t);
            rep.setObsCov(H, t);
            rep.setTransition(T, t);
            rep.setSelection(R, t);
            rep.setStateCov(Q, t);
            rep.setEndog(embedComplexVector(realgdp[t], realcons[t], realinv[t]), t);
        }

        return rep;
    }

    @Test
    void testExactDiffuseLocalLevelFilterAgainstKFAS() {
        double[] y = {10.2394, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;

        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLevel(y, sigma2Y, sigma2Mu);
        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.diffuse());

        List<String[]> csv = readExactDiffuseReferenceCsv("results_exact_initial_local_level_R.csv");
        String[] header = csv.get(0);
        int a1Col = csvColumnIndex(header, "a_1");
        int sumPCol = csvColumnIndex(header, "sumP");
        int sumPinfCol = csvColumnIndex(header, "sumPinf");
        int vCol = csvColumnIndex(header, "v_1");
        int fCol = csvColumnIndex(header, "F_1");
        int finfCol = csvColumnIndex(header, "Finf_1");
        int attCol = csvColumnIndex(header, "att_1");

        assertEquals(fr.nobs + 1, csv.size() - 1);
        for (int t = 0; t <= fr.nobs; t++) {
            String[] row = csv.get(t + 1);
            double a1 = csvValue(row, a1Col);
            if (!Double.isNaN(a1)) {
                assertRealComplexEquals(a1, fr.predictedState, vectorOffset(t, 0, rep.kStates), TOL_DIFFUSE,
                        "predictedState at t=" + t);
            }
            double sumP = csvValue(row, sumPCol);
            if (!Double.isNaN(sumP)) {
                assertEquals(sumP, sumRealSquareMatrix(fr.predictedStateCov, t, rep.kStates), TOL_DIFFUSE,
                        "predictedStateCov sum at t=" + t);
            }
            double sumPinf = csvValue(row, sumPinfCol);
            if (!Double.isNaN(sumPinf)) {
                assertEquals(sumPinf, sumRealSquareMatrix(fr.predictedDiffuseStateCov, t, rep.kStates), TOL_DIFFUSE,
                        "predictedDiffuseStateCov sum at t=" + t);
            }
            if (t == fr.nobs) {
                continue;
            }
            double v = csvValue(row, vCol);
            if (!Double.isNaN(v)) {
                assertRealComplexEquals(v, fr.forecastError, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                        "forecastError at t=" + t);
            }
            double f = csvValue(row, fCol);
            if (!Double.isNaN(f)) {
                assertRealComplexEquals(f, fr.forecastErrorCov, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                        "forecastErrorCov at t=" + t);
            }
            double finf = csvValue(row, finfCol);
            if (!Double.isNaN(finf)) {
                assertRealComplexEquals(finf, fr.forecastErrorDiffuseCov, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                        "forecastErrorDiffuseCov at t=" + t);
            }
            double att = csvValue(row, attCol);
            if (!Double.isNaN(att)) {
                assertRealComplexEquals(att, fr.filteredState, vectorOffset(t, 0, rep.kStates), TOL_DIFFUSE,
                        "filteredState at t=" + t);
            }
        }
        assertEquals(1, fr.nobsDiffuse);
    }

    @Test
    void testExactDiffuseLocalLevelSmootherAgainstKFAS() {
        double[] y = {10.2394, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;

        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLevel(y, sigma2Y, sigma2Mu);
        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.diffuse());
        ZSmootherResult sr = ZKalmanSmoother.smooth(rep, fr);

        List<String[]> csv = readExactDiffuseReferenceCsv("results_exact_initial_local_level_R.csv");
        String[] header = csv.get(0);
        int alphaCol = csvColumnIndex(header, "alphahat_1");
        int sumVCol = csvColumnIndex(header, "sumV");
        int epsCol = csvColumnIndex(header, "epshat_1");
        int vepsCol = csvColumnIndex(header, "Veps_1");

        for (int t = 0; t < fr.nobs; t++) {
            String[] row = csv.get(t + 1);
            assertRealComplexEquals(csvValue(row, alphaCol), sr.smoothedState, vectorOffset(t, 0, rep.kStates), TOL_DIFFUSE,
                    "smoothedState at t=" + t);
            assertEquals(csvValue(row, sumVCol), sumRealSquareMatrix(sr.smoothedStateCov, t, rep.kStates), TOL_DIFFUSE,
                    "smoothedStateCov sum at t=" + t);
            assertRealComplexEquals(csvValue(row, epsCol), sr.smoothedObsDisturbance, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                    "smoothedObsDisturbance at t=" + t);
            assertEquals(csvValue(row, vepsCol), sumRealSquareMatrix(sr.smoothedObsDisturbanceCov, t, rep.kEndog), TOL_DIFFUSE,
                    "smoothedObsDisturbanceCov sum at t=" + t);
        }
    }

    @Test
    void testExactDiffuseLocalLinearTrendFilterAgainstKFAS() {
        double[] y = {10.2394, 4.2039, 6.123123, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;
        double sigma2Beta = 2.334;

        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.diffuse());

        List<String[]> csv = readExactDiffuseReferenceCsv("results_exact_initial_local_linear_trend_R.csv");
        String[] header = csv.get(0);
        int a1Col = csvColumnIndex(header, "a_1");
        int a2Col = csvColumnIndex(header, "a_2");
        int sumPCol = csvColumnIndex(header, "sumP");
        int sumPinfCol = csvColumnIndex(header, "sumPinf");
        int vCol = csvColumnIndex(header, "v_1");
        int fCol = csvColumnIndex(header, "F_1");
        int finfCol = csvColumnIndex(header, "Finf_1");
        int att1Col = csvColumnIndex(header, "att_1");
        int att2Col = csvColumnIndex(header, "att_2");

        assertEquals(fr.nobs + 1, csv.size() - 1);
        for (int t = 0; t <= fr.nobs; t++) {
            String[] row = csv.get(t + 1);
            double a1 = csvValue(row, a1Col);
            double a2 = csvValue(row, a2Col);
            if (!Double.isNaN(a1)) {
                assertRealComplexEquals(a1, fr.predictedState, vectorOffset(t, 0, rep.kStates), TOL_DIFFUSE,
                        "predictedState[0] at t=" + t);
            }
            if (!Double.isNaN(a2)) {
                assertRealComplexEquals(a2, fr.predictedState, vectorOffset(t, 1, rep.kStates), TOL_DIFFUSE,
                        "predictedState[1] at t=" + t);
            }
            double sumP = csvValue(row, sumPCol);
            if (!Double.isNaN(sumP)) {
                assertEquals(sumP, sumRealSquareMatrix(fr.predictedStateCov, t, rep.kStates), TOL_DIFFUSE,
                        "predictedStateCov sum at t=" + t);
            }
            double sumPinf = csvValue(row, sumPinfCol);
            if (!Double.isNaN(sumPinf)) {
                assertEquals(sumPinf, sumRealSquareMatrix(fr.predictedDiffuseStateCov, t, rep.kStates), TOL_DIFFUSE,
                        "predictedDiffuseStateCov sum at t=" + t);
            }
            if (t == fr.nobs) {
                continue;
            }
            double v = csvValue(row, vCol);
            if (!Double.isNaN(v)) {
                assertRealComplexEquals(v, fr.forecastError, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                        "forecastError at t=" + t);
            }
            double f = csvValue(row, fCol);
            if (!Double.isNaN(f)) {
                assertRealComplexEquals(f, fr.forecastErrorCov, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                        "forecastErrorCov at t=" + t);
            }
            double finf = csvValue(row, finfCol);
            if (!Double.isNaN(finf)) {
                assertRealComplexEquals(finf, fr.forecastErrorDiffuseCov, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                        "forecastErrorDiffuseCov at t=" + t);
            }
            double att1 = csvValue(row, att1Col);
            double att2 = csvValue(row, att2Col);
            if (!Double.isNaN(att1)) {
                assertRealComplexEquals(att1, fr.filteredState, vectorOffset(t, 0, rep.kStates), TOL_DIFFUSE,
                        "filteredState[0] at t=" + t);
            }
            if (!Double.isNaN(att2)) {
                assertRealComplexEquals(att2, fr.filteredState, vectorOffset(t, 1, rep.kStates), TOL_DIFFUSE,
                        "filteredState[1] at t=" + t);
            }
        }
        assertEquals(2, fr.nobsDiffuse);
    }

    @Test
    void testExactDiffuseLocalLinearTrendMissingFilterAndSmootherAgainstKFAS() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;
        double sigma2Beta = 2.334;

        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        rep.setMissing(new boolean[]{true}, 1);

        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.diffuse());
        ZSmootherResult sr = ZKalmanSmoother.smooth(rep, fr);

        List<String[]> csv = readExactDiffuseReferenceCsv("results_exact_initial_local_linear_trend_missing_R.csv");
        String[] header = csv.get(0);
        int a1Col = csvColumnIndex(header, "a_1");
        int a2Col = csvColumnIndex(header, "a_2");
        int sumPCol = csvColumnIndex(header, "sumP");
        int sumPinfCol = csvColumnIndex(header, "sumPinf");
        int att1Col = csvColumnIndex(header, "att_1");
        int att2Col = csvColumnIndex(header, "att_2");
        int vCol = csvColumnIndex(header, "v_1");
        int fCol = csvColumnIndex(header, "F_1");
        int finfCol = csvColumnIndex(header, "Finf_1");
        int alpha1Col = csvColumnIndex(header, "alphahat_1");
        int alpha2Col = csvColumnIndex(header, "alphahat_2");
        int sumVCol = csvColumnIndex(header, "sumV");
        int epsCol = csvColumnIndex(header, "epshat_1");
        int vepsCol = csvColumnIndex(header, "Veps_1");

        assertEquals(fr.nobs + 1, csv.size() - 1);
        for (int t = 0; t <= fr.nobs; t++) {
            String[] row = csv.get(t + 1);
            double a1 = csvValue(row, a1Col);
            if (!Double.isNaN(a1)) {
                assertRealComplexEquals(a1, fr.predictedState, vectorOffset(t, 0, rep.kStates), TOL_DIFFUSE,
                        "predictedState[0] at t=" + t);
            }
            double a2 = csvValue(row, a2Col);
            if (!Double.isNaN(a2)) {
                assertRealComplexEquals(a2, fr.predictedState, vectorOffset(t, 1, rep.kStates), TOL_DIFFUSE,
                        "predictedState[1] at t=" + t);
            }
            double sumP = csvValue(row, sumPCol);
            if (!Double.isNaN(sumP)) {
                assertEquals(sumP, sumRealSquareMatrix(fr.predictedStateCov, t, rep.kStates), TOL_DIFFUSE,
                        "predictedStateCov sum at t=" + t);
            }
            double sumPinf = csvValue(row, sumPinfCol);
            if (!Double.isNaN(sumPinf)) {
                assertEquals(sumPinf, sumRealSquareMatrix(fr.predictedDiffuseStateCov, t, rep.kStates), TOL_DIFFUSE,
                        "predictedDiffuseStateCov sum at t=" + t);
            }
            if (t == fr.nobs) {
                continue;
            }
            if (t != 1) {
                double v = csvValue(row, vCol);
                if (!Double.isNaN(v)) {
                    assertRealComplexEquals(v, fr.forecastError, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                            "forecastError at t=" + t);
                }
                double f = csvValue(row, fCol);
                if (!Double.isNaN(f)) {
                    assertRealComplexEquals(f, fr.forecastErrorCov, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                            "forecastErrorCov at t=" + t);
                }
                double finf = csvValue(row, finfCol);
                if (!Double.isNaN(finf)) {
                    assertRealComplexEquals(finf, fr.forecastErrorDiffuseCov, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                            "forecastErrorDiffuseCov at t=" + t);
                }
            }
            double att1 = csvValue(row, att1Col);
            if (!Double.isNaN(att1)) {
                assertRealComplexEquals(att1, fr.filteredState, vectorOffset(t, 0, rep.kStates), TOL_DIFFUSE,
                        "filteredState[0] at t=" + t);
            }
            double att2 = csvValue(row, att2Col);
            if (!Double.isNaN(att2)) {
                assertRealComplexEquals(att2, fr.filteredState, vectorOffset(t, 1, rep.kStates), TOL_DIFFUSE,
                        "filteredState[1] at t=" + t);
            }

            double alpha1 = csvValue(row, alpha1Col);
            if (!Double.isNaN(alpha1)) {
                assertRealComplexEquals(alpha1, sr.smoothedState, vectorOffset(t, 0, rep.kStates), TOL_DIFFUSE,
                        "smoothedState[0] at t=" + t);
            }
            double alpha2 = csvValue(row, alpha2Col);
            if (!Double.isNaN(alpha2)) {
                assertRealComplexEquals(alpha2, sr.smoothedState, vectorOffset(t, 1, rep.kStates), TOL_DIFFUSE,
                        "smoothedState[1] at t=" + t);
            }
            double sumV = csvValue(row, sumVCol);
            if (!Double.isNaN(sumV)) {
                assertEquals(sumV, sumRealSquareMatrix(sr.smoothedStateCov, t, rep.kStates), TOL_DIFFUSE,
                        "smoothedStateCov sum at t=" + t);
            }
            double eps = csvValue(row, epsCol);
            if (!Double.isNaN(eps)) {
                assertRealComplexEquals(eps, sr.smoothedObsDisturbance, vectorOffset(t, 0, rep.kEndog), TOL_DIFFUSE,
                        "smoothedObsDisturbance at t=" + t);
            }
            double veps = csvValue(row, vepsCol);
            if (!Double.isNaN(veps)) {
                assertEquals(veps, sumRealSquareMatrix(sr.smoothedObsDisturbanceCov, t, rep.kEndog), TOL_DIFFUSE,
                        "smoothedObsDisturbanceCov sum at t=" + t);
            }
        }
        assertEquals(3, fr.nobsDiffuse);
    }

    @Test
    void testMultivariateVARReferenceParity() {
        ModelFixture rep = buildZMultivariateVARModel();
        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        ZSmootherResult sr = ZKalmanSmoother.smooth(rep, fr);

        double[] forecastRef = KalmanFilterRefData.MultivariateVAR.FORECAST;
        double[] smoothedStateRef = KalmanFilterRefData.MultivariateVAR.SMOOTHED_STATE;
        double[] smootherRRef = KalmanFilterRefData.MultivariateVAR.SMOOTHER_R;
        int skip = 5;

        for (int t = skip; t < rep.nobs; t++) {
            for (int i = 0; i < rep.kEndog; i++) {
                double forecast = forecastRef[t * rep.kEndog + i];
                if (!Double.isNaN(forecast)) {
                    assertRealComplexEquals(forecast, fr.forecast, vectorOffset(t, i, rep.kEndog), TOL_REF,
                            "forecast[" + i + "] at t=" + t);
                }
            }
        }
        for (int t = 1; t < rep.nobs; t++) {
            for (int i = 0; i < rep.kStates; i++) {
                double smoothed = smoothedStateRef[t * rep.kStates + i];
                if (!Double.isNaN(smoothed)) {
                    assertRealComplexEquals(smoothed, sr.smoothedState, vectorOffset(t, i, rep.kStates), TOL_REF,
                            "smoothedState[" + i + "] at t=" + t);
                }
            }
        }
        for (int t = skip; t < rep.nobs; t++) {
            for (int i = 0; i < rep.kStates; i++) {
                double r = smootherRRef[t * rep.kStates + i];
                if (!Double.isNaN(r)) {
                    assertRealComplexEquals(r, sr.scaledSmoothedEstimator, vectorOffset(t, i, rep.kStates), TOL_REF,
                            "scaledSmoothedEstimator[" + i + "] at t=" + t);
                }
            }
        }
    }

    @Test
    void testMultivariateMissingReferenceParity() {
        ModelFixture rep = buildZMultivariateMissingModel();
        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        ZSmootherResult sr = ZKalmanSmoother.smooth(rep, fr);

        double[] forecastRef = KalmanFilterRefData.MultivariateMissing.FORECAST;
        double[] smoothedStateRef = KalmanFilterRefData.MultivariateMissing.SMOOTHED_STATE;
        double[] smootherRRef = KalmanFilterRefData.MultivariateMissing.SMOOTHER_R;
        int skip = 5;

        for (int t = skip; t < rep.nobs; t++) {
            for (int i = 0; i < rep.kEndog; i++) {
                double forecast = forecastRef[t * rep.kEndog + i];
                if (!Double.isNaN(forecast)) {
                    assertRealComplexEquals(forecast, fr.forecast, vectorOffset(t, i, rep.kEndog), TOL_REF,
                            "forecast[" + i + "] at t=" + t);
                }
            }
        }
        for (int t = 1; t < rep.nobs; t++) {
            for (int i = 0; i < rep.kStates; i++) {
                double smoothed = smoothedStateRef[t * rep.kStates + i];
                if (!Double.isNaN(smoothed)) {
                    assertRealComplexEquals(smoothed, sr.smoothedState, vectorOffset(t, i, rep.kStates), TOL_REF,
                            "smoothedState[" + i + "] at t=" + t);
                }
            }
        }
        for (int t = skip; t < rep.nobs; t++) {
            for (int i = 0; i < rep.kStates; i++) {
                double r = smootherRRef[t * rep.kStates + i];
                if (!Double.isNaN(r)) {
                    assertRealComplexEquals(r, sr.scaledSmoothedEstimator, vectorOffset(t, i, rep.kStates), TOL_REF,
                            "scaledSmoothedEstimator[" + i + "] at t=" + t);
                }
            }
        }
    }

    @Test
    void testMultivariateMissingObsDisturbanceCovAgainstKFASCsv() {
        ModelFixture rep = buildZMultivariateMissingModel();
        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.approximateDiffuse());
        ZSmootherResult sr = ZKalmanSmoother.smooth(rep, fr);

        List<String[]> csv = readExactDiffuseReferenceCsv("results_smoothing_R.csv");
        String[] header = csv.get(0);
        int[] vepsCols = {
                csvColumnIndex(header, "Veps1"),
                csvColumnIndex(header, "Veps2"),
                csvColumnIndex(header, "Veps3")
        };

        assertEquals(fr.nobs + 1, csv.size(), "reference row count mismatch");
        for (int t = 0; t < fr.nobs; t++) {
            String[] row = csv.get(t + 1);
            for (int i = 0; i < rep.kEndog; i++) {
                double expectedVeps = csvValue(row, vepsCols[i]);
                if (!Double.isNaN(expectedVeps)) {
                    assertRealComplexEquals(expectedVeps,
                            sr.smoothedObsDisturbanceCov,
                            matrixOffset(t, i, i, rep.kEndog),
                            TOL_REF,
                            "smoothedObsDisturbanceCov[" + i + "," + i + "] at t=" + t);
                }
            }
        }
    }

    @Test
    void testMultivariateMissingKalmanGainParity() {
        ModelFixture realRep = buildRealMultivariateMissingModel();
        FilterResult realFr = KalmanFilter.filter(realRep, StateInitialization.approximateDiffuse());

        ModelFixture rep = buildZMultivariateMissingModel();
        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        double tol = 1e-10;
        assertEquals(realFr.perObsKalmanGain, fr.perObsKalmanGain);

        for (int t = 0; t < rep.nobs; t++) {
            int realKgOff = realFr.kalmanGainOffset(t);
            int zKgOff = fr.kalmanGainOffset(t);
            for (int i = 0; i < rep.kStates; i++) {
                for (int j = 0; j < rep.kEndog; j++) {
                    int realIdx = realKgOff + i * rep.kEndog + j;
                    int zIdx = zKgOff + i * 2 * rep.kEndog + j * 2;
                    assertEquals(realFr.kalmanGain[realIdx], fr.kalmanGain[zIdx], tol,
                            "kalmanGain[" + i + "," + j + "] at t=" + t + " Re");
                    assertEquals(0.0, fr.kalmanGain[zIdx + 1], tol,
                            "kalmanGain[" + i + "," + j + "] at t=" + t + " Im");
                }
            }
        }
    }

    @Test
    void testComplexDiffuseSmootherSpecWithoutOutputs() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        double sigma2Y = 1.0;
        double sigma2Mu = 0.5;
        double sigma2Beta = 0.2;

        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.diffuse());
        assertTrue(fr.nobsDiffuse > 0);

        ZSmootherResult full = ZKalmanSmoother.smooth(rep, fr, null, SmootherSpec.conventional());
        ZSmootherResult stateOnly = ZKalmanSmoother.smooth(rep, fr, null,
            SmootherSpec.conventional().stateOnly());

        assertStateOnlySmootherMatches(full, stateOnly, TOL_REF);
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuseEstimator, "scaledSmoothedDiffuseEstimator should be suppressed");
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuse1EstimatorCov, "scaledSmoothedDiffuse1EstimatorCov should be suppressed");
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuse2EstimatorCov, "scaledSmoothedDiffuse2EstimatorCov should be suppressed");
    }

    @Test
    void testComplexDiffuseBorrowedSmootherSpecWithoutOutputs() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        double sigma2Y = 1.0;
        double sigma2Mu = 0.5;
        double sigma2Beta = 0.2;

        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.diffuse());
        assertTrue(fr.nobsDiffuse > 0);

        ZKalmanSmoother.Pool pool = new ZKalmanSmoother.Pool();
        ZSmootherResult full = ZKalmanSmoother.smooth(rep, fr, pool, SmootherSpec.conventional());
        ZSmootherResult fullSnapshot = full.clone();
        long fullResultDoubles = pool.retainedResultDoubleCount();

        ZSmootherResult stateOnly = ZKalmanSmoother.smooth(rep, fr, pool, SmootherSpec.conventional().stateOnly());

        assertTrue(full == stateOnly);
        assertStateOnlySmootherMatches(fullSnapshot, stateOnly, TOL_REF);
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuseEstimator, "scaledSmoothedDiffuseEstimator should be suppressed");
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuse1EstimatorCov, "scaledSmoothedDiffuse1EstimatorCov should be suppressed");
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuse2EstimatorCov, "scaledSmoothedDiffuse2EstimatorCov should be suppressed");
        assertTrue(pool.retainedResultDoubleCount() < fullResultDoubles);
    }

    @Test
        void testComplexSmootherMethodContractsMirrorRealPath() {
        ModelFixture rep = buildZMultivariateVARModel();
        StateInitialization init = StateInitialization.approximateDiffuse();
        ZFilterResult compact = ZKalmanFilter.filter(
            rep, init, null, SmootherSpec.conventional().requiredFilterSpec());

        assertDoesNotThrow(() -> ZKalmanSmoother.smooth(rep, compact, null, SmootherSpec.conventional()));
        assertThrows(IllegalArgumentException.class,
            () -> ZKalmanSmoother.smooth(rep, compact, null, SmootherSpec.classical()));
        assertThrows(IllegalArgumentException.class,
            () -> ZKalmanSmoother.smooth(rep, compact, null, SmootherSpec.alternative()));

        assertDoesNotThrow(() -> ZKalmanSmoother.smooth(
            rep,
            ZKalmanFilter.filter(rep, init, null, SmootherSpec.classical().requiredFilterSpec()),
            null,
            SmootherSpec.classical()));
        assertDoesNotThrow(() -> ZKalmanSmoother.smooth(
            rep,
            ZKalmanFilter.filter(rep, init, null, SmootherSpec.alternative().requiredFilterSpec()),
            null,
            SmootherSpec.alternative()));
        }

        @Test
    void testComplexSmootherMethodVariantsAgree() {
        ModelFixture rep = buildZMultivariateVARModel();
        ZFilterResult fr = ZKalmanFilter.filter(rep, StateInitialization.approximateDiffuse());

        ZSmootherResult conventional = ZKalmanSmoother.smooth(rep, fr, null, SmootherSpec.conventional());
        ZSmootherResult classical = ZKalmanSmoother.smooth(rep, fr, null, SmootherSpec.classical());
        ZSmootherResult alternative = ZKalmanSmoother.smooth(rep, fr, null, SmootherSpec.alternative());

        assertSmootherOutputsEqual(conventional, classical, TOL_REF);
        assertSmootherOutputsEqual(conventional, alternative, TOL_REF);
    }
}