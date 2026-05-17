package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.AlignmentCase;
import com.curioloop.yum4j.fixtures.KalmanStatsmodelsFixtures.SurfacePoint;
import com.curioloop.yum4j.fixtures.StatsmodelsResources;
import com.curioloop.yum4j.ssm.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.ZKalmanFilterTest;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.ZSmootherResult;
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
    private static final String MACRODATA_MISSING_CASE_ID = "macrodata_diffuse_missing_conventional";
    private static final String MACRODATA_VAR_CASE_ID = "macrodata_var_diffuse_conventional";

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

    private static double sumRealSquareMatrix(double[] values, int t, int size) {
        return sumRealSquareMatrixAt(values, t * size * 2 * size, size);
    }

    private static double sumRealSquareMatrixAt(double[] values, int offset, int size) {
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

    private static void assertRealComplexClose(double expected, double[] values, int offset, double tol, String label) {
        double actual = values[offset];
        double scaledTol = scaledTolerance(expected, actual, tol);
        assertEquals(expected, actual, scaledTol, label + " Re");
        assertEquals(0.0, values[offset + 1], scaledTol, label + " Im");
    }

    private static void assertClose(double expected, double actual, double tol, String label) {
        assertEquals(expected, actual, scaledTolerance(expected, actual, tol), label);
    }

    private static double scaledTolerance(double expected, double actual, double tol) {
        return tol * Math.max(1.0, Math.max(Math.abs(expected), Math.abs(actual)));
    }

    private static void assertSuppressedEmpty(double[] values, String label) {
        assertEquals(0, values.length, label);
    }

    private static AlignmentCase alignmentCase(String id) {
        return KalmanStatsmodelsFixtures.cases().stream()
            .filter(fixture -> fixture.id().equals(id))
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException("Missing Kalman statsmodels fixture: " + id));
    }

    private static ModelFixture complexModel(AlignmentCase fixture) {
        KalmanStatsmodelsFixtures.Shape shape = fixture.shape();
        KalmanStatsmodelsFixtures.ModelSpec model = fixture.model();
        ModelFixture rep = ModelFixture.complex(shape.kEndog(), shape.kStates(), shape.kPosdef(), shape.nobs());
        double[] design = embedComplexMatrix(model.design(), shape.kEndog(), shape.kStates());
        double[] obsIntercept = embedComplexVector(defaulted(model.obsIntercept(), shape.kEndog()));
        double[] obsCov = embedComplexMatrix(model.obsCov(), shape.kEndog(), shape.kEndog());
        double[] transition = embedComplexMatrix(model.transition(), shape.kStates(), shape.kStates());
        double[] stateIntercept = embedComplexVector(defaulted(model.stateIntercept(), shape.kStates()));
        double[] selection = embedComplexMatrix(model.selection(), shape.kStates(), shape.kPosdef());
        double[] stateCov = embedComplexMatrix(model.stateCov(), shape.kPosdef(), shape.kPosdef());
        boolean[][] missing = model.missing();
        for (int t = 0; t < shape.nobs(); t++) {
            rep.setDesign(design, t);
            rep.setObsIntercept(obsIntercept, t);
            rep.setObsCov(obsCov, t);
            rep.setTransition(transition, t);
            rep.setStateIntercept(stateIntercept, t);
            rep.setSelection(selection, t);
            rep.setStateCov(stateCov, t);
            rep.setEndog(embedComplexVector(model.endog()[t]), t);
            if (missing != null) {
                rep.setMissing(missing[t], t);
            }
        }
        return rep;
    }

    private static double[] defaulted(double[] values, int length) {
        return values == null ? new double[length] : values;
    }

    private static void assertFilterSurface(AlignmentCase fixture, ZFilterResult result, String surface) {
        for (SurfacePoint point : KalmanStatsmodelsFixtures.surfaces(fixture)) {
            if (!point.surface().equals(surface)) {
                continue;
            }
            String label = point.caseId() + " " + surface + " t=" + point.t()
                + " row=" + point.row() + " col=" + point.col();
            if (surface.equals("log_likelihood_obs")) {
                assertClose(point.value(), result.logLikelihoodObs[result.logLikelihoodObsOffset(point.t())],
                    fixture.tolerance(), label);
            } else {
                assertRealComplexClose(point.value(), filterSurfaceArray(result, point),
                    filterSurfaceOffset(result, point), fixture.tolerance(), label);
            }
        }
    }

    private static double[] filterSurfaceArray(ZFilterResult result, SurfacePoint point) {
        return switch (point.surface()) {
            case "forecast" -> result.forecast;
            case "forecast_error" -> result.forecastError;
            case "forecast_error_cov" -> result.forecastErrorCov;
            case "predicted_state" -> result.predictedState;
            case "predicted_state_cov" -> result.predictedStateCov;
            case "filtered_state" -> result.filteredState;
            case "filtered_state_cov" -> result.filteredStateCov;
            case "standardized_forecast_error" -> result.standardizedForecastError;
            default -> throw new IllegalArgumentException("Unsupported filter surface: " + point.surface());
        };
    }

    private static int filterSurfaceOffset(ZFilterResult result, SurfacePoint point) {
        return switch (point.surface()) {
            case "forecast" -> result.forecastOffset(point.t()) + point.row() * 2;
            case "forecast_error" -> result.forecastErrorOffset(point.t()) + point.row() * 2;
            case "forecast_error_cov" -> result.forecastErrorCovOffset(point.t())
                + point.row() * 2 * result.kEndog + point.col() * 2;
            case "predicted_state" -> result.predictedStateOffset(point.t()) + point.row() * 2;
            case "predicted_state_cov" -> result.predictedStateCovOffset(point.t())
                + point.row() * 2 * result.kStates + point.col() * 2;
            case "filtered_state" -> result.filteredStateOffset(point.t()) + point.row() * 2;
            case "filtered_state_cov" -> result.filteredStateCovOffset(point.t())
                + point.row() * 2 * result.kStates + point.col() * 2;
            case "standardized_forecast_error" -> result.standardizedForecastErrorOffset(point.t()) + point.row() * 2;
            default -> throw new IllegalArgumentException("Unsupported filter surface: " + point.surface());
        };
    }

    private static void assertSmootherSurface(AlignmentCase fixture, ZSmootherResult result, String surface) {
        for (SurfacePoint point : KalmanStatsmodelsFixtures.smootherSurfaces(fixture)) {
            if (!point.surface().equals(surface)) {
                continue;
            }
            assertRealComplexClose(point.value(), smootherSurfaceArray(result, point),
                smootherSurfaceOffset(result, point), fixture.tolerance(),
                point.caseId() + " " + surface + " t=" + point.t()
                    + " row=" + point.row() + " col=" + point.col());
        }
    }

    private static double[] smootherSurfaceArray(ZSmootherResult result, SurfacePoint point) {
        return switch (point.surface()) {
            case "smoothed_state" -> result.smoothedState;
            case "smoothed_state_cov" -> result.smoothedStateCov;
            case "smoothed_measurement_disturbance" -> result.smoothedObsDisturbance;
            case "smoothed_state_disturbance" -> result.smoothedStateDisturbance;
            case "smoothed_measurement_disturbance_cov" -> result.smoothedObsDisturbanceCov;
            case "smoothed_state_disturbance_cov" -> result.smoothedStateDisturbanceCov;
            case "scaled_smoothed_estimator" -> result.scaledSmoothedEstimator;
            case "scaled_smoothed_estimator_cov" -> result.scaledSmoothedEstimatorCov;
            default -> throw new IllegalArgumentException("Unsupported smoother surface: " + point.surface());
        };
    }

    private static int smootherSurfaceOffset(ZSmootherResult result, SurfacePoint point) {
        return switch (point.surface()) {
            case "smoothed_state" -> result.smoothedStateOffset(point.t()) + point.row() * 2;
            case "smoothed_state_cov" -> result.smoothedStateCovOffset(point.t())
                + point.row() * 2 * result.kStates + point.col() * 2;
            case "smoothed_measurement_disturbance" -> result.smoothedObsDisturbanceOffset(point.t()) + point.row() * 2;
            case "smoothed_state_disturbance" -> result.smoothedStateDisturbanceOffset(point.t()) + point.row() * 2;
            case "smoothed_measurement_disturbance_cov" -> result.smoothedObsDisturbanceCovOffset(point.t())
                + point.row() * 2 * result.kEndog + point.col() * 2;
            case "smoothed_state_disturbance_cov" -> result.smoothedStateDisturbanceCovOffset(point.t())
                + point.row() * 2 * result.kPosdef + point.col() * 2;
            case "scaled_smoothed_estimator" -> result.scaledSmoothedEstimatorOffset(point.t()) + point.row() * 2;
            case "scaled_smoothed_estimator_cov" -> result.scaledSmoothedEstimatorCovOffset(point.t())
                + point.row() * 2 * result.kStates + point.col() * 2;
            default -> throw new IllegalArgumentException("Unsupported smoother surface: " + point.surface());
        };
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
        return complexModel(alignmentCase(MACRODATA_MISSING_CASE_ID));
    }

    private static ModelFixture buildRealMultivariateMissingModel() {
        return alignmentCase(MACRODATA_MISSING_CASE_ID).modelFixture();
    }

    private static ModelFixture buildZMultivariateVARModel() {
        return complexModel(alignmentCase(MACRODATA_VAR_CASE_ID));
    }

    @Test
    void testExactDiffuseLocalLevelFilterAgainstKFAS() {
        double[] y = {10.2394, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        double sigma2Y = 1.993;
        double sigma2Mu = 8.253;

        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLevel(y, sigma2Y, sigma2Mu);
        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
                assertEquals(sumPinf, sumRealSquareMatrixAt(fr.predictedDiffuseStateCov,
                                fr.predictedDiffuseStateCovOffset(t), rep.kStates), TOL_DIFFUSE,
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
                assertRealComplexEquals(finf, fr.forecastErrorDiffuseCov, fr.forecastErrorDiffuseCovOffset(t), TOL_DIFFUSE,
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
        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZSmootherResult sr = SmootherEngine.smoothComplex(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

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
        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
                assertEquals(sumPinf, sumRealSquareMatrixAt(fr.predictedDiffuseStateCov,
                                fr.predictedDiffuseStateCovOffset(t), rep.kStates), TOL_DIFFUSE,
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
                assertRealComplexEquals(finf, fr.forecastErrorDiffuseCov, fr.forecastErrorDiffuseCovOffset(t), TOL_DIFFUSE,
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

        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZSmootherResult sr = SmootherEngine.smoothComplex(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

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
                assertEquals(sumPinf, sumRealSquareMatrixAt(fr.predictedDiffuseStateCov,
                                fr.predictedDiffuseStateCovOffset(t), rep.kStates), TOL_DIFFUSE,
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
                    assertRealComplexEquals(finf, fr.forecastErrorDiffuseCov, fr.forecastErrorDiffuseCovOffset(t), TOL_DIFFUSE,
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
        AlignmentCase fixture = alignmentCase(MACRODATA_VAR_CASE_ID);
        ModelFixture rep = buildZMultivariateVARModel();
        SmootherOptions smootherOptions = fixture.smootherOptions();
        ZFilterResult fr = KalmanEngine.filterComplex(rep, fixture.initialState(rep), smootherOptions.requiredFilterOptions());
        ZSmootherResult sr = SmootherEngine.smoothComplex(rep, fr, smootherOptions);

        assertFilterSurface(fixture, fr, "forecast");
        assertSmootherSurface(fixture, sr, "smoothed_state");
        assertSmootherSurface(fixture, sr, "scaled_smoothed_estimator");
    }

    @Test
    void testComplexStateAutocovarianceMatchesRealEmbeddedModel() {
        AlignmentCase fixture = alignmentCase(MACRODATA_VAR_CASE_ID);
        ModelFixture realRep = fixture.modelFixture();
        ModelFixture complexRep = buildZMultivariateVARModel();
        SmootherOptions smootherOptions = SmootherOptions.conventional()
            .only(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);

        FilterResult realFr = KalmanEngine.filter(realRep, fixture.initialState(realRep),
            smootherOptions.requiredFilterOptions());
        ZFilterResult complexFr = KalmanEngine.filterComplex(complexRep, fixture.initialState(complexRep),
            smootherOptions.requiredFilterOptions());
        SmootherResult realSmoother = SmootherEngine.smooth(realRep, realFr, smootherOptions);
        ZSmootherResult complexSmoother = SmootherEngine.smoothComplex(complexRep, complexFr, smootherOptions);

        assertEquals(realRep.nobs * realRep.kStates * realRep.kStates,
            realSmoother.smoothedStateAutocovarianceLength());
        assertEquals(2 * complexRep.nobs * complexRep.kStates * complexRep.kStates,
            complexSmoother.smoothedStateAutocovarianceLength());
        for (int t = 0; t < realRep.nobs; t++) {
            for (int i = 0; i < realRep.kStates; i++) {
                for (int j = 0; j < realRep.kStates; j++) {
                    double expected = realSmoother.smoothedStateAutocovariance(i, j, t);
                    int complexOffset = complexSmoother.smoothedStateAutocovarianceOffset(t)
                        + i * 2 * complexRep.kStates + j * 2;
                    assertRealComplexClose(expected, complexSmoother.smoothedStateAutocovariance,
                        complexOffset, 1e-9, "smoothedStateAutocovariance t=" + t + " i=" + i + " j=" + j);
                }
            }
        }
    }

    @Test
    void testMultivariateMissingReferenceParity() {
        AlignmentCase fixture = alignmentCase(MACRODATA_MISSING_CASE_ID);
        ModelFixture rep = buildZMultivariateMissingModel();
        SmootherOptions smootherOptions = fixture.smootherOptions();
        ZFilterResult fr = KalmanEngine.filterComplex(rep, fixture.initialState(rep), smootherOptions.requiredFilterOptions());
        ZSmootherResult sr = SmootherEngine.smoothComplex(rep, fr, smootherOptions);

        assertFilterSurface(fixture, fr, "forecast");
        assertSmootherSurface(fixture, sr, "smoothed_state");
        assertSmootherSurface(fixture, sr, "scaled_smoothed_estimator");
    }

    @Test
    void testMultivariateMissingObsDisturbanceCovAgainstStatsmodelsFixture() {
        AlignmentCase fixture = alignmentCase(MACRODATA_MISSING_CASE_ID);
        ModelFixture rep = buildZMultivariateMissingModel();
        SmootherOptions smootherOptions = fixture.smootherOptions();
        ZFilterResult fr = KalmanEngine.filterComplex(rep, fixture.initialState(rep), smootherOptions.requiredFilterOptions());
        ZSmootherResult sr = SmootherEngine.smoothComplex(rep, fr, smootherOptions);

        assertSmootherSurface(fixture, sr, "smoothed_measurement_disturbance_cov");
    }

    @Test
    void testMultivariateMissingKalmanGainParity() {
        ModelFixture realRep = buildRealMultivariateMissingModel();
        FilterResult realFr = KalmanEngine.filter(realRep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        ModelFixture rep = buildZMultivariateMissingModel();
        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

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
    void testComplexDiffuseSmootherOptionsWithoutOutputs() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        double sigma2Y = 1.0;
        double sigma2Mu = 0.5;
        double sigma2Beta = 0.2;

        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertTrue(fr.nobsDiffuse > 0);

        ZSmootherResult full = SmootherEngine.smoothComplex(rep, fr, SmootherOptions.conventional());
        ZSmootherResult stateOnly = SmootherEngine.smoothComplex(rep, fr,
            SmootherOptions.conventional().only(SmootherOptions.Surface.STATE));

        assertStateOnlySmootherMatches(full, stateOnly, TOL_REF);
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuseEstimator, "scaledSmoothedDiffuseEstimator should be suppressed");
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuse1EstimatorCov, "scaledSmoothedDiffuse1EstimatorCov should be suppressed");
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuse2EstimatorCov, "scaledSmoothedDiffuse2EstimatorCov should be suppressed");
    }

    @Test
    void testComplexDiffuseBorrowedSmootherOptionsWithoutOutputs() {
        double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
        double sigma2Y = 1.0;
        double sigma2Mu = 0.5;
        double sigma2Beta = 0.2;

        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, sigma2Y, sigma2Mu, sigma2Beta);
        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        assertTrue(fr.nobsDiffuse > 0);

        SmootherEngine.Workspace workspace = SmootherEngine.workspace();
        ZSmootherResult full = SmootherEngine.smoothComplexBorrowedUnsafe(rep, fr, workspace, SmootherOptions.conventional());
        ZSmootherResult fullSnapshot = full.clone();
        long fullResultDoubles = workspace.retainedComplexSmootherResultDoubleCount();

        ZSmootherResult stateOnly = SmootherEngine.smoothComplexBorrowedUnsafe(rep, fr, workspace, SmootherOptions.conventional().only(SmootherOptions.Surface.STATE));

        assertTrue(full == stateOnly);
        assertStateOnlySmootherMatches(fullSnapshot, stateOnly, TOL_REF);
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuseEstimator, "scaledSmoothedDiffuseEstimator should be suppressed");
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuse1EstimatorCov, "scaledSmoothedDiffuse1EstimatorCov should be suppressed");
        assertSuppressedEmpty(stateOnly.scaledSmoothedDiffuse2EstimatorCov, "scaledSmoothedDiffuse2EstimatorCov should be suppressed");
        assertEquals(fullResultDoubles, workspace.retainedComplexSmootherResultDoubleCount());
    }

    @Test
        void testComplexSmootherMethodContractsMirrorRealPath() {
        ModelFixture rep = buildZMultivariateVARModel();
        InitialState init = InitialState.approximateDiffuse();
        ZFilterResult compact = KalmanEngine.filterComplex(
            rep, init, SmootherOptions.conventional().requiredFilterOptions());

        assertDoesNotThrow(() -> SmootherEngine.smoothComplex(rep, compact, SmootherOptions.conventional()));
        assertThrows(IllegalArgumentException.class,
            () -> SmootherEngine.smoothComplex(rep, compact, SmootherOptions.classical()));
        assertThrows(IllegalArgumentException.class,
            () -> SmootherEngine.smoothComplex(rep, compact, SmootherOptions.alternative()));
        assertThrows(IllegalArgumentException.class,
            () -> SmootherEngine.smoothComplex(rep, compact, SmootherOptions.univariate()));
        ZSmootherResult autocov = assertDoesNotThrow(() -> SmootherEngine.smoothComplex(rep, compact,
            SmootherOptions.conventional().only(SmootherOptions.Surface.STATE_AUTOCOVARIANCE)));
        assertEquals(2 * rep.nobs * rep.kStates * rep.kStates, autocov.smoothedStateAutocovarianceLength());

        assertDoesNotThrow(() -> SmootherEngine.smoothComplex(
            rep,
            KalmanEngine.filterComplex(rep, init, SmootherOptions.classical().requiredFilterOptions()),
            SmootherOptions.classical()));
        assertDoesNotThrow(() -> SmootherEngine.smoothComplex(
            rep,
            KalmanEngine.filterComplex(rep, init, SmootherOptions.alternative().requiredFilterOptions()),
            SmootherOptions.alternative()));
        }

        @Test
    void testComplexSmootherMethodVariantsAgree() {
        ModelFixture rep = buildZMultivariateVARModel();
        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());

        ZSmootherResult conventional = SmootherEngine.smoothComplex(rep, fr, SmootherOptions.conventional());
        ZSmootherResult classical = SmootherEngine.smoothComplex(rep, fr, SmootherOptions.classical());
        ZSmootherResult alternative = SmootherEngine.smoothComplex(rep, fr, SmootherOptions.alternative());

        assertSmootherOutputsEqual(conventional, classical, TOL_REF);
        assertSmootherOutputsEqual(conventional, alternative, TOL_REF);
    }
}