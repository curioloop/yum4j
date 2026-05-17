package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import com.curioloop.yum4j.ssm.kalman.filter.FilterMethod;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.ForecastErrorStrategy;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherMethod;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KalmanEngineTest {

    private static final double TOL = 1e-12;
    private static final double CHANDRASEKHAR_BREADTH_TOL = 1e-10;
    private static final double NON_DIAGONAL_COLLAPSED_LOG_DET_FACTOR = Math.log(0.7 * 1.1 * 0.8);

    private static ModelFixture buildScalarModel() {
        ModelFixture model = new ModelFixture(1, 1, 1, 4);
        for (int t = 0; t < model.nobs; t++) {
            model.setDesign(new double[]{1.0}, t);
            model.setObsIntercept(new double[]{0.0}, t);
            model.setObsCov(new double[]{1.0}, t);
            model.setTransition(new double[]{0.8}, t);
            model.setStateIntercept(new double[]{0.1}, t);
            model.setSelection(new double[]{1.0}, t);
            model.setStateCov(new double[]{0.25}, t);
            model.setEndog(new double[]{1.0 + 0.2 * t}, t);
        }
        return model;
    }

    private static ModelFixture buildWeakDiffuseSignalModel() {
        ModelFixture model = new ModelFixture(1, 1, 1, 4);
        for (int t = 0; t < model.nobs; t++) {
            model.setDesign(new double[]{1e-4}, t);
            model.setObsIntercept(new double[]{0.0}, t);
            model.setObsCov(new double[]{1.0}, t);
            model.setTransition(new double[]{0.1}, t);
            model.setStateIntercept(new double[]{0.0}, t);
            model.setSelection(new double[]{1.0}, t);
            model.setStateCov(new double[]{0.0}, t);
            model.setEndog(new double[]{0.1 * (t + 1)}, t);
        }
        return model;
    }

    private static ModelFixture buildMultivariateModel() {
        ModelFixture model = new ModelFixture(2, 2, 2, 4);
        for (int t = 0; t < model.nobs; t++) {
            model.setDesign(new double[]{1.0, 0.2, 0.1, 1.0}, t);
            model.setObsIntercept(new double[]{0.0, 0.0}, t);
            model.setObsCov(new double[]{0.6, 0.0, 0.0, 0.8}, t);
            model.setTransition(new double[]{0.7, 0.1, 0.0, 0.6}, t);
            model.setStateIntercept(new double[]{0.05, -0.02}, t);
            model.setSelection(new double[]{1.0, 0.0, 0.0, 1.0}, t);
            model.setStateCov(new double[]{0.2, 0.03, 0.03, 0.25}, t);
            model.setEndog(new double[]{1.0 + 0.2 * t, -0.4 + 0.1 * t}, t);
        }
        return model;
    }

    private static KalmanSSM buildConvergingChandrasekharModel(int kEndog, int kStates, int nobs) {
        int kPosdef = kStates;
        double[] design = new double[kEndog * kStates];
        for (int i = 0; i < Math.min(kEndog, kStates); i++) {
            design[i * kStates + i] = 1.0;
        }
        double[] obsCov = diagonal(kEndog, 1.0);
        double[] transition = new double[kStates * kStates];
        double[] selection = diagonal(kStates, 1.0);
        double[] stateCov = diagonal(kStates, 0.5);
        double[] endog = new double[kEndog * nobs];
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                endog[t * kEndog + i] = ((t & 1) == 0 ? 0.25 : -0.75) * (i + 1);
            }
        }
        return KalmanSSM.builder(kEndog, kStates, kPosdef, nobs)
            .design(design, false)
            .obsIntercept(new double[kEndog], false)
            .obsCov(obsCov, false)
            .transition(transition, false)
            .stateIntercept(new double[kStates], false)
            .selection(selection, false)
            .stateCovariance(stateCov, false)
            .endog(endog)
            .allObserved()
            .build();
    }

    private static KalmanSSM buildLuOnlyChandrasekharModel() {
        int nobs = 5;
        return KalmanSSM.builder(2, 2, 2, nobs)
            .design(new double[]{1.0, 0.0, 0.0, 1.0}, false)
            .obsIntercept(new double[]{0.0, 0.0}, false)
            .obsCov(new double[]{-2.0, 0.0, 0.0, -3.0}, false)
            .transition(new double[]{0.0, 0.0, 0.0, 0.0}, false)
            .stateIntercept(new double[]{0.0, 0.0}, false)
            .selection(new double[]{1.0, 0.0, 0.0, 1.0}, false)
            .stateCovariance(new double[]{0.5, 0.0, 0.0, 0.5}, false)
            .endog(new double[]{0.25, -0.5, -0.75, 1.0, 0.5, -0.25, -0.25, 0.75, 1.0, -1.0})
            .allObserved()
            .build();
    }

    private static KalmanSSM buildVarLikeChandrasekharModel(boolean timeVaryingIntercepts) {
        int nobs = 8;
        double[] obsIntercept = timeVaryingIntercepts ? new double[2 * nobs] : new double[]{0.03, -0.02};
        double[] stateIntercept = timeVaryingIntercepts ? new double[4 * nobs] : new double[]{0.01, -0.015, 0.0, 0.0};
        double[] endog = new double[2 * nobs];
        for (int t = 0; t < nobs; t++) {
            if (timeVaryingIntercepts) {
                obsIntercept[2 * t] = 0.03 - 0.004 * t;
                obsIntercept[2 * t + 1] = -0.02 + 0.006 * t;
                stateIntercept[4 * t] = 0.01 + 0.002 * t;
                stateIntercept[4 * t + 1] = -0.015 + 0.001 * t;
            }
            endog[2 * t] = 0.4 + 0.12 * t - 0.03 * (t % 3);
            endog[2 * t + 1] = -0.2 + 0.08 * t + (((t & 1) == 0) ? 0.02 : -0.02);
        }
        return KalmanSSM.builder(2, 4, 2, nobs)
            .design(new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, false)
            .obsIntercept(obsIntercept, timeVaryingIntercepts)
            .obsCov(new double[]{0.15, 0.04, 0.04, 0.20}, false)
            .transition(new double[]{
                0.48, 0.08, 0.12, -0.02,
                -0.05, 0.44, 0.03, 0.10,
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0
            }, false)
            .stateIntercept(stateIntercept, timeVaryingIntercepts)
            .selection(new double[]{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, false)
            .stateCovariance(new double[]{0.35, 0.06, 0.06, 0.28}, false)
            .endog(endog)
            .allObserved()
            .build();
    }

    private static KalmanSSM buildSarimaxLikeChandrasekharModel(boolean timeVaryingIntercepts) {
        int nobs = 10;
        int kStates = 6;
        double[] obsIntercept = timeVaryingIntercepts ? new double[nobs] : new double[]{0.02};
        double[] stateIntercept = timeVaryingIntercepts ? new double[kStates * nobs] : new double[]{0.01, -0.004, 0.0, 0.002, 0.0, 0.0};
        double[] endog = new double[nobs];
        for (int t = 0; t < nobs; t++) {
            if (timeVaryingIntercepts) {
                obsIntercept[t] = 0.02 + 0.003 * Math.sin(0.4 * t);
                stateIntercept[t * kStates] = 0.01 + 0.001 * t;
                stateIntercept[t * kStates + 1] = -0.004 + 0.0005 * t;
                stateIntercept[t * kStates + 3] = 0.002 - 0.0003 * t;
            }
            endog[t] = 0.8 + 0.07 * t - 0.03 * (t % 4) + 0.02 * Math.cos(0.6 * t);
        }
        return KalmanSSM.builder(1, kStates, 2, nobs)
            .design(new double[]{1.0, 0.35, -0.15, 0.25, 0.08, -0.04}, false)
            .obsIntercept(obsIntercept, timeVaryingIntercepts)
            .obsCov(new double[]{0.22}, false)
            .transition(new double[]{
                0.46, 0.18, -0.04, 0.08, 0.02, 0.01,
                0.10, 0.36, 0.09, -0.03, 0.00, 0.02,
                0.00, 0.12, 0.30, 0.04, -0.02, 0.00,
                0.05, 0.00, 0.00, 0.42, 0.16, -0.05,
                0.00, 0.03, 0.00, 0.09, 0.34, 0.11,
                0.00, 0.00, 0.02, -0.02, 0.08, 0.28
            }, false)
            .stateIntercept(stateIntercept, timeVaryingIntercepts)
            .selection(new double[]{
                1.0, 0.0,
                0.2, 0.0,
                0.0, 0.0,
                0.0, 1.0,
                0.0, 0.3,
                0.0, 0.0
            }, false)
            .stateCovariance(new double[]{0.28, 0.04, 0.04, 0.18}, false)
            .endog(endog)
            .allObserved()
            .build();
    }

    private static KalmanSSM buildVarmaxLikeChandrasekharModel() {
        int nobs = 9;
        double[] endog = new double[2 * nobs];
        for (int t = 0; t < nobs; t++) {
            endog[2 * t] = 0.35 + 0.09 * t + 0.04 * Math.sin(0.5 * t);
            endog[2 * t + 1] = -0.15 + 0.05 * t - 0.03 * Math.cos(0.4 * t);
        }
        return KalmanSSM.builder(2, 5, 2, nobs)
            .design(new double[]{
                1.0, 0.0, 0.35, 0.0, 0.10,
                0.0, 1.0, 0.0, 0.30, -0.08
            }, false)
            .obsIntercept(new double[]{0.01, -0.015}, false)
            .obsCov(new double[]{0.18, 0.035, 0.035, 0.24}, false)
            .transition(new double[]{
                0.42, 0.06, 0.10, 0.00, 0.04,
                -0.04, 0.38, 0.00, 0.08, -0.02,
                0.11, 0.00, 0.31, 0.03, 0.00,
                0.00, 0.10, -0.02, 0.29, 0.05,
                0.02, -0.01, 0.04, 0.02, 0.24
            }, false)
            .stateIntercept(new double[]{0.006, -0.004, 0.0, 0.0, 0.002}, false)
            .selection(new double[]{
                1.0, 0.0,
                0.0, 1.0,
                0.25, 0.0,
                0.0, 0.2,
                0.1, -0.1
            }, false)
            .stateCovariance(new double[]{0.30, 0.05, 0.05, 0.26}, false)
            .endog(endog)
            .allObserved()
            .build();
    }

    private static ModelFixture buildCollapsedEligibleModel() {
        ModelFixture model = new ModelFixture(3, 2, 2, 5);
        for (int t = 0; t < model.nobs; t++) {
            model.setDesign(new double[]{
                0.5, 0.2,
                0.0, 0.8,
                1.0, -0.5
            }, t);
            model.setObsIntercept(new double[]{0.0, 0.0, 0.0}, t);
            model.setObsCov(new double[]{
                0.2, 0.0, 0.0,
                0.0, 1.1, 0.0,
                0.0, 0.0, 0.5
            }, t);
            model.setTransition(new double[]{0.4, 0.5, 1.0, 0.0}, t);
            model.setStateIntercept(new double[]{0.03, -0.01}, t);
            model.setSelection(new double[]{1.0, 0.0, 0.0, 1.0}, t);
            model.setStateCov(new double[]{2.0, 0.0, 0.0, 1.0}, t);
            model.setEndog(new double[]{
                1.0 + 0.2 * t,
                -0.3 + 0.1 * t,
                0.6 - 0.15 * t
            }, t);
        }
        return model;
    }

    private static double[] diagonal(int dimension, double value) {
        double[] matrix = new double[dimension * dimension];
        for (int i = 0; i < dimension; i++) {
            matrix[i * dimension + i] = value;
        }
        return matrix;
    }

    private static ModelFixture scaledCovarianceModel(ModelFixture model, double scale) {
        ModelFixture copy = ModelFixture.copyOf(model);
        scaleInPlace(copy.obsCov, scale);
        scaleInPlace(copy.stateCov, scale);
        return copy;
    }

    private static double[] scaled(double[] values, double scale) {
        double[] copy = values.clone();
        scaleInPlace(copy, scale);
        return copy;
    }

    private static void scaleInPlace(double[] values, double scale) {
        for (int i = 0; i < values.length; i++) {
            values[i] *= scale;
        }
    }

    private static FilterOptions predictionOptions() {
        return FilterOptions.builder()
            .drop(FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
                FilterOptions.Surface.KALMAN_GAIN,
                FilterOptions.Surface.LIKELIHOOD)
            .build();
    }

    private static FilterOptions[] chandrasekharRetainedProfiles() {
        return new FilterOptions[]{
            FilterOptions.builder().retainAll().build(),
            FilterOptions.compact(),
            FilterOptions.likelihoodOnly(),
            predictionOptions(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE).build(),
            FilterOptions.standardizedForecastError(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.PREDICTED_STATE,
                FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
                FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE).build(),
            FilterOptions.builder().retainOnly(FilterOptions.Surface.KALMAN_GAIN).build()
        };
    }

    private static long expectedChandrasekharScratchDoubleCount(KalmanSSM model, FilterOptions options) {
        long kEndog = model.observationDimension();
        long kStates = model.stateCount();
        long total = 3L * kStates
            + 2L * kStates * kStates
            + 3L * kEndog
            + 5L * kEndog * kEndog
            + 7L * kStates * kEndog;
        if (options.includes(FilterOptions.Surface.FILTERED_STATE_COVARIANCE)) {
            total += kStates * kStates;
        }
        total += InitialState.requiredStationaryBackingLength(model);
        total += InitialState.requiredStationaryPivotLength(model) * ((long) Integer.BYTES) / Double.BYTES;
        return total;
    }

    private static FilterOptions[] collapsedExactDiffuseProfiles() {
        return new FilterOptions[]{
            FilterOptions.builder()
                .retainOnly(
                    FilterOptions.Surface.PREDICTED_STATE,
                    FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
                    FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE,
                    FilterOptions.Surface.FILTERED_STATE,
                    FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
                    FilterOptions.Surface.LIKELIHOOD)
                .build(),
            FilterOptions.likelihoodOnly(),
            FilterOptions.builder().retainOnly(FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE).build(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE).build(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE,
                FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE).build(),
            FilterOptions.standardizedForecastError(),
            FilterOptions.builder().retainOnly(FilterOptions.Surface.KALMAN_GAIN).build(),
            FilterOptions.builder().retainAll().build(),
            FilterOptions.compact()
        };
    }

    private static ModelFixture buildCollapsedMissingEligibleModel() {
        ModelFixture model = new ModelFixture(4, 2, 2, 6);
        for (int t = 0; t < model.nobs; t++) {
            model.setDesign(new double[]{
                0.5, 0.2,
                0.0, 0.8,
                1.0, -0.5,
                -0.3, 0.7
            }, t);
            model.setObsIntercept(new double[]{0.0, 0.0, 0.0, 0.0}, t);
            model.setObsCov(new double[]{
                0.2, 0.0, 0.0, 0.0,
                0.0, 1.1, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0,
                0.0, 0.0, 0.0, 0.9
            }, t);
            model.setTransition(new double[]{0.4, 0.5, 1.0, 0.0}, t);
            model.setStateIntercept(new double[]{0.03, -0.01}, t);
            model.setSelection(new double[]{1.0, 0.0, 0.0, 1.0}, t);
            model.setStateCov(new double[]{2.0, 0.0, 0.0, 1.0}, t);
            model.setEndog(new double[]{
                1.0 + 0.2 * t,
                -0.3 + 0.1 * t,
                0.6 - 0.15 * t,
                -0.8 + 0.05 * t
            }, t);
        }
        model.setMissing(new boolean[]{false, true, false, false}, 1);
        model.setMissing(new boolean[]{false, false, true, false}, 3);
        return model;
    }

    private static ModelFixture buildCollapsedMissingNonDiagonalEligibleModel() {
        ModelFixture model = buildCollapsedMissingEligibleModel();
        double[] obsCov = {
            0.49, 0.14, -0.07, 0.105,
            0.14, 1.25, 0.31, -0.025,
            -0.07, 0.31, 0.74, 0.17,
            0.105, -0.025, 0.17, 0.8975
        };
        for (int t = 0; t < model.nobs; t++) {
            model.setObsCov(obsCov, t);
        }
        return model;
    }

    private static ModelFixture buildCollapsedInterceptEligibleModel() {
        ModelFixture model = buildCollapsedEligibleModel();
        for (int t = 0; t < model.nobs; t++) {
            model.setObsIntercept(new double[]{
                0.15 - 0.02 * t,
                -0.08 + 0.03 * t,
                0.21 + 0.01 * t
            }, t);
        }
        return model;
    }


    private static ModelFixture buildCollapsedNonDiagonalEligibleModel() {
        ModelFixture model = new ModelFixture(3, 2, 2, 5);
        double[] obsCov = {
            0.49, 0.14, -0.07,
            0.14, 1.25, 0.31,
            -0.07, 0.31, 0.74
        };
        for (int t = 0; t < model.nobs; t++) {
            double[] design = collapsedBaseDesign();
            double[] endog = collapsedBaseEndog(t);
            model.setDesign(design, t);
            model.setObsIntercept(new double[]{0.0, 0.0, 0.0}, t);
            model.setObsCov(obsCov, t);
            setCollapsedStateSystem(model, t);
            model.setEndog(endog, t);
        }
        return model;
    }

    private static ModelFixture buildCollapsedWhitenedEquivalentModel() {
        ModelFixture model = new ModelFixture(3, 2, 2, 5);
        for (int t = 0; t < model.nobs; t++) {
            model.setDesign(whitenCollapsedDesign(collapsedBaseDesign()), t);
            model.setObsIntercept(new double[]{0.0, 0.0, 0.0}, t);
            model.setObsCov(new double[]{
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            }, t);
            setCollapsedStateSystem(model, t);
            model.setEndog(whitenCollapsedVector(collapsedBaseEndog(t)), t);
        }
        return model;
    }

    private static double[] collapsedBaseDesign() {
        return new double[]{
            0.5, 0.2,
            0.0, 0.8,
            1.0, -0.5
        };
    }

    private static double[] collapsedBaseEndog(int t) {
        return new double[]{
            1.0 + 0.2 * t,
            -0.3 + 0.1 * t,
            0.6 - 0.15 * t
        };
    }

    private static void setCollapsedStateSystem(ModelFixture model, int t) {
        model.setTransition(new double[]{0.4, 0.5, 1.0, 0.0}, t);
        model.setStateIntercept(new double[]{0.03, -0.01}, t);
        model.setSelection(new double[]{1.0, 0.0, 0.0, 1.0}, t);
        model.setStateCov(new double[]{2.0, 0.0, 0.0, 1.0}, t);
    }

    private static double[] whitenCollapsedDesign(double[] design) {
        double[] transformed = new double[design.length];
        for (int col = 0; col < 2; col++) {
            double first = design[col] / 0.7;
            double second = (design[2 + col] - 0.2 * first) / 1.1;
            double third = (design[4 + col] + 0.1 * first - 0.3 * second) / 0.8;
            transformed[col] = first;
            transformed[2 + col] = second;
            transformed[4 + col] = third;
        }
        return transformed;
    }

    private static double[] whitenCollapsedVector(double[] vector) {
        double first = vector[0] / 0.7;
        double second = (vector[1] - 0.2 * first) / 1.1;
        double third = (vector[2] + 0.1 * first - 0.3 * second) / 0.8;
        return new double[]{first, second, third};
    }

    private static ModelFixture buildSingularMultivariateModel() {
        ModelFixture model = new ModelFixture(2, 1, 1, 2);
        for (int t = 0; t < model.nobs; t++) {
            model.setDesign(new double[]{1.0, 1.0}, t);
            model.setObsIntercept(new double[]{0.0, 0.0}, t);
            model.setObsCov(new double[]{0.0, 0.0, 0.0, 0.0}, t);
            model.setTransition(new double[]{1.0}, t);
            model.setStateIntercept(new double[]{0.0}, t);
            model.setSelection(new double[]{1.0}, t);
            model.setStateCov(new double[]{0.0}, t);
            model.setEndog(new double[]{1.0 + t, 1.0 + t}, t);
        }
        return model;
    }

    private static ModelFixture buildComplexScalarModel() {
        ModelFixture model = ModelFixture.complex(1, 1, 1, 3);
        for (int t = 0; t < model.nobs; t++) {
            model.setDesign(new double[]{1.0, 0.0}, t);
            model.setObsIntercept(new double[]{0.0, 0.0}, t);
            model.setObsCov(new double[]{1.0, 0.0}, t);
            model.setTransition(new double[]{0.8, 0.0}, t);
            model.setStateIntercept(new double[]{0.0, 0.0}, t);
            model.setSelection(new double[]{1.0, 0.0}, t);
            model.setStateCov(new double[]{0.25, 0.0}, t);
            model.setEndog(new double[]{1.0 + 0.1 * t, 0.0}, t);
        }
        return model;
    }

    @Test
    void conventionalRouteMatchesKernelFilter() {
        ModelFixture model = buildScalarModel();
        InitialState kernelInit = InitialState.known(new double[]{0.0}, new double[]{1.0});
        FilterOptions kernelOptions = FilterOptions.defaults();

        FilterResult expected = KalmanEngine.filter(model, kernelInit, kernelOptions);
        FilterResult actual = KalmanEngine.filter(model,
            kernelInit,
            FilterOptions.builder()
                .method(FilterMethod.CONVENTIONAL)

                .build());

        assertFilterOutputsEqual(expected, actual);
    }

    @Test
    void univariateRouteMatchesKernelFilterAndUsesPool() {
        ModelFixture model = buildScalarModel();
        InitialState kernelInit = InitialState.known(new double[]{0.0}, new double[]{1.0});
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();

        FilterResult expected = KalmanEngine.filterUnivariate(model, kernelInit, FilterOptions.defaults());
        FilterResult actual = KalmanEngine.filterBorrowedUnsafe(model,
            kernelInit,
            workspace,
            FilterOptions.builder().method(FilterMethod.UNIVARIATE).build());

        assertFilterOutputsEqual(expected, actual);
        assertTrue(workspace.retainedUnivariateScratchDoubleCount() > 0);
    }

    @Test
    void engineWorkspaceBorrowedFilteringMatchesLegacyPoolRoutes() {
        ModelFixture conventionalModel = buildMultivariateModel();
        InitialState conventionalInit = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.1, 0.1, 1.0});
        FilterOptions conventionalOptions = FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build();
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();

        FilterResult conventionalExpected = KalmanEngine.filter(conventionalModel, conventionalInit, conventionalOptions);
        FilterResult conventionalActual = KalmanEngine.filterBorrowedUnsafe(conventionalModel, conventionalInit, workspace, conventionalOptions);

        assertFilterOutputsEqual(conventionalExpected, conventionalActual);
        assertTrue(workspace.retainedFilterResultDoubleCount() > 0);

        ModelFixture univariateModel = buildScalarModel();
        InitialState univariateInit = InitialState.known(new double[]{0.0}, new double[]{1.0});
        FilterOptions univariateOptions = FilterOptions.builder().method(FilterMethod.UNIVARIATE).build();

        FilterResult univariateExpected = KalmanEngine.filter(univariateModel, univariateInit, univariateOptions);
        FilterResult univariateActual = KalmanEngine.filterBorrowedUnsafe(univariateModel, univariateInit, workspace, univariateOptions);

        assertFilterOutputsEqual(univariateExpected, univariateActual);
        assertTrue(workspace.retainedUnivariateScratchDoubleCount() > 0);
    }

    @Test
    void engineWorkspaceBorrowedAdvancedRoutesUseDedicatedPools() {
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();

        ModelFixture collapsedModel = buildCollapsedEligibleModel();
        InitialState collapsedInit = InitialState.known(new double[]{0.2, -0.1}, new double[]{
            1.0, 0.1,
            0.1, 1.4
        });
        FilterOptions collapsedOptions = FilterOptions.builder().method(FilterMethod.COLLAPSED).build();
        FilterResult collapsedExpected = KalmanEngine.filter(collapsedModel, collapsedInit, collapsedOptions);
        FilterResult collapsedActual = KalmanEngine.filterBorrowedUnsafe(collapsedModel, collapsedInit, workspace, collapsedOptions);

        assertFilterOutputsEqual(collapsedExpected, collapsedActual);
        assertTrue(workspace.retainedCollapsedScratchDoubleCount() > 0);
        assertTrue(workspace.retainedCollapsedResultDoubleCount() > 0);

        KalmanSSM chandrasekharModel = buildConvergingChandrasekharModel(2, 2, 40);
        InitialState chandrasekharInit = InitialState.stationary(chandrasekharModel);
        FilterOptions chandrasekharOptions = FilterOptions.builder().method(FilterMethod.CHANDRASEKHAR).build();
        FilterResult chandrasekharExpected = KalmanEngine.filter(chandrasekharModel, chandrasekharInit, chandrasekharOptions);
        FilterResult chandrasekharActual = KalmanEngine.filterBorrowedUnsafe(chandrasekharModel, chandrasekharInit,
            workspace, chandrasekharOptions);

        assertFilterOutputsEqual(chandrasekharExpected, chandrasekharActual);
        assertTrue(workspace.retainedChandrasekharScratchDoubleCount() > 0);
        assertTrue(workspace.retainedChandrasekharResultDoubleCount() > 0);
    }

    @Test
    void engineWorkspaceAggregateMemoryDiagnosticsCoverAllFilterPools() {
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();

        ModelFixture realModel = buildMultivariateModel();
        InitialState realInit = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.1, 0.1, 1.0});
        KalmanEngine.filterBorrowedUnsafe(realModel, realInit, workspace,
            FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build());
        KalmanEngine.filterBorrowedUnsafe(buildScalarModel(), InitialState.known(new double[]{0.0}, new double[]{1.0}), workspace,
            FilterOptions.builder().method(FilterMethod.UNIVARIATE).build());
        KalmanEngine.filterComplexBorrowedUnsafe(buildComplexScalarModel(), InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0}), workspace,
            FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build());

        long summedRealScratch = workspace.retainedFilterScratchDoubleCount()
            + workspace.retainedUnivariateScratchDoubleCount()
            + workspace.retainedCollapsedScratchDoubleCount()
            + workspace.retainedChandrasekharScratchDoubleCount()
            + workspace.retainedTimingScratchDoubleCount();
        long expectedRealResult = workspace.retainedFilterResultDoubleCount()
            + workspace.retainedUnivariateResultDoubleCount()
            + workspace.retainedCollapsedResultDoubleCount()
            + workspace.retainedChandrasekharResultDoubleCount();

        assertTrue(workspace.retainedRealFilterScratchDoubleCount() > 0);
        assertTrue(workspace.retainedRealFilterScratchDoubleCount() <= summedRealScratch);
        assertEquals(expectedRealResult, workspace.retainedRealFilterResultDoubleCount());
        assertTrue(workspace.retainedFilterWorkspaceScratchDoubleCount()
            <= workspace.retainedRealFilterScratchDoubleCount() + workspace.retainedComplexFilterScratchDoubleCount());
        assertEquals(expectedRealResult + workspace.retainedComplexFilterResultDoubleCount(),
            workspace.retainedFilterWorkspaceResultDoubleCount());

        workspace.releaseRetainedResults();
        assertEquals(0L, workspace.retainedFilterWorkspaceResultDoubleCount());

        workspace.releaseRetainedScratch();
        assertEquals(0L, workspace.retainedFilterWorkspaceScratchDoubleCount());
    }

    @Test
    void autoRouteChoosesSupportedAdvancedRouteConservatively() {
        KalmanSSM chandrasekharModel = buildConvergingChandrasekharModel(2, 2, 40);
        InitialState chandrasekharInit = InitialState.stationary(chandrasekharModel);
        FilterOptions autoOptions = FilterOptions.builder().method(FilterMethod.AUTO).retainAll().build();
        FilterResult expectedChandrasekhar = KalmanEngine.filter(chandrasekharModel, chandrasekharInit,
            autoOptions.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());
        FilterResult actualAuto = KalmanEngine.filter(chandrasekharModel, chandrasekharInit, autoOptions);

        assertFilterOutputsEqual(expectedChandrasekhar, actualAuto);

        ModelFixture concentratedModel = buildMultivariateModel();
        InitialState concentratedInit = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.1, 0.1, 1.2});
        FilterOptions concentratedAutoOptions = FilterOptions.builder()
            .method(FilterMethod.AUTO)
            .concentratedLikelihood(true)
            .build();
        FilterResult expectedConventional = KalmanEngine.filter(concentratedModel, concentratedInit,
            concentratedAutoOptions.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actualConcentratedAuto = KalmanEngine.filter(concentratedModel, concentratedInit, concentratedAutoOptions);

        assertFilterOutputsEqual(expectedConventional, actualConcentratedAuto);
    }

    @Test
    void borrowedInitFilteredTimingReusesWorkspaceScratch() {
        ModelFixture model = buildScalarModel();
        InitialState init = InitialState.known(new double[]{0.0}, new double[]{1.0});
        FilterOptions options = FilterOptions.builder().timing(FilterOptions.Timing.INIT_FILTERED).build();
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();

        FilterResult expected = KalmanEngine.filter(model, init, options);
        FilterResult actual = KalmanEngine.filterBorrowedUnsafe(model, init, workspace, options);

        assertFilterOutputsEqual(expected, actual);
        assertTrue(workspace.retainedTimingScratchDoubleCount() > 0);
        assertEquals(5L, workspace.retainedTimingScratchDoubleCount());

        workspace.releaseRetainedScratch();
        KalmanEngine.filterBorrowedUnsafe(model, InitialState.diffuse(), workspace, options);
        assertEquals(6L, workspace.retainedTimingScratchDoubleCount());
    }

    @Test
    void chandrasekharRouteValidatesStatsmodelsContractBeforeKernelGate() {
        InitialState stationary = InitialState.stationary(1);
        FilterOptions options = FilterOptions.builder().method(FilterMethod.CHANDRASEKHAR).build();

        UnsupportedOperationException knownFailure = assertThrows(UnsupportedOperationException.class,
            () -> KalmanEngine.filter(buildScalarModel(),
                InitialState.known(new double[]{0.0}, new double[]{1.0}),
                options));
        assertTrue(knownFailure.getMessage().contains("stationary initialization"));

        ModelFixture missing = buildScalarModel();
        missing.setMissing(new boolean[]{true}, 1);
        UnsupportedOperationException missingFailure = assertThrows(UnsupportedOperationException.class,
            () -> KalmanEngine.filter(missing, stationary, options));
        assertTrue(missingFailure.getMessage().contains("missing observations"));

        UnsupportedOperationException filteredTimingFailure = assertThrows(UnsupportedOperationException.class,
            () -> KalmanEngine.filter(buildScalarModel(), stationary,
                options.toBuilder().timing(FilterOptions.Timing.INIT_FILTERED).build()));
        assertTrue(filteredTimingFailure.getMessage().contains("INIT_FILTERED timing"));

        ModelFixture timeVarying = buildScalarModel();
        timeVarying.setDesign(new double[]{0.9}, 2);
        UnsupportedOperationException timeVaryingFailure = assertThrows(UnsupportedOperationException.class,
            () -> KalmanEngine.filter(timeVarying, stationary, options));
        assertTrue(timeVaryingFailure.getMessage().contains("time-varying system matrices"));
    }

    @Test
    void chandrasekharRouteMatchesConventionalFilterForStationaryTimeInvariantModels() {
        FilterOptions options = FilterOptions.builder().retainAll().build();
        ModelFixture scalar = buildScalarModel();
        InitialState scalarStationary = InitialState.stationary(1);

        FilterResult expectedScalar = KalmanEngine.filter(scalar, scalarStationary,
            options.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actualScalar = KalmanEngine.filter(scalar, scalarStationary,
            options.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());
        assertCollapsedFilterOutputsEqual(expectedScalar, actualScalar);
        assertEquals(expectedScalar.converged(), actualScalar.converged());
        assertEquals(expectedScalar.periodConverged(), actualScalar.periodConverged());

        ModelFixture multivariate = buildMultivariateModel();
        InitialState multivariateStationary = InitialState.stationary(2);
        FilterResult expectedMultivariate = KalmanEngine.filter(multivariate, multivariateStationary,
            options.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actualMultivariate = KalmanEngine.filter(multivariate, multivariateStationary,
            options.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());
        assertCollapsedFilterOutputsEqual(expectedMultivariate, actualMultivariate);
        assertEquals(expectedMultivariate.converged(), actualMultivariate.converged());
        assertEquals(expectedMultivariate.periodConverged(), actualMultivariate.periodConverged());
    }

    @Test
    void chandrasekharRouteMatchesConventionalFilterForCompactRetainedSurfaces() {
        ModelFixture model = buildMultivariateModel();
        InitialState init = InitialState.stationary(2);
        FilterOptions[] profiles = {
            FilterOptions.compact(),
            FilterOptions.likelihoodOnly(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE).build(),
            FilterOptions.standardizedForecastError(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.PREDICTED_STATE,
                FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
                FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE).build(),
            FilterOptions.builder().retainOnly(FilterOptions.Surface.KALMAN_GAIN).build()
        };

        for (FilterOptions profile : profiles) {
            FilterResult expected = KalmanEngine.filter(model, init,
                profile.toBuilder().method(FilterMethod.CONVENTIONAL).build());
            FilterResult actual = KalmanEngine.filter(model, init,
                profile.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());

            assertCollapsedFilterRetainedSurfacesEqual(expected, actual);
            assertEquals(expected.converged(), actual.converged());
            assertEquals(expected.periodConverged(), actual.periodConverged());
        }
    }

    @Test
    void chandrasekharRouteMatchesConventionalConvergencePeriod() {
        KalmanSSM model = buildConvergingChandrasekharModel(2, 2, 6);
        InitialState init = InitialState.stationary(model);
        FilterOptions options = FilterOptions.builder().retainAll().build();

        FilterResult expected = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());

        assertCollapsedFilterOutputsEqual(expected, actual);
        assertTrue(actual.converged());
        assertEquals(expected.converged(), actual.converged());
        assertEquals(expected.periodConverged(), actual.periodConverged());
    }

    @Test
    void chandrasekharRouteHonorsLuSolveStrategy() {
        KalmanSSM model = buildLuOnlyChandrasekharModel();
        InitialState init = InitialState.stationary(model);
        FilterOptions options = FilterOptions.builder()
            .retainAll()
            .forecastErrorStrategy(ForecastErrorStrategy.LU_SOLVE)
            .build();

        FilterResult expected = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());

        assertCollapsedFilterOutputsEqual(expected, actual);
        assertEquals(expected.converged(), actual.converged());
        assertEquals(expected.periodConverged(), actual.periodConverged());
    }

    @Test
    void chandrasekharRouteAllowsTimeVaryingInterceptsForVarLikeModels() {
        KalmanSSM model = buildVarLikeChandrasekharModel(true);
        InitialState init = InitialState.stationary(model);
        FilterOptions options = FilterOptions.builder().retainAll().build();

        FilterResult expected = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());

        assertCollapsedFilterOutputsEqual(expected, actual);
        assertEquals(expected.converged(), actual.converged());
        assertEquals(expected.periodConverged(), actual.periodConverged());
    }

    @Test
    void chandrasekharRouteMatchesConventionalLikelihoodOnlyForVarLikeModels() {
        KalmanSSM model = buildVarLikeChandrasekharModel(false);
        InitialState init = InitialState.stationary(model);
        FilterOptions options = FilterOptions.likelihoodOnly();

        FilterResult expected = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());

        assertCollapsedFilterRetainedSurfacesEqual(expected, actual);
        assertEquals(0, actual.forecastLength());
        assertEquals(0, actual.predictedStateLength());
        assertEquals(0, actual.kalmanGainLength());
    }

    @Test
    void chandrasekharRouteMatchesConventionalFilterForSarimaxLikeLaggedLayouts() {
        KalmanSSM[] models = {
            buildSarimaxLikeChandrasekharModel(false),
            buildSarimaxLikeChandrasekharModel(true),
            buildVarmaxLikeChandrasekharModel()
        };

        for (KalmanSSM model : models) {
            assertChandrasekharMatchesConventional(model, FilterOptions.builder().retainAll().build());
        }
    }

    @Test
    void chandrasekharRouteMatchesConventionalFilterForBroaderMemoryProfiles() {
        KalmanSSM[] models = {
            buildSarimaxLikeChandrasekharModel(false),
            buildVarmaxLikeChandrasekharModel()
        };

        for (KalmanSSM model : models) {
            for (FilterOptions profile : chandrasekharRetainedProfiles()) {
                FilterResult actual = assertChandrasekharMatchesConventional(model, profile);
                assertEquals(profile.includes(FilterOptions.Surface.FORECAST_MEAN) ? model.observationDimension() * model.observationCount() : 0,
                    actual.forecastLength());
                assertEquals(profile.includes(FilterOptions.Surface.FILTERED_STATE) ? model.stateCount() * model.observationCount() : 0,
                    actual.filteredStateLength());
                assertEquals(profile.includes(FilterOptions.Surface.KALMAN_GAIN) ? model.stateCount() * model.observationDimension() * model.observationCount() : 0,
                    actual.kalmanGainLength());
            }
        }
    }

    @Test
    void chandrasekharRouteSupportsConcentratedLikelihood() {
        KalmanSSM model = buildConvergingChandrasekharModel(2, 2, 40);
        InitialState init = InitialState.stationary(model);
        FilterOptions profile = FilterOptions.builder()
            .retainAll()
            .concentratedLikelihood(true)
            .concentratedLikelihoodBurn(2)
            .build();

        FilterResult expected = KalmanEngine.filter(model, init,
            profile.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model, init,
            profile.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());

        assertTrue(actual.concentratedLikelihood());
        assertEquals(expected.scale(), actual.scale(), CHANDRASEKHAR_BREADTH_TOL);
        assertEquals(expected.scaleObservationCount(), actual.scaleObservationCount());
        assertEquals(expected.scaleObsLength(), actual.scaleObsLength());
        assertEquals(expected.scaleObsCountLength(), actual.scaleObsCountLength());
        assertFilterRetainedSurfacesEqual(expected, actual, CHANDRASEKHAR_BREADTH_TOL);
    }

    @Test
    void chandrasekharWorkspaceRetainsUnifiedScratchLayout() {
        KalmanSSM model = buildConvergingChandrasekharModel(2, 2, 6);
        InitialState init = InitialState.stationary(model);
        FilterOptions full = FilterOptions.builder().method(FilterMethod.CHANDRASEKHAR).retainAll().build();
        FilterOptions likelihoodOnly = FilterOptions.likelihoodOnly().toBuilder()
            .method(FilterMethod.CHANDRASEKHAR)
            .build();
        KalmanEngine.Workspace fullWorkspace = KalmanEngine.workspace();
        KalmanEngine.Workspace compactWorkspace = KalmanEngine.workspace();

        fullWorkspace.reserveChandrasekharFilter(model, init, full);
        compactWorkspace.reserveChandrasekharFilter(model, init, likelihoodOnly);

        long fullScratch = expectedChandrasekharScratchDoubleCount(model, full);
        long compactScratch = expectedChandrasekharScratchDoubleCount(model, likelihoodOnly);
        assertEquals(fullScratch, fullWorkspace.retainedChandrasekharScratchDoubleCount());
        assertEquals(compactScratch, compactWorkspace.retainedChandrasekharScratchDoubleCount());
        assertTrue(compactScratch < fullScratch);
    }

    @Test
    void collapsedRouteMatchesConventionalFilterForEligibleRealModel() {
        ModelFixture model = buildCollapsedEligibleModel();
        InitialState init = InitialState.known(new double[]{0.2, -0.1}, new double[]{
            1.0, 0.1,
            0.1, 1.4
        });
        FilterOptions options = FilterOptions.builder().retainAll().build();

        FilterResult expected = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.COLLAPSED).build());

        assertCollapsedFilterOutputsEqual(expected, actual);
    }

    @Test
    void collapsedRouteSupportsBorrowedObservations() {
        ModelFixture model = buildCollapsedEligibleModel();
        InitialState init = InitialState.known(new double[]{0.2, -0.1}, new double[]{
            1.0, 0.1,
            0.1, 1.4
        });
        double[] overriddenEndog = new double[model.kEndog * model.nobs];
        for (int t = 0; t < model.nobs; t++) {
            overriddenEndog[t * model.kEndog] = 0.7 + 0.05 * t;
            overriddenEndog[t * model.kEndog + 1] = -0.2 + 0.03 * t;
            overriddenEndog[t * model.kEndog + 2] = 0.4 - 0.07 * t;
        }
        FilterOptions conventional = FilterOptions.builder()
            .method(FilterMethod.CONVENTIONAL)
            .retainAll()
            .build();
        FilterOptions collapsed = conventional.toBuilder()
            .method(FilterMethod.COLLAPSED)
            .build();

        FilterResult expected = KalmanEngine.filterBorrowedUnsafe(model, init, overriddenEndog, 0,
            KalmanEngine.workspace(), conventional).clone();
        FilterResult actual = KalmanEngine.filterBorrowedUnsafe(model, init, overriddenEndog, 0,
            KalmanEngine.workspace(), collapsed);

        assertCollapsedFilterOutputsEqual(expected, actual);
    }

    @Test
    void collapsedRouteMatchesConventionalFilterWithSupportedMissingObservations() {
        ModelFixture model = buildCollapsedMissingEligibleModel();
        InitialState init = InitialState.known(new double[]{0.2, -0.1}, new double[]{
            1.0, 0.1,
            0.1, 1.4
        });
        FilterOptions options = FilterOptions.builder().retainAll().build();

        FilterResult expected = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.COLLAPSED).build());

        assertCollapsedFilterOutputsEqual(expected, actual);
    }

    @Test
    void collapsedRouteTreatsApproximateDiffuseAsFiniteCovariance() {
        ModelFixture model = buildCollapsedEligibleModel();
        InitialState init = InitialState.approximateDiffuse(2, 100.0);
        FilterOptions options = FilterOptions.builder().retainAll().build();

        FilterResult expected = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model, init,
            options.toBuilder().method(FilterMethod.COLLAPSED).build());

        assertCollapsedFilterOutputsEqual(expected, actual);
    }

    @Test
    void collapsedRouteMatchesConventionalExactDiffuseFilterForSupportedProfile() {
        ModelFixture model = buildCollapsedEligibleModel();
        InitialState init = InitialState.diffuse(2);

        for (FilterOptions profile : collapsedExactDiffuseProfiles()) {
            FilterResult expected = KalmanEngine.filter(model, init,
                profile.toBuilder().method(FilterMethod.CONVENTIONAL).build());
            FilterResult actual = KalmanEngine.filter(model, init,
                profile.toBuilder().method(FilterMethod.COLLAPSED).build());

            assertCollapsedFilterRetainedSurfacesEqual(expected, actual);
            assertEquals(profile.includes(FilterOptions.Surface.KALMAN_GAIN), actual.perObsKalmanGain);
        }
    }

    @Test
    void collapsedRouteMatchesConventionalExactDiffuseFilterWithSupportedMissingObservations() {
        ModelFixture model = buildCollapsedMissingEligibleModel();
        InitialState init = InitialState.diffuse(2);

        for (FilterOptions profile : collapsedExactDiffuseProfiles()) {
            FilterResult expected = KalmanEngine.filter(model, init,
                profile.toBuilder().method(FilterMethod.CONVENTIONAL).build());
            FilterResult actual = KalmanEngine.filter(model, init,
                profile.toBuilder().method(FilterMethod.COLLAPSED).build());

            assertCollapsedFilterRetainedSurfacesEqual(expected, actual);
            assertEquals(profile.includes(FilterOptions.Surface.KALMAN_GAIN), actual.perObsKalmanGain);
        }
    }

    @Test
    void collapsedRouteSupportsObservationIntercepts() {
        ModelFixture model = buildCollapsedInterceptEligibleModel();
        InitialState known = InitialState.known(new double[]{0.2, -0.1}, new double[]{
            1.0, 0.1,
            0.1, 1.4
        });
        InitialState diffuse = InitialState.diffuse(2);
        FilterOptions[] finiteProfiles = {
            FilterOptions.compact(),
            FilterOptions.likelihoodOnly(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE).build(),
            FilterOptions.standardizedForecastError(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.PREDICTED_STATE,
                FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
                FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE).build(),
            FilterOptions.builder().retainOnly(FilterOptions.Surface.KALMAN_GAIN).build()
        };

        for (FilterOptions profile : finiteProfiles) {
            FilterResult expected = KalmanEngine.filter(model, known,
                profile.toBuilder().method(FilterMethod.CONVENTIONAL).build());
            FilterResult actual = KalmanEngine.filter(model, known,
                profile.toBuilder().method(FilterMethod.COLLAPSED).build());

            assertCollapsedFilterRetainedSurfacesEqual(expected, actual);
        }
        for (FilterOptions profile : collapsedExactDiffuseProfiles()) {
            FilterResult expected = KalmanEngine.filter(model, diffuse,
                profile.toBuilder().method(FilterMethod.CONVENTIONAL).build());
            FilterResult actual = KalmanEngine.filter(model, diffuse,
                profile.toBuilder().method(FilterMethod.COLLAPSED).build());

            assertCollapsedFilterRetainedSurfacesEqual(expected, actual);
            assertEquals(profile.includes(FilterOptions.Surface.KALMAN_GAIN), actual.perObsKalmanGain);
        }
    }

    @Test
    void collapsedRouteMatchesWhitenedExactDiffuseFilterWithNonDiagonalObservationCovariance() {
        ModelFixture model = buildCollapsedNonDiagonalEligibleModel();
        ModelFixture whitened = buildCollapsedWhitenedEquivalentModel();
        InitialState init = InitialState.diffuse(2);
        FilterOptions[] profiles = {
            FilterOptions.builder()
                .retainOnly(
                    FilterOptions.Surface.PREDICTED_STATE,
                    FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
                    FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE,
                    FilterOptions.Surface.FILTERED_STATE,
                    FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
                    FilterOptions.Surface.LIKELIHOOD)
                .build(),
            FilterOptions.likelihoodOnly(),
            FilterOptions.builder().retainOnly(FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE).build(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE).build(),
            FilterOptions.compact()
        };

        for (FilterOptions profile : profiles) {
            FilterResult expected = KalmanEngine.filter(whitened, init,
                profile.toBuilder().method(FilterMethod.COLLAPSED).build());
            FilterResult actual = KalmanEngine.filter(model, init,
                profile.toBuilder().method(FilterMethod.COLLAPSED).build());

            assertCollapsedWhitenedSurfacesEqual(expected, actual, NON_DIAGONAL_COLLAPSED_LOG_DET_FACTOR);
        }
    }

    @Test
    void collapsedRouteReconstructsExactDiffuseNonDiagonalObservationSurfaces() {
        InitialState init = InitialState.diffuse(2);
        ModelFixture[] models = {
            buildCollapsedNonDiagonalEligibleModel(),
            buildCollapsedMissingNonDiagonalEligibleModel()
        };
        FilterOptions[] profiles = {
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE,
                FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE).build(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE,
                FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR).build(),
            FilterOptions.builder().retainAll().drop(FilterOptions.Surface.KALMAN_GAIN).build()
        };

        for (ModelFixture model : models) {
            FilterResult history = KalmanEngine.filter(model, init,
                FilterOptions.builder()
                    .method(FilterMethod.COLLAPSED)
                    .retainOnly(
                        FilterOptions.Surface.PREDICTED_STATE,
                        FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
                        FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)
                    .build());
            for (FilterOptions profile : profiles) {
                FilterResult actual = KalmanEngine.filter(model, init,
                    profile.toBuilder().method(FilterMethod.COLLAPSED).build());

                assertCollapsedNonDiagonalObservationSurfaces(model, history, actual);
                assertFalsePerObservationGain(actual);
            }
        }
    }

    @Test
    void collapsedRouteSupportsExactDiffuseNonDiagonalKalmanGain() {
        InitialState init = InitialState.diffuse(2);
        ModelFixture model = buildCollapsedNonDiagonalEligibleModel();
        FilterOptions profile = FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.KALMAN_GAIN)
            .build();

        FilterResult actual = KalmanEngine.filter(model, init,
            profile.toBuilder().method(FilterMethod.COLLAPSED).build());

        double[] expected = {
            0.7994336312354380, 0.0, 0.0,
            0.0, 0.8953024980911056, 0.0,
            0.4193050231373371, 0.2112256065049805, 0.0,
            0.5893597492629048, -0.1163910764155791, 0.0,
            0.3939241065403429, 0.1973794185066415, 0.0,
            0.5663948867300015, -0.1280040343718293, 0.0,
            0.3909669987991395, 0.1956390532860274, 0.0,
            0.5638224906430631, -0.1295085796704214, 0.0,
            0.3906212447139474, 0.1954428024441310, 0.0,
            0.5635191173855767, -0.1296806775825387, 0.0,
        };
        assertArrayEquals(expected,
            active(actual.kalmanGain, actual.kalmanGainBase(), actual.kalmanGainLength()), TOL);
        assertTrue(actual.perObsKalmanGain);
    }

    @Test
    void collapsedRouteMatchesConventionalFilterForCompactRetainedSurfaces() {
        ModelFixture model = buildCollapsedMissingEligibleModel();
        InitialState init = InitialState.known(new double[]{0.2, -0.1}, new double[]{
            1.0, 0.1,
            0.1, 1.4
        });
        FilterOptions[] profiles = {
            FilterOptions.compact(),
            FilterOptions.likelihoodOnly(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE).build(),
            FilterOptions.standardizedForecastError(),
            FilterOptions.builder().retainOnly(
                FilterOptions.Surface.PREDICTED_STATE,
                FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
                FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE).build(),
            FilterOptions.builder().retainOnly(FilterOptions.Surface.KALMAN_GAIN).build()
        };

        for (FilterOptions profile : profiles) {
            FilterResult expected = KalmanEngine.filter(model, init,
                profile.toBuilder().method(FilterMethod.CONVENTIONAL).build());
            FilterResult actual = KalmanEngine.filter(model, init,
                profile.toBuilder().method(FilterMethod.COLLAPSED).build());

            assertCollapsedFilterRetainedSurfacesEqual(expected, actual);
        }
    }

    @Test
    void collapsedRouteRejectsOutOfScopeModelsClearly() {
        InitialState known = InitialState.known(new double[]{0.2, -0.1}, new double[]{
            1.0, 0.1,
            0.1, 1.4
        });
        FilterOptions options = FilterOptions.builder().method(FilterMethod.COLLAPSED).build();

        UnsupportedOperationException lowDimension = assertThrows(UnsupportedOperationException.class,
            () -> KalmanEngine.filter(buildMultivariateModel(), known, options));
        assertTrue(lowDimension.getMessage().contains("observation dimension larger than state dimension"));

        ModelFixture missing = buildCollapsedEligibleModel();
        missing.setMissing(new boolean[]{false, true, false}, 1);
        FilterResult expected = KalmanEngine.filter(missing, known,
            options.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(missing, known, options);
        assertFilterOutputsEqual(expected, actual);

    }

    @Test
    void complexAdvancedFilterRoutesOpenConventionalSliceAndKeepAdvancedKernelsGated() {
        ModelFixture model = buildComplexScalarModel();
        InitialState init = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0});

        assertDoesNotThrow(() -> KalmanEngine.filterComplex(model, init,
            FilterOptions.builder().timing(FilterOptions.Timing.INIT_FILTERED).build()));

        assertDoesNotThrow(() -> KalmanEngine.filterComplex(model, init,
            FilterOptions.builder().method(FilterMethod.UNIVARIATE).build()));

        for (FilterMethod method : new FilterMethod[]{FilterMethod.COLLAPSED, FilterMethod.CHANDRASEKHAR}) {
            FilterOptions options = FilterOptions.builder().method(method).build();
            UnsupportedOperationException failure = assertThrows(UnsupportedOperationException.class,
                () -> KalmanEngine.filterComplex(model, init, options));
            assertTrue(failure.getMessage().contains(method + " filtering"));
            assertTrue(failure.getMessage().contains("real-valued models only"));
        }
    }

    @Test
    void conventionalRouteMatchesKernelExactDiffuseFilter() {
        ModelFixture model = buildScalarModel();
        InitialState kernelInit = InitialState.diffuse(1);
        FilterOptions kernelOptions = FilterOptions.defaults();

        FilterResult expected = KalmanEngine.filter(model, kernelInit, kernelOptions);
        FilterResult actual = KalmanEngine.filter(model,
            InitialState.diffuse(1),
            FilterOptions.builder()
                .method(FilterMethod.CONVENTIONAL)

                .build());

        assertFilterOutputsEqual(expected, actual);
    }

    @Test
    void univariateRouteMatchesKernelExactDiffuseFilter() {
        ModelFixture model = buildScalarModel();
        InitialState kernelInit = InitialState.diffuse(1);

        FilterResult expected = KalmanEngine.filterUnivariate(model, kernelInit, FilterOptions.defaults());
        FilterResult actual = KalmanEngine.filter(model,
            InitialState.diffuse(1),
            FilterOptions.builder()
                .method(FilterMethod.UNIVARIATE)

                .build());

        assertFilterOutputsEqual(expected, actual);
    }

    @Test
    void diffuseToleranceControlsExactDiffuseExit() {
        ModelFixture model = buildWeakDiffuseSignalModel();
        InitialState diffuse = InitialState.diffuse(1);
        FilterOptions lowTolerance = FilterOptions.builder()
            .method(FilterMethod.CONVENTIONAL)
            .diffuseTolerance(1e-10)
            .build();
        FilterOptions highTolerance = lowTolerance.toBuilder()
            .diffuseTolerance(1e-6)
            .build();

        FilterResult lowConventional = KalmanEngine.filter(model, diffuse, lowTolerance);
        FilterResult highConventional = KalmanEngine.filter(model, diffuse, highTolerance);
        FilterResult lowUnivariate = KalmanEngine.filter(model, diffuse,
            lowTolerance.toBuilder().method(FilterMethod.UNIVARIATE).build());
        FilterResult highUnivariate = KalmanEngine.filter(model, diffuse,
            highTolerance.toBuilder().method(FilterMethod.UNIVARIATE).build());

        assertEquals(1, lowConventional.nobsDiffuse());
        assertEquals(2, highConventional.nobsDiffuse());
        assertEquals(lowConventional.nobsDiffuse(), lowUnivariate.nobsDiffuse());
        assertEquals(highConventional.nobsDiffuse(), highUnivariate.nobsDiffuse());
    }

    @Test
    void convergenceToleranceControlsSteadyStateDiagnostics() {
        ModelFixture model = buildScalarModel();

        FilterResult result = KalmanEngine.filter(model,
            InitialState.known(new double[]{0.0}, new double[]{1.0}),
            FilterOptions.builder()
                .convergenceTolerance(1.0)
                .build());

        assertTrue(result.converged());
        assertTrue(result.periodConverged() > 0);
        assertTrue(result.periodConverged() < result.nobs);
    }

    @Test
    void filterOptionsControlRetainedSurfaces() {
        ModelFixture model = buildScalarModel();

        FilterResult result = KalmanEngine.builder(model)
            .initialState(InitialState.builder(1).known(0, 1, new double[]{0.0}, new double[]{1.0}).build())
            .options(FilterOptions.builder().retainNone().build())
            .filter();

        assertEquals(0, result.predictedStateLength());
        assertEquals(0, result.filteredStateLength());
        assertEquals(0, result.forecastLength());
        assertEquals(0, result.kalmanGainLength());
        assertEquals(0, result.logLikelihoodObsLength());
    }

    @Test
    void filterResultComputesStandardizedForecastErrors() {
        ModelFixture model = buildScalarModel();
        FilterResult result = KalmanEngine.filter(model,
            InitialState.known(new double[]{0.0}, new double[]{1.0}),
            FilterOptions.defaults());

        double[] standardized = result.standardizedForecastError();

        assertEquals(model.nobs, standardized.length);
        for (int t = 0; t < model.nobs; t++) {
            assertEquals(result.forecastError(0, t) / Math.sqrt(result.forecastErrorCov(0, 0, t)),
                standardized[t], TOL);
        }
    }

    @Test
    void filterResultComputesMultivariateStandardizedForecastErrors() {
        FilterResult result = new FilterResult(2, 1, 1);
        int errorOffset = result.forecastErrorOffset(0);
        int covarianceOffset = result.forecastErrorCovOffset(0);
        result.forecastError[errorOffset] = 4.0;
        result.forecastError[errorOffset + 1] = 7.0;
        result.forecastErrorCov[covarianceOffset] = 4.0;
        result.forecastErrorCov[covarianceOffset + 1] = 2.0;
        result.forecastErrorCov[covarianceOffset + 2] = 2.0;
        result.forecastErrorCov[covarianceOffset + 3] = 5.0;

        assertArrayEquals(new double[]{2.0, 2.5}, result.standardizedForecastError(), TOL);
    }

    @Test
    void standardizedForecastErrorMemoryKeepsForecastBlockOnly() {
        ModelFixture model = buildScalarModel();
        FilterResult result = KalmanEngine.filter(model,
            InitialState.known(new double[]{0.0}, new double[]{1.0}),
            FilterOptions.builder().retainOnly(FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR).build());

        assertEquals(model.nobs, result.standardizedForecastError().length);
        assertEquals(0, result.forecastLength());
        assertEquals(0, result.forecastErrorLength());
        assertEquals(0, result.forecastErrorCovLength());
        assertEquals(model.nobs, result.standardizedForecastErrorLength());
        assertEquals(0, result.predictedStateLength());
        assertEquals(0, result.filteredStateLength());
        assertEquals(0, result.kalmanGainLength());
        assertEquals(0, result.logLikelihoodObsLength());
    }

    @Test
    void luInversionMatchesCholesky() {
        ModelFixture model = buildMultivariateModel();
        InitialState initialState = InitialState.known(
            new double[]{0.0, 0.0},
            new double[]{1.0, 0.1, 0.1, 1.2});

        FilterResult cholesky = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder()
                .method(FilterMethod.CONVENTIONAL)
                .forecastErrorStrategy(ForecastErrorStrategy.CHOLESKY_SOLVE)
                .build());
        FilterResult lu = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder()
                .method(FilterMethod.CONVENTIONAL)
                .forecastErrorStrategy(ForecastErrorStrategy.LU_SOLVE)
                .build());

        assertFilterOutputsEqual(cholesky, lu);
    }

    @Test
    void singularFallbackRoutesToUnivariateFilter() {
        ModelFixture model = buildSingularMultivariateModel();
        InitialState initialState = InitialState.known(new double[]{0.0}, new double[]{1.0});

        FilterResult expected = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder()
                .method(FilterMethod.UNIVARIATE)
                .build());
        FilterResult actual = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder()
                .method(FilterMethod.CONVENTIONAL)
                .singularFallbackPolicy(FilterOptions.SingularFallback.UNIVARIATE)
                .build());

        assertFilterOutputsEqual(expected, actual);
    }

    @Test
    void singularFallbackTelemetryIsOptInAndPreservedByCopy() {
        ModelFixture model = buildSingularMultivariateModel();
        InitialState initialState = InitialState.known(new double[]{0.0}, new double[]{1.0});

        FilterResult noTelemetry = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder()
                .method(FilterMethod.CONVENTIONAL)
                .singularFallbackPolicy(FilterOptions.SingularFallback.UNIVARIATE)
                .build());
        assertFalse(noTelemetry.usedSingularFallback());
        assertEquals(0, noTelemetry.fallbackPeriodCount());
        assertArrayEquals(new int[0], noTelemetry.fallbackPeriods());

        FilterResult telemetry = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder()
                .method(FilterMethod.CONVENTIONAL)
                .singularFallbackPolicy(FilterOptions.SingularFallback.UNIVARIATE)
                .fallbackTelemetry(true)
                .build());

        assertTrue(telemetry.usedSingularFallback());
        assertTrue(telemetry.fullUnivariateFallback());
        assertEquals(model.nobs, telemetry.fallbackPeriodCount());
        assertArrayEquals(new int[]{0, 1}, telemetry.fallbackPeriods());
        assertArrayEquals(new int[]{0, 1}, telemetry.copy().fallbackPeriods());
    }

    @Test
    void concentratedSingularFallbackRoutesToUnivariateFilter() {
        ModelFixture model = buildSingularMultivariateModel();
        InitialState initialState = InitialState.known(new double[]{0.0}, new double[]{1.0});

        FilterResult expected = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder()
                .method(FilterMethod.UNIVARIATE)
                .concentratedLikelihood(true)
                .build());
        FilterResult actual = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder()
                .method(FilterMethod.CONVENTIONAL)
                .concentratedLikelihood(true)
                .singularFallbackPolicy(FilterOptions.SingularFallback.UNIVARIATE)
                .build());

        assertFilterOutputsEqual(expected, actual);
        assertEquals(expected.scale(), actual.scale(), TOL);
        assertEquals(expected.scaleObservationCount(), actual.scaleObservationCount());
        assertArrayEquals(active(expected.scaleObs, 0, expected.scaleObsLength()),
            active(actual.scaleObs, 0, actual.scaleObsLength()), TOL);
        assertArrayEquals(active(expected.scaleObsCount, 0, expected.scaleObsCountLength()),
            active(actual.scaleObsCount, 0, actual.scaleObsCountLength()));
    }

    @Test
    void univariateFilterDiagonalizesCorrelatedObservationCovariance() {
        ModelFixture model = buildMultivariateModel();
        for (int t = 0; t < model.nobs; t++) {
            model.setObsCov(new double[]{0.6, 0.2, 0.2, 0.8}, t);
        }
        InitialState initialState = InitialState.known(
            new double[]{0.0, 0.0},
            new double[]{1.0, 0.1, 0.1, 1.2});

        FilterResult expected = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build());
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();
        FilterResult actual = KalmanEngine.filterBorrowedUnsafe(model,
            initialState,
            workspace,
            FilterOptions.builder().method(FilterMethod.UNIVARIATE).build());

        assertFilterStateAndLikelihoodEqual(expected, actual);
        assertTrue(workspace.retainedUnivariateScratchDoubleCount() > 0);
    }

    @Test
    void univariateFilterDiagonalizesCorrelatedObservationCovarianceWithMissingData() {
        ModelFixture model = buildMultivariateModel();
        for (int t = 0; t < model.nobs; t++) {
            model.setObsCov(new double[]{0.7, 0.18, 0.18, 0.9}, t);
        }
        model.setMissing(new boolean[]{true, false}, 1);
        model.setMissing(new boolean[]{false, true}, 2);
        InitialState initialState = InitialState.known(
            new double[]{0.0, 0.0},
            new double[]{1.0, 0.1, 0.1, 1.2});

        FilterResult expected = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder().method(FilterMethod.UNIVARIATE).build());

        assertFilterStateAndLikelihoodEqual(expected, actual);
    }

    @Test
    void multivariateUnivariateRouteAcceptsDiagonalObservationCovariance() {
        ModelFixture model = buildMultivariateModel();
        InitialState initialState = InitialState.known(
            new double[]{0.0, 0.0},
            new double[]{1.0, 0.1, 0.1, 1.2});

        FilterResult expected = KalmanEngine.filterUnivariate(model,
            initialState,
            FilterOptions.defaults());
        FilterResult actual = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder()
                .method(FilterMethod.UNIVARIATE)

                .build());

        assertFilterOutputsEqual(expected, actual);
    }

    @Test
    void multivariateUnivariateRouteHandlesPartialMissingObservations() {
        ModelFixture model = buildMultivariateModel();
        model.setMissing(new boolean[]{true, false}, 1);
        InitialState initialState = InitialState.known(
            new double[]{0.0, 0.0},
            new double[]{1.0, 0.1, 0.1, 1.2});

        FilterResult expected = KalmanEngine.filterUnivariate(model,
            initialState,
            FilterOptions.defaults());
        FilterResult actual = KalmanEngine.filter(model,
            initialState,
            FilterOptions.builder()
                .method(FilterMethod.UNIVARIATE)

                .build());

        assertFilterOutputsEqual(expected, actual);
    }

    @Test
    void singularFallbackRejectsNonPositiveDefiniteCorrelatedObservationCovariance() {
        ModelFixture model = buildSingularMultivariateModel();
        model.setObsCov(new double[]{0.0, 0.1, 0.1, 0.0}, 0);

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
            () -> KalmanEngine.filter(model,
                InitialState.known(new double[]{0.0}, new double[]{1.0}),
                FilterOptions.builder()
                    .method(FilterMethod.CONVENTIONAL)
                    .singularFallbackPolicy(FilterOptions.SingularFallback.UNIVARIATE)
                    .build()));

        assertTrue(failure.getMessage().contains("positive definite active observation covariance"));
    }

    @Test
    void initialFilteredTimingAdvancesInitialStateBeforeFiltering() {
        ModelFixture model = buildScalarModel();

        FilterResult expected = KalmanEngine.filter(model,
            InitialState.known(new double[]{1.7}, new double[]{2.17}),
            FilterOptions.builder().build());
        FilterResult actual = KalmanEngine.filter(model,
            InitialState.known(new double[]{2.0}, new double[]{3.0}),
            FilterOptions.builder()
                .timing(FilterOptions.Timing.INIT_FILTERED)

                .build());

        assertFilterOutputsEqual(expected, actual);
    }

    @Test
    void initialFilteredTimingAdvancesExactDiffuseInitialization() {
        ModelFixture model = buildScalarModel();
        InitialState predictedDiffuse = InitialState.resolved(
            1,
            new double[]{0.1},
            new double[]{0.25},
            new double[]{0.64});
        FilterOptions conventionalOptions = FilterOptions.builder()
            .method(FilterMethod.CONVENTIONAL)

            .build();
        FilterOptions univariateOptions = conventionalOptions.toBuilder()
            .method(FilterMethod.UNIVARIATE)
            .build();

        FilterResult expectedConventional = KalmanEngine.filter(model, predictedDiffuse, conventionalOptions);
        FilterResult actualConventional = KalmanEngine.filter(model,
            InitialState.diffuse(1),
            conventionalOptions.toBuilder().timing(FilterOptions.Timing.INIT_FILTERED).build());
        FilterResult expectedUnivariate = KalmanEngine.filter(model, predictedDiffuse, univariateOptions);
        FilterResult actualUnivariate = KalmanEngine.filter(model,
            InitialState.diffuse(1),
            univariateOptions.toBuilder().timing(FilterOptions.Timing.INIT_FILTERED).build());

        assertFilterOutputsEqual(expectedConventional, actualConventional);
        assertFilterOutputsEqual(expectedUnivariate, actualUnivariate);
        assertEquals(expectedConventional.nobsDiffuse(), actualConventional.nobsDiffuse());
        assertEquals(expectedUnivariate.nobsDiffuse(), actualUnivariate.nobsDiffuse());
    }

    @Test
    void conventionalConcentratedLikelihoodMatchesScaledUnconcentratedModel() {
        ModelFixture model = buildMultivariateModel();
        double[] initialMean = {0.0, 0.0};
        double[] initialCovariance = {1.0, 0.1, 0.1, 1.2};
        InitialState initialState = InitialState.known(initialMean, initialCovariance);
        FilterOptions concentratedOptions = FilterOptions.builder()
            .method(FilterMethod.CONVENTIONAL)
            .retain(FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR)
            .concentratedLikelihood(true)
            .build();

        FilterResult concentrated = KalmanEngine.filter(model, initialState, concentratedOptions);
        double scale = concentrated.scale();
        ModelFixture scaledModel = scaledCovarianceModel(model, scale);
        InitialState scaledInitialState = InitialState.known(initialMean, scaled(initialCovariance, scale));
        FilterResult expected = KalmanEngine.filter(scaledModel, scaledInitialState,
            concentratedOptions.toBuilder().concentratedLikelihood(false).build());

        assertTrue(concentrated.concentratedLikelihood());
        assertTrue(scale > 0.0);
        assertEquals(model.kEndog * model.nobs, concentrated.scaleObservationCount());
        assertFilterOutputsEqual(expected, concentrated);
        assertArrayEquals(active(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            active(concentrated.logLikelihoodObs, concentrated.logLikelihoodObsBase(), concentrated.logLikelihoodObsLength()), TOL);
        assertArrayEquals(active(expected.standardizedForecastError,
                expected.standardizedForecastErrorBase(), expected.standardizedForecastErrorLength()),
            active(concentrated.standardizedForecastError,
                concentrated.standardizedForecastErrorBase(), concentrated.standardizedForecastErrorLength()), TOL);
    }

    @Test
    void conventionalConcentratedLikelihoodHonorsMissingObservationDenominator() {
        ModelFixture model = buildMultivariateModel();
        model.setMissing(new boolean[]{true, false}, 1);
        model.setMissing(new boolean[]{true, true}, 3);
        double[] initialMean = {0.0, 0.0};
        double[] initialCovariance = {1.0, 0.1, 0.1, 1.2};
        InitialState initialState = InitialState.known(initialMean, initialCovariance);
        FilterOptions concentratedOptions = FilterOptions.builder()
            .method(FilterMethod.CONVENTIONAL)
            .concentratedLikelihood(true)
            .build();

        FilterResult concentrated = KalmanEngine.filter(model, initialState, concentratedOptions);
        double scale = concentrated.scale();
        ModelFixture scaledModel = scaledCovarianceModel(model, scale);
        InitialState scaledInitialState = InitialState.known(initialMean, scaled(initialCovariance, scale));
        FilterResult expected = KalmanEngine.filter(scaledModel, scaledInitialState,
            concentratedOptions.toBuilder().concentratedLikelihood(false).build());

        assertEquals(5, concentrated.scaleObservationCount());
        assertFilterOutputsEqual(expected, concentrated);
        assertArrayEquals(active(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            active(concentrated.logLikelihoodObs, concentrated.logLikelihoodObsBase(), concentrated.logLikelihoodObsLength()), TOL);
    }

    @Test
    void concentratedLikelihoodBurnControlsScaleDenominator() {
        ModelFixture model = buildMultivariateModel();
        double[] initialMean = {0.0, 0.0};
        double[] initialCovariance = {1.0, 0.1, 0.1, 1.2};
        InitialState initialState = InitialState.known(initialMean, initialCovariance);
        int burn = 2;
        FilterOptions concentratedOptions = FilterOptions.builder()
            .method(FilterMethod.CONVENTIONAL)
            .concentratedLikelihood(true)
            .concentratedLikelihoodBurn(burn)
            .build();

        FilterResult concentrated = KalmanEngine.filter(model, initialState, concentratedOptions);
        double scale = concentrated.scale();
        ModelFixture scaledModel = scaledCovarianceModel(model, scale);
        InitialState scaledInitialState = InitialState.known(initialMean, scaled(initialCovariance, scale));
        FilterResult expected = KalmanEngine.filter(scaledModel, scaledInitialState,
            concentratedOptions.toBuilder().concentratedLikelihood(false).build());

        assertEquals((model.nobs - burn) * model.kEndog, concentrated.scaleObservationCount());
        assertFilterOutputsEqual(expected, concentrated);
        assertArrayEquals(active(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            active(concentrated.logLikelihoodObs, concentrated.logLikelihoodObsBase(), concentrated.logLikelihoodObsLength()), TOL);
    }

    @Test
    void univariateConcentratedLikelihoodMatchesScaledUnconcentratedModel() {
        ModelFixture model = buildMultivariateModel();
        double[] initialMean = {0.0, 0.0};
        double[] initialCovariance = {1.0, 0.1, 0.1, 1.2};
        InitialState initialState = InitialState.known(initialMean, initialCovariance);
        FilterOptions concentratedOptions = FilterOptions.builder()
            .method(FilterMethod.UNIVARIATE)
            .retain(FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR)
            .concentratedLikelihood(true)
            .build();

        FilterResult concentrated = KalmanEngine.filter(model, initialState, concentratedOptions);
        double scale = concentrated.scale();
        ModelFixture scaledModel = scaledCovarianceModel(model, scale);
        InitialState scaledInitialState = InitialState.known(initialMean, scaled(initialCovariance, scale));
        FilterResult expected = KalmanEngine.filter(scaledModel, scaledInitialState,
            concentratedOptions.toBuilder().concentratedLikelihood(false).build());

        assertTrue(concentrated.concentratedLikelihood());
        assertTrue(scale > 0.0);
        assertEquals(model.kEndog * model.nobs, concentrated.scaleObservationCount());
        assertEquals(model.nobs, concentrated.scaleObsCountLength());
        assertFilterOutputsEqual(expected, concentrated);
        assertArrayEquals(active(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            active(concentrated.logLikelihoodObs, concentrated.logLikelihoodObsBase(), concentrated.logLikelihoodObsLength()), TOL);
        assertArrayEquals(active(expected.standardizedForecastError,
                expected.standardizedForecastErrorBase(), expected.standardizedForecastErrorLength()),
            active(concentrated.standardizedForecastError,
                concentrated.standardizedForecastErrorBase(), concentrated.standardizedForecastErrorLength()), TOL);
    }

    @Test
    void univariateConcentratedLikelihoodHandlesCorrelatedObservationCovariance() {
        ModelFixture model = buildMultivariateModel();
        for (int t = 0; t < model.nobs; t++) {
            model.setObsCov(new double[]{0.6, 0.2, 0.2, 0.8}, t);
        }
        InitialState initialState = InitialState.known(
            new double[]{0.0, 0.0},
            new double[]{1.0, 0.1, 0.1, 1.2});
        FilterOptions baseOptions = FilterOptions.builder()
            .concentratedLikelihood(true)
            .build();

        FilterResult expected = KalmanEngine.filter(model, initialState,
            baseOptions.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model, initialState,
            baseOptions.toBuilder().method(FilterMethod.UNIVARIATE).build());

        assertEquals(expected.scale(), actual.scale(), TOL);
        assertEquals(expected.scaleObservationCount(), actual.scaleObservationCount());
        assertFilterStateAndLikelihoodEqual(expected, actual);
    }

    @Test
    void univariateConcentratedLikelihoodExcludesSingularScalarObservations() {
        ModelFixture model = new ModelFixture(2, 1, 1, 3);
        for (int t = 0; t < model.nobs; t++) {
            model.setDesign(new double[]{1.0, 0.0}, t);
            model.setObsIntercept(new double[]{0.0, 0.0}, t);
            model.setObsCov(new double[]{0.7, 0.0, 0.0, 0.0}, t);
            model.setTransition(new double[]{0.6}, t);
            model.setStateIntercept(new double[]{0.03}, t);
            model.setSelection(new double[]{1.0}, t);
            model.setStateCov(new double[]{0.2}, t);
            model.setEndog(new double[]{0.5 + 0.1 * t, -3.0 + t}, t);
        }
        double[] initialMean = {0.0};
        double[] initialCovariance = {1.0};
        InitialState initialState = InitialState.known(initialMean, initialCovariance);
        FilterOptions concentratedOptions = FilterOptions.builder()
            .method(FilterMethod.UNIVARIATE)
            .concentratedLikelihood(true)
            .build();

        FilterResult concentrated = KalmanEngine.filter(model, initialState, concentratedOptions);
        double scale = concentrated.scale();
        ModelFixture scaledModel = scaledCovarianceModel(model, scale);
        InitialState scaledInitialState = InitialState.known(initialMean, scaled(initialCovariance, scale));
        FilterResult expected = KalmanEngine.filter(scaledModel, scaledInitialState,
            concentratedOptions.toBuilder().concentratedLikelihood(false).build());

        assertEquals(model.nobs, concentrated.scaleObservationCount());
        for (int t = 0; t < model.nobs; t++) {
            assertEquals(1, concentrated.scaleObsCount(t));
        }
        assertFilterOutputsEqual(expected, concentrated);
        assertArrayEquals(active(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            active(concentrated.logLikelihoodObs, concentrated.logLikelihoodObsBase(), concentrated.logLikelihoodObsLength()), TOL);
    }

    @Test
    void exactDiffuseConcentratedLikelihoodMatchesScaledUnconcentratedModel() {
        ModelFixture model = buildScalarModel();
        InitialState initialState = InitialState.diffuse(1);
        FilterOptions concentratedOptions = FilterOptions.builder()
            .method(FilterMethod.CONVENTIONAL)
            .retain(FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR)
            .concentratedLikelihood(true)
            .build();

        FilterResult concentrated = KalmanEngine.filter(model, initialState, concentratedOptions);
        double scale = concentrated.scale();
        ModelFixture scaledModel = scaledCovarianceModel(model, scale);
        FilterResult expected = KalmanEngine.filter(scaledModel, initialState,
            concentratedOptions.toBuilder().concentratedLikelihood(false).build());

        assertTrue(concentrated.concentratedLikelihood());
        assertEquals(1, concentrated.nobsDiffuse());
        assertEquals(model.nobs - concentrated.nobsDiffuse(), concentrated.scaleObservationCount());
        assertEquals(0, concentrated.scaleObsCount(0));
        assertFilterOutputsEqual(expected, concentrated);
        assertArrayEquals(active(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            active(concentrated.logLikelihoodObs, concentrated.logLikelihoodObsBase(), concentrated.logLikelihoodObsLength()), TOL);
        assertArrayEquals(activeOrEmpty(expected.forecastErrorDiffuseCov,
                expected.forecastErrorDiffuseCovBase(), expected.forecastErrorDiffuseCovLength()),
            activeOrEmpty(concentrated.forecastErrorDiffuseCov,
                concentrated.forecastErrorDiffuseCovBase(), concentrated.forecastErrorDiffuseCovLength()), TOL);
    }

    @Test
    void exactDiffuseConcentratedLikelihoodAdjustsFiniteDiffuseUpdatesWithoutDenominator() {
        ModelFixture model = new ModelFixture(2, 1, 1, 3);
        for (int t = 0; t < model.nobs; t++) {
            model.setDesign(new double[]{1.0, 0.0}, t);
            model.setObsIntercept(new double[]{0.0, 0.0}, t);
            model.setObsCov(new double[]{0.6, 0.0, 0.0, 0.9}, t);
            model.setTransition(new double[]{0.5}, t);
            model.setStateIntercept(new double[]{0.02}, t);
            model.setSelection(new double[]{1.0}, t);
            model.setStateCov(new double[]{0.15}, t);
            model.setEndog(new double[]{0.4 + 0.1 * t, -0.3 + 0.05 * t}, t);
        }
        InitialState initialState = InitialState.diffuse(1);
        FilterOptions concentratedOptions = FilterOptions.builder()
            .method(FilterMethod.UNIVARIATE)
            .concentratedLikelihood(true)
            .build();

        FilterResult concentrated = KalmanEngine.filter(model, initialState, concentratedOptions);
        double scale = concentrated.scale();
        ModelFixture scaledModel = scaledCovarianceModel(model, scale);
        FilterResult expected = KalmanEngine.filter(scaledModel, initialState,
            concentratedOptions.toBuilder().concentratedLikelihood(false).build());

        assertEquals(1, concentrated.nobsDiffuse());
        assertEquals(2 * (model.nobs - concentrated.nobsDiffuse()), concentrated.scaleObservationCount());
        assertEquals(1, concentrated.scaleObsCount(0));
        for (int t = concentrated.nobsDiffuse(); t < model.nobs; t++) {
            assertEquals(2, concentrated.scaleObsCount(t));
        }
        assertFilterOutputsEqual(expected, concentrated);
        assertArrayEquals(active(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            active(concentrated.logLikelihoodObs, concentrated.logLikelihoodObsBase(), concentrated.logLikelihoodObsLength()), TOL);
    }

    @Test
    void unsupportedConcentratedLikelihoodRoutesFailClearly() {
        InitialState known = InitialState.known(new double[]{0.0}, new double[]{1.0});

        UnsupportedOperationException collapsedFailure = assertThrows(UnsupportedOperationException.class,
            () -> KalmanEngine.filter(buildCollapsedEligibleModel(),
                InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0, 0.0, 1.0}),
                FilterOptions.builder()
                    .method(FilterMethod.COLLAPSED)
                    .concentratedLikelihood(true)
                    .build()));
        assertTrue(collapsedFailure.getMessage().contains("does not support concentrated likelihood"));

        assertDoesNotThrow(
            () -> KalmanEngine.filter(buildScalarModel(), known,
                FilterOptions.builder()
                    .concentratedLikelihood(true)
                    .singularFallbackPolicy(FilterOptions.SingularFallback.UNIVARIATE)
                    .build()));
    }

    @Test
    void smootherEngineMatchesKernelSmootherAndUsesPools() {
        ModelFixture model = buildScalarModel();
        InitialState kernelInit = InitialState.known(new double[]{0.0}, new double[]{1.0});
        SmootherOptions kernelOptions = SmootherOptions.conventional().only(SmootherOptions.Surface.STATE);
        FilterResult expectedFilter = KalmanEngine.filter(model, kernelInit, kernelOptions.requiredFilterOptions());
        SmootherResult expected = SmootherEngine.smooth(model, expectedFilter, kernelOptions);
        SmootherEngine.Workspace workspace = SmootherEngine.workspace();

        SmootherResult actual = SmootherEngine.smooth(
            model,
            kernelInit,
            workspace,
            SmootherOptions.builder().retainOnly(SmootherOptions.Surface.STATE).build());

        assertArrayEquals(expected.smoothedState, actual.smoothedState, TOL);
        assertEquals(0, actual.smoothedStateCov.length);
        assertEquals(0L, workspace.retainedNestedFilterWorkspaceScratchDoubleCount());
        assertEquals(0L, workspace.retainedNestedFilterWorkspaceResultDoubleCount());
        assertTrue(workspace.retainedSmootherScratchDoubleCount() > 0);
    }

    @Test
    void smootherEngineRoutesStateAutocovarianceSurface() {
        ModelFixture model = buildScalarModel();
        InitialState init = InitialState.known(new double[]{0.0}, new double[]{1.0});
        SmootherOptions options = SmootherOptions.conventional().only(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);
        FilterResult expectedFilter = KalmanEngine.filter(model, init, options.requiredFilterOptions());
        SmootherResult expected = SmootherEngine.smooth(model, expectedFilter, options);

        SmootherResult actual = SmootherEngine.smooth(model,
            init,
            options);

        assertEquals(0, actual.smoothedStateLength());
        assertEquals(0, actual.smoothedStateCovLength());
        assertArrayEquals(active(expected.smoothedStateAutocovariance,
                expected.smoothedStateAutocovarianceBase(), expected.smoothedStateAutocovarianceLength()),
            active(actual.smoothedStateAutocovariance,
                actual.smoothedStateAutocovarianceBase(), actual.smoothedStateAutocovarianceLength()), TOL);
    }

    @Test
    void smootherEngineRoutesClassicalMethodThroughOptions() {
        assertSmootherEngineMatchesKernel(
            SmootherOptions.classical().only(SmootherOptions.Surface.STATE),
            SmootherOptions.builder()
                .method(SmootherMethod.CLASSICAL)
                .retainOnly(SmootherOptions.Surface.STATE)
                .build());
    }

    @Test
    void smootherEngineRoutesAlternativeDisturbanceMemoryThroughOptions() {
        assertSmootherEngineMatchesKernel(
            SmootherOptions.alternative().disturbanceOnly(),
            SmootherOptions.builder()
                .method(SmootherMethod.ALTERNATIVE)
                .retainOnly(SmootherOptions.Surface.DISTURBANCE)
                .build());
    }

    @Test
    void smootherEngineRoutesUnivariateMethodThroughOptions() {
        ModelFixture model = buildMultivariateModel();
        InitialState init = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0, 0.0, 1.0});
        SmootherOptions options = SmootherOptions.univariate().only(SmootherOptions.Surface.STATE);
        FilterResult expectedFilter = KalmanEngine.filterUnivariate(model, init, options.requiredFilterOptions());
        SmootherResult expected = SmootherEngine.smooth(model, expectedFilter, options);
        SmootherEngine.Workspace workspace = SmootherEngine.workspace();

        SmootherResult actual = SmootherEngine.smooth(model,
            init,
            workspace,
            options);

        assertTrue(expectedFilter.perObsKalmanGain);
        assertArrayEquals(active(expected.smoothedState, expected.smoothedStateBase(), expected.smoothedStateLength()),
            active(actual.smoothedState, actual.smoothedStateBase(), actual.smoothedStateLength()), TOL);
        assertEquals(0, actual.smoothedStateCov.length);
        assertEquals(0L, workspace.retainedNestedFilterWorkspaceScratchDoubleCount());
        assertEquals(0L, workspace.retainedNestedFilterWorkspaceResultDoubleCount());
        assertTrue(workspace.retainedSmootherScratchDoubleCount() > 0);
    }

    @Test
    void smootherEngineUnivariateStateAutocovarianceMatchesConventionalRoute() {
        ModelFixture model = buildScalarModel();
        InitialState init = InitialState.known(new double[]{0.0}, new double[]{1.0});
        SmootherOptions conventional = SmootherOptions.conventional().only(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);
        SmootherOptions univariate = SmootherOptions.univariate().only(SmootherOptions.Surface.STATE_AUTOCOVARIANCE);

        SmootherResult expected = SmootherEngine.smooth(model,
            init,
            conventional);
        SmootherResult actual = SmootherEngine.smooth(model,
            init,
            univariate);

        assertEquals(0, actual.smoothedStateLength());
        assertEquals(0, actual.smoothedStateCovLength());
        assertArrayEquals(active(expected.smoothedStateAutocovariance,
                expected.smoothedStateAutocovarianceBase(), expected.smoothedStateAutocovarianceLength()),
            active(actual.smoothedStateAutocovariance,
                actual.smoothedStateAutocovarianceBase(), actual.smoothedStateAutocovarianceLength()), TOL);
    }

    @Test
    void smootherEngineUnivariateExactDiffuseMatchesManualPath() {
        ModelFixture model = buildScalarModel();
        InitialState init = InitialState.diffuse();
        SmootherOptions options = SmootherOptions.univariate().only(SmootherOptions.Surface.STATE);
        FilterResult expectedFilter = KalmanEngine.filterUnivariate(model, init, options.requiredFilterOptions());
        SmootherResult expected = SmootherEngine.smooth(model, expectedFilter, options);

        SmootherResult actual = SmootherEngine.smooth(model,
            init,
            options);

        assertTrue(expectedFilter.perObsKalmanGain);
        assertArrayEquals(active(expected.smoothedState, expected.smoothedStateBase(), expected.smoothedStateLength()),
            active(actual.smoothedState, actual.smoothedStateBase(), actual.smoothedStateLength()), TOL);
    }

    @Test
    void smootherEngineUnivariateRejectsNonDiagonalObservationCovariance() {
        ModelFixture model = buildMultivariateModel();
        for (int t = 0; t < model.nobs; t++) {
            model.setObsCov(new double[]{0.6, 0.1, 0.1, 0.8}, t);
        }
        InitialState init = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0, 0.0, 1.0});

        UnsupportedOperationException failure = assertThrows(UnsupportedOperationException.class,
            () -> SmootherEngine.smooth(model, init, SmootherOptions.univariate()));

        assertTrue(failure.getMessage().contains("UNIVARIATE smoothing requires diagonal observation covariance"));
    }

    @Test
    void kalmanSmootherUnivariateMethodRequiresUnivariateFilterResult() {
        ModelFixture model = buildMultivariateModel();
        InitialState init = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0, 0.0, 1.0});
        FilterResult conventionalFilter = KalmanEngine.filter(model, init, SmootherOptions.conventional().requiredFilterOptions());

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
            () -> SmootherEngine.smooth(model, conventionalFilter, SmootherOptions.univariate()));

        assertTrue(failure.getMessage().contains("Univariate smoothing requires a univariate FilterResult"));
    }

    private static void assertFilterOutputsEqual(FilterResult expected, FilterResult actual) {
        assertEquals(expected.logLikelihood(), actual.logLikelihood(), TOL);
        assertArrayEquals(active(expected.forecastError, expected.forecastErrorBase(), expected.forecastErrorLength()),
            active(actual.forecastError, actual.forecastErrorBase(), actual.forecastErrorLength()), TOL);
        assertArrayEquals(active(expected.forecastErrorCov, expected.forecastErrorCovBase(), expected.forecastErrorCovLength()),
            active(actual.forecastErrorCov, actual.forecastErrorCovBase(), actual.forecastErrorCovLength()), TOL);
        assertArrayEquals(active(expected.predictedState, expected.predictedStateBase(), expected.predictedStateLength()),
            active(actual.predictedState, actual.predictedStateBase(), actual.predictedStateLength()), TOL);
        assertArrayEquals(active(expected.filteredState, expected.filteredStateBase(), expected.filteredStateLength()),
            active(actual.filteredState, actual.filteredStateBase(), actual.filteredStateLength()), TOL);
    }

    private static void assertFilterStateAndLikelihoodEqual(FilterResult expected, FilterResult actual) {
        assertEquals(expected.logLikelihood(), actual.logLikelihood(), TOL);
        assertArrayEquals(active(expected.logLikelihoodObs,
                expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            active(actual.logLikelihoodObs,
                actual.logLikelihoodObsBase(), actual.logLikelihoodObsLength()), TOL);
        assertArrayEquals(active(expected.predictedState, expected.predictedStateBase(), expected.predictedStateLength()),
            active(actual.predictedState, actual.predictedStateBase(), actual.predictedStateLength()), TOL);
        assertArrayEquals(active(expected.predictedStateCov, expected.predictedStateCovBase(), expected.predictedStateCovLength()),
            active(actual.predictedStateCov, actual.predictedStateCovBase(), actual.predictedStateCovLength()), TOL);
        assertArrayEquals(active(expected.filteredState, expected.filteredStateBase(), expected.filteredStateLength()),
            active(actual.filteredState, actual.filteredStateBase(), actual.filteredStateLength()), TOL);
        assertArrayEquals(active(expected.filteredStateCov, expected.filteredStateCovBase(), expected.filteredStateCovLength()),
            active(actual.filteredStateCov, actual.filteredStateCovBase(), actual.filteredStateCovLength()), TOL);
    }

    private static void assertCollapsedFilterOutputsEqual(FilterResult expected, FilterResult actual) {
        assertEquals(expected.logLikelihood(), actual.logLikelihood(), TOL);
        assertCollapsedFilterRetainedSurfacesEqual(expected, actual);
    }

    private static FilterResult assertChandrasekharMatchesConventional(KalmanSSM model, FilterOptions profile) {
        InitialState init = InitialState.stationary(model);
        FilterResult expected = KalmanEngine.filter(model, init,
            profile.toBuilder().method(FilterMethod.CONVENTIONAL).build());
        FilterResult actual = KalmanEngine.filter(model, init,
            profile.toBuilder().method(FilterMethod.CHANDRASEKHAR).build());

        if (profile.includes(FilterOptions.Surface.LIKELIHOOD)) {
            assertEquals(expected.logLikelihood(), actual.logLikelihood(), CHANDRASEKHAR_BREADTH_TOL);
        }
        assertFilterRetainedSurfacesEqual(expected, actual, CHANDRASEKHAR_BREADTH_TOL);
        assertEquals(expected.converged(), actual.converged());
        assertEquals(expected.periodConverged(), actual.periodConverged());
        return actual;
    }

    private static void assertCollapsedFilterRetainedSurfacesEqual(FilterResult expected, FilterResult actual) {
        assertFilterRetainedSurfacesEqual(expected, actual, TOL);
    }

    private static void assertCollapsedWhitenedSurfacesEqual(FilterResult expected,
                                                            FilterResult actual,
                                                            double logDetFactor) {
        assertArrayEquals(active(expected.forecast, expected.forecastBase(), expected.forecastLength()),
            active(actual.forecast, actual.forecastBase(), actual.forecastLength()), TOL);
        assertArrayEquals(active(expected.forecastError, expected.forecastErrorBase(), expected.forecastErrorLength()),
            active(actual.forecastError, actual.forecastErrorBase(), actual.forecastErrorLength()), TOL);
        assertArrayEquals(active(expected.forecastErrorCov, expected.forecastErrorCovBase(), expected.forecastErrorCovLength()),
            active(actual.forecastErrorCov, actual.forecastErrorCovBase(), actual.forecastErrorCovLength()), TOL);
        assertArrayEquals(active(expected.standardizedForecastError, expected.standardizedForecastErrorBase(), expected.standardizedForecastErrorLength()),
            active(actual.standardizedForecastError, actual.standardizedForecastErrorBase(), actual.standardizedForecastErrorLength()), TOL);
        assertArrayEquals(active(expected.predictedState, expected.predictedStateBase(), expected.predictedStateLength()),
            active(actual.predictedState, actual.predictedStateBase(), actual.predictedStateLength()), TOL);
        assertArrayEquals(active(expected.predictedStateCov, expected.predictedStateCovBase(), expected.predictedStateCovLength()),
            active(actual.predictedStateCov, actual.predictedStateCovBase(), actual.predictedStateCovLength()), TOL);
        assertArrayEquals(active(expected.filteredState, expected.filteredStateBase(), expected.filteredStateLength()),
            active(actual.filteredState, actual.filteredStateBase(), actual.filteredStateLength()), TOL);
        assertArrayEquals(active(expected.filteredStateCov, expected.filteredStateCovBase(), expected.filteredStateCovLength()),
            active(actual.filteredStateCov, actual.filteredStateCovBase(), actual.filteredStateCovLength()), TOL);
        assertArrayEquals(active(expected.kalmanGain, expected.kalmanGainBase(), expected.kalmanGainLength()),
            active(actual.kalmanGain, actual.kalmanGainBase(), actual.kalmanGainLength()), TOL);
        assertArrayEquals(activeOrEmpty(expected.predictedDiffuseStateCov,
            expected.predictedDiffuseStateCovBase(), expected.predictedDiffuseStateCovLength()),
            activeOrEmpty(actual.predictedDiffuseStateCov,
            actual.predictedDiffuseStateCovBase(), actual.predictedDiffuseStateCovLength()), TOL);
        assertArrayEquals(activeOrEmpty(expected.forecastErrorDiffuseCov,
            expected.forecastErrorDiffuseCovBase(), expected.forecastErrorDiffuseCovLength()),
            activeOrEmpty(actual.forecastErrorDiffuseCov,
            actual.forecastErrorDiffuseCovBase(), actual.forecastErrorDiffuseCovLength()), TOL);
        assertEquals(expected.nobsDiffuse(), actual.nobsDiffuse());
        assertEquals(expected.logLikelihoodObsLength(), actual.logLikelihoodObsLength());
        for (int t = 0; t < expected.nobs; t++) {
            if (expected.logLikelihoodObsLength() > 0) {
                assertEquals(expected.logLikelihoodObs[expected.logLikelihoodObsOffset(t)] - logDetFactor,
                    actual.logLikelihoodObs[actual.logLikelihoodObsOffset(t)], TOL);
            }
        }
    }

    private static void assertCollapsedNonDiagonalObservationSurfaces(KalmanSSM model,
                                                                     FilterResult history,
                                                                     FilterResult actual) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] expectedMean = new double[kEndog];
        double[] expectedError = new double[kEndog];
        double[] expectedCovariance = new double[kEndog * kEndog];
        double[] expectedDiffuseCovariance = new double[kEndog * kEndog];
        double[] expectedStandardized = new double[kEndog];
        for (int t = 0; t < model.observationCount(); t++) {
            computeExpectedObservationSurfaces(model, history, t, expectedMean, expectedError,
                expectedCovariance, expectedDiffuseCovariance);
            if (actual.forecastLength() > 0) {
                assertArrayEquals(expectedMean,
                    Arrays.copyOfRange(actual.forecast, actual.forecastOffset(t), actual.forecastOffset(t) + kEndog), TOL);
            }
            if (actual.forecastErrorLength() > 0) {
                assertArrayEquals(expectedError,
                    Arrays.copyOfRange(actual.forecastError, actual.forecastErrorOffset(t), actual.forecastErrorOffset(t) + kEndog), TOL);
            }
            if (actual.forecastErrorCovLength() > 0) {
                assertArrayEquals(expectedCovariance,
                    Arrays.copyOfRange(actual.forecastErrorCov, actual.forecastErrorCovOffset(t),
                        actual.forecastErrorCovOffset(t) + kEndog * kEndog), TOL);
            }
            if (actual.forecastErrorDiffuseCovLength() > 0) {
                assertArrayEquals(expectedDiffuseCovariance,
                    Arrays.copyOfRange(actual.forecastErrorDiffuseCov, actual.forecastErrorDiffuseCovOffset(t),
                        actual.forecastErrorDiffuseCovOffset(t) + kEndog * kEndog), TOL);
            }
            if (actual.standardizedForecastErrorLength() > 0) {
                standardizeExpectedForecastError(model, t, expectedError, expectedCovariance, expectedStandardized);
                assertArrayEquals(expectedStandardized,
                    Arrays.copyOfRange(actual.standardizedForecastError, actual.standardizedForecastErrorOffset(t),
                        actual.standardizedForecastErrorOffset(t) + kEndog), TOL);
            }
        }
        assertEquals(history.nobsDiffuse(), actual.nobsDiffuse());
        assertEquals(0, actual.kalmanGainLength());
    }

    private static void computeExpectedObservationSurfaces(KalmanSSM model,
                                                           FilterResult history,
                                                           int t,
                                                           double[] mean,
                                                           double[] error,
                                                           double[] covariance,
                                                           double[] diffuseCovariance) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        double[] obsIntercept = model.obsInterceptData();
        int interceptOffset = model.obsInterceptOffset(t);
        double[] endog = model.endogData();
        int endogOffset = model.endogOffset(t);
        int predOff = history.predictedStateOffset(t);
        int predCovOff = history.predictedStateCovOffset(t);
        int predDiffCovOff = history.predictedDiffuseStateCovOffset(t);
        Arrays.fill(diffuseCovariance, 0.0);
        for (int row = 0; row < kEndog; row++) {
            int rowOffset = designOffset + row * designLd;
            double forecastMean = obsIntercept[interceptOffset + row];
            for (int state = 0; state < kStates; state++) {
                forecastMean += design[rowOffset + state] * history.predictedState[predOff + state];
            }
            mean[row] = forecastMean;
            error[row] = model.isMissing(row, t) ? 0.0 : endog[endogOffset + row] - forecastMean;
        }
        copyModelObsCov(model, t, covariance);
        for (int row = 0; row < kEndog; row++) {
            int rowOffset = designOffset + row * designLd;
            for (int col = 0; col < kEndog; col++) {
                int colOffset = designOffset + col * designLd;
                double finiteValue = 0.0;
                double diffuseValue = 0.0;
                for (int left = 0; left < kStates; left++) {
                    double zLeft = design[rowOffset + left];
                    for (int right = 0; right < kStates; right++) {
                        double zRight = design[colOffset + right];
                        finiteValue += zLeft * history.predictedStateCov[predCovOff + left * kStates + right] * zRight;
                        diffuseValue += zLeft * history.predictedDiffuseStateCov[predDiffCovOff + left * kStates + right] * zRight;
                    }
                }
                covariance[row * kEndog + col] += finiteValue;
                diffuseCovariance[row * kEndog + col] = diffuseValue;
            }
        }
    }

    private static void copyModelObsCov(KalmanSSM model, int t, double[] target) {
        int kEndog = model.observationDimension();
        double[] obsCov = model.obsCovData();
        int obsCovOffset = model.obsCovOffset(t);
        int obsCovLd = model.obsCovLeadingDimension();
        for (int row = 0; row < kEndog; row++) {
            System.arraycopy(obsCov, obsCovOffset + row * obsCovLd, target, row * kEndog, kEndog);
        }
    }

    private static void standardizeExpectedForecastError(KalmanSSM model,
                                                         int t,
                                                         double[] error,
                                                         double[] covariance,
                                                         double[] standardized) {
        int dimension = model.observationDimension();
        int observed = 0;
        int[] observedIndex = new int[dimension];
        for (int obs = 0; obs < dimension; obs++) {
            if (!model.isMissing(obs, t)) {
                observedIndex[observed++] = obs;
            }
        }
        Arrays.fill(standardized, 0.0);
        if (observed == dimension) {
            standardizeExpectedForecastErrorBlock(error, covariance, standardized, dimension);
            return;
        }
        double[] selectedError = new double[observed];
        double[] selectedCovariance = new double[observed * observed];
        double[] selectedStandardized = new double[observed];
        for (int row = 0; row < observed; row++) {
            selectedError[row] = error[observedIndex[row]];
            for (int col = 0; col < observed; col++) {
                selectedCovariance[row * observed + col] = covariance[observedIndex[row] * dimension + observedIndex[col]];
            }
        }
        standardizeExpectedForecastErrorBlock(selectedError, selectedCovariance, selectedStandardized, observed);
        for (int obs = 0; obs < observed; obs++) {
            standardized[observedIndex[obs]] = selectedStandardized[obs];
        }
    }

    private static void standardizeExpectedForecastErrorBlock(double[] error,
                                                              double[] covariance,
                                                              double[] standardized,
                                                              int dimension) {
        double[] factor = Arrays.copyOf(covariance, dimension * dimension);
        for (int col = 0; col < dimension; col++) {
            for (int row = col; row < dimension; row++) {
                double value = factor[row * dimension + col];
                for (int inner = 0; inner < col; inner++) {
                    value -= factor[row * dimension + inner] * factor[col * dimension + inner];
                }
                if (row == col) {
                    factor[row * dimension + col] = Math.sqrt(value);
                } else {
                    factor[row * dimension + col] = value / factor[col * dimension + col];
                }
            }
            for (int row = 0; row < col; row++) {
                factor[row * dimension + col] = 0.0;
            }
        }
        for (int row = 0; row < dimension; row++) {
            double value = error[row];
            for (int col = 0; col < row; col++) {
                value -= factor[row * dimension + col] * standardized[col];
            }
            standardized[row] = value / factor[row * dimension + row];
        }
    }

    private static void assertFalsePerObservationGain(FilterResult result) {
        assertEquals(0, result.kalmanGainLength());
        assertEquals(false, result.perObsKalmanGain);
    }

    private static void assertFilterRetainedSurfacesEqual(FilterResult expected, FilterResult actual, double tolerance) {
        assertArrayEquals(active(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            active(actual.logLikelihoodObs, actual.logLikelihoodObsBase(), actual.logLikelihoodObsLength()), tolerance);
        assertArrayEquals(active(expected.forecast, expected.forecastBase(), expected.forecastLength()),
            active(actual.forecast, actual.forecastBase(), actual.forecastLength()), tolerance);
        assertArrayEquals(active(expected.forecastError, expected.forecastErrorBase(), expected.forecastErrorLength()),
            active(actual.forecastError, actual.forecastErrorBase(), actual.forecastErrorLength()), tolerance);
        assertArrayEquals(active(expected.forecastErrorCov, expected.forecastErrorCovBase(), expected.forecastErrorCovLength()),
            active(actual.forecastErrorCov, actual.forecastErrorCovBase(), actual.forecastErrorCovLength()), tolerance);
        assertArrayEquals(active(expected.standardizedForecastError, expected.standardizedForecastErrorBase(), expected.standardizedForecastErrorLength()),
            active(actual.standardizedForecastError, actual.standardizedForecastErrorBase(), actual.standardizedForecastErrorLength()), tolerance);
        assertArrayEquals(active(expected.predictedState, expected.predictedStateBase(), expected.predictedStateLength()),
            active(actual.predictedState, actual.predictedStateBase(), actual.predictedStateLength()), tolerance);
        assertArrayEquals(active(expected.predictedStateCov, expected.predictedStateCovBase(), expected.predictedStateCovLength()),
            active(actual.predictedStateCov, actual.predictedStateCovBase(), actual.predictedStateCovLength()), tolerance);
        assertArrayEquals(active(expected.filteredState, expected.filteredStateBase(), expected.filteredStateLength()),
            active(actual.filteredState, actual.filteredStateBase(), actual.filteredStateLength()), tolerance);
        assertArrayEquals(active(expected.filteredStateCov, expected.filteredStateCovBase(), expected.filteredStateCovLength()),
            active(actual.filteredStateCov, actual.filteredStateCovBase(), actual.filteredStateCovLength()), tolerance);
        assertArrayEquals(active(expected.kalmanGain, expected.kalmanGainBase(), expected.kalmanGainLength()),
            active(actual.kalmanGain, actual.kalmanGainBase(), actual.kalmanGainLength()), tolerance);
        assertArrayEquals(activeOrEmpty(expected.predictedDiffuseStateCov,
            expected.predictedDiffuseStateCovBase(), expected.predictedDiffuseStateCovLength()),
            activeOrEmpty(actual.predictedDiffuseStateCov,
            actual.predictedDiffuseStateCovBase(), actual.predictedDiffuseStateCovLength()), tolerance);
        assertArrayEquals(activeOrEmpty(expected.forecastErrorDiffuseCov,
            expected.forecastErrorDiffuseCovBase(), expected.forecastErrorDiffuseCovLength()),
            activeOrEmpty(actual.forecastErrorDiffuseCov,
            actual.forecastErrorDiffuseCovBase(), actual.forecastErrorDiffuseCovLength()), tolerance);
        assertEquals(expected.nobsDiffuse(), actual.nobsDiffuse());
    }

    private static void assertSmootherEngineMatchesKernel(SmootherOptions kernelOptions, SmootherOptions options) {
        ModelFixture model = buildScalarModel();
        InitialState kernelInit = InitialState.known(new double[]{0.0}, new double[]{1.0});
        FilterResult expectedFilter = KalmanEngine.filter(model, kernelInit, kernelOptions.requiredFilterOptions());
        SmootherResult expected = SmootherEngine.smooth(model, expectedFilter, kernelOptions);

        SmootherResult actual = SmootherEngine.smooth(model,
            kernelInit,
            options);

        assertArrayEquals(active(expected.smoothedState, expected.smoothedStateBase(), expected.smoothedStateLength()),
            active(actual.smoothedState, actual.smoothedStateBase(), actual.smoothedStateLength()), TOL);
        assertArrayEquals(active(expected.smoothedStateCov, expected.smoothedStateCovBase(), expected.smoothedStateCovLength()),
            active(actual.smoothedStateCov, actual.smoothedStateCovBase(), actual.smoothedStateCovLength()), TOL);
        assertArrayEquals(active(expected.smoothedObsDisturbance, expected.smoothedObsDisturbanceBase(), expected.smoothedObsDisturbanceLength()),
            active(actual.smoothedObsDisturbance, actual.smoothedObsDisturbanceBase(), actual.smoothedObsDisturbanceLength()), TOL);
        assertArrayEquals(active(expected.smoothedStateDisturbance, expected.smoothedStateDisturbanceBase(), expected.smoothedStateDisturbanceLength()),
            active(actual.smoothedStateDisturbance, actual.smoothedStateDisturbanceBase(), actual.smoothedStateDisturbanceLength()), TOL);
        assertArrayEquals(active(expected.smoothedObsDisturbanceCov, expected.smoothedObsDisturbanceCovBase(), expected.smoothedObsDisturbanceCovLength()),
            active(actual.smoothedObsDisturbanceCov, actual.smoothedObsDisturbanceCovBase(), actual.smoothedObsDisturbanceCovLength()), TOL);
        assertArrayEquals(active(expected.smoothedStateDisturbanceCov, expected.smoothedStateDisturbanceCovBase(), expected.smoothedStateDisturbanceCovLength()),
            active(actual.smoothedStateDisturbanceCov, actual.smoothedStateDisturbanceCovBase(), actual.smoothedStateDisturbanceCovLength()), TOL);
    }

    private static double[] active(double[] values, int base, int length) {
        return Arrays.copyOfRange(values, base, base + length);
    }

    private static int[] active(int[] values, int base, int length) {
        return Arrays.copyOfRange(values, base, base + length);
    }

    private static double[] activeOrEmpty(double[] values, int base, int length) {
        return length == 0 ? new double[0] : Arrays.copyOfRange(values, base, base + length);
    }
}
