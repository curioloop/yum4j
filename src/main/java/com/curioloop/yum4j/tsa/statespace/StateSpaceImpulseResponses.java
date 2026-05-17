package com.curioloop.yum4j.tsa.statespace;

import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;

import java.util.Arrays;

public final class StateSpaceImpulseResponses {

    private StateSpaceImpulseResponses() {
    }

    public static ImpulseResponse compute(KalmanSSM model,
                                          int steps,
                                          int impulse,
                                          boolean orthogonalized,
                                          boolean cumulative) {
        return compute(model, steps, impulse, orthogonalized, cumulative, 0);
    }

    public static ImpulseResponse compute(KalmanSSM model,
                                          int steps,
                                          int impulse,
                                          boolean orthogonalized,
                                          boolean cumulative,
                                          int anchor) {
        if (impulse < 0 || impulse >= model.stateDisturbanceCount()) {
            throw new IllegalArgumentException("impulse must be in [0, k_posdef)");
        }
        double[] impulseVector = new double[model.stateDisturbanceCount()];
        impulseVector[impulse] = 1.0;
        return compute(model, steps, impulseVector, orthogonalized, cumulative, anchor);
    }

    public static ImpulseResponse compute(KalmanSSM model,
                                          int steps,
                                          double[] impulse,
                                          boolean orthogonalized,
                                          boolean cumulative) {
        return compute(model, steps, impulse, orthogonalized, cumulative, 0);
    }

    public static ImpulseResponse compute(KalmanSSM model,
                                          int steps,
                                          double[] impulse,
                                          boolean orthogonalized,
                                          boolean cumulative,
                                          int anchor) {
        if (model == null) {
            throw new IllegalArgumentException("model must not be null");
        }
        if (model.complex()) {
            throw new IllegalArgumentException("complex state-space impulse responses are not supported");
        }
        if (steps < 0) {
            throw new IllegalArgumentException("steps must be non-negative");
        }
        int kPosdef = model.stateDisturbanceCount();
        if (impulse == null || impulse.length != kPosdef) {
            throw new IllegalArgumentException("impulse length must match k_posdef");
        }
        if (anchor < 0 || anchor >= model.observationCount()) {
            throw new IllegalArgumentException("anchor must be in [0, nobs)");
        }
        if (!timeInvariantImpulseSurfaces(model) && anchor + steps + 1 > model.observationCount()) {
            throw new IllegalArgumentException("time-varying impulse responses cannot exceed model observation count");
        }

        double[] pulse = orthogonalized ? orthogonalize(model, impulse) : impulse.clone();
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] state = selectedShock(model, anchor, pulse);
        double[][][] responses = new double[steps + 1][kEndog][1];
        double[] cumulativeValues = new double[kEndog];

        for (int step = 0; step <= steps; step++) {
            int time = Math.min(anchor + step, Math.max(0, model.observationCount() - 1));
            int designOffset = model.designOffset(time);
            for (int row = 0; row < kEndog; row++) {
                double value = 0.0;
                for (int col = 0; col < kStates; col++) {
                    value += model.designData()[designOffset + row * kStates + col] * state[col];
                }
                if (cumulative) {
                    cumulativeValues[row] += value;
                    value = cumulativeValues[row];
                }
                responses[step][row][0] = value;
            }
            if (step < steps) {
                state = transition(model, time, state);
            }
        }
        return new ImpulseResponse(responses, orthogonalized, cumulative);
    }

    private static boolean timeInvariantImpulseSurfaces(KalmanSSM model) {
        if (model.observationCount() <= 1) {
            return true;
        }
        return model.designOffset(0) == model.designOffset(1)
            && model.transitionOffset(0) == model.transitionOffset(1)
            && model.selectionOffset(0) == model.selectionOffset(1);
    }

    private static double[] orthogonalize(KalmanSSM model, double[] impulse) {
        int kPosdef = model.stateDisturbanceCount();
        double[] lower = choleskyLower(model.stateCovarianceData(), model.stateCovarianceOffset(0), kPosdef);
        double[] out = new double[kPosdef];
        for (int row = 0; row < kPosdef; row++) {
            double value = 0.0;
            for (int col = 0; col < kPosdef; col++) {
                value += lower[row * kPosdef + col] * impulse[col];
            }
            out[row] = value;
        }
        return out;
    }

    private static double[] selectedShock(KalmanSSM model, int time, double[] shock) {
        int kStates = model.stateCount();
        int kPosdef = model.stateDisturbanceCount();
        double[] state = new double[kStates];
        int selectionOffset = model.selectionOffset(time);
        for (int row = 0; row < kStates; row++) {
            double value = 0.0;
            for (int col = 0; col < kPosdef; col++) {
                value += model.selectionData()[selectionOffset + row * kPosdef + col] * shock[col];
            }
            state[row] = value;
        }
        return state;
    }

    private static double[] transition(KalmanSSM model, int time, double[] state) {
        int kStates = model.stateCount();
        double[] next = new double[kStates];
        int transitionOffset = model.transitionOffset(time);
        for (int row = 0; row < kStates; row++) {
            double value = 0.0;
            for (int col = 0; col < kStates; col++) {
                value += model.transitionData()[transitionOffset + row * kStates + col] * state[col];
            }
            next[row] = value;
        }
        return next;
    }

    private static double[] choleskyLower(double[] source, int offset, int dimension) {
        double[] lower = new double[dimension * dimension];
        Arrays.fill(lower, 0.0);
        for (int row = 0; row < dimension; row++) {
            for (int col = 0; col <= row; col++) {
                double sum = source[offset + row * dimension + col];
                for (int inner = 0; inner < col; inner++) {
                    sum -= lower[row * dimension + inner] * lower[col * dimension + inner];
                }
                if (row == col) {
                    if (sum <= 0.0 || !Double.isFinite(sum)) {
                        throw new IllegalArgumentException("state covariance is not positive definite");
                    }
                    lower[row * dimension + col] = Math.sqrt(sum);
                } else {
                    lower[row * dimension + col] = sum / lower[col * dimension + col];
                }
            }
        }
        return lower;
    }
}