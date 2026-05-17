package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;

/**
 * User-facing facade for Kalman filtering, smoothing, and simulation smoothing.
 *
 * <pre>{@code
 * FilterResult result = Kalman.filter()
 *     .model(model)
 *     .initialState(initialState)
 *     .likelihoodOnly()
 *     .run();
 *
 * try (KalmanWorkspace workspace = Kalman.workspace()) {
 *     SmootherResult smoothed = Kalman.smooth()
 *         .model(model)
 *         .initialState(initialState)
 *         .stateOnly()
 *         .run(workspace);
 * }
 * }</pre>
 */
public final class Kalman {
    private Kalman() {
    }

    public static KalmanWorkspace workspace() {
        return new KalmanWorkspace();
    }

    public static KalmanSSM.Builder model(int kEndog, int kStates, int kPosdef, int nobs) {
        return KalmanSSM.builder(kEndog, kStates, kPosdef, nobs);
    }

    public static KalmanFiltering filter() {
        return new KalmanFiltering();
    }

    public static KalmanSmoothing smooth() {
        return new KalmanSmoothing();
    }

    public static KalmanSimulation simulate() {
        return new KalmanSimulation();
    }
}