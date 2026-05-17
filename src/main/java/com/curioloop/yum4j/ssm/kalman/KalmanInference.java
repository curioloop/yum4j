package com.curioloop.yum4j.ssm.kalman;

/**
 * Contract for a configured Kalman inference that can run with an optional reusable workspace.
 *
 * @param <R> result type
 * @param <W> workspace type
 */
public interface KalmanInference<R, W> {

    default R run() {
        return run(null);
    }

    R run(W workspace);
}