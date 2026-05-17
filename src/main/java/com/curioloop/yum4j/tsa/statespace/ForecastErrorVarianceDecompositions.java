package com.curioloop.yum4j.tsa.statespace;

import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;

public final class ForecastErrorVarianceDecompositions {

    private ForecastErrorVarianceDecompositions() {
    }

    public static ForecastErrorVarianceDecomposition compute(KalmanSSM snapshot, int steps) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive");
        }
        int impulses = snapshot.stateDisturbanceCount();
        int responses = snapshot.observationDimension();
        double[][][] shares = new double[steps][responses][impulses];
        double[][] totals = new double[steps][responses];
        for (int impulse = 0; impulse < impulses; impulse++) {
            ImpulseResponse response = StateSpaceImpulseResponses.compute(snapshot, steps - 1, impulse, true, false);
            for (int horizon = 0; horizon < steps; horizon++) {
                double[][] values = response.response(horizon);
                for (int row = 0; row < responses; row++) {
                    double contribution = values[row][0] * values[row][0];
                    shares[horizon][row][impulse] += contribution;
                    totals[horizon][row] += contribution;
                }
            }
        }
        for (int horizon = 0; horizon < steps; horizon++) {
            for (int row = 0; row < responses; row++) {
                for (int impulse = 0; impulse < impulses; impulse++) {
                    shares[horizon][row][impulse] = totals[horizon][row] == 0.0
                        ? Double.NaN
                        : shares[horizon][row][impulse] / totals[horizon][row];
                }
            }
        }
        return new ForecastErrorVarianceDecomposition(shares);
    }
}