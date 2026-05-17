package com.curioloop.yum4j.tsa.prediction;

import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;

import java.util.Arrays;

public final class StatePredictionProjector {

    private StatePredictionProjector() {
    }

    public static double univariateMean(KalmanSSM snapshot,
                                        StatePredictionSurface surface,
                                        int time,
                                        boolean signalOnly) {
        int designOffset = snapshot.designOffset(time);
        int obsInterceptOffset = snapshot.obsInterceptOffset(time);
        double mean = signalOnly ? 0.0 : snapshot.obsInterceptData()[obsInterceptOffset];
        for (int state = 0; state < snapshot.stateCount(); state++) {
            mean += snapshot.designData()[designOffset + state]
                * surface.state()[surface.stateOffset() + state];
        }
        return mean;
    }

    public static double univariateVariance(KalmanSSM snapshot,
                                            StatePredictionSurface surface,
                                            int time,
                                            boolean signalOnly) {
        if (surface.covarianceOffset() < 0) {
            return Double.NaN;
        }
        int kStates = snapshot.stateCount();
        int designOffset = snapshot.designOffset(time);
        double value = 0.0;
        for (int row = 0; row < kStates; row++) {
            double left = snapshot.designData()[designOffset + row];
            for (int col = 0; col < kStates; col++) {
                value += left * surface.covariance()[surface.covarianceOffset() + row * kStates + col]
                    * snapshot.designData()[designOffset + col];
            }
        }
        return value + (signalOnly ? 0.0 : snapshot.obsCovData()[snapshot.obsCovOffset(time)]);
    }

    public static void multivariateMean(KalmanSSM snapshot,
                                        StatePredictionSurface surface,
                                        int time,
                                        boolean signalOnly,
                                        double[] mean) {
        int kEndog = snapshot.observationDimension();
        int kStates = snapshot.stateCount();
        if (mean.length != kEndog) {
            throw new IllegalArgumentException("mean length must match observation dimension");
        }
        int designOffset = snapshot.designOffset(time);
        int obsInterceptOffset = snapshot.obsInterceptOffset(time);
        for (int obs = 0; obs < kEndog; obs++) {
            double value = signalOnly ? 0.0 : snapshot.obsInterceptData()[obsInterceptOffset + obs];
            for (int state = 0; state < kStates; state++) {
                value += snapshot.designData()[designOffset + obs * kStates + state]
                    * surface.state()[surface.stateOffset() + state];
            }
            mean[obs] = value;
        }
    }

    public static void multivariateVariance(KalmanSSM snapshot,
                                            StatePredictionSurface surface,
                                            int time,
                                            boolean signalOnly,
                                            double[][] variance) {
        int kEndog = snapshot.observationDimension();
        int kStates = snapshot.stateCount();
        if (variance.length != kEndog) {
            throw new IllegalArgumentException("variance row count must match observation dimension");
        }
        if (surface.covarianceOffset() < 0) {
            for (double[] row : variance) {
                if (row.length != kEndog) {
                    throw new IllegalArgumentException("variance must be square with observation dimension");
                }
                Arrays.fill(row, Double.NaN);
            }
            return;
        }
        int designOffset = snapshot.designOffset(time);
        int obsCovOffset = snapshot.obsCovOffset(time);
        for (int left = 0; left < kEndog; left++) {
            if (variance[left].length != kEndog) {
                throw new IllegalArgumentException("variance must be square with observation dimension");
            }
            for (int right = 0; right < kEndog; right++) {
                double value = signalOnly ? 0.0 : snapshot.obsCovData()[obsCovOffset + left * kEndog + right];
                for (int row = 0; row < kStates; row++) {
                    double leftDesign = snapshot.designData()[designOffset + left * kStates + row];
                    for (int col = 0; col < kStates; col++) {
                        value += leftDesign * surface.covariance()[surface.covarianceOffset() + row * kStates + col]
                            * snapshot.designData()[designOffset + right * kStates + col];
                    }
                }
                variance[left][right] = value;
            }
        }
    }
}