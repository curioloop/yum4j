package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.arena.SmootherResultLayout;

import java.util.Arrays;

/**
 * Flat storage for smoothing outputs produced from a filter pass.
 */
public class SmootherResult extends SmootherResultBase<SmootherResult> {

    SmootherResult() {
        super();
    }

    public SmootherResult(int kEndog, int kStates, int kPosdef, int nobs) {
        this(kEndog, kStates, kPosdef, nobs, true, true, true, true, true);
    }

    public SmootherResult(int kEndog, int kStates, int kPosdef, int nobs,
                          boolean storeState,
                          boolean storeStateCovariance,
                          boolean storeDisturbance,
                          boolean storeDisturbanceCovariance,
                          boolean storeAuxiliary) {
        this(kEndog, kStates, kPosdef, nobs,
                    SmootherResultLayout.create(1, kEndog, kStates, kPosdef, nobs,
                        storeState, storeStateCovariance, storeDisturbance,
                        storeDisturbanceCovariance, storeAuxiliary));
    }

                public SmootherResult(int kEndog, int kStates, int kPosdef, int nobs, SmootherResultLayout layout) {
                super(kEndog, kStates, kPosdef, nobs, layout);
    }

    @Override
    protected int scalarWidth() {
        return 1;
    }

    @Override
    protected SmootherResult newResult(int kEndog, int kStates, int kPosdef, int nobs, SmootherResultLayout layout) {
        return new SmootherResult(kEndog, kStates, kPosdef, nobs, layout);
    }

    static double[] emptyArray() {
        return EMPTY;
    }

    public double smoothedState(int i, int t) {
        if (smoothedStateLength() == 0) {
            return 0.0;
        }
        return smoothedState[smoothedStateOffset(t) + i];
    }

    public double smoothedStateCov(int i, int j, int t) {
        if (smoothedStateCovLength() == 0) {
            return 0.0;
        }
        return smoothedStateCov[smoothedStateCovOffset(t) + i * kStates + j];
    }

    public double smoothedStateAutocovariance(int i, int j, int t) {
        if (smoothedStateAutocovarianceLength() == 0) {
            return 0.0;
        }
        return smoothedStateAutocovariance[smoothedStateAutocovarianceOffset(t) + i * kStates + j];
    }

    /**
     * Extracts Cov(alpha_t, alpha_{t-lag}) for models that embed fixed-size lag
     * blocks in the smoothed state vector.
     */
    public double[] smoothedStateAutocovariance(int lag, int blockSize) {
        return smoothedStateAutocovariance(lag, blockSize, null, null, null);
    }

    public double[] smoothedStateAutocovarianceAt(int lag, int blockSize, int time) {
        return smoothedStateAutocovariance(lag, blockSize, time, null, null);
    }

    public double[] smoothedStateAutocovarianceRange(int lag, int blockSize, int startInclusive, int endExclusive) {
        return smoothedStateAutocovariance(lag, blockSize, null, startInclusive, endExclusive);
    }

    public double[] smoothedStateAutocovariance(int lag,
                                                int blockSize,
                                                Integer time,
                                                Integer startInclusive,
                                                Integer endExclusive) {
        validateLaggedAutocovarianceRequest(lag, blockSize, time, startInclusive, endExclusive);
        int start = startInclusive == null ? 0 : startInclusive;
        int end = endExclusive == null ? nobs : endExclusive;
        if (time != null) {
            start = time;
            end = time + 1;
        }
        double[] values = new double[(end - start) * blockSize * blockSize];
        int outputOffset = 0;
        int absoluteLag = Math.abs(lag);
        for (int timeIndex = start; timeIndex < end; timeIndex++) {
            int covarianceTime = lag < 0 ? timeIndex + absoluteLag : timeIndex;
            if (covarianceTime < 0 || covarianceTime >= nobs || (lag > 0 && timeIndex < absoluteLag)) {
                Arrays.fill(values, outputOffset, outputOffset + blockSize * blockSize, Double.NaN);
            } else {
                int rowBlock = lag < 0 ? absoluteLag : 0;
                int colBlock = lag < 0 ? 0 : absoluteLag;
                copyLaggedBlock(covarianceTime, rowBlock, colBlock, blockSize, values, outputOffset);
            }
            outputOffset += blockSize * blockSize;
        }
        return values;
    }

    private void validateLaggedAutocovarianceRequest(int lag,
                                                     int blockSize,
                                                     Integer time,
                                                     Integer startInclusive,
                                                     Integer endExclusive) {
        if (smoothedStateCovLength() == 0) {
            throw new IllegalStateException("smoothed state covariance was not retained");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be positive");
        }
        if ((Math.abs(lag) + 1) * blockSize > kStates) {
            throw new IllegalArgumentException("lagged block is outside the smoothed state dimension");
        }
        if (time != null && (startInclusive != null || endExclusive != null)) {
            throw new IllegalArgumentException("Cannot specify both time and start/end");
        }
        if (time != null && time < 0) {
            throw new IllegalArgumentException("Negative time is not allowed");
        }
        if (startInclusive != null && startInclusive < 0) {
            throw new IllegalArgumentException("Negative time is not allowed");
        }
        if (endExclusive != null && endExclusive < 0) {
            throw new IllegalArgumentException("Negative time is not allowed");
        }
        int start = startInclusive == null ? 0 : startInclusive;
        int end = endExclusive == null ? nobs : endExclusive;
        if (end < start) {
            throw new IllegalArgumentException("end must be after start");
        }
        if (time != null && time >= nobs) {
            throw new IllegalArgumentException("time is outside the smoothed sample");
        }
        if (end > nobs) {
            throw new IllegalArgumentException("end is outside the smoothed sample");
        }
    }

    private void copyLaggedBlock(int covarianceTime,
                                 int rowBlock,
                                 int colBlock,
                                 int blockSize,
                                 double[] target,
                                 int targetOffset) {
        int covarianceOffset = smoothedStateCovOffset(covarianceTime);
        int rowStart = rowBlock * blockSize;
        int colStart = colBlock * blockSize;
        for (int rowIndex = 0; rowIndex < blockSize; rowIndex++) {
            int sourceOffset = covarianceOffset + (rowStart + rowIndex) * kStates + colStart;
            System.arraycopy(smoothedStateCov, sourceOffset, target, targetOffset + rowIndex * blockSize, blockSize);
        }
    }

    public double smoothedObsDisturbance(int i, int t) {
        if (smoothedObsDisturbanceLength() == 0) {
            return 0.0;
        }
        return smoothedObsDisturbance[smoothedObsDisturbanceOffset(t) + i];
    }

    public double smoothedStateDisturbance(int i, int t) {
        if (smoothedStateDisturbanceLength() == 0) {
            return 0.0;
        }
        return smoothedStateDisturbance[smoothedStateDisturbanceOffset(t) + i];
    }

    public double smoothedObsDisturbanceCov(int i, int j, int t) {
        if (smoothedObsDisturbanceCovLength() == 0) {
            return 0.0;
        }
        return smoothedObsDisturbanceCov[smoothedObsDisturbanceCovOffset(t) + i * kEndog + j];
    }

    public double smoothedStateDisturbanceCov(int i, int j, int t) {
        if (smoothedStateDisturbanceCovLength() == 0) {
            return 0.0;
        }
        return smoothedStateDisturbanceCov[smoothedStateDisturbanceCovOffset(t) + i * kPosdef + j];
    }
}
