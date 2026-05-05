package com.curioloop.yum4j.kalman.smooth;

import com.curioloop.yum4j.kalman.arena.SmootherResultLayout;

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

    public SmootherResult(int kEndog, int kStates, int kPosdef, int nobs, SmootherResultShape shape) {
                this(kEndog, kStates, kPosdef, nobs,
                    SmootherResultLayout.create(1, kEndog, kStates, kPosdef, nobs,
                        shape.storesState(),
                        shape.storesStateCovariance(),
                        shape.storesDisturbance(),
                        shape.storesDisturbanceCovariance(),
                        shape.storesAuxiliary()));
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
