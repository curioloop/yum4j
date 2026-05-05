package com.curioloop.yum4j.kalman.smooth;

import com.curioloop.yum4j.kalman.arena.SmootherResultLayout;

/**
 * Complex-valued smoothing output stored in interleaved flat arrays.
 */
public class ZSmootherResult extends SmootherResultBase<ZSmootherResult> {
    ZSmootherResult() {
        super();
    }

    public ZSmootherResult(int kEndog, int kStates, int kPosdef, int nobs) {
        this(kEndog, kStates, kPosdef, nobs, true, true, true, true, true);
    }

    public ZSmootherResult(int kEndog, int kStates, int kPosdef, int nobs,
                           boolean storeState,
                           boolean storeStateCovariance,
                           boolean storeDisturbance,
                           boolean storeDisturbanceCovariance,
                           boolean storeAuxiliary) {
        this(kEndog, kStates, kPosdef, nobs,
                    SmootherResultLayout.create(2, kEndog, kStates, kPosdef, nobs,
                        storeState, storeStateCovariance, storeDisturbance,
                        storeDisturbanceCovariance, storeAuxiliary));
    }

    public ZSmootherResult(int kEndog, int kStates, int kPosdef, int nobs, SmootherResultShape shape) {
                this(kEndog, kStates, kPosdef, nobs,
                    SmootherResultLayout.create(2, kEndog, kStates, kPosdef, nobs,
                        shape.storesState(),
                        shape.storesStateCovariance(),
                        shape.storesDisturbance(),
                        shape.storesDisturbanceCovariance(),
                        shape.storesAuxiliary()));
                }

                public ZSmootherResult(int kEndog, int kStates, int kPosdef, int nobs, SmootherResultLayout layout) {
                super(kEndog, kStates, kPosdef, nobs, layout);
    }

    @Override
    protected int scalarWidth() {
        return 2;
    }

    @Override
    protected ZSmootherResult newResult(int kEndog, int kStates, int kPosdef, int nobs, SmootherResultLayout layout) {
        return new ZSmootherResult(kEndog, kStates, kPosdef, nobs, layout);
    }

    static double[] emptyArray() {
        return EMPTY;
    }
}
