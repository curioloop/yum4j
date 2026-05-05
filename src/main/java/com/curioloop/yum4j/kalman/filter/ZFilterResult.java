package com.curioloop.yum4j.kalman.filter;

import com.curioloop.yum4j.kalman.arena.FilterResultLayout;

/**
 * Complex-valued filter output stored in interleaved flat arrays.
 */
public class ZFilterResult extends FilterResultBase<ZFilterResult> {
    ZFilterResult() {
        super();
    }

    public ZFilterResult(int kEndog, int kStates, int nobs) {
        this(kEndog, kStates, nobs, true, true, true, true, true);
    }

    public ZFilterResult(int kEndog, int kStates, int nobs,
                         boolean storeForecast,
                         boolean storePredicted,
                         boolean storeFiltered,
                         boolean storeGain,
                         boolean storeLikelihood) {
                this(kEndog, kStates, nobs,
                    FilterResultLayout.create(2, kEndog, kStates, nobs,
                        storeForecast, storePredicted, storeFiltered, storeGain, storeLikelihood));
    }

    public ZFilterResult(int kEndog, int kStates, int nobs, FilterResultShape shape) {
                this(kEndog, kStates, nobs,
                    FilterResultLayout.create(2, kEndog, kStates, nobs,
                        shape.storesForecast(),
                        shape.storesPredictedState(),
                        shape.storesFilteredState(),
                        shape.storesKalmanGain(),
                        shape.storesLikelihood()));
                }

                public ZFilterResult(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
                super(kEndog, kStates, nobs, layout);
    }

    @Override
    protected int scalarWidth() {
        return 2;
    }

    @Override
    protected ZFilterResult newResult(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
        return new ZFilterResult(kEndog, kStates, nobs, layout);
    }

    static double[] emptyArray() {
        return EMPTY;
    }

    public double[] predictedDiffuseStateCov(int i, int j, int t) {
        if (predictedDiffuseStateCov == null || predictedDiffuseStateCovLength() == 0) {
            return new double[]{0.0, 0.0};
        }
        int off = predictedDiffuseStateCovOffset(t) + i * 2 * kStates + j * 2;
        return new double[]{predictedDiffuseStateCov[off], predictedDiffuseStateCov[off + 1]};
    }

    public double[] forecastErrorDiffuseCov(int i, int j, int t) {
        if (forecastErrorDiffuseCov == null || forecastErrorDiffuseCovLength() == 0) {
            return new double[]{0.0, 0.0};
        }
        int off = forecastErrorDiffuseCovOffset(t) + i * 2 * kEndog + j * 2;
        return new double[]{forecastErrorDiffuseCov[off], forecastErrorDiffuseCov[off + 1]};
    }
}
