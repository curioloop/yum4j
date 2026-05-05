package com.curioloop.yum4j.kalman.filter;

import com.curioloop.yum4j.kalman.arena.FilterResultLayout;

/**
 * Flat storage for one Kalman filter run.
 *
 * <p>Arrays are grouped by predicted, filtered, forecast, and optional diffuse
 * quantities, matching the package-level separation used by the reference design.</p>
 */
public class FilterResult extends FilterResultBase<FilterResult> {

    FilterResult() {
        super();
    }

    public FilterResult(int kEndog, int kStates, int nobs) {
        this(kEndog, kStates, nobs, true, true, true, true, true);
    }

    public FilterResult(int kEndog, int kStates, int nobs,
                        boolean storeForecast,
                        boolean storePredicted,
                        boolean storeFiltered,
                        boolean storeGain,
                        boolean storeLikelihood) {
                this(kEndog, kStates, nobs,
                    FilterResultLayout.create(1, kEndog, kStates, nobs,
                        storeForecast, storePredicted, storeFiltered, storeGain, storeLikelihood));
    }

    public FilterResult(int kEndog, int kStates, int nobs, FilterResultShape shape) {
                this(kEndog, kStates, nobs,
                    FilterResultLayout.create(1, kEndog, kStates, nobs,
                        shape.storesForecast(),
                        shape.storesPredictedState(),
                        shape.storesFilteredState(),
                        shape.storesKalmanGain(),
                        shape.storesLikelihood()));
                }

                public FilterResult(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
                super(kEndog, kStates, nobs, layout);
    }

    @Override
    protected int scalarWidth() {
        return 1;
    }

    @Override
    protected FilterResult newResult(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
        return new FilterResult(kEndog, kStates, nobs, layout);
    }

    static double[] emptyArray() {
        return EMPTY;
    }

    public double predictedState(int i, int t) {
        if (predictedStateLength() == 0) {
            return 0.0;
        }
        return predictedState[predictedStateOffset(t) + i];
    }

    public double predictedStateCov(int i, int j, int t) {
        if (predictedStateCovLength() == 0) {
            return 0.0;
        }
        return predictedStateCov[predictedStateCovOffset(t) + i * kStates + j];
    }

    public double filteredState(int i, int t) {
        if (filteredStateLength() == 0) {
            return 0.0;
        }
        return filteredState[filteredStateOffset(t) + i];
    }

    public double filteredStateCov(int i, int j, int t) {
        if (filteredStateCovLength() == 0) {
            return 0.0;
        }
        return filteredStateCov[filteredStateCovOffset(t) + i * kStates + j];
    }

    public double forecast(int i, int t) {
        if (forecastLength() == 0) {
            return 0.0;
        }
        return forecast[forecastOffset(t) + i];
    }

    public double forecastError(int i, int t) {
        if (forecastErrorLength() == 0) {
            return 0.0;
        }
        return forecastError[forecastErrorOffset(t) + i];
    }

    public double forecastErrorCov(int i, int j, int t) {
        if (forecastErrorCovLength() == 0) {
            return 0.0;
        }
        return forecastErrorCov[forecastErrorCovOffset(t) + i * kEndog + j];
    }

    public double kalmanGain(int i, int j, int t) {
        if (kalmanGainLength() == 0) {
            return 0.0;
        }
        return kalmanGain[kalmanGainOffset(t) + i * kEndog + j];
    }

    public double predictedDiffuseStateCov(int i, int j, int t) {
        if (predictedDiffuseStateCov == null || predictedDiffuseStateCovLength() == 0) {
            return 0.0;
        }
        return predictedDiffuseStateCov[predictedDiffuseStateCovOffset(t) + i * kStates + j];
    }

    public double forecastErrorDiffuseCov(int i, int j, int t) {
        if (forecastErrorDiffuseCov == null || forecastErrorDiffuseCovLength() == 0) {
            return 0.0;
        }
        return forecastErrorDiffuseCov[forecastErrorDiffuseCovOffset(t) + i * kEndog + j];
    }
}
