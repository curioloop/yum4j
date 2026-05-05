package com.curioloop.yum4j.kalman.filter;

/**
 * Explicit storage shape for filter results.
 */
public value class FilterResultShape {

    private static final int FORECAST_MASK = 0x01;
    private static final int PREDICTED_STATE_MASK = 0x02;
    private static final int FILTERED_STATE_MASK = 0x04;
    private static final int KALMAN_GAIN_MASK = 0x08;
    private static final int LIKELIHOOD_MASK = 0x10;
    private static final int ALL_MASK = FORECAST_MASK
            | PREDICTED_STATE_MASK
            | FILTERED_STATE_MASK
            | KALMAN_GAIN_MASK
            | LIKELIHOOD_MASK;
    private static final FilterResultShape[] CACHE = createCache();

    private int mask;

    private FilterResultShape(int mask) {
        this.mask = mask & ALL_MASK;
    }

    public static FilterResultShape of(boolean storeForecast,
                                       boolean storePredictedState,
                                       boolean storeFilteredState,
                                       boolean storeKalmanGain,
                                       boolean storeLikelihood) {
        int mask = 0;
        if (storeForecast) {
            mask |= FORECAST_MASK;
        }
        if (storePredictedState) {
            mask |= PREDICTED_STATE_MASK;
        }
        if (storeFilteredState) {
            mask |= FILTERED_STATE_MASK;
        }
        if (storeKalmanGain) {
            mask |= KALMAN_GAIN_MASK;
        }
        if (storeLikelihood) {
            mask |= LIKELIHOOD_MASK;
        }
        return CACHE[mask];
    }

    public boolean storesForecast() {
        return (mask & FORECAST_MASK) != 0;
    }

    public boolean storesPredictedState() {
        return (mask & PREDICTED_STATE_MASK) != 0;
    }

    public boolean storesFilteredState() {
        return (mask & FILTERED_STATE_MASK) != 0;
    }

    public boolean storesKalmanGain() {
        return (mask & KALMAN_GAIN_MASK) != 0;
    }

    public boolean storesLikelihood() {
        return (mask & LIKELIHOOD_MASK) != 0;
    }

    public int layoutOptions() {
        return mask;
    }

    public int diffuseLayoutOptions() {
        return mask & (FORECAST_MASK | PREDICTED_STATE_MASK);
    }

    private static FilterResultShape[] createCache() {
        FilterResultShape[] cache = new FilterResultShape[ALL_MASK + 1];
        for (int mask = 0; mask < cache.length; mask++) {
            cache[mask] = new FilterResultShape(mask);
        }
        return cache;
    }
}