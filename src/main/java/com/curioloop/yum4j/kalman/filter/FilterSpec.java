package com.curioloop.yum4j.kalman.filter;

import com.curioloop.yum4j.kalman.arena.FilterDiffuseLayout;
import com.curioloop.yum4j.kalman.arena.FilterResultLayout;

/**
 * Typed filter configuration replacing raw bitmask flags at the public API boundary.
 */
public value class FilterSpec {

    public enum Storage {
        FORECAST,
        PREDICTED_STATE,
        FILTERED_STATE,
        LIKELIHOOD,
        KALMAN_GAIN
    }

    private static final int STORAGE_MASK_ALL = (1 << Storage.values().length) - 1;
    private static final FilterSpec[][] CACHE = createCache();

    private boolean forceSymmetry;
    private int storedMask;

    private FilterSpec(boolean forceSymmetry, int storedMask) {
        this.forceSymmetry = forceSymmetry;
        this.storedMask = storedMask & STORAGE_MASK_ALL;
    }

    public static FilterSpec defaults() {
        return cached(true, STORAGE_MASK_ALL);
    }

    public static FilterSpec full() {
        return defaults();
    }

    public static FilterSpec compact() {
        return cached(true, 0);
    }

    public static FilterSpec conventionalSmoothing() {
        return cached(true, STORAGE_MASK_ALL & ~(bit(Storage.FILTERED_STATE) | bit(Storage.LIKELIHOOD)));
    }

    public static FilterSpec classicalSmoothing() {
        return cached(true, STORAGE_MASK_ALL & ~bit(Storage.LIKELIHOOD));
    }

    public FilterSpec withForceSymmetry(boolean enabled) {
        return cached(enabled, storedMask);
    }

    public FilterSpec without(Storage storage, Storage... more) {
        int nextMask = storedMask & ~bit(storage);
        for (Storage item : more) {
            nextMask &= ~bit(item);
        }
        return cached(forceSymmetry, nextMask);
    }

    public boolean forceSymmetry() {
        return forceSymmetry;
    }

    public boolean stores(Storage storage) {
        return (storedMask & bit(storage)) != 0;
    }

    public FilterResultShape resultShape(boolean hasMissingObservations) {
        boolean storeForecast = stores(Storage.FORECAST);
        boolean storePredicted = storesPredictedState(hasMissingObservations);
        return FilterResultShape.of(
                storeForecast,
                storePredicted,
                stores(Storage.FILTERED_STATE),
                stores(Storage.KALMAN_GAIN),
                stores(Storage.LIKELIHOOD));
    }

        FilterResultLayout createResultLayout(int scalarWidth,
                          int kEndog,
                          int kStates,
                          int nobs,
                          boolean hasMissingObservations) {
        return FilterResultLayout.create(
            scalarWidth,
            kEndog,
            kStates,
            nobs,
            stores(Storage.FORECAST),
            storesPredictedState(hasMissingObservations),
            stores(Storage.FILTERED_STATE),
            stores(Storage.KALMAN_GAIN),
            stores(Storage.LIKELIHOOD));
        }

        FilterDiffuseLayout createDiffuseLayout(int scalarWidth,
                            int kEndog,
                            int kStates,
                            int nobs,
                            boolean hasMissingObservations) {
        return FilterDiffuseLayout.create(
            scalarWidth,
            kEndog,
            kStates,
            nobs,
            storesPredictedState(hasMissingObservations),
            stores(Storage.FORECAST));
        }

    int stabilityMask() {
        return forceSymmetry ? KalmanFilter.STABILITY_FORCE_SYMMETRY : 0;
    }

    int conserveMemoryMask() {
        int mask = KalmanFilter.MEMORY_STORE_ALL;
        if (!stores(Storage.FORECAST)) {
            mask |= KalmanFilter.MEMORY_NO_FORECAST;
        }
        if (!stores(Storage.PREDICTED_STATE)) {
            mask |= KalmanFilter.MEMORY_NO_PREDICTED;
        }
        if (!stores(Storage.FILTERED_STATE)) {
            mask |= KalmanFilter.MEMORY_NO_FILTERED;
        }
        if (!stores(Storage.LIKELIHOOD)) {
            mask |= KalmanFilter.MEMORY_NO_LIKELIHOOD;
        }
        if (!stores(Storage.KALMAN_GAIN)) {
            mask |= KalmanFilter.MEMORY_NO_GAIN;
        }
        return mask;
    }

    private static int bit(Storage storage) {
        return 1 << storage.ordinal();
    }

    private boolean storesPredictedState(boolean hasMissingObservations) {
        return stores(Storage.PREDICTED_STATE)
                || (stores(Storage.FORECAST) && hasMissingObservations);
    }

    private static FilterSpec cached(boolean forceSymmetry, int storedMask) {
        return CACHE[forceSymmetry ? 1 : 0][storedMask & STORAGE_MASK_ALL];
    }

    private static FilterSpec[][] createCache() {
        FilterSpec[][] cache = new FilterSpec[2][STORAGE_MASK_ALL + 1];
        for (int symmetry = 0; symmetry < cache.length; symmetry++) {
            boolean forceSymmetry = symmetry != 0;
            for (int mask = 0; mask <= STORAGE_MASK_ALL; mask++) {
                cache[symmetry][mask] = new FilterSpec(forceSymmetry, mask);
            }
        }
        return cache;
    }
}