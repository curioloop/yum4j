package com.curioloop.yum4j.kalman.smooth;

/**
 * Explicit storage shape for smoother results.
 */
public value class SmootherResultShape {

    private static final int STATE_MASK = 0x01;
    private static final int STATE_COV_MASK = 0x02;
    private static final int DISTURBANCE_MASK = 0x04;
    private static final int DISTURBANCE_COV_MASK = 0x08;
    private static final int AUXILIARY_MASK = 0x10;
    private static final int ALL_MASK = STATE_MASK
            | STATE_COV_MASK
            | DISTURBANCE_MASK
            | DISTURBANCE_COV_MASK
            | AUXILIARY_MASK;
    private static final SmootherResultShape[] CACHE = createCache();

    private int mask;

    private SmootherResultShape(int mask) {
        this.mask = mask & ALL_MASK;
    }

    public static SmootherResultShape of(boolean storeState,
                                         boolean storeStateCovariance,
                                         boolean storeDisturbance,
                                         boolean storeDisturbanceCovariance,
                                         boolean storeAuxiliary) {
        int mask = 0;
        if (storeState) {
            mask |= STATE_MASK;
        }
        if (storeStateCovariance) {
            mask |= STATE_COV_MASK;
        }
        if (storeDisturbance) {
            mask |= DISTURBANCE_MASK;
        }
        if (storeDisturbanceCovariance) {
            mask |= DISTURBANCE_COV_MASK;
        }
        if (storeAuxiliary) {
            mask |= AUXILIARY_MASK;
        }
        return CACHE[mask];
    }

    public boolean storesState() {
        return (mask & STATE_MASK) != 0;
    }

    public boolean storesStateCovariance() {
        return (mask & STATE_COV_MASK) != 0;
    }

    public boolean storesDisturbance() {
        return (mask & DISTURBANCE_MASK) != 0;
    }

    public boolean storesDisturbanceCovariance() {
        return (mask & DISTURBANCE_COV_MASK) != 0;
    }

    public boolean storesAuxiliary() {
        return (mask & AUXILIARY_MASK) != 0;
    }

    public int layoutOptions() {
        return mask;
    }

    private static SmootherResultShape[] createCache() {
        SmootherResultShape[] cache = new SmootherResultShape[ALL_MASK + 1];
        for (int mask = 0; mask < cache.length; mask++) {
            cache[mask] = new SmootherResultShape(mask);
        }
        return cache;
    }
}