package com.curioloop.yum4j.kalman.smooth;

import com.curioloop.yum4j.kalman.arena.SmootherResultLayout;
import com.curioloop.yum4j.kalman.filter.FilterSpec;

/**
 * Typed smoother configuration replacing raw method and output bitmasks.
 */
public value class SmootherSpec {

    public enum Method {
        CONVENTIONAL,
        CLASSICAL,
        ALTERNATIVE
    }

    public enum Output {
        STATE,
        STATE_COVARIANCE,
        DISTURBANCE,
        DISTURBANCE_COVARIANCE
    }

    private static final int OUTPUT_STATE_MASK = KalmanSmoother.SMOOTHER_STATE;
    private static final int OUTPUT_STATE_COV_MASK = KalmanSmoother.SMOOTHER_STATE_COV;
    private static final int OUTPUT_DISTURBANCE_MASK = KalmanSmoother.SMOOTHER_DISTURBANCE;
    private static final int OUTPUT_DISTURBANCE_COV_MASK = KalmanSmoother.SMOOTHER_DISTURBANCE_COV;
    private static final int OUTPUT_MASK_ALL = OUTPUT_STATE_MASK
            | OUTPUT_STATE_COV_MASK
            | OUTPUT_DISTURBANCE_MASK
            | OUTPUT_DISTURBANCE_COV_MASK;
    private static final SmootherSpec[][] CACHE = createCache();

    private Method method;
    private int outputsMask;

    private SmootherSpec(Method method, int outputsMask) {
        this.method = method;
        this.outputsMask = outputsMask & OUTPUT_MASK_ALL;
    }

    public static SmootherSpec conventional() {
        return cached(Method.CONVENTIONAL, OUTPUT_MASK_ALL);
    }

    public static SmootherSpec classical() {
        return cached(Method.CLASSICAL, OUTPUT_MASK_ALL);
    }

    public static SmootherSpec alternative() {
        return cached(Method.ALTERNATIVE, OUTPUT_MASK_ALL);
    }

    public SmootherSpec without(Output output, Output... more) {
        int nextMask = outputsMask & ~bit(output);
        for (Output item : more) {
            nextMask &= ~bit(item);
        }
        return cached(method, nextMask);
    }

    public SmootherSpec withoutCovariances() {
        return cached(method, outputsMask & ~(OUTPUT_STATE_COV_MASK | OUTPUT_DISTURBANCE_COV_MASK));
    }

    public SmootherSpec stateOnly() {
        return cached(method, OUTPUT_STATE_MASK);
    }

    public SmootherSpec disturbanceOnly() {
        return cached(method, OUTPUT_DISTURBANCE_MASK);
    }

    public Method method() {
        return method;
    }

    public boolean includes(Output output) {
        return (outputsMask & bit(output)) != 0;
    }

    public boolean requiresFilteredStateHistory() {
        return method != Method.CONVENTIONAL;
    }

    public FilterSpec requiredFilterSpec() {
        return switch (method) {
            case CONVENTIONAL -> FilterSpec.conventionalSmoothing();
            case CLASSICAL -> FilterSpec.classicalSmoothing();
            case ALTERNATIVE -> FilterSpec.classicalSmoothing();
        };
    }

    public SmootherResultShape resultShape() {
        return SmootherResultShape.of(
                includes(Output.STATE),
                includes(Output.STATE_COVARIANCE),
                includes(Output.DISTURBANCE),
                includes(Output.DISTURBANCE_COVARIANCE),
                outputsMask == OUTPUT_MASK_ALL);
    }

    SmootherResultLayout createResultLayout(int scalarWidth,
                                            int kEndog,
                                            int kStates,
                                            int kPosdef,
                                            int nobs) {
        return SmootherResultLayout.create(
                scalarWidth,
                kEndog,
                kStates,
                kPosdef,
                nobs,
                includes(Output.STATE),
                includes(Output.STATE_COVARIANCE),
                includes(Output.DISTURBANCE),
                includes(Output.DISTURBANCE_COVARIANCE),
                outputsMask == OUTPUT_MASK_ALL);
    }

    int smoothMethodMask() {
        return switch (method) {
            case CONVENTIONAL -> KalmanSmoother.SMOOTH_CONVENTIONAL;
            case CLASSICAL -> KalmanSmoother.SMOOTH_CLASSICAL;
            case ALTERNATIVE -> KalmanSmoother.SMOOTH_ALTERNATIVE;
        };
    }

    int outputMask() {
        return outputsMask;
    }

    private static int bit(Output output) {
        return switch (output) {
            case STATE -> OUTPUT_STATE_MASK;
            case STATE_COVARIANCE -> OUTPUT_STATE_COV_MASK;
            case DISTURBANCE -> OUTPUT_DISTURBANCE_MASK;
            case DISTURBANCE_COVARIANCE -> OUTPUT_DISTURBANCE_COV_MASK;
        };
    }

    private static SmootherSpec cached(Method method, int outputsMask) {
        return CACHE[method.ordinal()][outputsMask & OUTPUT_MASK_ALL];
    }

    private static SmootherSpec[][] createCache() {
        SmootherSpec[][] cache = new SmootherSpec[Method.values().length][OUTPUT_MASK_ALL + 1];
        for (Method method : Method.values()) {
            for (int mask = 0; mask <= OUTPUT_MASK_ALL; mask++) {
                cache[method.ordinal()][mask] = new SmootherSpec(method, mask);
            }
        }
        return cache;
    }
}