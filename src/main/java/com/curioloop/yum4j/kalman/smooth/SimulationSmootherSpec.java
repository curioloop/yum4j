package com.curioloop.yum4j.kalman.smooth;

import com.curioloop.yum4j.kalman.filter.FilterSpec;

/**
 * Typed output selection for simulation smoothing.
 */
public value class SimulationSmootherSpec {

    public enum Output {
        STATE,
        DISTURBANCE
    }

    private static final int OUTPUT_STATE_MASK = 0x01;
    private static final int OUTPUT_DISTURBANCE_MASK = 0x02;
    private static final int OUTPUT_MASK_ALL = OUTPUT_STATE_MASK | OUTPUT_DISTURBANCE_MASK;
    private static final SimulationSmootherSpec[][][] CACHE = createCache();

    private int outputsMask;
    private boolean univariateFilter;
    private boolean generatedOutputs;

    private SimulationSmootherSpec(int outputsMask, boolean univariateFilter, boolean generatedOutputs) {
        this.outputsMask = outputsMask & OUTPUT_MASK_ALL;
        this.univariateFilter = univariateFilter;
        this.generatedOutputs = generatedOutputs;
    }

    public static SimulationSmootherSpec all() {
        return cached(OUTPUT_MASK_ALL, false, false);
    }

    public static SimulationSmootherSpec stateOnly() {
        return cached(OUTPUT_STATE_MASK, false, false);
    }

    public static SimulationSmootherSpec disturbanceOnly() {
        return cached(OUTPUT_DISTURBANCE_MASK, false, false);
    }

    public SimulationSmootherSpec without(Output output, Output... more) {
        int nextMask = outputsMask & ~bit(output);
        for (Output item : more) {
            nextMask &= ~bit(item);
        }
        return cached(nextMask, univariateFilter, generatedOutputs);
    }

    public SimulationSmootherSpec withUnivariateFilter() {
        return cached(outputsMask, true, generatedOutputs);
    }

    public SimulationSmootherSpec withGeneratedOutputs() {
        return cached(outputsMask, univariateFilter, true);
    }

    public SimulationSmootherSpec withoutGeneratedOutputs() {
        return cached(outputsMask, univariateFilter, false);
    }

    public boolean includes(Output output) {
        return (outputsMask & bit(output)) != 0;
    }

    public boolean usesUnivariateFilter() {
        return univariateFilter;
    }

    public boolean storesGeneratedOutputs() {
        return generatedOutputs;
    }

    public FilterSpec requiredFilterSpec() {
        return FilterSpec.conventionalSmoothing();
    }

    public SmootherSpec requiredSmootherSpec() {
        SmootherSpec spec = SmootherSpec.conventional().withoutCovariances();
        boolean state = includes(Output.STATE);
        boolean disturbance = includes(Output.DISTURBANCE);
        if (state && disturbance) {
            return spec;
        }
        if (state) {
            return spec.stateOnly();
        }
        if (disturbance) {
            return spec.disturbanceOnly();
        }
        throw new IllegalStateException("SimulationSmootherSpec must include at least one output");
    }

    private static int bit(Output output) {
        return switch (output) {
            case STATE -> OUTPUT_STATE_MASK;
            case DISTURBANCE -> OUTPUT_DISTURBANCE_MASK;
        };
    }

    private static SimulationSmootherSpec cached(int outputsMask, boolean univariateFilter, boolean generatedOutputs) {
        return CACHE[generatedOutputs ? 1 : 0][univariateFilter ? 1 : 0][outputsMask & OUTPUT_MASK_ALL];
    }

    private static SimulationSmootherSpec[][][] createCache() {
        SimulationSmootherSpec[][][] cache = new SimulationSmootherSpec[2][2][OUTPUT_MASK_ALL + 1];
        for (int generated = 0; generated < cache.length; generated++) {
            boolean storesGenerated = generated != 0;
            for (int univariate = 0; univariate < cache[generated].length; univariate++) {
                boolean usesUnivariate = univariate != 0;
                for (int mask = 0; mask <= OUTPUT_MASK_ALL; mask++) {
                    cache[generated][univariate][mask] = new SimulationSmootherSpec(mask, usesUnivariate, storesGenerated);
                }
            }
        }
        return cache;
    }
}