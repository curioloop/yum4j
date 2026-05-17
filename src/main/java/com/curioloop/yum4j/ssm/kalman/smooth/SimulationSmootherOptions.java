package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;

import java.util.Objects;

public final class SimulationSmootherOptions {

    public enum Surface {
        STATE,
        DISTURBANCE,
        GENERATED_OUTPUTS
    }

    public enum Method {
        KFS,
        CFA
    }

    private static final int BIT_STATE = 1 << Surface.STATE.ordinal();
    private static final int BIT_DISTURBANCE = 1 << Surface.DISTURBANCE.ordinal();
    private static final int SURFACE_MASK_ALL = BIT_STATE | BIT_DISTURBANCE;
    private static final SimulationSmootherOptions DEFAULTS = builder().build();

    private final Method method;
    private final int surfaceMask;
    private final boolean univariateFilter;

    private SimulationSmootherOptions(Builder builder) {
        this.method = Objects.requireNonNull(builder.method, "method");
        int normalizedSurfaceMask = builder.surfaceMask & ((1 << Surface.values().length) - 1);
        if (this.method == Method.CFA && normalizedSurfaceMask == SURFACE_MASK_ALL) {
            normalizedSurfaceMask = BIT_STATE;
        }
        this.surfaceMask = normalizedSurfaceMask;
        this.univariateFilter = builder.univariateFilter;
    }

    public static SimulationSmootherOptions defaults() {
        return DEFAULTS;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static SimulationSmootherOptions stateOnly() {
        return builder().retainOnly(Surface.STATE).build();
    }

    public static SimulationSmootherOptions disturbanceOnly() {
        return builder().retainOnly(Surface.DISTURBANCE).build();
    }

    public boolean includes(Surface surface) {
        return (surfaceMask & bit(Objects.requireNonNull(surface, "surface"))) != 0;
    }

    public Method method() {
        return method;
    }

    public boolean univariateFilter() {
        return univariateFilter;
    }

    public boolean usesUnivariateFilter() {
        return univariateFilter;
    }

    public boolean storesGeneratedOutputs() {
        return includes(Surface.GENERATED_OUTPUTS);
    }

    public FilterOptions requiredFilterOptions() {
        requireKfsMethod();
        return FilterOptions.builder()
            .drop(FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
                FilterOptions.Surface.LIKELIHOOD)
            .build();
    }

    public SmootherOptions requiredSmootherOptions() {
        requireKfsMethod();
        boolean storeState = includes(Surface.STATE);
        boolean storeDisturbance = includes(Surface.DISTURBANCE);
        if (storeState && storeDisturbance) {
            return SmootherOptions.conventional().withoutCovariances();
        }
        if (storeState) {
            return SmootherOptions.builder()
                .retainOnly(SmootherOptions.Surface.STATE)
                .build();
        }
        if (storeDisturbance) {
            return SmootherOptions.builder()
                .retainOnly(SmootherOptions.Surface.DISTURBANCE)
                .build();
        }
        throw new IllegalStateException("Simulation smoother options must include state or disturbance output");
    }

    public Builder toBuilder() {
        return new Builder()
            .method(method)
            .surfaceMask(surfaceMask)
            .univariateFilter(univariateFilter);
    }

    public SimulationSmootherOptions with(Surface surface, Surface... more) {
        return toBuilder().retain(surface, more).build();
    }

    public SimulationSmootherOptions without(Surface surface, Surface... more) {
        return toBuilder().drop(surface, more).build();
    }

    public SimulationSmootherOptions withUnivariateFilter() {
        return toBuilder().univariateFilter(true).build();
    }

    public SimulationSmootherOptions withMethod(Method method) {
        return toBuilder().method(method).build();
    }

    public SimulationSmootherOptions withGeneratedOutputs() {
        return with(Surface.GENERATED_OUTPUTS);
    }

    public SimulationSmootherOptions withoutGeneratedOutputs() {
        return without(Surface.GENERATED_OUTPUTS);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof SimulationSmootherOptions that)) {
            return false;
        }
        return surfaceMask == that.surfaceMask && univariateFilter == that.univariateFilter && method == that.method;
    }

    @Override
    public int hashCode() {
        return Objects.hash(method, surfaceMask, univariateFilter);
    }

    private static int bit(Surface surface) {
        return 1 << surface.ordinal();
    }

    private static int maskOf(Surface surface, Surface... more) {
        int mask = bit(Objects.requireNonNull(surface, "surface"));
        for (int i = 0; i < more.length; i++) {
            mask |= bit(Objects.requireNonNull(more[i], "more[" + i + "]"));
        }
        return mask;
    }

    private void requireKfsMethod() {
        if (method != Method.KFS) {
            throw new UnsupportedOperationException("KFS filter and smoother history is not available for CFA simulation smoothing");
        }
    }

    public static final class Builder {
        private Method method = Method.KFS;
        private int surfaceMask = SURFACE_MASK_ALL;
        private boolean univariateFilter;

        private Builder() {
        }

        public Builder method(Method method) {
            this.method = Objects.requireNonNull(method, "method");
            return this;
        }

        public Builder retainOnly(Surface surface, Surface... more) {
            this.surfaceMask = maskOf(surface, more);
            return this;
        }

        public Builder retain(Surface surface, Surface... more) {
            this.surfaceMask |= maskOf(surface, more);
            return this;
        }

        public Builder drop(Surface surface, Surface... more) {
            this.surfaceMask &= ~maskOf(surface, more);
            return this;
        }

        Builder surfaceMask(int surfaceMask) {
            this.surfaceMask = surfaceMask & ((1 << Surface.values().length) - 1);
            return this;
        }

        public Builder univariateFilter(boolean univariateFilter) {
            this.univariateFilter = univariateFilter;
            return this;
        }

        public SimulationSmootherOptions build() {
            return new SimulationSmootherOptions(this);
        }
    }
}