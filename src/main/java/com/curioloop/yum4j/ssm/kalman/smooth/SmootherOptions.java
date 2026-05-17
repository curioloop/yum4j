package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.arena.SmootherResultLayout;
import com.curioloop.yum4j.ssm.kalman.filter.FilterMethod;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;

import java.util.Objects;

public final class SmootherOptions {

    public enum Surface {
        STATE,
        STATE_COVARIANCE,
        STATE_AUTOCOVARIANCE,
        DISTURBANCE,
        DISTURBANCE_COVARIANCE
    }

    private static final int SURFACE_MASK_ALL = (1 << Surface.values().length) - 1;
    private static final int SURFACE_MASK_DEFAULT = bit(Surface.STATE)
        | bit(Surface.STATE_COVARIANCE)
        | bit(Surface.DISTURBANCE)
        | bit(Surface.DISTURBANCE_COVARIANCE);
    private static final SmootherOptions DEFAULTS = builder().build();

    private final SmootherMethod method;
    private final int surfaceMask;

    private SmootherOptions(Builder builder) {
        this.method = Objects.requireNonNull(builder.method, "method");
        this.surfaceMask = builder.surfaceMask & SURFACE_MASK_ALL;
    }

    public static SmootherOptions defaults() {
        return DEFAULTS;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static SmootherOptions stateOnly() {
        return builder().retainOnly(Surface.STATE).build();
    }

    public static SmootherOptions disturbanceOnly() {
        return builder().retainOnly(Surface.DISTURBANCE).build();
    }

    public static SmootherOptions conventional() {
        return builder().method(SmootherMethod.CONVENTIONAL).build();
    }

    public static SmootherOptions classical() {
        return builder().method(SmootherMethod.CLASSICAL).build();
    }

    public static SmootherOptions alternative() {
        return builder().method(SmootherMethod.ALTERNATIVE).build();
    }

    public static SmootherOptions univariate() {
        return builder().method(SmootherMethod.UNIVARIATE).build();
    }

    public SmootherMethod method() {
        return method;
    }

    public boolean includes(Surface surface) {
        return (surfaceMask & bit(Objects.requireNonNull(surface, "surface"))) != 0;
    }

    public SmootherOptions with(Surface surface, Surface... more) {
        return toBuilder().retain(surface, more).build();
    }

    public SmootherOptions without(Surface surface, Surface... more) {
        return toBuilder().drop(surface, more).build();
    }

    public SmootherOptions only(Surface surface, Surface... more) {
        return toBuilder().retainOnly(surface, more).build();
    }

    public SmootherOptions withoutCovariances() {
        return without(Surface.STATE_COVARIANCE, Surface.STATE_AUTOCOVARIANCE, Surface.DISTURBANCE_COVARIANCE);
    }

    public boolean requiresFilteredStateHistory() {
        return method != SmootherMethod.CONVENTIONAL && method != SmootherMethod.UNIVARIATE;
    }

    public FilterOptions requiredFilterOptions() {
        return switch (method) {
            case CONVENTIONAL -> conventionalFilterOptions();
            case CLASSICAL, ALTERNATIVE -> classicalFilterOptions();
            case UNIVARIATE -> conventionalFilterOptions()
                .toBuilder()
                .method(FilterMethod.UNIVARIATE)
                .build();
        };
    }

    private static FilterOptions conventionalFilterOptions() {
        return FilterOptions.builder()
            .drop(FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
                FilterOptions.Surface.LIKELIHOOD)
            .build();
    }

    private static FilterOptions classicalFilterOptions() {
        return FilterOptions.builder()
            .drop(FilterOptions.Surface.LIKELIHOOD)
            .build();
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
            includes(Surface.STATE),
            includes(Surface.STATE_COVARIANCE),
            includes(Surface.STATE_AUTOCOVARIANCE),
            includes(Surface.DISTURBANCE),
            includes(Surface.DISTURBANCE_COVARIANCE),
            includesAllCoreSurfaces());
    }

    int smoothMethodMask() {
        return switch (method) {
            case CONVENTIONAL -> KalmanSmoother.SMOOTH_CONVENTIONAL;
            case CLASSICAL -> KalmanSmoother.SMOOTH_CLASSICAL;
            case ALTERNATIVE -> KalmanSmoother.SMOOTH_ALTERNATIVE;
            case UNIVARIATE -> KalmanSmoother.SMOOTH_UNIVARIATE;
        };
    }

    int outputMask() {
        int mask = 0;
        if (includes(Surface.STATE)) {
            mask |= KalmanSmoother.SMOOTHER_STATE;
        }
        if (includes(Surface.STATE_COVARIANCE)) {
            mask |= KalmanSmoother.SMOOTHER_STATE_COV;
        }
        if (includes(Surface.STATE_AUTOCOVARIANCE)) {
            mask |= KalmanSmoother.SMOOTHER_STATE_AUTOCOV;
        }
        if (includes(Surface.DISTURBANCE)) {
            mask |= KalmanSmoother.SMOOTHER_DISTURBANCE;
        }
        if (includes(Surface.DISTURBANCE_COVARIANCE)) {
            mask |= KalmanSmoother.SMOOTHER_DISTURBANCE_COV;
        }
        return mask;
    }

    public Builder toBuilder() {
        return new Builder()
            .method(method)
            .surfaceMask(surfaceMask);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof SmootherOptions that)) {
            return false;
        }
        return surfaceMask == that.surfaceMask && method == that.method;
    }

    @Override
    public int hashCode() {
        return Objects.hash(method, surfaceMask);
    }

    private static int bit(Surface surface) {
        return 1 << surface.ordinal();
    }

    private boolean includesAllCoreSurfaces() {
        return includes(Surface.STATE)
            && includes(Surface.STATE_COVARIANCE)
            && includes(Surface.DISTURBANCE)
            && includes(Surface.DISTURBANCE_COVARIANCE);
    }

    private static int maskOf(Surface surface, Surface... more) {
        int mask = bit(Objects.requireNonNull(surface, "surface"));
        for (int i = 0; i < more.length; i++) {
            mask |= bit(Objects.requireNonNull(more[i], "more[" + i + "]"));
        }
        return mask;
    }

    public static final class Builder {
        private SmootherMethod method = SmootherMethod.CONVENTIONAL;
        private int surfaceMask = SURFACE_MASK_DEFAULT;

        private Builder() {
        }

        public Builder method(SmootherMethod method) {
            this.method = Objects.requireNonNull(method, "method");
            return this;
        }

        public Builder retainAll() {
            this.surfaceMask = SURFACE_MASK_ALL;
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
            this.surfaceMask = surfaceMask & SURFACE_MASK_ALL;
            return this;
        }

        public SmootherOptions build() {
            return new SmootherOptions(this);
        }
    }
}
