package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.arena.FilterDiffuseLayout;
import com.curioloop.yum4j.ssm.kalman.arena.FilterResultLayout;

import java.util.Objects;

public final class FilterOptions {

    public enum Surface {
        FORECAST_MEAN,
        FORECAST_ERROR,
        FORECAST_COVARIANCE,
        STANDARDIZED_FORECAST_ERROR,
        PREDICTED_STATE,
        PREDICTED_STATE_COVARIANCE,
        FILTERED_STATE,
        FILTERED_STATE_COVARIANCE,
        KALMAN_GAIN,
        LIKELIHOOD,
        PREDICTED_DIFFUSE_STATE_COVARIANCE,
        FORECAST_ERROR_DIFFUSE_COVARIANCE
    }

    public enum Timing {
        INIT_PREDICTED,
        INIT_FILTERED
    }

    public enum SingularFallback {
        DISABLED,
        UNIVARIATE
    }

    public static final double DEFAULT_DIFFUSE_TOLERANCE = 1e-10;
    public static final double DEFAULT_CONVERGENCE_TOLERANCE = 1e-19;

    private static final ForecastErrorSolver.Method[] FORECAST_ERROR_AUTO_SEQUENCE = { ForecastErrorSolver.Method.AUTO };
    private static final ForecastErrorSolver.Method[] FORECAST_ERROR_CHOLESKY_SEQUENCE = { ForecastErrorSolver.Method.CHOLESKY_SOLVE };
    private static final ForecastErrorSolver.Method[] FORECAST_ERROR_LU_SEQUENCE = { ForecastErrorSolver.Method.LU_SOLVE };
    private static final ForecastErrorSolver.Method[] FORECAST_ERROR_UNIVARIATE_SEQUENCE = { ForecastErrorSolver.Method.UNIVARIATE };
    private static final ForecastErrorSolver.Method[] FORECAST_ERROR_CHOLESKY_LU_SEQUENCE = { ForecastErrorSolver.Method.CHOLESKY_SOLVE, ForecastErrorSolver.Method.LU_SOLVE };
    private static final ForecastErrorSolver.Method[] FORECAST_ERROR_AUTO_LU_SEQUENCE = { ForecastErrorSolver.Method.AUTO, ForecastErrorSolver.Method.LU_SOLVE };
    private static final ForecastErrorSolver.Method[] FORECAST_ERROR_LU_CHOLESKY_SEQUENCE = { ForecastErrorSolver.Method.LU_SOLVE, ForecastErrorSolver.Method.CHOLESKY_SOLVE };

    private static final int SURFACE_MASK_ALL = (1 << Surface.values().length) - 1;
    private static final int SURFACE_MASK_DEFAULT = SURFACE_MASK_ALL & ~bit(Surface.STANDARDIZED_FORECAST_ERROR);
    private static final int SURFACE_MASK_COMPACT = 0;
    private static final FilterOptions DEFAULTS = builder().build();

    private final FilterMethod method;
    private final int routeFlags;
    private final ForecastErrorStrategy forecastErrorStrategy;
    private final Timing timing;
    private final SingularFallback singularFallbackPolicy;
    private final int surfaceMask;
    private final boolean forceSymmetry;
    private final double diffuseTolerance;
    private final double convergenceTolerance;
    private final boolean concentratedLikelihood;
    private final int concentratedLikelihoodBurn;
    private final boolean fallbackTelemetry;

    private FilterOptions(Builder builder) {
        this.method = Objects.requireNonNull(builder.method, "method");
        this.routeFlags = FilterRouteSupport.validateRouteFlags(builder.routeFlags);
        this.forecastErrorStrategy = Objects.requireNonNull(builder.forecastErrorStrategy, "forecastErrorStrategy");
        this.timing = Objects.requireNonNull(builder.timing, "timing");
        this.singularFallbackPolicy = Objects.requireNonNull(builder.singularFallbackPolicy, "singularFallbackPolicy");
        this.surfaceMask = builder.surfaceMask & SURFACE_MASK_ALL;
        this.forceSymmetry = builder.forceSymmetry;
        this.diffuseTolerance = validatePositiveFinite("diffuseTolerance", builder.diffuseTolerance);
        this.convergenceTolerance = validatePositiveFinite("convergenceTolerance", builder.convergenceTolerance);
        this.concentratedLikelihood = builder.concentratedLikelihood;
        this.concentratedLikelihoodBurn = validateNonNegative("concentratedLikelihoodBurn", builder.concentratedLikelihoodBurn);
        this.fallbackTelemetry = builder.fallbackTelemetry;
    }

    public static FilterOptions defaults() {
        return DEFAULTS;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static FilterOptions compact() {
        return builder().retainNone().build();
    }

    public static FilterOptions storeAll() {
        return builder().retainAll().build();
    }

    public static FilterOptions likelihoodOnly() {
        return builder().retainOnly(Surface.LIKELIHOOD).build();
    }

    public static FilterOptions standardizedForecastError() {
        return builder().retainOnly(Surface.STANDARDIZED_FORECAST_ERROR).build();
    }

    public FilterMethod method() {
        return method;
    }

    int routeFlags() {
        return routeFlags;
    }

    public ForecastErrorStrategy forecastErrorStrategy() {
        return forecastErrorStrategy;
    }

    ForecastErrorSolver.Method[] forecastErrorSequenceRef() {
        return sequenceForStrategy(forecastErrorStrategy);
    }

    public Timing timing() {
        return timing;
    }

    public SingularFallback singularFallbackPolicy() {
        return singularFallbackPolicy;
    }

    public boolean allowsUnivariateFallback() {
        return singularFallbackPolicy == SingularFallback.UNIVARIATE;
    }

    public boolean includes(Surface surface) {
        return (surfaceMask & bit(Objects.requireNonNull(surface, "surface"))) != 0;
    }

    public boolean stores(Surface surface, Surface... more) {
        if (includes(surface)) {
            return true;
        }
        for (Surface item : more) {
            if (includes(item)) {
                return true;
            }
        }
        return false;
    }

    public FilterOptions with(Surface surface, Surface... more) {
        return toBuilder().retain(surface, more).build();
    }

    public FilterOptions without(Surface surface, Surface... more) {
        return toBuilder().drop(surface, more).build();
    }

    public FilterOptions withForceSymmetry(boolean enabled) {
        return toBuilder().forceSymmetry(enabled).build();
    }

    public FilterOptions withDiffuseTolerance(double tolerance) {
        return toBuilder().diffuseTolerance(tolerance).build();
    }

    public FilterOptions withConvergenceTolerance(double tolerance) {
        return toBuilder().convergenceTolerance(tolerance).build();
    }

    public FilterOptions withForecastErrorStrategy(ForecastErrorStrategy strategy) {
        return toBuilder().forecastErrorStrategy(strategy).build();
    }

    public FilterOptions withForecastSurfaces(boolean mean, boolean error, boolean covariance) {
        return withForecastSurfaces(mean, error, covariance, false, false);
    }

    public FilterOptions withForecastSurfaces(boolean mean,
                                              boolean error,
                                              boolean covariance,
                                              boolean standardized,
                                              boolean diffuseCovariance) {
        Builder builder = toBuilder().drop(
            Surface.FORECAST_MEAN,
            Surface.FORECAST_ERROR,
            Surface.FORECAST_COVARIANCE,
            Surface.STANDARDIZED_FORECAST_ERROR,
            Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);
        if (mean) {
            builder.retain(Surface.FORECAST_MEAN);
        }
        if (error) {
            builder.retain(Surface.FORECAST_ERROR);
        }
        if (covariance) {
            builder.retain(Surface.FORECAST_COVARIANCE);
        }
        if (standardized) {
            builder.retain(Surface.STANDARDIZED_FORECAST_ERROR);
        }
        if (diffuseCovariance) {
            builder.retain(Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);
        }
        return builder.build();
    }

    public boolean forceSymmetry() {
        return forceSymmetry;
    }

    public double diffuseTolerance() {
        return diffuseTolerance;
    }

    public double convergenceTolerance() {
        return convergenceTolerance;
    }

    public boolean concentratedLikelihood() {
        return concentratedLikelihood;
    }

    public int concentratedLikelihoodBurn() {
        return concentratedLikelihoodBurn;
    }

    public boolean fallbackTelemetry() {
        return fallbackTelemetry;
    }

    public Builder toBuilder() {
        Builder builder = new Builder()
            .method(method)
            .routeFlags(routeFlags)
            .forecastErrorStrategy(forecastErrorStrategy)
            .timing(timing)
            .singularFallbackPolicy(singularFallbackPolicy)
            .surfaceMask(surfaceMask)
            .forceSymmetry(forceSymmetry)
            .diffuseTolerance(diffuseTolerance)
            .convergenceTolerance(convergenceTolerance)
            .concentratedLikelihood(concentratedLikelihood)
            .concentratedLikelihoodBurn(concentratedLikelihoodBurn)
            .fallbackTelemetry(fallbackTelemetry);
        return builder;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof FilterOptions that)) {
            return false;
        }
        return surfaceMask == that.surfaceMask
            && forceSymmetry == that.forceSymmetry
            && Double.compare(diffuseTolerance, that.diffuseTolerance) == 0
            && Double.compare(convergenceTolerance, that.convergenceTolerance) == 0
            && concentratedLikelihood == that.concentratedLikelihood
            && concentratedLikelihoodBurn == that.concentratedLikelihoodBurn
            && fallbackTelemetry == that.fallbackTelemetry
            && method == that.method
            && routeFlags == that.routeFlags
            && forecastErrorStrategy == that.forecastErrorStrategy
            && timing == that.timing
            && singularFallbackPolicy == that.singularFallbackPolicy;
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(method, routeFlags, timing, singularFallbackPolicy, surfaceMask, forceSymmetry,
            diffuseTolerance, convergenceTolerance, concentratedLikelihood, concentratedLikelihoodBurn,
            fallbackTelemetry, forecastErrorStrategy);
        return result;
    }

    private static ForecastErrorSolver.Method[] sequenceForStrategy(ForecastErrorStrategy strategy) {
        return switch (Objects.requireNonNull(strategy, "strategy")) {
            case AUTO -> FORECAST_ERROR_AUTO_SEQUENCE;
            case CHOLESKY_SOLVE -> FORECAST_ERROR_CHOLESKY_SEQUENCE;
            case LU_SOLVE -> FORECAST_ERROR_LU_SEQUENCE;
            case UNIVARIATE -> FORECAST_ERROR_UNIVARIATE_SEQUENCE;
            case CHOLESKY_THEN_LU -> FORECAST_ERROR_CHOLESKY_LU_SEQUENCE;
            case AUTO_THEN_LU -> FORECAST_ERROR_AUTO_LU_SEQUENCE;
            case LU_THEN_CHOLESKY -> FORECAST_ERROR_LU_CHOLESKY_SEQUENCE;
        };
    }

    private static double validatePositiveFinite(String name, double value) {
        if (!(value > 0.0) || !Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be positive and finite");
        }
        return value;
    }

    private static int validateNonNegative(String name, int value) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative");
        }
        return value;
    }

    public boolean storesForecastMean() {
        return includes(Surface.FORECAST_MEAN);
    }

    public boolean storesForecastError() {
        return includes(Surface.FORECAST_ERROR);
    }

    public boolean storesForecastCovariance() {
        return includes(Surface.FORECAST_COVARIANCE);
    }

    public boolean storesStandardizedForecastError() {
        return includes(Surface.STANDARDIZED_FORECAST_ERROR);
    }

    public boolean storesForecastDiffuseCovariance() {
        return includes(Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);
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
            storesForecastMean(),
            storesForecastError(),
            storesForecastCovariance(),
            storesStandardizedForecastError(),
            storesPredictedState(hasMissingObservations),
            storesPredictedStateCovariance(hasMissingObservations),
            includes(Surface.FILTERED_STATE),
            includes(Surface.FILTERED_STATE_COVARIANCE),
            includes(Surface.KALMAN_GAIN),
            includes(Surface.LIKELIHOOD));
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
            storesPredictedDiffuseCovariance(hasMissingObservations),
            storesForecastDiffuseCovariance());
    }

    int stabilityMask() {
        return forceSymmetry ? KalmanFilter.STABILITY_FORCE_SYMMETRY : 0;
    }

    int conserveMemoryMask() {
        int mask = KalmanFilter.MEMORY_STORE_ALL;
        if (!storesForecastBlock()) {
            mask |= KalmanFilter.MEMORY_NO_FORECAST;
        }
        if (!storesPredictedBlock()) {
            mask |= KalmanFilter.MEMORY_NO_PREDICTED;
        }
        if (!storesFilteredBlock()) {
            mask |= KalmanFilter.MEMORY_NO_FILTERED;
        }
        if (!includes(Surface.LIKELIHOOD)) {
            mask |= KalmanFilter.MEMORY_NO_LIKELIHOOD;
        }
        if (!includes(Surface.KALMAN_GAIN)) {
            mask |= KalmanFilter.MEMORY_NO_GAIN;
        }
        return mask;
    }

    private boolean storesForecastBlock() {
        return includes(Surface.FORECAST_MEAN)
            || includes(Surface.FORECAST_ERROR)
            || includes(Surface.FORECAST_COVARIANCE)
            || includes(Surface.STANDARDIZED_FORECAST_ERROR)
            || includes(Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);
    }

    private boolean storesPredictedBlock() {
        return includes(Surface.PREDICTED_STATE)
            || includes(Surface.PREDICTED_STATE_COVARIANCE)
            || includes(Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE);
    }

    private boolean storesFilteredBlock() {
        return includes(Surface.FILTERED_STATE)
            || includes(Surface.FILTERED_STATE_COVARIANCE);
    }

    private boolean storesPredictedState(boolean hasMissingObservations) {
        return includes(Surface.PREDICTED_STATE) || storesInternalPredictedHistory(hasMissingObservations);
    }

    private boolean storesPredictedStateCovariance(boolean hasMissingObservations) {
        return includes(Surface.PREDICTED_STATE_COVARIANCE) || storesInternalPredictedHistory(hasMissingObservations);
    }

    private boolean storesPredictedDiffuseCovariance(boolean hasMissingObservations) {
        return includes(Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE)
            || storesInternalPredictedHistory(hasMissingObservations);
    }

    private boolean storesInternalPredictedHistory(boolean hasMissingObservations) {
        return storesForecastBlock() && hasMissingObservations;
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

    public static final class Builder {
        private FilterMethod method = FilterMethod.CONVENTIONAL;
        private int routeFlags = FilterRouteSupport.ROUTE_CONVENTIONAL;
        private ForecastErrorStrategy forecastErrorStrategy = ForecastErrorStrategy.AUTO;
        private Timing timing = Timing.INIT_PREDICTED;
        private SingularFallback singularFallbackPolicy = SingularFallback.DISABLED;
        private int surfaceMask = SURFACE_MASK_DEFAULT;
        private boolean forceSymmetry = true;
        private double diffuseTolerance = DEFAULT_DIFFUSE_TOLERANCE;
        private double convergenceTolerance = DEFAULT_CONVERGENCE_TOLERANCE;
        private boolean concentratedLikelihood;
        private int concentratedLikelihoodBurn;
        private boolean fallbackTelemetry;

        private Builder() {
        }

        public Builder method(FilterMethod method) {
            this.method = Objects.requireNonNull(method, "method");
            this.routeFlags = FilterRouteSupport.routeFlagsOf(method);
            return this;
        }

        public Builder tryMethods(FilterMethod preferred, FilterMethod... fallbackMethods) {
            this.method = Objects.requireNonNull(preferred, "preferred");
            if (preferred == FilterMethod.AUTO) {
                throw new IllegalArgumentException("AUTO cannot be used as a preferred explicit filter route");
            }
            this.routeFlags = FilterRouteSupport.routeFlagsOf(preferred, fallbackMethods);
            return this;
        }

        Builder routeFlags(int routeFlags) {
            this.routeFlags = FilterRouteSupport.validateRouteFlags(routeFlags);
            return this;
        }

        public Builder forecastErrorStrategy(ForecastErrorStrategy strategy) {
            this.forecastErrorStrategy = Objects.requireNonNull(strategy, "strategy");
            return this;
        }

        public Builder timing(Timing timing) {
            this.timing = Objects.requireNonNull(timing, "timing");
            return this;
        }

        public Builder singularFallbackPolicy(SingularFallback singularFallbackPolicy) {
            this.singularFallbackPolicy = Objects.requireNonNull(singularFallbackPolicy, "singularFallbackPolicy");
            return this;
        }

        public Builder retainAll() {
            this.surfaceMask = SURFACE_MASK_ALL;
            return this;
        }

        public Builder retainNone() {
            this.surfaceMask = SURFACE_MASK_COMPACT;
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

        public Builder forceSymmetry(boolean forceSymmetry) {
            this.forceSymmetry = forceSymmetry;
            return this;
        }

        public Builder diffuseTolerance(double diffuseTolerance) {
            this.diffuseTolerance = diffuseTolerance;
            return this;
        }

        public Builder convergenceTolerance(double convergenceTolerance) {
            this.convergenceTolerance = convergenceTolerance;
            return this;
        }

        public Builder concentratedLikelihood(boolean concentratedLikelihood) {
            this.concentratedLikelihood = concentratedLikelihood;
            return this;
        }

        public Builder concentratedLikelihoodBurn(int concentratedLikelihoodBurn) {
            this.concentratedLikelihoodBurn = concentratedLikelihoodBurn;
            return this;
        }

        public Builder fallbackTelemetry(boolean fallbackTelemetry) {
            this.fallbackTelemetry = fallbackTelemetry;
            return this;
        }

        public FilterOptions build() {
            return new FilterOptions(this);
        }
    }
}
