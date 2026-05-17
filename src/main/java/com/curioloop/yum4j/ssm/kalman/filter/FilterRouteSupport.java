package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;

import java.util.Objects;

final class FilterRouteSupport {

    private static final int AUTO_MIN_PERIODS = 32;
    static final int ROUTE_CONVENTIONAL = 1;
    static final int ROUTE_UNIVARIATE = 1 << 1;
    static final int ROUTE_COLLAPSED = 1 << 2;
    static final int ROUTE_CHANDRASEKHAR = 1 << 3;
    static final int ROUTE_ALL = ROUTE_CONVENTIONAL | ROUTE_UNIVARIATE | ROUTE_COLLAPSED | ROUTE_CHANDRASEKHAR;

    private static final FilterMethod[] FALLBACK_ORDER = {
        FilterMethod.CHANDRASEKHAR,
        FilterMethod.COLLAPSED,
        FilterMethod.UNIVARIATE,
        FilterMethod.CONVENTIONAL
    };

    enum Reason {
        SUPPORTED,
        COMPLEX_ROUTE_UNSUPPORTED,
        OVERRIDDEN_OBSERVATIONS_UNSUPPORTED,
        CONCENTRATED_UNSUPPORTED,
        CONCENTRATED_FALLBACK_UNSUPPORTED,
        INIT_FILTERED_UNSUPPORTED,
        DIFFUSE_UNSUPPORTED,
        STATIONARY_REQUIRED,
        MISSING_UNSUPPORTED,
        TIME_VARYING_UNSUPPORTED,
        DIMENSION_UNSUPPORTED,
        NON_DIAGONAL_OBSERVATION_UNSUPPORTED,
        NON_DIAGONAL_EXACT_DIFFUSE_GAIN_UNSUPPORTED
    }

    record Capability(boolean supported, Reason reason, String message) {
        static Capability available() {
            return new Capability(true, Reason.SUPPORTED, null);
        }

        static Capability rejected(Reason reason, String message) {
            return new Capability(false, reason, message);
        }

        RuntimeException exception(FilterMethod method) {
            return new UnsupportedOperationException(message == null
                ? method + " filtering is not supported for this model/options profile"
                : message);
        }
    }

    record StructuralProfile(boolean hasMissingObservations,
                             boolean hasNonDiagonalObservationCovariance,
                             boolean hasTimeVaryingSystemMatrices) {
    }

    private FilterRouteSupport() {
    }

    static Capability check(KalmanSSM model,
                            InitialState initialState,
                            FilterOptions options,
                            FilterMethod method,
                            boolean complex,
                            boolean overriddenObservations) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(method, "method");
        return check(model, initialState, options, method, complex, overriddenObservations, analyze(model));
    }

    static Reason reason(KalmanSSM model,
                         InitialState initialState,
                         FilterOptions options,
                         FilterMethod method,
                         boolean complex,
                         boolean overriddenObservations) {
        return check(model, initialState, options, method, complex, overriddenObservations).reason();
    }

    static StructuralProfile analyze(KalmanSSM model) {
        Objects.requireNonNull(model, "model");
        return new StructuralProfile(
            KalmanSSMSupport.hasMissingObservations(model),
            KalmanSSMSupport.hasNonDiagonalObservationCovariance(model),
            hasTimeVaryingSystemMatrices(model));
    }

    static int routeFlagsOf(FilterMethod method) {
        return switch (Objects.requireNonNull(method, "method")) {
            case AUTO -> ROUTE_ALL;
            case CONVENTIONAL -> ROUTE_CONVENTIONAL;
            case UNIVARIATE -> ROUTE_UNIVARIATE;
            case COLLAPSED -> ROUTE_COLLAPSED;
            case CHANDRASEKHAR -> ROUTE_CHANDRASEKHAR;
        };
    }

    static int routeFlagsOf(FilterMethod first, FilterMethod... more) {
        int flags = routeFlagsOf(first);
        for (int i = 0; i < more.length; i++) {
            FilterMethod method = Objects.requireNonNull(more[i], "more[" + i + "]");
            if (method == FilterMethod.AUTO) {
                throw new IllegalArgumentException("AUTO cannot be combined with explicit filter routes");
            }
            flags |= routeFlagsOf(method);
        }
        return flags;
    }

    static boolean includesRoute(int flags, FilterMethod method) {
        return (flags & routeFlagsOf(method)) != 0;
    }

    static int validateRouteFlags(int flags) {
        if ((flags & ~ROUTE_ALL) != 0) {
            throw new IllegalArgumentException("Unknown filter route flags: " + flags);
        }
        return flags == 0 ? ROUTE_CONVENTIONAL : flags;
    }

    static Capability check(KalmanSSM model,
                            InitialState initialState,
                            FilterOptions options,
                            FilterMethod method,
                            boolean complex,
                            boolean overriddenObservations,
                            StructuralProfile profile) {
        return switch (method) {
            case AUTO -> Capability.available();
            case CONVENTIONAL -> conventional(options, complex);
            case UNIVARIATE -> univariate(initialState, complex, profile);
            case COLLAPSED -> collapsed(model, initialState, options, complex, overriddenObservations, profile);
            case CHANDRASEKHAR -> chandrasekhar(model, initialState, options, complex, overriddenObservations, profile);
        };
    }

    static FilterMethod resolveAuto(KalmanSSM model,
                                    InitialState initialState,
                                    FilterOptions options,
                                    boolean complex,
                                    boolean overriddenObservations) {
        return resolveAuto(model, initialState, options, complex, overriddenObservations, analyze(model));
    }

    static FilterMethod resolveAuto(KalmanSSM model,
                                    InitialState initialState,
                                    FilterOptions options,
                                    boolean complex,
                                    boolean overriddenObservations,
                                    StructuralProfile profile) {
        if (complex || overriddenObservations) {
            return FilterMethod.CONVENTIONAL;
        }
        if (initialState.mayResolveDiffuse() || options.timing() == FilterOptions.Timing.INIT_FILTERED) {
            return FilterMethod.CONVENTIONAL;
        }
        if (options.singularFallbackPolicy() != FilterOptions.SingularFallback.DISABLED) {
            return FilterMethod.CONVENTIONAL;
        }
        if (model.observationCount() >= AUTO_MIN_PERIODS
            && model.stateCount() <= model.observationDimension()
                && check(model, initialState, options, FilterMethod.CHANDRASEKHAR, false, false, profile).supported()) {
            return FilterMethod.CHANDRASEKHAR;
        }
        if (model.observationCount() >= AUTO_MIN_PERIODS
                && collapsedAutoWorthwhile(options)
                && check(model, initialState, options, FilterMethod.COLLAPSED, false, false, profile).supported()) {
            return FilterMethod.COLLAPSED;
        }
        if (model.observationCount() >= AUTO_MIN_PERIODS
            && model.stateCount() <= model.observationDimension()
                && !profile.hasNonDiagonalObservationCovariance()
                && check(model, initialState, options, FilterMethod.UNIVARIATE, false, false, profile).supported()) {
            return FilterMethod.UNIVARIATE;
        }
        return FilterMethod.CONVENTIONAL;
    }

    static FilterMethod resolveProfile(KalmanSSM model,
                                       InitialState initialState,
                                       FilterOptions options,
                                       boolean complex,
                                       boolean overriddenObservations,
                                       StructuralProfile profile) {
        FilterMethod requested = options.method();
        if (requested == FilterMethod.AUTO) {
            return resolveAuto(model, initialState, options, complex, overriddenObservations, profile);
        }

        int flags = options.routeFlags();
        if (includesRoute(flags, requested)
                && check(model, initialState, options, requested, complex, overriddenObservations, profile).supported()) {
            return requested;
        }
        for (FilterMethod candidate : FALLBACK_ORDER) {
            if (candidate != requested
                    && includesRoute(flags, candidate)
                    && check(model, initialState, options, candidate, complex, overriddenObservations, profile).supported()) {
                return candidate;
            }
        }
        return requested;
    }

    private static Capability conventional(FilterOptions options, boolean complex) {
        return Capability.available();
    }

    private static Capability univariate(InitialState initialState,
                                         boolean complex,
                                         StructuralProfile profile) {
        return Capability.available();
    }

    private static Capability collapsed(KalmanSSM model,
                                        InitialState initialState,
                                        FilterOptions options,
                                        boolean complex,
                                        boolean overriddenObservations,
                                        StructuralProfile profile) {
        if (complex) {
            return Capability.rejected(Reason.COMPLEX_ROUTE_UNSUPPORTED,
                "COLLAPSED filtering is implemented for real-valued models only");
        }
        if (options.concentratedLikelihood()) {
            return Capability.rejected(Reason.CONCENTRATED_UNSUPPORTED,
                "COLLAPSED filtering does not support concentrated likelihood");
        }
        if (model.observationDimension() <= model.stateCount()) {
            return Capability.rejected(Reason.DIMENSION_UNSUPPORTED,
                "COLLAPSED filtering requires observation dimension larger than state dimension");
        }
        return Capability.available();
    }

    private static Capability chandrasekhar(KalmanSSM model,
                                            InitialState initialState,
                                            FilterOptions options,
                                            boolean complex,
                                            boolean overriddenObservations,
                                            StructuralProfile profile) {
        if (complex) {
            return Capability.rejected(Reason.COMPLEX_ROUTE_UNSUPPORTED,
                "CHANDRASEKHAR filtering is implemented for real-valued models only");
        }
        if (overriddenObservations) {
            return Capability.rejected(Reason.OVERRIDDEN_OBSERVATIONS_UNSUPPORTED,
                "CHANDRASEKHAR filtering with overridden observations is not implemented yet");
        }
        if (options.timing() == FilterOptions.Timing.INIT_FILTERED) {
            return Capability.rejected(Reason.INIT_FILTERED_UNSUPPORTED,
                "CHANDRASEKHAR filtering cannot be used with INIT_FILTERED timing");
        }
        if (initialState.mayResolveDiffuse()) {
            return Capability.rejected(Reason.DIFFUSE_UNSUPPORTED,
                "CHANDRASEKHAR filtering cannot be used with diffuse initialization");
        }
        if (!initialState.isStationary()) {
            return Capability.rejected(Reason.STATIONARY_REQUIRED,
                "CHANDRASEKHAR filtering requires stationary initialization");
        }
        if (profile.hasMissingObservations()) {
            return Capability.rejected(Reason.MISSING_UNSUPPORTED,
                "CHANDRASEKHAR filtering cannot be used with missing observations");
        }
        if (profile.hasTimeVaryingSystemMatrices()) {
            return Capability.rejected(Reason.TIME_VARYING_UNSUPPORTED,
                "CHANDRASEKHAR filtering cannot be used with time-varying system matrices except intercepts");
        }
        return Capability.available();
    }

    private static boolean collapsedAutoWorthwhile(FilterOptions options) {
        return !options.stores(FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE,
                FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE)
            && !options.includes(FilterOptions.Surface.KALMAN_GAIN);
    }

    private static boolean hasTimeVaryingSystemMatrices(KalmanSSM model) {
        int nobs = model.observationCount();
        for (int t = 1; t < nobs; t++) {
            if (!sameBlock(model.designData(), model.designOffset(0), model.designOffset(t),
                    model.designLeadingDimension(), model.observationDimension(), model.stateCount())) {
                return true;
            }
            if (!sameBlock(model.obsCovData(), model.obsCovOffset(0), model.obsCovOffset(t),
                    model.obsCovLeadingDimension(), model.observationDimension(), model.observationDimension())) {
                return true;
            }
            if (!sameBlock(model.transitionData(), model.transitionOffset(0), model.transitionOffset(t),
                    model.transitionLeadingDimension(), model.stateCount(), model.stateCount())) {
                return true;
            }
            if (!sameBlock(model.selectionData(), model.selectionOffset(0), model.selectionOffset(t),
                    model.selectionLeadingDimension(), model.stateCount(), model.stateDisturbanceCount())) {
                return true;
            }
            if (!sameBlock(model.stateCovarianceData(), model.stateCovarianceOffset(0), model.stateCovarianceOffset(t),
                    model.stateCovarianceLeadingDimension(), model.stateDisturbanceCount(), model.stateDisturbanceCount())) {
                return true;
            }
        }
        return false;
    }

    private static boolean sameBlock(double[] data,
                                     int firstOffset,
                                     int secondOffset,
                                     int leadingDimension,
                                     int rows,
                                     int cols) {
        for (int row = 0; row < rows; row++) {
            int firstRow = firstOffset + row * leadingDimension;
            int secondRow = secondOffset + row * leadingDimension;
            for (int col = 0; col < cols; col++) {
                if (data[firstRow + col] != data[secondRow + col]) {
                    return false;
                }
            }
        }
        return true;
    }
}
