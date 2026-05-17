package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class FilterRouteSupportTest {

    @Test
    void capabilityMatrixExposesSupportedAndGatedReasonCodes() {
        KalmanSSM conventional = model(2, 2, 2, 4, false, false, false);
        InitialState known = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0, 0.0, 1.0});

        assertReason(conventional, known, FilterOptions.defaults(), FilterMethod.CONVENTIONAL,
            false, false, FilterRouteSupport.Reason.SUPPORTED);
        assertReason(conventional, known, FilterOptions.defaults(), FilterMethod.UNIVARIATE,
            false, false, FilterRouteSupport.Reason.SUPPORTED);
        assertReason(conventional, known, FilterOptions.defaults(), FilterMethod.UNIVARIATE,
            true, false, FilterRouteSupport.Reason.SUPPORTED);
        assertReason(conventional, known, FilterOptions.builder().concentratedLikelihood(true).build(),
            FilterMethod.CONVENTIONAL, false, false, FilterRouteSupport.Reason.SUPPORTED);
        assertReason(conventional, known, FilterOptions.builder()
                .concentratedLikelihood(true)
                .singularFallbackPolicy(FilterOptions.SingularFallback.UNIVARIATE)
                .build(),
            FilterMethod.CONVENTIONAL, false, false,
            FilterRouteSupport.Reason.SUPPORTED);
        assertReason(conventional, known, FilterOptions.builder().concentratedLikelihood(true).build(),
            FilterMethod.CONVENTIONAL, true, false, FilterRouteSupport.Reason.SUPPORTED);
        assertReason(conventional, known, FilterOptions.defaults(), FilterMethod.UNIVARIATE,
            false, true, FilterRouteSupport.Reason.SUPPORTED);

        KalmanSSM collapsed = model(3, 2, 2, 4, false, false, false);
        assertReason(collapsed, known, FilterOptions.builder().method(FilterMethod.COLLAPSED).build(),
            FilterMethod.COLLAPSED, false, false, FilterRouteSupport.Reason.SUPPORTED);
        assertReason(collapsed, known, FilterOptions.builder().method(FilterMethod.COLLAPSED).build(),
            FilterMethod.COLLAPSED, true, false, FilterRouteSupport.Reason.COMPLEX_ROUTE_UNSUPPORTED);
        assertReason(collapsed, known, FilterOptions.builder().method(FilterMethod.COLLAPSED).build(),
            FilterMethod.COLLAPSED, false, true, FilterRouteSupport.Reason.SUPPORTED);
        assertReason(collapsed, known, FilterOptions.builder()
                .method(FilterMethod.COLLAPSED)
                .concentratedLikelihood(true)
                .build(),
            FilterMethod.COLLAPSED, false, false, FilterRouteSupport.Reason.CONCENTRATED_UNSUPPORTED);
        assertReason(conventional, known, FilterOptions.builder().method(FilterMethod.COLLAPSED).build(),
            FilterMethod.COLLAPSED, false, false, FilterRouteSupport.Reason.DIMENSION_UNSUPPORTED);

        KalmanSSM nonDiagonalCollapsed = model(3, 2, 2, 4, true, false, false);
        FilterOptions gainProfile = FilterOptions.builder()
            .method(FilterMethod.COLLAPSED)
            .retainOnly(FilterOptions.Surface.KALMAN_GAIN)
            .build();
        assertReason(nonDiagonalCollapsed, InitialState.diffuse(2), gainProfile,
            FilterMethod.COLLAPSED, false, false,
            FilterRouteSupport.Reason.SUPPORTED);
        assertReason(nonDiagonalCollapsed, InitialState.diffuse(2), FilterOptions.builder()
                .method(FilterMethod.COLLAPSED)
                .retainAll()
                .drop(FilterOptions.Surface.KALMAN_GAIN)
                .build(),
            FilterMethod.COLLAPSED, false, false, FilterRouteSupport.Reason.SUPPORTED);
    }

    @Test
    void chandrasekharReasonCodesCaptureStatsmodelsStyleRestrictions() {
        KalmanSSM stationary = model(2, 2, 2, 4, false, false, false);
        InitialState stationaryInit = InitialState.stationary(stationary);
        FilterOptions options = FilterOptions.builder().method(FilterMethod.CHANDRASEKHAR).build();

        assertReason(stationary, stationaryInit, options, FilterMethod.CHANDRASEKHAR,
            false, false, FilterRouteSupport.Reason.SUPPORTED);
        assertReason(stationary, stationaryInit, options, FilterMethod.CHANDRASEKHAR,
            true, false, FilterRouteSupport.Reason.COMPLEX_ROUTE_UNSUPPORTED);
        assertReason(stationary, stationaryInit, options, FilterMethod.CHANDRASEKHAR,
            false, true, FilterRouteSupport.Reason.OVERRIDDEN_OBSERVATIONS_UNSUPPORTED);
        assertReason(stationary, stationaryInit, options.toBuilder().concentratedLikelihood(true).build(),
            FilterMethod.CHANDRASEKHAR, false, false, FilterRouteSupport.Reason.SUPPORTED);
        assertReason(stationary, stationaryInit, options.toBuilder().timing(FilterOptions.Timing.INIT_FILTERED).build(),
            FilterMethod.CHANDRASEKHAR, false, false, FilterRouteSupport.Reason.INIT_FILTERED_UNSUPPORTED);
        assertReason(stationary, InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0, 0.0, 1.0}),
            options, FilterMethod.CHANDRASEKHAR, false, false, FilterRouteSupport.Reason.STATIONARY_REQUIRED);
        assertReason(stationary, InitialState.diffuse(2), options, FilterMethod.CHANDRASEKHAR,
            false, false, FilterRouteSupport.Reason.DIFFUSE_UNSUPPORTED);

        KalmanSSM missing = model(2, 2, 2, 4, false, true, false);
        assertReason(missing, InitialState.stationary(missing), options, FilterMethod.CHANDRASEKHAR,
            false, false, FilterRouteSupport.Reason.MISSING_UNSUPPORTED);

        KalmanSSM timeVarying = model(2, 2, 2, 4, false, false, true);
        assertReason(timeVarying, InitialState.stationary(timeVarying), options, FilterMethod.CHANDRASEKHAR,
            false, false, FilterRouteSupport.Reason.TIME_VARYING_UNSUPPORTED);
    }

    @Test
    void explicitAutoRemainsConventionalWhenDefaultOptionsAreUsed() {
        KalmanSSM stationary = model(2, 2, 2, 64, false, false, false);
        InitialState stationaryInit = InitialState.stationary(stationary);

        assertEquals(FilterMethod.CHANDRASEKHAR, FilterRouteSupport.resolveAuto(stationary, stationaryInit,
            FilterOptions.builder().method(FilterMethod.AUTO).retainAll().build(), false, false));
        assertEquals(FilterMethod.CONVENTIONAL, FilterRouteSupport.resolveAuto(stationary, InitialState.diffuse(2),
            FilterOptions.builder().method(FilterMethod.AUTO).retainAll().build(), false, false));
        assertEquals(FilterMethod.CHANDRASEKHAR, FilterRouteSupport.resolveAuto(stationary, stationaryInit,
            FilterOptions.builder().method(FilterMethod.AUTO).retainAll().concentratedLikelihood(true).build(), false, false));
        assertEquals(FilterMethod.CONVENTIONAL, FilterRouteSupport.resolveAuto(stationary, stationaryInit,
            FilterOptions.builder().method(FilterMethod.AUTO).retainAll().timing(FilterOptions.Timing.INIT_FILTERED).build(), false, false));
        assertEquals(FilterMethod.CONVENTIONAL, FilterRouteSupport.resolveAuto(stationary, stationaryInit,
            FilterOptions.builder().method(FilterMethod.AUTO).retainAll().build(), false, true));
        assertEquals(FilterMethod.CONVENTIONAL, FilterRouteSupport.resolveAuto(stationary, stationaryInit,
            FilterOptions.builder().method(FilterMethod.AUTO).retainAll().build(), true, false));
        assertEquals(FilterMethod.CONVENTIONAL, FilterOptions.defaults().method());
    }

    @Test
    void explicitRouteProfilesTrySupportedFallbacks() {
        KalmanSSM conventionalOnly = model(2, 2, 2, 8, false, false, false);
        InitialState known = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0, 0.0, 1.0});
        FilterRouteSupport.StructuralProfile profile = FilterRouteSupport.analyze(conventionalOnly);
        FilterOptions profileOptions = FilterOptions.builder()
            .tryMethods(FilterMethod.COLLAPSED, FilterMethod.UNIVARIATE, FilterMethod.CONVENTIONAL)
            .build();

        assertEquals(FilterMethod.UNIVARIATE, FilterRouteSupport.resolveProfile(conventionalOnly, known,
            profileOptions, false, false, profile));
        assertEquals(profileOptions, profileOptions.toBuilder().build());

        FilterOptions strictCollapsed = FilterOptions.builder()
            .method(FilterMethod.COLLAPSED)
            .build();
        assertEquals(FilterMethod.COLLAPSED, FilterRouteSupport.resolveProfile(conventionalOnly, known,
            strictCollapsed, false, false, profile));
    }

    @Test
    void forecastErrorStrategyUsesCanonicalSolveSequences() {
        FilterOptions options = FilterOptions.builder()
            .forecastErrorStrategy(ForecastErrorStrategy.AUTO_THEN_LU)
            .build();

        assertEquals(ForecastErrorStrategy.AUTO_THEN_LU, options.forecastErrorStrategy());
        assertArrayEquals(new ForecastErrorSolver.Method[]{ ForecastErrorSolver.Method.AUTO, ForecastErrorSolver.Method.LU_SOLVE },
            options.forecastErrorSequenceRef());
        assertEquals(options, FilterOptions.builder()
            .forecastErrorStrategy(ForecastErrorStrategy.AUTO_THEN_LU)
            .build());
    }

    @Test
    void staticNonDiagonalUnivariateTransformReleasesTransientObservationWorkspace() {
        KalmanSSM shortModel = model(3, 2, 2, 4, true, false, false);
        KalmanSSM longModel = model(3, 2, 2, 40, true, false, false);
        InitialState initialState = InitialState.known(new double[]{0.0, 0.0}, new double[]{1.0, 0.0, 0.0, 1.0});
        FilterOptions options = FilterOptions.builder()
            .method(FilterMethod.UNIVARIATE)
            .retainOnly(FilterOptions.Surface.LIKELIHOOD)
            .build();

        UnivariateFilter.Pool shortPool = new UnivariateFilter.Pool();
        UnivariateFilter.filter(shortModel, initialState, shortPool, options);
        UnivariateFilter.Pool longPool = new UnivariateFilter.Pool();
        UnivariateFilter.filter(longModel, initialState, longPool, options);

        assertEquals(0L, shortPool.retainedObservationTransformDoubleCount());
        assertEquals(0L, longPool.retainedObservationTransformDoubleCount());
        assertEquals(shortPool.retainedScratchDoubleCount(), longPool.retainedScratchDoubleCount());
    }

    private static void assertReason(KalmanSSM model,
                                     InitialState initialState,
                                     FilterOptions options,
                                     FilterMethod method,
                                     boolean complex,
                                     boolean overriddenObservations,
                                     FilterRouteSupport.Reason expected) {
        assertEquals(expected, FilterRouteSupport.reason(model, initialState, options, method, complex,
            overriddenObservations));
    }

    private static KalmanSSM model(int kEndog,
                                         int kStates,
                                         int kPosdef,
                                         int nobs,
                                         boolean nonDiagonalObservationCovariance,
                                         boolean missingObservation,
                                         boolean timeVaryingTransition) {
        double[] design = new double[kEndog * kStates];
        for (int i = 0; i < Math.min(kEndog, kStates); i++) {
            design[i * kStates + i] = 1.0;
        }
        double[] obsCov = new double[kEndog * kEndog];
        for (int i = 0; i < kEndog; i++) {
            obsCov[i * kEndog + i] = 0.5 + 0.1 * i;
        }
        if (nonDiagonalObservationCovariance && kEndog > 1) {
            obsCov[1] = 0.05;
            obsCov[kEndog] = 0.05;
        }
        double[] transition = timeVaryingTransition
            ? new double[kStates * kStates * nobs]
            : new double[kStates * kStates];
        for (int t = 0; t < (timeVaryingTransition ? nobs : 1); t++) {
            int offset = t * kStates * kStates;
            for (int i = 0; i < kStates; i++) {
                transition[offset + i * kStates + i] = timeVaryingTransition && t > 0 ? 0.1 * t : 0.0;
            }
        }
        double[] selection = new double[kStates * kPosdef];
        for (int i = 0; i < Math.min(kStates, kPosdef); i++) {
            selection[i * kPosdef + i] = 1.0;
        }
        double[] stateCov = new double[kPosdef * kPosdef];
        for (int i = 0; i < kPosdef; i++) {
            stateCov[i * kPosdef + i] = 0.2;
        }
        double[] endog = new double[kEndog * nobs];
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                endog[t * kEndog + i] = 0.1 * (t + 1) * (i + 1);
            }
        }
        KalmanSSM.Builder builder = KalmanSSM.builder(kEndog, kStates, kPosdef, nobs)
            .design(design, false)
            .obsCov(obsCov, false)
            .transition(transition, timeVaryingTransition)
            .selection(selection, false)
            .stateCovariance(stateCov, false)
            .endog(endog);
        if (missingObservation) {
            boolean[] missing = new boolean[kEndog * nobs];
            missing[Math.min(kEndog * nobs - 1, kEndog)] = true;
            builder.missing(missing);
        } else {
            builder.allObserved();
        }
        return builder.build();
    }
}