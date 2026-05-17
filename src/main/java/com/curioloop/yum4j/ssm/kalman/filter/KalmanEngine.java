package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Objects;

public final class KalmanEngine {

    private KalmanEngine() {
    }

    public static final class Workspace {
        private final DoubleArena filterScratchArena = new DoubleArena();
        private final DoubleArena filterResultArena = new DoubleArena();
        private final KalmanFilter.Pool filterPool = new KalmanFilter.Pool(filterScratchArena, false, filterResultArena, false);
        private final UnivariateFilter.Pool univariateFilterPool = new UnivariateFilter.Pool(filterScratchArena, false, filterResultArena, false);
        private final CollapsedKalmanFilter.Pool collapsedFilterPool = new CollapsedKalmanFilter.Pool(filterScratchArena, false, filterResultArena, false);
        private final ChandrasekharKalmanFilter.Pool chandrasekharFilterPool = new ChandrasekharKalmanFilter.Pool(filterScratchArena, false, filterResultArena, false);
        private final ZKalmanFilter.Pool complexFilterPool = new ZKalmanFilter.Pool(filterScratchArena, false, filterResultArena, false);
        private final InitialTimingWorkspace timingWorkspace = new InitialTimingWorkspace();

        public void reserveFilter(KalmanSSM model,
                                  FilterOptions options) {
            filterPool.reserve(model, options);
            refreshFilterScratchBackings();
        }

        public void reserveUnivariateFilter(KalmanSSM model,
                                            FilterOptions options) {
            univariateFilterPool.reserve(model, options);
            refreshFilterScratchBackings();
        }

        public void reserveComplexFilter(KalmanSSM model,
                                         FilterOptions options) {
            complexFilterPool.reserve(model, options);
            refreshFilterScratchBackings();
        }

        public void reserveCollapsedFilter(KalmanSSM model,
                                           InitialState initialState,
                                           FilterOptions options) {
            collapsedFilterPool.reserve(model, initialState, options);
            refreshFilterScratchBackings();
        }

        public void reserveChandrasekharFilter(KalmanSSM model,
                                               InitialState initialState,
                                               FilterOptions options) {
            chandrasekharFilterPool.reserve(model, initialState, options);
            refreshFilterScratchBackings();
        }

        public long retainedRealFilterScratchDoubleCount() {
            return (usesRealFilterScratchArena() ? filterScratchArena.retainedDoubleCount() : 0L)
                + filterPool.retainedAuxiliaryScratchDoubleCount()
                + univariateFilterPool.retainedAuxiliaryScratchDoubleCount()
                + collapsedFilterPool.retainedAuxiliaryScratchDoubleCount()
                + chandrasekharFilterPool.retainedAuxiliaryScratchDoubleCount()
                + timingWorkspace.retainedDoubleCount();
        }

        public long retainedRealFilterResultDoubleCount() {
            return (usesRealFilterResultArena() ? filterResultArena.retainedDoubleCount() : 0L)
                + collapsedFilterPool.retainedAuxiliaryResultDoubleCount();
        }

        public long retainedRealFilterTotalByteCount() {
            return (retainedRealFilterScratchDoubleCount() + retainedRealFilterResultDoubleCount()) * Double.BYTES;
        }

        public long retainedComplexFilterWorkspaceScratchDoubleCount() {
            return complexFilterPool.retainedScratchDoubleCount() + timingWorkspace.retainedDoubleCount();
        }

        public long retainedComplexFilterWorkspaceResultDoubleCount() {
            return complexFilterPool.usesResultArena() ? filterResultArena.retainedDoubleCount() : 0L;
        }

        public long retainedComplexFilterWorkspaceTotalByteCount() {
            return (retainedComplexFilterWorkspaceScratchDoubleCount()
                + retainedComplexFilterWorkspaceResultDoubleCount()) * Double.BYTES;
        }

        public long retainedFilterWorkspaceScratchDoubleCount() {
            return (usesAnyFilterScratchArena() ? filterScratchArena.retainedDoubleCount() : 0L)
                + filterPool.retainedAuxiliaryScratchDoubleCount()
                + univariateFilterPool.retainedAuxiliaryScratchDoubleCount()
                + collapsedFilterPool.retainedAuxiliaryScratchDoubleCount()
                + chandrasekharFilterPool.retainedAuxiliaryScratchDoubleCount()
                + complexFilterPool.retainedAuxiliaryScratchDoubleCount()
                + timingWorkspace.retainedDoubleCount();
        }

        public long retainedFilterWorkspaceResultDoubleCount() {
            return (usesAnyFilterResultArena() ? filterResultArena.retainedDoubleCount() : 0L)
                + collapsedFilterPool.retainedAuxiliaryResultDoubleCount();
        }

        public long retainedFilterWorkspaceTotalByteCount() {
            return (retainedFilterWorkspaceScratchDoubleCount() + retainedFilterWorkspaceResultDoubleCount()) * Double.BYTES;
        }

        public void releaseRetainedScratch() {
            filterPool.releaseRetainedScratch();
            univariateFilterPool.releaseRetainedScratch();
            collapsedFilterPool.releaseRetainedScratch();
            chandrasekharFilterPool.releaseRetainedScratch();
            complexFilterPool.releaseRetainedScratch();
            filterScratchArena.release();
            timingWorkspace.release();
        }

        public void releaseRetainedResults() {
            filterPool.releaseRetainedResults();
            univariateFilterPool.releaseRetainedResults();
            collapsedFilterPool.releaseRetainedResults();
            chandrasekharFilterPool.releaseRetainedResults();
            complexFilterPool.releaseRetainedResults();
            filterResultArena.release();
        }

        public long retainedFilterScratchDoubleCount() {
            return filterPool.retainedScratchDoubleCount();
        }

        public long retainedFilterResultDoubleCount() {
            return filterPool.usesResultArena() ? filterResultArena.retainedDoubleCount() : 0L;
        }

        public long retainedFilterTotalByteCount() {
            return filterPool.retainedTotalByteCount();
        }

        public void releaseFilterRetainedScratch() {
            filterPool.releaseRetainedScratch();
            releaseSharedFilterScratchIfUnused();
        }

        public void releaseFilterRetainedResults() {
            filterPool.releaseRetainedResults();
            releaseSharedFilterResultIfUnused();
        }

        public long retainedUnivariateScratchDoubleCount() {
            return univariateFilterPool.retainedScratchDoubleCount();
        }

        public long retainedUnivariateResultDoubleCount() {
            return univariateFilterPool.usesResultArena() ? filterResultArena.retainedDoubleCount() : 0L;
        }

        public long retainedUnivariateTotalByteCount() {
            return univariateFilterPool.retainedTotalByteCount();
        }

        public void releaseUnivariateRetainedScratch() {
            univariateFilterPool.releaseRetainedScratch();
            releaseSharedFilterScratchIfUnused();
        }

        public void releaseUnivariateRetainedResults() {
            univariateFilterPool.releaseRetainedResults();
            releaseSharedFilterResultIfUnused();
        }

        public long retainedCollapsedScratchDoubleCount() {
            return collapsedFilterPool.retainedScratchDoubleCount();
        }

        public long retainedCollapsedResultDoubleCount() {
            return (collapsedFilterPool.usesResultArena() ? filterResultArena.retainedDoubleCount() : 0L)
                + collapsedFilterPool.retainedAuxiliaryResultDoubleCount();
        }

        public long retainedCollapsedTotalByteCount() {
            return collapsedFilterPool.retainedTotalByteCount();
        }

        public void releaseCollapsedRetainedScratch() {
            collapsedFilterPool.releaseRetainedScratch();
            releaseSharedFilterScratchIfUnused();
        }

        public void releaseCollapsedRetainedResults() {
            collapsedFilterPool.releaseRetainedResults();
            releaseSharedFilterResultIfUnused();
        }

        public long retainedChandrasekharScratchDoubleCount() {
            return chandrasekharFilterPool.retainedScratchDoubleCount();
        }

        public long retainedChandrasekharResultDoubleCount() {
            return chandrasekharFilterPool.usesResultArena() ? filterResultArena.retainedDoubleCount() : 0L;
        }

        public long retainedChandrasekharTotalByteCount() {
            return chandrasekharFilterPool.retainedTotalByteCount();
        }

        public void releaseChandrasekharRetainedScratch() {
            chandrasekharFilterPool.releaseRetainedScratch();
            releaseSharedFilterScratchIfUnused();
        }

        public void releaseChandrasekharRetainedResults() {
            chandrasekharFilterPool.releaseRetainedResults();
            releaseSharedFilterResultIfUnused();
        }

        public long retainedTimingScratchDoubleCount() {
            return timingWorkspace.retainedDoubleCount();
        }

        public long retainedTimingScratchByteCount() {
            return timingWorkspace.retainedByteCount();
        }

        public void releaseTimingRetainedScratch() {
            timingWorkspace.release();
        }

        public long retainedComplexFilterScratchDoubleCount() {
            return complexFilterPool.retainedScratchDoubleCount();
        }

        public long retainedComplexFilterResultDoubleCount() {
            return complexFilterPool.usesResultArena() ? filterResultArena.retainedDoubleCount() : 0L;
        }

        public long retainedComplexFilterTotalByteCount() {
            return complexFilterPool.retainedTotalByteCount();
        }

        public void releaseComplexFilterRetainedScratch() {
            complexFilterPool.releaseRetainedScratch();
            releaseSharedFilterScratchIfUnused();
        }

        public void releaseComplexFilterRetainedResults() {
            complexFilterPool.releaseRetainedResults();
            releaseSharedFilterResultIfUnused();
        }

        private void invalidateBorrowedFilterResults() {
            filterPool.invalidateBorrowedResult();
            univariateFilterPool.invalidateBorrowedResult();
            collapsedFilterPool.invalidateBorrowedResult();
            chandrasekharFilterPool.invalidateBorrowedResult();
            complexFilterPool.invalidateBorrowedResult();
        }

        private void refreshFilterScratchBackings() {
            filterPool.refreshScratchBacking();
            univariateFilterPool.refreshScratchBacking();
            collapsedFilterPool.refreshScratchBacking();
            chandrasekharFilterPool.refreshScratchBacking();
            complexFilterPool.refreshScratchBacking();
        }

        private boolean usesRealFilterScratchArena() {
            return filterPool.usesScratchArena()
                || univariateFilterPool.usesScratchArena()
                || collapsedFilterPool.usesScratchArena()
                || chandrasekharFilterPool.usesScratchArena();
        }

        private boolean usesAnyFilterScratchArena() {
            return usesRealFilterScratchArena() || complexFilterPool.usesScratchArena();
        }

        private void releaseSharedFilterScratchIfUnused() {
            if (!usesAnyFilterScratchArena()) {
                filterScratchArena.release();
            }
        }

        private boolean usesRealFilterResultArena() {
            return filterPool.usesResultArena()
                || univariateFilterPool.usesResultArena()
                || collapsedFilterPool.usesResultArena()
                || chandrasekharFilterPool.usesResultArena();
        }

        private boolean usesAnyFilterResultArena() {
            return usesRealFilterResultArena() || complexFilterPool.usesResultArena();
        }

        private void releaseSharedFilterResultIfUnused() {
            if (!usesAnyFilterResultArena()) {
                filterResultArena.release();
            }
        }
    }

    public static Workspace workspace() {
        return new Workspace();
    }

    private static final class InitialTimingWorkspace {
        private double[] filteredState;
        private double[] filteredStateCov;
        private double[] filteredDiffuseStateCov;
        private double[] predictedState;
        private double[] predictedStateCov;
        private double[] predictedDiffuseStateCov;
        private double[] covarianceWork;
        private InitialState.StationaryWorkspace stationaryWorkspace;

        void ensure(KalmanSSM model, boolean needDiffuse) {
            int kStates = model.stateCount();
            int scalarWidth = model.complex() ? 2 : 1;
            int stateLength = scalarWidth * kStates;
            int covarianceLength = scalarWidth * kStates * kStates;
            filteredState = ensure(filteredState, stateLength);
            filteredStateCov = ensure(filteredStateCov, covarianceLength);
            predictedState = ensure(predictedState, stateLength);
            predictedStateCov = ensure(predictedStateCov, covarianceLength);
            if (needDiffuse) {
                filteredDiffuseStateCov = ensure(filteredDiffuseStateCov, covarianceLength);
                predictedDiffuseStateCov = ensure(predictedDiffuseStateCov, covarianceLength);
                covarianceWork = null;
            } else {
                filteredDiffuseStateCov = null;
                predictedDiffuseStateCov = null;
                covarianceWork = ensure(covarianceWork, covarianceLength);
            }
        }

        InitialState.StationaryWorkspace stationaryWorkspace() {
            if (stationaryWorkspace == null) {
                stationaryWorkspace = new InitialState.StationaryWorkspace();
            }
            return stationaryWorkspace;
        }

        long retainedDoubleCount() {
            return doubleCount(filteredState)
                + doubleCount(filteredStateCov)
                + doubleCount(filteredDiffuseStateCov)
                + doubleCount(predictedState)
                + doubleCount(predictedStateCov)
                + doubleCount(predictedDiffuseStateCov)
                + doubleCount(covarianceWork)
                + (stationaryWorkspace == null ? 0L : stationaryWorkspace.retainedDoubleCount());
        }

        long retainedByteCount() {
            return retainedDoubleCount() * Double.BYTES;
        }

        void release() {
            filteredState = null;
            filteredStateCov = null;
            filteredDiffuseStateCov = null;
            predictedState = null;
            predictedStateCov = null;
            predictedDiffuseStateCov = null;
            covarianceWork = null;
            if (stationaryWorkspace != null) {
                stationaryWorkspace.release();
                stationaryWorkspace = null;
            }
        }

        private static double[] ensure(double[] values, int length) {
            return values == null || values.length < length ? new double[length] : values;
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }
    }

    public static Builder builder(KalmanSSM model) {
        return new Builder(model);
    }

    public static FilterResult filter(KalmanSSM model, InitialState initialState) {
        return filter(model, initialState, null, FilterOptions.defaults());
    }

    public static FilterResult filter(KalmanSSM model,
                                      InitialState initialState,
                                      FilterOptions options) {
        return filter(model, initialState, null, options);
    }

    public static FilterResult filterBorrowedUnsafe(KalmanSSM model,
                                                    InitialState initialState,
                                                    Workspace workspace,
                                                    FilterOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateBorrowedFilterResults();
        FilterResult result = filter(model, initialState,
            workspace.filterPool,
            workspace.univariateFilterPool,
            workspace.collapsedFilterPool,
            workspace.chandrasekharFilterPool,
            workspace.timingWorkspace,
            options);
        workspace.refreshFilterScratchBackings();
        return result;
    }

    public static FilterResult filterBorrowedUnsafe(KalmanSSM model,
                                                    InitialState initialState,
                                                    double[] endog,
                                                    int endogBase,
                                                    Workspace workspace,
                                                    FilterOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateBorrowedFilterResults();
        FilterResult result = filter(model, initialState, endog, endogBase,
            workspace.filterPool,
            workspace.univariateFilterPool,
            workspace.collapsedFilterPool,
            workspace.chandrasekharFilterPool,
            workspace.timingWorkspace,
            options);
        workspace.refreshFilterScratchBackings();
        return result;
    }

    public static FilterResult filterUnivariate(KalmanSSM model,
                                                InitialState initialState,
                                                FilterOptions options) {
        return filterUnivariate(model, initialState, null, options);
    }

    public static FilterResult filterUnivariateBorrowedUnsafe(KalmanSSM model,
                                                              InitialState initialState,
                                                              Workspace workspace,
                                                              FilterOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateBorrowedFilterResults();
        FilterResult result = filterUnivariate(model, initialState, workspace.univariateFilterPool, options);
        workspace.refreshFilterScratchBackings();
        return result;
    }

    public static FilterResult filterUnivariateBorrowedUnsafe(KalmanSSM model,
                                                              InitialState initialState,
                                                              double[] endog,
                                                              int endogBase,
                                                              Workspace workspace,
                                                              FilterOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateBorrowedFilterResults();
        FilterResult result = filterUnivariate(model, initialState, endog, endogBase,
            workspace.univariateFilterPool, options);
        workspace.refreshFilterScratchBackings();
        return result;
    }

    static FilterResult filterUnivariate(KalmanSSM model,
                                         InitialState initialState,
                                         UnivariateFilter.Pool pool,
                                         FilterOptions options) {
        return filter(model, initialState, null, pool, forceMethod(options, FilterMethod.UNIVARIATE));
    }

    static FilterResult filterUnivariate(KalmanSSM model,
                                         InitialState initialState,
                                         double[] endog,
                                         int endogBase,
                                         UnivariateFilter.Pool pool,
                                         FilterOptions options) {
        return filter(model, initialState, endog, endogBase, null, pool, forceMethod(options, FilterMethod.UNIVARIATE));
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState initialState,
                               KalmanFilter.Pool pool,
                               FilterOptions options) {
        return filter(model, initialState, pool, null, options);
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState initialState,
                               KalmanFilter.Pool pool,
                               UnivariateFilter.Pool univariatePool,
                               FilterOptions options) {
        return filter(model, initialState, pool, univariatePool, null, null, options);
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState initialState,
                               KalmanFilter.Pool pool,
                               UnivariateFilter.Pool univariatePool,
                               CollapsedKalmanFilter.Pool collapsedPool,
                               ChandrasekharKalmanFilter.Pool chandrasekharPool,
                               FilterOptions options) {
        return filter(model, initialState, pool, univariatePool, collapsedPool, chandrasekharPool, null, options);
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState initialState,
                               KalmanFilter.Pool pool,
                               UnivariateFilter.Pool univariatePool,
                               CollapsedKalmanFilter.Pool collapsedPool,
                               ChandrasekharKalmanFilter.Pool chandrasekharPool,
                               InitialTimingWorkspace timingWorkspace,
                               FilterOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        FilterOptions resolvedOptions = options == null ? FilterOptions.defaults() : options;
        validateSupportedOptions(resolvedOptions);
        InitialState effectiveInitialState = resolveInitialStateForTiming(model, initialState,
            resolvedOptions.timing(), timingWorkspace);
        FilterRouteSupport.StructuralProfile routeProfile = FilterRouteSupport.analyze(model);
        FilterMethod method = FilterRouteSupport.resolveProfile(model, effectiveInitialState,
            resolvedOptions, false, false, routeProfile);
        validateRoute(model, effectiveInitialState, resolvedOptions, method, false, false, routeProfile);
        validateConcentratedLikelihood(resolvedOptions, effectiveInitialState, method, false);
        FilterOptions spec = resolvedOptions;
        return switch (method) {
            case AUTO, CONVENTIONAL -> {
                if (pool != null) {
                    pool.reserve(model, spec);
                }
                try {
                    yield KalmanFilter.filter(model, effectiveInitialState, pool, spec);
                } catch (KalmanFilter.SingularForecastException failure) {
                    if (!resolvedOptions.allowsUnivariateFallback()) {
                        throw failure;
                    }
                    if (univariatePool != null) {
                        univariatePool.reserve(model, spec);
                    }
                    FilterResult result = UnivariateFilter.filter(model, effectiveInitialState, univariatePool, spec);
                    if (resolvedOptions.fallbackTelemetry()) {
                        result.markFullUnivariateFallback();
                    }
                    yield result;
                }
            }
            case UNIVARIATE -> {
                if (univariatePool != null) {
                    univariatePool.reserve(model, spec);
                }
                yield UnivariateFilter.filter(model, effectiveInitialState, univariatePool, spec);
            }
            case COLLAPSED -> CollapsedKalmanFilter.filter(model, effectiveInitialState, pool, collapsedPool, spec);
            case CHANDRASEKHAR -> ChandrasekharKalmanFilter.filter(model, effectiveInitialState, pool, chandrasekharPool, spec);
        };
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState initialState,
                               double[] endog,
                               int endogBase,
                               KalmanFilter.Pool pool,
                               UnivariateFilter.Pool univariatePool,
                               FilterOptions options) {
        return filter(model, initialState, endog, endogBase, pool, univariatePool, null, null, options);
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState initialState,
                               double[] endog,
                               int endogBase,
                               KalmanFilter.Pool pool,
                               UnivariateFilter.Pool univariatePool,
                               CollapsedKalmanFilter.Pool collapsedPool,
                               ChandrasekharKalmanFilter.Pool chandrasekharPool,
                               FilterOptions options) {
        return filter(model, initialState, endog, endogBase, pool, univariatePool,
            collapsedPool, chandrasekharPool, null, options);
        }

        static FilterResult filter(KalmanSSM model,
                       InitialState initialState,
                       double[] endog,
                       int endogBase,
                       KalmanFilter.Pool pool,
                       UnivariateFilter.Pool univariatePool,
                       CollapsedKalmanFilter.Pool collapsedPool,
                       ChandrasekharKalmanFilter.Pool chandrasekharPool,
                       InitialTimingWorkspace timingWorkspace,
                       FilterOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        Objects.requireNonNull(endog, "endog");
        FilterOptions resolvedOptions = options == null ? FilterOptions.defaults() : options;
        validateSupportedOptions(resolvedOptions);
        InitialState effectiveInitialState = resolveInitialStateForTiming(model, initialState,
            resolvedOptions.timing(), timingWorkspace);
        FilterRouteSupport.StructuralProfile routeProfile = FilterRouteSupport.analyze(model);
        FilterMethod method = FilterRouteSupport.resolveProfile(model, effectiveInitialState,
            resolvedOptions, false, true, routeProfile);
        validateRoute(model, effectiveInitialState, resolvedOptions, method, false, true, routeProfile);
        validateConcentratedLikelihood(resolvedOptions, effectiveInitialState, method, false);
        FilterOptions spec = resolvedOptions;
        return switch (method) {
            case AUTO, CONVENTIONAL -> {
                if (pool != null) {
                    pool.reserve(model, spec);
                }
                try {
                    yield KalmanFilter.filter(model, effectiveInitialState, endog, endogBase, pool, spec);
                } catch (KalmanFilter.SingularForecastException failure) {
                    if (!resolvedOptions.allowsUnivariateFallback()) {
                        throw failure;
                    }
                    if (univariatePool != null) {
                        univariatePool.reserve(model, spec);
                    }
                    FilterResult result = UnivariateFilter.filter(model, effectiveInitialState, endog, endogBase, univariatePool, spec);
                    if (resolvedOptions.fallbackTelemetry()) {
                        result.markFullUnivariateFallback();
                    }
                    yield result;
                }
            }
            case UNIVARIATE -> {
                if (univariatePool != null) {
                    univariatePool.reserve(model, spec);
                }
                yield UnivariateFilter.filter(model, effectiveInitialState, endog, endogBase, univariatePool, spec);
            }
            case COLLAPSED -> CollapsedKalmanFilter.filter(model, effectiveInitialState, endog, endogBase,
                pool, collapsedPool, spec);
            case CHANDRASEKHAR -> throw FilterRouteSupport
                .check(model, effectiveInitialState, resolvedOptions, method, false, true, routeProfile)
                .exception(method);
        };
    }

    public static ZFilterResult filterComplex(KalmanSSM model,
                                              InitialState initialState,
                                              FilterOptions options) {
        return filterComplex(model, initialState, null, options);
    }

    static ZFilterResult filterComplex(KalmanSSM model,
                                       InitialState initialState,
                                       ZKalmanFilter.Pool pool,
                                       FilterOptions options) {
        return filterComplex(model, initialState, pool, null, options);
    }

    private static ZFilterResult filterComplex(KalmanSSM model,
                                               InitialState initialState,
                                               ZKalmanFilter.Pool pool,
                                               InitialTimingWorkspace timingWorkspace,
                                               FilterOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        FilterOptions resolvedOptions = options == null ? FilterOptions.defaults() : options;
        validateSupportedOptions(resolvedOptions);
        InitialState effectiveInitialState = resolveInitialStateForTiming(model, initialState,
            resolvedOptions.timing(), timingWorkspace);
        FilterRouteSupport.StructuralProfile routeProfile = FilterRouteSupport.analyze(model);
        FilterMethod method = FilterRouteSupport.resolveProfile(model, effectiveInitialState,
            resolvedOptions, true, false, routeProfile);
        validateRoute(model, effectiveInitialState, resolvedOptions, method, true, false, routeProfile);
        validateConcentratedLikelihood(resolvedOptions, effectiveInitialState, method, true);
        FilterOptions spec = resolvedOptions;
        return switch (method) {
            case AUTO, CONVENTIONAL -> ZKalmanFilter.filter(model, effectiveInitialState, pool, spec);
            case UNIVARIATE -> ZKalmanFilter.filterUnivariate(model, effectiveInitialState, pool, spec);
            case COLLAPSED, CHANDRASEKHAR -> throw FilterRouteSupport
                .check(model, effectiveInitialState, resolvedOptions, method, true, false, routeProfile)
                .exception(method);
        };
    }

    static ZFilterResult filterComplex(KalmanSSM model,
                                       InitialState initialState,
                                       double[] endog,
                                       int endogBase,
                                       ZKalmanFilter.Pool pool,
                                       FilterOptions options) {
        return filterComplex(model, initialState, endog, endogBase, pool, null, options);
    }

    private static ZFilterResult filterComplex(KalmanSSM model,
                                               InitialState initialState,
                                               double[] endog,
                                               int endogBase,
                                               ZKalmanFilter.Pool pool,
                                               InitialTimingWorkspace timingWorkspace,
                                               FilterOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        Objects.requireNonNull(endog, "endog");
        FilterOptions resolvedOptions = options == null ? FilterOptions.defaults() : options;
        validateSupportedOptions(resolvedOptions);
        InitialState effectiveInitialState = resolveInitialStateForTiming(model, initialState,
            resolvedOptions.timing(), timingWorkspace);
        FilterRouteSupport.StructuralProfile routeProfile = FilterRouteSupport.analyze(model);
        FilterMethod method = FilterRouteSupport.resolveProfile(model, effectiveInitialState,
            resolvedOptions, true, true, routeProfile);
        validateRoute(model, effectiveInitialState, resolvedOptions, method, true, true, routeProfile);
        validateConcentratedLikelihood(resolvedOptions, effectiveInitialState, method, true);
        FilterOptions spec = resolvedOptions;
        return switch (method) {
            case AUTO, CONVENTIONAL -> ZKalmanFilter.filter(model, effectiveInitialState, endog, endogBase, pool, spec);
            case UNIVARIATE -> ZKalmanFilter.filterUnivariate(model, effectiveInitialState, endog, endogBase, pool, spec);
            case COLLAPSED, CHANDRASEKHAR -> throw FilterRouteSupport
                .check(model, effectiveInitialState, resolvedOptions, method, true, true, routeProfile)
                .exception(method);
        };
    }

    public static ZFilterResult filterComplexBorrowedUnsafe(KalmanSSM model,
                                                            InitialState initialState,
                                                            Workspace workspace,
                                                            FilterOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateBorrowedFilterResults();
        ZFilterResult result = filterComplex(model, initialState, workspace.complexFilterPool,
            workspace.timingWorkspace, options);
        workspace.refreshFilterScratchBackings();
        return result;
    }

    public static ZFilterResult filterComplexBorrowedUnsafe(KalmanSSM model,
                                                            InitialState initialState,
                                                            double[] endog,
                                                            int endogBase,
                                                            Workspace workspace,
                                                            FilterOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateBorrowedFilterResults();
        ZFilterResult result = filterComplex(model, initialState, endog, endogBase,
            workspace.complexFilterPool, workspace.timingWorkspace, options);
        workspace.refreshFilterScratchBackings();
        return result;
    }

    private static void validateRoute(KalmanSSM model,
                                      InitialState initialState,
                                      FilterOptions options,
                                      FilterMethod method,
                                      boolean complex,
                                      boolean overriddenObservations,
                                      FilterRouteSupport.StructuralProfile routeProfile) {
        FilterRouteSupport.Capability capability = FilterRouteSupport.check(model, initialState, options,
            method, complex, overriddenObservations, routeProfile);
        if (!capability.supported()) {
            throw capability.exception(method);
        }
    }

    private static FilterOptions forceMethod(FilterOptions options, FilterMethod method) {
        FilterOptions resolvedOptions = options == null ? FilterOptions.defaults() : options;
        return resolvedOptions.method() == method
            ? resolvedOptions
            : resolvedOptions.toBuilder().method(method).build();
    }

    private static void validateSupportedOptions(FilterOptions options) {
        for (ForecastErrorSolver.Method method : options.forecastErrorSequenceRef()) {
            if (method != ForecastErrorSolver.Method.AUTO
                && method != ForecastErrorSolver.Method.CHOLESKY_SOLVE
                && method != ForecastErrorSolver.Method.LU_SOLVE
                && method != ForecastErrorSolver.Method.UNIVARIATE) {
                throw new UnsupportedOperationException(
                    method + " forecast-error solving is not wired into KalmanEngine yet");
            }
        }
        if (options.singularFallbackPolicy() != FilterOptions.SingularFallback.DISABLED
                && options.singularFallbackPolicy() != FilterOptions.SingularFallback.UNIVARIATE) {
            throw new UnsupportedOperationException(
                options.singularFallbackPolicy() + " singular fallback is not wired into KalmanEngine yet");
        }
    }

    private static void validateConcentratedLikelihood(FilterOptions options,
                                                       InitialState initialState,
                                                       FilterMethod method,
                                                       boolean complex) {
        if (!options.concentratedLikelihood()) {
            return;
        }
        if (method != FilterMethod.CONVENTIONAL && method != FilterMethod.UNIVARIATE
            && method != FilterMethod.CHANDRASEKHAR && method != FilterMethod.AUTO) {
            throw new UnsupportedOperationException(
            "Generic concentrated likelihood is currently implemented for conventional, univariate, and Chandrasekhar filtering only");
        }
    }

    private static InitialState resolveInitialStateForTiming(KalmanSSM model,
                                                             InitialState initialState,
                                                             FilterOptions.Timing timing,
                                                             InitialTimingWorkspace workspace) {
        if (timing == FilterOptions.Timing.INIT_PREDICTED) {
            return initialState;
        }
        if (timing != FilterOptions.Timing.INIT_FILTERED) {
            throw new UnsupportedOperationException(timing + " timing is not supported");
        }
        int kStates = model.stateCount();
        boolean mayResolveDiffuse = initialState.mayResolveDiffuse();
        boolean mayResolveStationary = initialState.mayResolveStationary();
        if (workspace != null) {
            workspace.ensure(model, mayResolveDiffuse);
        }
        int scalarWidth = model.complex() ? 2 : 1;
        int stateLength = scalarWidth * kStates;
        int covarianceLength = scalarWidth * kStates * kStates;
        double[] filteredState = workspace == null ? new double[stateLength] : workspace.filteredState;
        double[] filteredStateCov = workspace == null ? new double[covarianceLength] : workspace.filteredStateCov;
        double[] filteredDiffuseStateCov = mayResolveDiffuse
            ? (workspace == null ? new double[covarianceLength] : workspace.filteredDiffuseStateCov)
            : null;
        InitialState.StationaryWorkspace stationaryWorkspace = workspace == null || !mayResolveStationary
            ? null
            : workspace.stationaryWorkspace();
        boolean diffuse = initialState.resolveInto(model, 0, stationaryWorkspace,
            filteredState, 0,
            filteredStateCov, 0,
            filteredDiffuseStateCov, 0);

        double[] predictedState = workspace == null ? new double[stateLength] : workspace.predictedState;
        double[] predictedStateCov = workspace == null ? new double[covarianceLength] : workspace.predictedStateCov;
        double[] predictedDiffuseStateCov = diffuse
            ? (workspace == null ? new double[covarianceLength] : workspace.predictedDiffuseStateCov)
            : null;
        predictInitialState(model, filteredState, predictedState, scalarWidth);
        double[] covarianceWork = diffuse
            ? predictedDiffuseStateCov
            : (workspace == null ? new double[covarianceLength] : workspace.covarianceWork);
        propagateInitialCovariance(model, filteredStateCov, predictedStateCov, covarianceWork, scalarWidth, true);

        if (diffuse) {
            propagateInitialCovariance(model, filteredDiffuseStateCov, predictedDiffuseStateCov, filteredStateCov,
                scalarWidth, false);
        }
        return InitialState.resolved(kStates,
            predictedState,
            predictedStateCov,
            predictedDiffuseStateCov)
            .withDiffuseTolerance(initialState.diffuseTolerance());
    }

    private static void predictInitialState(KalmanSSM model,
                                            double[] filteredState,
                                            double[] predictedState,
                                            int scalarWidth) {
        int kStates = model.stateCount();
        int transitionOffset = model.transitionOffset(0);
        int transitionLd = model.transitionLeadingDimension();
        double[] transition = model.transitionData();
        int stateInterceptOffset = model.stateInterceptOffset(0);
        double[] stateIntercept = model.stateInterceptData();
        for (int row = 0; row < kStates; row++) {
            int out = row * scalarWidth;
            predictedState[out] = stateIntercept[stateInterceptOffset + out];
            if (scalarWidth == 2) {
                predictedState[out + 1] = stateIntercept[stateInterceptOffset + out + 1];
            }
            int rowOffset = transitionOffset + row * scalarWidth * transitionLd;
            for (int col = 0; col < kStates; col++) {
                int transitionElement = rowOffset + col * scalarWidth;
                int stateElement = col * scalarWidth;
                addProduct(predictedState, out,
                    transition[transitionElement], scalarWidth == 1 ? 0.0 : transition[transitionElement + 1],
                    filteredState[stateElement], scalarWidth == 1 ? 0.0 : filteredState[stateElement + 1],
                    scalarWidth);
            }
        }
    }

    private static void propagateInitialCovariance(KalmanSSM model,
                                                   double[] source,
                                                   double[] target,
                                                   double[] work,
                                                   int scalarWidth,
                                                   boolean addStateNoise) {
        int kStates = model.stateCount();
        double[] transition = model.transitionData();
        int transitionOffset = model.transitionOffset(0);
        int transitionLd = model.transitionLeadingDimension();
        if (scalarWidth == 1 && work != null && work.length >= kStates * kStates) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans,
                kStates, kStates, kStates,
                1.0, transition, transitionOffset, transitionLd,
                source, 0, kStates,
                0.0, work, 0, kStates);
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans,
                kStates, kStates, kStates,
                1.0, work, 0, kStates,
                transition, transitionOffset, transitionLd,
                0.0, target, 0, kStates);
            if (addStateNoise && model.stateDisturbanceCount() > 0) {
                addInitialStateNoise(model, target, work, scalarWidth);
            }
            return;
        }
        for (int row = 0; row < kStates; row++) {
            int transitionRow = transitionOffset + row * scalarWidth * transitionLd;
            for (int col = 0; col < kStates; col++) {
                int targetElement = row * scalarWidth * kStates + col * scalarWidth;
                double sumRe = 0.0;
                double sumIm = 0.0;
                int transitionCol = transitionOffset + col * scalarWidth * transitionLd;
                for (int left = 0; left < kStates; left++) {
                    int leftTransition = transitionRow + left * scalarWidth;
                    double leftRe = transition[leftTransition];
                    double leftIm = scalarWidth == 1 ? 0.0 : transition[leftTransition + 1];
                    for (int right = 0; right < kStates; right++) {
                        int sourceElement = left * scalarWidth * kStates + right * scalarWidth;
                        int rightTransition = transitionCol + right * scalarWidth;
                        double sourceRe = source[sourceElement];
                        double sourceIm = scalarWidth == 1 ? 0.0 : source[sourceElement + 1];
                        double rightRe = transition[rightTransition];
                        double rightIm = scalarWidth == 1 ? 0.0 : -transition[rightTransition + 1];
                        double firstRe = leftRe * sourceRe - leftIm * sourceIm;
                        double firstIm = leftRe * sourceIm + leftIm * sourceRe;
                        sumRe += firstRe * rightRe - firstIm * rightIm;
                        sumIm += firstRe * rightIm + firstIm * rightRe;
                    }
                }
                target[targetElement] = sumRe;
                if (scalarWidth == 2) {
                    target[targetElement + 1] = sumIm;
                }
            }
        }
        if (addStateNoise && model.stateDisturbanceCount() > 0) {
            addInitialStateNoise(model, target, work, scalarWidth);
        }
    }

    private static void addInitialStateNoise(KalmanSSM model,
                                             double[] target,
                                             double[] work,
                                             int scalarWidth) {
        int kStates = model.stateCount();
        int kPosdef = model.stateDisturbanceCount();
        double[] selection = model.selectionData();
        int selectionOffset = model.selectionOffset(0);
        int selectionLd = model.selectionLeadingDimension();
        double[] stateCovariance = model.stateCovarianceData();
        int stateCovarianceOffset = model.stateCovarianceOffset(0);
        int stateCovarianceLd = model.stateCovarianceLeadingDimension();
        if (scalarWidth == 1 && work != null && work.length >= kStates * kPosdef) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans,
                kStates, kPosdef, kPosdef,
                1.0, selection, selectionOffset, selectionLd,
                stateCovariance, stateCovarianceOffset, stateCovarianceLd,
                0.0, work, 0, kPosdef);
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans,
                kStates, kStates, kPosdef,
                1.0, work, 0, kPosdef,
                selection, selectionOffset, selectionLd,
                1.0, target, 0, kStates);
            return;
        }
        for (int row = 0; row < kStates; row++) {
            int selectionRow = selectionOffset + row * scalarWidth * selectionLd;
            for (int col = 0; col < kStates; col++) {
                int targetElement = row * scalarWidth * kStates + col * scalarWidth;
                double sumRe = 0.0;
                double sumIm = 0.0;
                int selectionCol = selectionOffset + col * scalarWidth * selectionLd;
                for (int left = 0; left < kPosdef; left++) {
                    int leftSelection = selectionRow + left * scalarWidth;
                    double leftRe = selection[leftSelection];
                    double leftIm = scalarWidth == 1 ? 0.0 : selection[leftSelection + 1];
                    for (int right = 0; right < kPosdef; right++) {
                        int covarianceElement = stateCovarianceOffset + left * scalarWidth * stateCovarianceLd + right * scalarWidth;
                        int rightSelection = selectionCol + right * scalarWidth;
                        double covarianceRe = stateCovariance[covarianceElement];
                        double covarianceIm = scalarWidth == 1 ? 0.0 : stateCovariance[covarianceElement + 1];
                        double rightRe = selection[rightSelection];
                        double rightIm = scalarWidth == 1 ? 0.0 : -selection[rightSelection + 1];
                        double firstRe = leftRe * covarianceRe - leftIm * covarianceIm;
                        double firstIm = leftRe * covarianceIm + leftIm * covarianceRe;
                        sumRe += firstRe * rightRe - firstIm * rightIm;
                        sumIm += firstRe * rightIm + firstIm * rightRe;
                    }
                }
                target[targetElement] += sumRe;
                if (scalarWidth == 2) {
                    target[targetElement + 1] += sumIm;
                }
            }
        }
    }

    private static void addProduct(double[] target,
                                   int targetOffset,
                                   double leftRe,
                                   double leftIm,
                                   double rightRe,
                                   double rightIm,
                                   int scalarWidth) {
        target[targetOffset] += leftRe * rightRe - leftIm * rightIm;
        if (scalarWidth == 2) {
            target[targetOffset + 1] += leftRe * rightIm + leftIm * rightRe;
        }
    }

    public static final class Builder {
        private final KalmanSSM model;
        private InitialState initialState;
        private FilterOptions options = FilterOptions.defaults();
        private Workspace workspace;

        private Builder(KalmanSSM model) {
            this.model = Objects.requireNonNull(model, "model");
        }

        public Builder initialState(InitialState initialState) {
            this.initialState = Objects.requireNonNull(initialState, "initialState");
            return this;
        }

        public Builder options(FilterOptions options) {
            this.options = Objects.requireNonNull(options, "options");
            return this;
        }

        public Builder workspace(Workspace workspace) {
            this.workspace = Objects.requireNonNull(workspace, "workspace");
            return this;
        }

        public FilterResult filter() {
            if (workspace == null) {
                return KalmanEngine.filter(model, requireInitialState(), options);
            }
            FilterResult result = KalmanEngine.filterBorrowedUnsafe(model, requireInitialState(), workspace, options).copy();
            workspace.releaseRetainedResults();
            return result;
        }

        public ZFilterResult filterComplex() {
            if (workspace == null) {
                return KalmanEngine.filterComplex(model, requireInitialState(), options);
            }
            ZFilterResult result = KalmanEngine.filterComplexBorrowedUnsafe(model, requireInitialState(), workspace, options).copy();
            workspace.releaseRetainedResults();
            return result;
        }

        private InitialState requireInitialState() {
            if (initialState == null) {
                throw new IllegalStateException("initialState must be set before filtering");
            }
            return initialState;
        }
    }
}
