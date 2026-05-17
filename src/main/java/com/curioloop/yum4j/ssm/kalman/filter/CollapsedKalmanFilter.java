package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.arena.CollapsedScratchLayout;
import com.curioloop.yum4j.ssm.kalman.arena.CollapsedIntScratchLayout;
import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.arena.FilterResultLayout;
import com.curioloop.yum4j.ssm.kalman.arena.FilterDiffuseLayout;
import com.curioloop.yum4j.ssm.kalman.arena.IntArena;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;
import java.util.Objects;

final class CollapsedKalmanFilter {

    private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);

    private CollapsedKalmanFilter() {
    }

    static final class Pool {
        private KalmanSSM collapsedModel;
        private double[] collapsedModelDesign;
        private double[] collapsedModelObsIntercept;
        private double[] collapsedModelObsCov;
        private double[] collapsedModelEndog;
        private double[] scratchBacking;
        private int[] intScratchBacking;
        private CollapsedScratchLayout scratchLayout;
        private CollapsedIntScratchLayout intScratchLayout;
        private FilterResult pooledResult;
        private double[] resultBacking;
        private double[] diffuseResultBacking;
        private final DoubleArena scratchArena;
        private final IntArena intScratchArena = new IntArena();
        private final DoubleArena likelihoodAdjustmentArena = new DoubleArena();
        private final DoubleArena resultArena;
        private final boolean ownsScratchArena;
        private final boolean ownsResultArena;
        private double[] logLikelihoodAdjustmentBacking;
        private FilterResultLayout resultLayout;
        private FilterDiffuseLayout diffuseResultLayout;
        private KalmanFilter.Pool internalFilterPool;
        private KalmanFilter.Pool replayPool;
        private int diffuseResultBase;
        private int scratchKEndog;
        private int scratchKStates;
        private int scratchKPosdef;
        private int scratchNobs;
        private boolean scratchNeedsForecast;
        private boolean scratchNeedsForecastCovariance;
        private boolean scratchNeedsForecastDiffuseCovariance;
        private boolean scratchNeedsStandardizedForecastError;
        private boolean scratchNeedsKalmanGain;
        private long collapsedModelOwnedDoubleCount;

        Pool() {
            this(new DoubleArena(), true, new DoubleArena(), true);
        }

        Pool(DoubleArena scratchArena, boolean ownsScratchArena) {
            this(scratchArena, ownsScratchArena, new DoubleArena(), true);
        }

        Pool(DoubleArena scratchArena, boolean ownsScratchArena,
             DoubleArena resultArena, boolean ownsResultArena) {
            this.scratchArena = Objects.requireNonNull(scratchArena, "scratchArena");
            this.resultArena = Objects.requireNonNull(resultArena, "resultArena");
            this.ownsScratchArena = ownsScratchArena;
            this.ownsResultArena = ownsResultArena;
        }

        void reserve(KalmanSSM model, InitialState init, FilterOptions options) {
            Objects.requireNonNull(model, "model");
            Objects.requireNonNull(init, "init");
            FilterOptions resolvedOptions = options == null ? FilterOptions.defaults() : options;
            validateSupportedSlice(model, init, resolvedOptions);
            if (!canCollapseEveryPeriod(model)) {
                return;
            }
            ensureCollapse(model, resolvedOptions);
            boolean hasMissingObservations = KalmanSSMSupport.hasMissingObservations(model);
            FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(1,
                model.observationDimension(),
                model.stateCount(),
                model.observationCount(),
                hasMissingObservations);
            ensureResult(retainedLayout);
            if (init.mayResolveDiffuse()) {
                FilterDiffuseLayout retainedDiffuseLayout = resolvedOptions.createDiffuseLayout(1,
                    model.observationDimension(),
                    model.stateCount(),
                    model.observationCount(),
                    hasMissingObservations);
                ensureDiffuseResult(retainedDiffuseLayout);
                boolean nonDiagonalObservationCovariance = KalmanSSMSupport.hasNonDiagonalObservationCovariance(model);
                boolean nativeExactDiffuseObservationSurfaces = nonDiagonalObservationCovariance
                    && needsExactDiffuseObservationSurfaceReplay(resolvedOptions)
                    && !resolvedOptions.includes(FilterOptions.Surface.KALMAN_GAIN);
                if (needsExactDiffuseObservationSurfaceReplay(resolvedOptions) && !nativeExactDiffuseObservationSurfaces) {
                    ensureReplayPool().reserve(model, replayOptions(resolvedOptions));
                }
            }
        }

        FilterResult borrowResult(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
            ensureResult(layout);
            if (pooledResult == null) {
                pooledResult = new FilterResult();
            }
            return pooledResult.reuse(kEndog, kStates, nobs, resultBacking, resultLayout);
        }

        FilterResult borrowDiffuseResult(int kEndog,
                                         int kStates,
                                         int nobs,
                                         FilterResultLayout layout,
                                         FilterDiffuseLayout diffuseLayout) {
            ensureResult(layout);
            ensureDiffuseResult(diffuseLayout);
            if (pooledResult == null) {
                pooledResult = new FilterResult();
            }
            return pooledResult
                .reuse(kEndog, kStates, nobs, resultBacking, resultLayout)
                .reuseDiffuse(diffuseResultBacking, diffuseResultBase, diffuseResultLayout);
        }

        long retainedScratchDoubleCount() {
            return (usesScratchArena() ? scratchArena.retainedDoubleCount() : 0L)
                + retainedAuxiliaryScratchDoubleCount();
        }

        long retainedAuxiliaryScratchDoubleCount() {
            return (collapsedModel == null ? 0L : collapsedModelOwnedDoubleCount)
                + intArenaDoubleEquivalentCount()
                + (internalFilterPool == null ? 0L : internalFilterPool.retainedScratchDoubleCount())
                + (replayPool == null ? 0L : replayPool.retainedScratchDoubleCount());
        }

        boolean usesScratchArena() {
            return scratchBacking != null;
        }

        void refreshScratchBacking() {
            if (scratchLayout != null) {
                scratchBacking = scratchArena.backing();
            }
        }

        long retainedResultDoubleCount() {
            return resultArena.retainedDoubleCount()
                + retainedAuxiliaryResultDoubleCount();
        }

        long retainedAuxiliaryResultDoubleCount() {
            return (internalFilterPool == null ? 0L : internalFilterPool.retainedResultDoubleCount())
                + (replayPool == null ? 0L : replayPool.retainedResultDoubleCount());
        }

        long retainedTotalByteCount() {
            return (retainedScratchDoubleCount() + retainedResultDoubleCount()) * Double.BYTES;
        }

        void releaseRetainedScratch() {
            collapsedModel = null;
            collapsedModelDesign = null;
            collapsedModelObsIntercept = null;
            collapsedModelObsCov = null;
            collapsedModelEndog = null;
            scratchBacking = null;
            intScratchBacking = null;
            if (ownsScratchArena) {
                scratchArena.release();
            }
            intScratchArena.release();
            releaseLogLikelihoodAdjustment();
            scratchLayout = null;
            intScratchLayout = null;
            scratchKEndog = 0;
            scratchKStates = 0;
            scratchKPosdef = 0;
            scratchNobs = 0;
            scratchNeedsForecast = false;
            scratchNeedsForecastCovariance = false;
            scratchNeedsForecastDiffuseCovariance = false;
            scratchNeedsStandardizedForecastError = false;
            scratchNeedsKalmanGain = false;
            collapsedModelOwnedDoubleCount = 0L;
            if (replayPool != null) {
                replayPool.releaseRetainedScratch();
            }
            if (internalFilterPool != null) {
                internalFilterPool.releaseRetainedScratch();
            }
        }

        void releaseRetainedResults() {
            invalidateBorrowedResult();
            pooledResult = null;
            if (ownsResultArena) {
                resultArena.release();
            }
            if (internalFilterPool != null) {
                internalFilterPool.releaseRetainedResults();
            }
            if (replayPool != null) {
                replayPool.releaseRetainedResults();
            }
        }

        boolean usesResultArena() {
            return resultBacking != null || diffuseResultBacking != null;
        }

        void invalidateBorrowedResult() {
            resultBacking = null;
            diffuseResultBacking = null;
            resultLayout = null;
            diffuseResultLayout = null;
            diffuseResultBase = 0;
        }

        KalmanSSM ensureCollapsedModel(KalmanSSM source) {
            int kStates = source.stateCount();
            int kPosdef = source.stateDisturbanceCount();
            int nobs = source.observationCount();
            collapsedModelDesign = ensure(collapsedModelDesign, kStates * kStates * nobs);
            collapsedModelObsIntercept = ensure(collapsedModelObsIntercept, kStates);
            collapsedModelObsCov = ensure(collapsedModelObsCov, kStates * kStates);
            collapsedModelEndog = ensure(collapsedModelEndog, kStates * nobs);
            Arrays.fill(collapsedModelObsIntercept, 0, kStates, 0.0);
            setIdentity(collapsedModelObsCov, kStates);
            collapsedModel = KalmanSSM.builder(kStates, kStates, kPosdef, nobs)
                .design(collapsedModelDesign, true)
                .obsIntercept(collapsedModelObsIntercept, false)
                .obsCov(collapsedModelObsCov, false)
                .transition(source.transitionData(), isTransitionTimeVarying(source))
                .stateIntercept(source.stateInterceptData(), isStateInterceptTimeVarying(source))
                .selection(source.selectionData(), isSelectionTimeVarying(source))
                .stateCovariance(source.stateCovarianceData(), isStateCovarianceTimeVarying(source))
                .endog(collapsedModelEndog)
                .allObserved()
                .build();
            collapsedModelOwnedDoubleCount = doubleCount(collapsedModelDesign)
                + doubleCount(collapsedModelObsIntercept)
                + doubleCount(collapsedModelObsCov)
                + doubleCount(collapsedModelEndog);
            return collapsedModel;
        }

        private void ensureCollapse(KalmanSSM source, FilterOptions options) {
            int kEndog = source.observationDimension();
            int kStates = source.stateCount();
            int kPosdef = source.stateDisturbanceCount();
            int nobs = source.observationCount();
            ensureCollapsedModel(source);
            boolean needKalmanGain = options.includes(FilterOptions.Surface.KALMAN_GAIN);
            boolean needStandardizedForecastError = options.storesStandardizedForecastError();
            boolean needForecastDiffuseCovariance = options.storesForecastDiffuseCovariance();
            boolean needForecastCovariance = options.storesForecastCovariance()
                || needStandardizedForecastError
                || needKalmanGain;
            boolean needForecast = options.storesForecastMean()
                || options.storesForecastError()
                || needForecastCovariance
                || needForecastDiffuseCovariance
                || needStandardizedForecastError
                || needKalmanGain;
            int nextKEndog = Math.max(scratchKEndog, kEndog);
            int nextKStates = Math.max(scratchKStates, kStates);
            int nextKPosdef = Math.max(scratchKPosdef, kPosdef);
            int nextNobs = Math.max(scratchNobs, nobs);
            boolean nextNeedsForecast = scratchNeedsForecast || needForecast;
            boolean nextNeedsForecastCovariance = scratchNeedsForecastCovariance || needForecastCovariance;
            boolean nextNeedsForecastDiffuseCovariance = scratchNeedsForecastDiffuseCovariance || needForecastDiffuseCovariance;
            boolean nextNeedsStandardizedForecastError = scratchNeedsStandardizedForecastError || needStandardizedForecastError;
            boolean nextNeedsKalmanGain = scratchNeedsKalmanGain || needKalmanGain;
            if (scratchLayout == null
                    || nextKEndog != scratchKEndog
                    || nextKStates != scratchKStates
                    || nextKPosdef != scratchKPosdef
                    || nextNobs != scratchNobs
                    || nextNeedsForecast != scratchNeedsForecast
                    || nextNeedsForecastCovariance != scratchNeedsForecastCovariance
                    || nextNeedsForecastDiffuseCovariance != scratchNeedsForecastDiffuseCovariance
                    || nextNeedsStandardizedForecastError != scratchNeedsStandardizedForecastError
                    || nextNeedsKalmanGain != scratchNeedsKalmanGain) {
                scratchLayout = CollapsedScratchLayout.create(nextKEndog, nextKStates, nextKPosdef, nextNobs,
                    nextNeedsForecast,
                    nextNeedsForecastCovariance,
                    nextNeedsForecastDiffuseCovariance,
                    nextNeedsStandardizedForecastError,
                    nextNeedsKalmanGain);
                intScratchLayout = CollapsedIntScratchLayout.create(nextKEndog);
                scratchKEndog = nextKEndog;
                scratchKStates = nextKStates;
                scratchKPosdef = nextKPosdef;
                scratchNobs = nextNobs;
                scratchNeedsForecast = nextNeedsForecast;
                scratchNeedsForecastCovariance = nextNeedsForecastCovariance;
                scratchNeedsForecastDiffuseCovariance = nextNeedsForecastDiffuseCovariance;
                scratchNeedsStandardizedForecastError = nextNeedsStandardizedForecastError;
                scratchNeedsKalmanGain = nextNeedsKalmanGain;
            }
            scratchBacking = scratchArena.ensureCapacity(scratchLayout.totalLength());
            intScratchBacking = intScratchArena.ensureCapacity(intScratchLayout.totalLength());
        }

        private void ensureResult(FilterResultLayout layout) {
            resultLayout = layout;
            ensureResultArena();
        }

        private void ensureDiffuseResult(FilterDiffuseLayout layout) {
            diffuseResultLayout = layout;
            ensureResultArena();
        }

        private void ensureResultArena() {
            int regularLength = resultLayout == null ? 0 : resultLayout.totalLength();
            int diffuseLength = diffuseResultLayout == null ? 0 : diffuseResultLayout.totalLength();
            diffuseResultBase = regularLength;
            double[] backing = resultArena.ensureCapacity(regularLength + diffuseLength);
            resultBacking = backing;
            diffuseResultBacking = diffuseResultLayout == null ? null : backing;
        }

        private KalmanFilter.Pool ensureReplayPool() {
            if (replayPool == null) {
                replayPool = new KalmanFilter.Pool();
            }
            return replayPool;
        }

        private KalmanFilter.Pool ensureInternalFilterPool() {
            if (internalFilterPool == null) {
                internalFilterPool = new KalmanFilter.Pool(scratchArena, false);
            }
            return internalFilterPool;
        }

        private void releaseReplayRetainedStorage() {
            if (replayPool != null) {
                replayPool.releaseRetainedScratch();
                replayPool.releaseRetainedResults();
            }
            if (internalFilterPool != null) {
                internalFilterPool.releaseRetainedScratch();
                internalFilterPool.releaseRetainedResults();
            }
        }

        private double[] borrowLogLikelihoodAdjustment(int nobs) {
            logLikelihoodAdjustmentBacking = likelihoodAdjustmentArena.borrow(nobs);
            return logLikelihoodAdjustmentBacking;
        }

        private void releaseLogLikelihoodAdjustment() {
            logLikelihoodAdjustmentBacking = null;
            likelihoodAdjustmentArena.release();
        }

        private double[] ensure(double[] values, int length) {
            return values == null || values.length < length ? new double[length] : values;
        }

        private int[] ensure(int[] values, int length) {
            return values == null || values.length < length ? new int[length] : values;
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }

        private static long intCount(int[] values) {
            return values == null ? 0L : values.length * ((long) Integer.BYTES) / Double.BYTES;
        }

        private long intArenaDoubleEquivalentCount() {
            return intScratchArena.retainedIntCount() * ((long) Integer.BYTES) / Double.BYTES;
        }

        private static long modelDoubleCount(KalmanSSM model) {
            return model == null ? 0L : model.retainedModelDoubleCount();
        }
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState init,
                               KalmanFilter.Pool pool,
                               FilterOptions options) {
        return filter(model, init, pool, null, options);
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState init,
                               KalmanFilter.Pool pool,
                               Pool collapsedPool,
                               FilterOptions options) {
        return filter(model, init, null, 0, pool, collapsedPool, options);
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState init,
                               double[] endogOverride,
                               int endogBase,
                               KalmanFilter.Pool pool,
                               Pool collapsedPool,
                               FilterOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        KalmanSSMSupport.requireRealStorage(model, "model");
        if (endogOverride != null) {
            Objects.requireNonNull(endogOverride, "endogOverride");
        }
        FilterOptions resolvedOptions = options == null ? FilterOptions.defaults() : options;
        validateSupportedSlice(model, init, resolvedOptions);
        if (!canCollapseEveryPeriod(model)) {
            FilterOptions fallbackOptions = resolvedOptions.toBuilder()
                .method(FilterMethod.CONVENTIONAL)
                .build();
            if (endogOverride == null) {
                return KalmanFilter.filter(model, init, pool, fallbackOptions);
            }
            return KalmanFilter.filter(model, init, endogOverride, endogBase, pool, fallbackOptions);
        }
        try {
            boolean exactDiffuse = init.mayResolveDiffuse();
            boolean nonDiagonalObservationCovariance = exactDiffuse
                && KalmanSSMSupport.hasNonDiagonalObservationCovariance(model);
            boolean nativeExactDiffuseObservationSurfaces = nonDiagonalObservationCovariance
                && needsExactDiffuseObservationSurfaceReplay(resolvedOptions);
            boolean nativeExactDiffuseKalmanGain = nativeExactDiffuseObservationSurfaces
                && resolvedOptions.includes(FilterOptions.Surface.KALMAN_GAIN);
            boolean replayExactDiffuseObservationSurfaces = exactDiffuse
                && needsExactDiffuseObservationSurfaceReplay(resolvedOptions)
                && !nativeExactDiffuseObservationSurfaces;
            boolean preserveMissingForecastHistory = exactDiffuse && preservesMissingForecastHistory(model, resolvedOptions);
            boolean hasMissingObservations = KalmanSSMSupport.hasMissingObservations(model);
            FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(1,
                model.observationDimension(),
                model.stateCount(),
                model.observationCount(),
                hasMissingObservations);
            FilterDiffuseLayout retainedDiffuseLayout = exactDiffuse
                ? resolvedOptions.createDiffuseLayout(1,
                    model.observationDimension(),
                    model.stateCount(),
                    model.observationCount(),
                    hasMissingObservations)
                : null;
            CollapsedData collapsedData = collapse(model, resolvedOptions, collapsedPool, endogOverride, endogBase);
            FilterMethod internalMethod = FilterRouteSupport.includesRoute(resolvedOptions.routeFlags(), FilterMethod.UNIVARIATE)
                ? FilterMethod.UNIVARIATE
                : FilterMethod.CONVENTIONAL;
            FilterOptions internalOptions = internalOptions(resolvedOptions,
                retainedLayout,
                retainedDiffuseLayout,
                internalMethod,
                replayExactDiffuseObservationSurfaces,
                nativeExactDiffuseObservationSurfaces,
                nativeExactDiffuseKalmanGain);
            KalmanFilter.Pool internalPool = collapsedPool == null ? pool : collapsedPool.ensureInternalFilterPool();
            FilterResult collapsed = KalmanFilter.filter(collapsedData.model, init, internalPool, internalOptions);
            if (collapsedPool != null) {
                collapsedPool.refreshScratchBacking();
            }
            FilterResult exactDiffuseObservationSurfaces = replayExactDiffuseObservationSurfaces
                ? replayExactDiffuseObservationSurfaces(model, init, endogOverride, endogBase, resolvedOptions, collapsedPool)
                : null;

            FilterResult result = collapsedPool == null
                ? new FilterResult(model.observationDimension(), model.stateCount(), model.observationCount(), retainedLayout)
                : collapsedPool.borrowResult(model.observationDimension(), model.stateCount(), model.observationCount(), retainedLayout);
            if (exactDiffuse) {
                if (collapsedPool == null) {
                    FilterDiffuseLayout diffuseLayout = Objects.requireNonNull(retainedDiffuseLayout, "retainedDiffuseLayout");
                    result.allocateDiffuse(diffuseLayout.predictedDiffuseStateCovLength() > 0,
                        diffuseLayout.forecastErrorDiffuseCovLength() > 0);
                } else {
                    FilterDiffuseLayout diffuseLayout = Objects.requireNonNull(retainedDiffuseLayout, "retainedDiffuseLayout");
                    result = collapsedPool.borrowDiffuseResult(model.observationDimension(), model.stateCount(),
                        model.observationCount(), retainedLayout, diffuseLayout);
                }
            }
            copyStateSurfaces(collapsed, result);
            if (exactDiffuseObservationSurfaces == null) {
                rebuildObservationSurfaces(model, collapsed, result, resolvedOptions,
                    collapsedData.logLikelihoodAdjustment, collapsedData.logLikelihoodAdjustmentBase,
                    endogOverride, endogBase, nativeExactDiffuseKalmanGain, collapsedData.model, collapsedPool);
            } else {
                copyObservationSurfaces(exactDiffuseObservationSurfaces, result);
                rebuildLikelihood(collapsed, result,
                    collapsedData.logLikelihoodAdjustment, collapsedData.logLikelihoodAdjustmentBase);
            }
            result.converged = collapsed.converged;
            result.periodConverged = collapsed.periodConverged;
            result.nobsDiffuse = collapsed.nobsDiffuse;
            result.perObsKalmanGain = exactDiffuseObservationSurfaces != null
                && result.kalmanGainLength() > 0
                && exactDiffuseObservationSurfaces.perObsKalmanGain
                || nativeExactDiffuseKalmanGain
                && result.kalmanGainLength() > 0
                && collapsed.perObsKalmanGain;
            return result.applyConserveMemory(conserveMemoryMask(resolvedOptions, preserveMissingForecastHistory));
        } finally {
            releaseInternalRetainedStorage(collapsedPool == null ? pool : collapsedPool.internalFilterPool);
            if (collapsedPool != null) {
                collapsedPool.releaseLogLikelihoodAdjustment();
                collapsedPool.releaseReplayRetainedStorage();
            }
        }
    }

    private static void releaseInternalRetainedStorage(KalmanFilter.Pool pool) {
        if (pool != null) {
            pool.releaseRetainedScratch();
            pool.releaseRetainedResults();
        }
    }

    private static void validateSupportedSlice(KalmanSSM model, InitialState init, FilterOptions options) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        if (options.concentratedLikelihood()) {
            throw new UnsupportedOperationException("COLLAPSED filtering does not support concentrated likelihood");
        }
        if (kEndog <= kStates) {
            throw new UnsupportedOperationException(
                "COLLAPSED filtering requires observation dimension larger than state dimension");
        }
    }

    private static boolean canCollapseEveryPeriod(KalmanSSM model) {
        int kStates = model.stateCount();
        for (int t = 0; t < model.observationCount(); t++) {
            if (observedDimension(model, t) <= kStates) {
                return false;
            }
        }
        return true;
    }

    private static boolean needsExactDiffuseObservationSurfaceReplay(FilterOptions options) {
        return options.stores(FilterOptions.Surface.FORECAST_MEAN,
            FilterOptions.Surface.FORECAST_ERROR,
            FilterOptions.Surface.FORECAST_COVARIANCE,
            FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR,
            FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE)
            || options.includes(FilterOptions.Surface.KALMAN_GAIN);
    }

    private static FilterOptions internalOptions(FilterOptions options,
                                                 FilterResultLayout retainedLayout,
                                                 FilterDiffuseLayout retainedDiffuseLayout,
                                                 FilterMethod method,
                                                 boolean replayExactDiffuseObservationSurfaces,
                                                 boolean nativeExactDiffuseObservationSurfaces,
                                                 boolean nativeExactDiffuseKalmanGain) {
        FilterOptions.Builder builder = options.toBuilder()
            .method(method)
            .retainNone();
        if (needsInternalPredictedState(retainedLayout, replayExactDiffuseObservationSurfaces)) {
            builder.retain(FilterOptions.Surface.PREDICTED_STATE);
        }
        if (needsInternalPredictedStateCov(retainedLayout, replayExactDiffuseObservationSurfaces, nativeExactDiffuseKalmanGain)) {
            builder.retain(FilterOptions.Surface.PREDICTED_STATE_COVARIANCE);
        }
        if (retainedLayout.filteredStateLength() > 0) {
            builder.retain(FilterOptions.Surface.FILTERED_STATE);
        }
        if (retainedLayout.filteredStateCovLength() > 0) {
            builder.retain(FilterOptions.Surface.FILTERED_STATE_COVARIANCE);
        }
        if (retainedLayout.logLikelihoodObsLength() > 0) {
            builder.retain(FilterOptions.Surface.LIKELIHOOD);
        }
        if (needsInternalPredictedDiffuseStateCov(retainedDiffuseLayout,
                replayExactDiffuseObservationSurfaces,
                nativeExactDiffuseObservationSurfaces)) {
            builder.retain(FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE);
        }
        if (nativeExactDiffuseKalmanGain) {
            builder.retain(FilterOptions.Surface.KALMAN_GAIN);
        }
        return builder.build();
    }

    private static boolean needsInternalPredictedState(FilterResultLayout retainedLayout,
                                                       boolean replayExactDiffuseObservationSurfaces) {
        return retainedLayout.predictedStateLength() > 0
            || !replayExactDiffuseObservationSurfaces
                && (retainedLayout.forecastLength() > 0
                    || retainedLayout.forecastErrorLength() > 0
                    || retainedLayout.standardizedForecastErrorLength() > 0
                    || retainedLayout.kalmanGainLength() > 0);
    }

    private static boolean needsInternalPredictedStateCov(FilterResultLayout retainedLayout,
                                                          boolean replayExactDiffuseObservationSurfaces,
                                                          boolean nativeExactDiffuseKalmanGain) {
        return retainedLayout.predictedStateCovLength() > 0
            || !replayExactDiffuseObservationSurfaces
                && (retainedLayout.forecastErrorCovLength() > 0
                    || retainedLayout.standardizedForecastErrorLength() > 0
                    || retainedLayout.kalmanGainLength() > 0)
            || nativeExactDiffuseKalmanGain;
    }

    private static boolean needsInternalPredictedDiffuseStateCov(FilterDiffuseLayout retainedDiffuseLayout,
                                                                 boolean replayExactDiffuseObservationSurfaces,
                                                                 boolean nativeExactDiffuseObservationSurfaces) {
        if (retainedDiffuseLayout == null) {
            return false;
        }
        return retainedDiffuseLayout.predictedDiffuseStateCovLength() > 0
            || !replayExactDiffuseObservationSurfaces
                && nativeExactDiffuseObservationSurfaces
                && retainedDiffuseLayout.forecastErrorDiffuseCovLength() > 0;
    }

    private static boolean preservesMissingForecastHistory(KalmanSSM model, FilterOptions options) {
        return KalmanSSMSupport.hasMissingObservations(model)
            && options.stores(FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE,
                FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);
    }

    private static int conserveMemoryMask(FilterOptions options, boolean preserveMissingForecastHistory) {
        int mask = options.conserveMemoryMask();
        if (preserveMissingForecastHistory) {
            mask &= ~KalmanFilter.MEMORY_NO_PREDICTED;
        }
        return mask;
    }

    private static CollapsedData collapse(KalmanSSM source,
                                          FilterOptions options,
                                          Pool pool,
                                          double[] endogOverride,
                                          int endogBase) {
        int kEndog = source.observationDimension();
        int kStates = source.stateCount();
        int kPosdef = source.stateDisturbanceCount();
        int nobs = source.observationCount();
        if (pool != null) {
            pool.ensureCollapse(source, options);
        }
        KalmanSSM collapsed = pool == null ? new KalmanSSM(kStates, kStates, kPosdef, nobs) : pool.ensureCollapsedModel(source);
        CollapsedScratchLayout layout = pool == null ? null : pool.scratchLayout;
        double[] scratch = pool == null ? null : pool.scratchBacking;
        double[] logLikelihoodAdjustment = pool == null ? new double[nobs] : pool.borrowLogLikelihoodAdjustment(nobs);
        int logLikelihoodAdjustmentBase = 0;
        double[] design = pool == null ? new double[kEndog * kStates] : scratch;
        int designBase = pool == null ? 0 : layout.designBase();
        double[] obsCov = pool == null ? new double[kEndog * kEndog] : scratch;
        int obsCovBase = pool == null ? 0 : layout.obsCovBase();
        double[] obsCovFactor = pool == null ? new double[kEndog * kEndog] : scratch;
        int obsCovFactorBase = pool == null ? 0 : layout.obsCovFactorBase();
        double[] hInvZ = pool == null ? new double[kEndog * kStates] : scratch;
        int hInvZBase = pool == null ? 0 : layout.hInvZBase();
        double[] information = pool == null ? new double[kStates * kStates] : scratch;
        int informationBase = pool == null ? 0 : layout.informationBase();
        double[] collapseFactor = pool == null ? new double[kStates * kStates] : scratch;
        int collapseFactorBase = pool == null ? 0 : layout.collapseFactorBase();
        double[] collapsedDesign = pool == null ? new double[kStates * kStates] : scratch;
        int collapsedDesignBase = pool == null ? 0 : layout.collapsedDesignBase();
        double[] collapsedObsCov = pool == null ? new double[kStates * kStates] : scratch;
        int collapsedObsCovBase = pool == null ? 0 : layout.collapsedObsCovBase();
        double[] collapsedEndog = pool == null ? new double[kStates] : scratch;
        int collapsedEndogBase = pool == null ? 0 : layout.collapsedEndogBase();
        double[] zHy = pool == null ? new double[kStates] : scratch;
        int zHyBase = pool == null ? 0 : layout.zHyBase();
        double[] reconstructed = pool == null ? new double[kStates] : scratch;
        int reconstructedBase = pool == null ? 0 : layout.reconstructedBase();
        double[] residual = pool == null ? new double[kEndog] : scratch;
        int residualBase = pool == null ? 0 : layout.residualBase();
        double[] solvedResidual = pool == null ? new double[kEndog] : scratch;
        int solvedResidualBase = pool == null ? 0 : layout.solvedResidualBase();
        double[] observedEndog = pool == null ? new double[kEndog] : scratch;
        int observedEndogBase = pool == null ? 0 : layout.observedEndogBase();
        double[] transition = pool == null ? new double[kStates * kStates] : null;
        double[] stateIntercept = pool == null ? new double[kStates] : null;
        double[] selection = pool == null ? new double[kStates * kPosdef] : null;
        double[] stateCov = pool == null ? new double[kPosdef * kPosdef] : null;
        int[] selectedIndex = pool == null ? new int[kEndog] : pool.intScratchBacking;
        int selectedIndexBase = pool == null ? 0 : pool.intScratchLayout.selectedIndexBase();

        setIdentity(collapsedObsCov, collapsedObsCovBase, kStates);
        for (int t = 0; t < nobs; t++) {
            int ke = selectObserved(source, t, selectedIndex, selectedIndexBase);
            copyObservedDesign(source, t, selectedIndex, selectedIndexBase, ke, design, designBase);
            copyObservedObsCov(source, t, selectedIndex, selectedIndexBase, ke, obsCov, obsCovBase);
            copyObservedEndog(source, t, selectedIndex, selectedIndexBase, ke, observedEndog, observedEndogBase, endogOverride, endogBase);
            System.arraycopy(obsCov, obsCovBase, obsCovFactor, obsCovFactorBase, ke * ke);
            if (BLAS.dpotrf(BLAS.Uplo.Lower, ke, obsCovFactor, obsCovFactorBase, ke) != 0) {
                throw new KalmanFilter.SingularForecastException(
                    "Observation covariance is not positive definite for COLLAPSED filtering at t=" + t);
            }
            double logDetH = 0.0;
            for (int row = 0; row < ke; row++) {
                double diagonal = obsCovFactor[obsCovFactorBase + row * ke + row];
                if (!(diagonal * diagonal > options.diffuseTolerance())) {
                    throw new KalmanFilter.SingularForecastException(
                        "Observation covariance is singular for COLLAPSED filtering at t=" + t);
                }
                logDetH += Math.log(diagonal);
            }
            logDetH *= 2.0;

            System.arraycopy(design, designBase, hInvZ, hInvZBase, ke * kStates);
            BLAS.dpotrs(BLAS.Uplo.Lower, ke, kStates,
                obsCovFactor, obsCovFactorBase, ke,
                hInvZ, hInvZBase, kStates);
            multiplyTransposedLeft(design, designBase, hInvZ, hInvZBase, ke, kStates, information, informationBase);
            System.arraycopy(information, informationBase, collapseFactor, collapseFactorBase, kStates * kStates);
            if (BLAS.dpotrf(BLAS.Uplo.Upper, kStates, collapseFactor, collapseFactorBase, kStates) != 0
                    || !BLAS.dpotri(BLAS.Uplo.Upper, kStates, collapseFactor, collapseFactorBase, kStates)
                    || BLAS.dpotrf(BLAS.Uplo.Upper, kStates, collapseFactor, collapseFactorBase, kStates) != 0) {
                throw new KalmanFilter.SingularForecastException(
                    "Collapsed design information is not positive definite at t=" + t);
            }

            setIdentity(collapsedDesign, collapsedDesignBase, kStates);
            if (!BLAS.dtrtrs(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                    kStates, kStates,
                    collapseFactor, collapseFactorBase, kStates,
                    collapsedDesign, collapsedDesignBase, kStates)) {
                throw new KalmanFilter.SingularForecastException(
                    "Collapsed design transformation is singular at t=" + t);
            }

            Arrays.fill(zHy, zHyBase, zHyBase + kStates, 0.0);
            for (int row = 0; row < ke; row++) {
                double value = observedEndog[observedEndogBase + row];
                for (int col = 0; col < kStates; col++) {
                    zHy[zHyBase + col] += hInvZ[hInvZBase + row * kStates + col] * value;
                }
            }
            multiplyUpper(collapseFactor, collapseFactorBase, zHy, zHyBase, collapsedEndog, collapsedEndogBase, kStates);

            multiplyUpperTransposed(collapseFactor, collapseFactorBase, collapsedEndog, collapsedEndogBase, reconstructed, reconstructedBase, kStates);
            for (int row = 0; row < ke; row++) {
                double value = observedEndog[observedEndogBase + row];
                for (int col = 0; col < kStates; col++) {
                    value -= design[designBase + row * kStates + col] * reconstructed[reconstructedBase + col];
                }
                residual[residualBase + row] = value;
                solvedResidual[solvedResidualBase + row] = value;
            }
            BLAS.dpotrs(BLAS.Uplo.Lower, ke, 1,
                obsCovFactor, obsCovFactorBase, ke,
                solvedResidual, solvedResidualBase, 1);
            double residualQuadratic = BLAS.ddot(ke, residual, residualBase, 1, solvedResidual, solvedResidualBase, 1);
            logLikelihoodAdjustment[logLikelihoodAdjustmentBase + t] = -0.5 * (residualQuadratic + (ke - kStates) * LOG_TWO_PI + logDetH);

            copySurface(collapsedDesign, collapsedDesignBase, collapsed.designData(), collapsed.designOffset(t), kStates * kStates);
            copySurface(collapsedObsCov, collapsedObsCovBase, collapsed.obsCovData(), collapsed.obsCovOffset(t), kStates * kStates);
            copySurface(collapsedEndog, collapsedEndogBase, collapsed.endogData(), collapsed.endogOffset(t), kStates);
            if (pool == null) {
                copyStateSystem(source, collapsed, t, transition, stateIntercept, selection, stateCov);
            }
        }

        return new CollapsedData(collapsed, logLikelihoodAdjustment, logLikelihoodAdjustmentBase);
    }

    private static FilterResult replayExactDiffuseObservationSurfaces(KalmanSSM model,
                                                                      InitialState init,
                                                                      double[] endogOverride,
                                                                      int endogBase,
                                                                      FilterOptions options,
                                                                      Pool pool) {
        FilterOptions replayOptions = replayOptions(options);
        KalmanFilter.Pool replayPool = pool == null ? null : pool.ensureReplayPool();
        if (endogOverride == null) {
            return KalmanFilter.filter(model, init, replayPool, replayOptions);
        }
        return KalmanFilter.filter(model, init, endogOverride, endogBase, replayPool, replayOptions);
    }

    private static FilterOptions replayOptions(FilterOptions options) {
        FilterOptions.Builder builder = options.toBuilder()
            .method(FilterMethod.CONVENTIONAL)
            .retainNone();
        retainIfRequested(builder, options, FilterOptions.Surface.FORECAST_MEAN);
        retainIfRequested(builder, options, FilterOptions.Surface.FORECAST_ERROR);
        retainIfRequested(builder, options, FilterOptions.Surface.FORECAST_COVARIANCE);
        retainIfRequested(builder, options, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR);
        retainIfRequested(builder, options, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);
        retainIfRequested(builder, options, FilterOptions.Surface.KALMAN_GAIN);
        return builder.build();
    }

    private static void retainIfRequested(FilterOptions.Builder builder,
                                          FilterOptions options,
                                          FilterOptions.Surface surface) {
        if (options.includes(surface)) {
            builder.retain(surface);
        }
    }

    private static void copyStateSurfaces(FilterResult source, FilterResult target) {
        copySurface(source.predictedState, source.predictedStateBase(), target.predictedState,
            target.predictedStateBase(), target.predictedStateLength());
        copySurface(source.predictedStateCov, source.predictedStateCovBase(), target.predictedStateCov,
            target.predictedStateCovBase(), target.predictedStateCovLength());
        copySurface(source.filteredState, source.filteredStateBase(), target.filteredState,
            target.filteredStateBase(), target.filteredStateLength());
        copySurface(source.filteredStateCov, source.filteredStateCovBase(), target.filteredStateCov,
            target.filteredStateCovBase(), target.filteredStateCovLength());
        copySurface(source.predictedDiffuseStateCov, source.predictedDiffuseStateCovBase(), target.predictedDiffuseStateCov,
            target.predictedDiffuseStateCovBase(), target.predictedDiffuseStateCovLength());
    }

    private static void copyObservationSurfaces(FilterResult source, FilterResult target) {
        copySurface(source.forecast, source.forecastBase(), target.forecast,
            target.forecastBase(), target.forecastLength());
        copySurface(source.forecastError, source.forecastErrorBase(), target.forecastError,
            target.forecastErrorBase(), target.forecastErrorLength());
        copySurface(source.forecastErrorCov, source.forecastErrorCovBase(), target.forecastErrorCov,
            target.forecastErrorCovBase(), target.forecastErrorCovLength());
        copySurface(source.standardizedForecastError, source.standardizedForecastErrorBase(), target.standardizedForecastError,
            target.standardizedForecastErrorBase(), target.standardizedForecastErrorLength());
        copySurface(source.forecastErrorDiffuseCov, source.forecastErrorDiffuseCovBase(), target.forecastErrorDiffuseCov,
            target.forecastErrorDiffuseCovBase(), target.forecastErrorDiffuseCovLength());
        copySurface(source.kalmanGain, source.kalmanGainBase(), target.kalmanGain,
            target.kalmanGainBase(), target.kalmanGainLength());
    }

    private static void rebuildLikelihood(FilterResult collapsed,
                                          FilterResult result,
                                          double[] logLikelihoodAdjustment,
                                          int logLikelihoodAdjustmentBase) {
        if (result.logLikelihoodObsLength() == 0) {
            return;
        }
        for (int t = 0; t < result.nobs; t++) {
            result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] =
                collapsed.logLikelihoodObs[collapsed.logLikelihoodObsOffset(t)]
                    + logLikelihoodAdjustment[logLikelihoodAdjustmentBase + t];
        }
    }

    private static void rebuildObservationSurfaces(KalmanSSM model,
                                                   FilterResult collapsed,
                                                   FilterResult result,
                                                   FilterOptions options,
                                                   double[] logLikelihoodAdjustment,
                                                   int logLikelihoodAdjustmentBase,
                                                   double[] endogOverride,
                                                   int endogBase,
                                                   boolean copyCollapsedKalmanGain,
                                                   KalmanSSM collapsedModel,
                                                   Pool pool) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        int nobs = model.observationCount();
        boolean storeForecastMean = result.forecastLength() > 0;
        boolean storeForecastError = result.forecastErrorLength() > 0;
        boolean storeForecastCovariance = result.forecastErrorCovLength() > 0;
        boolean storeStandardizedForecastError = result.standardizedForecastErrorLength() > 0;
        boolean storeForecastDiffuseCovariance = result.forecastErrorDiffuseCovLength() > 0;
        boolean storeGain = result.kalmanGainLength() > 0;
        boolean storeLikelihood = result.logLikelihoodObsLength() > 0;
        boolean needForecast = storeForecastMean || storeForecastError || storeStandardizedForecastError || storeGain;
        boolean needForecastCovariance = storeForecastCovariance || storeStandardizedForecastError || storeGain;
        if (!needForecast && !needForecastCovariance && !storeForecastDiffuseCovariance && !storeGain) {
            rebuildLikelihood(collapsed, result, logLikelihoodAdjustment, logLikelihoodAdjustmentBase);
            return;
        }

        CollapsedScratchLayout layout = pool == null ? null : pool.scratchLayout;
        double[] scratch = pool == null ? null : pool.scratchBacking;
        double[] forecastMean = pool == null ? new double[kEndog] : scratch;
        int forecastMeanBase = pool == null ? 0 : layout.forecastMeanBase();
        double[] forecastError = pool == null ? new double[kEndog] : scratch;
        int forecastErrorBase = pool == null ? 0 : layout.forecastErrorBase();
        double[] forecastErrorCov = pool == null ? new double[kEndog * kEndog] : scratch;
        int forecastErrorCovBase = pool == null ? 0 : layout.forecastErrorCovBase();
        double[] forecastErrorDiffuseCov = storeForecastDiffuseCovariance
            ? (pool == null ? new double[kEndog * kEndog] : scratch)
            : null;
        int forecastErrorDiffuseCovBase = pool == null ? 0 : layout.forecastErrorDiffuseCovBase();
        double[] factor = pool == null ? new double[kEndog * kEndog] : scratch;
        int factorBase = pool == null ? 0 : layout.factorBase();
        double[] zP = pool == null ? new double[kEndog * kStates] : scratch;
        int zPBase = pool == null ? 0 : layout.zPBase();
        double[] solvedError = pool == null ? new double[kEndog] : scratch;
        int solvedErrorBase = pool == null ? 0 : layout.solvedErrorBase();
        double[] covarianceWork = pool == null ? new double[kEndog * kEndog] : scratch;
        int covarianceWorkBase = pool == null ? 0 : layout.covarianceWorkBase();
        double[] selectedError = pool == null ? new double[kEndog] : scratch;
        int selectedErrorBase = pool == null ? 0 : layout.selectedErrorBase();
        double[] selectedStandardized = pool == null ? new double[kEndog] : scratch;
        int selectedStandardizedBase = pool == null ? 0 : layout.selectedStandardizedBase();
        double[] selectedCovariance = pool == null ? new double[kEndog * kEndog] : scratch;
        int selectedCovarianceBase = pool == null ? 0 : layout.selectedCovarianceBase();
        int[] selectedIndex = pool == null ? new int[kEndog] : pool.intScratchBacking;
        int selectedIndexBase = pool == null ? 0 : pool.intScratchLayout.selectedIndexBase();
        int[] pivots = needsLu(options.forecastErrorSequenceRef())
            ? (pool == null ? new int[kEndog] : pool.intScratchBacking)
            : null;
        int pivotsBase = pool == null ? 0 : pool.intScratchLayout.pivotsBase();
        ForecastErrorSolver.Result solverResult = new ForecastErrorSolver.Result();

        for (int t = 0; t < nobs; t++) {
            int predOff = collapsed.predictedStateOffset(t);
            int predCovOff = collapsed.predictedStateCovOffset(t);
            if (needForecast) {
                computeForecast(model, collapsed, t, predOff, endogOverride, endogBase,
                    forecastMean, forecastMeanBase, forecastError, forecastErrorBase);
            }
            if (needForecastCovariance) {
                computeForecastCovariance(model, collapsed, t, predCovOff,
                    forecastErrorCov, forecastErrorCovBase);
            }
            if (storeForecastDiffuseCovariance) {
                computeForecastDiffuseCovariance(model, collapsed, t,
                    forecastErrorDiffuseCov, forecastErrorDiffuseCovBase);
            }
            if (storeForecastMean) {
                System.arraycopy(forecastMean, forecastMeanBase, result.forecast, result.forecastOffset(t), kEndog);
            }
            if (storeForecastError) {
                System.arraycopy(forecastError, forecastErrorBase, result.forecastError, result.forecastErrorOffset(t), kEndog);
            }
            if (storeForecastCovariance) {
                System.arraycopy(forecastErrorCov, forecastErrorCovBase, result.forecastErrorCov, result.forecastErrorCovOffset(t), kEndog * kEndog);
            }
            if (storeForecastDiffuseCovariance) {
                System.arraycopy(forecastErrorDiffuseCov, forecastErrorDiffuseCovBase, result.forecastErrorDiffuseCov,
                    result.forecastErrorDiffuseCovOffset(t), kEndog * kEndog);
            }
            if (storeStandardizedForecastError) {
                storeStandardizedForecastError(model, t,
                    forecastError, forecastErrorBase,
                    forecastErrorCov, forecastErrorCovBase,
                    factor, factorBase,
                    selectedError, selectedErrorBase,
                    selectedStandardized, selectedStandardizedBase,
                    selectedCovariance, selectedCovarianceBase,
                    selectedIndex, selectedIndexBase,
                    result.standardizedForecastError, result.standardizedForecastErrorOffset(t));
            }
            if (storeGain) {
                if (copyCollapsedKalmanGain) {
                    copyCollapsedKalmanGain(collapsedModel, collapsed, options, t, predCovOff,
                        forecastErrorCov, forecastErrorCovBase,
                        covarianceWork, covarianceWorkBase,
                        zP, zPBase,
                        solvedError, solvedErrorBase,
                        pivots, pivotsBase, solverResult,
                        result.kalmanGain, result.kalmanGainOffset(t), kEndog, kStates);
                } else {
                    computeFullKalmanGain(model, collapsed, options, t, predCovOff,
                        forecastError, forecastErrorBase,
                        forecastErrorCov, forecastErrorCovBase,
                        covarianceWork, covarianceWorkBase,
                        selectedCovariance, selectedCovarianceBase,
                        zP, zPBase,
                        solvedError, solvedErrorBase,
                        selectedError, selectedErrorBase,
                        selectedIndex, selectedIndexBase,
                        pivots, pivotsBase, solverResult,
                        result.kalmanGain, result.kalmanGainOffset(t));
                }
            }
            if (storeLikelihood) {
                result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] =
                    collapsed.logLikelihoodObs[collapsed.logLikelihoodObsOffset(t)]
                        + logLikelihoodAdjustment[logLikelihoodAdjustmentBase + t];
            }
        }
    }

    private static void copyCollapsedKalmanGain(KalmanSSM collapsedModel,
                                                FilterResult collapsed,
                                                FilterOptions options,
                                                int t,
                                                int predCovOff,
                                                double[] forecastErrorCov,
                                                int forecastErrorCovBase,
                                                double[] covarianceWork,
                                                int covarianceWorkBase,
                                                double[] zP,
                                                int zPBase,
                                                double[] solvedError,
                                                int solvedErrorBase,
                                                int[] pivots,
                                                int pivotsBase,
                                                ForecastErrorSolver.Result solverResult,
                                                double[] kalmanGain,
                                                int kalmanGainOffset,
                                                int kEndog,
                                                int kStates) {
        Arrays.fill(kalmanGain, kalmanGainOffset, kalmanGainOffset + kStates * kEndog, 0.0);
        if (collapsed.kalmanGainLength() == 0) {
            return;
        }
        if (t >= collapsed.nobsDiffuse) {
            computeCollapsedConventionalKalmanGain(collapsedModel, collapsed, options, t, predCovOff,
                forecastErrorCov, forecastErrorCovBase,
                covarianceWork, covarianceWorkBase,
                zP, zPBase,
                solvedError, solvedErrorBase,
                pivots, pivotsBase, solverResult,
                kalmanGain, kalmanGainOffset, kEndog, kStates);
            return;
        }
        int columns = Math.min(collapsed.kEndog, kEndog);
        int collapsedOffset = collapsed.kalmanGainOffset(t);
        for (int row = 0; row < kStates; row++) {
            System.arraycopy(collapsed.kalmanGain, collapsedOffset + row * collapsed.kEndog,
                kalmanGain, kalmanGainOffset + row * kEndog,
                columns);
        }
    }

    private static void computeCollapsedConventionalKalmanGain(KalmanSSM collapsedModel,
                                                               FilterResult collapsed,
                                                               FilterOptions options,
                                                               int t,
                                                               int predCovOff,
                                                               double[] forecastErrorCov,
                                                               int forecastErrorCovBase,
                                                               double[] covarianceWork,
                                                               int covarianceWorkBase,
                                                               double[] zP,
                                                               int zPBase,
                                                               double[] solvedError,
                                                               int solvedErrorBase,
                                                               int[] pivots,
                                                               int pivotsBase,
                                                               ForecastErrorSolver.Result solverResult,
                                                               double[] kalmanGain,
                                                               int kalmanGainOffset,
                                                               int kEndog,
                                                               int kStates) {
        int collapsedEndog = collapsedModel.observationDimension();
        double[] design = collapsedModel.designData();
        int designOffset = collapsedModel.designOffset(t);
        int designLd = collapsedModel.designLeadingDimension();
        double[] obsCov = collapsedModel.obsCovData();
        int obsCovOffset = collapsedModel.obsCovOffset(t);
        int obsCovLd = collapsedModel.obsCovLeadingDimension();
        for (int row = 0; row < collapsedEndog; row++) {
            int rowOffset = designOffset + row * designLd;
            for (int col = 0; col < kStates; col++) {
                double value = 0.0;
                for (int inner = 0; inner < kStates; inner++) {
                    value += design[rowOffset + inner] * collapsed.predictedStateCov[predCovOff + inner * kStates + col];
                }
                zP[zPBase + row * kStates + col] = value;
            }
        }
        for (int row = 0; row < collapsedEndog; row++) {
            int rowOffset = designOffset + row * designLd;
            int obsCovRowOffset = obsCovOffset + row * obsCovLd;
            for (int col = 0; col < collapsedEndog; col++) {
                double value = obsCov[obsCovRowOffset + col];
                for (int inner = 0; inner < kStates; inner++) {
                    value += design[rowOffset + inner] * zP[zPBase + col * kStates + inner];
                }
                forecastErrorCov[forecastErrorCovBase + row * collapsedEndog + col] = value;
            }
        }
        Arrays.fill(solvedError, solvedErrorBase, solvedErrorBase + collapsedEndog, 0.0);
        ForecastErrorSolver.solveMultivariate(options.forecastErrorSequenceRef(),
            covarianceWork, covarianceWorkBase,
            collapsedEndog,
            forecastErrorCov, forecastErrorCovBase,
            zP, zPBase,
            kStates,
            kStates,
            solvedError, solvedErrorBase,
            solvedError, solvedErrorBase,
            pivots,
            pivotsBase,
            options.diffuseTolerance(),
            solverResult);
        double[] transition = collapsedModel.transitionData();
        int transitionOffset = collapsedModel.transitionOffset(t);
        int transitionLd = collapsedModel.transitionLeadingDimension();
        int columns = Math.min(collapsedEndog, kEndog);
        for (int row = 0; row < kStates; row++) {
            int transitionRowOffset = transitionOffset + row * transitionLd;
            for (int obs = 0; obs < columns; obs++) {
                double value = 0.0;
                for (int col = 0; col < kStates; col++) {
                    value += transition[transitionRowOffset + col] * zP[zPBase + obs * kStates + col];
                }
                kalmanGain[kalmanGainOffset + row * kEndog + obs] = value;
            }
        }
    }

    private static void computeForecastDiffuseCovariance(KalmanSSM model,
                                                         FilterResult collapsed,
                                                         int t,
                                                         double[] out,
                                                         int outBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        Arrays.fill(out, outBase, outBase + kEndog * kEndog, 0.0);
        if (collapsed.predictedDiffuseStateCovLength() == 0) {
            return;
        }
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        int predDiffCovOff = collapsed.predictedDiffuseStateCovOffset(t);
        for (int row = 0; row < kEndog; row++) {
            int rowOffset = designOffset + row * designLd;
            for (int col = 0; col < kEndog; col++) {
                int colOffset = designOffset + col * designLd;
                double value = 0.0;
                for (int left = 0; left < kStates; left++) {
                    double zLeft = design[rowOffset + left];
                    for (int right = 0; right < kStates; right++) {
                        value += zLeft * collapsed.predictedDiffuseStateCov[predDiffCovOff + left * kStates + right]
                            * design[colOffset + right];
                    }
                }
                out[outBase + row * kEndog + col] = value;
            }
        }
    }

    private static void computeForecast(KalmanSSM model,
                                        FilterResult collapsed,
                                        int t,
                                        int predOff,
                                        double[] endogOverride,
                                        int endogBase,
                                        double[] forecastMean,
                                        int forecastMeanBase,
                                        double[] forecastError) {
        computeForecast(model, collapsed, t, predOff, endogOverride, endogBase,
            forecastMean, forecastMeanBase, forecastError, 0);
    }

    private static void computeForecast(KalmanSSM model,
                                        FilterResult collapsed,
                                        int t,
                                        int predOff,
                                        double[] endogOverride,
                                        int endogBase,
                                        double[] forecastMean,
                                        int forecastMeanBase,
                                        double[] forecastError,
                                        int forecastErrorBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        double[] obsIntercept = model.obsInterceptData();
        int interceptOffset = model.obsInterceptOffset(t);
        double[] endog = endogOverride == null ? model.endogData() : endogOverride;
        int endogOffset = endogOverride == null ? model.endogOffset(t) : endogBase + t * kEndog;
        for (int row = 0; row < kEndog; row++) {
            double mean = obsIntercept[interceptOffset + row];
            int rowOffset = designOffset + row * designLd;
            for (int col = 0; col < kStates; col++) {
                mean += design[rowOffset + col] * collapsed.predictedState[predOff + col];
            }
            forecastMean[forecastMeanBase + row] = mean;
            forecastError[forecastErrorBase + row] = model.isMissing(row, t) ? 0.0 : endog[endogOffset + row] - mean;
        }
    }

    private static void storeStandardizedForecastError(KalmanSSM model,
                                                       int t,
                                                       double[] forecastError,
                                                       int forecastErrorBase,
                                                       double[] forecastErrorCov,
                                                       int forecastErrorCovBase,
                                                       double[] factor,
                                                       int factorBase,
                                                       double[] selectedError,
                                                       int selectedErrorBase,
                                                       double[] selectedStandardized,
                                                       int selectedStandardizedBase,
                                                       double[] selectedCovariance,
                                                       int selectedCovarianceBase,
                                                       int[] selectedIndex,
                                                       int selectedIndexBase,
                                                       double[] target,
                                                       int targetOffset) {
        int kEndog = model.observationDimension();
        int observed = selectObserved(model, t, selectedIndex, selectedIndexBase);
        if (observed == kEndog) {
            if (kEndog == 1) {
                target[targetOffset] = ForecastErrorSolver.standardizeScalar(
                    forecastError[forecastErrorBase], forecastErrorCov[forecastErrorCovBase], 0.0);
            } else if (!ForecastErrorSolver.standardizeCholesky(forecastErrorCov, forecastErrorCovBase, kEndog,
                    forecastError, forecastErrorBase, target, targetOffset, kEndog, factor, factorBase, 0.0)) {
                Arrays.fill(target, targetOffset, targetOffset + kEndog, Double.NaN);
            }
            return;
        }

        Arrays.fill(target, targetOffset, targetOffset + kEndog, 0.0);
        copyObservedVector(forecastError, forecastErrorBase, selectedIndex, selectedIndexBase, observed,
            selectedError, selectedErrorBase);
        copyObservedSquare(forecastErrorCov, forecastErrorCovBase, kEndog, selectedIndex, selectedIndexBase, observed,
            selectedCovariance, selectedCovarianceBase);
        if (observed == 1) {
            target[targetOffset + selectedIndex[selectedIndexBase]] = ForecastErrorSolver.standardizeScalar(
                selectedError[selectedErrorBase], selectedCovariance[selectedCovarianceBase], 0.0);
        } else if (ForecastErrorSolver.standardizeCholesky(selectedCovariance, selectedCovarianceBase, observed,
                selectedError, selectedErrorBase, selectedStandardized, selectedStandardizedBase,
                observed, factor, factorBase, 0.0)) {
            for (int obs = 0; obs < observed; obs++) {
                target[targetOffset + selectedIndex[selectedIndexBase + obs]] = selectedStandardized[selectedStandardizedBase + obs];
            }
        } else {
            for (int obs = 0; obs < observed; obs++) {
                target[targetOffset + selectedIndex[selectedIndexBase + obs]] = Double.NaN;
            }
        }
    }

    private static void computeForecastCovariance(KalmanSSM model,
                                                  FilterResult collapsed,
                                                  int t,
                                                  int predCovOff,
                                                  double[] forecastErrorCov,
                                                  int forecastErrorCovBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        copySquare(model.obsCovData(), model.obsCovOffset(t), model.obsCovLeadingDimension(),
            kEndog, forecastErrorCov, forecastErrorCovBase, kEndog);
        for (int row = 0; row < kEndog; row++) {
            int rowOffset = designOffset + row * designLd;
            for (int col = 0; col < kEndog; col++) {
                int colOffset = designOffset + col * designLd;
                double value = 0.0;
                for (int left = 0; left < kStates; left++) {
                    double zLeft = design[rowOffset + left];
                    for (int right = 0; right < kStates; right++) {
                        value += zLeft * collapsed.predictedStateCov[predCovOff + left * kStates + right]
                            * design[colOffset + right];
                    }
                }
                forecastErrorCov[forecastErrorCovBase + row * kEndog + col] += value;
            }
        }
    }

    private static void computeFullKalmanGain(KalmanSSM model,
                                              FilterResult collapsed,
                                              FilterOptions options,
                                              int t,
                                              int predCovOff,
                                              double[] forecastError,
                                              int forecastErrorBase,
                                              double[] forecastErrorCov,
                                              int forecastErrorCovBase,
                                              double[] covarianceWork,
                                              int covarianceWorkBase,
                                              double[] selectedCovariance,
                                              int selectedCovarianceBase,
                                              double[] zP,
                                              int zPBase,
                                              double[] solvedError,
                                              int solvedErrorBase,
                                              double[] selectedError,
                                              int selectedErrorBase,
                                              int[] selectedIndex,
                                              int selectedIndexBase,
                                              int[] pivots,
                                              int pivotsBase,
                                              ForecastErrorSolver.Result solverResult,
                                              double[] kalmanGain,
                                              int kalmanGainOffset) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        int observed = selectObserved(model, t, selectedIndex, selectedIndexBase);
        for (int row = 0; row < observed; row++) {
            int rowOffset = designOffset + selectedIndex[selectedIndexBase + row] * designLd;
            for (int col = 0; col < kStates; col++) {
                double value = 0.0;
                for (int inner = 0; inner < kStates; inner++) {
                    value += design[rowOffset + inner] * collapsed.predictedStateCov[predCovOff + inner * kStates + col];
                }
                zP[zPBase + row * kStates + col] = value;
            }
        }
        double[] originalError = forecastError;
        int originalErrorBase = forecastErrorBase;
        double[] originalCovariance = forecastErrorCov;
        int originalCovarianceBase = forecastErrorCovBase;
        if (observed < kEndog) {
            copyObservedVector(forecastError, forecastErrorBase, selectedIndex, selectedIndexBase, observed,
                selectedError, selectedErrorBase);
            copyObservedSquare(forecastErrorCov, forecastErrorCovBase, kEndog, selectedIndex, selectedIndexBase, observed,
                selectedCovariance, selectedCovarianceBase);
            originalError = selectedError;
            originalErrorBase = selectedErrorBase;
            originalCovariance = selectedCovariance;
            originalCovarianceBase = selectedCovarianceBase;
        }
        System.arraycopy(originalError, originalErrorBase, solvedError, solvedErrorBase, observed);
        ForecastErrorSolver.solveMultivariate(options.forecastErrorSequenceRef(),
            covarianceWork, covarianceWorkBase,
            observed,
            originalCovariance, originalCovarianceBase,
            zP, zPBase,
            kStates,
            kStates,
            solvedError, solvedErrorBase,
            originalError, originalErrorBase,
            pivots,
            pivotsBase,
            options.diffuseTolerance(),
            solverResult);

        Arrays.fill(kalmanGain, kalmanGainOffset, kalmanGainOffset + kStates * kEndog, 0.0);
        double[] transition = model.transitionData();
        int transitionOffset = model.transitionOffset(t);
        int transitionLd = model.transitionLeadingDimension();
        for (int row = 0; row < kStates; row++) {
            int transitionRowOffset = transitionOffset + row * transitionLd;
            for (int obs = 0; obs < observed; obs++) {
                double value = 0.0;
                for (int col = 0; col < kStates; col++) {
                    value += transition[transitionRowOffset + col] * zP[zPBase + obs * kStates + col];
                }
                kalmanGain[kalmanGainOffset + row * kEndog + selectedIndex[selectedIndexBase + obs]] = value;
            }
        }
    }

    private static int observedDimension(KalmanSSM model, int t) {
        return model.observationDimension() - model.missingCount(t);
    }

    private static int selectObserved(KalmanSSM model, int t, int[] selectedIndex) {
        return selectObserved(model, t, selectedIndex, 0);
    }

    private static int selectObserved(KalmanSSM model, int t, int[] selectedIndex, int selectedIndexBase) {
        int observed = 0;
        for (int obs = 0; obs < model.observationDimension(); obs++) {
            if (!model.isMissing(obs, t)) {
                selectedIndex[selectedIndexBase + observed++] = obs;
            }
        }
        return observed;
    }

    private static void copyObservedDesign(KalmanSSM source,
                                           int t,
                                           int[] selectedIndex,
                                           int observed,
                                           double[] target) {
        copyObservedDesign(source, t, selectedIndex, 0, observed, target, 0);
    }

    private static void copyObservedDesign(KalmanSSM source,
                                           int t,
                                           int[] selectedIndex,
                                           int selectedIndexBase,
                                           int observed,
                                           double[] target,
                                           int targetBase) {
        double[] design = source.designData();
        int sourceOffset = source.designOffset(t);
        int sourceLd = source.designLeadingDimension();
        int kStates = source.stateCount();
        for (int row = 0; row < observed; row++) {
            System.arraycopy(design, sourceOffset + selectedIndex[selectedIndexBase + row] * sourceLd,
                target, targetBase + row * kStates, kStates);
        }
    }

    private static void copyObservedObsCov(KalmanSSM source,
                                           int t,
                                           int[] selectedIndex,
                                           int observed,
                                           double[] target) {
        copyObservedObsCov(source, t, selectedIndex, 0, observed, target, 0);
    }

    private static void copyObservedObsCov(KalmanSSM source,
                                           int t,
                                           int[] selectedIndex,
                                           int selectedIndexBase,
                                           int observed,
                                           double[] target,
                                           int targetBase) {
        double[] obsCov = source.obsCovData();
        int sourceOffset = source.obsCovOffset(t);
        int sourceLd = source.obsCovLeadingDimension();
        for (int row = 0; row < observed; row++) {
            int sourceRow = selectedIndex[selectedIndexBase + row];
            for (int col = 0; col < observed; col++) {
                target[targetBase + row * observed + col] = obsCov[
                    sourceOffset + sourceRow * sourceLd + selectedIndex[selectedIndexBase + col]];
            }
        }
    }

    private static void copyObservedEndog(KalmanSSM source,
                                          int t,
                                          int[] selectedIndex,
                                          int observed,
                                          double[] target,
                                          double[] endogOverride,
                                          int endogBase) {
        copyObservedEndog(source, t, selectedIndex, 0, observed, target, 0, endogOverride, endogBase);
    }

    private static void copyObservedEndog(KalmanSSM source,
                                          int t,
                                          int[] selectedIndex,
                                          int selectedIndexBase,
                                          int observed,
                                          double[] target,
                                          int targetBase,
                                          double[] endogOverride,
                                          int endogBase) {
        double[] endog = endogOverride == null ? source.endogData() : endogOverride;
        double[] obsIntercept = source.obsInterceptData();
        int sourceOffset = endogOverride == null ? source.endogOffset(t) : endogBase + t * source.observationDimension();
        int interceptOffset = source.obsInterceptOffset(t);
        for (int row = 0; row < observed; row++) {
            int sourceIndex = selectedIndex[selectedIndexBase + row];
            target[targetBase + row] = endog[sourceOffset + sourceIndex] - obsIntercept[interceptOffset + sourceIndex];
        }
    }

    private static void copyObservedVector(double[] source,
                                           int[] selectedIndex,
                                           int observed,
                                           double[] target) {
        copyObservedVector(source, 0, selectedIndex, 0, observed, target, 0);
    }

    private static void copyObservedVector(double[] source,
                                           int sourceBase,
                                           int[] selectedIndex,
                                           int selectedIndexBase,
                                           int observed,
                                           double[] target,
                                           int targetBase) {
        for (int row = 0; row < observed; row++) {
            target[targetBase + row] = source[sourceBase + selectedIndex[selectedIndexBase + row]];
        }
    }

    private static void copyObservedSquare(double[] source,
                                           int sourceLd,
                                           int[] selectedIndex,
                                           int observed,
                                           double[] target) {
        copyObservedSquare(source, 0, sourceLd, selectedIndex, 0, observed, target, 0);
    }

    private static void copyObservedSquare(double[] source,
                                           int sourceBase,
                                           int sourceLd,
                                           int[] selectedIndex,
                                           int selectedIndexBase,
                                           int observed,
                                           double[] target,
                                           int targetBase) {
        for (int row = 0; row < observed; row++) {
            int sourceRow = selectedIndex[selectedIndexBase + row];
            for (int col = 0; col < observed; col++) {
                target[targetBase + row * observed + col] = source[
                    sourceBase + sourceRow * sourceLd + selectedIndex[selectedIndexBase + col]];
            }
        }
    }

    private static void copyStateSystem(KalmanSSM source,
                                        KalmanSSM target,
                                        int t,
                                        double[] transition,
                                        double[] stateIntercept,
                                        double[] selection,
                                        double[] stateCov) {
        int kStates = source.stateCount();
        int kPosdef = source.stateDisturbanceCount();
        copySquare(source.transitionData(), source.transitionOffset(t), source.transitionLeadingDimension(),
            kStates, transition, 0, kStates);
        System.arraycopy(source.stateInterceptData(), source.stateInterceptOffset(t), stateIntercept, 0, kStates);
        copyRectangular(source.selectionData(), source.selectionOffset(t), source.selectionLeadingDimension(),
            kStates, kPosdef, selection, 0, kPosdef);
        copySquare(source.stateCovarianceData(), source.stateCovarianceOffset(t), source.stateCovarianceLeadingDimension(),
            kPosdef, stateCov, 0, kPosdef);
        target.setTransition(transition, t);
        target.setStateIntercept(stateIntercept, t);
        target.setSelection(selection, t);
        target.setStateCov(stateCov, t);
    }

    private static boolean isTransitionTimeVarying(KalmanSSM source) {
        return source.observationCount() > 1 && source.transitionOffset(1) != source.transitionOffset(0);
    }

    private static boolean isStateInterceptTimeVarying(KalmanSSM source) {
        return source.observationCount() > 1 && source.stateInterceptOffset(1) != source.stateInterceptOffset(0);
    }

    private static boolean isSelectionTimeVarying(KalmanSSM source) {
        return source.observationCount() > 1 && source.selectionOffset(1) != source.selectionOffset(0);
    }

    private static boolean isStateCovarianceTimeVarying(KalmanSSM source) {
        return source.observationCount() > 1 && source.stateCovarianceOffset(1) != source.stateCovarianceOffset(0);
    }

    private static void multiplyTransposedLeft(double[] left, double[] right, int rows, int cols, double[] out) {
        multiplyTransposedLeft(left, 0, right, 0, rows, cols, out, 0);
    }

    private static void multiplyTransposedLeft(double[] left,
                                               int leftBase,
                                               double[] right,
                                               int rightBase,
                                               int rows,
                                               int cols,
                                               double[] out,
                                               int outBase) {
        Arrays.fill(out, outBase, outBase + cols * cols, 0.0);
        for (int row = 0; row < cols; row++) {
            for (int col = 0; col < cols; col++) {
                double value = 0.0;
                for (int inner = 0; inner < rows; inner++) {
                    value += left[leftBase + inner * cols + row] * right[rightBase + inner * cols + col];
                }
                out[outBase + row * cols + col] = value;
            }
        }
    }

    private static void multiplyUpper(double[] upper, double[] vector, double[] out, int n) {
        multiplyUpper(upper, 0, vector, 0, out, 0, n);
    }

    private static void multiplyUpper(double[] upper,
                                      int upperBase,
                                      double[] vector,
                                      int vectorBase,
                                      double[] out,
                                      int outBase,
                                      int n) {
        for (int row = 0; row < n; row++) {
            double value = 0.0;
            for (int col = row; col < n; col++) {
                value += upper[upperBase + row * n + col] * vector[vectorBase + col];
            }
            out[outBase + row] = value;
        }
    }

    private static void multiplyUpperTransposed(double[] upper, double[] vector, double[] out, int n) {
        multiplyUpperTransposed(upper, 0, vector, 0, out, 0, n);
    }

    private static void multiplyUpperTransposed(double[] upper,
                                                int upperBase,
                                                double[] vector,
                                                int vectorBase,
                                                double[] out,
                                                int outBase,
                                                int n) {
        for (int row = 0; row < n; row++) {
            double value = 0.0;
            for (int col = 0; col <= row; col++) {
                value += upper[upperBase + col * n + row] * vector[vectorBase + col];
            }
            out[outBase + row] = value;
        }
    }

    private static void setIdentity(double[] matrix, int n) {
        setIdentity(matrix, 0, n);
    }

    private static void setIdentity(double[] matrix, int matrixBase, int n) {
        Arrays.fill(matrix, matrixBase, matrixBase + n * n, 0.0);
        for (int i = 0; i < n; i++) {
            matrix[matrixBase + i * n + i] = 1.0;
        }
    }

    private static void copyRectangular(double[] source,
                                        int sourceOffset,
                                        int sourceLd,
                                        int rows,
                                        int cols,
                                        double[] target,
                                        int targetOffset,
                                        int targetLd) {
        for (int row = 0; row < rows; row++) {
            System.arraycopy(source, sourceOffset + row * sourceLd, target, targetOffset + row * targetLd, cols);
        }
    }

    private static void copySquare(double[] source,
                                   int sourceOffset,
                                   int sourceLd,
                                   int size,
                                   double[] target,
                                   int targetOffset,
                                   int targetLd) {
        copyRectangular(source, sourceOffset, sourceLd, size, size, target, targetOffset, targetLd);
    }

    private static void copySurface(double[] source, int sourceBase, double[] target, int targetBase, int length) {
        if (length > 0) {
            System.arraycopy(source, sourceBase, target, targetBase, length);
        }
    }

    private static boolean needsLu(ForecastErrorSolver.Method[] sequence) {
        for (ForecastErrorSolver.Method method : sequence) {
            if (method == ForecastErrorSolver.Method.LU_SOLVE) {
                return true;
            }
        }
        return false;
    }

    private record CollapsedData(KalmanSSM model, double[] logLikelihoodAdjustment, int logLikelihoodAdjustmentBase) {
    }
}
