package com.curioloop.yum4j.kalman.filter;

import com.curioloop.yum4j.kalman.arena.FilterScratchLayout;
import com.curioloop.yum4j.kalman.arena.FilterSelectionLayout;
import com.curioloop.yum4j.kalman.arena.FilterDiffuseLayout;
import com.curioloop.yum4j.kalman.arena.FilterResultLayout;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.internal.StateSpaceModelSupport;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Objects;

public final class KalmanFilter {

    private static final double TOLERANCE_DIFFUSE = 1e-7;

    static final int FILTER_CONVENTIONAL = 0x01;
    static final int FILTER_UNIVARIATE = 0x10;

    static final int INVERT_UNIVARIATE = 0x01;
    static final int SOLVE_CHOLESKY = 0x08;

    static final int STABILITY_FORCE_SYMMETRY = 0x01;

    static final int MEMORY_STORE_ALL = 0;
    static final int MEMORY_NO_FORECAST = 0x03;
    static final int MEMORY_NO_PREDICTED = 0x0C;
    static final int MEMORY_NO_FILTERED = 0x30;
    static final int MEMORY_NO_LIKELIHOOD = 0x40;
    static final int MEMORY_NO_GAIN = 0x80;
    static final int MEMORY_NO_SMOOTHING = 0x100;
    static final int MEMORY_CONSERVE = 0x1FF;

    public static final class Pool {
        public UnivariateFilter.Pool diffusePool;
        public int[] selectedIndex;
        public double[] convergedForecastErrorCov;
        public double[] convergedFilteredStateCov;
        public double[] convergedPredictedStateCov;
        public double[] convergedKalmanGain;
        public double[] scratchBacking;
        public double[] selectionBacking;
        public double[] resultBacking;
        public double[] diffuseResultBacking;
        public FilterScratchLayout scratchLayout;
        public FilterSelectionLayout selectionLayout;
        public FilterResultLayout resultLayout;
        public FilterDiffuseLayout diffuseResultLayout;

        private FilterResult pooledResult;
        private StateInitialization.StationaryWorkspace stationaryWorkspace;
        private int scratchKStates;
        private int scratchKEndog;
        private int scratchKPosdef;
        private boolean scratchPredicted;
        private boolean scratchFiltered;
        private boolean scratchDiffusePredicted;
        private int selectionKStates;
        private int selectionKEndog;

        void ensure(int kStates, int kEndog, int kPosdef) {
            ensureScratchArena(kStates, kEndog, kPosdef, false, false, false);
        }

        void ensureSelection(int kStates, int kEndog) {
            int nextSelectionKStates = Math.max(selectionKStates, kStates);
            int nextSelectionKEndog = Math.max(selectionKEndog, kEndog);
            if (selectedIndex == null || selectedIndex.length < kEndog) selectedIndex = new int[kEndog];
            if (selectionLayout == null
                    || nextSelectionKStates != selectionKStates
                    || nextSelectionKEndog != selectionKEndog) {
                selectionLayout = FilterSelectionLayout.create(1, nextSelectionKEndog, nextSelectionKStates);
                selectionKStates = nextSelectionKStates;
                selectionKEndog = nextSelectionKEndog;
            }
            int totalLength = selectionLayout.totalLength();
            if (selectionBacking == null || selectionBacking.length < totalLength) {
                selectionBacking = new double[totalLength];
            }
        }

        void ensureConvergedSnapshots(int kStates, int kEndog) {
            int ksKe = kStates * kEndog;
            int ksKs = kStates * kStates;
            int keKe = kEndog * kEndog;
            if (convergedForecastErrorCov == null || convergedForecastErrorCov.length < keKe)
                convergedForecastErrorCov = new double[keKe];
            if (convergedFilteredStateCov == null || convergedFilteredStateCov.length < ksKs)
                convergedFilteredStateCov = new double[ksKs];
            if (convergedPredictedStateCov == null || convergedPredictedStateCov.length < ksKs)
                convergedPredictedStateCov = new double[ksKs];
            if (convergedKalmanGain == null || convergedKalmanGain.length < ksKe)
                convergedKalmanGain = new double[ksKe];
        }

        void ensurePooledScratch(int kStates,
                                 int kEndog,
                                 int kPosdef,
                                 boolean needPredictedScratch,
                                 boolean needFilteredScratch) {
            ensureScratchArena(kStates, kEndog, kPosdef, needPredictedScratch, needFilteredScratch, false);
        }

        void ensureDiffusePredictedScratch(int kStates, int kEndog, int kPosdef) {
            ensureScratchArena(kStates, kEndog, kPosdef, false, false, true);
        }

        private void ensureScratchArena(int kStates,
                                        int kEndog,
                                        int kPosdef,
                                        boolean needPredictedScratch,
                                        boolean needFilteredScratch,
                                        boolean needDiffusePredictedScratch) {
            int nextScratchKStates = Math.max(scratchKStates, kStates);
            int nextScratchKEndog = Math.max(scratchKEndog, kEndog);
            int nextScratchKPosdef = Math.max(scratchKPosdef, kPosdef);
            boolean nextScratchPredicted = scratchPredicted || needPredictedScratch;
            boolean nextScratchFiltered = scratchFiltered || needFilteredScratch;
            boolean nextScratchDiffusePredicted = scratchDiffusePredicted || needDiffusePredictedScratch;
            if (scratchLayout == null
                    || nextScratchKStates != scratchKStates
                    || nextScratchKEndog != scratchKEndog
                    || nextScratchKPosdef != scratchKPosdef
                    || nextScratchPredicted != scratchPredicted
                    || nextScratchFiltered != scratchFiltered
                    || nextScratchDiffusePredicted != scratchDiffusePredicted) {
                scratchLayout = FilterScratchLayout.create(1,
                    nextScratchKEndog,
                    nextScratchKStates,
                    nextScratchKPosdef,
                    nextScratchPredicted,
                    nextScratchFiltered,
                    nextScratchDiffusePredicted);
                scratchKStates = nextScratchKStates;
                scratchKEndog = nextScratchKEndog;
                scratchKPosdef = nextScratchKPosdef;
                scratchPredicted = nextScratchPredicted;
                scratchFiltered = nextScratchFiltered;
                scratchDiffusePredicted = nextScratchDiffusePredicted;
            }
            int totalLength = scratchLayout.totalLength();
            if (scratchBacking == null || scratchBacking.length < totalLength) {
                scratchBacking = new double[totalLength];
            }
        }

        void ensureResult(FilterResultLayout layout) {
            if (!layout.equals(resultLayout)) {
                resultLayout = layout;
            }
            int totalLength = resultLayout.totalLength();
            if (resultBacking == null || resultBacking.length != totalLength) {
                resultBacking = new double[totalLength];
            }
        }

        void ensureDiffuseResult(FilterDiffuseLayout layout) {
            if (!layout.equals(diffuseResultLayout)) {
                diffuseResultLayout = layout;
            }
            int totalLength = diffuseResultLayout.totalLength();
            if (diffuseResultBacking == null || diffuseResultBacking.length != totalLength) {
                diffuseResultBacking = new double[totalLength];
            }
        }

        FilterResult borrowResult(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
            ensureResult(layout);
            if (pooledResult == null) {
                pooledResult = new FilterResult();
            }
            return pooledResult.reuse(kEndog, kStates, nobs,
                    resultBacking, resultLayout);
        }

        FilterResult borrowDiffuseResult(int kEndog,
                                         int kStates,
                                         int nobs,
                                         FilterResultLayout layout,
                                         FilterDiffuseLayout diffuseLayout) {
            FilterResult result = borrowResult(kEndog, kStates, nobs, layout);
            ensureDiffuseResult(diffuseLayout);
            return result.reuseDiffuse(diffuseResultBacking, diffuseResultLayout);
        }

        public void reserve(StateSpaceModel model,
                            FilterSpec spec) {
            Objects.requireNonNull(model, "model");
            requireRealInputs(model);
            FilterSpec resolvedSpec = spec == null ? FilterSpec.defaults() : spec;
            FilterResultLayout retainedLayout = resolvedSpec.createResultLayout(
                1,
                model.observationDimension(),
                model.stateCount(),
                model.observationCount(),
                StateSpaceModelSupport.hasMissingObservations(model));
            releaseRetainedScratch();
                diffuseResultBacking = null;
                diffuseResultLayout = null;
            ensure(model.stateCount(), model.observationDimension(), model.stateDisturbanceCount());

            ensurePooledScratch(model.stateCount(),
                model.observationDimension(),
                model.stateDisturbanceCount(),
                    retainedLayout.predictedStateLength() == 0,
                    retainedLayout.filteredStateLength() == 0);
            ensureResult(retainedLayout);
        }

        public long retainedScratchDoubleCount() {
            return doubleCount(scratchBacking)
                + doubleCount(selectionBacking)
                    + (stationaryWorkspace == null ? 0L : stationaryWorkspace.retainedDoubleCount())
                    + intCount(selectedIndex)
                    + doubleCount(convergedForecastErrorCov)
                    + doubleCount(convergedFilteredStateCov)
                    + doubleCount(convergedPredictedStateCov)
                    + doubleCount(convergedKalmanGain)
                    + (diffusePool == null ? 0L : diffusePool.retainedScratchDoubleCount());
        }

        public long retainedResultDoubleCount() {
            return doubleCount(resultBacking) + doubleCount(diffuseResultBacking);
        }

        public long retainedTotalByteCount() {
            return (retainedScratchDoubleCount() + retainedResultDoubleCount()) * Double.BYTES;
        }

        public void releaseRetainedResults() {
            resultBacking = null;
            diffuseResultBacking = null;
            pooledResult = null;
        }

        public void releaseRetainedScratch() {
            if (diffusePool != null) {
                diffusePool.releaseRetainedScratch();
            }
            selectedIndex = null;
            convergedForecastErrorCov = null;
            convergedFilteredStateCov = null;
            convergedPredictedStateCov = null;
            convergedKalmanGain = null;
            scratchBacking = null;
            selectionBacking = null;
            scratchLayout = null;
            selectionLayout = null;
            if (stationaryWorkspace != null) {
                stationaryWorkspace.release();
                stationaryWorkspace = null;
            }
            scratchKStates = 0;
            scratchKEndog = 0;
            scratchKPosdef = 0;
            scratchPredicted = false;
            scratchFiltered = false;
            scratchDiffusePredicted = false;
            selectionKStates = 0;
            selectionKEndog = 0;
        }

        void ensureStationary(StateSpaceModel model) {
            if (stationaryWorkspace == null) {
                stationaryWorkspace = new StateInitialization.StationaryWorkspace();
            }
            stationaryWorkspace.ensureCapacity(model);
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }

        private static long intCount(int[] values) {
            return values == null ? 0L : values.length * ((long) Integer.BYTES) / Double.BYTES;
        }

        private int tmpKBase() {
            return scratchLayout.tmpKBase();
        }

        private int tmpPZtBase() {
            return scratchLayout.tmpPZtBase();
        }

        private int tmpFcopyBase() {
            return scratchLayout.tmpFcopyBase();
        }

        private int tmpTPBase() {
            return scratchLayout.tmpTPBase();
        }

        private int tmpVBase() {
            return scratchLayout.tmpVBase();
        }

        private int tmpRQBase() {
            return scratchLayout.tmpRQBase();
        }

        private int rollingPredictedStateBase() {
            return scratchLayout.rollingPredictedStateBase();
        }

        private int rollingPredictedStateCovBase() {
            return scratchLayout.rollingPredictedStateCovBase();
        }

        private int rollingPredictedDiffuseStateCovBase() {
            return scratchLayout.rollingPredictedDiffuseStateCovBase();
        }

        private int scratchFilteredStateBase() {
            return scratchLayout.scratchFilteredStateBase();
        }

        private int scratchFilteredStateCovBase() {
            return scratchLayout.scratchFilteredStateCovBase();
        }

        private int selectedDesignBase() {
            return selectionLayout.selectedDesignBase();
        }

        private int selectedObsCovBase() {
            return selectionLayout.selectedObsCovBase();
        }

        private int selectedObsInterceptBase() {
            return selectionLayout.selectedObsInterceptBase();
        }

        private int selectedEndogBase() {
            return selectionLayout.selectedEndogBase();
        }
    }

    public static FilterResult filter(StateSpaceModel model,
                                      StateInitialization init) {
        return filter(model, init, null, FilterSpec.defaults());
    }

    public static FilterResult filter(StateSpaceModel model,
                                      StateInitialization init,
                                      Pool pool) {
        return filter(model, init, pool, FilterSpec.defaults());
    }

    /**
     * When a non-null pool is supplied, the returned result borrows pool-owned storage
     * and is invalidated by the next call that reuses the same pool. Call clone() or copy()
     * on the result when an independent snapshot is required.
     */
    public static FilterResult filter(StateSpaceModel model,
                                      StateInitialization init,
                                      Pool pool,
                                      FilterSpec spec) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        requireRealInputs(model);
        FilterSpec resolvedSpec = spec == null ? FilterSpec.defaults() : spec;
        FilterResultLayout retainedLayout = resolvedSpec.createResultLayout(
            1,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            StateSpaceModelSupport.hasMissingObservations(model));
        return filterInternal(model, init, pool,
            retainedLayout, resolvedSpec.stabilityMask(), resolvedSpec.conserveMemoryMask(), pool != null, null, 0);
    }

    public static FilterResult filter(StateSpaceModel model,
                                      StateInitialization init,
                                      double[] endog,
                                      int endogBase,
                                      Pool pool,
                                      FilterSpec spec) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(endog, "endog");
        requireRealInputs(model);
        FilterSpec resolvedSpec = spec == null ? FilterSpec.defaults() : spec;
        FilterResultLayout retainedLayout = resolvedSpec.createResultLayout(
            1,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            StateSpaceModelSupport.hasMissingObservations(model));
        return filterInternal(model, init, pool,
            retainedLayout, resolvedSpec.stabilityMask(), resolvedSpec.conserveMemoryMask(), pool != null, endog, endogBase);
    }

    private static FilterResult filterInternal(StateSpaceModel model,
                                               StateInitialization init,
                                               Pool pool,
                               FilterResultLayout retainedLayout,
                                               int stabilityMethod,
                                               int conserveMemory,
                                               boolean reuseResult,
                                               double[] endogOverride,
                                               int endogBase) {
        StateSpaceModel stateSpace = model;
        StateSpaceModel observationModel = model;
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int kEndog2 = kEndog * kEndog;
        int kStates2 = kStates * kStates;
        double[] transition = stateSpace.transitionData();
        int transitionLd = stateSpace.transitionLeadingDimension();
        double[] stateIntercept = stateSpace.stateInterceptData();
        double[] selection = stateSpace.selectionData();
        int selectionLd = stateSpace.selectionLeadingDimension();
        double[] stateCovariance = stateSpace.stateCovarianceData();
        int stateCovarianceLd = stateSpace.stateCovarianceLeadingDimension();
        double[] design = observationModel.designData();
        int designLd = observationModel.designLeadingDimension();
        double[] obsIntercept = observationModel.obsInterceptData();
        double[] obsCov = observationModel.obsCovData();
        double[] endog = endogOverride != null ? endogOverride : observationModel.endogData();

        if (pool == null) pool = new Pool();
        pool.ensure(kStates, kEndog, kPosdef);
        double[] scratch = pool.scratchBacking;

        boolean storeForecast = retainedLayout.forecastLength() > 0;
        boolean retainPredictedHistory = retainedLayout.predictedStateLength() > 0;
        boolean storeFiltered = retainedLayout.filteredStateLength() > 0;
        boolean storeLikelihood = retainedLayout.logLikelihoodObsLength() > 0;
        boolean storeGain = retainedLayout.kalmanGainLength() > 0;
        boolean steadyStateEligible = model.isTimeInvariant();

        if (reuseResult) {
            pool.ensurePooledScratch(kStates, kEndog, kPosdef, !retainPredictedHistory, !storeFiltered);
        }

        StateInitialization.StationaryWorkspace stationaryWorkspace = null;

        int tmpKBase = pool.tmpKBase();
        int tmpPZtBase = pool.tmpPZtBase();
        int tmpFcopyBase = pool.tmpFcopyBase();
        int tmpTPBase = pool.tmpTPBase();
        int tmpVBase = pool.tmpVBase();
        int tmpRQBase = pool.tmpRQBase();
        int rollingPredictedStateBase = reuseResult ? pool.rollingPredictedStateBase() : 0;
        int rollingPredictedStateCovBase = reuseResult ? pool.rollingPredictedStateCovBase() : 0;
        int scratchFilteredStateBase = reuseResult ? pool.scratchFilteredStateBase() : 0;
        int scratchFilteredStateCovBase = reuseResult ? pool.scratchFilteredStateCovBase() : 0;

        FilterResult result = reuseResult
            ? pool.borrowResult(kEndog, kStates, nobs, retainedLayout)
            : new FilterResult(kEndog, kStates, nobs, retainedLayout);
        double[] predictedState = retainPredictedHistory
                ? result.predictedState
            : (reuseResult ? scratch : new double[kStates * 2]);
        double[] predictedStateCov = retainPredictedHistory
                ? result.predictedStateCov
            : (reuseResult ? scratch : new double[kStates2 * 2]);
        double[] filteredState = storeFiltered
                ? result.filteredState
            : (reuseResult ? scratch : new double[kStates]);
        double[] filteredStateCov = storeFiltered
                ? result.filteredStateCov
            : (reuseResult ? scratch : new double[kStates2]);
        double[] forecast = result.forecast;
        double[] forecastError = result.forecastError;
        double[] forecastErrorCov = result.forecastErrorCov;
        double[] kalmanGain = result.kalmanGain;
        int predictedStateBase = retainPredictedHistory ? result.predictedStateBase() : rollingPredictedStateBase;
        int predictedStateCovBase = retainPredictedHistory ? result.predictedStateCovBase() : rollingPredictedStateCovBase;
        int logLikelihoodObsBase = storeLikelihood ? result.logLikelihoodObsBase() : 0;

        if (init.mayResolveDiffuse()) {
            return filterDiffuse(model, init, pool,
                retainedLayout, conserveMemory, reuseResult, endogOverride, endogBase);
        }

        if (init.mayResolveStationary()) {
            pool.ensureStationary(model);
            stationaryWorkspace = pool.stationaryWorkspace;
        }

        init.resolveInto(model, 0, stationaryWorkspace,
            predictedState, predictedStateBase,
            predictedStateCov, predictedStateCovBase,
            null, 0);

        boolean converged = false;
        int periodConverged = nobs;
        double tolerance = 1e-19;
        boolean previousFullyObserved = true;

        for (int t = 0; t < nobs; t++) {
            int predOff = retainPredictedHistory ? result.predictedStateOffset(t) : rollingPredictedStateBase + (t & 1) * kStates;
            int predCovOff = retainPredictedHistory ? result.predictedStateCovOffset(t) : rollingPredictedStateCovBase + (t & 1) * kStates2;
            int filtOff = storeFiltered ? result.filteredStateOffset(t) : scratchFilteredStateBase;
            int filtCovOff = storeFiltered ? result.filteredStateCovOffset(t) : scratchFilteredStateCovBase;
            int foreOff = storeForecast ? result.forecastOffset(t) : 0;
            int foreErrOff = storeForecast ? result.forecastErrorOffset(t) : 0;
            int foreErrCovOff = storeForecast ? result.forecastErrorCovOffset(t) : 0;
            int nextPredOff = retainPredictedHistory ? result.predictedStateOffset(t + 1) : rollingPredictedStateBase + ((t + 1) & 1) * kStates;
            int nextPredCovOff = retainPredictedHistory ? result.predictedStateCovOffset(t + 1) : rollingPredictedStateCovBase + ((t + 1) & 1) * kStates2;
            int kgOff = storeGain ? result.kalmanGainOffset(t) : 0;
            int endogOff = endogOverride != null ? endogBase + t * kEndog : observationModel.endogOffset(t);

            int ke = selectObservations(observationModel, pool, t, endog, endogOff);
            boolean fullyObserved = ke == kEndog;
            boolean useConverged = steadyStateEligible && converged && fullyObserved && previousFullyObserved;
            if (converged && !fullyObserved) {
                converged = false;
            }

            if (ke == 0) {
                if (storeForecast) {
                    for (int i = 0; i < kEndog; i++) {
                        forecast[foreOff + i] = 0.0;
                        forecastError[foreErrOff + i] = 0.0;
                    }
                    for (int i = 0; i < kEndog2; i++) {
                        forecastErrorCov[foreErrCovOff + i] = 0.0;
                    }
                }
                BLAS.dcopy(kStates, predictedState, predOff, 1,
                        filteredState, filtOff, 1);
                BLAS.dcopy(kStates2, predictedStateCov, predCovOff, 1,
                        filteredStateCov, filtCovOff, 1);
                if (storeGain) {
                    for (int i = 0; i < kStates * kEndog; i++) {
                        kalmanGain[kgOff + i] = 0.0;
                    }
                }
                if (storeLikelihood) {
                    result.logLikelihoodObs[logLikelihoodObsBase + t] = 0.0;
                }

                BLAS.dcopy(kStates, stateIntercept, stateSpace.stateInterceptOffset(t), 1,
                        predictedState, nextPredOff, 1);
                BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                        predictedState, predOff, 1,
                        1.0, predictedState, nextPredOff, 1);

                if (!useConverged) {
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                            predictedStateCov, predCovOff, kStates,
                            0.0, scratch, tmpTPBase, kStates);
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                            1.0, scratch, tmpTPBase, kStates,
                        transition, stateSpace.transitionOffset(t), transitionLd,
                            0.0, predictedStateCov, nextPredCovOff, kStates);
                    if (kPosdef > 0) {
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                            0.0, scratch, tmpRQBase, kPosdef);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kPosdef,
                            1.0, scratch, tmpRQBase, kPosdef,
                        selection, stateSpace.selectionOffset(t), selectionLd,
                                1.0, predictedStateCov, nextPredCovOff, kStates);
                    }
                    if ((stabilityMethod & STABILITY_FORCE_SYMMETRY) != 0) {
                        for (int i = 0; i < kStates; i++) {
                            for (int j = i + 1; j < kStates; j++) {
                                double avg = (predictedStateCov[nextPredCovOff + i * kStates + j] +
                                        predictedStateCov[nextPredCovOff + j * kStates + i]) * 0.5;
                                predictedStateCov[nextPredCovOff + i * kStates + j] = avg;
                                predictedStateCov[nextPredCovOff + j * kStates + i] = avg;
                            }
                        }
                    }
                } else {
                    BLAS.dcopy(kStates2, pool.convergedPredictedStateCov, 0, 1,
                            predictedStateCov, nextPredCovOff, 1);
                }
                previousFullyObserved = false;
                continue;
            }

            double[] Z = design;
            int zOff = observationModel.designOffset(t);
            int zLd = designLd;
            double[] H = obsCov;
            int hOff = observationModel.obsCovOffset(t);
            double[] d = obsIntercept;
            int dOff = observationModel.obsInterceptOffset(t);

            if (ke < kEndog) {
                Z = pool.selectionBacking;
                zOff = pool.selectedDesignBase();
                zLd = kStates;
                H = pool.selectionBacking;
                hOff = pool.selectedObsCovBase();
                d = pool.selectionBacking;
                dOff = pool.selectedObsInterceptBase();
            }

            int ke2 = ke * ke;

            // Forecast step:
            // f_t = d_t + Z_t a_t
            // v_t = y_t - f_t
            if (storeForecast) {
                BLAS.dcopy(ke, d, dOff, 1, forecast, foreOff, 1);
                BLAS.dgemv(BLAS.Trans.NoTrans, ke, kStates, 1.0,
                        Z, zOff, zLd,
                    predictedState, predOff, 1,
                    1.0, forecast, foreOff, 1);

                BLAS.dcopy(ke, ke < kEndog ? pool.selectionBacking : endog,
                    ke < kEndog ? pool.selectedEndogBase() : endogOff, 1,
                    forecastError, foreErrOff, 1);
                BLAS.daxpy(ke, -1.0, forecast, foreOff, 1,
                    forecastError, foreErrOff, 1);
            } else {
                BLAS.dcopy(ke, d, dOff, 1, scratch, tmpVBase, 1);
                BLAS.dgemv(BLAS.Trans.NoTrans, ke, kStates, 1.0,
                        Z, zOff, zLd,
                    predictedState, predOff, 1,
                        1.0, scratch, tmpVBase, 1);
                BLAS.dcopy(ke, ke < kEndog ? pool.selectionBacking : endog,
                    ke < kEndog ? pool.selectedEndogBase() : endogOff, 1,
                        scratch, tmpKBase, 1);
                BLAS.daxpy(ke, -1.0, scratch, tmpVBase, 1, scratch, tmpKBase, 1);
                if (!storeForecast) {
                    BLAS.dcopy(ke, scratch, tmpKBase, 1, scratch, tmpVBase, 1);
                }
            }

            // Innovation covariance and cross term:
            // M_t = P_t Z_t'
            // F_t = Z_t P_t Z_t' + H_t
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, ke, kStates,
                    1.0, predictedStateCov, predCovOff, kStates,
                    Z, zOff, zLd,
                    0.0, scratch, tmpPZtBase, ke);

            if (!useConverged) {
                if (storeForecast) {
                    BLAS.dcopy(ke2, H, hOff, 1,
                        forecastErrorCov, foreErrCovOff, 1);
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, kStates,
                            1.0, Z, zOff, zLd,
                            scratch, tmpPZtBase, ke,
                        1.0, forecastErrorCov, foreErrCovOff, ke);
                } else {
                    BLAS.dcopy(ke2, H, hOff, 1, scratch, tmpFcopyBase, 1);
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, kStates,
                            1.0, Z, zOff, zLd,
                            scratch, tmpPZtBase, ke,
                            1.0, scratch, tmpFcopyBase, ke);
                }
            } else {
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, ke, kStates,
                        1.0, pool.convergedPredictedStateCov, 0, kStates,
                        Z, zOff, zLd,
                        0.0, scratch, tmpPZtBase, ke);
                if (storeForecast) {
                    BLAS.dcopy(ke2, H, hOff, 1,
                        forecastErrorCov, foreErrCovOff, 1);
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, kStates,
                            1.0, Z, zOff, zLd,
                            scratch, tmpPZtBase, ke,
                        1.0, forecastErrorCov, foreErrCovOff, ke);
                    BLAS.dcopy(ke2, forecastErrorCov, foreErrCovOff, 1,
                            scratch, tmpFcopyBase, 1);
                } else {
                    BLAS.dcopy(ke2, H, hOff, 1, scratch, tmpFcopyBase, 1);
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, kStates,
                            1.0, Z, zOff, zLd,
                            scratch, tmpPZtBase, ke,
                            1.0, scratch, tmpFcopyBase, ke);
                }
            }

            double logDetF = 0.0;
            double vFv = 0.0;
            double savedForecastError = 0.0;
            if (!storeForecast && ke == 1) {
                savedForecastError = scratch[tmpKBase];
            }

            if (storeForecast) {
                BLAS.dcopy(ke2, forecastErrorCov, foreErrCovOff, 1, scratch, tmpFcopyBase, 1);
            }

            if (ke == 1) {
                double Ft = scratch[tmpFcopyBase];
                if (Ft <= TOLERANCE_DIFFUSE) {
                    BLAS.dcopy(kStates, predictedState, predOff, 1,
                        filteredState, filtOff, 1);
                    BLAS.dcopy(kStates2, predictedStateCov, predCovOff, 1,
                        filteredStateCov, filtCovOff, 1);
                } else {
                double FtInv = 1.0 / Ft;
                logDetF = Math.log(Ft);

                BLAS.dcopy(kStates, scratch, tmpPZtBase, 1, scratch, tmpKBase, 1);
                BLAS.dscal(kStates, FtInv, scratch, tmpKBase, 1);

                BLAS.dcopy(kStates, predictedState, predOff, 1,
                    filteredState, filtOff, 1);
                if (storeForecast) {
                    BLAS.daxpy(kStates, forecastError[foreErrOff] * FtInv,
                        scratch, tmpPZtBase, 1, filteredState, filtOff, 1);
                } else {
                    BLAS.daxpy(kStates, savedForecastError * FtInv,
                        scratch, tmpPZtBase, 1, filteredState, filtOff, 1);
                }

                BLAS.dcopy(kStates2, predictedStateCov, predCovOff, 1,
                    filteredStateCov, filtCovOff, 1);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, ke,
                        -FtInv, scratch, tmpPZtBase, ke,
                        scratch, tmpPZtBase, ke,
                    1.0, filteredStateCov, filtCovOff, kStates);

                if (storeGain) {
                    BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                            transition, stateSpace.transitionOffset(t), transitionLd,
                        scratch, tmpKBase, 1,
                        0.0, kalmanGain, kgOff, 1);
                }

                if (storeForecast) {
                    vFv = forecastError[foreErrOff] * FtInv * forecastError[foreErrOff];
                } else {
                    vFv = savedForecastError * FtInv * savedForecastError;
                }
                }
            } else {
                BLAS.dpotrf(BLAS.Uplo.Lower, ke, scratch, tmpFcopyBase, ke);

                for (int i = 0; i < ke; i++) {
                    logDetF += Math.log(scratch[tmpFcopyBase + i * ke + i]);
                }
                logDetF *= 2.0;

                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, kStates, kStates,
                        1.0, Z, zOff, zLd,
                    predictedStateCov, predCovOff, kStates,
                        0.0, scratch, tmpKBase, kStates);

                BLAS.dpotrs(BLAS.Uplo.Lower, ke, kStates,
                        scratch, tmpFcopyBase, ke, scratch, tmpKBase, kStates);

                BLAS.dcopy(kStates, predictedState, predOff, 1,
                    filteredState, filtOff, 1);
                double[] vSrc = storeForecast ? forecastError : scratch;
                int vSrcOff = storeForecast ? foreErrOff : 0;
                BLAS.dgemv(BLAS.Trans.Trans, ke, kStates, 1.0,
                    scratch, tmpKBase, kStates,
                        vSrc, vSrcOff, 1,
                    1.0, filteredState, filtOff, 1);

                BLAS.dcopy(kStates2, predictedStateCov, predCovOff, 1,
                    filteredStateCov, filtCovOff, 1);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                        -1.0, scratch, tmpPZtBase, ke,
                        scratch, tmpKBase, kStates,
                    1.0, filteredStateCov, filtCovOff, kStates);

                if (storeGain) {
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, ke, kStates,
                            1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                            scratch, tmpKBase, kStates,
                        0.0, kalmanGain, kgOff, ke);
                }

                if (storeForecast) {
                    BLAS.dcopy(ke, forecastError, foreErrOff, 1, scratch, tmpVBase, 1);
                }
                BLAS.dcopy(ke, scratch, tmpVBase, 1, scratch, tmpKBase, 1);
                BLAS.dpotrs(BLAS.Uplo.Lower, ke, 1,
                        scratch, tmpFcopyBase, ke, scratch, tmpVBase, 1);
                if (storeForecast) {
                    vFv = BLAS.ddot(ke, forecastError, foreErrOff, 1, scratch, tmpVBase, 1);
                } else {
                    vFv = BLAS.ddot(ke, scratch, tmpKBase, 1, scratch, tmpVBase, 1);
                }
            }

            if ((stabilityMethod & STABILITY_FORCE_SYMMETRY) != 0) {
                for (int i = 0; i < kStates; i++) {
                    for (int j = i + 1; j < kStates; j++) {
                        double avg = (filteredStateCov[filtCovOff + i * kStates + j] +
                                filteredStateCov[filtCovOff + j * kStates + i]) * 0.5;
                        filteredStateCov[filtCovOff + i * kStates + j] = avg;
                        filteredStateCov[filtCovOff + j * kStates + i] = avg;
                    }
                }
            }

            if (storeLikelihood) {
                result.logLikelihoodObs[logLikelihoodObsBase + t] = -0.5 * (ke * Math.log(2 * Math.PI) + logDetF + vFv);
            }

            if (useConverged) {
                BLAS.dcopy(kStates2, pool.convergedFilteredStateCov, 0, 1,
                        filteredStateCov, filtCovOff, 1);
                if (storeGain) {
                    BLAS.dcopy(kStates * kEndog, pool.convergedKalmanGain, 0, 1,
                            kalmanGain, kgOff, 1);
                }
            }

                // Time update:
                // a_{t+1} = c_t + T_t a_{t|t}
                // P_{t+1} = T_t P_{t|t} T_t' + R_t Q_t R_t'
                BLAS.dcopy(kStates, stateIntercept, stateSpace.stateInterceptOffset(t), 1,
                    predictedState, nextPredOff, 1);
            BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                    filteredState, filtOff, 1,
                    1.0, predictedState, nextPredOff, 1);

            if (!useConverged) {
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                    filteredStateCov, filtCovOff, kStates,
                        0.0, scratch, tmpTPBase, kStates);

                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                        1.0, scratch, tmpTPBase, kStates,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                    0.0, predictedStateCov, nextPredCovOff, kStates);

                if (kPosdef > 0) {
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                            0.0, scratch, tmpRQBase, kPosdef);

                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kPosdef,
                            1.0, scratch, tmpRQBase, kPosdef,
                        selection, stateSpace.selectionOffset(t), selectionLd,
                            1.0, predictedStateCov, nextPredCovOff, kStates);
                }

                if ((stabilityMethod & STABILITY_FORCE_SYMMETRY) != 0) {
                    for (int i = 0; i < kStates; i++) {
                        for (int j = i + 1; j < kStates; j++) {
                            double avg = (predictedStateCov[nextPredCovOff + i * kStates + j] +
                                    predictedStateCov[nextPredCovOff + j * kStates + i]) * 0.5;
                            predictedStateCov[nextPredCovOff + i * kStates + j] = avg;
                            predictedStateCov[nextPredCovOff + j * kStates + i] = avg;
                        }
                    }
                }

                if (steadyStateEligible && t > 0 && fullyObserved && previousFullyObserved) {
                    double maxDiff = 0.0;
                    for (int i = 0; i < kStates2; i++) {
                        double diff = Math.abs(predictedStateCov[nextPredCovOff + i] -
                                predictedStateCov[predCovOff + i]);
                        if (diff > maxDiff) maxDiff = diff;
                    }
                    if (maxDiff < tolerance) {
                        converged = true;
                        periodConverged = t + 1;
                        pool.ensureConvergedSnapshots(kStates, kEndog);
                        if (storeForecast) {
                            BLAS.dcopy(ke2, forecastErrorCov, foreErrCovOff, 1,
                                    pool.convergedForecastErrorCov, 0, 1);
                        } else {
                            BLAS.dcopy(ke2, scratch, tmpFcopyBase, 1,
                                    pool.convergedForecastErrorCov, 0, 1);
                        }
                        BLAS.dcopy(kStates2, filteredStateCov, filtCovOff, 1,
                                pool.convergedFilteredStateCov, 0, 1);
                        BLAS.dcopy(kStates2, predictedStateCov, nextPredCovOff, 1,
                                pool.convergedPredictedStateCov, 0, 1);
                        if (storeGain) {
                            BLAS.dcopy(kStates * kEndog, kalmanGain, kgOff, 1,
                                    pool.convergedKalmanGain, 0, 1);
                        }
                    }
                }
            } else {
                BLAS.dcopy(kStates2, pool.convergedPredictedStateCov, 0, 1,
                        predictedStateCov, nextPredCovOff, 1);
            }

            previousFullyObserved = fullyObserved;
        }

        for (int t = 0; t < nobs; t++) {
            if (observationModel.missingCount(t) == 0) continue;

            int foreOff = result.forecastOffset(t);
            int foreErrOff = result.forecastErrorOffset(t);
            int foreErrCovOff = result.forecastErrorCovOffset(t);
            int kgOff = result.kalmanGainOffset(t);
            int predOff = result.predictedStateOffset(t);
            int predCovOff = result.predictedStateCovOffset(t);

            if (storeForecast) {
                BLAS.dcopy(kEndog, obsIntercept, observationModel.obsInterceptOffset(t), 1,
                        result.forecast, foreOff, 1);
                BLAS.dgemv(BLAS.Trans.NoTrans, kEndog, kStates, 1.0,
                        design, observationModel.designOffset(t), designLd,
                        result.predictedState, predOff, 1,
                        1.0, result.forecast, foreOff, 1);

                if (observationModel.missingCount(t) < kEndog) {
                    int ke = selectObservations(observationModel, pool, t, endog,
                            endogOverride != null ? endogBase + t * kEndog : observationModel.endogOffset(t));
                    int[] selectedIndex = pool.selectedIndex;
                    BLAS.dcopy(ke, result.forecastError, foreErrOff, 1, scratch, tmpVBase, 1);
                    for (int i = 0; i < kEndog; i++) result.forecastError[foreErrOff + i] = 0.0;
                    for (int j = 0; j < ke; j++) result.forecastError[foreErrOff + selectedIndex[j]] = scratch[tmpVBase + j];
                } else {
                    for (int i = 0; i < kEndog; i++) result.forecastError[foreErrOff + i] = 0.0;
                }

                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kEndog, kStates, 1.0,
                        result.predictedStateCov, predCovOff, kStates,
                        design, observationModel.designOffset(t), designLd,
                    0.0, scratch, tmpPZtBase, kEndog);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kEndog, kEndog, kStates, 1.0,
                        design, observationModel.designOffset(t), designLd,
                    scratch, tmpPZtBase, kEndog,
                        0.0, result.forecastErrorCov, foreErrCovOff, kEndog);
                BLAS.daxpy(kEndog2, 1.0, obsCov, observationModel.obsCovOffset(t), 1,
                        result.forecastErrorCov, foreErrCovOff, 1);
            }

            if (observationModel.missingCount(t) < kEndog && storeGain) {
                int ke = selectObservations(observationModel, pool, t, endog,
                        endogOverride != null ? endogBase + t * kEndog : observationModel.endogOffset(t));
                int[] selectedIndex = pool.selectedIndex;
                BLAS.dcopy(kStates * ke, result.kalmanGain, kgOff, 1, scratch, tmpKBase, 1);
                for (int i = 0; i < kStates * kEndog; i++) result.kalmanGain[kgOff + i] = 0.0;
                for (int i = 0; i < kStates; i++) {
                    for (int j = 0; j < ke; j++) {
                        result.kalmanGain[kgOff + i * kEndog + selectedIndex[j]] = scratch[tmpKBase + i * ke + j];
                    }
                }
            } else if (storeGain) {
                for (int i = 0; i < kStates * kEndog; i++) result.kalmanGain[kgOff + i] = 0.0;
            }
        }

        result.applyConserveMemory(conserveMemory);

        result.converged = converged;
        result.periodConverged = periodConverged;
        return result;
    }

    private static void requireRealInputs(StateSpaceModel model) {
        StateSpaceModelSupport.requireRealStorage(model, "model");
    }

    private static FilterResult filterDiffuse(StateSpaceModel model,
                                              StateInitialization init,
                                              Pool pool,
                                              FilterResultLayout retainedLayout,
                                              int conserveMemory,
                                              boolean reuseResult,
                                              double[] endogOverride,
                                              int endogBase) {
        StateSpaceModel stateSpace = model;
        StateSpaceModel observationModel = model;
        if (pool.diffusePool == null) {
            pool.diffusePool = new UnivariateFilter.Pool();
        }

        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int nobs = observationModel.observationCount();

        boolean storePredicted = retainedLayout.predictedStateLength() > 0;
        boolean storeFiltered = retainedLayout.filteredStateLength() > 0;
        FilterDiffuseLayout diffuseLayout = FilterDiffuseLayout.create(
            1,
            kEndog,
            kStates,
            nobs,
            storePredicted,
            retainedLayout.forecastLength() > 0);

        if (reuseResult) {
            pool.ensurePooledScratch(kStates, kEndog, stateSpace.stateDisturbanceCount(), !storePredicted, !storeFiltered);
            if (!storePredicted) {
                pool.ensureDiffusePredictedScratch(kStates, kEndog, stateSpace.stateDisturbanceCount());
            }
        }

        FilterResult result;
        if (reuseResult) {
            result = pool.borrowDiffuseResult(kEndog, kStates, nobs, retainedLayout, diffuseLayout);
            result.clearActiveSurfaces();
        } else {
            result = new FilterResult(kEndog, kStates, nobs, retainedLayout);
        }

        UnivariateFilter.filterDiffuse(model, init, pool.diffusePool, result,
            endogOverride, endogBase,
            reuseResult ? pool.scratchBacking : null,
            reuseResult ? pool.rollingPredictedStateBase() : 0,
            reuseResult ? pool.scratchBacking : null,
            reuseResult ? pool.rollingPredictedStateCovBase() : 0,
            reuseResult ? pool.scratchBacking : null,
            reuseResult ? pool.rollingPredictedDiffuseStateCovBase() : 0,
            reuseResult ? pool.scratchBacking : null,
            reuseResult ? pool.scratchFilteredStateBase() : 0,
            reuseResult ? pool.scratchBacking : null,
            reuseResult ? pool.scratchFilteredStateCovBase() : 0);
        result.converged = false;
        result.periodConverged = nobs;
        if (reuseResult) {
            result.applyConserveMemory(conserveMemory);
        }
        return result;
    }

    private static int selectObservations(StateSpaceModel observationModel,
                                          Pool pool,
                                          int t,
                                          double[] endog,
                                          int endogOff) {
        int kEndog = observationModel.observationDimension();
        int missing = observationModel.missingCount(t);
        if (missing == 0) {
            return kEndog;
        }
        if (missing == kEndog) {
            return 0;
        }

        int kStates = observationModel.stateCount();
        pool.ensureSelection(kStates, kEndog);
        int selectedDesignBase = pool.selectedDesignBase();
        int selectedObsInterceptBase = pool.selectedObsInterceptBase();
        int selectedEndogBase = pool.selectedEndogBase();
        int selectedObsCovBase = pool.selectedObsCovBase();
        int ke = 0;
        for (int i = 0; i < kEndog; i++) {
            if (!observationModel.isMissing(i, t)) {
                pool.selectedIndex[ke++] = i;
            }
        }

        double[] design = observationModel.designData();
        int designOff = observationModel.designOffset(t);
        int designLd = observationModel.designLeadingDimension();
        double[] obsIntercept = observationModel.obsInterceptData();
        int obsInterceptOff = observationModel.obsInterceptOffset(t);
        double[] obsCov = observationModel.obsCovData();
        int obsCovOff = observationModel.obsCovOffset(t);
        int obsCovLd = observationModel.obsCovLeadingDimension();

        for (int row = 0; row < ke; row++) {
            int source = pool.selectedIndex[row];
            System.arraycopy(design, designOff + source * designLd,
                    pool.selectionBacking, selectedDesignBase + row * kStates, kStates);
            pool.selectionBacking[selectedObsInterceptBase + row] = obsIntercept[obsInterceptOff + source];
            pool.selectionBacking[selectedEndogBase + row] = endog[endogOff + source];
        }

        for (int row = 0; row < ke; row++) {
            int sourceRow = pool.selectedIndex[row];
            int sourceRowOff = obsCovOff + sourceRow * obsCovLd;
            int targetRowOff = selectedObsCovBase + row * ke;
            for (int col = 0; col < ke; col++) {
                int sourceCol = pool.selectedIndex[col];
                pool.selectionBacking[targetRowOff + col] = obsCov[sourceRowOff + sourceCol];
            }
        }
        return ke;
    }
}
