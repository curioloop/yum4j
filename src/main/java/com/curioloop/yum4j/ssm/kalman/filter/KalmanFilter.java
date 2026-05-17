package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.arena.FilterScratchLayout;
import com.curioloop.yum4j.ssm.kalman.arena.FilterSelectionLayout;
import com.curioloop.yum4j.ssm.kalman.arena.FilterDiffuseLayout;
import com.curioloop.yum4j.ssm.kalman.arena.FilterResultLayout;
import com.curioloop.yum4j.ssm.kalman.arena.IntArena;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Objects;

public final class KalmanFilter {

    public static final class SingularForecastException extends ArithmeticException {
        SingularForecastException(String message) {
            super(message);
        }
    }

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

    static final class Pool {
        private final DoubleArena scratchArena;
        private final DoubleArena resultArena;
        private final DoubleArena forecastErrorCovWorkArena = new DoubleArena();
        private final DoubleArena forecastErrorWorkArena = new DoubleArena();
        private final IntArena selectedIndexArena = new IntArena();
        private final IntArena forecastErrorPivotArena = new IntArena();
        private final boolean ownsScratchArena;
        private final boolean ownsResultArena;
        UnivariateFilter.Pool diffusePool;
        public int[] selectedIndex;
        public double[] convergedForecastErrorCov;
        public double[] convergedFilteredStateCov;
        public double[] convergedPredictedStateCov;
        public double[] convergedKalmanGain;
        public double[] forecastErrorCovWork;
        public double[] forecastErrorWork;
        public int[] forecastErrorPivots;
        public double[] scratchBacking;
        public double[] selectionBacking;
        public double[] resultBacking;
        public double[] diffuseResultBacking;
        public FilterScratchLayout scratchLayout;
        public FilterSelectionLayout selectionLayout;
        public FilterResultLayout resultLayout;
        public FilterDiffuseLayout diffuseResultLayout;
        private int diffuseResultBase;

        private FilterResult pooledResult;
        private ForecastErrorSolver.Result forecastErrorResult;
        private InitialState.StationaryWorkspace stationaryWorkspace;
        private int scratchKStates;
        private int scratchKEndog;
        private int scratchKPosdef;
        private boolean scratchPredicted;
        private boolean scratchFiltered;
        private boolean scratchDiffusePredicted;
        private int selectionKStates;
        private int selectionKEndog;
        private int selectionBase;

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

        void ensure(int kStates, int kEndog, int kPosdef) {
            ensureScratchArena(kStates, kEndog, kPosdef, false, false, false);
        }

        void ensureSelection(int kStates, int kEndog) {
            int nextSelectionKStates = Math.max(selectionKStates, kStates);
            int nextSelectionKEndog = Math.max(selectionKEndog, kEndog);
            selectedIndex = selectedIndexArena.ensureCapacity(kEndog);
            if (selectionLayout == null
                    || nextSelectionKStates != selectionKStates
                    || nextSelectionKEndog != selectionKEndog) {
                selectionLayout = FilterSelectionLayout.create(1, nextSelectionKEndog, nextSelectionKStates);
                selectionKStates = nextSelectionKStates;
                selectionKEndog = nextSelectionKEndog;
            }
            int totalLength = selectionLayout.totalLength();
            selectionBase = scratchLayout == null ? 0 : scratchLayout.totalLength();
            double[] backing = scratchArena.ensureCapacity(selectionBase + totalLength);
            scratchBacking = backing;
            selectionBacking = backing;
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

        void releaseConvergedSnapshots() {
            convergedForecastErrorCov = null;
            convergedFilteredStateCov = null;
            convergedPredictedStateCov = null;
            convergedKalmanGain = null;
        }

        ForecastErrorSolver.Result forecastErrorResult() {
            if (forecastErrorResult == null) {
                forecastErrorResult = new ForecastErrorSolver.Result();
            }
            return forecastErrorResult;
        }

        double[] ensureForecastErrorCovWork(int kEndog) {
            int covarianceLength = kEndog * kEndog;
            forecastErrorCovWork = forecastErrorCovWorkArena.ensureCapacity(covarianceLength);
            return forecastErrorCovWork;
        }

        double[] ensureForecastErrorWork(int kEndog) {
            forecastErrorWork = forecastErrorWorkArena.ensureCapacity(kEndog);
            return forecastErrorWork;
        }

        int[] ensureForecastErrorPivots(int kEndog) {
            forecastErrorPivots = forecastErrorPivotArena.ensureCapacity(kEndog);
            return forecastErrorPivots;
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
            int totalLength = scratchLayout.totalLength()
                + (selectionLayout == null ? 0 : selectionLayout.totalLength());
            selectionBase = scratchLayout.totalLength();
            double[] backing = scratchArena.ensureCapacity(totalLength);
            scratchBacking = backing;
            if (selectionLayout != null) {
                selectionBacking = backing;
            }
        }

        void ensureResult(FilterResultLayout layout) {
            if (!layout.equals(resultLayout)) {
                resultLayout = layout;
            }
            ensureResultArena();
        }

        void ensureDiffuseResult(FilterDiffuseLayout layout) {
            if (!layout.equals(diffuseResultLayout)) {
                diffuseResultLayout = layout;
            }
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
            ensureResult(layout);
            ensureDiffuseResult(diffuseLayout);
            if (pooledResult == null) {
                pooledResult = new FilterResult();
            }
            return pooledResult
                .reuse(kEndog, kStates, nobs, resultBacking, resultLayout)
                .reuseDiffuse(diffuseResultBacking, diffuseResultBase, diffuseResultLayout);
        }

        public void reserve(KalmanSSM model,
                            FilterOptions spec) {
            Objects.requireNonNull(model, "model");
            requireRealInputs(model);
            FilterOptions resolvedOptions = spec == null ? FilterOptions.defaults() : spec;
            FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(
                1,
                model.observationDimension(),
                model.stateCount(),
                model.observationCount(),
                KalmanSSMSupport.hasMissingObservations(model));
            ensure(model.stateCount(), model.observationDimension(), model.stateDisturbanceCount());

            ensurePooledScratch(model.stateCount(),
                model.observationDimension(),
                model.stateDisturbanceCount(),
                    retainedLayout.predictedStateLength() == 0,
                    retainedLayout.filteredStateLength() == 0);
            ensureResult(retainedLayout);
        }

        public long retainedScratchDoubleCount() {
            return (usesScratchArena() ? scratchArena.retainedDoubleCount() : 0L)
                    + retainedAuxiliaryScratchDoubleCount();
        }

        long retainedAuxiliaryScratchDoubleCount() {
            return (stationaryWorkspace == null ? 0L : stationaryWorkspace.retainedDoubleCount())
                    + intArenaDoubleEquivalentCount(selectedIndexArena)
                    + doubleCount(convergedForecastErrorCov)
                    + doubleCount(convergedFilteredStateCov)
                    + doubleCount(convergedPredictedStateCov)
                    + doubleCount(convergedKalmanGain)
                    + forecastErrorCovWorkArena.retainedDoubleCount()
                    + forecastErrorWorkArena.retainedDoubleCount()
                    + intArenaDoubleEquivalentCount(forecastErrorPivotArena)
                    + (diffusePool == null ? 0L : diffusePool.retainedScratchDoubleCount());
        }

        boolean usesScratchArena() {
            return scratchBacking != null;
        }

        void refreshScratchBacking() {
            if (scratchLayout != null) {
                scratchBacking = scratchArena.backing();
                if (selectionLayout != null) {
                    selectionBacking = scratchBacking;
                }
            }
        }

        public long retainedResultDoubleCount() {
            return resultArena.retainedDoubleCount();
        }

        public long retainedTotalByteCount() {
            return (retainedScratchDoubleCount() + retainedResultDoubleCount()) * Double.BYTES;
        }

        public void releaseRetainedResults() {
            invalidateBorrowedResult();
            pooledResult = null;
            if (ownsResultArena) {
                resultArena.release();
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

        public void releaseRetainedScratch() {
            if (diffusePool != null) {
                diffusePool.releaseRetainedScratch();
            }
            selectedIndex = null;
            selectedIndexArena.release();
            releaseConvergedSnapshots();
            forecastErrorCovWork = null;
            forecastErrorCovWorkArena.release();
            forecastErrorWork = null;
            forecastErrorWorkArena.release();
            forecastErrorPivots = null;
            forecastErrorPivotArena.release();
            forecastErrorResult = null;
            scratchBacking = null;
            selectionBacking = null;
            if (ownsScratchArena) {
                scratchArena.release();
            }
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
            selectionBase = 0;
        }

        void ensureStationary(KalmanSSM model) {
            if (stationaryWorkspace == null) {
                stationaryWorkspace = new InitialState.StationaryWorkspace();
            }
            stationaryWorkspace.ensureCapacity(model);
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }

        private static long intArenaDoubleEquivalentCount(IntArena arena) {
            return arena.retainedIntCount() * ((long) Integer.BYTES) / Double.BYTES;
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
            return selectionBase + selectionLayout.selectedDesignBase();
        }

        private int selectedObsCovBase() {
            return selectionBase + selectionLayout.selectedObsCovBase();
        }

        private int selectedObsInterceptBase() {
            return selectionBase + selectionLayout.selectedObsInterceptBase();
        }

        private int selectedEndogBase() {
            return selectionBase + selectionLayout.selectedEndogBase();
        }
    }

    /**
     * When a non-null pool is supplied, the returned result borrows pool-owned storage
     * and is invalidated by the next call that reuses the same pool. Call clone() or copy()
     * on the result when an independent snapshot is required.
     */
    static FilterResult filter(KalmanSSM model,
                               InitialState init,
                               Pool pool,
                               FilterOptions spec) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        requireRealInputs(model);
        FilterOptions resolvedOptions = spec == null ? FilterOptions.defaults() : spec;
        boolean hasMissingObservations = KalmanSSMSupport.hasMissingObservations(model);
        FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(
            1,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            hasMissingObservations);
        FilterDiffuseLayout retainedDiffuseLayout = init.mayResolveDiffuse()
            ? resolvedOptions.createDiffuseLayout(
                1,
                model.observationDimension(),
                model.stateCount(),
                model.observationCount(),
                hasMissingObservations)
            : null;
        return filterInternal(model, init, pool,
            retainedLayout,
            retainedDiffuseLayout,
            resolvedOptions.stabilityMask(),
            resolvedOptions.conserveMemoryMask(),
            resolvedOptions.diffuseTolerance(),
            resolvedOptions.convergenceTolerance(),
            resolvedOptions.forecastErrorSequenceRef(),
            resolvedOptions.concentratedLikelihood(),
            resolvedOptions.concentratedLikelihoodBurn(),
            pool != null,
            null,
            0);
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState init,
                               double[] endog,
                               int endogBase,
                               Pool pool,
                               FilterOptions spec) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(endog, "endog");
        requireRealInputs(model);
        FilterOptions resolvedOptions = spec == null ? FilterOptions.defaults() : spec;
        boolean hasMissingObservations = KalmanSSMSupport.hasMissingObservations(model);
        FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(
            1,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            hasMissingObservations);
        FilterDiffuseLayout retainedDiffuseLayout = init.mayResolveDiffuse()
            ? resolvedOptions.createDiffuseLayout(
                1,
                model.observationDimension(),
                model.stateCount(),
                model.observationCount(),
                hasMissingObservations)
            : null;
        return filterInternal(model, init, pool,
            retainedLayout,
            retainedDiffuseLayout,
            resolvedOptions.stabilityMask(),
            resolvedOptions.conserveMemoryMask(),
            resolvedOptions.diffuseTolerance(),
            resolvedOptions.convergenceTolerance(),
            resolvedOptions.forecastErrorSequenceRef(),
            resolvedOptions.concentratedLikelihood(),
            resolvedOptions.concentratedLikelihoodBurn(),
            pool != null,
            endog,
            endogBase);
    }

    private static FilterResult filterInternal(KalmanSSM model,
                                               InitialState init,
                                               Pool pool,
                               FilterResultLayout retainedLayout,
                                               FilterDiffuseLayout retainedDiffuseLayout,
                                               int stabilityMethod,
                                               int conserveMemory,
                                               double diffuseTolerance,
                                               double convergenceTolerance,
                                               ForecastErrorSolver.Method[] forecastErrorSequence,
                                               boolean concentratedLikelihood,
                                               int concentratedLikelihoodBurn,
                                               boolean reuseResult,
                                               double[] endogOverride,
                                               int endogBase) {
        KalmanSSM stateSpace = model;
        KalmanSSM observationModel = model;
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
        ForecastErrorSolver.Result forecastErrorSolve = pool.forecastErrorResult();

        boolean storeForecastMean = retainedLayout.forecastLength() > 0;
        boolean storeForecastError = retainedLayout.forecastErrorLength() > 0;
        boolean storeForecastCovariance = retainedLayout.forecastErrorCovLength() > 0;
        boolean storeStandardizedForecastError = retainedLayout.standardizedForecastErrorLength() > 0;
        boolean storePredictedState = retainedLayout.predictedStateLength() > 0;
        boolean storePredictedCovariance = retainedLayout.predictedStateCovLength() > 0;
        boolean storeFilteredState = retainedLayout.filteredStateLength() > 0;
        boolean storeFilteredCovariance = retainedLayout.filteredStateCovLength() > 0;
        boolean storeLikelihood = retainedLayout.logLikelihoodObsLength() > 0;
        boolean storeGain = retainedLayout.kalmanGainLength() > 0;
        boolean steadyStateEligible = model.isTimeInvariant();

        boolean usePredictedScratch = !storePredictedState || !storePredictedCovariance;
        boolean useFilteredScratch = !storeFilteredState || !storeFilteredCovariance;
        if (usePredictedScratch || useFilteredScratch) {
            pool.ensurePooledScratch(kStates, kEndog, kPosdef, usePredictedScratch, useFilteredScratch);
        }
        double[] scratch = pool.scratchBacking;

        InitialState.StationaryWorkspace stationaryWorkspace = null;

        int tmpKBase = pool.tmpKBase();
        int tmpPZtBase = pool.tmpPZtBase();
        int tmpFcopyBase = pool.tmpFcopyBase();
        int tmpTPBase = pool.tmpTPBase();
        int tmpVBase = pool.tmpVBase();
        int tmpRQBase = pool.tmpRQBase();
        int rollingPredictedStateBase = usePredictedScratch ? pool.rollingPredictedStateBase() : 0;
        int rollingPredictedStateCovBase = usePredictedScratch ? pool.rollingPredictedStateCovBase() : 0;
        int scratchFilteredStateBase = useFilteredScratch ? pool.scratchFilteredStateBase() : 0;
        int scratchFilteredStateCovBase = useFilteredScratch ? pool.scratchFilteredStateCovBase() : 0;

        FilterResult result = reuseResult
            ? pool.borrowResult(kEndog, kStates, nobs, retainedLayout)
            : new FilterResult(kEndog, kStates, nobs, retainedLayout);
        if (concentratedLikelihood) {
            result.concentratedLikelihood = true;
            result.scale = 1.0;
            result.scaleObservationCount = 0;
            if (storeLikelihood) {
                result.retainScaleObs(nobs);
                result.retainScaleObsCount(nobs);
            } else {
                result.dropScaleObs();
                result.dropScaleObsCount();
            }
        } else {
            result.concentratedLikelihood = false;
            result.scale = 1.0;
            result.scaleObservationCount = 0;
            result.dropScaleObs();
            result.dropScaleObsCount();
        }
        double[] predictedState = storePredictedState
                ? result.predictedState
            : scratch;
        double[] predictedStateCov = storePredictedCovariance
                ? result.predictedStateCov
            : scratch;
        double[] filteredState = storeFilteredState
                ? result.filteredState
            : scratch;
        double[] filteredStateCov = storeFilteredCovariance
                ? result.filteredStateCov
            : scratch;
        double[] forecast = result.forecast;
        double[] forecastError = result.forecastError;
        double[] forecastErrorCov = result.forecastErrorCov;
        double[] standardizedForecastError = result.standardizedForecastError;
        double[] kalmanGain = result.kalmanGain;
        int predictedStateBase = storePredictedState ? result.predictedStateBase() : rollingPredictedStateBase;
        int predictedStateCovBase = storePredictedCovariance ? result.predictedStateCovBase() : rollingPredictedStateCovBase;
        int logLikelihoodObsBase = storeLikelihood ? result.logLikelihoodObsBase() : 0;

        if (init.mayResolveDiffuse()) {
            return filterDiffuse(model, init, pool,
                retainedLayout, Objects.requireNonNull(retainedDiffuseLayout, "retainedDiffuseLayout"), conserveMemory, diffuseTolerance,
                concentratedLikelihood, concentratedLikelihoodBurn, reuseResult, endogOverride, endogBase);
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
        boolean previousFullyObserved = true;
        double concentratedScaleSum = 0.0;
        int concentratedScaleObservationCount = 0;
        double concentratedTotalScaleSum = 0.0;
        int concentratedTotalObservationCount = 0;
        double logLikelihoodSum = 0.0;
        int concentratedScaleStart = Math.max(0, concentratedLikelihoodBurn);

        for (int t = 0; t < nobs; t++) {
            int predOff = storePredictedState ? result.predictedStateOffset(t) : rollingPredictedStateBase + (t & 1) * kStates;
            int predCovOff = storePredictedCovariance ? result.predictedStateCovOffset(t) : rollingPredictedStateCovBase + (t & 1) * kStates2;
            int filtOff = storeFilteredState ? result.filteredStateOffset(t) : scratchFilteredStateBase;
            int filtCovOff = storeFilteredCovariance ? result.filteredStateCovOffset(t) : scratchFilteredStateCovBase;
            int foreOff = storeForecastMean ? result.forecastOffset(t) : 0;
            int foreErrOff = storeForecastError ? result.forecastErrorOffset(t) : 0;
            int foreErrCovOff = storeForecastCovariance ? result.forecastErrorCovOffset(t) : 0;
            int stdForeErrOff = storeStandardizedForecastError ? result.standardizedForecastErrorOffset(t) : 0;
            int nextPredOff = storePredictedState ? result.predictedStateOffset(t + 1) : rollingPredictedStateBase + ((t + 1) & 1) * kStates;
            int nextPredCovOff = storePredictedCovariance ? result.predictedStateCovOffset(t + 1) : rollingPredictedStateCovBase + ((t + 1) & 1) * kStates2;
            int kgOff = storeGain ? result.kalmanGainOffset(t) : 0;
            int endogOff = endogOverride != null ? endogBase + t * kEndog : observationModel.endogOffset(t);

            int ke = selectObservations(observationModel, pool, t, endog, endogOff);
            boolean fullyObserved = ke == kEndog;
            boolean useConverged = steadyStateEligible && converged && fullyObserved && previousFullyObserved;
            if (converged && !fullyObserved) {
                converged = false;
            }

            if (ke == 0) {
                if (storeForecastMean) {
                    for (int i = 0; i < kEndog; i++) {
                        forecast[foreOff + i] = 0.0;
                    }
                }
                if (storeForecastError) {
                    for (int i = 0; i < kEndog; i++) {
                        forecastError[foreErrOff + i] = 0.0;
                    }
                }
                if (storeForecastCovariance) {
                    for (int i = 0; i < kEndog2; i++) {
                        forecastErrorCov[foreErrCovOff + i] = 0.0;
                    }
                }
                if (storeStandardizedForecastError) {
                    for (int i = 0; i < kEndog; i++) {
                        standardizedForecastError[stdForeErrOff + i] = 0.0;
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
            double[] forecastMean = storeForecastMean ? forecast : scratch;
            int forecastMeanOff = storeForecastMean ? foreOff : tmpVBase;
            BLAS.dcopy(ke, d, dOff, 1, forecastMean, forecastMeanOff, 1);
            BLAS.dgemv(BLAS.Trans.NoTrans, ke, kStates, 1.0,
                    Z, zOff, zLd,
                predictedState, predOff, 1,
                    1.0, forecastMean, forecastMeanOff, 1);

            if (storeForecastError) {
                BLAS.dcopy(ke, ke < kEndog ? pool.selectionBacking : endog,
                    ke < kEndog ? pool.selectedEndogBase() : endogOff, 1,
                    forecastError, foreErrOff, 1);
                BLAS.daxpy(ke, -1.0, forecastMean, forecastMeanOff, 1,
                    forecastError, foreErrOff, 1);
            } else {
                BLAS.dcopy(ke, ke < kEndog ? pool.selectionBacking : endog,
                    ke < kEndog ? pool.selectedEndogBase() : endogOff, 1,
                        scratch, tmpKBase, 1);
                BLAS.daxpy(ke, -1.0, forecastMean, forecastMeanOff, 1, scratch, tmpKBase, 1);
                BLAS.dcopy(ke, scratch, tmpKBase, 1, scratch, tmpVBase, 1);
            }

            // Innovation covariance and cross term:
            // M_t = P_t Z_t'
            // F_t = Z_t P_t Z_t' + H_t
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, ke, kStates,
                    1.0, predictedStateCov, predCovOff, kStates,
                    Z, zOff, zLd,
                    0.0, scratch, tmpPZtBase, ke);

            if (!useConverged) {
                if (storeForecastCovariance) {
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
                if (storeForecastCovariance) {
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
            if (!storeForecastError && ke == 1) {
                savedForecastError = scratch[tmpVBase];
            }

            if (storeForecastCovariance) {
                BLAS.dcopy(ke2, forecastErrorCov, foreErrCovOff, 1, scratch, tmpFcopyBase, 1);
            }

            if (ke == 1) {
                double Ft = scratch[tmpFcopyBase];
                double error = storeForecastError ? forecastError[foreErrOff] : savedForecastError;
                if (storeStandardizedForecastError) {
                    standardizedForecastError[stdForeErrOff] = ForecastErrorSolver.standardizeScalar(error, Ft, 0.0);
                }
                ForecastErrorSolver.solveScalar(error,
                    Ft,
                    predictedState,
                    predOff,
                    predictedStateCov,
                    predCovOff,
                    scratch,
                    tmpPZtBase,
                    kStates,
                    transition,
                    stateSpace.transitionOffset(t),
                    transitionLd,
                    filteredState,
                    filtOff,
                    filteredStateCov,
                    filtCovOff,
                    kalmanGain,
                    kgOff,
                    storeGain,
                    diffuseTolerance,
                    forecastErrorSolve);
                logDetF = forecastErrorSolve.logDeterminant;
                vFv = forecastErrorSolve.quadraticForm;
            } else {
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, kStates, kStates,
                        1.0, Z, zOff, zLd,
                    predictedStateCov, predCovOff, kStates,
                        0.0, scratch, tmpKBase, kStates);
                double[] originalError;
                int originalErrorOffset;
                if (storeForecastError) {
                    originalError = forecastError;
                    originalErrorOffset = foreErrOff;
                    BLAS.dcopy(ke, forecastError, foreErrOff, 1, scratch, tmpVBase, 1);
                } else {
                    originalError = pool.ensureForecastErrorWork(ke);
                    originalErrorOffset = 0;
                    BLAS.dcopy(ke, scratch, tmpVBase, 1, originalError, originalErrorOffset, 1);
                }
                double[] originalCovariance = scratch;
                int originalCovarianceOffset = tmpFcopyBase;
                if (needsCovarianceSnapshot(forecastErrorSequence)) {
                    originalCovariance = pool.ensureForecastErrorCovWork(ke);
                    originalCovarianceOffset = 0;
                    BLAS.dcopy(ke2, scratch, tmpFcopyBase, 1, originalCovariance, originalCovarianceOffset, 1);
                }
                if (storeStandardizedForecastError) {
                    double[] factor = originalCovariance == scratch
                        ? pool.ensureForecastErrorCovWork(ke)
                        : scratch;
                    int factorOffset = originalCovariance == scratch ? 0 : tmpFcopyBase;
                    if (!ForecastErrorSolver.standardizeCholesky(originalCovariance,
                            originalCovarianceOffset,
                            ke,
                            originalError,
                            originalErrorOffset,
                            standardizedForecastError,
                            stdForeErrOff,
                            ke,
                            factor,
                            factorOffset,
                            0.0)) {
                        fillNaN(standardizedForecastError, stdForeErrOff, ke);
                    }
                }
                ForecastErrorSolver.solveMultivariate(forecastErrorSequence,
                    scratch,
                    tmpFcopyBase,
                    ke,
                    originalCovariance,
                    originalCovarianceOffset,
                    scratch,
                    tmpKBase,
                    kStates,
                    kStates,
                    scratch,
                    tmpVBase,
                    originalError,
                    originalErrorOffset,
                    needsLu(forecastErrorSequence) ? pool.ensureForecastErrorPivots(ke) : null,
                    diffuseTolerance,
                    forecastErrorSolve);
                logDetF = forecastErrorSolve.logDeterminant;
                vFv = forecastErrorSolve.quadraticForm;

                BLAS.dcopy(kStates, predictedState, predOff, 1,
                    filteredState, filtOff, 1);
                BLAS.dgemv(BLAS.Trans.NoTrans, kStates, ke, 1.0,
                    scratch, tmpPZtBase, ke,
                        scratch, tmpVBase, 1,
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

            double logLikelihood = -0.5 * (ke * Math.log(2 * Math.PI) + logDetF);
            if (!concentratedLikelihood) {
                logLikelihood += -0.5 * vFv;
            }
            if (storeLikelihood) {
                result.logLikelihoodObs[logLikelihoodObsBase + t] = logLikelihood;
            } else {
                logLikelihoodSum += logLikelihood;
            }
            if (concentratedLikelihood) {
                concentratedTotalScaleSum += vFv;
                concentratedTotalObservationCount += ke;
                if (t >= concentratedScaleStart) {
                    concentratedScaleSum += vFv;
                    concentratedScaleObservationCount += ke;
                }
                if (storeLikelihood) {
                    result.scaleObs[t] = vFv;
                    result.scaleObsCount[t] = ke;
                }
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
                    if (maxDiff < convergenceTolerance) {
                        converged = true;
                        periodConverged = t + 1;
                        pool.ensureConvergedSnapshots(kStates, kEndog);
                        if (storeForecastCovariance) {
                            BLAS.dcopy(ke2, forecastErrorCov, foreErrCovOff, 1,
                                    pool.convergedForecastErrorCov, 0, 1);
                        } else {
                            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, ke, kStates,
                                1.0, predictedStateCov, predCovOff, kStates,
                                Z, zOff, zLd,
                                0.0, scratch, tmpPZtBase, ke);
                            BLAS.dcopy(ke2, H, hOff, 1,
                                pool.convergedForecastErrorCov, 0, 1);
                            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, kStates,
                                1.0, Z, zOff, zLd,
                                scratch, tmpPZtBase, ke,
                                1.0, pool.convergedForecastErrorCov, 0, ke);
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

            int foreOff = storeForecastMean ? result.forecastOffset(t) : 0;
            int foreErrOff = storeForecastError ? result.forecastErrorOffset(t) : 0;
            int foreErrCovOff = storeForecastCovariance ? result.forecastErrorCovOffset(t) : 0;
            int stdForeErrOff = storeStandardizedForecastError ? result.standardizedForecastErrorOffset(t) : 0;
            int kgOff = result.kalmanGainOffset(t);
            int predOff = result.predictedStateOffset(t);
            int predCovOff = result.predictedStateCovOffset(t);

            if (storeForecastMean) {
                BLAS.dcopy(kEndog, obsIntercept, observationModel.obsInterceptOffset(t), 1,
                        result.forecast, foreOff, 1);
                BLAS.dgemv(BLAS.Trans.NoTrans, kEndog, kStates, 1.0,
                        design, observationModel.designOffset(t), designLd,
                        result.predictedState, predOff, 1,
                        1.0, result.forecast, foreOff, 1);
            }

            if (storeForecastError) {
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
            }

            if (storeForecastCovariance) {
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

            if (storeStandardizedForecastError) {
                if (observationModel.missingCount(t) < kEndog) {
                    int ke = selectObservations(observationModel, pool, t, endog,
                            endogOverride != null ? endogBase + t * kEndog : observationModel.endogOffset(t));
                    int[] selectedIndex = pool.selectedIndex;
                    BLAS.dcopy(ke, result.standardizedForecastError, stdForeErrOff, 1, scratch, tmpVBase, 1);
                    for (int i = 0; i < kEndog; i++) result.standardizedForecastError[stdForeErrOff + i] = 0.0;
                    for (int j = 0; j < ke; j++) {
                        result.standardizedForecastError[stdForeErrOff + selectedIndex[j]] = scratch[tmpVBase + j];
                    }
                } else {
                    for (int i = 0; i < kEndog; i++) result.standardizedForecastError[stdForeErrOff + i] = 0.0;
                }
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

        if (concentratedLikelihood) {
            double scale = concentratedScaleObservationCount == 0
                ? 1.0
                : concentratedScaleSum / concentratedScaleObservationCount;
            result.scale = scale;
            result.scaleObservationCount = concentratedScaleObservationCount;
            if (!storeLikelihood) {
                logLikelihoodSum += -0.5 * (concentratedTotalObservationCount * Math.log(scale)
                    + concentratedTotalScaleSum / scale);
            }
            applyConcentratedScale(observationModel, result, scale, storeLikelihood);
        }

        if (!storeLikelihood) {
            result.setLogLikelihood(logLikelihoodSum);
        }

        result.applyConserveMemory(conserveMemory);

        result.converged = converged;
        result.periodConverged = periodConverged;
        if (reuseResult) {
            pool.releaseConvergedSnapshots();
        }
        return result;
    }

    private static void requireRealInputs(KalmanSSM model) {
        KalmanSSMSupport.requireRealStorage(model, "model");
    }

    private static void fillNaN(double[] values, int offset, int length) {
        for (int i = 0; i < length; i++) {
            values[offset + i] = Double.NaN;
        }
    }

    static void applyConcentratedScale(KalmanSSM observationModel,
                                       FilterResultBase<?> result,
                                       double scale,
                                       boolean storeLikelihood) {
        if (storeLikelihood) {
            double logScale = Math.log(scale);
            for (int t = 0; t < result.nobs; t++) {
                int observed = result.scaleObsCountLength() > 0
                    ? result.scaleObsCount(t)
                    : result.kEndog - observationModel.missingCount(t);
                double quadratic = result.scaleObs(t);
                result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] +=
                    -0.5 * (observed * logScale + quadratic / scale);
            }
        }
        scaleSurface(result.predictedStateCov, result.predictedStateCovBase(), result.predictedStateCovLength(), scale);
        scaleSurface(result.filteredStateCov, result.filteredStateCovBase(), result.filteredStateCovLength(), scale);
        scaleSurface(result.forecastErrorCov, result.forecastErrorCovBase(), result.forecastErrorCovLength(), scale);
        if (result.standardizedForecastErrorLength() > 0) {
            double standardizationScale = 1.0 / Math.sqrt(scale);
            BLAS.dscal(result.standardizedForecastErrorLength(), standardizationScale,
                result.standardizedForecastError, result.standardizedForecastErrorBase(), 1);
        }
    }

    private static void scaleSurface(double[] values, int base, int length, double scale) {
        if (values == null || length == 0) {
            return;
        }
        BLAS.dscal(length, scale, values, base, 1);
    }

    private static boolean needsLu(ForecastErrorSolver.Method[] sequence) {
        if (sequence == null) {
            return false;
        }
        for (ForecastErrorSolver.Method method : sequence) {
            if (method == ForecastErrorSolver.Method.LU_SOLVE) {
                return true;
            }
        }
        return false;
    }

    private static boolean needsCovarianceSnapshot(ForecastErrorSolver.Method[] sequence) {
        if (sequence == null || sequence.length <= 1) {
            return false;
        }
        int factorizingMethods = 0;
        for (ForecastErrorSolver.Method method : sequence) {
            ForecastErrorSolver.Method actual = method == ForecastErrorSolver.Method.AUTO ? ForecastErrorSolver.Method.CHOLESKY_SOLVE : method;
            if (actual == ForecastErrorSolver.Method.CHOLESKY_SOLVE || actual == ForecastErrorSolver.Method.LU_SOLVE) {
                factorizingMethods++;
            }
        }
        return factorizingMethods > 1;
    }

    private static FilterResult filterDiffuse(KalmanSSM model,
                                              InitialState init,
                                              Pool pool,
                                              FilterResultLayout retainedLayout,
                                              FilterDiffuseLayout retainedDiffuseLayout,
                                              int conserveMemory,
                                              double diffuseTolerance,
                                              boolean concentratedLikelihood,
                                              int concentratedLikelihoodBurn,
                                              boolean reuseResult,
                                              double[] endogOverride,
                                              int endogBase) {
        KalmanSSM stateSpace = model;
        KalmanSSM observationModel = model;
        if (pool.diffusePool == null) {
            pool.diffusePool = new UnivariateFilter.Pool();
        }

        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int nobs = observationModel.observationCount();

        boolean storePredictedState = retainedLayout.predictedStateLength() > 0;
        boolean storePredictedCovariance = retainedLayout.predictedStateCovLength() > 0;
        boolean storeFilteredState = retainedLayout.filteredStateLength() > 0;
        boolean storeFilteredCovariance = retainedLayout.filteredStateCovLength() > 0;
        FilterDiffuseLayout diffuseLayout = retainedDiffuseLayout;

        boolean usePredictedScratch = !storePredictedState || !storePredictedCovariance;
        boolean useFilteredScratch = !storeFilteredState || !storeFilteredCovariance;
        if (usePredictedScratch || useFilteredScratch) {
            pool.ensurePooledScratch(kStates, kEndog, stateSpace.stateDisturbanceCount(),
                    usePredictedScratch, useFilteredScratch);
        }
        if (usePredictedScratch) {
            pool.ensureDiffusePredictedScratch(kStates, kEndog, stateSpace.stateDisturbanceCount());
        }

        FilterResult result;
        if (reuseResult) {
            result = pool.borrowDiffuseResult(kEndog, kStates, nobs, retainedLayout, diffuseLayout);
            result.clearActiveSurfaces();
        } else {
            result = new FilterResult(kEndog, kStates, nobs, retainedLayout);
            result.allocateDiffuse(diffuseLayout.predictedDiffuseStateCovLength() > 0,
                diffuseLayout.forecastErrorDiffuseCovLength() > 0);
        }

        UnivariateFilter.filterDiffuse(model, init, pool.diffusePool, result,
            diffuseTolerance,
            concentratedLikelihood,
            concentratedLikelihoodBurn,
            endogOverride, endogBase,
            storePredictedState ? null : pool.scratchBacking,
            storePredictedState ? 0 : pool.rollingPredictedStateBase(),
            storePredictedCovariance ? null : pool.scratchBacking,
            storePredictedCovariance ? 0 : pool.rollingPredictedStateCovBase(),
            usePredictedScratch ? pool.scratchBacking : null,
            usePredictedScratch ? pool.rollingPredictedDiffuseStateCovBase() : 0,
            storeFilteredState ? null : pool.scratchBacking,
            storeFilteredState ? 0 : pool.scratchFilteredStateBase(),
            storeFilteredCovariance ? null : pool.scratchBacking,
            storeFilteredCovariance ? 0 : pool.scratchFilteredStateCovBase());
        result.converged = false;
        result.periodConverged = nobs;
        if (reuseResult) {
            result.applyConserveMemory(conserveMemory);
            pool.releaseConvergedSnapshots();
        }
        return result;
    }

    private static int selectObservations(KalmanSSM observationModel,
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
