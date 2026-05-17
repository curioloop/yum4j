package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.arena.FilterSelectionLayout;
import com.curioloop.yum4j.ssm.kalman.arena.FilterDiffuseLayout;
import com.curioloop.yum4j.ssm.kalman.arena.FilterResultLayout;
import com.curioloop.yum4j.ssm.kalman.arena.IntArena;
import com.curioloop.yum4j.ssm.kalman.arena.ZFilterScratchLayout;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;

import java.util.Arrays;
import java.util.Objects;

public final class ZKalmanFilter {

    static final class Pool {
        private final DoubleArena scratchArena;
        private final DoubleArena resultArena;
        private final IntArena selectedIndexArena = new IntArena();
        private final boolean ownsScratchArena;
        private final boolean ownsResultArena;
        public int[] selectedIndex;
        public double[] convergedForecastErrorCov;
        public double[] convergedFilteredStateCov;
        public double[] convergedPredictedStateCov;
        public double[] convergedKalmanGain;
        public double[] scratchBacking;
        public double[] selectionBacking;
        public double[] resultBacking;
        public double[] diffuseResultBacking;
        public ZFilterScratchLayout scratchLayout;
        public FilterSelectionLayout selectionLayout;
        public FilterResultLayout resultLayout;
        public FilterDiffuseLayout diffuseResultLayout;
        private int diffuseResultBase;

        private ZFilterResult pooledResult;
        private InitialState.StationaryWorkspace stationaryWorkspace;
        private ObservationTransformWorkspace observationTransformWorkspace;
        private int scratchKStates;
        private int scratchKEndog;
        private int scratchKPosdef;
        private boolean scratchPredicted;
        private boolean scratchFiltered;
        private boolean scratchDiffusePredicted;
        private int scratchMode = -1;
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
            ensureConventionalScratch(kStates, kEndog, kPosdef, false, false, false);
        }

        void ensureSelection(int kStates, int kEndog) {
            int nextSelectionKStates = Math.max(selectionKStates, kStates);
            int nextSelectionKEndog = Math.max(selectionKEndog, kEndog);
            selectedIndex = selectedIndexArena.ensureCapacity(kEndog);
            if (selectionLayout == null
                    || nextSelectionKStates != selectionKStates
                    || nextSelectionKEndog != selectionKEndog) {
                selectionLayout = FilterSelectionLayout.create(2, nextSelectionKEndog, nextSelectionKStates);
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
            int ksKe = 2 * kStates * kEndog;
            int ksKs = 2 * kStates * kStates;
            int keKe = 2 * kEndog * kEndog;
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

        void ensurePooledScratch(int kStates,
                                 int kEndog,
                                 int kPosdef,
                                 boolean needPredictedScratch,
                                 boolean needFilteredScratch) {
            ensureConventionalScratch(kStates, kEndog, kPosdef,
                needPredictedScratch, needFilteredScratch, false);
        }

        void ensureDiffusePredictedScratch(int kStates, int kEndog, int kPosdef) {
            ensureExactDiffuseScratch(kStates, kEndog, kPosdef, false, false, true);
        }

        void ensureConventionalScratch(int kStates,
                                       int kEndog,
                                       int kPosdef,
                                       boolean needPredictedScratch,
                                       boolean needFilteredScratch,
                                       boolean needDiffusePredictedScratch) {
            ensureScratchArena(ZFilterScratchLayout.MODE_CONVENTIONAL,
                kStates, kEndog, kPosdef,
                needPredictedScratch, needFilteredScratch, needDiffusePredictedScratch);
        }

        void ensureUnivariateScratch(int kStates,
                                     int kEndog,
                                     int kPosdef,
                                     boolean needPredictedScratch,
                                     boolean needFilteredScratch) {
            ensureScratchArena(ZFilterScratchLayout.MODE_UNIVARIATE,
                kStates, kEndog, kPosdef,
                needPredictedScratch, needFilteredScratch, false);
        }

        void ensureExactDiffuseScratch(int kStates,
                                       int kEndog,
                                       int kPosdef,
                                       boolean needPredictedScratch,
                                       boolean needFilteredScratch,
                                       boolean needDiffusePredictedScratch) {
            ensureScratchArena(ZFilterScratchLayout.MODE_EXACT_DIFFUSE,
                kStates, kEndog, kPosdef,
                needPredictedScratch, needFilteredScratch, needDiffusePredictedScratch);
        }

        private void ensureScratchArena(int mode,
                                        int kStates,
                                        int kEndog,
                                        int kPosdef,
                                        boolean needPredictedScratch,
                                        boolean needFilteredScratch,
                                        boolean needDiffusePredictedScratch) {
            int nextScratchKStates = Math.max(scratchKStates, kStates);
            int nextScratchKEndog = Math.max(scratchKEndog, kEndog);
            int nextScratchKPosdef = Math.max(scratchKPosdef, kPosdef);
            boolean nextScratchPredicted = needPredictedScratch;
            boolean nextScratchFiltered = needFilteredScratch;
            boolean nextScratchDiffusePredicted = needDiffusePredictedScratch;
            if (scratchLayout == null
                    || nextScratchKStates != scratchKStates
                    || nextScratchKEndog != scratchKEndog
                    || nextScratchKPosdef != scratchKPosdef
                    || nextScratchPredicted != scratchPredicted
                    || nextScratchFiltered != scratchFiltered
                    || nextScratchDiffusePredicted != scratchDiffusePredicted
                    || mode != scratchMode) {
                scratchLayout = switch (mode) {
                    case ZFilterScratchLayout.MODE_CONVENTIONAL -> ZFilterScratchLayout.createConventional(
                        nextScratchKEndog,
                        nextScratchKStates,
                        nextScratchKPosdef,
                        nextScratchPredicted,
                        nextScratchFiltered,
                        nextScratchDiffusePredicted);
                    case ZFilterScratchLayout.MODE_UNIVARIATE -> ZFilterScratchLayout.createUnivariate(
                        nextScratchKEndog,
                        nextScratchKStates,
                        nextScratchKPosdef,
                        nextScratchPredicted,
                        nextScratchFiltered);
                    case ZFilterScratchLayout.MODE_EXACT_DIFFUSE -> ZFilterScratchLayout.createExactDiffuse(
                        nextScratchKEndog,
                        nextScratchKStates,
                        nextScratchKPosdef,
                        nextScratchPredicted,
                        nextScratchFiltered,
                        nextScratchDiffusePredicted);
                    default -> throw new IllegalArgumentException("Unknown complex filter scratch mode: " + mode);
                };
                scratchKStates = nextScratchKStates;
                scratchKEndog = nextScratchKEndog;
                scratchKPosdef = nextScratchKPosdef;
                scratchPredicted = nextScratchPredicted;
                scratchFiltered = nextScratchFiltered;
                scratchDiffusePredicted = nextScratchDiffusePredicted;
                scratchMode = mode;
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

        ZFilterResult borrowResult(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
            ensureResult(layout);
            if (pooledResult == null) {
                pooledResult = new ZFilterResult();
            }
            return pooledResult.reuse(kEndog, kStates, nobs,
                    resultBacking, resultLayout);
        }

        ZFilterResult borrowDiffuseResult(int kEndog,
                                          int kStates,
                                          int nobs,
                                          FilterResultLayout layout,
                                          FilterDiffuseLayout diffuseLayout) {
            ensureResult(layout);
            ensureDiffuseResult(diffuseLayout);
            if (pooledResult == null) {
                pooledResult = new ZFilterResult();
            }
            return pooledResult
                .reuse(kEndog, kStates, nobs, resultBacking, resultLayout)
                .reuseDiffuse(diffuseResultBacking, diffuseResultBase, diffuseResultLayout);
        }

        public void reserve(KalmanSSM model,
                            FilterOptions spec) {
            requireComplexInputs(model);
            FilterOptions resolvedOptions = spec == null ? FilterOptions.defaults() : spec;
            FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(
                2,
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
                    + retainedObservationTransformDoubleCount();
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

        long retainedObservationTransformDoubleCount() {
            return observationTransformWorkspace == null ? 0L : observationTransformWorkspace.retainedDoubleCount();
        }

        public long retainedResultDoubleCount() {
            return resultArena.retainedDoubleCount();
        }

        public long retainedTotalByteCount() {
            long observationTransformBytes = observationTransformWorkspace == null
                ? 0L
                : observationTransformWorkspace.retainedByteCount();
            return (retainedScratchDoubleCount() - retainedObservationTransformDoubleCount()
                + retainedResultDoubleCount()) * Double.BYTES
                + observationTransformBytes;
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
            selectedIndex = null;
            selectedIndexArena.release();
            releaseConvergedSnapshots();
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
            if (observationTransformWorkspace != null) {
                releaseObservationTransformWorkspace();
            }
            scratchKStates = 0;
            scratchKEndog = 0;
            scratchKPosdef = 0;
            scratchPredicted = false;
            scratchFiltered = false;
            scratchDiffusePredicted = false;
            scratchMode = -1;
            selectionKStates = 0;
            selectionKEndog = 0;
            selectionBase = 0;
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }

        private static long intArenaDoubleEquivalentCount(IntArena arena) {
            return arena.retainedIntCount() * ((long) Integer.BYTES) / Double.BYTES;
        }

        void ensureStationary(KalmanSSM model) {
            if (stationaryWorkspace == null) {
                stationaryWorkspace = new InitialState.StationaryWorkspace();
            }
            stationaryWorkspace.ensureCapacity(model);
        }

        ObservationTransformWorkspace observationTransformWorkspace() {
            if (observationTransformWorkspace == null) {
                observationTransformWorkspace = new ObservationTransformWorkspace();
            }
            return observationTransformWorkspace;
        }

        void releaseObservationTransformWorkspace() {
            if (observationTransformWorkspace != null) {
                observationTransformWorkspace.release();
                observationTransformWorkspace = null;
            }
        }

        int tmpABase() { return scratchLayout.tmpABase(); }
        int tmpKBase() { return scratchLayout.tmpKBase(); }
        int tmpPZtBase() { return scratchLayout.tmpPZtBase(); }
        int tmpFcopyBase() { return scratchLayout.tmpFcopyBase(); }
        int tmpTPBase() { return scratchLayout.tmpTPBase(); }
        int tmpVBase() { return scratchLayout.tmpVBase(); }
        int tmpRQBase() { return scratchLayout.tmpRQBase(); }
        int tmpPInfBase() { return scratchLayout.tmpPInfBase(); }
        int tmpPStarBase() { return scratchLayout.tmpPStarBase(); }
        int tmpMStarBase() { return scratchLayout.tmpMStarBase(); }
        int tmpMInfBase() { return scratchLayout.tmpMInfBase(); }
        int tmpK0Base() { return scratchLayout.tmpK0Base(); }
        int tmpK1Base() { return scratchLayout.tmpK1Base(); }
        int rollingPredictedStateBase() { return scratchLayout.rollingPredictedStateBase(); }
        int rollingPredictedStateCovBase() { return scratchLayout.rollingPredictedStateCovBase(); }
        int rollingPredictedDiffuseStateCovBase() { return scratchLayout.rollingPredictedDiffuseStateCovBase(); }
        int scratchFilteredStateBase() { return scratchLayout.scratchFilteredStateBase(); }
        int scratchFilteredStateCovBase() { return scratchLayout.scratchFilteredStateCovBase(); }
        int selectedDesignBase() { return selectionBase + selectionLayout.selectedDesignBase(); }
        int selectedObsCovBase() { return selectionBase + selectionLayout.selectedObsCovBase(); }
        int selectedObsInterceptBase() { return selectionBase + selectionLayout.selectedObsInterceptBase(); }
        int selectedEndogBase() { return selectionBase + selectionLayout.selectedEndogBase(); }
    }

    private static void requireComplexInputs(KalmanSSM model) {
        KalmanSSMSupport.requireComplexStorage(model, "model");
    }

    static ZFilterResult filter(KalmanSSM model,
                                InitialState init,
                                Pool pool,
                                FilterOptions spec) {
        requireComplexInputs(model);
        Objects.requireNonNull(init, "init");
        FilterOptions resolvedOptions = spec == null ? FilterOptions.defaults() : spec;
        FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(
            2,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            KalmanSSMSupport.hasMissingObservations(model));
        FilterDiffuseLayout retainedDiffuseLayout = resolvedOptions.createDiffuseLayout(
            2,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            KalmanSSMSupport.hasMissingObservations(model));
        return filterInternal(model, init, pool,
            retainedLayout,
            retainedDiffuseLayout,
            resolvedOptions.stabilityMask(),
            resolvedOptions.conserveMemoryMask(),
            resolvedOptions.diffuseTolerance(),
            resolvedOptions.convergenceTolerance(),
            resolvedOptions.concentratedLikelihood(),
            resolvedOptions.concentratedLikelihoodBurn(),
            pool != null,
            null,
            0);
    }

    static ZFilterResult filter(KalmanSSM model,
                                InitialState init,
                                double[] endog,
                                int endogBase,
                                Pool pool,
                                FilterOptions spec) {
        requireComplexInputs(model);
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(endog, "endog");
        FilterOptions resolvedOptions = spec == null ? FilterOptions.defaults() : spec;
        FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(
            2,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            KalmanSSMSupport.hasMissingObservations(model));
        FilterDiffuseLayout retainedDiffuseLayout = resolvedOptions.createDiffuseLayout(
            2,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            KalmanSSMSupport.hasMissingObservations(model));
        return filterInternal(model, init, pool,
            retainedLayout,
            retainedDiffuseLayout,
            resolvedOptions.stabilityMask(),
            resolvedOptions.conserveMemoryMask(),
            resolvedOptions.diffuseTolerance(),
            resolvedOptions.convergenceTolerance(),
            resolvedOptions.concentratedLikelihood(),
            resolvedOptions.concentratedLikelihoodBurn(),
            pool != null,
            endog,
            endogBase);
    }

    static ZFilterResult filterUnivariate(KalmanSSM model,
                                          InitialState init,
                                          Pool pool,
                                          FilterOptions spec) {
        requireComplexInputs(model);
        Objects.requireNonNull(init, "init");
        FilterOptions resolvedOptions = spec == null ? FilterOptions.defaults() : spec;
        FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(
            2,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            KalmanSSMSupport.hasMissingObservations(model));
        FilterDiffuseLayout retainedDiffuseLayout = resolvedOptions.createDiffuseLayout(
            2,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            KalmanSSMSupport.hasMissingObservations(model));
        ZFilterResult result = filterUnivariateInternal(model, init, pool,
            retainedLayout,
            retainedDiffuseLayout,
            resolvedOptions.stabilityMask(),
            resolvedOptions.diffuseTolerance(),
            resolvedOptions.convergenceTolerance(),
            resolvedOptions.concentratedLikelihood(),
            resolvedOptions.concentratedLikelihoodBurn(),
            pool != null,
            null,
            0);
        result.applyConserveMemory(resolvedOptions.conserveMemoryMask());
        return result;
    }

    static ZFilterResult filterUnivariate(KalmanSSM model,
                                          InitialState init,
                                          double[] endog,
                                          int endogBase,
                                          Pool pool,
                                          FilterOptions spec) {
        requireComplexInputs(model);
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(endog, "endog");
        FilterOptions resolvedOptions = spec == null ? FilterOptions.defaults() : spec;
        FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(
            2,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            KalmanSSMSupport.hasMissingObservations(model));
        FilterDiffuseLayout retainedDiffuseLayout = resolvedOptions.createDiffuseLayout(
            2,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            KalmanSSMSupport.hasMissingObservations(model));
        ZFilterResult result = filterUnivariateInternal(model, init, pool,
            retainedLayout,
            retainedDiffuseLayout,
            resolvedOptions.stabilityMask(),
            resolvedOptions.diffuseTolerance(),
            resolvedOptions.convergenceTolerance(),
            resolvedOptions.concentratedLikelihood(),
            resolvedOptions.concentratedLikelihoodBurn(),
            pool != null,
            endog,
            endogBase);
        result.applyConserveMemory(resolvedOptions.conserveMemoryMask());
        return result;
    }

    private static ZFilterResult filterUnivariateInternal(KalmanSSM model,
                                                          InitialState init,
                                                          Pool pool,
                                                          FilterResultLayout retainedLayout,
                                                          FilterDiffuseLayout retainedDiffuseLayout,
                                                          int stabilityMethod,
                                                          double diffuseTolerance,
                                                          double convergenceTolerance,
                                                          boolean concentratedLikelihood,
                                                          int concentratedLikelihoodBurn,
                                                          boolean reuseResult,
                                                          double[] endogOverride,
                                                          int endogBase) {
        if (pool == null) pool = new Pool();
        boolean transformed = needsObservationTransform(model);
        try {
            model = prepareUnivariateObservationModel(model, pool, endogOverride, endogBase);
            endogOverride = null;
            endogBase = 0;
            KalmanSSM stateSpace = model;
        KalmanSSM observationModel = model;
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int ksKs = kStates * 2 * kStates;
        int keKe = kEndog * 2 * kEndog;
        if (init.mayResolveDiffuse()) {
            return filterDiffuse(model, init, pool, kEndog, kStates, kPosdef, nobs, ksKs, keKe,
                retainedLayout, retainedDiffuseLayout, stabilityMethod, 0, diffuseTolerance,
                concentratedLikelihood, concentratedLikelihoodBurn, reuseResult, endogOverride, endogBase);
        }
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
        int obsCovLd = observationModel.obsCovLeadingDimension();
        double[] endog = endogOverride != null ? endogOverride : observationModel.endogData();

        if (pool == null) pool = new Pool();
        boolean storeForecastMean = retainedLayout.forecastLength() > 0;
        boolean storeForecastError = retainedLayout.forecastErrorLength() > 0;
        boolean storeForecastCovariance = retainedLayout.forecastErrorCovLength() > 0;
        boolean storeStandardizedForecastError = retainedLayout.standardizedForecastErrorLength() > 0;
        boolean storeForecast = storeForecastMean || storeForecastError || storeForecastCovariance || storeStandardizedForecastError;
        boolean storePredictedState = retainedLayout.predictedStateLength() > 0;
        boolean storePredictedCovariance = retainedLayout.predictedStateCovLength() > 0;
        boolean storeFilteredState = retainedLayout.filteredStateLength() > 0;
        boolean storeFilteredCovariance = retainedLayout.filteredStateCovLength() > 0;
        boolean storeLikelihood = retainedLayout.logLikelihoodObsLength() > 0;
        boolean storeGain = retainedLayout.kalmanGainLength() > 0;
        boolean usePredictedScratch = !storePredictedState || !storePredictedCovariance;
        boolean useFilteredScratch = !storeFilteredState || !storeFilteredCovariance;
        pool.ensureUnivariateScratch(kStates, kEndog, kPosdef,
            usePredictedScratch,
            useFilteredScratch);
        double[] scratch = pool.scratchBacking;

        int tmpABase = pool.tmpABase();
        int tmpPBase = pool.tmpPStarBase();
        int tmpMBase = pool.tmpMStarBase();
        int tmpTPBase = pool.tmpTPBase();
        int tmpRQBase = pool.tmpRQBase();
        int rollingPredictedStateBase = usePredictedScratch ? pool.rollingPredictedStateBase() : 0;
        int rollingPredictedStateCovBase = usePredictedScratch ? pool.rollingPredictedStateCovBase() : 0;
        int scratchFilteredStateBase = useFilteredScratch ? pool.scratchFilteredStateBase() : 0;
        int scratchFilteredStateCovBase = useFilteredScratch ? pool.scratchFilteredStateCovBase() : 0;

        InitialState.StationaryWorkspace stationaryWorkspace = null;
        if (init.mayResolveStationary()) {
            pool.ensureStationary(model);
            stationaryWorkspace = pool.stationaryWorkspace;
        }

        ZFilterResult result = reuseResult
            ? pool.borrowResult(kEndog, kStates, nobs, retainedLayout)
            : new ZFilterResult(kEndog, kStates, nobs, retainedLayout);
        result.perObsKalmanGain = true;
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
        int predictedStateBase = storePredictedState ? result.predictedStateBase() : rollingPredictedStateBase;
        int predictedStateCovBase = storePredictedCovariance ? result.predictedStateCovBase() : rollingPredictedStateCovBase;
        int filteredStateBase = storeFilteredState ? result.filteredStateBase() : scratchFilteredStateBase;
        int filteredStateCovBase = storeFilteredCovariance ? result.filteredStateCovBase() : scratchFilteredStateCovBase;

        init.resolveInto(model, 0, stationaryWorkspace,
            predictedState, predictedStateBase,
            predictedStateCov, predictedStateCovBase,
            null, 0);

        double concentratedScaleSum = 0.0;
        int concentratedScaleObservationCount = 0;
        int concentratedScaleStart = Math.max(0, concentratedLikelihoodBurn);

        for (int period = 0; period < nobs; period++) {
            int predOff = storePredictedState
                ? result.predictedStateOffset(period)
                : rollingPredictedStateBase + (period & 1) * 2 * kStates;
            int predCovOff = storePredictedCovariance
                ? result.predictedStateCovOffset(period)
                : rollingPredictedStateCovBase + (period & 1) * ksKs;
            int filtOff = storeFilteredState ? result.filteredStateOffset(period) : filteredStateBase;
            int filtCovOff = storeFilteredCovariance ? result.filteredStateCovOffset(period) : filteredStateCovBase;
            int foreOff = storeForecastMean ? result.forecastOffset(period) : 0;
            int foreErrOff = storeForecastError ? result.forecastErrorOffset(period) : 0;
            int foreErrCovOff = storeForecastCovariance ? result.forecastErrorCovOffset(period) : 0;
            int stdForeErrOff = storeStandardizedForecastError ? result.standardizedForecastErrorOffset(period) : 0;
            int nextPredOff = storePredictedState
                ? result.predictedStateOffset(period + 1)
                : rollingPredictedStateBase + ((period + 1) & 1) * 2 * kStates;
            int nextPredCovOff = storePredictedCovariance
                ? result.predictedStateCovOffset(period + 1)
                : rollingPredictedStateCovBase + ((period + 1) & 1) * ksKs;
            int kgOff = storeGain ? result.kalmanGainOffset(period) : 0;
            int endogOff = endogOverride != null
                ? endogBase + period * 2 * kEndog
                : observationModel.endogOffset(period);

            ZLAS.zcopy(kStates, predictedState, predOff, 1, scratch, tmpABase, 1);
            ZLAS.zcopy(kStates * kStates, predictedStateCov, predCovOff, 1, scratch, tmpPBase, 1);

            zeroComplex(result.forecast, foreOff, storeForecastMean ? kEndog : 0);
            zeroComplex(result.forecastError, foreErrOff, storeForecastError ? kEndog : 0);
            zeroComplex(result.forecastErrorCov, foreErrCovOff, storeForecastCovariance ? kEndog * kEndog : 0);
            zeroComplex(result.standardizedForecastError, stdForeErrOff,
                storeStandardizedForecastError ? kEndog : 0);
            zeroComplex(result.kalmanGain, kgOff, storeGain ? kStates * kEndog : 0);

            double logLikelihoodRe = 0.0;
            double periodScaleContribution = 0.0;
            int periodScaleObservationCount = 0;

            for (int endogIndex = 0; endogIndex < kEndog; endogIndex++) {
                if (observationModel.isMissing(endogIndex, period)) {
                    continue;
                }
                int designRowOff = observationModel.designOffset(period) + endogIndex * 2 * designLd;
                int obsInterceptOff = observationModel.obsInterceptOffset(period) + endogIndex * 2;
                int obsCovDiagOff = observationModel.obsCovOffset(period) + endogIndex * 2 * obsCovLd + endogIndex * 2;
                int endogValueOff = endogOff + endogIndex * 2;

                double meanRe = obsIntercept[obsInterceptOff];
                double meanIm = obsIntercept[obsInterceptOff + 1];
                for (int state = 0; state < kStates; state++) {
                    int designOff = designRowOff + state * 2;
                    int stateOff = tmpABase + state * 2;
                    double productRe = design[designOff] * scratch[stateOff]
                        - design[designOff + 1] * scratch[stateOff + 1];
                    double productIm = design[designOff] * scratch[stateOff + 1]
                        + design[designOff + 1] * scratch[stateOff];
                    meanRe += productRe;
                    meanIm += productIm;
                }
                double forecastErrorRe = endog[endogValueOff] - meanRe;
                double forecastErrorIm = endog[endogValueOff + 1] - meanIm;
                if (storeForecastMean) {
                    result.forecast[foreOff + endogIndex * 2] = meanRe;
                    result.forecast[foreOff + endogIndex * 2 + 1] = meanIm;
                }
                if (storeForecastError) {
                    result.forecastError[foreErrOff + endogIndex * 2] = forecastErrorRe;
                    result.forecastError[foreErrOff + endogIndex * 2 + 1] = forecastErrorIm;
                }

                for (int state = 0; state < kStates; state++) {
                    double projectionRe = 0.0;
                    double projectionIm = 0.0;
                    for (int innerState = 0; innerState < kStates; innerState++) {
                        int covOff = tmpPBase + state * 2 * kStates + innerState * 2;
                        int designOff = designRowOff + innerState * 2;
                        double designRe = design[designOff];
                        double designIm = design[designOff + 1];
                        projectionRe += scratch[covOff] * designRe - scratch[covOff + 1] * designIm;
                        projectionIm += scratch[covOff] * designIm + scratch[covOff + 1] * designRe;
                    }
                    int projectionOff = tmpMBase + state * 2;
                    scratch[projectionOff] = projectionRe;
                    scratch[projectionOff + 1] = projectionIm;
                }

                double forecastCovRe = obsCov[obsCovDiagOff];
                double forecastCovIm = obsCov[obsCovDiagOff + 1];
                for (int state = 0; state < kStates; state++) {
                    int designOff = designRowOff + state * 2;
                    int projectionOff = tmpMBase + state * 2;
                    forecastCovRe += design[designOff] * scratch[projectionOff]
                        - design[designOff + 1] * scratch[projectionOff + 1];
                    forecastCovIm += design[designOff] * scratch[projectionOff + 1]
                        + design[designOff + 1] * scratch[projectionOff];
                }
                double forecastCovMag2 = forecastCovRe * forecastCovRe + forecastCovIm * forecastCovIm;
                if (forecastCovMag2 <= diffuseTolerance * diffuseTolerance) {
                    continue;
                }
                double forecastCovInvRe = forecastCovRe / forecastCovMag2;
                double forecastCovInvIm = -forecastCovIm / forecastCovMag2;
                if (storeForecastCovariance) {
                    int forecastCovOff = foreErrCovOff + endogIndex * 2 * kEndog + endogIndex * 2;
                    result.forecastErrorCov[forecastCovOff] = forecastCovRe;
                    result.forecastErrorCov[forecastCovOff + 1] = forecastCovIm;
                }
                if (storeStandardizedForecastError) {
                    storeStandardizedScalarComplex(result.standardizedForecastError,
                        stdForeErrOff + endogIndex * 2,
                        forecastErrorRe, forecastErrorIm,
                        forecastCovRe, forecastCovIm);
                }

                double gainScaledErrorRe = forecastCovInvRe * forecastErrorRe - forecastCovInvIm * forecastErrorIm;
                double gainScaledErrorIm = forecastCovInvRe * forecastErrorIm + forecastCovInvIm * forecastErrorRe;
                double quadraticRe = forecastErrorRe * gainScaledErrorRe + forecastErrorIm * gainScaledErrorIm;
                if (storeGain) {
                    for (int state = 0; state < kStates; state++) {
                        int projectionOff = tmpMBase + state * 2;
                        int gainOff = kgOff + state * 2 * kEndog + endogIndex * 2;
                        result.kalmanGain[gainOff] = scratch[projectionOff] * forecastCovInvRe
                            - scratch[projectionOff + 1] * forecastCovInvIm;
                        result.kalmanGain[gainOff + 1] = scratch[projectionOff] * forecastCovInvIm
                            + scratch[projectionOff + 1] * forecastCovInvRe;
                    }
                }

                for (int state = 0; state < kStates; state++) {
                    int projectionOff = tmpMBase + state * 2;
                    int stateOff = tmpABase + state * 2;
                    double updateRe = scratch[projectionOff] * gainScaledErrorRe
                        - scratch[projectionOff + 1] * gainScaledErrorIm;
                    double updateIm = scratch[projectionOff] * gainScaledErrorIm
                        + scratch[projectionOff + 1] * gainScaledErrorRe;
                    scratch[stateOff] += updateRe;
                    scratch[stateOff + 1] += updateIm;
                }

                for (int row = 0; row < kStates; row++) {
                    int projectionOff = tmpMBase + row * 2;
                    double gainRe = scratch[projectionOff] * forecastCovInvRe
                        - scratch[projectionOff + 1] * forecastCovInvIm;
                    double gainIm = scratch[projectionOff] * forecastCovInvIm
                        + scratch[projectionOff + 1] * forecastCovInvRe;
                    for (int col = 0; col < kStates; col++) {
                        int rowProjectionOff = tmpMBase + col * 2;
                        double rowProjectionRe = scratch[rowProjectionOff];
                        double rowProjectionIm = scratch[rowProjectionOff + 1];
                        int covOff = tmpPBase + row * 2 * kStates + col * 2;
                        double reductionRe = gainRe * rowProjectionRe - gainIm * rowProjectionIm;
                        double reductionIm = gainRe * rowProjectionIm + gainIm * rowProjectionRe;
                        scratch[covOff] -= reductionRe;
                        scratch[covOff + 1] -= reductionIm;
                    }
                }
                if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                    symmetrizeComplexSymmetric(scratch, tmpPBase, kStates);
                }

                double logDet = Math.log(Math.sqrt(forecastCovMag2));
                if (concentratedLikelihood) {
                    logLikelihoodRe -= 0.5 * (Math.log(2 * Math.PI) + logDet);
                    periodScaleContribution += quadraticRe;
                    periodScaleObservationCount++;
                } else {
                    logLikelihoodRe -= 0.5 * (Math.log(2 * Math.PI) + logDet + quadraticRe);
                }
            }

            ZLAS.zcopy(kStates, scratch, tmpABase, 1, filteredState, filtOff, 1);
            ZLAS.zcopy(kStates * kStates, scratch, tmpPBase, 1, filteredStateCov, filtCovOff, 1);
            if (storeLikelihood) {
                result.logLikelihoodObs[result.logLikelihoodObsOffset(period)] = logLikelihoodRe;
            }
            if (concentratedLikelihood) {
                if (period >= concentratedScaleStart) {
                    concentratedScaleSum += periodScaleContribution;
                    concentratedScaleObservationCount += periodScaleObservationCount;
                }
                if (storeLikelihood) {
                    result.scaleObs[period] = periodScaleContribution;
                    result.scaleObsCount[period] = periodScaleObservationCount;
                }
            }

            ZLAS.zcopy(kStates, stateIntercept, stateSpace.stateInterceptOffset(period), 1,
                predictedState, nextPredOff, 1);
            ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                transition, stateSpace.transitionOffset(period), transitionLd,
                scratch, tmpABase, 1,
                1.0, 0.0, predictedState, nextPredOff, 1);

            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, 0.0, transition, stateSpace.transitionOffset(period) >> 1, transitionLd,
                scratch, tmpPBase >> 1, kStates,
                0.0, 0.0, scratch, tmpTPBase >> 1, kStates);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                1.0, 0.0, scratch, tmpTPBase >> 1, kStates,
                transition, stateSpace.transitionOffset(period) >> 1, transitionLd,
                0.0, 0.0, predictedStateCov, nextPredCovOff >> 1, kStates);
            if (kPosdef > 0) {
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                    1.0, 0.0, selection, stateSpace.selectionOffset(period) >> 1, selectionLd,
                    stateCovariance, stateSpace.stateCovarianceOffset(period) >> 1, stateCovarianceLd,
                    0.0, 0.0, scratch, tmpRQBase >> 1, kPosdef);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kPosdef,
                    1.0, 0.0, scratch, tmpRQBase >> 1, kPosdef,
                    selection, stateSpace.selectionOffset(period) >> 1, selectionLd,
                    1.0, 0.0, predictedStateCov, nextPredCovOff >> 1, kStates);
            }
            if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                symmetrizeComplexSymmetric(predictedStateCov, nextPredCovOff, kStates);
            }
        }

        if (concentratedLikelihood) {
            double scale = concentratedScaleObservationCount == 0
                ? 1.0
                : concentratedScaleSum / concentratedScaleObservationCount;
            result.scale = scale;
            result.scaleObservationCount = concentratedScaleObservationCount;
            KalmanFilter.applyConcentratedScale(observationModel, result, scale, storeLikelihood);
        }
        result.converged = false;
        result.periodConverged = nobs;
        return result;
        } finally {
            if (reuseResult) {
                pool.releaseConvergedSnapshots();
            }
            if (transformed) {
                pool.releaseObservationTransformWorkspace();
            }
        }
    }

    private static ZFilterResult filterInternal(KalmanSSM model,
                                                InitialState init,
                                                Pool pool,
                            FilterResultLayout retainedLayout,
                                                FilterDiffuseLayout retainedDiffuseLayout,
                                                int stabilityMethod,
                                                int conserveMemory,
                                                double diffuseTolerance,
                                                double convergenceTolerance,
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
        int keKe = kEndog * 2 * kEndog;
        int ksKs = kStates * 2 * kStates;
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
        boolean storeForecast = retainedLayout.forecastLength() > 0;
        boolean storeStandardizedForecastError = retainedLayout.standardizedForecastErrorLength() > 0;
        boolean storePredictedState = retainedLayout.predictedStateLength() > 0;
        boolean storePredictedCovariance = retainedLayout.predictedStateCovLength() > 0;
        boolean storeFilteredState = retainedLayout.filteredStateLength() > 0;
        boolean storeFilteredCovariance = retainedLayout.filteredStateCovLength() > 0;
        boolean storeLikelihood = retainedLayout.logLikelihoodObsLength() > 0;
        boolean storeGain = retainedLayout.kalmanGainLength() > 0;
        boolean steadyStateEligible = model.isTimeInvariant();

        if (init.mayResolveDiffuse()) {
            return filterDiffuse(model, init, pool, kEndog, kStates, kPosdef, nobs, ksKs, keKe,
                    retainedLayout, retainedDiffuseLayout, stabilityMethod, conserveMemory, diffuseTolerance,
                    concentratedLikelihood, concentratedLikelihoodBurn, reuseResult, endogOverride, endogBase);
        }

        boolean usePredictedScratch = !storePredictedState || !storePredictedCovariance;
        boolean useFilteredScratch = !storeFilteredState || !storeFilteredCovariance;
        pool.ensureConventionalScratch(kStates, kEndog, kPosdef,
            usePredictedScratch,
            useFilteredScratch,
            false);
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

        ZFilterResult result = reuseResult
            ? pool.borrowResult(kEndog, kStates, nobs, retainedLayout)
            : new ZFilterResult(kEndog, kStates, nobs, retainedLayout);
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
        double[] kalmanGain = result.kalmanGain;
        int predictedStateBase = storePredictedState ? result.predictedStateBase() : rollingPredictedStateBase;
        int predictedStateCovBase = storePredictedCovariance ? result.predictedStateCovBase() : rollingPredictedStateCovBase;
        int filteredStateBase = storeFilteredState ? result.filteredStateBase() : scratchFilteredStateBase;
        int filteredStateCovBase = storeFilteredCovariance ? result.filteredStateCovBase() : scratchFilteredStateCovBase;
        int logLikelihoodObsBase = storeLikelihood ? result.logLikelihoodObsBase() : 0;

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
        int concentratedScaleStart = Math.max(0, concentratedLikelihoodBurn);

        for (int t = 0; t < nobs; t++) {
            int predOff = storePredictedState ? result.predictedStateOffset(t) : rollingPredictedStateBase + (t & 1) * 2 * kStates;
            int predCovOff = storePredictedCovariance ? result.predictedStateCovOffset(t) : rollingPredictedStateCovBase + (t & 1) * ksKs;
            int filtOff = storeFilteredState ? result.filteredStateOffset(t) : filteredStateBase;
            int filtCovOff = storeFilteredCovariance ? result.filteredStateCovOffset(t) : filteredStateCovBase;
            int foreOff = storeForecast ? result.forecastOffset(t) : 0;
            int foreErrOff = storeForecast ? result.forecastErrorOffset(t) : 0;
            int foreErrCovOff = storeForecast ? result.forecastErrorCovOffset(t) : 0;
            int stdForeErrOff = storeStandardizedForecastError ? result.standardizedForecastErrorOffset(t) : 0;
            int nextPredOff = storePredictedState ? result.predictedStateOffset(t + 1) : rollingPredictedStateBase + ((t + 1) & 1) * 2 * kStates;
            int nextPredCovOff = storePredictedCovariance ? result.predictedStateCovOffset(t + 1) : rollingPredictedStateCovBase + ((t + 1) & 1) * ksKs;
            int kgOff = storeGain ? result.kalmanGainOffset(t) : 0;
            int endogOff = endogOverride != null ? endogBase + t * 2 * kEndog : observationModel.endogOffset(t);

            int ke = selectObservations(observationModel, pool, t, endog, endogOff);
            boolean fullyObserved = ke == kEndog;
            boolean useConverged = steadyStateEligible && converged && fullyObserved && previousFullyObserved;
            if (converged && !fullyObserved) {
                converged = false;
            }

            if (ke == 0) {
                if (storeForecast) {
                    for (int i = 0; i < 2 * kEndog; i++) {
                        forecast[foreOff + i] = 0.0;
                        forecastError[foreErrOff + i] = 0.0;
                    }
                    for (int i = 0; i < keKe; i++) {
                        forecastErrorCov[foreErrCovOff + i] = 0.0;
                    }
                }
                ZLAS.zcopy(kStates, predictedState, predOff, 1,
                        filteredState, filtOff, 1);
                ZLAS.zcopy(kStates * kStates, predictedStateCov, predCovOff, 1,
                        filteredStateCov, filtCovOff, 1);
                if (storeGain) {
                    for (int i = 0; i < kStates * 2 * kEndog; i++) {
                        kalmanGain[kgOff + i] = 0.0;
                    }
                }
                zeroComplex(result.standardizedForecastError, stdForeErrOff,
                    storeStandardizedForecastError ? kEndog : 0);
                if (storeLikelihood) {
                    result.logLikelihoodObs[logLikelihoodObsBase + t] = 0.0;
                }

                ZLAS.zcopy(kStates, stateIntercept, stateSpace.stateInterceptOffset(t), 1,
                        predictedState, nextPredOff, 1);
                ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                        predictedState, predOff, 1,
                        1.0, 0.0, predictedState, nextPredOff, 1);

                if (!converged) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                            predictedStateCov, predCovOff >> 1, kStates,
                            0.0, 0.0, scratch, tmpTPBase >> 1, kStates);
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpTPBase >> 1, kStates,
                        transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                            0.0, 0.0, predictedStateCov, nextPredCovOff >> 1, kStates);
                    if (kPosdef > 0) {
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                            0.0, 0.0, scratch, tmpRQBase >> 1, kPosdef);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kPosdef,
                            1.0, 0.0, scratch, tmpRQBase >> 1, kPosdef,
                        selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                                1.0, 0.0, predictedStateCov, nextPredCovOff >> 1, kStates);
                    }
                    if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                        symmetrizeComplexSymmetric(predictedStateCov, nextPredCovOff, kStates);
                    }
                } else {
                    ZLAS.zcopy(kStates * kStates, predictedStateCov, predCovOff, 1,
                            predictedStateCov, nextPredCovOff, 1);
                }
                previousFullyObserved = false;
                continue;
            }

            double[] Z; int zOff; int zLd;
            double[] H; int hOff;
            double[] d; int dOff;
            double[] y; int yOff;
            if (ke < kEndog) {
                Z = pool.selectionBacking; zOff = pool.selectedDesignBase(); zLd = kStates;
                H = pool.selectionBacking; hOff = pool.selectedObsCovBase();
                d = pool.selectionBacking; dOff = pool.selectedObsInterceptBase();
                y = pool.selectionBacking; yOff = pool.selectedEndogBase();
            } else {
                Z = design; zOff = observationModel.designOffset(t); zLd = designLd;
                H = obsCov; hOff = observationModel.obsCovOffset(t);
                d = obsIntercept; dOff = observationModel.obsInterceptOffset(t);
                y = endog; yOff = endogOff;
            }

            // Complex forecast/update path mirrors statsmodels' complex recursion,
            // which extends the real formulas with plain transposes.
                if (storeForecast) {
                ZLAS.zcopy(ke, d, dOff, 1,
                    forecast, foreOff, 1);
                ZLAS.zgemv(BLAS.Trans.NoTrans, ke, kStates, 1.0, 0.0,
                    Z, zOff, zLd,
                    predictedState, predOff, 1,
                    1.0, 0.0, forecast, foreOff, 1);

                ZLAS.zcopy(ke, y, yOff, 1,
                    forecastError, foreErrOff, 1);
                ZLAS.zaxpy(ke, -1.0, 0.0,
                    forecast, foreOff >> 1, 1,
                    forecastError, foreErrOff >> 1, 1);
                } else {
                ZLAS.zcopy(ke, d, dOff, 1, scratch, tmpVBase, 1);
                ZLAS.zgemv(BLAS.Trans.NoTrans, ke, kStates, 1.0, 0.0,
                    Z, zOff, zLd,
                    predictedState, predOff, 1,
                    1.0, 0.0, scratch, tmpVBase, 1);
                ZLAS.zcopy(ke, y, yOff, 1, scratch, tmpKBase, 1);
                ZLAS.zaxpy(ke, -1.0, 0.0,
                    scratch, tmpVBase >> 1, 1,
                    scratch, tmpKBase >> 1, 1);
                ZLAS.zcopy(ke, scratch, tmpKBase, 1, scratch, tmpVBase, 1);
                }

            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, ke, kStates,
                    1.0, 0.0, predictedStateCov, predCovOff >> 1, kStates,
                        Z, zOff >> 1, zLd,
                        0.0, 0.0, scratch, tmpPZtBase >> 1, ke);

            if (!useConverged) {
                if (storeForecast) {
                    ZLAS.zcopy(ke * ke, H, hOff, 1,
                        forecastErrorCov, foreErrCovOff, 1);
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, kStates,
                        1.0, 0.0, Z, zOff >> 1, zLd,
                        scratch, tmpPZtBase >> 1, ke,
                        1.0, 0.0, forecastErrorCov, foreErrCovOff >> 1, ke);
                } else {
                    ZLAS.zcopy(ke * ke, H, hOff, 1, scratch, tmpFcopyBase, 1);
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, kStates,
                        1.0, 0.0, Z, zOff >> 1, zLd,
                        scratch, tmpPZtBase >> 1, ke,
                        1.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                }
            } else {
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, ke, kStates,
                    1.0, 0.0, pool.convergedPredictedStateCov, 0, kStates,
                    Z, zOff >> 1, zLd,
                    0.0, 0.0, scratch, tmpPZtBase >> 1, ke);
                if (storeForecast) {
                    ZLAS.zcopy(ke * ke, H, hOff, 1,
                        forecastErrorCov, foreErrCovOff, 1);
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, kStates,
                        1.0, 0.0, Z, zOff >> 1, zLd,
                        scratch, tmpPZtBase >> 1, ke,
                        1.0, 0.0, forecastErrorCov, foreErrCovOff >> 1, ke);
                    ZLAS.zcopy(ke * ke, forecastErrorCov, foreErrCovOff, 1,
                        scratch, tmpFcopyBase, 1);
                } else {
                    ZLAS.zcopy(ke * ke, H, hOff, 1, scratch, tmpFcopyBase, 1);
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, kStates,
                        1.0, 0.0, Z, zOff >> 1, zLd,
                        scratch, tmpPZtBase >> 1, ke,
                        1.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                }
            }

            double logDetF = 0.0;
            double vFv = 0.0;

                if (storeForecast) {
                ZLAS.zcopy(ke * ke, forecastErrorCov, foreErrCovOff, 1, scratch, tmpFcopyBase, 1);
                }

            if (ke == 1) {
                double FtRe = scratch[tmpFcopyBase];
                double FtIm = scratch[tmpFcopyBase + 1];
                double FtMag2 = FtRe * FtRe + FtIm * FtIm;
                double FtInvRe = FtRe / FtMag2;
                double FtInvIm = -FtIm / FtMag2;
                double FtMag = Math.sqrt(FtRe * FtRe + FtIm * FtIm);
                logDetF = Math.log(FtMag);

                ZLAS.zcopy(kStates, scratch, tmpPZtBase, 1, scratch, tmpKBase, 1);
                ZLAS.zscal(kStates, FtInvRe, FtInvIm, scratch, tmpKBase, 1);

                if (storeGain) {
                    ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                            transition, stateSpace.transitionOffset(t), transitionLd,
                            scratch, tmpKBase, 1,
                            0.0, 0.0, kalmanGain, kgOff, 1);
                }

                ZLAS.zcopy(kStates, predictedState, predOff, 1,
                            filteredState, filtOff, 1);
                double vRe = storeForecast ? forecastError[foreErrOff] : scratch[tmpVBase];
                double vIm = storeForecast ? forecastError[foreErrOff + 1] : scratch[tmpVBase + 1];
                double vFtInvRe = vRe * FtInvRe - vIm * FtInvIm;
                double vFtInvIm = vRe * FtInvIm + vIm * FtInvRe;
                if (storeStandardizedForecastError) {
                    storeStandardizedScalarComplex(result.standardizedForecastError, stdForeErrOff,
                        vRe, vIm, FtRe, FtIm);
                }
                ZLAS.zaxpy(kStates, vFtInvRe, vFtInvIm,
                            scratch, tmpPZtBase >> 1, 1,
                            filteredState, filtOff >> 1, 1);

                ZLAS.zcopy(kStates * kStates, predictedStateCov, predCovOff, 1,
                            filteredStateCov, filtCovOff, 1);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, ke,
                            -1.0, 0.0, scratch, tmpKBase >> 1, ke,
                            scratch, tmpPZtBase >> 1, ke,
                            1.0, 0.0, filteredStateCov, filtCovOff >> 1, kStates);

                vFv = vRe * vFtInvRe + vIm * vFtInvIm;
            } else {
                ZLAS.zpotrf(BLAS.Uplo.Lower, ke, scratch, tmpFcopyBase >> 1, ke);

                if (storeStandardizedForecastError) {
                    double[] vSrc = storeForecast ? forecastError : scratch;
                    int vSrcOff = storeForecast ? foreErrOff : tmpVBase;
                    ZLAS.zcopy(ke, vSrc, vSrcOff, 1,
                        result.standardizedForecastError, stdForeErrOff, 1);
                    solveLowerTriangularInPlace(scratch, tmpFcopyBase,
                        result.standardizedForecastError, stdForeErrOff, ke);
                }

                for (int i = 0; i < ke; i++) {
                    int diagOff = tmpFcopyBase + (i * ke + i) * 2;
                    double diagRe = scratch[diagOff];
                    double diagIm = scratch[diagOff + 1];
                    logDetF += Math.log(Math.sqrt(diagRe * diagRe + diagIm * diagIm));
                }
                logDetF *= 2.0;

                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, kStates, kStates,
                            1.0, 0.0, Z, zOff >> 1, zLd,
                            predictedStateCov, predCovOff >> 1, kStates,
                            0.0, 0.0, scratch, tmpKBase >> 1, kStates);

                ZLAS.zpotrs(BLAS.Uplo.Lower, ke, kStates,
                             scratch, tmpFcopyBase >> 1, ke, scratch, tmpKBase >> 1, kStates);

                if (storeGain) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, ke, kStates,
                            1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                            scratch, tmpKBase >> 1, kStates,
                            0.0, 0.0, kalmanGain, kgOff >> 1, ke);
                }

                ZLAS.zcopy(kStates, predictedState, predOff, 1,
                            filteredState, filtOff, 1);
                double[] vSrc = storeForecast ? forecastError : scratch;
                int vSrcOff = storeForecast ? foreErrOff : tmpVBase;
                ZLAS.zgemv(BLAS.Trans.Trans, ke, kStates, 1.0, 0.0,
                            scratch, tmpKBase, kStates,
                            vSrc, vSrcOff, 1,
                            1.0, 0.0, filteredState, filtOff, 1);

                ZLAS.zcopy(kStates * kStates, predictedStateCov, predCovOff, 1,
                            filteredStateCov, filtCovOff, 1);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                            -1.0, 0.0, scratch, tmpPZtBase >> 1, ke,
                            scratch, tmpKBase >> 1, kStates,
                            1.0, 0.0, filteredStateCov, filtCovOff >> 1, kStates);

                if (storeForecast) {
                    ZLAS.zcopy(ke, forecastError, foreErrOff, 1, scratch, tmpVBase, 1);
                } else {
                    ZLAS.zcopy(ke, scratch, tmpVBase, 1, scratch, tmpKBase, 1);
                }
                ZLAS.zpotrs(BLAS.Uplo.Lower, ke, 1,
                         scratch, tmpFcopyBase >> 1, ke, scratch, tmpVBase >> 1, 1);
                double[] vFvArr = new double[2];
                if (storeForecast) {
                    ZLAS.zdotc(ke, forecastError, foreErrOff >> 1, 1,
                        scratch, tmpVBase >> 1, 1, vFvArr);
                } else {
                    ZLAS.zdotc(ke, scratch, tmpKBase >> 1, 1,
                        scratch, tmpVBase >> 1, 1, vFvArr);
                }
                vFv = vFvArr[0];
            }

            if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                symmetrizeComplexSymmetric(filteredStateCov, filtCovOff, kStates);
            }

            if (storeLikelihood) {
                double logLikelihood = -0.5 * (ke * Math.log(2 * Math.PI) + logDetF);
                if (!concentratedLikelihood) {
                    logLikelihood += -0.5 * vFv;
                }
                result.logLikelihoodObs[logLikelihoodObsBase + t] = logLikelihood;
            }
            if (concentratedLikelihood) {
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
                ZLAS.zcopy(kStates * kStates, pool.convergedFilteredStateCov, 0, 1,
                    filteredStateCov, filtCovOff, 1);
                if (storeGain) {
                    ZLAS.zcopy(kStates * kEndog, pool.convergedKalmanGain, 0, 1,
                        kalmanGain, kgOff, 1);
                }
            }

            ZLAS.zcopy(kStates, stateIntercept, stateSpace.stateInterceptOffset(t), 1,
                        predictedState, nextPredOff, 1);
            ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                        transition, stateSpace.transitionOffset(t), transitionLd,
                        filteredState, filtOff, 1,
                        1.0, 0.0, predictedState, nextPredOff, 1);

            if (!converged) {
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                            filteredStateCov, filtCovOff >> 1, kStates,
                            0.0, 0.0, scratch, tmpTPBase >> 1, kStates);

                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpTPBase >> 1, kStates,
                            transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                            0.0, 0.0, predictedStateCov, nextPredCovOff >> 1, kStates);

                if (kPosdef > 0) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                                stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                                0.0, 0.0, scratch, tmpRQBase >> 1, kPosdef);

                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kPosdef,
                                1.0, 0.0, scratch, tmpRQBase >> 1, kPosdef,
                                selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                                1.0, 0.0, predictedStateCov, nextPredCovOff >> 1, kStates);
                }

                if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                    for (int i = 0; i < kStates; i++) {
                        for (int j = i + 1; j < kStates; j++) {
                            int ij = nextPredCovOff + (i * 2 * kStates + j * 2);
                            int ji = nextPredCovOff + (j * 2 * kStates + i * 2);
                            double avgRe = (predictedStateCov[ij] + predictedStateCov[ji]) * 0.5;
                            double avgIm = (predictedStateCov[ij + 1] + predictedStateCov[ji + 1]) * 0.5;
                            predictedStateCov[ij] = avgRe;
                            predictedStateCov[ij + 1] = avgIm;
                            predictedStateCov[ji] = avgRe;
                            predictedStateCov[ji + 1] = avgIm;
                        }
                    }
                }

                if (steadyStateEligible && t > 0 && fullyObserved && previousFullyObserved) {
                    double maxDiff = 0.0;
                    for (int i = 0; i < ksKs; i++) {
                        double diff = Math.abs(predictedStateCov[nextPredCovOff + i] -
                                               predictedStateCov[predCovOff + i]);
                        if (diff > maxDiff) maxDiff = diff;
                    }
                    if (maxDiff < convergenceTolerance) {
                        converged = true;
                        periodConverged = t + 1;
                        pool.ensureConvergedSnapshots(kStates, kEndog);
                        if (storeForecast) {
                            ZLAS.zcopy(ke * ke, forecastErrorCov, foreErrCovOff, 1,
                                pool.convergedForecastErrorCov, 0, 1);
                        } else {
                            ZLAS.zcopy(ke * ke, scratch, tmpFcopyBase, 1,
                                pool.convergedForecastErrorCov, 0, 1);
                        }
                        ZLAS.zcopy(kStates * kStates, filteredStateCov, filtCovOff, 1,
                            pool.convergedFilteredStateCov, 0, 1);
                        ZLAS.zcopy(kStates * kStates, predictedStateCov, nextPredCovOff, 1,
                            pool.convergedPredictedStateCov, 0, 1);
                        if (storeGain) {
                            ZLAS.zcopy(kStates * kEndog, kalmanGain, kgOff, 1,
                                pool.convergedKalmanGain, 0, 1);
                        }
                    }
                }
            } else {
                ZLAS.zcopy(kStates * kStates, pool.convergedPredictedStateCov, 0, 1,
                            predictedStateCov, nextPredCovOff, 1);
            }

            previousFullyObserved = fullyObserved;
        }

        for (int t = 0; t < nobs; t++) {
            if (observationModel.missingCount(t) == 0 || (!storeForecast && !storeGain)) continue;

            if (storeForecast) {
                int predOff = result.predictedStateOffset(t);
                int predCovOff = result.predictedStateCovOffset(t);
                int foreOff = result.forecastOffset(t);
                int foreErrOff = result.forecastErrorOffset(t);
                int foreErrCovOff = result.forecastErrorCovOffset(t);

                ZLAS.zcopy(kEndog, obsIntercept, observationModel.obsInterceptOffset(t), 1,
                        forecast, foreOff, 1);
                ZLAS.zgemv(BLAS.Trans.NoTrans, kEndog, kStates, 1.0, 0.0,
                        design, observationModel.designOffset(t), designLd,
                        result.predictedState, predOff, 1,
                        1.0, 0.0, forecast, foreOff, 1);

                if (observationModel.missingCount(t) < kEndog) {
                    int ke = selectObservations(observationModel, pool, t, endog,
                            endogOverride != null ? endogBase + t * 2 * kEndog : observationModel.endogOffset(t));
                    int[] selectedIndex = pool.selectedIndex;
                    ZLAS.zcopy(ke, forecastError, foreErrOff, 1, scratch, tmpVBase, 1);
                    for (int i = 0; i < 2 * kEndog; i++) forecastError[foreErrOff + i] = 0.0;
                    for (int j = 0; j < ke; j++) {
                        int idx = selectedIndex[j];
                        forecastError[foreErrOff + idx * 2] = scratch[tmpVBase + j * 2];
                        forecastError[foreErrOff + idx * 2 + 1] = scratch[tmpVBase + j * 2 + 1];
                    }
                } else {
                    for (int i = 0; i < 2 * kEndog; i++) forecastError[foreErrOff + i] = 0.0;
                }

                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kEndog, kStates,
                        1.0, 0.0, result.predictedStateCov, predCovOff >> 1, kStates,
                        design, observationModel.designOffset(t) >> 1, designLd,
                    0.0, 0.0, scratch, tmpPZtBase >> 1, kEndog);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kEndog, kEndog, kStates,
                        1.0, 0.0, design, observationModel.designOffset(t) >> 1, designLd,
                    scratch, tmpPZtBase >> 1, kEndog,
                        0.0, 0.0, forecastErrorCov, foreErrCovOff >> 1, kEndog);
                ZLAS.zaxpy(kEndog * kEndog, 1.0, 0.0,
                        obsCov, observationModel.obsCovOffset(t) >> 1, 1,
                        forecastErrorCov, foreErrCovOff >> 1, 1);
            }

            if (storeGain) {
                int kgOff = result.kalmanGainOffset(t);
                if (observationModel.missingCount(t) < kEndog) {
                    int ke = selectObservations(observationModel, pool, t, endog,
                            endogOverride != null ? endogBase + t * 2 * kEndog : observationModel.endogOffset(t));
                    int[] selectedIndex = pool.selectedIndex;
                    ZLAS.zcopy(kStates * ke, kalmanGain, kgOff, 1, scratch, tmpKBase, 1);
                    for (int i = 0; i < kStates * 2 * kEndog; i++) kalmanGain[kgOff + i] = 0.0;
                    for (int i = 0; i < kStates; i++) {
                        for (int j = 0; j < ke; j++) {
                            int idx = selectedIndex[j];
                            int sourceOff = tmpKBase + i * 2 * ke + j * 2;
                            kalmanGain[kgOff + i * 2 * kEndog + idx * 2] = scratch[sourceOff];
                            kalmanGain[kgOff + i * 2 * kEndog + idx * 2 + 1] = scratch[sourceOff + 1];
                        }
                    }
                } else {
                    for (int i = 0; i < kStates * 2 * kEndog; i++) kalmanGain[kgOff + i] = 0.0;
                }
            }
        }

        if (concentratedLikelihood) {
            double scale = concentratedScaleObservationCount == 0
                ? 1.0
                : concentratedScaleSum / concentratedScaleObservationCount;
            result.scale = scale;
            result.scaleObservationCount = concentratedScaleObservationCount;
            KalmanFilter.applyConcentratedScale(observationModel, result, scale, storeLikelihood);
        }

        result.applyConserveMemory(conserveMemory);
        result.converged = converged;
        result.periodConverged = periodConverged;
        if (reuseResult) {
            pool.releaseConvergedSnapshots();
        }
        return result;
    }

    private static KalmanSSM prepareUnivariateObservationModel(KalmanSSM model,
                                                                     Pool pool,
                                                                     double[] endogOverride,
                                                                     int endogBase) {
        if (!needsObservationTransform(model)) {
            return model;
        }
        return pool.observationTransformWorkspace().transform(model, endogOverride, endogBase);
    }

    private static boolean needsObservationTransform(KalmanSSM model) {
        int kEndog = model.observationDimension();
        int nobs = model.observationCount();
        for (int t = 0; t < nobs; t++) {
            if (kEndog - model.missingCount(t) <= 1) {
                continue;
            }
            int obsCovOffset = model.obsCovOffset(t);
            int obsCovLd = model.obsCovLeadingDimension();
            for (int row = 0; row < kEndog; row++) {
                if (model.isMissing(row, t)) {
                    continue;
                }
                for (int col = 0; col < kEndog; col++) {
                    if (row == col || model.isMissing(col, t)) {
                        continue;
                    }
                    int off = obsCovOffset + row * 2 * obsCovLd + col * 2;
                    if (model.obsCovData()[off] != 0.0 || model.obsCovData()[off + 1] != 0.0) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private static ZFilterResult filterDiffuse(KalmanSSM model,
                                               InitialState init,
                                               Pool pool,
                                               int kEndog, int kStates, int kPosdef,
                                               int nobs, int ksKs, int keKe,
                                               FilterResultLayout retainedLayout,
                                               FilterDiffuseLayout retainedDiffuseLayout,
                                               int stabilityMethod, int conserveMemory,
                                               double diffuseTolerance,
                                               boolean concentratedLikelihood,
                                               int concentratedLikelihoodBurn,
                                               boolean reuseResult,
                                               double[] endogOverride,
                                               int endogBase) {
        KalmanSSM stateSpace = model;
        KalmanSSM observationModel = model;
        double[] transition = stateSpace.transitionData();
        int transitionLd = stateSpace.transitionLeadingDimension();
        double[] stateIntercept = stateSpace.stateInterceptData();
        double[] design = observationModel.designData();
        int designLd = observationModel.designLeadingDimension();
        double[] obsIntercept = observationModel.obsInterceptData();
        double[] obsCov = observationModel.obsCovData();
        int obsCovLd = observationModel.obsCovLeadingDimension();
        double[] endog = endogOverride != null ? endogOverride : observationModel.endogData();
        boolean storePredictedState = retainedLayout.predictedStateLength() > 0;
        boolean storePredictedCovariance = retainedLayout.predictedStateCovLength() > 0;
        boolean storePredictedDiffuseCovariance = retainedDiffuseLayout.predictedDiffuseStateCovLength() > 0;
        boolean storeForecast = retainedLayout.forecastLength() > 0;
        boolean storeForecastDiffuseCovariance = retainedDiffuseLayout.forecastErrorDiffuseCovLength() > 0;
        boolean storeFilteredState = retainedLayout.filteredStateLength() > 0;
        boolean storeFilteredCovariance = retainedLayout.filteredStateCovLength() > 0;
        boolean storeGain = retainedLayout.kalmanGainLength() > 0;
        boolean storeLikelihood = retainedLayout.logLikelihoodObsLength() > 0;
        FilterDiffuseLayout diffuseLayout = retainedDiffuseLayout;
        boolean usePredictedScratch = !storePredictedState || !storePredictedCovariance;
        boolean usePredictedDiffuseScratch = !storePredictedDiffuseCovariance;
        pool.ensureExactDiffuseScratch(kStates, kEndog, kPosdef,
            usePredictedScratch,
            false,
            usePredictedDiffuseScratch);
        ZFilterResult result = reuseResult
            ? pool.borrowDiffuseResult(kEndog, kStates, nobs, retainedLayout, diffuseLayout)
            : new ZFilterResult(kEndog, kStates, nobs, retainedLayout);
        if (reuseResult) {
            result.clearActiveSurfaces();
        }
        if (!reuseResult) {
            result.allocateDiffuse(storePredictedDiffuseCovariance, storeForecastDiffuseCovariance);
        }
        result.perObsKalmanGain = true;
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
        double[] periodScaleObs = concentratedLikelihood && !storeLikelihood ? new double[nobs] : result.scaleObs;
        int[] periodScaleObsCount = concentratedLikelihood && !storeLikelihood ? new int[nobs] : result.scaleObsCount;
        double[] scratch = pool.scratchBacking;
        InitialState.StationaryWorkspace stationaryWorkspace = null;
        if (init.mayResolveStationary()) {
            pool.ensureStationary(model);
            stationaryWorkspace = pool.stationaryWorkspace;
        }

        double[] predictedState = storePredictedState
            ? result.predictedState
            : scratch;
        double[] predictedStateCov = storePredictedCovariance
            ? result.predictedStateCov
            : scratch;
        double[] predictedDiffuseStateCov = storePredictedDiffuseCovariance
            ? result.predictedDiffuseStateCov
            : scratch;
        int predictedStateBase = storePredictedState ? result.predictedStateBase()
            : pool.rollingPredictedStateBase();
        int predictedStateCovBase = storePredictedCovariance ? result.predictedStateCovBase()
            : pool.rollingPredictedStateCovBase();
        int predictedDiffuseStateCovBase = storePredictedDiffuseCovariance ? result.predictedDiffuseStateCovBase()
            : pool.rollingPredictedDiffuseStateCovBase();
        int tmpABase = pool.tmpABase();
        int tmpPInfBase = pool.tmpPInfBase();
        int tmpPStarBase = pool.tmpPStarBase();
        int tmpMStarBase = pool.tmpMStarBase();
        int tmpMInfBase = pool.tmpMInfBase();
        int tmpK0Base = pool.tmpK0Base();
        int tmpK1Base = pool.tmpK1Base();

        zero(predictedState, predictedStateBase, 2 * kStates);
        zero(predictedStateCov, predictedStateCovBase, ksKs);
        zero(predictedDiffuseStateCov, predictedDiffuseStateCovBase, ksKs);
        boolean diffuse = init.resolveInto(model, 0, stationaryWorkspace,
            predictedState, predictedStateBase,
            predictedStateCov, predictedStateCovBase,
            predictedDiffuseStateCov, predictedDiffuseStateCovBase);
        int nobsDiffuse = diffuse ? 1 : 0;

        for (int t = 0; t < nobs; t++) {
        int predOff = storePredictedState ? result.predictedStateOffset(t)
            : predictedStateBase + (t & 1) * 2 * kStates;
        int predCovOff = storePredictedCovariance ? result.predictedStateCovOffset(t)
            : predictedStateCovBase + (t & 1) * ksKs;
        int predDiffCovOff = storePredictedDiffuseCovariance ? result.predictedDiffuseStateCovOffset(t)
            : predictedDiffuseStateCovBase + (t & 1) * ksKs;
            int filtOff = storeFilteredState ? result.filteredStateOffset(t) : 0;
            int filtCovOff = storeFilteredCovariance ? result.filteredStateCovOffset(t) : 0;
            int foreOff = result.forecastOffset(t);
            int foreErrOff = result.forecastErrorOffset(t);
            int foreErrCovOff = result.forecastErrorCovOffset(t);
            int foreErrDiffCovOff = result.forecastErrorDiffuseCovOffset(t);
        int nextPredOff = storePredictedState ? result.predictedStateOffset(t + 1)
            : predictedStateBase + ((t + 1) & 1) * 2 * kStates;
        int nextPredCovOff = storePredictedCovariance ? result.predictedStateCovOffset(t + 1)
            : predictedStateCovBase + ((t + 1) & 1) * ksKs;
        int nextPredDiffCovOff = storePredictedDiffuseCovariance ? result.predictedDiffuseStateCovOffset(t + 1)
            : predictedDiffuseStateCovBase + ((t + 1) & 1) * ksKs;
            int kgOff = result.kalmanGainOffset(t);
            int endogOff = endogOverride != null ? endogBase + t * 2 * kEndog : observationModel.endogOffset(t);

            if (observationModel.missingCount(t) == kEndog) {
                if (storeForecast) {
                    zero(result.forecast, foreOff, 2 * kEndog);
                    zero(result.forecastError, foreErrOff, 2 * kEndog);
                    zero(result.forecastErrorCov, foreErrCovOff, keKe);
                }
                if (diffuse && storeForecastDiffuseCovariance) {
                    zero(result.forecastErrorDiffuseCov, foreErrDiffCovOff, keKe);
                }
                if (storeFilteredState) {
                    ZLAS.zcopy(kStates, predictedState, predOff, 1,
                            result.filteredState, filtOff, 1);
                }
                if (storeFilteredCovariance) {
                    ZLAS.zcopy(kStates * kStates, predictedStateCov, predCovOff, 1,
                            result.filteredStateCov, filtCovOff, 1);
                }
                if (storeGain) {
                    zero(result.kalmanGain, kgOff, kStates * 2 * kEndog);
                }
                if (storeLikelihood) {
                    result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] = 0.0;
                }

                ZLAS.zcopy(kStates, stateIntercept, stateSpace.stateInterceptOffset(t), 1,
                        predictedState, nextPredOff, 1);
                ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                        predictedState, predOff, 1,
                        1.0, 0.0, predictedState, nextPredOff, 1);

                propagateHermitian(stateSpace, t, predictedStateCov, predCovOff,
                        predictedStateCov, nextPredCovOff, pool, kStates, kPosdef, true);
                if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                    symmetrizeComplexSymmetric(predictedStateCov, nextPredCovOff, kStates);
                }

                if (diffuse) {
                    propagateHermitian(stateSpace, t, predictedDiffuseStateCov, predDiffCovOff,
                            predictedDiffuseStateCov, nextPredDiffCovOff, pool, kStates, 0, false);
                    if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                        symmetrizeComplexSymmetric(predictedDiffuseStateCov, nextPredDiffCovOff, kStates);
                    }
                    double pInfNorm = squaredNorm(predictedDiffuseStateCov, nextPredDiffCovOff, ksKs);
                    if (t + 1 >= nobsDiffuse) {
                        if (pInfNorm < diffuseTolerance) {
                            diffuse = false;
                            result.nobsDiffuse = nobsDiffuse;
                        } else {
                            nobsDiffuse = t + 2;
                        }
                    }
                }
                continue;
            }

            ZLAS.zcopy(kStates, predictedState, predOff, 1, scratch, tmpABase, 1);
            ZLAS.zcopy(kStates * kStates, predictedStateCov, predCovOff, 1, scratch, tmpPStarBase, 1);
            if (diffuse) {
                ZLAS.zcopy(kStates * kStates, predictedDiffuseStateCov, predDiffCovOff, 1, scratch, tmpPInfBase, 1);
            }

            if (storeForecast) {
                zero(result.forecast, foreOff, 2 * kEndog);
                zero(result.forecastError, foreErrOff, 2 * kEndog);
                zero(result.forecastErrorCov, foreErrCovOff, keKe);
            }
            if (diffuse && storeForecastDiffuseCovariance) {
                zero(result.forecastErrorDiffuseCov, foreErrDiffCovOff, keKe);
            }
            if (storeGain) {
                zero(result.kalmanGain, kgOff, kStates * 2 * kEndog);
            }

            double logLik = 0.0;
            double periodScaleContribution = 0.0;
            int periodScaleObservationCount = 0;

            for (int i = 0; i < kEndog; i++) {
                if (observationModel.isMissing(i, t)) {
                    continue;
                }

                int zRowOff = observationModel.designOffset(t) + i * 2 * designLd;
                int yOff = endogOff + i * 2;
                int dOff = observationModel.obsInterceptOffset(t) + i * 2;
                int hDiagOff = observationModel.obsCovOffset(t) + i * 2 * obsCovLd + i * 2;

                double vRe = endog[yOff] - obsIntercept[dOff];
                double vIm = endog[yOff + 1] - obsIntercept[dOff + 1];
                for (int j = 0; j < kStates; j++) {
                    int zOff = zRowOff + j * 2;
                    int aOff = tmpABase + j * 2;
                    double prodRe = design[zOff] * scratch[aOff] - design[zOff + 1] * scratch[aOff + 1];
                    double prodIm = design[zOff] * scratch[aOff + 1] + design[zOff + 1] * scratch[aOff];
                    vRe -= prodRe;
                    vIm -= prodIm;
                }
                if (storeForecast) {
                    result.forecast[foreOff + i * 2] = endog[yOff] - vRe;
                    result.forecast[foreOff + i * 2 + 1] = endog[yOff + 1] - vIm;
                    result.forecastError[foreErrOff + i * 2] = vRe;
                    result.forecastError[foreErrOff + i * 2 + 1] = vIm;
                }

                for (int j = 0; j < kStates; j++) {
                    int mOff = j * 2;
                    double mStarRe = 0.0;
                    double mStarIm = 0.0;
                    double mInfRe = 0.0;
                    double mInfIm = 0.0;
                    for (int k = 0; k < kStates; k++) {
                        int pOff = j * 2 * kStates + k * 2;
                        int zOff = zRowOff + k * 2;
                        double zRe = design[zOff];
                        double zIm = -design[zOff + 1];
                        double pStarRe = scratch[tmpPStarBase + pOff];
                        double pStarIm = scratch[tmpPStarBase + pOff + 1];
                        mStarRe += pStarRe * zRe - pStarIm * zIm;
                        mStarIm += pStarRe * zIm + pStarIm * zRe;
                        if (diffuse) {
                            double pInfRe = scratch[tmpPInfBase + pOff];
                            double pInfIm = scratch[tmpPInfBase + pOff + 1];
                            mInfRe += pInfRe * zRe - pInfIm * zIm;
                            mInfIm += pInfRe * zIm + pInfIm * zRe;
                        }
                    }
                    scratch[tmpMStarBase + mOff] = mStarRe;
                    scratch[tmpMStarBase + mOff + 1] = mStarIm;
                    if (diffuse) {
                        scratch[tmpMInfBase + mOff] = mInfRe;
                        scratch[tmpMInfBase + mOff + 1] = mInfIm;
                    }
                }

                double fStarRe = obsCov[hDiagOff];
                double fInfRe = 0.0;
                for (int j = 0; j < kStates; j++) {
                    int zOff = zRowOff + j * 2;
                    int mOff = j * 2;
                    double zRe = design[zOff];
                    double zIm = design[zOff + 1];
                    double mStarRe = scratch[tmpMStarBase + mOff];
                    double mStarIm = scratch[tmpMStarBase + mOff + 1];
                    fStarRe += zRe * mStarRe - zIm * mStarIm;
                    if (diffuse) {
                        double mInfRe = scratch[tmpMInfBase + mOff];
                        double mInfIm = scratch[tmpMInfBase + mOff + 1];
                        fInfRe += zRe * mInfRe - zIm * mInfIm;
                    }
                }

                double fStar = fStarRe;
                double fInf = fInfRe;

                if (diffuse && fInf > diffuseTolerance) {
                    double f1 = 1.0 / fInf;
                    double f2 = -fStar * f1 * f1;
                    if (storeForecastDiffuseCovariance) {
                        result.forecastErrorDiffuseCov[foreErrDiffCovOff + i * 2 * kEndog + i * 2] = fInf;
                    }
                    if (storeForecast) {
                        result.forecastErrorCov[foreErrCovOff + i * 2 * kEndog + i * 2] = fStar;
                    }

                    for (int j = 0; j < kStates; j++) {
                        int off = j * 2;
                        double mInfRe = scratch[tmpMInfBase + off];
                        double mInfIm = scratch[tmpMInfBase + off + 1];
                        double mStarRe = scratch[tmpMStarBase + off];
                        double mStarIm = scratch[tmpMStarBase + off + 1];
                        scratch[tmpK0Base + off] = mInfRe * f1;
                        scratch[tmpK0Base + off + 1] = mInfIm * f1;
                        scratch[tmpK1Base + off] = mStarRe * f1 + mInfRe * f2;
                        scratch[tmpK1Base + off + 1] = mStarIm * f1 + mInfIm * f2;
                        if (storeGain) {
                            result.kalmanGain[kgOff + j * 2 * kEndog + i * 2] = scratch[tmpK0Base + off];
                            result.kalmanGain[kgOff + j * 2 * kEndog + i * 2 + 1] = scratch[tmpK0Base + off + 1];
                        }
                    }

                    axpyVector(scratch, tmpABase, scratch, tmpK0Base, vRe, vIm, kStates);
                    rank1Subtract(scratch, tmpPStarBase, scratch, tmpMStarBase, scratch, tmpK0Base, kStates);
                    rank1Subtract(scratch, tmpPStarBase, scratch, tmpMInfBase, scratch, tmpK1Base, kStates);
                    rank1Subtract(scratch, tmpPInfBase, scratch, tmpMInfBase, scratch, tmpK0Base, kStates);

                    if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                        symmetrizeComplexSymmetric(scratch, tmpPStarBase, kStates);
                        symmetrizeComplexSymmetric(scratch, tmpPInfBase, kStates);
                    }

                    logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(fInf));
                } else if (fStar > diffuseTolerance) {
                    double fStarInv = 1.0 / fStar;
                    double quadratic = (vRe * vRe + vIm * vIm) * fStarInv;
                    if (storeForecast) {
                        result.forecastErrorCov[foreErrCovOff + i * 2 * kEndog + i * 2] = fStar;
                    }

                    for (int j = 0; j < kStates; j++) {
                        int off = j * 2;
                        scratch[tmpK0Base + off] = scratch[tmpMStarBase + off] * fStarInv;
                        scratch[tmpK0Base + off + 1] = scratch[tmpMStarBase + off + 1] * fStarInv;
                        if (storeGain) {
                            result.kalmanGain[kgOff + j * 2 * kEndog + i * 2] = scratch[tmpK0Base + off];
                            result.kalmanGain[kgOff + j * 2 * kEndog + i * 2 + 1] = scratch[tmpK0Base + off + 1];
                        }
                    }

                    axpyVector(scratch, tmpABase, scratch, tmpK0Base, vRe, vIm, kStates);
                    rank1Subtract(scratch, tmpPStarBase, scratch, tmpMStarBase, scratch, tmpK0Base, kStates);
                    if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                        symmetrizeComplexSymmetric(scratch, tmpPStarBase, kStates);
                    }

                    logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(fStar));
                    if (!concentratedLikelihood) {
                        logLik -= 0.5 * quadratic;
                    } else {
                        periodScaleContribution += quadratic;
                        periodScaleObservationCount++;
                    }
                }
            }

            if (storeFilteredState) {
                ZLAS.zcopy(kStates, scratch, tmpABase, 1, result.filteredState, filtOff, 1);
            }
            if (storeFilteredCovariance) {
                ZLAS.zcopy(kStates * kStates, scratch, tmpPStarBase, 1, result.filteredStateCov, filtCovOff, 1);
            }
            if (storeLikelihood) {
                result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] = logLik;
            }
            if (concentratedLikelihood) {
                periodScaleObs[t] = periodScaleContribution;
                periodScaleObsCount[t] = periodScaleObservationCount;
            }

            ZLAS.zcopy(kStates, stateIntercept, stateSpace.stateInterceptOffset(t), 1,
                    predictedState, nextPredOff, 1);
            ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                    scratch, tmpABase, 1,
                    1.0, 0.0, predictedState, nextPredOff, 1);

            propagateHermitian(stateSpace, t, scratch, tmpPStarBase,
                    predictedStateCov, nextPredCovOff, pool, kStates, kPosdef, true);
            if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                symmetrizeComplexSymmetric(predictedStateCov, nextPredCovOff, kStates);
            }

            if (diffuse) {
                propagateHermitian(stateSpace, t, scratch, tmpPInfBase,
                        predictedDiffuseStateCov, nextPredDiffCovOff, pool, kStates, 0, false);
                if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                    symmetrizeComplexSymmetric(predictedDiffuseStateCov, nextPredDiffCovOff, kStates);
                }

                double pInfNorm = squaredNorm(predictedDiffuseStateCov, nextPredDiffCovOff, ksKs);
                if (t + 1 >= nobsDiffuse) {
                    if (pInfNorm < diffuseTolerance) {
                        diffuse = false;
                        result.nobsDiffuse = nobsDiffuse;
                    } else {
                        nobsDiffuse = t + 2;
                    }
                }
            }
        }

        if (diffuse) {
            result.nobsDiffuse = nobsDiffuse;
        }

        if (concentratedLikelihood) {
            int concentratedScaleStart = Math.max(Math.max(0, concentratedLikelihoodBurn), result.nobsDiffuse);
            double concentratedScaleSum = 0.0;
            int concentratedScaleObservationCount = 0;
            for (int t = concentratedScaleStart; t < nobs; t++) {
                concentratedScaleSum += periodScaleObs[t];
                concentratedScaleObservationCount += periodScaleObsCount[t];
            }
            double scale = concentratedScaleObservationCount == 0
                ? 1.0
                : concentratedScaleSum / concentratedScaleObservationCount;
            result.scale = scale;
            result.scaleObservationCount = concentratedScaleObservationCount;
            KalmanFilter.applyConcentratedScale(observationModel, result, scale, storeLikelihood);
        }

        result.applyConserveMemory(conserveMemory);
        result.converged = false;
        result.periodConverged = nobs;
        if (reuseResult) {
            pool.releaseConvergedSnapshots();
        }
        return result;
    }

    private static void propagateHermitian(KalmanSSM stateSpace, int t,
                                           double[] source, int sourceOff,
                                           double[] target, int targetOff,
                                           Pool pool, int kStates, int kPosdef,
                                           boolean addStateNoise) {
        double[] transition = stateSpace.transitionData();
        int transitionLd = stateSpace.transitionLeadingDimension();
        double[] selection = stateSpace.selectionData();
        int selectionLd = stateSpace.selectionLeadingDimension();
        double[] stateCovariance = stateSpace.stateCovarianceData();
        int stateCovarianceLd = stateSpace.stateCovarianceLeadingDimension();
        double[] scratch = pool.scratchBacking;
        int tmpTPBase = pool.tmpTPBase();
        int tmpRQBase = pool.tmpRQBase();

        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                source, sourceOff >> 1, kStates,
                0.0, 0.0, scratch, tmpTPBase >> 1, kStates);
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                1.0, 0.0, scratch, tmpTPBase >> 1, kStates,
                transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                0.0, 0.0, target, targetOff >> 1, kStates);
        if (addStateNoise && kPosdef > 0) {
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                    1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                    stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                    0.0, 0.0, scratch, tmpRQBase >> 1, kPosdef);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kPosdef,
                    1.0, 0.0, scratch, tmpRQBase >> 1, kPosdef,
                    selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                    1.0, 0.0, target, targetOff >> 1, kStates);
        }
    }

    private static void axpyVector(double[] target, int targetBase,
                                   double[] vector, int vectorBase,
                                   double scalarRe, double scalarIm, int n) {
        for (int i = 0; i < n; i++) {
            int off = i * 2;
            double vRe = vector[vectorBase + off];
            double vIm = vector[vectorBase + off + 1];
            target[targetBase + off] += vRe * scalarRe - vIm * scalarIm;
            target[targetBase + off + 1] += vRe * scalarIm + vIm * scalarRe;
        }
    }

    private static void rank1Subtract(double[] matrix, int matrixBase,
                                      double[] left, int leftBase,
                                      double[] right, int rightBase,
                                      int n) {
        for (int row = 0; row < n; row++) {
            int leftOff = row * 2;
            double leftRe = left[leftBase + leftOff];
            double leftIm = left[leftBase + leftOff + 1];
            for (int col = 0; col < n; col++) {
                int rightOff = col * 2;
                double rightRe = right[rightBase + rightOff];
                double rightIm = -right[rightBase + rightOff + 1];
                int matOff = matrixBase + row * 2 * n + col * 2;
                matrix[matOff] -= leftRe * rightRe - leftIm * rightIm;
                matrix[matOff + 1] -= leftRe * rightIm + leftIm * rightRe;
            }
        }
    }

    private static void symmetrizeHermitian(double[] matrix, int offset, int n) {
        for (int i = 0; i < n; i++) {
            int diag = offset + i * 2 * n + i * 2;
            matrix[diag + 1] = 0.0;
            for (int j = i + 1; j < n; j++) {
                int ij = offset + i * 2 * n + j * 2;
                int ji = offset + j * 2 * n + i * 2;
                double avgRe = (matrix[ij] + matrix[ji]) * 0.5;
                double avgIm = (matrix[ij + 1] - matrix[ji + 1]) * 0.5;
                matrix[ij] = avgRe;
                matrix[ij + 1] = avgIm;
                matrix[ji] = avgRe;
                matrix[ji + 1] = -avgIm;
            }
        }
    }

    private static void symmetrizeComplexSymmetric(double[] matrix, int offset, int n) {
        for (int row = 0; row < n; row++) {
            for (int col = row + 1; col < n; col++) {
                int upper = offset + row * 2 * n + col * 2;
                int lower = offset + col * 2 * n + row * 2;
                double avgRe = (matrix[upper] + matrix[lower]) * 0.5;
                double avgIm = (matrix[upper + 1] + matrix[lower + 1]) * 0.5;
                matrix[upper] = avgRe;
                matrix[upper + 1] = avgIm;
                matrix[lower] = avgRe;
                matrix[lower + 1] = avgIm;
            }
        }
    }

    private static void mirrorHermitianLowerToUpper(double[] matrix, int offset, int n) {
        for (int row = 0; row < n; row++) {
            int diag = offset + row * 2 * n + row * 2;
            matrix[diag + 1] = 0.0;
            for (int col = 0; col < row; col++) {
                int lower = offset + row * 2 * n + col * 2;
                int upper = offset + col * 2 * n + row * 2;
                matrix[upper] = matrix[lower];
                matrix[upper + 1] = -matrix[lower + 1];
            }
        }
    }

    private static double squaredNorm(double[] values, int offset, int length) {
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            double v = values[offset + i];
            sum += v * v;
        }
        return sum;
    }

    private static void zero(double[] values, int offset, int length) {
        for (int i = 0; i < length; i++) {
            values[offset + i] = 0.0;
        }
    }

    private static void zeroComplex(double[] values, int offset, int count) {
        if (count <= 0) {
            return;
        }
        zero(values, offset, 2 * count);
    }

    private static void storeStandardizedScalarComplex(double[] target,
                                                       int offset,
                                                       double valueRe,
                                                       double valueIm,
                                                       double covarianceRe,
                                                       double covarianceIm) {
        double covarianceAbs = Math.hypot(covarianceRe, covarianceIm);
        if (covarianceAbs == 0.0) {
            target[offset] = 0.0;
            target[offset + 1] = 0.0;
            return;
        }
        double sqrtRe = Math.sqrt(Math.max(0.0, (covarianceAbs + covarianceRe) * 0.5));
        double sqrtIm = Math.copySign(Math.sqrt(Math.max(0.0, (covarianceAbs - covarianceRe) * 0.5)), covarianceIm);
        double denom = sqrtRe * sqrtRe + sqrtIm * sqrtIm;
        target[offset] = (valueRe * sqrtRe + valueIm * sqrtIm) / denom;
        target[offset + 1] = (valueIm * sqrtRe - valueRe * sqrtIm) / denom;
    }

    private static void solveLowerTriangularInPlace(double[] lower,
                                                    int lowerOffset,
                                                    double[] rhs,
                                                    int rhsOffset,
                                                    int n) {
        for (int row = 0; row < n; row++) {
            double valueRe = rhs[rhsOffset + row * 2];
            double valueIm = rhs[rhsOffset + row * 2 + 1];
            for (int col = 0; col < row; col++) {
                int lowerCell = lowerOffset + row * 2 * n + col * 2;
                int rhsCell = rhsOffset + col * 2;
                double lowerRe = lower[lowerCell];
                double lowerIm = lower[lowerCell + 1];
                double rhsRe = rhs[rhsCell];
                double rhsIm = rhs[rhsCell + 1];
                valueRe -= lowerRe * rhsRe - lowerIm * rhsIm;
                valueIm -= lowerRe * rhsIm + lowerIm * rhsRe;
            }
            int diagonal = lowerOffset + row * 2 * n + row * 2;
            double diagonalRe = lower[diagonal];
            double diagonalIm = lower[diagonal + 1];
            double scale = diagonalRe * diagonalRe + diagonalIm * diagonalIm;
            rhs[rhsOffset + row * 2] = (valueRe * diagonalRe + valueIm * diagonalIm) / scale;
            rhs[rhsOffset + row * 2 + 1] = (valueIm * diagonalRe - valueRe * diagonalIm) / scale;
        }
    }

    private static final class ObservationTransformWorkspace {
        private KalmanSSM transformedModel;
        private double[] design;
        private double[] obsIntercept;
        private double[] obsCov;
        private double[] endog;
        private boolean[] missing;
        private int[] nmissing;
        private int[] active;
        private double[] factor;
        private double[] diagonal;
        private double[] rhs;
        private int kEndog;
        private int kStates;
        private int kPosdef;
        private int nobs;
        private boolean hasMissingObservations;
        private boolean observationBlocksTimeVarying;
        private boolean transitionTimeVarying;
        private boolean stateInterceptTimeVarying;
        private boolean selectionTimeVarying;
        private boolean stateCovarianceTimeVarying;

        KalmanSSM transform(KalmanSSM source, double[] endogOverride, int endogBase) {
            ensure(source);
            bindStateSystem(source);
            copyMissing(source);
            if (observationBlocksTimeVarying) {
                for (int t = 0; t < nobs; t++) {
                    copyObservationBlock(source, endogOverride, endogBase, t);
                    transformObservationBlock(t);
                }
            } else {
                copyObservationBlock(source, endogOverride, endogBase, 0);
                transformObservationBlock(0);
                for (int t = 1; t < nobs; t++) {
                    transformEndogBlock(source, endogOverride, endogBase, t, activeCount(t));
                }
            }
            return transformedModel;
        }

        long retainedDoubleCount() {
            return doubleCount(design)
                + doubleCount(obsIntercept)
                + doubleCount(obsCov)
                + doubleCount(endog)
                + doubleCount(factor)
                + doubleCount(diagonal)
                + doubleCount(rhs);
        }

        long retainedByteCount() {
            return retainedDoubleCount() * Double.BYTES
                + booleanByteCount(missing)
                + intByteCount(nmissing)
                + intByteCount(active);
        }

        void release() {
            transformedModel = null;
            design = null;
            obsIntercept = null;
            obsCov = null;
            endog = null;
            missing = null;
            nmissing = null;
            active = null;
            factor = null;
            diagonal = null;
            rhs = null;
            kEndog = 0;
            kStates = 0;
            kPosdef = 0;
            nobs = 0;
            hasMissingObservations = false;
            observationBlocksTimeVarying = false;
            transitionTimeVarying = false;
            stateInterceptTimeVarying = false;
            selectionTimeVarying = false;
            stateCovarianceTimeVarying = false;
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }

        private static long booleanByteCount(boolean[] values) {
            return values == null ? 0L : values.length;
        }

        private static long intByteCount(int[] values) {
            return values == null ? 0L : values.length * (long) Integer.BYTES;
        }

        private void ensure(KalmanSSM source) {
            int nextKEndog = source.observationDimension();
            int nextKStates = source.stateCount();
            int nextKPosdef = source.stateDisturbanceCount();
            int nextNobs = source.observationCount();
            boolean nextHasMissingObservations = KalmanSSMSupport.hasMissingObservations(source);
            boolean nextObservationBlocksTimeVarying = nextHasMissingObservations
                || isDesignTimeVarying(source)
                || isObsInterceptTimeVarying(source)
                || isObsCovTimeVarying(source);
            boolean nextTransitionTimeVarying = isTransitionTimeVarying(source);
            boolean nextStateInterceptTimeVarying = isStateInterceptTimeVarying(source);
            boolean nextSelectionTimeVarying = isSelectionTimeVarying(source);
            boolean nextStateCovarianceTimeVarying = isStateCovarianceTimeVarying(source);
            if (transformedModel != null
                    && kEndog == nextKEndog
                    && kStates == nextKStates
                    && kPosdef == nextKPosdef
                    && nobs == nextNobs
                    && hasMissingObservations == nextHasMissingObservations
                    && observationBlocksTimeVarying == nextObservationBlocksTimeVarying
                    && transitionTimeVarying == nextTransitionTimeVarying
                    && stateInterceptTimeVarying == nextStateInterceptTimeVarying
                    && selectionTimeVarying == nextSelectionTimeVarying
                    && stateCovarianceTimeVarying == nextStateCovarianceTimeVarying) {
                return;
            }
            kEndog = nextKEndog;
            kStates = nextKStates;
            kPosdef = nextKPosdef;
            nobs = nextNobs;
            hasMissingObservations = nextHasMissingObservations;
            observationBlocksTimeVarying = nextObservationBlocksTimeVarying;
            transitionTimeVarying = nextTransitionTimeVarying;
            stateInterceptTimeVarying = nextStateInterceptTimeVarying;
            selectionTimeVarying = nextSelectionTimeVarying;
            stateCovarianceTimeVarying = nextStateCovarianceTimeVarying;
            int observationBlockCount = observationBlocksTimeVarying ? nobs : 1;
            design = new double[2 * kEndog * kStates * observationBlockCount];
            obsIntercept = new double[2 * kEndog * observationBlockCount];
            obsCov = new double[2 * kEndog * kEndog * observationBlockCount];
            endog = new double[2 * kEndog * nobs];
            missing = hasMissingObservations ? new boolean[kEndog * nobs] : null;
            nmissing = hasMissingObservations ? new int[nobs] : null;
            active = new int[kEndog];
            factor = new double[2 * kEndog * kEndog];
            diagonal = new double[2 * kEndog];
            rhs = new double[2 * kEndog];
            KalmanSSM.Builder builder = KalmanSSM.complexBuilder(kEndog, kStates, kPosdef, nobs)
                .design(design, observationBlocksTimeVarying)
                .obsIntercept(obsIntercept, observationBlocksTimeVarying)
                .obsCov(obsCov, observationBlocksTimeVarying)
                .transition(source.transitionData(), transitionTimeVarying)
                .stateIntercept(source.stateInterceptData(), stateInterceptTimeVarying)
                .selection(source.selectionData(), selectionTimeVarying)
                .stateCovariance(source.stateCovarianceData(), stateCovarianceTimeVarying)
                .endog(endog);
            transformedModel = hasMissingObservations
                ? builder.missing(missing, nmissing).build()
                : builder.allObserved().build();
        }

        private void bindStateSystem(KalmanSSM source) {
            transformedModel.transition = source.transitionData();
            transformedModel.stateIntercept = source.stateInterceptData();
            transformedModel.selection = source.selectionData();
            transformedModel.stateCov = source.stateCovarianceData();
        }

        private void copyMissing(KalmanSSM source) {
            if (!hasMissingObservations) {
                return;
            }
            for (int t = 0; t < nobs; t++) {
                int targetOffset = t * kEndog;
                int count = 0;
                for (int obs = 0; obs < kEndog; obs++) {
                    boolean value = source.isMissing(obs, t);
                    missing[targetOffset + obs] = value;
                    if (value) {
                        count++;
                    }
                }
                nmissing[t] = count;
            }
        }

        private void copyObservationBlock(KalmanSSM source,
                                          double[] endogOverride,
                                          int endogBase,
                                          int t) {
            int designTargetOffset = observationBlockOffset(t, 2 * kEndog * kStates);
            int obsInterceptTargetOffset = observationBlockOffset(t, 2 * kEndog);
            int obsCovTargetOffset = observationBlockOffset(t, 2 * kEndog * kEndog);
            int endogTargetOffset = endogOffset(t);
            copyComplexRectangularBlock(source.designData(), source.designOffset(t), source.designLeadingDimension(),
                kEndog, kStates, design, designTargetOffset, kStates);
            System.arraycopy(source.obsInterceptData(), source.obsInterceptOffset(t),
                obsIntercept, obsInterceptTargetOffset, 2 * kEndog);
            copyComplexRectangularBlock(source.obsCovData(), source.obsCovOffset(t), source.obsCovLeadingDimension(),
                kEndog, kEndog, obsCov, obsCovTargetOffset, kEndog);
            double[] sourceEndog = endogOverride == null ? source.endogData() : endogOverride;
            int sourceEndogOffset = endogOverride == null ? source.endogOffset(t) : endogBase + t * 2 * kEndog;
            System.arraycopy(sourceEndog, sourceEndogOffset, endog, endogTargetOffset, 2 * kEndog);
        }

        private void copyEndogBlock(KalmanSSM source,
                                    double[] endogOverride,
                                    int endogBase,
                                    int t) {
            double[] sourceEndog = endogOverride == null ? source.endogData() : endogOverride;
            int sourceEndogOffset = endogOverride == null ? source.endogOffset(t) : endogBase + t * 2 * kEndog;
            System.arraycopy(sourceEndog, sourceEndogOffset, endog, endogOffset(t), 2 * kEndog);
        }

        private void transformEndogBlock(KalmanSSM source,
                                         double[] endogOverride,
                                         int endogBase,
                                         int t,
                                         int activeCount) {
            double[] sourceEndog = endogOverride == null ? source.endogData() : endogOverride;
            int sourceEndogOffset = endogOverride == null ? source.endogOffset(t) : endogBase + t * 2 * kEndog;
            int targetEndogOffset = endogOffset(t);
            for (int row = 0; row < activeCount; row++) {
                int sourceCell = sourceEndogOffset + active[row] * 2;
                rhs[row * 2] = sourceEndog[sourceCell];
                rhs[row * 2 + 1] = sourceEndog[sourceCell + 1];
            }
            solveLowerInPlace(activeCount, rhs);
            for (int row = 0; row < activeCount; row++) {
                int target = targetEndogOffset + active[row] * 2;
                endog[target] = rhs[row * 2];
                endog[target + 1] = rhs[row * 2 + 1];
            }
        }

        private void transformObservationBlock(int t) {
            int activeCount = activeCount(t);
            if (activeCount <= 1) {
                return;
            }
            int obsCovOffset = observationBlockOffset(t, 2 * kEndog * kEndog);
            if (activeObservationCovarianceIsDiagonal(obsCovOffset, activeCount)) {
                return;
            }
            Arrays.fill(factor, 0.0);
            for (int row = 0; row < activeCount; row++) {
                int sourceRow = active[row];
                for (int col = 0; col < activeCount; col++) {
                    int sourceCol = active[col];
                    int source = obsCovOffset + sourceRow * 2 * kEndog + sourceCol * 2;
                    int target = row * 2 * activeCount + col * 2;
                    factor[target] = obsCov[source];
                    factor[target + 1] = obsCov[source + 1];
                }
            }
            ldlFactor(activeCount, t);
            int currentEndogOffset = endogOffset(t);
            solveActiveVector(endog, currentEndogOffset, activeCount);
            int obsInterceptOffset = observationBlockOffset(t, 2 * kEndog);
            solveActiveVector(obsIntercept, obsInterceptOffset, activeCount);
            int designOffset = observationBlockOffset(t, 2 * kEndog * kStates);
            for (int state = 0; state < kStates; state++) {
                for (int row = 0; row < activeCount; row++) {
                    int src = designOffset + active[row] * 2 * kStates + state * 2;
                    rhs[row * 2] = design[src];
                    rhs[row * 2 + 1] = design[src + 1];
                }
                solveLowerInPlace(activeCount, rhs);
                for (int row = 0; row < activeCount; row++) {
                    int dst = designOffset + active[row] * 2 * kStates + state * 2;
                    design[dst] = rhs[row * 2];
                    design[dst + 1] = rhs[row * 2 + 1];
                }
            }
            Arrays.fill(obsCov, obsCovOffset, obsCovOffset + 2 * kEndog * kEndog, 0.0);
            for (int row = 0; row < activeCount; row++) {
                int obs = active[row];
                int obsCovCell = obsCovOffset + obs * 2 * kEndog + obs * 2;
                int diagonalCell = row * 2;
                obsCov[obsCovCell] = diagonal[diagonalCell];
                obsCov[obsCovCell + 1] = diagonal[diagonalCell + 1];
            }
        }

        private void ldlFactor(int activeCount, int t) {
            double tolerance = 1e-15;
            for (int col = 0; col < activeCount; col++) {
                int diagonalCell = col * 2 * activeCount + col * 2;
                double pivotRe = factor[diagonalCell];
                double pivotIm = factor[diagonalCell + 1];
                if (!(pivotRe > tolerance) || !Double.isFinite(pivotRe) || !Double.isFinite(pivotIm)) {
                    throw new IllegalArgumentException(
                        "UNIVARIATE filtering requires positive definite active observation covariance at t=" + t);
                }

                for (int row = 0; row < col; row++) {
                    int lowerCell = col * 2 * activeCount + row * 2;
                    int previousDiagonalCell = row * 2 * activeCount + row * 2;
                    double lowerRe = factor[lowerCell];
                    double lowerIm = factor[lowerCell + 1];
                    double previousDiagonalRe = factor[previousDiagonalCell];
                    double previousDiagonalIm = factor[previousDiagonalCell + 1];
                    double partialRe = lowerRe * previousDiagonalRe - lowerIm * previousDiagonalIm;
                    double partialIm = lowerRe * previousDiagonalIm + lowerIm * previousDiagonalRe;
                    rhs[row * 2] = partialRe;
                    rhs[row * 2 + 1] = partialIm;
                    pivotRe -= lowerRe * partialRe - lowerIm * partialIm;
                    pivotIm -= lowerRe * partialIm + lowerIm * partialRe;
                }

                factor[diagonalCell] = pivotRe;
                factor[diagonalCell + 1] = pivotIm;
                diagonal[col * 2] = pivotRe;
                diagonal[col * 2 + 1] = pivotIm;
                double pivotScale = pivotRe * pivotRe + pivotIm * pivotIm;
                if (!(pivotScale > 0.0) || !Double.isFinite(pivotScale)) {
                    throw new IllegalArgumentException(
                        "UNIVARIATE filtering requires nonsingular active observation covariance at t=" + t);
                }

                for (int row = col + 1; row < activeCount; row++) {
                    int lowerCell = row * 2 * activeCount + col * 2;
                    double valueRe = factor[lowerCell];
                    double valueIm = factor[lowerCell + 1];
                    for (int k = 0; k < col; k++) {
                        int previousLowerCell = row * 2 * activeCount + k * 2;
                        double lowerRe = factor[previousLowerCell];
                        double lowerIm = factor[previousLowerCell + 1];
                        double partialRe = rhs[k * 2];
                        double partialIm = rhs[k * 2 + 1];
                        valueRe -= lowerRe * partialRe - lowerIm * partialIm;
                        valueIm -= lowerRe * partialIm + lowerIm * partialRe;
                    }
                    factor[lowerCell] = (valueRe * pivotRe + valueIm * pivotIm) / pivotScale;
                    factor[lowerCell + 1] = (valueIm * pivotRe - valueRe * pivotIm) / pivotScale;
                }
            }

            for (int row = 0; row < activeCount; row++) {
                int diagonalCell = row * 2 * activeCount + row * 2;
                factor[diagonalCell] = 1.0;
                factor[diagonalCell + 1] = 0.0;
                for (int col = row + 1; col < activeCount; col++) {
                    int upper = row * 2 * activeCount + col * 2;
                    factor[upper] = 0.0;
                    factor[upper + 1] = 0.0;
                }
            }
        }

        private int activeCount(int t) {
            int activeCount = 0;
            int missingOffset = hasMissingObservations ? t * kEndog : 0;
            for (int obs = 0; obs < kEndog; obs++) {
                if (!hasMissingObservations || !missing[missingOffset + obs]) {
                    active[activeCount++] = obs;
                }
            }
            return activeCount;
        }

        private void solveActiveVector(double[] values, int offset, int activeCount) {
            for (int row = 0; row < activeCount; row++) {
                int source = offset + active[row] * 2;
                rhs[row * 2] = values[source];
                rhs[row * 2 + 1] = values[source + 1];
            }
            solveLowerInPlace(activeCount, rhs);
            for (int row = 0; row < activeCount; row++) {
                int target = offset + active[row] * 2;
                values[target] = rhs[row * 2];
                values[target + 1] = rhs[row * 2 + 1];
            }
        }

        private void solveLowerInPlace(int activeCount, double[] values) {
            for (int row = 0; row < activeCount; row++) {
                double valueRe = values[row * 2];
                double valueIm = values[row * 2 + 1];
                for (int col = 0; col < row; col++) {
                    int lower = row * 2 * activeCount + col * 2;
                    double lowerRe = factor[lower];
                    double lowerIm = factor[lower + 1];
                    double rhsRe = values[col * 2];
                    double rhsIm = values[col * 2 + 1];
                    valueRe -= lowerRe * rhsRe - lowerIm * rhsIm;
                    valueIm -= lowerRe * rhsIm + lowerIm * rhsRe;
                }
                values[row * 2] = valueRe;
                values[row * 2 + 1] = valueIm;
            }
        }

        private boolean activeObservationCovarianceIsDiagonal(int obsCovOffset, int activeCount) {
            for (int row = 0; row < activeCount; row++) {
                int sourceRow = active[row];
                for (int col = 0; col < activeCount; col++) {
                    if (row == col) {
                        continue;
                    }
                    int sourceCol = active[col];
                    int off = obsCovOffset + sourceRow * 2 * kEndog + sourceCol * 2;
                    if (obsCov[off] != 0.0 || obsCov[off + 1] != 0.0) {
                        return false;
                    }
                }
            }
            return true;
        }

        private int observationBlockOffset(int t, int blockLength) {
            return (observationBlocksTimeVarying ? t : 0) * blockLength;
        }

        private int endogOffset(int t) {
            return t * 2 * kEndog;
        }

        private static boolean isTransitionTimeVarying(KalmanSSM source) {
            return source.observationCount() > 1 && source.transitionOffset(1) != source.transitionOffset(0);
        }

        private static boolean isDesignTimeVarying(KalmanSSM source) {
            return source.observationCount() > 1 && source.designOffset(1) != source.designOffset(0);
        }

        private static boolean isObsInterceptTimeVarying(KalmanSSM source) {
            return source.observationCount() > 1 && source.obsInterceptOffset(1) != source.obsInterceptOffset(0);
        }

        private static boolean isObsCovTimeVarying(KalmanSSM source) {
            return source.observationCount() > 1 && source.obsCovOffset(1) != source.obsCovOffset(0);
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
    }

    private static void copyComplexRectangularBlock(double[] source,
                                                    int sourceOffset,
                                                    int sourceLeadingDimension,
                                                    int rows,
                                                    int cols,
                                                    double[] target,
                                                    int targetOffset,
                                                    int targetLeadingDimension) {
        for (int row = 0; row < rows; row++) {
            System.arraycopy(source, sourceOffset + row * 2 * sourceLeadingDimension,
                target, targetOffset + row * 2 * targetLeadingDimension, 2 * cols);
        }
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
        int ke = 0;
        for (int i = 0; i < kEndog; i++) {
            if (!observationModel.isMissing(i, t)) {
                pool.selectedIndex[ke++] = i;
            }
        }

        double[] design = observationModel.designData();
        int designOff = observationModel.designOffset(t);
        int designLd = observationModel.designLeadingDimension();
        int selectedDesignBase = pool.selectedDesignBase();
        double[] obsIntercept = observationModel.obsInterceptData();
        int obsInterceptOff = observationModel.obsInterceptOffset(t);
        int selectedObsInterceptBase = pool.selectedObsInterceptBase();
        double[] obsCov = observationModel.obsCovData();
        int obsCovOff = observationModel.obsCovOffset(t);
        int obsCovLd = observationModel.obsCovLeadingDimension();
        int selectedObsCovBase = pool.selectedObsCovBase();
        int selectedEndogBase = pool.selectedEndogBase();

        for (int row = 0; row < ke; row++) {
            int source = pool.selectedIndex[row];
            System.arraycopy(design, designOff + source * 2 * designLd,
                pool.selectionBacking, selectedDesignBase + row * 2 * kStates, 2 * kStates);
            System.arraycopy(obsIntercept, obsInterceptOff + source * 2,
                pool.selectionBacking, selectedObsInterceptBase + row * 2, 2);
            System.arraycopy(endog, endogOff + source * 2,
                pool.selectionBacking, selectedEndogBase + row * 2, 2);
        }

        for (int row = 0; row < ke; row++) {
            int sourceRow = pool.selectedIndex[row];
            int sourceRowOff = obsCovOff + sourceRow * 2 * obsCovLd;
            int targetRowOff = selectedObsCovBase + row * 2 * ke;
            for (int col = 0; col < ke; col++) {
                int sourceCol = pool.selectedIndex[col];
                System.arraycopy(obsCov, sourceRowOff + sourceCol * 2,
                        pool.selectionBacking, targetRowOff + col * 2, 2);
            }
        }
        return ke;
    }

}
