package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.arena.FilterDiffuseLayout;
import com.curioloop.yum4j.ssm.kalman.arena.FilterResultLayout;
import com.curioloop.yum4j.ssm.kalman.arena.UnivariateScratchLayout;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;
import java.util.Objects;

public final class UnivariateFilter {

    private static final double DEFAULT_DIFFUSE_TOLERANCE = FilterOptions.DEFAULT_DIFFUSE_TOLERANCE;

    static final class Pool {
        private final DoubleArena scratchArena;
        private final DoubleArena resultArena;
        private final boolean ownsScratchArena;
        private final boolean ownsResultArena;
        public double[] scratchBacking;
        public UnivariateScratchLayout scratchLayout;
        public double[] resultBacking;
        public double[] diffuseResultBacking;
        public FilterResultLayout resultLayout;
        public FilterDiffuseLayout diffuseResultLayout;
        private int diffuseResultBase;

        private InitialState.StationaryWorkspace stationaryWorkspace;
        private ObservationTransformWorkspace observationTransformWorkspace;
        private FilterResult pooledResult;
        private int scratchKStates;
        private int scratchKEndog;
        private int scratchKPosdef;
        private boolean scratchPredicted;
        private boolean scratchFiltered;
        private boolean scratchDiffusePredicted;
        private boolean scratchExactDiffuse;

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
            ensureScratchArena(kStates, kEndog, kPosdef, false, false, false, false);
        }

        void ensurePooledScratch(int kStates,
                                 int kEndog,
                                 int kPosdef,
                                 boolean needPredictedScratch,
                                 boolean needFilteredScratch,
                                 boolean needDiffusePredictedScratch) {
            ensureScratchArena(kStates, kEndog,
                kPosdef, needPredictedScratch, needFilteredScratch, needDiffusePredictedScratch, false);
        }

        void ensureExactDiffuseScratch(int kStates,
                                       int kEndog,
                                       int kPosdef,
                                       boolean needPredictedScratch,
                                       boolean needFilteredScratch,
                                       boolean needDiffusePredictedScratch) {
            ensureScratchArena(kStates, kEndog,
                kPosdef, needPredictedScratch, needFilteredScratch, needDiffusePredictedScratch, true);
        }

        private void ensureScratchArena(int kStates,
                                        int kEndog,
                                        int kPosdef,
                                        boolean needPredictedScratch,
                                        boolean needFilteredScratch,
                                        boolean needDiffusePredictedScratch,
                                        boolean exactDiffuse) {
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
                    || nextScratchDiffusePredicted != scratchDiffusePredicted
                    || exactDiffuse != scratchExactDiffuse) {
                scratchLayout = UnivariateScratchLayout.create(nextScratchKEndog,
                    nextScratchKStates,
                    nextScratchKPosdef,
                    exactDiffuse,
                    nextScratchPredicted,
                    nextScratchFiltered,
                    nextScratchDiffusePredicted);
                scratchKStates = nextScratchKStates;
                scratchKEndog = nextScratchKEndog;
                scratchKPosdef = nextScratchKPosdef;
                scratchPredicted = nextScratchPredicted;
                scratchFiltered = nextScratchFiltered;
                scratchDiffusePredicted = nextScratchDiffusePredicted;
                scratchExactDiffuse = exactDiffuse;
            }
            int totalLength = scratchLayout.totalLength();
            scratchBacking = scratchArena.ensureCapacity(totalLength);
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

        public void reserve(KalmanSSM model) {
            reserve(model, null);
        }

        public void reserve(KalmanSSM model, FilterOptions spec) {
            Objects.requireNonNull(model, "model");
            requireRealInputs(model);
            FilterOptions resolvedOptions = spec == null ? FilterOptions.defaults() : spec;
            FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(
                1,
                model.observationDimension(),
                model.stateCount(),
                model.observationCount(),
                hasMissingObservations(model));
            ensurePooledScratch(model.stateCount(),
                model.observationDimension(),
                model.stateDisturbanceCount(),
                retainedLayout.predictedStateLength() == 0,
                false,
                false);
            ensureResult(retainedLayout);
        }

        public long retainedScratchDoubleCount() {
            return (usesScratchArena() ? scratchArena.retainedDoubleCount() : 0L)
                + retainedAuxiliaryScratchDoubleCount();
        }

        long retainedAuxiliaryScratchDoubleCount() {
            return (stationaryWorkspace == null ? 0L : stationaryWorkspace.retainedDoubleCount())
                + (observationTransformWorkspace == null ? 0L : observationTransformWorkspace.retainedDoubleCount());
        }

        boolean usesScratchArena() {
            return scratchBacking != null;
        }

        void refreshScratchBacking() {
            if (scratchLayout != null) {
                scratchBacking = scratchArena.backing();
            }
        }

        long retainedObservationTransformDoubleCount() {
            return observationTransformWorkspace == null ? 0L : observationTransformWorkspace.retainedDoubleCount();
        }

        public long retainedResultDoubleCount() {
            return resultArena.retainedDoubleCount();
        }

        public long retainedTotalByteCount() {
            long scratchDoubleCount = doubleCount(scratchBacking)
                + (stationaryWorkspace == null ? 0L : stationaryWorkspace.retainedDoubleCount());
            long observationTransformBytes = observationTransformWorkspace == null
                ? 0L
                : observationTransformWorkspace.retainedByteCount();
            return (scratchDoubleCount + retainedResultDoubleCount()) * Double.BYTES
                + observationTransformBytes;
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }

        public void releaseRetainedScratch() {
            scratchBacking = null;
            if (ownsScratchArena) {
                scratchArena.release();
            }
            scratchLayout = null;
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
            scratchExactDiffuse = false;
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
        int tmpPBase() { return scratchLayout.tmpPBase(); }
        int tmpPZiBase() { return scratchLayout.tmpPZiBase(); }
        int tmpRQBase() { return scratchLayout.tmpRQBase(); }
        int tmpPInfBase() { return scratchLayout.tmpPInfBase(); }
        int tmpPStarBase() { return scratchLayout.tmpPStarBase(); }
        int tmpTPBase() { return scratchLayout.tmpTPBase(); }
        int tmpMStarBase() { return scratchLayout.tmpMStarBase(); }
        int tmpMInfBase() { return scratchLayout.tmpMInfBase(); }
        int tmpK0Base() { return scratchLayout.tmpK0Base(); }
        int tmpK1Base() { return scratchLayout.tmpK1Base(); }
        int rollingPredictedStateBase() { return scratchLayout.rollingPredictedStateBase(); }
        int rollingPredictedStateCovBase() { return scratchLayout.rollingPredictedStateCovBase(); }
        int rollingPredictedDiffuseStateCovBase() { return scratchLayout.rollingPredictedDiffuseStateCovBase(); }
        int scratchFilteredStateBase() { return scratchLayout.scratchFilteredStateBase(); }
        int scratchFilteredStateCovBase() { return scratchLayout.scratchFilteredStateCovBase(); }
    }

    private record PreparedObservationModel(KalmanSSM model,
                                            double[] endogOverride,
                                            int endogBase,
                                            boolean transformed) {
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState init,
                               Pool pool,
                               FilterOptions spec) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        requireRealInputs(model);
        FilterOptions resolvedOptions = spec == null ? FilterOptions.defaults() : spec;
        boolean hasMissingObservations = hasMissingObservations(model);
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
        Pool actualPool = pool == null ? new Pool() : pool;
        PreparedObservationModel preparedModel = prepareObservationModel(model, null, 0, actualPool);
        try {
            FilterResult result = filterInternal(preparedModel.model(), init, actualPool, retainedLayout, retainedDiffuseLayout,
                resolvedOptions.diffuseTolerance(), resolvedOptions.concentratedLikelihood(),
                resolvedOptions.concentratedLikelihoodBurn(),
                preparedModel.endogOverride(), preparedModel.endogBase());
            result.applyConserveMemory(resolvedOptions.conserveMemoryMask());
            return result;
        } finally {
            if (preparedModel.transformed()) {
                actualPool.releaseObservationTransformWorkspace();
            }
        }
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
        boolean hasMissingObservations = hasMissingObservations(model);
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
        Pool actualPool = pool == null ? new Pool() : pool;
        PreparedObservationModel preparedModel = prepareObservationModel(model, endog, endogBase, actualPool);
        try {
            FilterResult result = filterInternal(preparedModel.model(), init, actualPool, retainedLayout, retainedDiffuseLayout,
                resolvedOptions.diffuseTolerance(), resolvedOptions.concentratedLikelihood(),
                resolvedOptions.concentratedLikelihoodBurn(),
                preparedModel.endogOverride(), preparedModel.endogBase());
            result.applyConserveMemory(resolvedOptions.conserveMemoryMask());
            return result;
        } finally {
            if (preparedModel.transformed()) {
                actualPool.releaseObservationTransformWorkspace();
            }
        }
    }

    private static PreparedObservationModel prepareObservationModel(KalmanSSM model,
                                                                    double[] endogOverride,
                                                                    int endogBase,
                                                                    Pool pool) {
        if (!KalmanSSMSupport.hasNonDiagonalObservationCovariance(model)) {
            return new PreparedObservationModel(model, endogOverride, endogBase, false);
        }
        KalmanSSM transformed = pool.observationTransformWorkspace()
            .transform(model, endogOverride, endogBase);
        return new PreparedObservationModel(transformed, null, 0, true);
    }

    private static FilterResult filterInternal(KalmanSSM model,
                                               InitialState init,
                                               Pool pool,
                               FilterResultLayout retainedLayout,
                                               FilterDiffuseLayout retainedDiffuseLayout,
                                               double diffuseTolerance,
                                               boolean concentratedLikelihood,
                                               int concentratedLikelihoodBurn,
                                               double[] endogOverride,
                                               int endogBase) {
        KalmanSSM stateSpace = model;
        KalmanSSM observationModel = model;
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
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
        int obsCovLd = observationModel.obsCovLeadingDimension();
        double[] endog = endogOverride != null ? endogOverride : observationModel.endogData();

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
        boolean diffuseInit = init.mayResolveDiffuse();
        boolean usePredictedScratch = !storePredictedState || !storePredictedCovariance;
        boolean useFilteredScratch = !storeFilteredState || !storeFilteredCovariance;

        if (pool == null) pool = new Pool();
        if (diffuseInit) {
            pool.ensureExactDiffuseScratch(kStates,
                kEndog,
                kPosdef,
                usePredictedScratch,
                useFilteredScratch,
                usePredictedScratch);
        } else {
            pool.ensurePooledScratch(kStates,
                kEndog,
                kPosdef,
                usePredictedScratch,
                useFilteredScratch,
                false);
        }
        double[] scratch = pool.scratchBacking;
        int tmpABase = pool.tmpABase();
        int tmpPBase = pool.tmpPBase();
        int tmpPZiBase = pool.tmpPZiBase();
        int tmpRQBase = pool.tmpRQBase();

        if (diffuseInit) {
            FilterResult result = pool.borrowDiffuseResult(kEndog, kStates, nobs,
                retainedLayout, Objects.requireNonNull(retainedDiffuseLayout, "retainedDiffuseLayout"));
            result.clearActiveSurfaces();
            return filterDiffuse(model, init, pool,
                result,
                diffuseTolerance,
                concentratedLikelihood,
                concentratedLikelihoodBurn,
                endogOverride, endogBase,
                storePredictedState ? null : scratch,
                storePredictedState ? 0 : pool.rollingPredictedStateBase(),
                storePredictedCovariance ? null : scratch,
                storePredictedCovariance ? 0 : pool.rollingPredictedStateCovBase(),
                usePredictedScratch ? scratch : null,
                usePredictedScratch ? pool.rollingPredictedDiffuseStateCovBase() : 0,
                storeFilteredState ? null : scratch,
                storeFilteredState ? 0 : pool.scratchFilteredStateBase(),
                storeFilteredCovariance ? null : scratch,
                storeFilteredCovariance ? 0 : pool.scratchFilteredStateCovBase());
        }

            InitialState.StationaryWorkspace stationaryWorkspace = null;
            if (init.mayResolveStationary()) {
                pool.ensureStationary(model);
                stationaryWorkspace = pool.stationaryWorkspace;
            }

        FilterResult result = pool.borrowResult(kEndog, kStates, nobs, retainedLayout);
        result.clearActiveSurfaces();
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
        double[] predictedState = storePredictedState ? result.predictedState : scratch;
        double[] predictedStateCov = storePredictedCovariance ? result.predictedStateCov : scratch;
        int predictedStateBase = storePredictedState ? result.predictedStateBase() : pool.rollingPredictedStateBase();
        int predictedStateCovBase = storePredictedCovariance ? result.predictedStateCovBase() : pool.rollingPredictedStateCovBase();

        init.resolveInto(model, 0, stationaryWorkspace,
            predictedState, predictedStateBase,
            predictedStateCov, predictedStateCovBase,
            null, 0);

        double concentratedScaleSum = 0.0;
        int concentratedScaleObservationCount = 0;
        double concentratedTotalScaleSum = 0.0;
        int concentratedTotalObservationCount = 0;
        double logLikelihoodSum = 0.0;
        int concentratedScaleStart = Math.max(0, concentratedLikelihoodBurn);

        for (int t = 0; t < nobs; t++) {
            int predOff = storePredictedState ? result.predictedStateOffset(t) : predictedStateBase + (t & 1) * kStates;
            int predCovOff = storePredictedCovariance ? result.predictedStateCovOffset(t) : predictedStateCovBase + (t & 1) * kStates2;
            int filtOff = storeFilteredState ? result.filteredStateOffset(t) : pool.scratchFilteredStateBase();
            int filtCovOff = storeFilteredCovariance ? result.filteredStateCovOffset(t) : pool.scratchFilteredStateCovBase();
            int foreOff = storeForecastMean ? result.forecastOffset(t) : 0;
            int foreErrOff = storeForecastError ? result.forecastErrorOffset(t) : 0;
            int foreErrCovOff = storeForecastCovariance ? result.forecastErrorCovOffset(t) : 0;
            int stdForeErrOff = storeStandardizedForecastError ? result.standardizedForecastErrorOffset(t) : 0;
            int nextPredOff = storePredictedState ? result.predictedStateOffset(t + 1) : predictedStateBase + ((t + 1) & 1) * kStates;
            int nextPredCovOff = storePredictedCovariance ? result.predictedStateCovOffset(t + 1) : predictedStateCovBase + ((t + 1) & 1) * kStates2;
            int kgOff = storeGain ? result.kalmanGainOffset(t) : 0;
            int endogOff = endogOverride != null ? endogBase + t * kEndog : observationModel.endogOffset(t);

            BLAS.dcopy(kStates, predictedState, predOff, 1, scratch, tmpABase, 1);
            BLAS.dcopy(kStates2, predictedStateCov, predCovOff, 1, scratch, tmpPBase, 1);

            double logLik = 0.0;
            double periodScaleContribution = 0.0;
            int periodScaleObservationCount = 0;

            if (storeForecastMean) {
                for (int i = 0; i < kEndog; i++) {
                    result.forecast[foreOff + i] = 0.0;
                }
            }
            if (storeForecastError) {
                for (int i = 0; i < kEndog; i++) {
                    result.forecastError[foreErrOff + i] = 0.0;
                }
            }
            if (storeForecastCovariance) {
                for (int i = 0; i < kEndog * kEndog; i++) {
                    result.forecastErrorCov[foreErrCovOff + i] = 0.0;
                }
            }
            if (storeStandardizedForecastError) {
                for (int i = 0; i < kEndog; i++) {
                    result.standardizedForecastError[stdForeErrOff + i] = 0.0;
                }
            }
            if (storeGain) {
                for (int i = 0; i < kStates * kEndog; i++) {
                    result.kalmanGain[kgOff + i] = 0.0;
                }
            }

            if (observationModel.missingCount(t) == kEndog && storeForecast) {
                storeMissingForecastBlock(
                    observationModel,
                    design,
                    designLd,
                    obsIntercept,
                    obsCov,
                    obsCovLd,
                    t,
                    kEndog,
                    kStates,
                    predictedState,
                    predOff,
                    predictedStateCov,
                    predCovOff,
                    null,
                    0,
                    false,
                    result,
                    storeForecastMean,
                    storeForecastError,
                    storeForecastCovariance,
                    storeStandardizedForecastError,
                    false,
                    foreOff,
                    foreErrOff,
                    foreErrCovOff,
                    stdForeErrOff,
                    0);
            }

            for (int i = 0; i < kEndog; i++) {
                if (observationModel.isMissing(i, t)) {
                    continue;
                }

                // Scalar update for one observation at a time:
                // v_{t,i} = y_{t,i} - d_{t,i} - Z_{t,i} a_t
                // F_{t,i} = H_{t,ii} + Z_{t,i} P_t Z_{t,i}'
                // a_t <- a_t + P_t Z_{t,i}' F_{t,i}^{-1} v_{t,i}
                int zRowOff = observationModel.designOffset(t) + i * designLd;

                double vi = endog[endogOff + i]
                        - obsIntercept[observationModel.obsInterceptOffset(t) + i];
                vi -= BLAS.ddot(kStates, design, zRowOff, 1, scratch, tmpABase, 1);
                if (storeForecastMean) {
                    result.forecast[foreOff + i] = endog[endogOff + i] - vi;
                }
                if (storeForecastError) {
                    result.forecastError[foreErrOff + i] = vi;
                }

                BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                    scratch, tmpPBase, kStates,
                    design, zRowOff, 1,
                    0.0, scratch, tmpPZiBase, 1);

                double Fi = obsCov[observationModel.obsCovOffset(t) + i * obsCovLd + i];
                Fi += BLAS.ddot(kStates, design, zRowOff, 1, scratch, tmpPZiBase, 1);

                if (Fi <= diffuseTolerance) {
                    continue;
                }
                if (storeForecastCovariance) {
                    result.forecastErrorCov[foreErrCovOff + i * kEndog + i] = Fi;
                }
                if (storeStandardizedForecastError) {
                    result.standardizedForecastError[stdForeErrOff + i] = ForecastErrorSolver.standardizeScalar(vi, Fi, 0.0);
                }

                double FiInv = 1.0 / Fi;
                double quadratic = vi * vi * FiInv;
                if (storeGain) {
                    for (int j = 0; j < kStates; j++) {
                        result.kalmanGain[kgOff + j * kEndog + i] = scratch[tmpPZiBase + j] * FiInv;
                    }
                }

                BLAS.daxpy(kStates, FiInv * vi, scratch, tmpPZiBase, 1, scratch, tmpABase, 1);

                BLAS.dger(kStates, kStates, -FiInv,
                    scratch, tmpPZiBase, 1,
                    scratch, tmpPZiBase, 1,
                    scratch, tmpPBase, kStates);

                symmetrize(scratch, tmpPBase, kStates);

                if (concentratedLikelihood) {
                    logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(Fi));
                    periodScaleContribution += quadratic;
                    periodScaleObservationCount++;
                } else {
                    logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(Fi) + quadratic);
                }
            }

            BLAS.dcopy(kStates, scratch, tmpABase, 1,
                storeFilteredState ? result.filteredState : scratch, filtOff, 1);
            BLAS.dcopy(kStates2, scratch, tmpPBase, 1,
                storeFilteredCovariance ? result.filteredStateCov : scratch, filtCovOff, 1);

            if (storeLikelihood) {
                result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] = logLik;
            } else {
                logLikelihoodSum += logLik;
            }
            if (concentratedLikelihood) {
                concentratedTotalScaleSum += periodScaleContribution;
                concentratedTotalObservationCount += periodScaleObservationCount;
                if (t >= concentratedScaleStart) {
                    concentratedScaleSum += periodScaleContribution;
                    concentratedScaleObservationCount += periodScaleObservationCount;
                }
                if (storeLikelihood) {
                    result.scaleObs[t] = periodScaleContribution;
                    result.scaleObsCount[t] = periodScaleObservationCount;
                }
            }

            BLAS.dcopy(kStates, stateIntercept, stateSpace.stateInterceptOffset(t), 1,
                    predictedState, nextPredOff, 1);
            BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                    scratch, tmpABase, 1,
                    1.0, predictedState, nextPredOff, 1);

            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                    scratch, tmpPBase, kStates,
                    0.0, scratch, tmpRQBase, kStates);
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                    1.0, scratch, tmpRQBase, kStates,
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

            symmetrize(predictedStateCov, nextPredCovOff, kStates);
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
            KalmanFilter.applyConcentratedScale(observationModel, result, scale, storeLikelihood);
        }

        if (!storeLikelihood) {
            result.setLogLikelihood(logLikelihoodSum);
        }

        return result;
    }

    static FilterResult filterDiffuse(KalmanSSM model,
                                      InitialState init,
                                      Pool pool,
                                      FilterResult result) {
        return filterDiffuse(model, init, pool, result,
            DEFAULT_DIFFUSE_TOLERANCE,
            false,
            0,
            null, 0,
                null, 0,
                null, 0,
                null, 0,
                null, 0,
                null, 0);
    }

    static FilterResult filterDiffuse(KalmanSSM model,
                                      InitialState init,
                                      Pool pool,
                                      FilterResult result,
                                      double diffuseTolerance,
                                      boolean concentratedLikelihood,
                                      int concentratedLikelihoodBurn,
                                      double[] endogOverride,
                                      int endogBase,
                                      double[] predictedStateScratch,
                                      int predictedStateScratchBase,
                                      double[] predictedStateCovScratch,
                                      int predictedStateCovScratchBase,
                                      double[] predictedDiffuseStateCovScratch,
                                      int predictedDiffuseStateCovScratchBase,
                                      double[] filteredStateScratch,
                                      int filteredStateScratchBase,
                                      double[] filteredStateCovScratch,
                                      int filteredStateCovScratchBase) {
        KalmanSSM stateSpace = model;
        KalmanSSM observationModel = model;
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
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
        int obsCovLd = observationModel.obsCovLeadingDimension();
        double[] endog = endogOverride != null ? endogOverride : observationModel.endogData();
        if (pool == null) {
            pool = new Pool();
        }
        InitialState.StationaryWorkspace stationaryWorkspace = null;
        if (init.mayResolveStationary()) {
            pool.ensureStationary(model);
            stationaryWorkspace = pool.stationaryWorkspace;
        }
        boolean storeForecastMean = result.forecastLength() > 0;
        boolean storeForecastError = result.forecastErrorLength() > 0;
        boolean storeForecastCovariance = result.forecastErrorCovLength() > 0;
        boolean storeStandardizedForecastError = result.standardizedForecastErrorLength() > 0;
        boolean storeForecast = storeForecastMean || storeForecastError || storeForecastCovariance || storeStandardizedForecastError;
        boolean storePredictedState = result.predictedStateLength() > 0;
        boolean storePredictedCovariance = result.predictedStateCovLength() > 0;
        boolean storePredictedDiffuseCovariance = result.predictedDiffuseStateCovLength() > 0;
        boolean storeFilteredState = result.filteredStateLength() > 0;
        boolean storeFilteredCovariance = result.filteredStateCovLength() > 0;
        boolean storeLikelihood = result.logLikelihoodObsLength() > 0;
        boolean storeGain = result.kalmanGainLength() > 0;

        boolean needPredictedScratch = (!storePredictedState && !hasScratch(predictedStateScratch, predictedStateScratchBase, kStates * 2))
            || (!storePredictedCovariance && !hasScratch(predictedStateCovScratch, predictedStateCovScratchBase, kStates2 * 2));
        boolean needDiffusePredictedScratch = !storePredictedDiffuseCovariance
            && !hasScratch(predictedDiffuseStateCovScratch, predictedDiffuseStateCovScratchBase, kStates2 * 2);
        boolean needFilteredScratch = (!storeFilteredState && !hasScratch(filteredStateScratch, filteredStateScratchBase, kStates))
            || (!storeFilteredCovariance && !hasScratch(filteredStateCovScratch, filteredStateCovScratchBase, kStates2));
        pool.ensureExactDiffuseScratch(kStates,
            kEndog,
            kPosdef,
            needPredictedScratch,
            needFilteredScratch,
            needDiffusePredictedScratch);
        double[] scratch = pool.scratchBacking;
        if (!storePredictedState || !storePredictedCovariance || !storePredictedDiffuseCovariance) {
            if (!hasScratch(predictedStateScratch, predictedStateScratchBase, kStates * 2)) {
                predictedStateScratch = scratch;
                predictedStateScratchBase = pool.rollingPredictedStateBase();
            }
            if (!hasScratch(predictedStateCovScratch, predictedStateCovScratchBase, kStates2 * 2)) {
                predictedStateCovScratch = scratch;
                predictedStateCovScratchBase = pool.rollingPredictedStateCovBase();
            }
            if (!hasScratch(predictedDiffuseStateCovScratch, predictedDiffuseStateCovScratchBase, kStates2 * 2)) {
                predictedDiffuseStateCovScratch = scratch;
                predictedDiffuseStateCovScratchBase = pool.rollingPredictedDiffuseStateCovBase();
            }
        }
        if (!storeFilteredState || !storeFilteredCovariance) {
            if (!hasScratch(filteredStateScratch, filteredStateScratchBase, kStates)) {
                filteredStateScratch = scratch;
                filteredStateScratchBase = pool.scratchFilteredStateBase();
            }
            if (!hasScratch(filteredStateCovScratch, filteredStateCovScratchBase, kStates2)) {
                filteredStateCovScratch = scratch;
                filteredStateCovScratchBase = pool.scratchFilteredStateCovBase();
            }
        }
        int tmpABase = pool.tmpABase();
        int tmpRQBase = pool.tmpRQBase();
        int tmpPInfBase = pool.tmpPInfBase();
        int tmpPStarBase = pool.tmpPStarBase();
        int tmpTPBase = pool.tmpTPBase();
        int tmpMStarBase = pool.tmpMStarBase();
        int tmpMInfBase = pool.tmpMInfBase();
        int tmpK0Base = pool.tmpK0Base();
        int tmpK1Base = pool.tmpK1Base();

        boolean storeForecastDiffuseCovariance = result.forecastErrorDiffuseCovLength() > 0
            || result.forecastErrorDiffuseCov == null && storeForecast;
        result.allocateDiffuse(storePredictedDiffuseCovariance, storeForecastDiffuseCovariance);
        storeForecastDiffuseCovariance = result.forecastErrorDiffuseCovLength() > 0;
        result.converged = false;
        result.periodConverged = nobs;
        result.nobsDiffuse = 0;
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
            : predictedStateScratch;
        double[] predictedStateCov = storePredictedCovariance
            ? result.predictedStateCov
            : predictedStateCovScratch;
        double[] predictedDiffuseStateCov = storePredictedDiffuseCovariance
            ? result.predictedDiffuseStateCov
            : predictedDiffuseStateCovScratch;
        double[] filteredState = storeFilteredState
            ? result.filteredState
            : filteredStateScratch;
        double[] filteredStateCov = storeFilteredCovariance
            ? result.filteredStateCov
            : filteredStateCovScratch;
        int predictedStateBase = storePredictedState
            ? result.predictedStateBase()
            : predictedStateScratchBase;
        int predictedStateCovBase = storePredictedCovariance
            ? result.predictedStateCovBase()
            : predictedStateCovScratchBase;
        int predictedDiffuseStateCovBase = storePredictedDiffuseCovariance
            ? result.predictedDiffuseStateCovBase()
            : predictedDiffuseStateCovScratchBase;
        int filteredStateBase = storeFilteredState
            ? result.filteredStateBase()
            : filteredStateScratchBase;
        int filteredStateCovBase = storeFilteredCovariance
            ? result.filteredStateCovBase()
            : filteredStateCovScratchBase;

        if (!storePredictedState) {
            Arrays.fill(predictedState, predictedStateBase, predictedStateBase + kStates * 2, 0.0);
        }
        if (!storePredictedCovariance) {
            Arrays.fill(predictedStateCov, predictedStateCovBase, predictedStateCovBase + kStates2 * 2, 0.0);
        }
        if (!storePredictedDiffuseCovariance) {
            Arrays.fill(predictedDiffuseStateCov, predictedDiffuseStateCovBase, predictedDiffuseStateCovBase + kStates2 * 2, 0.0);
        }
        Arrays.fill(predictedState, predictedStateBase, predictedStateBase + kStates, 0.0);
        Arrays.fill(predictedStateCov, predictedStateCovBase, predictedStateCovBase + kStates2, 0.0);
        Arrays.fill(predictedDiffuseStateCov, predictedDiffuseStateCovBase, predictedDiffuseStateCovBase + kStates2, 0.0);
        boolean diffuse = init.resolveInto(model, 0, stationaryWorkspace,
            predictedState, predictedStateBase,
            predictedStateCov, predictedStateCovBase,
            predictedDiffuseStateCov, predictedDiffuseStateCovBase);
        int nobsDiffuse = diffuse ? 1 : 0;
        int kEndog2 = kEndog * kEndog;
        double concentratedScaleSum = 0.0;
        int concentratedScaleObservationCount = 0;
        int concentratedScaleStart = Math.max(0, concentratedLikelihoodBurn);

        for (int t = 0; t < nobs; t++) {
            int predOff = storePredictedState ? result.predictedStateOffset(t) : predictedStateBase + (t & 1) * kStates;
            int predCovOff = storePredictedCovariance ? result.predictedStateCovOffset(t) : predictedStateCovBase + (t & 1) * kStates2;
            int predDiffCovOff = storePredictedDiffuseCovariance ? result.predictedDiffuseStateCovOffset(t) : predictedDiffuseStateCovBase + (t & 1) * kStates2;
            int filtOff = storeFilteredState ? result.filteredStateOffset(t) : filteredStateBase;
            int filtCovOff = storeFilteredCovariance ? result.filteredStateCovOffset(t) : filteredStateCovBase;
            int foreOff = storeForecastMean ? result.forecastOffset(t) : 0;
            int foreErrOff = storeForecastError ? result.forecastErrorOffset(t) : 0;
            int foreErrCovOff = storeForecastCovariance ? result.forecastErrorCovOffset(t) : 0;
            int stdForeErrOff = storeStandardizedForecastError ? result.standardizedForecastErrorOffset(t) : 0;
            int foreErrDiffCovOff = storeForecastDiffuseCovariance ? result.forecastErrorDiffuseCovOffset(t) : 0;
            int nextPredOff = storePredictedState ? result.predictedStateOffset(t + 1) : predictedStateBase + ((t + 1) & 1) * kStates;
            int nextPredCovOff = storePredictedCovariance ? result.predictedStateCovOffset(t + 1) : predictedStateCovBase + ((t + 1) & 1) * kStates2;
            int nextPredDiffCovOff = storePredictedDiffuseCovariance ? result.predictedDiffuseStateCovOffset(t + 1) : predictedDiffuseStateCovBase + ((t + 1) & 1) * kStates2;
            int kgOff = storeGain ? result.kalmanGainOffset(t) : 0;
            int endogOff = endogOverride != null ? endogBase + t * kEndog : observationModel.endogOffset(t);

            int ke = kEndog - observationModel.missingCount(t);

            if (ke == 0) {
                if (storeForecast) {
                    storeMissingForecastBlock(
                        observationModel,
                        design,
                        designLd,
                        obsIntercept,
                        obsCov,
                        obsCovLd,
                        t,
                        kEndog,
                        kStates,
                        predictedState,
                        predOff,
                        predictedStateCov,
                        predCovOff,
                        diffuse ? predictedDiffuseStateCov : null,
                        predDiffCovOff,
                        diffuse,
                        result,
                        storeForecastMean,
                        storeForecastError,
                        storeForecastCovariance,
                        storeStandardizedForecastError,
                        storeForecastDiffuseCovariance,
                        foreOff,
                        foreErrOff,
                        foreErrCovOff,
                        stdForeErrOff,
                        foreErrDiffCovOff);
                }
                BLAS.dcopy(kStates, predictedState, predOff, 1,
                    filteredState, filtOff, 1);
                BLAS.dcopy(kStates2, predictedStateCov, predCovOff, 1,
                    filteredStateCov, filtCovOff, 1);
                if (storeLikelihood) {
                    result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] = 0.0;
                }
                if (storeGain) {
                    for (int i = 0; i < kStates * kEndog; i++) {
                        result.kalmanGain[kgOff + i] = 0.0;
                    }
                }

                BLAS.dcopy(kStates, stateIntercept, stateSpace.stateInterceptOffset(t), 1,
                        predictedState, nextPredOff, 1);
                BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                        transition, stateSpace.transitionOffset(t), transitionLd,
                        predictedState, predOff, 1,
                        1.0, predictedState, nextPredOff, 1);

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

                if (diffuse) {
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                            predictedDiffuseStateCov, predDiffCovOff, kStates,
                        0.0, scratch, tmpTPBase, kStates);
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                        1.0, scratch, tmpTPBase, kStates,
                            transition, stateSpace.transitionOffset(t), transitionLd,
                            0.0, predictedDiffuseStateCov, nextPredDiffCovOff, kStates);

                    double pInfNorm = 0.0;
                    for (int i = 0; i < kStates2; i++) {
                        pInfNorm += predictedDiffuseStateCov[nextPredDiffCovOff + i]
                                * predictedDiffuseStateCov[nextPredDiffCovOff + i];
                    }

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
            BLAS.dcopy(kStates, predictedState, predOff, 1, scratch, tmpABase, 1);
            BLAS.dcopy(kStates2, predictedStateCov, predCovOff, 1, scratch, tmpPStarBase, 1);
            if (diffuse) {
                BLAS.dcopy(kStates2, predictedDiffuseStateCov, predDiffCovOff, 1, scratch, tmpPInfBase, 1);
            }

            double logLik = 0.0;
            double periodScaleContribution = 0.0;
            int periodScaleObservationCount = 0;

            if (storeForecastMean) {
                for (int i = 0; i < kEndog; i++) {
                    result.forecast[foreOff + i] = 0.0;
                }
            }
            if (storeForecastError) {
                for (int i = 0; i < kEndog; i++) {
                    result.forecastError[foreErrOff + i] = 0.0;
                }
            }
            if (storeForecastCovariance) {
                for (int i = 0; i < kEndog2; i++) {
                    result.forecastErrorCov[foreErrCovOff + i] = 0.0;
                }
            }
            if (storeStandardizedForecastError) {
                for (int i = 0; i < kEndog; i++) {
                    result.standardizedForecastError[stdForeErrOff + i] = 0.0;
                }
            }
            if (storeGain) {
                for (int i = 0; i < kStates * kEndog; i++) {
                    result.kalmanGain[kgOff + i] = 0.0;
                }
            }

            if (diffuse) {
                for (int i = 0; i < kEndog; i++) {
                    if (observationModel.isMissing(i, t)) continue;

                    int zRowOff = observationModel.designOffset(t) + i * designLd;
                    double vi = endog[endogOff + i] - obsIntercept[observationModel.obsInterceptOffset(t) + i];
                    for (int j = 0; j < kStates; j++)
                        vi -= design[zRowOff + j] * scratch[tmpABase + j];
                    if (storeForecastMean) {
                        result.forecast[foreOff + i] = endog[endogOff + i] - vi;
                    }
                    if (storeForecastError) {
                        result.forecastError[foreErrOff + i] = vi;
                    }

                    for (int j = 0; j < kStates; j++) {
                        double mStarJ = 0.0;
                        double mInfJ = 0.0;
                        for (int k = 0; k < kStates; k++) {
                            mStarJ += scratch[tmpPStarBase + j * kStates + k] * design[zRowOff + k];
                            mInfJ += scratch[tmpPInfBase + j * kStates + k] * design[zRowOff + k];
                        }
                        scratch[tmpMStarBase + j] = mStarJ;
                        scratch[tmpMInfBase + j] = mInfJ;
                    }

                    double Fstar = obsCov[observationModel.obsCovOffset(t) + i * obsCovLd + i];
                    for (int j = 0; j < kStates; j++)
                        Fstar += design[zRowOff + j] * scratch[tmpMStarBase + j];

                    double Finf = 0.0;
                    for (int j = 0; j < kStates; j++)
                        Finf += design[zRowOff + j] * scratch[tmpMInfBase + j];

                    if (Finf > diffuseTolerance) {
                        double F1 = 1.0 / Finf;
                        double F12 = -Fstar / (Finf * Finf);

                        for (int j = 0; j < kStates; j++) {
                            scratch[tmpK0Base + j] = scratch[tmpMInfBase + j] * F1;
                            scratch[tmpK1Base + j] = scratch[tmpMStarBase + j] * F1
                                    + scratch[tmpMInfBase + j] * F12;
                        }

                        for (int j = 0; j < kStates; j++)
                            scratch[tmpABase + j] += scratch[tmpK0Base + j] * vi;

                        for (int j = 0; j < kStates; j++) {
                            for (int k = 0; k < kStates; k++) {
                                scratch[tmpPStarBase + j * kStates + k] -=
                                        scratch[tmpMStarBase + j] * scratch[tmpK0Base + k]
                                                + scratch[tmpMInfBase + j] * scratch[tmpK1Base + k];
                                scratch[tmpPInfBase + j * kStates + k] -=
                                        scratch[tmpMInfBase + j] * scratch[tmpK0Base + k];
                            }
                        }

                        symmetrize(scratch, tmpPStarBase, kStates);
                        symmetrize(scratch, tmpPInfBase, kStates);

                        logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(Finf));
                        if (storeForecastDiffuseCovariance) {
                            result.forecastErrorDiffuseCov[foreErrDiffCovOff + i * kEndog + i] = Finf;
                        }
                        if (storeForecastCovariance) {
                            result.forecastErrorCov[foreErrCovOff + i * kEndog + i] = Fstar;
                        }
                        if (storeStandardizedForecastError) {
                            result.standardizedForecastError[stdForeErrOff + i] = 0.0;
                        }
                        if (storeGain) {
                            for (int j = 0; j < kStates; j++) {
                                result.kalmanGain[kgOff + j * kEndog + i] = scratch[tmpK0Base + j];
                            }
                        }
                    } else if (Fstar > diffuseTolerance) {
                        double FstarInv = 1.0 / Fstar;
                        double quadratic = vi * vi * FstarInv;
                        if (storeForecastCovariance) {
                            result.forecastErrorCov[foreErrCovOff + i * kEndog + i] = Fstar;
                        }
                        if (storeStandardizedForecastError) {
                            result.standardizedForecastError[stdForeErrOff + i] = ForecastErrorSolver.standardizeScalar(vi, Fstar, 0.0);
                        }
                        if (storeGain) {
                            for (int j = 0; j < kStates; j++) {
                                result.kalmanGain[kgOff + j * kEndog + i] = scratch[tmpMStarBase + j] * FstarInv;
                            }
                        }

                        for (int j = 0; j < kStates; j++)
                            scratch[tmpABase + j] += scratch[tmpMStarBase + j] * FstarInv * vi;

                        for (int j = 0; j < kStates; j++) {
                            for (int k = 0; k < kStates; k++) {
                                scratch[tmpPStarBase + j * kStates + k] -=
                                        scratch[tmpMStarBase + j] * FstarInv * scratch[tmpMStarBase + k];
                            }
                        }

                        symmetrize(scratch, tmpPStarBase, kStates);

                        if (concentratedLikelihood) {
                            logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(Fstar));
                            periodScaleContribution += quadratic;
                            periodScaleObservationCount++;
                        } else {
                            logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(Fstar) + quadratic);
                        }
                    }
                }
            } else {
                for (int i = 0; i < kEndog; i++) {
                    if (observationModel.isMissing(i, t)) continue;

                    int zRowOff = observationModel.designOffset(t) + i * designLd;
                    double vi = endog[endogOff + i] - obsIntercept[observationModel.obsInterceptOffset(t) + i];
                    for (int j = 0; j < kStates; j++)
                        vi -= design[zRowOff + j] * scratch[tmpABase + j];
                    if (storeForecastMean) {
                        result.forecast[foreOff + i] = endog[endogOff + i] - vi;
                    }
                    if (storeForecastError) {
                        result.forecastError[foreErrOff + i] = vi;
                    }

                    for (int j = 0; j < kStates; j++) {
                        double mStarJ = 0.0;
                        for (int k = 0; k < kStates; k++)
                            mStarJ += scratch[tmpPStarBase + j * kStates + k] * design[zRowOff + k];
                        scratch[tmpMStarBase + j] = mStarJ;
                    }

                    double Fi = obsCov[observationModel.obsCovOffset(t) + i * obsCovLd + i];
                    for (int j = 0; j < kStates; j++)
                        Fi += design[zRowOff + j] * scratch[tmpMStarBase + j];

                    if (Fi <= diffuseTolerance) continue;
                    if (storeForecastCovariance) {
                        result.forecastErrorCov[foreErrCovOff + i * kEndog + i] = Fi;
                    }
                    if (storeStandardizedForecastError) {
                        result.standardizedForecastError[stdForeErrOff + i] = ForecastErrorSolver.standardizeScalar(vi, Fi, 0.0);
                    }

                    double FiInv = 1.0 / Fi;
                    double quadratic = vi * vi * FiInv;
                    if (storeGain) {
                        for (int j = 0; j < kStates; j++) {
                            result.kalmanGain[kgOff + j * kEndog + i] = scratch[tmpMStarBase + j] * FiInv;
                        }
                    }
                    for (int j = 0; j < kStates; j++)
                        scratch[tmpABase + j] += scratch[tmpMStarBase + j] * FiInv * vi;

                    for (int j = 0; j < kStates; j++) {
                        for (int k = 0; k < kStates; k++)
                            scratch[tmpPStarBase + j * kStates + k] -=
                                    scratch[tmpMStarBase + j] * FiInv * scratch[tmpMStarBase + k];
                    }

                    symmetrize(scratch, tmpPStarBase, kStates);

                    if (concentratedLikelihood) {
                        logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(Fi));
                        periodScaleContribution += quadratic;
                        periodScaleObservationCount++;
                    } else {
                        logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(Fi) + quadratic);
                    }
                }
            }

            BLAS.dcopy(kStates, scratch, tmpABase, 1, filteredState, filtOff, 1);
            BLAS.dcopy(kStates2, scratch, tmpPStarBase, 1, filteredStateCov, filtCovOff, 1);
            if (storeLikelihood) {
                result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] = logLik;
            }
            if (concentratedLikelihood) {
                if (!diffuse && t >= concentratedScaleStart) {
                    concentratedScaleSum += periodScaleContribution;
                    concentratedScaleObservationCount += periodScaleObservationCount;
                }
                if (storeLikelihood) {
                    result.scaleObs[t] = periodScaleContribution;
                    result.scaleObsCount[t] = periodScaleObservationCount;
                }
            }

            BLAS.dcopy(kStates, stateIntercept, stateSpace.stateInterceptOffset(t), 1,
                    predictedState, nextPredOff, 1);
            BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                    scratch, tmpABase, 1,
                    1.0, predictedState, nextPredOff, 1);

            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                    scratch, tmpPStarBase, kStates,
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

            if (diffuse) {
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                        scratch, tmpPInfBase, kStates,
                        0.0, scratch, tmpTPBase, kStates);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                        1.0, scratch, tmpTPBase, kStates,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                    0.0, predictedDiffuseStateCov, nextPredDiffCovOff, kStates);

                double pInfNorm = 0.0;
                for (int i = 0; i < kStates2; i++)
                    pInfNorm += predictedDiffuseStateCov[nextPredDiffCovOff + i] *
                        predictedDiffuseStateCov[nextPredDiffCovOff + i];

                if (t + 1 >= nobsDiffuse) {
                    if (pInfNorm < diffuseTolerance) {
                        diffuse = false;
                        result.nobsDiffuse = nobsDiffuse;
                    } else {
                        nobsDiffuse = t + 2;
                    }
                }
            }

            symmetrize(predictedStateCov, nextPredCovOff, kStates);
        }

        if (concentratedLikelihood) {
            double scale = concentratedScaleObservationCount == 0
                ? 1.0
                : concentratedScaleSum / concentratedScaleObservationCount;
            result.scale = scale;
            result.scaleObservationCount = concentratedScaleObservationCount;
            KalmanFilter.applyConcentratedScale(observationModel, result, scale, storeLikelihood);
        }

        return result;
    }

    private static boolean hasScratch(double[] scratch, int base, int length) {
        return scratch != null && base >= 0 && scratch.length >= base + length;
    }

    private static void storeMissingForecastBlock(KalmanSSM observationModel,
                                                  double[] design,
                                                  int designLd,
                                                  double[] obsIntercept,
                                                  double[] obsCov,
                                                  int obsCovLd,
                                                  int t,
                                                  int kEndog,
                                                  int kStates,
                                                  double[] predictedState,
                                                  int predOff,
                                                  double[] predictedStateCov,
                                                  int predCovOff,
                                                  double[] predictedDiffuseStateCov,
                                                  int predDiffCovOff,
                                                  boolean diffuse,
                                                  FilterResult result,
                                                  boolean storeForecastMean,
                                                  boolean storeForecastError,
                                                  boolean storeForecastCovariance,
                                                  boolean storeStandardizedForecastError,
                                                  boolean storeForecastDiffuseCovariance,
                                                  int foreOff,
                                                  int foreErrOff,
                                                  int foreErrCovOff,
                                                  int stdForeErrOff,
                                                  int foreErrDiffCovOff) {
        int designOff = observationModel.designOffset(t);
        int obsInterceptOff = observationModel.obsInterceptOffset(t);
        int obsCovOff = observationModel.obsCovOffset(t);
        for (int row = 0; row < kEndog; row++) {
            int zRowOff = designOff + row * designLd;
            double mean = obsIntercept[obsInterceptOff + row];
            for (int state = 0; state < kStates; state++) {
                mean += design[zRowOff + state] * predictedState[predOff + state];
            }
            if (storeForecastMean) {
                result.forecast[foreOff + row] = mean;
            }
            if (storeForecastError) {
                result.forecastError[foreErrOff + row] = 0.0;
            }
            if (storeStandardizedForecastError) {
                result.standardizedForecastError[stdForeErrOff + row] = 0.0;
            }

            for (int col = 0; col < kEndog; col++) {
                int zColOff = designOff + col * designLd;
                double covariance = obsCov[obsCovOff + row * obsCovLd + col];
                double diffuseCovariance = 0.0;
                for (int i = 0; i < kStates; i++) {
                    double rowProjection = 0.0;
                    double rowDiffuseProjection = 0.0;
                    for (int j = 0; j < kStates; j++) {
                        rowProjection += predictedStateCov[predCovOff + i * kStates + j] * design[zColOff + j];
                        if (diffuse) {
                            rowDiffuseProjection += predictedDiffuseStateCov[predDiffCovOff + i * kStates + j] * design[zColOff + j];
                        }
                    }
                    covariance += design[zRowOff + i] * rowProjection;
                    if (diffuse) {
                        diffuseCovariance += design[zRowOff + i] * rowDiffuseProjection;
                    }
                }
                if (storeForecastCovariance) {
                    result.forecastErrorCov[foreErrCovOff + row * kEndog + col] = covariance;
                }
                if (diffuse && storeForecastDiffuseCovariance) {
                    result.forecastErrorDiffuseCov[foreErrDiffCovOff + row * kEndog + col] = diffuseCovariance;
                }
            }
        }
    }

    private static void symmetrize(double[] matrix, int offset, int n) {
        for (int row = 0; row < n; row++) {
            for (int col = row + 1; col < n; col++) {
                double avg = (matrix[offset + row * n + col] + matrix[offset + col * n + row]) * 0.5;
                matrix[offset + row * n + col] = avg;
                matrix[offset + col * n + row] = avg;
            }
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
                    transformEndogBlock(source, endogOverride, endogBase, t, kEndog);
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
            design = new double[kEndog * kStates * observationBlockCount];
            obsIntercept = new double[kEndog * observationBlockCount];
            obsCov = new double[kEndog * kEndog * observationBlockCount];
            endog = new double[kEndog * nobs];
            missing = hasMissingObservations ? new boolean[kEndog * nobs] : null;
            nmissing = hasMissingObservations ? new int[nobs] : null;
            active = new int[kEndog];
            factor = new double[kEndog * kEndog];
            diagonal = new double[kEndog];
            rhs = new double[kEndog];
            KalmanSSM.Builder builder = KalmanSSM.builder(kEndog, kStates, kPosdef, nobs)
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
            int designTargetOffset = observationBlockOffset(t, kEndog * kStates);
            int obsInterceptTargetOffset = observationBlockOffset(t, kEndog);
            int obsCovTargetOffset = observationBlockOffset(t, kEndog * kEndog);
            int endogTargetOffset = endogOffset(t);
            copyRectangularBlock(source.designData(), source.designOffset(t), source.designLeadingDimension(),
                kEndog, kStates, design, designTargetOffset, kStates);
            copyVector(source.obsInterceptData(), source.obsInterceptOffset(t), kEndog,
                obsIntercept, obsInterceptTargetOffset);
            copySquareBlock(source.obsCovData(), source.obsCovOffset(t), source.obsCovLeadingDimension(),
                kEndog, obsCov, obsCovTargetOffset, kEndog);
            double[] sourceEndog = endogOverride == null ? source.endogData() : endogOverride;
            int sourceEndogOffset = endogOverride == null ? source.endogOffset(t) : endogBase + t * kEndog;
            copyVector(sourceEndog, sourceEndogOffset, kEndog, endog, endogTargetOffset);
        }

        private void copyEndogBlock(KalmanSSM source,
                                    double[] endogOverride,
                                    int endogBase,
                                    int t) {
            double[] sourceEndog = endogOverride == null ? source.endogData() : endogOverride;
            int sourceEndogOffset = endogOverride == null ? source.endogOffset(t) : endogBase + t * kEndog;
            copyVector(sourceEndog, sourceEndogOffset, kEndog, endog, endogOffset(t));
        }

        private void transformEndogBlock(KalmanSSM source,
                                         double[] endogOverride,
                                         int endogBase,
                                         int t,
                                         int activeCount) {
            double[] sourceEndog = endogOverride == null ? source.endogData() : endogOverride;
            int sourceEndogOffset = endogOverride == null ? source.endogOffset(t) : endogBase + t * kEndog;
            int targetEndogOffset = endogOffset(t);
            for (int row = 0; row < activeCount; row++) {
                rhs[row] = sourceEndog[sourceEndogOffset + active[row]];
            }
            solveLowerInPlace(activeCount, rhs);
            for (int row = 0; row < activeCount; row++) {
                endog[targetEndogOffset + active[row]] = rhs[row];
            }
        }

        private void transformObservationBlock(int t) {
            int activeCount = 0;
            int missingOffset = hasMissingObservations ? t * kEndog : 0;
            for (int obs = 0; obs < kEndog; obs++) {
                if (!hasMissingObservations || !missing[missingOffset + obs]) {
                    active[activeCount++] = obs;
                }
            }
            if (activeCount <= 1) {
                return;
            }
            int obsCovOffset = observationBlockOffset(t, kEndog * kEndog);
            if (activeObservationCovarianceIsDiagonal(obsCovOffset, activeCount)) {
                return;
            }
            for (int row = 0; row < activeCount; row++) {
                int sourceRow = active[row];
                for (int col = 0; col < activeCount; col++) {
                    int sourceCol = active[col];
                    factor[row * activeCount + col] = obsCov[obsCovOffset + sourceRow * kEndog + sourceCol];
                }
            }
            ldlFactor(activeCount, t);
            int currentEndogOffset = endogOffset(t);
            solveActiveVector(endog, currentEndogOffset, activeCount);
            int obsInterceptOffset = observationBlockOffset(t, kEndog);
            solveActiveVector(obsIntercept, obsInterceptOffset, activeCount);
            int designOffset = observationBlockOffset(t, kEndog * kStates);
            for (int state = 0; state < kStates; state++) {
                for (int row = 0; row < activeCount; row++) {
                    rhs[row] = design[designOffset + active[row] * kStates + state];
                }
                solveLowerInPlace(activeCount, rhs);
                for (int row = 0; row < activeCount; row++) {
                    design[designOffset + active[row] * kStates + state] = rhs[row];
                }
            }
            Arrays.fill(obsCov, obsCovOffset, obsCovOffset + kEndog * kEndog, 0.0);
            for (int row = 0; row < activeCount; row++) {
                int obs = active[row];
                obsCov[obsCovOffset + obs * kEndog + obs] = diagonal[row];
            }
        }

        private int observationBlockOffset(int t, int blockLength) {
            return (observationBlocksTimeVarying ? t : 0) * blockLength;
        }

        private int endogOffset(int t) {
            return t * kEndog;
        }

        private boolean activeObservationCovarianceIsDiagonal(int obsCovOffset, int activeCount) {
            for (int row = 0; row < activeCount; row++) {
                int sourceRow = active[row];
                for (int col = 0; col < activeCount; col++) {
                    if (row != col) {
                        int sourceCol = active[col];
                        if (obsCov[obsCovOffset + sourceRow * kEndog + sourceCol] != 0.0) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        private void solveActiveVector(double[] values, int offset, int activeCount) {
            for (int row = 0; row < activeCount; row++) {
                rhs[row] = values[offset + active[row]];
            }
            solveLowerInPlace(activeCount, rhs);
            for (int row = 0; row < activeCount; row++) {
                values[offset + active[row]] = rhs[row];
            }
        }

        private void ldlFactor(int activeCount, int t) {
            for (int row = 0; row < activeCount; row++) {
                for (int col = 0; col < row; col++) {
                    double value = factor[row * activeCount + col];
                    for (int k = 0; k < col; k++) {
                        value -= factor[row * activeCount + k] * factor[col * activeCount + k] * diagonal[k];
                    }
                    value /= diagonal[col];
                    factor[row * activeCount + col] = value;
                }
                double pivot = factor[row * activeCount + row];
                for (int k = 0; k < row; k++) {
                    double lower = factor[row * activeCount + k];
                    pivot -= lower * lower * diagonal[k];
                }
                if (!(pivot > 0.0) || !Double.isFinite(pivot)) {
                    throw new IllegalArgumentException(
                        "UNIVARIATE filtering requires positive definite active observation covariance at t=" + t);
                }
                diagonal[row] = pivot;
                factor[row * activeCount + row] = 1.0;
                for (int col = row + 1; col < activeCount; col++) {
                    factor[row * activeCount + col] = 0.0;
                }
            }
        }

        private void solveLowerInPlace(int activeCount, double[] values) {
            for (int row = 0; row < activeCount; row++) {
                double value = values[row];
                for (int col = 0; col < row; col++) {
                    value -= factor[row * activeCount + col] * values[col];
                }
                values[row] = value;
            }
        }
    }

    private static void copyVector(double[] source, int sourceOffset, int width,
                                   double[] target, int targetOffset) {
        System.arraycopy(source, sourceOffset, target, targetOffset, width);
    }

    private static void copySquareBlock(double[] source,
                                        int sourceOffset,
                                        int sourceLeadingDimension,
                                        int width,
                                        double[] target,
                                        int targetOffset,
                                        int targetLeadingDimension) {
        copyRectangularBlock(source, sourceOffset, sourceLeadingDimension, width, width,
            target, targetOffset, targetLeadingDimension);
    }

    private static void copyRectangularBlock(double[] source,
                                             int sourceOffset,
                                             int sourceLeadingDimension,
                                             int rows,
                                             int cols,
                                             double[] target,
                                             int targetOffset,
                                             int targetLeadingDimension) {
        for (int row = 0; row < rows; row++) {
            System.arraycopy(source, sourceOffset + row * sourceLeadingDimension,
                target, targetOffset + row * targetLeadingDimension, cols);
        }
    }

    private static long doubleCount(double[] values) {
        return values == null ? 0L : values.length;
    }

    private static long intByteCount(int[] values) {
        return values == null ? 0L : (long) values.length * Integer.BYTES;
    }

    private static long booleanByteCount(boolean[] values) {
        return values == null ? 0L : values.length;
    }

    private static boolean hasMissingObservations(KalmanSSM model) {
        return KalmanSSMSupport.hasMissingObservations(model);
    }

    private static void requireRealInputs(KalmanSSM model) {
        KalmanSSMSupport.requireRealStorage(model, "model");
    }
}
