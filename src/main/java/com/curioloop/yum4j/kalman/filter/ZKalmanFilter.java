package com.curioloop.yum4j.kalman.filter;

import com.curioloop.yum4j.kalman.arena.FilterSelectionLayout;
import com.curioloop.yum4j.kalman.arena.FilterDiffuseLayout;
import com.curioloop.yum4j.kalman.arena.FilterResultLayout;
import com.curioloop.yum4j.kalman.arena.ZFilterScratchLayout;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.internal.StateSpaceModelSupport;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;

import java.util.Objects;

public final class ZKalmanFilter {

    public static final class Pool {
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

        private ZFilterResult pooledResult;
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
                selectionLayout = FilterSelectionLayout.create(2, nextSelectionKEndog, nextSelectionKStates);
                selectionKStates = nextSelectionKStates;
                selectionKEndog = nextSelectionKEndog;
            }
            int totalLength = selectionLayout.totalLength();
            if (selectionBacking == null || selectionBacking.length < totalLength) {
                selectionBacking = new double[totalLength];
            }
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
                scratchLayout = ZFilterScratchLayout.create(
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
            ZFilterResult result = borrowResult(kEndog, kStates, nobs, layout);
            ensureDiffuseResult(diffuseLayout);
            return result.reuseDiffuse(diffuseResultBacking, diffuseResultLayout);
        }

        public void reserve(StateSpaceModel model,
                            FilterSpec spec) {
            requireComplexInputs(model);
            FilterSpec resolvedSpec = spec == null ? FilterSpec.defaults() : spec;
            FilterResultLayout retainedLayout = resolvedSpec.createResultLayout(
                2,
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
                    + doubleCount(convergedKalmanGain);
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

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }

        private static long intCount(int[] values) {
            return values == null ? 0L : values.length * ((long) Integer.BYTES) / Double.BYTES;
        }

        void ensureStationary(StateSpaceModel model) {
            if (stationaryWorkspace == null) {
                stationaryWorkspace = new StateInitialization.StationaryWorkspace();
            }
            stationaryWorkspace.ensureCapacity(model);
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
        int selectedDesignBase() { return selectionLayout.selectedDesignBase(); }
        int selectedObsCovBase() { return selectionLayout.selectedObsCovBase(); }
        int selectedObsInterceptBase() { return selectionLayout.selectedObsInterceptBase(); }
        int selectedEndogBase() { return selectionLayout.selectedEndogBase(); }
    }

    private static void requireComplexInputs(StateSpaceModel model) {
        StateSpaceModelSupport.requireComplexStorage(model, "model");
    }

    public static ZFilterResult filter(StateSpaceModel model,
                                       StateInitialization init) {
        return filter(model, init, null, FilterSpec.defaults());
    }

    public static ZFilterResult filter(StateSpaceModel model,
                                       StateInitialization init,
                                       Pool pool) {
        return filter(model, init, pool, FilterSpec.defaults());
    }

    public static ZFilterResult filter(StateSpaceModel model,
                                       StateInitialization init,
                                       Pool pool,
                                       FilterSpec spec) {
        requireComplexInputs(model);
        Objects.requireNonNull(init, "init");
        FilterSpec resolvedSpec = spec == null ? FilterSpec.defaults() : spec;
        FilterResultLayout retainedLayout = resolvedSpec.createResultLayout(
            2,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            StateSpaceModelSupport.hasMissingObservations(model));
        return filterInternal(model, init, pool,
            retainedLayout, resolvedSpec.stabilityMask(), resolvedSpec.conserveMemoryMask(), pool != null, null, 0);
    }

    public static ZFilterResult filter(StateSpaceModel model,
                                       StateInitialization init,
                                       double[] endog,
                                       int endogBase,
                                       Pool pool,
                                       FilterSpec spec) {
        requireComplexInputs(model);
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(endog, "endog");
        FilterSpec resolvedSpec = spec == null ? FilterSpec.defaults() : spec;
        FilterResultLayout retainedLayout = resolvedSpec.createResultLayout(
            2,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            StateSpaceModelSupport.hasMissingObservations(model));
        return filterInternal(model, init, pool,
            retainedLayout, resolvedSpec.stabilityMask(), resolvedSpec.conserveMemoryMask(), pool != null, endog, endogBase);
    }

    private static ZFilterResult filterInternal(StateSpaceModel model,
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

        ZFilterResult result = reuseResult
            ? pool.borrowResult(kEndog, kStates, nobs, retainedLayout)
            : new ZFilterResult(kEndog, kStates, nobs, retainedLayout);
        double[] predictedState = retainPredictedHistory
                ? result.predictedState
            : (reuseResult ? scratch : new double[4 * kStates]);
        double[] predictedStateCov = retainPredictedHistory
                ? result.predictedStateCov
            : (reuseResult ? scratch : new double[2 * ksKs]);
        double[] filteredState = storeFiltered
                ? result.filteredState
            : (reuseResult ? scratch : new double[2 * kStates]);
        double[] filteredStateCov = storeFiltered
                ? result.filteredStateCov
            : (reuseResult ? scratch : new double[ksKs]);
        double[] forecast = result.forecast;
        double[] forecastError = result.forecastError;
        double[] forecastErrorCov = result.forecastErrorCov;
        double[] kalmanGain = result.kalmanGain;
        int predictedStateBase = retainPredictedHistory ? result.predictedStateBase() : rollingPredictedStateBase;
        int predictedStateCovBase = retainPredictedHistory ? result.predictedStateCovBase() : rollingPredictedStateCovBase;
        int filteredStateBase = storeFiltered ? result.filteredStateBase() : scratchFilteredStateBase;
        int filteredStateCovBase = storeFiltered ? result.filteredStateCovBase() : scratchFilteredStateCovBase;
        int logLikelihoodObsBase = storeLikelihood ? result.logLikelihoodObsBase() : 0;

        if (init.mayResolveDiffuse()) {
            return filterDiffuse(model, init, pool, kEndog, kStates, kPosdef, nobs, ksKs, keKe,
                    retainedLayout, stabilityMethod, conserveMemory, reuseResult, endogOverride, endogBase);
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
            int predOff = retainPredictedHistory ? result.predictedStateOffset(t) : rollingPredictedStateBase + (t & 1) * 2 * kStates;
            int predCovOff = retainPredictedHistory ? result.predictedStateCovOffset(t) : rollingPredictedStateCovBase + (t & 1) * ksKs;
            int filtOff = storeFiltered ? result.filteredStateOffset(t) : filteredStateBase;
            int filtCovOff = storeFiltered ? result.filteredStateCovOffset(t) : filteredStateCovBase;
            int foreOff = storeForecast ? result.forecastOffset(t) : 0;
            int foreErrOff = storeForecast ? result.forecastErrorOffset(t) : 0;
            int foreErrCovOff = storeForecast ? result.forecastErrorCovOffset(t) : 0;
            int nextPredOff = retainPredictedHistory ? result.predictedStateOffset(t + 1) : rollingPredictedStateBase + ((t + 1) & 1) * 2 * kStates;
            int nextPredCovOff = retainPredictedHistory ? result.predictedStateCovOffset(t + 1) : rollingPredictedStateCovBase + ((t + 1) & 1) * ksKs;
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
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpTPBase >> 1, kStates,
                        transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                            0.0, 0.0, predictedStateCov, nextPredCovOff >> 1, kStates);
                    if (kPosdef > 0) {
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                            0.0, 0.0, scratch, tmpRQBase >> 1, kPosdef);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kPosdef,
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
                                double avgIm = (predictedStateCov[ij + 1] - predictedStateCov[ji + 1]) * 0.5;
                                predictedStateCov[ij] = avgRe;
                                predictedStateCov[ij + 1] = avgIm;
                                predictedStateCov[ji] = avgRe;
                                predictedStateCov[ji + 1] = -avgIm;
                            }
                        }
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

            // Complex forecast/update path mirrors the real recursion, with
            // Hermitian products replacing plain transposes.
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

            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, ke, kStates,
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
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, ke, kStates,
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
                ZLAS.zaxpy(kStates, vFtInvRe, vFtInvIm,
                            scratch, tmpPZtBase >> 1, 1,
                            filteredState, filtOff >> 1, 1);

                ZLAS.zcopy(kStates * kStates, predictedStateCov, predCovOff, 1,
                            filteredStateCov, filtCovOff, 1);
                ZLAS.zherk(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, kStates, ke,
                            -FtInvRe, scratch, tmpPZtBase >> 1, ke,
                            1.0, filteredStateCov, filtCovOff >> 1, kStates);

                vFv = vRe * vFtInvRe + vIm * vFtInvIm;
            } else {
                ZLAS.zpotrf(BLAS.Uplo.Lower, ke, scratch, tmpFcopyBase >> 1, ke);

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
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, ke, kStates,
                            1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                            scratch, tmpKBase >> 1, kStates,
                            0.0, 0.0, kalmanGain, kgOff >> 1, ke);
                }

                ZLAS.zcopy(kStates, predictedState, predOff, 1,
                            filteredState, filtOff, 1);
                double[] vSrc = storeForecast ? forecastError : scratch;
                int vSrcOff = storeForecast ? foreErrOff : tmpVBase;
                ZLAS.zgemv(BLAS.Trans.Conj, ke, kStates, 1.0, 0.0,
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
                for (int i = 0; i < kStates; i++) {
                    for (int j = i + 1; j < kStates; j++) {
                        int ij = filtCovOff + (i * 2 * kStates + j * 2);
                        int ji = filtCovOff + (j * 2 * kStates + i * 2);
                        double avgRe = (filteredStateCov[ij] + filteredStateCov[ji]) * 0.5;
                        double avgIm = (filteredStateCov[ij + 1] - filteredStateCov[ji + 1]) * 0.5;
                        filteredStateCov[ij] = avgRe;
                        filteredStateCov[ij + 1] = avgIm;
                        filteredStateCov[ji] = avgRe;
                        filteredStateCov[ji + 1] = -avgIm;
                    }
                }
            }

            if (storeLikelihood) {
                result.logLikelihoodObs[logLikelihoodObsBase + t] = -0.5 * (ke * Math.log(2 * Math.PI) + logDetF + vFv);
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

                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpTPBase >> 1, kStates,
                            transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                            0.0, 0.0, predictedStateCov, nextPredCovOff >> 1, kStates);

                if (kPosdef > 0) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                                stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                                0.0, 0.0, scratch, tmpRQBase >> 1, kPosdef);

                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kPosdef,
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
                            double avgIm = (predictedStateCov[ij + 1] - predictedStateCov[ji + 1]) * 0.5;
                            predictedStateCov[ij] = avgRe;
                            predictedStateCov[ij + 1] = avgIm;
                            predictedStateCov[ji] = avgRe;
                            predictedStateCov[ji + 1] = -avgIm;
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
                    if (maxDiff < tolerance) {
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

                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kEndog, kStates,
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

        result.applyConserveMemory(conserveMemory);
        result.converged = converged;
        result.periodConverged = periodConverged;
        return result;
    }

    private static ZFilterResult filterDiffuse(StateSpaceModel model,
                                               StateInitialization init,
                                               Pool pool,
                                               int kEndog, int kStates, int kPosdef,
                                               int nobs, int ksKs, int keKe,
                                               FilterResultLayout retainedLayout,
                                               int stabilityMethod, int conserveMemory,
                                               boolean reuseResult,
                                               double[] endogOverride,
                                               int endogBase) {
        StateSpaceModel stateSpace = model;
        StateSpaceModel observationModel = model;
        double[] transition = stateSpace.transitionData();
        int transitionLd = stateSpace.transitionLeadingDimension();
        double[] stateIntercept = stateSpace.stateInterceptData();
        double[] design = observationModel.designData();
        int designLd = observationModel.designLeadingDimension();
        double[] obsIntercept = observationModel.obsInterceptData();
        double[] obsCov = observationModel.obsCovData();
        int obsCovLd = observationModel.obsCovLeadingDimension();
        double[] endog = endogOverride != null ? endogOverride : observationModel.endogData();
        boolean storePredicted = retainedLayout.predictedStateLength() > 0;
        boolean storeForecast = retainedLayout.forecastLength() > 0;
        boolean storeFiltered = retainedLayout.filteredStateLength() > 0;
        boolean storeGain = retainedLayout.kalmanGainLength() > 0;
        boolean storeLikelihood = retainedLayout.logLikelihoodObsLength() > 0;
        FilterDiffuseLayout diffuseLayout = FilterDiffuseLayout.create(
                2,
                kEndog,
                kStates,
                nobs,
                storePredicted,
                storeForecast);
        if (reuseResult && !storePredicted) {
            pool.ensurePooledScratch(kStates, kEndog, kPosdef, true, false);
            pool.ensureDiffusePredictedScratch(kStates, kEndog, kPosdef);
        }
        ZFilterResult result = reuseResult
            ? pool.borrowDiffuseResult(kEndog, kStates, nobs, retainedLayout, diffuseLayout)
            : new ZFilterResult(kEndog, kStates, nobs, retainedLayout);
        if (reuseResult) {
            result.clearActiveSurfaces();
        }
        if (!reuseResult) {
            result.allocateDiffuse(storePredicted, storeForecast);
        }
        result.perObsKalmanGain = true;
        double[] scratch = pool.scratchBacking;
        StateInitialization.StationaryWorkspace stationaryWorkspace = null;
        if (init.mayResolveStationary()) {
            pool.ensureStationary(model);
            stationaryWorkspace = pool.stationaryWorkspace;
        }

        double[] predictedState = storePredicted
            ? result.predictedState
            : (reuseResult ? pool.scratchBacking : new double[4 * kStates]);
        double[] predictedStateCov = storePredicted
            ? result.predictedStateCov
            : (reuseResult ? pool.scratchBacking : new double[2 * ksKs]);
        double[] predictedDiffuseStateCov = storePredicted
            ? result.predictedDiffuseStateCov
            : (reuseResult ? pool.scratchBacking : new double[2 * ksKs]);
        int predictedStateBase = storePredicted ? result.predictedStateBase()
            : (reuseResult ? pool.rollingPredictedStateBase() : 0);
        int predictedStateCovBase = storePredicted ? result.predictedStateCovBase()
            : (reuseResult ? pool.rollingPredictedStateCovBase() : 0);
        int predictedDiffuseStateCovBase = storePredicted ? result.predictedDiffuseStateCovBase()
            : (reuseResult ? pool.rollingPredictedDiffuseStateCovBase() : 0);
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
        int predOff = storePredicted ? result.predictedStateOffset(t)
            : predictedStateBase + (t & 1) * 2 * kStates;
        int predCovOff = storePredicted ? result.predictedStateCovOffset(t)
            : predictedStateCovBase + (t & 1) * ksKs;
        int predDiffCovOff = storePredicted ? result.predictedDiffuseStateCovOffset(t)
            : predictedDiffuseStateCovBase + (t & 1) * ksKs;
            int filtOff = result.filteredStateOffset(t);
            int filtCovOff = result.filteredStateCovOffset(t);
            int foreOff = result.forecastOffset(t);
            int foreErrOff = result.forecastErrorOffset(t);
            int foreErrCovOff = result.forecastErrorCovOffset(t);
            int foreErrDiffCovOff = result.forecastErrorDiffuseCovOffset(t);
        int nextPredOff = storePredicted ? result.predictedStateOffset(t + 1)
            : predictedStateBase + ((t + 1) & 1) * 2 * kStates;
        int nextPredCovOff = storePredicted ? result.predictedStateCovOffset(t + 1)
            : predictedStateCovBase + ((t + 1) & 1) * ksKs;
        int nextPredDiffCovOff = storePredicted ? result.predictedDiffuseStateCovOffset(t + 1)
            : predictedDiffuseStateCovBase + ((t + 1) & 1) * ksKs;
            int kgOff = result.kalmanGainOffset(t);
            int endogOff = endogOverride != null ? endogBase + t * 2 * kEndog : observationModel.endogOffset(t);

            if (observationModel.missingCount(t) == kEndog) {
                if (storeForecast) {
                    zero(result.forecast, foreOff, 2 * kEndog);
                    zero(result.forecastError, foreErrOff, 2 * kEndog);
                    zero(result.forecastErrorCov, foreErrCovOff, keKe);
                }
                if (diffuse && storeForecast) {
                    zero(result.forecastErrorDiffuseCov, foreErrDiffCovOff, keKe);
                }
                if (storeFiltered) {
                    ZLAS.zcopy(kStates, predictedState, predOff, 1,
                            result.filteredState, filtOff, 1);
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
                    symmetrizeHermitian(predictedStateCov, nextPredCovOff, kStates);
                }

                if (diffuse) {
                    propagateHermitian(stateSpace, t, predictedDiffuseStateCov, predDiffCovOff,
                            predictedDiffuseStateCov, nextPredDiffCovOff, pool, kStates, 0, false);
                    if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                        symmetrizeHermitian(predictedDiffuseStateCov, nextPredDiffCovOff, kStates);
                    }
                    double pInfNorm = squaredNorm(predictedDiffuseStateCov, nextPredDiffCovOff, ksKs);
                    if (t + 1 >= nobsDiffuse) {
                        if (pInfNorm < UnivariateFilterTolerance.DIFFUSE) {
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
            if (diffuse && storeForecast) {
                zero(result.forecastErrorDiffuseCov, foreErrDiffCovOff, keKe);
            }
            if (storeGain) {
                zero(result.kalmanGain, kgOff, kStates * 2 * kEndog);
            }

            double logLik = 0.0;

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

                if (diffuse && fInf > UnivariateFilterTolerance.DIFFUSE) {
                    double f1 = 1.0 / fInf;
                    double f2 = -fStar * f1 * f1;
                    if (storeForecast) {
                        result.forecastErrorDiffuseCov[foreErrDiffCovOff + i * 2 * kEndog + i * 2] = fInf;
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
                        symmetrizeHermitian(scratch, tmpPStarBase, kStates);
                        symmetrizeHermitian(scratch, tmpPInfBase, kStates);
                    }

                    logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(fInf));
                } else if (fStar > UnivariateFilterTolerance.DIFFUSE) {
                    double fStarInv = 1.0 / fStar;
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
                        symmetrizeHermitian(scratch, tmpPStarBase, kStates);
                    }

                    logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(fStar)
                            + (vRe * vRe + vIm * vIm) * fStarInv);
                }
            }

            if (storeFiltered) {
                ZLAS.zcopy(kStates, scratch, tmpABase, 1, result.filteredState, filtOff, 1);
                ZLAS.zcopy(kStates * kStates, scratch, tmpPStarBase, 1, result.filteredStateCov, filtCovOff, 1);
            }
            if (storeLikelihood) {
                result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] = logLik;
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
                symmetrizeHermitian(predictedStateCov, nextPredCovOff, kStates);
            }

            if (diffuse) {
                propagateHermitian(stateSpace, t, scratch, tmpPInfBase,
                        predictedDiffuseStateCov, nextPredDiffCovOff, pool, kStates, 0, false);
                if ((stabilityMethod & KalmanFilter.STABILITY_FORCE_SYMMETRY) != 0) {
                    symmetrizeHermitian(predictedDiffuseStateCov, nextPredDiffCovOff, kStates);
                }

                double pInfNorm = squaredNorm(predictedDiffuseStateCov, nextPredDiffCovOff, ksKs);
                if (t + 1 >= nobsDiffuse) {
                    if (pInfNorm < UnivariateFilterTolerance.DIFFUSE) {
                        diffuse = false;
                        result.nobsDiffuse = nobsDiffuse;
                    } else {
                        nobsDiffuse = t + 2;
                    }
                }
            }
        }

        result.applyConserveMemory(conserveMemory);
        result.converged = false;
        result.periodConverged = nobs;
        return result;
    }

    private static void propagateHermitian(StateSpaceModel stateSpace, int t,
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
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
                1.0, 0.0, scratch, tmpTPBase >> 1, kStates,
                transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                0.0, 0.0, target, targetOff >> 1, kStates);
        if (addStateNoise && kPosdef > 0) {
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                    1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                    stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                    0.0, 0.0, scratch, tmpRQBase >> 1, kPosdef);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kPosdef,
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

    private static final class UnivariateFilterTolerance {
        private static final double DIFFUSE = 1e-7;

        private UnivariateFilterTolerance() {
        }
    }

}
