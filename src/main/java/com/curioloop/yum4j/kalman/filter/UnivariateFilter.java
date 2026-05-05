package com.curioloop.yum4j.kalman.filter;

import com.curioloop.yum4j.kalman.arena.FilterResultLayout;
import com.curioloop.yum4j.kalman.arena.UnivariateScratchLayout;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.internal.StateSpaceModelSupport;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;
import java.util.Objects;

public final class UnivariateFilter {

    private static final double toleranceDiffuse = 1e-7;

    public static final class Pool {
        public double[] scratchBacking;
        public UnivariateScratchLayout scratchLayout;

        private StateInitialization.StationaryWorkspace stationaryWorkspace;
        private int scratchKStates;
        private int scratchKEndog;

        void ensure(int kStates, int kEndog, int kPosdef) {
            int nextScratchKStates = Math.max(scratchKStates, kStates);
            int nextScratchKEndog = Math.max(scratchKEndog, kEndog);
            if (scratchLayout == null
                    || nextScratchKStates != scratchKStates
                    || nextScratchKEndog != scratchKEndog) {
                scratchLayout = UnivariateScratchLayout.create(nextScratchKEndog, nextScratchKStates);
                scratchKStates = nextScratchKStates;
                scratchKEndog = nextScratchKEndog;
            }
            int totalLength = scratchLayout.totalLength();
            if (scratchBacking == null || scratchBacking.length < totalLength) {
                scratchBacking = new double[totalLength];
            }
        }

        public void reserve(StateSpaceModel model) {
            reserve(model, null);
        }

        public void reserve(StateSpaceModel model, FilterSpec spec) {
            Objects.requireNonNull(model, "model");
            requireRealInputs(model);
            releaseRetainedScratch();
            ensure(model.stateCount(), model.observationDimension(), model.stateDisturbanceCount());
        }

        public long retainedScratchDoubleCount() {
            return doubleCount(scratchBacking)
                + (stationaryWorkspace == null ? 0L : stationaryWorkspace.retainedDoubleCount());
        }

        public long retainedTotalByteCount() {
            return retainedScratchDoubleCount() * Double.BYTES;
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }

        public void releaseRetainedScratch() {
            scratchBacking = null;
            scratchLayout = null;
            if (stationaryWorkspace != null) {
                stationaryWorkspace.release();
                stationaryWorkspace = null;
            }
            scratchKStates = 0;
            scratchKEndog = 0;
        }

        void ensureStationary(StateSpaceModel model) {
            if (stationaryWorkspace == null) {
                stationaryWorkspace = new StateInitialization.StationaryWorkspace();
            }
            stationaryWorkspace.ensureCapacity(model);
        }

        int tmpABase() { return scratchLayout.tmpABase(); }
        int tmpPBase() { return scratchLayout.tmpPBase(); }
        int tmpPZiBase() { return scratchLayout.tmpPZiBase(); }
        int tmpRQBase() { return scratchLayout.tmpRQBase(); }
        int tmpKBase() { return scratchLayout.tmpKBase(); }
        int tmpFBase() { return scratchLayout.tmpFBase(); }
        int tmpPInfBase() { return scratchLayout.tmpPInfBase(); }
        int tmpPStarBase() { return scratchLayout.tmpPStarBase(); }
        int tmpTPBase() { return scratchLayout.tmpTPBase(); }
        int tmpMStarBase() { return scratchLayout.tmpMStarBase(); }
        int tmpMInfBase() { return scratchLayout.tmpMInfBase(); }
        int tmpK0Base() { return scratchLayout.tmpK0Base(); }
        int tmpK1Base() { return scratchLayout.tmpK1Base(); }
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

    public static FilterResult filter(StateSpaceModel model,
                                      StateInitialization init,
                                      Pool pool,
                                      FilterSpec spec) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        requireRealInputs(model);
        FilterSpec resolvedSpec = spec == null ? FilterSpec.defaults() : spec;
        boolean hasMissingObservations = hasMissingObservations(model);
        FilterResultLayout retainedLayout = resolvedSpec.createResultLayout(
            1,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            hasMissingObservations);
        FilterResult result = filterInternal(model, init, pool, retainedLayout, null, 0);
        result.applyConserveMemory(resolvedSpec.conserveMemoryMask());
        return result;
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
        boolean hasMissingObservations = hasMissingObservations(model);
        FilterResultLayout retainedLayout = resolvedSpec.createResultLayout(
            1,
            model.observationDimension(),
            model.stateCount(),
            model.observationCount(),
            hasMissingObservations);
        FilterResult result = filterInternal(model, init, pool, retainedLayout, endog, endogBase);
        result.applyConserveMemory(resolvedSpec.conserveMemoryMask());
        return result;
    }

    private static FilterResult filterInternal(StateSpaceModel model,
                                               StateInitialization init,
                                               Pool pool,
                               FilterResultLayout retainedLayout,
                                               double[] endogOverride,
                                               int endogBase) {
        StateSpaceModel stateSpace = model;
        StateSpaceModel observationModel = model;
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

        if (pool == null) pool = new Pool();
        pool.ensure(kStates, kEndog, kPosdef);
        double[] scratch = pool.scratchBacking;
        int tmpABase = pool.tmpABase();
        int tmpPBase = pool.tmpPBase();
        int tmpPZiBase = pool.tmpPZiBase();
        int tmpRQBase = pool.tmpRQBase();

        boolean storeForecast = retainedLayout.forecastLength() > 0;
        boolean storePredicted = retainedLayout.predictedStateLength() > 0;
        boolean storeFiltered = retainedLayout.filteredStateLength() > 0;
        boolean storeLikelihood = retainedLayout.logLikelihoodObsLength() > 0;
        boolean storeGain = retainedLayout.kalmanGainLength() > 0;

        if (init.mayResolveDiffuse()) {
            return filterDiffuse(model, init, pool,
            new FilterResult(kEndog, kStates, nobs, retainedLayout),
                endogOverride, endogBase,
                null, 0,
                null, 0,
                null, 0,
                null, 0,
                null, 0);
        }

            StateInitialization.StationaryWorkspace stationaryWorkspace = null;
            if (init.mayResolveStationary()) {
                pool.ensureStationary(model);
                stationaryWorkspace = pool.stationaryWorkspace;
            }

        FilterResult result = new FilterResult(kEndog, kStates, nobs, retainedLayout);
        result.perObsKalmanGain = true;
        double[] predictedState = storePredicted ? result.predictedState : new double[kStates * 2];
        double[] predictedStateCov = storePredicted ? result.predictedStateCov : new double[kStates2 * 2];
        int predictedStateBase = storePredicted ? result.predictedStateBase() : 0;
        int predictedStateCovBase = storePredicted ? result.predictedStateCovBase() : 0;

        init.resolveInto(model, 0, stationaryWorkspace,
            predictedState, predictedStateBase,
            predictedStateCov, predictedStateCovBase,
            null, 0);

        for (int t = 0; t < nobs; t++) {
            int predOff = storePredicted ? result.predictedStateOffset(t) : (t & 1) * kStates;
            int predCovOff = storePredicted ? result.predictedStateCovOffset(t) : (t & 1) * kStates2;
            int filtOff = storeFiltered ? result.filteredStateOffset(t) : 0;
            int filtCovOff = storeFiltered ? result.filteredStateCovOffset(t) : 0;
            int foreOff = storeForecast ? result.forecastOffset(t) : 0;
            int foreErrOff = storeForecast ? result.forecastErrorOffset(t) : 0;
            int foreErrCovOff = storeForecast ? result.forecastErrorCovOffset(t) : 0;
            int nextPredOff = storePredicted ? result.predictedStateOffset(t + 1) : ((t + 1) & 1) * kStates;
            int nextPredCovOff = storePredicted ? result.predictedStateCovOffset(t + 1) : ((t + 1) & 1) * kStates2;
            int kgOff = storeGain ? result.kalmanGainOffset(t) : 0;
            int endogOff = endogOverride != null ? endogBase + t * kEndog : observationModel.endogOffset(t);

            BLAS.dcopy(kStates, predictedState, predOff, 1, scratch, tmpABase, 1);
            BLAS.dcopy(kStates2, predictedStateCov, predCovOff, 1, scratch, tmpPBase, 1);

            double logLik = 0.0;

            if (storeForecast) {
                for (int i = 0; i < kEndog; i++) {
                    result.forecast[foreOff + i] = 0.0;
                    result.forecastError[foreErrOff + i] = 0.0;
                }
                for (int i = 0; i < kEndog * kEndog; i++) {
                    result.forecastErrorCov[foreErrCovOff + i] = 0.0;
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
                    foreOff,
                    foreErrOff,
                    foreErrCovOff,
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
                for (int j = 0; j < kStates; j++) {
                    vi -= design[zRowOff + j] * scratch[tmpABase + j];
                }
                if (storeForecast) {
                    result.forecast[foreOff + i] = endog[endogOff + i] - vi;
                    result.forecastError[foreErrOff + i] = vi;
                }

                for (int j = 0; j < kStates; j++) {
                    scratch[tmpPZiBase + j] = 0.0;
                    for (int k = 0; k < kStates; k++) {
                        scratch[tmpPZiBase + j] += scratch[tmpPBase + j * kStates + k] * design[zRowOff + k];
                    }
                }

                double Fi = obsCov[observationModel.obsCovOffset(t) + i * obsCovLd + i];
                for (int j = 0; j < kStates; j++) {
                    Fi += design[zRowOff + j] * scratch[tmpPZiBase + j];
                }

                if (Fi <= toleranceDiffuse) {
                    continue;
                }
                if (storeForecast) {
                    result.forecastErrorCov[foreErrCovOff + i * kEndog + i] = Fi;
                }

                double FiInv = 1.0 / Fi;
                if (storeGain) {
                    for (int j = 0; j < kStates; j++) {
                        result.kalmanGain[kgOff + j * kEndog + i] = scratch[tmpPZiBase + j] * FiInv;
                    }
                }

                for (int j = 0; j < kStates; j++) {
                    scratch[tmpABase + j] += scratch[tmpPZiBase + j] * FiInv * vi;
                }

                for (int j = 0; j < kStates; j++) {
                    for (int k = 0; k < kStates; k++) {
                        scratch[tmpPBase + j * kStates + k] -= scratch[tmpPZiBase + j] * FiInv * scratch[tmpPZiBase + k];
                    }
                }

                symmetrize(scratch, tmpPBase, kStates);

                logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(Fi) + vi * vi * FiInv);
            }

            if (storeFiltered) {
                BLAS.dcopy(kStates, scratch, tmpABase, 1, result.filteredState, filtOff, 1);
                BLAS.dcopy(kStates2, scratch, tmpPBase, 1, result.filteredStateCov, filtCovOff, 1);
            }

            if (storeLikelihood) {
                result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] = logLik;
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

        return result;
    }

    static FilterResult filterDiffuse(StateSpaceModel model,
                                      StateInitialization init,
                                      Pool pool,
                                      FilterResult result) {
        return filterDiffuse(model, init, pool, result,
            null, 0,
                null, 0,
                null, 0,
                null, 0,
                null, 0,
                null, 0);
    }

    static FilterResult filterDiffuse(StateSpaceModel model,
                                      StateInitialization init,
                                      Pool pool,
                                      FilterResult result,
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
        StateSpaceModel stateSpace = model;
        StateSpaceModel observationModel = model;
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
        pool.ensure(kStates, kEndog, kPosdef);
        StateInitialization.StationaryWorkspace stationaryWorkspace = null;
        if (init.mayResolveStationary()) {
            pool.ensureStationary(model);
            stationaryWorkspace = pool.stationaryWorkspace;
        }
        double[] scratch = pool.scratchBacking;
        int tmpABase = pool.tmpABase();
        int tmpRQBase = pool.tmpRQBase();
        int tmpPInfBase = pool.tmpPInfBase();
        int tmpPStarBase = pool.tmpPStarBase();
        int tmpTPBase = pool.tmpTPBase();
        int tmpMStarBase = pool.tmpMStarBase();
        int tmpMInfBase = pool.tmpMInfBase();
        int tmpK0Base = pool.tmpK0Base();
        int tmpK1Base = pool.tmpK1Base();
        boolean storeForecast = result.forecastLength() > 0
            || result.forecastErrorLength() > 0
            || result.forecastErrorCovLength() > 0;
        boolean storePredicted = result.predictedStateLength() > 0
            || result.predictedStateCovLength() > 0;
        boolean storeFiltered = result.filteredStateLength() > 0
            || result.filteredStateCovLength() > 0;
        boolean storeLikelihood = result.logLikelihoodObsLength() > 0;
        boolean storeGain = result.kalmanGainLength() > 0;

        result.allocateDiffuse(storePredicted, storeForecast);
        result.converged = false;
        result.periodConverged = nobs;
        result.nobsDiffuse = 0;
        result.perObsKalmanGain = true;
        double[] predictedState = storePredicted
            ? result.predictedState
            : scratchOrAllocate(predictedStateScratch, predictedStateScratchBase, kStates * 2);
        double[] predictedStateCov = storePredicted
            ? result.predictedStateCov
            : scratchOrAllocate(predictedStateCovScratch, predictedStateCovScratchBase, kStates2 * 2);
        double[] predictedDiffuseStateCov = storePredicted
            ? result.predictedDiffuseStateCov
            : scratchOrAllocate(predictedDiffuseStateCovScratch, predictedDiffuseStateCovScratchBase, kStates2 * 2);
        double[] filteredState = storeFiltered
            ? result.filteredState
            : scratchOrAllocate(filteredStateScratch, filteredStateScratchBase, kStates);
        double[] filteredStateCov = storeFiltered
            ? result.filteredStateCov
            : scratchOrAllocate(filteredStateCovScratch, filteredStateCovScratchBase, kStates2);
        int predictedStateBase = storePredicted
            ? result.predictedStateBase()
            : scratchBase(predictedState, predictedStateScratch, predictedStateScratchBase, kStates * 2);
        int predictedStateCovBase = storePredicted
            ? result.predictedStateCovBase()
            : scratchBase(predictedStateCov, predictedStateCovScratch, predictedStateCovScratchBase, kStates2 * 2);
        int predictedDiffuseStateCovBase = storePredicted
            ? result.predictedDiffuseStateCovBase()
            : scratchBase(predictedDiffuseStateCov, predictedDiffuseStateCovScratch, predictedDiffuseStateCovScratchBase, kStates2 * 2);
        int filteredStateBase = storeFiltered
            ? result.filteredStateBase()
            : scratchBase(filteredState, filteredStateScratch, filteredStateScratchBase, kStates);
        int filteredStateCovBase = storeFiltered
            ? result.filteredStateCovBase()
            : scratchBase(filteredStateCov, filteredStateCovScratch, filteredStateCovScratchBase, kStates2);

        if (!storePredicted) {
            Arrays.fill(predictedState, predictedStateBase, predictedStateBase + kStates * 2, 0.0);
            Arrays.fill(predictedStateCov, predictedStateCovBase, predictedStateCovBase + kStates2 * 2, 0.0);
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

        for (int t = 0; t < nobs; t++) {
            int predOff = storePredicted ? result.predictedStateOffset(t) : predictedStateBase + (t & 1) * kStates;
            int predCovOff = storePredicted ? result.predictedStateCovOffset(t) : predictedStateCovBase + (t & 1) * kStates2;
            int predDiffCovOff = storePredicted ? result.predictedDiffuseStateCovOffset(t) : predictedDiffuseStateCovBase + (t & 1) * kStates2;
            int filtOff = storeFiltered ? result.filteredStateOffset(t) : filteredStateBase;
            int filtCovOff = storeFiltered ? result.filteredStateCovOffset(t) : filteredStateCovBase;
            int foreOff = storeForecast ? result.forecastOffset(t) : 0;
            int foreErrOff = storeForecast ? result.forecastErrorOffset(t) : 0;
            int foreErrCovOff = storeForecast ? result.forecastErrorCovOffset(t) : 0;
            int foreErrDiffCovOff = storeForecast ? result.forecastErrorDiffuseCovOffset(t) : 0;
            int nextPredOff = storePredicted ? result.predictedStateOffset(t + 1) : predictedStateBase + ((t + 1) & 1) * kStates;
            int nextPredCovOff = storePredicted ? result.predictedStateCovOffset(t + 1) : predictedStateCovBase + ((t + 1) & 1) * kStates2;
            int nextPredDiffCovOff = storePredicted ? result.predictedDiffuseStateCovOffset(t + 1) : predictedDiffuseStateCovBase + ((t + 1) & 1) * kStates2;
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
                        foreOff,
                        foreErrOff,
                        foreErrCovOff,
                        foreErrDiffCovOff);
                }
                if (storeFiltered) {
                    BLAS.dcopy(kStates, predictedState, predOff, 1,
                        filteredState, filtOff, 1);
                    BLAS.dcopy(kStates2, predictedStateCov, predCovOff, 1,
                        filteredStateCov, filtCovOff, 1);
                }
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
                        if (pInfNorm < toleranceDiffuse) {
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

            if (storeForecast) {
                for (int i = 0; i < kEndog; i++) {
                    result.forecast[foreOff + i] = 0.0;
                    result.forecastError[foreErrOff + i] = 0.0;
                }
                for (int i = 0; i < kEndog2; i++) {
                    result.forecastErrorCov[foreErrCovOff + i] = 0.0;
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
                    if (storeForecast) {
                        result.forecast[foreOff + i] = endog[endogOff + i] - vi;
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

                    if (Finf > toleranceDiffuse) {
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
                        if (storeForecast) {
                            result.forecastErrorDiffuseCov[foreErrDiffCovOff + i * kEndog + i] = Finf;
                            result.forecastErrorCov[foreErrCovOff + i * kEndog + i] = Fstar;
                        }
                        if (storeGain) {
                            for (int j = 0; j < kStates; j++) {
                                result.kalmanGain[kgOff + j * kEndog + i] = scratch[tmpK0Base + j];
                            }
                        }
                    } else if (Fstar > toleranceDiffuse) {
                        double FstarInv = 1.0 / Fstar;
                        if (storeForecast) {
                            result.forecastErrorCov[foreErrCovOff + i * kEndog + i] = Fstar;
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

                        logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(Fstar) + vi * vi * FstarInv);
                    }
                }
            } else {
                for (int i = 0; i < kEndog; i++) {
                    if (observationModel.isMissing(i, t)) continue;

                    int zRowOff = observationModel.designOffset(t) + i * designLd;
                    double vi = endog[endogOff + i] - obsIntercept[observationModel.obsInterceptOffset(t) + i];
                    for (int j = 0; j < kStates; j++)
                        vi -= design[zRowOff + j] * scratch[tmpABase + j];
                    if (storeForecast) {
                        result.forecast[foreOff + i] = endog[endogOff + i] - vi;
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

                    if (Fi <= toleranceDiffuse) continue;
                    if (storeForecast) {
                        result.forecastErrorCov[foreErrCovOff + i * kEndog + i] = Fi;
                    }

                    double FiInv = 1.0 / Fi;
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

                    logLik -= 0.5 * (Math.log(2 * Math.PI) + Math.log(Fi) + vi * vi * FiInv);
                }
            }

            if (storeFiltered) {
                BLAS.dcopy(kStates, scratch, tmpABase, 1, result.filteredState, filtOff, 1);
                BLAS.dcopy(kStates2, scratch, tmpPStarBase, 1, result.filteredStateCov, filtCovOff, 1);
            } else {
                BLAS.dcopy(kStates, scratch, tmpABase, 1, filteredState, filtOff, 1);
                BLAS.dcopy(kStates2, scratch, tmpPStarBase, 1, filteredStateCov, filtCovOff, 1);
            }
            if (storeLikelihood) {
                result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] = logLik;
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
                    if (pInfNorm < toleranceDiffuse) {
                        diffuse = false;
                        result.nobsDiffuse = nobsDiffuse;
                    } else {
                        nobsDiffuse = t + 2;
                    }
                }
            }

            symmetrize(predictedStateCov, nextPredCovOff, kStates);
        }

        return result;
    }

    private static double[] scratchOrAllocate(double[] scratch, int base, int length) {
        return scratch != null && scratch.length >= base + length ? scratch : new double[length];
    }

    private static int scratchBase(double[] actual, double[] supplied, int suppliedBase, int requiredLength) {
        return actual == supplied && supplied != null && supplied.length >= suppliedBase + requiredLength
            ? suppliedBase
            : 0;
    }

    private static void storeMissingForecastBlock(StateSpaceModel observationModel,
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
                                                  int foreOff,
                                                  int foreErrOff,
                                                  int foreErrCovOff,
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
            result.forecast[foreOff + row] = mean;
            result.forecastError[foreErrOff + row] = 0.0;

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
                result.forecastErrorCov[foreErrCovOff + row * kEndog + col] = covariance;
                if (diffuse) {
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

    private static boolean hasMissingObservations(StateSpaceModel model) {
        return StateSpaceModelSupport.hasMissingObservations(model);
    }

    private static void requireRealInputs(StateSpaceModel model) {
        StateSpaceModelSupport.requireRealStorage(model, "model");
    }
}
