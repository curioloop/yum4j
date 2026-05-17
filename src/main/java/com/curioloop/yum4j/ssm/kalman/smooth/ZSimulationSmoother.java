package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.arena.SimulationResultLayout;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;

import java.util.Objects;
import java.util.Random;

/**
 * Complex-valued KFS simulation smoother built on the complex filter and smoother.
 */
public final class ZSimulationSmoother {

    private static void requireComplexInputs(KalmanSSM stateSpace,
                                             KalmanSSM observationModel) {
        KalmanSSMSupport.requireComplexStorage(stateSpace, "stateSpace");
        KalmanSSMSupport.requireComplexStorage(observationModel, "observationModel");
    }

    public static final class CombineWorkspace {
        public double[] tmpCholeskyWork;
        double[] generatedObsScratch;
        double[] generatedMeasurementScratch;
        double[] generatedStateDisturbanceScratch;
        double[] generatedInitialStateScratch;
        double[] generatedCurrentStateScratch;
        double[] generatedNextStateScratch;

        void ensure(int kEndog, int kStates, int kPosdef, int nobs,
                    boolean useBorrowedObsBank,
                    boolean storeGeneratedOutputs) {
            int maxCholeskyDim = Math.max(kEndog, Math.max(kStates, kPosdef));
            if (maxCholeskyDim > 1 && (tmpCholeskyWork == null || tmpCholeskyWork.length < maxCholeskyDim * 2 * maxCholeskyDim)) {
                tmpCholeskyWork = new double[maxCholeskyDim * 2 * maxCholeskyDim];
            }
            generatedObsScratch = storeGeneratedOutputs || useBorrowedObsBank
                ? ZSimulationSmootherResult.emptyArray()
                : Pool.exactSized(generatedObsScratch, 2 * kEndog * nobs);
            if (storeGeneratedOutputs) {
                generatedInitialStateScratch = ZSimulationSmootherResult.emptyArray();
                generatedCurrentStateScratch = ZSimulationSmootherResult.emptyArray();
                generatedNextStateScratch = ZSimulationSmootherResult.emptyArray();
            } else {
                generatedInitialStateScratch = Pool.exactSized(generatedInitialStateScratch, 2 * kStates);
                generatedCurrentStateScratch = Pool.exactSized(generatedCurrentStateScratch, 2 * kStates);
                generatedNextStateScratch = Pool.exactSized(generatedNextStateScratch, 2 * kStates);
            }
            if (storeGeneratedOutputs) {
                generatedMeasurementScratch = ZSimulationSmootherResult.emptyArray();
                generatedStateDisturbanceScratch = ZSimulationSmootherResult.emptyArray();
            } else {
                generatedMeasurementScratch = Pool.exactSized(generatedMeasurementScratch, 2 * kEndog);
                generatedStateDisturbanceScratch = Pool.exactSized(generatedStateDisturbanceScratch, 2 * kPosdef);
            }
        }

        void releaseBackings() {
            tmpCholeskyWork = null;
            generatedObsScratch = null;
            generatedMeasurementScratch = null;
            generatedStateDisturbanceScratch = null;
            generatedInitialStateScratch = null;
            generatedCurrentStateScratch = null;
            generatedNextStateScratch = null;
        }

        long retainedDoubleCount() {
            return Pool.doubleCount(tmpCholeskyWork)
                + Pool.doubleCount(generatedObsScratch)
                + Pool.doubleCount(generatedMeasurementScratch)
                + Pool.doubleCount(generatedStateDisturbanceScratch)
                + Pool.doubleCount(generatedInitialStateScratch)
                + Pool.doubleCount(generatedCurrentStateScratch)
                + Pool.doubleCount(generatedNextStateScratch);
        }
    }

    static final class Pool {
        private KalmanEngine.Workspace actualFilterWorkspace;
        private KalmanEngine.Workspace generatedFilterWorkspace;
        private ZKalmanSmoother.Pool actualSmootherPool;
        private ZKalmanSmoother.Pool generatedSmootherPool;
        private CombineWorkspace combineWorkspace;
        private ZCFASimulationSmoother.Workspace cfaWorkspace;
        private double[] measurementDisturbanceVariates;
        private double[] stateDisturbanceVariates;
        private double[] initialStateVariates;
        private boolean retainReusableWorkspaces;
        private boolean retainCfaPosteriorWorkspace;

        private ZSimulationSmootherResult pooledResult;
        private ZSmootherResult generatedSmoothTarget;
        private final DoubleArena resultArena;
        private final boolean ownsResultArena;
        private double[] resultBacking;
        private SimulationResultLayout resultLayout;
        private int resultLayoutKEndog = -1;
        private int resultLayoutKStates = -1;
        private int resultLayoutKPosdef = -1;
        private int resultLayoutNobs = -1;
        private int resultLayoutOptions = -1;

        Pool() {
            this(new DoubleArena(), true);
        }

        Pool(DoubleArena resultArena, boolean ownsResultArena) {
            this.resultArena = Objects.requireNonNull(resultArena, "resultArena");
            this.ownsResultArena = ownsResultArena;
        }

        void ensure(int kEndog, int kStates, int kPosdef) {
            if (actualFilterWorkspace == null) {
                actualFilterWorkspace = KalmanEngine.workspace();
            }
            if (actualSmootherPool == null) {
                actualSmootherPool = new ZKalmanSmoother.Pool();
            }
            if (generatedSmootherPool == null) {
                generatedSmootherPool = new ZKalmanSmoother.Pool();
            }
            if (combineWorkspace == null) {
                combineWorkspace = new CombineWorkspace();
            }
            generatedFilterWorkspace = actualFilterWorkspace;
            generatedSmootherPool.shareScratchFrom(actualSmootherPool);
        }

        void ensureResult(int kEndog, int kStates, int kPosdef, int nobs,
                          boolean storeGeneratedOutputs,
                          boolean storeState, boolean storeDisturbance) {
            int nextLayoutOptions = (storeGeneratedOutputs ? 1 : 0)
                | (storeState ? 1 << 1 : 0)
                | (storeDisturbance ? 1 << 2 : 0);
            if (resultLayout == null
                    || resultLayoutKEndog != kEndog
                    || resultLayoutKStates != kStates
                    || resultLayoutKPosdef != kPosdef
                    || resultLayoutNobs != nobs
                    || resultLayoutOptions != nextLayoutOptions) {
                resultLayout = SimulationResultLayout.create(2, kEndog, kStates, kPosdef, nobs,
                    storeGeneratedOutputs, storeState, storeDisturbance);
                resultLayoutKEndog = kEndog;
                resultLayoutKStates = kStates;
                resultLayoutKPosdef = kPosdef;
                resultLayoutNobs = nobs;
                resultLayoutOptions = nextLayoutOptions;
            }
            int totalLength = resultLayout.totalLength();
            resultBacking = resultArena.ensureCapacity(totalLength);
        }

        void ensureCombineWorkspace(int kEndog, int kStates, int kPosdef, int nobs,
                                    boolean useBorrowedObsBank,
                                    boolean storeGeneratedOutputs) {
            combineWorkspace.ensure(kEndog, kStates, kPosdef, nobs, useBorrowedObsBank, storeGeneratedOutputs);
        }

        void ensureVariates(int kEndog, int kStates, int kPosdef, int nobs) {
            measurementDisturbanceVariates = exactSized(measurementDisturbanceVariates, 2 * kEndog * nobs);
            stateDisturbanceVariates = exactSized(stateDisturbanceVariates, 2 * kPosdef * nobs);
            initialStateVariates = exactSized(initialStateVariates, 2 * kStates);
        }

        void releaseVariates() {
            measurementDisturbanceVariates = null;
            stateDisturbanceVariates = null;
            initialStateVariates = null;
        }

        void releaseTransientNestedResults() {
            if (actualFilterWorkspace != null) {
                actualFilterWorkspace.releaseComplexFilterRetainedResults();
            }
            if (actualSmootherPool != null) {
                actualSmootherPool.releaseRetainedResults();
            }
            if (generatedSmootherPool != null) {
                generatedSmootherPool.releaseRetainedResults();
            }
        }

        void releaseTransientNestedScratch() {
            if (actualFilterWorkspace != null) {
                actualFilterWorkspace.releaseComplexFilterRetainedScratch();
            }
            if (generatedFilterWorkspace != null && !generatedFilterWorkspaceShared()) {
                generatedFilterWorkspace.releaseComplexFilterRetainedScratch();
            }
            if (actualSmootherPool != null) {
                actualSmootherPool.releaseRetainedScratch();
            }
            if (generatedSmootherPool != null) {
                generatedSmootherPool.releaseRetainedScratch();
            }
        }

        void releaseCombineWorkspace() {
            if (combineWorkspace != null) {
                combineWorkspace.releaseBackings();
            }
            releaseCfaWorkspace();
        }

        ZCFASimulationSmoother.Workspace cfaWorkspace(int order) {
            if (cfaWorkspace == null) {
                cfaWorkspace = new ZCFASimulationSmoother.Workspace();
            }
            cfaWorkspace.ensure(order);
            return cfaWorkspace;
        }

        void releaseCfaWorkspace() {
            if (cfaWorkspace != null) {
                cfaWorkspace.release();
            }
        }

        void retainReusableWorkspaces(boolean retainReusableWorkspaces) {
            this.retainReusableWorkspaces = retainReusableWorkspaces;
            if (!retainReusableWorkspaces) {
                retainCfaPosteriorWorkspace = false;
            }
        }

        boolean retainsReusableWorkspaces() {
            return retainReusableWorkspaces;
        }

        void retainCfaPosteriorWorkspace(boolean retainCfaPosteriorWorkspace) {
            this.retainCfaPosteriorWorkspace = retainCfaPosteriorWorkspace;
        }

        boolean retainsCfaPosteriorWorkspace() {
            return retainCfaPosteriorWorkspace;
        }

        void releaseTransientRunState() {
            releaseTransientNestedResults();
            releaseTransientNestedScratch();
            if (!retainReusableWorkspaces) {
                releaseCombineWorkspace();
            }
            releaseVariates();
        }

        private boolean generatedFilterWorkspaceShared() {
            return generatedFilterWorkspace == actualFilterWorkspace;
        }

        private boolean generatedSmootherScratchShared() {
            return generatedSmootherPool == actualSmootherPool
                || (generatedSmootherPool != null
                    && actualSmootherPool != null
                    && generatedSmootherPool.sharesScratchWith(actualSmootherPool));
        }

        ZSimulationSmootherResult borrowResult(int kEndog, int kStates, int kPosdef, int nobs,
                                               boolean storeGeneratedOutputs,
                                               boolean storeState, boolean storeDisturbance) {
            ensureResult(kEndog, kStates, kPosdef, nobs, storeGeneratedOutputs, storeState, storeDisturbance);
            if (pooledResult == null) {
                pooledResult = new ZSimulationSmootherResult();
            }
            return pooledResult.reuse(kEndog, kStates, kPosdef, nobs,
                resultBacking, resultLayout);
        }

        ZSmootherResult borrowGeneratedSmoothTarget(ZSimulationSmootherResult result,
                                                    int kEndog, int kStates, int kPosdef, int nobs,
                                                    boolean storeState, boolean storeDisturbance) {
            if (generatedSmoothTarget == null) {
                generatedSmoothTarget = new ZSmootherResult();
            }
            return generatedSmoothTarget.reuseExternal(kEndog, kStates, kPosdef, nobs,
                storeState,
                storeDisturbance,
                result.simulatedState,
                result.simulatedStateBase(),
                result.simulatedMeasurementDisturbance,
                result.simulatedMeasurementDisturbanceBase(),
                result.simulatedStateDisturbance,
                result.simulatedStateDisturbanceBase());
        }

        private static double[] exactSized(double[] values, int length) {
            if (length == 0) {
                return ZSimulationSmootherResult.emptyArray();
            }
            if (values == null || values.length != length) {
                return new double[length];
            }
            return values;
        }

        public long retainedWorkspaceDoubleCount() {
            return (combineWorkspace == null ? 0L : combineWorkspace.retainedDoubleCount())
                + (cfaWorkspace == null ? 0L : cfaWorkspace.retainedDoubleCount());
        }

        public long retainedCombineWorkspaceDoubleCount() {
            return combineWorkspace == null ? 0L : combineWorkspace.retainedDoubleCount();
        }

        public long retainedCfaWorkspaceDoubleCount() {
            return cfaWorkspace == null ? 0L : cfaWorkspace.retainedDoubleCount();
        }

        public long retainedVariateDoubleCount() {
            return doubleCount(measurementDisturbanceVariates)
                + doubleCount(stateDisturbanceVariates)
                + doubleCount(initialStateVariates);
        }

        public long retainedResultDoubleCount() {
            return usesResultArena() ? resultArena.retainedDoubleCount() : 0L;
        }

        public void releaseRetainedResults() {
            invalidateBorrowedResult();
            if (ownsResultArena) {
                resultArena.release();
            }
            resultLayout = null;
            resultLayoutKEndog = -1;
            resultLayoutKStates = -1;
            resultLayoutKPosdef = -1;
            resultLayoutNobs = -1;
            resultLayoutOptions = -1;
        }

        boolean usesResultArena() {
            return resultBacking != null;
        }

        void invalidateBorrowedResult() {
            resultBacking = null;
            pooledResult = null;
            generatedSmoothTarget = null;
        }

        public long retainedNestedScratchDoubleCount() {
            long total = 0L;
            if (actualFilterWorkspace != null) {
                total += actualFilterWorkspace.retainedComplexFilterScratchDoubleCount();
            }
            if (generatedFilterWorkspace != null && !generatedFilterWorkspaceShared()) {
                total += generatedFilterWorkspace.retainedComplexFilterScratchDoubleCount();
            }
            if (actualSmootherPool != null) {
                total += actualSmootherPool.retainedScratchDoubleCount();
            }
            if (generatedSmootherPool != null && !generatedSmootherScratchShared()) {
                total += generatedSmootherPool.retainedScratchDoubleCount();
            }
            return total;
        }

        public long retainedNestedResultDoubleCount() {
            long total = 0L;
            if (actualFilterWorkspace != null) {
                total += actualFilterWorkspace.retainedComplexFilterResultDoubleCount();
            }
            if (generatedFilterWorkspace != null && !generatedFilterWorkspaceShared()) {
                total += generatedFilterWorkspace.retainedComplexFilterResultDoubleCount();
            }
            if (actualSmootherPool != null) {
                total += actualSmootherPool.retainedResultDoubleCount();
            }
            if (generatedSmootherPool != null) {
                total += generatedSmootherPool.retainedResultDoubleCount();
            }
            return total;
        }

        public long retainedTotalByteCount() {
            return (retainedWorkspaceDoubleCount()
                + retainedVariateDoubleCount()
                + retainedResultDoubleCount()
                + retainedNestedScratchDoubleCount()
                + retainedNestedResultDoubleCount()) * Double.BYTES;
        }

        public void releaseRetainedScratch() {
            releaseCombineWorkspace();
            if (actualFilterWorkspace != null) {
                actualFilterWorkspace.releaseComplexFilterRetainedScratch();
            }
            if (generatedFilterWorkspace != null && !generatedFilterWorkspaceShared()) {
                generatedFilterWorkspace.releaseComplexFilterRetainedScratch();
            }
            if (actualSmootherPool != null) {
                actualSmootherPool.releaseRetainedScratch();
            }
            if (generatedSmootherPool != null) {
                generatedSmootherPool.releaseRetainedScratch();
            }
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }

        public void reserve(KalmanSSM model,
                            InitialState init) {
            reserve(model, init, SimulationSmootherOptions.defaults());
        }

        public void reserve(KalmanSSM model,
                            InitialState init,
                            SimulationSmootherOptions spec) {
            requireComplexInputs(model, model);
            Objects.requireNonNull(init, "init");
            SimulationSmootherOptions resolvedOptions = spec == null ? SimulationSmootherOptions.defaults() : spec;
            requireSupportedOptions(resolvedOptions);

            ensure(model.observationDimension(), model.stateCount(), model.stateDisturbanceCount());
            boolean storeGeneratedOutputs = resolvedOptions.storesGeneratedOutputs();
            boolean storeState = resolvedOptions.includes(SimulationSmootherOptions.Surface.STATE);
            boolean storeDisturbance = resolvedOptions.includes(SimulationSmootherOptions.Surface.DISTURBANCE);

            actualFilterWorkspace.reserveComplexFilter(model, resolvedOptions.requiredFilterOptions());
            actualSmootherPool.releaseRetainedResults();
            generatedSmootherPool.releaseRetainedResults();
            generatedSmootherPool.shareScratchFrom(actualSmootherPool);
            ensureResult(model.observationDimension(), model.stateCount(), model.stateDisturbanceCount(), model.observationCount(),
                storeGeneratedOutputs, storeState, storeDisturbance);
        }
    }

    private ZSimulationSmoother() {
    }

    static ZSimulationSmootherResult simulate(KalmanSSM model,
                                              InitialState init,
                                              double[] measurementDisturbanceVariates,
                                              double[] stateDisturbanceVariates,
                                              double[] initialStateVariates,
                                              Pool pool,
                                              SimulationSmootherOptions spec) {
        requireComplexInputs(model, model);
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(measurementDisturbanceVariates, "measurementDisturbanceVariates");
        Objects.requireNonNull(stateDisturbanceVariates, "stateDisturbanceVariates");
        Objects.requireNonNull(initialStateVariates, "initialStateVariates");
        SimulationSmootherOptions resolvedOptions = spec == null ? SimulationSmootherOptions.defaults() : spec;

        boolean reuseResult = pool != null;
        Pool workspace = reuseResult ? pool : new Pool();
        workspace.reserve(model, init, resolvedOptions);

        return simulateInternal(model, init,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            workspace,
            resolvedOptions,
            reuseResult);
    }

    /**
     * When a non-null pool is supplied, the returned result borrows pool-owned storage
     * and is invalidated by the next call that reuses the same pool. Call clone() or copy()
     * on the result when an independent snapshot is required.
     */
    private static ZSimulationSmootherResult simulateInternal(KalmanSSM model,
                                                              InitialState init,
                                                              double[] measurementDisturbanceVariates,
                                                              double[] stateDisturbanceVariates,
                                                              double[] initialStateVariates,
                                                              Pool workspace,
                                                              SimulationSmootherOptions resolvedOptions,
                                                              boolean reuseResult) {
                                    KalmanSSM stateSpace = model;
                                    KalmanSSM observationModel = model;
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();

        requireLength("measurement disturbance variates", measurementDisturbanceVariates, 2 * nobs * kEndog);
        requireLength("state disturbance variates", stateDisturbanceVariates, 2 * nobs * kPosdef);
        requireLength("initial state variates", initialStateVariates, 2 * kStates);

        boolean storeState = resolvedOptions.includes(SimulationSmootherOptions.Surface.STATE);
        boolean storeDisturbance = resolvedOptions.includes(SimulationSmootherOptions.Surface.DISTURBANCE);
        boolean storeGeneratedOutputs = resolvedOptions.storesGeneratedOutputs();
        SmootherOptions resolvedSmootherOptions = resolvedOptions.requiredSmootherOptions();

        ZFilterResult actualFilter = filter(model, init, resolvedOptions, workspace.actualFilterWorkspace, null, 0);
        double[] initialState = actualFilter.predictedState;
        int initialStateBase = actualFilter.predictedStateOffset(0);
        double[] initialStateCov = actualFilter.predictedStateCov;
        int initialStateCovBase = actualFilter.predictedStateCovOffset(0);
        ZSmootherResult actualSmooth = ZKalmanSmoother.smooth(model, actualFilter,
            workspace.actualSmootherPool, resolvedSmootherOptions);

        boolean useFilterForecastErrorObsBank = !storeGeneratedOutputs
            && canBorrowGeneratedObsFromFilter(actualFilter, kEndog, nobs);
        boolean useResultObsBank = !useFilterForecastErrorObsBank
            && !storeGeneratedOutputs
            && canBorrowGeneratedObsFromResult(kEndog, kStates, storeState, storeDisturbance);

        ZSimulationSmootherResult result = reuseResult
            ? workspace.borrowResult(kEndog, kStates, kPosdef, nobs,
                storeGeneratedOutputs, storeState, storeDisturbance)
            : new ZSimulationSmootherResult(kEndog, kStates, kPosdef, nobs,
                storeGeneratedOutputs, storeState, storeDisturbance);

        try {
            workspace.ensureCombineWorkspace(kEndog, kStates, kPosdef, nobs,
                useFilterForecastErrorObsBank || useResultObsBank,
                storeGeneratedOutputs);

            double[] generatedObsResultBank = useResultObsBank
                ? borrowedGeneratedObsResultBank(result, kEndog, kStates, storeState, storeDisturbance)
                : null;
            int generatedObsResultBase = generatedObsResultBank == null
                ? 0
                : borrowedGeneratedObsResultBase(result, kEndog, kStates, storeState, storeDisturbance);

            double[] generatedObs = storeGeneratedOutputs
                ? result.generatedObs
                : (useFilterForecastErrorObsBank ? actualFilter.forecastError
                    : (generatedObsResultBank != null ? generatedObsResultBank : workspace.combineWorkspace.generatedObsScratch));
            int generatedObsBase = storeGeneratedOutputs
                ? result.generatedObsBase()
                : (useFilterForecastErrorObsBank
                    ? actualFilter.forecastErrorBase()
                    : (generatedObsResultBank != null ? generatedObsResultBase : 0));
            double[] generatedStatePath = storeGeneratedOutputs
                ? result.generatedState
                : ZSimulationSmootherResult.emptyArray();
            int generatedStateBase = storeGeneratedOutputs ? result.generatedStateBase() : 0;
            double[] generatedMeasurement = storeGeneratedOutputs
                ? result.generatedMeasurementDisturbance
                : workspace.combineWorkspace.generatedMeasurementScratch;
            int generatedMeasurementBase = storeGeneratedOutputs ? result.generatedMeasurementDisturbanceBase() : 0;
            double[] generatedStateDisturbance = storeGeneratedOutputs
                ? result.generatedStateDisturbance
                : workspace.combineWorkspace.generatedStateDisturbanceScratch;
            int generatedStateDisturbanceBase = storeGeneratedOutputs ? result.generatedStateDisturbanceBase() : 0;
            double[] generatedInitialState = workspace.combineWorkspace.generatedInitialStateScratch;
            int simulatedMeasurementBase = result.simulatedMeasurementDisturbanceBase();
            int simulatedStateDisturbanceBase = result.simulatedStateDisturbanceBase();
            int simulatedStateBase = result.simulatedStateBase();

            double[] tmpCholeskyWork = Math.max(kEndog, Math.max(kStates, kPosdef)) <= 1
                ? null
                : workspace.combineWorkspace.tmpCholeskyWork;

            if (storeGeneratedOutputs) {
                initializeGeneratedState(initialState, initialStateBase,
                    initialStateCov, initialStateCovBase,
                    kStates, initialStateVariates,
                    generatedStatePath, generatedStateBase,
                    tmpCholeskyWork);

                for (int t = 0; t < nobs; t++) {
                    int measOff = t * 2 * kEndog;
                    int stateOff = t * 2 * kPosdef;
                    int generatedStateOff = t * 2 * kStates;
                    int nextGeneratedStateOff = (t + 1) * 2 * kStates;

                    transformVariates(observationModel.obsCovData(), observationModel.obsCovOffset(t), kEndog,
                        measurementDisturbanceVariates, measOff,
                        generatedMeasurement, generatedMeasurementBase + measOff,
                        tmpCholeskyWork);
                    if (kPosdef > 0) {
                        transformVariates(stateSpace.stateCovarianceData(), stateSpace.stateCovarianceOffset(t), kPosdef,
                            stateDisturbanceVariates, stateOff,
                            generatedStateDisturbance, generatedStateDisturbanceBase + stateOff,
                            tmpCholeskyWork);
                    }

                    generateObservation(observationModel, kEndog, kStates, t,
                        generatedStatePath, generatedStateBase + generatedStateOff,
                        generatedMeasurement, generatedMeasurementBase + measOff,
                        generatedObs, generatedObsBase + measOff);
                    generateState(stateSpace, kStates, kPosdef, t,
                        generatedStatePath, generatedStateBase + generatedStateOff,
                        generatedStateDisturbance, generatedStateDisturbanceBase + stateOff,
                        generatedStatePath, generatedStateBase + nextGeneratedStateOff);
                }
            } else {
                double[] currentGeneratedState = workspace.combineWorkspace.generatedCurrentStateScratch;
                double[] nextGeneratedState = workspace.combineWorkspace.generatedNextStateScratch;
                initializeGeneratedState(initialState, initialStateBase,
                    initialStateCov, initialStateCovBase,
                    kStates, initialStateVariates,
                    currentGeneratedState, 0,
                    tmpCholeskyWork);
                if (storeState) {
                    ZLAS.zcopy(kStates, currentGeneratedState, 0, 1, generatedInitialState, 0, 1);
                }

                for (int t = 0; t < nobs; t++) {
                    int measOff = t * 2 * kEndog;
                    int stateOff = t * 2 * kPosdef;

                    transformVariates(observationModel.obsCovData(), observationModel.obsCovOffset(t), kEndog,
                        measurementDisturbanceVariates, measOff,
                        generatedMeasurement, 0,
                        tmpCholeskyWork);
                    if (kPosdef > 0) {
                        transformVariates(stateSpace.stateCovarianceData(), stateSpace.stateCovarianceOffset(t), kPosdef,
                            stateDisturbanceVariates, stateOff,
                            generatedStateDisturbance, 0,
                            tmpCholeskyWork);
                    }

                    generateObservation(observationModel, kEndog, kStates, t,
                        currentGeneratedState, 0,
                        generatedMeasurement, 0,
                        generatedObs, generatedObsBase + measOff);
                    generateState(stateSpace, kStates, kPosdef, t,
                        currentGeneratedState, 0,
                        generatedStateDisturbance, 0,
                        nextGeneratedState, 0);
                    double[] swap = currentGeneratedState;
                    currentGeneratedState = nextGeneratedState;
                    nextGeneratedState = swap;
                }
            }

            ZFilterResult generatedFilter = filter(observationModel, init, resolvedOptions,
                workspace.generatedFilterWorkspace, generatedObs, generatedObsBase);
            ZSmootherResult generatedSmooth = reuseResult
                ? workspace.borrowGeneratedSmoothTarget(result, kEndog, kStates, kPosdef, nobs, storeState, storeDisturbance)
                : new ZSmootherResult().reuseExternal(kEndog, kStates, kPosdef, nobs,
                    storeState,
                    storeDisturbance,
                    result.simulatedState,
                    result.simulatedStateBase(),
                    result.simulatedMeasurementDisturbance,
                    result.simulatedMeasurementDisturbanceBase(),
                    result.simulatedStateDisturbance,
                    result.simulatedStateDisturbanceBase());
            ZKalmanSmoother.smoothInto(stateSpace, observationModel, generatedFilter,
                generatedSmooth, workspace.generatedSmootherPool, resolvedSmootherOptions);

            if (storeState) {
                if (storeGeneratedOutputs) {
                    for (int t = 0; t < nobs; t++) {
                        int off = t * 2 * kStates;
                        int actualStateOff = actualSmooth.smoothedStateOffset(t);
                        for (int i = 0; i < 2 * kStates; i++) {
                            result.simulatedState[simulatedStateBase + off + i] = generatedStatePath[generatedStateBase + off + i]
                                - result.simulatedState[simulatedStateBase + off + i]
                                + actualSmooth.smoothedState[actualStateOff + i];
                        }
                    }
                } else {
                    double[] currentGeneratedState = workspace.combineWorkspace.generatedCurrentStateScratch;
                    double[] nextGeneratedState = workspace.combineWorkspace.generatedNextStateScratch;
                    ZLAS.zcopy(kStates, generatedInitialState, 0, 1, currentGeneratedState, 0, 1);

                    for (int t = 0; t < nobs; t++) {
                        int off = t * 2 * kStates;
                        int stateOff = t * 2 * kPosdef;
                        int actualStateOff = actualSmooth.smoothedStateOffset(t);
                        for (int i = 0; i < 2 * kStates; i++) {
                            result.simulatedState[simulatedStateBase + off + i] = currentGeneratedState[i]
                                - result.simulatedState[simulatedStateBase + off + i]
                                + actualSmooth.smoothedState[actualStateOff + i];
                        }
                        if (kPosdef > 0) {
                            transformVariates(stateSpace.stateCovarianceData(), stateSpace.stateCovarianceOffset(t), kPosdef,
                                stateDisturbanceVariates, stateOff,
                                generatedStateDisturbance, 0,
                                tmpCholeskyWork);
                        }
                        generateState(stateSpace, kStates, kPosdef, t,
                            currentGeneratedState, 0,
                            generatedStateDisturbance, 0,
                            nextGeneratedState, 0);
                        double[] swap = currentGeneratedState;
                        currentGeneratedState = nextGeneratedState;
                        nextGeneratedState = swap;
                    }
                }
            }

            if (storeDisturbance) {
                for (int t = 0; t < nobs; t++) {
                    int obsOff = t * 2 * kEndog;
                    int stateOff = t * 2 * kPosdef;
                    if (!storeGeneratedOutputs) {
                        transformVariates(observationModel.obsCovData(), observationModel.obsCovOffset(t), kEndog,
                            measurementDisturbanceVariates, obsOff,
                            generatedMeasurement, 0,
                            tmpCholeskyWork);
                        if (kPosdef > 0) {
                            transformVariates(stateSpace.stateCovarianceData(), stateSpace.stateCovarianceOffset(t), kPosdef,
                                stateDisturbanceVariates, stateOff,
                                generatedStateDisturbance, 0,
                                tmpCholeskyWork);
                        }
                    }
                    int actualObsOff = actualSmooth.smoothedObsDisturbanceOffset(t);
                    int generatedMeasurementOff = storeGeneratedOutputs ? generatedMeasurementBase + obsOff : 0;
                    for (int i = 0; i < 2 * kEndog; i++) {
                        result.simulatedMeasurementDisturbance[simulatedMeasurementBase + obsOff + i] = generatedMeasurement[generatedMeasurementOff + i]
                            - result.simulatedMeasurementDisturbance[simulatedMeasurementBase + obsOff + i]
                            + actualSmooth.smoothedObsDisturbance[actualObsOff + i];
                    }
                    int actualStateDistOff = actualSmooth.smoothedStateDisturbanceOffset(t);
                    int generatedStateDisturbanceOff = storeGeneratedOutputs ? generatedStateDisturbanceBase + stateOff : 0;
                    for (int i = 0; i < 2 * kPosdef; i++) {
                        result.simulatedStateDisturbance[simulatedStateDisturbanceBase + stateOff + i] = generatedStateDisturbance[generatedStateDisturbanceOff + i]
                            - result.simulatedStateDisturbance[simulatedStateDisturbanceBase + stateOff + i]
                            + actualSmooth.smoothedStateDisturbance[actualStateDistOff + i];
                    }
                }
            }

            return result;
        } finally {
            if (reuseResult) {
                workspace.releaseTransientRunState();
            }
        }
    }

    static ZSimulationSmootherResult simulate(KalmanSSM model,
                                              InitialState init,
                                              Random rng,
                                              Pool pool,
                                              SimulationSmootherOptions spec) {
        requireComplexInputs(model, model);
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(rng, "rng");

        SimulationSmootherOptions resolvedOptions = spec == null ? SimulationSmootherOptions.defaults() : spec;
        boolean reuseResult = pool != null;
        Pool workspace = reuseResult ? pool : new Pool();
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        int kPosdef = model.stateDisturbanceCount();
        int nobs = model.observationCount();
        workspace.reserve(model, init, resolvedOptions);
        workspace.ensureVariates(kEndog, kStates, kPosdef, nobs);

        double[] measurementDisturbanceVariates = reuseResult
            ? workspace.measurementDisturbanceVariates
            : new double[2 * nobs * kEndog];
        double[] stateDisturbanceVariates = reuseResult
            ? workspace.stateDisturbanceVariates
            : new double[2 * nobs * kPosdef];
        double[] initialStateVariates = reuseResult
            ? workspace.initialStateVariates
            : new double[2 * kStates];
        fillGaussian(measurementDisturbanceVariates, rng);
        fillGaussian(stateDisturbanceVariates, rng);
        fillGaussian(initialStateVariates, rng);
        return simulateInternal(model, init,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            workspace,
            resolvedOptions,
            reuseResult);
    }

    private static void requireSupportedOptions(SimulationSmootherOptions spec) {
        if (spec.usesUnivariateFilter()) {
            throw new IllegalArgumentException("Complex simulation smoothing does not support the univariate filter profile");
        }
    }

    private static ZFilterResult filter(KalmanSSM model,
                                        InitialState init,
                                        SimulationSmootherOptions spec,
                        KalmanEngine.Workspace filterWorkspace,
                                        double[] endogOverride,
                                        int endogBase) {
        FilterOptions filterOptions = spec.requiredFilterOptions();
        return endogOverride == null
            ? KalmanEngine.filterComplexBorrowedUnsafe(model, init, filterWorkspace, filterOptions)
            : KalmanEngine.filterComplexBorrowedUnsafe(model, init, endogOverride, endogBase, filterWorkspace, filterOptions);
    }

    private static boolean canBorrowGeneratedObsFromResult(int kEndog, int kStates,
                                                           boolean storeState, boolean storeDisturbance) {
        return storeDisturbance || (storeState && kStates >= kEndog);
    }

    private static boolean canBorrowGeneratedObsFromFilter(ZFilterResult filter, int kEndog, int nobs) {
        return filter.forecastError != null
            && filter.forecastError.length - filter.forecastErrorBase() >= 2 * kEndog * nobs;
    }

    private static double[] borrowedGeneratedObsResultBank(ZSimulationSmootherResult result,
                                                           int kEndog, int kStates,
                                                           boolean storeState, boolean storeDisturbance) {
        if (storeDisturbance) {
            return result.simulatedMeasurementDisturbance;
        }
        if (storeState && kStates >= kEndog) {
            return result.simulatedState;
        }
        return null;
    }

    private static int borrowedGeneratedObsResultBase(ZSimulationSmootherResult result,
                                                      int kEndog, int kStates,
                                                      boolean storeState, boolean storeDisturbance) {
        if (storeDisturbance) {
            return result.simulatedMeasurementDisturbanceBase();
        }
        if (storeState && kStates >= kEndog) {
            return result.simulatedStateBase();
        }
        return 0;
    }

    private static void requireLength(String name, double[] values, int expected) {
        if (values.length != expected) {
            throw new IllegalArgumentException(name + " length must be " + expected + ", got " + values.length);
        }
    }

    private static void fillGaussian(double[] target, Random rng) {
        for (int i = 0; i < target.length; i++) {
            target[i] = rng.nextGaussian();
        }
    }

    private static void initializeGeneratedState(double[] initialState,
                                                 int initialStateOff,
                                                 double[] initialStateCov,
                                                 int initialStateCovOff,
                                                 int kStates,
                                                 double[] initialStateVariates,
                                                 double[] target,
                                                 int targetOff,
                                                 double[] tmpCholeskyWork) {
        transformVariates(initialStateCov, initialStateCovOff, kStates,
            initialStateVariates, 0,
            target, targetOff,
            tmpCholeskyWork);
        ZLAS.zaxpy(kStates, 1.0, 0.0, initialState, initialStateOff / 2, 1, target, targetOff / 2, 1);
    }

    private static void transformVariates(double[] covariance, int covarianceOff, int dim,
                                          double[] variates, int variatesOff,
                                          double[] target, int targetOff,
                                          double[] tmpCholesky) {
        if (dim == 0) {
            return;
        }

        ZLAS.zcopy(dim, variates, variatesOff, 1, target, targetOff, 1);
        if (dim == 1) {
            ZLAS.zscal(dim, Math.sqrt(covariance[covarianceOff]), 0.0, target, targetOff, 1);
            return;
        }

        ZLAS.zcopy(dim * dim, covariance, covarianceOff, 1, tmpCholesky, 0, 1);
        ZLAS.zpotrf(BLAS.Uplo.Lower, dim, tmpCholesky, 0, dim);
        ZLAS.ztrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
            dim, tmpCholesky, 0, dim, target, targetOff, 1);
    }

    private static void generateObservation(KalmanSSM observationModel,
                                            int kEndog,
                                            int kStates,
                                            int t,
                                            double[] state, int stateOff,
                                            double[] disturbance, int disturbanceOff,
                                            double[] target, int targetOff) {
        ZLAS.zcopy(kEndog, observationModel.obsInterceptData(), observationModel.obsInterceptOffset(t), 1, target, targetOff, 1);
        ZLAS.zaxpy(kEndog, 1.0, 0.0, disturbance, disturbanceOff / 2, 1, target, targetOff / 2, 1);
        ZLAS.zgemv(BLAS.Trans.NoTrans, kEndog, kStates, 1.0, 0.0,
            observationModel.designData(), observationModel.designOffset(t), observationModel.designLeadingDimension(),
            state, stateOff, 1,
            1.0, 0.0, target, targetOff, 1);
    }

    private static void generateState(KalmanSSM stateSpace,
                                      int kStates,
                                      int kPosdef,
                                      int t,
                                      double[] state, int stateOff,
                                      double[] disturbance, int disturbanceOff,
                                      double[] target, int targetOff) {
        ZLAS.zcopy(kStates, stateSpace.stateInterceptData(), stateSpace.stateInterceptOffset(t), 1, target, targetOff, 1);
        if (kPosdef > 0) {
            ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kPosdef, 1.0, 0.0,
                stateSpace.selectionData(), stateSpace.selectionOffset(t), stateSpace.selectionLeadingDimension(),
                disturbance, disturbanceOff, 1,
                1.0, 0.0, target, targetOff, 1);
        }
        ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
            stateSpace.transitionData(), stateSpace.transitionOffset(t), stateSpace.transitionLeadingDimension(),
            state, stateOff, 1,
            1.0, 0.0, target, targetOff, 1);
    }

}