package com.curioloop.yum4j.kalman.smooth;

import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.kalman.filter.ZKalmanFilter;
import com.curioloop.yum4j.kalman.arena.SimulationResultLayout;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.internal.StateSpaceModelSupport;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;

import java.util.Objects;
import java.util.Random;

/**
 * Complex-valued KFS simulation smoother built on the complex filter and smoother.
 */
public final class ZSimulationSmoother {

    private static void requireComplexInputs(StateSpaceModel stateSpace,
                                             StateSpaceModel observationModel) {
        StateSpaceModelSupport.requireComplexStorage(stateSpace, "stateSpace");
        StateSpaceModelSupport.requireComplexStorage(observationModel, "observationModel");
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

    public static final class Pool {
        private ZKalmanFilter.Pool actualFilterPool;
        private ZKalmanFilter.Pool generatedFilterPool;
        private ZKalmanSmoother.Pool actualSmootherPool;
        private ZKalmanSmoother.Pool generatedSmootherPool;
        private CombineWorkspace combineWorkspace;
        private double[] measurementDisturbanceVariates;
        private double[] stateDisturbanceVariates;
        private double[] initialStateVariates;

        private ZSimulationSmootherResult pooledResult;
        private ZSmootherResult generatedSmoothTarget;
        private double[] resultBacking;
        private SimulationResultLayout resultLayout;
        private int resultLayoutKEndog = -1;
        private int resultLayoutKStates = -1;
        private int resultLayoutKPosdef = -1;
        private int resultLayoutNobs = -1;
        private int resultLayoutOptions = -1;

        void ensure(int kEndog, int kStates, int kPosdef) {
            if (actualFilterPool == null) {
                actualFilterPool = new ZKalmanFilter.Pool();
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
            generatedFilterPool = actualFilterPool;
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
            if (resultBacking == null || resultBacking.length != totalLength) {
                resultBacking = new double[totalLength];
            }
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
            if (actualFilterPool != null) {
                actualFilterPool.releaseRetainedResults();
            }
            if (actualSmootherPool != null) {
                actualSmootherPool.releaseRetainedResults();
            }
            if (generatedSmootherPool != null) {
                generatedSmootherPool.releaseRetainedResults();
            }
        }

        void releaseCombineWorkspace() {
            if (combineWorkspace != null) {
                combineWorkspace.releaseBackings();
            }
        }

        void releaseTransientRunState() {
            releaseTransientNestedResults();
            releaseCombineWorkspace();
            releaseVariates();
        }

        private boolean generatedFilterPoolShared() {
            return generatedFilterPool == actualFilterPool;
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
            return combineWorkspace == null ? 0L : combineWorkspace.retainedDoubleCount();
        }

        public long retainedVariateDoubleCount() {
            return doubleCount(measurementDisturbanceVariates)
                + doubleCount(stateDisturbanceVariates)
                + doubleCount(initialStateVariates);
        }

        public long retainedResultDoubleCount() {
            return doubleCount(resultBacking);
        }

        public long retainedNestedScratchDoubleCount() {
            long total = 0L;
            if (actualFilterPool != null) {
                total += actualFilterPool.retainedScratchDoubleCount();
            }
            if (generatedFilterPool != null && !generatedFilterPoolShared()) {
                total += generatedFilterPool.retainedScratchDoubleCount();
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
            if (actualFilterPool != null) {
                total += actualFilterPool.retainedResultDoubleCount();
            }
            if (generatedFilterPool != null && !generatedFilterPoolShared()) {
                total += generatedFilterPool.retainedResultDoubleCount();
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
            if (actualFilterPool != null) {
                actualFilterPool.releaseRetainedScratch();
            }
            if (generatedFilterPool != null && !generatedFilterPoolShared()) {
                generatedFilterPool.releaseRetainedScratch();
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

        public void reserve(StateSpaceModel model,
                            StateInitialization init) {
            reserve(model, init, SimulationSmootherSpec.all());
        }

        public void reserve(StateSpaceModel model,
                            StateInitialization init,
                            SimulationSmootherSpec spec) {
            requireComplexInputs(model, model);
            Objects.requireNonNull(init, "init");
            SimulationSmootherSpec resolvedSpec = spec == null ? SimulationSmootherSpec.all() : spec;
            requireSupportedSpec(resolvedSpec);

            ensure(model.observationDimension(), model.stateCount(), model.stateDisturbanceCount());
            boolean storeGeneratedOutputs = resolvedSpec.storesGeneratedOutputs();
            boolean storeState = resolvedSpec.includes(SimulationSmootherSpec.Output.STATE);
            boolean storeDisturbance = resolvedSpec.includes(SimulationSmootherSpec.Output.DISTURBANCE);

            actualFilterPool.reserve(model, resolvedSpec.requiredFilterSpec());
            actualSmootherPool.releaseRetainedResults();
            generatedSmootherPool.releaseRetainedResults();
            actualSmootherPool.releaseRetainedScratch();
            generatedSmootherPool.shareScratchFrom(actualSmootherPool);
            ensureResult(model.observationDimension(), model.stateCount(), model.stateDisturbanceCount(), model.observationCount(),
                storeGeneratedOutputs, storeState, storeDisturbance);
        }
    }

    private ZSimulationSmoother() {
    }

    public static ZSimulationSmootherResult simulate(StateSpaceModel model,
                                                     StateInitialization init,
                                                     double[] measurementDisturbanceVariates,
                                                     double[] stateDisturbanceVariates,
                                                     double[] initialStateVariates) {
        return simulate(model, init, measurementDisturbanceVariates,
            stateDisturbanceVariates, initialStateVariates, null, SimulationSmootherSpec.all());
    }

    public static ZSimulationSmootherResult simulate(StateSpaceModel model,
                                                     StateInitialization init,
                                                     double[] measurementDisturbanceVariates,
                                                     double[] stateDisturbanceVariates,
                                                     double[] initialStateVariates,
                                                     Pool pool) {
        return simulate(model, init, measurementDisturbanceVariates,
            stateDisturbanceVariates, initialStateVariates, pool, SimulationSmootherSpec.all());
    }

    public static ZSimulationSmootherResult simulate(StateSpaceModel model,
                                                     StateInitialization init,
                                                     double[] measurementDisturbanceVariates,
                                                     double[] stateDisturbanceVariates,
                                                     double[] initialStateVariates,
                                                     SimulationSmootherSpec spec) {
        return simulate(model, init, measurementDisturbanceVariates,
            stateDisturbanceVariates, initialStateVariates, null, spec);
    }

    public static ZSimulationSmootherResult simulate(StateSpaceModel model,
                                                     StateInitialization init,
                                                     double[] measurementDisturbanceVariates,
                                                     double[] stateDisturbanceVariates,
                                                     double[] initialStateVariates,
                                                     Pool pool,
                                                     SimulationSmootherSpec spec) {
        requireComplexInputs(model, model);
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(measurementDisturbanceVariates, "measurementDisturbanceVariates");
        Objects.requireNonNull(stateDisturbanceVariates, "stateDisturbanceVariates");
        Objects.requireNonNull(initialStateVariates, "initialStateVariates");
        SimulationSmootherSpec resolvedSpec = spec == null ? SimulationSmootherSpec.all() : spec;

        boolean reuseResult = pool != null;
        Pool workspace = reuseResult ? pool : new Pool();
        workspace.reserve(model, init, resolvedSpec);

        return simulateInternal(model, init,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            workspace,
            resolvedSpec,
            reuseResult);
    }

    /**
     * When a non-null pool is supplied, the returned result borrows pool-owned storage
     * and is invalidated by the next call that reuses the same pool. Call clone() or copy()
     * on the result when an independent snapshot is required.
     */
    private static ZSimulationSmootherResult simulateInternal(StateSpaceModel model,
                                                              StateInitialization init,
                                                              double[] measurementDisturbanceVariates,
                                                              double[] stateDisturbanceVariates,
                                                              double[] initialStateVariates,
                                                              Pool workspace,
                                                              SimulationSmootherSpec resolvedSpec,
                                                              boolean reuseResult) {
                                    StateSpaceModel stateSpace = model;
                                    StateSpaceModel observationModel = model;
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();

        requireLength("measurement disturbance variates", measurementDisturbanceVariates, 2 * nobs * kEndog);
        requireLength("state disturbance variates", stateDisturbanceVariates, 2 * nobs * kPosdef);
        requireLength("initial state variates", initialStateVariates, 2 * kStates);

        boolean storeState = resolvedSpec.includes(SimulationSmootherSpec.Output.STATE);
        boolean storeDisturbance = resolvedSpec.includes(SimulationSmootherSpec.Output.DISTURBANCE);
        boolean storeGeneratedOutputs = resolvedSpec.storesGeneratedOutputs();
        boolean useFilterForecastErrorObsBank = !storeGeneratedOutputs;
        boolean useResultObsBank = !storeGeneratedOutputs
            && canBorrowGeneratedObsFromResult(kEndog, kStates, storeState, storeDisturbance);
        SmootherSpec resolvedSmootherSpec = resolvedSpec.requiredSmootherSpec();

        ZFilterResult actualFilter = filter(model, init, resolvedSpec, workspace.actualFilterPool, null, 0);
        double[] initialState = actualFilter.predictedState;
        int initialStateBase = actualFilter.predictedStateOffset(0);
        double[] initialStateCov = actualFilter.predictedStateCov;
        int initialStateCovBase = actualFilter.predictedStateCovOffset(0);
        ZSmootherResult actualSmooth = ZKalmanSmoother.smooth(model, actualFilter,
            workspace.actualSmootherPool, resolvedSmootherSpec);

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
                : (useFilterForecastErrorObsBank ? workspace.actualFilterPool.resultBacking
                    : (generatedObsResultBank != null ? generatedObsResultBank : workspace.combineWorkspace.generatedObsScratch));
            int generatedObsBase = storeGeneratedOutputs
                ? result.generatedObsBase()
                : (useFilterForecastErrorObsBank
                    ? workspace.actualFilterPool.resultLayout.forecastErrorBase()
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

            ZFilterResult generatedFilter = filter(observationModel, init, resolvedSpec,
                workspace.generatedFilterPool, generatedObs, generatedObsBase);
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
                generatedSmooth, workspace.generatedSmootherPool, resolvedSmootherSpec);

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

    public static ZSimulationSmootherResult simulate(StateSpaceModel model,
                                                     StateInitialization init,
                                                     Random rng) {
        return simulate(model, init, rng, null, SimulationSmootherSpec.all());
    }

    public static ZSimulationSmootherResult simulate(StateSpaceModel model,
                                                     StateInitialization init,
                                                     Random rng,
                                                     Pool pool) {
        return simulate(model, init, rng, pool, SimulationSmootherSpec.all());
    }

    public static ZSimulationSmootherResult simulate(StateSpaceModel model,
                                                     StateInitialization init,
                                                     Random rng,
                                                     SimulationSmootherSpec spec) {
        return simulate(model, init, rng, null, spec);
    }

    public static ZSimulationSmootherResult simulate(StateSpaceModel model,
                                                     StateInitialization init,
                                                     Random rng,
                                                     Pool pool,
                                                     SimulationSmootherSpec spec) {
        requireComplexInputs(model, model);
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(rng, "rng");

        SimulationSmootherSpec resolvedSpec = spec == null ? SimulationSmootherSpec.all() : spec;
        boolean reuseResult = pool != null;
        Pool workspace = reuseResult ? pool : new Pool();
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        int kPosdef = model.stateDisturbanceCount();
        int nobs = model.observationCount();
        workspace.reserve(model, init, resolvedSpec);
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
            resolvedSpec,
            reuseResult);
    }

    private static void requireSupportedSpec(SimulationSmootherSpec spec) {
        if (spec.usesUnivariateFilter()) {
            throw new IllegalArgumentException("Complex simulation smoothing does not support the univariate filter profile");
        }
    }

    private static ZFilterResult filter(StateSpaceModel model,
                                        StateInitialization init,
                                        SimulationSmootherSpec spec,
                                        ZKalmanFilter.Pool filterPool,
                                        double[] endogOverride,
                                        int endogBase) {
        FilterSpec filterSpec = spec.requiredFilterSpec();
        return endogOverride == null
            ? ZKalmanFilter.filter(model, init, filterPool, filterSpec)
            : ZKalmanFilter.filter(model, init, endogOverride, endogBase, filterPool, filterSpec);
    }

    private static boolean canBorrowGeneratedObsFromResult(int kEndog, int kStates,
                                                           boolean storeState, boolean storeDisturbance) {
        return storeDisturbance || (storeState && kStates >= kEndog);
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

    private static void generateObservation(StateSpaceModel observationModel,
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

    private static void generateState(StateSpaceModel stateSpace,
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