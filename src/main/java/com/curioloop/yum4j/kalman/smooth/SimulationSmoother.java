package com.curioloop.yum4j.kalman.smooth;

import com.curioloop.yum4j.kalman.filter.FilterResult;
import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.filter.KalmanFilter;
import com.curioloop.yum4j.kalman.filter.UnivariateFilter;
import com.curioloop.yum4j.kalman.arena.SimulationResultLayout;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.internal.StateSpaceModelSupport;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Random;
import java.util.Objects;

/**
 * Real-valued KFS simulation smoother built on the existing filter/smoother.
 */
public final class SimulationSmoother {

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
            if (maxCholeskyDim > 1 && (tmpCholeskyWork == null || tmpCholeskyWork.length < maxCholeskyDim * maxCholeskyDim)) {
                tmpCholeskyWork = new double[maxCholeskyDim * maxCholeskyDim];
            }
            generatedObsScratch = storeGeneratedOutputs || useBorrowedObsBank
                ? SimulationSmootherResult.emptyArray()
                : Pool.exactSized(generatedObsScratch, kEndog * nobs);
            if (storeGeneratedOutputs) {
                generatedInitialStateScratch = SimulationSmootherResult.emptyArray();
                generatedCurrentStateScratch = SimulationSmootherResult.emptyArray();
                generatedNextStateScratch = SimulationSmootherResult.emptyArray();
            } else {
                generatedInitialStateScratch = Pool.exactSized(generatedInitialStateScratch, kStates);
                generatedCurrentStateScratch = Pool.exactSized(generatedCurrentStateScratch, kStates);
                generatedNextStateScratch = Pool.exactSized(generatedNextStateScratch, kStates);
            }
            if (storeGeneratedOutputs) {
                generatedMeasurementScratch = SimulationSmootherResult.emptyArray();
                generatedStateDisturbanceScratch = SimulationSmootherResult.emptyArray();
            } else {
                generatedMeasurementScratch = Pool.exactSized(generatedMeasurementScratch, kEndog);
                generatedStateDisturbanceScratch = Pool.exactSized(generatedStateDisturbanceScratch, kPosdef);
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
        private KalmanFilter.Pool actualFilterPool;
        private KalmanFilter.Pool generatedFilterPool;
        private UnivariateFilter.Pool actualUnivariateFilterPool;
        private UnivariateFilter.Pool generatedUnivariateFilterPool;
        private KalmanSmoother.Pool actualSmootherPool;
        private KalmanSmoother.Pool generatedSmootherPool;
        private CombineWorkspace combineWorkspace;
        private double[] measurementDisturbanceVariates;
        private double[] stateDisturbanceVariates;
        private double[] initialStateVariates;

        private SimulationSmootherResult pooledResult;
        private SmootherResult generatedSmoothTarget;
        private double[] resultBacking;
        private SimulationResultLayout resultLayout;
        private int resultLayoutKEndog = -1;
        private int resultLayoutKStates = -1;
        private int resultLayoutKPosdef = -1;
        private int resultLayoutNobs = -1;
        private int resultLayoutOptions = -1;

        void ensure(int kEndog, int kStates, int kPosdef) {
            if (actualFilterPool == null) {
                actualFilterPool = new KalmanFilter.Pool();
            }
            if (actualUnivariateFilterPool == null) {
                actualUnivariateFilterPool = new UnivariateFilter.Pool();
            }
            if (actualSmootherPool == null) {
                actualSmootherPool = new KalmanSmoother.Pool();
            }
            if (generatedSmootherPool == null) {
                generatedSmootherPool = new KalmanSmoother.Pool();
            }
            if (combineWorkspace == null) {
                combineWorkspace = new CombineWorkspace();
            }
            generatedFilterPool = actualFilterPool;
            generatedUnivariateFilterPool = actualUnivariateFilterPool;
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
                resultLayout = SimulationResultLayout.create(1, kEndog, kStates, kPosdef, nobs,
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
            measurementDisturbanceVariates = exactSized(measurementDisturbanceVariates, kEndog * nobs);
            stateDisturbanceVariates = exactSized(stateDisturbanceVariates, kPosdef * nobs);
            initialStateVariates = exactSized(initialStateVariates, kStates);
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

        private boolean generatedFilterPoolsShared() {
            return generatedFilterPool == actualFilterPool
                && generatedUnivariateFilterPool == actualUnivariateFilterPool;
        }

        private boolean generatedSmootherScratchShared() {
            return generatedSmootherPool == actualSmootherPool
                || (generatedSmootherPool != null
                    && actualSmootherPool != null
                    && generatedSmootherPool.sharesScratchWith(actualSmootherPool));
        }

        private static long retainedFilterScratchDoubleCount(KalmanFilter.Pool filterPool,
                                                             UnivariateFilter.Pool univariateFilterPool) {
            return (filterPool == null ? 0L : filterPool.retainedScratchDoubleCount())
                + (univariateFilterPool == null ? 0L : univariateFilterPool.retainedScratchDoubleCount());
        }

        private static long retainedFilterResultDoubleCount(KalmanFilter.Pool filterPool) {
            return filterPool == null ? 0L : filterPool.retainedResultDoubleCount();
        }

        private static void releaseFilterScratch(KalmanFilter.Pool filterPool,
                                                 UnivariateFilter.Pool univariateFilterPool) {
            if (filterPool != null) {
                filterPool.releaseRetainedScratch();
            }
            if (univariateFilterPool != null) {
                univariateFilterPool.releaseRetainedScratch();
            }
        }

        SimulationSmootherResult borrowResult(int kEndog, int kStates, int kPosdef, int nobs,
                                              boolean storeGeneratedOutputs,
                                              boolean storeState, boolean storeDisturbance) {
            ensureResult(kEndog, kStates, kPosdef, nobs, storeGeneratedOutputs, storeState, storeDisturbance);
            if (pooledResult == null) {
                pooledResult = new SimulationSmootherResult();
            }
            return pooledResult.reuse(kEndog, kStates, kPosdef, nobs,
                resultBacking, resultLayout);
        }

        SmootherResult borrowGeneratedSmoothTarget(SimulationSmootherResult result,
                                                   int kEndog, int kStates, int kPosdef, int nobs,
                                                   boolean storeState, boolean storeDisturbance) {
            if (generatedSmoothTarget == null) {
                generatedSmoothTarget = new SmootherResult();
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
                return SimulationSmootherResult.emptyArray();
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
            if (actualFilterPool != null || actualUnivariateFilterPool != null) {
                total += retainedFilterScratchDoubleCount(actualFilterPool, actualUnivariateFilterPool);
            }
            if ((generatedFilterPool != null || generatedUnivariateFilterPool != null)
                    && !generatedFilterPoolsShared()) {
                total += retainedFilterScratchDoubleCount(generatedFilterPool, generatedUnivariateFilterPool);
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
                total += retainedFilterResultDoubleCount(actualFilterPool);
            }
            if (generatedFilterPool != null && !generatedFilterPoolsShared()) {
                total += retainedFilterResultDoubleCount(generatedFilterPool);
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
            if (actualFilterPool != null || actualUnivariateFilterPool != null) {
                releaseFilterScratch(actualFilterPool, actualUnivariateFilterPool);
            }
            if ((generatedFilterPool != null || generatedUnivariateFilterPool != null)
                    && !generatedFilterPoolsShared()) {
                releaseFilterScratch(generatedFilterPool, generatedUnivariateFilterPool);
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
            Objects.requireNonNull(model, "model");
            Objects.requireNonNull(init, "init");
            requireRealInputs(model);
            SimulationSmootherSpec resolvedSpec = spec == null ? SimulationSmootherSpec.all() : spec;
            ensure(model.observationDimension(), model.stateCount(), model.stateDisturbanceCount());
            boolean storeGeneratedOutputs = resolvedSpec.storesGeneratedOutputs();
            boolean storeState = resolvedSpec.includes(SimulationSmootherSpec.Output.STATE);
            boolean storeDisturbance = resolvedSpec.includes(SimulationSmootherSpec.Output.DISTURBANCE);

            if (!resolvedSpec.usesUnivariateFilter()) {
                actualUnivariateFilterPool.releaseRetainedScratch();
                actualFilterPool.reserve(model, resolvedSpec.requiredFilterSpec());
            } else {
                actualFilterPool.releaseRetainedScratch();
                actualFilterPool.releaseRetainedResults();
                actualUnivariateFilterPool.reserve(model, resolvedSpec.requiredFilterSpec());
            }

            actualSmootherPool.releaseRetainedResults();
            generatedSmootherPool.releaseRetainedResults();
            actualSmootherPool.releaseRetainedScratch();
            generatedSmootherPool.shareScratchFrom(actualSmootherPool);
            ensureResult(model.observationDimension(), model.stateCount(), model.stateDisturbanceCount(), model.observationCount(),
                storeGeneratedOutputs, storeState, storeDisturbance);
        }
    }

    private SimulationSmoother() {
    }

    public static SimulationSmootherResult simulate(StateSpaceModel model,
                                                    StateInitialization init,
                                                    double[] measurementDisturbanceVariates,
                                                    double[] stateDisturbanceVariates,
                                                    double[] initialStateVariates) {
        return simulate(model, init, measurementDisturbanceVariates,
                stateDisturbanceVariates, initialStateVariates, null, SimulationSmootherSpec.all());
    }

    public static SimulationSmootherResult simulate(StateSpaceModel model,
                                                    StateInitialization init,
                                                    double[] measurementDisturbanceVariates,
                                                    double[] stateDisturbanceVariates,
                                                    double[] initialStateVariates,
                                                    Pool pool) {
        return simulate(model, init, measurementDisturbanceVariates,
                stateDisturbanceVariates, initialStateVariates, pool, SimulationSmootherSpec.all());
    }

    public static SimulationSmootherResult simulate(StateSpaceModel model,
                                                    StateInitialization init,
                                                    double[] measurementDisturbanceVariates,
                                                    double[] stateDisturbanceVariates,
                                                    double[] initialStateVariates,
                                                    SimulationSmootherSpec spec) {
        return simulate(model, init, measurementDisturbanceVariates,
                stateDisturbanceVariates, initialStateVariates, null, spec);
    }

    public static SimulationSmootherResult simulate(StateSpaceModel model,
                                                    StateInitialization init,
                                                    double[] measurementDisturbanceVariates,
                                                    double[] stateDisturbanceVariates,
                                                    double[] initialStateVariates,
                                                    Pool pool,
                                                    SimulationSmootherSpec spec) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(measurementDisturbanceVariates, "measurementDisturbanceVariates");
        Objects.requireNonNull(stateDisturbanceVariates, "stateDisturbanceVariates");
        Objects.requireNonNull(initialStateVariates, "initialStateVariates");
        requireRealInputs(model);
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
        private static SimulationSmootherResult simulateInternal(StateSpaceModel model,
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

        requireLength("measurement disturbance variates", measurementDisturbanceVariates, nobs * kEndog);
        requireLength("state disturbance variates", stateDisturbanceVariates, nobs * kPosdef);
        requireLength("initial state variates", initialStateVariates, kStates);

        boolean storeState = resolvedSpec.includes(SimulationSmootherSpec.Output.STATE);
        boolean storeDisturbance = resolvedSpec.includes(SimulationSmootherSpec.Output.DISTURBANCE);
        boolean storeGeneratedOutputs = resolvedSpec.storesGeneratedOutputs();
        boolean useFilterForecastErrorObsBank = !storeGeneratedOutputs && !resolvedSpec.usesUnivariateFilter();
        boolean useResultObsBank = !storeGeneratedOutputs && resolvedSpec.usesUnivariateFilter()
            && canBorrowGeneratedObsFromResult(kEndog, kStates, storeState, storeDisturbance);
        SmootherSpec resolvedSmootherSpec = resolvedSpec.requiredSmootherSpec();

        FilterResult actualFilter = filter(model, init, resolvedSpec,
            workspace.actualFilterPool, workspace.actualUnivariateFilterPool, null, 0);
        double[] initialState = actualFilter.predictedState;
        int initialStateBase = actualFilter.predictedStateOffset(0);
        double[] initialStateCov = actualFilter.predictedStateCov;
        int initialStateCovBase = actualFilter.predictedStateCovOffset(0);
        SmootherResult actualSmooth = KalmanSmoother.smooth(model, actualFilter,
            workspace.actualSmootherPool, resolvedSmootherSpec);

        SimulationSmootherResult result = reuseResult
            ? workspace.borrowResult(kEndog, kStates, kPosdef, nobs,
            storeGeneratedOutputs, storeState, storeDisturbance)
            : new SimulationSmootherResult(kEndog, kStates, kPosdef, nobs,
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
                : SimulationSmootherResult.emptyArray();
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
                int measOff = t * kEndog;
                int stateOff = t * kPosdef;
                int generatedStateOff = t * kStates;
                int nextGeneratedStateOff = (t + 1) * kStates;
                int disturbanceObsOff = storeGeneratedOutputs ? generatedMeasurementBase + measOff : 0;
                int disturbanceStateOff = storeGeneratedOutputs ? generatedStateDisturbanceBase + stateOff : 0;

                transformVariates(observationModel.obsCovData(), observationModel.obsCovOffset(t), kEndog,
                    measurementDisturbanceVariates, measOff,
                    generatedMeasurement, disturbanceObsOff,
                    tmpCholeskyWork);
                if (kPosdef > 0) {
                    transformVariates(stateSpace.stateCovarianceData(), stateSpace.stateCovarianceOffset(t), kPosdef,
                        stateDisturbanceVariates, stateOff,
                        generatedStateDisturbance, disturbanceStateOff,
                        tmpCholeskyWork);
                }

                generateObservation(observationModel, kEndog, kStates, t,
                    generatedStatePath, generatedStateBase + generatedStateOff,
                    generatedMeasurement, disturbanceObsOff,
                        generatedObs, generatedObsBase + measOff);
                generateState(stateSpace, kStates, kPosdef, t,
                    generatedStatePath, generatedStateBase + generatedStateOff,
                    generatedStateDisturbance, disturbanceStateOff,
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
                    BLAS.dcopy(kStates, currentGeneratedState, 0, 1, generatedInitialState, 0, 1);
                }

                for (int t = 0; t < nobs; t++) {
                int measOff = t * kEndog;
                int stateOff = t * kPosdef;
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

            FilterResult generatedFilter = filter(observationModel, init, resolvedSpec,
                workspace.generatedFilterPool, workspace.generatedUnivariateFilterPool, generatedObs, generatedObsBase);
            SmootherResult generatedSmooth = reuseResult
                ? workspace.borrowGeneratedSmoothTarget(result, kEndog, kStates, kPosdef, nobs, storeState, storeDisturbance)
                : new SmootherResult().reuseExternal(kEndog, kStates, kPosdef, nobs,
                    storeState,
                    storeDisturbance,
                    result.simulatedState,
                    result.simulatedStateBase(),
                    result.simulatedMeasurementDisturbance,
                    result.simulatedMeasurementDisturbanceBase(),
                    result.simulatedStateDisturbance,
                    result.simulatedStateDisturbanceBase());
            KalmanSmoother.smoothInto(stateSpace, observationModel, generatedFilter,
                generatedSmooth, workspace.generatedSmootherPool, resolvedSmootherSpec);

            if (storeState) {
                if (storeGeneratedOutputs) {
                    for (int t = 0; t < nobs; t++) {
                        int off = t * kStates;
                        int actualStateOff = actualSmooth.smoothedStateOffset(t);
                        for (int i = 0; i < kStates; i++) {
                            result.simulatedState[simulatedStateBase + off + i] = generatedStatePath[generatedStateBase + off + i]
                                    - result.simulatedState[simulatedStateBase + off + i]
                                    + actualSmooth.smoothedState[actualStateOff + i];
                        }
                    }
                } else {
                    double[] currentGeneratedState = workspace.combineWorkspace.generatedCurrentStateScratch;
                    double[] nextGeneratedState = workspace.combineWorkspace.generatedNextStateScratch;
                    BLAS.dcopy(kStates, generatedInitialState, 0, 1, currentGeneratedState, 0, 1);

                    for (int t = 0; t < nobs; t++) {
                        int off = t * kStates;
                        int stateOff = t * kPosdef;
                        int actualStateOff = actualSmooth.smoothedStateOffset(t);
                        for (int i = 0; i < kStates; i++) {
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
                    int obsOff = t * kEndog;
                    int stateOff = t * kPosdef;
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
                    for (int i = 0; i < kEndog; i++) {
                        result.simulatedMeasurementDisturbance[simulatedMeasurementBase + obsOff + i] = generatedMeasurement[generatedMeasurementOff + i]
                                - result.simulatedMeasurementDisturbance[simulatedMeasurementBase + obsOff + i]
                                + actualSmooth.smoothedObsDisturbance[actualObsOff + i];
                    }
                    int actualStateDistOff = actualSmooth.smoothedStateDisturbanceOffset(t);
                    int generatedStateDisturbanceOff = storeGeneratedOutputs ? generatedStateDisturbanceBase + stateOff : 0;
                    for (int i = 0; i < kPosdef; i++) {
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

    public static SimulationSmootherResult simulate(StateSpaceModel model,
                                                    StateInitialization init,
                                                    Random rng) {
        return simulate(model, init, rng, null, SimulationSmootherSpec.all());
    }

    public static SimulationSmootherResult simulate(StateSpaceModel model,
                                                    StateInitialization init,
                                                    Random rng,
                                                    Pool pool) {
        return simulate(model, init, rng, pool, SimulationSmootherSpec.all());
    }

    public static SimulationSmootherResult simulate(StateSpaceModel model,
                                                    StateInitialization init,
                                                    Random rng,
                                                    SimulationSmootherSpec spec) {
        return simulate(model, init, rng, null, spec);
    }

    public static SimulationSmootherResult simulate(StateSpaceModel model,
                                                    StateInitialization init,
                                                    Random rng,
                                                    Pool pool,
                                                    SimulationSmootherSpec spec) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(rng, "rng");
        requireRealInputs(model);

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
            : new double[nobs * kEndog];
        double[] stateDisturbanceVariates = reuseResult
            ? workspace.stateDisturbanceVariates
            : new double[nobs * kPosdef];
        double[] initialStateVariates = reuseResult
            ? workspace.initialStateVariates
            : new double[kStates];
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

    private static FilterResult filter(StateSpaceModel model,
                                       StateInitialization init,
                                       SimulationSmootherSpec spec,
                                       KalmanFilter.Pool filterPool,
                                       UnivariateFilter.Pool univariateFilterPool,
                                       double[] endogOverride,
                                       int endogBase) {
        FilterSpec filterSpec = spec.requiredFilterSpec();
        if (spec.usesUnivariateFilter()) {
            return endogOverride == null
                ? UnivariateFilter.filter(model, init,
                    univariateFilterPool, filterSpec)
                : UnivariateFilter.filter(model, init, endogOverride, endogBase,
                    univariateFilterPool, filterSpec);
        }
        return endogOverride == null
            ? KalmanFilter.filter(model, init, filterPool, filterSpec)
            : KalmanFilter.filter(model, init, endogOverride, endogBase, filterPool, filterSpec);
    }

    private static boolean canBorrowGeneratedObsFromResult(int kEndog, int kStates,
                                                           boolean storeState, boolean storeDisturbance) {
        return storeDisturbance || (storeState && kStates >= kEndog);
    }

    private static double[] borrowedGeneratedObsResultBank(SimulationSmootherResult result,
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

    private static int borrowedGeneratedObsResultBase(SimulationSmootherResult result,
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

    private static void requireRealInputs(StateSpaceModel model) {
        StateSpaceModelSupport.requireRealStorage(model, "model");
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
        BLAS.daxpy(kStates, 1.0, initialState, initialStateOff, 1, target, targetOff, 1);
    }

    private static void transformVariates(double[] covariance, int covarianceOff, int dim,
                                          double[] variates, int variatesOff,
                                          double[] target, int targetOff,
                                          double[] tmpCholesky) {
        if (dim == 0) {
            return;
        }

        BLAS.dcopy(dim, variates, variatesOff, 1, target, targetOff, 1);
        if (dim == 1) {
            target[targetOff] *= Math.sqrt(covariance[covarianceOff]);
            return;
        }

        BLAS.dcopy(dim * dim, covariance, covarianceOff, 1, tmpCholesky, 0, 1);
        BLAS.dpotrf(BLAS.Uplo.Lower, dim, tmpCholesky, 0, dim);
        BLAS.dtrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                dim, tmpCholesky, 0, dim, target, targetOff, 1);
    }

    private static void generateObservation(StateSpaceModel observationModel,
                                            int kEndog,
                                            int kStates,
                                            int t,
                                            double[] state, int stateOff,
                                            double[] disturbance, int disturbanceOff,
                                            double[] target, int targetOff) {
        BLAS.dcopy(kEndog, observationModel.obsInterceptData(), observationModel.obsInterceptOffset(t), 1, target, targetOff, 1);
        BLAS.daxpy(kEndog, 1.0, disturbance, disturbanceOff, 1, target, targetOff, 1);
        BLAS.dgemv(BLAS.Trans.NoTrans, kEndog, kStates, 1.0,
                observationModel.designData(), observationModel.designOffset(t), observationModel.designLeadingDimension(),
                state, stateOff, 1,
                1.0, target, targetOff, 1);
    }

    private static void generateState(StateSpaceModel stateSpace,
                                      int kStates,
                                      int kPosdef,
                                      int t,
                                      double[] state, int stateOff,
                                      double[] disturbance, int disturbanceOff,
                                      double[] target, int targetOff) {
        BLAS.dcopy(kStates, stateSpace.stateInterceptData(), stateSpace.stateInterceptOffset(t), 1, target, targetOff, 1);
        if (kPosdef > 0) {
            BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kPosdef, 1.0,
                    stateSpace.selectionData(), stateSpace.selectionOffset(t), stateSpace.selectionLeadingDimension(),
                    disturbance, disturbanceOff, 1,
                    1.0, target, targetOff, 1);
        }
        BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                stateSpace.transitionData(), stateSpace.transitionOffset(t), stateSpace.transitionLeadingDimension(),
                state, stateOff, 1,
                1.0, target, targetOff, 1);
    }

}