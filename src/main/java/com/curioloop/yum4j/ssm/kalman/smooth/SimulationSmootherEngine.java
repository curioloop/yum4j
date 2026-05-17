package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;

import java.util.Objects;
import java.util.Random;

public final class SimulationSmootherEngine {

    private SimulationSmootherEngine() {
    }

    public static final class Workspace {
        private final DoubleArena resultArena = new DoubleArena();
        private final SimulationSmoother.Pool pool = new SimulationSmoother.Pool(resultArena, false);
        private final ZSimulationSmoother.Pool complexPool = new ZSimulationSmoother.Pool(resultArena, false);

        public void reserve(KalmanSSM model,
                            InitialState initialState,
                            SimulationSmootherOptions options) {
            pool.reserve(model, initialState, options);
        }

        public long retainedWorkspaceDoubleCount() {
            return pool.retainedWorkspaceDoubleCount();
        }

        public long retainedCombineWorkspaceDoubleCount() {
            return pool.retainedCombineWorkspaceDoubleCount();
        }

        public long retainedCfaWorkspaceDoubleCount() {
            return pool.retainedCfaWorkspaceDoubleCount();
        }

        public long retainedVariateDoubleCount() {
            return pool.retainedVariateDoubleCount();
        }

        public long retainedResultDoubleCount() {
            return pool.retainedResultDoubleCount();
        }

        public long retainedNestedScratchDoubleCount() {
            return pool.retainedNestedScratchDoubleCount();
        }

        public long retainedNestedResultDoubleCount() {
            return pool.retainedNestedResultDoubleCount();
        }

        public long retainedTotalByteCount() {
            return pool.retainedTotalByteCount();
        }

        public void releaseRetainedScratch() {
            pool.releaseRetainedScratch();
        }

        public void retainReusableWorkspaces() {
            pool.retainReusableWorkspaces(true);
            complexPool.retainReusableWorkspaces(true);
        }

        public void retainCfaPosteriorWorkspace() {
            retainReusableWorkspaces();
            pool.retainCfaPosteriorWorkspace(true);
            complexPool.retainCfaPosteriorWorkspace(true);
        }

        public void trimReusableWorkspaces() {
            pool.retainReusableWorkspaces(false);
            complexPool.retainReusableWorkspaces(false);
            pool.releaseCombineWorkspace();
            complexPool.releaseCombineWorkspace();
        }

        public void releaseRetainedResults() {
            pool.releaseRetainedResults();
            releaseSharedResultArenaIfUnused();
        }

        public void reserveComplex(KalmanSSM model,
                                   InitialState initialState,
                                   SimulationSmootherOptions options) {
            complexPool.reserve(model, initialState, options);
        }

        public long retainedComplexWorkspaceDoubleCount() {
            return complexPool.retainedWorkspaceDoubleCount();
        }

        public long retainedComplexCombineWorkspaceDoubleCount() {
            return complexPool.retainedCombineWorkspaceDoubleCount();
        }

        public long retainedComplexCfaWorkspaceDoubleCount() {
            return complexPool.retainedCfaWorkspaceDoubleCount();
        }

        public long retainedComplexVariateDoubleCount() {
            return complexPool.retainedVariateDoubleCount();
        }

        public long retainedComplexResultDoubleCount() {
            return complexPool.retainedResultDoubleCount();
        }

        public long retainedComplexNestedScratchDoubleCount() {
            return complexPool.retainedNestedScratchDoubleCount();
        }

        public long retainedComplexNestedResultDoubleCount() {
            return complexPool.retainedNestedResultDoubleCount();
        }

        public long retainedComplexTotalByteCount() {
            return complexPool.retainedTotalByteCount();
        }

        public void releaseComplexRetainedScratch() {
            complexPool.releaseRetainedScratch();
        }

        public void releaseComplexRetainedResults() {
            complexPool.releaseRetainedResults();
            releaseSharedResultArenaIfUnused();
        }

        private void invalidateComplexBorrowedResult() {
            complexPool.invalidateBorrowedResult();
        }

        private void invalidateRealBorrowedResult() {
            pool.invalidateBorrowedResult();
        }

        private void releaseSharedResultArenaIfUnused() {
            if (!pool.usesResultArena() && !complexPool.usesResultArena()) {
                resultArena.release();
            }
        }
    }

    public static Workspace workspace() {
        return new Workspace();
    }

    public static SimulationSmootherResult simulate(KalmanSSM model,
                                                    InitialState initialState,
                                                    Random rng,
                                                    Workspace workspace,
                                                    SimulationSmootherOptions options) {
        SimulationSmootherResult result = simulateBorrowedUnsafe(model, initialState, rng, workspace, options).copy();
        workspace.releaseRetainedResults();
        return result;
    }

    public static SimulationSmootherResult simulateBorrowedUnsafe(KalmanSSM model,
                                                                  InitialState initialState,
                                                                  Random rng,
                                                                  Workspace workspace,
                                                                  SimulationSmootherOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateComplexBorrowedResult();
        return simulateBorrowed(model, initialState, rng, workspace.pool, options);
    }

    public static ZSimulationSmootherResult simulateComplex(KalmanSSM model,
                                                            InitialState initialState,
                                                            Random rng,
                                                            Workspace workspace,
                                                            SimulationSmootherOptions options) {
        ZSimulationSmootherResult result = simulateComplexBorrowedUnsafe(model, initialState, rng, workspace, options).copy();
        workspace.releaseComplexRetainedResults();
        return result;
    }

    public static ZSimulationSmootherResult simulateComplexBorrowedUnsafe(KalmanSSM model,
                                                                          InitialState initialState,
                                                                          Random rng,
                                                                          Workspace workspace,
                                                                          SimulationSmootherOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateRealBorrowedResult();
        return simulateComplexBorrowed(model, initialState, rng, workspace.complexPool, options);
    }

    public static SimulationSmootherResult simulate(KalmanSSM model,
                                                    InitialState initialState,
                                                    Random rng,
                                                    SimulationSmootherOptions options) {
        return simulate(model, initialState, rng, (SimulationSmoother.Pool) null, options);
    }

    static SimulationSmootherResult simulate(KalmanSSM model,
                                             InitialState initialState,
                                             Random rng,
                                             SimulationSmoother.Pool pool,
                                             SimulationSmootherOptions options) {
        return simulateBorrowed(model, initialState, rng, pool, options).copy();
    }

    static SimulationSmootherResult simulateBorrowed(KalmanSSM model,
                                                     InitialState initialState,
                                                     Random rng,
                                                     SimulationSmoother.Pool pool,
                                                     SimulationSmootherOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        Objects.requireNonNull(rng, "rng");
        SimulationSmootherOptions spec = toRealSpec(model, initialState, options);
        if (spec.method() == SimulationSmootherOptions.Method.CFA) {
            return CFASimulationSmoother.simulate(model, initialState, rng, pool, spec);
        }
        return SimulationSmoother.simulate(model, initialState, rng, pool, spec);
    }

    public static ZSimulationSmootherResult simulateComplex(KalmanSSM model,
                                                            InitialState initialState,
                                                            Random rng,
                                                            SimulationSmootherOptions options) {
        return simulateComplex(model, initialState, rng, (ZSimulationSmoother.Pool) null, options);
    }

    static ZSimulationSmootherResult simulateComplex(KalmanSSM model,
                                                     InitialState initialState,
                                                     Random rng,
                                                     ZSimulationSmoother.Pool pool,
                                                     SimulationSmootherOptions options) {
        return simulateComplexBorrowed(model, initialState, rng, pool, options).copy();
    }

    static ZSimulationSmootherResult simulateComplexBorrowed(KalmanSSM model,
                                                             InitialState initialState,
                                                             Random rng,
                                                             ZSimulationSmoother.Pool pool,
                                                             SimulationSmootherOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        Objects.requireNonNull(rng, "rng");
        SimulationSmootherOptions spec = toComplexSpec(model, initialState, options);
        if (spec.method() == SimulationSmootherOptions.Method.CFA) {
            return ZCFASimulationSmoother.simulate(model, initialState, rng, pool, spec);
        }
        return ZSimulationSmoother.simulate(model, initialState, rng, pool, spec);
    }

    public static SimulationSmootherResult simulate(KalmanSSM model,
                                                    InitialState initialState,
                                                    double[] measurementDisturbanceVariates,
                                                    double[] stateDisturbanceVariates,
                                                    double[] initialStateVariates,
                                                    SimulationSmootherOptions options) {
        return simulate(model, initialState,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            (SimulationSmoother.Pool) null,
            options);
    }

    static SimulationSmootherResult simulate(KalmanSSM model,
                                             InitialState initialState,
                                             double[] measurementDisturbanceVariates,
                                             double[] stateDisturbanceVariates,
                                             double[] initialStateVariates,
                                             SimulationSmoother.Pool pool,
                                             SimulationSmootherOptions options) {
        return simulateBorrowed(model, initialState,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            pool,
            options).copy();
    }

    public static SimulationSmootherResult simulate(KalmanSSM model,
                                                    InitialState initialState,
                                                    double[] measurementDisturbanceVariates,
                                                    double[] stateDisturbanceVariates,
                                                    double[] initialStateVariates,
                                                    Workspace workspace,
                                                    SimulationSmootherOptions options) {
        SimulationSmootherResult result = simulateBorrowedUnsafe(model,
            initialState,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            workspace,
            options).copy();
        workspace.releaseRetainedResults();
        return result;
    }

    public static SimulationSmootherResult simulateBorrowedUnsafe(KalmanSSM model,
                                                                  InitialState initialState,
                                                                  double[] measurementDisturbanceVariates,
                                                                  double[] stateDisturbanceVariates,
                                                                  double[] initialStateVariates,
                                                                  Workspace workspace,
                                                                  SimulationSmootherOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateComplexBorrowedResult();
        return simulateBorrowed(model,
            initialState,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            workspace.pool,
            options);
    }

    static SimulationSmootherResult simulateBorrowed(KalmanSSM model,
                                                     InitialState initialState,
                                                     double[] measurementDisturbanceVariates,
                                                     double[] stateDisturbanceVariates,
                                                     double[] initialStateVariates,
                                                     SimulationSmoother.Pool pool,
                                                     SimulationSmootherOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        SimulationSmootherOptions spec = toRealSpec(model, initialState, options);
        if (spec.method() == SimulationSmootherOptions.Method.CFA) {
            return CFASimulationSmoother.simulate(model,
                initialState,
                stateDisturbanceVariates,
                pool,
                spec);
        }
        return SimulationSmoother.simulate(model,
            initialState,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            pool,
            spec);
    }

    public static ZSimulationSmootherResult simulateComplex(KalmanSSM model,
                                                            InitialState initialState,
                                                            double[] measurementDisturbanceVariates,
                                                            double[] stateDisturbanceVariates,
                                                            double[] initialStateVariates,
                                                            SimulationSmootherOptions options) {
        return simulateComplex(model, initialState,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            (ZSimulationSmoother.Pool) null,
            options);
    }

    static ZSimulationSmootherResult simulateComplex(KalmanSSM model,
                                                     InitialState initialState,
                                                     double[] measurementDisturbanceVariates,
                                                     double[] stateDisturbanceVariates,
                                                     double[] initialStateVariates,
                                                     ZSimulationSmoother.Pool pool,
                                                     SimulationSmootherOptions options) {
        return simulateComplexBorrowed(model, initialState,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            pool,
            options).copy();
    }

    public static ZSimulationSmootherResult simulateComplex(KalmanSSM model,
                                                            InitialState initialState,
                                                            double[] measurementDisturbanceVariates,
                                                            double[] stateDisturbanceVariates,
                                                            double[] initialStateVariates,
                                                            Workspace workspace,
                                                            SimulationSmootherOptions options) {
        ZSimulationSmootherResult result = simulateComplexBorrowedUnsafe(model,
            initialState,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            workspace,
            options).copy();
        workspace.releaseComplexRetainedResults();
        return result;
    }

    public static ZSimulationSmootherResult simulateComplexBorrowedUnsafe(KalmanSSM model,
                                                                          InitialState initialState,
                                                                          double[] measurementDisturbanceVariates,
                                                                          double[] stateDisturbanceVariates,
                                                                          double[] initialStateVariates,
                                                                          Workspace workspace,
                                                                          SimulationSmootherOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateRealBorrowedResult();
        return simulateComplexBorrowed(model,
            initialState,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            workspace.complexPool,
            options);
    }

    static ZSimulationSmootherResult simulateComplexBorrowed(KalmanSSM model,
                                                             InitialState initialState,
                                                             double[] measurementDisturbanceVariates,
                                                             double[] stateDisturbanceVariates,
                                                             double[] initialStateVariates,
                                                             ZSimulationSmoother.Pool pool,
                                                             SimulationSmootherOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        SimulationSmootherOptions spec = toComplexSpec(model, initialState, options);
        if (spec.method() == SimulationSmootherOptions.Method.CFA) {
            return ZCFASimulationSmoother.simulate(model,
                initialState,
                stateDisturbanceVariates,
                pool,
                spec);
        }
        return ZSimulationSmoother.simulate(model,
            initialState,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            pool,
            spec);
    }

    private static SimulationSmootherOptions toRealSpec(KalmanSSM model,
                                                        InitialState initialState,
                                                        SimulationSmootherOptions options) {
        SimulationSmootherOptions resolvedOptions = options == null ? SimulationSmootherOptions.defaults() : options;
        if (resolvedOptions.method() == SimulationSmootherOptions.Method.CFA) {
            CFASimulationSmoother.validateContract(model, initialState, resolvedOptions);
        }
        return resolvedOptions;
    }

    private static SimulationSmootherOptions toComplexSpec(KalmanSSM model,
                                                          InitialState initialState,
                                                          SimulationSmootherOptions options) {
        SimulationSmootherOptions resolvedOptions = options == null ? SimulationSmootherOptions.defaults() : options;
        if (resolvedOptions.method() == SimulationSmootherOptions.Method.CFA) {
            ZCFASimulationSmoother.validateContract(model, initialState, resolvedOptions);
        }
        return resolvedOptions;
    }
}