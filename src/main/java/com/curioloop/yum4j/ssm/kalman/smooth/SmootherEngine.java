package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;

import java.util.Objects;

public final class SmootherEngine {

    private SmootherEngine() {
    }

    public static final class Workspace {
        private final KalmanEngine.Workspace filterWorkspace;
        private final DoubleArena smootherScratchArena = new DoubleArena();
        private final DoubleArena smootherResultArena = new DoubleArena();
        private final KalmanSmoother.ScratchArena realSmootherScratch = new KalmanSmoother.ScratchArena(smootherScratchArena, false);
        private final ZKalmanSmoother.ScratchArena complexSmootherScratch = new ZKalmanSmoother.ScratchArena(smootherScratchArena, false);
        private final KalmanSmoother.Pool smootherPool = new KalmanSmoother.Pool(smootherResultArena, false);
        private final ZKalmanSmoother.Pool complexSmootherPool = new ZKalmanSmoother.Pool(smootherResultArena, false);

        private Workspace() {
            this(KalmanEngine.workspace());
        }

        private Workspace(KalmanEngine.Workspace filterWorkspace) {
            this.filterWorkspace = Objects.requireNonNull(filterWorkspace, "filterWorkspace");
            smootherPool.attachScratch(realSmootherScratch);
            complexSmootherPool.attachScratch(complexSmootherScratch);
        }

        public void reserveSmoother(KalmanSSM model,
                                    SmootherOptions options) {
            smootherPool.reserve(model, options);
            refreshSmootherScratchBackings();
        }

        public long retainedSmootherScratchDoubleCount() {
            return smootherPool.retainedScratchDoubleCount();
        }

        public long retainedSmootherResultDoubleCount() {
            return smootherPool.usesResultArena() ? smootherResultArena.retainedDoubleCount() : 0L;
        }

        public long retainedSmootherTotalByteCount() {
            return smootherPool.retainedTotalByteCount();
        }

        public void releaseSmootherRetainedScratch() {
            smootherPool.releaseRetainedScratch();
            releaseSharedSmootherScratchIfUnused();
        }

        public void releaseSmootherRetainedResults() {
            smootherPool.releaseRetainedResults();
            releaseSharedSmootherResultIfUnused();
        }

        public void reserveComplexSmoother(KalmanSSM model,
                                           SmootherOptions options) {
            complexSmootherPool.reserve(model, options);
            refreshSmootherScratchBackings();
        }

        public long retainedComplexSmootherScratchDoubleCount() {
            return complexSmootherPool.retainedScratchDoubleCount();
        }

        public long retainedComplexSmootherResultDoubleCount() {
            return complexSmootherPool.usesResultArena() ? smootherResultArena.retainedDoubleCount() : 0L;
        }

        public long retainedComplexSmootherTotalByteCount() {
            return complexSmootherPool.retainedTotalByteCount();
        }

        public void releaseComplexSmootherRetainedScratch() {
            complexSmootherPool.releaseRetainedScratch();
            releaseSharedSmootherScratchIfUnused();
        }

        public void releaseComplexSmootherRetainedResults() {
            complexSmootherPool.releaseRetainedResults();
            releaseSharedSmootherResultIfUnused();
        }

        public void releaseRetainedScratch() {
            filterWorkspace.releaseRetainedScratch();
            smootherPool.releaseRetainedScratch();
            complexSmootherPool.releaseRetainedScratch();
            releaseSharedSmootherScratchIfUnused();
        }

        public void releaseRetainedResults() {
            filterWorkspace.releaseRetainedResults();
            smootherPool.releaseRetainedResults();
            complexSmootherPool.releaseRetainedResults();
            releaseSharedSmootherResultIfUnused();
        }

        public long retainedSmootherWorkspaceScratchDoubleCount() {
            return smootherScratchArena.retainedDoubleCount();
        }

        public long retainedSmootherWorkspaceResultDoubleCount() {
            return usesAnySmootherResultArena() ? smootherResultArena.retainedDoubleCount() : 0L;
        }

        public long retainedSmootherWorkspaceTotalByteCount() {
            return (retainedSmootherWorkspaceScratchDoubleCount()
                + retainedSmootherWorkspaceResultDoubleCount()) * Double.BYTES;
        }

        public long retainedNestedFilterWorkspaceScratchDoubleCount() {
            return filterWorkspace.retainedFilterWorkspaceScratchDoubleCount();
        }

        public long retainedNestedFilterWorkspaceResultDoubleCount() {
            return filterWorkspace.retainedFilterWorkspaceResultDoubleCount();
        }

        public long retainedNestedFilterWorkspaceTotalByteCount() {
            return filterWorkspace.retainedFilterWorkspaceTotalByteCount();
        }

        private void refreshSmootherScratchBackings() {
            smootherPool.refreshScratchBacking();
            complexSmootherPool.refreshScratchBacking();
        }

        private void invalidateBorrowedSmootherResults() {
            smootherPool.invalidateBorrowedResult();
            complexSmootherPool.invalidateBorrowedResult();
        }

        private void releaseSharedSmootherScratchIfUnused() {
            if (!smootherPool.usesScratchArena() && !complexSmootherPool.usesScratchArena()) {
                smootherScratchArena.release();
            }
        }

        private boolean usesAnySmootherResultArena() {
            return smootherPool.usesResultArena() || complexSmootherPool.usesResultArena();
        }

        private void releaseSharedSmootherResultIfUnused() {
            if (!usesAnySmootherResultArena()) {
                smootherResultArena.release();
            }
        }
    }

    public static Workspace workspace() {
        return new Workspace();
    }

    public static Workspace workspace(KalmanEngine.Workspace filterWorkspace) {
        return new Workspace(filterWorkspace);
    }

    public static SmootherResult smooth(KalmanSSM model,
                                        InitialState initialState,
                                        SmootherOptions options) {
        return smooth(model, initialState, workspace(), options);
    }

    public static SmootherResult smooth(KalmanSSM model,
                                        InitialState initialState,
                                        Workspace workspace,
                                        SmootherOptions options) {
        SmootherResult result = smoothBorrowedUnsafe(model, initialState, workspace, options).copy();
        workspace.releaseSmootherRetainedResults();
        return result;
    }

    public static SmootherResult smoothBorrowedUnsafe(KalmanSSM model,
                                                      InitialState initialState,
                                                      Workspace workspace,
                                                      SmootherOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        SmootherOptions spec = resolveSmootherOptions(options);
        FilterOptions requiredFilterOptions = spec.requiredFilterOptions();
        if (spec.method() == SmootherMethod.UNIVARIATE) {
            KalmanSSMSupport.requireDiagonalObservationCovariance(model, "UNIVARIATE smoothing");
        }
        workspace.invalidateBorrowedSmootherResults();
        workspace.smootherPool.reserve(model, spec);
        FilterResult filterResult = KalmanEngine.filterBorrowedUnsafe(model,
            initialState,
            workspace.filterWorkspace,
            requiredFilterOptions);
        workspace.filterWorkspace.releaseRetainedScratch();
        try {
            SmootherResult result = KalmanSmoother.smooth(model, filterResult, workspace.smootherPool, spec);
            workspace.refreshSmootherScratchBackings();
            return result;
        } finally {
            workspace.filterWorkspace.releaseRetainedResults();
        }
    }

    public static SmootherResult smooth(KalmanSSM model,
                                        FilterResult filterResult,
                                        SmootherOptions options) {
        return smoothBorrowedUnsafe(model, filterResult, workspace(), options).copy();
    }

    public static SmootherResult smoothBorrowedUnsafe(KalmanSSM model,
                                                      FilterResult filterResult,
                                                      Workspace workspace,
                                                      SmootherOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateBorrowedSmootherResults();
        SmootherResult result = smoothBorrowed(model, filterResult, workspace.smootherPool, options);
        workspace.refreshSmootherScratchBackings();
        return result;
    }

    static SmootherResult smooth(KalmanSSM model,
                                 FilterResult filterResult,
                                 KalmanSmoother.Pool smootherPool,
                                 SmootherOptions options) {
        return smoothBorrowed(model, filterResult, smootherPool, options).copy();
    }

    static SmootherResult smoothBorrowed(KalmanSSM model,
                                         FilterResult filterResult,
                                         KalmanSmoother.Pool smootherPool,
                                         SmootherOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(filterResult, "filterResult");
        SmootherOptions spec = resolveSmootherOptions(options);
        if (spec.method() == SmootherMethod.UNIVARIATE) {
            KalmanSSMSupport.requireDiagonalObservationCovariance(model, "UNIVARIATE smoothing");
        }
        if (smootherPool != null) {
            smootherPool.reserve(model, spec);
        }
        return KalmanSmoother.smooth(model, filterResult, smootherPool, spec);
    }

    public static ZSmootherResult smoothComplex(KalmanSSM model,
                                                InitialState initialState,
                                                SmootherOptions options) {
        return smoothComplex(model, initialState, workspace(), options);
    }

    public static ZSmootherResult smoothComplex(KalmanSSM model,
                                                InitialState initialState,
                                                Workspace workspace,
                                                SmootherOptions options) {
        ZSmootherResult result = smoothComplexBorrowedUnsafe(model, initialState, workspace, options).copy();
        workspace.releaseComplexSmootherRetainedResults();
        return result;
    }

    public static ZSmootherResult smoothComplexBorrowedUnsafe(KalmanSSM model,
                                                              InitialState initialState,
                                                              Workspace workspace,
                                                              SmootherOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        SmootherOptions spec = resolveSmootherOptions(options);
        FilterOptions requiredFilterOptions = spec.requiredFilterOptions();
        if (spec.method() == SmootherMethod.UNIVARIATE) {
            KalmanSSMSupport.requireDiagonalObservationCovariance(model, "UNIVARIATE smoothing");
        }
        workspace.invalidateBorrowedSmootherResults();
        workspace.complexSmootherPool.reserve(model, spec);
        ZFilterResult filterResult = KalmanEngine.filterComplexBorrowedUnsafe(model,
            initialState,
            workspace.filterWorkspace,
            requiredFilterOptions);
        workspace.filterWorkspace.releaseRetainedScratch();
        try {
            ZSmootherResult result = ZKalmanSmoother.smooth(model, filterResult, workspace.complexSmootherPool, spec);
            workspace.refreshSmootherScratchBackings();
            return result;
        } finally {
            workspace.filterWorkspace.releaseRetainedResults();
        }
    }

    public static ZSmootherResult smoothComplex(KalmanSSM model,
                                                ZFilterResult filterResult,
                                                SmootherOptions options) {
        return smoothComplexBorrowedUnsafe(model, filterResult, workspace(), options).copy();
    }

    public static ZSmootherResult smoothComplexBorrowedUnsafe(KalmanSSM model,
                                                              ZFilterResult filterResult,
                                                              Workspace workspace,
                                                              SmootherOptions options) {
        Objects.requireNonNull(workspace, "workspace");
        workspace.invalidateBorrowedSmootherResults();
        ZSmootherResult result = smoothComplexBorrowed(model, filterResult, workspace.complexSmootherPool, options);
        workspace.refreshSmootherScratchBackings();
        return result;
    }

    static ZSmootherResult smoothComplex(KalmanSSM model,
                                         ZFilterResult filterResult,
                                         ZKalmanSmoother.Pool smootherPool,
                                         SmootherOptions options) {
        return smoothComplexBorrowed(model, filterResult, smootherPool, options).copy();
    }

    static ZSmootherResult smoothComplexBorrowed(KalmanSSM model,
                                                 ZFilterResult filterResult,
                                                 ZKalmanSmoother.Pool smootherPool,
                                                 SmootherOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(filterResult, "filterResult");
        SmootherOptions spec = resolveSmootherOptions(options);
        if (spec.method() == SmootherMethod.UNIVARIATE) {
            KalmanSSMSupport.requireDiagonalObservationCovariance(model, "UNIVARIATE smoothing");
        }
        if (smootherPool != null) {
            smootherPool.reserve(model, spec);
        }
        return ZKalmanSmoother.smooth(model, filterResult, smootherPool, spec);
    }

    private static SmootherOptions resolveSmootherOptions(SmootherOptions options) {
        return options == null ? SmootherOptions.defaults() : options;
    }

}