package com.curioloop.yum4j.kalman.smooth;

import com.curioloop.yum4j.kalman.arena.SmootherDiffuseLayout;
import com.curioloop.yum4j.kalman.arena.SmootherResultLayout;
import com.curioloop.yum4j.kalman.arena.ZSmootherScratchLayout;
import com.curioloop.yum4j.kalman.filter.FilterResultShape;
import com.curioloop.yum4j.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.kalman.internal.StateSpaceModelSupport;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;

import java.util.Objects;

public final class ZKalmanSmoother {

    private static final int SMOOTHER_STATE = 0x01;
    private static final int SMOOTHER_STATE_COV = 0x02;
    private static final int SMOOTHER_DISTURBANCE = 0x04;
    private static final int SMOOTHER_DISTURBANCE_COV = 0x08;
    private static final int SMOOTHER_ALL = 0x0F;

    private static long doubleCount(double[] values) {
        return values == null ? 0L : values.length;
    }

    static final class ScratchArena {
        double[] scratchBacking;
        ZSmootherScratchLayout scratchLayout;
        int scratchKStates;
        int scratchKEndog;
        int scratchKPosdef;
        boolean scratchObservationFactor;
        boolean scratchForecastFactor;
        boolean scratchTransitionGain;
        boolean scratchDisturbanceCovariance;
        boolean scratchFactor;
        boolean scratchDiffuse;
        boolean scratchBorrowedSmoothingError;

        long retainedDoubleCount() {
            return doubleCount(scratchBacking);
        }

        void releaseBackings() {
            scratchBacking = null;
            scratchLayout = null;
            scratchKStates = 0;
            scratchKEndog = 0;
            scratchKPosdef = 0;
            scratchObservationFactor = false;
            scratchForecastFactor = false;
            scratchTransitionGain = false;
            scratchDisturbanceCovariance = false;
            scratchFactor = false;
            scratchDiffuse = false;
            scratchBorrowedSmoothingError = false;
        }

        void ensureMainScratch(int kStates,
                               int kEndog,
                               int kPosdef,
                               boolean needObservationFactor,
                               boolean needForecastFactor,
                       boolean needTransitionGain,
                               boolean needDisturbanceCovariance,
                       boolean needFactorScratch,
                               boolean needDiffuseScratch,
                               boolean needBorrowedSmoothingError) {
            int nextKStates = Math.max(scratchKStates, kStates);
            int nextKEndog = Math.max(scratchKEndog, kEndog);
            int nextKPosdef = Math.max(scratchKPosdef, kPosdef);
            boolean nextObservationFactor = scratchObservationFactor || needObservationFactor;
            boolean nextForecastFactor = scratchForecastFactor || needForecastFactor;
            boolean nextTransitionGain = scratchTransitionGain || needTransitionGain;
            boolean nextDisturbanceCovariance = scratchDisturbanceCovariance || needDisturbanceCovariance;
            boolean nextFactorScratch = scratchFactor || needFactorScratch;
            boolean nextDiffuse = scratchDiffuse || needDiffuseScratch;
            boolean nextBorrowedSmoothingError = scratchBorrowedSmoothingError || needBorrowedSmoothingError;
            if (scratchLayout == null
                    || nextKStates != scratchKStates
                    || nextKEndog != scratchKEndog
                    || nextKPosdef != scratchKPosdef
                    || nextObservationFactor != scratchObservationFactor
                    || nextForecastFactor != scratchForecastFactor
                || nextTransitionGain != scratchTransitionGain
                    || nextDisturbanceCovariance != scratchDisturbanceCovariance
                || nextFactorScratch != scratchFactor
                    || nextDiffuse != scratchDiffuse
                    || nextBorrowedSmoothingError != scratchBorrowedSmoothingError) {
                scratchLayout = ZSmootherScratchLayout.create(
                        nextKEndog,
                        nextKStates,
                        nextKPosdef,
                        nextObservationFactor,
                        nextForecastFactor,
                nextTransitionGain,
                        nextDisturbanceCovariance,
                nextFactorScratch,
                        nextDiffuse,
                        nextBorrowedSmoothingError);
                scratchKStates = nextKStates;
                scratchKEndog = nextKEndog;
                scratchKPosdef = nextKPosdef;
                scratchObservationFactor = nextObservationFactor;
                scratchForecastFactor = nextForecastFactor;
            scratchTransitionGain = nextTransitionGain;
                scratchDisturbanceCovariance = nextDisturbanceCovariance;
            scratchFactor = nextFactorScratch;
                scratchDiffuse = nextDiffuse;
                scratchBorrowedSmoothingError = nextBorrowedSmoothingError;
            }
            int totalLength = scratchLayout.totalLength();
            if (scratchBacking == null || scratchBacking.length < totalLength) {
                scratchBacking = new double[totalLength];
            }
        }

        void ensureDiffuse(int kStates) {
            ensureMainScratch(kStates, 0, 0,
                    false, false, false, false, false, true, false);
        }
    }

    public static final class Pool {
        private ZSmootherResult pooledResult;
        private final ZSmootherResult retainedResultView = new ZSmootherResult();
        private ScratchArena scratch;
        private double[] resultBacking;
        private double[] diffuseResultBacking;
        private SmootherResultLayout resultLayout;
        private SmootherDiffuseLayout diffuseResultLayout;
        private int resultKEndog = -1;
        private int resultKStates = -1;
        private int resultKPosdef = -1;
        private int resultNobs = -1;

        private ScratchArena scratchArena() {
            if (scratch == null) {
                scratch = new ScratchArena();
            }
            return scratch;
        }

        void attachScratch(ScratchArena scratch) {
            ScratchArena nextScratch = Objects.requireNonNull(scratch, "scratch");
            if (this.scratch == nextScratch) {
                return;
            }
            this.scratch = nextScratch;
        }

        public void shareScratchFrom(Pool source) {
            Objects.requireNonNull(source, "source");
            attachScratch(source.scratchArena());
        }

        public boolean sharesScratchWith(Pool other) {
            return other != null && scratch != null && scratch == other.scratch;
        }

        public ZSmootherResult retainedResult() {
            if (resultBacking == null || resultLayout == null) {
                return null;
            }
            retainedResultView.reuse(resultKEndog, resultKStates, resultKPosdef, resultNobs, resultBacking, resultLayout);
            bindRetainedDiffuseAuxiliary(retainedResultView);
            return retainedResultView;
        }

        void bindRetainedDiffuseAuxiliary(ZSmootherResult target) {
            Objects.requireNonNull(target, "target");
            if (diffuseResultBacking == null || diffuseResultLayout == null) {
                return;
            }
            target.reuseDiffuseAuxiliary(diffuseResultBacking, diffuseResultLayout);
        }

        void ensure(int kStates, int kEndog, int kPosdef) {
            scratchArena().ensureMainScratch(kStates, kEndog, kPosdef,
                    false, false, false, false, false, false, false);
        }

        void ensureObservationFactorScratch(int kStates, int kEndog) {
            scratchArena().ensureMainScratch(kStates, kEndog, 0,
                    true, false, false, false, false, false, false);
        }

        void ensureForecastFactorScratch(int kEndog) {
            scratchArena().ensureMainScratch(0, kEndog, 0,
                    false, true, false, false, false, false, false);
        }

        void ensureTransitionGainScratch(int kStates, int kEndog) {
            scratchArena().ensureMainScratch(kStates, kEndog, 0,
                    false, false, true, false, false, false, false);
        }

        void ensureDiffuse(int kStates) {
            scratchArena().ensureDiffuse(kStates);
        }

        void ensureDisturbanceCovarianceScratch(int kStates, int kEndog) {
            scratchArena().ensureMainScratch(kStates, kEndog, 0,
                    false, false, false, true, false, false, false);
        }

        void ensureFactorScratch(int kStates) {
            scratchArena().ensureMainScratch(kStates, 0, 0,
                    false, false, false, false, true, false, false);
        }

        void ensurePooledScratch(int kEndog) {
            scratchArena().ensureMainScratch(0, kEndog, 0,
                    false, false, false, false, false, false, true);
        }

        void ensureResult(int kEndog, int kStates, int kPosdef, int nobs,
                  SmootherResultLayout layout) {
            if (resultLayout == null
                    || resultKEndog != kEndog
                    || resultKStates != kStates
                    || resultKPosdef != kPosdef
                    || resultNobs != nobs
                    || !layout.equals(resultLayout)) {
                resultLayout = layout;
                resultKEndog = kEndog;
                resultKStates = kStates;
                resultKPosdef = kPosdef;
                resultNobs = nobs;
            }
            int totalLength = resultLayout.totalLength();
            if (resultBacking == null || resultBacking.length != totalLength) {
                resultBacking = new double[totalLength];
            }
        }

        void ensureDiffuseResult(int kStates, int nobs) {
            SmootherDiffuseLayout nextLayout = SmootherDiffuseLayout.create(2, kStates, nobs);
            if (!nextLayout.equals(diffuseResultLayout)) {
                diffuseResultLayout = nextLayout;
            }
            int totalLength = diffuseResultLayout.totalLength();
            if (diffuseResultBacking == null || diffuseResultBacking.length != totalLength) {
                diffuseResultBacking = new double[totalLength];
            }
        }

        ZSmootherResult borrowResult(int kEndog, int kStates, int kPosdef, int nobs,
                                     SmootherResultLayout layout) {
            ensureResult(kEndog, kStates, kPosdef, nobs, layout);
            if (pooledResult == null) {
                pooledResult = new ZSmootherResult();
            }
            pooledResult.reuse(kEndog, kStates, kPosdef, nobs, resultBacking, resultLayout);
            bindRetainedDiffuseAuxiliary(pooledResult);
            return pooledResult;
        }

        ZSmootherResult borrowDiffuseResult(int kEndog, int kStates, int kPosdef, int nobs,
                                            SmootherResultLayout layout) {
            ZSmootherResult result = borrowResult(kEndog, kStates, kPosdef, nobs, layout);
            ensureDiffuseResult(kStates, nobs);
            bindRetainedDiffuseAuxiliary(result);
            return result;
        }

        public void reserve(StateSpaceModel model,
                            SmootherSpec spec) {
            requireComplexInputs(model, model);
            SmootherSpec resolvedSpec = spec == null ? SmootherSpec.conventional() : spec;
            SmootherResultLayout retainedLayout = resolvedSpec.createResultLayout(
                2,
                model.observationDimension(),
                model.stateCount(),
                model.stateDisturbanceCount(),
                model.observationCount());
            if (scratch != null) {
                scratch.releaseBackings();
            }
            ensureResult(model.observationDimension(), model.stateCount(),
                model.stateDisturbanceCount(), model.observationCount(), retainedLayout);
        }

        public long retainedScratchDoubleCount() {
            return scratch == null ? 0L : scratch.retainedDoubleCount();
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
            resultLayout = null;
            diffuseResultLayout = null;
            resultKEndog = -1;
            resultKStates = -1;
            resultKPosdef = -1;
            resultNobs = -1;
            pooledResult = null;
        }

        public void releaseRetainedScratch() {
            if (scratch != null) {
                scratch.releaseBackings();
                scratch = null;
            }
        }

    }

    private static void requireComplexInputs(StateSpaceModel stateSpace,
                                             StateSpaceModel observationModel) {
        StateSpaceModelSupport.requireComplexStorage(stateSpace, "stateSpace");
        StateSpaceModelSupport.requireComplexStorage(observationModel, "observationModel");
    }

    public static ZSmootherResult smooth(StateSpaceModel model,
                                         ZFilterResult fr) {
        return smooth(model, fr, null, SmootherSpec.conventional());
    }

    public static ZSmootherResult smooth(StateSpaceModel model,
                                         ZFilterResult fr,
                                         Pool pool) {
        return smooth(model, fr, pool, SmootherSpec.conventional());
    }

    /**
     * When a non-null pool is supplied, the returned result borrows pool-owned storage
     * and is invalidated by the next call that reuses the same pool. Call clone() or copy()
     * on the result when an independent snapshot is required.
     */
    public static ZSmootherResult smooth(StateSpaceModel model,
                                         ZFilterResult fr,
                                         Pool pool,
                                         SmootherSpec spec) {
        requireComplexInputs(model, model);
        Objects.requireNonNull(fr, "fr");
        SmootherSpec resolvedSpec = spec == null ? SmootherSpec.conventional() : spec;
        return smoothInternal(model, model, fr, pool,
            resolvedSpec.smoothMethodMask(), resolvedSpec.outputMask(), pool != null, null);
    }

    static ZSmootherResult smoothInto(StateSpaceModel stateSpace,
                                      StateSpaceModel observationModel,
                                      ZFilterResult fr,
                                      ZSmootherResult target,
                                      Pool pool,
                                      SmootherSpec spec) {
        requireComplexInputs(stateSpace, observationModel);
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(fr, "fr");
        SmootherSpec resolvedSpec = spec == null ? SmootherSpec.conventional() : spec;
        return smoothInternal(stateSpace, observationModel, fr, pool,
            resolvedSpec.smoothMethodMask(), resolvedSpec.outputMask(), pool != null, target);
    }

    private static ZSmootherResult prepareResult(Pool pool,
                                                 ZSmootherResult target,
                                                 boolean reuseResult,
                                                 int kEndog,
                                                 int kStates,
                                                 int kPosdef,
                                                 int nobs,
                                                 boolean doState,
                                                 boolean doStateCov,
                                                 boolean doDisturbance,
                                                 boolean doDisturbanceCov,
                                                 boolean storeAuxiliary) {
        SmootherResultLayout retainedLayout = SmootherResultLayout.create(
            2,
            kEndog,
            kStates,
            kPosdef,
            nobs,
            doState,
            doStateCov,
            doDisturbance,
            doDisturbanceCov,
            storeAuxiliary);
        if (target != null) {
            return target;
        }
        return reuseResult
            ? pool.borrowResult(kEndog, kStates, kPosdef, nobs, retainedLayout)
            : new ZSmootherResult(kEndog, kStates, kPosdef, nobs, retainedLayout);
    }

    private static ZSmootherResult smoothInternal(StateSpaceModel stateSpace,
                                                  StateSpaceModel observationModel,
                                                  ZFilterResult fr,
                                                  Pool pool,
                                                  int smoothMethod, int smootherOutput,
                                                  boolean reuseResult,
                                                  ZSmootherResult target) {
        if (pool == null) {
            pool = new Pool();
        }
        int kStates = stateSpace.stateCount();
        int kEndog = observationModel.observationDimension();
        int kPosdef = stateSpace.stateDisturbanceCount();
        pool.ensure(kStates, kEndog, kPosdef);
        requireFilterStorage(fr, smoothMethod);

        if (fr.nobsDiffuse > 0) {
            pool.ensureDiffuse(kStates);
            return smoothDiffuse(stateSpace, observationModel, fr, pool, smootherOutput, reuseResult, target);
        }

        if ((smoothMethod & KalmanSmoother.SMOOTH_CLASSICAL) != 0) {
            return smoothClassical(stateSpace, observationModel, fr, pool,
                smootherOutput, reuseResult, target);
        }
        if ((smoothMethod & KalmanSmoother.SMOOTH_ALTERNATIVE) != 0) {
            return smoothAlternative(stateSpace, observationModel, fr, pool,
                smootherOutput, reuseResult, target);
        }
        return smoothConventionalCore(stateSpace, observationModel, fr, pool,
            smootherOutput, reuseResult, target);
    }

    private static ZSmootherResult smoothClassical(StateSpaceModel stateSpace,
                                                   StateSpaceModel observationModel,
                                                   ZFilterResult fr,
                                                   Pool pool,
                                                   int smootherOutput,
                                                   boolean reuseResult,
                                                   ZSmootherResult target) {
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int ksKs = 2 * kStates * kStates;
        int ksKsC = kStates * kStates;
        double[] transition = stateSpace.transitionData();
        int transitionLd = stateSpace.transitionLeadingDimension();
        double[] selection = stateSpace.selectionData();
        int selectionLd = stateSpace.selectionLeadingDimension();
        double[] stateCovariance = stateSpace.stateCovarianceData();
        int stateCovarianceLd = stateSpace.stateCovarianceLeadingDimension();
        double[] design = observationModel.designData();
        int designLd = observationModel.designLeadingDimension();
        double[] obsCov = observationModel.obsCovData();
        int obsCovLd = observationModel.obsCovLeadingDimension();

        boolean doState = (smootherOutput & SMOOTHER_STATE) != 0;
        boolean doStateCov = (smootherOutput & SMOOTHER_STATE_COV) != 0;
        boolean doDisturbance = (smootherOutput & SMOOTHER_DISTURBANCE) != 0;
        boolean doDisturbanceCov = (smootherOutput & SMOOTHER_DISTURBANCE_COV) != 0;
        boolean storeAuxiliary = smootherOutput == SMOOTHER_ALL;
        boolean needSmoothedState = doState || doDisturbance;
        boolean needSmoothedStateCov = doStateCov || doDisturbanceCov;

        pool.ensureObservationFactorScratch(kStates, kEndog);
        pool.ensureForecastFactorScratch(kEndog);
        pool.ensureFactorScratch(kStates);
        if (fr.perObsKalmanGain) {
            pool.ensureTransitionGainScratch(kStates, kEndog);
        }
        if (doDisturbanceCov) {
            pool.ensureDisturbanceCovarianceScratch(kStates, kEndog);
        }
        if ((reuseResult || target != null) && !storeAuxiliary && doDisturbance) {
            pool.ensurePooledScratch(kEndog);
        }

        ZSmootherResult result = prepareResult(pool, target, reuseResult,
            kEndog, kStates, kPosdef, nobs,
            doState, doStateCov, doDisturbance, doDisturbanceCov, storeAuxiliary);
        ScratchArena scratchArena = pool.scratch;
        ZSmootherScratchLayout layout = scratchArena.scratchLayout;
        double[] scratch = scratchArena.scratchBacking;
        int rBase = layout.rBase();
        int nBase = layout.nBase();
        int lBase = layout.lBase();
        int tmpNpBase = layout.tmpNpBase();
        int tmpRqBase = layout.tmpRqBase();
        int tmpFivBase = layout.tmpFivBase();
        int tmpFiZBase = layout.tmpFiZBase();
        int tmpFcopyBase = layout.tmpFcopyBase();
        int tmpTkBase = fr.perObsKalmanGain ? layout.tmpTkBase() : -1;
        int tmpKhBase = doDisturbanceCov ? layout.tmpKhBase() : -1;
        int tmpFiHBase = doDisturbanceCov ? layout.tmpFiHBase() : -1;
        int tmpJBase = layout.tmpJBase();
        int selectedDesignBase = layout.selectedDesignBase();
        int selectedObsCovBase = layout.selectedObsCovBase();
        int tmpKgBase = layout.tmpKgBase();
        int borrowedSmoothingErrorBase = (!storeAuxiliary && reuseResult && doDisturbance)
            ? layout.scratchSmoothingErrorBase()
            : 0;
        double[] smoothingError = (storeAuxiliary || doDisturbance)
            ? (storeAuxiliary ? result.smoothingError : (reuseResult ? scratch : new double[2 * kEndog]))
            : null;
        double[] nextSmoothedState = (!doState && needSmoothedState) ? new double[2 * kStates] : null;
        double[] nextSmoothedStateCov = (!doStateCov && needSmoothedStateCov) ? new double[ksKs] : null;

        for (int t = nobs - 1; t >= 0; t--) {
            int filtOff = fr.filteredStateOffset(t);
            int filtCovOff = fr.filteredStateCovOffset(t);
            int foreErrOff = fr.forecastErrorOffset(t);
            int foreErrCovOff = fr.forecastErrorCovOffset(t);
            int kgOff = fr.kalmanGainOffset(t);

            int ke = selectObservations(observationModel, scratchArena, layout, t);
            int observedBase = ke > 0 && ke < kEndog
                ? observedIndexBase(selectedObsCovBase, ke)
                : -1;

            if (t < nobs - 1) {
                int nextPredOff = fr.predictedStateOffset(t + 1);
                int nextPredCovOff = fr.predictedStateCovOffset(t + 1);
                double[] smoothedState = doState ? result.smoothedState : nextSmoothedState;
                int smoothedStateOff = doState ? result.smoothedStateOffset(t + 1) : 0;
                double[] smoothedStateCov = doStateCov ? result.smoothedStateCov : nextSmoothedStateCov;
                int smoothedStateCovOff = doStateCov ? result.smoothedStateCovOffset(t + 1) : 0;

                if (needSmoothedState) {
                    for (int i = 0; i < 2 * kStates; i++) {
                        scratch[rBase + i] = smoothedState[smoothedStateOff + i] - fr.predictedState[nextPredOff + i];
                    }
                }

                if (needSmoothedState || needSmoothedStateCov) {
                    ZLAS.zcopy(ksKsC, fr.predictedStateCov, nextPredCovOff, 1, scratch, tmpJBase, 1);
                    ZLAS.zpotrf(BLAS.Uplo.Lower, kStates, scratch, tmpJBase >> 1, kStates);
                }
                if (needSmoothedState) {
                    ZLAS.zpotrs(BLAS.Uplo.Lower, kStates, 1,
                        scratch, tmpJBase >> 1, kStates, scratch, rBase >> 1, 1);
                }

                if (needSmoothedStateCov) {
                    ZLAS.zcopy(ksKsC, fr.predictedStateCov, nextPredCovOff, 1, scratch, nBase, 1);
                    ZLAS.zaxpy(ksKsC, -1.0, 0.0, smoothedStateCov, smoothedStateCovOff >> 1, 1,
                        scratch, nBase >> 1, 1);
                    ZLAS.zpotrs(BLAS.Uplo.Lower, kStates, kStates,
                        scratch, tmpJBase >> 1, kStates, scratch, nBase >> 1, kStates);
                    conjugateTransposeInPlace(scratch, nBase, kStates);
                    ZLAS.zpotrs(BLAS.Uplo.Lower, kStates, kStates,
                        scratch, tmpJBase >> 1, kStates, scratch, nBase >> 1, kStates);
                    symmetrizeHermitian(scratch, nBase, kStates);
                }
            } else {
                zero(scratch, rBase, 2 * kStates);
                zero(scratch, nBase, ksKs);
            }

            if (ke == 0) {
                if (needSmoothedState) {
                    double[] smoothedState = doState ? result.smoothedState : nextSmoothedState;
                    int smoothedStateOff = doState ? result.smoothedStateOffset(t) : 0;
                    ZLAS.zcopy(kStates, fr.filteredState, filtOff, 1,
                        smoothedState, smoothedStateOff, 1);
                    if (t < nobs - 1) {
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
                            1.0, 0.0, fr.filteredStateCov, filtCovOff >> 1, kStates,
                            transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                            0.0, 0.0, scratch, tmpRqBase >> 1, kStates);
                        ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                            scratch, tmpRqBase, kStates,
                            scratch, rBase, 1,
                            1.0, 0.0, smoothedState, smoothedStateOff, 1);
                    }
                }
                if (needSmoothedStateCov) {
                    double[] smoothedStateCov = doStateCov ? result.smoothedStateCov : nextSmoothedStateCov;
                    int smoothedStateCovOff = doStateCov ? result.smoothedStateCovOffset(t) : 0;
                    if (t < nobs - 1) {
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
                            1.0, 0.0, fr.filteredStateCov, filtCovOff >> 1, kStates,
                            transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                            0.0, 0.0, scratch, tmpRqBase >> 1, kStates);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, nBase >> 1, kStates,
                            scratch, tmpRqBase >> 1, kStates,
                            0.0, 0.0, scratch, tmpJBase >> 1, kStates);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            -1.0, 0.0, scratch, tmpRqBase >> 1, kStates,
                            scratch, tmpJBase >> 1, kStates,
                            0.0, 0.0, smoothedStateCov, smoothedStateCovOff >> 1, kStates);
                        ZLAS.zaxpy(ksKsC, 1.0, 0.0, fr.filteredStateCov, filtCovOff >> 1, 1,
                            smoothedStateCov, smoothedStateCovOff >> 1, 1);
                    } else {
                        ZLAS.zcopy(ksKsC, fr.filteredStateCov, filtCovOff, 1,
                            smoothedStateCov, smoothedStateCovOff, 1);
                    }
                    symmetrizeHermitian(smoothedStateCov, smoothedStateCovOff, kStates);
                }

                if (storeAuxiliary) {
                    ZLAS.zcopy(ksKsC, transition, stateSpace.transitionOffset(t), 1,
                        result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                    ZLAS.zcopy(kStates, scratch, rBase, 1,
                        result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                    ZLAS.zcopy(ksKsC, scratch, nBase, 1,
                        result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
                }

                if (doDisturbance) {
                    zero(result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t), 2 * kEndog);
                }

                if (doDisturbance && kPosdef > 0) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                        0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                    ZLAS.zgemv(BLAS.Trans.Conj, kStates, kPosdef, 1.0, 0.0,
                        scratch, tmpRqBase, kPosdef,
                        scratch, rBase, 1,
                        0.0, 0.0, result.smoothedStateDisturbance, result.smoothedStateDisturbanceOffset(t), 1);
                }

                if (doDisturbanceCov) {
                    ZLAS.zcopy(kEndog * kEndog, obsCov, observationModel.obsCovOffset(t), 1,
                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                    if (kPosdef > 0) {
                        if (!doDisturbance) {
                            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                                stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                                0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                        }
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                            1.0, 0.0, scratch, nBase >> 1, kStates,
                            scratch, tmpRqBase >> 1, kPosdef,
                            0.0, 0.0, scratch, tmpNpBase >> 1, kPosdef);
                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                            -1.0, 0.0, scratch, tmpRqBase >> 1, kPosdef,
                            scratch, tmpNpBase >> 1, kPosdef,
                            0.0, 0.0, result.smoothedStateDisturbanceCov,
                            result.smoothedStateDisturbanceCovOffset(t) >> 1, kPosdef);
                        ZLAS.zaxpy(kPosdef * kPosdef, 1.0, 0.0,
                            stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, 1,
                            result.smoothedStateDisturbanceCov,
                            result.smoothedStateDisturbanceCovOffset(t) >> 1, 1);
                    }
                }
                continue;
            }

            double[] Z;
            int zOff;
            int zLd;
            double[] H;
            int hOff;
            int hLd;

            if (ke < kEndog) {
                Z = scratch;
                zOff = selectedDesignBase;
                zLd = kStates;
                H = scratch;
                hOff = selectedObsCovBase;
                hLd = ke;
            } else {
                Z = design;
                zOff = observationModel.designOffset(t);
                zLd = designLd;
                H = obsCov;
                hOff = observationModel.obsCovOffset(t);
                hLd = obsCovLd;
            }

            if (ke < kEndog) {
                copyObservedSquareMatrix(
                    fr.forecastErrorCov, foreErrCovOff, kEndog,
                    scratch, observedBase, ke,
                    scratch, tmpFcopyBase, ke);
            } else {
                ZLAS.zcopy(ke * ke, fr.forecastErrorCov, foreErrCovOff, 1, scratch, tmpFcopyBase, 1);
            }

            if (ke == 1) {
                double FtRe = scratch[tmpFcopyBase];
                double FtIm = scratch[tmpFcopyBase + 1];
                double FtMag2 = FtRe * FtRe + FtIm * FtIm;
                double FtInvRe = FtRe / FtMag2;
                double FtInvIm = -FtIm / FtMag2;
                if (ke < kEndog) {
                    copyObservedVector(
                        fr.forecastError, foreErrOff,
                        scratch, observedBase, ke,
                        scratch, tmpFivBase);
                    double vRe = scratch[tmpFivBase];
                    double vIm = scratch[tmpFivBase + 1];
                    scratch[tmpFivBase] = vRe * FtInvRe - vIm * FtInvIm;
                    scratch[tmpFivBase + 1] = vRe * FtInvIm + vIm * FtInvRe;
                } else {
                    double vRe = fr.forecastError[foreErrOff];
                    double vIm = fr.forecastError[foreErrOff + 1];
                    scratch[tmpFivBase] = vRe * FtInvRe - vIm * FtInvIm;
                    scratch[tmpFivBase + 1] = vRe * FtInvIm + vIm * FtInvRe;
                }
                for (int j = 0; j < kStates; j++) {
                    double zr = Z[zOff + j * 2];
                    double zi = Z[zOff + j * 2 + 1];
                    scratch[tmpFiZBase + j * 2] = zr * FtInvRe - zi * FtInvIm;
                    scratch[tmpFiZBase + j * 2 + 1] = zr * FtInvIm + zi * FtInvRe;
                }
            } else {
                ZLAS.zpotrf(BLAS.Uplo.Lower, ke, scratch, tmpFcopyBase >> 1, ke);
                if (ke < kEndog) {
                    copyObservedVector(
                        fr.forecastError, foreErrOff,
                        scratch, observedBase, ke,
                        scratch, tmpFivBase);
                } else {
                    ZLAS.zcopy(ke, fr.forecastError, foreErrOff, 1, scratch, tmpFivBase, 1);
                }
                ZLAS.zpotrs(BLAS.Uplo.Lower, ke, 1,
                    scratch, tmpFcopyBase >> 1, ke, scratch, tmpFivBase >> 1, 1);
                ZLAS.zcopy(ke * kStates, Z, zOff, 1, scratch, tmpFiZBase, 1);
                ZLAS.zpotrs(BLAS.Uplo.Lower, ke, kStates,
                    scratch, tmpFcopyBase >> 1, ke, scratch, tmpFiZBase >> 1, kStates);
            }

            double[] K;
            int kOff;
            int kLd;
            if (ke < kEndog) {
                copyObservedColumns(kStates,
                    fr.kalmanGain, kgOff, kEndog,
                    scratch, observedBase, ke,
                    scratch, tmpKgBase, ke);
                K = scratch;
                kOff = tmpKgBase;
                kLd = ke;
            } else {
                K = fr.kalmanGain;
                kOff = kgOff;
                kLd = kEndog;
            }

            if (fr.perObsKalmanGain) {
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                    1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                    K, kOff >> 1, kLd,
                    0.0, 0.0, scratch, tmpTkBase >> 1, ke);
                K = scratch;
                kOff = tmpTkBase;
                kLd = ke;
            }

            if (smoothingError != null) {
                int smoothingErrorOff = storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase;
                ZLAS.zgemv(BLAS.Trans.Conj, kStates, ke, -1.0, 0.0,
                    K, kOff, kLd,
                    scratch, rBase, 1,
                    0.0, 0.0, smoothingError, smoothingErrorOff, 1);
                ZLAS.zaxpy(ke, 1.0, 0.0, scratch, tmpFivBase >> 1, 1,
                    smoothingError, smoothingErrorOff >> 1, 1);

                if (ke < kEndog) {
                    System.arraycopy(smoothingError, smoothingErrorOff, scratch, tmpRqBase, 2 * ke);
                    scatterObservedVector(
                        scratch, tmpRqBase,
                        scratch, observedBase, ke,
                        smoothingError, smoothingErrorOff, kEndog);
                }
            }

            ZLAS.zcopy(ksKsC, transition, stateSpace.transitionOffset(t), 1, scratch, lBase, 1);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                -1.0, 0.0, K, kOff >> 1, kLd,
                Z, zOff >> 1, zLd,
                1.0, 0.0, scratch, lBase >> 1, kStates);

            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
                1.0, 0.0, fr.filteredStateCov, filtCovOff >> 1, kStates,
                transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                0.0, 0.0, scratch, tmpRqBase >> 1, kStates);

            if (needSmoothedState) {
                double[] smoothedState = doState ? result.smoothedState : nextSmoothedState;
                int smoothedStateOff = doState ? result.smoothedStateOffset(t) : 0;
                ZLAS.zcopy(kStates, fr.filteredState, filtOff, 1,
                    smoothedState, smoothedStateOff, 1);
                ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                    scratch, tmpRqBase, kStates,
                    scratch, rBase, 1,
                    1.0, 0.0, smoothedState, smoothedStateOff, 1);
            }

            if (needSmoothedStateCov) {
                double[] smoothedStateCov = doStateCov ? result.smoothedStateCov : nextSmoothedStateCov;
                int smoothedStateCovOff = doStateCov ? result.smoothedStateCovOffset(t) : 0;
                if (t < nobs - 1) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, scratch, nBase >> 1, kStates,
                        scratch, tmpRqBase >> 1, kStates,
                        0.0, 0.0, scratch, tmpNpBase >> 1, kStates);
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        -1.0, 0.0, scratch, tmpRqBase >> 1, kStates,
                        scratch, tmpNpBase >> 1, kStates,
                        0.0, 0.0, smoothedStateCov, smoothedStateCovOff >> 1, kStates);
                    ZLAS.zaxpy(ksKsC, 1.0, 0.0, fr.filteredStateCov, filtCovOff >> 1, 1,
                        smoothedStateCov, smoothedStateCovOff >> 1, 1);
                } else {
                    ZLAS.zcopy(ksKsC, fr.filteredStateCov, filtCovOff, 1,
                        smoothedStateCov, smoothedStateCovOff, 1);
                }
                symmetrizeHermitian(smoothedStateCov, smoothedStateCovOff, kStates);
            }

            if (storeAuxiliary) {
                ZLAS.zcopy(ksKsC, scratch, lBase, 1,
                    result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                ZLAS.zcopy(kStates, scratch, rBase, 1,
                    result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                ZLAS.zcopy(ksKsC, scratch, nBase, 1,
                    result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
            }

            if (doDisturbance) {
                int smoothingErrorOff = storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase;
                if (ke < kEndog) {
                    copyObservedVector(
                        smoothingError, smoothingErrorOff,
                        scratch, observedBase, ke,
                        scratch, tmpFivBase);
                    ZLAS.zgemv(BLAS.Trans.NoTrans, ke, ke, 1.0, 0.0,
                        H, hOff, hLd,
                        scratch, tmpFivBase, 1,
                        0.0, 0.0, scratch, tmpFiZBase, 1);
                    scatterObservedVector(
                        scratch, tmpFiZBase,
                        scratch, observedBase, ke,
                        result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t), kEndog);
                } else {
                    ZLAS.zgemv(BLAS.Trans.NoTrans, ke, ke, 1.0, 0.0,
                        H, hOff, hLd,
                        smoothingError, smoothingErrorOff, 1,
                        0.0, 0.0, result.smoothedObsDisturbance,
                        result.smoothedObsDisturbanceOffset(t), 1);
                }

                if (kPosdef > 0) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                        0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                    ZLAS.zgemv(BLAS.Trans.Conj, kStates, kPosdef, 1.0, 0.0,
                        scratch, tmpRqBase, kPosdef,
                        scratch, rBase, 1,
                        0.0, 0.0, result.smoothedStateDisturbance,
                        result.smoothedStateDisturbanceOffset(t), 1);
                }
            }

            if (doDisturbanceCov) {
                double FtRe = ke == 1 ? scratch[tmpFcopyBase] : 0.0;
                double FtIm = ke == 1 ? scratch[tmpFcopyBase + 1] : 0.0;
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, ke,
                    1.0, 0.0, K, kOff >> 1, kLd,
                    H, hOff >> 1, hLd,
                    0.0, 0.0, scratch, tmpKhBase >> 1, ke);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                    1.0, 0.0, scratch, nBase >> 1, kStates,
                    scratch, tmpKhBase >> 1, ke,
                    0.0, 0.0, scratch, tmpNpBase >> 1, ke);

                if (ke >= kEndog) {
                    ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ke, ke, kStates,
                        -1.0, 0.0, scratch, tmpKhBase >> 1, ke,
                        scratch, tmpNpBase >> 1, ke,
                        0.0, 0.0, result.smoothedObsDisturbanceCov,
                        result.smoothedObsDisturbanceCovOffset(t) >> 1, kEndog);
                    ZLAS.zaxpy(ke * ke, 1.0, 0.0, H, hOff >> 1, 1,
                        result.smoothedObsDisturbanceCov,
                        result.smoothedObsDisturbanceCovOffset(t) >> 1, 1);
                }

                if (ke == 1) {
                    double FtMag2 = FtRe * FtRe + FtIm * FtIm;
                    double FtInvRe = FtRe / FtMag2;
                    double FtInvIm = -FtIm / FtMag2;
                    double hRe = H[hOff];
                    double hIm = H[hOff + 1];
                    double hfiRe = hRe * FtInvRe - hIm * FtInvIm;
                    double hfiIm = hRe * FtInvIm + hIm * FtInvRe;
                    double hfhRe = hfiRe * hRe - hfiIm * hIm;
                    double hfhIm = hfiRe * hIm + hfiIm * hRe;
                    if (ke < kEndog) {
                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ke, ke, kStates,
                            -1.0, 0.0, scratch, tmpKhBase >> 1, ke,
                            scratch, tmpNpBase >> 1, ke,
                            0.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                        ZLAS.zaxpy(ke * ke, 1.0, 0.0, H, hOff >> 1, 1,
                            scratch, tmpFcopyBase >> 1, 1);
                        scratch[tmpFcopyBase] -= hfhRe;
                        scratch[tmpFcopyBase + 1] -= hfhIm;
                    } else {
                        int obsCovOff = result.smoothedObsDisturbanceCovOffset(t);
                        result.smoothedObsDisturbanceCov[obsCovOff] -= hfhRe;
                        result.smoothedObsDisturbanceCov[obsCovOff + 1] -= hfhIm;
                    }
                } else {
                    ZLAS.zcopy(ke * ke, H, hOff, 1, scratch, tmpFiHBase, 1);
                    ZLAS.zpotrs(BLAS.Uplo.Lower, ke, ke,
                        scratch, tmpFcopyBase >> 1, ke, scratch, tmpFiHBase >> 1, ke);
                    if (ke < kEndog) {
                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ke, ke, kStates,
                            -1.0, 0.0, scratch, tmpKhBase >> 1, ke,
                            scratch, tmpNpBase >> 1, ke,
                            0.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                        ZLAS.zaxpy(ke * ke, 1.0, 0.0, H, hOff >> 1, 1,
                            scratch, tmpFcopyBase >> 1, 1);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                            -1.0, 0.0, H, hOff >> 1, hLd,
                            scratch, tmpFiHBase >> 1, ke,
                            1.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                    } else {
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                            -1.0, 0.0, H, hOff >> 1, hLd,
                            scratch, tmpFiHBase >> 1, ke,
                            1.0, 0.0, result.smoothedObsDisturbanceCov,
                            result.smoothedObsDisturbanceCovOffset(t) >> 1, kEndog);
                    }
                }

                if (ke < kEndog) {
                    ZLAS.zcopy(kEndog * kEndog, obsCov, observationModel.obsCovOffset(t), 1,
                        result.smoothedObsDisturbanceCov,
                        result.smoothedObsDisturbanceCovOffset(t), 1);
                    scatterObservedSquareMatrix(
                        scratch, tmpFcopyBase, ke,
                        scratch, observedBase, ke,
                        result.smoothedObsDisturbanceCov,
                        result.smoothedObsDisturbanceCovOffset(t), kEndog);
                    restoreMissingSquareEntries(observationModel, t,
                        obsCov, observationModel.obsCovOffset(t), obsCovLd,
                        result.smoothedObsDisturbanceCov,
                        result.smoothedObsDisturbanceCovOffset(t), kEndog);
                }

                if (kPosdef > 0) {
                    if (!doDisturbance) {
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                            1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                            stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                            0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                    }
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                        1.0, 0.0, scratch, nBase >> 1, kStates,
                        scratch, tmpRqBase >> 1, kPosdef,
                        0.0, 0.0, scratch, tmpNpBase >> 1, kPosdef);
                    ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                        -1.0, 0.0, scratch, tmpRqBase >> 1, kPosdef,
                        scratch, tmpNpBase >> 1, kPosdef,
                        0.0, 0.0, result.smoothedStateDisturbanceCov,
                        result.smoothedStateDisturbanceCovOffset(t) >> 1, kPosdef);
                    ZLAS.zaxpy(kPosdef * kPosdef, 1.0, 0.0,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, 1,
                        result.smoothedStateDisturbanceCov,
                        result.smoothedStateDisturbanceCovOffset(t) >> 1, 1);
                }
            }
        }

        zero(scratch, rBase, 2 * kStates);
        zero(scratch, nBase, ksKs);
        if (storeAuxiliary) {
            ZLAS.zcopy(kStates, scratch, rBase, 1,
                result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(nobs), 1);
            ZLAS.zcopy(ksKsC, scratch, nBase, 1,
                result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(nobs), 1);

            shiftLeft(result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorBase(), 2 * kStates, nobs * 2 * kStates);
            shiftLeft(result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovBase(), ksKs, nobs * ksKs);
        }

        return result.trimOutputMask(smootherOutput);
    }

    private static ZSmootherResult smoothAlternative(StateSpaceModel stateSpace,
                                                     StateSpaceModel observationModel,
                                                     ZFilterResult fr,
                                                     Pool pool,
                                                     int smootherOutput,
                                                     boolean reuseResult,
                                                     ZSmootherResult target) {
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int ksKs = 2 * kStates * kStates;
        int ksKsC = kStates * kStates;
        double[] transition = stateSpace.transitionData();
        int transitionLd = stateSpace.transitionLeadingDimension();
        double[] selection = stateSpace.selectionData();
        int selectionLd = stateSpace.selectionLeadingDimension();
        double[] stateCovariance = stateSpace.stateCovarianceData();
        int stateCovarianceLd = stateSpace.stateCovarianceLeadingDimension();
        double[] design = observationModel.designData();
        int designLd = observationModel.designLeadingDimension();
        double[] obsCov = observationModel.obsCovData();
        int obsCovLd = observationModel.obsCovLeadingDimension();

        boolean doState = (smootherOutput & SMOOTHER_STATE) != 0;
        boolean doStateCov = (smootherOutput & SMOOTHER_STATE_COV) != 0;
        boolean doDisturbance = (smootherOutput & SMOOTHER_DISTURBANCE) != 0;
        boolean doDisturbanceCov = (smootherOutput & SMOOTHER_DISTURBANCE_COV) != 0;
        boolean storeAuxiliary = smootherOutput == SMOOTHER_ALL;

        pool.ensureObservationFactorScratch(kStates, kEndog);
        pool.ensureForecastFactorScratch(kEndog);
        if (fr.perObsKalmanGain) {
            pool.ensureTransitionGainScratch(kStates, kEndog);
        }
        if (doDisturbanceCov) {
            pool.ensureDisturbanceCovarianceScratch(kStates, kEndog);
        }
        if ((reuseResult || target != null) && !storeAuxiliary && doDisturbance) {
            pool.ensurePooledScratch(kEndog);
        }

        ZSmootherResult result = prepareResult(pool, target, reuseResult,
            kEndog, kStates, kPosdef, nobs,
            doState, doStateCov, doDisturbance, doDisturbanceCov, storeAuxiliary);
        ScratchArena scratchArena = pool.scratch;
        ZSmootherScratchLayout layout = scratchArena.scratchLayout;
        double[] scratch = scratchArena.scratchBacking;
        int hatRBase = layout.rBase();
        int hatNBase = layout.nBase();
        int tildeRBase = layout.rPrevBase();
        int tildeNBase = layout.nPrevBase();
        int lBase = layout.lBase();
        int tmpNpBase = layout.tmpNpBase();
        int tmpRqBase = layout.tmpRqBase();
        int tmpFivBase = layout.tmpFivBase();
        int tmpFiZBase = layout.tmpFiZBase();
        int tmpFcopyBase = layout.tmpFcopyBase();
        int tmpTkBase = fr.perObsKalmanGain ? layout.tmpTkBase() : -1;
        int tmpKhBase = doDisturbanceCov ? layout.tmpKhBase() : -1;
        int tmpFiHBase = doDisturbanceCov ? layout.tmpFiHBase() : -1;
        int selectedDesignBase = layout.selectedDesignBase();
        int selectedObsCovBase = layout.selectedObsCovBase();
        int tmpKgBase = layout.tmpKgBase();
        int borrowedSmoothingErrorBase = (!storeAuxiliary && reuseResult && doDisturbance)
            ? layout.scratchSmoothingErrorBase()
            : 0;
        double[] smoothingError = (storeAuxiliary || doDisturbance)
            ? (storeAuxiliary ? result.smoothingError : (reuseResult ? scratch : new double[2 * kEndog]))
            : null;
        double[] nextScaledSmoothedEstimator = (!storeAuxiliary && doDisturbance && kPosdef > 0)
            ? new double[2 * kStates]
            : null;
        double[] nextScaledSmoothedEstimatorCov = (!storeAuxiliary && doDisturbanceCov)
            ? new double[ksKs]
            : null;

        zero(scratch, hatRBase, 2 * kStates);
        zero(scratch, hatNBase, ksKs);
        zero(scratch, tildeRBase, 2 * kStates);
        zero(scratch, tildeNBase, ksKs);
        if (storeAuxiliary) {
            ZLAS.zcopy(kStates, scratch, tildeRBase, 1,
                result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(nobs), 1);
            ZLAS.zcopy(ksKsC, scratch, tildeNBase, 1,
                result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(nobs), 1);
        }

        for (int t = nobs - 1; t >= 0; t--) {
            int filtOff = fr.filteredStateOffset(t);
            int filtCovOff = fr.filteredStateCovOffset(t);
            int foreErrOff = fr.forecastErrorOffset(t);
            int foreErrCovOff = fr.forecastErrorCovOffset(t);
            int kgOff = fr.kalmanGainOffset(t);
            int smoothingErrorOff = storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase;

            if (!storeAuxiliary) {
                if (nextScaledSmoothedEstimator != null) {
                    ZLAS.zcopy(kStates, scratch, tildeRBase, 1, nextScaledSmoothedEstimator, 0, 1);
                }
                if (nextScaledSmoothedEstimatorCov != null) {
                    ZLAS.zcopy(ksKsC, scratch, tildeNBase, 1, nextScaledSmoothedEstimatorCov, 0, 1);
                }
            }

            int ke = selectObservations(observationModel, scratchArena, layout, t);
            int observedBase = ke > 0 && ke < kEndog
                ? observedIndexBase(selectedObsCovBase, ke)
                : -1;

            if (doState) {
                int smoothedStateOff = result.smoothedStateOffset(t);
                ZLAS.zcopy(kStates, fr.filteredState, filtOff, 1,
                    result.smoothedState, smoothedStateOff, 1);
                ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                    fr.filteredStateCov, filtCovOff, kStates,
                    scratch, hatRBase, 1,
                    1.0, 0.0, result.smoothedState, smoothedStateOff, 1);
            }

            if (doStateCov) {
                int smoothedStateCovOff = result.smoothedStateCovOffset(t);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, 0.0, scratch, hatNBase >> 1, kStates,
                    fr.filteredStateCov, filtCovOff >> 1, kStates,
                    0.0, 0.0, scratch, tmpNpBase >> 1, kStates);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    -1.0, 0.0, fr.filteredStateCov, filtCovOff >> 1, kStates,
                    scratch, tmpNpBase >> 1, kStates,
                    0.0, 0.0, result.smoothedStateCov, smoothedStateCovOff >> 1, kStates);
                ZLAS.zaxpy(ksKsC, 1.0, 0.0, fr.filteredStateCov, filtCovOff >> 1, 1,
                    result.smoothedStateCov, smoothedStateCovOff >> 1, 1);
                symmetrizeHermitian(result.smoothedStateCov, smoothedStateCovOff, kStates);
            }

            if (ke == 0) {
                ZLAS.zcopy(kStates, scratch, hatRBase, 1, scratch, tildeRBase, 1);
                ZLAS.zcopy(ksKsC, scratch, hatNBase, 1, scratch, tildeNBase, 1);
                ZLAS.zcopy(ksKsC, transition, stateSpace.transitionOffset(t), 1, scratch, lBase, 1);

                if (doDisturbance) {
                    zero(result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t), 2 * kEndog);
                }

                if (doDisturbance && kPosdef > 0) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                        0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                    ZLAS.zgemv(BLAS.Trans.Conj, kStates, kPosdef, 1.0, 0.0,
                        scratch, tmpRqBase, kPosdef,
                        storeAuxiliary ? result.scaledSmoothedEstimator : nextScaledSmoothedEstimator,
                        storeAuxiliary ? result.scaledSmoothedEstimatorOffset(t + 1) : 0, 1,
                        0.0, 0.0, result.smoothedStateDisturbance,
                        result.smoothedStateDisturbanceOffset(t), 1);
                }

                if (doDisturbanceCov) {
                    ZLAS.zcopy(kEndog * kEndog, obsCov, observationModel.obsCovOffset(t), 1,
                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                    if (kPosdef > 0) {
                        if (!doDisturbance) {
                            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                                stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                                0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                        }
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                            1.0, 0.0, scratch, hatNBase >> 1, kStates,
                            scratch, tmpRqBase >> 1, kPosdef,
                            0.0, 0.0, scratch, tmpNpBase >> 1, kPosdef);
                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                            -1.0, 0.0, scratch, tmpRqBase >> 1, kPosdef,
                            scratch, tmpNpBase >> 1, kPosdef,
                            0.0, 0.0, result.smoothedStateDisturbanceCov,
                            result.smoothedStateDisturbanceCovOffset(t) >> 1, kPosdef);
                        ZLAS.zaxpy(kPosdef * kPosdef, 1.0, 0.0,
                            stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, 1,
                            result.smoothedStateDisturbanceCov,
                            result.smoothedStateDisturbanceCovOffset(t) >> 1, 1);
                    }
                }

                if (storeAuxiliary) {
                    ZLAS.zcopy(ksKsC, scratch, lBase, 1,
                        result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                    ZLAS.zcopy(kStates, scratch, tildeRBase, 1,
                        result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                    ZLAS.zcopy(ksKsC, scratch, tildeNBase, 1,
                        result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
                }

                ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                    scratch, tildeRBase, 1,
                    0.0, 0.0, scratch, hatRBase, 1);
                ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                    scratch, tildeNBase >> 1, kStates,
                    0.0, 0.0, scratch, tmpNpBase >> 1, kStates);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, 0.0, scratch, tmpNpBase >> 1, kStates,
                    transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                    0.0, 0.0, scratch, hatNBase >> 1, kStates);
                continue;
            }

            double[] Z;
            int zOff;
            int zLd;
            double[] H;
            int hOff;
            int hLd;
            if (ke < kEndog) {
                Z = scratch;
                zOff = selectedDesignBase;
                zLd = kStates;
                H = scratch;
                hOff = selectedObsCovBase;
                hLd = ke;
            } else {
                Z = design;
                zOff = observationModel.designOffset(t);
                zLd = designLd;
                H = obsCov;
                hOff = observationModel.obsCovOffset(t);
                hLd = obsCovLd;
            }

            if (ke < kEndog) {
                copyObservedSquareMatrix(
                    fr.forecastErrorCov, foreErrCovOff, kEndog,
                    scratch, observedBase, ke,
                    scratch, tmpFcopyBase, ke);
            } else {
                ZLAS.zcopy(ke * ke, fr.forecastErrorCov, foreErrCovOff, 1, scratch, tmpFcopyBase, 1);
            }

            if (ke == 1) {
                double FtRe = scratch[tmpFcopyBase];
                double FtIm = scratch[tmpFcopyBase + 1];
                double FtMag2 = FtRe * FtRe + FtIm * FtIm;
                double FtInvRe = FtRe / FtMag2;
                double FtInvIm = -FtIm / FtMag2;
                if (ke < kEndog) {
                    copyObservedVector(
                        fr.forecastError, foreErrOff,
                        scratch, observedBase, ke,
                        scratch, tmpFivBase);
                    double vRe = scratch[tmpFivBase];
                    double vIm = scratch[tmpFivBase + 1];
                    scratch[tmpFivBase] = vRe * FtInvRe - vIm * FtInvIm;
                    scratch[tmpFivBase + 1] = vRe * FtInvIm + vIm * FtInvRe;
                } else {
                    double vRe = fr.forecastError[foreErrOff];
                    double vIm = fr.forecastError[foreErrOff + 1];
                    scratch[tmpFivBase] = vRe * FtInvRe - vIm * FtInvIm;
                    scratch[tmpFivBase + 1] = vRe * FtInvIm + vIm * FtInvRe;
                }
                for (int j = 0; j < kStates; j++) {
                    double zr = Z[zOff + j * 2];
                    double zi = Z[zOff + j * 2 + 1];
                    scratch[tmpFiZBase + j * 2] = zr * FtInvRe - zi * FtInvIm;
                    scratch[tmpFiZBase + j * 2 + 1] = zr * FtInvIm + zi * FtInvRe;
                }
            } else {
                ZLAS.zpotrf(BLAS.Uplo.Lower, ke, scratch, tmpFcopyBase >> 1, ke);
                if (ke < kEndog) {
                    copyObservedVector(
                        fr.forecastError, foreErrOff,
                        scratch, observedBase, ke,
                        scratch, tmpFivBase);
                } else {
                    ZLAS.zcopy(ke, fr.forecastError, foreErrOff, 1, scratch, tmpFivBase, 1);
                }
                ZLAS.zpotrs(BLAS.Uplo.Lower, ke, 1,
                    scratch, tmpFcopyBase >> 1, ke, scratch, tmpFivBase >> 1, 1);
                ZLAS.zcopy(ke * kStates, Z, zOff, 1, scratch, tmpFiZBase, 1);
                ZLAS.zpotrs(BLAS.Uplo.Lower, ke, kStates,
                    scratch, tmpFcopyBase >> 1, ke, scratch, tmpFiZBase >> 1, kStates);
            }

            double[] K;
            int kOff;
            int kLd;
            if (ke < kEndog) {
                copyObservedColumns(kStates,
                    fr.kalmanGain, kgOff, kEndog,
                    scratch, observedBase, ke,
                    scratch, tmpKgBase, ke);
                K = scratch;
                kOff = tmpKgBase;
                kLd = ke;
            } else {
                K = fr.kalmanGain;
                kOff = kgOff;
                kLd = kEndog;
            }

            if (fr.perObsKalmanGain) {
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                    1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                    K, kOff >> 1, kLd,
                    0.0, 0.0, scratch, tmpTkBase >> 1, ke);
                K = scratch;
                kOff = tmpTkBase;
                kLd = ke;
            }

            if (smoothingError != null) {
                ZLAS.zgemv(BLAS.Trans.Conj, kStates, ke, -1.0, 0.0,
                    K, kOff, kLd,
                    scratch, tildeRBase, 1,
                    0.0, 0.0, smoothingError, smoothingErrorOff, 1);
                ZLAS.zaxpy(ke, 1.0, 0.0, scratch, tmpFivBase >> 1, 1,
                    smoothingError, smoothingErrorOff >> 1, 1);

                if (ke < kEndog) {
                    System.arraycopy(smoothingError, smoothingErrorOff, scratch, tmpRqBase, 2 * ke);
                    scatterObservedVector(
                        scratch, tmpRqBase,
                        scratch, observedBase, ke,
                        smoothingError, smoothingErrorOff, kEndog);
                }
            }

            ZLAS.zcopy(ksKsC, transition, stateSpace.transitionOffset(t), 1, scratch, lBase, 1);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                -1.0, 0.0, K, kOff >> 1, kLd,
                Z, zOff >> 1, zLd,
                1.0, 0.0, scratch, lBase >> 1, kStates);

            int predCovOff = fr.predictedStateCovOffset(t);
            ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, ke,
                1.0, 0.0, Z, zOff >> 1, zLd,
                scratch, tmpFiZBase >> 1, kStates,
                0.0, 0.0, scratch, tmpRqBase >> 1, kStates);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                -1.0, 0.0, fr.predictedStateCov, predCovOff >> 1, kStates,
                scratch, tmpRqBase >> 1, kStates,
                0.0, 0.0, scratch, tmpNpBase >> 1, kStates);
            for (int i = 0; i < kStates; i++) {
                scratch[tmpNpBase + i * 2 * kStates + i * 2] += 1.0;
            }

            ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                scratch, tmpNpBase, kStates,
                scratch, hatRBase, 1,
                0.0, 0.0, scratch, tildeRBase, 1);
            ZLAS.zgemv(BLAS.Trans.Conj, ke, kStates, 1.0, 0.0,
                Z, zOff, zLd,
                scratch, tmpFivBase, 1,
                1.0, 0.0, scratch, tildeRBase, 1);

            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, 0.0, scratch, hatNBase >> 1, kStates,
                scratch, tmpNpBase >> 1, kStates,
                0.0, 0.0, scratch, tmpRqBase >> 1, kStates);
            ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, 0.0, scratch, tmpNpBase >> 1, kStates,
                scratch, tmpRqBase >> 1, kStates,
                0.0, 0.0, scratch, tildeNBase >> 1, kStates);
            ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, ke,
                1.0, 0.0, Z, zOff >> 1, zLd,
                scratch, tmpFiZBase >> 1, kStates,
                1.0, 0.0, scratch, tildeNBase >> 1, kStates);

            if (storeAuxiliary) {
                ZLAS.zcopy(ksKsC, scratch, lBase, 1,
                    result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                ZLAS.zcopy(kStates, scratch, tildeRBase, 1,
                    result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                ZLAS.zcopy(ksKsC, scratch, tildeNBase, 1,
                    result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
            }

            if (doDisturbance) {
                if (ke < kEndog) {
                    copyObservedVector(
                        smoothingError, smoothingErrorOff,
                        scratch, observedBase, ke,
                        scratch, tmpFivBase);
                    ZLAS.zgemv(BLAS.Trans.NoTrans, ke, ke, 1.0, 0.0,
                        H, hOff, hLd,
                        scratch, tmpFivBase, 1,
                        0.0, 0.0, scratch, tmpFiZBase, 1);
                    scatterObservedVector(
                        scratch, tmpFiZBase,
                        scratch, observedBase, ke,
                        result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t), kEndog);
                } else {
                    ZLAS.zgemv(BLAS.Trans.NoTrans, ke, ke, 1.0, 0.0,
                        H, hOff, hLd,
                        smoothingError, smoothingErrorOff, 1,
                        0.0, 0.0, result.smoothedObsDisturbance,
                        result.smoothedObsDisturbanceOffset(t), 1);
                }

                if (kPosdef > 0) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                        0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                    ZLAS.zgemv(BLAS.Trans.Conj, kStates, kPosdef, 1.0, 0.0,
                        scratch, tmpRqBase, kPosdef,
                        storeAuxiliary ? result.scaledSmoothedEstimator : nextScaledSmoothedEstimator,
                        storeAuxiliary ? result.scaledSmoothedEstimatorOffset(t + 1) : 0, 1,
                        0.0, 0.0, result.smoothedStateDisturbance,
                        result.smoothedStateDisturbanceOffset(t), 1);
                }
            }

            if (doDisturbanceCov) {
                double[] scaledSmoothedEstimatorCov = storeAuxiliary
                    ? result.scaledSmoothedEstimatorCov
                    : nextScaledSmoothedEstimatorCov;
                int scaledSmoothedEstimatorCovOff = storeAuxiliary
                    ? result.scaledSmoothedEstimatorCovOffset(t + 1)
                    : 0;
                double FtRe = ke == 1 ? scratch[tmpFcopyBase] : 0.0;
                double FtIm = ke == 1 ? scratch[tmpFcopyBase + 1] : 0.0;
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, ke,
                    1.0, 0.0, K, kOff >> 1, kLd,
                    H, hOff >> 1, hLd,
                    0.0, 0.0, scratch, tmpKhBase >> 1, ke);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                    1.0, 0.0, scaledSmoothedEstimatorCov, scaledSmoothedEstimatorCovOff >> 1, kStates,
                    scratch, tmpKhBase >> 1, ke,
                    0.0, 0.0, scratch, tmpNpBase >> 1, ke);

                if (ke >= kEndog) {
                    ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ke, ke, kStates,
                        -1.0, 0.0, scratch, tmpKhBase >> 1, ke,
                        scratch, tmpNpBase >> 1, ke,
                        0.0, 0.0, result.smoothedObsDisturbanceCov,
                        result.smoothedObsDisturbanceCovOffset(t) >> 1, kEndog);
                    ZLAS.zaxpy(ke * ke, 1.0, 0.0, H, hOff >> 1, 1,
                        result.smoothedObsDisturbanceCov,
                        result.smoothedObsDisturbanceCovOffset(t) >> 1, 1);
                }

                if (ke == 1) {
                    double FtMag2 = FtRe * FtRe + FtIm * FtIm;
                    double FtInvRe = FtRe / FtMag2;
                    double FtInvIm = -FtIm / FtMag2;
                    double hRe = H[hOff];
                    double hIm = H[hOff + 1];
                    double hfiRe = hRe * FtInvRe - hIm * FtInvIm;
                    double hfiIm = hRe * FtInvIm + hIm * FtInvRe;
                    double hfhRe = hfiRe * hRe - hfiIm * hIm;
                    double hfhIm = hfiRe * hIm + hfiIm * hRe;
                    if (ke < kEndog) {
                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ke, ke, kStates,
                            -1.0, 0.0, scratch, tmpKhBase >> 1, ke,
                            scratch, tmpNpBase >> 1, ke,
                            0.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                        ZLAS.zaxpy(ke * ke, 1.0, 0.0, H, hOff >> 1, 1,
                            scratch, tmpFcopyBase >> 1, 1);
                        scratch[tmpFcopyBase] -= hfhRe;
                        scratch[tmpFcopyBase + 1] -= hfhIm;
                    } else {
                        int obsCovOff = result.smoothedObsDisturbanceCovOffset(t);
                        result.smoothedObsDisturbanceCov[obsCovOff] -= hfhRe;
                        result.smoothedObsDisturbanceCov[obsCovOff + 1] -= hfhIm;
                    }
                } else {
                    ZLAS.zcopy(ke * ke, H, hOff, 1, scratch, tmpFiHBase, 1);
                    ZLAS.zpotrs(BLAS.Uplo.Lower, ke, ke,
                        scratch, tmpFcopyBase >> 1, ke, scratch, tmpFiHBase >> 1, ke);
                    if (ke < kEndog) {
                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ke, ke, kStates,
                            -1.0, 0.0, scratch, tmpKhBase >> 1, ke,
                            scratch, tmpNpBase >> 1, ke,
                            0.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                        ZLAS.zaxpy(ke * ke, 1.0, 0.0, H, hOff >> 1, 1,
                            scratch, tmpFcopyBase >> 1, 1);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                            -1.0, 0.0, H, hOff >> 1, hLd,
                            scratch, tmpFiHBase >> 1, ke,
                            1.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                    } else {
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                            -1.0, 0.0, H, hOff >> 1, hLd,
                            scratch, tmpFiHBase >> 1, ke,
                            1.0, 0.0, result.smoothedObsDisturbanceCov,
                            result.smoothedObsDisturbanceCovOffset(t) >> 1, kEndog);
                    }
                }

                if (ke < kEndog) {
                    ZLAS.zcopy(kEndog * kEndog, obsCov, observationModel.obsCovOffset(t), 1,
                        result.smoothedObsDisturbanceCov,
                        result.smoothedObsDisturbanceCovOffset(t), 1);
                    scatterObservedSquareMatrix(
                        scratch, tmpFcopyBase, ke,
                        scratch, observedBase, ke,
                        result.smoothedObsDisturbanceCov,
                        result.smoothedObsDisturbanceCovOffset(t), kEndog);
                    restoreMissingSquareEntries(observationModel, t,
                        obsCov, observationModel.obsCovOffset(t), obsCovLd,
                        result.smoothedObsDisturbanceCov,
                        result.smoothedObsDisturbanceCovOffset(t), kEndog);
                }

                if (kPosdef > 0) {
                    if (!doDisturbance) {
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                            1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                            stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                            0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                    }
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                        1.0, 0.0, scaledSmoothedEstimatorCov, scaledSmoothedEstimatorCovOff >> 1, kStates,
                        scratch, tmpRqBase >> 1, kPosdef,
                        0.0, 0.0, scratch, tmpNpBase >> 1, kPosdef);
                    ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                        -1.0, 0.0, scratch, tmpRqBase >> 1, kPosdef,
                        scratch, tmpNpBase >> 1, kPosdef,
                        0.0, 0.0, result.smoothedStateDisturbanceCov,
                        result.smoothedStateDisturbanceCovOffset(t) >> 1, kPosdef);
                    ZLAS.zaxpy(kPosdef * kPosdef, 1.0, 0.0,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, 1,
                        result.smoothedStateDisturbanceCov,
                        result.smoothedStateDisturbanceCovOffset(t) >> 1, 1);
                }
            }

            ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                transition, stateSpace.transitionOffset(t), transitionLd,
                scratch, tildeRBase, 1,
                0.0, 0.0, scratch, hatRBase, 1);
            ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                scratch, tildeNBase >> 1, kStates,
                0.0, 0.0, scratch, tmpNpBase >> 1, kStates);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, 0.0, scratch, tmpNpBase >> 1, kStates,
                transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                0.0, 0.0, scratch, hatNBase >> 1, kStates);
        }

        if (storeAuxiliary) {
            shiftLeft(result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorBase(), 2 * kStates, nobs * 2 * kStates);
            shiftLeft(result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovBase(), ksKs, nobs * ksKs);
        }

        return result.trimOutputMask(smootherOutput);
    }

    private static ZSmootherResult smoothConventionalCore(StateSpaceModel stateSpace,
                                                          StateSpaceModel observationModel,
                                                          ZFilterResult fr,
                                                          Pool pool,
                                                          int smootherOutput,
                                                          boolean reuseResult,
                                                          ZSmootherResult target) {
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int ksKs = kStates * 2 * kStates;
        int ksKsC = kStates * kStates;
        double[] transition = stateSpace.transitionData();
        int transitionLd = stateSpace.transitionLeadingDimension();
        double[] selection = stateSpace.selectionData();
        int selectionLd = stateSpace.selectionLeadingDimension();
        double[] stateCovariance = stateSpace.stateCovarianceData();
        int stateCovarianceLd = stateSpace.stateCovarianceLeadingDimension();
        double[] design = observationModel.designData();
        int designLd = observationModel.designLeadingDimension();
        double[] obsCov = observationModel.obsCovData();
        int obsCovLd = observationModel.obsCovLeadingDimension();

        pool.ensureObservationFactorScratch(kStates, kEndog);

        boolean doState = (smootherOutput & SMOOTHER_STATE) != 0;
        boolean doStateCov = (smootherOutput & SMOOTHER_STATE_COV) != 0;
        boolean doDisturbance = (smootherOutput & SMOOTHER_DISTURBANCE) != 0;
        boolean doDisturbanceCov = (smootherOutput & SMOOTHER_DISTURBANCE_COV) != 0;
        boolean storeAuxiliary = smootherOutput == SMOOTHER_ALL;

        pool.ensureForecastFactorScratch(kEndog);
        if (fr.perObsKalmanGain) {
            pool.ensureTransitionGainScratch(kStates, kEndog);
        }
        if (doDisturbanceCov) {
            pool.ensureDisturbanceCovarianceScratch(kStates, kEndog);
        }

        if ((reuseResult || target != null) && !storeAuxiliary && doDisturbance) {
            pool.ensurePooledScratch(kEndog);
        }

        ZSmootherResult result = prepareResult(pool, target, reuseResult,
            kEndog, kStates, kPosdef, nobs,
            doState, doStateCov, doDisturbance, doDisturbanceCov, storeAuxiliary);
        ScratchArena scratchArena = pool.scratch;
        ZSmootherScratchLayout layout = scratchArena.scratchLayout;
        double[] scratch = scratchArena.scratchBacking;
        int rBase = layout.rBase();
        int rPrevBase = layout.rPrevBase();
        int nBase = layout.nBase();
        int nPrevBase = layout.nPrevBase();
        int lBase = layout.lBase();
        int tmpNpBase = layout.tmpNpBase();
        int tmpRqBase = layout.tmpRqBase();
        int tmpFivBase = layout.tmpFivBase();
        int tmpFiZBase = layout.tmpFiZBase();
        int selectedDesignBase = layout.selectedDesignBase();
        int selectedObsCovBase = layout.selectedObsCovBase();
        int tmpKgBase = layout.tmpKgBase();
        int tmpFcopyBase = layout.tmpFcopyBase();
        int tmpTkBase = fr.perObsKalmanGain ? layout.tmpTkBase() : -1;
        int tmpKhBase = doDisturbanceCov ? layout.tmpKhBase() : -1;
        int tmpFiHBase = doDisturbanceCov ? layout.tmpFiHBase() : -1;
        int borrowedSmoothingErrorBase = (!storeAuxiliary && reuseResult && doDisturbance)
            ? layout.scratchSmoothingErrorBase()
            : 0;
        double[] smoothingError = (storeAuxiliary || doDisturbance)
            ? (storeAuxiliary ? result.smoothingError : (reuseResult ? scratch : new double[2 * kEndog]))
                : null;

        zero(scratch, rBase, 2 * kStates);
        zero(scratch, nBase, ksKs);

        for (int t = nobs - 1; t >= 0; t--) {
            int predOff = fr.predictedStateOffset(t);
            int predCovOff = fr.predictedStateCovOffset(t);
            int foreErrOff = fr.forecastErrorOffset(t);
            int foreErrCovOff = fr.forecastErrorCovOffset(t);
            int kgOff = fr.kalmanGainOffset(t);
            int smoothedStateOff = result.smoothedStateOffset(t);
            int smoothedStateCovOff = result.smoothedStateCovOffset(t);
            int innovationsTransitionOff = result.innovationsTransitionOffset(t);
            int scaledSmoothedEstimatorOff = result.scaledSmoothedEstimatorOffset(t);
            int scaledSmoothedEstimatorCovOff = result.scaledSmoothedEstimatorCovOffset(t);
            int smoothingErrorOff = storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase;
            int smoothedObsDisturbanceOff = result.smoothedObsDisturbanceOffset(t);
            int smoothedStateDisturbanceOff = result.smoothedStateDisturbanceOffset(t);
            int smoothedObsDisturbanceCovOff = result.smoothedObsDisturbanceCovOffset(t);
            int smoothedStateDisturbanceCovOff = result.smoothedStateDisturbanceCovOffset(t);

            int ke = selectObservations(observationModel, scratchArena, layout, t);
            int observedBase = ke > 0 && ke < kEndog
                ? observedIndexBase(selectedObsCovBase, ke)
                : -1;
            if (ke == 0) {
                ZLAS.zcopy(kStates * kStates, transition, stateSpace.transitionOffset(t), 1,
                        scratch, lBase, 1);
                ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                    transition, stateSpace.transitionOffset(t), transitionLd,
                        scratch, rBase, 1,
                        0.0, 0.0, scratch, rPrevBase, 1);
                ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                        scratch, nBase >> 1, kStates,
                        0.0, 0.0, scratch, tmpNpBase >> 1, kStates);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, scratch, tmpNpBase >> 1, kStates,
                    transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                        0.0, 0.0, scratch, nPrevBase >> 1, kStates);

                if (doState) {
                    ZLAS.zcopy(kStates, fr.predictedState, predOff, 1,
                        result.smoothedState, smoothedStateOff, 1);
                    ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                            fr.predictedStateCov, predCovOff, kStates,
                            scratch, rPrevBase, 1,
                        1.0, 0.0, result.smoothedState, smoothedStateOff, 1);
                }
                if (doStateCov) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            -1.0, 0.0, fr.predictedStateCov, predCovOff >> 1, kStates,
                            scratch, nPrevBase >> 1, kStates,
                            0.0, 0.0, scratch, tmpNpBase >> 1, kStates);
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpNpBase >> 1, kStates,
                            fr.predictedStateCov, predCovOff >> 1, kStates,
                            0.0, 0.0, result.smoothedStateCov, smoothedStateCovOff >> 1, kStates);
                    ZLAS.zaxpy(ksKsC, 1.0, 0.0, fr.predictedStateCov, predCovOff >> 1, 1,
                            result.smoothedStateCov, smoothedStateCovOff >> 1, 1);
                    for (int i = 0; i < kStates; i++) {
                        for (int j = i + 1; j < kStates; j++) {
                            int ij = smoothedStateCovOff + i * 2 * kStates + j * 2;
                            int ji = smoothedStateCovOff + j * 2 * kStates + i * 2;
                            double avgRe = (result.smoothedStateCov[ij] + result.smoothedStateCov[ji]) * 0.5;
                            double avgIm = (result.smoothedStateCov[ij + 1] - result.smoothedStateCov[ji + 1]) * 0.5;
                            result.smoothedStateCov[ij] = avgRe;
                            result.smoothedStateCov[ij + 1] = avgIm;
                            result.smoothedStateCov[ji] = avgRe;
                            result.smoothedStateCov[ji + 1] = -avgIm;
                        }
                    }
                }

                if (storeAuxiliary) {
                    ZLAS.zcopy(kStates * kStates, scratch, lBase, 1,
                result.innovationsTransition, innovationsTransitionOff, 1);
                    ZLAS.zcopy(kStates, scratch, rPrevBase, 1,
                result.scaledSmoothedEstimator, scaledSmoothedEstimatorOff, 1);
                    ZLAS.zcopy(kStates * kStates, scratch, nPrevBase, 1,
                result.scaledSmoothedEstimatorCov, scaledSmoothedEstimatorCovOff, 1);
                }

                if (doDisturbance && kPosdef > 0) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                            0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                    ZLAS.zgemv(BLAS.Trans.Conj, kStates, kPosdef, 1.0, 0.0,
                            scratch, tmpRqBase, kPosdef,
                            scratch, rBase, 1,
                            0.0, 0.0, result.smoothedStateDisturbance, smoothedStateDisturbanceOff, 1);
                }

                if (doDisturbanceCov) {
                    ZLAS.zcopy(kEndog * kEndog, obsCov, observationModel.obsCovOffset(t), 1,
                            result.smoothedObsDisturbanceCov, smoothedObsDisturbanceCovOff, 1);
                    if (kPosdef > 0) {
                        if (!doDisturbance) {
                            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                    1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                                    stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                                    0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                        }
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                                1.0, 0.0, scratch, nBase >> 1, kStates,
                                scratch, tmpRqBase >> 1, kPosdef,
                                0.0, 0.0, scratch, tmpNpBase >> 1, kPosdef);
                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                                -1.0, 0.0, scratch, tmpRqBase >> 1, kPosdef,
                                scratch, tmpNpBase >> 1, kPosdef,
                            0.0, 0.0, result.smoothedStateDisturbanceCov, smoothedStateDisturbanceCovOff >> 1, kPosdef);
                        ZLAS.zaxpy(kPosdef * kPosdef, 1.0, 0.0, stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, 1,
                            result.smoothedStateDisturbanceCov, smoothedStateDisturbanceCovOff >> 1, 1);
                    }
                }

                int tmpRBase = rBase;
                rBase = rPrevBase;
                rPrevBase = tmpRBase;
                int tmpNBase = nBase;
                nBase = nPrevBase;
                nPrevBase = tmpNBase;
                continue;
            }

            double[] Z; int zOff; int zLd;
            double[] H; int hOff; int hLd;
            if (ke < kEndog) {
                Z = scratch; zOff = selectedDesignBase; zLd = kStates;
                H = scratch; hOff = selectedObsCovBase; hLd = ke;
            } else {
                Z = design; zOff = observationModel.designOffset(t); zLd = designLd;
                H = obsCov; hOff = observationModel.obsCovOffset(t); hLd = obsCovLd;
            }

            if (ke < kEndog) {
                copyObservedSquareMatrix(
                    fr.forecastErrorCov, foreErrCovOff, kEndog,
                    scratch, observedBase, ke,
                    scratch, tmpFcopyBase, ke);
            } else {
                ZLAS.zcopy(kEndog * kEndog, fr.forecastErrorCov, foreErrCovOff, 1, scratch, tmpFcopyBase, 1);
            }

            if (ke == 1) {
                double FtRe = scratch[tmpFcopyBase];
                double FtIm = scratch[tmpFcopyBase + 1];
                double FtMag2 = FtRe * FtRe + FtIm * FtIm;
                double FtInvRe = FtRe / FtMag2;
                double FtInvIm = -FtIm / FtMag2;
                if (ke < kEndog) {
                    copyObservedVector(
                        fr.forecastError, foreErrOff,
                        scratch, observedBase, ke,
                        scratch, tmpFivBase);
                    double vRe = scratch[tmpFivBase];
                    double vIm = scratch[tmpFivBase + 1];
                    scratch[tmpFivBase] = vRe * FtInvRe - vIm * FtInvIm;
                    scratch[tmpFivBase + 1] = vRe * FtInvIm + vIm * FtInvRe;
                } else {
                    double vRe = fr.forecastError[foreErrOff];
                    double vIm = fr.forecastError[foreErrOff + 1];
                    scratch[tmpFivBase] = vRe * FtInvRe - vIm * FtInvIm;
                    scratch[tmpFivBase + 1] = vRe * FtInvIm + vIm * FtInvRe;
                }
                for (int j = 0; j < kStates; j++) {
                    double zr = Z[zOff + j * 2];
                    double zi = Z[zOff + j * 2 + 1];
                    scratch[tmpFiZBase + j * 2] = zr * FtInvRe - zi * FtInvIm;
                    scratch[tmpFiZBase + j * 2 + 1] = zr * FtInvIm + zi * FtInvRe;
                }
            } else {
                ZLAS.zpotrf(BLAS.Uplo.Lower, ke, scratch, tmpFcopyBase >> 1, ke);
                if (ke < kEndog) {
                    copyObservedVector(
                        fr.forecastError, foreErrOff,
                        scratch, observedBase, ke,
                        scratch, tmpFivBase);
                } else {
                    ZLAS.zcopy(ke, fr.forecastError, foreErrOff, 1, scratch, tmpFivBase, 1);
                }
                ZLAS.zpotrs(BLAS.Uplo.Lower, ke, 1,
                    scratch, tmpFcopyBase >> 1, ke, scratch, tmpFivBase >> 1, 1);
                ZLAS.zcopy(ke * kStates, Z, zOff, 1, scratch, tmpFiZBase, 1);
                ZLAS.zpotrs(BLAS.Uplo.Lower, ke, kStates,
                    scratch, tmpFcopyBase >> 1, ke, scratch, tmpFiZBase >> 1, kStates);
            }

            double[] K; int kOff; int kLd;
            if (ke < kEndog) {
                copyObservedColumns(kStates,
                    fr.kalmanGain, kgOff, kEndog,
                    scratch, observedBase, ke,
                    scratch, tmpKgBase, ke);
                K = scratch; kOff = tmpKgBase; kLd = ke;
            } else {
                K = fr.kalmanGain; kOff = kgOff; kLd = kEndog;
            }

            if (fr.perObsKalmanGain) {
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                    1.0, 0.0, transition, stateSpace.transitionOffset(t) >> 1, transitionLd,
                    K, kOff >> 1, kLd,
                    0.0, 0.0, scratch, tmpTkBase >> 1, ke);
                K = scratch;
                kOff = tmpTkBase;
                kLd = ke;
            }

            if (smoothingError != null) {
                ZLAS.zgemv(BLAS.Trans.Conj, kStates, ke, -1.0, 0.0,
                    K, kOff, kLd,
                    scratch, rBase, 1,
                    0.0, 0.0, smoothingError, smoothingErrorOff, 1);
                ZLAS.zaxpy(ke, 1.0, 0.0, scratch, tmpFivBase >> 1, 1,
                    smoothingError, smoothingErrorOff >> 1, 1);

                if (ke < kEndog) {
                    System.arraycopy(smoothingError, smoothingErrorOff, scratch, tmpRqBase, 2 * ke);
                    scatterObservedVector(
                        scratch, tmpRqBase,
                        scratch, observedBase, ke,
                        smoothingError, smoothingErrorOff, kEndog);
                }
            }

            ZLAS.zcopy(kStates * kStates, transition, stateSpace.transitionOffset(t), 1,
                    scratch, lBase, 1);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                    -1.0, 0.0, K, kOff >> 1, kLd,
                    Z, zOff >> 1, zLd,
                    1.0, 0.0, scratch, lBase >> 1, kStates);

            ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                    scratch, lBase, kStates,
                    scratch, rBase, 1,
                    0.0, 0.0, scratch, rPrevBase, 1);
            ZLAS.zgemv(BLAS.Trans.Conj, ke, kStates, 1.0, 0.0,
                    Z, zOff, zLd,
                    scratch, tmpFivBase, 1,
                    1.0, 0.0, scratch, rPrevBase, 1);

            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, 0.0, scratch, nBase >> 1, kStates,
                    scratch, lBase >> 1, kStates,
                    0.0, 0.0, scratch, tmpNpBase >> 1, kStates);
            ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, 0.0, scratch, lBase >> 1, kStates,
                    scratch, tmpNpBase >> 1, kStates,
                    0.0, 0.0, scratch, nPrevBase >> 1, kStates);
            ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, ke,
                    1.0, 0.0, Z, zOff >> 1, zLd,
                    scratch, tmpFiZBase >> 1, kStates,
                    1.0, 0.0, scratch, nPrevBase >> 1, kStates);

            if (doState) {
                ZLAS.zcopy(kStates, fr.predictedState, predOff, 1,
                    result.smoothedState, smoothedStateOff, 1);
                ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                        fr.predictedStateCov, predCovOff, kStates,
                    scratch, rPrevBase, 1,
                    1.0, 0.0, result.smoothedState, smoothedStateOff, 1);
            }

            if (doStateCov) {
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        -1.0, 0.0, fr.predictedStateCov, predCovOff >> 1, kStates,
                    scratch, nPrevBase >> 1, kStates,
                    0.0, 0.0, scratch, tmpNpBase >> 1, kStates);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, 0.0, scratch, tmpNpBase >> 1, kStates,
                        fr.predictedStateCov, predCovOff >> 1, kStates,
                    0.0, 0.0, result.smoothedStateCov, smoothedStateCovOff >> 1, kStates);
                ZLAS.zaxpy(ksKsC, 1.0, 0.0, fr.predictedStateCov, predCovOff >> 1, 1,
                    result.smoothedStateCov, smoothedStateCovOff >> 1, 1);
                for (int i = 0; i < kStates; i++) {
                    for (int j = i + 1; j < kStates; j++) {
                    int ij = smoothedStateCovOff + i * 2 * kStates + j * 2;
                    int ji = smoothedStateCovOff + j * 2 * kStates + i * 2;
                        double avgRe = (result.smoothedStateCov[ij] + result.smoothedStateCov[ji]) * 0.5;
                        double avgIm = (result.smoothedStateCov[ij + 1] - result.smoothedStateCov[ji + 1]) * 0.5;
                        result.smoothedStateCov[ij] = avgRe;
                        result.smoothedStateCov[ij + 1] = avgIm;
                        result.smoothedStateCov[ji] = avgRe;
                        result.smoothedStateCov[ji + 1] = -avgIm;
                    }
                }
            }

            if (storeAuxiliary) {
                ZLAS.zcopy(kStates * kStates, scratch, lBase, 1,
                    result.innovationsTransition, innovationsTransitionOff, 1);
                ZLAS.zcopy(kStates, scratch, rPrevBase, 1,
                    result.scaledSmoothedEstimator, scaledSmoothedEstimatorOff, 1);
                ZLAS.zcopy(kStates * kStates, scratch, nPrevBase, 1,
                    result.scaledSmoothedEstimatorCov, scaledSmoothedEstimatorCovOff, 1);
            }

            if (doDisturbance) {
                if (ke < kEndog) {
                    copyObservedVector(
                        smoothingError, smoothingErrorOff,
                        scratch, observedBase, ke,
                        scratch, tmpFivBase);
                    ZLAS.zgemv(BLAS.Trans.NoTrans, ke, ke, 1.0, 0.0,
                            H, hOff, hLd,
                            scratch, tmpFivBase, 1,
                            0.0, 0.0, scratch, tmpFiZBase, 1);
                    scatterObservedVector(
                        scratch, tmpFiZBase,
                        scratch, observedBase, ke,
                        result.smoothedObsDisturbance, smoothedObsDisturbanceOff, kEndog);
                } else {
                    ZLAS.zgemv(BLAS.Trans.NoTrans, ke, ke, 1.0, 0.0,
                            H, hOff, hLd,
                            smoothingError, storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase, 1,
                            0.0, 0.0, result.smoothedObsDisturbance, smoothedObsDisturbanceOff, 1);
                }

                if (kPosdef > 0) {
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                            0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                    ZLAS.zgemv(BLAS.Trans.Conj, kStates, kPosdef, 1.0, 0.0,
                            scratch, tmpRqBase, kPosdef,
                            scratch, rBase, 1,
                            0.0, 0.0, result.smoothedStateDisturbance, smoothedStateDisturbanceOff, 1);
                }
            }

            if (doDisturbanceCov) {
                double FtRe = ke == 1 ? scratch[tmpFcopyBase] : 0.0;
                double FtIm = ke == 1 ? scratch[tmpFcopyBase + 1] : 0.0;
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, ke,
                        1.0, 0.0, K, kOff >> 1, kLd,
                        H, hOff >> 1, hLd,
                        0.0, 0.0, scratch, tmpKhBase >> 1, ke);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                        1.0, 0.0, scratch, nBase >> 1, kStates,
                        scratch, tmpKhBase >> 1, ke,
                        0.0, 0.0, scratch, tmpNpBase >> 1, ke);

                if (ke >= kEndog) {
                    ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ke, ke, kStates,
                        -1.0, 0.0, scratch, tmpKhBase >> 1, ke,
                        scratch, tmpNpBase >> 1, ke,
                        0.0, 0.0, result.smoothedObsDisturbanceCov, smoothedObsDisturbanceCovOff >> 1, kEndog);
                    ZLAS.zaxpy(ke * ke, 1.0, 0.0, H, hOff >> 1, 1,
                            result.smoothedObsDisturbanceCov, smoothedObsDisturbanceCovOff >> 1, 1);
                }

                if (ke == 1) {
                    double FtMag2 = FtRe * FtRe + FtIm * FtIm;
                    double FtInvRe = FtRe / FtMag2;
                    double FtInvIm = -FtIm / FtMag2;
                    double hRe = H[hOff];
                    double hIm = H[hOff + 1];
                    double hfiRe = hRe * FtInvRe - hIm * FtInvIm;
                    double hfiIm = hRe * FtInvIm + hIm * FtInvRe;
                    double hfhRe = hfiRe * hRe - hfiIm * hIm;
                    double hfhIm = hfiRe * hIm + hfiIm * hRe;
                    if (ke < kEndog) {
                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ke, ke, kStates,
                                -1.0, 0.0, scratch, tmpKhBase >> 1, ke,
                                scratch, tmpNpBase >> 1, ke,
                                0.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                        ZLAS.zaxpy(ke * ke, 1.0, 0.0, H, hOff >> 1, 1,
                            scratch, tmpFcopyBase >> 1, 1);
                        scratch[tmpFcopyBase] -= hfhRe;
                        scratch[tmpFcopyBase + 1] -= hfhIm;
                    } else {
                        result.smoothedObsDisturbanceCov[smoothedObsDisturbanceCovOff] -= hfhRe;
                        result.smoothedObsDisturbanceCov[smoothedObsDisturbanceCovOff + 1] -= hfhIm;
                    }
                } else {
                    ZLAS.zcopy(ke * ke, H, hOff, 1, scratch, tmpFiHBase, 1);
                    ZLAS.zpotrs(BLAS.Uplo.Lower, ke, ke,
                            scratch, tmpFcopyBase >> 1, ke, scratch, tmpFiHBase >> 1, ke);
                    if (ke < kEndog) {
                    ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, ke, ke, kStates,
                        -1.0, 0.0, scratch, tmpKhBase >> 1, ke,
                        scratch, tmpNpBase >> 1, ke,
                        0.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                    ZLAS.zaxpy(ke * ke, 1.0, 0.0, H, hOff >> 1, 1,
                        scratch, tmpFcopyBase >> 1, 1);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                                -1.0, 0.0, H, hOff >> 1, hLd,
                                scratch, tmpFiHBase >> 1, ke,
                                1.0, 0.0, scratch, tmpFcopyBase >> 1, ke);
                    } else {
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                                -1.0, 0.0, H, hOff >> 1, hLd,
                                scratch, tmpFiHBase >> 1, ke,
                                1.0, 0.0, result.smoothedObsDisturbanceCov, smoothedObsDisturbanceCovOff >> 1, kEndog);
                    }
                }

                if (ke < kEndog) {
                    ZLAS.zcopy(kEndog * kEndog, obsCov, observationModel.obsCovOffset(t), 1,
                        result.smoothedObsDisturbanceCov, smoothedObsDisturbanceCovOff, 1);
                    scatterObservedSquareMatrix(
                        scratch, tmpFcopyBase, ke,
                        scratch, observedBase, ke,
                        result.smoothedObsDisturbanceCov, smoothedObsDisturbanceCovOff, kEndog);
                    restoreMissingSquareEntries(observationModel, t,
                        obsCov, observationModel.obsCovOffset(t), obsCovLd,
                        result.smoothedObsDisturbanceCov, smoothedObsDisturbanceCovOff, kEndog);
                }

                if (kPosdef > 0) {
                    if (!doDisturbance) {
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                                stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                                0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                    }
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                            1.0, 0.0, scratch, nBase >> 1, kStates,
                            scratch, tmpRqBase >> 1, kPosdef,
                            0.0, 0.0, scratch, tmpNpBase >> 1, kPosdef);
                    ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                            -1.0, 0.0, scratch, tmpRqBase >> 1, kPosdef,
                            scratch, tmpNpBase >> 1, kPosdef,
                            0.0, 0.0, result.smoothedStateDisturbanceCov, smoothedStateDisturbanceCovOff >> 1, kPosdef);
                            ZLAS.zaxpy(kPosdef * kPosdef, 1.0, 0.0, stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, 1,
                            result.smoothedStateDisturbanceCov, smoothedStateDisturbanceCovOff >> 1, 1);
                }
            }

            int tmpRBase = rBase;
            rBase = rPrevBase;
            rPrevBase = tmpRBase;
            int tmpNBase = nBase;
            nBase = nPrevBase;
            nPrevBase = tmpNBase;
        }

        if (storeAuxiliary) {
            ZLAS.zcopy(kStates, scratch, rBase, 1,
                result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(nobs), 1);
            ZLAS.zcopy(kStates * kStates, scratch, nBase, 1,
                result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(nobs), 1);

            shiftLeft(result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorBase(), 2 * kStates, nobs * 2 * kStates);
            shiftLeft(result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovBase(), ksKs, nobs * ksKs);
        }

        return result.trimOutputMask(smootherOutput);
    }

    private static ZSmootherResult smoothDiffuse(StateSpaceModel stateSpace,
                                                 StateSpaceModel observationModel,
                                                 ZFilterResult fr,
                                                 Pool pool,
                                                 int smootherOutput,
                                                 boolean reuseResult,
                                                 ZSmootherResult target) {
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int ksKs = kStates * 2 * kStates;
        int nobsDiffuse = fr.nobsDiffuse;
        double toleranceDiffuse = 1e-10;
        double[] transition = stateSpace.transitionData();
        int transitionLd = stateSpace.transitionLeadingDimension();
        double[] selection = stateSpace.selectionData();
        int selectionLd = stateSpace.selectionLeadingDimension();
        double[] stateCovariance = stateSpace.stateCovarianceData();
        int stateCovarianceLd = stateSpace.stateCovarianceLeadingDimension();
        double[] design = observationModel.designData();
        int designLd = observationModel.designLeadingDimension();
        double[] obsCov = observationModel.obsCovData();
        int obsCovLd = observationModel.obsCovLeadingDimension();

        boolean doState = (smootherOutput & SMOOTHER_STATE) != 0;
        boolean doStateCov = (smootherOutput & SMOOTHER_STATE_COV) != 0;
        boolean doDisturbance = (smootherOutput & SMOOTHER_DISTURBANCE) != 0;
        boolean doDisturbanceCov = (smootherOutput & SMOOTHER_DISTURBANCE_COV) != 0;
        boolean storeAuxiliary = smootherOutput == SMOOTHER_ALL;

        SmootherResultLayout retainedLayout = SmootherResultLayout.create(
            2,
            kEndog,
            kStates,
            kPosdef,
            nobs,
            doState,
            doStateCov,
            doDisturbance,
            doDisturbanceCov,
            storeAuxiliary);
        ZSmootherResult result = target != null
            ? target
            : (reuseResult
            ? pool.borrowDiffuseResult(kEndog, kStates, kPosdef, nobs, retainedLayout)
            : new ZSmootherResult(kEndog, kStates, kPosdef, nobs, retainedLayout));
        result.allocateDiffuse();

        ScratchArena scratchArena = pool.scratch;
        ZSmootherScratchLayout scratchLayout = scratchArena.scratchLayout;
        double[] scratch = scratchArena.scratchBacking;
        int rBase = scratchLayout.rBase();
        int nBase = scratchLayout.nBase();
        int rPrevBase = scratchLayout.rPrevBase();
        int nPrevBase = scratchLayout.nPrevBase();
        int lBase = scratchLayout.lBase();
        int tmpNpBase = scratchLayout.tmpNpBase();
        int tmpRqBase = scratchLayout.tmpRqBase();
        int rInfBase = scratchLayout.rInfBase();
        int rInfPrevBase = scratchLayout.rInfPrevBase();
        int nInf1Base = scratchLayout.nInf1Base();
        int nInf1PrevBase = scratchLayout.nInf1PrevBase();
        int nInf2Base = scratchLayout.nInf2Base();
        int nInf2PrevBase = scratchLayout.nInf2PrevBase();
        int l0Base = scratchLayout.l0Base();
        int l1Base = scratchLayout.l1Base();
        int k0Base = scratchLayout.k0Base();
        int k1Base = scratchLayout.k1Base();
        int tmpMInfBase = scratchLayout.tmpMInfBase();
        int tmpMStarBase = scratchLayout.tmpMStarBase();
        int tmpQuadBase = scratchLayout.tmpQuadBase();
        int tmpTtNBase = scratchLayout.tmpTtNBase();
        int tmpTtNInf1Base = scratchLayout.tmpTtNInf1Base();
        int tmpTtNInf2Base = scratchLayout.tmpTtNInf2Base();
        int tmpPNInf1Base = scratchLayout.tmpPNInf1Base();
        int tmpPInfNInf2Base = scratchLayout.tmpPInfNInf2Base();

        zero(scratch, rBase, 2 * kStates);
        zero(scratch, nBase, ksKs);
        zero(scratch, rInfBase, 2 * kStates);
        zero(scratch, nInf1Base, ksKs);
        zero(scratch, nInf2Base, ksKs);

        for (int t = nobs - 1; t >= 0; t--) {
            int predOff = fr.predictedStateOffset(t);
            int predCovOff = fr.predictedStateCovOffset(t);
            int predDiffCovOff = fr.predictedDiffuseStateCovOffset(t);
            int foreErrOff = fr.forecastErrorOffset(t);
            int foreErrCovOff = fr.forecastErrorCovOffset(t);
            int foreErrDiffCovOff = fr.forecastErrorDiffuseCovOffset(t);
            int smoothedObsDisturbanceOff = result.smoothedObsDisturbanceOffset(t);
            int smoothedObsDisturbanceCovOff = result.smoothedObsDisturbanceCovOffset(t);
            int smoothedStateDisturbanceOff = result.smoothedStateDisturbanceOffset(t);
            int smoothedStateDisturbanceCovOff = result.smoothedStateDisturbanceCovOffset(t);
            int scaledSmoothedEstimatorNextOff = result.scaledSmoothedEstimatorOffset(t + 1);
            int scaledSmoothedEstimatorCovNextOff = result.scaledSmoothedEstimatorCovOffset(t + 1);
            int scaledSmoothedDiffuseEstimatorNextOff = result.scaledSmoothedDiffuseEstimatorOffset(t + 1);
            int scaledSmoothedDiffuse1NextOff = result.scaledSmoothedDiffuse1EstimatorCovOffset(t + 1);
            int scaledSmoothedDiffuse2NextOff = result.scaledSmoothedDiffuse2EstimatorCovOffset(t + 1);
            int smoothedStateOff = result.smoothedStateOffset(t);
            int smoothedStateCovOff = result.smoothedStateCovOffset(t);
            boolean isDiffuse = t < nobsDiffuse;

            ZLAS.zcopy(kStates, scratch, rBase, 1, scratch, rPrevBase, 1);
            ZLAS.zcopy(kStates * kStates, scratch, nBase, 1, scratch, nPrevBase, 1);
            if (isDiffuse) {
                ZLAS.zcopy(kStates, scratch, rInfBase, 1, scratch, rInfPrevBase, 1);
                ZLAS.zcopy(kStates * kStates, scratch, nInf1Base, 1, scratch, nInf1PrevBase, 1);
                ZLAS.zcopy(kStates * kStates, scratch, nInf2Base, 1, scratch, nInf2PrevBase, 1);
            }

            if (doDisturbance) {
                zero(result.smoothedObsDisturbance, smoothedObsDisturbanceOff, 2 * kEndog);
            }
            if (doDisturbanceCov) {
                ZLAS.zcopy(kEndog * kEndog, obsCov, observationModel.obsCovOffset(t), 1,
                        result.smoothedObsDisturbanceCov, smoothedObsDisturbanceCovOff, 1);
            }

            if ((doDisturbance || doDisturbanceCov) && kPosdef > 0) {
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                        1.0, 0.0, selection, stateSpace.selectionOffset(t) >> 1, selectionLd,
                        stateCovariance, stateSpace.stateCovarianceOffset(t) >> 1, stateCovarianceLd,
                        0.0, 0.0, scratch, tmpRqBase >> 1, kPosdef);
                if (doDisturbance) {
                ZLAS.zgemv(BLAS.Trans.Conj, kStates, kPosdef, 1.0, 0.0,
                        scratch, tmpRqBase, kPosdef,
                        scratch, rBase, 1,
                        0.0, 0.0, result.smoothedStateDisturbance, smoothedStateDisturbanceOff, 1);
                }
                if (doDisturbanceCov) {
                ZLAS.zcopy(kPosdef * kPosdef, stateCovariance, stateSpace.stateCovarianceOffset(t), 1,
                        result.smoothedStateDisturbanceCov, smoothedStateDisturbanceCovOff, 1);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                        1.0, 0.0, scratch, nBase >> 1, kStates,
                        scratch, tmpRqBase >> 1, kPosdef,
                    0.0, 0.0, scratch, tmpPNInf1Base >> 1, kPosdef);
                ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                        -1.0, 0.0, scratch, tmpRqBase >> 1, kPosdef,
                    scratch, tmpPNInf1Base >> 1, kPosdef,
                        1.0, 0.0, result.smoothedStateDisturbanceCov, smoothedStateDisturbanceCovOff >> 1, kPosdef);
                }
            }

            if (observationModel.missingCount(t) == kEndog) {
                ZLAS.zcopy(kStates * kStates, transition, stateSpace.transitionOffset(t), 1, scratch, l0Base, 1);
            } else {
                for (int i = kEndog - 1; i >= 0; i--) {
                    if (observationModel.isMissing(i, t)) {
                        continue;
                    }

                    int zRowOff = observationModel.designOffset(t) + i * 2 * designLd;
                    int hDiagOff = observationModel.obsCovOffset(t) + i * 2 * obsCovLd + i * 2;
                    int vOff = foreErrOff + i * 2;
                    double viRe = fr.forecastError[vOff];
                    double viIm = fr.forecastError[vOff + 1];
                    double hii = obsCov[hDiagOff];

                        computeM(fr.predictedStateCov, predCovOff, design, zRowOff, scratch, tmpMStarBase, kStates);
                    double fStar = fr.forecastErrorCov[foreErrCovOff + i * 2 * kEndog + i * 2];
                    if (Double.isNaN(fStar) || fStar == 0.0) {
                        fStar = fr.forecastErrorCov[fr.forecastErrorCovOffset(t) + i * 2 * kEndog + i * 2];
                    }

                    double fInf = 0.0;
                    if (isDiffuse) {
                        computeM(fr.predictedDiffuseStateCov, predDiffCovOff, design, zRowOff, scratch, tmpMInfBase, kStates);
                        fInf = fr.forecastErrorDiffuseCov[foreErrDiffCovOff + i * 2 * kEndog + i * 2];
                    }

                    if (isDiffuse && fInf > toleranceDiffuse) {
                        double f1 = 1.0 / fInf;
                        double f2 = -fStar * f1 * f1;

                        for (int j = 0; j < kStates; j++) {
                            int off = j * 2;
                            scratch[k0Base + off] = scratch[tmpMInfBase + off] * f1;
                            scratch[k0Base + off + 1] = scratch[tmpMInfBase + off + 1] * f1;
                            scratch[k1Base + off] = scratch[tmpMStarBase + off] * f1 + scratch[tmpMInfBase + off] * f2;
                            scratch[k1Base + off + 1] = scratch[tmpMStarBase + off + 1] * f1 + scratch[tmpMInfBase + off + 1] * f2;
                        }

                        setIdentity(scratch, l0Base, kStates);
                        zero(scratch, l1Base, ksKs);
                        for (int row = 0; row < kStates; row++) {
                            int k0Off = row * 2;
                            int k1Off = row * 2;
                            for (int col = 0; col < kStates; col++) {
                                int zOff = zRowOff + col * 2;
                                int lOff = row * 2 * kStates + col * 2;
                            subtractProduct(scratch, l0Base + lOff,
                                scratch[k0Base + k0Off], scratch[k0Base + k0Off + 1],
                                design[zOff], design[zOff + 1]);
                            subtractProduct(scratch, l1Base + lOff,
                                scratch[k1Base + k1Off], scratch[k1Base + k1Off + 1],
                                design[zOff], design[zOff + 1]);
                            }
                        }

                        if (doDisturbance && hii != 0.0) {
                            double dot = dotHermitian(scratch, k0Base, scratch, rPrevBase, kStates);
                            result.smoothedObsDisturbance[smoothedObsDisturbanceOff + i * 2] = -hii * dot;
                        }
                        if (doDisturbanceCov) {
                            double quad = quadraticForm(scratch, nPrevBase, scratch, k0Base, scratch, tmpQuadBase, kStates);
                            result.smoothedObsDisturbanceCov[smoothedObsDisturbanceCovOff + i * 2 * kEndog + i * 2] = hii * (1.0 - hii * quad);
                        }

                        ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                            scratch, l0Base, kStates,
                            scratch, rInfPrevBase, 1,
                            0.0, 0.0, scratch, rInfBase, 1);
                        addConjScaledRow(scratch, rInfBase, design, zRowOff, viRe * f1, viIm * f1, kStates);
                        ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                            scratch, l1Base, kStates,
                            scratch, rPrevBase, 1,
                            0.0, 0.0, scratch, tmpMStarBase, 1);
                        ZLAS.zaxpy(kStates, 1.0, 0.0, scratch, tmpMStarBase >> 1, 1, scratch, rInfBase >> 1, 1);

                        ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                            scratch, l0Base, kStates,
                                scratch, rPrevBase, 1,
                                0.0, 0.0, scratch, rBase, 1);

                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, l0Base >> 1, kStates,
                            scratch, nInf2PrevBase >> 1, kStates,
                            0.0, 0.0, scratch, tmpTtNInf2Base >> 1, kStates);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpTtNInf2Base >> 1, kStates,
                            scratch, l0Base >> 1, kStates,
                            0.0, 0.0, scratch, nInf2Base >> 1, kStates);

                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, l0Base >> 1, kStates,
                            scratch, nInf1PrevBase >> 1, kStates,
                            0.0, 0.0, scratch, tmpTtNInf2Base >> 1, kStates);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpTtNInf2Base >> 1, kStates,
                            scratch, l1Base >> 1, kStates,
                            0.0, 0.0, scratch, tmpPInfNInf2Base >> 1, kStates);
                        addWithConjTranspose(scratch, tmpPInfNInf2Base, scratch, nInf2Base, kStates);

                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, l1Base >> 1, kStates,
                                scratch, nPrevBase >> 1, kStates,
                            0.0, 0.0, scratch, tmpTtNInf2Base >> 1, kStates);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpTtNInf2Base >> 1, kStates,
                            scratch, l1Base >> 1, kStates,
                            0.0, 0.0, scratch, tmpPInfNInf2Base >> 1, kStates);
                        ZLAS.zaxpy(kStates * kStates, 1.0, 0.0, scratch, tmpPInfNInf2Base >> 1, 1, scratch, nInf2Base >> 1, 1);
                        addOuterConjZ(scratch, nInf2Base, design, zRowOff, f2,
                            scratch, tmpMStarBase, kStates);

                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, l0Base >> 1, kStates,
                            scratch, nInf1PrevBase >> 1, kStates,
                            0.0, 0.0, scratch, tmpTtNInf1Base >> 1, kStates);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpTtNInf1Base >> 1, kStates,
                            scratch, l0Base >> 1, kStates,
                            0.0, 0.0, scratch, nInf1Base >> 1, kStates);
                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, l1Base >> 1, kStates,
                                scratch, nPrevBase >> 1, kStates,
                            0.0, 0.0, scratch, tmpTtNInf1Base >> 1, kStates);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpTtNInf1Base >> 1, kStates,
                            scratch, l0Base >> 1, kStates,
                            0.0, 0.0, scratch, tmpPInfNInf2Base >> 1, kStates);
                        ZLAS.zaxpy(kStates * kStates, 1.0, 0.0, scratch, tmpPInfNInf2Base >> 1, 1, scratch, nInf1Base >> 1, 1);
                        addOuterConjZ(scratch, nInf1Base, design, zRowOff, f1,
                            scratch, tmpMStarBase, kStates);

                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, l0Base >> 1, kStates,
                                scratch, nPrevBase >> 1, kStates,
                            0.0, 0.0, scratch, tmpTtNBase >> 1, kStates);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpTtNBase >> 1, kStates,
                            scratch, l0Base >> 1, kStates,
                                0.0, 0.0, scratch, nBase >> 1, kStates);

                        int tmpRBase = rBase;
                        rBase = rPrevBase;
                        rPrevBase = tmpRBase;
                        int tmpNBase = nBase;
                        nBase = nPrevBase;
                        nPrevBase = tmpNBase;
                        int tmpRInfBase = rInfPrevBase;
                        rInfPrevBase = rInfBase;
                        rInfBase = tmpRInfBase;
                        int tmpNInf1Base = nInf1PrevBase;
                        nInf1PrevBase = nInf1Base;
                        nInf1Base = tmpNInf1Base;
                        int tmpNInf2Base = nInf2PrevBase;
                        nInf2PrevBase = nInf2Base;
                        nInf2Base = tmpNInf2Base;
                    } else if (fStar > 0.0) {
                        double fStarInv = 1.0 / fStar;

                        for (int j = 0; j < kStates; j++) {
                            int off = j * 2;
                            scratch[k0Base + off] = scratch[tmpMStarBase + off] * fStarInv;
                            scratch[k0Base + off + 1] = scratch[tmpMStarBase + off + 1] * fStarInv;
                        }

                        setIdentity(scratch, l0Base, kStates);
                        for (int row = 0; row < kStates; row++) {
                            int k0Off = row * 2;
                            for (int col = 0; col < kStates; col++) {
                                int zOff = zRowOff + col * 2;
                                int lOff = row * 2 * kStates + col * 2;
                                subtractProduct(scratch, l0Base + lOff,
                                        scratch[k0Base + k0Off], scratch[k0Base + k0Off + 1],
                                        design[zOff], design[zOff + 1]);
                            }
                        }

                        if (doDisturbance && hii != 0.0) {
                            double dot = dotHermitian(scratch, k0Base, scratch, rPrevBase, kStates);
                            result.smoothedObsDisturbance[smoothedObsDisturbanceOff + i * 2] = hii * (viRe * fStarInv - dot);
                        }
                        if (doDisturbanceCov) {
                            double quad = quadraticForm(scratch, nPrevBase, scratch, k0Base, scratch, tmpQuadBase, kStates);
                            result.smoothedObsDisturbanceCov[smoothedObsDisturbanceCovOff + i * 2 * kEndog + i * 2] =
                                    hii * (1.0 - hii * (fStarInv + quad));
                        }

                        ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                                scratch, l0Base, kStates,
                                scratch, rPrevBase, 1,
                                0.0, 0.0, scratch, rBase, 1);
                        addConjScaledRow(scratch, rBase, design, zRowOff, viRe * fStarInv, viIm * fStarInv, kStates);

                        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, 0.0, scratch, l0Base >> 1, kStates,
                                scratch, nPrevBase >> 1, kStates,
                                0.0, 0.0, scratch, tmpTtNBase >> 1, kStates);
                        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, 0.0, scratch, tmpTtNBase >> 1, kStates,
                                scratch, l0Base >> 1, kStates,
                                0.0, 0.0, scratch, nBase >> 1, kStates);
                        addOuterConjZ(scratch, nBase, design, zRowOff, fStarInv,
                            scratch, tmpMStarBase, kStates);

                        if (isDiffuse) {
                            ZLAS.zcopy(kStates, scratch, rInfPrevBase, 1, scratch, rInfBase, 1);
                            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                    1.0, 0.0, scratch, nInf1PrevBase >> 1, kStates,
                                    scratch, l0Base >> 1, kStates,
                                    0.0, 0.0, scratch, nInf1Base >> 1, kStates);
                            ZLAS.zcopy(kStates * kStates, scratch, nInf2PrevBase, 1, scratch, nInf2Base, 1);
                        }

                        int tmpRBase = rBase;
                        rBase = rPrevBase;
                        rPrevBase = tmpRBase;
                        int tmpNBase = nBase;
                        nBase = nPrevBase;
                        nPrevBase = tmpNBase;
                        if (isDiffuse) {
                            int tmpRInfBase = rInfPrevBase;
                            rInfPrevBase = rInfBase;
                            rInfBase = tmpRInfBase;
                            int tmpNInf1Base = nInf1PrevBase;
                            nInf1PrevBase = nInf1Base;
                            nInf1Base = tmpNInf1Base;
                            int tmpNInf2Base = nInf2PrevBase;
                            nInf2PrevBase = nInf2Base;
                            nInf2Base = tmpNInf2Base;
                        }
                    }
                }

                ZLAS.zcopy(kStates * kStates, transition, stateSpace.transitionOffset(t), 1, scratch, l0Base, 1);
            }

            if (storeAuxiliary) {
                ZLAS.zcopy(kStates, scratch, rPrevBase, 1,
                    result.scaledSmoothedEstimator, scaledSmoothedEstimatorNextOff, 1);
                ZLAS.zcopy(kStates * kStates, scratch, nPrevBase, 1,
                    result.scaledSmoothedEstimatorCov, scaledSmoothedEstimatorCovNextOff, 1);
            }
            if (isDiffuse) {
                ZLAS.zcopy(kStates, scratch, rInfPrevBase, 1,
                        result.scaledSmoothedDiffuseEstimator, scaledSmoothedDiffuseEstimatorNextOff, 1);
                ZLAS.zcopy(kStates * kStates, scratch, nInf1PrevBase, 1,
                        result.scaledSmoothedDiffuse1EstimatorCov, scaledSmoothedDiffuse1NextOff, 1);
                ZLAS.zcopy(kStates * kStates, scratch, nInf2PrevBase, 1,
                        result.scaledSmoothedDiffuse2EstimatorCov, scaledSmoothedDiffuse2NextOff, 1);
            }

            if (doState) {
                ZLAS.zcopy(kStates, fr.predictedState, predOff, 1,
                        result.smoothedState, smoothedStateOff, 1);
                ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                        fr.predictedStateCov, predCovOff, kStates,
                        scratch, rPrevBase, 1,
                        1.0, 0.0, result.smoothedState, smoothedStateOff, 1);
                if (isDiffuse) {
                    ZLAS.zgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0, 0.0,
                            fr.predictedDiffuseStateCov, predDiffCovOff, kStates,
                            scratch, rInfPrevBase, 1,
                            1.0, 0.0, result.smoothedState, smoothedStateOff, 1);
                }

                if (isDiffuse && doDisturbance && t + 1 < nobs && kPosdef == kStates) {
                    boolean selectionIdentity = true;
                    int selOff = stateSpace.selectionOffset(t);
                    for (int row = 0; row < kStates && selectionIdentity; row++) {
                        for (int col = 0; col < kStates; col++) {
                            int off = selOff + row * 2 * selectionLd + col * 2;
                            double expectedRe = row == col ? 1.0 : 0.0;
                            if (Math.abs(selection[off] - expectedRe) > 1e-12
                                    || Math.abs(selection[off + 1]) > 1e-12) {
                                selectionIdentity = false;
                                break;
                            }
                        }
                    }
                    if (selectionIdentity) {
                        int transOff = stateSpace.transitionOffset(t);
                        int nextSmoothedStateOff = result.smoothedStateOffset(t + 1);
                        for (int row = 0; row < kStates; row++) {
                            int etaOff = smoothedStateDisturbanceOff + row * 2;
                            int nextStateOff = nextSmoothedStateOff + row * 2;
                            double etaRe = result.smoothedState[nextStateOff];
                            double etaIm = result.smoothedState[nextStateOff + 1];
                            for (int col = 0; col < kStates; col++) {
                                int tOff = transOff + row * 2 * transitionLd + col * 2;
                                int sOff = smoothedStateOff + col * 2;
                                double tr = transition[tOff];
                                double ti = transition[tOff + 1];
                                double sr = result.smoothedState[sOff];
                                double si = result.smoothedState[sOff + 1];
                                etaRe -= tr * sr - ti * si;
                                etaIm -= tr * si + ti * sr;
                            }
                            result.smoothedStateDisturbance[etaOff] = etaRe;
                            result.smoothedStateDisturbance[etaOff + 1] = etaIm;
                        }
                    }
                }
            }

            if (doStateCov) {
                ZLAS.zcopy(kStates * kStates, fr.predictedStateCov, predCovOff, 1,
                        result.smoothedStateCov, smoothedStateCovOff, 1);

                ZLAS.zcopy(kStates * kStates, fr.predictedStateCov, predCovOff, 1, scratch, tmpNpBase, 1);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, scratch, tmpNpBase >> 1, kStates,
                        scratch, nPrevBase >> 1, kStates,
                    0.0, 0.0, scratch, tmpTtNBase >> 1, kStates);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, 0.0, scratch, tmpTtNBase >> 1, kStates,
                        scratch, tmpNpBase >> 1, kStates,
                    0.0, 0.0, scratch, tmpPNInf1Base >> 1, kStates);
                ZLAS.zaxpy(kStates * kStates, -1.0, 0.0, scratch, tmpPNInf1Base >> 1, 1,
                        result.smoothedStateCov, smoothedStateCovOff >> 1, 1);

                if (isDiffuse) {
                    ZLAS.zcopy(kStates * kStates, fr.predictedDiffuseStateCov, predDiffCovOff, 1,
                        scratch, tmpPInfNInf2Base, 1);

                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, scratch, tmpPInfNInf2Base >> 1, kStates,
                        scratch, nInf1PrevBase >> 1, kStates,
                        0.0, 0.0, scratch, tmpTtNBase >> 1, kStates);
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, scratch, tmpTtNBase >> 1, kStates,
                            scratch, tmpNpBase >> 1, kStates,
                        0.0, 0.0, scratch, tmpPNInf1Base >> 1, kStates);
                    ZLAS.zaxpy(kStates * kStates, -1.0, 0.0, scratch, tmpPNInf1Base >> 1, 1,
                            result.smoothedStateCov, smoothedStateCovOff >> 1, 1);

                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
                            1.0, 0.0, scratch, tmpNpBase >> 1, kStates,
                        scratch, nInf1PrevBase >> 1, kStates,
                        0.0, 0.0, scratch, tmpTtNBase >> 1, kStates);
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, scratch, tmpTtNBase >> 1, kStates,
                        scratch, tmpPInfNInf2Base >> 1, kStates,
                        0.0, 0.0, scratch, tmpPNInf1Base >> 1, kStates);
                    ZLAS.zaxpy(kStates * kStates, -1.0, 0.0, scratch, tmpPNInf1Base >> 1, 1,
                            result.smoothedStateCov, smoothedStateCovOff >> 1, 1);

                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, scratch, tmpPInfNInf2Base >> 1, kStates,
                        scratch, nInf2PrevBase >> 1, kStates,
                        0.0, 0.0, scratch, tmpTtNBase >> 1, kStates);
                    ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, scratch, tmpTtNBase >> 1, kStates,
                        scratch, tmpPInfNInf2Base >> 1, kStates,
                        0.0, 0.0, scratch, tmpPNInf1Base >> 1, kStates);
                    ZLAS.zaxpy(kStates * kStates, -1.0, 0.0, scratch, tmpPNInf1Base >> 1, 1,
                            result.smoothedStateCov, smoothedStateCovOff >> 1, 1);
                }

                symmetrizeHermitian(result.smoothedStateCov, smoothedStateCovOff, kStates);
            }

            ZLAS.zcopy(kStates * kStates, transition, stateSpace.transitionOffset(t), 1, scratch, lBase, 1);

            ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                scratch, lBase, kStates,
                scratch, rPrevBase, 1,
                0.0, 0.0, scratch, rBase, 1);
            ZLAS.zcopy(kStates, scratch, rBase, 1, scratch, rPrevBase, 1);

            ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, 0.0, scratch, lBase >> 1, kStates,
                scratch, nPrevBase >> 1, kStates,
                0.0, 0.0, scratch, tmpTtNBase >> 1, kStates);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, 0.0, scratch, tmpTtNBase >> 1, kStates,
                scratch, lBase >> 1, kStates,
                0.0, 0.0, scratch, nBase >> 1, kStates);
            ZLAS.zcopy(kStates * kStates, scratch, nBase, 1, scratch, nPrevBase, 1);

            if (isDiffuse) {
                ZLAS.zgemv(BLAS.Trans.Conj, kStates, kStates, 1.0, 0.0,
                        scratch, lBase, kStates,
                    scratch, rInfPrevBase, 1,
                    0.0, 0.0, scratch, rInfBase, 1);
                ZLAS.zcopy(kStates, scratch, rInfBase, 1, scratch, rInfPrevBase, 1);

                ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, scratch, lBase >> 1, kStates,
                    scratch, nInf1PrevBase >> 1, kStates,
                    0.0, 0.0, scratch, tmpTtNInf1Base >> 1, kStates);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, 0.0, scratch, tmpTtNInf1Base >> 1, kStates,
                        scratch, lBase >> 1, kStates,
                    0.0, 0.0, scratch, nInf1Base >> 1, kStates);
                ZLAS.zcopy(kStates * kStates, scratch, nInf1Base, 1, scratch, nInf1PrevBase, 1);

                ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, 0.0, scratch, lBase >> 1, kStates,
                    scratch, nInf2PrevBase >> 1, kStates,
                    0.0, 0.0, scratch, tmpTtNInf2Base >> 1, kStates);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                    1.0, 0.0, scratch, tmpTtNInf2Base >> 1, kStates,
                        scratch, lBase >> 1, kStates,
                    0.0, 0.0, scratch, nInf2Base >> 1, kStates);
                ZLAS.zcopy(kStates * kStates, scratch, nInf2Base, 1, scratch, nInf2PrevBase, 1);
            }
        }

        if (storeAuxiliary) {
            shiftLeft(result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorBase(), 2 * kStates, nobs * 2 * kStates);
            shiftLeft(result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovBase(), ksKs, nobs * ksKs);
        }
        shiftLeft(result.scaledSmoothedDiffuseEstimator, result.scaledSmoothedDiffuseEstimatorBase(), 2 * kStates, nobs * 2 * kStates);
        shiftLeft(result.scaledSmoothedDiffuse1EstimatorCov, result.scaledSmoothedDiffuse1EstimatorCovBase(), ksKs, nobs * ksKs);
        shiftLeft(result.scaledSmoothedDiffuse2EstimatorCov, result.scaledSmoothedDiffuse2EstimatorCovBase(), ksKs, nobs * ksKs);

        return result.trimOutputMask(smootherOutput);
    }

    private static void requireFilterStorage(ZFilterResult fr, int smoothMethod) {
        FilterResultShape filterShape = fr.resultShape();
        boolean needsFilteredState = fr.nobsDiffuse == 0
            && !fr.perObsKalmanGain
            && (smoothMethod & (KalmanSmoother.SMOOTH_CLASSICAL | KalmanSmoother.SMOOTH_ALTERNATIVE)) != 0;
        boolean needsKalmanGain = fr.nobsDiffuse == 0;

        if (!filterShape.storesPredictedState()
                || !filterShape.storesForecast()
                || (needsKalmanGain && !filterShape.storesKalmanGain())
                || (needsFilteredState && !filterShape.storesFilteredState())) {
            throw new IllegalArgumentException("ZFilterResult does not retain the storage required for smoothing");
        }
        if (fr.nobsDiffuse > 0 && ((fr.predictedDiffuseStateCov == null || fr.predictedDiffuseStateCovLength() == 0)
                || (fr.forecastErrorDiffuseCov == null || fr.forecastErrorDiffuseCovLength() == 0))) {
            throw new IllegalArgumentException("Diffuse smoothing requires diffuse filter storage");
        }
    }

    private static void copyObservedRows(double[] source,
                                         int sourceOffset,
                                         int sourceLd,
                                         int rowWidth,
                                         double[] observedIndices,
                                         int observedOffset,
                                         int observedCount,
                                         double[] target,
                                         int targetOffset) {
        for (int targetRow = 0; targetRow < observedCount; targetRow++) {
            int sourceRow = (int) observedIndices[observedOffset + targetRow];
            System.arraycopy(source, sourceOffset + sourceRow * 2 * sourceLd,
                target, targetOffset + targetRow * 2 * rowWidth, 2 * rowWidth);
        }
    }

    private static void copyObservedColumns(int rowCount,
                                            double[] source,
                                            int sourceOffset,
                                            int sourceLd,
                                            double[] observedIndices,
                                            int observedOffset,
                                            int observedCount,
                                            double[] target,
                                            int targetOffset,
                                            int targetLd) {
        for (int row = 0; row < rowCount; row++) {
            int sourceRowOff = sourceOffset + row * 2 * sourceLd;
            int targetRowOff = targetOffset + row * 2 * targetLd;
            for (int targetCol = 0; targetCol < observedCount; targetCol++) {
                int sourceCol = (int) observedIndices[observedOffset + targetCol];
                int sourceOff = sourceRowOff + sourceCol * 2;
                int targetOff = targetRowOff + targetCol * 2;
                target[targetOff] = source[sourceOff];
                target[targetOff + 1] = source[sourceOff + 1];
            }
        }
    }

    private static void copyObservedVector(double[] source,
                                           int sourceOffset,
                                           double[] observedIndices,
                                           int observedOffset,
                                           int observedCount,
                                           double[] target,
                                           int targetOffset) {
        for (int targetIndex = 0; targetIndex < observedCount; targetIndex++) {
            int sourceIndex = (int) observedIndices[observedOffset + targetIndex];
            int sourceOff = sourceOffset + sourceIndex * 2;
            int targetOff = targetOffset + targetIndex * 2;
            target[targetOff] = source[sourceOff];
            target[targetOff + 1] = source[sourceOff + 1];
        }
    }

    private static void copyObservedSquareMatrix(double[] source,
                                                 int sourceOffset,
                                                 int sourceLd,
                                                 double[] observedIndices,
                                                 int observedOffset,
                                                 int observedCount,
                                                 double[] target,
                                                 int targetOffset,
                                                 int targetLd) {
        for (int targetRow = 0; targetRow < observedCount; targetRow++) {
            int sourceRow = (int) observedIndices[observedOffset + targetRow];
            int sourceRowOff = sourceOffset + sourceRow * 2 * sourceLd;
            int targetRowOff = targetOffset + targetRow * 2 * targetLd;
            for (int targetCol = 0; targetCol < observedCount; targetCol++) {
                int sourceCol = (int) observedIndices[observedOffset + targetCol];
                int sourceOff = sourceRowOff + sourceCol * 2;
                int targetOff = targetRowOff + targetCol * 2;
                target[targetOff] = source[sourceOff];
                target[targetOff + 1] = source[sourceOff + 1];
            }
        }
    }

    private static void scatterObservedVector(double[] source,
                                              int sourceOffset,
                                              double[] observedIndices,
                                              int observedOffset,
                                              int observedCount,
                                              double[] target,
                                              int targetOffset,
                                              int targetCount) {
        zero(target, targetOffset, 2 * targetCount);
        for (int sourceIndex = 0; sourceIndex < observedCount; sourceIndex++) {
            int targetIndex = (int) observedIndices[observedOffset + sourceIndex];
            int sourceOff = sourceOffset + sourceIndex * 2;
            int targetOff = targetOffset + targetIndex * 2;
            target[targetOff] = source[sourceOff];
            target[targetOff + 1] = source[sourceOff + 1];
        }
    }

    private static void scatterObservedSquareMatrix(double[] source,
                                                    int sourceOffset,
                                                    int sourceLd,
                                                    double[] observedIndices,
                                                    int observedOffset,
                                                    int observedCount,
                                                    double[] target,
                                                    int targetOffset,
                                                    int targetLd) {
        for (int sourceRow = 0; sourceRow < observedCount; sourceRow++) {
            int targetRow = (int) observedIndices[observedOffset + sourceRow];
            int targetRowOff = targetOffset + targetRow * 2 * targetLd;
            int sourceRowOff = sourceOffset + sourceRow * 2 * sourceLd;
            for (int sourceCol = 0; sourceCol < observedCount; sourceCol++) {
                int targetCol = (int) observedIndices[observedOffset + sourceCol];
                int targetOff = targetRowOff + targetCol * 2;
                int sourceOff = sourceRowOff + sourceCol * 2;
                target[targetOff] = source[sourceOff];
                target[targetOff + 1] = source[sourceOff + 1];
            }
        }
    }

    private static void restoreMissingSquareEntries(StateSpaceModel observationModel,
                                                    int t,
                                                    double[] source,
                                                    int sourceOffset,
                                                    int sourceLd,
                                                    double[] target,
                                                    int targetOffset,
                                                    int targetLd) {
        int kEndog = observationModel.observationDimension();
        for (int row = 0; row < kEndog; row++) {
            boolean missingRow = observationModel.isMissing(row, t);
            for (int col = 0; col < kEndog; col++) {
                if (!missingRow && !observationModel.isMissing(col, t)) {
                    continue;
                }
                int sourceOff = sourceOffset + row * 2 * sourceLd + col * 2;
                int targetOff = targetOffset + row * 2 * targetLd + col * 2;
                target[targetOff] = source[sourceOff];
                target[targetOff + 1] = source[sourceOff + 1];
            }
        }
    }

    private static int selectObservations(StateSpaceModel observationModel,
                                          ScratchArena scratchArena,
                                          ZSmootherScratchLayout layout,
                                          int t) {
        int kEndog = observationModel.observationDimension();
        int missing = observationModel.missingCount(t);
        if (missing == 0) {
            return kEndog;
        }
        if (missing == kEndog) {
            return 0;
        }

        int kStates = observationModel.stateCount();
        int ke = kEndog - missing;

        double[] scratch = scratchArena.scratchBacking;
        int observedOffset = observedIndexBase(layout.selectedObsCovBase(), ke);
        int observedCount = 0;
        for (int i = 0; i < kEndog; i++) {
            if (!observationModel.isMissing(i, t)) {
                scratch[observedOffset + observedCount] = i;
                observedCount++;
            }
        }
        double[] design = observationModel.designData();
        int designOff = observationModel.designOffset(t);
        int designLd = observationModel.designLeadingDimension();
        double[] obsCov = observationModel.obsCovData();
        int obsCovOff = observationModel.obsCovOffset(t);
        int obsCovLd = observationModel.obsCovLeadingDimension();
        copyObservedRows(
            design, designOff, designLd, kStates,
            scratch, observedOffset, ke,
            scratch, layout.selectedDesignBase());
        copyObservedSquareMatrix(
            obsCov, obsCovOff, obsCovLd,
            scratch, observedOffset, ke,
            scratch, layout.selectedObsCovBase(), ke);
        return ke;
    }

    private static int observedIndexBase(int selectedObsCovBase, int observedCount) {
        return selectedObsCovBase + 2 * observedCount * observedCount;
    }

        private static void computeM(double[] matrix, int matrixOff, double[] design, int zRowOff,
                     double[] target, int targetOffset, int kStates) {
        for (int row = 0; row < kStates; row++) {
            double sumRe = 0.0;
            double sumIm = 0.0;
            for (int col = 0; col < kStates; col++) {
            int mOff = matrixOff + row * 2 * kStates + col * 2;
            int zOff = zRowOff + col * 2;
            double zRe = design[zOff];
            double zIm = -design[zOff + 1];
            double mRe = matrix[mOff];
            double mIm = matrix[mOff + 1];
            sumRe += mRe * zRe - mIm * zIm;
            sumIm += mRe * zIm + mIm * zRe;
            }
            target[targetOffset + row * 2] = sumRe;
            target[targetOffset + row * 2 + 1] = sumIm;
        }
        }

        private static void addConjScaledRow(double[] target, int targetOffset, double[] design, int zRowOff,
                         double scalarRe, double scalarIm, int kStates) {
        for (int i = 0; i < kStates; i++) {
            int zOff = zRowOff + i * 2;
            int tOff = targetOffset + i * 2;
            double zRe = design[zOff];
            double zIm = -design[zOff + 1];
            target[tOff] += zRe * scalarRe - zIm * scalarIm;
            target[tOff + 1] += zRe * scalarIm + zIm * scalarRe;
        }
        }

        private static void addOuterConjZ(double[] matrix, int matrixOffset,
                          double[] design, int zRowOff,
                          double alpha,
                          double[] scratch, int conjRowOffset,
                          int kStates) {
        conjugateRow(design, zRowOff, scratch, conjRowOffset, kStates);
        ZLAS.zgerc(kStates, kStates, alpha, 0.0,
            scratch, conjRowOffset, 1,
            scratch, conjRowOffset, 1,
            matrix, matrixOffset, kStates);
        }

        private static void conjugateRow(double[] source, int sourceOffset,
                         double[] target, int targetOffset,
                         int n) {
        for (int i = 0; i < n; i++) {
            int src = sourceOffset + i * 2;
            int dst = targetOffset + i * 2;
            target[dst] = source[src];
            target[dst + 1] = -source[src + 1];
        }
        }

        private static void subtractProduct(double[] matrix, int off,
                        double leftRe, double leftIm,
                        double rightRe, double rightIm) {
        matrix[off] -= leftRe * rightRe - leftIm * rightIm;
        matrix[off + 1] -= leftRe * rightIm + leftIm * rightRe;
        }

        private static void setIdentity(double[] matrix, int offset, int n) {
        zero(matrix, offset, n * 2 * n);
        for (int i = 0; i < n; i++) {
            matrix[offset + i * 2 * n + i * 2] = 1.0;
        }
        }

        private static double dotHermitian(double[] left, int leftOffset, double[] right, int rightOffset, int n) {
        double[] out = new double[2];
        ZLAS.zdotc(n, left, leftOffset >> 1, 1, right, rightOffset >> 1, 1, out);
        return out[0];
        }

        private static double quadraticForm(double[] matrix, int matrixOffset,
                                            double[] vector, int vectorOffset,
                                            double[] workspace, int workspaceOffset,
                                            int n) {
        ZLAS.zgemv(BLAS.Trans.NoTrans, n, n, 1.0, 0.0,
            matrix, matrixOffset, n,
            vector, vectorOffset, 1,
            0.0, 0.0, workspace, workspaceOffset, 1);
        return dotHermitian(vector, vectorOffset, workspace, workspaceOffset, n);
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

    private static void addWithConjTranspose(double[] source, int sourceOffset,
                                             double[] target, int targetOffset,
                                             int n) {
        for (int i = 0; i < n; i++) {
            int srcRow = sourceOffset + i * 2 * n;
            int dstRow = targetOffset + i * 2 * n;
            for (int j = 0; j < n; j++) {
                int ij = srcRow + j * 2;
                int ji = sourceOffset + j * 2 * n + i * 2;
                int out = dstRow + j * 2;
                target[out] += source[ij] + source[ji];
                target[out + 1] += source[ij + 1] - source[ji + 1];
            }
        }
    }

    private static void conjugateTransposeInPlace(double[] matrix, int offset, int n) {
        for (int i = 0; i < n; i++) {
            int diag = offset + i * 2 * n + i * 2;
            matrix[diag + 1] = -matrix[diag + 1];
            for (int j = i + 1; j < n; j++) {
                int ij = offset + i * 2 * n + j * 2;
                int ji = offset + j * 2 * n + i * 2;
                double ijRe = matrix[ij];
                double ijIm = matrix[ij + 1];
                double jiRe = matrix[ji];
                double jiIm = matrix[ji + 1];
                matrix[ij] = jiRe;
                matrix[ij + 1] = -jiIm;
                matrix[ji] = ijRe;
                matrix[ji + 1] = -ijIm;
            }
        }
    }

        private static void shiftLeft(double[] values, int base, int stride, int length) {
        System.arraycopy(values, base + stride, values, base, length);
        }

        private static void zero(double[] values, int offset, int length) {
        for (int i = 0; i < length; i++) {
            values[offset + i] = 0.0;
        }
        }
}
