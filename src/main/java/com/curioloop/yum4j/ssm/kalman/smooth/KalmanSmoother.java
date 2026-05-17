package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.arena.SmootherDiffuseLayout;
import com.curioloop.yum4j.ssm.kalman.arena.SmootherResultLayout;
import com.curioloop.yum4j.ssm.kalman.arena.SmootherScratchLayout;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;
import java.util.Objects;

public final class KalmanSmoother {

        static final int SMOOTH_CONVENTIONAL = 0x01;
        static final int SMOOTH_CLASSICAL = 0x02;
        static final int SMOOTH_ALTERNATIVE = 0x04;
        static final int SMOOTH_UNIVARIATE = 0x08;

        static final int SMOOTHER_STATE = 0x01;
        static final int SMOOTHER_STATE_COV = 0x02;
        static final int SMOOTHER_DISTURBANCE = 0x04;
        static final int SMOOTHER_DISTURBANCE_COV = 0x08;
        static final int SMOOTHER_STATE_AUTOCOV = 0x10;
        static final int SMOOTHER_ALL = 0x0F;

        static boolean storesAuxiliary(int smootherOutput) {
                return (smootherOutput & SMOOTHER_ALL) == SMOOTHER_ALL;
        }

        private static long doubleCount(double[] values) {
                return values == null ? 0L : values.length;
        }

        static final class ScratchArena {
                private final DoubleArena arena;
                private final boolean ownsArena;
                double[] scratchBacking;
                SmootherScratchLayout scratchLayout;
                int scratchKStates;
                int scratchKEndog;
                boolean scratchObservationFactor;
                boolean scratchForecastFactor;
                boolean scratchTransitionGain;
                boolean scratchDisturbanceCovariance;
                boolean scratchFactor;
                boolean scratchDiffuse;
                boolean scratchBorrowedSmoothingError;

                ScratchArena() {
                        this(new DoubleArena(), true);
                }

                ScratchArena(DoubleArena arena, boolean ownsArena) {
                        this.arena = Objects.requireNonNull(arena, "arena");
                        this.ownsArena = ownsArena;
                }

                long retainedDoubleCount() {
                        return usesScratchArena() ? arena.retainedDoubleCount() : 0L;
                }

                boolean ownsArena() {
                        return ownsArena;
                }

                boolean usesScratchArena() {
                        return scratchBacking != null;
                }

                void refreshScratchBacking() {
                        if (scratchLayout != null) {
                                scratchBacking = arena.backing();
                        }
                }

                void ensureMainScratch(int kStates,
                                       int kEndog,
                                       boolean needObservationFactor,
                                       boolean needForecastFactor,
                                       boolean needTransitionGain,
                                       boolean needDisturbanceCovariance,
                                       boolean needFactor,
                                       boolean needDiffuseScratch,
                                       boolean needBorrowedSmoothingError) {
                        int nextKStates = Math.max(scratchKStates, kStates);
                        int nextKEndog = Math.max(scratchKEndog, kEndog);
                        boolean nextObservationFactor = scratchObservationFactor || needObservationFactor;
                        boolean nextForecastFactor = scratchForecastFactor || needForecastFactor;
                        boolean nextTransitionGain = scratchTransitionGain || needTransitionGain;
                        boolean nextDisturbanceCovariance = scratchDisturbanceCovariance || needDisturbanceCovariance;
                        boolean nextFactor = scratchFactor || needFactor;
                        boolean nextDiffuse = scratchDiffuse || needDiffuseScratch;
                        boolean nextBorrowedSmoothingError = scratchBorrowedSmoothingError || needBorrowedSmoothingError;
                        if (scratchLayout == null
                                        || nextKStates != scratchKStates
                                        || nextKEndog != scratchKEndog
                                        || nextObservationFactor != scratchObservationFactor
                                        || nextForecastFactor != scratchForecastFactor
                                        || nextTransitionGain != scratchTransitionGain
                                        || nextDisturbanceCovariance != scratchDisturbanceCovariance
                                        || nextFactor != scratchFactor
                                        || nextDiffuse != scratchDiffuse
                                        || nextBorrowedSmoothingError != scratchBorrowedSmoothingError) {
                                scratchLayout = SmootherScratchLayout.create(
                                                nextKEndog,
                                                nextKStates,
                                                nextObservationFactor,
                                                nextForecastFactor,
                                                nextTransitionGain,
                                                nextDisturbanceCovariance,
                                                nextFactor,
                                                nextDiffuse,
                                                nextBorrowedSmoothingError);
                                scratchKStates = nextKStates;
                                scratchKEndog = nextKEndog;
                                scratchObservationFactor = nextObservationFactor;
                                scratchForecastFactor = nextForecastFactor;
                                scratchTransitionGain = nextTransitionGain;
                                scratchDisturbanceCovariance = nextDisturbanceCovariance;
                                scratchFactor = nextFactor;
                                scratchDiffuse = nextDiffuse;
                                scratchBorrowedSmoothingError = nextBorrowedSmoothingError;
                        }
                        int totalLength = scratchLayout.totalLength();
                        scratchBacking = arena.ensureCapacity(totalLength);
                }

                void ensureDiffuse(int kStates) {
                        ensureMainScratch(kStates, 0,
                                        false, false, false, false, false, true, false);
                }

                void releaseBackings() {
                        scratchBacking = null;
                        if (ownsArena) {
                                arena.release();
                        }
                        scratchLayout = null;
                        scratchKStates = 0;
                        scratchKEndog = 0;
                        scratchObservationFactor = false;
                        scratchForecastFactor = false;
                        scratchTransitionGain = false;
                        scratchDisturbanceCovariance = false;
                        scratchFactor = false;
                        scratchDiffuse = false;
                        scratchBorrowedSmoothingError = false;
                }
        }

        static final class Pool {
        private SmootherResult pooledResult;
        private final SmootherResult retainedResultView = new SmootherResult();
        private final DoubleArena resultArena;
        private final boolean ownsResultArena;
        private ScratchArena scratch;
        private double[] resultBacking;
        private double[] diffuseResultBacking;
        private SmootherResultLayout resultLayout;
        private SmootherDiffuseLayout diffuseResultLayout;
        private int resultKEndog = -1;
        private int resultKStates = -1;
        private int resultKPosdef = -1;
        private int resultNobs = -1;
        private int diffuseResultBase;

        Pool() {
                this(new DoubleArena(), true);
        }

        Pool(DoubleArena resultArena, boolean ownsResultArena) {
                this.resultArena = Objects.requireNonNull(resultArena, "resultArena");
                this.ownsResultArena = ownsResultArena;
        }

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

        boolean usesScratchArena() {
                return scratch != null && scratch.usesScratchArena();
        }

        void refreshScratchBacking() {
                if (scratch != null) {
                        scratch.refreshScratchBacking();
                }
        }

        public SmootherResult retainedResult() {
                if (resultBacking == null || resultLayout == null) {
                        return null;
                }
                retainedResultView.reuse(resultKEndog, resultKStates, resultKPosdef, resultNobs, resultBacking, resultLayout);
                bindRetainedDiffuseAuxiliary(retainedResultView);
                return retainedResultView;
        }

        void bindRetainedDiffuseAuxiliary(SmootherResult target) {
                Objects.requireNonNull(target, "target");
                if (diffuseResultBacking == null || diffuseResultLayout == null) {
                        return;
                }
                target.reuseDiffuseAuxiliary(diffuseResultBacking, diffuseResultBase, diffuseResultLayout);
        }

        void ensure(int kStates, int kEndog, int kPosdef) {
            scratchArena().ensureMainScratch(kStates, kEndog,
                            false, false, false, false, false, false, false);
                }

                void ensureObservationFactorScratch(int kStates, int kEndog) {
                        scratchArena().ensureMainScratch(kStates, kEndog,
                                        true, false, false, false, false, false, false);
                }

                void ensureForecastFactorScratch(int kEndog) {
                        scratchArena().ensureMainScratch(0, kEndog,
                                        false, true, false, false, false, false, false);
                }

                void ensureTransitionGainScratch(int kStates, int kEndog) {
                        scratchArena().ensureMainScratch(kStates, kEndog,
                                        false, false, true, false, false, false, false);
                }

                void ensureDisturbanceCovarianceScratch(int kStates, int kEndog) {
                        scratchArena().ensureMainScratch(kStates, kEndog,
                                        false, false, false, true, false, false, false);
                }

                void ensureFactorScratch(int kStates) {
                        scratchArena().ensureMainScratch(kStates, 0,
                                        false, false, false, false, true, false, false);
                }

                void ensurePooledScratch(int kEndog) {
                        scratchArena().ensureMainScratch(0, kEndog,
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
                        ensureResultArena();
                }

                void ensureDiffuseResult(int kStates, int nobs) {
                        SmootherDiffuseLayout nextLayout = SmootherDiffuseLayout.create(1, kStates, nobs);
                        if (!nextLayout.equals(diffuseResultLayout)) {
                                diffuseResultLayout = nextLayout;
                        }
                        ensureResultArena();
                        if (pooledResult != null && resultLayout != null) {
                                pooledResult.reuse(resultKEndog, resultKStates, resultKPosdef, resultNobs, resultBacking, resultLayout);
                        }
                }

                private void ensureResultArena() {
                        int regularLength = resultLayout == null ? 0 : resultLayout.totalLength();
                        int diffuseLength = diffuseResultLayout == null ? 0 : diffuseResultLayout.totalLength();
                        diffuseResultBase = regularLength;
                        double[] backing = resultArena.ensureCapacity(regularLength + diffuseLength);
                        resultBacking = backing;
                        diffuseResultBacking = diffuseResultLayout == null ? null : backing;
                }

                SmootherResult borrowResult(int kEndog, int kStates, int kPosdef, int nobs,
                                                                        SmootherResultLayout layout) {
                        ensureResult(kEndog, kStates, kPosdef, nobs, layout);
                        if (pooledResult == null) {
                                pooledResult = new SmootherResult();
                        }
                        pooledResult.reuse(kEndog, kStates, kPosdef, nobs, resultBacking, resultLayout);
                        bindRetainedDiffuseAuxiliary(pooledResult);
                        return pooledResult;
                }

                public void reserve(KalmanSSM model,
                                    SmootherOptions spec) {
                        Objects.requireNonNull(model, "model");
                        requireRealInputs(model, model);
                        SmootherOptions resolvedOptions = spec == null ? SmootherOptions.conventional() : spec;
                        SmootherResultLayout retainedLayout = resolvedOptions.createResultLayout(
                                        1,
                                        model.observationDimension(),
                                        model.stateCount(),
                                        model.stateDisturbanceCount(),
                                        model.observationCount());
                        ensureResult(model.observationDimension(), model.stateCount(),
                                        model.stateDisturbanceCount(), model.observationCount(), retainedLayout);
                }

                void ensureDiffuse(int kStates) {
                        scratchArena().ensureDiffuse(kStates);
        }

        public long retainedScratchDoubleCount() {
                return scratch == null ? 0L : scratch.retainedDoubleCount();
        }

        public long retainedResultDoubleCount() {
                        return resultArena.retainedDoubleCount();
        }

        public long retainedTotalByteCount() {
                return (retainedScratchDoubleCount() + retainedResultDoubleCount()) * Double.BYTES;
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
                                resultKEndog = -1;
                                resultKStates = -1;
                                resultKPosdef = -1;
                                resultNobs = -1;
                }

                public void releaseRetainedScratch() {
                        if (scratch != null) {
                                boolean ownsScratch = scratch.ownsArena();
                                scratch.releaseBackings();
                                if (ownsScratch) {
                                        scratch = null;
                                }
                        }
                }
    }

        /**
         * When a non-null pool is supplied, the returned result borrows pool-owned storage
         * and is invalidated by the next call that reuses the same pool. Call clone() or copy()
         * on the result when an independent snapshot is required.
         */
        static SmootherResult smooth(KalmanSSM model,
                                     FilterResult fr,
                                     Pool pool,
                                     SmootherOptions spec) {
                Objects.requireNonNull(model, "model");
                Objects.requireNonNull(fr, "fr");
                requireRealInputs(model, model);
                SmootherOptions resolvedOptions = spec == null ? SmootherOptions.conventional() : spec;
                                return smoothInternal(model, model, fr, pool,
                                                resolvedOptions.smoothMethodMask(), resolvedOptions.outputMask(), pool != null, null);
        }

        static SmootherResult smoothInto(KalmanSSM stateSpace,
                                         KalmanSSM observationModel,
                                         FilterResult fr,
                                         SmootherResult target,
                                         Pool pool,
                                         SmootherOptions spec) {
                Objects.requireNonNull(stateSpace, "stateSpace");
                Objects.requireNonNull(observationModel, "observationModel");
                Objects.requireNonNull(target, "target");
                Objects.requireNonNull(fr, "fr");
                requireRealInputs(stateSpace, observationModel);
                SmootherOptions resolvedOptions = spec == null ? SmootherOptions.conventional() : spec;
                return smoothInternal(stateSpace, observationModel, fr, pool,
                                resolvedOptions.smoothMethodMask(), resolvedOptions.outputMask(), pool != null, target);
        }

        private static SmootherResult smoothInternal(KalmanSSM stateSpace,
                                                                                                 KalmanSSM observationModel,
                                                                                                 FilterResult fr,
                                                                                                 Pool pool,
                                                                                                                                                                                                 int smoothMethod, int smootherOutput,
                                                                                                                                                                                                 boolean reuseResult,
                                                                                                                                                                                                 SmootherResult target) {
        if (pool == null) pool = new Pool();
        pool.ensure(stateSpace.stateCount(), observationModel.observationDimension(), stateSpace.stateDisturbanceCount());

                requireFilterStorage(fr, smoothMethod);
                if ((smoothMethod & SMOOTH_UNIVARIATE) != 0 && !fr.perObsKalmanGain) {
                        throw new IllegalArgumentException("Univariate smoothing requires a univariate FilterResult");
                }
                if ((smootherOutput & SMOOTHER_STATE_AUTOCOV) != 0
                                && (smoothMethod & (SMOOTH_CONVENTIONAL | SMOOTH_UNIVARIATE)) == 0) {
                        throw new UnsupportedOperationException(
                                        "State autocovariance smoothing is implemented for conventional and univariate real smoothing only");
                }
        if (fr.nobsDiffuse > 0) {
                        pool.ensureDiffuse(stateSpace.stateCount());
                        if (!storesAuxiliary(smootherOutput) && (smootherOutput & SMOOTHER_DISTURBANCE) != 0) {
                                                pool.ensurePooledScratch(observationModel.observationDimension());
                        }
            return smoothDiffuse(stateSpace, observationModel, fr, pool, smootherOutput, reuseResult, target);
        }

                pool.ensureObservationFactorScratch(stateSpace.stateCount(), observationModel.observationDimension());

                if (!storesAuxiliary(smootherOutput) && (smootherOutput & SMOOTHER_DISTURBANCE) != 0) {
                        pool.ensurePooledScratch(observationModel.observationDimension());
                }

        if (fr.perObsKalmanGain) {
                        return smoothUnivariate(stateSpace, observationModel, fr, pool, smootherOutput, reuseResult, target);
        }

        if ((smoothMethod & SMOOTH_CLASSICAL) != 0) {
                        return smoothClassical(stateSpace, observationModel, fr, pool, smootherOutput, reuseResult, target);
        }
        if ((smoothMethod & SMOOTH_ALTERNATIVE) != 0) {
                        return smoothAlternative(stateSpace, observationModel, fr, pool, smootherOutput, reuseResult, target);
        }
                return smoothConventional(stateSpace, observationModel, fr, pool, smootherOutput, reuseResult, target);
    }

    private static SmootherResult prepareResult(SmootherResult target,
                                                Pool pool,
                                                boolean reuseResult,
                                                int kEndog,
                                                int kStates,
                                                int kPosdef,
                                                int nobs,
                                                boolean doState,
                                                boolean doStateCov,
                                                boolean doStateAutocov,
                                                boolean doDisturbance,
                                                boolean doDisturbanceCov,
                                                boolean storeAuxiliary) {
        SmootherResultLayout layout = SmootherResultLayout.create(
                1,
                kEndog,
                kStates,
                kPosdef,
                nobs,
                doState,
                doStateCov,
                doStateAutocov,
                doDisturbance,
                doDisturbanceCov,
                storeAuxiliary);
        if (target != null) {
            return target;
        }
        return reuseResult
                ? pool.borrowResult(kEndog, kStates, kPosdef, nobs, layout)
                : new SmootherResult(kEndog, kStates, kPosdef, nobs, layout);
    }

        private static void computeConventionalStateAutocovariance(FilterResult fr,
                                                                   SmootherResult result,
                                                                   int t,
                                                                   int kStates,
                                                                   double[] scratch,
                                                                   int lBase,
                                                                   int nBase,
                                                                   int tmpBase) {
                computeConventionalStateAutocovariance(fr, result, t, kStates,
                        scratch, lBase, scratch, nBase, scratch, tmpBase);
        }

        private static void computeConventionalStateAutocovariance(FilterResult fr,
                                                                   SmootherResult result,
                                                                   int t,
                                                                   int kStates,
                                                                   double[] lArray,
                                                                   int lBase,
                                                                   double[] nArray,
                                                                   int nBase,
                                                                   double[] tmpArray,
                                                                   int tmpBase) {
                int predCovOff = fr.predictedStateCovOffset(t);
                int nextPredCovOff = fr.predictedStateCovOffset(t + 1);
                int autocovOff = result.smoothedStateAutocovarianceOffset(t);

                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, lArray, lBase, kStates,
                        fr.predictedStateCov, predCovOff, kStates,
                        0.0, result.smoothedStateAutocovariance, autocovOff, kStates);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, nArray, nBase, kStates,
                        result.smoothedStateAutocovariance, autocovOff, kStates,
                        0.0, tmpArray, tmpBase, kStates);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        -1.0, fr.predictedStateCov, nextPredCovOff, kStates,
                        tmpArray, tmpBase, kStates,
                        1.0, result.smoothedStateAutocovariance, autocovOff, kStates);
        }

        private static void computeDiffuseStateAutocovarianceTail(KalmanSSM stateSpace,
                                                                  KalmanSSM observationModel,
                                                                  FilterResult fr,
                                                                  SmootherResult result,
                                                                  double[] scratch,
                                                                  int measurementTransitionBase,
                                                                  int singleObservationTransitionBase,
                                                                  int innovationsTransitionBase,
                                                                  int tmpBase) {
                int kEndog = observationModel.observationDimension();
                int kStates = stateSpace.stateCount();
                int kStates2 = kStates * kStates;
                double[] transition = stateSpace.transitionData();
                int transitionLd = stateSpace.transitionLeadingDimension();
                double[] design = observationModel.designData();
                int designLd = observationModel.designLeadingDimension();
                double[] nextMeasuredN = new double[kStates2];
                double[] currentPredictedN = new double[kStates2];
                double[] measuredN = new double[kStates2];

                for (int t = observationModel.observationCount() - 1; t >= fr.nobsDiffuse; t--) {
                        System.arraycopy(currentPredictedN, 0, measuredN, 0, kStates2);
                        setIdentity(scratch, measurementTransitionBase, kStates);
                        int kgOff = fr.kalmanGainOffset(t);
                        int foreErrCovOff = fr.forecastErrorCovOffset(t);
                        int designOff = observationModel.designOffset(t);
                        for (int obs = kEndog - 1; obs >= 0; obs--) {
                                if (observationModel.isMissing(obs, t)) {
                                        continue;
                                }
                                double Fi = fr.forecastErrorCov[foreErrCovOff + obs * kEndog + obs];
                                if (Fi <= 0.0) {
                                        continue;
                                }
                                double FiInv = 1.0 / Fi;
                                setIdentity(scratch, singleObservationTransitionBase, kStates);
                                int zRowOff = designOff + obs * designLd;
                                for (int row = 0; row < kStates; row++) {
                                        double gain = fr.kalmanGain[kgOff + row * kEndog + obs];
                                        for (int col = 0; col < kStates; col++) {
                                                scratch[singleObservationTransitionBase + row * kStates + col] -=
                                                        gain * design[zRowOff + col];
                                        }
                                }
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans,
                                        kStates, kStates, kStates,
                                        1.0, scratch, measurementTransitionBase, kStates,
                                        scratch, singleObservationTransitionBase, kStates,
                                        0.0, scratch, tmpBase, kStates);
                                BLAS.dcopy(kStates2, scratch, tmpBase, 1,
                                        scratch, measurementTransitionBase, 1);

                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans,
                                        kStates, kStates, kStates,
                                        1.0, measuredN, 0, kStates,
                                        scratch, singleObservationTransitionBase, kStates,
                                        0.0, scratch, tmpBase, kStates);
                                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans,
                                        kStates, kStates, kStates,
                                        1.0, scratch, singleObservationTransitionBase, kStates,
                                        scratch, tmpBase, kStates,
                                        0.0, measuredN, 0, kStates);
                                for (int row = 0; row < kStates; row++) {
                                        double zi = design[zRowOff + row] * FiInv;
                                        for (int col = 0; col < kStates; col++) {
                                                measuredN[row * kStates + col] += zi * design[zRowOff + col];
                                        }
                                }
                        }

                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans,
                                kStates, kStates, kStates,
                                1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                                scratch, measurementTransitionBase, kStates,
                                0.0, scratch, innovationsTransitionBase, kStates);
                        computeConventionalStateAutocovariance(fr, result, t, kStates,
                                scratch, innovationsTransitionBase,
                                nextMeasuredN, 0,
                                scratch, tmpBase);

                        System.arraycopy(measuredN, 0, nextMeasuredN, 0, kStates2);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans,
                                kStates, kStates, kStates,
                                1.0, measuredN, 0, kStates,
                                transition, stateSpace.transitionOffset(t), transitionLd,
                                0.0, scratch, tmpBase, kStates);
                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans,
                                kStates, kStates, kStates,
                                1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                                scratch, tmpBase, kStates,
                                0.0, currentPredictedN, 0, kStates);
                }
        }

        private static SmootherResult smoothConventional(KalmanSSM stateSpace,
                                                         KalmanSSM observationModel,
                                                         FilterResult fr,
                                                         Pool pool, int smootherOutput,
                                                         boolean reuseResult,
                                                         SmootherResult target) {
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int kEndog2 = kEndog * kEndog;
        int kStates2 = kStates * kStates;
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
        boolean doStateAutocov = (smootherOutput & SMOOTHER_STATE_AUTOCOV) != 0;
        boolean doDisturbance = (smootherOutput & SMOOTHER_DISTURBANCE) != 0;
        boolean doDisturbanceCov = (smootherOutput & SMOOTHER_DISTURBANCE_COV) != 0;
        boolean storeAuxiliary = storesAuxiliary(smootherOutput);

        pool.ensureForecastFactorScratch(kEndog);
        if (fr.perObsKalmanGain) {
                pool.ensureTransitionGainScratch(kStates, kEndog);
        }
        if (doDisturbanceCov) {
                pool.ensureDisturbanceCovarianceScratch(kStates, kEndog);
        }

        SmootherResult result = prepareResult(target, pool, reuseResult,
                kEndog, kStates, kPosdef, nobs,
                doState, doStateCov, doStateAutocov, doDisturbance, doDisturbanceCov, storeAuxiliary);
        ScratchArena scratchArena = pool.scratch;
        SmootherScratchLayout layout = scratchArena.scratchLayout;
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
        int tmpFcopyBase = layout.tmpFcopyBase();
        int selectedDesignBase = layout.selectedDesignBase();
        int selectedObsCovBase = layout.selectedObsCovBase();
        int tmpKgBase = layout.tmpKgBase();
        int tmpTkBase = fr.perObsKalmanGain ? layout.tmpTkBase() : -1;
        int tmpKhBase = doDisturbanceCov ? layout.tmpKhBase() : -1;
        int tmpFiHBase = doDisturbanceCov ? layout.tmpFiHBase() : -1;
        int borrowedSmoothingErrorBase = (!storeAuxiliary && doDisturbance)
                ? layout.scratchSmoothingErrorBase()
                : 0;
        double[] smoothingError = (storeAuxiliary || doDisturbance)
                ? (storeAuxiliary ? result.smoothingError : scratch)
                : null;

        for (int i = 0; i < kStates; i++) scratch[rBase + i] = 0.0;
        for (int i = 0; i < kStates2; i++) scratch[nBase + i] = 0.0;

        for (int t = nobs - 1; t >= 0; t--) {
                int predOff = fr.predictedStateOffset(t);
                int predCovOff = fr.predictedStateCovOffset(t);
                int foreErrOff = fr.forecastErrorOffset(t);
                int foreErrCovOff = fr.forecastErrorCovOffset(t);
                int kgOff = fr.kalmanGainOffset(t);

                int ke = selectObservations(observationModel, scratchArena, layout, t);

                if (ke == 0) {
                        BLAS.dcopy(kStates2, transition, stateSpace.transitionOffset(t), 1,
                                scratch, lBase, 1);
                        BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                                transition, stateSpace.transitionOffset(t), transitionLd,
                                scratch, rBase, 1,
                                0.0, scratch, rPrevBase, 1);
                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                                scratch, nBase, kStates,
                                0.0, scratch, tmpNpBase, kStates);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, tmpNpBase, kStates,
                                transition, stateSpace.transitionOffset(t), transitionLd,
                                0.0, scratch, nPrevBase, kStates);

                        if (doState) {
                                BLAS.dcopy(kStates, fr.predictedState, predOff, 1,
                                        result.smoothedState, result.smoothedStateOffset(t), 1);
                                BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                                        fr.predictedStateCov, predCovOff, kStates,
                                        scratch, rPrevBase, 1,
                                        1.0, result.smoothedState, result.smoothedStateOffset(t), 1);
                        }
                        if (doStateCov) {
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                        -1.0, fr.predictedStateCov, predCovOff, kStates,
                                        scratch, nPrevBase, kStates,
                                        0.0, scratch, tmpNpBase, kStates);
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                        1.0, scratch, tmpNpBase, kStates,
                                        fr.predictedStateCov, predCovOff, kStates,
                                        0.0, result.smoothedStateCov, result.smoothedStateCovOffset(t), kStates);
                                BLAS.daxpy(kStates2, 1.0, fr.predictedStateCov, predCovOff, 1,
                                        result.smoothedStateCov, result.smoothedStateCovOffset(t), 1);
                        }

                        if (doStateAutocov) {
                                computeConventionalStateAutocovariance(fr, result, t, kStates,
                                        scratch, lBase, nBase, tmpNpBase);
                        }

                        if (storeAuxiliary) {
                                int smoothingErrorOff = result.smoothingErrorOffset(t);
                                for (int i = 0; i < kEndog; i++) {
                                        result.smoothingError[smoothingErrorOff + i] = 0.0;
                                }
                                BLAS.dcopy(kStates2, scratch, lBase, 1,
                                        result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                                BLAS.dcopy(kStates, scratch, rPrevBase, 1,
                                        result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                                BLAS.dcopy(kStates2, scratch, nPrevBase, 1,
                                        result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
                        }

                        if (doDisturbance) {
                                for (int i = 0; i < kEndog; i++) {
                                        result.smoothedObsDisturbance[result.smoothedObsDisturbanceOffset(t) + i] = 0.0;
                                }
                        }
                        if (doDisturbance && kPosdef > 0) {
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                        0.0, scratch, tmpRqBase, kPosdef);
                                BLAS.dgemv(BLAS.Trans.Trans, kStates, kPosdef, 1.0,
                                        scratch, tmpRqBase, kPosdef,
                                        scratch, rBase, 1,
                                        0.0, result.smoothedStateDisturbance, result.smoothedStateDisturbanceOffset(t), 1);
                        }

                        if (doDisturbanceCov) {
                                BLAS.dcopy(kEndog2, obsCov, observationModel.obsCovOffset(t), 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                                if (kPosdef > 0) {
                                        if (!doDisturbance) {
                                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                                        0.0, scratch, tmpRqBase, kPosdef);
                                        }
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                                                1.0, scratch, nBase, kStates,
                                                scratch, tmpRqBase, kPosdef,
                                                0.0, scratch, tmpNpBase, kPosdef);
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                                                -1.0, scratch, tmpRqBase, kPosdef,
                                                scratch, tmpNpBase, kPosdef,
                                                0.0, result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), kPosdef);
                                        BLAS.daxpy(kPosdef * kPosdef, 1.0, stateCovariance, stateSpace.stateCovarianceOffset(t), 1,
                                                result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), 1);
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

                double[] Z = design;
                int zOff = observationModel.designOffset(t);
                int zLd = designLd;
                double[] H = obsCov;
                int hOff = observationModel.obsCovOffset(t);
                int hLd = obsCovLd;

                if (ke < kEndog) {
                        Z = scratch;
                        zOff = selectedDesignBase;
                        zLd = kStates;
                        H = scratch;
                        hOff = selectedObsCovBase;
                        hLd = ke;
                }

                int ke2 = ke * ke;

                if (ke < kEndog) {
                        copyObservedSquareMatrix(observationModel, t,
                                fr.forecastErrorCov, foreErrCovOff, kEndog,
                                scratch, tmpFcopyBase, ke);
                } else {
                        BLAS.dcopy(ke2, fr.forecastErrorCov, foreErrCovOff, 1, scratch, tmpFcopyBase, 1);
                }

                if (ke == 1) {
                        double Ft = scratch[tmpFcopyBase];
                        double FtInv = 1.0 / Ft;
                        if (ke < kEndog) {
                                copyObservedVector(observationModel, t, fr.forecastError, foreErrOff, scratch, tmpFivBase);
                                scratch[tmpFivBase] *= FtInv;
                        } else {
                                scratch[tmpFivBase] = fr.forecastError[foreErrOff] * FtInv;
                        }
                        for (int j = 0; j < kStates; j++) {
                                scratch[tmpFiZBase + j] = Z[zOff + j] * FtInv;
                        }
                } else {
                        BLAS.dpotrf(BLAS.Uplo.Lower, ke, scratch, tmpFcopyBase, ke);
                        if (ke < kEndog) {
                                copyObservedVector(observationModel, t, fr.forecastError, foreErrOff, scratch, tmpFivBase);
                        } else {
                                BLAS.dcopy(ke, fr.forecastError, foreErrOff, 1, scratch, tmpFivBase, 1);
                        }
                        BLAS.dpotrs(BLAS.Uplo.Lower, ke, 1,
                                scratch, tmpFcopyBase, ke, scratch, tmpFivBase, 1);
                        BLAS.dcopy(ke * kStates, Z, zOff, 1, scratch, tmpFiZBase, 1);
                        BLAS.dpotrs(BLAS.Uplo.Lower, ke, kStates,
                                scratch, tmpFcopyBase, ke, scratch, tmpFiZBase, kStates);
                }

                double[] K;
                int kOff;
                int kLd;
                if (ke < kEndog) {
                        copyObservedColumns(observationModel, t, kStates,
                                fr.kalmanGain, kgOff, kEndog,
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
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                                1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                                K, kOff, kLd,
                                0.0, scratch, tmpTkBase, ke);
                        K = scratch;
                        kOff = tmpTkBase;
                        kLd = ke;
                }

                if (smoothingError != null) {
                        int smoothingErrorOff = storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase;
                        BLAS.dgemv(BLAS.Trans.Trans, kStates, ke, -1.0,
                                K, kOff, kLd,
                                scratch, rBase, 1,
                                0.0, smoothingError, smoothingErrorOff, 1);
                        BLAS.daxpy(ke, 1.0, scratch, tmpFivBase, 1,
                                smoothingError, smoothingErrorOff, 1);

                        if (ke < kEndog) {
                                scatterObservedVectorInPlace(observationModel, t,
                                        smoothingError, smoothingErrorOff);
                        }
                }

                BLAS.dcopy(kStates2, transition, stateSpace.transitionOffset(t), 1,
                        scratch, lBase, 1);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                        -1.0, K, kOff, kLd,
                        Z, zOff, zLd,
                        1.0, scratch, lBase, kStates);

                BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                        scratch, lBase, kStates,
                        scratch, rBase, 1,
                        0.0, scratch, rPrevBase, 1);
                BLAS.dgemv(BLAS.Trans.Trans, ke, kStates, 1.0,
                        Z, zOff, zLd,
                        scratch, tmpFivBase, 1,
                        1.0, scratch, rPrevBase, 1);

                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, scratch, nBase, kStates,
                        scratch, lBase, kStates,
                        0.0, scratch, tmpNpBase, kStates);
                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, scratch, lBase, kStates,
                        scratch, tmpNpBase, kStates,
                        0.0, scratch, nPrevBase, kStates);
                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                        1.0, Z, zOff, zLd,
                        scratch, tmpFiZBase, kStates,
                        1.0, scratch, nPrevBase, kStates);

                if (doState) {
                        BLAS.dcopy(kStates, fr.predictedState, predOff, 1,
                                result.smoothedState, result.smoothedStateOffset(t), 1);
                        BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                                fr.predictedStateCov, predCovOff, kStates,
                                scratch, rPrevBase, 1,
                                1.0, result.smoothedState, result.smoothedStateOffset(t), 1);
                }

                if (doStateCov) {
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                -1.0, fr.predictedStateCov, predCovOff, kStates,
                                scratch, nPrevBase, kStates,
                                0.0, scratch, tmpNpBase, kStates);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, tmpNpBase, kStates,
                                fr.predictedStateCov, predCovOff, kStates,
                                0.0, result.smoothedStateCov, result.smoothedStateCovOffset(t), kStates);
                        BLAS.daxpy(kStates2, 1.0, fr.predictedStateCov, predCovOff, 1,
                                result.smoothedStateCov, result.smoothedStateCovOffset(t), 1);
                }

                if (doStateAutocov) {
                        computeConventionalStateAutocovariance(fr, result, t, kStates,
                                scratch, lBase, nBase, tmpNpBase);
                }

                if (storeAuxiliary) {
                        BLAS.dcopy(kStates2, scratch, lBase, 1,
                                result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                        BLAS.dcopy(kStates, scratch, rPrevBase, 1,
                                result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                        BLAS.dcopy(kStates2, scratch, nPrevBase, 1,
                                result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
                }

                if (doDisturbance) {
                        if (ke < kEndog) {
                                int smoothingErrorOff = storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase;
                                copyObservedVector(observationModel, t,
                                        smoothingError, smoothingErrorOff,
                                        scratch, tmpFivBase);
                                BLAS.dgemv(BLAS.Trans.NoTrans, ke, ke, 1.0,
                                        H, hOff, hLd,
                                        scratch, tmpFivBase, 1,
                                        0.0, scratch, tmpFiZBase, 1);
                                scatterObservedVector(observationModel, t,
                                        scratch, tmpFiZBase,
                                        result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t));
                        } else {
                                BLAS.dgemv(BLAS.Trans.NoTrans, ke, ke, 1.0,
                                        H, hOff, hLd,
                                        smoothingError, storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase, 1,
                                        0.0, result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t), 1);
                        }

                        if (kPosdef > 0) {
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                        0.0, scratch, tmpRqBase, kPosdef);
                                BLAS.dgemv(BLAS.Trans.Trans, kStates, kPosdef, 1.0,
                                        scratch, tmpRqBase, kPosdef,
                                        scratch, rBase, 1,
                                        0.0, result.smoothedStateDisturbance, result.smoothedStateDisturbanceOffset(t), 1);
                        }
                }

                if (doDisturbanceCov) {
                        double FtInv = ke == 1 ? 1.0 / scratch[tmpFcopyBase] : 0.0;
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, ke,
                                1.0, K, kOff, kLd,
                                H, hOff, hLd,
                                0.0, scratch, tmpKhBase, ke);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                                1.0, scratch, nBase, kStates,
                                scratch, tmpKhBase, ke,
                                0.0, scratch, tmpNpBase, ke);
                        if (ke >= kEndog) {
                                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, ke, ke, kStates,
                                        -1.0, scratch, tmpKhBase, ke,
                                        scratch, tmpNpBase, ke,
                                        0.0, result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), kEndog);
                                BLAS.daxpy(kEndog2, 1.0, H, hOff, 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                        }

                        if (ke == 1) {
                                if (ke < kEndog) {
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, ke, ke, kStates,
                                                -1.0, scratch, tmpKhBase, ke,
                                                scratch, tmpNpBase, ke,
                                                0.0, scratch, tmpFcopyBase, ke);
                                        BLAS.daxpy(ke2, 1.0, H, hOff, 1, scratch, tmpFcopyBase, 1);
                                        scratch[tmpFcopyBase] -= H[hOff] * FtInv * H[hOff];
                                } else {
                                        for (int ri = 0; ri < ke; ri++) {
                                                for (int ci = 0; ci < ke; ci++) {
                                                        int idx = result.smoothedObsDisturbanceCovOffset(t) + ri * kEndog + ci;
                                                        result.smoothedObsDisturbanceCov[idx] -=
                                                                H[hOff + ri * hLd + ci] * FtInv * H[hOff + ci * hLd + ri];
                                                }
                                        }
                                }
                        } else {
                                BLAS.dcopy(ke2, H, hOff, 1, scratch, tmpFiHBase, 1);
                                BLAS.dpotrs(BLAS.Uplo.Lower, ke, ke,
                                        scratch, tmpFcopyBase, ke, scratch, tmpFiHBase, ke);
                                if (ke < kEndog) {
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, ke, ke, kStates,
                                                -1.0, scratch, tmpKhBase, ke,
                                                scratch, tmpNpBase, ke,
                                                0.0, scratch, tmpFcopyBase, ke);
                                        BLAS.daxpy(ke2, 1.0, H, hOff, 1, scratch, tmpFcopyBase, 1);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                                                -1.0, H, hOff, hLd,
                                                scratch, tmpFiHBase, ke,
                                                1.0, scratch, tmpFcopyBase, ke);
                                } else {
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                                                -1.0, H, hOff, hLd,
                                                scratch, tmpFiHBase, ke,
                                                1.0, result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), kEndog);
                                }
                        }

                        if (ke < kEndog) {
                                BLAS.dcopy(kEndog2, obsCov, observationModel.obsCovOffset(t), 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                                scatterObservedSquareMatrix(observationModel, t,
                                        scratch, tmpFcopyBase, ke,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), kEndog);
                        }

                        if (kPosdef > 0) {
                                if (!doDisturbance) {
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                                1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                                stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                                0.0, scratch, tmpRqBase, kPosdef);
                                }
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                                        1.0, scratch, nBase, kStates,
                                        scratch, tmpRqBase, kPosdef,
                                        0.0, scratch, tmpNpBase, kPosdef);
                                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                                        -1.0, scratch, tmpRqBase, kPosdef,
                                        scratch, tmpNpBase, kPosdef,
                                        0.0, result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), kPosdef);
                                BLAS.daxpy(kPosdef * kPosdef, 1.0, stateCovariance, stateSpace.stateCovarianceOffset(t), 1,
                                        result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), 1);
                        }
                }

                int tmpRBase = rBase;
                rBase = rPrevBase;
                rPrevBase = tmpRBase;
                int tmpNBase = nBase;
                nBase = nPrevBase;
                nPrevBase = tmpNBase;
        }

        for (int i = 0; i < kStates; i++) scratch[rBase + i] = 0.0;
        for (int i = 0; i < kStates2; i++) scratch[nBase + i] = 0.0;
        if (storeAuxiliary) {
                BLAS.dcopy(kStates, scratch, rBase, 1,
                                result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(nobs), 1);
                BLAS.dcopy(kStates2, scratch, nBase, 1,
                                result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(nobs), 1);

                int scaledEstimatorBase = result.scaledSmoothedEstimatorBase();
                int scaledEstimatorCovBase = result.scaledSmoothedEstimatorCovBase();
                for (int i = 0; i < nobs * kStates; i++) {
                        result.scaledSmoothedEstimator[scaledEstimatorBase + i] = result.scaledSmoothedEstimator[scaledEstimatorBase + i + kStates];
                }
                for (int i = 0; i < nobs * kStates2; i++) {
                        result.scaledSmoothedEstimatorCov[scaledEstimatorCovBase + i] = result.scaledSmoothedEstimatorCov[scaledEstimatorCovBase + i + kStates2];
                }
        }

        return result.trimOutputMask(smootherOutput);
    }

        private static SmootherResult smoothClassical(KalmanSSM stateSpace,
                                                      KalmanSSM observationModel,
                                                      FilterResult fr,
                                                      Pool pool, int smootherOutput,
                                                      boolean reuseResult,
                                                      SmootherResult target) {
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int kEndog2 = kEndog * kEndog;
        int kStates2 = kStates * kStates;
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
        boolean doStateAutocov = (smootherOutput & SMOOTHER_STATE_AUTOCOV) != 0;
        boolean doDisturbance = (smootherOutput & SMOOTHER_DISTURBANCE) != 0;
        boolean doDisturbanceCov = (smootherOutput & SMOOTHER_DISTURBANCE_COV) != 0;
        boolean storeAuxiliary = storesAuxiliary(smootherOutput);
        boolean needSmoothedState = doState || doDisturbance;
        boolean needSmoothedStateCov = doStateCov || doDisturbanceCov;

        pool.ensureForecastFactorScratch(kEndog);
        pool.ensureFactorScratch(kStates);
        if (fr.perObsKalmanGain) {
                pool.ensureTransitionGainScratch(kStates, kEndog);
        }
        if (doDisturbanceCov) {
                pool.ensureDisturbanceCovarianceScratch(kStates, kEndog);
        }

        SmootherResult result = prepareResult(target, pool, reuseResult,
                kEndog, kStates, kPosdef, nobs,
                doState, doStateCov, doStateAutocov, doDisturbance, doDisturbanceCov, storeAuxiliary);
        ScratchArena scratchArena = pool.scratch;
        SmootherScratchLayout layout = scratchArena.scratchLayout;
        double[] scratch = scratchArena.scratchBacking;
        int rBase = layout.rBase();
        int nBase = layout.nBase();
        int lBase = layout.lBase();
        int tmpNpBase = layout.tmpNpBase();
        int tmpRqBase = layout.tmpRqBase();
        int tmpFivBase = layout.tmpFivBase();
        int tmpFiZBase = layout.tmpFiZBase();
        int tmpFcopyBase = layout.tmpFcopyBase();
        int selectedDesignBase = layout.selectedDesignBase();
        int selectedObsCovBase = layout.selectedObsCovBase();
        int tmpKgBase = layout.tmpKgBase();
        int tmpTkBase = fr.perObsKalmanGain ? layout.tmpTkBase() : -1;
        int tmpKhBase = doDisturbanceCov ? layout.tmpKhBase() : -1;
        int tmpFiHBase = doDisturbanceCov ? layout.tmpFiHBase() : -1;
        int tmpJBase = layout.tmpJBase();
        int borrowedSmoothingErrorBase = (!storeAuxiliary && doDisturbance)
                ? layout.scratchSmoothingErrorBase()
                : 0;
        double[] smoothingError = (storeAuxiliary || doDisturbance)
                ? (storeAuxiliary ? result.smoothingError : scratch)
                : null;
        double[] nextSmoothedState = (!doState && needSmoothedState) ? new double[kStates] : null;
        double[] nextSmoothedStateCov = (!doStateCov && needSmoothedStateCov) ? new double[kStates2] : null;

        for (int t = nobs - 1; t >= 0; t--) {
                int filtOff = fr.filteredStateOffset(t);
                int filtCovOff = fr.filteredStateCovOffset(t);
                int foreErrOff = fr.forecastErrorOffset(t);
                int foreErrCovOff = fr.forecastErrorCovOffset(t);
                int kgOff = fr.kalmanGainOffset(t);

                int ke = selectObservations(observationModel, scratchArena, layout, t);

                if (t < nobs - 1) {
                        int nextPredOff = fr.predictedStateOffset(t + 1);
                        int nextPredCovOff = fr.predictedStateCovOffset(t + 1);
                        double[] smoothedState = doState ? result.smoothedState : nextSmoothedState;
                        int smoothedStateOff = doState ? result.smoothedStateOffset(t + 1) : 0;
                        double[] smoothedStateCov = doStateCov ? result.smoothedStateCov : nextSmoothedStateCov;
                        int smoothedStateCovOff = doStateCov ? result.smoothedStateCovOffset(t + 1) : 0;

                        if (needSmoothedState) {
                                for (int i = 0; i < kStates; i++) {
                                        scratch[rBase + i] = smoothedState[smoothedStateOff + i] - fr.predictedState[nextPredOff + i];
                                }
                        }

                        if (needSmoothedState || needSmoothedStateCov) {
                                BLAS.dcopy(kStates2, fr.predictedStateCov, nextPredCovOff, 1, scratch, tmpJBase, 1);
                                BLAS.dpotrf(BLAS.Uplo.Lower, kStates, scratch, tmpJBase, kStates);
                        }
                        if (needSmoothedState) {
                                BLAS.dpotrs(BLAS.Uplo.Lower, kStates, 1, scratch, tmpJBase, kStates, scratch, rBase, 1);
                        }

                        if (needSmoothedStateCov) {
                                BLAS.dcopy(kStates2, fr.predictedStateCov, nextPredCovOff, 1, scratch, nBase, 1);
                                BLAS.daxpy(kStates2, -1.0, smoothedStateCov, smoothedStateCovOff, 1, scratch, nBase, 1);
                                BLAS.dpotrs(BLAS.Uplo.Lower, kStates, kStates, scratch, tmpJBase, kStates, scratch, nBase, kStates);
                                for (int i = 0; i < kStates; i++) {
                                        for (int j = i + 1; j < kStates; j++) {
                                                double tmp = scratch[nBase + i * kStates + j];
                                                scratch[nBase + i * kStates + j] = scratch[nBase + j * kStates + i];
                                                scratch[nBase + j * kStates + i] = tmp;
                                        }
                                }
                                BLAS.dpotrs(BLAS.Uplo.Lower, kStates, kStates, scratch, tmpJBase, kStates, scratch, nBase, kStates);
                                symmetrizeMatrix(scratch, nBase, kStates);
                        }
                } else {
                        for (int i = 0; i < kStates; i++) scratch[rBase + i] = 0.0;
                        for (int i = 0; i < kStates2; i++) scratch[nBase + i] = 0.0;
                }

                if (ke == 0) {
                        if (needSmoothedState) {
                                double[] smoothedState = doState ? result.smoothedState : nextSmoothedState;
                                int smoothedStateOff = doState ? result.smoothedStateOffset(t) : 0;
                                BLAS.dcopy(kStates, fr.filteredState, filtOff, 1,
                                                smoothedState, smoothedStateOff, 1);
                                if (t < nobs - 1) {
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates, 1.0,
                                                fr.filteredStateCov, filtCovOff, kStates,
                                                transition, stateSpace.transitionOffset(t), transitionLd,
                                                0.0, scratch, tmpRqBase, kStates);
                                        BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                                                scratch, tmpRqBase, kStates,
                                                scratch, rBase, 1,
                                                1.0, smoothedState, smoothedStateOff, 1);
                                }
                        }
                        if (needSmoothedStateCov) {
                                double[] smoothedStateCov = doStateCov ? result.smoothedStateCov : nextSmoothedStateCov;
                                int smoothedStateCovOff = doStateCov ? result.smoothedStateCovOffset(t) : 0;
                                if (t < nobs - 1) {
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates, 1.0,
                                                fr.filteredStateCov, filtCovOff, kStates,
                                                transition, stateSpace.transitionOffset(t), transitionLd,
                                                0.0, scratch, tmpRqBase, kStates);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates, 1.0,
                                                scratch, nBase, kStates,
                                                scratch, tmpRqBase, kStates,
                                                0.0, scratch, tmpJBase, kStates);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates, -1.0,
                                                scratch, tmpRqBase, kStates,
                                                scratch, tmpJBase, kStates,
                                                0.0, smoothedStateCov, smoothedStateCovOff, kStates);
                                        BLAS.daxpy(kStates2, 1.0, fr.filteredStateCov, filtCovOff, 1,
                                                smoothedStateCov, smoothedStateCovOff, 1);
                                } else {
                                        BLAS.dcopy(kStates2, fr.filteredStateCov, filtCovOff, 1,
                                                smoothedStateCov, smoothedStateCovOff, 1);
                                }
                                symmetrizeMatrix(smoothedStateCov, smoothedStateCovOff, kStates);
                        }

                        if (storeAuxiliary) {
                                BLAS.dcopy(kStates2, transition, stateSpace.transitionOffset(t), 1,
                                        result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                                BLAS.dcopy(kStates, scratch, rBase, 1,
                                        result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                                BLAS.dcopy(kStates2, scratch, nBase, 1,
                                        result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
                        }

                        if (doDisturbance) {
                                for (int i = 0; i < kEndog; i++) {
                                        result.smoothedObsDisturbance[result.smoothedObsDisturbanceOffset(t) + i] = 0.0;
                                }
                        }

                        if (doDisturbance && kPosdef > 0) {
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                        0.0, scratch, tmpRqBase, kPosdef);
                                BLAS.dgemv(BLAS.Trans.Trans, kStates, kPosdef, 1.0,
                                        scratch, tmpRqBase, kPosdef,
                                        scratch, rBase, 1,
                                        0.0, result.smoothedStateDisturbance, result.smoothedStateDisturbanceOffset(t), 1);
                        }

                        if (doDisturbanceCov) {
                                BLAS.dcopy(kEndog2, obsCov, observationModel.obsCovOffset(t), 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                                if (kPosdef > 0) {
                                        if (!doDisturbance) {
                                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                                        0.0, scratch, tmpRqBase, kPosdef);
                                        }
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                                                1.0, scratch, nBase, kStates,
                                                scratch, tmpRqBase, kPosdef,
                                                0.0, scratch, tmpNpBase, kPosdef);
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                                                -1.0, scratch, tmpRqBase, kPosdef,
                                                scratch, tmpNpBase, kPosdef,
                                                0.0, result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), kPosdef);
                                        BLAS.daxpy(kPosdef * kPosdef, 1.0, stateCovariance, stateSpace.stateCovarianceOffset(t), 1,
                                                result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), 1);
                                }
                        }

                        continue;
                }

                double[] Z = design;
                int zOff = observationModel.designOffset(t);
                int zLd = designLd;
                double[] H = obsCov;
                int hOff = observationModel.obsCovOffset(t);
                int hLd = obsCovLd;

                if (ke < kEndog) {
                        Z = scratch;
                        zOff = selectedDesignBase;
                        zLd = kStates;
                        H = scratch;
                        hOff = selectedObsCovBase;
                        hLd = ke;
                }

                int ke2 = ke * ke;
                int smoothingErrorOff = storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase;

                if (ke < kEndog) {
                        copyObservedSquareMatrix(observationModel, t,
                                fr.forecastErrorCov, foreErrCovOff, kEndog,
                                scratch, tmpFcopyBase, ke);
                } else {
                        BLAS.dcopy(ke2, fr.forecastErrorCov, foreErrCovOff, 1, scratch, tmpFcopyBase, 1);
                }

                if (ke == 1) {
                        double Ft = scratch[tmpFcopyBase];
                        double FtInv = 1.0 / Ft;
                        if (ke < kEndog) {
                                copyObservedVector(observationModel, t, fr.forecastError, foreErrOff, scratch, tmpFivBase);
                                scratch[tmpFivBase] *= FtInv;
                        } else {
                                scratch[tmpFivBase] = fr.forecastError[foreErrOff] * FtInv;
                        }
                        for (int j = 0; j < kStates; j++) {
                                scratch[tmpFiZBase + j] = Z[zOff + j] * FtInv;
                        }
                } else {
                        BLAS.dpotrf(BLAS.Uplo.Lower, ke, scratch, tmpFcopyBase, ke);
                        if (ke < kEndog) {
                                copyObservedVector(observationModel, t, fr.forecastError, foreErrOff, scratch, tmpFivBase);
                        } else {
                                BLAS.dcopy(ke, fr.forecastError, foreErrOff, 1, scratch, tmpFivBase, 1);
                        }
                        BLAS.dpotrs(BLAS.Uplo.Lower, ke, 1,
                                scratch, tmpFcopyBase, ke, scratch, tmpFivBase, 1);
                        BLAS.dcopy(ke * kStates, Z, zOff, 1, scratch, tmpFiZBase, 1);
                        BLAS.dpotrs(BLAS.Uplo.Lower, ke, kStates,
                                scratch, tmpFcopyBase, ke, scratch, tmpFiZBase, kStates);
                }

                double[] K;
                int kOff;
                int kLd;
                if (ke < kEndog) {
                        copyObservedColumns(observationModel, t, kStates,
                                fr.kalmanGain, kgOff, kEndog,
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
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                                1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                                K, kOff, kLd,
                                0.0, scratch, tmpTkBase, ke);
                        K = scratch;
                        kOff = tmpTkBase;
                        kLd = ke;
                }

                if (smoothingError != null) {
                        BLAS.dgemv(BLAS.Trans.Trans, kStates, ke, -1.0,
                                K, kOff, kLd,
                                scratch, rBase, 1,
                                0.0, smoothingError, smoothingErrorOff, 1);
                        BLAS.daxpy(ke, 1.0, scratch, tmpFivBase, 1,
                                smoothingError, smoothingErrorOff, 1);

                        if (ke < kEndog) {
                                scatterObservedVectorInPlace(observationModel, t,
                                        smoothingError, smoothingErrorOff);
                        }
                }

                BLAS.dcopy(kStates2, transition, stateSpace.transitionOffset(t), 1,
                        scratch, lBase, 1);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                        -1.0, K, kOff, kLd,
                        Z, zOff, zLd,
                        1.0, scratch, lBase, kStates);

                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates, 1.0,
                        fr.filteredStateCov, filtCovOff, kStates,
                        transition, stateSpace.transitionOffset(t), transitionLd,
                        0.0, scratch, tmpRqBase, kStates);

                if (needSmoothedState) {
                        double[] smoothedState = doState ? result.smoothedState : nextSmoothedState;
                        int smoothedStateOff = doState ? result.smoothedStateOffset(t) : 0;
                        BLAS.dcopy(kStates, fr.filteredState, filtOff, 1,
                                smoothedState, smoothedStateOff, 1);
                        BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                                scratch, tmpRqBase, kStates,
                                scratch, rBase, 1,
                                1.0, smoothedState, smoothedStateOff, 1);
                }

                if (needSmoothedStateCov) {
                        double[] smoothedStateCov = doStateCov ? result.smoothedStateCov : nextSmoothedStateCov;
                        int smoothedStateCovOff = doStateCov ? result.smoothedStateCovOffset(t) : 0;
                        if (t < nobs - 1) {
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates, 1.0,
                                        scratch, nBase, kStates,
                                        scratch, tmpRqBase, kStates,
                                        0.0, scratch, tmpNpBase, kStates);
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates, -1.0,
                                        scratch, tmpRqBase, kStates,
                                        scratch, tmpNpBase, kStates,
                                        0.0, smoothedStateCov, smoothedStateCovOff, kStates);
                                BLAS.daxpy(kStates2, 1.0, fr.filteredStateCov, filtCovOff, 1,
                                        smoothedStateCov, smoothedStateCovOff, 1);
                        } else {
                                BLAS.dcopy(kStates2, fr.filteredStateCov, filtCovOff, 1,
                                        smoothedStateCov, smoothedStateCovOff, 1);
                        }
                }

                if (storeAuxiliary) {
                        BLAS.dcopy(kStates2, scratch, lBase, 1,
                                result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                        BLAS.dcopy(kStates, scratch, rBase, 1,
                                result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                        BLAS.dcopy(kStates2, scratch, nBase, 1,
                                result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
                }

                if (doDisturbance) {
                        if (ke < kEndog) {
                                copyObservedVector(observationModel, t,
                                        smoothingError, smoothingErrorOff,
                                        scratch, tmpFivBase);
                                BLAS.dgemv(BLAS.Trans.NoTrans, ke, ke, 1.0,
                                        H, hOff, hLd,
                                        scratch, tmpFivBase, 1,
                                        0.0, scratch, tmpFiZBase, 1);
                                scatterObservedVector(observationModel, t,
                                        scratch, tmpFiZBase,
                                        result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t));
                        } else {
                                BLAS.dgemv(BLAS.Trans.NoTrans, ke, ke, 1.0,
                                        H, hOff, hLd,
                                        smoothingError, smoothingErrorOff, 1,
                                        0.0, result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t), 1);
                        }

                        if (kPosdef > 0) {
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                        0.0, scratch, tmpRqBase, kPosdef);
                                BLAS.dgemv(BLAS.Trans.Trans, kStates, kPosdef, 1.0,
                                        scratch, tmpRqBase, kPosdef,
                                        scratch, rBase, 1,
                                        0.0, result.smoothedStateDisturbance, result.smoothedStateDisturbanceOffset(t), 1);
                        }
                }

                if (doDisturbanceCov) {
                        double FtInv = ke == 1 ? 1.0 / scratch[tmpFcopyBase] : 0.0;
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, ke,
                                1.0, K, kOff, kLd,
                                H, hOff, hLd,
                                0.0, scratch, tmpKhBase, ke);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                                1.0, scratch, nBase, kStates,
                                scratch, tmpKhBase, ke,
                                0.0, scratch, tmpNpBase, ke);
                        if (ke >= kEndog) {
                                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, ke, ke, kStates,
                                        -1.0, scratch, tmpKhBase, ke,
                                        scratch, tmpNpBase, ke,
                                        0.0, result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), kEndog);
                                BLAS.daxpy(kEndog2, 1.0, H, hOff, 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                        }

                        if (ke == 1) {
                                if (ke < kEndog) {
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, ke, ke, kStates,
                                                -1.0, scratch, tmpKhBase, ke,
                                                scratch, tmpNpBase, ke,
                                                0.0, scratch, tmpFcopyBase, ke);
                                        BLAS.daxpy(ke2, 1.0, H, hOff, 1, scratch, tmpFcopyBase, 1);
                                        scratch[tmpFcopyBase] -= H[hOff] * FtInv * H[hOff];
                                } else {
                                        for (int ri = 0; ri < ke; ri++) {
                                                for (int ci = 0; ci < ke; ci++) {
                                                        int idx = result.smoothedObsDisturbanceCovOffset(t) + ri * kEndog + ci;
                                                        result.smoothedObsDisturbanceCov[idx] -=
                                                                H[hOff + ri * hLd + ci] * FtInv * H[hOff + ci * hLd + ri];
                                                }
                                        }
                                }
                        } else {
                                BLAS.dcopy(ke2, H, hOff, 1, scratch, tmpFiHBase, 1);
                                BLAS.dpotrs(BLAS.Uplo.Lower, ke, ke,
                                        scratch, tmpFcopyBase, ke, scratch, tmpFiHBase, ke);
                                if (ke < kEndog) {
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, ke, ke, kStates,
                                                -1.0, scratch, tmpKhBase, ke,
                                                scratch, tmpNpBase, ke,
                                                0.0, scratch, tmpFcopyBase, ke);
                                        BLAS.daxpy(ke2, 1.0, H, hOff, 1, scratch, tmpFcopyBase, 1);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                                                -1.0, H, hOff, hLd,
                                                scratch, tmpFiHBase, ke,
                                                1.0, scratch, tmpFcopyBase, ke);
                                } else {
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                                                -1.0, H, hOff, hLd,
                                                scratch, tmpFiHBase, ke,
                                                1.0, result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), kEndog);
                                }
                        }

                        if (ke < kEndog) {
                                BLAS.dcopy(kEndog2, obsCov, observationModel.obsCovOffset(t), 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                                scatterObservedSquareMatrix(observationModel, t,
                                        scratch, tmpFcopyBase, ke,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), kEndog);
                        }

                        if (kPosdef > 0) {
                                if (!doDisturbance) {
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                                1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                                stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                                0.0, scratch, tmpRqBase, kPosdef);
                                }
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                                        1.0, scratch, nBase, kStates,
                                        scratch, tmpRqBase, kPosdef,
                                        0.0, scratch, tmpNpBase, kPosdef);
                                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                                        -1.0, scratch, tmpRqBase, kPosdef,
                                        scratch, tmpNpBase, kPosdef,
                                        0.0, result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), kPosdef);
                                BLAS.daxpy(kPosdef * kPosdef, 1.0, stateCovariance, stateSpace.stateCovarianceOffset(t), 1,
                                        result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), 1);
                        }
                }
        }

        for (int i = 0; i < kStates; i++) scratch[rBase + i] = 0.0;
        for (int i = 0; i < kStates2; i++) scratch[nBase + i] = 0.0;
        if (storeAuxiliary) {
                BLAS.dcopy(kStates, scratch, rBase, 1,
                                result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(nobs), 1);
                BLAS.dcopy(kStates2, scratch, nBase, 1,
                                result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(nobs), 1);

                int scaledEstimatorBase = result.scaledSmoothedEstimatorBase();
                int scaledEstimatorCovBase = result.scaledSmoothedEstimatorCovBase();
                for (int i = 0; i < nobs * kStates; i++) {
                        result.scaledSmoothedEstimator[scaledEstimatorBase + i] = result.scaledSmoothedEstimator[scaledEstimatorBase + i + kStates];
                }
                for (int i = 0; i < nobs * kStates2; i++) {
                        result.scaledSmoothedEstimatorCov[scaledEstimatorCovBase + i] = result.scaledSmoothedEstimatorCov[scaledEstimatorCovBase + i + kStates2];
                }
        }

        return result.trimOutputMask(smootherOutput);
    }

        private static SmootherResult smoothAlternative(KalmanSSM stateSpace,
                                                        KalmanSSM observationModel,
                                                        FilterResult fr,
                                                        Pool pool, int smootherOutput,
                                                        boolean reuseResult,
                                                        SmootherResult target) {
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int kEndog2 = kEndog * kEndog;
        int kStates2 = kStates * kStates;
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
        boolean doStateAutocov = (smootherOutput & SMOOTHER_STATE_AUTOCOV) != 0;
        boolean doDisturbance = (smootherOutput & SMOOTHER_DISTURBANCE) != 0;
        boolean doDisturbanceCov = (smootherOutput & SMOOTHER_DISTURBANCE_COV) != 0;
        boolean storeAuxiliary = storesAuxiliary(smootherOutput);

        pool.ensureForecastFactorScratch(kEndog);
        if (doDisturbanceCov) {
                pool.ensureDisturbanceCovarianceScratch(kStates, kEndog);
        }

        SmootherResult result = prepareResult(target, pool, reuseResult,
                kEndog, kStates, kPosdef, nobs,
                doState, doStateCov, doStateAutocov, doDisturbance, doDisturbanceCov, storeAuxiliary);
        ScratchArena scratchArena = pool.scratch;
        SmootherScratchLayout layout = scratchArena.scratchLayout;
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
        int selectedDesignBase = layout.selectedDesignBase();
        int selectedObsCovBase = layout.selectedObsCovBase();
        int tmpKgBase = layout.tmpKgBase();
        int tmpTkBase = fr.perObsKalmanGain ? layout.tmpTkBase() : -1;
        int tmpKhBase = doDisturbanceCov ? layout.tmpKhBase() : -1;
        int tmpFiHBase = doDisturbanceCov ? layout.tmpFiHBase() : -1;
        int borrowedSmoothingErrorBase = (!storeAuxiliary && doDisturbance)
                ? layout.scratchSmoothingErrorBase()
                : 0;
        double[] smoothingError = (storeAuxiliary || doDisturbance)
                ? (storeAuxiliary ? result.smoothingError : scratch)
                : null;
        double[] nextScaledSmoothedEstimator = (!storeAuxiliary && doDisturbance && kPosdef > 0)
                ? new double[kStates]
                : null;
        double[] nextScaledSmoothedEstimatorCov = (!storeAuxiliary && doDisturbanceCov)
                ? new double[kStates2]
                : null;

        for (int i = 0; i < kStates; i++) scratch[hatRBase + i] = 0.0;
        for (int i = 0; i < kStates2; i++) scratch[hatNBase + i] = 0.0;

        for (int i = 0; i < kStates; i++) scratch[tildeRBase + i] = 0.0;
        for (int i = 0; i < kStates2; i++) scratch[tildeNBase + i] = 0.0;
        if (storeAuxiliary) {
                BLAS.dcopy(kStates, scratch, tildeRBase, 1,
                        result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(nobs), 1);
                BLAS.dcopy(kStates2, scratch, tildeNBase, 1,
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
                                BLAS.dcopy(kStates, scratch, tildeRBase, 1, nextScaledSmoothedEstimator, 0, 1);
                        }
                        if (nextScaledSmoothedEstimatorCov != null) {
                                BLAS.dcopy(kStates2, scratch, tildeNBase, 1, nextScaledSmoothedEstimatorCov, 0, 1);
                        }
                }

                int ke = selectObservations(observationModel, scratchArena, layout, t);

                if (doState) {
                        BLAS.dcopy(kStates, fr.filteredState, filtOff, 1,
                                result.smoothedState, result.smoothedStateOffset(t), 1);
                        BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                                fr.filteredStateCov, filtCovOff, kStates,
                                scratch, hatRBase, 1,
                                1.0, result.smoothedState, result.smoothedStateOffset(t), 1);
                }

                if (doStateCov) {
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates, 1.0,
                                scratch, hatNBase, kStates,
                                fr.filteredStateCov, filtCovOff, kStates,
                                0.0, scratch, tmpNpBase, kStates);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates, -1.0,
                                fr.filteredStateCov, filtCovOff, kStates,
                                scratch, tmpNpBase, kStates,
                                0.0, result.smoothedStateCov, result.smoothedStateCovOffset(t), kStates);
                        BLAS.daxpy(kStates2, 1.0, fr.filteredStateCov, filtCovOff, 1,
                                result.smoothedStateCov, result.smoothedStateCovOffset(t), 1);
                        symmetrizeMatrix(result.smoothedStateCov, result.smoothedStateCovOffset(t), kStates);
                }

                if (ke == 0) {
                        BLAS.dcopy(kStates, scratch, hatRBase, 1, scratch, tildeRBase, 1);
                        BLAS.dcopy(kStates2, scratch, hatNBase, 1, scratch, tildeNBase, 1);

                        BLAS.dcopy(kStates2, transition, stateSpace.transitionOffset(t), 1,
                                scratch, lBase, 1);

                        if (doDisturbance) {
                                for (int i = 0; i < kEndog; i++) {
                                        result.smoothedObsDisturbance[result.smoothedObsDisturbanceOffset(t) + i] = 0.0;
                                }
                        }

                        if (doDisturbance && kPosdef > 0) {
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                        0.0, scratch, tmpRqBase, kPosdef);
                                BLAS.dgemv(BLAS.Trans.Trans, kStates, kPosdef, 1.0,
                                        scratch, tmpRqBase, kPosdef,
                                        storeAuxiliary ? result.scaledSmoothedEstimator : nextScaledSmoothedEstimator,
                                        storeAuxiliary ? result.scaledSmoothedEstimatorOffset(t + 1) : 0, 1,
                                        0.0, result.smoothedStateDisturbance, result.smoothedStateDisturbanceOffset(t), 1);
                        }

                        if (doDisturbanceCov) {
                                BLAS.dcopy(kEndog2, obsCov, observationModel.obsCovOffset(t), 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                                if (kPosdef > 0) {
                                        if (!doDisturbance) {
                                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                                        0.0, scratch, tmpRqBase, kPosdef);
                                        }
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                                                1.0, scratch, hatNBase, kStates,
                                                scratch, tmpRqBase, kPosdef,
                                                0.0, scratch, tmpNpBase, kPosdef);
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                                                -1.0, scratch, tmpRqBase, kPosdef,
                                                scratch, tmpNpBase, kPosdef,
                                                0.0, result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), kPosdef);
                                        BLAS.daxpy(kPosdef * kPosdef, 1.0, stateCovariance, stateSpace.stateCovarianceOffset(t), 1,
                                                result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), 1);
                                }
                        }

                        if (storeAuxiliary) {
                                BLAS.dcopy(kStates2, scratch, lBase, 1,
                                        result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                                BLAS.dcopy(kStates, scratch, tildeRBase, 1,
                                        result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                                BLAS.dcopy(kStates2, scratch, tildeNBase, 1,
                                        result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
                        }

                        BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                                transition, stateSpace.transitionOffset(t), transitionLd,
                                scratch, tildeRBase, 1,
                                0.0, scratch, hatRBase, 1);
                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                                scratch, tildeNBase, kStates,
                                0.0, scratch, tmpNpBase, kStates);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, tmpNpBase, kStates,
                                transition, stateSpace.transitionOffset(t), transitionLd,
                                0.0, scratch, hatNBase, kStates);
                        continue;
                }

                double[] Z = design;
                int zOff = observationModel.designOffset(t);
                int zLd = designLd;
                double[] H = obsCov;
                int hOff = observationModel.obsCovOffset(t);
                int hLd = obsCovLd;

                if (ke < kEndog) {
                        Z = scratch;
                        zOff = selectedDesignBase;
                        zLd = kStates;
                        H = scratch;
                        hOff = selectedObsCovBase;
                        hLd = ke;
                }

                int ke2 = ke * ke;

                if (ke < kEndog) {
                        copyObservedSquareMatrix(observationModel, t,
                                fr.forecastErrorCov, foreErrCovOff, kEndog,
                                scratch, tmpFcopyBase, ke);
                } else {
                        BLAS.dcopy(ke2, fr.forecastErrorCov, foreErrCovOff, 1, scratch, tmpFcopyBase, 1);
                }

                if (ke == 1) {
                        double Ft = scratch[tmpFcopyBase];
                        double FtInv = 1.0 / Ft;
                        if (ke < kEndog) {
                                copyObservedVector(observationModel, t, fr.forecastError, foreErrOff, scratch, tmpFivBase);
                                scratch[tmpFivBase] *= FtInv;
                        } else {
                                scratch[tmpFivBase] = fr.forecastError[foreErrOff] * FtInv;
                        }
                        for (int j = 0; j < kStates; j++) {
                                scratch[tmpFiZBase + j] = Z[zOff + j] * FtInv;
                        }
                } else {
                        BLAS.dpotrf(BLAS.Uplo.Lower, ke, scratch, tmpFcopyBase, ke);
                        if (ke < kEndog) {
                                copyObservedVector(observationModel, t, fr.forecastError, foreErrOff, scratch, tmpFivBase);
                        } else {
                                BLAS.dcopy(ke, fr.forecastError, foreErrOff, 1, scratch, tmpFivBase, 1);
                        }
                        BLAS.dpotrs(BLAS.Uplo.Lower, ke, 1,
                                scratch, tmpFcopyBase, ke, scratch, tmpFivBase, 1);
                        BLAS.dcopy(ke * kStates, Z, zOff, 1, scratch, tmpFiZBase, 1);
                        BLAS.dpotrs(BLAS.Uplo.Lower, ke, kStates,
                                scratch, tmpFcopyBase, ke, scratch, tmpFiZBase, kStates);
                }

                double[] K;
                int kOff;
                int kLd;
                if (ke < kEndog) {
                        copyObservedColumns(observationModel, t, kStates,
                                fr.kalmanGain, kgOff, kEndog,
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
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                                1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                                K, kOff, kLd,
                                0.0, scratch, tmpTkBase, ke);
                        K = scratch;
                        kOff = tmpTkBase;
                        kLd = ke;
                }

                if (smoothingError != null) {
                        BLAS.dgemv(BLAS.Trans.Trans, kStates, ke, -1.0,
                                K, kOff, kLd,
                                scratch, tildeRBase, 1,
                                0.0, smoothingError, smoothingErrorOff, 1);
                        BLAS.daxpy(ke, 1.0, scratch, tmpFivBase, 1,
                                smoothingError, smoothingErrorOff, 1);

                        if (ke < kEndog) {
                                scatterObservedVectorInPlace(observationModel, t,
                                        smoothingError, smoothingErrorOff);
                        }
                }

                BLAS.dcopy(kStates2, transition, stateSpace.transitionOffset(t), 1,
                        scratch, lBase, 1);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                        -1.0, K, kOff, kLd,
                        Z, zOff, zLd,
                        1.0, scratch, lBase, kStates);

                int predCovOff = fr.predictedStateCovOffset(t);
                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                        1.0, Z, zOff, zLd,
                        scratch, tmpFiZBase, kStates,
                        0.0, scratch, tmpRqBase, kStates);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        -1.0, fr.predictedStateCov, predCovOff, kStates,
                        scratch, tmpRqBase, kStates,
                        0.0, scratch, tmpNpBase, kStates);
                for (int i = 0; i < kStates; i++) {
                        scratch[tmpNpBase + i * kStates + i] += 1.0;
                }

                BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                        scratch, tmpNpBase, kStates,
                        scratch, hatRBase, 1,
                        0.0, scratch, tildeRBase, 1);
                BLAS.dgemv(BLAS.Trans.Trans, ke, kStates, 1.0,
                        Z, zOff, zLd,
                        scratch, tmpFivBase, 1,
                        1.0, scratch, tildeRBase, 1);

                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, scratch, hatNBase, kStates,
                        scratch, tmpNpBase, kStates,
                        0.0, scratch, tmpRqBase, kStates);
                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, scratch, tmpNpBase, kStates,
                        scratch, tmpRqBase, kStates,
                        0.0, scratch, tildeNBase, kStates);
                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, ke,
                        1.0, Z, zOff, zLd,
                        scratch, tmpFiZBase, kStates,
                        1.0, scratch, tildeNBase, kStates);

                if (storeAuxiliary) {
                        BLAS.dcopy(kStates2, scratch, lBase, 1,
                                result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                        BLAS.dcopy(kStates, scratch, tildeRBase, 1,
                                result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                        BLAS.dcopy(kStates2, scratch, tildeNBase, 1,
                                result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
                }

                if (doDisturbance) {
                        if (ke < kEndog) {
                                copyObservedVector(observationModel, t,
                                        smoothingError, smoothingErrorOff,
                                        scratch, tmpFivBase);
                                BLAS.dgemv(BLAS.Trans.NoTrans, ke, ke, 1.0,
                                        H, hOff, hLd,
                                        scratch, tmpFivBase, 1,
                                        0.0, scratch, tmpFiZBase, 1);
                                scatterObservedVector(observationModel, t,
                                        scratch, tmpFiZBase,
                                        result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t));
                        } else {
                                BLAS.dgemv(BLAS.Trans.NoTrans, ke, ke, 1.0,
                                        H, hOff, hLd,
                                        smoothingError, smoothingErrorOff, 1,
                                        0.0, result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t), 1);
                        }

                        if (kPosdef > 0) {
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                        0.0, scratch, tmpRqBase, kPosdef);
                                BLAS.dgemv(BLAS.Trans.Trans, kStates, kPosdef, 1.0,
                                        scratch, tmpRqBase, kPosdef,
                                        storeAuxiliary ? result.scaledSmoothedEstimator : nextScaledSmoothedEstimator,
                                        storeAuxiliary ? result.scaledSmoothedEstimatorOffset(t + 1) : 0, 1,
                                        0.0, result.smoothedStateDisturbance, result.smoothedStateDisturbanceOffset(t), 1);
                        }
                }

                if (doDisturbanceCov) {
                        double FtInv = ke == 1 ? 1.0 / scratch[tmpFcopyBase] : 0.0;
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, ke,
                                1.0, K, kOff, kLd,
                                H, hOff, hLd,
                                0.0, scratch, tmpKhBase, ke);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, ke, kStates,
                                1.0,
                                storeAuxiliary ? result.scaledSmoothedEstimatorCov : nextScaledSmoothedEstimatorCov,
                                storeAuxiliary ? result.scaledSmoothedEstimatorCovOffset(t + 1) : 0, kStates,
                                scratch, tmpKhBase, ke,
                                0.0, scratch, tmpNpBase, ke);
                        if (ke >= kEndog) {
                                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, ke, ke, kStates,
                                        -1.0, scratch, tmpKhBase, ke,
                                        scratch, tmpNpBase, ke,
                                        0.0, result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), kEndog);
                                BLAS.daxpy(kEndog2, 1.0, H, hOff, 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                        }

                        if (ke == 1) {
                                if (ke < kEndog) {
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, ke, ke, kStates,
                                                -1.0, scratch, tmpKhBase, ke,
                                                scratch, tmpNpBase, ke,
                                                0.0, scratch, tmpFcopyBase, ke);
                                        BLAS.daxpy(ke2, 1.0, H, hOff, 1, scratch, tmpFcopyBase, 1);
                                        scratch[tmpFcopyBase] -= H[hOff] * FtInv * H[hOff];
                                } else {
                                        for (int ri = 0; ri < ke; ri++) {
                                                for (int ci = 0; ci < ke; ci++) {
                                                        int idx = result.smoothedObsDisturbanceCovOffset(t) + ri * kEndog + ci;
                                                        result.smoothedObsDisturbanceCov[idx] -=
                                                                H[hOff + ri * hLd + ci] * FtInv * H[hOff + ci * hLd + ri];
                                                }
                                        }
                                }
                        } else {
                                BLAS.dcopy(ke2, H, hOff, 1, scratch, tmpFiHBase, 1);
                                BLAS.dpotrs(BLAS.Uplo.Lower, ke, ke,
                                        scratch, tmpFcopyBase, ke, scratch, tmpFiHBase, ke);
                                if (ke < kEndog) {
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, ke, ke, kStates,
                                                -1.0, scratch, tmpKhBase, ke,
                                                scratch, tmpNpBase, ke,
                                                0.0, scratch, tmpFcopyBase, ke);
                                        BLAS.daxpy(ke2, 1.0, H, hOff, 1, scratch, tmpFcopyBase, 1);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                                                -1.0, H, hOff, hLd,
                                                scratch, tmpFiHBase, ke,
                                                1.0, scratch, tmpFcopyBase, ke);
                                } else {
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, ke, ke, ke,
                                                -1.0, H, hOff, hLd,
                                                scratch, tmpFiHBase, ke,
                                                1.0, result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), kEndog);
                                }
                        }

                        if (ke < kEndog) {
                                BLAS.dcopy(kEndog2, obsCov, observationModel.obsCovOffset(t), 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                                scatterObservedSquareMatrix(observationModel, t,
                                        scratch, tmpFcopyBase, ke,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), kEndog);
                        }

                        if (kPosdef > 0) {
                                if (!doDisturbance) {
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                                1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                                stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                                0.0, scratch, tmpRqBase, kPosdef);
                                }
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                                        1.0,
                                        storeAuxiliary ? result.scaledSmoothedEstimatorCov : nextScaledSmoothedEstimatorCov,
                                        storeAuxiliary ? result.scaledSmoothedEstimatorCovOffset(t + 1) : 0, kStates,
                                        scratch, tmpRqBase, kPosdef,
                                        0.0, scratch, tmpNpBase, kPosdef);
                                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                                        -1.0, scratch, tmpRqBase, kPosdef,
                                        scratch, tmpNpBase, kPosdef,
                                        0.0, result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), kPosdef);
                                BLAS.daxpy(kPosdef * kPosdef, 1.0, stateCovariance, stateSpace.stateCovarianceOffset(t), 1,
                                        result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), 1);
                        }
                }

                BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                        transition, stateSpace.transitionOffset(t), transitionLd,
                        scratch, tildeRBase, 1,
                        0.0, scratch, hatRBase, 1);
                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                        scratch, tildeNBase, kStates,
                        0.0, scratch, tmpNpBase, kStates);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, scratch, tmpNpBase, kStates,
                        transition, stateSpace.transitionOffset(t), transitionLd,
                        0.0, scratch, hatNBase, kStates);
        }

        if (storeAuxiliary) {
                int scaledEstimatorBase = result.scaledSmoothedEstimatorBase();
                int scaledEstimatorCovBase = result.scaledSmoothedEstimatorCovBase();
                for (int i = 0; i < nobs * kStates; i++) {
                        result.scaledSmoothedEstimator[scaledEstimatorBase + i] = result.scaledSmoothedEstimator[scaledEstimatorBase + i + kStates];
                }
                for (int i = 0; i < nobs * kStates2; i++) {
                        result.scaledSmoothedEstimatorCov[scaledEstimatorCovBase + i] = result.scaledSmoothedEstimatorCov[scaledEstimatorCovBase + i + kStates2];
                }
        }

        return result.trimOutputMask(smootherOutput);
    }

        private static void setIdentity(double[] matrix, int offset, int n) {
                for (int i = 0; i < n * n; i++) {
                        matrix[offset + i] = 0.0;
                }
                for (int i = 0; i < n; i++) {
                        matrix[offset + i * n + i] = 1.0;
                }
        }

        private static double dotProduct(double[] a, int aOffset, double[] b, int bOffset, int n) {
                return BLAS.ddot(n, a, aOffset, 1, b, bOffset, 1);
        }

        private static double quadraticForm(double[] matrix, int matrixOffset, double[] vector, int vectorOffset, int n) {
                double sum = 0.0;
                for (int row = 0; row < n; row++) {
                        double rowDot = 0.0;
                        for (int col = 0; col < n; col++) {
                                rowDot += matrix[matrixOffset + row * n + col] * vector[vectorOffset + col];
                        }
                        sum += vector[vectorOffset + row] * rowDot;
                }
                return sum;
        }

        private static SmootherResult smoothDiffuse(KalmanSSM stateSpace,
                                                    KalmanSSM observationModel,
                                                    FilterResult fr,
                                                    Pool pool, int smootherOutput,
                                                    boolean reuseResult,
                                                    SmootherResult target) {
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int kStates2 = kStates * kStates;
        int kEndog2 = kEndog * kEndog;
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
        boolean doStateAutocov = (smootherOutput & SMOOTHER_STATE_AUTOCOV) != 0;
        boolean doDisturbance = (smootherOutput & SMOOTHER_DISTURBANCE) != 0;
        boolean doDisturbanceCov = (smootherOutput & SMOOTHER_DISTURBANCE_COV) != 0;
        boolean storeAuxiliary = storesAuxiliary(smootherOutput) || doStateAutocov;

        SmootherResult result = prepareResult(target, pool, reuseResult,
                        kEndog, kStates, kPosdef, nobs,
                        doState, doStateCov, doStateAutocov, doDisturbance, doDisturbanceCov, storeAuxiliary);
        if (doStateAutocov) {
                Arrays.fill(result.smoothedStateAutocovariance, result.smoothedStateAutocovarianceBase(),
                                result.smoothedStateAutocovarianceBase() + result.smoothedStateAutocovarianceLength(), 0.0);
        }
        ScratchArena scratchArena = pool.scratch;
        SmootherScratchLayout layout = scratchArena.scratchLayout;
        double[] scratch = scratchArena.scratchBacking;
        int rBase = layout.rBase();
        int nBase = layout.nBase();
        int rPrevBase = layout.rPrevBase();
        int nPrevBase = layout.nPrevBase();
        int lBase = layout.lBase();
        int tmpNpBase = layout.tmpNpBase();
        int tmpRqBase = layout.tmpRqBase();
        int rInfBase = layout.rInfBase();
        int rInfPrevBase = layout.rInfPrevBase();
        int nInf1Base = layout.nInf1Base();
        int nInf1PrevBase = layout.nInf1PrevBase();
        int nInf2Base = layout.nInf2Base();
        int nInf2PrevBase = layout.nInf2PrevBase();
        int l0Base = layout.l0Base();
        int l1Base = layout.l1Base();
        int k0Base = layout.k0Base();
        int k1Base = layout.k1Base();
        int tmpMInfBase = layout.tmpMInfBase();
        int tmpMStarBase = layout.tmpMStarBase();
        int tmpTtNBase = layout.tmpTtNBase();
        int tmpTtNInf1Base = layout.tmpTtNInf1Base();
        int tmpTtNInf2Base = layout.tmpTtNInf2Base();
        int tmpPNInf1Base = layout.tmpPNInf1Base();
        int tmpPInfNInf2Base = layout.tmpPInfNInf2Base();
        if (storeAuxiliary) {
                        if (reuseResult) {
                                pool.ensureDiffuseResult(kStates, nobs);
                                pool.bindRetainedDiffuseAuxiliary(result);
                                Arrays.fill(result.smoothingError, 0.0);
                                Arrays.fill(result.innovationsTransition, 0.0);
                                Arrays.fill(result.scaledSmoothedDiffuseEstimator, 0.0);
                                Arrays.fill(result.scaledSmoothedDiffuse1EstimatorCov, 0.0);
                                Arrays.fill(result.scaledSmoothedDiffuse2EstimatorCov, 0.0);
                        } else {
                                result.allocateDiffuse();
                        }
        }

        for (int i = 0; i < kStates; i++) scratch[rBase + i] = 0.0;
        for (int i = 0; i < kStates2; i++) scratch[nBase + i] = 0.0;
        for (int i = 0; i < kStates; i++) scratch[rInfBase + i] = 0.0;
        for (int i = 0; i < kStates2; i++) scratch[nInf1Base + i] = 0.0;
        for (int i = 0; i < kStates2; i++) scratch[nInf2Base + i] = 0.0;

        for (int t = nobs - 1; t >= 0; t--) {
                int predOff = fr.predictedStateOffset(t);
                int predCovOff = fr.predictedStateCovOffset(t);
                int predDiffCovOff = fr.predictedDiffuseStateCovOffset(t);
                int foreErrOff = fr.forecastErrorOffset(t);
                int foreErrDiffCovOff = fr.forecastErrorDiffuseCovOffset(t);
                boolean isDiffuse = t < nobsDiffuse;

                int ke = kEndog - observationModel.missingCount(t);

                BLAS.dcopy(kStates, scratch, rBase, 1, scratch, rPrevBase, 1);
                BLAS.dcopy(kStates2, scratch, nBase, 1, scratch, nPrevBase, 1);
                if (isDiffuse) {
                        BLAS.dcopy(kStates, scratch, rInfBase, 1, scratch, rInfPrevBase, 1);
                        BLAS.dcopy(kStates2, scratch, nInf1Base, 1, scratch, nInf1PrevBase, 1);
                        BLAS.dcopy(kStates2, scratch, nInf2Base, 1, scratch, nInf2PrevBase, 1);
                }
                if (doDisturbance) {
                        for (int i = 0; i < kEndog; i++) {
                                result.smoothedObsDisturbance[result.smoothedObsDisturbanceOffset(t) + i] = 0.0;
                        }
                }
                if (doDisturbanceCov) {
                        BLAS.dcopy(kEndog2, obsCov, observationModel.obsCovOffset(t), 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                }

                if ((doDisturbance || doDisturbanceCov) && kPosdef > 0) {
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                0.0, scratch, tmpRqBase, kPosdef);
                        if (doDisturbance) {
                                BLAS.dgemv(BLAS.Trans.Trans, kStates, kPosdef, 1.0,
                                        scratch, tmpRqBase, kPosdef,
                                        scratch, rBase, 1,
                                        0.0, result.smoothedStateDisturbance, result.smoothedStateDisturbanceOffset(t), 1);
                        }
                        if (doDisturbanceCov) {
                                BLAS.dcopy(kPosdef * kPosdef, stateCovariance, stateSpace.stateCovarianceOffset(t), 1,
                                        result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), 1);
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                                        1.0, scratch, nBase, kStates,
                                        scratch, tmpRqBase, kPosdef,
                                        0.0, scratch, tmpPNInf1Base, kPosdef);
                                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                                        -1.0, scratch, tmpRqBase, kPosdef,
                                        scratch, tmpPNInf1Base, kPosdef,
                                        1.0, result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), kPosdef);
                        }
                }

                if (ke == 0) {
                        if (!doStateAutocov || isDiffuse) {
                                BLAS.dcopy(kStates2, transition, stateSpace.transitionOffset(t), 1,
                                        scratch, l0Base, 1);
                        }
                } else {
                        for (int i = kEndog - 1; i >= 0; i--) {
                                if (observationModel.isMissing(i, t)) {
                                        continue;
                                }

                                int zRowOff = observationModel.designOffset(t) + i * designLd;
                                int obsCovOff = observationModel.obsCovOffset(t);
                                double vi = fr.forecastError[foreErrOff + i];
                                double hii = obsCov[obsCovOff + i * obsCovLd + i];

                                for (int j = 0; j < kStates; j++) {
                                        double mStar = 0.0;
                                        for (int k = 0; k < kStates; k++) {
                                                mStar += fr.predictedStateCov[predCovOff + j * kStates + k]
                                                        * design[zRowOff + k];
                                        }
                                        scratch[tmpMStarBase + j] = mStar;
                                }

                                double Fstar = hii;
                                for (int j = 0; j < kStates; j++) {
                                        Fstar += design[zRowOff + j] * scratch[tmpMStarBase + j];
                                }

                                double Finf = 0.0;
                                if (isDiffuse) {
                                        for (int j = 0; j < kStates; j++) {
                                                double mInf = 0.0;
                                                for (int k = 0; k < kStates; k++) {
                                                        mInf += fr.predictedDiffuseStateCov[predDiffCovOff + j * kStates + k]
                                                                * design[zRowOff + k];
                                                }
                                                scratch[tmpMInfBase + j] = mInf;
                                        }
                                        Finf = fr.forecastErrorDiffuseCov[foreErrDiffCovOff + i * kEndog + i];
                                }

                                if (isDiffuse && Finf > toleranceDiffuse) {
                                        double F1 = 1.0 / Finf;
                                        double F2 = -Fstar * F1 * F1;

                                        for (int j = 0; j < kStates; j++) {
                                                scratch[k0Base + j] = scratch[tmpMInfBase + j] * F1;
                                                scratch[k1Base + j] = scratch[tmpMStarBase + j] * F1 + scratch[tmpMInfBase + j] * F2;
                                        }

                                        setIdentity(scratch, l0Base, kStates);
                                        for (int j = 0; j < kStates2; j++) {
                                                scratch[l1Base + j] = 0.0;
                                        }
                                        for (int row = 0; row < kStates; row++) {
                                                for (int col = 0; col < kStates; col++) {
                                                        scratch[l0Base + row * kStates + col] -= scratch[k0Base + row] * design[zRowOff + col];
                                                        scratch[l1Base + row * kStates + col] -= scratch[k1Base + row] * design[zRowOff + col];
                                                }
                                        }

                                        if (doDisturbance && hii != 0.0) {
                                                result.smoothedObsDisturbance[result.smoothedObsDisturbanceOffset(t) + i] =
                                                        -hii * dotProduct(scratch, k0Base, scratch, rPrevBase, kStates);
                                        }
                                        if (doDisturbanceCov) {
                                                result.smoothedObsDisturbanceCov[result.smoothedObsDisturbanceCovOffset(t) + i * kEndog + i] =
                                                        hii * (1.0 - hii * quadraticForm(scratch, nPrevBase, scratch, k0Base, kStates));
                                        }

                                        BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                                                scratch, l0Base, kStates,
                                                scratch, rInfPrevBase, 1,
                                                0.0, scratch, rInfBase, 1);
                                        for (int j = 0; j < kStates; j++) {
                                                scratch[rInfBase + j] += design[zRowOff + j] * vi * F1;
                                        }
                                        BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                                                scratch, l1Base, kStates,
                                                scratch, rPrevBase, 1,
                                                0.0, scratch, tmpMStarBase, 1);
                                        for (int j = 0; j < kStates; j++) {
                                                scratch[rInfBase + j] += scratch[tmpMStarBase + j];
                                        }

                                        BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                                                scratch, l0Base, kStates,
                                                scratch, rPrevBase, 1,
                                                0.0, scratch, rBase, 1);

                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, l0Base, kStates,
                                                scratch, nInf2PrevBase, kStates,
                                                0.0, scratch, tmpTtNInf2Base, kStates);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, tmpTtNInf2Base, kStates,
                                                scratch, l0Base, kStates,
                                                0.0, scratch, nInf2Base, kStates);

                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, l0Base, kStates,
                                                scratch, nInf1PrevBase, kStates,
                                                0.0, scratch, tmpTtNInf2Base, kStates);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, tmpTtNInf2Base, kStates,
                                                scratch, l1Base, kStates,
                                                0.0, scratch, tmpPInfNInf2Base, kStates);
                                        for (int j = 0; j < kStates2; j++) {
                                                scratch[nInf2Base + j] += scratch[tmpPInfNInf2Base + j];
                                        }

                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.Trans, kStates, kStates, kStates,
                                                1.0, scratch, l1Base, kStates,
                                                scratch, nInf1PrevBase, kStates,
                                                0.0, scratch, tmpTtNInf2Base, kStates);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, tmpTtNInf2Base, kStates,
                                                scratch, l0Base, kStates,
                                                0.0, scratch, tmpPInfNInf2Base, kStates);
                                        for (int j = 0; j < kStates2; j++) {
                                                scratch[nInf2Base + j] += scratch[tmpPInfNInf2Base + j];
                                        }

                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, l1Base, kStates,
                                                scratch, nPrevBase, kStates,
                                                0.0, scratch, tmpTtNInf2Base, kStates);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, tmpTtNInf2Base, kStates,
                                                scratch, l1Base, kStates,
                                                0.0, scratch, tmpPInfNInf2Base, kStates);
                                        for (int j = 0; j < kStates2; j++) {
                                                scratch[nInf2Base + j] += scratch[tmpPInfNInf2Base + j];
                                        }
                                        BLAS.dger(kStates, kStates, F2,
                                                design, zRowOff, 1,
                                                design, zRowOff, 1,
                                                scratch, nInf2Base, kStates);

                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, l0Base, kStates,
                                                scratch, nInf1PrevBase, kStates,
                                                0.0, scratch, tmpTtNInf1Base, kStates);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, tmpTtNInf1Base, kStates,
                                                scratch, l0Base, kStates,
                                                0.0, scratch, nInf1Base, kStates);
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, l1Base, kStates,
                                                scratch, nPrevBase, kStates,
                                                0.0, scratch, tmpTtNInf1Base, kStates);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, tmpTtNInf1Base, kStates,
                                                scratch, l0Base, kStates,
                                                0.0, scratch, tmpPInfNInf2Base, kStates);
                                        for (int j = 0; j < kStates2; j++) {
                                                scratch[nInf1Base + j] += scratch[tmpPInfNInf2Base + j];
                                        }
                                        BLAS.dger(kStates, kStates, F1,
                                                design, zRowOff, 1,
                                                design, zRowOff, 1,
                                                scratch, nInf1Base, kStates);

                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, l0Base, kStates,
                                                scratch, nPrevBase, kStates,
                                                0.0, scratch, tmpTtNBase, kStates);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, tmpTtNBase, kStates,
                                                scratch, l0Base, kStates,
                                                0.0, scratch, nBase, kStates);

                                        int tmpRBase = rPrevBase;
                                        rPrevBase = rBase;
                                        rBase = tmpRBase;
                                        int tmpNBase = nPrevBase;
                                        nPrevBase = nBase;
                                        nBase = tmpNBase;
                                        int tmpRInfBase = rInfPrevBase;
                                        rInfPrevBase = rInfBase;
                                        rInfBase = tmpRInfBase;
                                        int tmpNInf1Base = nInf1PrevBase;
                                        nInf1PrevBase = nInf1Base;
                                        nInf1Base = tmpNInf1Base;
                                        int tmpNInf2Base = nInf2PrevBase;
                                        nInf2PrevBase = nInf2Base;
                                        nInf2Base = tmpNInf2Base;
                                } else if (Fstar > 0.0) {
                                        double FstarInv = 1.0 / Fstar;

                                        for (int j = 0; j < kStates; j++) {
                                                scratch[k0Base + j] = scratch[tmpMStarBase + j] * FstarInv;
                                        }

                                        setIdentity(scratch, l0Base, kStates);
                                        for (int row = 0; row < kStates; row++) {
                                                for (int col = 0; col < kStates; col++) {
                                                        scratch[l0Base + row * kStates + col] -= scratch[k0Base + row] * design[zRowOff + col];
                                                }
                                        }

                                        if (doDisturbance && hii != 0.0) {
                                                result.smoothedObsDisturbance[result.smoothedObsDisturbanceOffset(t) + i] =
                                                        hii * (vi * FstarInv - dotProduct(scratch, k0Base, scratch, rPrevBase, kStates));
                                        }
                                        if (doDisturbanceCov) {
                                                result.smoothedObsDisturbanceCov[result.smoothedObsDisturbanceCovOffset(t) + i * kEndog + i] =
                                                        hii * (1.0 - hii * (FstarInv + quadraticForm(scratch, nPrevBase, scratch, k0Base, kStates)));
                                        }

                                        BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                                                scratch, l0Base, kStates,
                                                scratch, rPrevBase, 1,
                                                0.0, scratch, rBase, 1);
                                        for (int j = 0; j < kStates; j++) {
                                                scratch[rBase + j] += design[zRowOff + j] * vi * FstarInv;
                                        }

                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, l0Base, kStates,
                                                scratch, nPrevBase, kStates,
                                                0.0, scratch, tmpTtNBase, kStates);
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                1.0, scratch, tmpTtNBase, kStates,
                                                scratch, l0Base, kStates,
                                                0.0, scratch, nBase, kStates);
                                        BLAS.dger(kStates, kStates, FstarInv,
                                                design, zRowOff, 1,
                                                design, zRowOff, 1,
                                                scratch, nBase, kStates);

                                        if (isDiffuse) {
                                                System.arraycopy(scratch, rInfPrevBase, scratch, rInfBase, kStates);
                                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                                        1.0, scratch, nInf1PrevBase, kStates,
                                                        scratch, l0Base, kStates,
                                                        0.0, scratch, nInf1Base, kStates);
                                                System.arraycopy(scratch, nInf2PrevBase, scratch, nInf2Base, kStates2);
                                        }

                                        int tmpRBase = rPrevBase;
                                        rPrevBase = rBase;
                                        rBase = tmpRBase;
                                        int tmpNBase = nPrevBase;
                                        nPrevBase = nBase;
                                        nBase = tmpNBase;
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

                        BLAS.dcopy(kStates2, transition, stateSpace.transitionOffset(t), 1,
                                scratch, l0Base, 1);
                }

                if (storeAuxiliary) {
                        BLAS.dcopy(kStates, scratch, rPrevBase, 1,
                                result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t + 1), 1);
                        BLAS.dcopy(kStates2, scratch, nPrevBase, 1,
                                result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t + 1), 1);
                        if (isDiffuse) {
                                BLAS.dcopy(kStates, scratch, rInfPrevBase, 1,
                                        result.scaledSmoothedDiffuseEstimator, result.scaledSmoothedDiffuseEstimatorOffset(t + 1), 1);
                                BLAS.dcopy(kStates2, scratch, nInf1PrevBase, 1,
                                        result.scaledSmoothedDiffuse1EstimatorCov, result.scaledSmoothedDiffuse1EstimatorCovOffset(t + 1), 1);
                                BLAS.dcopy(kStates2, scratch, nInf2PrevBase, 1,
                                        result.scaledSmoothedDiffuse2EstimatorCov, result.scaledSmoothedDiffuse2EstimatorCovOffset(t + 1), 1);
                        }
                }

                if (doState) {
                        BLAS.dcopy(kStates, fr.predictedState, predOff, 1,
                                result.smoothedState, result.smoothedStateOffset(t), 1);
                        BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                                fr.predictedStateCov, predCovOff, kStates,
                                scratch, rPrevBase, 1,
                                1.0, result.smoothedState, result.smoothedStateOffset(t), 1);
                        if (isDiffuse) {
                                BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                                        fr.predictedDiffuseStateCov, predDiffCovOff, kStates,
                                        scratch, rInfPrevBase, 1,
                                        1.0, result.smoothedState, result.smoothedStateOffset(t), 1);
                        }

                        if (isDiffuse && doDisturbance && t + 1 < nobs && kPosdef == kStates) {
                                boolean selectionIdentity = true;
                                int selOff = stateSpace.selectionOffset(t);
                                for (int row = 0; row < kStates && selectionIdentity; row++) {
                                        for (int col = 0; col < kStates; col++) {
                                                double expected = row == col ? 1.0 : 0.0;
                                                if (Math.abs(selection[selOff + row * selectionLd + col] - expected) > 1e-12) {
                                                        selectionIdentity = false;
                                                        break;
                                                }
                                        }
                                }
                                if (selectionIdentity) {
                                        int transOff = stateSpace.transitionOffset(t);
                                        for (int row = 0; row < kStates; row++) {
                                                double eta = result.smoothedState[result.smoothedStateOffset(t + 1) + row];
                                                for (int col = 0; col < kStates; col++) {
                                                        eta -= transition[transOff + row * transitionLd + col]
                                                                        * result.smoothedState[result.smoothedStateOffset(t) + col];
                                                }
                                                result.smoothedStateDisturbance[result.smoothedStateDisturbanceOffset(t) + row] = eta;
                                        }
                                }
                        }
                }

                if (doStateCov) {
                        BLAS.dcopy(kStates2, fr.predictedStateCov, predCovOff, 1,
                                result.smoothedStateCov, result.smoothedStateCovOffset(t), 1);

                        BLAS.dcopy(kStates2, fr.predictedStateCov, predCovOff, 1,
                                scratch, tmpNpBase, 1);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, tmpNpBase, kStates,
                                scratch, nPrevBase, kStates,
                                0.0, scratch, tmpTtNBase, kStates);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, tmpTtNBase, kStates,
                                scratch, tmpNpBase, kStates,
                                0.0, scratch, tmpPNInf1Base, kStates);
                        for (int i = 0; i < kStates2; i++) {
                                result.smoothedStateCov[result.smoothedStateCovOffset(t) + i] -= scratch[tmpPNInf1Base + i];
                        }

                        if (isDiffuse) {
                                BLAS.dcopy(kStates2, fr.predictedDiffuseStateCov, predDiffCovOff, 1,
                                        scratch, tmpPInfNInf2Base, 1);

                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                        1.0, scratch, tmpPInfNInf2Base, kStates,
                                        scratch, nInf1PrevBase, kStates,
                                        0.0, scratch, tmpTtNBase, kStates);
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                        1.0, scratch, tmpTtNBase, kStates,
                                        scratch, tmpNpBase, kStates,
                                        0.0, scratch, tmpPNInf1Base, kStates);
                                for (int i = 0; i < kStates2; i++) {
                                        result.smoothedStateCov[result.smoothedStateCovOffset(t) + i] -= scratch[tmpPNInf1Base + i];
                                }

                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kStates, kStates,
                                        1.0, scratch, tmpNpBase, kStates,
                                        scratch, nInf1PrevBase, kStates,
                                        0.0, scratch, tmpTtNBase, kStates);
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                        1.0, scratch, tmpTtNBase, kStates,
                                        scratch, tmpPInfNInf2Base, kStates,
                                        0.0, scratch, tmpPNInf1Base, kStates);
                                for (int i = 0; i < kStates2; i++) {
                                        result.smoothedStateCov[result.smoothedStateCovOffset(t) + i] -= scratch[tmpPNInf1Base + i];
                                }

                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                        1.0, scratch, tmpPInfNInf2Base, kStates,
                                        scratch, nInf2PrevBase, kStates,
                                        0.0, scratch, tmpTtNBase, kStates);
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                        1.0, scratch, tmpTtNBase, kStates,
                                        scratch, tmpPInfNInf2Base, kStates,
                                        0.0, scratch, tmpPNInf1Base, kStates);
                                for (int i = 0; i < kStates2; i++) {
                                        result.smoothedStateCov[result.smoothedStateCovOffset(t) + i] -= scratch[tmpPNInf1Base + i];
                                }
                        }

                        for (int i = 0; i < kStates; i++) {
                                for (int j = i + 1; j < kStates; j++) {
                                        int smoothedStateCovOff = result.smoothedStateCovOffset(t);
                                        double avg = (result.smoothedStateCov[smoothedStateCovOff + i * kStates + j]
                                                        + result.smoothedStateCov[smoothedStateCovOff + j * kStates + i]) * 0.5;
                                        result.smoothedStateCov[smoothedStateCovOff + i * kStates + j] = avg;
                                        result.smoothedStateCov[smoothedStateCovOff + j * kStates + i] = avg;
                                }
                        }
                }

                        BLAS.dcopy(kStates2, transition, stateSpace.transitionOffset(t), 1, scratch, lBase, 1);

                BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                        scratch, lBase, kStates,
                        scratch, rPrevBase, 1,
                        0.0, scratch, rBase, 1);
                System.arraycopy(scratch, rBase, scratch, rPrevBase, kStates);

                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, scratch, lBase, kStates,
                        scratch, nPrevBase, kStates,
                        0.0, scratch, tmpTtNBase, kStates);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, scratch, tmpTtNBase, kStates,
                        scratch, lBase, kStates,
                        0.0, scratch, nBase, kStates);
                System.arraycopy(scratch, nBase, scratch, nPrevBase, kStates2);

                if (isDiffuse) {
                        BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                                scratch, lBase, kStates,
                                scratch, rInfPrevBase, 1,
                                0.0, scratch, rInfBase, 1);
                        System.arraycopy(scratch, rInfBase, scratch, rInfPrevBase, kStates);

                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, lBase, kStates,
                                scratch, nInf1PrevBase, kStates,
                                0.0, scratch, tmpTtNInf1Base, kStates);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, tmpTtNInf1Base, kStates,
                                scratch, lBase, kStates,
                                0.0, scratch, nInf1Base, kStates);
                        System.arraycopy(scratch, nInf1Base, scratch, nInf1PrevBase, kStates2);

                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, lBase, kStates,
                                scratch, nInf2PrevBase, kStates,
                                0.0, scratch, tmpTtNInf2Base, kStates);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, tmpTtNInf2Base, kStates,
                                scratch, lBase, kStates,
                                0.0, scratch, nInf2Base, kStates);
                        System.arraycopy(scratch, nInf2Base, scratch, nInf2PrevBase, kStates2);
                }
        }

        if (storeAuxiliary) {
                int scaledEstimatorBase = result.scaledSmoothedEstimatorBase();
                int scaledEstimatorCovBase = result.scaledSmoothedEstimatorCovBase();
                int scaledDiffuseBase = result.scaledSmoothedDiffuseEstimatorBase();
                int scaledDiffuse1Base = result.scaledSmoothedDiffuse1EstimatorCovBase();
                int scaledDiffuse2Base = result.scaledSmoothedDiffuse2EstimatorCovBase();
                Arrays.fill(result.scaledSmoothedEstimator,
                        result.scaledSmoothedEstimatorOffset(nobs),
                        result.scaledSmoothedEstimatorOffset(nobs) + kStates, 0.0);
                Arrays.fill(result.scaledSmoothedEstimatorCov,
                        result.scaledSmoothedEstimatorCovOffset(nobs),
                        result.scaledSmoothedEstimatorCovOffset(nobs) + kStates2, 0.0);
                Arrays.fill(result.scaledSmoothedDiffuseEstimator,
                        result.scaledSmoothedDiffuseEstimatorOffset(nobs),
                        result.scaledSmoothedDiffuseEstimatorOffset(nobs) + kStates, 0.0);
                Arrays.fill(result.scaledSmoothedDiffuse1EstimatorCov,
                        result.scaledSmoothedDiffuse1EstimatorCovOffset(nobs),
                        result.scaledSmoothedDiffuse1EstimatorCovOffset(nobs) + kStates2, 0.0);
                Arrays.fill(result.scaledSmoothedDiffuse2EstimatorCov,
                        result.scaledSmoothedDiffuse2EstimatorCovOffset(nobs),
                        result.scaledSmoothedDiffuse2EstimatorCovOffset(nobs) + kStates2, 0.0);
                for (int i = 0; i < nobs * kStates; i++) {
                        result.scaledSmoothedEstimator[scaledEstimatorBase + i] = result.scaledSmoothedEstimator[scaledEstimatorBase + i + kStates];
                }
                for (int i = 0; i < nobs * kStates2; i++) {
                        result.scaledSmoothedEstimatorCov[scaledEstimatorCovBase + i] = result.scaledSmoothedEstimatorCov[scaledEstimatorCovBase + i + kStates2];
                }
                for (int i = 0; i < nobs * kStates; i++) {
                        result.scaledSmoothedDiffuseEstimator[scaledDiffuseBase + i] = result.scaledSmoothedDiffuseEstimator[scaledDiffuseBase + i + kStates];
                }
                for (int i = 0; i < nobs * kStates2; i++) {
                        result.scaledSmoothedDiffuse1EstimatorCov[scaledDiffuse1Base + i] = result.scaledSmoothedDiffuse1EstimatorCov[scaledDiffuse1Base + i + kStates2];
                }
                for (int i = 0; i < nobs * kStates2; i++) {
                        result.scaledSmoothedDiffuse2EstimatorCov[scaledDiffuse2Base + i] = result.scaledSmoothedDiffuse2EstimatorCov[scaledDiffuse2Base + i + kStates2];
                }
        }

        if (doStateAutocov) {
                computeDiffuseStateAutocovarianceTail(stateSpace, observationModel, fr, result, scratch,
                        l0Base, l1Base, tmpTtNBase, tmpNpBase);
        }

        return result.trimOutputMask(smootherOutput);
    }

        private static SmootherResult smoothUnivariate(KalmanSSM stateSpace,
                                                       KalmanSSM observationModel,
                                                       FilterResult fr,
                                                       Pool pool, int smootherOutput,
                                                       boolean reuseResult,
                                                       SmootherResult target) {
        int kEndog = observationModel.observationDimension();
        int kStates = stateSpace.stateCount();
        int kPosdef = stateSpace.stateDisturbanceCount();
        int nobs = observationModel.observationCount();
        int kEndog2 = kEndog * kEndog;
        int kStates2 = kStates * kStates;
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
        boolean doStateAutocov = (smootherOutput & SMOOTHER_STATE_AUTOCOV) != 0;
        boolean doDisturbance = (smootherOutput & SMOOTHER_DISTURBANCE) != 0;
        boolean doDisturbanceCov = (smootherOutput & SMOOTHER_DISTURBANCE_COV) != 0;
        boolean storeAuxiliary = storesAuxiliary(smootherOutput);
        boolean needInnovationsTransition = storeAuxiliary || doStateAutocov;

        if (doDisturbanceCov) {
                pool.ensureDisturbanceCovarianceScratch(kStates, kEndog);
        }
        pool.ensureObservationFactorScratch(kEndog, kStates);

        SmootherResult result = prepareResult(target, pool, reuseResult,
                kEndog, kStates, kPosdef, nobs,
                doState, doStateCov, doStateAutocov, doDisturbance, doDisturbanceCov, storeAuxiliary);
        ScratchArena scratchArena = pool.scratch;
        SmootherScratchLayout layout = scratchArena.scratchLayout;
        double[] scratch = scratchArena.scratchBacking;
        int rBase = layout.rBase();
        int nBase = layout.nBase();
        int rPrevBase = layout.rPrevBase();
        int nPrevBase = layout.nPrevBase();
        int lBase = layout.lBase();
        int tmpNpBase = layout.tmpNpBase();
        int tmpRqBase = layout.tmpRqBase();
        int tmpFiZBase = layout.tmpFiZBase();
        int tmpKgBase = layout.tmpKgBase();
        int tmpKhBase = doDisturbanceCov ? layout.tmpKhBase() : -1;
        int borrowedSmoothingErrorBase = (!storeAuxiliary && doDisturbance)
                ? layout.scratchSmoothingErrorBase()
                : 0;
        double[] smoothingError = (storeAuxiliary || doDisturbance)
                ? (storeAuxiliary ? result.smoothingError : scratch)
                : null;
        double[] nextScaledSmoothedEstimator = (!storeAuxiliary && doDisturbance && kPosdef > 0)
                ? new double[kStates]
                : null;
        double[] nextScaledSmoothedEstimatorCov = (!storeAuxiliary && doDisturbanceCov)
                ? new double[kStates2]
                : null;
        double[] autocovInputN = doStateAutocov ? new double[kStates2] : null;
        double[] autocovMeasurementTransition = needInnovationsTransition ? new double[kStates2] : null;
        double[] autocovWork = needInnovationsTransition ? new double[kStates2] : null;

        for (int i = 0; i < kStates; i++) scratch[rBase + i] = 0.0;
        for (int i = 0; i < kStates2; i++) scratch[nBase + i] = 0.0;

        for (int t = nobs - 1; t >= 0; t--) {
                int predOff = fr.predictedStateOffset(t);
                int predCovOff = fr.predictedStateCovOffset(t);
                int foreErrOff = fr.forecastErrorOffset(t);
                int foreErrCovOff = fr.forecastErrorCovOffset(t);
                int kgOff = fr.kalmanGainOffset(t);

                int ke = kEndog - observationModel.missingCount(t);

                if (ke == 0) {
                        BLAS.dcopy(kStates2, transition, stateSpace.transitionOffset(t), 1,
                                scratch, lBase, 1);

                        if (doState) {
                                BLAS.dcopy(kStates, fr.predictedState, predOff, 1,
                                        result.smoothedState, result.smoothedStateOffset(t), 1);
                                BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                                        fr.predictedStateCov, predCovOff, kStates,
                                        scratch, rBase, 1,
                                        1.0, result.smoothedState, result.smoothedStateOffset(t), 1);
                        }
                        if (doStateCov) {
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                        -1.0, fr.predictedStateCov, predCovOff, kStates,
                                        scratch, nBase, kStates,
                                        0.0, scratch, tmpNpBase, kStates);
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                        1.0, scratch, tmpNpBase, kStates,
                                        fr.predictedStateCov, predCovOff, kStates,
                                        0.0, result.smoothedStateCov, result.smoothedStateCovOffset(t), kStates);
                                BLAS.daxpy(kStates2, 1.0, fr.predictedStateCov, predCovOff, 1,
                                        result.smoothedStateCov, result.smoothedStateCovOffset(t), 1);
                        }

                        if (doStateAutocov) {
                                computeConventionalStateAutocovariance(fr, result, t, kStates,
                                        scratch, lBase,
                                        autocovInputN, 0,
                                        scratch, tmpNpBase);
                                BLAS.dcopy(kStates2, scratch, nBase, 1,
                                        autocovInputN, 0, 1);
                        }

                        if (storeAuxiliary) {
                                for (int i = 0; i < kEndog; i++) {
                                        result.smoothingError[result.smoothingErrorOffset(t) + i] = 0.0;
                                }
                                BLAS.dcopy(kStates2, scratch, lBase, 1,
                                        result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                                BLAS.dcopy(kStates, scratch, rBase, 1,
                                        result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                                BLAS.dcopy(kStates2, scratch, nBase, 1,
                                        result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
                        }

                        if (doDisturbance) {
                                for (int i = 0; i < kEndog; i++) {
                                        result.smoothedObsDisturbance[result.smoothedObsDisturbanceOffset(t) + i] = 0.0;
                                }
                        }

                        BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                                transition, stateSpace.transitionOffset(t), transitionLd,
                                scratch, rBase, 1,
                                0.0, scratch, rPrevBase, 1);
                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                                scratch, nBase, kStates,
                                0.0, scratch, tmpNpBase, kStates);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, tmpNpBase, kStates,
                                transition, stateSpace.transitionOffset(t), transitionLd,
                                0.0, scratch, nPrevBase, kStates);

                        if (doDisturbance && kPosdef > 0) {
                                double[] nextScaledSmoothedEstimatorSource = storeAuxiliary ? result.scaledSmoothedEstimator : nextScaledSmoothedEstimator;
                                int nextScaledSmoothedEstimatorOff = storeAuxiliary ? result.scaledSmoothedEstimatorOffset(t + 1) : 0;
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                        0.0, scratch, tmpRqBase, kPosdef);
                                BLAS.dgemv(BLAS.Trans.Trans, kStates, kPosdef, 1.0,
                                        scratch, tmpRqBase, kPosdef,
                                        nextScaledSmoothedEstimatorSource, nextScaledSmoothedEstimatorOff, 1,
                                        0.0, result.smoothedStateDisturbance, result.smoothedStateDisturbanceOffset(t), 1);
                        }

                        if (doDisturbanceCov) {
                                double[] nextScaledSmoothedEstimatorCovSource = storeAuxiliary ? result.scaledSmoothedEstimatorCov : nextScaledSmoothedEstimatorCov;
                                int nextScaledSmoothedEstimatorCovOff = storeAuxiliary ? result.scaledSmoothedEstimatorCovOffset(t + 1) : 0;
                                BLAS.dcopy(kEndog2, obsCov, observationModel.obsCovOffset(t), 1,
                                        result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);
                                if (kPosdef > 0) {
                                        if (!doDisturbance) {
                                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                                        0.0, scratch, tmpRqBase, kPosdef);
                                        }
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                                                1.0, nextScaledSmoothedEstimatorCovSource, nextScaledSmoothedEstimatorCovOff, kStates,
                                                scratch, tmpRqBase, kPosdef,
                                                0.0, scratch, tmpNpBase, kPosdef);
                                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                                                -1.0, scratch, tmpRqBase, kPosdef,
                                                scratch, tmpNpBase, kPosdef,
                                                0.0, result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), kPosdef);
                                        BLAS.daxpy(kPosdef * kPosdef, 1.0, stateCovariance, stateSpace.stateCovarianceOffset(t), 1,
                                                result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), 1);
                                }
                        }

                        if (!storeAuxiliary) {
                                if (nextScaledSmoothedEstimator != null) {
                                        BLAS.dcopy(kStates, scratch, rBase, 1, nextScaledSmoothedEstimator, 0, 1);
                                }
                                if (nextScaledSmoothedEstimatorCov != null) {
                                        BLAS.dcopy(kStates2, scratch, nBase, 1, nextScaledSmoothedEstimatorCov, 0, 1);
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

                if (needInnovationsTransition) {
                        setIdentity(autocovMeasurementTransition, 0, kStates);
                }

                if (smoothingError != null) {
                        int smoothingErrorOff = storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase;
                        for (int i = 0; i < kEndog; i++) smoothingError[smoothingErrorOff + i] = 0.0;
                }

                for (int i = kEndog - 1; i >= 0; i--) {
                        if (observationModel.isMissing(i, t)) continue;

                        int zRowOff = observationModel.designOffset(t) + i * designLd;
                        double vi = fr.forecastError[foreErrOff + i];
                        double Fi = fr.forecastErrorCov[foreErrCovOff + i * kEndog + i];
                        if (Fi <= 0.0) continue;
                        double FiInv = 1.0 / Fi;

                        for (int j = 0; j < kStates; j++) {
                                        scratch[tmpKgBase + j] = fr.kalmanGain[kgOff + j * kEndog + i];
                        }

                        for (int j = 0; j < kStates; j++) {
                                for (int k = 0; k < kStates; k++) {
                                        scratch[lBase + j * kStates + k] = (j == k) ? 1.0 : 0.0;
                                }
                                for (int k = 0; k < kStates; k++) {
                                                scratch[lBase + j * kStates + k] -= scratch[tmpKgBase + j] * design[zRowOff + k];
                                }
                        }

                        if (needInnovationsTransition) {
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans,
                                        kStates, kStates, kStates,
                                        1.0, autocovMeasurementTransition, 0, kStates,
                                        scratch, lBase, kStates,
                                        0.0, autocovWork, 0, kStates);
                                BLAS.dcopy(kStates2, autocovWork, 0, 1,
                                        autocovMeasurementTransition, 0, 1);
                        }

                        if (smoothingError != null) {
                                int smoothingErrorOff = storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase;
                                smoothingError[smoothingErrorOff + i] = vi * FiInv;
                                for (int j = 0; j < kStates; j++) {
                                        smoothingError[smoothingErrorOff + i] -= scratch[tmpKgBase + j] * scratch[rBase + j];
                                }
                        }

                        for (int j = 0; j < kStates; j++) {
                                scratch[tmpFiZBase + j] = design[zRowOff + j] * FiInv;
                        }

                        BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                                scratch, lBase, kStates,
                                scratch, rBase, 1,
                                0.0, scratch, rPrevBase, 1);
                        BLAS.daxpy(kStates, vi, scratch, tmpFiZBase, 1, scratch, rPrevBase, 1);

                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, nBase, kStates,
                                scratch, lBase, kStates,
                                0.0, scratch, tmpNpBase, kStates);
                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, lBase, kStates,
                                scratch, tmpNpBase, kStates,
                                0.0, scratch, nPrevBase, kStates);
                        for (int j = 0; j < kStates; j++) {
                                for (int k = 0; k < kStates; k++) {
                                        scratch[nPrevBase + j * kStates + k] += scratch[tmpFiZBase + j] * design[zRowOff + k];
                                }
                        }

                        int tmpRBase = rBase;
                        rBase = rPrevBase;
                        rPrevBase = tmpRBase;
                        int tmpNBase = nBase;
                        nBase = nPrevBase;
                        nPrevBase = tmpNBase;
                }

                if (doState) {
                        BLAS.dcopy(kStates, fr.predictedState, predOff, 1,
                                result.smoothedState, result.smoothedStateOffset(t), 1);
                        BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
                                fr.predictedStateCov, predCovOff, kStates,
                                scratch, rBase, 1,
                                1.0, result.smoothedState, result.smoothedStateOffset(t), 1);
                }

                if (doStateCov) {
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                -1.0, fr.predictedStateCov, predCovOff, kStates,
                                scratch, nBase, kStates,
                                0.0, scratch, tmpNpBase, kStates);
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                                1.0, scratch, tmpNpBase, kStates,
                                fr.predictedStateCov, predCovOff, kStates,
                                0.0, result.smoothedStateCov, result.smoothedStateCovOffset(t), kStates);
                        BLAS.daxpy(kStates2, 1.0, fr.predictedStateCov, predCovOff, 1,
                                result.smoothedStateCov, result.smoothedStateCovOffset(t), 1);
                }

                if (needInnovationsTransition) {
                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans,
                                kStates, kStates, kStates,
                                1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                                autocovMeasurementTransition, 0, kStates,
                                0.0, scratch, lBase, kStates);
                }
                if (doStateAutocov) {
                        computeConventionalStateAutocovariance(fr, result, t, kStates,
                                scratch, lBase,
                                autocovInputN, 0,
                                scratch, tmpNpBase);
                        BLAS.dcopy(kStates2, scratch, nBase, 1,
                                autocovInputN, 0, 1);
                }

                if (storeAuxiliary) {
                        BLAS.dcopy(kStates2, scratch, lBase, 1,
                                result.innovationsTransition, result.innovationsTransitionOffset(t), 1);
                        BLAS.dcopy(kStates, scratch, rBase, 1,
                                result.scaledSmoothedEstimator, result.scaledSmoothedEstimatorOffset(t), 1);
                        BLAS.dcopy(kStates2, scratch, nBase, 1,
                                result.scaledSmoothedEstimatorCov, result.scaledSmoothedEstimatorCovOffset(t), 1);
                }

                BLAS.dgemv(BLAS.Trans.Trans, kStates, kStates, 1.0,
                        transition, stateSpace.transitionOffset(t), transitionLd,
                        scratch, rBase, 1,
                        0.0, scratch, rPrevBase, 1);
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, scratch, nBase, kStates,
                        transition, stateSpace.transitionOffset(t), transitionLd,
                        0.0, scratch, tmpNpBase, kStates);
                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                        1.0, transition, stateSpace.transitionOffset(t), transitionLd,
                        scratch, tmpNpBase, kStates,
                        0.0, scratch, nPrevBase, kStates);

                if (doDisturbance) {
                        BLAS.dgemv(BLAS.Trans.NoTrans, kEndog, kEndog, 1.0,
                                obsCov, observationModel.obsCovOffset(t), obsCovLd,
                                smoothingError, storeAuxiliary ? result.smoothingErrorOffset(t) : borrowedSmoothingErrorBase, 1,
                                0.0, result.smoothedObsDisturbance, result.smoothedObsDisturbanceOffset(t), 1);

                        if (kPosdef > 0) {
                                double[] nextScaledSmoothedEstimatorSource = storeAuxiliary ? result.scaledSmoothedEstimator : nextScaledSmoothedEstimator;
                                int nextScaledSmoothedEstimatorOff = storeAuxiliary ? result.scaledSmoothedEstimatorOffset(t + 1) : 0;
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                        1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                        stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                        0.0, scratch, tmpRqBase, kPosdef);
                                BLAS.dgemv(BLAS.Trans.Trans, kStates, kPosdef, 1.0,
                                        scratch, tmpRqBase, kPosdef,
                                        nextScaledSmoothedEstimatorSource, nextScaledSmoothedEstimatorOff, 1,
                                        0.0, result.smoothedStateDisturbance, result.smoothedStateDisturbanceOffset(t), 1);
                        }
                }

                if (doDisturbanceCov) {
                        BLAS.dcopy(kEndog2, obsCov, observationModel.obsCovOffset(t), 1,
                                result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), 1);

                        for (int i = kEndog - 1; i >= 0; i--) {
                                if (observationModel.isMissing(i, t)) continue;

                                double Fi = fr.forecastErrorCov[foreErrCovOff + i * kEndog + i];
                                if (Fi <= 0.0) continue;
                                double FiInv = 1.0 / Fi;

                                for (int j = 0; j < kStates; j++) {
                                        scratch[tmpKgBase + j] = fr.kalmanGain[kgOff + j * kEndog + i];
                                }

                                for (int j = 0; j < kStates; j++) {
                                        scratch[tmpKhBase + j * kEndog + i] = scratch[tmpKgBase + j];
                                }

                                int obsCovOff = observationModel.obsCovOffset(t);
                                for (int r = 0; r < kEndog; r++) {
                                        for (int c = 0; c < kEndog; c++) {
                                                result.smoothedObsDisturbanceCov[result.smoothedObsDisturbanceCovOffset(t) + r * kEndog + c] -=
                                                        obsCov[obsCovOff + r * obsCovLd + i] * FiInv *
                                                        obsCov[obsCovOff + c * obsCovLd + i];
                                        }
                                }
                        }

                        double[] nextScaledSmoothedEstimatorCovSource = storeAuxiliary ? result.scaledSmoothedEstimatorCov : nextScaledSmoothedEstimatorCov;
                        int nextScaledSmoothedEstimatorCovOff = storeAuxiliary ? result.scaledSmoothedEstimatorCovOffset(t + 1) : 0;

                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kEndog, kStates,
                                1.0, nextScaledSmoothedEstimatorCovSource, nextScaledSmoothedEstimatorCovOff, kStates,
                                scratch, tmpKhBase, kEndog,
                                0.0, scratch, tmpNpBase, kEndog);
                        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kEndog, kEndog, kStates,
                                -1.0, scratch, tmpKhBase, kEndog,
                                scratch, tmpNpBase, kEndog,
                                1.0, result.smoothedObsDisturbanceCov, result.smoothedObsDisturbanceCovOffset(t), kEndog);

                        if (kPosdef > 0) {
                                if (!doDisturbance) {
                                        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                                                1.0, selection, stateSpace.selectionOffset(t), selectionLd,
                                                stateCovariance, stateSpace.stateCovarianceOffset(t), stateCovarianceLd,
                                                0.0, scratch, tmpRqBase, kPosdef);
                                }
                                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kStates,
                                        1.0, nextScaledSmoothedEstimatorCovSource, nextScaledSmoothedEstimatorCovOff, kStates,
                                        scratch, tmpRqBase, kPosdef,
                                        0.0, scratch, tmpNpBase, kPosdef);
                                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, kPosdef, kPosdef, kStates,
                                        -1.0, scratch, tmpRqBase, kPosdef,
                                        scratch, tmpNpBase, kPosdef,
                                        0.0, result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), kPosdef);
                                BLAS.daxpy(kPosdef * kPosdef, 1.0, stateCovariance, stateSpace.stateCovarianceOffset(t), 1,
                                        result.smoothedStateDisturbanceCov, result.smoothedStateDisturbanceCovOffset(t), 1);
                        }
                }

                if (!storeAuxiliary) {
                        if (nextScaledSmoothedEstimator != null) {
                                BLAS.dcopy(kStates, scratch, rBase, 1, nextScaledSmoothedEstimator, 0, 1);
                        }
                        if (nextScaledSmoothedEstimatorCov != null) {
                                BLAS.dcopy(kStates2, scratch, nBase, 1, nextScaledSmoothedEstimatorCov, 0, 1);
                        }
                }

                int tmpRBase = rBase;
                rBase = rPrevBase;
                rPrevBase = tmpRBase;
                int tmpNBase = nBase;
                nBase = nPrevBase;
                nPrevBase = tmpNBase;
        }

        return result.trimOutputMask(smootherOutput);
    }

        private static void requireFilterStorage(FilterResult fr, int smoothMethod) {
                boolean needsFilteredState = fr.nobsDiffuse == 0
                                && !fr.perObsKalmanGain
                                && (smoothMethod & (SMOOTH_CLASSICAL | SMOOTH_ALTERNATIVE)) != 0;
                boolean needsKalmanGain = fr.nobsDiffuse == 0;

                if (!(fr.predictedStateLength() > 0 || fr.predictedStateCovLength() > 0)
                                || !(fr.forecastLength() > 0 || fr.forecastErrorLength() > 0
                                                || fr.forecastErrorCovLength() > 0
                                                || fr.standardizedForecastErrorLength() > 0
                                                || fr.forecastErrorDiffuseCovLength() > 0)
                                || (needsKalmanGain && fr.kalmanGainLength() == 0)
                                || (needsFilteredState && !(fr.filteredStateLength() > 0 || fr.filteredStateCovLength() > 0))) {
                        throw new IllegalArgumentException("FilterResult does not retain the storage required for smoothing");
                }
                if (fr.nobsDiffuse > 0 && ((fr.predictedDiffuseStateCov == null || fr.predictedDiffuseStateCov.length == 0)
                                || (fr.forecastErrorDiffuseCov == null || fr.forecastErrorDiffuseCov.length == 0))) {
                        throw new IllegalArgumentException("Diffuse smoothing requires diffuse filter storage");
                }
        }

        private static void requireRealInputs(KalmanSSM stateSpace,
                                              KalmanSSM observationModel) {
                KalmanSSMSupport.requireRealStorage(stateSpace, "stateSpace");
                KalmanSSMSupport.requireRealStorage(observationModel, "observationModel");
        }

        private static void copyObservedRows(KalmanSSM observationModel,
                                             int t,
                                             double[] source,
                                             int sourceOffset,
                                             int sourceLd,
                                             int rowWidth,
                                             double[] target,
                                             int targetOffset) {
                int targetRow = 0;
                int kEndog = observationModel.observationDimension();
                for (int sourceRow = 0; sourceRow < kEndog; sourceRow++) {
                        if (observationModel.isMissing(sourceRow, t)) {
                                continue;
                        }
                        System.arraycopy(source, sourceOffset + sourceRow * sourceLd,
                                        target, targetOffset + targetRow * rowWidth, rowWidth);
                        targetRow++;
                }
        }

        private static void copyObservedColumns(KalmanSSM observationModel,
                                                int t,
                                                int rowCount,
                                                double[] source,
                                                int sourceOffset,
                                                int sourceLd,
                                                double[] target,
                                                int targetOffset,
                                                int targetLd) {
                int kEndog = observationModel.observationDimension();
                for (int row = 0; row < rowCount; row++) {
                        int sourceRowOff = sourceOffset + row * sourceLd;
                        int targetRowOff = targetOffset + row * targetLd;
                        int targetCol = 0;
                        for (int sourceCol = 0; sourceCol < kEndog; sourceCol++) {
                                if (observationModel.isMissing(sourceCol, t)) {
                                        continue;
                                }
                                target[targetRowOff + targetCol] = source[sourceRowOff + sourceCol];
                                targetCol++;
                        }
                }
        }

        private static void copyObservedVector(KalmanSSM observationModel,
                                               int t,
                                               double[] source,
                                               int sourceOffset,
                                               double[] target,
                                               int targetOffset) {
                int targetIndex = 0;
                int kEndog = observationModel.observationDimension();
                for (int sourceIndex = 0; sourceIndex < kEndog; sourceIndex++) {
                        if (observationModel.isMissing(sourceIndex, t)) {
                                continue;
                        }
                        target[targetOffset + targetIndex] = source[sourceOffset + sourceIndex];
                        targetIndex++;
                }
        }

        private static void copyObservedSquareMatrix(KalmanSSM observationModel,
                                                     int t,
                                                     double[] source,
                                                     int sourceOffset,
                                                     int sourceLd,
                                                     double[] target,
                                                     int targetOffset,
                                                     int targetLd) {
                int targetRow = 0;
                int kEndog = observationModel.observationDimension();
                for (int sourceRow = 0; sourceRow < kEndog; sourceRow++) {
                        if (observationModel.isMissing(sourceRow, t)) {
                                continue;
                        }
                        int sourceRowOff = sourceOffset + sourceRow * sourceLd;
                        int targetRowOff = targetOffset + targetRow * targetLd;
                        int targetCol = 0;
                        for (int sourceCol = 0; sourceCol < kEndog; sourceCol++) {
                                if (observationModel.isMissing(sourceCol, t)) {
                                        continue;
                                }
                                target[targetRowOff + targetCol] = source[sourceRowOff + sourceCol];
                                targetCol++;
                        }
                        targetRow++;
                }
        }

        private static void scatterObservedVector(KalmanSSM observationModel,
                                                  int t,
                                                  double[] source,
                                                  int sourceOffset,
                                                  double[] target,
                                                  int targetOffset) {
                int sourceIndex = 0;
                int kEndog = observationModel.observationDimension();
                Arrays.fill(target, targetOffset, targetOffset + kEndog, 0.0);
                for (int targetIndex = 0; targetIndex < kEndog; targetIndex++) {
                        if (observationModel.isMissing(targetIndex, t)) {
                                continue;
                        }
                        target[targetOffset + targetIndex] = source[sourceOffset + sourceIndex];
                        sourceIndex++;
                }
        }

        private static void scatterObservedVectorInPlace(KalmanSSM observationModel,
                                                         int t,
                                                         double[] values,
                                                         int offset) {
                int kEndog = observationModel.observationDimension();
                int sourceIndex = kEndog - observationModel.missingCount(t) - 1;
                for (int targetIndex = kEndog - 1; targetIndex >= 0; targetIndex--) {
                        if (observationModel.isMissing(targetIndex, t)) {
                                values[offset + targetIndex] = 0.0;
                        } else {
                                values[offset + targetIndex] = values[offset + sourceIndex--];
                        }
                }
        }

        private static void scatterObservedSquareMatrix(KalmanSSM observationModel,
                                                        int t,
                                                        double[] source,
                                                        int sourceOffset,
                                                        int sourceLd,
                                                        double[] target,
                                                        int targetOffset,
                                                        int targetLd) {
                int sourceRow = 0;
                int kEndog = observationModel.observationDimension();
                for (int targetRow = 0; targetRow < kEndog; targetRow++) {
                        if (observationModel.isMissing(targetRow, t)) {
                                continue;
                        }
                        int targetRowOff = targetOffset + targetRow * targetLd;
                        int sourceRowOff = sourceOffset + sourceRow * sourceLd;
                        int sourceCol = 0;
                        for (int targetCol = 0; targetCol < kEndog; targetCol++) {
                                if (observationModel.isMissing(targetCol, t)) {
                                        continue;
                                }
                                target[targetRowOff + targetCol] = source[sourceRowOff + sourceCol];
                                sourceCol++;
                        }
                        sourceRow++;
                }
        }

        private static int selectObservations(KalmanSSM observationModel,
                                              ScratchArena scratchArena,
                                              SmootherScratchLayout layout,
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
                int ke = 0;
                for (int i = 0; i < kEndog; i++) {
                        if (!observationModel.isMissing(i, t)) {
                                ke++;
                        }
                }

                double[] scratch = scratchArena.scratchBacking;
                double[] design = observationModel.designData();
                int designOff = observationModel.designOffset(t);
                int designLd = observationModel.designLeadingDimension();
                double[] obsCov = observationModel.obsCovData();
                int obsCovOff = observationModel.obsCovOffset(t);
                int obsCovLd = observationModel.obsCovLeadingDimension();
                copyObservedRows(observationModel, t,
                                design, designOff, designLd, kStates,
                                scratch, layout.selectedDesignBase());
                copyObservedSquareMatrix(observationModel, t,
                                obsCov, obsCovOff, obsCovLd,
                                scratch, layout.selectedObsCovBase(), ke);
                return ke;
        }

        private static void symmetrizeMatrix(double[] matrix, int offset, int n) {
                for (int i = 0; i < n; i++) {
                        for (int j = i + 1; j < n; j++) {
                                double avg = 0.5 * (matrix[offset + i * n + j] + matrix[offset + j * n + i]);
                                matrix[offset + i * n + j] = avg;
                                matrix[offset + j * n + i] = avg;
                        }
                }
        }
}
