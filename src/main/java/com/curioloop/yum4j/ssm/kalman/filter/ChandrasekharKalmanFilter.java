package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSMSupport;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.arena.ChandrasekharScratchLayout;
import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.arena.FilterResultLayout;
import com.curioloop.yum4j.ssm.kalman.arena.IntArena;
import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;
import java.util.Objects;

final class ChandrasekharKalmanFilter {

    private ChandrasekharKalmanFilter() {
    }

    static final class Pool {
        private final DoubleArena scratchArena;
        private final IntArena intScratchArena = new IntArena();
        private final DoubleArena resultArena;
        private final boolean ownsScratchArena;
        private final boolean ownsResultArena;
        private double[] resultBacking;
        private FilterResultLayout resultLayout;
        private FilterResult pooledResult;
        private InitialState.StationaryWorkspace stationaryWorkspace;
        private ForecastErrorSolver.Result solverResult;
        private double[] scratchBacking;
        private ChandrasekharScratchLayout scratchLayout;
        private int scratchKEndog;
        private int scratchKStates;
        private boolean scratchFilteredCov;
        private int scratchLuInverseWorkLength;
        private int[] pivots;

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

        void reserve(KalmanSSM model, InitialState init, FilterOptions options) {
            Objects.requireNonNull(model, "model");
            Objects.requireNonNull(init, "init");
            FilterOptions resolvedOptions = options == null ? FilterOptions.defaults() : options;
            KalmanSSMSupport.requireRealStorage(model, "model");
            validateContract(model, init, resolvedOptions);
            int kEndog = model.observationDimension();
            int kStates = model.stateCount();
            FilterResultLayout retainedLayout = resolvedOptions.createResultLayout(1, kEndog, kStates,
                model.observationCount(), false);
            ensureResult(retainedLayout);
            int luInverseWorkLength = needsLu(resolvedOptions.forecastErrorSequenceRef()) ? luInverseWorkSize(kEndog) : 0;
            ensureScratch(kEndog, kStates, retainedLayout.filteredStateCovLength() > 0, luInverseWorkLength);
            stationaryWorkspace().ensureCapacity(model);
            if (luInverseWorkLength > 0) {
                ensurePivots(kEndog);
            }
        }

        FilterResult borrowResult(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
            ensureResult(layout);
            if (pooledResult == null) {
                pooledResult = new FilterResult();
            }
            return pooledResult.reuse(kEndog, kStates, nobs, resultBacking, resultLayout);
        }

        InitialState.StationaryWorkspace stationaryWorkspace() {
            if (stationaryWorkspace == null) {
                stationaryWorkspace = new InitialState.StationaryWorkspace();
            }
            return stationaryWorkspace;
        }

        ForecastErrorSolver.Result solverResult() {
            if (solverResult == null) {
                solverResult = new ForecastErrorSolver.Result();
            }
            return solverResult;
        }

        long retainedScratchDoubleCount() {
            return (usesScratchArena() ? scratchArena.retainedDoubleCount() : 0L)
                + retainedAuxiliaryScratchDoubleCount();
        }

        long retainedAuxiliaryScratchDoubleCount() {
            return intScratchArena.retainedByteCount() / Double.BYTES
                + (stationaryWorkspace == null ? 0L : stationaryWorkspace.retainedDoubleCount());
        }

        boolean usesScratchArena() {
            return scratchBacking != null;
        }

        void refreshScratchBacking() {
            if (scratchLayout != null) {
                scratchBacking = scratchArena.backing();
            }
        }

        long retainedResultDoubleCount() {
            return resultArena.retainedDoubleCount();
        }

        long retainedTotalByteCount() {
            return (retainedScratchDoubleCount() + retainedResultDoubleCount()) * Double.BYTES;
        }

        void releaseRetainedScratch() {
            scratchBacking = null;
            if (ownsScratchArena) {
                scratchArena.release();
            }
            intScratchArena.release();
            scratchLayout = null;
            scratchKEndog = 0;
            scratchKStates = 0;
            scratchFilteredCov = false;
            scratchLuInverseWorkLength = 0;
            pivots = null;
            solverResult = null;
            if (stationaryWorkspace != null) {
                stationaryWorkspace.release();
                stationaryWorkspace = null;
            }
        }

        void releaseRetainedResults() {
            invalidateBorrowedResult();
            pooledResult = null;
            if (ownsResultArena) {
                resultArena.release();
            }
        }

        boolean usesResultArena() {
            return resultBacking != null;
        }

        void invalidateBorrowedResult() {
            resultBacking = null;
            resultLayout = null;
        }

        private void ensureScratch(int kEndog, int kStates, boolean needFilteredCov, int luInverseWorkLength) {
            int nextKEndog = Math.max(scratchKEndog, kEndog);
            int nextKStates = Math.max(scratchKStates, kStates);
            boolean nextFilteredCov = scratchFilteredCov || needFilteredCov;
            int nextLuInverseWorkLength = Math.max(scratchLuInverseWorkLength, luInverseWorkLength);
            if (scratchLayout == null
                    || nextKEndog != scratchKEndog
                    || nextKStates != scratchKStates
                    || nextFilteredCov != scratchFilteredCov
                    || nextLuInverseWorkLength != scratchLuInverseWorkLength) {
                scratchLayout = ChandrasekharScratchLayout.create(
                    nextKEndog,
                    nextKStates,
                    nextFilteredCov,
                    nextLuInverseWorkLength);
                scratchKEndog = nextKEndog;
                scratchKStates = nextKStates;
                scratchFilteredCov = nextFilteredCov;
                scratchLuInverseWorkLength = nextLuInverseWorkLength;
            }
            int totalLength = scratchLayout.totalLength();
            scratchBacking = scratchArena.ensureCapacity(totalLength);
        }

        private void ensureResult(FilterResultLayout layout) {
            resultLayout = layout;
            int totalLength = layout.totalLength();
            resultBacking = resultArena.ensureCapacity(totalLength);
        }

        private int[] ensurePivots(int kEndog) {
            pivots = intScratchArena.ensureCapacity(kEndog);
            return pivots;
        }

        private static long doubleCount(double[] values) {
            return values == null ? 0L : values.length;
        }

    }

    static FilterResult filter(KalmanSSM model,
                               InitialState init,
                               KalmanFilter.Pool pool,
                               FilterOptions options) {
        return filter(model, init, pool, null, options);
    }

    static FilterResult filter(KalmanSSM model,
                               InitialState init,
                               KalmanFilter.Pool pool,
                               Pool chandrasekharPool,
                               FilterOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(options, "options");
        KalmanSSMSupport.requireRealStorage(model, "model");
        validateContract(model, init, options);
        return filterStationaryTimeInvariant(model, init, options, chandrasekharPool);
    }

    private static FilterResult filterStationaryTimeInvariant(KalmanSSM model,
                                                              InitialState init,
                                                              FilterOptions options,
                                                              Pool pool) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        int nobs = model.observationCount();
        int kEndog2 = kEndog * kEndog;
        int kStates2 = kStates * kStates;

        FilterResultLayout retainedLayout = options.createResultLayout(1, kEndog, kStates, nobs, false);
        FilterResult result = pool == null
            ? new FilterResult(kEndog, kStates, nobs, retainedLayout)
            : pool.borrowResult(kEndog, kStates, nobs, retainedLayout);

        boolean storeForecastMean = result.forecastLength() > 0;
        boolean storeForecastError = result.forecastErrorLength() > 0;
        boolean storeForecastCovariance = result.forecastErrorCovLength() > 0;
        boolean storeStandardizedForecastError = result.standardizedForecastErrorLength() > 0;
        boolean storePredictedState = result.predictedStateLength() > 0;
        boolean storePredictedCovariance = result.predictedStateCovLength() > 0;
        boolean storeFilteredState = result.filteredStateLength() > 0;
        boolean storeFilteredCovariance = result.filteredStateCovLength() > 0;
        boolean storeGain = result.kalmanGainLength() > 0;
        boolean storeLikelihood = result.logLikelihoodObsLength() > 0;
        boolean concentratedLikelihood = options.concentratedLikelihood();
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

        boolean needsLu = needsLu(options.forecastErrorSequenceRef());
        int luInverseWorkLength = needsLu ? luInverseWorkSize(kEndog) : 0;
        ChandrasekharScratchLayout scratchLayout;
        double[] scratch;
        if (pool != null) {
            pool.ensureScratch(kEndog, kStates, storeFilteredCovariance, luInverseWorkLength);
            scratchLayout = pool.scratchLayout;
            scratch = pool.scratchBacking;
        } else {
            scratchLayout = ChandrasekharScratchLayout.create(kEndog, kStates,
                storeFilteredCovariance, luInverseWorkLength);
            scratch = new double[scratchLayout.totalLength()];
        }

        int currentStateBase = scratchLayout.currentStateBase();
        int currentCovBase = scratchLayout.currentCovBase();
        int filteredStateBase = scratchLayout.filteredStateBase();
        int filteredCovBase = scratchLayout.filteredCovBase();
        int nextStateBase = scratchLayout.nextStateBase();
        int nextCovBase = scratchLayout.nextCovBase();
        int forecastMeanBase = scratchLayout.forecastMeanBase();
        int forecastErrorBase = scratchLayout.forecastErrorBase();
        int forecastCovBase = scratchLayout.forecastCovBase();
        int forecastCovWorkBase = scratchLayout.forecastCovWorkBase();
        int forecastErrorSolvedBase = scratchLayout.forecastErrorSolvedBase();
        int predictedDesignTransposedBase = scratchLayout.predictedDesignTransposedBase();
        int solvedDesignPredictedBase = scratchLayout.solvedDesignPredictedBase();
        int factorMWBase = scratchLayout.factorMWBase();
        int forecastInverseDesignBase = scratchLayout.forecastInverseDesignBase();
        int previousForecastInverseDesignBase = scratchLayout.previousForecastInverseDesignBase();
        int kalmanGainBase = scratchLayout.kalmanGainBase();
        int factorMBase = scratchLayout.factorMBase();
        int factorWBase = scratchLayout.factorWBase();
        int factorWNextBase = scratchLayout.factorWNextBase();
        int tmpMBase = scratchLayout.tmpMBase();
        int tmpMWZBase = scratchLayout.tmpMWZBase();
        int transitionMinusKZBase = scratchLayout.transitionMinusKZBase();
        int luInverseWorkBase = needsLu ? scratchLayout.luInverseWorkBase() : -1;

        InitialState.StationaryWorkspace stationaryWorkspace = pool == null
            ? new InitialState.StationaryWorkspace()
            : pool.stationaryWorkspace();
        stationaryWorkspace.ensureCapacity(model);
        init.resolveInto(model, 0, stationaryWorkspace,
            scratch, currentStateBase,
            scratch, currentCovBase,
            null, 0);

        int[] pivots = needsLu ? (pool == null ? new int[kEndog] : pool.ensurePivots(kEndog)) : null;
        ForecastErrorSolver.Result solverResult = pool == null ? new ForecastErrorSolver.Result() : pool.solverResult();

        boolean converged = false;
        int periodConverged = nobs;
        double concentratedScaleSum = 0.0;
        int concentratedScaleObservationCount = 0;
        int concentratedScaleStart = Math.max(0, options.concentratedLikelihoodBurn());

        for (int t = 0; t < nobs; t++) {
            if (storePredictedState) {
                System.arraycopy(scratch, currentStateBase, result.predictedState, result.predictedStateOffset(t), kStates);
            }
            if (storePredictedCovariance) {
                System.arraycopy(scratch, currentCovBase, result.predictedStateCov, result.predictedStateCovOffset(t), kStates2);
            }

            computeForecast(model, t, scratch, currentStateBase, scratch, forecastMeanBase, scratch, forecastErrorBase);
            computePredictedDesignTransposed(model, t, scratch, currentCovBase, scratch, predictedDesignTransposedBase);
            computeForecastCovariance(model, t, scratch, predictedDesignTransposedBase, scratch, forecastCovBase);

            if (storeForecastMean) {
                System.arraycopy(scratch, forecastMeanBase, result.forecast, result.forecastOffset(t), kEndog);
            }
            if (storeForecastError) {
                System.arraycopy(scratch, forecastErrorBase, result.forecastError, result.forecastErrorOffset(t), kEndog);
            }
            if (storeForecastCovariance) {
                System.arraycopy(scratch, forecastCovBase, result.forecastErrorCov, result.forecastErrorCovOffset(t), kEndog2);
            }
            if (storeStandardizedForecastError) {
                int offset = result.standardizedForecastErrorOffset(t);
                if (kEndog == 1) {
                    result.standardizedForecastError[offset] = ForecastErrorSolver.standardizeScalar(
                        scratch[forecastErrorBase], scratch[forecastCovBase], 0.0);
                } else if (!ForecastErrorSolver.standardizeCholesky(scratch, forecastCovBase, kEndog,
                        scratch, forecastErrorBase,
                        result.standardizedForecastError, offset,
                        kEndog, scratch, scratchLayout.standardizedWorkBase(), 0.0)) {
                    Arrays.fill(result.standardizedForecastError, offset, offset + kEndog, Double.NaN);
                }
            }

            copyTranspose(scratch, predictedDesignTransposedBase, kStates, kEndog, scratch, solvedDesignPredictedBase);
            BLAS.dcopy(kEndog, scratch, forecastErrorBase, 1, scratch, forecastErrorSolvedBase, 1);
            ForecastErrorSolver.solveMultivariate(options.forecastErrorSequenceRef(),
                scratch, forecastCovWorkBase,
                kEndog,
                scratch, forecastCovBase,
                scratch, solvedDesignPredictedBase,
                kStates,
                kStates,
                scratch, forecastErrorSolvedBase,
                scratch, forecastErrorBase,
                pivots,
                options.diffuseTolerance(),
                solverResult);
            invertForecastCovariance(scratch, forecastCovWorkBase, kEndog,
                solverResult.method, pivots, scratch, luInverseWorkBase, luInverseWorkLength);
            computeForecastInverseDesign(model, t, scratch, forecastCovWorkBase, scratch, forecastInverseDesignBase);

            computeFilteredState(scratch, currentStateBase,
                scratch, predictedDesignTransposedBase,
                scratch, forecastErrorSolvedBase,
                scratch, filteredStateBase,
                kStates, kEndog);
            if (storeFilteredCovariance) {
                computeFilteredCovariance(scratch, currentCovBase,
                    scratch, predictedDesignTransposedBase,
                    scratch, solvedDesignPredictedBase,
                    scratch, filteredCovBase,
                    kStates, kEndog);
            }
            computeKalmanGain(model, t, scratch, solvedDesignPredictedBase, scratch, kalmanGainBase);

            if (storeFilteredState) {
                System.arraycopy(scratch, filteredStateBase, result.filteredState, result.filteredStateOffset(t), kStates);
            }
            if (storeFilteredCovariance) {
                System.arraycopy(scratch, filteredCovBase, result.filteredStateCov, result.filteredStateCovOffset(t), kStates2);
            }
            if (storeGain) {
                System.arraycopy(scratch, kalmanGainBase, result.kalmanGain, result.kalmanGainOffset(t), kStates * kEndog);
            }
            if (storeLikelihood) {
                double logLikelihood = -0.5 * (kEndog * Math.log(2.0 * Math.PI)
                    + solverResult.logDeterminant);
                if (!concentratedLikelihood) {
                    logLikelihood += -0.5 * solverResult.quadraticForm;
                }
                result.logLikelihoodObs[result.logLikelihoodObsOffset(t)] =
                    logLikelihood;
            }
            if (concentratedLikelihood) {
                if (t >= concentratedScaleStart) {
                    concentratedScaleSum += solverResult.quadraticForm;
                    concentratedScaleObservationCount += kEndog;
                }
                if (storeLikelihood) {
                    result.scaleObs[t] = solverResult.quadraticForm;
                    result.scaleObsCount[t] = kEndog;
                }
            }

            predictState(model, t, scratch, filteredStateBase, scratch, nextStateBase);
            if (t == 0) {
                initializeChandrasekharFactors(model, t,
                    scratch, forecastCovWorkBase,
                    scratch, predictedDesignTransposedBase,
                    scratch, factorMBase,
                    scratch, factorWBase);
            } else {
                updateChandrasekharFactors(model, t,
                    scratch, factorMBase,
                    scratch, factorWBase,
                    scratch, previousForecastInverseDesignBase,
                    scratch, kalmanGainBase,
                    scratch, factorMWBase,
                    scratch, tmpMBase,
                    scratch, tmpMWZBase,
                    scratch, factorWNextBase,
                    scratch, transitionMinusKZBase);
                int factorWSwap = factorWBase;
                factorWBase = factorWNextBase;
                factorWNextBase = factorWSwap;
            }
            int forecastInverseSwap = forecastInverseDesignBase;
            forecastInverseDesignBase = previousForecastInverseDesignBase;
            previousForecastInverseDesignBase = forecastInverseSwap;
            predictCovarianceChandrasekhar(scratch, currentCovBase,
                scratch, factorMBase,
                scratch, factorWBase,
                scratch, factorMWBase,
                scratch, nextCovBase,
                kStates, kEndog);

            if (!converged && t > 0 && maxAbsDifference(scratch, nextCovBase, scratch, currentCovBase,
                    kStates2) < options.convergenceTolerance()) {
                converged = true;
                periodConverged = t + 1;
            }

            int stateSwap = currentStateBase;
            currentStateBase = nextStateBase;
            nextStateBase = stateSwap;
            int covSwap = currentCovBase;
            currentCovBase = nextCovBase;
            nextCovBase = covSwap;
            transitionMinusKZBase = nextCovBase;
        }

        if (storePredictedState) {
            System.arraycopy(scratch, currentStateBase, result.predictedState, result.predictedStateOffset(nobs), kStates);
        }
        if (storePredictedCovariance) {
            System.arraycopy(scratch, currentCovBase, result.predictedStateCov, result.predictedStateCovOffset(nobs), kStates2);
        }
        if (concentratedLikelihood) {
            double scale = concentratedScaleObservationCount == 0
                ? 1.0
                : concentratedScaleSum / concentratedScaleObservationCount;
            result.scale = scale;
            result.scaleObservationCount = concentratedScaleObservationCount;
            KalmanFilter.applyConcentratedScale(model, result, scale, storeLikelihood);
        }
        result.converged = converged;
        result.periodConverged = periodConverged;
        result.nobsDiffuse = 0;
        result.perObsKalmanGain = false;
        return result.applyConserveMemory(options.conserveMemoryMask());
    }

    static void validateContract(KalmanSSM model, InitialState init, FilterOptions options) {
        if (KalmanSSMSupport.hasMissingObservations(model)) {
            throw new UnsupportedOperationException(
                "CHANDRASEKHAR filtering cannot be used with missing observations");
        }
        if (options.timing() == FilterOptions.Timing.INIT_FILTERED) {
            throw new UnsupportedOperationException(
                "CHANDRASEKHAR filtering cannot be used with INIT_FILTERED timing");
        }
        if (!init.isStationary()) {
            throw new UnsupportedOperationException(
                "CHANDRASEKHAR filtering requires stationary initialization");
        }
        if (init.mayResolveDiffuse()) {
            throw new UnsupportedOperationException(
                "CHANDRASEKHAR filtering cannot be used with diffuse initialization");
        }
        if (hasTimeVaryingSystemMatrices(model)) {
            throw new UnsupportedOperationException(
                "CHANDRASEKHAR filtering cannot be used with time-varying system matrices except intercepts");
        }
    }

    private static boolean hasTimeVaryingSystemMatrices(KalmanSSM model) {
        int nobs = model.observationCount();
        for (int t = 1; t < nobs; t++) {
            if (!sameBlock(model.designData(), model.designOffset(0), model.designOffset(t),
                    model.designLeadingDimension(), model.observationDimension(), model.stateCount())) {
                return true;
            }
            if (!sameBlock(model.obsCovData(), model.obsCovOffset(0), model.obsCovOffset(t),
                    model.obsCovLeadingDimension(), model.observationDimension(), model.observationDimension())) {
                return true;
            }
            if (!sameBlock(model.transitionData(), model.transitionOffset(0), model.transitionOffset(t),
                    model.transitionLeadingDimension(), model.stateCount(), model.stateCount())) {
                return true;
            }
            if (!sameBlock(model.selectionData(), model.selectionOffset(0), model.selectionOffset(t),
                    model.selectionLeadingDimension(), model.stateCount(), model.stateDisturbanceCount())) {
                return true;
            }
            if (!sameBlock(model.stateCovarianceData(), model.stateCovarianceOffset(0), model.stateCovarianceOffset(t),
                    model.stateCovarianceLeadingDimension(), model.stateDisturbanceCount(), model.stateDisturbanceCount())) {
                return true;
            }
        }
        return false;
    }

    private static boolean sameBlock(double[] data,
                                     int firstOffset,
                                     int secondOffset,
                                     int leadingDimension,
                                     int rows,
                                     int cols) {
        for (int row = 0; row < rows; row++) {
            int firstRow = firstOffset + row * leadingDimension;
            int secondRow = secondOffset + row * leadingDimension;
            for (int col = 0; col < cols; col++) {
                if (data[firstRow + col] != data[secondRow + col]) {
                    return false;
                }
            }
        }
        return true;
    }

    private static void computeForecast(KalmanSSM model,
                                        int t,
                                        double[] predictedState,
                                        int predictedStateBase,
                                        double[] forecastMean,
                                        int forecastMeanBase,
                                        double[] forecastError,
                                        int forecastErrorBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        double[] obsIntercept = model.obsInterceptData();
        int interceptOffset = model.obsInterceptOffset(t);
        double[] endog = model.endogData();
        int endogOffset = model.endogOffset(t);
        BLAS.dcopy(kEndog, obsIntercept, interceptOffset, 1, forecastMean, forecastMeanBase, 1);
        BLAS.dgemv(BLAS.Trans.NoTrans, kEndog, kStates, 1.0,
            design, designOffset, designLd,
            predictedState, predictedStateBase, 1,
            1.0,
            forecastMean, forecastMeanBase, 1);
        BLAS.dcopy(kEndog, endog, endogOffset, 1, forecastError, forecastErrorBase, 1);
        BLAS.daxpy(kEndog, -1.0, forecastMean, forecastMeanBase, 1, forecastError, forecastErrorBase, 1);
    }

    private static void computePredictedDesignTransposed(KalmanSSM model,
                                                         int t,
                                                         double[] predictedCov,
                                                         int predictedCovBase,
                                                         double[] out,
                                                         int outBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kEndog, kStates,
            1.0, predictedCov, predictedCovBase, kStates,
            design, designOffset, designLd,
            0.0, out, outBase, kEndog);
    }

    private static void computeForecastCovariance(KalmanSSM model,
                                                  int t,
                                                  double[] predictedDesignTransposed,
                                                  int predictedDesignTransposedBase,
                                                  double[] out,
                                                  int outBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        BLAS.dlacpy(BLAS.Uplo.All, kEndog, kEndog,
            model.obsCovData(), model.obsCovOffset(t), model.obsCovLeadingDimension(),
            out, outBase, kEndog);
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kEndog, kEndog, kStates,
            1.0, design, designOffset, designLd,
            predictedDesignTransposed, predictedDesignTransposedBase, kEndog,
            1.0, out, outBase, kEndog);
    }

    private static void invertForecastCovariance(double[] inverse,
                                                 int inverseBase,
                                                 int dimension,
                                                 ForecastErrorSolver.Method method,
                                                 int[] pivots,
                                                 double[] luWork,
                                                 int luWorkBase,
                                                 int luWorkLength) {
        if (method == ForecastErrorSolver.Method.CHOLESKY_SOLVE) {
            if (!BLAS.dpotri(BLAS.Uplo.Lower, dimension, inverse, inverseBase, dimension)) {
                throw new KalmanFilter.SingularForecastException(
                    "Forecast-error covariance is singular for CHANDRASEKHAR filtering");
            }
            for (int row = 0; row < dimension; row++) {
                for (int col = row + 1; col < dimension; col++) {
                    inverse[inverseBase + row * dimension + col] = inverse[inverseBase + col * dimension + row];
                }
            }
        } else if (method == ForecastErrorSolver.Method.LU_SOLVE) {
            if (pivots == null || pivots.length < dimension) {
                throw new IllegalArgumentException("LU inverse requires pivot workspace of length " + dimension);
            }
            if (luWork == null || luWorkBase < 0 || luWorkLength < dimension) {
                throw new IllegalArgumentException("LU inverse requires double workspace of length " + dimension);
            }
            if (!BLAS.dgetri(dimension, inverse, inverseBase, dimension, pivots, 0,
                    luWork, luWorkBase, luWorkLength)) {
                throw new KalmanFilter.SingularForecastException(
                    "Forecast-error covariance is singular for CHANDRASEKHAR filtering");
            }
        } else {
            throw new UnsupportedOperationException(method + " inverse is not implemented for CHANDRASEKHAR filtering");
        }
    }

    private static int luInverseWorkSize(int dimension) {
        double[] query = new double[1];
        BLAS.dgetri(dimension, null, 0, dimension, null, 0, query, 0, -1);
        return Math.max(dimension, (int) query[0]);
    }

    private static void computeForecastInverseDesign(KalmanSSM model,
                                                     int t,
                                                     double[] forecastCovInverse,
                                                     int forecastCovInverseBase,
                                                     double[] out,
                                                     int outBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kEndog, kStates, kEndog,
            1.0, forecastCovInverse, forecastCovInverseBase, kEndog,
            design, designOffset, designLd,
            0.0, out, outBase, kStates);
    }

    private static void computeFilteredState(double[] predictedState,
                                             int predictedStateBase,
                                             double[] predictedDesignTransposed,
                                             int predictedDesignTransposedBase,
                                             double[] solvedForecastError,
                                             int solvedForecastErrorBase,
                                             double[] filteredState,
                                             int filteredStateBase,
                                             int kStates,
                                             int kEndog) {
        BLAS.dcopy(kStates, predictedState, predictedStateBase, 1, filteredState, filteredStateBase, 1);
        BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kEndog, 1.0,
            predictedDesignTransposed, predictedDesignTransposedBase, kEndog,
            solvedForecastError, solvedForecastErrorBase, 1,
            1.0,
            filteredState, filteredStateBase, 1);
    }

    private static void computeFilteredCovariance(double[] predictedCov,
                                                  int predictedCovBase,
                                                  double[] predictedDesignTransposed,
                                                  int predictedDesignTransposedBase,
                                                  double[] solvedDesignPredicted,
                                                  int solvedDesignPredictedBase,
                                                  double[] filteredCov,
                                                  int filteredCovBase,
                                                  int kStates,
                                                  int kEndog) {
        BLAS.dcopy(kStates * kStates, predictedCov, predictedCovBase, 1, filteredCov, filteredCovBase, 1);
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kEndog,
            -1.0, predictedDesignTransposed, predictedDesignTransposedBase, kEndog,
            solvedDesignPredicted, solvedDesignPredictedBase, kStates,
            1.0, filteredCov, filteredCovBase, kStates);
    }

    private static void computeKalmanGain(KalmanSSM model,
                                          int t,
                                          double[] solvedDesignPredicted,
                                          int solvedDesignPredictedBase,
                                          double[] kalmanGain,
                                          int kalmanGainBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] transition = model.transitionData();
        int transitionOffset = model.transitionOffset(t);
        int transitionLd = model.transitionLeadingDimension();
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kStates, kEndog, kStates,
            1.0, transition, transitionOffset, transitionLd,
            solvedDesignPredicted, solvedDesignPredictedBase, kStates,
            0.0, kalmanGain, kalmanGainBase, kEndog);
    }

    private static void predictState(KalmanSSM model,
                                     int t,
                                     double[] filteredState,
                                     int filteredStateBase,
                                     double[] nextState,
                                     int nextStateBase) {
        int kStates = model.stateCount();
        double[] stateIntercept = model.stateInterceptData();
        BLAS.dcopy(kStates, stateIntercept, model.stateInterceptOffset(t), 1, nextState, nextStateBase, 1);
        double[] transition = model.transitionData();
        int transitionOffset = model.transitionOffset(t);
        int transitionLd = model.transitionLeadingDimension();
        BLAS.dgemv(BLAS.Trans.NoTrans, kStates, kStates, 1.0,
            transition, transitionOffset, transitionLd,
            filteredState, filteredStateBase, 1,
            1.0,
            nextState, nextStateBase, 1);
    }

    private static void initializeChandrasekharFactors(KalmanSSM model,
                                                       int t,
                                                       double[] forecastCovInverse,
                                                       int forecastCovInverseBase,
                                                       double[] predictedDesignTransposed,
                                                       int predictedDesignTransposedBase,
                                                       double[] factorM,
                                                       int factorMBase,
                                                       double[] factorW,
                                                       int factorWBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        BLAS.dcopy(kEndog * kEndog, forecastCovInverse, forecastCovInverseBase, 1, factorM, factorMBase, 1);
        BLAS.dscal(kEndog * kEndog, -1.0, factorM, factorMBase, 1);
        double[] transition = model.transitionData();
        int transitionOffset = model.transitionOffset(t);
        int transitionLd = model.transitionLeadingDimension();
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kEndog, kStates,
            1.0, transition, transitionOffset, transitionLd,
            predictedDesignTransposed, predictedDesignTransposedBase, kEndog,
            0.0, factorW, factorWBase, kEndog);
    }

    private static void updateChandrasekharFactors(KalmanSSM model,
                                                   int t,
                                                   double[] factorM,
                                                   int factorMBase,
                                                   double[] factorW,
                                                   int factorWBase,
                                                   double[] previousForecastInverseDesign,
                                                   int previousForecastInverseDesignBase,
                                                   double[] kalmanGain,
                                                   int kalmanGainBase,
                                                   double[] factorMW,
                                                   int factorMWBase,
                                                   double[] tmpM,
                                                   int tmpMBase,
                                                   double[] tmpMWZ,
                                                   int tmpMWZBase,
                                                   double[] factorWNext,
                                                   int factorWNextBase,
                                                   double[] transitionMinusKZ,
                                                   int transitionMinusKZBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        multiplyMByWTransposed(factorM, factorMBase, factorW, factorWBase,
            factorMW, factorMWBase, kEndog, kStates);
        multiplyForecastInverseDesignByTransposed(previousForecastInverseDesign, previousForecastInverseDesignBase,
            factorMW, factorMWBase, tmpM, tmpMBase, kEndog, kStates);
        multiplyMWByDesignTransposed(model, t, factorMW, factorMWBase, tmpMWZ, tmpMWZBase, kEndog, kStates);
        addProductToM(factorM, factorMBase, tmpMWZ, tmpMWZBase, tmpM, tmpMBase, kEndog);

        computeTransitionMinusKZ(model, t, kalmanGain, kalmanGainBase, transitionMinusKZ, transitionMinusKZBase);
        multiplySquareByRectangular(transitionMinusKZ, transitionMinusKZBase,
            factorW, factorWBase,
            factorWNext, factorWNextBase,
            kStates, kEndog);
    }

    private static void predictCovarianceChandrasekhar(double[] currentCov,
                                                       int currentCovBase,
                                                       double[] factorM,
                                                       int factorMBase,
                                                       double[] factorW,
                                                       int factorWBase,
                                                       double[] factorMW,
                                                       int factorMWBase,
                                                       double[] nextCov,
                                                       int nextCovBase,
                                                       int kStates,
                                                       int kEndog) {
        multiplyMByWTransposed(factorM, factorMBase, factorW, factorWBase,
            factorMW, factorMWBase, kEndog, kStates);
        BLAS.dcopy(kStates * kStates, currentCov, currentCovBase, 1, nextCov, nextCovBase, 1);
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kEndog,
            1.0, factorW, factorWBase, kEndog,
            factorMW, factorMWBase, kStates,
            1.0, nextCov, nextCovBase, kStates);
    }

    private static void multiplyMByWTransposed(double[] factorM,
                                               int factorMBase,
                                               double[] factorW,
                                               int factorWBase,
                                               double[] out,
                                               int outBase,
                                               int kEndog,
                                               int kStates) {
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kEndog, kStates, kEndog,
            1.0, factorM, factorMBase, kEndog,
            factorW, factorWBase, kEndog,
            0.0, out, outBase, kStates);
    }

    private static void multiplyForecastInverseDesignByTransposed(double[] forecastInverseDesign,
                                                                  int forecastInverseDesignBase,
                                                                  double[] factorMW,
                                                                  int factorMWBase,
                                                                  double[] out,
                                                                  int outBase,
                                                                  int kEndog,
                                                                  int kStates) {
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kEndog, kEndog, kStates,
            1.0, forecastInverseDesign, forecastInverseDesignBase, kStates,
            factorMW, factorMWBase, kStates,
            0.0, out, outBase, kEndog);
    }

    private static void multiplyMWByDesignTransposed(KalmanSSM model,
                                                     int t,
                                                     double[] factorMW,
                                                     int factorMWBase,
                                                     double[] out,
                                                     int outBase,
                                                     int kEndog,
                                                     int kStates) {
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, kEndog, kEndog, kStates,
            1.0, factorMW, factorMWBase, kStates,
            design, designOffset, designLd,
            0.0, out, outBase, kEndog);
    }

    private static void addProductToM(double[] factorM,
                                      int factorMBase,
                                      double[] left,
                                      int leftBase,
                                      double[] right,
                                      int rightBase,
                                      int kEndog) {
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kEndog, kEndog, kEndog,
            1.0, left, leftBase, kEndog,
            right, rightBase, kEndog,
            1.0, factorM, factorMBase, kEndog);
    }

    private static void computeTransitionMinusKZ(KalmanSSM model,
                                                 int t,
                                                 double[] kalmanGain,
                                                 int kalmanGainBase,
                                                 double[] out,
                                                 int outBase) {
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        double[] design = model.designData();
        int designOffset = model.designOffset(t);
        int designLd = model.designLeadingDimension();
        BLAS.dlacpy(BLAS.Uplo.All, kStates, kStates,
            model.transitionData(), model.transitionOffset(t), model.transitionLeadingDimension(),
            out, outBase, kStates);
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kEndog,
            -1.0, kalmanGain, kalmanGainBase, kEndog,
            design, designOffset, designLd,
            1.0, out, outBase, kStates);
    }

    private static void multiplySquareByRectangular(double[] square,
                                                    int squareBase,
                                                    double[] rectangular,
                                                    int rectangularBase,
                                                    double[] out,
                                                    int outBase,
                                                    int kStates,
                                                    int kEndog) {
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kEndog, kStates,
            1.0, square, squareBase, kStates,
            rectangular, rectangularBase, kEndog,
            0.0, out, outBase, kEndog);
    }

    private static void copyTranspose(double[] source, int sourceBase, int rows, int cols,
                                      double[] target, int targetBase) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                target[targetBase + col * rows + row] = source[sourceBase + row * cols + col];
            }
        }
    }

    private static double maxAbsDifference(double[] left, int leftBase,
                                           double[] right, int rightBase,
                                           int length) {
        double max = 0.0;
        for (int i = 0; i < length; i++) {
            double diff = Math.abs(left[leftBase + i] - right[rightBase + i]);
            if (diff > max) {
                max = diff;
            }
        }
        return max;
    }

    private static boolean needsLu(ForecastErrorSolver.Method[] sequence) {
        for (ForecastErrorSolver.Method method : sequence) {
            if (method == ForecastErrorSolver.Method.LU_SOLVE) {
                return true;
            }
        }
        return false;
    }
}