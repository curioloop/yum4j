package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.arena.FilterDiffuseLayout;
import com.curioloop.yum4j.ssm.kalman.arena.FilterResultLayout;
import com.curioloop.yum4j.ssm.kalman.arena.ResultArrays;

import java.util.Arrays;

abstract class FilterResultBase<T extends FilterResultBase<T>> implements Cloneable {

    protected static final double[] EMPTY = ResultArrays.emptyArray();
    protected static final int[] EMPTY_INT = new int[0];

    public int kEndog;
    public int kStates;
    public int nobs;

    public double[] predictedState;
    public double[] predictedStateCov;
    public double[] filteredState;
    public double[] filteredStateCov;
    public double[] forecast;
    public double[] forecastError;
    public double[] forecastErrorCov;
    public double[] standardizedForecastError;
    public double[] kalmanGain;
    public double[] logLikelihoodObs;
    public double[] scaleObs;
    public int[] scaleObsCount;
    public boolean concentratedLikelihood;
    private double logLikelihood;
    public double scale;
    public int scaleObservationCount;
    public boolean converged;
    public int periodConverged;

    public double[] predictedDiffuseStateCov;
    public double[] forecastErrorDiffuseCov;
    public int nobsDiffuse;

    public boolean perObsKalmanGain;
    public int[] fallbackPeriods;
    public int fallbackPeriodCount;
    public int fallbackRouteCode;

    private int predictedStateBase;
    private int predictedStateLength;
    private int predictedStateCovBase;
    private int predictedStateCovLength;
    private int filteredStateBase;
    private int filteredStateLength;
    private int filteredStateCovBase;
    private int filteredStateCovLength;
    private int forecastBase;
    private int forecastLength;
    private int forecastErrorBase;
    private int forecastErrorLength;
    private int forecastErrorCovBase;
    private int forecastErrorCovLength;
    private int standardizedForecastErrorBase;
    private int standardizedForecastErrorLength;
    private int kalmanGainBase;
    private int kalmanGainLength;
    private int logLikelihoodObsBase;
    private int logLikelihoodObsLength;
    private int scaleObsLength;
    private int scaleObsCountLength;
    private int predictedDiffuseStateCovBase;
    private int predictedDiffuseStateCovLength;
    private int forecastErrorDiffuseCovBase;
    private int forecastErrorDiffuseCovLength;

    protected FilterResultBase() {
        this.kEndog = 0;
        this.kStates = 0;
        this.nobs = 0;
        this.predictedState = EMPTY;
        this.predictedStateCov = EMPTY;
        this.filteredState = EMPTY;
        this.filteredStateCov = EMPTY;
        this.forecast = EMPTY;
        this.forecastError = EMPTY;
        this.forecastErrorCov = EMPTY;
        this.standardizedForecastError = EMPTY;
        this.kalmanGain = EMPTY;
        this.logLikelihoodObs = EMPTY;
        this.scaleObs = EMPTY;
        this.scaleObsCount = EMPTY_INT;
        this.concentratedLikelihood = false;
        this.logLikelihood = 0.0;
        this.scale = 1.0;
        this.scaleObservationCount = 0;
        this.predictedDiffuseStateCov = null;
        this.forecastErrorDiffuseCov = null;
        this.nobsDiffuse = 0;
        this.fallbackPeriods = EMPTY_INT;
        this.fallbackPeriodCount = 0;
        this.fallbackRouteCode = 0;
    }

    protected FilterResultBase(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
        this();
        this.kEndog = kEndog;
        this.kStates = kStates;
        this.nobs = nobs;
        int predictedStateLength = layout.predictedStateLength();
        int predictedStateCovLength = layout.predictedStateCovLength();
        int filteredStateLength = layout.filteredStateLength();
        int filteredStateCovLength = layout.filteredStateCovLength();
        int forecastLength = layout.forecastLength();
        int forecastErrorLength = layout.forecastErrorLength();
        int forecastErrorCovLength = layout.forecastErrorCovLength();
        int standardizedForecastErrorLength = layout.standardizedForecastErrorLength();
        int kalmanGainLength = layout.kalmanGainLength();
        int likelihoodLength = layout.logLikelihoodObsLength();
        setPredictedStateSurface(predictedStateLength > 0 ? new double[predictedStateLength] : EMPTY, 0, predictedStateLength);
        setPredictedStateCovSurface(predictedStateCovLength > 0 ? new double[predictedStateCovLength] : EMPTY, 0, predictedStateCovLength);
        setFilteredStateSurface(filteredStateLength > 0 ? new double[filteredStateLength] : EMPTY, 0, filteredStateLength);
        setFilteredStateCovSurface(filteredStateCovLength > 0 ? new double[filteredStateCovLength] : EMPTY, 0, filteredStateCovLength);
        setForecastSurface(forecastLength > 0 ? new double[forecastLength] : EMPTY, 0, forecastLength);
        setForecastErrorSurface(forecastErrorLength > 0 ? new double[forecastErrorLength] : EMPTY, 0, forecastErrorLength);
        setForecastErrorCovSurface(forecastErrorCovLength > 0 ? new double[forecastErrorCovLength] : EMPTY, 0, forecastErrorCovLength);
        setStandardizedForecastErrorSurface(standardizedForecastErrorLength > 0 ? new double[standardizedForecastErrorLength] : EMPTY,
            0, standardizedForecastErrorLength);
        setKalmanGainSurface(kalmanGainLength > 0 ? new double[kalmanGainLength] : EMPTY, 0, kalmanGainLength);
        setLogLikelihoodObsSurface(likelihoodLength > 0 ? new double[likelihoodLength] : EMPTY, 0, likelihoodLength);
    }

    protected abstract int scalarWidth();

    protected abstract T newResult(int kEndog, int kStates, int nobs, FilterResultLayout layout);

    @SuppressWarnings("unchecked")
    private T self() {
        return (T) this;
    }

    final T reuse(int kEndog, int kStates, int nobs, double[] backing, FilterResultLayout layout) {
        bind(kEndog, kStates, nobs, backing, layout);
        this.predictedDiffuseStateCov = null;
        this.forecastErrorDiffuseCov = null;
        this.nobsDiffuse = 0;
        this.converged = false;
        this.periodConverged = 0;
        this.perObsKalmanGain = false;
        this.concentratedLikelihood = false;
        this.logLikelihood = 0.0;
        this.scale = 1.0;
        this.scaleObservationCount = 0;
        this.scaleObsLength = 0;
        this.scaleObsCountLength = 0;
        this.fallbackPeriods = EMPTY_INT;
        this.fallbackPeriodCount = 0;
        this.fallbackRouteCode = 0;
        return self();
    }

    final T reuseDiffuse(double[] backing, FilterDiffuseLayout diffuseLayout) {
        return reuseDiffuse(backing, 0, diffuseLayout);
    }

    final T reuseDiffuse(double[] backing, int backingBase, FilterDiffuseLayout diffuseLayout) {
        int predictedDiffuseBase = backingBase + diffuseLayout.predictedDiffuseStateCovBase();
        int predictedDiffuseLength = diffuseLayout.predictedDiffuseStateCovLength();
        int diffuseForecastBase = backingBase + diffuseLayout.forecastErrorDiffuseCovBase();
        int diffuseForecastLength = diffuseLayout.forecastErrorDiffuseCovLength();
        setPredictedDiffuseStateCovSurface(predictedDiffuseLength == 0 ? EMPTY : backing, predictedDiffuseBase, predictedDiffuseLength);
        setForecastErrorDiffuseCovSurface(diffuseForecastLength == 0 ? EMPTY : backing, diffuseForecastBase, diffuseForecastLength);
        return self();
    }

    public final T copy() {
        T copy = newResult(kEndog, kStates, nobs, currentLayout());
        ResultArrays.copySurface(predictedState, predictedStateBase(), predictedStateLength(), copy.predictedState, copy.predictedStateBase());
        ResultArrays.copySurface(predictedStateCov, predictedStateCovBase(), predictedStateCovLength(), copy.predictedStateCov, copy.predictedStateCovBase());
        ResultArrays.copySurface(filteredState, filteredStateBase(), filteredStateLength(), copy.filteredState, copy.filteredStateBase());
        ResultArrays.copySurface(filteredStateCov, filteredStateCovBase(), filteredStateCovLength(), copy.filteredStateCov, copy.filteredStateCovBase());
        ResultArrays.copySurface(forecast, forecastBase(), forecastLength(), copy.forecast, copy.forecastBase());
        ResultArrays.copySurface(forecastError, forecastErrorBase(), forecastErrorLength(), copy.forecastError, copy.forecastErrorBase());
        ResultArrays.copySurface(forecastErrorCov, forecastErrorCovBase(), forecastErrorCovLength(), copy.forecastErrorCov, copy.forecastErrorCovBase());
        ResultArrays.copySurface(standardizedForecastError, standardizedForecastErrorBase(), standardizedForecastErrorLength(),
            copy.standardizedForecastError, copy.standardizedForecastErrorBase());
        ResultArrays.copySurface(kalmanGain, kalmanGainBase(), kalmanGainLength(), copy.kalmanGain, copy.kalmanGainBase());
        ResultArrays.copySurface(logLikelihoodObs, logLikelihoodObsBase(), logLikelihoodObsLength(), copy.logLikelihoodObs, copy.logLikelihoodObsBase());
        copy.concentratedLikelihood = concentratedLikelihood;
        copy.setLogLikelihood(logLikelihood());
        copy.scale = scale;
        copy.scaleObservationCount = scaleObservationCount;
        if (scaleObsLength() > 0) {
            copy.retainScaleObs(scaleObsLength());
            System.arraycopy(scaleObs, 0, copy.scaleObs, 0, scaleObsLength());
        }
        if (scaleObsCountLength() > 0) {
            copy.retainScaleObsCount(scaleObsCountLength());
            System.arraycopy(scaleObsCount, 0, copy.scaleObsCount, 0, scaleObsCountLength());
        }
        copy.converged = converged;
        copy.periodConverged = periodConverged;
        if (predictedDiffuseStateCov != null || forecastErrorDiffuseCov != null) {
            copy.allocateDiffuse(predictedDiffuseStateCovLength() > 0, forecastErrorDiffuseCovLength() > 0);
            ResultArrays.copySurface(predictedDiffuseStateCov, predictedDiffuseStateCovBase(), predictedDiffuseStateCovLength(),
                copy.predictedDiffuseStateCov, copy.predictedDiffuseStateCovBase());
            ResultArrays.copySurface(forecastErrorDiffuseCov, forecastErrorDiffuseCovBase(), forecastErrorDiffuseCovLength(),
                copy.forecastErrorDiffuseCov, copy.forecastErrorDiffuseCovBase());
        }
        copy.nobsDiffuse = nobsDiffuse;
        copy.perObsKalmanGain = perObsKalmanGain;
        copyFallbackTelemetryTo(copy);
        return copy;
    }

    @Override
    public final T clone() {
        return copy();
    }

    private FilterResultLayout currentLayout() {
        return FilterResultLayout.create(
            scalarWidth(),
            kEndog,
            kStates,
            nobs,
            forecastLength() > 0,
            forecastErrorLength() > 0,
            forecastErrorCovLength() > 0,
            standardizedForecastErrorLength() > 0,
            predictedStateLength() > 0,
            predictedStateCovLength() > 0,
            filteredStateLength() > 0,
            filteredStateCovLength() > 0,
            kalmanGainLength() > 0,
            logLikelihoodObsLength() > 0);
    }

    public final void allocateDiffuse() {
        allocateDiffuse(true, true);
    }

    public final void allocateDiffuse(boolean storePredicted, boolean storeForecast) {
        if (predictedDiffuseStateCov != null || forecastErrorDiffuseCov != null) {
            return;
        }
        FilterDiffuseLayout layout = FilterDiffuseLayout.create(scalarWidth(), kEndog, kStates, nobs,
            storePredicted, storeForecast);
        double[] backing = ResultArrays.allocate(layout.totalLength());
        setPredictedDiffuseStateCovSurface(
            layout.predictedDiffuseStateCovLength() > 0 ? backing : EMPTY,
            layout.predictedDiffuseStateCovBase(),
            layout.predictedDiffuseStateCovLength());
        setForecastErrorDiffuseCovSurface(
            layout.forecastErrorDiffuseCovLength() > 0 ? backing : EMPTY,
            layout.forecastErrorDiffuseCovBase(),
            layout.forecastErrorDiffuseCovLength());
    }

    public final void suppressPredictedState() {
        setPredictedStateSurface(EMPTY, 0, 0);
        setPredictedStateCovSurface(EMPTY, 0, 0);
        if (predictedDiffuseStateCov != null) {
            setPredictedDiffuseStateCovSurface(EMPTY, 0, 0);
        }
    }

    public final void suppressFilteredState() {
        setFilteredStateSurface(EMPTY, 0, 0);
        setFilteredStateCovSurface(EMPTY, 0, 0);
    }

    public final void suppressForecast() {
        setForecastSurface(EMPTY, 0, 0);
        setForecastErrorSurface(EMPTY, 0, 0);
        setForecastErrorCovSurface(EMPTY, 0, 0);
        setStandardizedForecastErrorSurface(EMPTY, 0, 0);
        if (forecastErrorDiffuseCov != null) {
            setForecastErrorDiffuseCovSurface(EMPTY, 0, 0);
        }
    }

    public final void suppressKalmanGain() {
        setKalmanGainSurface(EMPTY, 0, 0);
    }

    public final void suppressLikelihood() {
        logLikelihood = logLikelihood();
        setLogLikelihoodObsSurface(EMPTY, 0, 0);
    }

    final T clearActiveSurfaces() {
        clearSurface(predictedState, predictedStateBase(), predictedStateLength());
        clearSurface(predictedStateCov, predictedStateCovBase(), predictedStateCovLength());
        clearSurface(filteredState, filteredStateBase(), filteredStateLength());
        clearSurface(filteredStateCov, filteredStateCovBase(), filteredStateCovLength());
        clearSurface(forecast, forecastBase(), forecastLength());
        clearSurface(forecastError, forecastErrorBase(), forecastErrorLength());
        clearSurface(forecastErrorCov, forecastErrorCovBase(), forecastErrorCovLength());
        clearSurface(standardizedForecastError, standardizedForecastErrorBase(), standardizedForecastErrorLength());
        clearSurface(kalmanGain, kalmanGainBase(), kalmanGainLength());
        clearSurface(logLikelihoodObs, logLikelihoodObsBase(), logLikelihoodObsLength());
        clearSurface(scaleObs, 0, scaleObsLength());
        clearSurface(scaleObsCount, 0, scaleObsCountLength());
        clearSurface(predictedDiffuseStateCov, predictedDiffuseStateCovBase(), predictedDiffuseStateCovLength());
        clearSurface(forecastErrorDiffuseCov, forecastErrorDiffuseCovBase(), forecastErrorDiffuseCovLength());
        concentratedLikelihood = false;
        logLikelihood = 0.0;
        scale = 1.0;
        scaleObservationCount = 0;
        nobsDiffuse = 0;
        return self();
    }

    final T applyConserveMemory(int conserveMemory) {
        if ((conserveMemory & KalmanFilter.MEMORY_NO_FORECAST) != 0) {
            suppressForecast();
        }
        if ((conserveMemory & KalmanFilter.MEMORY_NO_PREDICTED) != 0) {
            suppressPredictedState();
        }
        if ((conserveMemory & KalmanFilter.MEMORY_NO_FILTERED) != 0) {
            suppressFilteredState();
        }
        if ((conserveMemory & KalmanFilter.MEMORY_NO_LIKELIHOOD) != 0) {
            suppressLikelihood();
        }
        if ((conserveMemory & KalmanFilter.MEMORY_NO_GAIN) != 0) {
            suppressKalmanGain();
        }
        return self();
    }

    public final double logLikelihood() {
        if (logLikelihoodObsLength() == 0) {
            return logLikelihood;
        }
        double sum = 0.0;
        for (int t = 0; t < nobs; t++) {
            sum += logLikelihoodObs[logLikelihoodObsOffset(t)];
        }
        return sum;
    }

    final void setLogLikelihood(double logLikelihood) {
        this.logLikelihood = logLikelihood;
    }

    final void markFullUnivariateFallback() {
        fallbackRouteCode = 1;
        fallbackPeriodCount = nobs;
        if (nobs == 0) {
            fallbackPeriods = EMPTY_INT;
            return;
        }
        fallbackPeriods = new int[nobs];
        for (int period = 0; period < nobs; period++) {
            fallbackPeriods[period] = period;
        }
    }

    public final boolean usedSingularFallback() {
        return fallbackPeriodCount > 0;
    }

    public final int fallbackPeriodCount() {
        return fallbackPeriodCount;
    }

    public final int[] fallbackPeriods() {
        return Arrays.copyOf(fallbackPeriods, fallbackPeriodCount);
    }

    public final boolean fullUnivariateFallback() {
        return fallbackRouteCode == 1 && fallbackPeriodCount == nobs;
    }

    private void copyFallbackTelemetryTo(T copy) {
        copy.fallbackRouteCode = fallbackRouteCode;
        copy.fallbackPeriodCount = fallbackPeriodCount;
        if (fallbackPeriodCount == 0) {
            copy.fallbackPeriods = EMPTY_INT;
        } else {
            copy.fallbackPeriods = Arrays.copyOf(fallbackPeriods, fallbackPeriodCount);
        }
    }

    public final boolean concentratedLikelihood() {
        return concentratedLikelihood;
    }

    public final double scale() {
        return scale;
    }

    public final int scaleObservationCount() {
        return scaleObservationCount;
    }

    public final int scaleObsLength() {
        return scaleObsLength;
    }

    public final double scaleObs(int t) {
        if (scaleObsLength() == 0) {
            return 0.0;
        }
        return scaleObs[t];
    }

    public final int scaleObsCountLength() {
        return scaleObsCountLength;
    }

    public final int scaleObsCount(int t) {
        if (scaleObsCountLength() == 0) {
            return 0;
        }
        return scaleObsCount[t];
    }

    final void retainScaleObs(int length) {
        if (length <= 0) {
            scaleObsLength = 0;
            return;
        }
        if (scaleObs == null || scaleObs == EMPTY || scaleObs.length < length) {
            scaleObs = new double[length];
        } else {
            Arrays.fill(scaleObs, 0, length, 0.0);
        }
        scaleObsLength = length;
    }

    final void dropScaleObs() {
        scaleObsLength = 0;
    }

    final void retainScaleObsCount(int length) {
        if (length <= 0) {
            scaleObsCountLength = 0;
            return;
        }
        if (scaleObsCount == null || scaleObsCount == EMPTY_INT || scaleObsCount.length < length) {
            scaleObsCount = new int[length];
        } else {
            Arrays.fill(scaleObsCount, 0, length, 0);
        }
        scaleObsCountLength = length;
    }

    final void dropScaleObsCount() {
        scaleObsCountLength = 0;
    }

    public final boolean converged() {
        return converged;
    }

    public final int periodConverged() {
        return periodConverged;
    }

    public final int nobsDiffuse() {
        return nobsDiffuse;
    }

    public final int predictedStateBase() {
        return predictedStateBase;
    }

    public final int predictedStateLength() {
        return predictedStateLength;
    }

    public final int predictedStateOffset(int t) {
        return predictedStateBase() + t * scalarWidth() * kStates;
    }

    public final int predictedStateCovBase() {
        return predictedStateCovBase;
    }

    public final int predictedStateCovLength() {
        return predictedStateCovLength;
    }

    public final int predictedStateCovOffset(int t) {
        return predictedStateCovBase() + t * kStates * scalarWidth() * kStates;
    }

    public final int filteredStateBase() {
        return filteredStateBase;
    }

    public final int filteredStateLength() {
        return filteredStateLength;
    }

    public final int filteredStateOffset(int t) {
        return filteredStateBase() + t * scalarWidth() * kStates;
    }

    public final int filteredStateCovBase() {
        return filteredStateCovBase;
    }

    public final int filteredStateCovLength() {
        return filteredStateCovLength;
    }

    public final int filteredStateCovOffset(int t) {
        return filteredStateCovBase() + t * kStates * scalarWidth() * kStates;
    }

    public final int forecastBase() {
        return forecastBase;
    }

    public final int forecastLength() {
        return forecastLength;
    }

    public final int forecastOffset(int t) {
        return forecastBase() + t * scalarWidth() * kEndog;
    }

    public final int forecastErrorBase() {
        return forecastErrorBase;
    }

    public final int forecastErrorLength() {
        return forecastErrorLength;
    }

    public final int forecastErrorOffset(int t) {
        return forecastErrorBase() + t * scalarWidth() * kEndog;
    }

    public final int forecastErrorCovBase() {
        return forecastErrorCovBase;
    }

    public final int forecastErrorCovLength() {
        return forecastErrorCovLength;
    }

    public final int forecastErrorCovOffset(int t) {
        return forecastErrorCovBase() + t * kEndog * scalarWidth() * kEndog;
    }

    public final int standardizedForecastErrorBase() {
        return standardizedForecastErrorBase;
    }

    public final int standardizedForecastErrorLength() {
        return standardizedForecastErrorLength;
    }

    public final int standardizedForecastErrorOffset(int t) {
        return standardizedForecastErrorBase() + t * scalarWidth() * kEndog;
    }

    public final int kalmanGainBase() {
        return kalmanGainBase;
    }

    public final int kalmanGainLength() {
        return kalmanGainLength;
    }

    public final int kalmanGainOffset(int t) {
        return kalmanGainBase() + t * kStates * scalarWidth() * kEndog;
    }

    public final int logLikelihoodObsBase() {
        return logLikelihoodObsBase;
    }

    public final int logLikelihoodObsLength() {
        return logLikelihoodObsLength;
    }

    public final int logLikelihoodObsOffset(int t) {
        return logLikelihoodObsBase() + t;
    }

    public final int predictedDiffuseStateCovBase() {
        return predictedDiffuseStateCovBase;
    }

    public final int predictedDiffuseStateCovLength() {
        return predictedDiffuseStateCovLength;
    }

    public final int predictedDiffuseStateCovOffset(int t) {
        return predictedDiffuseStateCovBase() + t * kStates * scalarWidth() * kStates;
    }

    public final int forecastErrorDiffuseCovBase() {
        return forecastErrorDiffuseCovBase;
    }

    public final int forecastErrorDiffuseCovLength() {
        return forecastErrorDiffuseCovLength;
    }

    public final int forecastErrorDiffuseCovOffset(int t) {
        return forecastErrorDiffuseCovBase() + t * kEndog * scalarWidth() * kEndog;
    }

    private void bind(int kEndog, int kStates, int nobs, double[] backing, FilterResultLayout layout) {
        this.kEndog = kEndog;
        this.kStates = kStates;
        this.nobs = nobs;
        setPredictedStateSurface(layout.predictedStateLength() == 0 ? EMPTY : backing, layout.predictedStateBase(), layout.predictedStateLength());
        setPredictedStateCovSurface(layout.predictedStateCovLength() == 0 ? EMPTY : backing, layout.predictedStateCovBase(), layout.predictedStateCovLength());
        setFilteredStateSurface(layout.filteredStateLength() == 0 ? EMPTY : backing, layout.filteredStateBase(), layout.filteredStateLength());
        setFilteredStateCovSurface(layout.filteredStateCovLength() == 0 ? EMPTY : backing, layout.filteredStateCovBase(), layout.filteredStateCovLength());
        setForecastSurface(layout.forecastLength() == 0 ? EMPTY : backing, layout.forecastBase(), layout.forecastLength());
        setForecastErrorSurface(layout.forecastErrorLength() == 0 ? EMPTY : backing, layout.forecastErrorBase(), layout.forecastErrorLength());
        setForecastErrorCovSurface(layout.forecastErrorCovLength() == 0 ? EMPTY : backing, layout.forecastErrorCovBase(), layout.forecastErrorCovLength());
        setStandardizedForecastErrorSurface(layout.standardizedForecastErrorLength() == 0 ? EMPTY : backing,
            layout.standardizedForecastErrorBase(), layout.standardizedForecastErrorLength());
        setKalmanGainSurface(layout.kalmanGainLength() == 0 ? EMPTY : backing, layout.kalmanGainBase(), layout.kalmanGainLength());
        setLogLikelihoodObsSurface(layout.logLikelihoodObsLength() == 0 ? EMPTY : backing, layout.logLikelihoodObsBase(), layout.logLikelihoodObsLength());
    }

    private static void clearSurface(double[] values, int base, int length) {
        if (values == null || length == 0) {
            return;
        }
        Arrays.fill(values, base, base + length, 0.0);
    }

    private static void clearSurface(int[] values, int base, int length) {
        if (values == null || length == 0) {
            return;
        }
        Arrays.fill(values, base, base + length, 0);
    }

    private void setPredictedStateSurface(double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        predictedState = assigned;
        predictedStateBase = assignedBase;
        predictedStateLength = assignedLength;
    }

    private void setPredictedStateCovSurface(double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        predictedStateCov = assigned;
        predictedStateCovBase = assignedBase;
        predictedStateCovLength = assignedLength;
    }

    private void setFilteredStateSurface(double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        filteredState = assigned;
        filteredStateBase = assignedBase;
        filteredStateLength = assignedLength;
    }

    private void setFilteredStateCovSurface(double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        filteredStateCov = assigned;
        filteredStateCovBase = assignedBase;
        filteredStateCovLength = assignedLength;
    }

    private void setForecastSurface(double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        forecast = assigned;
        forecastBase = assignedBase;
        forecastLength = assignedLength;
    }

    private void setForecastErrorSurface(double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        forecastError = assigned;
        forecastErrorBase = assignedBase;
        forecastErrorLength = assignedLength;
    }

    private void setForecastErrorCovSurface(double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        forecastErrorCov = assigned;
        forecastErrorCovBase = assignedBase;
        forecastErrorCovLength = assignedLength;
    }

    private void setStandardizedForecastErrorSurface(double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        standardizedForecastError = assigned;
        standardizedForecastErrorBase = assignedBase;
        standardizedForecastErrorLength = assignedLength;
    }

    private void setKalmanGainSurface(double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        kalmanGain = assigned;
        kalmanGainBase = assignedBase;
        kalmanGainLength = assignedLength;
    }

    private void setLogLikelihoodObsSurface(double[] surface, int base, int length) {
        double[] assigned = surface == null ? EMPTY : surface;
        int assignedBase = assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == EMPTY ? 0 : length;
        logLikelihoodObs = assigned;
        logLikelihoodObsBase = assignedBase;
        logLikelihoodObsLength = assignedLength;
    }

    private void setPredictedDiffuseStateCovSurface(double[] surface, int base, int length) {
        double[] assigned = surface;
        int assignedBase = assigned == null || assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == null || assigned == EMPTY ? 0 : length;
        predictedDiffuseStateCov = assigned;
        predictedDiffuseStateCovBase = assignedBase;
        predictedDiffuseStateCovLength = assignedLength;
    }

    private void setForecastErrorDiffuseCovSurface(double[] surface, int base, int length) {
        double[] assigned = surface;
        int assignedBase = assigned == null || assigned == EMPTY ? 0 : base;
        int assignedLength = assigned == null || assigned == EMPTY ? 0 : length;
        forecastErrorDiffuseCov = assigned;
        forecastErrorDiffuseCovBase = assignedBase;
        forecastErrorDiffuseCovLength = assignedLength;
    }
}