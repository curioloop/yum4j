package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.arena.FilterResultLayout;

import java.util.Objects;

/**
 * Flat storage for one Kalman filter run.
 *
 * <p>Arrays are grouped by predicted, filtered, forecast, and optional diffuse
 * quantities, matching the package-level separation used by the reference design.</p>
 */
public class FilterResult extends FilterResultBase<FilterResult> {

    FilterResult() {
        super();
    }

    public FilterResult(int kEndog, int kStates, int nobs) {
        this(kEndog, kStates, nobs, true, true, true, true, true);
    }

    public FilterResult(int kEndog, int kStates, int nobs,
                        boolean storeForecast,
                        boolean storePredicted,
                        boolean storeFiltered,
                        boolean storeGain,
                        boolean storeLikelihood) {
                this(kEndog, kStates, nobs,
                    FilterResultLayout.create(1, kEndog, kStates, nobs,
                        storeForecast, storePredicted, storeFiltered, storeGain, storeLikelihood));
    }

                public FilterResult(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
                super(kEndog, kStates, nobs, layout);
    }

    @Override
    protected int scalarWidth() {
        return 1;
    }

    @Override
    protected FilterResult newResult(int kEndog, int kStates, int nobs, FilterResultLayout layout) {
        return new FilterResult(kEndog, kStates, nobs, layout);
    }

    static double[] emptyArray() {
        return EMPTY;
    }

    public double predictedState(int i, int t) {
        if (predictedStateLength() == 0) {
            return 0.0;
        }
        return predictedState[predictedStateOffset(t) + i];
    }

    public double predictedStateCov(int i, int j, int t) {
        if (predictedStateCovLength() == 0) {
            return 0.0;
        }
        return predictedStateCov[predictedStateCovOffset(t) + i * kStates + j];
    }

    public double filteredState(int i, int t) {
        if (filteredStateLength() == 0) {
            return 0.0;
        }
        return filteredState[filteredStateOffset(t) + i];
    }

    public double filteredStateCov(int i, int j, int t) {
        if (filteredStateCovLength() == 0) {
            return 0.0;
        }
        return filteredStateCov[filteredStateCovOffset(t) + i * kStates + j];
    }

    public double forecast(int i, int t) {
        if (forecastLength() == 0) {
            return 0.0;
        }
        return forecast[forecastOffset(t) + i];
    }

    public double forecastError(int i, int t) {
        if (forecastErrorLength() == 0) {
            return 0.0;
        }
        return forecastError[forecastErrorOffset(t) + i];
    }

    public double forecastErrorCov(int i, int j, int t) {
        if (forecastErrorCovLength() == 0) {
            return 0.0;
        }
        return forecastErrorCov[forecastErrorCovOffset(t) + i * kEndog + j];
    }

    public double kalmanGain(int i, int j, int t) {
        if (kalmanGainLength() == 0) {
            return 0.0;
        }
        return kalmanGain[kalmanGainOffset(t) + i * kEndog + j];
    }

    public double[] standardizedForecastError() {
        return standardizedForecastError(0.0);
    }

    public double[] standardizedForecastError(double tolerance) {
        int length = standardizedForecastErrorOutputLength();
        if (length == 0) {
            return emptyArray();
        }
        double[] standardized = new double[length];
        copyStandardizedForecastError(standardized, 0, tolerance, null);
        return standardized;
    }

    public int standardizedForecastErrorOutputLength() {
        if (standardizedForecastErrorLength() > 0) {
            return standardizedForecastErrorLength();
        }
        return forecastErrorLength() == 0 || forecastErrorCovLength() == 0 ? 0 : kEndog * nobs;
    }

    public boolean hasRetainedStandardizedForecastError() {
        return standardizedForecastErrorLength() > 0;
    }

    public double[] retainedStandardizedForecastErrorBackingOrNull() {
        return hasRetainedStandardizedForecastError() ? standardizedForecastError : null;
    }

    public void copyStandardizedForecastError(double[] target, int targetOffset) {
        copyStandardizedForecastError(target, targetOffset, 0.0, null);
    }

    public void copyStandardizedForecastError(double[] target,
                                              int targetOffset,
                                              double tolerance,
                                              double[] factorScratch) {
        Objects.requireNonNull(target, "target");
        int length = standardizedForecastErrorOutputLength();
        if (targetOffset < 0 || target.length < targetOffset + length) {
            throw new IllegalArgumentException("target is too small for standardized forecast errors");
        }
        if (length == 0) {
            return;
        }
        if (standardizedForecastErrorLength() > 0) {
            System.arraycopy(standardizedForecastError,
                standardizedForecastErrorBase(),
                target,
                targetOffset,
                standardizedForecastErrorLength());
            return;
        }
        if (kEndog == 1) {
            standardizeScalarForecastErrors(target, targetOffset, tolerance);
        } else {
            double[] factor = factorScratch == null || factorScratch.length < kEndog * kEndog
                ? new double[kEndog * kEndog]
                : factorScratch;
            standardizeMultivariateForecastErrors(target, targetOffset, factor, tolerance);
        }
    }

    public double standardizedForecastError(int i, int t) {
        if (standardizedForecastErrorLength() == 0) {
            return 0.0;
        }
        return standardizedForecastError[standardizedForecastErrorOffset(t) + i];
    }

    private void standardizeScalarForecastErrors(double[] standardized, int standardizedOffset, double tolerance) {
        for (int t = 0; t < nobs; t++) {
            standardized[standardizedOffset + t] = diffuseForecastError(t, tolerance)
                ? 0.0
                : ForecastErrorSolver.standardizeScalar(
                    forecastError(0, t),
                    forecastErrorCov(0, 0, t),
                    tolerance);
        }
    }

    private void standardizeMultivariateForecastErrors(double[] standardized,
                                                       int standardizedOffset,
                                                       double[] factor,
                                                       double tolerance) {
        for (int t = 0; t < nobs; t++) {
            int outBase = standardizedOffset + t * kEndog;
            if (diffuseForecastError(t, tolerance)) {
                fillNaN(standardized, outBase, kEndog);
                continue;
            }
            if (!ForecastErrorSolver.standardizeCholesky(
                    forecastErrorCov,
                    forecastErrorCovOffset(t),
                    kEndog,
                    forecastError,
                    forecastErrorOffset(t),
                    standardized,
                    outBase,
                    kEndog,
                    factor,
                    tolerance)) {
                fillNaN(standardized, outBase, kEndog);
            }
        }
    }

    private static void fillNaN(double[] values, int offset, int length) {
        for (int i = 0; i < length; i++) {
            values[offset + i] = Double.NaN;
        }
    }

    private boolean diffuseForecastError(int t, double tolerance) {
        if (t >= nobsDiffuse || forecastErrorDiffuseCov == null || forecastErrorDiffuseCovLength() == 0) {
            return false;
        }
        double threshold = Math.max(0.0, tolerance);
        int offset = forecastErrorDiffuseCovOffset(t);
        for (int i = 0; i < kEndog * kEndog; i++) {
            if (Math.abs(forecastErrorDiffuseCov[offset + i]) > threshold) {
                return true;
            }
        }
        return false;
    }

    public double predictedDiffuseStateCov(int i, int j, int t) {
        if (predictedDiffuseStateCov == null || predictedDiffuseStateCovLength() == 0) {
            return 0.0;
        }
        return predictedDiffuseStateCov[predictedDiffuseStateCovOffset(t) + i * kStates + j];
    }

    public double forecastErrorDiffuseCov(int i, int j, int t) {
        if (forecastErrorDiffuseCov == null || forecastErrorDiffuseCovLength() == 0) {
            return 0.0;
        }
        return forecastErrorDiffuseCov[forecastErrorDiffuseCovOffset(t) + i * kEndog + j];
    }
}
