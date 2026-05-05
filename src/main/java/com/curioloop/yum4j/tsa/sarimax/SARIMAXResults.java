package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.kalman.filter.FilterResult;
import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.mle.MLECovariance;
import com.curioloop.yum4j.kalman.mle.MLEResults;
import com.curioloop.yum4j.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
import com.curioloop.yum4j.optim.Optimization;

import java.util.Arrays;

public final class SARIMAXResults extends MLEResults {

    private final SARIMAX model;
    private final FilterResult filterResult;
    private final int logLikelihoodBurn;
    private final double scale;
    private final int nobsEffective;
    private final int kDiffuseStates;
    private final int dfModel;

    SARIMAXResults(SARIMAX model,
                   Optimization optimization,
                   double[] params,
                   double[] unconstrainedParams,
                   FilterResult filterResult,
                   MLECovariance covType) {
        super(optimization, params, unconstrainedParams, model.parameterNames(), covType);
        this.model = model;
        this.filterResult = filterResult.copy();
        this.logLikelihoodBurn = Math.min(model.likelihoodBurn(), this.filterResult.logLikelihoodObs.length);
        this.scale = model.spec().concentrateScale()
            ? model.concentratedScale(this.filterResult, logLikelihoodBurn)
            : (params.length == 0 ? 1.0 : params[params.length - 1]);
        if (model.spec().concentrateScale()) {
            model.applyConcentratedScale(this.filterResult, scale);
        }
        applyLikelihoodBurn(this.filterResult, logLikelihoodBurn);
        this.nobsEffective = model.effectiveObservationCount() - logLikelihoodBurn;
        this.kDiffuseStates = model.diffuseStateCount();
        this.dfModel = params.length + kDiffuseStates + (model.spec().concentrateScale() ? 1 : 0);
    }

    public FilterResult filterResult() {
        return filterResult.copy();
    }

    public double[] logLikelihoodObs() {
        return Arrays.copyOf(filterResult.logLikelihoodObs, filterResult.logLikelihoodObs.length);
    }

    public int logLikelihoodBurn() {
        return logLikelihoodBurn;
    }

    public int kDiffuseStates() {
        return kDiffuseStates;
    }

    public double[] fittedValues() {
        double[] fitted = new double[filterResult.nobs];
        for (int t = 0; t < fitted.length; t++) {
            fitted[t] = filterResult.forecast(0, t);
        }
        return fitted;
    }

    public double[] residuals() {
        double[] residuals = new double[filterResult.nobs];
        double[] effectiveEndog = model.effectiveEndog();
        for (int t = 0; t < residuals.length; t++) {
            residuals[t] = Double.isNaN(effectiveEndog[t])
                ? Double.NaN
                : filterResult.forecastError(0, t);
        }
        return residuals;
    }

    public double[] standardizedForecastsError() {
        double[] standardized = new double[filterResult.nobs];
        double[] effectiveEndog = model.effectiveEndog();
        for (int t = 0; t < standardized.length; t++) {
            double forecastVariance = filterResult.forecastErrorCov(0, 0, t);
            if (Double.isNaN(effectiveEndog[t]) || !(forecastVariance > 0.0) || !Double.isFinite(forecastVariance)) {
                standardized[t] = Double.NaN;
                continue;
            }
            standardized[t] = filterResult.forecastError(0, t) / Math.sqrt(forecastVariance);
        }
        return standardized;
    }

    public SmootherResult smooth() {
        return model.smooth(paramsInternal(), SmootherSpec.conventional());
    }

    public SmootherResult smooth(SmootherSpec smootherSpec) {
        return model.smooth(paramsInternal(), smootherSpec == null ? SmootherSpec.conventional() : smootherSpec);
    }

    public SARIMAXPrediction predict(int start, int end) {
        return predict(start, end, false, null);
    }

    public SARIMAXPrediction predict(int start, int end, boolean dynamic, double[][] futureExog) {
        int nobs = model.spec().observationCount();
        if (start < 0) {
            throw new IllegalArgumentException("start must be non-negative");
        }
        if (end < start) {
            throw new IllegalArgumentException("end must be greater than or equal to start");
        }
        if (!dynamic && end < nobs) {
            return slicePrediction(filterResult, start, end, false);
        }

        int requiredLength = end + 1;
        double[] extendedEndog = Arrays.copyOf(model.spec().endog(), Math.max(requiredLength, nobs));
        for (int index = nobs; index < extendedEndog.length; index++) {
            extendedEndog[index] = Double.NaN;
        }
        if (dynamic) {
            for (int index = Math.min(start, nobs); index < Math.min(nobs, extendedEndog.length); index++) {
                extendedEndog[index] = Double.NaN;
            }
        }

        int outOfSample = Math.max(0, requiredLength - nobs);
        double[][] extendedExog = extendExog(outOfSample, futureExog);
        SARIMAX extendedModel = model.extend(extendedEndog, extendedExog);
        FilterResult predictionFilter = extendedModel.filter(paramsInternal(), FilterSpec.full());
        if (model.spec().concentrateScale()) {
            extendedModel.applyConcentratedScale(predictionFilter, scale);
        }
        return slicePrediction(predictionFilter, start, end, dynamic);
    }

    public SARIMAXPrediction forecast(int steps, double[][] futureExog) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive");
        }
        int start = model.spec().observationCount();
        int end = start + steps - 1;
        return predict(start, end, false, futureExog);
    }

    @Override
    public String summary() {
        return super.summary();
    }

    @Override
    public String summary(double alpha) {
        return super.summary(alpha);
    }

    private double[][] extendExog(int outOfSample, double[][] futureExog) {
        double[][] baseExog = model.spec().exog();
        if (baseExog == null) {
            if (futureExog != null && futureExog.length > 0) {
                throw new IllegalArgumentException("futureExog provided for a model without exogenous regressors");
            }
            return null;
        }
        if (outOfSample == 0) {
            return baseExog;
        }
        if (futureExog == null || futureExog.length != outOfSample) {
            throw new IllegalArgumentException("futureExog row count must match out-of-sample horizon");
        }
        int width = baseExog[0].length;
        double[][] extended = Arrays.copyOf(baseExog, baseExog.length + outOfSample);
        for (int row = 0; row < futureExog.length; row++) {
            if (futureExog[row] == null || futureExog[row].length != width) {
                throw new IllegalArgumentException("futureExog rows must match exogenous dimension");
            }
            extended[baseExog.length + row] = Arrays.copyOf(futureExog[row], width);
        }
        return extended;
    }

    private static SARIMAXPrediction slicePrediction(FilterResult filterResult, int start, int end, boolean dynamic) {
        double[] mean = new double[end - start + 1];
        double[] variance = new double[end - start + 1];
        for (int t = start; t <= end; t++) {
            mean[t - start] = filterResult.forecast(0, t);
            variance[t - start] = filterResult.forecastErrorCov(0, 0, t);
        }
        return new SARIMAXPrediction(start, end, dynamic, mean, variance);
    }

    private static void applyLikelihoodBurn(FilterResult filterResult, int burn) {
        Arrays.fill(filterResult.logLikelihoodObs, 0, burn, 0.0);
    }

    @Override
    protected double[] computeCovParamsOim() {
        return model.covarianceFromObservedInformation(paramsInternal());
    }

    @Override
    protected double[] computeCovParamsApprox() {
        if (paramsInternal().length == 0) {
            return new double[0];
        }
        return model.covarianceFromApproximateInformation(paramsInternal());
    }

    @Override
    protected double[] computeCovParamsRobustOim() {
        double[] params = paramsInternal();
        if (params.length == 0) {
            return new double[0];
        }

        return nobsEffective <= 0
            ? nanCovariance(params.length)
            : model.robustCovarianceFromObservedInformation(params);
    }

    @Override
    protected double[] computeCovParamsRobustApprox() {
        double[] params = paramsInternal();
        if (params.length == 0) {
            return new double[0];
        }

        return nobsEffective <= 0
            ? nanCovariance(params.length)
            : model.robustCovarianceFromApproximateInformation(params);
    }

    @Override
    protected double[] computeCovParamsOpg() {
        return model.covarianceFromOpg(paramsInternal());
    }

    @Override
    protected double logLikelihoodInternal() {
        return filterResult.logLikelihood();
    }

    @Override
    protected double scaleInternal() {
        return scale;
    }

    @Override
    protected int nobsEffectiveInternal() {
        return nobsEffective;
    }

    @Override
    protected int dfModelInternal() {
        return dfModel;
    }

    @Override
    protected int observationCountInternal() {
        return model.spec().observationCount();
    }

    @Override
    protected String modelNameInternal() {
        return "SARIMAX";
    }

    @Override
    protected boolean summaryIncludesScaleInternal() {
        return model.spec().concentrateScale();
    }

    @Override
    protected String summaryDiagnosticsTextInternal() {
        return "Diagnostics: unavailable in current SARIMAXResults summary surface";
    }

    private static double[] nanCovariance(int dimension) {
        double[] nan = new double[dimension * dimension];
        Arrays.fill(nan, Double.NaN);
        return nan;
    }
}