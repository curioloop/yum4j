package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.ssm.kalman.KalmanFiltering;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.mle.FixedParameterResults;
import com.curioloop.yum4j.ssm.kalman.mle.FixedParameters;
import com.curioloop.yum4j.ssm.kalman.mle.MLEResults;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherDiagnostics;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.stats.test.Box;
import com.curioloop.yum4j.stats.test.BreakVar;
import com.curioloop.yum4j.stats.test.JarqueBera;
import com.curioloop.yum4j.tsa.diagnostics.DiagnosticSupport;
import com.curioloop.yum4j.tsa.lifecycle.ExogenousDataSupport;
import com.curioloop.yum4j.tsa.lifecycle.TimeSeriesArraySupport;
import com.curioloop.yum4j.tsa.prediction.PredictionInformationSet;
import com.curioloop.yum4j.tsa.prediction.PredictionKind;
import com.curioloop.yum4j.tsa.prediction.PredictionToolkit;
import com.curioloop.yum4j.tsa.prediction.StatePredictionProjector;
import com.curioloop.yum4j.tsa.prediction.StatePredictionSurface;
import com.curioloop.yum4j.tsa.prediction.StatePredictionSurfaces;
import com.curioloop.yum4j.tsa.statespace.ImpulseResponse;
import com.curioloop.yum4j.tsa.statespace.ImpulseResponseRepetitions;
import com.curioloop.yum4j.tsa.statespace.SimulationAnchor;
import com.curioloop.yum4j.tsa.statespace.SimulationSupport;
import com.curioloop.yum4j.tsa.statespace.SimulationSmootherRepetitions;
import com.curioloop.yum4j.tsa.statespace.StateSpaceImpulseResponses;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

public final class SARIMAXResults extends MLEResults {

    private enum FilterResultOwnership {
        COPY,
        ADOPT
    }

    private final SARIMAX model;
    private final FilterResult filterResult;
    private final int logLikelihoodBurn;
    private final double scale;
    private final int nobsEffective;
    private final int kDiffuseStates;
    private final int dfModel;
    private final FixedParameters fixedParameters;

    SARIMAXResults(SARIMAX model,
                   Optimization optimization,
                   double[] params,
                   double[] unconstrainedParams,
                   FilterResult filterResult,
                   MLEResults.Covariance covType) {
        this(model, optimization, params, unconstrainedParams, filterResult, covType, FixedParameters.none());
    }

    SARIMAXResults(SARIMAX model,
                   Optimization optimization,
                   double[] params,
                   double[] unconstrainedParams,
                   FilterResult filterResult,
                   MLEResults.Covariance covType,
                   FixedParameters fixedParameters) {
            this(model, optimization, params, unconstrainedParams, filterResult, covType, fixedParameters,
                FilterResultOwnership.COPY);
            }

            static SARIMAXResults adoptFilterResult(SARIMAX model,
                                Optimization optimization,
                                double[] params,
                                double[] unconstrainedParams,
                                FilterResult filterResult,
                                MLEResults.Covariance covType,
                                FixedParameters fixedParameters) {
            return new SARIMAXResults(model, optimization, params, unconstrainedParams, filterResult, covType,
                fixedParameters, FilterResultOwnership.ADOPT);
            }

            private SARIMAXResults(SARIMAX model,
                       Optimization optimization,
                       double[] params,
                       double[] unconstrainedParams,
                       FilterResult filterResult,
                       MLEResults.Covariance covType,
                       FixedParameters fixedParameters,
                       FilterResultOwnership ownership) {
        super(optimization, params, unconstrainedParams, model.parameterNames(), covType);
        this.model = model;
        this.fixedParameters = (fixedParameters == null ? FixedParameters.none() : fixedParameters).validate(params.length);
            FilterResult ownedFilterResult = Objects.requireNonNull(filterResult, "filterResult");
            this.filterResult = ownership == FilterResultOwnership.COPY ? ownedFilterResult.copy() : ownedFilterResult;
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
        this.dfModel = FixedParameterResults.dfModel(params.length, this.fixedParameters,
            kDiffuseStates + (model.spec().concentrateScale() ? 1 : 0));
    }

    public FixedParameters fixedParameters() {
        return fixedParameters;
    }

    public int fixedParameterCount() {
        return fixedParameters.size();
    }

    public int freeParameterCount() {
        return FixedParameterResults.freeParameterCount(paramsInternal().length, fixedParameters);
    }

    public boolean isFixedParameter(int index) {
        return fixedParameters.isFixed(index);
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
        requireForecastMean("fitted values are not available when forecast means were not retained");
        double[] fitted = new double[filterResult.nobs];
        for (int t = 0; t < fitted.length; t++) {
            fitted[t] = filterResult.forecast(0, t);
        }
        return fitted;
    }

    public double[] residuals() {
        requireForecastMean("residuals are not available when forecast means were not retained");
        requireForecastError("residuals are not available when forecast errors were not retained");
        double[] residuals = new double[filterResult.nobs];
        double[] effectiveEndog = model.effectiveEndog();
        for (int t = 0; t < residuals.length; t++) {
            residuals[t] = Double.isNaN(effectiveEndog[t])
                ? Double.NaN
                : filterResult.forecastError(0, t);
        }
        return residuals;
    }

    public double[] standardizedForecastError() {
        double[] standardized = new double[filterResult.standardizedForecastErrorOutputLength()];
        filterResult.copyStandardizedForecastError(standardized, 0);
        if (standardized.length == 0) {
            return standardized;
        }
        double[] effectiveEndog = model.effectiveEndog();
        for (int t = 0; t < standardized.length; t++) {
            if (Double.isNaN(effectiveEndog[t])) {
                standardized[t] = Double.NaN;
            }
        }
        return standardized;
    }

    public SmootherResult smooth() {
        return model.smooth(paramsInternal(), SmootherOptions.defaults());
    }

    public SmootherResult smooth(SmootherOptions smootherOptions) {
        return model.smooth(paramsInternal(), smootherOptions == null ? SmootherOptions.defaults() : smootherOptions);
    }

    public SARIMAXResults append(double[] endog) {
        return append(endog, null);
    }

    public SARIMAXResults append(double[] endog, double[][] exog) {
        double[] newEndog = TimeSeriesArraySupport.copyNonEmpty(endog, "endog");
        double[] combinedEndog = TimeSeriesArraySupport.concat(model.spec().endog(), newEndog);
        double[][] combinedExog = concatExog(newEndog.length, exog);
        SARIMAX appendedModel = model.extend(combinedEndog, combinedExog, model.spec().trendOffset());
        return resultFor(appendedModel, appendedModel.filter(paramsInternal(), resultFilterOptions(appendedModel)));
    }

    public SARIMAXResults appendRefit(double[] endog) {
        return appendRefit(endog, null, null);
    }

    public SARIMAXResults appendRefit(double[] endog, double[][] exog) {
        return appendRefit(endog, exog, null);
    }

    public SARIMAXResults appendRefit(double[] endog, double[][] exog, SARIMAXFitOptions fitOptions) {
        double[] newEndog = TimeSeriesArraySupport.copyNonEmpty(endog, "endog");
        double[] combinedEndog = TimeSeriesArraySupport.concat(model.spec().endog(), newEndog);
        double[][] combinedExog = concatExog(newEndog.length, exog);
        return model.extend(combinedEndog, combinedExog, model.spec().trendOffset())
            .fit(refitOptions(fitOptions));
    }

    public SARIMAXResults extend(double[] endog) {
        return extend(endog, null);
    }

    public SARIMAXResults extend(double[] endog, double[][] exog) {
        double[] newEndog = TimeSeriesArraySupport.copyNonEmpty(endog, "endog");
        double[][] newExog = validateNewExog(newEndog.length, exog);
        SARIMAX extendedModel = model.extend(newEndog, newExog, model.spec().trendOffset() + filterResult.nobs);
        FilterResult extendedFilter = new KalmanFiltering()
            .model(extendedModel.snapshotModel(paramsInternal()))
            .initialState(finalPredictedInitialState())
            .options(resultFilterOptions(extendedModel))
            .filter();
        return resultFor(extendedModel, extendedFilter);
    }

    public SARIMAXResults extendRefit(double[] endog) {
        return extendRefit(endog, null, null);
    }

    public SARIMAXResults extendRefit(double[] endog, double[][] exog) {
        return extendRefit(endog, exog, null);
    }

    public SARIMAXResults extendRefit(double[] endog, double[][] exog, SARIMAXFitOptions fitOptions) {
        double[] newEndog = TimeSeriesArraySupport.copyNonEmpty(endog, "endog");
        double[][] newExog = validateNewExog(newEndog.length, exog);
        return model.extend(newEndog, newExog, model.spec().trendOffset() + filterResult.nobs)
            .fit(refitOptions(fitOptions));
    }

    public SARIMAXResults apply(double[] endog) {
        return apply(endog, null);
    }

    public SARIMAXResults apply(double[] endog, double[][] exog) {
        double[] newEndog = TimeSeriesArraySupport.copyNonEmpty(endog, "endog");
        double[][] newExog = validateNewExog(newEndog.length, exog);
        SARIMAX appliedModel = model.extend(newEndog, newExog, model.spec().trendOffset());
        return resultFor(appliedModel, appliedModel.filter(paramsInternal(), resultFilterOptions(appliedModel)));
    }

    public SARIMAXResults applyRefit(double[] endog) {
        return applyRefit(endog, null, null);
    }

    public SARIMAXResults applyRefit(double[] endog, double[][] exog) {
        return applyRefit(endog, exog, null);
    }

    public SARIMAXResults applyRefit(double[] endog, double[][] exog, SARIMAXFitOptions fitOptions) {
        double[] newEndog = TimeSeriesArraySupport.copyNonEmpty(endog, "endog");
        double[][] newExog = validateNewExog(newEndog.length, exog);
        return model.extend(newEndog, newExog, model.spec().trendOffset())
            .fit(refitOptions(fitOptions));
    }

    public SimulationSmootherResult simulate(int nsimulations) {
        return simulate(nsimulations, (double[][]) null, (Random) null, SimulationSmootherOptions.defaults().withGeneratedOutputs());
    }

    public SimulationSmootherResult simulate(int nsimulations, Random random) {
        return simulate(nsimulations, (double[][]) null, random, SimulationSmootherOptions.defaults().withGeneratedOutputs());
    }

    public SimulationSmootherResult simulate(int nsimulations,
                                             double[][] exog,
                                             Random random,
                                             SimulationSmootherOptions options) {
        SimulationSupport.requirePositiveSimulations(nsimulations);
        return SimulationSupport.simulate(simulationSnapshot(nsimulations, exog), initialPredictedState(), random, options);
    }

    public SimulationSmootherResult simulate(int nsimulations,
                                             SimulationAnchor anchor,
                                             Random random,
                                             SimulationSmootherOptions options) {
        return simulate(nsimulations, null, simulationInitialState(anchor), random, options);
    }

    public SimulationSmootherResult simulate(int nsimulations,
                                             int anchor,
                                             double[][] exog,
                                             Random random,
                                             SimulationSmootherOptions options) {
        return simulate(nsimulations, exog, predictedInitialStateAt(anchor), random, options);
    }

    public SimulationSmootherRepetitions simulateRepetitions(int nsimulations, int repetitions) {
        return simulateRepetitions(nsimulations, repetitions, SimulationAnchor.INITIAL, new Random(),
            SimulationSmootherOptions.defaults().withGeneratedOutputs());
    }

    public SimulationSmootherRepetitions simulateRepetitions(int nsimulations,
                                                             int repetitions,
                                                             SimulationAnchor anchor,
                                                             Random random,
                                                             SimulationSmootherOptions options) {
        SimulationSupport.requirePositiveRepetitions(repetitions);
        SimulationSmootherResult[] results = new SimulationSmootherResult[repetitions];
        Random rng = random == null ? new Random() : random;
        for (int repetition = 0; repetition < repetitions; repetition++) {
            results[repetition] = simulate(nsimulations, anchor, rng, options);
        }
        return new SimulationSmootherRepetitions(results);
    }

    public SimulationSmootherResult simulate(int nsimulations,
                                             double[] measurementDisturbanceVariates,
                                             double[] stateDisturbanceVariates,
                                             double[] initialStateVariates) {
        return simulate(nsimulations,
            null,
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            SimulationSmootherOptions.defaults().withGeneratedOutputs());
    }

    public SimulationSmootherResult simulate(int nsimulations,
                                             double[][] exog,
                                             double[] measurementDisturbanceVariates,
                                             double[] stateDisturbanceVariates,
                                             double[] initialStateVariates,
                                             SimulationSmootherOptions options) {
        SimulationSupport.requirePositiveSimulations(nsimulations);
        return SimulationSupport.simulate(simulationSnapshot(nsimulations, exog), initialPredictedState(),
            measurementDisturbanceVariates, stateDisturbanceVariates, initialStateVariates, options);
    }

    public ImpulseResponse impulseResponses(int steps) {
        return impulseResponses(steps, 0, false, false);
    }

    public ImpulseResponse impulseResponses(int steps, int impulse) {
        return impulseResponses(steps, impulse, false, false);
    }

    public ImpulseResponse impulseResponses(int steps,
                                            int impulse,
                                            boolean orthogonalized,
                                            boolean cumulative) {
        return StateSpaceImpulseResponses.compute(model.snapshotModel(paramsInternal()),
            steps, impulse, orthogonalized, cumulative);
    }

    public ImpulseResponse impulseResponses(int steps,
                                            int impulse,
                                            boolean orthogonalized,
                                            boolean cumulative,
                                            int anchor) {
        return StateSpaceImpulseResponses.compute(model.snapshotModel(paramsInternal()),
            steps, impulse, orthogonalized, cumulative, anchor);
    }

    public ImpulseResponse impulseResponses(int steps,
                                            double[] impulse,
                                            boolean orthogonalized,
                                            boolean cumulative) {
        return StateSpaceImpulseResponses.compute(model.snapshotModel(paramsInternal()),
            steps, impulse, orthogonalized, cumulative);
    }

    public ImpulseResponse impulseResponses(int steps,
                                            double[] impulse,
                                            boolean orthogonalized,
                                            boolean cumulative,
                                            int anchor) {
        return StateSpaceImpulseResponses.compute(model.snapshotModel(paramsInternal()),
            steps, impulse, orthogonalized, cumulative, anchor);
    }

    public ImpulseResponseRepetitions impulseResponseRepetitions(int steps,
                                                                 int impulse,
                                                                 int repetitions,
                                                                 boolean orthogonalized,
                                                                 boolean cumulative) {
        SimulationSupport.requirePositiveRepetitions(repetitions);
        ImpulseResponse[] responses = new ImpulseResponse[repetitions];
        for (int repetition = 0; repetition < repetitions; repetition++) {
            responses[repetition] = impulseResponses(steps, impulse, orthogonalized, cumulative);
        }
        return new ImpulseResponseRepetitions(responses);
    }

    public JarqueBera testNormality() {
        return JarqueBera.test(DiagnosticSupport.finiteValues(residuals()));
    }

    public BreakVar testHeteroskedasticity() {
        return testHeteroskedasticity(1.0 / 3.0, BreakVar.Alternative.TWO_SIDED, true);
    }

    public BreakVar testHeteroskedasticity(double subsetLen,
                                           BreakVar.Alternative alternative,
                                           boolean useF) {
        return BreakVar.test(DiagnosticSupport.finiteValues(residuals()), subsetLen, alternative, useF);
    }

    public Box testSerialCorrelation(int... lags) {
        return Box.test(DiagnosticSupport.finiteValues(residuals()), Box.Statistic.LJUNG_BOX,
            Math.max(0, model.spec().order().autoregressive() + model.spec().order().movingAverage()),
            false, 0, lags == null ? new int[0] : lags);
    }

    public SmootherDiagnostics.SmoothedStateWeights smoothedStateWeights() {
        return smoothedStateWeights(SmootherOptions.defaults());
    }

    public SmootherDiagnostics.SmoothedStateWeights smoothedStateWeights(SmootherOptions options) {
        return SmootherDiagnostics.computeSmoothedStateWeights(model.snapshotModel(paramsInternal()),
            initialPredictedState(), options);
    }

    public SmootherDiagnostics.SmoothedDecomposition smoothedDecomposition() {
        return smoothedDecomposition(SmootherOptions.defaults());
    }

    public SmootherDiagnostics.SmoothedDecomposition smoothedDecomposition(SmootherOptions options) {
        return SmootherDiagnostics.getSmoothedDecomposition(model.snapshotModel(paramsInternal()),
            initialPredictedState(), options);
    }

    public SmootherDiagnostics.NewsImpact news(SARIMAXResults updated) {
        return news(updated, SmootherOptions.defaults());
    }

    public SmootherDiagnostics.NewsImpact news(SARIMAXResults updated, SmootherOptions options) {
        if (updated == null) {
            throw new IllegalArgumentException("updated results must not be null");
        }
        return SmootherDiagnostics.news(model.snapshotModel(paramsInternal()),
            initialPredictedState(),
            updated.model.snapshotModel(updated.paramsInternal()),
            updated.initialPredictedState(),
            options);
    }

    public SARIMAXPrediction predict() {
        return predict(0, model.effectiveObservationCount() - 1, -1, null);
    }

    public SARIMAXPrediction predict(int start, int end) {
        return predict(start, end, -1, null);
    }

    public SARIMAXPrediction predict(int start,
                                     int end,
                                     PredictionInformationSet informationSet,
                                     boolean signalOnly) {
        return predict(start, end, -1, null, false, informationSet, signalOnly);
    }

    public SARIMAXPrediction predict(int start, int end, int dynamicStart, double[][] futureExog) {
        return predict(start, end, dynamicStart, futureExog, false, PredictionInformationSet.PREDICTED, false);
    }

    public SARIMAXPrediction predict(int start,
                                     int end,
                                     int dynamicStart,
                                     double[][] futureExog,
                                     PredictionInformationSet informationSet,
                                     boolean signalOnly) {
        return predict(start, end, dynamicStart, futureExog, false, informationSet, signalOnly);
    }

    private SARIMAXPrediction predict(int start,
                                      int end,
                                      int dynamicStart,
                                      double[][] futureExog,
                                      boolean forecastOnly,
                                      PredictionInformationSet informationSet,
                                      boolean signalOnly) {
        PredictionInformationSet resolvedInformationSet = PredictionToolkit.resolveInformationSet(informationSet);
        int nobs = model.effectiveObservationCount();
        int originalNobs = model.spec().observationCount();
        int observationOffset = originalNobs - nobs;
        int effectiveDynamicStart = PredictionToolkit.effectiveDynamicStart(start, end, dynamicStart);
        boolean dynamic = effectiveDynamicStart >= 0;
        PredictionKind kind = PredictionToolkit.predictionKind(start, end, effectiveDynamicStart, nobs, forecastOnly);

        if (resolvedInformationSet != PredictionInformationSet.PREDICTED && (dynamic || end >= nobs)) {
            throw new IllegalArgumentException("filtered and smoothed predictions are only available for non-dynamic in-sample ranges");
        }

        if (!dynamic && end < nobs) {
            if (resolvedInformationSet == PredictionInformationSet.PREDICTED && !signalOnly) {
                requireForecastMean("in-sample prediction is not available when forecast means were not retained");
                return slicePrediction(filterResult, start, end, -1, kind);
            }
            return sliceStatePrediction(model.snapshotModel(paramsInternal()), filterResult, null,
                start, end, -1, kind, resolvedInformationSet, signalOnly);
        }

        int requiredLength = end + 1;
        int outOfSample = Math.max(0, requiredLength - nobs);
        if (!dynamic && start >= nobs && originalNobs == nobs && !model.spec().concentrateScale()) {
            return warmStartedOutOfSamplePrediction(start, end, outOfSample, futureExog, kind,
                resolvedInformationSet, signalOnly);
        }

        return fullRefilterPrediction(start, end, effectiveDynamicStart, outOfSample, originalNobs,
            observationOffset, futureExog, kind, resolvedInformationSet, signalOnly);
    }

    private SARIMAXPrediction fullRefilterPrediction(int start,
                                                     int end,
                                                     int effectiveDynamicStart,
                                                     int outOfSample,
                                                     int originalNobs,
                                                     int observationOffset,
                                                     double[][] futureExog,
                                                     PredictionKind kind,
                                                     PredictionInformationSet informationSet,
                                                     boolean signalOnly) {
        int nobs = model.effectiveObservationCount();
        boolean dynamic = effectiveDynamicStart >= 0;
        double[] extendedEndog = Arrays.copyOf(model.spec().endog(), originalNobs + outOfSample);
        for (int index = originalNobs; index < extendedEndog.length; index++) {
            extendedEndog[index] = Double.NaN;
        }
        if (dynamic) {
            for (int index = Math.min(effectiveDynamicStart, nobs); index < nobs; index++) {
                int originalIndex = index + observationOffset;
                if (originalIndex >= 0 && originalIndex < extendedEndog.length) {
                    extendedEndog[originalIndex] = Double.NaN;
                }
            }
        }

        double[][] extendedExog = extendExog(outOfSample, futureExog);
        SARIMAX extendedModel = model.extend(extendedEndog, extendedExog);
        FilterResult predictionFilter = extendedModel.filter(paramsInternal(), FilterOptions.defaults());
        if (model.spec().concentrateScale()) {
            extendedModel.applyConcentratedScale(predictionFilter, scale);
        }
        if (signalOnly) {
            return sliceStatePrediction(extendedModel.snapshotModel(paramsInternal()), predictionFilter, null,
                start, end, effectiveDynamicStart, kind, informationSet, true);
        }
        return slicePrediction(predictionFilter, start, end, effectiveDynamicStart, kind);
    }

    private SARIMAXPrediction warmStartedOutOfSamplePrediction(int start,
                                                               int end,
                                                               int outOfSample,
                                                               double[][] futureExog,
                                                               PredictionKind kind,
                                                               PredictionInformationSet informationSet,
                                                               boolean signalOnly) {
        double[][] extendedExog = extendExog(outOfSample, futureExog);
        if (filterResult.predictedStateLength() == 0 || filterResult.predictedStateCovLength() == 0) {
            int nobs = model.effectiveObservationCount();
            return fullRefilterPrediction(start, end, -1, outOfSample, nobs, 0, futureExog, kind,
                informationSet, signalOnly);
        }
        int nobs = model.effectiveObservationCount();
        double[] futureEndog = new double[outOfSample];
        Arrays.fill(futureEndog, Double.NaN);
        double[][] futureOnlyExog = extendedExog == null ? null : TimeSeriesArraySupport.sliceRows(extendedExog, nobs, outOfSample);
        SARIMAX futureModel = model.extend(futureEndog, futureOnlyExog, model.spec().trendOffset() + nobs);
        FilterResult futureFilter = new KalmanFiltering()
            .model(futureModel.snapshotModel(paramsInternal()))
            .initialState(finalPredictedInitialState())
            .options(FilterOptions.defaults())
            .filter();
        int localStart = start - nobs;
        int localEnd = end - nobs;
        if (signalOnly) {
            return sliceStatePrediction(futureModel.snapshotModel(paramsInternal()), futureFilter, null,
                localStart, localEnd, start, end, -1, kind, informationSet, true);
        }
        return slicePrediction(futureFilter, localStart, localEnd, start, end, -1, kind);
    }

    public SARIMAXPrediction forecast(int steps) {
        return forecast(steps, null);
    }

    public SARIMAXPrediction forecast(int steps, double[][] futureExog) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive");
        }
        int start = model.effectiveObservationCount();
        int end = start + steps - 1;
        return predict(start, end, -1, futureExog, true, PredictionInformationSet.PREDICTED, false);
    }

    public SARIMAXPrediction forecast(int steps, double[][] futureExog, boolean signalOnly) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive");
        }
        int start = model.effectiveObservationCount();
        int end = start + steps - 1;
        return predict(start, end, -1, futureExog, true, PredictionInformationSet.PREDICTED, signalOnly);
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
        return ExogenousDataSupport.extendFutureRows(model.spec().exog(), outOfSample, futureExog);
    }

    private double[][] concatExog(int newRows, double[][] exog) {
        return ExogenousDataSupport.appendRows(model.spec().exog(), newRows, exog,
            "exog row count must match new endog length");
    }

    private double[][] validateNewExog(int rows, double[][] exog) {
        return ExogenousDataSupport.validateRows(model.spec().exog(), rows, exog,
            "exog row count must match new endog length");
    }

    private SARIMAXResults resultFor(SARIMAX targetModel, FilterResult targetFilter) {
        return adoptFilterResult(targetModel,
            optimization(),
            paramsInternal(),
            unconstrainedParamsInternal(),
            targetFilter,
            MLEResults.Covariance.fromId(covType()),
            fixedParameters);
    }

    private SARIMAXFitOptions refitOptions(SARIMAXFitOptions options) {
        SARIMAXFitOptions resolved = options == null ? SARIMAXFitOptions.DEFAULT : options;
        SARIMAXFitOptions.Builder builder = resolved.toBuilder()
            .fixedParameters(resolved.fixedParameters().isEmpty() ? fixedParameters : resolved.fixedParameters());
        if (resolved.startParams() == null) {
            builder.startParams(paramsInternal());
        }
        return builder.build();
    }

    private FilterOptions resultFilterOptions(SARIMAX targetModel) {
        return targetModel.spec().concentrateScale()
            ? targetModel.realConcentratedLikelihoodOptions(FilterOptions.defaults())
            : FilterOptions.defaults();
    }

    private InitialState finalPredictedInitialState() {
        if (filterResult.predictedStateLength() == 0 || filterResult.predictedStateCovLength() == 0) {
            throw new IllegalArgumentException("extend requires retained final predicted state and covariance");
        }
        int kStates = model.stateCount();
        double[] state = new double[kStates];
        double[] covariance = new double[kStates * kStates];
        System.arraycopy(filterResult.predictedState, filterResult.predictedStateOffset(filterResult.nobs), state, 0, kStates);
        System.arraycopy(filterResult.predictedStateCov, filterResult.predictedStateCovOffset(filterResult.nobs), covariance, 0, covariance.length);
        return InitialState.known(kStates, state, covariance);
    }

    private InitialState initialPredictedState() {
        if (filterResult.predictedStateLength() == 0 || filterResult.predictedStateCovLength() == 0) {
            throw new IllegalArgumentException("simulation requires retained initial predicted state and covariance");
        }
        int kStates = model.stateCount();
        double[] state = new double[kStates];
        double[] covariance = new double[kStates * kStates];
        System.arraycopy(filterResult.predictedState, filterResult.predictedStateOffset(0), state, 0, kStates);
        System.arraycopy(filterResult.predictedStateCov, filterResult.predictedStateCovOffset(0), covariance, 0, covariance.length);
        return InitialState.known(kStates, state, covariance);
    }

    private InitialState simulationInitialState(SimulationAnchor anchor) {
        SimulationAnchor resolved = anchor == null ? SimulationAnchor.INITIAL : anchor;
        return switch (resolved) {
            case INITIAL -> initialPredictedState();
            case FINAL -> finalPredictedInitialState();
        };
    }

    private InitialState predictedInitialStateAt(int anchor) {
        if (anchor < 0 || anchor > filterResult.nobs) {
            throw new IllegalArgumentException("anchor must be in [0, nobs]");
        }
        if (filterResult.predictedStateLength() == 0 || filterResult.predictedStateCovLength() == 0) {
            throw new IllegalArgumentException("anchored simulation requires retained predicted state and covariance");
        }
        int kStates = model.stateCount();
        double[] state = new double[kStates];
        double[] covariance = new double[kStates * kStates];
        System.arraycopy(filterResult.predictedState, filterResult.predictedStateOffset(anchor), state, 0, kStates);
        System.arraycopy(filterResult.predictedStateCov, filterResult.predictedStateCovOffset(anchor), covariance, 0, covariance.length);
        return InitialState.known(kStates, state, covariance);
    }

    private SimulationSmootherResult simulate(int nsimulations,
                                             double[][] exog,
                                             InitialState initialState,
                                             Random random,
                                             SimulationSmootherOptions options) {
        SimulationSupport.requirePositiveSimulations(nsimulations);
        return SimulationSupport.simulate(simulationSnapshot(nsimulations, exog), initialState, random, options);
    }

    private KalmanSSM simulationSnapshot(int nsimulations, double[][] exog) {
        if (nsimulations == model.spec().observationCount() && exog == null) {
            return model.snapshotModel(paramsInternal());
        }
        double[] simulationEndog = new double[nsimulations];
        Arrays.fill(simulationEndog, Double.NaN);
        double[][] simulationExog = validateSimulationExog(nsimulations, exog);
        return model.extend(simulationEndog, simulationExog, model.spec().trendOffset()).snapshotModel(paramsInternal());
    }

    private double[][] validateSimulationExog(int rows, double[][] exog) {
        return ExogenousDataSupport.validateRows(model.spec().exog(), rows, exog,
            "exog row count must match simulation length");
    }

    private static SARIMAXPrediction slicePrediction(FilterResult filterResult, int start, int end, int dynamicStart) {
        return slicePrediction(filterResult, start, end, dynamicStart,
            dynamicStart >= start && dynamicStart <= end ? PredictionKind.DYNAMIC_IN_SAMPLE : PredictionKind.IN_SAMPLE);
    }

    private static SARIMAXPrediction slicePrediction(FilterResult filterResult,
                                                     int start,
                                                     int end,
                                                     int dynamicStart,
                                                     PredictionKind kind) {
        return slicePrediction(filterResult, start, end, start, end, dynamicStart, kind);
    }

    private static SARIMAXPrediction slicePrediction(FilterResult filterResult,
                                                     int localStart,
                                                     int localEnd,
                                                     int absoluteStart,
                                                     int absoluteEnd,
                                                     int dynamicStart,
                                                     PredictionKind kind) {
        if (filterResult.forecastLength() == 0) {
            throw new IllegalArgumentException("prediction is not available when forecast means were not retained");
        }
        double[] mean = new double[localEnd - localStart + 1];
        double[] variance = new double[localEnd - localStart + 1];
        boolean hasForecastCovariance = filterResult.forecastErrorCovLength() > 0;
        for (int t = localStart; t <= localEnd; t++) {
            mean[t - localStart] = filterResult.forecast(0, t);
            variance[t - localStart] = hasForecastCovariance
                ? filterResult.forecastErrorCov(0, 0, t)
                : Double.NaN;
        }
        return new SARIMAXPrediction(absoluteStart, absoluteEnd, dynamicStart, kind, mean, variance);
    }

    private SARIMAXPrediction sliceStatePrediction(KalmanSSM snapshot,
                                                   FilterResult filterResult,
                                                   SmootherResult smootherResult,
                                                   int start,
                                                   int end,
                                                   int dynamicStart,
                                                   PredictionKind kind,
                                                   PredictionInformationSet informationSet,
                                                   boolean signalOnly) {
        return sliceStatePrediction(snapshot, filterResult, smootherResult,
            start, end, start, end, dynamicStart, kind, informationSet, signalOnly);
    }

    private SARIMAXPrediction sliceStatePrediction(KalmanSSM snapshot,
                                                   FilterResult filterResult,
                                                   SmootherResult smootherResult,
                                                   int localStart,
                                                   int localEnd,
                                                   int absoluteStart,
                                                   int absoluteEnd,
                                                   int dynamicStart,
                                                   PredictionKind kind,
                                                   PredictionInformationSet informationSet,
                                                   boolean signalOnly) {
        SmootherResult resolvedSmoother = smootherResult;
        if (informationSet == PredictionInformationSet.SMOOTHED && resolvedSmoother == null) {
            resolvedSmoother = smooth(SmootherOptions.builder()
                .retainOnly(SmootherOptions.Surface.STATE, SmootherOptions.Surface.STATE_COVARIANCE)
                .build());
        }
        double[] mean = new double[localEnd - localStart + 1];
        double[] variance = new double[localEnd - localStart + 1];
        for (int t = localStart; t <= localEnd; t++) {
            StatePredictionSurface surface = StatePredictionSurfaces.resolve(filterResult, resolvedSmoother, informationSet, t);
            mean[t - localStart] = StatePredictionProjector.univariateMean(snapshot, surface, t, signalOnly);
            variance[t - localStart] = StatePredictionProjector.univariateVariance(snapshot, surface, t, signalOnly);
        }
        return new SARIMAXPrediction(absoluteStart, absoluteEnd, dynamicStart, kind, informationSet, signalOnly, mean, variance);
    }

    private static void applyLikelihoodBurn(FilterResult filterResult, int burn) {
        Arrays.fill(filterResult.logLikelihoodObs, 0, burn, 0.0);
    }

    private void requireForecastMean(String message) {
        if (filterResult.forecastLength() == 0) {
            throw new IllegalArgumentException(message);
        }
    }

    private void requireForecastError(String message) {
        if (filterResult.forecastErrorLength() == 0) {
            throw new IllegalArgumentException(message);
        }
    }

    @Override
    protected double[] computeCovParamsOim() {
        return FixedParameterResults.adjustedCovariance(model.covarianceFromObservedInformation(paramsInternal()), fixedParameters);
    }

    @Override
    protected double[] computeCovParamsApprox() {
        if (paramsInternal().length == 0) {
            return new double[0];
        }
        return FixedParameterResults.adjustedCovariance(model.covarianceFromApproximateInformation(paramsInternal()), fixedParameters);
    }

    @Override
    protected double[] computeCovParamsRobustOim() {
        double[] params = paramsInternal();
        if (params.length == 0) {
            return new double[0];
        }

        return nobsEffective <= 0
            ? nanCovariance(params.length)
            : FixedParameterResults.adjustedCovariance(model.robustCovarianceFromObservedInformation(params), fixedParameters);
    }

    @Override
    protected double[] computeCovParamsRobustApprox() {
        double[] params = paramsInternal();
        if (params.length == 0) {
            return new double[0];
        }

        return nobsEffective <= 0
            ? nanCovariance(params.length)
            : FixedParameterResults.adjustedCovariance(model.robustCovarianceFromApproximateInformation(params), fixedParameters);
    }

    @Override
    protected double[] computeCovParamsOpg() {
        return FixedParameterResults.adjustedCovariance(model.covarianceFromOpg(paramsInternal()), fixedParameters);
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