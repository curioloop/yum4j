package com.curioloop.yum4j.tsa.varmax;

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
import com.curioloop.yum4j.stats.test.GrangerCausality;
import com.curioloop.yum4j.stats.test.JarqueBera;
import com.curioloop.yum4j.tsa.diagnostics.DiagnosticSupport;
import com.curioloop.yum4j.tsa.diagnostics.InstantaneousCausality;
import com.curioloop.yum4j.tsa.lifecycle.ExogenousDataSupport;
import com.curioloop.yum4j.tsa.lifecycle.TimeSeriesArraySupport;
import com.curioloop.yum4j.tsa.prediction.PredictionInformationSet;
import com.curioloop.yum4j.tsa.prediction.PredictionKind;
import com.curioloop.yum4j.tsa.prediction.PredictionToolkit;
import com.curioloop.yum4j.tsa.prediction.StatePredictionProjector;
import com.curioloop.yum4j.tsa.prediction.StatePredictionSurface;
import com.curioloop.yum4j.tsa.prediction.StatePredictionSurfaces;
import com.curioloop.yum4j.tsa.statespace.ForecastErrorVarianceDecompositions;
import com.curioloop.yum4j.tsa.statespace.ForecastErrorVarianceDecomposition;
import com.curioloop.yum4j.tsa.statespace.ImpulseResponse;
import com.curioloop.yum4j.tsa.statespace.ImpulseResponseRepetitions;
import com.curioloop.yum4j.tsa.statespace.SimulationAnchor;
import com.curioloop.yum4j.tsa.statespace.SimulationSupport;
import com.curioloop.yum4j.tsa.statespace.SimulationSmootherRepetitions;
import com.curioloop.yum4j.tsa.statespace.StateSpaceImpulseResponses;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

public final class VARMAXResults extends MLEResults {

    private enum FilterResultOwnership {
        COPY,
        ADOPT
    }

    private final VARMAX model;
    private final FilterResult filterResult;
    private final int logLikelihoodBurn;
    private final int nobsEffective;
    private final int dfModel;
    private final FixedParameters fixedParameters;

    VARMAXResults(VARMAX model,
                  Optimization optimization,
                  double[] params,
                  double[] unconstrainedParams,
                  FilterResult filterResult,
                  MLEResults.Covariance covType) {
        this(model, optimization, params, unconstrainedParams, filterResult, covType, FixedParameters.none());
    }

    VARMAXResults(VARMAX model,
                  Optimization optimization,
                  double[] params,
                  double[] unconstrainedParams,
                  FilterResult filterResult,
                  MLEResults.Covariance covType,
                  FixedParameters fixedParameters) {
            this(model, optimization, params, unconstrainedParams, filterResult, covType, fixedParameters,
                FilterResultOwnership.COPY);
            }

            static VARMAXResults adoptFilterResult(VARMAX model,
                               Optimization optimization,
                               double[] params,
                               double[] unconstrainedParams,
                               FilterResult filterResult,
                               MLEResults.Covariance covType,
                               FixedParameters fixedParameters) {
            return new VARMAXResults(model, optimization, params, unconstrainedParams, filterResult, covType,
                fixedParameters, FilterResultOwnership.ADOPT);
            }

            private VARMAXResults(VARMAX model,
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
        applyLikelihoodBurn(this.filterResult, logLikelihoodBurn);
        this.nobsEffective = Math.max(0, model.effectiveObservationCount() - logLikelihoodBurn);
        this.dfModel = FixedParameterResults.dfModel(params.length, this.fixedParameters, 0);
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

    public SmootherResult smooth() {
        return model.smooth(paramsInternal(), SmootherOptions.defaults());
    }

    public SmootherResult smooth(SmootherOptions smootherOptions) {
        return model.smooth(paramsInternal(), smootherOptions == null ? SmootherOptions.defaults() : smootherOptions);
    }

    public VARMAXResults append(double[][] endog) {
        return append(endog, null);
    }

    public VARMAXResults append(double[][] endog, double[][] exog) {
        double[][] newEndog = TimeSeriesArraySupport.copyRows(endog, "endog");
        double[][] combinedEndog = TimeSeriesArraySupport.concatRows(model.spec().endog(), newEndog);
        double[][] combinedExog = model.extendExog(newEndog.length, exog);
        VARMAX appendedModel = model.extend(combinedEndog, combinedExog, model.spec().trendOffset());
        return resultFor(appendedModel, appendedModel.filter(paramsInternal(), FilterOptions.defaults()));
    }

    public VARMAXResults appendRefit(double[][] endog) {
        return appendRefit(endog, null, null);
    }

    public VARMAXResults appendRefit(double[][] endog, double[][] exog) {
        return appendRefit(endog, exog, null);
    }

    public VARMAXResults appendRefit(double[][] endog, double[][] exog, VARMAXFitOptions fitOptions) {
        double[][] newEndog = TimeSeriesArraySupport.copyRows(endog, "endog");
        double[][] combinedEndog = TimeSeriesArraySupport.concatRows(model.spec().endog(), newEndog);
        double[][] combinedExog = model.extendExog(newEndog.length, exog);
        return model.extend(combinedEndog, combinedExog, model.spec().trendOffset())
            .fit(refitOptions(fitOptions));
    }

    public VARMAXResults extend(double[][] endog) {
        return extend(endog, null);
    }

    public VARMAXResults extend(double[][] endog, double[][] exog) {
        double[][] newEndog = TimeSeriesArraySupport.copyRows(endog, "endog");
        double[][] extendedExog = model.extendExog(newEndog.length, exog);
        double[][] newExog = extendedExog == null ? null : TimeSeriesArraySupport.sliceRows(extendedExog, model.spec().observationCount(), newEndog.length);
        if (filterResult.filteredStateLength() == 0 || filterResult.filteredStateCovLength() == 0) {
            throw new IllegalArgumentException("extend requires retained final filtered state and covariance");
        }
        InitialState initialState = forecastInitialState(extendedExog, model.spec().observationCount());
        VARMAX extendedModel = model.extend(newEndog, newExog, model.spec().trendOffset() + model.spec().observationCount());
        FilterResult extendedFilter = new KalmanFiltering()
            .model(extendedModel.snapshotModel(paramsInternal()))
            .initialState(initialState)
            .options(FilterOptions.defaults())
            .filter();
        return resultFor(extendedModel, extendedFilter);
    }

    public VARMAXResults extendRefit(double[][] endog) {
        return extendRefit(endog, null, null);
    }

    public VARMAXResults extendRefit(double[][] endog, double[][] exog) {
        return extendRefit(endog, exog, null);
    }

    public VARMAXResults extendRefit(double[][] endog, double[][] exog, VARMAXFitOptions fitOptions) {
        double[][] newEndog = TimeSeriesArraySupport.copyRows(endog, "endog");
        double[][] extendedExog = model.extendExog(newEndog.length, exog);
        double[][] newExog = extendedExog == null ? null : TimeSeriesArraySupport.sliceRows(extendedExog, model.spec().observationCount(), newEndog.length);
        return model.extend(newEndog, newExog, model.spec().trendOffset() + model.spec().observationCount())
            .fit(refitOptions(fitOptions));
    }

    public VARMAXResults apply(double[][] endog) {
        return apply(endog, null);
    }

    public VARMAXResults apply(double[][] endog, double[][] exog) {
        double[][] newEndog = TimeSeriesArraySupport.copyRows(endog, "endog");
        double[][] extendedExog = model.extendExog(newEndog.length, exog);
        double[][] newExog = extendedExog == null ? null : TimeSeriesArraySupport.sliceRows(extendedExog, model.spec().observationCount(), newEndog.length);
        VARMAX appliedModel = model.extend(newEndog, newExog, model.spec().trendOffset());
        return resultFor(appliedModel, appliedModel.filter(paramsInternal(), FilterOptions.defaults()));
    }

    public VARMAXResults applyRefit(double[][] endog) {
        return applyRefit(endog, null, null);
    }

    public VARMAXResults applyRefit(double[][] endog, double[][] exog) {
        return applyRefit(endog, exog, null);
    }

    public VARMAXResults applyRefit(double[][] endog, double[][] exog, VARMAXFitOptions fitOptions) {
        double[][] newEndog = TimeSeriesArraySupport.copyRows(endog, "endog");
        double[][] extendedExog = model.extendExog(newEndog.length, exog);
        double[][] newExog = extendedExog == null ? null : TimeSeriesArraySupport.sliceRows(extendedExog, model.spec().observationCount(), newEndog.length);
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
        int anchorIndex = simulationAnchorIndex(anchor);
        return simulate(nsimulations, null, model.spec().trendOffset() + anchorIndex,
            simulationInitialStateAt(anchorIndex, null), random, options);
    }

    public SimulationSmootherResult simulate(int nsimulations,
                                             SimulationAnchor anchor,
                                             double[][] exog,
                                             Random random,
                                             SimulationSmootherOptions options) {
        int anchorIndex = simulationAnchorIndex(anchor);
        return simulate(nsimulations, exog, model.spec().trendOffset() + anchorIndex,
            simulationInitialStateAt(anchorIndex, exog), random, options);
    }

    public SimulationSmootherResult simulate(int nsimulations,
                                             int anchor,
                                             double[][] exog,
                                             Random random,
                                             SimulationSmootherOptions options) {
        return simulate(nsimulations, exog, model.spec().trendOffset() + anchor,
            simulationInitialStateAt(anchor, exog), random, options);
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
        return simulate(nsimulations,
            exog,
            model.spec().trendOffset(),
            initialPredictedState(),
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            options);
    }

    public SimulationSmootherResult simulate(int nsimulations,
                                             int anchor,
                                             double[][] exog,
                                             double[] measurementDisturbanceVariates,
                                             double[] stateDisturbanceVariates,
                                             double[] initialStateVariates,
                                             SimulationSmootherOptions options) {
        return simulate(nsimulations,
            exog,
            model.spec().trendOffset() + anchor,
            simulationInitialStateAt(anchor, exog),
            measurementDisturbanceVariates,
            stateDisturbanceVariates,
            initialStateVariates,
            options);
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
        if (repetitions < 1) {
            throw new IllegalArgumentException("repetitions must be positive");
        }
        ImpulseResponse[] responses = new ImpulseResponse[repetitions];
        for (int repetition = 0; repetition < repetitions; repetition++) {
            responses[repetition] = impulseResponses(steps, impulse, orthogonalized, cumulative);
        }
        return new ImpulseResponseRepetitions(responses);
    }

    public ForecastErrorVarianceDecomposition fevd(int steps) {
        return ForecastErrorVarianceDecompositions.compute(model.snapshotModel(paramsInternal()), steps);
    }

    public GrangerCausality testCausality(int caused, int causing, int maxLag) {
        double[][] residuals = residuals();
        int dimension = model.observationDimension();
        return GrangerCausality.test(DiagnosticSupport.finitePairMatrix(residuals, caused, causing, dimension),
            DiagnosticSupport.usablePairCount(residuals, caused, causing, dimension), maxLag);
    }

    public InstantaneousCausality testInstantaneousCausality(int caused, int causing) {
        return DiagnosticSupport.instantaneousCausality(residuals(), caused, causing, model.observationDimension());
    }

    public JarqueBera[] testNormality() {
        double[][] values = DiagnosticSupport.finiteColumns(residuals(), model.observationDimension());
        JarqueBera[] out = new JarqueBera[values.length];
        for (int col = 0; col < values.length; col++) {
            out[col] = JarqueBera.test(values[col]);
        }
        return out;
    }

    public BreakVar[] testHeteroskedasticity() {
        return testHeteroskedasticity(1.0 / 3.0, BreakVar.Alternative.TWO_SIDED, true);
    }

    public BreakVar[] testHeteroskedasticity(double subsetLen,
                                             BreakVar.Alternative alternative,
                                             boolean useF) {
        double[][] values = DiagnosticSupport.finiteColumns(residuals(), model.observationDimension());
        BreakVar[] out = new BreakVar[values.length];
        for (int col = 0; col < values.length; col++) {
            out[col] = BreakVar.test(values[col], subsetLen, alternative, useF);
        }
        return out;
    }

    public Box[] testSerialCorrelation(int... lags) {
        double[][] values = DiagnosticSupport.finiteColumns(residuals(), model.observationDimension());
        Box[] out = new Box[values.length];
        int adjustDF = model.spec().order().autoregressive() + model.spec().order().movingAverage();
        for (int col = 0; col < values.length; col++) {
            out[col] = Box.test(values[col], Box.Statistic.LJUNG_BOX, adjustDF, false, 0,
                lags == null ? new int[0] : lags);
        }
        return out;
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

    public SmootherDiagnostics.NewsImpact news(VARMAXResults updated) {
        return news(updated, SmootherOptions.defaults());
    }

    public SmootherDiagnostics.NewsImpact news(VARMAXResults updated, SmootherOptions options) {
        if (updated == null) {
            throw new IllegalArgumentException("updated results must not be null");
        }
        return SmootherDiagnostics.news(model.snapshotModel(paramsInternal()),
            initialPredictedState(),
            updated.model.snapshotModel(updated.paramsInternal()),
            updated.initialPredictedState(),
            options);
    }

    public double[][] coefficientMatricesVAR() {
        return coefficientMatrices(model.spec().order().autoregressive(), model.observationDimension(), varBlockOffset());
    }

    public double[][] coefficientMatricesVMA() {
        return coefficientMatrices(model.spec().order().movingAverage(), model.observationDimension(), vmaBlockOffset());
    }

    public double[][] fittedValues() {
        requireForecastMean("fitted values are not available when forecast means were not retained");
        int nobs = filterResult.nobs;
        int kEndog = model.observationDimension();
        double[][] fitted = new double[nobs][kEndog];
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                fitted[t][i] = filterResult.forecast(i, t);
            }
        }
        return fitted;
    }

    public double[][] residuals() {
        requireForecastError("residuals are not available when forecast errors were not retained");
        int nobs = filterResult.nobs;
        int kEndog = model.observationDimension();
        double[][] residuals = new double[nobs][kEndog];
        double[][] endog = model.effectiveEndog();
        for (int t = 0; t < nobs; t++) {
            for (int i = 0; i < kEndog; i++) {
                residuals[t][i] = Double.isNaN(endog[t][i]) ? Double.NaN : filterResult.forecastError(i, t);
            }
        }
        return residuals;
    }

    public double[][] standardizedForecastError() {
        double[] flat = filterResult.standardizedForecastError();
        int kEndog = model.observationDimension();
        double[][] values = new double[flat.length / kEndog][kEndog];
        for (int t = 0; t < values.length; t++) {
            System.arraycopy(flat, t * kEndog, values[t], 0, kEndog);
        }
        return values;
    }

    public VARMAXPrediction predict() {
        return predict(0, model.spec().observationCount() - 1, -1, null);
    }

    public VARMAXPrediction predict(int start, int end) {
        return predict(start, end, -1, null);
    }

    public VARMAXPrediction predict(int start,
                                    int end,
                                    PredictionInformationSet informationSet,
                                    boolean signalOnly) {
        return predict(start, end, -1, null, false, informationSet, signalOnly);
    }

    public VARMAXPrediction predict(int start, int end, int dynamicStart, double[][] futureExog) {
        return predict(start, end, dynamicStart, futureExog, false, PredictionInformationSet.PREDICTED, false);
    }

    public VARMAXPrediction predict(int start,
                                    int end,
                                    int dynamicStart,
                                    double[][] futureExog,
                                    PredictionInformationSet informationSet,
                                    boolean signalOnly) {
        return predict(start, end, dynamicStart, futureExog, false, informationSet, signalOnly);
    }

    private VARMAXPrediction predict(int start,
                                     int end,
                                     int dynamicStart,
                                     double[][] futureExog,
                                     boolean forecastOnly,
                                     PredictionInformationSet informationSet,
                                     boolean signalOnly) {
        PredictionInformationSet resolvedInformationSet = PredictionToolkit.resolveInformationSet(informationSet);
        int nobs = model.spec().observationCount();
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
        if (!dynamic && start >= nobs) {
            return warmStartedOutOfSamplePrediction(start, end, outOfSample, futureExog, kind, resolvedInformationSet, signalOnly);
        }
        return fullRefilterPrediction(start, end, effectiveDynamicStart, requiredLength, outOfSample, futureExog, nobs, kind,
            resolvedInformationSet, signalOnly);
    }

    private VARMAXPrediction fullRefilterPrediction(int start,
                                                    int end,
                                                    int effectiveDynamicStart,
                                                    int requiredLength,
                                                    int outOfSample,
                                                    double[][] futureExog,
                                                    int nobs,
                                                    PredictionKind kind,
                                                    PredictionInformationSet informationSet,
                                                    boolean signalOnly) {
        double[][] extendedEndog = VARMAXSupport.appendMissingRows(model.spec().endog(), Math.max(requiredLength, nobs));
        if (effectiveDynamicStart >= 0) {
            for (int index = Math.min(effectiveDynamicStart, nobs); index < Math.min(nobs, extendedEndog.length); index++) {
                Arrays.fill(extendedEndog[index], Double.NaN);
            }
        }
        double[][] extendedExog = model.extendExog(outOfSample, futureExog);
        VARMAX extendedModel = model.extend(extendedEndog, extendedExog, model.spec().trendOffset());
        FilterResult predictionFilter = extendedModel.filter(paramsInternal(), FilterOptions.defaults());
        if (signalOnly) {
            return sliceStatePrediction(extendedModel.snapshotModel(paramsInternal()), predictionFilter, null,
                start, end, effectiveDynamicStart, kind, informationSet, true);
        }
        return slicePrediction(predictionFilter, start, end, effectiveDynamicStart, kind);
    }

    private VARMAXPrediction warmStartedOutOfSamplePrediction(int start,
                                                              int end,
                                                              int outOfSample,
                                                              double[][] futureExog,
                                                              PredictionKind kind,
                                                              PredictionInformationSet informationSet,
                                                              boolean signalOnly) {
        double[][] extendedExog = model.extendExog(outOfSample, futureExog);
        if (filterResult.filteredStateLength() == 0 || filterResult.filteredStateCovLength() == 0) {
            return fullRefilterPrediction(start, end, -1, end + 1, outOfSample, futureExog, model.spec().observationCount(), kind,
                informationSet, signalOnly);
        }
        int nobs = model.spec().observationCount();
        InitialState initialState = forecastInitialState(extendedExog, nobs);
        double[][] futureEndog = TimeSeriesArraySupport.missingRows(outOfSample, model.observationDimension());
        double[][] futureOnlyExog = TimeSeriesArraySupport.sliceRows(extendedExog, nobs, outOfSample);
        VARMAX futureModel = model.extend(futureEndog, futureOnlyExog, model.spec().trendOffset() + nobs);
        FilterResult futureFilter = new KalmanFiltering()
            .model(futureModel.snapshotModel(paramsInternal()))
            .initialState(initialState)
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

    private InitialState forecastInitialState(double[][] extendedExog, int nobs) {
        double[][] transitionEndog = VARMAXSupport.appendMissingRows(model.spec().endog(), nobs + 1);
        double[][] transitionExog = extendedExog == null ? null : TimeSeriesArraySupport.sliceRows(extendedExog, 0, nobs + 1);
        VARMAX transitionModel = model.extend(transitionEndog, transitionExog, model.spec().trendOffset());
        KalmanSSM snapshot = transitionModel.snapshotModel(paramsInternal());
        int last = nobs - 1;
        double[] state = nextPredictedState(snapshot, last);
        double[] covariance = nextPredictedStateCov(snapshot, last);
        return InitialState.known(model.stateCount(), state, covariance);
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

    private int simulationAnchorIndex(SimulationAnchor anchor) {
        SimulationAnchor resolved = anchor == null ? SimulationAnchor.INITIAL : anchor;
        return switch (resolved) {
            case INITIAL -> 0;
            case FINAL -> filterResult.nobs;
        };
    }

    private InitialState simulationInitialStateAt(int anchor, double[][] exog) {
        if (anchor == filterResult.nobs && model.spec().exog() != null && exog != null) {
            return forecastInitialState(model.extendExog(exog.length, exog), filterResult.nobs);
        }
        return predictedInitialStateAt(anchor);
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
        return simulate(nsimulations, exog, model.spec().trendOffset(), initialState, random, options);
    }

    private SimulationSmootherResult simulate(int nsimulations,
                                             double[][] exog,
                                             int trendOffset,
                                             InitialState initialState,
                                             Random random,
                                             SimulationSmootherOptions options) {
        SimulationSupport.requirePositiveSimulations(nsimulations);
        return SimulationSupport.simulate(simulationSnapshot(nsimulations, exog, trendOffset), initialState, random, options);
    }

    private SimulationSmootherResult simulate(int nsimulations,
                                             double[][] exog,
                                             int trendOffset,
                                             InitialState initialState,
                                             double[] measurementDisturbanceVariates,
                                             double[] stateDisturbanceVariates,
                                             double[] initialStateVariates,
                                             SimulationSmootherOptions options) {
        SimulationSupport.requirePositiveSimulations(nsimulations);
        return SimulationSupport.simulate(simulationSnapshot(nsimulations, exog, trendOffset), initialState,
            measurementDisturbanceVariates, stateDisturbanceVariates, initialStateVariates, options);
    }

    private KalmanSSM simulationSnapshot(int nsimulations, double[][] exog) {
        return simulationSnapshot(nsimulations, exog, model.spec().trendOffset());
    }

    private KalmanSSM simulationSnapshot(int nsimulations, double[][] exog, int trendOffset) {
        if (trendOffset == model.spec().trendOffset() && nsimulations == model.spec().observationCount() && exog == null) {
            return model.snapshotModel(paramsInternal());
        }
        double[][] simulationEndog = TimeSeriesArraySupport.missingRows(nsimulations, model.observationDimension());
        double[][] simulationExog = validateSimulationExog(nsimulations, exog);
        return model.extend(simulationEndog, simulationExog, trendOffset).snapshotModel(paramsInternal());
    }

    private double[][] validateSimulationExog(int rows, double[][] exog) {
        return ExogenousDataSupport.validateRows(model.spec().exog(), rows, exog,
            "exog row count must match simulation length");
    }

    private double[] nextPredictedState(KalmanSSM snapshot, int transitionTime) {
        int kStates = model.stateCount();
        double[] state = new double[kStates];
        int filteredOffset = filterResult.filteredStateOffset(transitionTime);
        int transitionOffset = snapshot.transitionOffset(transitionTime);
        int interceptOffset = snapshot.stateInterceptOffset(transitionTime);
        for (int row = 0; row < kStates; row++) {
            double value = snapshot.stateInterceptData()[interceptOffset + row];
            for (int col = 0; col < kStates; col++) {
                value += snapshot.transitionData()[transitionOffset + row * kStates + col]
                    * filterResult.filteredState[filteredOffset + col];
            }
            state[row] = value;
        }
        return state;
    }

    private double[] nextPredictedStateCov(KalmanSSM snapshot, int transitionTime) {
        int kStates = model.stateCount();
        int kPosdef = snapshot.stateDisturbanceCount();
        double[] covariance = new double[kStates * kStates];
        double[] tmp = new double[kStates * kStates];
        int filteredCovOffset = filterResult.filteredStateCovOffset(transitionTime);
        int transitionOffset = snapshot.transitionOffset(transitionTime);
        for (int row = 0; row < kStates; row++) {
            for (int col = 0; col < kStates; col++) {
                double value = 0.0;
                for (int inner = 0; inner < kStates; inner++) {
                    value += snapshot.transitionData()[transitionOffset + row * kStates + inner]
                        * filterResult.filteredStateCov[filteredCovOffset + inner * kStates + col];
                }
                tmp[row * kStates + col] = value;
            }
        }
        for (int row = 0; row < kStates; row++) {
            for (int col = 0; col < kStates; col++) {
                double value = 0.0;
                for (int inner = 0; inner < kStates; inner++) {
                    value += tmp[row * kStates + inner]
                        * snapshot.transitionData()[transitionOffset + col * kStates + inner];
                }
                covariance[row * kStates + col] = value;
            }
        }
        int selectionOffset = snapshot.selectionOffset(transitionTime);
        int stateCovOffset = snapshot.stateCovarianceOffset(transitionTime);
        for (int row = 0; row < kStates; row++) {
            for (int col = 0; col < kStates; col++) {
                double noise = 0.0;
                for (int left = 0; left < kPosdef; left++) {
                    double rowSelection = snapshot.selectionData()[selectionOffset + row * kPosdef + left];
                    for (int right = 0; right < kPosdef; right++) {
                        noise += rowSelection
                            * snapshot.stateCovarianceData()[stateCovOffset + left * kPosdef + right]
                            * snapshot.selectionData()[selectionOffset + col * kPosdef + right];
                    }
                }
                covariance[row * kStates + col] += noise;
            }
        }
        return covariance;
    }

    public VARMAXPrediction forecast(int steps) {
        return forecast(steps, null);
    }

    public VARMAXPrediction forecast(int steps, double[][] futureExog) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive");
        }
        int start = model.spec().observationCount();
        int end = start + steps - 1;
        return predict(start, end, -1, futureExog, true, PredictionInformationSet.PREDICTED, false);
    }

    public VARMAXPrediction forecast(int steps, double[][] futureExog, boolean signalOnly) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive");
        }
        int start = model.spec().observationCount();
        int end = start + steps - 1;
        return predict(start, end, -1, futureExog, true, PredictionInformationSet.PREDICTED, signalOnly);
    }


    private int varBlockOffset() {
        return model.spec().trendPowers().length * model.observationDimension();
    }

    private int vmaBlockOffset() {
        int k = model.observationDimension();
        return varBlockOffset() + k * k * model.spec().order().autoregressive();
    }

    private VARMAXResults resultFor(VARMAX targetModel, FilterResult targetFilter) {
        return adoptFilterResult(targetModel,
            optimization(),
            paramsInternal(),
            unconstrainedParamsInternal(),
            targetFilter,
            MLEResults.Covariance.fromId(covType()),
            fixedParameters);
    }

    private VARMAXFitOptions refitOptions(VARMAXFitOptions options) {
        VARMAXFitOptions resolved = options == null ? VARMAXFitOptions.DEFAULT : options;
        VARMAXFitOptions.Builder builder = resolved.toBuilder()
            .fixedParameters(resolved.fixedParameters().isEmpty() ? fixedParameters : resolved.fixedParameters());
        if (resolved.startParams() == null) {
            builder.startParams(paramsInternal());
        }
        return builder.build();
    }

    private double[][] coefficientMatrices(int order, int kEndog, int offset) {
        if (order == 0) {
            return new double[0][0];
        }
        double[] params = paramsInternal();
        double[][] matrices = new double[order][kEndog * kEndog];
        for (int lag = 0; lag < order; lag++) {
            for (int equation = 0; equation < kEndog; equation++) {
                for (int source = 0; source < kEndog; source++) {
                    matrices[lag][equation * kEndog + source] =
                        params[offset + equation * order * kEndog + lag * kEndog + source];
                }
            }
        }
        return matrices;
    }

    private static VARMAXPrediction slicePrediction(FilterResult filterResult, int start, int end, int dynamicStart) {
        return slicePrediction(filterResult, start, end, start, end, dynamicStart,
            dynamicStart >= start && dynamicStart <= end ? PredictionKind.DYNAMIC_IN_SAMPLE : PredictionKind.IN_SAMPLE);
    }

    private static VARMAXPrediction slicePrediction(FilterResult filterResult,
                                                   int start,
                                                   int end,
                                                   int dynamicStart,
                                                   PredictionKind kind) {
        return slicePrediction(filterResult, start, end, start, end, dynamicStart, kind);
    }

    private static VARMAXPrediction slicePrediction(FilterResult filterResult,
                                                   int localStart,
                                                   int localEnd,
                                                   int absoluteStart,
                                                   int absoluteEnd,
                                                   int dynamicStart,
                                                   PredictionKind kind) {
        if (filterResult.forecastLength() == 0) {
            throw new IllegalArgumentException("prediction is not available when forecast means were not retained");
        }
        int kEndog = filterResult.kEndog;
        int rows = localEnd - localStart + 1;
        double[][] mean = new double[rows][kEndog];
        double[][][] variance = new double[rows][kEndog][kEndog];
        boolean hasForecastCovariance = filterResult.forecastErrorCovLength() > 0;
        for (int t = localStart; t <= localEnd; t++) {
            for (int i = 0; i < kEndog; i++) {
                mean[t - localStart][i] = filterResult.forecast(i, t);
                for (int j = 0; j < kEndog; j++) {
                    variance[t - localStart][i][j] = hasForecastCovariance
                        ? filterResult.forecastErrorCov(i, j, t)
                        : Double.NaN;
                }
            }
        }
        return new VARMAXPrediction(absoluteStart, absoluteEnd, dynamicStart, kind, mean, variance);
    }

    private VARMAXPrediction sliceStatePrediction(KalmanSSM snapshot,
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

    private VARMAXPrediction sliceStatePrediction(KalmanSSM snapshot,
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
        int kEndog = snapshot.observationDimension();
        int rows = localEnd - localStart + 1;
        double[][] mean = new double[rows][kEndog];
        double[][][] variance = new double[rows][kEndog][kEndog];
        for (int t = localStart; t <= localEnd; t++) {
            StatePredictionSurface surface = StatePredictionSurfaces.resolve(filterResult, resolvedSmoother, informationSet, t);
            int row = t - localStart;
            StatePredictionProjector.multivariateMean(snapshot, surface, t, signalOnly, mean[row]);
            StatePredictionProjector.multivariateVariance(snapshot, surface, t, signalOnly, variance[row]);
        }
        return new VARMAXPrediction(absoluteStart, absoluteEnd, dynamicStart, kind,
            informationSet, signalOnly, mean, variance);
    }

    private static void applyLikelihoodBurn(FilterResult filterResult, int burn) {
        Arrays.fill(filterResult.logLikelihoodObs, 0, Math.min(burn, filterResult.logLikelihoodObs.length), 0.0);
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
        return paramsInternal().length == 0 ? new double[0]
            : FixedParameterResults.adjustedCovariance(model.covarianceFromApproximateInformation(paramsInternal()), fixedParameters);
    }

    @Override
    protected double[] computeCovParamsOpg() {
        return FixedParameterResults.adjustedCovariance(model.covarianceFromOpg(paramsInternal()), fixedParameters);
    }

    @Override
    protected double[] computeCovParamsRobustOim() {
        return nobsEffective <= 0 ? nanCovariance(paramsInternal().length)
            : FixedParameterResults.adjustedCovariance(model.robustCovarianceFromObservedInformation(paramsInternal()), fixedParameters);
    }

    @Override
    protected double[] computeCovParamsRobustApprox() {
        return nobsEffective <= 0 ? nanCovariance(paramsInternal().length)
            : FixedParameterResults.adjustedCovariance(model.robustCovarianceFromApproximateInformation(paramsInternal()), fixedParameters);
    }

    @Override
    protected double logLikelihoodInternal() {
        return filterResult.logLikelihood();
    }

    @Override
    protected double scaleInternal() {
        return 1.0;
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
        return "VARMAX";
    }

    @Override
    protected String summaryDiagnosticsTextInternal() {
        return "Diagnostics: unavailable in current VARMAXResults summary surface";
    }

    private static double[] nanCovariance(int dimension) {
        double[] nan = new double[dimension * dimension];
        Arrays.fill(nan, Double.NaN);
        return nan;
    }

}
