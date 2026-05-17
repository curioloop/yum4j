package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.ssm.kalman.arena.DoubleArena;
import com.curioloop.yum4j.ssm.kalman.arena.IntArena;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.mle.ComplexMLE;
import com.curioloop.yum4j.ssm.kalman.mle.FixedParameterMLE;
import com.curioloop.yum4j.ssm.kalman.mle.RealMLE;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;
import com.curioloop.yum4j.linalg.mat.LU;
import com.curioloop.yum4j.optim.NumericalGradient;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;

import java.util.Arrays;

public final class SARIMAX {

    private static final FilterOptions CONCENTRATED_LIKELIHOOD_OPTIONS = FilterOptions.builder()
        .drop(FilterOptions.Surface.PREDICTED_STATE,
            FilterOptions.Surface.PREDICTED_STATE_COVARIANCE,
            FilterOptions.Surface.PREDICTED_DIFFUSE_STATE_COVARIANCE,
            FilterOptions.Surface.FILTERED_STATE,
            FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
            FilterOptions.Surface.KALMAN_GAIN)
        .build();
    private static final double[] UNIT_POLYNOMIAL = {1.0};
    private static final double[] UNIT_COMPLEX_POLYNOMIAL = {1.0, 0.0};
    private static final double[] EMPTY_COMPLEX_ARRAY = new double[0];
    private static final double DEFAULT_STATE_REGRESSION_APPROXIMATE_DIFFUSE_VARIANCE = 1e10;

    private final SARIMAXSpec spec;
    private final SARIMAXSupport.Meta meta;
    private final ParameterLayout parameterLayout;
    private final InitialState initialization;
    private final SARIMAXSupport.StationaryTransformWorkspace transformWorkspace = new SARIMAXSupport.StationaryTransformWorkspace();
    private final PhaseWorkspace phaseWorkspace = new PhaseWorkspace();
    private final AnalyticComplexWorkspace analyticComplexWorkspace = new AnalyticComplexWorkspace();
    private final RealDelegate realDelegate;
    private final ComplexDelegate complexDelegate;

    public SARIMAX(SARIMAXSpec spec) {
        this(spec, SARIMAXSupport.analyze(spec));
    }

    private SARIMAX(SARIMAXSpec spec, SARIMAXSupport.Meta meta) {
        this.spec = spec;
        this.meta = meta;
        this.parameterLayout = ParameterLayout.of(meta);
        this.initialization = buildInitialization();
        this.realDelegate = new RealDelegate();
        this.complexDelegate = new ComplexDelegate();
    }

    public SARIMAXSpec spec() {
        return spec;
    }

    public int paramCount() {
        return realDelegate.paramCount();
    }

    public String[] parameterNames() {
        return realDelegate.parameterNames();
    }

    public double[] startParams() {
        return realDelegate.startParams();
    }

    public void trimWorkspaces() {
        realDelegate.trimDelegateWorkspace();
        complexDelegate.trimDelegateWorkspace();
        transformWorkspace.release();
        phaseWorkspace.release();
        analyticComplexWorkspace.release();
    }

    public boolean complex() {
        return realDelegate.complex();
    }

    public int observationCount() {
        return realDelegate.observationCount();
    }

    public int observationDimension() {
        return realDelegate.observationDimension();
    }

    public int stateCount() {
        return realDelegate.stateCount();
    }

    public double[] unconstrainedStartParams() {
        return untransformParams(startParams());
    }

    public double[] transformParams(double[] unconstrainedParams) {
        return realDelegate.transformParams(unconstrainedParams);
    }

    public double[] untransformParams(double[] params) {
        return realDelegate.untransformParams(params);
    }

    public double logLikelihood(double[] params) {
        return realDelegate.logLikelihood(params);
    }

    public double logLikelihood(double[] params, FilterOptions filterOptions) {
        return realDelegate.logLikelihood(params, filterOptions);
    }

    public double[] logLikelihoodObs(double[] params) {
        return realDelegate.logLikelihoodObs(params);
    }

    public double[] logLikelihoodObs(double[] params, FilterOptions filterOptions) {
        return realDelegate.logLikelihoodObs(params, filterOptions);
    }

    public FilterResult filter(double[] params) {
        return realDelegate.filter(params);
    }

    public FilterResult filter(double[] params, FilterOptions filterOptions) {
        return realDelegate.filter(params, filterOptions);
    }

    public SmootherResult smooth(double[] params) {
        return realDelegate.smooth(params);
    }

    public SmootherResult smooth(double[] params, SmootherOptions smootherOptions) {
        return realDelegate.smooth(params, smootherOptions);
    }

    public FilterResult predict(double[] params) {
        return realDelegate.predict(params);
    }

    public FilterResult predict(double[] params, FilterOptions filterOptions) {
        return realDelegate.predict(params, filterOptions);
    }

    public double[] score(double[] params) {
        return realDelegate.score(params);
    }

    public double[] score(double[] params, NumericalGradient gradientScheme) {
        return realDelegate.score(params, gradientScheme);
    }

    public double[] scoreObs(double[] params) {
        return realDelegate.scoreObs(params);
    }

    public double[] opgInformationMatrix(double[] params) {
        return realDelegate.opgInformationMatrix(params);
    }

    public double[] observedInformationMatrix(double[] params) {
        return realDelegate.observedInformationMatrix(params);
    }

    public double[] covarianceFromObservedInformation(double[] params) {
        return realDelegate.covarianceFromObservedInformation(params);
    }

    public double[] covarianceFromOpg(double[] params) {
        return realDelegate.covarianceFromOpg(params);
    }

    public double[] robustCovarianceFromObservedInformation(double[] params) {
        return realDelegate.robustCovarianceFromObservedInformation(params);
    }

    public double[] covarianceFromApproximateInformation(double[] params) {
        return realDelegate.covarianceFromApproximateInformation(params);
    }

    public double[] robustCovarianceFromApproximateInformation(double[] params) {
        return realDelegate.robustCovarianceFromApproximateInformation(params);
    }

    public double[] standardErrors(double[] covariance) {
        return realDelegate.standardErrors(covariance);
    }

    public Univariate.Objective objectiveFunction() {
        return realDelegate.objectiveFunction();
    }

    public Univariate.Objective transformedObjectiveFunction() {
        return realDelegate.transformedObjectiveFunction();
    }

    public Univariate objective() {
        return realDelegate.objective();
    }

    public Univariate transformedObjective() {
        return realDelegate.transformedObjective();
    }

    public KalmanSSM snapshotModel(double[] params) {
        return realDelegate.snapshotModel(params);
    }

    private KalmanSSM snapshotComplexModel(double[] params) {
        return complexDelegate.snapshotComplexModel(params);
    }

    private KalmanSSM snapshotComplexModel(double[] params, int perturbIndex, double perturbStep) {
        return complexDelegate.snapshotComplexModel(params, perturbIndex, perturbStep);
    }

    private KalmanSSM borrowedComplexModel(double[] params, int perturbIndex, double perturbStep) {
        return complexDelegate.borrowedComplexModel(params, perturbIndex, perturbStep);
    }

    private double[] harveyObservedInformationMatrixSurface(double[] params) {
        int paramCount = params.length;
        if (paramCount == 0) {
            return new double[0];
        }
        if (spec.concentrateScale()) {
            return null;
        }

        if (supportsAnalyticComplexStepOim()) {
            double[] analytic = analyticComplexStepHarveyInformationMatrix(params);
            if (analytic != null) {
                return analytic;
            }
        }

        FilterResult result = realDelegate.filter(params, FilterOptions.defaults());
        ForecastDerivativeSurfaces partials = forecastDerivativeSurfaces(params, result);
        if (partials == null) {
            return null;
        }

        int kEndog = result.kEndog;
        int start = Math.max(Math.max(0, likelihoodBurnInternal(params, null)), result.nobsDiffuse);
        double[] information = new double[paramCount * paramCount];
        double[] inverseForecastCov = new double[kEndog * kEndog];
        double[] transformedForecastCovPartials = new double[paramCount * kEndog * kEndog];
        double[] transformedForecastErrorPartials = new double[paramCount * kEndog];

        for (int t = start; t < result.nobs; t++) {
            int forecastCovOffset = result.forecastErrorCovOffset(t);
            if (!invertForecastCovariance(result.forecastErrorCov, forecastCovOffset, kEndog, inverseForecastCov)) {
                return null;
            }

            for (int paramIndex = 0; paramIndex < paramCount; paramIndex++) {
                int covOffset = partials.forecastErrorCovOffset(t, paramIndex, kEndog, paramCount);
                int transformedCovOffset = paramIndex * kEndog * kEndog;
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kEndog, kEndog, kEndog,
                    1.0, inverseForecastCov, 0, kEndog,
                    partials.forecastErrorCovPartials, covOffset, kEndog,
                    0.0, transformedForecastCovPartials, transformedCovOffset, kEndog);

                int errorOffset = partials.forecastErrorOffset(t, paramIndex, kEndog, paramCount);
                int transformedErrorOffset = paramIndex * kEndog;
                BLAS.dgemv(BLAS.Trans.NoTrans, kEndog, kEndog, 1.0,
                    inverseForecastCov, 0, kEndog,
                    partials.forecastErrorPartials, errorOffset, 1,
                    0.0, transformedForecastErrorPartials, transformedErrorOffset, 1);
            }

            for (int i = 0; i < paramCount; i++) {
                int leftCovOffset = i * kEndog * kEndog;
                int leftErrorOffset = partials.forecastErrorOffset(t, i, kEndog, paramCount);
                for (int j = 0; j < paramCount; j++) {
                    int rightCovOffset = j * kEndog * kEndog;
                    int rightErrorOffset = j * kEndog;
                    information[i * paramCount + j] += 0.5 * traceProduct(
                        transformedForecastCovPartials, leftCovOffset,
                        transformedForecastCovPartials, rightCovOffset,
                        kEndog);
                    information[i * paramCount + j] += BLAS.ddot(
                        kEndog,
                        partials.forecastErrorPartials, leftErrorOffset, 1,
                        transformedForecastErrorPartials, rightErrorOffset, 1);
                }
            }
        }
        return information;
    }

    private IllegalStateException unavailableObservedInformationException() {
        String reason = spec.concentrateScale()
            ? "observed information is unavailable when concentrateScale=true"
            : "Harvey observed information could not be formed for this evaluation point";
        return new IllegalStateException(
            reason + "; choose covarianceFromApproximateInformation or robustCovarianceFromApproximateInformation explicitly");
    }

    private boolean supportsAnalyticApproximateHessian() {
        return observationDimension() == 1
            && !meta.stateRegression()
            && !meta.timeVaryingRegression()
            && !spec.concentrateScale()
            && !spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE);
    }

    private double[] approximateHessianComplexStepSurface(double[] params) {
        int paramCount = params.length;
        double[] hessian = new double[paramCount * paramCount];
        if (paramCount == 0) {
            return hessian;
        }

        phaseWorkspace.ensureParamCount(paramCount);
        double[] steps = phaseWorkspace.steps;
        double[] plus = phaseWorkspace.plusParams;
        double[] minus = phaseWorkspace.minusParams;
        for (int i = 0; i < paramCount; i++) {
            steps[i] = approximateHessianStepSize(params[i]);
        }

        for (int i = 0; i < paramCount; i++) {
            for (int j = i; j < paramCount; j++) {
                System.arraycopy(params, 0, plus, 0, paramCount);
                plus[j] += steps[j];
                double[] llfPlus = analyticComplexLogLikelihood(plus, i, steps[i]);
                if (llfPlus == null) {
                    return null;
                }

                System.arraycopy(params, 0, minus, 0, paramCount);
                minus[j] -= steps[j];
                double[] llfMinus = analyticComplexLogLikelihood(minus, i, steps[i]);
                if (llfMinus == null) {
                    return null;
                }

                double value = (llfPlus[1] - llfMinus[1]) / (2.0 * steps[i] * steps[j]);
                hessian[i * paramCount + j] = value;
                hessian[j * paramCount + i] = value;
            }
        }
        return hessian;
    }

    private boolean supportsAnalyticComplexStepOim() {
        return observationDimension() == 1
            && !meta.stateRegression()
            && !meta.timeVaryingRegression()
            && !spec.concentrateScale()
            && !spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE);
    }

    private double[] analyticComplexStepHarveyInformationMatrix(double[] params) {
        int paramCount = params.length;
        FilterResult baseline = realDelegate.filter(params, FilterOptions.defaults());
        int start = Math.max(Math.max(0, likelihoodBurnInternal(params, null)), baseline.nobsDiffuse);
        double[] forecastErrorPartials = new double[baseline.nobs * paramCount];
        double[] forecastVariancePartials = new double[baseline.nobs * paramCount];

        for (int paramIndex = 0; paramIndex < paramCount; paramIndex++) {
            double step = harveyDerivativeStepSize(params[paramIndex]);
            if (!writeAnalyticComplexForecastPartials(params, paramIndex, step,
                    forecastErrorPartials, forecastVariancePartials, paramCount, baseline.nobs)) {
                return null;
            }
        }

        double[] information = new double[paramCount * paramCount];
        for (int t = start; t < baseline.nobs; t++) {
            int forecastCovOffset = baseline.forecastErrorCovOffset(t);
            double forecastVariance = baseline.forecastErrorCov[forecastCovOffset];
            if (!(forecastVariance > 0.0) || !Double.isFinite(forecastVariance)) {
                return null;
            }
            double inverseForecastVariance = 1.0 / forecastVariance;
            for (int i = 0; i < paramCount; i++) {
                double leftVariancePartial = forecastVariancePartials[t * paramCount + i];
                double leftErrorPartial = forecastErrorPartials[t * paramCount + i];
                for (int j = 0; j < paramCount; j++) {
                    double rightVariancePartial = forecastVariancePartials[t * paramCount + j];
                    double rightErrorPartial = forecastErrorPartials[t * paramCount + j];
                    information[i * paramCount + j] += 0.5 * leftVariancePartial * rightVariancePartial
                        * inverseForecastVariance * inverseForecastVariance;
                    information[i * paramCount + j] += leftErrorPartial * rightErrorPartial * inverseForecastVariance;
                }
            }
        }
        return information;
    }

    private boolean writeAnalyticComplexForecastPartials(double[] params,
                                                         int perturbIndex,
                                                         double perturbStep,
                                                         double[] forecastErrorPartials,
                                                         double[] forecastVariancePartials,
                                                         int paramCount,
                                                         int nobs) {
        AnalyticComplexWorkspace workspace = analyticComplexWorkspace;
        workspace.ensureForecastSurface(nobs);
        if (!runAnalyticComplexFilter(params, perturbIndex, perturbStep, workspace,
                null, 0, 0,
                null, null,
                workspace.forecastError, workspace.forecastVariance)) {
            return false;
        }
        for (int t = 0; t < nobs; t++) {
            forecastErrorPartials[t * paramCount + perturbIndex] = workspace.forecastError[t * 2 + 1] / perturbStep;
            forecastVariancePartials[t * paramCount + perturbIndex] = workspace.forecastVariance[t * 2 + 1] / perturbStep;
        }
        return true;
    }

    private ForecastDerivativeSurfaces forecastDerivativeSurfaces(double[] params, FilterResult baseline) {
        int nobs = baseline.nobs;
        int kEndog = baseline.kEndog;
        int paramCount = params.length;
        double[] forecastErrorPartials = new double[nobs * paramCount * kEndog];
        double[] forecastErrorCovPartials = new double[nobs * paramCount * kEndog * kEndog];

        double[] perturbed = Arrays.copyOf(params, params.length);
        for (int paramIndex = 0; paramIndex < paramCount; paramIndex++) {
            double step = harveyDerivativeStepSize(params[paramIndex]);
            perturbed[paramIndex] = params[paramIndex] + step;
            FilterResult shifted = realDelegate.filter(perturbed, FilterOptions.defaults());
            perturbed[paramIndex] = params[paramIndex];

            for (int t = 0; t < nobs; t++) {
                int targetErrorOffset = (t * paramCount + paramIndex) * kEndog;
                int baseErrorOffset = baseline.forecastErrorOffset(t);
                int shiftedErrorOffset = shifted.forecastErrorOffset(t);
                for (int row = 0; row < kEndog; row++) {
                    forecastErrorPartials[targetErrorOffset + row] =
                        (shifted.forecastError[shiftedErrorOffset + row] - baseline.forecastError[baseErrorOffset + row]) / step;
                }

                int targetCovOffset = (t * paramCount + paramIndex) * kEndog * kEndog;
                int baseCovOffset = baseline.forecastErrorCovOffset(t);
                int shiftedCovOffset = shifted.forecastErrorCovOffset(t);
                int covLength = kEndog * kEndog;
                for (int index = 0; index < covLength; index++) {
                    forecastErrorCovPartials[targetCovOffset + index] =
                        (shifted.forecastErrorCov[shiftedCovOffset + index] - baseline.forecastErrorCov[baseCovOffset + index]) / step;
                }
            }
        }
        return new ForecastDerivativeSurfaces(forecastErrorPartials, forecastErrorCovPartials);
    }

    private boolean supportsAnalyticComplexStepScore() {
        return observationDimension() == 1
            && !meta.stateRegression()
            && !meta.timeVaryingRegression()
            && !spec.concentrateScale()
            && !spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE);
    }

    private double[] analyticComplexStepScoreObs(double[] params) {
        double[] analytic = analyticComplexStepScoreObsOrNull(params);
        return analytic != null ? analytic : realDelegate.scoreObs(params);
    }

    private double[] analyticComplexStepScoreObsOrNull(double[] params) {
        int nobs = observationCount();
        int paramCount = params.length;
        double[] scoreObs = new double[nobs * paramCount];
        if (nobs == 0 || paramCount == 0) {
            return scoreObs;
        }

        for (int paramIndex = 0; paramIndex < paramCount; paramIndex++) {
            double step = complexStepSize(params[paramIndex]);
            if (!runAnalyticComplexFilter(params, paramIndex, step, analyticComplexWorkspace,
                    scoreObs, paramCount, paramIndex,
                    null, null,
                    null, null)) {
                return null;
            }
        }
        return scoreObs;
    }

    private double[] analyticComplexLogLikelihood(double[] params, int perturbIndex, double perturbStep) {
        AnalyticComplexWorkspace workspace = analyticComplexWorkspace;
        if (!runAnalyticComplexFilter(params, perturbIndex, perturbStep, workspace,
                null, 0, 0,
                null, workspace.logLikelihoodSum,
                null, null)) {
            return null;
        }
        return complexScalar(workspace.logLikelihoodSum[0], workspace.logLikelihoodSum[1]);
    }

    private boolean runAnalyticComplexFilter(double[] params,
                                             int perturbIndex,
                                             double perturbStep,
                                             AnalyticComplexWorkspace workspace,
                                             double[] scoreObs,
                                             int scoreParamCount,
                                             int scoreParamIndex,
                                             double[] logLikelihoodObs,
                                             double[] logLikelihoodSum,
                                             double[] forecastErrorOut,
                                             double[] forecastVarianceOut) {
        KalmanSSM model = borrowedComplexModel(params, perturbIndex, perturbStep);
        int nobs = model.observationCount();
        int stateCount = model.stateCount();
        int kPosdef = model.stateDisturbanceCount();
        int stateLength = stateCount * 2;
        int covarianceLength = stateCount * stateCount * 2;
        workspace.ensureFilter(stateCount, kPosdef);
        if (!analyticInitializationInto(model, workspace)) {
            return false;
        }

        if (logLikelihoodObs != null) {
            Arrays.fill(logLikelihoodObs, 0, nobs * 2, 0.0);
        }
        if (logLikelihoodSum != null) {
            logLikelihoodSum[0] = 0.0;
            logLikelihoodSum[1] = 0.0;
        }
        if (forecastErrorOut != null) {
            Arrays.fill(forecastErrorOut, 0, nobs * 2, 0.0);
        }
        if (forecastVarianceOut != null) {
            Arrays.fill(forecastVarianceOut, 0, nobs * 2, 0.0);
        }

        boolean computeLikelihood = scoreObs != null || logLikelihoodObs != null || logLikelihoodSum != null;
        int burn = computeLikelihood ? Math.min(Math.max(0, likelihoodBurnInternal(params, model)), nobs) : 0;
        double logTwoPi = Math.log(2.0 * Math.PI);
        double[] state = workspace.state0;
        double[] covariance = workspace.covariance0;
        double[] nextState = workspace.state1;
        double[] nextCovariance = workspace.covariance1;
        double[] dot = workspace.dot;
        double[] gainNumerator = workspace.gainNumerator;
        double[] filteredStateWorkspace = workspace.filteredState;
        double[] filteredCovarianceWorkspace = workspace.filteredCovariance;
        double[] transitionTimesCovariance = workspace.transitionTimesCovariance;
        double[] selectedCovariance = workspace.selectedCovariance;

        for (int t = 0; t < nobs; t++) {
            int designOffset = model.designOffset(t);
            int obsInterceptOffset = model.obsInterceptOffset(t);
            int obsCovOffset = model.obsCovOffset(t);
            int transitionOffset = model.transitionOffset(t);
            int stateInterceptOffset = model.stateInterceptOffset(t);
            int selectionOffset = model.selectionOffset(t);
            int stateCovOffset = model.stateCovarianceOffset(t);
            int endogOffset = model.endogOffset(t);

            boolean missing = model.isMissing(0, t) || Double.isNaN(model.endog[endogOffset]);
            double[] filteredState = state;
            double[] filteredCovariance = covariance;

            if (!missing) {
                ZLAS.zdotu(stateCount, model.design, designOffset >> 1, 1, state, 0, 1, dot);
                double forecastErrorRe = model.endog[endogOffset] - model.obsIntercept[obsInterceptOffset] - dot[0];
                double forecastErrorIm = model.endog[endogOffset + 1] - model.obsIntercept[obsInterceptOffset + 1] - dot[1];

                ZLAS.zgemv(BLAS.Trans.NoTrans, stateCount, stateCount, 1.0, 0.0,
                    covariance, 0, stateCount,
                    model.design, designOffset, 1,
                    0.0, 0.0, gainNumerator, 0, 1);
                ZLAS.zdotu(stateCount, model.design, designOffset >> 1, 1, gainNumerator, 0, 1, dot);
                double forecastVarianceRe = model.obsCov[obsCovOffset] + dot[0];
                double forecastVarianceIm = model.obsCov[obsCovOffset + 1] + dot[1];
                if (!Double.isFinite(forecastVarianceRe)
                        || !Double.isFinite(forecastVarianceIm)
                        || isNearZero(forecastVarianceRe, forecastVarianceIm)) {
                    return false;
                }

                if (forecastErrorOut != null) {
                    int offset = t * 2;
                    forecastErrorOut[offset] = forecastErrorRe;
                    forecastErrorOut[offset + 1] = forecastErrorIm;
                    forecastVarianceOut[offset] = forecastVarianceRe;
                    forecastVarianceOut[offset + 1] = forecastVarianceIm;
                }

                double denom = forecastVarianceRe * forecastVarianceRe + forecastVarianceIm * forecastVarianceIm;
                if (!(denom > 0.0) || !Double.isFinite(denom)) {
                    return false;
                }
                double inverseForecastVarianceRe = forecastVarianceRe / denom;
                double inverseForecastVarianceIm = -forecastVarianceIm / denom;
                double updateScaleRe = inverseForecastVarianceRe * forecastErrorRe - inverseForecastVarianceIm * forecastErrorIm;
                double updateScaleIm = inverseForecastVarianceRe * forecastErrorIm + inverseForecastVarianceIm * forecastErrorRe;

                System.arraycopy(state, 0, filteredStateWorkspace, 0, stateLength);
                ZLAS.zaxpy(stateCount, updateScaleRe, updateScaleIm,
                    gainNumerator, 0, 1,
                    filteredStateWorkspace, 0, 1);
                filteredState = filteredStateWorkspace;

                System.arraycopy(covariance, 0, filteredCovarianceWorkspace, 0, covarianceLength);
                ZLAS.zgeru(stateCount, stateCount, -inverseForecastVarianceRe, -inverseForecastVarianceIm,
                    gainNumerator, 0, 1,
                    gainNumerator, 0, 1,
                    filteredCovarianceWorkspace, 0, stateCount);
                filteredCovariance = filteredCovarianceWorkspace;

                if (computeLikelihood && t >= burn) {
                    double forecastErrorSquareRe = forecastErrorRe * forecastErrorRe - forecastErrorIm * forecastErrorIm;
                    double forecastErrorSquareIm = 2.0 * forecastErrorRe * forecastErrorIm;
                    double quadraticRe = forecastErrorSquareRe * inverseForecastVarianceRe
                        - forecastErrorSquareIm * inverseForecastVarianceIm;
                    double quadraticIm = forecastErrorSquareRe * inverseForecastVarianceIm
                        + forecastErrorSquareIm * inverseForecastVarianceRe;
                    double magnitudeSquared = forecastVarianceRe * forecastVarianceRe + forecastVarianceIm * forecastVarianceIm;
                    if (!(magnitudeSquared > 0.0) || !Double.isFinite(magnitudeSquared)) {
                        return false;
                    }
                    double logLikelihoodRe = -0.5 * (logTwoPi + 0.5 * Math.log(magnitudeSquared) + quadraticRe);
                    double logLikelihoodIm = -0.5 * (Math.atan2(forecastVarianceIm, forecastVarianceRe) + quadraticIm);
                    if (scoreObs != null) {
                        scoreObs[t * scoreParamCount + scoreParamIndex] = logLikelihoodIm / perturbStep;
                    }
                    if (logLikelihoodObs != null) {
                        int offset = t * 2;
                        logLikelihoodObs[offset] = logLikelihoodRe;
                        logLikelihoodObs[offset + 1] = logLikelihoodIm;
                    }
                    if (logLikelihoodSum != null) {
                        logLikelihoodSum[0] += logLikelihoodRe;
                        logLikelihoodSum[1] += logLikelihoodIm;
                    }
                }
            }

            System.arraycopy(model.stateIntercept, stateInterceptOffset, nextState, 0, stateLength);
            ZLAS.zgemv(BLAS.Trans.NoTrans, stateCount, stateCount, 1.0, 0.0,
                model.transition, transitionOffset, model.transitionLeadingDimension(),
                filteredState, 0, 1,
                1.0, 0.0, nextState, 0, 1);
            complexMatrixMultiplyInto(
                model.transition, transitionOffset >> 1, model.transitionLeadingDimension(),
                filteredCovariance, 0, stateCount,
                stateCount, stateCount, stateCount,
                transitionTimesCovariance, 0, stateCount,
                0.0, 0.0);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, stateCount, stateCount, stateCount,
                1.0, 0.0, transitionTimesCovariance, 0, stateCount,
                model.transition, transitionOffset >> 1, model.transitionLeadingDimension(),
                0.0, 0.0, nextCovariance, 0, stateCount);
            if (kPosdef > 0) {
                complexMatrixMultiplyInto(
                    model.selection, selectionOffset >> 1, model.selectionLeadingDimension(),
                    model.stateCov, stateCovOffset >> 1, model.stateCovarianceLeadingDimension(),
                    stateCount, kPosdef, kPosdef,
                    selectedCovariance, 0, kPosdef,
                    0.0, 0.0);
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, stateCount, stateCount, kPosdef,
                    1.0, 0.0, selectedCovariance, 0, kPosdef,
                    model.selection, selectionOffset >> 1, model.selectionLeadingDimension(),
                    1.0, 0.0, nextCovariance, 0, stateCount);
            }

            double[] stateSwap = state;
            state = nextState;
            nextState = stateSwap;
            double[] covarianceSwap = covariance;
            covariance = nextCovariance;
            nextCovariance = covarianceSwap;
        }
        return true;
    }

    private boolean analyticInitializationInto(KalmanSSM model, AnalyticComplexWorkspace workspace) {
        if (spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)) {
            return false;
        }

        int stateCount = model.stateCount();
        double[] state = workspace.state0;
        double[] covariance = workspace.covariance0;
        Arrays.fill(state, 0, stateCount * 2, 0.0);
        Arrays.fill(covariance, 0, stateCount * stateCount * 2, 0.0);

        boolean approximateDiffuseOnly = spec.hasApproximateDiffuseVariance()
            && !spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE);
        if (!spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY) || approximateDiffuseOnly) {
            fillComplexDiagonal(covariance, 0, stateCount, stateCount, spec.approximateDiffuseVariance());
            return true;
        }

        if (meta.totalDiff() > 0) {
            fillComplexDiagonal(covariance, 0, stateCount, meta.totalDiff(), spec.approximateDiffuseVariance());
        }

        int stationaryStart = meta.totalDiff();
        int stationaryDimension = meta.totalDiff() == 0 && !meta.stateRegression()
            ? stateCount
            : meta.totalOrder();
        if (stationaryDimension > 0
                && !solveAnalyticStationaryBlockInto(model, stationaryStart, stationaryDimension, state, covariance, workspace)) {
            return false;
        }

        if (meta.stateRegression()) {
            int regressionStart = meta.totalDiff() + meta.totalOrder();
            fillComplexDiagonal(covariance, regressionStart, stateCount, stateCount - regressionStart,
                stateRegressionApproximateDiffuseVariance());
        }
        return true;
    }

    private boolean solveAnalyticStationaryBlockInto(KalmanSSM model,
                                                     int startState,
                                                     int blockDimension,
                                                     double[] targetState,
                                                     double[] targetCovariance,
                                                     AnalyticComplexWorkspace workspace) {
        int kPosdef = model.stateDisturbanceCount();
        workspace.ensureStationary(blockDimension, kPosdef);
        copyComplexSquareBlockInto(
            model.transition,
            model.transitionOffset(0),
            model.transitionLeadingDimension(),
            startState,
            blockDimension,
            workspace.stationaryTransition);
        copyComplexVectorBlockInto(
            model.stateIntercept,
            model.stateInterceptOffset(0),
            startState,
            blockDimension,
            workspace.stationaryStateIntercept);
        copyComplexRectangularBlockInto(
            model.selection,
            model.selectionOffset(0),
            model.selectionLeadingDimension(),
            startState,
            blockDimension,
            kPosdef,
            workspace.stationarySelection);
        copyComplexSquareBlockInto(
            model.stateCov,
            model.stateCovarianceOffset(0),
            model.stateCovarianceLeadingDimension(),
            0,
            kPosdef,
            workspace.stationaryStateCovariance);

        if (!solveAnalyticStationaryMeanInto(
                workspace.stationaryTransition,
                workspace.stationaryStateIntercept,
                blockDimension,
                workspace)) {
            return false;
        }
        if (!solveAnalyticStationaryCovarianceInto(
                workspace.stationaryTransition,
                workspace.stationarySelection,
                workspace.stationaryStateCovariance,
                blockDimension,
                kPosdef,
                workspace)) {
            return false;
        }
        int stationaryStateLength = blockDimension * 2;
        int stationaryCovarianceLength = blockDimension * blockDimension * 2;
        if (!allFinite(workspace.stationaryState, 0, stationaryStateLength)
                || !allFinite(workspace.stationaryCovarianceVector, 0, stationaryCovarianceLength)) {
            return false;
        }

        System.arraycopy(workspace.stationaryState, 0, targetState, startState * 2, stationaryStateLength);
        writeComplexSquareBlock(workspace.stationaryCovarianceVector, blockDimension,
            targetCovariance, targetState.length / 2, startState);
        return true;
    }

    private boolean solveAnalyticStationaryMeanInto(double[] transition,
                                                    double[] stateIntercept,
                                                    int dimension,
                                                    AnalyticComplexWorkspace workspace) {
        if (dimension == 0) {
            return true;
        }

        double[] system = workspace.stationaryMeanSystem;
        for (int row = 0; row < dimension; row++) {
            int rowOffset = row * dimension * 2;
            for (int col = 0; col < dimension; col++) {
                int off = rowOffset + col * 2;
                system[off] = -transition[off];
                system[off + 1] = -transition[off + 1];
            }
            system[rowOffset + row * 2] += 1.0;
        }

        System.arraycopy(stateIntercept, 0, workspace.stationaryState, 0, dimension * 2);
        return solveRealifiedComplexSystemInto(system, workspace.stationaryState, dimension, workspace);
    }

    private boolean solveAnalyticStationaryCovarianceInto(double[] transition,
                                                          double[] selection,
                                                          double[] stateCovariance,
                                                          int dimension,
                                                          int kPosdef,
                                                          AnalyticComplexWorkspace workspace) {
        if (dimension == 0) {
            return true;
        }

        double[] rhs = workspace.stationaryCovarianceRhs;
        Arrays.fill(rhs, 0, dimension * dimension * 2, 0.0);
        if (kPosdef > 0) {
            complexMatrixMultiplyInto(selection, 0, kPosdef,
                stateCovariance, 0, kPosdef,
                dimension, kPosdef, kPosdef,
                workspace.stationarySelectedCovariance, 0, kPosdef,
                0.0, 0.0);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, dimension, dimension, kPosdef,
                1.0, 0.0, workspace.stationarySelectedCovariance, 0, kPosdef,
                selection, 0, kPosdef,
                0.0, 0.0, rhs, 0, dimension);
        }

        int systemDimension = dimension * dimension;
        double[] system = workspace.stationaryCovarianceSystem;
        double[] vector = workspace.stationaryCovarianceVector;
        Arrays.fill(system, 0, systemDimension * systemDimension * 2, 0.0);
        for (int row = 0; row < dimension; row++) {
            for (int col = 0; col < dimension; col++) {
                int equation = row * dimension + col;
                int vectorOffset = equation * 2;
                int rhsOffset = (row * dimension + col) * 2;
                vector[vectorOffset] = rhs[rhsOffset];
                vector[vectorOffset + 1] = rhs[rhsOffset + 1];
                system[(equation * systemDimension + equation) * 2] = 1.0;
                for (int leftIndex = 0; leftIndex < dimension; leftIndex++) {
                    int leftOffset = (row * dimension + leftIndex) * 2;
                    double leftRe = transition[leftOffset];
                    double leftIm = transition[leftOffset + 1];
                    if (leftRe == 0.0 && leftIm == 0.0) {
                        continue;
                    }
                    for (int rightIndex = 0; rightIndex < dimension; rightIndex++) {
                        int rightOffset = (col * dimension + rightIndex) * 2;
                        double rightRe = transition[rightOffset];
                        double rightIm = transition[rightOffset + 1];
                        if (rightRe == 0.0 && rightIm == 0.0) {
                            continue;
                        }
                        int coefficientOffset = (equation * systemDimension + leftIndex * dimension + rightIndex) * 2;
                        double productRe = leftRe * rightRe - leftIm * rightIm;
                        double productIm = leftRe * rightIm + leftIm * rightRe;
                        system[coefficientOffset] -= productRe;
                        system[coefficientOffset + 1] -= productIm;
                    }
                }
            }
        }

        if (!solveRealifiedComplexSystemInto(system, vector, systemDimension, workspace)) {
            return false;
        }
        symmetrizeComplexTranspose(vector, dimension);
        return true;
    }

    private boolean solveRealifiedComplexSystemInto(double[] complexMatrix,
                                                   double[] complexVector,
                                                   int dimension,
                                                   AnalyticComplexWorkspace workspace) {
        int realDimension = dimension * 2;
        workspace.ensureRealifiedCapacity(realDimension);
        double[] realMatrix = workspace.realifiedMatrix;
        double[] realVector = workspace.realifiedVector;
        int[] pivots = workspace.realifiedPivots;
        for (int row = 0; row < dimension; row++) {
            realVector[row] = complexVector[row * 2];
            realVector[dimension + row] = complexVector[row * 2 + 1];
            for (int col = 0; col < dimension; col++) {
                int complexOffset = (row * dimension + col) * 2;
                double re = complexMatrix[complexOffset];
                double im = complexMatrix[complexOffset + 1];
                realMatrix[row * realDimension + col] = re;
                realMatrix[row * realDimension + dimension + col] = -im;
                realMatrix[(dimension + row) * realDimension + col] = im;
                realMatrix[(dimension + row) * realDimension + dimension + col] = re;
            }
        }

        if (BLAS.dgetrf(realDimension, realDimension, realMatrix, 0, realDimension, pivots, 0) != 0) {
            return false;
        }
        BLAS.dgetrs(BLAS.Trans.NoTrans, realDimension, 1, realMatrix, 0, realDimension, pivots, 0, realVector, 0, 1);
        for (int row = 0; row < dimension; row++) {
            complexVector[row * 2] = realVector[row];
            complexVector[row * 2 + 1] = realVector[dimension + row];
        }
        return allFinite(realVector, 0, realDimension);
    }

    private static void copyComplexVectorBlockInto(double[] source,
                                                   int offset,
                                                   int startState,
                                                   int blockDimension,
                                                   double[] target) {
        int length = blockDimension * 2;
        Arrays.fill(target, 0, length, 0.0);
        if (source != null) {
            System.arraycopy(source, offset + startState * 2, target, 0, length);
        }
    }

    private static void copyComplexRectangularBlockInto(double[] source,
                                                        int offset,
                                                        int leadingDimension,
                                                        int startRow,
                                                        int rows,
                                                        int cols,
                                                        double[] target) {
        int length = rows * cols * 2;
        Arrays.fill(target, 0, length, 0.0);
        if (source == null) {
            return;
        }
        for (int row = 0; row < rows; row++) {
            int sourceRow = offset + (startRow + row) * leadingDimension * 2;
            int targetRow = row * cols * 2;
            System.arraycopy(source, sourceRow, target, targetRow, cols * 2);
        }
    }

    private static void copyComplexSquareBlockInto(double[] source,
                                                   int offset,
                                                   int leadingDimension,
                                                   int startState,
                                                   int blockDimension,
                                                   double[] target) {
        int length = blockDimension * blockDimension * 2;
        Arrays.fill(target, 0, length, 0.0);
        if (source == null) {
            return;
        }
        for (int row = 0; row < blockDimension; row++) {
            int sourceRow = offset + (startState + row) * leadingDimension * 2 + startState * 2;
            int targetRow = row * blockDimension * 2;
            System.arraycopy(source, sourceRow, target, targetRow, blockDimension * 2);
        }
    }

    private static void writeComplexSquareBlock(double[] source,
                                                int sourceDimension,
                                                double[] target,
                                                int targetDimension,
                                                int startState) {
        for (int row = 0; row < sourceDimension; row++) {
            int sourceRow = row * sourceDimension * 2;
            int targetRow = ((startState + row) * targetDimension + startState) * 2;
            System.arraycopy(source, sourceRow, target, targetRow, sourceDimension * 2);
        }
    }

    private static void fillComplexDiagonal(double[] matrix,
                                            int startState,
                                            int totalDimension,
                                            int blockDimension,
                                            double value) {
        for (int i = 0; i < blockDimension; i++) {
            int diagonal = ((startState + i) * totalDimension + startState + i) * 2;
            matrix[diagonal] = value;
            matrix[diagonal + 1] = 0.0;
        }
    }

    private static void symmetrizeComplexTranspose(double[] matrix, int dimension) {
        for (int row = 0; row < dimension; row++) {
            for (int col = row + 1; col < dimension; col++) {
                int upper = (row * dimension + col) * 2;
                int lower = (col * dimension + row) * 2;
                double avgRe = 0.5 * (matrix[upper] + matrix[lower]);
                double avgIm = 0.5 * (matrix[upper + 1] + matrix[lower + 1]);
                matrix[upper] = avgRe;
                matrix[upper + 1] = avgIm;
                matrix[lower] = avgRe;
                matrix[lower + 1] = avgIm;
            }
        }
    }

    public SARIMAXResults fit() {
        return fit(SARIMAXFitOptions.DEFAULT);
    }

    public SARIMAXResults fit(SARIMAXFitOptions options) {
        SARIMAXFitOptions fitOptions = options == null ? SARIMAXFitOptions.DEFAULT : options;
        double[] startParams = fitOptions.startParams() == null ? startParams() : fitOptions.startParams();
        FixedParameterMLE.Result fit = FixedParameterMLE.optimize(new FixedParameterMLE.Model() {
            @Override
            public double[] transformParams(double[] unconstrainedParams) {
                return SARIMAX.this.transformParams(unconstrainedParams);
            }

            @Override
            public double[] untransformParams(double[] params) {
                return SARIMAX.this.untransformParams(params);
            }

            @Override
            public double logLikelihood(double[] params) {
                return SARIMAX.this.logLikelihood(params);
            }
        }, startParams, fitOptions.fixedParameters(), fitOptions.optimizer());

        Optimization optimization = fit.optimization();
        double[] unconstrained = fit.unconstrainedParams();
        double[] params = fit.params();
        FilterOptions resultOptions = spec.concentrateScale()
            ? realConcentratedLikelihoodOptions(FilterOptions.defaults())
            : FilterOptions.defaults();
        FilterResult filterResult = filter(params, resultOptions);
        return SARIMAXResults.adoptFilterResult(
            this,
            optimization,
            params,
            unconstrained,
            filterResult,
            fitOptions.covarianceType(),
            fitOptions.fixedParameters());
    }

    SARIMAX extend(double[] endog, double[][] exog) {
        return extend(endog, exog, spec.trendOffset());
    }

    SARIMAX extend(double[] endog, double[][] exog, int trendOffset) {
        SARIMAXSpec.Builder builder = SARIMAXSpec.builder(spec.order(), endog)
            .seasonalOrder(spec.seasonalOrder())
            .trendPowers(spec.trendPowers())
            .trendOffset(trendOffset)
            .autoregressiveLags(spec.autoregressiveLags())
            .movingAverageLags(spec.movingAverageLags())
            .measurementError(spec.measurementError())
            .concentrateScale(spec.concentrateScale())
            .mleRegression(spec.mleRegression())
            .timeVaryingRegression(spec.timeVaryingRegression())
            .simpleDifferencing(spec.simpleDifferencing())
            .hamiltonRepresentation(spec.hamiltonRepresentation());
        if (spec.hasApproximateDiffuseVariance()) {
            builder.approximateDiffuseVariance(spec.approximateDiffuseVariance());
        }
        if (exog != null) {
            builder.exog(exog);
        }
        for (SARIMAXOption option : spec.options()) {
            builder.include(option);
        }
        SARIMAXSpec extendedSpec = builder.build();
        return new SARIMAX(extendedSpec, SARIMAXSupport.extend(meta, extendedSpec));
    }

    private double[] transformParamsInternal(double[] unconstrainedParams) {
        double[] constrained = new double[unconstrainedParams.length];
        copyParameterBlock(unconstrainedParams, constrained, parameterLayout.trendOffset(), parameterLayout.trendLength());
        copyParameterBlock(unconstrainedParams, constrained, parameterLayout.exogOffset(), parameterLayout.exogLength());

        transformStationaryBlock(
            unconstrainedParams,
            constrained,
            parameterLayout.arOffset(),
            parameterLayout.arLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY),
            1.0,
            transformWorkspace);
        transformStationaryBlock(
            unconstrainedParams,
            constrained,
            parameterLayout.maOffset(),
            parameterLayout.maLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_INVERTIBILITY),
            -1.0,
            transformWorkspace);
        transformStationaryBlock(
            unconstrainedParams,
            constrained,
            parameterLayout.seasonalArOffset(),
            parameterLayout.seasonalArLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY),
            1.0,
            transformWorkspace);
        transformStationaryBlock(
            unconstrainedParams,
            constrained,
            parameterLayout.seasonalMaOffset(),
            parameterLayout.seasonalMaLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_INVERTIBILITY),
            -1.0,
            transformWorkspace);

        if (parameterLayout.timeVaryingRegressionLength() > 0) {
            for (int index = 0; index < parameterLayout.timeVaryingRegressionLength(); index++) {
                int paramIndex = parameterLayout.timeVaryingRegressionOffset() + index;
                constrained[paramIndex] = Math.pow(unconstrainedParams[paramIndex], 2);
            }
        }

        if (parameterLayout.hasMeasurementError()) {
            int index = parameterLayout.measurementErrorOffset();
            constrained[index] = Math.pow(unconstrainedParams[index], 2);
        }

        if (parameterLayout.hasScale()) {
            int index = parameterLayout.scaleOffset();
            constrained[index] = Math.pow(unconstrainedParams[index], 2);
        }
        return constrained;
    }

    private double[] untransformParamsInternal(double[] params) {
        double[] unconstrained = new double[params.length];
        copyParameterBlock(params, unconstrained, parameterLayout.trendOffset(), parameterLayout.trendLength());
        copyParameterBlock(params, unconstrained, parameterLayout.exogOffset(), parameterLayout.exogLength());

        untransformStationaryBlock(
            params,
            unconstrained,
            parameterLayout.arOffset(),
            parameterLayout.arLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY),
            1.0,
            transformWorkspace);
        untransformStationaryBlock(
            params,
            unconstrained,
            parameterLayout.maOffset(),
            parameterLayout.maLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_INVERTIBILITY),
            -1.0,
            transformWorkspace);
        untransformStationaryBlock(
            params,
            unconstrained,
            parameterLayout.seasonalArOffset(),
            parameterLayout.seasonalArLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY),
            1.0,
            transformWorkspace);
        untransformStationaryBlock(
            params,
            unconstrained,
            parameterLayout.seasonalMaOffset(),
            parameterLayout.seasonalMaLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_INVERTIBILITY),
            -1.0,
            transformWorkspace);

        if (parameterLayout.timeVaryingRegressionLength() > 0) {
            for (int index = 0; index < parameterLayout.timeVaryingRegressionLength(); index++) {
                int paramIndex = parameterLayout.timeVaryingRegressionOffset() + index;
                unconstrained[paramIndex] = Math.sqrt(Math.max(params[paramIndex], 1e-8));
            }
        }

        if (parameterLayout.hasMeasurementError()) {
            int index = parameterLayout.measurementErrorOffset();
            unconstrained[index] = Math.sqrt(Math.max(params[index], 1e-8));
        }

        if (parameterLayout.hasScale()) {
            int index = parameterLayout.scaleOffset();
            unconstrained[index] = Math.sqrt(Math.max(params[index], 1e-8));
        }
        return unconstrained;
    }

    private double[] adjustLogLikelihoodObsInternal(FilterResult filterResult,
                                                    double[] logLikelihoodObs,
                                                    double[] params,
                                                    KalmanSSM model,
                                                    int burn) {
        if (!spec.concentrateScale()) {
            return logLikelihoodObs;
        }
        if (filterResult.concentratedLikelihood()) {
            return logLikelihoodObs;
        }
        return concentratedLogLikelihoodObs(filterResult, logLikelihoodObs, concentratedScale(filterResult, burn), burn);
    }

    private double adjustedLogLikelihoodInternal(FilterResult filterResult, int burn) {
        if (!spec.concentrateScale()) {
            int start = Math.max(0, burn);
            if (start == 0) {
                return filterResult.logLikelihood();
            }
            double sum = 0.0;
            for (int t = Math.min(start, filterResult.nobs); t < filterResult.nobs; t++) {
                sum += filterResult.logLikelihoodObs[filterResult.logLikelihoodObsOffset(t)];
            }
            return sum;
        }
        if (filterResult.concentratedLikelihood()) {
            return logLikelihoodAfterBurn(filterResult, burn);
        }
        double scale = concentratedScale(filterResult, burn);
        double logScale = Math.log(scale);
        double[] effectiveEndog = meta.endog();
        double sum = 0.0;
        int nobs = Math.min(filterResult.nobs, effectiveEndog.length);
        int concentratedStart = Math.max(Math.max(0, burn), filterResult.nobsDiffuse);
        for (int t = Math.max(0, burn); t < nobs; t++) {
            double value = filterResult.logLikelihoodObs[filterResult.logLikelihoodObsOffset(t)];
            double forecastVariance = filterResult.forecastErrorCov(0, 0, t);
            if (t >= concentratedStart && !Double.isNaN(effectiveEndog[t]) && forecastVariance > 0.0 && Double.isFinite(forecastVariance)) {
                double error = filterResult.forecastError(0, t);
                double scaleObs = error * error / forecastVariance;
                value += -0.5 * (logScale + scaleObs / scale - scaleObs);
            }
            sum += value;
        }
        return sum;
    }

    private void updateModelInternal(double[] params, KalmanSSM model) {
        updateModelInternal(params, model, new RealUpdateWorkspace());
    }

    private void updateModelInternal(double[] params, KalmanSSM model, RealUpdateWorkspace workspace) {
        ParameterLayout layout = parameterLayout;
        double measurementVariance = layout.hasMeasurementError() ? params[layout.measurementErrorOffset()] : 0.0;
        double variance = layout.hasScale() ? params[layout.scaleOffset()] : 1.0;

        double[] reducedArPolynomial = UNIT_POLYNOMIAL;
        int reducedArLength = 1;
        if (meta.reducedArOrder() > 0) {
            reducedArLength = buildReducedPolynomial(params,
                layout.arOffset(), layout.arLags(),
                layout.seasonalArOffset(), layout.seasonalArLags(),
                true,
                workspace);
            reducedArPolynomial = workspace.product;
        }

        if (meta.kExog() > 0 && !meta.stateRegression()) {
            for (int t = 0; t < meta.nobs(); t++) {
                model.obsIntercept[model.obsInterceptOffset(t)] = dot(params, layout.exogOffset(), meta.exog()[t], layout.exogLength());
            }
        }

        if (meta.kTrend() > 0) {
            double arScale = 0.0;
            for (int i = 0; i < reducedArLength; i++) {
                arScale += reducedArPolynomial[i];
            }
            if (!(Math.abs(arScale) > 0.0)) {
                arScale = 1.0;
            }
            for (int t = 0; t < meta.nobs(); t++) {
                double trendValue = dot(params, layout.trendOffset(), meta.trendData()[t], layout.trendLength());
                if (spec.hamiltonRepresentation()) {
                    model.obsIntercept[model.obsInterceptOffset(t)] += trendValue / arScale;
                } else {
                    model.stateIntercept[model.stateInterceptOffset(t) + meta.totalDiff()] = trendValue;
                }
            }
        }

        if (meta.reducedArOrder() > 0) {
            int transitionBase = model.transitionOffset(0);
            for (int i = 1; i < reducedArLength; i++) {
                if (spec.hamiltonRepresentation()) {
                    model.transition[transitionBase + meta.totalDiff() * meta.kStates() + meta.totalDiff() + i - 1] = -reducedArPolynomial[i];
                } else {
                    model.transition[transitionBase + (meta.totalDiff() + i - 1) * meta.kStates() + meta.totalDiff()] = -reducedArPolynomial[i];
                }
            }
        }

        if (meta.reducedMaOrder() > 0) {
            int reducedMaLength = buildReducedPolynomial(params,
                layout.maOffset(), layout.maLags(),
                layout.seasonalMaOffset(), layout.seasonalMaLags(),
                false,
                workspace);
            double[] reducedMa = workspace.product;
            for (int i = 1; i < reducedMaLength; i++) {
                if (spec.hamiltonRepresentation()) {
                    model.design[model.designOffset(0) + meta.totalDiff() + i] = reducedMa[i];
                } else {
                    int selectionBase = model.selectionOffset(0);
                    model.selection[selectionBase + meta.totalDiff() + i] = reducedMa[i];
                }
            }
        }

        if (meta.kPosdef() > 0) {
            model.stateCov[model.stateCovarianceOffset(0)] = spec.concentrateScale() ? 1.0 : Math.max(variance, 1e-8);
            if (meta.timeVaryingRegression()) {
                int diagonalOffset = meta.totalOrder() > 0 ? 1 : 0;
                for (int index = 0; index < meta.kExog(); index++) {
                    int matrixIndex = (diagonalOffset + index) * meta.kPosdef() + diagonalOffset + index;
                    model.stateCov[model.stateCovarianceOffset(0) + matrixIndex] = Math.max(params[layout.timeVaryingRegressionOffset() + index], 1e-8);
                }
            }
        }
        model.obsCov[model.obsCovOffset(0)] = spec.measurementError() ? Math.max(measurementVariance, 1e-8) : 0.0;
    }

    private void updateModelComplexInternal(double[] params, int perturbIndex, double perturbStep, KalmanSSM model) {
        updateModelComplexInternal(params, perturbIndex, perturbStep, model, new ComplexUpdateWorkspace());
    }

    private void updateModelComplexInternal(double[] params,
                                            int perturbIndex,
                                            double perturbStep,
                                            KalmanSSM model,
                                            ComplexUpdateWorkspace workspace) {
        ParameterLayout layout = parameterLayout;
        double[] trendParams = complexSlice(workspace.trendParams, params, perturbIndex, perturbStep,
            layout.trendOffset(), layout.trendEnd());
        workspace.trendParams = trendParams;
        double[] exogParams = meta.stateRegression()
            ? EMPTY_COMPLEX_ARRAY
            : complexSlice(workspace.exogParams, params, perturbIndex, perturbStep, layout.exogOffset(), layout.exogEnd());
        workspace.exogParams = exogParams;
        double[] arParams = complexSlice(workspace.arParams, params, perturbIndex, perturbStep, layout.arOffset(), layout.arEnd());
        workspace.arParams = arParams;
        double[] maParams = complexSlice(workspace.maParams, params, perturbIndex, perturbStep, layout.maOffset(), layout.maEnd());
        workspace.maParams = maParams;
        double[] seasonalArParams = complexSlice(workspace.seasonalArParams, params, perturbIndex, perturbStep,
            layout.seasonalArOffset(), layout.seasonalArEnd());
        workspace.seasonalArParams = seasonalArParams;
        double[] seasonalMaParams = complexSlice(workspace.seasonalMaParams, params, perturbIndex, perturbStep,
            layout.seasonalMaOffset(), layout.seasonalMaEnd());
        workspace.seasonalMaParams = seasonalMaParams;
        double[] exogVariances = meta.timeVaryingRegression()
            ? complexSlice(workspace.exogVariances, params, perturbIndex, perturbStep,
                layout.timeVaryingRegressionOffset(), layout.timeVaryingRegressionEnd())
            : EMPTY_COMPLEX_ARRAY;
        workspace.exogVariances = exogVariances;
        double measurementVarianceRe = spec.measurementError() ? params[layout.measurementErrorOffset()] : 0.0;
        double measurementVarianceIm = spec.measurementError()
            ? imaginaryStep(perturbIndex, layout.measurementErrorOffset(), perturbStep)
            : 0.0;
        double varianceRe = spec.concentrateScale() ? 1.0 : params[layout.scaleOffset()];
        double varianceIm = spec.concentrateScale() ? 0.0 : imaginaryStep(perturbIndex, layout.scaleOffset(), perturbStep);

        double[] reducedArPolynomial = UNIT_COMPLEX_POLYNOMIAL;
        int reducedArLength = 1;
        if (meta.reducedArOrder() > 0) {
            reducedArLength = buildReducedComplexPolynomial(
                arParams,
                spec.autoregressiveLags(),
                seasonalArParams,
                spec.seasonalAutoregressiveLags(),
                true,
                workspace);
            reducedArPolynomial = workspace.product;
        }

        if (meta.kExog() > 0 && !meta.stateRegression()) {
            for (int t = 0; t < meta.nobs(); t++) {
                dotComplexReal(exogParams, meta.exog()[t], workspace.scalar0);
                writeComplexScalar(model.obsIntercept, model.obsInterceptOffset(t), workspace.scalar0[0], workspace.scalar0[1]);
            }
        }

        if (meta.kTrend() > 0) {
            sumComplex(reducedArPolynomial, reducedArLength, workspace.scalar1);
            if (isNearZero(workspace.scalar1[0], workspace.scalar1[1])) {
                workspace.scalar1[0] = 1.0;
                workspace.scalar1[1] = 0.0;
            }
            for (int t = 0; t < meta.nobs(); t++) {
                dotComplexReal(trendParams, meta.trendData()[t], workspace.scalar0);
                if (spec.hamiltonRepresentation()) {
                    divideComplex(workspace.scalar0[0], workspace.scalar0[1], workspace.scalar1[0], workspace.scalar1[1], workspace.scalar2);
                    addComplexScalar(model.obsIntercept, model.obsInterceptOffset(t), workspace.scalar2[0], workspace.scalar2[1]);
                } else {
                    writeComplexScalar(
                        model.stateIntercept,
                        complexVectorOffset(model.stateInterceptOffset(t), meta.totalDiff()),
                        workspace.scalar0[0],
                        workspace.scalar0[1]);
                }
            }
        }

        if (meta.reducedArOrder() > 0) {
            for (int t = 0; t < meta.nobs(); t++) {
                int transitionBase = model.transitionOffset(t);
                for (int i = 1; i < reducedArLength; i++) {
                    int row = spec.hamiltonRepresentation() ? meta.totalDiff() : meta.totalDiff() + i - 1;
                    int col = spec.hamiltonRepresentation() ? meta.totalDiff() + i - 1 : meta.totalDiff();
                    int off = complexMatrixOffset(transitionBase, meta.kStates(), row, col);
                    writeComplexScalar(model.transition, off, -reducedArPolynomial[i * 2], -reducedArPolynomial[i * 2 + 1]);
                }
            }
        }

        if (meta.reducedMaOrder() > 0) {
            int reducedMaLength = buildReducedComplexPolynomial(
                maParams,
                spec.movingAverageLags(),
                seasonalMaParams,
                spec.seasonalMovingAverageLags(),
                false,
                workspace);
            double[] reducedMa = workspace.product;
            for (int t = 0; t < meta.nobs(); t++) {
                for (int i = 1; i < reducedMaLength; i++) {
                    if (spec.hamiltonRepresentation()) {
                        int off = complexVectorOffset(model.designOffset(t), meta.totalDiff() + i);
                        writeComplexScalar(model.design, off, reducedMa[i * 2], reducedMa[i * 2 + 1]);
                    } else {
                        int selectionBase = model.selectionOffset(t);
                        int off = complexMatrixOffset(selectionBase, meta.kPosdef(), meta.totalDiff() + i, 0);
                        writeComplexScalar(model.selection, off, reducedMa[i * 2], reducedMa[i * 2 + 1]);
                    }
                }
            }
        }

        if (meta.kPosdef() > 0) {
            positiveContinuation(varianceRe, varianceIm, workspace.scalar0);
            double stateVarianceRe = spec.concentrateScale() ? 1.0 : workspace.scalar0[0];
            double stateVarianceIm = spec.concentrateScale() ? 0.0 : workspace.scalar0[1];
            for (int t = 0; t < meta.nobs(); t++) {
                int stateCovBase = model.stateCovarianceOffset(t);
                writeComplexScalar(model.stateCov, stateCovBase, stateVarianceRe, stateVarianceIm);
                if (meta.timeVaryingRegression()) {
                    int diagonalOffset = meta.totalOrder() > 0 ? 1 : 0;
                    for (int index = 0; index < meta.kExog(); index++) {
                        int row = diagonalOffset + index;
                        int off = complexMatrixOffset(stateCovBase, meta.kPosdef(), row, row);
                        int complexIndex = index * 2;
                        positiveContinuation(exogVariances[complexIndex], exogVariances[complexIndex + 1], workspace.scalar0);
                        writeComplexScalar(model.stateCov, off, workspace.scalar0[0], workspace.scalar0[1]);
                    }
                }
            }
        }
        double observationVarianceRe = 0.0;
        double observationVarianceIm = 0.0;
        if (spec.measurementError()) {
            positiveContinuation(measurementVarianceRe, measurementVarianceIm, workspace.scalar0);
            observationVarianceRe = workspace.scalar0[0];
            observationVarianceIm = workspace.scalar0[1];
        }
        for (int t = 0; t < meta.nobs(); t++) {
            writeComplexScalar(model.obsCov, model.obsCovOffset(t), observationVarianceRe, observationVarianceIm);
        }
    }

    private InitialState initializationInternal(double[] params, KalmanSSM model) {
        return initialization;
    }

    private InitialState buildInitialization() {
        if (spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY) && meta.totalDiff() == 0 && !meta.stateRegression()) {
            if (spec.hasApproximateDiffuseVariance() && !spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)) {
                return InitialState.approximateDiffuse(meta.kStates(), spec.approximateDiffuseVariance());
            }
            return InitialState.stationary(meta.kStates());
        }
        if (spec.hasApproximateDiffuseVariance() && !spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)) {
            return InitialState.approximateDiffuse(meta.kStates(), spec.approximateDiffuseVariance());
        }
        if (!spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY)) {
            return spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)
                ? InitialState.diffuse(meta.kStates())
                : InitialState.approximateDiffuse(meta.kStates(), spec.approximateDiffuseVariance());
        }
        InitialState.Builder composite = InitialState.builder(meta.kStates());
        if (meta.totalDiff() > 0) {
            addDiffuseBlock(composite, 0, meta.totalDiff());
        }
        composite.stationary(meta.totalDiff(), meta.totalDiff() + meta.totalOrder());
        if (meta.stateRegression()) {
            addDiffuseBlock(composite, meta.totalDiff() + meta.totalOrder(), meta.kStates());
        }
        return composite.build();
    }

    private void addDiffuseBlock(InitialState.Builder builder, int startInclusive, int endExclusive) {
        if (spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)) {
            builder.diffuse(startInclusive, endExclusive);
        } else {
            builder.approximateDiffuse(startInclusive, endExclusive, approximateDiffuseVariance(startInclusive));
        }
    }

    private double approximateDiffuseVariance(int startInclusive) {
        if (spec.hasApproximateDiffuseVariance()) {
            return spec.approximateDiffuseVariance();
        }
        return meta.stateRegression() && startInclusive >= meta.totalDiff() + meta.totalOrder()
            ? DEFAULT_STATE_REGRESSION_APPROXIMATE_DIFFUSE_VARIANCE
            : spec.approximateDiffuseVariance();
    }

    private double stateRegressionApproximateDiffuseVariance() {
        return spec.hasApproximateDiffuseVariance()
            ? spec.approximateDiffuseVariance()
            : DEFAULT_STATE_REGRESSION_APPROXIMATE_DIFFUSE_VARIANCE;
    }

    private static FilterOptions requireConcentratedLikelihoodSurfaces(FilterOptions options) {
        return options.toBuilder()
            .retain(FilterOptions.Surface.FORECAST_MEAN,
                FilterOptions.Surface.FORECAST_ERROR,
                FilterOptions.Surface.FORECAST_COVARIANCE,
                FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE,
                FilterOptions.Surface.LIKELIHOOD)
            .build();
    }

    FilterOptions realConcentratedLikelihoodOptions(FilterOptions options) {
        return requireConcentratedLikelihoodSurfaces(options)
            .toBuilder()
            .concentratedLikelihood(true)
            .concentratedLikelihoodBurn(likelihoodBurn())
            .build();
    }

    private int likelihoodBurnInternal(double[] params, KalmanSSM model) {
        if (spec.hasLogLikelihoodBurn()) {
            return spec.logLikelihoodBurn();
        }
        if (spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)) {
            return 0;
        }
        int burn = meta.kStates();
        if (spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY)) {
            burn -= meta.totalOrder();
        }
        return Math.max(0, burn);
    }

    int likelihoodBurn() {
        return likelihoodBurnInternal(startParams(), null);
    }

    double concentratedScale(FilterResult filterResult, int burn) {
        if (filterResult.concentratedLikelihood()) {
            return filterResult.scale();
        }
        int start = Math.max(Math.max(0, burn), filterResult.nobsDiffuse);
        int observedCount = 0;
        double scaleSum = 0.0;
        double[] effectiveEndog = meta.endog();
        int nobs = Math.min(filterResult.nobs, effectiveEndog.length);
        for (int t = start; t < nobs; t++) {
            double forecastVariance = filterResult.forecastErrorCov(0, 0, t);
            if (Double.isNaN(effectiveEndog[t]) || !(forecastVariance > 0.0) || !Double.isFinite(forecastVariance)) {
                continue;
            }
            double error = filterResult.forecastError(0, t);
            scaleSum += error * error / forecastVariance;
            observedCount++;
        }
        if (observedCount == 0) {
            return 1.0;
        }
        return Math.max(scaleSum / observedCount, 1e-8);
    }

    void applyConcentratedScale(FilterResult filterResult, double scale) {
        if (!spec.concentrateScale()) {
            return;
        }
        if (filterResult.concentratedLikelihood()) {
            return;
        }
        double[] adjusted = concentratedLogLikelihoodObs(filterResult, filterResult.logLikelihoodObs, scale, 0);
        System.arraycopy(adjusted, 0, filterResult.logLikelihoodObs, 0, adjusted.length);
        scaleSurface(filterResult.predictedStateCov, scale);
        scaleSurface(filterResult.filteredStateCov, scale);
        scaleSurface(filterResult.forecastErrorCov, scale);
    }

    private static double logLikelihoodAfterBurn(FilterResult filterResult, int burn) {
        int start = Math.max(0, burn);
        if (start == 0) {
            return filterResult.logLikelihood();
        }
        double sum = 0.0;
        for (int t = Math.min(start, filterResult.nobs); t < filterResult.nobs; t++) {
            sum += filterResult.logLikelihoodObs[filterResult.logLikelihoodObsOffset(t)];
        }
        return sum;
    }

    int diffuseStateCount() {
        if (!spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)) {
            return 0;
        }
        if (!spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY)) {
            return meta.kStates();
        }
        return meta.totalDiff() + (meta.stateRegression() ? meta.kExog() : 0);
    }

    int effectiveObservationCount() {
        return meta.nobs();
    }

    double[] effectiveEndog() {
        return Arrays.copyOf(meta.endog(), meta.endog().length);
    }

    private static double dot(double[] left, int leftOffset, double[] right, int length) {
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            sum += left[leftOffset + i] * right[i];
        }
        return sum;
    }

    private static void copyParameterBlock(double[] source, double[] target, int offset, int length) {
        if (length > 0) {
            System.arraycopy(source, offset, target, offset, length);
        }
    }

    private void transformStationaryBlock(double[] source,
                                          double[] target,
                                          int offset,
                                          int length,
                                          boolean transform,
                                          double sign,
                                          SARIMAXSupport.StationaryTransformWorkspace workspace) {
        if (length == 0) {
            return;
        }
        if (transform) {
            SARIMAXSupport.constrainStationary(
                sign,
                source,
                offset,
                length,
                target,
                offset,
                workspace);
        } else {
            System.arraycopy(source, offset, target, offset, length);
        }
    }

    private void untransformStationaryBlock(double[] source,
                                            double[] target,
                                            int offset,
                                            int length,
                                            boolean transform,
                                            double sign,
                                            SARIMAXSupport.StationaryTransformWorkspace workspace) {
        if (length == 0) {
            return;
        }
        if (transform) {
            SARIMAXSupport.unconstrainStationary(
                sign,
                source,
                offset,
                length,
                target,
                offset,
                workspace);
        } else {
            System.arraycopy(source, offset, target, offset, length);
        }
    }

    private static int buildReducedPolynomial(double[] params,
                                              int leftParamOffset,
                                              int[] leftLags,
                                              int rightParamOffset,
                                              int[] rightLags,
                                              boolean autoregressive,
                                              RealUpdateWorkspace workspace) {
        int leftLength = lagPolynomialLength(leftLags);
        int rightLength = lagPolynomialLength(rightLags);
        int productLength = leftLength + rightLength - 1;
        workspace.ensure(leftLength, rightLength, productLength);
        buildLagPolynomial(params, leftParamOffset, leftLags, autoregressive, workspace.leftPolynomial, leftLength);
        buildLagPolynomial(params, rightParamOffset, rightLags, autoregressive, workspace.rightPolynomial, rightLength);
        multiply(workspace.leftPolynomial, leftLength, workspace.rightPolynomial, rightLength, workspace.product, productLength);
        return productLength;
    }

    private static int buildReducedComplexPolynomial(double[] leftParams,
                                                     int[] leftLags,
                                                     double[] rightParams,
                                                     int[] rightLags,
                                                     boolean autoregressive,
                                                     ComplexUpdateWorkspace workspace) {
        int leftLength = lagPolynomialLength(leftLags);
        int rightLength = lagPolynomialLength(rightLags);
        int productLength = leftLength + rightLength - 1;
        workspace.ensurePolynomialCapacity(leftLength, rightLength, productLength);
        buildLagPolynomialComplex(leftParams, leftLags, autoregressive, workspace.leftPolynomial, leftLength);
        buildLagPolynomialComplex(rightParams, rightLags, autoregressive, workspace.rightPolynomial, rightLength);
        multiplyComplex(workspace.leftPolynomial, leftLength, workspace.rightPolynomial, rightLength, workspace.product, productLength);
        return productLength;
    }

    private static void buildLagPolynomial(double[] params,
                                           int paramOffset,
                                           int[] lags,
                                           boolean autoregressive,
                                           double[] polynomial,
                                           int length) {
        Arrays.fill(polynomial, 0, length, 0.0);
        polynomial[0] = 1.0;
        for (int i = 0; i < lags.length; i++) {
            polynomial[lags[i]] = autoregressive ? -params[paramOffset + i] : params[paramOffset + i];
        }
    }

    private static void buildLagPolynomialComplex(double[] params,
                                                  int[] lags,
                                                  boolean autoregressive,
                                                  double[] polynomial,
                                                  int length) {
        Arrays.fill(polynomial, 0, length * 2, 0.0);
        polynomial[0] = 1.0;
        for (int i = 0; i < lags.length; i++) {
            int off = lags[i] * 2;
            int paramOffset = i * 2;
            polynomial[off] = autoregressive ? -params[paramOffset] : params[paramOffset];
            polynomial[off + 1] = autoregressive ? -params[paramOffset + 1] : params[paramOffset + 1];
        }
    }

    private static void multiply(double[] left,
                                 int leftLength,
                                 double[] right,
                                 int rightLength,
                                 double[] product,
                                 int productLength) {
        Arrays.fill(product, 0, productLength, 0.0);
        for (int i = 0; i < leftLength; i++) {
            for (int j = 0; j < rightLength; j++) {
                product[i + j] += left[i] * right[j];
            }
        }
    }

    private static void multiplyComplex(double[] left,
                                        int leftLength,
                                        double[] right,
                                        int rightLength,
                                        double[] product,
                                        int productLength) {
        Arrays.fill(product, 0, productLength * 2, 0.0);
        for (int i = 0; i < leftLength; i++) {
            double leftRe = left[i * 2];
            double leftIm = left[i * 2 + 1];
            for (int j = 0; j < rightLength; j++) {
                double rightRe = right[j * 2];
                double rightIm = right[j * 2 + 1];
                int off = (i + j) * 2;
                product[off] += leftRe * rightRe - leftIm * rightIm;
                product[off + 1] += leftRe * rightIm + leftIm * rightRe;
            }
        }
    }

    private static int lagPolynomialLength(int[] lags) {
        return lags.length == 0 ? 1 : lags[lags.length - 1] + 1;
    }

    private double[] concentratedLogLikelihoodObs(FilterResult filterResult,
                                                  double[] logLikelihoodObs,
                                                  double scale,
                                                  int burn) {
        double[] adjusted = Arrays.copyOf(logLikelihoodObs, logLikelihoodObs.length);
        double[] effectiveEndog = meta.endog();
        double logScale = Math.log(scale);
        int nobs = Math.min(adjusted.length, effectiveEndog.length);
        int start = Math.max(Math.max(0, burn), filterResult.nobsDiffuse);
        for (int t = start; t < nobs; t++) {
            double forecastVariance = filterResult.forecastErrorCov(0, 0, t);
            if (Double.isNaN(effectiveEndog[t]) || !(forecastVariance > 0.0) || !Double.isFinite(forecastVariance)) {
                continue;
            }
            double error = filterResult.forecastError(0, t);
            double scaleObs = error * error / forecastVariance;
            adjusted[t] += -0.5 * (logScale + scaleObs / scale - scaleObs);
        }
        return adjusted;
    }

    private double concentratedScale(ZFilterResult filterResult, int burn) {
        int start = Math.max(Math.max(0, burn), filterResult.nobsDiffuse);
        int observedCount = 0;
        double scaleSum = 0.0;
        double[] effectiveEndog = meta.endog();
        int nobs = Math.min(filterResult.nobs, effectiveEndog.length);
        for (int t = start; t < nobs; t++) {
            int forecastCovOffset = filterResult.forecastErrorCovOffset(t);
            double forecastVariance = filterResult.forecastErrorCov[forecastCovOffset];
            if (Double.isNaN(effectiveEndog[t]) || !(forecastVariance > 0.0) || !Double.isFinite(forecastVariance)) {
                continue;
            }
            int forecastErrorOffset = filterResult.forecastErrorOffset(t);
            double errorRe = filterResult.forecastError[forecastErrorOffset];
            double errorIm = filterResult.forecastError[forecastErrorOffset + 1];
            scaleSum += (errorRe * errorRe + errorIm * errorIm) / forecastVariance;
            observedCount++;
        }
        if (observedCount == 0) {
            return 1.0;
        }
        return Math.max(scaleSum / observedCount, 1e-8);
    }

    private double[] concentratedLogLikelihoodObs(ZFilterResult filterResult,
                                                  double[] logLikelihoodObs,
                                                  double scale,
                                                  int burn) {
        double[] adjusted = Arrays.copyOf(logLikelihoodObs, logLikelihoodObs.length);
        double[] effectiveEndog = meta.endog();
        double logScale = Math.log(scale);
        int nobs = Math.min(adjusted.length, effectiveEndog.length);
        int start = Math.max(Math.max(0, burn), filterResult.nobsDiffuse);
        for (int t = start; t < nobs; t++) {
            int forecastCovOffset = filterResult.forecastErrorCovOffset(t);
            double forecastVariance = filterResult.forecastErrorCov[forecastCovOffset];
            if (Double.isNaN(effectiveEndog[t]) || !(forecastVariance > 0.0) || !Double.isFinite(forecastVariance)) {
                continue;
            }
            int forecastErrorOffset = filterResult.forecastErrorOffset(t);
            double errorRe = filterResult.forecastError[forecastErrorOffset];
            double errorIm = filterResult.forecastError[forecastErrorOffset + 1];
            double scaleObs = (errorRe * errorRe + errorIm * errorIm) / forecastVariance;
            adjusted[t] += -0.5 * (logScale + scaleObs / scale - scaleObs);
        }
        return adjusted;
    }

    private double adjustedLogLikelihoodInternal(ZFilterResult filterResult, int burn) {
        if (!spec.concentrateScale()) {
            int start = Math.max(0, burn);
            if (start == 0) {
                return filterResult.logLikelihood();
            }
            double sum = 0.0;
            for (int t = Math.min(start, filterResult.nobs); t < filterResult.nobs; t++) {
                sum += filterResult.logLikelihoodObs[filterResult.logLikelihoodObsOffset(t)];
            }
            return sum;
        }
        double scale = concentratedScale(filterResult, burn);
        double logScale = Math.log(scale);
        double[] effectiveEndog = meta.endog();
        double sum = 0.0;
        int nobs = Math.min(filterResult.nobs, effectiveEndog.length);
        int concentratedStart = Math.max(Math.max(0, burn), filterResult.nobsDiffuse);
        for (int t = Math.max(0, burn); t < nobs; t++) {
            double value = filterResult.logLikelihoodObs[filterResult.logLikelihoodObsOffset(t)];
            int forecastCovOffset = filterResult.forecastErrorCovOffset(t);
            double forecastVariance = filterResult.forecastErrorCov[forecastCovOffset];
            if (t >= concentratedStart && !Double.isNaN(effectiveEndog[t]) && forecastVariance > 0.0 && Double.isFinite(forecastVariance)) {
                int forecastErrorOffset = filterResult.forecastErrorOffset(t);
                double errorRe = filterResult.forecastError[forecastErrorOffset];
                double errorIm = filterResult.forecastError[forecastErrorOffset + 1];
                double scaleObs = (errorRe * errorRe + errorIm * errorIm) / forecastVariance;
                value += -0.5 * (logScale + scaleObs / scale - scaleObs);
            }
            sum += value;
        }
        return sum;
    }

    private static KalmanSSM toComplexStateSpace(KalmanSSM source) {
        KalmanSSM expanded = KalmanSSM.copyOf(source);
        int secondIndex = expanded.observationCount() > 1 ? 1 : 0;
        KalmanSSM.Builder builder = KalmanSSM.complexBuilder(
            expanded.observationDimension(),
            expanded.stateCount(),
            expanded.stateDisturbanceCount(),
            expanded.observationCount())
            .design(embedRealSurface(expanded.designData()), isTimeVarying(expanded.designOffset(0), expanded.designOffset(secondIndex)))
            .obsIntercept(embedRealSurface(expanded.obsInterceptData()), isTimeVarying(expanded.obsInterceptOffset(0), expanded.obsInterceptOffset(secondIndex)))
            .obsCov(embedRealSurface(expanded.obsCovData()), isTimeVarying(expanded.obsCovOffset(0), expanded.obsCovOffset(secondIndex)))
            .transition(embedRealSurface(expanded.transitionData()), isTimeVarying(expanded.transitionOffset(0), expanded.transitionOffset(secondIndex)))
            .stateIntercept(embedRealSurface(expanded.stateInterceptData()), isTimeVarying(expanded.stateInterceptOffset(0), expanded.stateInterceptOffset(secondIndex)))
            .selection(embedRealSurface(expanded.selectionData()), isTimeVarying(expanded.selectionOffset(0), expanded.selectionOffset(secondIndex)))
            .stateCovariance(embedRealSurface(expanded.stateCovarianceData()), isTimeVarying(expanded.stateCovarianceOffset(0), expanded.stateCovarianceOffset(secondIndex)))
            .endog(embedRealSurface(expanded.endogData()), expanded.endogOffset(0) * 2);
        boolean[] missing = expanded.getMissing();
        int[] nmissing = expanded.getNmissing();
        if (missing != null || nmissing != null) {
            builder.missing(
                missing == null ? null : Arrays.copyOf(missing, missing.length),
                nmissing == null ? null : Arrays.copyOf(nmissing, nmissing.length));
        } else {
            builder.allObserved();
        }
        return builder.build();
    }

    private static double[] embedRealSurface(double[] source) {
        if (source == null) {
            return null;
        }
        double[] target = new double[source.length * 2];
        embedRealSurface(source, target);
        return target;
    }

    private static void embedRealSurface(double[] source, double[] target) {
        if (source == null || target == null) {
            if (source != null || target != null) {
                throw new IllegalStateException("source and target surfaces must agree on nullability");
            }
            return;
        }
        if (target.length != source.length * 2) {
            throw new IllegalArgumentException("complex target length does not match embedded real surface");
        }
        for (int index = 0; index < source.length; index++) {
            int complexIndex = index * 2;
            target[complexIndex] = source[index];
            target[complexIndex + 1] = 0.0;
        }
    }

    private static boolean isTimeVarying(int offset0, int offset1) {
        return offset0 != offset1;
    }

    private static double[] complexSlice(double[] params, int perturbIndex, double perturbStep, int startInclusive, int endExclusive) {
        double[] complex = new double[Math.max(0, endExclusive - startInclusive) * 2];
        for (int index = startInclusive; index < endExclusive; index++) {
            int offset = (index - startInclusive) * 2;
            complex[offset] = params[index];
            complex[offset + 1] = imaginaryStep(perturbIndex, index, perturbStep);
        }
        return complex;
    }

    private static double[] complexSlice(double[] target,
                                         double[] params,
                                         int perturbIndex,
                                         double perturbStep,
                                         int startInclusive,
                                         int endExclusive) {
        int length = Math.max(0, endExclusive - startInclusive) * 2;
        if (length == 0) {
            return EMPTY_COMPLEX_ARRAY;
        }
        if (target.length < length) {
            target = new double[length];
        }
        for (int index = startInclusive; index < endExclusive; index++) {
            int offset = (index - startInclusive) * 2;
            target[offset] = params[index];
            target[offset + 1] = imaginaryStep(perturbIndex, index, perturbStep);
        }
        return target;
    }

    private static double[] sumScoreObs(double[] scoreObs, int paramCount) {
        double[] score = new double[paramCount];
        if (paramCount == 0) {
            return score;
        }
        for (int offset = 0; offset < scoreObs.length; offset += paramCount) {
            for (int index = 0; index < paramCount; index++) {
                score[index] += scoreObs[offset + index];
            }
        }
        return score;
    }

    private static double harveyDerivativeStepSize(double value) {
        return NumericalGradient.SQRT_EPSILON * Math.max(Math.abs(value), 0.1);
    }

    private static double approximateHessianStepSize(double value) {
        return NumericalGradient.CBRT_EPSILON * Math.max(Math.abs(value), 0.1);
    }

    private static double complexStepSize(double value) {
        return 1e-20 * Math.max(1.0, Math.abs(value));
    }

    private static void complexMatrixMultiplyInto(double[] left,
                                                  int leftOffset,
                                                  int leftLeadingDimension,
                                                  double[] right,
                                                  int rightOffset,
                                                  int rightLeadingDimension,
                                                  int rows,
                                                  int inner,
                                                  int cols,
                                                  double[] target,
                                                  int targetOffset,
                                                  int targetLeadingDimension,
                                                  double betaRe,
                                                  double betaIm) {
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, rows, cols, inner,
            1.0, 0.0, left, leftOffset, leftLeadingDimension,
            right, rightOffset, rightLeadingDimension,
            betaRe, betaIm, target, targetOffset, targetLeadingDimension);
    }

    private static boolean invertForecastCovariance(double[] matrix, int offset, int dimension, double[] inverse) {
        if (dimension == 1) {
            double value = matrix[offset];
            if (!(value > 0.0) || !Double.isFinite(value)) {
                return false;
            }
            inverse[0] = 1.0 / value;
            return true;
        }

        double[] copy = new double[dimension * dimension];
        for (int row = 0; row < dimension; row++) {
            System.arraycopy(matrix, offset + row * dimension, copy, row * dimension, dimension);
        }
        try {
            double[] solved = LU.decompose(copy, dimension).inverse(new double[dimension * dimension]);
            System.arraycopy(solved, 0, inverse, 0, solved.length);
            return allFinite(inverse);
        } catch (ArithmeticException ignored) {
            return false;
        }
    }

    private static double traceProduct(double[] left,
                                       int leftOffset,
                                       double[] right,
                                       int rightOffset,
                                       int dimension) {
        double trace = 0.0;
        for (int row = 0; row < dimension; row++) {
            for (int col = 0; col < dimension; col++) {
                trace += left[leftOffset + row * dimension + col] * right[rightOffset + col * dimension + row];
            }
        }
        return trace;
    }

    private static double[] lagPolynomialComplex(double[] params, int[] lags, boolean autoregressive) {
        int maxLag = 0;
        for (int lag : lags) {
            maxLag = Math.max(maxLag, lag);
        }
        double[] polynomial = new double[(maxLag + 1) * 2];
        polynomial[0] = 1.0;
        for (int i = 0; i < lags.length; i++) {
            int off = lags[i] * 2;
            polynomial[off] = autoregressive ? -params[i * 2] : params[i * 2];
            polynomial[off + 1] = autoregressive ? -params[i * 2 + 1] : params[i * 2 + 1];
        }
        return polynomial;
    }

    private static double[] multiplyComplex(double[] left, double[] right) {
        double[] result = new double[left.length + right.length - 2];
        int leftLength = left.length / 2;
        int rightLength = right.length / 2;
        for (int i = 0; i < leftLength; i++) {
            double leftRe = left[i * 2];
            double leftIm = left[i * 2 + 1];
            for (int j = 0; j < rightLength; j++) {
                double rightRe = right[j * 2];
                double rightIm = right[j * 2 + 1];
                int off = (i + j) * 2;
                result[off] += leftRe * rightRe - leftIm * rightIm;
                result[off + 1] += leftRe * rightIm + leftIm * rightRe;
            }
        }
        return result;
    }

    private static double[] dotComplexReal(double[] left, double[] right) {
        double sumRe = 0.0;
        double sumIm = 0.0;
        for (int i = 0; i < right.length; i++) {
            sumRe += left[i * 2] * right[i];
            sumIm += left[i * 2 + 1] * right[i];
        }
        return complexScalar(sumRe, sumIm);
    }

    private static void dotComplexReal(double[] left, double[] right, double[] target) {
        double sumRe = 0.0;
        double sumIm = 0.0;
        for (int i = 0; i < right.length; i++) {
            sumRe += left[i * 2] * right[i];
            sumIm += left[i * 2 + 1] * right[i];
        }
        target[0] = sumRe;
        target[1] = sumIm;
    }

    private static double[] sumComplex(double[] values) {
        double sumRe = 0.0;
        double sumIm = 0.0;
        for (int i = 0; i < values.length; i += 2) {
            sumRe += values[i];
            sumIm += values[i + 1];
        }
        return complexScalar(sumRe, sumIm);
    }

    private static void sumComplex(double[] values, int complexLength, double[] target) {
        double sumRe = 0.0;
        double sumIm = 0.0;
        for (int i = 0; i < complexLength; i++) {
            int off = i * 2;
            sumRe += values[off];
            sumIm += values[off + 1];
        }
        target[0] = sumRe;
        target[1] = sumIm;
    }

    private static double[] divideComplex(double leftRe, double leftIm, double rightRe, double rightIm) {
        double denom = rightRe * rightRe + rightIm * rightIm;
        if (!(denom > 0.0)) {
            return complexScalar(0.0, 0.0);
        }
        return complexScalar(
            (leftRe * rightRe + leftIm * rightIm) / denom,
            (leftIm * rightRe - leftRe * rightIm) / denom);
    }

    private static void divideComplex(double leftRe, double leftIm, double rightRe, double rightIm, double[] target) {
        double denom = rightRe * rightRe + rightIm * rightIm;
        if (!(denom > 0.0)) {
            target[0] = 0.0;
            target[1] = 0.0;
            return;
        }
        target[0] = (leftRe * rightRe + leftIm * rightIm) / denom;
        target[1] = (leftIm * rightRe - leftRe * rightIm) / denom;
    }

    private static double[] positiveContinuation(double re, double im) {
        if (im == 0.0 && re < 1e-8) {
            return complexScalar(1e-8, 0.0);
        }
        return complexScalar(re, im);
    }

    private static void positiveContinuation(double re, double im, double[] target) {
        if (im == 0.0 && re < 1e-8) {
            target[0] = 1e-8;
            target[1] = 0.0;
            return;
        }
        target[0] = re;
        target[1] = im;
    }

    private static double[] complexScalar(double re, double im) {
        return new double[]{re, im};
    }

    private static double imaginaryStep(int perturbIndex, int parameterIndex, double perturbStep) {
        return perturbIndex == parameterIndex ? perturbStep : 0.0;
    }

    private static void writeComplexScalar(double[] values, int offset, double re, double im) {
        values[offset] = re;
        values[offset + 1] = im;
    }

    private static void addComplexScalar(double[] values, int offset, double re, double im) {
        values[offset] += re;
        values[offset + 1] += im;
    }

    private static int complexVectorOffset(int base, int index) {
        return base + index * 2;
    }

    private static int complexMatrixOffset(int base, int leadingDimension, int row, int col) {
        return base + row * 2 * leadingDimension + col * 2;
    }

    private static boolean isNearZero(double re, double im) {
        return re * re + im * im < 1e-24;
    }

    private static boolean allFinite(double[] values) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        return true;
    }

    private static boolean allFinite(double[] values, int offset, int length) {
        for (int index = 0; index < length; index++) {
            if (!Double.isFinite(values[offset + index])) {
                return false;
            }
        }
        return true;
    }

    private record ParameterLayout(int trendOffset,
                                   int trendLength,
                                   int exogOffset,
                                   int exogLength,
                                   int arOffset,
                                   int arLength,
                                   int maOffset,
                                   int maLength,
                                   int seasonalArOffset,
                                   int seasonalArLength,
                                   int seasonalMaOffset,
                                   int seasonalMaLength,
                                   int timeVaryingRegressionOffset,
                                   int timeVaryingRegressionLength,
                                   int measurementErrorOffset,
                                   int scaleOffset,
                                   int parameterCount,
                                   int[] arLags,
                                   int[] maLags,
                                   int[] seasonalArLags,
                                   int[] seasonalMaLags) {

        static ParameterLayout of(SARIMAXSupport.Meta meta) {
            SARIMAXSpec spec = meta.spec();
            int offset = 0;
            int trendOffset = offset;
            int trendLength = meta.kTrend();
            offset += trendLength;

            int exogOffset = offset;
            int exogLength = meta.stateRegression() ? 0 : meta.kExog();
            offset += exogLength;

            int arOffset = offset;
            int arLength = spec.autoregressiveParameterCount();
            offset += arLength;

            int maOffset = offset;
            int maLength = spec.movingAverageParameterCount();
            offset += maLength;

            int seasonalArOffset = offset;
            int seasonalArLength = spec.seasonalAutoregressiveParameterCount();
            offset += seasonalArLength;

            int seasonalMaOffset = offset;
            int seasonalMaLength = spec.seasonalMovingAverageParameterCount();
            offset += seasonalMaLength;

            int timeVaryingRegressionOffset = offset;
            int timeVaryingRegressionLength = meta.timeVaryingRegression() ? meta.kExog() : 0;
            offset += timeVaryingRegressionLength;

            int measurementErrorOffset = spec.measurementError() ? offset++ : -1;
            int scaleOffset = spec.concentrateScale() ? -1 : offset++;

            return new ParameterLayout(
                trendOffset,
                trendLength,
                exogOffset,
                exogLength,
                arOffset,
                arLength,
                maOffset,
                maLength,
                seasonalArOffset,
                seasonalArLength,
                seasonalMaOffset,
                seasonalMaLength,
                timeVaryingRegressionOffset,
                timeVaryingRegressionLength,
                measurementErrorOffset,
                scaleOffset,
                offset,
                spec.autoregressiveLags(),
                spec.movingAverageLags(),
                spec.seasonalAutoregressiveLags(),
                spec.seasonalMovingAverageLags());
        }

        int trendEnd() {
            return trendOffset + trendLength;
        }

        int exogEnd() {
            return exogOffset + exogLength;
        }

        int arEnd() {
            return arOffset + arLength;
        }

        int maEnd() {
            return maOffset + maLength;
        }

        int seasonalArEnd() {
            return seasonalArOffset + seasonalArLength;
        }

        int seasonalMaEnd() {
            return seasonalMaOffset + seasonalMaLength;
        }

        int timeVaryingRegressionEnd() {
            return timeVaryingRegressionOffset + timeVaryingRegressionLength;
        }

        boolean hasMeasurementError() {
            return measurementErrorOffset >= 0;
        }

        boolean hasScale() {
            return scaleOffset >= 0;
        }
    }

    private static final class RealUpdateWorkspace {
        private final DoubleArena leftPolynomialArena = new DoubleArena();
        private final DoubleArena rightPolynomialArena = new DoubleArena();
        private final DoubleArena productArena = new DoubleArena();
        private double[] leftPolynomial = UNIT_POLYNOMIAL;
        private double[] rightPolynomial = UNIT_POLYNOMIAL;
        private double[] product = UNIT_POLYNOMIAL;

        private void ensure(int leftLength, int rightLength, int productLength) {
            leftPolynomial = leftPolynomialArena.borrow(leftLength);
            rightPolynomial = rightPolynomialArena.borrow(rightLength);
            product = productArena.borrow(productLength);
        }

        private void release() {
            leftPolynomialArena.release();
            rightPolynomialArena.release();
            productArena.release();
            leftPolynomial = UNIT_POLYNOMIAL;
            rightPolynomial = UNIT_POLYNOMIAL;
            product = UNIT_POLYNOMIAL;
        }
    }

    private static final class ComplexUpdateWorkspace {
        private double[] trendParams = EMPTY_COMPLEX_ARRAY;
        private double[] exogParams = EMPTY_COMPLEX_ARRAY;
        private double[] arParams = EMPTY_COMPLEX_ARRAY;
        private double[] maParams = EMPTY_COMPLEX_ARRAY;
        private double[] seasonalArParams = EMPTY_COMPLEX_ARRAY;
        private double[] seasonalMaParams = EMPTY_COMPLEX_ARRAY;
        private double[] exogVariances = EMPTY_COMPLEX_ARRAY;
        private final DoubleArena leftPolynomialArena = new DoubleArena();
        private final DoubleArena rightPolynomialArena = new DoubleArena();
        private final DoubleArena productArena = new DoubleArena();
        private double[] leftPolynomial = UNIT_COMPLEX_POLYNOMIAL;
        private double[] rightPolynomial = UNIT_COMPLEX_POLYNOMIAL;
        private double[] product = UNIT_COMPLEX_POLYNOMIAL;
        private final double[] scalar0 = new double[2];
        private final double[] scalar1 = new double[2];
        private final double[] scalar2 = new double[2];

        private void ensurePolynomialCapacity(int leftLength, int rightLength, int productLength) {
            leftPolynomial = leftPolynomialArena.borrow(leftLength * 2);
            rightPolynomial = rightPolynomialArena.borrow(rightLength * 2);
            product = productArena.borrow(productLength * 2);
        }

        private void release() {
            trendParams = EMPTY_COMPLEX_ARRAY;
            exogParams = EMPTY_COMPLEX_ARRAY;
            arParams = EMPTY_COMPLEX_ARRAY;
            maParams = EMPTY_COMPLEX_ARRAY;
            seasonalArParams = EMPTY_COMPLEX_ARRAY;
            seasonalMaParams = EMPTY_COMPLEX_ARRAY;
            exogVariances = EMPTY_COMPLEX_ARRAY;
            leftPolynomialArena.release();
            rightPolynomialArena.release();
            productArena.release();
            leftPolynomial = UNIT_COMPLEX_POLYNOMIAL;
            rightPolynomial = UNIT_COMPLEX_POLYNOMIAL;
            product = UNIT_COMPLEX_POLYNOMIAL;
        }
    }

    private static final class AnalyticComplexWorkspace {
        private static final int[] EMPTY_INT_ARRAY = new int[0];

        private final DoubleArena state0Arena = new DoubleArena();
        private final DoubleArena state1Arena = new DoubleArena();
        private final DoubleArena covariance0Arena = new DoubleArena();
        private final DoubleArena covariance1Arena = new DoubleArena();
        private final DoubleArena filteredStateArena = new DoubleArena();
        private final DoubleArena filteredCovarianceArena = new DoubleArena();
        private final DoubleArena gainNumeratorArena = new DoubleArena();
        private final DoubleArena transitionTimesCovarianceArena = new DoubleArena();
        private final DoubleArena selectedCovarianceArena = new DoubleArena();
        private final DoubleArena logLikelihoodObsArena = new DoubleArena();
        private final DoubleArena forecastErrorArena = new DoubleArena();
        private final DoubleArena forecastVarianceArena = new DoubleArena();
        private final DoubleArena stationaryTransitionArena = new DoubleArena();
        private final DoubleArena stationaryStateInterceptArena = new DoubleArena();
        private final DoubleArena stationarySelectionArena = new DoubleArena();
        private final DoubleArena stationaryStateCovarianceArena = new DoubleArena();
        private final DoubleArena stationaryMeanSystemArena = new DoubleArena();
        private final DoubleArena stationaryStateArena = new DoubleArena();
        private final DoubleArena stationaryCovarianceRhsArena = new DoubleArena();
        private final DoubleArena stationarySelectedCovarianceArena = new DoubleArena();
        private final DoubleArena stationaryCovarianceSystemArena = new DoubleArena();
        private final DoubleArena stationaryCovarianceVectorArena = new DoubleArena();
        private final DoubleArena realifiedMatrixArena = new DoubleArena();
        private final DoubleArena realifiedVectorArena = new DoubleArena();
        private final IntArena realifiedPivotArena = new IntArena();

        private double[] state0 = EMPTY_COMPLEX_ARRAY;
        private double[] state1 = EMPTY_COMPLEX_ARRAY;
        private double[] covariance0 = EMPTY_COMPLEX_ARRAY;
        private double[] covariance1 = EMPTY_COMPLEX_ARRAY;
        private double[] filteredState = EMPTY_COMPLEX_ARRAY;
        private double[] filteredCovariance = EMPTY_COMPLEX_ARRAY;
        private double[] gainNumerator = EMPTY_COMPLEX_ARRAY;
        private double[] transitionTimesCovariance = EMPTY_COMPLEX_ARRAY;
        private double[] selectedCovariance = EMPTY_COMPLEX_ARRAY;
        private double[] logLikelihoodObs = EMPTY_COMPLEX_ARRAY;
        private double[] forecastError = EMPTY_COMPLEX_ARRAY;
        private double[] forecastVariance = EMPTY_COMPLEX_ARRAY;
        private double[] stationaryTransition = EMPTY_COMPLEX_ARRAY;
        private double[] stationaryStateIntercept = EMPTY_COMPLEX_ARRAY;
        private double[] stationarySelection = EMPTY_COMPLEX_ARRAY;
        private double[] stationaryStateCovariance = EMPTY_COMPLEX_ARRAY;
        private double[] stationaryMeanSystem = EMPTY_COMPLEX_ARRAY;
        private double[] stationaryState = EMPTY_COMPLEX_ARRAY;
        private double[] stationaryCovarianceRhs = EMPTY_COMPLEX_ARRAY;
        private double[] stationarySelectedCovariance = EMPTY_COMPLEX_ARRAY;
        private double[] stationaryCovarianceSystem = EMPTY_COMPLEX_ARRAY;
        private double[] stationaryCovarianceVector = EMPTY_COMPLEX_ARRAY;
        private double[] realifiedMatrix = EMPTY_COMPLEX_ARRAY;
        private double[] realifiedVector = EMPTY_COMPLEX_ARRAY;
        private int[] realifiedPivots = EMPTY_INT_ARRAY;
        private final double[] dot = new double[2];
        private final double[] logLikelihoodSum = new double[2];

        private void ensureFilter(int stateCount, int kPosdef) {
            int stateLength = stateCount * 2;
            int covarianceLength = stateCount * stateCount * 2;
            state0 = borrow(state0Arena, stateLength);
            state1 = borrow(state1Arena, stateLength);
            covariance0 = borrow(covariance0Arena, covarianceLength);
            covariance1 = borrow(covariance1Arena, covarianceLength);
            filteredState = borrow(filteredStateArena, stateLength);
            filteredCovariance = borrow(filteredCovarianceArena, covarianceLength);
            gainNumerator = borrow(gainNumeratorArena, stateLength);
            transitionTimesCovariance = borrow(transitionTimesCovarianceArena, covarianceLength);
            selectedCovariance = kPosdef > 0
                ? borrow(selectedCovarianceArena, stateCount * kPosdef * 2)
                : EMPTY_COMPLEX_ARRAY;
        }

        private void ensureLogLikelihoodObs(int nobs) {
            logLikelihoodObs = borrow(logLikelihoodObsArena, nobs * 2);
        }

        private void ensureForecastSurface(int nobs) {
            forecastError = borrow(forecastErrorArena, nobs * 2);
            forecastVariance = borrow(forecastVarianceArena, nobs * 2);
        }

        private void ensureStationary(int dimension, int kPosdef) {
            int squareLength = dimension * dimension * 2;
            int systemDimension = dimension * dimension;
            stationaryTransition = borrow(stationaryTransitionArena, squareLength);
            stationaryStateIntercept = borrow(stationaryStateInterceptArena, dimension * 2);
            stationarySelection = kPosdef > 0
                ? borrow(stationarySelectionArena, dimension * kPosdef * 2)
                : EMPTY_COMPLEX_ARRAY;
            stationaryStateCovariance = kPosdef > 0
                ? borrow(stationaryStateCovarianceArena, kPosdef * kPosdef * 2)
                : EMPTY_COMPLEX_ARRAY;
            stationaryMeanSystem = borrow(stationaryMeanSystemArena, squareLength);
            stationaryState = borrow(stationaryStateArena, dimension * 2);
            stationaryCovarianceRhs = borrow(stationaryCovarianceRhsArena, squareLength);
            stationarySelectedCovariance = kPosdef > 0
                ? borrow(stationarySelectedCovarianceArena, dimension * kPosdef * 2)
                : EMPTY_COMPLEX_ARRAY;
            stationaryCovarianceSystem = borrow(stationaryCovarianceSystemArena, systemDimension * systemDimension * 2);
            stationaryCovarianceVector = borrow(stationaryCovarianceVectorArena, systemDimension * 2);
        }

        private void ensureRealifiedCapacity(int realDimension) {
            realifiedMatrix = borrow(realifiedMatrixArena, realDimension * realDimension);
            realifiedVector = borrow(realifiedVectorArena, realDimension);
            realifiedPivots = realDimension == 0 ? EMPTY_INT_ARRAY : realifiedPivotArena.borrow(realDimension);
        }

        private void release() {
            state0Arena.release();
            state1Arena.release();
            covariance0Arena.release();
            covariance1Arena.release();
            filteredStateArena.release();
            filteredCovarianceArena.release();
            gainNumeratorArena.release();
            transitionTimesCovarianceArena.release();
            selectedCovarianceArena.release();
            logLikelihoodObsArena.release();
            forecastErrorArena.release();
            forecastVarianceArena.release();
            stationaryTransitionArena.release();
            stationaryStateInterceptArena.release();
            stationarySelectionArena.release();
            stationaryStateCovarianceArena.release();
            stationaryMeanSystemArena.release();
            stationaryStateArena.release();
            stationaryCovarianceRhsArena.release();
            stationarySelectedCovarianceArena.release();
            stationaryCovarianceSystemArena.release();
            stationaryCovarianceVectorArena.release();
            realifiedMatrixArena.release();
            realifiedVectorArena.release();
            realifiedPivotArena.release();
            state0 = EMPTY_COMPLEX_ARRAY;
            state1 = EMPTY_COMPLEX_ARRAY;
            covariance0 = EMPTY_COMPLEX_ARRAY;
            covariance1 = EMPTY_COMPLEX_ARRAY;
            filteredState = EMPTY_COMPLEX_ARRAY;
            filteredCovariance = EMPTY_COMPLEX_ARRAY;
            gainNumerator = EMPTY_COMPLEX_ARRAY;
            transitionTimesCovariance = EMPTY_COMPLEX_ARRAY;
            selectedCovariance = EMPTY_COMPLEX_ARRAY;
            logLikelihoodObs = EMPTY_COMPLEX_ARRAY;
            forecastError = EMPTY_COMPLEX_ARRAY;
            forecastVariance = EMPTY_COMPLEX_ARRAY;
            stationaryTransition = EMPTY_COMPLEX_ARRAY;
            stationaryStateIntercept = EMPTY_COMPLEX_ARRAY;
            stationarySelection = EMPTY_COMPLEX_ARRAY;
            stationaryStateCovariance = EMPTY_COMPLEX_ARRAY;
            stationaryMeanSystem = EMPTY_COMPLEX_ARRAY;
            stationaryState = EMPTY_COMPLEX_ARRAY;
            stationaryCovarianceRhs = EMPTY_COMPLEX_ARRAY;
            stationarySelectedCovariance = EMPTY_COMPLEX_ARRAY;
            stationaryCovarianceSystem = EMPTY_COMPLEX_ARRAY;
            stationaryCovarianceVector = EMPTY_COMPLEX_ARRAY;
            realifiedMatrix = EMPTY_COMPLEX_ARRAY;
            realifiedVector = EMPTY_COMPLEX_ARRAY;
            realifiedPivots = EMPTY_INT_ARRAY;
        }

        private static double[] borrow(DoubleArena arena, int length) {
            return length == 0 ? EMPTY_COMPLEX_ARRAY : arena.borrow(length);
        }
    }

    private static final class PhaseWorkspace {
        private final DoubleArena stepsArena = new DoubleArena();
        private final DoubleArena plusParamsArena = new DoubleArena();
        private final DoubleArena minusParamsArena = new DoubleArena();
        private double[] steps = EMPTY_COMPLEX_ARRAY;
        private double[] plusParams = EMPTY_COMPLEX_ARRAY;
        private double[] minusParams = EMPTY_COMPLEX_ARRAY;

        private void ensureParamCount(int paramCount) {
            steps = stepsArena.borrow(paramCount);
            plusParams = plusParamsArena.borrow(paramCount);
            minusParams = minusParamsArena.borrow(paramCount);
        }

        private void release() {
            stepsArena.release();
            plusParamsArena.release();
            minusParamsArena.release();
            steps = EMPTY_COMPLEX_ARRAY;
            plusParams = EMPTY_COMPLEX_ARRAY;
            minusParams = EMPTY_COMPLEX_ARRAY;
        }
    }

    private record ForecastDerivativeSurfaces(double[] forecastErrorPartials,
                                              double[] forecastErrorCovPartials) {
        private int forecastErrorOffset(int t, int paramIndex, int kEndog, int paramCount) {
            return (t * paramCount + paramIndex) * kEndog;
        }

        private int forecastErrorCovOffset(int t, int paramIndex, int kEndog, int paramCount) {
            return (t * paramCount + paramIndex) * kEndog * kEndog;
        }
    }

    private static void scaleSurface(double[] values, double scale) {
        if (values == null) {
            return;
        }
        for (int i = 0; i < values.length; i++) {
            values[i] *= scale;
        }
    }

    private final class RealDelegate extends RealMLE {

        private final RealUpdateWorkspace updateWorkspace = new RealUpdateWorkspace();

        private RealDelegate() {
            super(meta.templateModel(), meta.startParams(), meta.parameterNames());
        }

        private void trimDelegateWorkspace() {
            trimWorkspaces();
            updateWorkspace.release();
        }

        @Override
        public double[] transformParams(double[] unconstrainedParams) {
            return transformParamsInternal(unconstrainedParams);
        }

        @Override
        public double[] untransformParams(double[] params) {
            return untransformParamsInternal(params);
        }

        @Override
        protected FilterOptions likelihoodOptions() {
            return spec.concentrateScale() ? realConcentratedLikelihoodOptions(CONCENTRATED_LIKELIHOOD_OPTIONS) : super.likelihoodOptions();
        }

        @Override
        protected FilterOptions likelihoodOptions(FilterOptions options) {
            FilterOptions resolved = super.likelihoodOptions(options);
            return spec.concentrateScale() ? realConcentratedLikelihoodOptions(resolved) : resolved;
        }

        @Override
        protected double[] adjustLogLikelihoodObs(FilterResult filterResult,
                                                  double[] logLikelihoodObs,
                                                  double[] params,
                                                  KalmanSSM model,
                                                  int burn) {
            return adjustLogLikelihoodObsInternal(filterResult, logLikelihoodObs, params, model, burn);
        }

        @Override
        protected double adjustedLogLikelihood(FilterResult filterResult,
                                               double[] params,
                                               KalmanSSM model,
                                               int burn) {
            return adjustedLogLikelihoodInternal(filterResult, burn);
        }

        @Override
        protected void updateModel(double[] params, KalmanSSM model) {
            updateModelInternal(params, model, updateWorkspace);
        }

        @Override
        protected InitialState initialState(double[] params, KalmanSSM model) {
            return initializationInternal(params, model);
        }

        @Override
        protected int likelihoodBurn(double[] params, KalmanSSM model) {
            return likelihoodBurnInternal(params, model);
        }

        @Override
        protected double[] scoreSurface(double[] params, NumericalGradient gradientScheme) {
            if (gradientScheme == NumericalGradient.CENTRAL && supportsAnalyticComplexStepScore()) {
                return sumScoreObs(analyticComplexStepScoreObs(params), params.length);
            }
            return super.scoreSurface(params, gradientScheme);
        }

        @Override
        protected double[] scoreObsSurface(double[] params) {
            if (supportsAnalyticComplexStepScore()) {
                return analyticComplexStepScoreObs(params);
            }
            return super.scoreObsSurface(params);
        }

        @Override
        protected double[] approximateHessianSurface(double[] params) {
            if (supportsAnalyticApproximateHessian()) {
                double[] analytic = approximateHessianComplexStepSurface(params);
                if (analytic != null) {
                    return analytic;
                }
            }
            return super.approximateHessianSurface(params);
        }

        @Override
        protected double[] observedInformationMatrixSurface(double[] params) {
            double[] information = harveyObservedInformationMatrixSurface(params);
            if (information == null) {
                throw unavailableObservedInformationException();
            }
            return information;
        }

        @Override
        protected double[] invertObservedInformationSurface(double[] information, int dimension) {
            return pseudoInverseInformationMatrix(information, dimension);
        }
    }

    private final class ComplexDelegate extends ComplexMLE {

        private final ComplexUpdateWorkspace updateWorkspace = new ComplexUpdateWorkspace();

        private int perturbIndex = -1;
        private double perturbStep = 0.0;

        private ComplexDelegate() {
            super(toComplexStateSpace(meta.templateModel()), meta.startParams(), meta.parameterNames());
        }

        private void trimDelegateWorkspace() {
            trimWorkspaces();
            updateWorkspace.release();
        }

        private KalmanSSM snapshotComplexModel(double[] params) {
            return snapshotComplexModel(params, -1, 0.0);
        }

        private KalmanSSM snapshotComplexModel(double[] params, int perturbIndex, double perturbStep) {
            int previousIndex = this.perturbIndex;
            double previousStep = this.perturbStep;
            this.perturbIndex = perturbIndex;
            this.perturbStep = perturbStep;
            try {
                return super.snapshotModel(params);
            } finally {
                this.perturbIndex = previousIndex;
                this.perturbStep = previousStep;
            }
        }

        private KalmanSSM borrowedComplexModel(double[] params, int perturbIndex, double perturbStep) {
            int previousIndex = this.perturbIndex;
            double previousStep = this.perturbStep;
            this.perturbIndex = perturbIndex;
            this.perturbStep = perturbStep;
            try {
                return super.borrowedModel(params);
            } finally {
                this.perturbIndex = previousIndex;
                this.perturbStep = previousStep;
            }
        }

        @Override
        public double[] transformParams(double[] unconstrainedParams) {
            return transformParamsInternal(unconstrainedParams);
        }

        @Override
        public double[] untransformParams(double[] params) {
            return untransformParamsInternal(params);
        }

        @Override
        protected FilterOptions likelihoodOptions() {
            return spec.concentrateScale() ? CONCENTRATED_LIKELIHOOD_OPTIONS : super.likelihoodOptions();
        }

        @Override
        protected FilterOptions likelihoodOptions(FilterOptions options) {
            FilterOptions resolved = super.likelihoodOptions(options);
            return spec.concentrateScale() ? requireConcentratedLikelihoodSurfaces(resolved) : resolved;
        }

        @Override
        protected double[] adjustLogLikelihoodObs(ZFilterResult filterResult,
                                                  double[] logLikelihoodObs,
                                                  double[] params,
                                                  KalmanSSM model,
                                                  int burn) {
            if (!spec.concentrateScale()) {
                return logLikelihoodObs;
            }
            return concentratedLogLikelihoodObs(filterResult, logLikelihoodObs, concentratedScale(filterResult, burn), burn);
        }

        @Override
        protected double adjustedLogLikelihood(ZFilterResult filterResult,
                                               double[] params,
                                               KalmanSSM model,
                                               int burn) {
            return adjustedLogLikelihoodInternal(filterResult, burn);
        }

        @Override
        protected void updateModel(double[] params, KalmanSSM model) {
            updateModelComplexInternal(params, perturbIndex, perturbStep, model, updateWorkspace);
        }

        @Override
        protected InitialState initialState(double[] params, KalmanSSM model) {
            return initializationInternal(params, model);
        }

        @Override
        protected int likelihoodBurn(double[] params, KalmanSSM model) {
            return likelihoodBurnInternal(params, model);
        }
    }
}