package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.kalman.filter.FilterResult;
import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.mle.ComplexMLE;
import com.curioloop.yum4j.kalman.mle.RealMLE;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;
import com.curioloop.yum4j.linalg.mat.LU;
import com.curioloop.yum4j.optim.NumericalGradient;
import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;

import java.util.Arrays;

public final class SARIMAX {

    private static final FilterSpec CONCENTRATED_LIKELIHOOD_SPEC = FilterSpec.full().without(
        FilterSpec.Storage.PREDICTED_STATE,
        FilterSpec.Storage.FILTERED_STATE,
        FilterSpec.Storage.KALMAN_GAIN);
    private static final double[] UNIT_POLYNOMIAL = {1.0};

    private final SARIMAXSpec spec;
    private final SARIMAXSupport.Meta meta;
    private final ParameterLayout parameterLayout;
    private final ThreadLocal<SARIMAXSupport.StationaryTransformWorkspace> stationaryTransformWorkspace;
    private final ThreadLocal<RealUpdateWorkspace> realUpdateWorkspace;
    private final RealDelegate realDelegate;
    private final ComplexDelegate complexDelegate;

    public SARIMAX(SARIMAXSpec spec) {
        this(spec, SARIMAXSupport.analyze(spec));
    }

    private SARIMAX(SARIMAXSpec spec, SARIMAXSupport.Meta meta) {
        this.spec = spec;
        this.meta = meta;
        this.parameterLayout = ParameterLayout.of(meta);
        this.stationaryTransformWorkspace = ThreadLocal.withInitial(SARIMAXSupport.StationaryTransformWorkspace::new);
        this.realUpdateWorkspace = ThreadLocal.withInitial(RealUpdateWorkspace::new);
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

    public double[] logLikelihoodObs(double[] params) {
        return realDelegate.logLikelihoodObs(params);
    }

    public FilterResult filter(double[] params) {
        return realDelegate.filter(params);
    }

    public FilterResult filter(double[] params, FilterSpec filterSpec) {
        return realDelegate.filter(params, filterSpec);
    }

    public SmootherResult smooth(double[] params) {
        return realDelegate.smooth(params);
    }

    public SmootherResult smooth(double[] params, SmootherSpec smootherSpec) {
        return realDelegate.smooth(params, smootherSpec);
    }

    public FilterResult predict(double[] params) {
        return realDelegate.predict(params);
    }

    public FilterResult predict(double[] params, FilterSpec filterSpec) {
        return realDelegate.predict(params, filterSpec);
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

    public Optimization optimize() {
        return realDelegate.optimize();
    }

    public Optimization optimize(Minimizer<?, ?, ?> optimizer) {
        return realDelegate.optimize(optimizer);
    }

    public Optimization optimizeTransformed(double[] unconstrainedStartParams) {
        return realDelegate.optimizeTransformed(unconstrainedStartParams);
    }

    public Optimization optimizeTransformed(double[] unconstrainedStartParams, Minimizer<?, ?, ?> optimizer) {
        return realDelegate.optimizeTransformed(unconstrainedStartParams, optimizer);
    }

    public StateSpaceModel snapshotModel(double[] params) {
        return realDelegate.snapshotModel(params);
    }

    StateSpaceModel snapshotComplexModel(double[] params) {
        return complexDelegate.snapshotComplexModel(params);
    }

    StateSpaceModel snapshotComplexModel(double[] params, int perturbIndex, double perturbStep) {
        return complexDelegate.snapshotComplexModel(params, perturbIndex, perturbStep);
    }

    double complexLogLikelihood(double[] params) {
        return complexDelegate.logLikelihood(params);
    }

    double[] complexLogLikelihoodObs(double[] params) {
        return complexDelegate.logLikelihoodObs(params);
    }

    ZFilterResult filterComplex(double[] params, FilterSpec filterSpec) {
        return complexDelegate.filter(params, filterSpec);
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

        FilterResult result = realDelegate.filter(params, FilterSpec.full());
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

        double[] steps = new double[paramCount];
        for (int i = 0; i < paramCount; i++) {
            steps[i] = approximateHessianStepSize(params[i]);
        }

        for (int i = 0; i < paramCount; i++) {
            for (int j = i; j < paramCount; j++) {
                double[] plus = Arrays.copyOf(params, paramCount);
                plus[j] += steps[j];
                double[] llfPlus = analyticComplexLogLikelihood(plus, i, steps[i]);
                if (llfPlus == null) {
                    return null;
                }

                double[] minus = Arrays.copyOf(params, paramCount);
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
        FilterResult baseline = realDelegate.filter(params, FilterSpec.full());
        int start = Math.max(Math.max(0, likelihoodBurnInternal(params, null)), baseline.nobsDiffuse);
        double[] forecastErrorPartials = new double[baseline.nobs * paramCount];
        double[] forecastVariancePartials = new double[baseline.nobs * paramCount];

        for (int paramIndex = 0; paramIndex < paramCount; paramIndex++) {
            double step = harveyDerivativeStepSize(params[paramIndex]);
            ComplexForecastSurface surface = analyticComplexForecastSurface(params, paramIndex, step);
            if (surface == null) {
                return null;
            }
            for (int t = 0; t < baseline.nobs; t++) {
                forecastErrorPartials[t * paramCount + paramIndex] = surface.forecastError()[t * 2 + 1] / step;
                forecastVariancePartials[t * paramCount + paramIndex] = surface.forecastVariance()[t * 2 + 1] / step;
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

    private ComplexForecastSurface analyticComplexForecastSurface(double[] params, int perturbIndex, double perturbStep) {
        StateSpaceModel model = snapshotComplexModel(params, perturbIndex, perturbStep);
        AnalyticInitialization initialization = analyticInitialization(model);
        if (initialization == null) {
            return null;
        }

        int nobs = model.observationCount();
        int stateCount = model.stateCount();
        int kPosdef = model.stateDisturbanceCount();
        double[] state = Arrays.copyOf(initialization.state(), initialization.state().length);
        double[] covariance = Arrays.copyOf(initialization.covariance(), initialization.covariance().length);
        double[] forecastError = new double[nobs * 2];
        double[] forecastVariance = new double[nobs * 2];
        double[] dot = new double[2];

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
                double[] h = complexScalar(model.obsCov[obsCovOffset], model.obsCov[obsCovOffset + 1]);
                double[] d = complexScalar(model.obsIntercept[obsInterceptOffset], model.obsIntercept[obsInterceptOffset + 1]);
                double[] y = complexScalar(model.endog[endogOffset], model.endog[endogOffset + 1]);
                ZLAS.zdotu(stateCount, model.design, designOffset >> 1, 1, state, 0, 1, dot);
                double[] predictedObservation = complexScalar(dot[0], dot[1]);
                double[] currentForecastError = complexSubtract(y, complexAdd(d, predictedObservation));
                double[] gainNumerator = new double[stateCount * 2];
                ZLAS.zgemv(BLAS.Trans.NoTrans, stateCount, stateCount, 1.0, 0.0,
                    covariance, 0, stateCount,
                    model.design, designOffset, 1,
                    0.0, 0.0, gainNumerator, 0, 1);
                ZLAS.zdotu(stateCount, model.design, designOffset >> 1, 1, gainNumerator, 0, 1, dot);
                double[] currentForecastVariance = complexAdd(h, complexScalar(dot[0], dot[1]));
                if (!isFiniteComplex(currentForecastVariance) || isNearZero(currentForecastVariance[0], currentForecastVariance[1])) {
                    return null;
                }
                forecastError[t * 2] = currentForecastError[0];
                forecastError[t * 2 + 1] = currentForecastError[1];
                forecastVariance[t * 2] = currentForecastVariance[0];
                forecastVariance[t * 2 + 1] = currentForecastVariance[1];

                double[] forecastVarianceInv = reciprocalComplex(currentForecastVariance[0], currentForecastVariance[1]);
                double[] kalmanGain = complexVectorScale(gainNumerator, forecastVarianceInv[0], forecastVarianceInv[1]);
                filteredState = complexVectorAdd(state, complexVectorScale(kalmanGain, currentForecastError[0], currentForecastError[1]));
                filteredCovariance = Arrays.copyOf(covariance, covariance.length);
                ZLAS.zgeru(stateCount, stateCount, -forecastVarianceInv[0], -forecastVarianceInv[1],
                    gainNumerator, 0, 1,
                    gainNumerator, 0, 1,
                    filteredCovariance, 0, stateCount);
            }

            state = new double[stateCount * 2];
            System.arraycopy(model.stateIntercept, stateInterceptOffset, state, 0, stateCount * 2);
            ZLAS.zgemv(BLAS.Trans.NoTrans, stateCount, stateCount, 1.0, 0.0,
                model.transition, transitionOffset, model.transitionLeadingDimension(),
                filteredState, 0, 1,
                1.0, 0.0, state, 0, 1);
            double[] transitionTimesCovariance = complexMatrixMultiply(
                model.transition, transitionOffset >> 1, model.transitionLeadingDimension(),
                filteredCovariance, 0, stateCount,
                stateCount, stateCount, stateCount);
            double[] predictedCovariance = new double[stateCount * stateCount * 2];
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, stateCount, stateCount, stateCount,
                1.0, 0.0, transitionTimesCovariance, 0, stateCount,
                model.transition, transitionOffset >> 1, model.transitionLeadingDimension(),
                0.0, 0.0, predictedCovariance, 0, stateCount);
            if (kPosdef > 0) {
                double[] selectedCovariance = complexMatrixMultiply(
                    model.selection, selectionOffset >> 1, model.selectionLeadingDimension(),
                    model.stateCov, stateCovOffset >> 1, model.stateCovarianceLeadingDimension(),
                    stateCount, kPosdef, kPosdef);
                double[] stateInnovation = new double[stateCount * stateCount * 2];
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, stateCount, stateCount, kPosdef,
                    1.0, 0.0, selectedCovariance, 0, kPosdef,
                    model.selection, selectionOffset >> 1, model.selectionLeadingDimension(),
                    0.0, 0.0, stateInnovation, 0, stateCount);
                predictedCovariance = complexMatrixAdd(predictedCovariance, stateInnovation);
            }
            covariance = predictedCovariance;
        }

        return new ComplexForecastSurface(forecastError, forecastVariance);
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
            FilterResult shifted = realDelegate.filter(perturbed, FilterSpec.full());
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
            double[] logLikelihoodObs = analyticComplexLogLikelihoodObs(params, paramIndex, step);
            if (logLikelihoodObs == null) {
                return null;
            }
            for (int t = 0; t < nobs; t++) {
                scoreObs[t * paramCount + paramIndex] = logLikelihoodObs[t * 2 + 1] / step;
            }
        }
        return scoreObs;
    }

    double[] analyticComplexStepScoreObsForTesting(double[] params) {
        return analyticComplexStepScoreObsOrNull(params);
    }

    double[] numericScoreObsForTesting(double[] params) {
        return realDelegate.scoreObs(params);
    }

    private double[] analyticComplexLogLikelihood(double[] params, int perturbIndex, double perturbStep) {
        double[] logLikelihoodObs = analyticComplexLogLikelihoodObs(params, perturbIndex, perturbStep);
        if (logLikelihoodObs == null) {
            return null;
        }
        return sumComplex(logLikelihoodObs);
    }

    private double[] analyticComplexLogLikelihoodObs(double[] params, int perturbIndex, double perturbStep) {
        StateSpaceModel model = snapshotComplexModel(params, perturbIndex, perturbStep);
        AnalyticInitialization initialization = analyticInitialization(model);
        if (initialization == null) {
            return null;
        }

        int nobs = model.observationCount();
        int stateCount = model.stateCount();
        int kPosdef = model.stateDisturbanceCount();
        int burn = Math.min(Math.max(0, likelihoodBurnInternal(params, model)), nobs);

        double[] state = Arrays.copyOf(initialization.state(), initialization.state().length);
        double[] covariance = Arrays.copyOf(initialization.covariance(), initialization.covariance().length);
        double[] logLikelihoodObs = new double[nobs * 2];
        double[] dot = new double[2];
        double logTwoPi = Math.log(2.0 * Math.PI);

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
                double[] h = complexScalar(model.obsCov[obsCovOffset], model.obsCov[obsCovOffset + 1]);
                double[] d = complexScalar(model.obsIntercept[obsInterceptOffset], model.obsIntercept[obsInterceptOffset + 1]);
                double[] y = complexScalar(model.endog[endogOffset], model.endog[endogOffset + 1]);
                ZLAS.zdotu(stateCount, model.design, designOffset >> 1, 1, state, 0, 1, dot);
                double[] predictedObservation = complexScalar(dot[0], dot[1]);
                double[] forecastError = complexSubtract(y, complexAdd(d, predictedObservation));
                double[] gainNumerator = new double[stateCount * 2];
                ZLAS.zgemv(BLAS.Trans.NoTrans, stateCount, stateCount, 1.0, 0.0,
                    covariance, 0, stateCount,
                    model.design, designOffset, 1,
                    0.0, 0.0, gainNumerator, 0, 1);
                ZLAS.zdotu(stateCount, model.design, designOffset >> 1, 1, gainNumerator, 0, 1, dot);
                double[] forecastVariance = complexAdd(h, complexScalar(dot[0], dot[1]));
                if (!isFiniteComplex(forecastVariance) || isNearZero(forecastVariance[0], forecastVariance[1])) {
                    return null;
                }
                double[] forecastVarianceInv = reciprocalComplex(forecastVariance[0], forecastVariance[1]);
                double[] kalmanGain = complexVectorScale(gainNumerator, forecastVarianceInv[0], forecastVarianceInv[1]);
                filteredState = complexVectorAdd(state, complexVectorScale(kalmanGain, forecastError[0], forecastError[1]));
                filteredCovariance = Arrays.copyOf(covariance, covariance.length);
                ZLAS.zgeru(stateCount, stateCount, -forecastVarianceInv[0], -forecastVarianceInv[1],
                    gainNumerator, 0, 1,
                    gainNumerator, 0, 1,
                    filteredCovariance, 0, stateCount);

                double[] quadratic = complexMultiply(forecastError, complexMultiply(forecastError, forecastVarianceInv));
                double[] logLikelihood = complexScale(
                    complexAdd(
                        complexScalar(logTwoPi, 0.0),
                        complexAdd(logComplex(forecastVariance[0], forecastVariance[1]), quadratic)),
                    -0.5);
                if (t >= burn) {
                    logLikelihoodObs[t * 2] = logLikelihood[0];
                    logLikelihoodObs[t * 2 + 1] = logLikelihood[1];
                }
            }

            state = new double[stateCount * 2];
            System.arraycopy(model.stateIntercept, stateInterceptOffset, state, 0, stateCount * 2);
            ZLAS.zgemv(BLAS.Trans.NoTrans, stateCount, stateCount, 1.0, 0.0,
                model.transition, transitionOffset, model.transitionLeadingDimension(),
                filteredState, 0, 1,
                1.0, 0.0, state, 0, 1);
            double[] transitionTimesCovariance = complexMatrixMultiply(
                model.transition, transitionOffset >> 1, model.transitionLeadingDimension(),
                filteredCovariance, 0, stateCount,
                stateCount, stateCount, stateCount);
            double[] predictedCovariance = new double[stateCount * stateCount * 2];
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, stateCount, stateCount, stateCount,
                1.0, 0.0, transitionTimesCovariance, 0, stateCount,
                model.transition, transitionOffset >> 1, model.transitionLeadingDimension(),
                0.0, 0.0, predictedCovariance, 0, stateCount);
            if (kPosdef > 0) {
                double[] selectedCovariance = complexMatrixMultiply(
                    model.selection, selectionOffset >> 1, model.selectionLeadingDimension(),
                    model.stateCov, stateCovOffset >> 1, model.stateCovarianceLeadingDimension(),
                    stateCount, kPosdef, kPosdef);
                double[] stateInnovation = new double[stateCount * stateCount * 2];
                ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, stateCount, stateCount, kPosdef,
                    1.0, 0.0, selectedCovariance, 0, kPosdef,
                    model.selection, selectionOffset >> 1, model.selectionLeadingDimension(),
                    0.0, 0.0, stateInnovation, 0, stateCount);
                predictedCovariance = complexMatrixAdd(predictedCovariance, stateInnovation);
            }
            covariance = predictedCovariance;
        }

        return logLikelihoodObs;
    }

    private AnalyticInitialization analyticInitialization(StateSpaceModel model) {
        if (spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)) {
            return null;
        }

        int stateCount = model.stateCount();
        double[] state = new double[stateCount * 2];
        double[] covariance = new double[stateCount * stateCount * 2];

        boolean approximateDiffuseOnly = spec.hasApproximateDiffuseVariance()
            && !spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE);
        if (!spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY) || approximateDiffuseOnly) {
            fillComplexDiagonal(covariance, 0, stateCount, stateCount, spec.approximateDiffuseVariance());
            return new AnalyticInitialization(state, covariance);
        }

        if (meta.totalDiff() > 0) {
            fillComplexDiagonal(covariance, 0, stateCount, meta.totalDiff(), spec.approximateDiffuseVariance());
        }

        int stationaryStart = meta.totalDiff();
        int stationaryDimension = meta.totalDiff() == 0 && !meta.stateRegression()
            ? stateCount
            : meta.totalOrder();
        if (stationaryDimension > 0
                && !solveAnalyticStationaryBlock(model, stationaryStart, stationaryDimension, state, covariance)) {
            return null;
        }

        if (meta.stateRegression()) {
            int regressionStart = meta.totalDiff() + meta.totalOrder();
            fillComplexDiagonal(covariance, regressionStart, stateCount, stateCount - regressionStart, spec.approximateDiffuseVariance());
        }
        return new AnalyticInitialization(state, covariance);
    }

    private boolean solveAnalyticStationaryBlock(StateSpaceModel model,
                                                 int startState,
                                                 int blockDimension,
                                                 double[] targetState,
                                                 double[] targetCovariance) {
        double[] transition = copyComplexSquareBlock(
            model.transition,
            model.transitionOffset(0),
            model.transitionLeadingDimension(),
            startState,
            blockDimension);
        double[] stateIntercept = copyComplexVectorBlock(
            model.stateIntercept,
            model.stateInterceptOffset(0),
            startState,
            blockDimension);
        double[] selection = copyComplexRectangularBlock(
            model.selection,
            model.selectionOffset(0),
            model.selectionLeadingDimension(),
            startState,
            blockDimension,
            model.stateDisturbanceCount());
        double[] stateCovariance = copyComplexSquareBlock(
            model.stateCov,
            model.stateCovarianceOffset(0),
            model.stateCovarianceLeadingDimension(),
            0,
            model.stateDisturbanceCount());

        double[] stationaryState = solveAnalyticStationaryMean(transition, stateIntercept, blockDimension);
        double[] stationaryCovariance = solveAnalyticStationaryCovariance(
            transition,
            selection,
            stateCovariance,
            blockDimension,
            model.stateDisturbanceCount());
        if (stationaryState == null || stationaryCovariance == null
                || !allFinite(stationaryState) || !allFinite(stationaryCovariance)) {
            return false;
        }

        System.arraycopy(stationaryState, 0, targetState, startState * 2, stationaryState.length);
        writeComplexSquareBlock(stationaryCovariance, blockDimension, targetCovariance, targetState.length / 2, startState);
        return true;
    }

    private static double[] solveAnalyticStationaryMean(double[] transition,
                                                        double[] stateIntercept,
                                                        int dimension) {
        if (dimension == 0) {
            return new double[0];
        }

        double[] system = new double[dimension * dimension * 2];
        for (int row = 0; row < dimension; row++) {
            int rowOffset = row * dimension * 2;
            for (int col = 0; col < dimension; col++) {
                int off = rowOffset + col * 2;
                system[off] = -transition[off];
                system[off + 1] = -transition[off + 1];
            }
            system[rowOffset + row * 2] += 1.0;
        }

        double[] solution = Arrays.copyOf(stateIntercept, stateIntercept.length);
        return solveRealifiedComplexSystem(system, solution, dimension) ? solution : null;
    }

    private static double[] solveAnalyticStationaryCovariance(double[] transition,
                                                              double[] selection,
                                                              double[] stateCovariance,
                                                              int dimension,
                                                              int kPosdef) {
        if (dimension == 0) {
            return new double[0];
        }

        double[] rhs = new double[dimension * dimension * 2];
        if (kPosdef > 0) {
            double[] selected = complexMatrixMultiply(selection, stateCovariance, dimension, kPosdef, kPosdef);
            ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, dimension, dimension, kPosdef,
                1.0, 0.0, selected, 0, kPosdef,
                selection, 0, kPosdef,
                0.0, 0.0, rhs, 0, dimension);
        }

        int systemDimension = dimension * dimension;
        double[] system = new double[systemDimension * systemDimension * 2];
        double[] vector = new double[systemDimension * 2];
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

        if (!solveRealifiedComplexSystem(system, vector, systemDimension)) {
            return null;
        }

        double[] covariance = new double[dimension * dimension * 2];
        for (int row = 0; row < dimension; row++) {
            int sourceOffset = row * dimension * 2;
            int targetOffset = row * dimension * 2;
            System.arraycopy(vector, sourceOffset, covariance, targetOffset, dimension * 2);
        }
        symmetrizeComplexTranspose(covariance, dimension);
        return covariance;
    }

    private static boolean solveRealifiedComplexSystem(double[] complexMatrix,
                                                       double[] complexVector,
                                                       int dimension) {
        int realDimension = dimension * 2;
        double[] realMatrix = new double[realDimension * realDimension];
        double[] realVector = new double[realDimension];
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

        int[] pivots = new int[realDimension];
        if (BLAS.dgetrf(realDimension, realDimension, realMatrix, 0, realDimension, pivots, 0) != 0) {
            return false;
        }
        BLAS.dgetrs(BLAS.Trans.NoTrans, realDimension, 1, realMatrix, 0, realDimension, pivots, 0, realVector, 0, 1);
        for (int row = 0; row < dimension; row++) {
            complexVector[row * 2] = realVector[row];
            complexVector[row * 2 + 1] = realVector[dimension + row];
        }
        return allFinite(realVector);
    }

    private static double[] copyComplexVectorBlock(double[] source,
                                                   int offset,
                                                   int startState,
                                                   int blockDimension) {
        double[] values = new double[blockDimension * 2];
        if (source != null) {
            System.arraycopy(source, offset + startState * 2, values, 0, blockDimension * 2);
        }
        return values;
    }

    private static double[] copyComplexRectangularBlock(double[] source,
                                                        int offset,
                                                        int leadingDimension,
                                                        int startRow,
                                                        int rows,
                                                        int cols) {
        double[] values = new double[rows * cols * 2];
        if (source == null) {
            return values;
        }
        for (int row = 0; row < rows; row++) {
            int sourceRow = offset + (startRow + row) * leadingDimension * 2;
            int targetRow = row * cols * 2;
            System.arraycopy(source, sourceRow, values, targetRow, cols * 2);
        }
        return values;
    }

    private static double[] copyComplexSquareBlock(double[] source,
                                                   int offset,
                                                   int leadingDimension,
                                                   int startState,
                                                   int blockDimension) {
        double[] values = new double[blockDimension * blockDimension * 2];
        if (source == null) {
            return values;
        }
        for (int row = 0; row < blockDimension; row++) {
            int sourceRow = offset + (startState + row) * leadingDimension * 2 + startState * 2;
            int targetRow = row * blockDimension * 2;
            System.arraycopy(source, sourceRow, values, targetRow, blockDimension * 2);
        }
        return values;
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

    public SARIMAXResults fit(Minimizer<?, ?, ?> optimizer) {
        return fit(SARIMAXFitOptions.builder().optimizer(optimizer).build());
    }

    public SARIMAXResults fit(SARIMAXFitOptions options) {
        SARIMAXFitOptions fitOptions = options == null ? SARIMAXFitOptions.DEFAULT : options;
        Optimization optimization = realDelegate.optimize(fitOptions.optimizer());

        double[] unconstrained = optimization.solution();
        double[] params = transformParams(unconstrained);
        FilterResult filterResult = filter(params, FilterSpec.full());
        return new SARIMAXResults(
            this,
            optimization,
            params,
            unconstrained,
            filterResult,
            fitOptions.covarianceType());
    }

    SARIMAX extend(double[] endog, double[][] exog) {
        SARIMAXSpec.Builder builder = SARIMAXSpec.builder(spec.order(), endog)
            .seasonalOrder(spec.seasonalOrder())
            .trendPowers(spec.trendPowers())
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
            1.0);
        transformStationaryBlock(
            unconstrainedParams,
            constrained,
            parameterLayout.maOffset(),
            parameterLayout.maLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_INVERTIBILITY),
            -1.0);
        transformStationaryBlock(
            unconstrainedParams,
            constrained,
            parameterLayout.seasonalArOffset(),
            parameterLayout.seasonalArLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY),
            1.0);
        transformStationaryBlock(
            unconstrainedParams,
            constrained,
            parameterLayout.seasonalMaOffset(),
            parameterLayout.seasonalMaLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_INVERTIBILITY),
            -1.0);

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
            1.0);
        untransformStationaryBlock(
            params,
            unconstrained,
            parameterLayout.maOffset(),
            parameterLayout.maLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_INVERTIBILITY),
            -1.0);
        untransformStationaryBlock(
            params,
            unconstrained,
            parameterLayout.seasonalArOffset(),
            parameterLayout.seasonalArLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY),
            1.0);
        untransformStationaryBlock(
            params,
            unconstrained,
            parameterLayout.seasonalMaOffset(),
            parameterLayout.seasonalMaLength(),
            spec.hasOption(SARIMAXOption.ENFORCE_INVERTIBILITY),
            -1.0);

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
                                                    StateSpaceModel model,
                                                    int burn) {
        if (!spec.concentrateScale()) {
            return logLikelihoodObs;
        }
        return concentratedLogLikelihoodObs(filterResult, logLikelihoodObs, concentratedScale(filterResult, burn));
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
        double scale = concentratedScale(filterResult, burn);
        double logScale = Math.log(scale);
        double[] effectiveEndog = meta.endog();
        double sum = 0.0;
        int nobs = Math.min(filterResult.nobs, effectiveEndog.length);
        for (int t = Math.max(0, burn); t < nobs; t++) {
            double value = filterResult.logLikelihoodObs[filterResult.logLikelihoodObsOffset(t)];
            double forecastVariance = filterResult.forecastErrorCov(0, 0, t);
            if (!Double.isNaN(effectiveEndog[t]) && forecastVariance > 0.0 && Double.isFinite(forecastVariance)) {
                double error = filterResult.forecastError(0, t);
                double scaleObs = error * error / forecastVariance;
                value += -0.5 * (logScale + scaleObs / scale - scaleObs);
            }
            sum += value;
        }
        return sum;
    }

    private void updateModelInternal(double[] params, StateSpaceModel model) {
        ParameterLayout layout = parameterLayout;
        double measurementVariance = layout.hasMeasurementError() ? params[layout.measurementErrorOffset()] : 0.0;
        double variance = layout.hasScale() ? params[layout.scaleOffset()] : 1.0;

        RealUpdateWorkspace workspace = realUpdateWorkspace.get();
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

    private void updateModelComplexInternal(double[] params, int perturbIndex, double perturbStep, StateSpaceModel model) {
        ParameterLayout layout = parameterLayout;
        double[] trendParams = complexSlice(params, perturbIndex, perturbStep, layout.trendOffset(), layout.trendEnd());
        double[] exogParams = meta.stateRegression()
            ? new double[0]
            : complexSlice(params, perturbIndex, perturbStep, layout.exogOffset(), layout.exogEnd());
        double[] arParams = complexSlice(params, perturbIndex, perturbStep, layout.arOffset(), layout.arEnd());
        double[] maParams = complexSlice(params, perturbIndex, perturbStep, layout.maOffset(), layout.maEnd());
        double[] seasonalArParams = complexSlice(params, perturbIndex, perturbStep, layout.seasonalArOffset(), layout.seasonalArEnd());
        double[] seasonalMaParams = complexSlice(params, perturbIndex, perturbStep, layout.seasonalMaOffset(), layout.seasonalMaEnd());
        double[] exogVariances = meta.timeVaryingRegression()
            ? complexSlice(params, perturbIndex, perturbStep, layout.timeVaryingRegressionOffset(), layout.timeVaryingRegressionEnd())
            : new double[0];
        double[] measurementVariance = spec.measurementError()
            ? complexScalar(params[layout.measurementErrorOffset()], imaginaryStep(perturbIndex, layout.measurementErrorOffset(), perturbStep))
            : complexScalar(0.0, 0.0);
        double[] variance = spec.concentrateScale()
            ? complexScalar(1.0, 0.0)
            : complexScalar(params[layout.scaleOffset()], imaginaryStep(perturbIndex, layout.scaleOffset(), perturbStep));

        double[] reducedArPolynomial = meta.reducedArOrder() > 0
            ? multiplyComplex(
                lagPolynomialComplex(arParams, spec.autoregressiveLags(), true),
                lagPolynomialComplex(seasonalArParams, spec.seasonalAutoregressiveLags(), true))
            : complexScalar(1.0, 0.0);

        if (meta.kExog() > 0 && !meta.stateRegression()) {
            for (int t = 0; t < meta.nobs(); t++) {
                double[] obsIntercept = dotComplexReal(exogParams, meta.exog()[t]);
                writeComplexScalar(model.obsIntercept, model.obsInterceptOffset(t), obsIntercept[0], obsIntercept[1]);
            }
        }

        if (meta.kTrend() > 0) {
            double[] arScale = sumComplex(reducedArPolynomial);
            if (isNearZero(arScale[0], arScale[1])) {
                arScale[0] = 1.0;
                arScale[1] = 0.0;
            }
            for (int t = 0; t < meta.nobs(); t++) {
                double[] trendValue = dotComplexReal(trendParams, meta.trendData()[t]);
                if (spec.hamiltonRepresentation()) {
                    double[] scaledTrend = divideComplex(trendValue[0], trendValue[1], arScale[0], arScale[1]);
                    addComplexScalar(model.obsIntercept, model.obsInterceptOffset(t), scaledTrend[0], scaledTrend[1]);
                } else {
                    writeComplexScalar(
                        model.stateIntercept,
                        complexVectorOffset(model.stateInterceptOffset(t), meta.totalDiff()),
                        trendValue[0],
                        trendValue[1]);
                }
            }
        }

        if (meta.reducedArOrder() > 0) {
            for (int t = 0; t < meta.nobs(); t++) {
                int transitionBase = model.transitionOffset(t);
                for (int i = 1; i < reducedArPolynomial.length / 2; i++) {
                    int row = spec.hamiltonRepresentation() ? meta.totalDiff() : meta.totalDiff() + i - 1;
                    int col = spec.hamiltonRepresentation() ? meta.totalDiff() + i - 1 : meta.totalDiff();
                    int off = complexMatrixOffset(transitionBase, meta.kStates(), row, col);
                    writeComplexScalar(model.transition, off, -reducedArPolynomial[i * 2], -reducedArPolynomial[i * 2 + 1]);
                }
            }
        }

        if (meta.reducedMaOrder() > 0) {
            double[] reducedMa = multiplyComplex(
                lagPolynomialComplex(maParams, spec.movingAverageLags(), false),
                lagPolynomialComplex(seasonalMaParams, spec.seasonalMovingAverageLags(), false));
            for (int t = 0; t < meta.nobs(); t++) {
                for (int i = 1; i < reducedMa.length / 2; i++) {
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
            double[] stateVariance = spec.concentrateScale() ? complexScalar(1.0, 0.0) : positiveContinuation(variance[0], variance[1]);
            for (int t = 0; t < meta.nobs(); t++) {
                int stateCovBase = model.stateCovarianceOffset(t);
                writeComplexScalar(model.stateCov, stateCovBase, stateVariance[0], stateVariance[1]);
                if (meta.timeVaryingRegression()) {
                    int diagonalOffset = meta.totalOrder() > 0 ? 1 : 0;
                    for (int index = 0; index < meta.kExog(); index++) {
                        int row = diagonalOffset + index;
                        int off = complexMatrixOffset(stateCovBase, meta.kPosdef(), row, row);
                        int complexIndex = index * 2;
                        double[] exogVariance = positiveContinuation(exogVariances[complexIndex], exogVariances[complexIndex + 1]);
                        writeComplexScalar(model.stateCov, off, exogVariance[0], exogVariance[1]);
                    }
                }
            }
        }
        double[] observationVariance = spec.measurementError()
            ? positiveContinuation(measurementVariance[0], measurementVariance[1])
            : complexScalar(0.0, 0.0);
        for (int t = 0; t < meta.nobs(); t++) {
            writeComplexScalar(model.obsCov, model.obsCovOffset(t), observationVariance[0], observationVariance[1]);
        }
    }

    private StateInitialization initializationInternal(double[] params, StateSpaceModel model) {
        if (spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY) && meta.totalDiff() == 0 && !meta.stateRegression()) {
            if (spec.hasApproximateDiffuseVariance() && !spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)) {
                return StateInitialization.approximateDiffuse(meta.kStates(), spec.approximateDiffuseVariance());
            }
            return StateInitialization.stationary(meta.kStates());
        }
        StateInitialization diffuse = spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)
            ? StateInitialization.diffuse()
            : StateInitialization.approximateDiffuse(spec.approximateDiffuseVariance());
        if (spec.hasApproximateDiffuseVariance() && !spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)) {
            return StateInitialization.approximateDiffuse(meta.kStates(), spec.approximateDiffuseVariance());
        }
        if (!spec.hasOption(SARIMAXOption.ENFORCE_STATIONARITY)) {
            return spec.hasOption(SARIMAXOption.USE_EXACT_DIFFUSE)
                ? StateInitialization.diffuse(meta.kStates())
                : StateInitialization.approximateDiffuse(meta.kStates(), spec.approximateDiffuseVariance());
        }
        StateInitialization composite = new StateInitialization(meta.kStates());
        if (meta.totalDiff() > 0) {
            composite.set(0, meta.totalDiff(), diffuse);
        }
        composite.set(meta.totalDiff(), meta.totalDiff() + meta.totalOrder(), StateInitialization.stationary(meta.totalOrder()));
        if (meta.stateRegression()) {
            composite.set(meta.totalDiff() + meta.totalOrder(), meta.kStates(), diffuse);
        }
        return composite;
    }

    private int likelihoodBurnInternal(double[] params, StateSpaceModel model) {
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
        double[] adjusted = concentratedLogLikelihoodObs(filterResult, filterResult.logLikelihoodObs, scale);
        System.arraycopy(adjusted, 0, filterResult.logLikelihoodObs, 0, adjusted.length);
        scaleSurface(filterResult.predictedStateCov, scale);
        scaleSurface(filterResult.filteredStateCov, scale);
        scaleSurface(filterResult.forecastErrorCov, scale);
        scaleSurface(filterResult.predictedDiffuseStateCov, scale);
        scaleSurface(filterResult.forecastErrorDiffuseCov, scale);
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
                                          double sign) {
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
                stationaryTransformWorkspace.get());
        } else {
            System.arraycopy(source, offset, target, offset, length);
        }
    }

    private void untransformStationaryBlock(double[] source,
                                            double[] target,
                                            int offset,
                                            int length,
                                            boolean transform,
                                            double sign) {
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
                stationaryTransformWorkspace.get());
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

    private static int lagPolynomialLength(int[] lags) {
        return lags.length == 0 ? 1 : lags[lags.length - 1] + 1;
    }

    private double[] concentratedLogLikelihoodObs(FilterResult filterResult, double[] logLikelihoodObs, double scale) {
        double[] adjusted = Arrays.copyOf(logLikelihoodObs, logLikelihoodObs.length);
        double[] effectiveEndog = meta.endog();
        double logScale = Math.log(scale);
        int nobs = Math.min(adjusted.length, effectiveEndog.length);
        for (int t = 0; t < nobs; t++) {
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

    private double[] concentratedLogLikelihoodObs(ZFilterResult filterResult, double[] logLikelihoodObs, double scale) {
        double[] adjusted = Arrays.copyOf(logLikelihoodObs, logLikelihoodObs.length);
        double[] effectiveEndog = meta.endog();
        double logScale = Math.log(scale);
        int nobs = Math.min(adjusted.length, effectiveEndog.length);
        for (int t = 0; t < nobs; t++) {
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
        for (int t = Math.max(0, burn); t < nobs; t++) {
            double value = filterResult.logLikelihoodObs[filterResult.logLikelihoodObsOffset(t)];
            int forecastCovOffset = filterResult.forecastErrorCovOffset(t);
            double forecastVariance = filterResult.forecastErrorCov[forecastCovOffset];
            if (!Double.isNaN(effectiveEndog[t]) && forecastVariance > 0.0 && Double.isFinite(forecastVariance)) {
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

    private static StateSpaceModel toComplexStateSpace(StateSpaceModel source) {
        StateSpaceModel expanded = StateSpaceModel.copyOf(source);
        int secondIndex = expanded.observationCount() > 1 ? 1 : 0;
        StateSpaceModel.Builder builder = StateSpaceModel.complexBuilder(
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

    private static double[] complexAdd(double[] left, double[] right) {
        return complexScalar(left[0] + right[0], left[1] + right[1]);
    }

    private static double[] complexSubtract(double[] left, double[] right) {
        return complexScalar(left[0] - right[0], left[1] - right[1]);
    }

    private static double[] complexScale(double[] value, double scale) {
        return complexScalar(value[0] * scale, value[1] * scale);
    }

    private static double[] complexMultiply(double[] left, double[] right) {
        return complexScalar(
            left[0] * right[0] - left[1] * right[1],
            left[0] * right[1] + left[1] * right[0]);
    }

    private static double[] reciprocalComplex(double re, double im) {
        double denom = re * re + im * im;
        if (!(denom > 0.0) || !Double.isFinite(denom)) {
            return complexScalar(Double.NaN, Double.NaN);
        }
        return complexScalar(re / denom, -im / denom);
    }

    private static double[] logComplex(double re, double im) {
        double magnitudeSquared = re * re + im * im;
        if (!(magnitudeSquared > 0.0) || !Double.isFinite(magnitudeSquared)) {
            return complexScalar(Double.NaN, Double.NaN);
        }
        return complexScalar(0.5 * Math.log(magnitudeSquared), Math.atan2(im, re));
    }

    private static boolean isFiniteComplex(double[] value) {
        return value.length >= 2 && Double.isFinite(value[0]) && Double.isFinite(value[1]);
    }

    private static double[] complexVectorScale(double[] vector, double scaleRe, double scaleIm) {
        double[] result = Arrays.copyOf(vector, vector.length);
        ZLAS.zscal(vector.length >> 1, scaleRe, scaleIm, result, 0, 1);
        return result;
    }

    private static double[] complexVectorAdd(double[] left, double[] right) {
        double[] result = Arrays.copyOf(left, left.length);
        ZLAS.zaxpy(left.length >> 1, 1.0, 0.0, right, 0, 1, result, 0, 1);
        return result;
    }

    private static double[] complexMatrixAdd(double[] left, double[] right) {
        double[] result = Arrays.copyOf(left, left.length);
        ZLAS.zaxpy(left.length >> 1, 1.0, 0.0, right, 0, 1, result, 0, 1);
        return result;
    }

    private static double[] complexMatrixMultiply(double[] left, double[] right, int rows, int inner, int cols) {
        return complexMatrixMultiply(left, 0, inner, right, 0, cols, rows, inner, cols);
    }

    private static double[] complexMatrixMultiply(double[] left,
                                                  int leftOffset,
                                                  int leftLeadingDimension,
                                                  double[] right,
                                                  int rightOffset,
                                                  int rightLeadingDimension,
                                                  int rows,
                                                  int inner,
                                                  int cols) {
        double[] result = new double[rows * cols * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, rows, cols, inner,
            1.0, 0.0, left, leftOffset, leftLeadingDimension,
            right, rightOffset, rightLeadingDimension,
            0.0, 0.0, result, 0, cols);
        return result;
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

    private static double[] sumComplex(double[] values) {
        double sumRe = 0.0;
        double sumIm = 0.0;
        for (int i = 0; i < values.length; i += 2) {
            sumRe += values[i];
            sumIm += values[i + 1];
        }
        return complexScalar(sumRe, sumIm);
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

    private static double[] positiveContinuation(double re, double im) {
        if (im == 0.0 && re < 1e-8) {
            return complexScalar(1e-8, 0.0);
        }
        return complexScalar(re, im);
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
        private double[] leftPolynomial = new double[1];
        private double[] rightPolynomial = new double[1];
        private double[] product = new double[1];

        private void ensure(int leftLength, int rightLength, int productLength) {
            if (leftPolynomial.length < leftLength) {
                leftPolynomial = new double[leftLength];
            }
            if (rightPolynomial.length < rightLength) {
                rightPolynomial = new double[rightLength];
            }
            if (product.length < productLength) {
                product = new double[productLength];
            }
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

    private record ComplexForecastSurface(double[] forecastError, double[] forecastVariance) {}

    private record AnalyticInitialization(double[] state, double[] covariance) {}

    private static void scaleSurface(double[] values, double scale) {
        if (values == null) {
            return;
        }
        for (int i = 0; i < values.length; i++) {
            values[i] *= scale;
        }
    }

    private final class RealDelegate extends RealMLE {

        private RealDelegate() {
            super(meta.templateModel(), meta.startParams(), meta.parameterNames());
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
        protected FilterSpec likelihoodSpec() {
            return spec.concentrateScale() ? CONCENTRATED_LIKELIHOOD_SPEC : super.likelihoodSpec();
        }

        @Override
        protected double[] adjustLogLikelihoodObs(FilterResult filterResult,
                                                  double[] logLikelihoodObs,
                                                  double[] params,
                                                  StateSpaceModel model,
                                                  int burn) {
            return adjustLogLikelihoodObsInternal(filterResult, logLikelihoodObs, params, model, burn);
        }

        @Override
        protected double adjustedLogLikelihood(FilterResult filterResult,
                                               double[] params,
                                               StateSpaceModel model,
                                               int burn) {
            return adjustedLogLikelihoodInternal(filterResult, burn);
        }

        @Override
        protected void updateModel(double[] params, StateSpaceModel model) {
            updateModelInternal(params, model);
        }

        @Override
        protected StateInitialization initialization(double[] params, StateSpaceModel model) {
            return initializationInternal(params, model);
        }

        @Override
        protected int likelihoodBurn(double[] params, StateSpaceModel model) {
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

        private int perturbIndex = -1;
        private double perturbStep = 0.0;

        private ComplexDelegate() {
            super(toComplexStateSpace(meta.templateModel()), meta.startParams(), meta.parameterNames());
        }

        private StateSpaceModel snapshotComplexModel(double[] params) {
            return snapshotComplexModel(params, -1, 0.0);
        }

        private StateSpaceModel snapshotComplexModel(double[] params, int perturbIndex, double perturbStep) {
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

        @Override
        public double[] transformParams(double[] unconstrainedParams) {
            return transformParamsInternal(unconstrainedParams);
        }

        @Override
        public double[] untransformParams(double[] params) {
            return untransformParamsInternal(params);
        }

        @Override
        protected FilterSpec likelihoodSpec() {
            return spec.concentrateScale() ? CONCENTRATED_LIKELIHOOD_SPEC : super.likelihoodSpec();
        }

        @Override
        protected double[] adjustLogLikelihoodObs(ZFilterResult filterResult,
                                                  double[] logLikelihoodObs,
                                                  double[] params,
                                                  StateSpaceModel model,
                                                  int burn) {
            if (!spec.concentrateScale()) {
                return logLikelihoodObs;
            }
            return concentratedLogLikelihoodObs(filterResult, logLikelihoodObs, concentratedScale(filterResult, burn));
        }

        @Override
        protected double adjustedLogLikelihood(ZFilterResult filterResult,
                                               double[] params,
                                               StateSpaceModel model,
                                               int burn) {
            return adjustedLogLikelihoodInternal(filterResult, burn);
        }

        @Override
        protected void updateModel(double[] params, StateSpaceModel model) {
            updateModelComplexInternal(params, perturbIndex, perturbStep, model);
        }

        @Override
        protected StateInitialization initialization(double[] params, StateSpaceModel model) {
            return initializationInternal(params, model);
        }

        @Override
        protected int likelihoodBurn(double[] params, StateSpaceModel model) {
            return likelihoodBurnInternal(params, model);
        }
    }
}