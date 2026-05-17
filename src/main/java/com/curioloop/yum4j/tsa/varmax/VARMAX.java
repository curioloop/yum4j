package com.curioloop.yum4j.tsa.varmax;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.mle.FixedParameterMLE;
import com.curioloop.yum4j.ssm.kalman.mle.RealMLE;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.optim.NumericalGradient;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;

import java.util.Arrays;
import java.util.logging.Logger;

public final class VARMAX {

    private static final Logger LOGGER = Logger.getLogger(VARMAX.class.getName());

    private final VARMAXSpec spec;
    private final VARMAXSupport.Meta meta;
    private final InitialState initialization;
    private final RealDelegate realDelegate;

    public VARMAX(VARMAXSpec spec) {
        this(spec, VARMAXSupport.analyze(spec), true);
    }

    private VARMAX(VARMAXSpec spec, VARMAXSupport.Meta meta, boolean warnIdentification) {
        if (warnIdentification && spec.order().autoregressive() > 0 && spec.order().movingAverage() > 0) {
            LOGGER.warning("Estimation of VARMA(p,q) models is not generically robust, due especially to identification issues.");
        }
        this.spec = spec;
        this.meta = meta;
        this.initialization = buildInitialization();
        this.realDelegate = new RealDelegate();
    }

    public VARMAXSpec spec() {
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

    public double[] unconstrainedStartParams() {
        return untransformParams(startParams());
    }

    public void trimWorkspaces() {
        realDelegate.trimWorkspaces();
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

    public int likelihoodBurn() {
        return spec.logLikelihoodBurn();
    }

    public int effectiveObservationCount() {
        int observed = 0;
        for (double[] row : meta.endog()) {
            boolean anyObserved = false;
            for (double value : row) {
                anyObserved |= !Double.isNaN(value);
            }
            if (anyObserved) {
                observed++;
            }
        }
        return observed;
    }

    public double[][] effectiveEndog() {
        return copyMatrix(meta.endog());
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

    boolean stateInterceptTimeVarying() {
        return VARMAXSupport.stateInterceptTimeVarying(meta);
    }

    public VARMAXResults fit() {
        return fit(VARMAXFitOptions.DEFAULT);
    }

    public VARMAXResults fit(VARMAXFitOptions options) {
        VARMAXFitOptions fitOptions = options == null ? VARMAXFitOptions.DEFAULT : options;
        double[] startParams = fitOptions.startParams() == null ? startParams() : fitOptions.startParams();
        FixedParameterMLE.Result fit = FixedParameterMLE.optimize(new FixedParameterMLE.Model() {
            @Override
            public double[] transformParams(double[] unconstrainedParams) {
                return VARMAX.this.transformParams(unconstrainedParams);
            }

            @Override
            public double[] untransformParams(double[] params) {
                return VARMAX.this.untransformParams(params);
            }

            @Override
            public double logLikelihood(double[] params) {
                return VARMAX.this.logLikelihood(params);
            }
        }, startParams, fitOptions.fixedParameters(), fitOptions.optimizer());
        Optimization optimization = fit.optimization();
        double[] unconstrained = fit.unconstrainedParams();
        double[] params = fit.params();
        FilterResult filterResult = filter(params, FilterOptions.defaults());
        return VARMAXResults.adoptFilterResult(this, optimization, params, unconstrained, filterResult,
            fitOptions.covarianceType(), fitOptions.fixedParameters());
    }

    VARMAX extend(double[][] endog, double[][] exog, int trendOffset) {
        VARMAXSpec.Builder builder = VARMAXSpec.builder(spec.order(), endog)
            .observationIntercept(spec.observationIntercept())
            .trendPowers(spec.trendPowers())
            .trendOffset(trendOffset)
            .errorCovariance(spec.errorCovariance())
            .measurementError(spec.measurementError())
            .logLikelihoodBurn(spec.logLikelihoodBurn())
            .endogNames(spec.endogNames())
            .exogNames(spec.exogNames());
        if (exog != null) {
            builder.exog(exog);
        }
        if (spec.hasApproximateDiffuseVariance()) {
            builder.approximateDiffuseVariance(spec.approximateDiffuseVariance());
        }
        for (VARMAXOption option : spec.options()) {
            builder.include(option);
        }
        VARMAXSpec extendedSpec = builder.build();
        return new VARMAX(extendedSpec, VARMAXSupport.extend(meta, extendedSpec), false);
    }

    double[][] extendExog(int outOfSample, double[][] futureExog) {
        double[][] baseExog = spec.exog();
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

    private InitialState buildInitialization() {
        if (spec.hasOption(VARMAXOption.USE_EXACT_DIFFUSE)) {
            return InitialState.diffuse(meta.kStates());
        }
        if (spec.hasApproximateDiffuseVariance()) {
            return InitialState.approximateDiffuse(meta.kStates(), spec.approximateDiffuseVariance());
        }
        return InitialState.stationary(meta.kStates());
    }

    private static double[][] copyMatrix(double[][] source) {
        double[][] copy = new double[source.length][];
        for (int i = 0; i < source.length; i++) {
            copy[i] = source[i].clone();
        }
        return copy;
    }

    private final class RealDelegate extends RealMLE {
        private RealDelegate() {
            super(meta.templateModel(), meta.startParams(), meta.parameterNames());
        }

        @Override
        protected void updateModel(double[] params, KalmanSSM model) {
            VARMAXSupport.updateModel(meta, params, model);
        }

        @Override
        protected InitialState initialState(double[] params, KalmanSSM model) {
            return initialization;
        }

        @Override
        public double[] transformParams(double[] unconstrainedParams) {
            return VARMAXSupport.transformParams(meta, unconstrainedParams);
        }

        @Override
        public double[] untransformParams(double[] params) {
            return VARMAXSupport.untransformParams(meta, params);
        }

        @Override
        protected int likelihoodBurn(double[] params, KalmanSSM model) {
            return spec.logLikelihoodBurn();
        }
    }
}
