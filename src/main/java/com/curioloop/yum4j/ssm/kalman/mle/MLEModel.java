package com.curioloop.yum4j.ssm.kalman.mle;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.linalg.Decomposer;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.mat.LU;
import com.curioloop.yum4j.linalg.mat.SVD;
import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.optim.NumericalGradient;
import com.curioloop.yum4j.optim.NumericalHessian;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;
import com.curioloop.yum4j.optim.cmaes.CMAESProblem;
import com.curioloop.yum4j.optim.lbfgsb.LBFGSBProblem;
import com.curioloop.yum4j.optim.slsqp.SLSQPProblem;
import com.curioloop.yum4j.optim.subplex.SubplexProblem;

import java.util.Arrays;
import java.util.Objects;

public abstract class MLEModel<F, S> {

    private static final FilterOptions LIKELIHOOD_OPTIONS = FilterOptions.likelihoodOnly();

    private static final FilterOptions PREDICTION_OPTIONS = FilterOptions.builder()
        .drop(FilterOptions.Surface.FILTERED_STATE,
            FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
            FilterOptions.Surface.KALMAN_GAIN,
            FilterOptions.Surface.LIKELIHOOD)
        .build();

    private final KalmanSSM templateModel;
    private final double[] startParams;
    private final String[] parameterNames;
    private EvaluationContext reusableContext;

    protected MLEModel(KalmanSSM templateModel,
                       double[] startParams,
                       String... parameterNames) {
        Objects.requireNonNull(templateModel, "templateModel");
        requireSupported(templateModel);
        this.templateModel = cloneWithLayout(templateModel);
        this.startParams = validateInitialParams(copyParams(startParams, "startParams"), "startParams");
        this.parameterNames = normalizeParameterNames(this.startParams.length, parameterNames);
    }

    public final int paramCount() {
        return startParams.length;
    }

    public final String[] parameterNames() {
        return Arrays.copyOf(parameterNames, parameterNames.length);
    }

    public final double[] startParams() {
        return Arrays.copyOf(startParams, startParams.length);
    }

    public final boolean complex() {
        return templateModel.complex();
    }

    public final int observationCount() {
        return templateModel.observationCount();
    }

    public final int observationDimension() {
        return templateModel.observationDimension();
    }

    public final int stateCount() {
        return templateModel.stateCount();
    }

    public double[] transformParams(double[] unconstrainedParams) {
        return validateParams(copyParams(unconstrainedParams, "unconstrainedParams"), "unconstrainedParams");
    }

    public double[] untransformParams(double[] params) {
        return validateParams(copyParams(params, "params"), "params");
    }

    protected abstract void requireSupported(KalmanSSM templateModel);

    protected abstract void updateModel(double[] params, KalmanSSM model);

    protected abstract InitialState initialState(double[] params, KalmanSSM model);

    protected abstract F filter(EvaluationContext context, FilterOptions options);

    protected abstract F filterBorrowed(EvaluationContext context, FilterOptions options);

    protected abstract S smooth(EvaluationContext context, SmootherOptions options);

    protected abstract double logLikelihood(F filterResult);

    protected abstract double logLikelihood(F filterResult, int burn);

    protected abstract double[] logLikelihoodObs(F filterResult);

    protected LogLikelihoodObsView logLikelihoodObsView(F filterResult) {
        double[] values = logLikelihoodObs(filterResult);
        return new LogLikelihoodObsView(values, 0, values.length);
    }

    protected int likelihoodBurn(double[] params, KalmanSSM model) {
        return 0;
    }

    protected FilterOptions likelihoodOptions() {
        return LIKELIHOOD_OPTIONS;
    }

    protected double[] adjustLogLikelihoodObs(F filterResult,
                                              double[] logLikelihoodObs,
                                              double[] params,
                                              KalmanSSM model,
                                              int burn) {
        return logLikelihoodObs;
    }

    protected double adjustedLogLikelihood(F filterResult,
                                           double[] params,
                                           KalmanSSM model,
                                           int burn) {
        return logLikelihood(filterResult, burn);
    }

    public final double logLikelihood(double[] params) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        int burn = likelihoodBurn(prepared.params, prepared.model);
        F filterResult = filterBorrowed(prepared.context, likelihoodOptions(null));
        return adjustedLogLikelihood(filterResult, prepared.params, prepared.model, burn);
    }

    public final double logLikelihood(double[] params, FilterOptions options) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        int burn = likelihoodBurn(prepared.params, prepared.model);
        F filterResult = filterBorrowed(prepared.context, likelihoodOptions(options));
        return adjustedLogLikelihood(filterResult, prepared.params, prepared.model, burn);
    }

    public final double[] logLikelihoodObs(double[] params) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        int burn = likelihoodBurn(prepared.params, prepared.model);
        F filterResult = filterBorrowed(prepared.context, likelihoodOptions(null));
        double[] logLikelihoodObs = adjustLogLikelihoodObs(
            filterResult,
            logLikelihoodObs(filterResult),
            prepared.params,
            prepared.model,
            burn);
        zeroLeadingLogLikelihood(logLikelihoodObs, burn);
        return logLikelihoodObs;
    }

    public final double[] logLikelihoodObs(double[] params, FilterOptions options) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        int burn = likelihoodBurn(prepared.params, prepared.model);
        F filterResult = filterBorrowed(prepared.context, likelihoodOptions(options));
        double[] logLikelihoodObs = adjustLogLikelihoodObs(
            filterResult,
            logLikelihoodObs(filterResult),
            prepared.params,
            prepared.model,
            burn);
        zeroLeadingLogLikelihood(logLikelihoodObs, burn);
        return logLikelihoodObs;
    }

    public final void trimWorkspaces() {
        if (reusableContext != null) {
            resetEvaluationContext(reusableContext);
            reusableContext = null;
        }
    }

    public final F filter(double[] params) {
        return filter(params, FilterOptions.defaults());
    }

    public final F filter(double[] params, FilterOptions options) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        return filter(prepared.context, options == null ? FilterOptions.defaults() : options);
    }

    public final S smooth(double[] params) {
        return smooth(params, SmootherOptions.defaults());
    }

    public final S smooth(double[] params, SmootherOptions options) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        SmootherOptions resolvedOptions = options == null ? SmootherOptions.defaults() : options;
        return smooth(prepared.context, resolvedOptions);
    }

    public final F predict(double[] params) {
        return predict(params, PREDICTION_OPTIONS);
    }

    public final F predict(double[] params, FilterOptions options) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        FilterOptions resolvedOptions = options == null ? PREDICTION_OPTIONS : options;
        return filter(prepared.context, resolvedOptions);
    }

    public final double[] score(double[] params) {
        return score(params, NumericalGradient.CENTRAL);
    }

    public final double[] score(double[] params, NumericalGradient gradientScheme) {
        Objects.requireNonNull(gradientScheme, "gradientScheme");
        double[] point = validateParams(copyParams(params, "params"), "params");
        return scoreSurface(point, gradientScheme);
    }

    public final double[] scoreObs(double[] params) {
        double[] point = validateParams(copyParams(params, "params"), "params");
        return scoreObsSurface(point);
    }

    public final double[] opgInformationMatrix(double[] params) {
        double[] point = validateParams(copyParams(params, "params"), "params");
        return opgInformationMatrixSurface(point);
    }

    public final double[] observedInformationMatrix(double[] params) {
        double[] point = validateParams(copyParams(params, "params"), "params");
        return observedInformationMatrixSurface(point);
    }

    public final double[] covarianceFromObservedInformation(double[] params) {
        double[] point = validateParams(copyParams(params, "params"), "params");
        return covarianceFromObservedInformationSurface(point, observedInformationMatrixSurface(point));
    }

    public final double[] covarianceFromApproximateInformation(double[] params) {
        double[] point = validateParams(copyParams(params, "params"), "params");
        return covarianceFromInformationSurface(point, approximateInformationMatrixSurface(point));
    }

    public final double[] covarianceFromOpg(double[] params) {
        double[] point = validateParams(copyParams(params, "params"), "params");
        return covarianceFromInformationSurface(point, opgInformationMatrixSurface(point));
    }

    public final double[] robustCovarianceFromObservedInformation(double[] params) {
        double[] point = validateParams(copyParams(params, "params"), "params");
        return robustCovarianceFromObservedInformationSurfaces(
            point,
            observedInformationMatrixSurface(point),
            opgInformationMatrixSurface(point));
    }

    public final double[] robustCovarianceFromApproximateInformation(double[] params) {
        double[] point = validateParams(copyParams(params, "params"), "params");
        return robustCovarianceFromInformationSurfaces(
            point,
            approximateInformationMatrixSurface(point),
            opgInformationMatrixSurface(point));
    }

    protected double[] scoreSurface(double[] point, NumericalGradient gradientScheme) {
        double[] gradient = new double[point.length];
        gradientScheme.wrap((x, n) -> logLikelihood(copyParams(x, n, "params"))).evaluate(point, point.length, gradient);
        return gradient;
    }

    protected double[] scoreObsSurface(double[] point) {
        return scoreObsAt(point);
    }

    protected double[] opgInformationMatrixSurface(double[] point) {
        double[] scoreObs = scoreObsSurface(point);
        double[] information = new double[point.length * point.length];
        for (int row = 0; row < observationCount(); row++) {
            int offset = row * point.length;
            for (int i = 0; i < point.length; i++) {
                double left = scoreObs[offset + i];
                for (int j = 0; j < point.length; j++) {
                    information[i * point.length + j] += left * scoreObs[offset + j];
                }
            }
        }
        return information;
    }

    protected double[] approximateHessianSurface(double[] point) {
        double[] hessian = new double[point.length * point.length];
        NumericalHessian.CENTRAL.compute(
            (x, n) -> logLikelihood(copyParams(x, n, "params")),
            point,
            point.length,
            hessian);
        return hessian;
    }

    protected double[] observedInformationMatrixSurface(double[] point) {
        return approximateInformationMatrixSurface(point);
    }

    protected double[] approximateInformationMatrixSurface(double[] point) {
        double[] information = approximateHessianSurface(point);
        for (int index = 0; index < information.length; index++) {
            information[index] = -information[index];
        }
        return information;
    }

    protected double[] covarianceFromInformationSurface(double[] point, double[] information) {
        if (point.length == 0) {
            return new double[0];
        }
        return invertInformationSurface(symmetrize(information, point.length), point.length);
    }

    protected double[] covarianceFromObservedInformationSurface(double[] point, double[] information) {
        if (point.length == 0) {
            return new double[0];
        }
        return invertObservedInformationSurface(symmetrize(information, point.length), point.length);
    }

    protected double[] robustCovarianceFromInformationSurfaces(double[] point,
                                                               double[] observedInformation,
                                                               double[] opgInformation) {
        if (point.length == 0) {
            return new double[0];
        }

        double[] inverseInformation = covarianceFromInformationSurface(point, observedInformation);
        if (!allFinite(inverseInformation)) {
            return nanMatrix(point.length);
        }

        return symmetrize(
            multiply(multiply(inverseInformation, opgInformation, point.length), inverseInformation, point.length),
            point.length);
    }

    protected double[] robustCovarianceFromObservedInformationSurfaces(double[] point,
                                                                       double[] observedInformation,
                                                                       double[] opgInformation) {
        if (point.length == 0) {
            return new double[0];
        }

        double[] inverseInformation = covarianceFromObservedInformationSurface(point, observedInformation);
        if (!allFinite(inverseInformation)) {
            return nanMatrix(point.length);
        }

        return symmetrize(
            multiply(multiply(inverseInformation, opgInformation, point.length), inverseInformation, point.length),
            point.length);
    }

    protected double[] invertInformationSurface(double[] information, int dimension) {
        return invertInformationMatrix(information, dimension);
    }

    protected double[] invertObservedInformationSurface(double[] information, int dimension) {
        return invertInformationSurface(information, dimension);
    }

    private double[] scoreObsAt(double[] point) {
        int nobs = observationCount();
        double[] scoreObs = new double[nobs * point.length];
        if (nobs == 0 || point.length == 0) {
            return scoreObs;
        }

        double[] x = Arrays.copyOf(point, point.length);
        for (int index = 0; index < x.length; index++) {
            double xi = x[index];
            double h = gradientStep(xi);

            x[index] = xi + h;
            LogLikelihoodObsView fPlus = logLikelihoodObsViewAt(x);
            for (int t = 0; t < nobs; t++) {
                scoreObs[t * point.length + index] = fPlus.valueAt(t);
            }

            x[index] = xi - h;
            LogLikelihoodObsView fMinus = logLikelihoodObsViewAt(x);

            x[index] = xi;
            for (int t = 0; t < nobs; t++) {
                scoreObs[t * point.length + index] = (scoreObs[t * point.length + index] - fMinus.valueAt(t)) / (2.0 * h);
            }
        }
        return scoreObs;
    }

    private LogLikelihoodObsView logLikelihoodObsViewAt(double[] params) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        int burn = likelihoodBurn(prepared.params, prepared.model);
        F filterResult = filterBorrowed(prepared.context, likelihoodOptions(null));
        LogLikelihoodObsView view = logLikelihoodObsView(filterResult);
        double[] adjusted = adjustLogLikelihoodObs(
            filterResult,
            view.contiguousValues(),
            prepared.params,
            prepared.model,
            burn);
        LogLikelihoodObsView adjustedView = adjusted == view.values
            ? view
            : new LogLikelihoodObsView(adjusted, 0, adjusted.length);
        zeroLeadingLogLikelihood(adjustedView.values, adjustedView.offset, burn, adjustedView.nobs);
        return adjustedView;
    }

    public final double[] standardErrors(double[] covariance) {
        Objects.requireNonNull(covariance, "covariance");
        int dimension = (int) Math.round(Math.sqrt(covariance.length));
        double[] stderr = new double[dimension];
        for (int index = 0; index < dimension; index++) {
            double variance = covariance[index * dimension + index];
            stderr[index] = variance >= 0.0 && Double.isFinite(variance)
                ? Math.sqrt(variance)
                : Double.NaN;
        }
        return stderr;
    }

    public final Univariate.Objective objectiveFunction() {
        return (x, n) -> -logLikelihood(copyParams(x, n, "params"));
    }

    public final Univariate.Objective transformedObjectiveFunction() {
        return (x, n) -> -logLikelihood(resolveTransformedParams(x, n));
    }

    public final Univariate objective() {
        return NumericalGradient.CENTRAL.wrap(objectiveFunction());
    }

    public final Univariate transformedObjective() {
        return NumericalGradient.CENTRAL.wrap(transformedObjectiveFunction());
    }

    public final Optimization optimize() {
        return optimize((Minimizer<?, ?, ?>) null);
    }

    public final Optimization optimize(Minimizer<?, ?, ?> optimizer) {
        return optimize(startParams, optimizer);
    }

    public final Optimization optimize(double[] params) {
        return optimize(params, null);
    }

    public final Optimization optimize(double[] params, Minimizer<?, ?, ?> optimizer) {
        double[] point = validateParams(copyParams(params, "params"), "params");
        return optimizeTransformed(untransformParams(point), optimizer);
    }

    public final Optimization optimizeTransformed(double[] unconstrainedStartParams) {
        return optimizeTransformed(unconstrainedStartParams, null);
    }

    public final Optimization optimizeTransformed(double[] unconstrainedStartParams, Minimizer<?, ?, ?> optimizer) {
        double[] point = validateParams(copyParams(unconstrainedStartParams, "unconstrainedStartParams"), "unconstrainedStartParams");
        return solveOptimization(point, optimizer == null ? defaultOptimizer() : optimizer);
    }

    private static LBFGSBProblem defaultOptimizer() {
        return Minimizer.lbfgsb().gradientTolerance(1e-8);
    }

    private Optimization solveOptimization(double[] unconstrainedStartParams, Minimizer<?, ?, ?> optimizer) {
        double[] start = unconstrainedStartParams.clone();
        if (optimizer instanceof LBFGSBProblem problem) {
            return problem.objective(transformedObjective())
                .initialPoint(start)
                .solve();
        }
        if (optimizer instanceof SLSQPProblem problem) {
            return problem.objective(transformedObjective())
                .initialPoint(start)
                .solve();
        }
        if (optimizer instanceof SubplexProblem problem) {
            return problem.objective(transformedObjectiveFunction())
                .initialPoint(start)
                .solve();
        }
        if (optimizer instanceof CMAESProblem problem) {
            return problem.objective(transformedObjectiveFunction())
                .initialPoint(start)
                .solve();
        }
        throw new IllegalArgumentException(
            "Unsupported MLE optimizer type: " + optimizer.getClass().getName()
                + ". Expected LBFGSBProblem, SLSQPProblem, SubplexProblem, or CMAESProblem.");
    }

    public final KalmanSSM snapshotModel(double[] params) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        return KalmanSSM.copyOf(prepared.model);
    }

    protected final KalmanSSM borrowedModel(double[] params) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        return prepared.model;
    }

    private PreparedEvaluation prepareNoCopy(double[] params, String name) {
        double[] resolvedParams = validateParams(params, name);
        EvaluationContext context = reusableEvaluationContext();
        updateModel(resolvedParams, context.model);
        InitialState resolvedInitialState = Objects.requireNonNull(
            initialState(resolvedParams, context.model),
            "initialState");
        context.initialState = resolvedInitialState;
        return new PreparedEvaluation(resolvedParams, context, context.model, context.initialState);
    }

    protected FilterOptions likelihoodOptions(FilterOptions options) {
        FilterOptions resolved = options == null ? likelihoodOptions() : options;
        if (resolved.includes(FilterOptions.Surface.LIKELIHOOD)) {
            return resolved;
        }
        return resolved.toBuilder()
            .retain(FilterOptions.Surface.LIKELIHOOD)
            .build();
    }

    private static void zeroLeadingLogLikelihood(double[] logLikelihoodObs, int burn) {
        zeroLeadingLogLikelihood(logLikelihoodObs, 0, burn, logLikelihoodObs.length);
    }

    private static void zeroLeadingLogLikelihood(double[] logLikelihoodObs, int offset, int burn, int nobs) {
        int limit = Math.min(Math.max(0, burn), nobs);
        Arrays.fill(logLikelihoodObs, offset, offset + limit, 0.0);
    }

    protected static final class LogLikelihoodObsView {
        private final double[] values;
        private final int offset;
        private final int nobs;

        protected LogLikelihoodObsView(double[] values, int offset, int nobs) {
            this.values = values;
            this.offset = offset;
            this.nobs = nobs;
        }

        private double valueAt(int t) {
            return values[offset + t];
        }

        private double[] contiguousValues() {
            if (offset == 0 && nobs == values.length) {
                return values;
            }
            return Arrays.copyOfRange(values, offset, offset + nobs);
        }
    }

    private static double[] invertInformationMatrix(double[] information, int dimension) {
        try {
            double[] inverse = LU.decompose(Arrays.copyOf(information, information.length), dimension)
                .inverse(new double[information.length]);
            return symmetrize(inverse, dimension);
        } catch (ArithmeticException ignored) {
            return nanMatrix(dimension);
        }
    }

    protected static double[] pseudoInverseInformationMatrix(double[] information, int dimension) {
        try {
            SVD svd = Decomposer.svd(
                Arrays.copyOf(information, information.length),
                dimension,
                dimension,
                SVD.Opts.WANT_U,
                SVD.Opts.WANT_V);
            if (!svd.ok() || svd.U() == null || svd.VT() == null) {
                return nanMatrix(dimension);
            }

            double[] singularValues = svd.singularValues();
            double[] u = svd.U();
            double[] vt = svd.VT();
            double[] inverse = new double[dimension * dimension];
            double maxSingular = 0.0;
            for (double singularValue : singularValues) {
                maxSingular = Math.max(maxSingular, singularValue);
            }
            double cutoff = 1e-15 * maxSingular;

            for (int i = 0; i < singularValues.length; i++) {
                double singularValue = singularValues[i];
                if (!(singularValue > cutoff)) {
                    continue;
                }
                double inverseSingular = 1.0 / singularValue;
                for (int row = 0; row < dimension; row++) {
                    double v = vt[i * dimension + row];
                    if (v == 0.0) {
                        continue;
                    }
                    for (int col = 0; col < dimension; col++) {
                        inverse[row * dimension + col] += v * inverseSingular * u[col * dimension + i];
                    }
                }
            }
            return symmetrize(inverse, dimension);
        } catch (RuntimeException ignored) {
            return nanMatrix(dimension);
        }
    }

    private static double[] multiply(double[] left, double[] right, int dimension) {
        double[] product = new double[dimension * dimension];
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, dimension, dimension, dimension,
            1.0, left, 0, dimension,
            right, 0, dimension,
            0.0, product, 0, dimension);
        return product;
    }

    private static boolean allFinite(double[] values) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        return true;
    }

    private static double[] symmetrize(double[] matrix, int dimension) {
        double[] symmetric = Arrays.copyOf(matrix, matrix.length);
        for (int row = 0; row < dimension; row++) {
            for (int col = row + 1; col < dimension; col++) {
                double value = 0.5 * (symmetric[row * dimension + col] + symmetric[col * dimension + row]);
                symmetric[row * dimension + col] = value;
                symmetric[col * dimension + row] = value;
            }
        }
        return symmetric;
    }

    private static double[] nanMatrix(int dimension) {
        double[] nan = new double[dimension * dimension];
        Arrays.fill(nan, Double.NaN);
        return nan;
    }

    private static double gradientStep(double value) {
        double step = NumericalGradient.CBRT_EPSILON * Math.max(1.0, Math.abs(value));
        double adjusted = (value + step) - value;
        return adjusted == 0.0 ? step : adjusted;
    }

    private double[] resolveTransformedParams(double[] params, int n) {
        double[] unconstrained = copyParams(params, n, "params");
        double[] transformed = Objects.requireNonNull(transformParams(unconstrained), "transformParams");
        return validateParams(copyParams(transformed, "transformedParams"), "transformedParams");
    }

    private EvaluationContext reusableEvaluationContext() {
        if (reusableContext == null) {
            reusableContext = createEvaluationContext(cloneWithLayout(templateModel));
        } else {
            resetEvaluationContext(reusableContext);
            reusableContext.model.copyDataFromSameLayout(templateModel);
            reusableContext.initialState = null;
        }
        return reusableContext;
    }

    protected EvaluationContext createEvaluationContext(KalmanSSM model) {
        return new EvaluationContext(model);
    }

    protected void resetEvaluationContext(EvaluationContext context) {
    }

    private static KalmanSSM cloneWithLayout(KalmanSSM source) {
        int secondIndex = source.observationCount() > 1 ? 1 : 0;
        boolean[] missing = source.getMissing();
        int[] nmissing = source.getNmissing();
        KalmanSSM.Builder builder = (source.complex()
            ? KalmanSSM.complexBuilder(
                source.observationDimension(),
                source.stateCount(),
                source.stateDisturbanceCount(),
                source.observationCount())
            : KalmanSSM.builder(
                source.observationDimension(),
                source.stateCount(),
                source.stateDisturbanceCount(),
                source.observationCount()))
            .design(Arrays.copyOf(source.designData(), source.designData().length), isTimeVarying(source.designOffset(0), source.designOffset(secondIndex)))
            .obsIntercept(Arrays.copyOf(source.obsInterceptData(), source.obsInterceptData().length), isTimeVarying(source.obsInterceptOffset(0), source.obsInterceptOffset(secondIndex)))
            .obsCov(Arrays.copyOf(source.obsCovData(), source.obsCovData().length), isTimeVarying(source.obsCovOffset(0), source.obsCovOffset(secondIndex)))
            .transition(Arrays.copyOf(source.transitionData(), source.transitionData().length), isTimeVarying(source.transitionOffset(0), source.transitionOffset(secondIndex)))
            .stateIntercept(Arrays.copyOf(source.stateInterceptData(), source.stateInterceptData().length), isTimeVarying(source.stateInterceptOffset(0), source.stateInterceptOffset(secondIndex)))
            .selection(Arrays.copyOf(source.selectionData(), source.selectionData().length), isTimeVarying(source.selectionOffset(0), source.selectionOffset(secondIndex)))
            .stateCovariance(Arrays.copyOf(source.stateCovarianceData(), source.stateCovarianceData().length), isTimeVarying(source.stateCovarianceOffset(0), source.stateCovarianceOffset(secondIndex)))
            .endog(Arrays.copyOf(source.endogData(), source.endogData().length), source.endogOffset(0));
        if (missing != null || nmissing != null) {
            builder.missing(
                missing == null ? null : Arrays.copyOf(missing, missing.length),
                nmissing == null ? null : Arrays.copyOf(nmissing, nmissing.length));
        } else {
            builder.allObserved();
        }
        return builder.build();
    }

    private static boolean isTimeVarying(int offset0, int offset1) {
        return offset0 != offset1;
    }

    private static double[] copyParams(double[] params, String name) {
        Objects.requireNonNull(params, name);
        return Arrays.copyOf(params, params.length);
    }

    private static double[] copyParams(double[] params, int n, String name) {
        Objects.requireNonNull(params, name);
        if (n < 0 || n > params.length) {
            throw new IllegalArgumentException(name + " has invalid effective length " + n);
        }
        return Arrays.copyOf(params, n);
    }

    private double[] validateParams(double[] params, String name) {
        Objects.requireNonNull(params, name);
        if (params.length != startParams.length) {
            throw new IllegalArgumentException(name + " length=" + params.length + " but expected " + startParams.length);
        }
        for (int i = 0; i < params.length; i++) {
            if (!Double.isFinite(params[i])) {
                throw new IllegalArgumentException(name + "[" + i + "] must be finite");
            }
        }
        return params;
    }

    private static double[] validateInitialParams(double[] params, String name) {
        for (int i = 0; i < params.length; i++) {
            if (!Double.isFinite(params[i])) {
                throw new IllegalArgumentException(name + "[" + i + "] must be finite");
            }
        }
        return params;
    }

    private static String[] normalizeParameterNames(int paramCount, String[] parameterNames) {
        if (parameterNames == null || parameterNames.length == 0) {
            String[] generated = new String[paramCount];
            for (int i = 0; i < paramCount; i++) {
                generated[i] = "param" + i;
            }
            return generated;
        }
        if (parameterNames.length != paramCount) {
            throw new IllegalArgumentException(
                "parameterNames length=" + parameterNames.length + " but expected " + paramCount);
        }
        String[] copy = Arrays.copyOf(parameterNames, parameterNames.length);
        for (int i = 0; i < copy.length; i++) {
            if (copy[i] == null || copy[i].isBlank()) {
                throw new IllegalArgumentException("parameterNames[" + i + "] must not be blank");
            }
        }
        return copy;
    }

    private record PreparedEvaluation(double[] params,
                                      EvaluationContext context,
                                      KalmanSSM model,
                                      InitialState initialState) {
    }

    protected static class EvaluationContext {
        protected final KalmanSSM model;
        protected InitialState initialState;

        protected EvaluationContext(KalmanSSM model) {
            this.model = model;
        }
    }
}