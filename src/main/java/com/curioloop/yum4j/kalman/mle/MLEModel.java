package com.curioloop.yum4j.kalman.mle;

import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
import com.curioloop.yum4j.linalg.Decomposer;
import com.curioloop.yum4j.linalg.mat.LU;
import com.curioloop.yum4j.linalg.mat.SVD;
import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.optim.NumericalGradient;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;
import com.curioloop.yum4j.optim.cmaes.CMAESProblem;
import com.curioloop.yum4j.optim.lbfgsb.LBFGSBProblem;
import com.curioloop.yum4j.optim.slsqp.SLSQPProblem;
import com.curioloop.yum4j.optim.subplex.SubplexProblem;

import java.util.Arrays;
import java.util.Objects;

public abstract class MLEModel<F, S> implements MLEBackend<F, S> {

    private static final FilterSpec LIKELIHOOD_SPEC = FilterSpec.full().without(
        FilterSpec.Storage.FORECAST,
        FilterSpec.Storage.PREDICTED_STATE,
        FilterSpec.Storage.FILTERED_STATE,
        FilterSpec.Storage.KALMAN_GAIN);

    private static final FilterSpec PREDICTION_SPEC = FilterSpec.full().without(
        FilterSpec.Storage.FILTERED_STATE,
        FilterSpec.Storage.LIKELIHOOD,
        FilterSpec.Storage.KALMAN_GAIN);

    private final StateSpaceModel templateModel;
    private final StateSpaceModel workingModel;
    private final double[] startParams;
    private final String[] parameterNames;

    protected MLEModel(StateSpaceModel templateModel,
                       double[] startParams,
                       String... parameterNames) {
        Objects.requireNonNull(templateModel, "templateModel");
        requireSupported(templateModel);
        this.templateModel = cloneWithLayout(templateModel);
        this.workingModel = cloneWithLayout(templateModel);
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
        return workingModel.complex();
    }

    public final int observationCount() {
        return workingModel.observationCount();
    }

    public final int observationDimension() {
        return workingModel.observationDimension();
    }

    public final int stateCount() {
        return workingModel.stateCount();
    }

    public double[] transformParams(double[] unconstrainedParams) {
        return validateParams(copyParams(unconstrainedParams, "unconstrainedParams"), "unconstrainedParams");
    }

    public double[] untransformParams(double[] params) {
        return validateParams(copyParams(params, "params"), "params");
    }

    protected abstract void updateModel(double[] params, StateSpaceModel model);

    protected abstract StateInitialization initialization(double[] params, StateSpaceModel model);

    protected int likelihoodBurn(double[] params, StateSpaceModel model) {
        return 0;
    }

    protected FilterSpec likelihoodSpec() {
        return LIKELIHOOD_SPEC;
    }

    protected double[] adjustLogLikelihoodObs(F filterResult,
                                              double[] logLikelihoodObs,
                                              double[] params,
                                              StateSpaceModel model,
                                              int burn) {
        return logLikelihoodObs;
    }

    protected double adjustedLogLikelihood(F filterResult,
                                           double[] params,
                                           StateSpaceModel model,
                                           int burn) {
        return logLikelihood(filterResult, burn);
    }

    public final double logLikelihood(double[] params) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        int burn = likelihoodBurn(prepared.params, prepared.model);
        F filterResult = filterBorrowed(prepared.model, prepared.initialization, likelihoodSpec());
        return adjustedLogLikelihood(filterResult, prepared.params, prepared.model, burn);
    }

    public final double[] logLikelihoodObs(double[] params) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        int burn = likelihoodBurn(prepared.params, prepared.model);
        F filterResult = filterBorrowed(prepared.model, prepared.initialization, likelihoodSpec());
        double[] logLikelihoodObs = adjustLogLikelihoodObs(
            filterResult,
            logLikelihoodObs(filterResult),
            prepared.params,
            prepared.model,
            burn);
        zeroLeadingLogLikelihood(logLikelihoodObs, burn);
        return logLikelihoodObs;
    }

    public final F filter(double[] params) {
        return filter(params, FilterSpec.defaults());
    }

    public final F filter(double[] params, FilterSpec spec) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        return filter(prepared.model, prepared.initialization, spec == null ? FilterSpec.defaults() : spec);
    }

    public final S smooth(double[] params) {
        return smooth(params, SmootherSpec.conventional());
    }

    public final S smooth(double[] params, SmootherSpec spec) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        return smooth(prepared.model, prepared.initialization, spec == null ? SmootherSpec.conventional() : spec);
    }

    public final F predict(double[] params) {
        return predict(params, PREDICTION_SPEC);
    }

    public final F predict(double[] params, FilterSpec spec) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        return filter(prepared.model, prepared.initialization, spec == null ? PREDICTION_SPEC : spec);
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
        return NumericalHessian.central((x, n) -> logLikelihood(copyParams(x, n, "params")), point);
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
            double[] fPlus = logLikelihoodObs(x);

            x[index] = xi - h;
            double[] fMinus = logLikelihoodObs(x);

            x[index] = xi;
            for (int t = 0; t < nobs; t++) {
                scoreObs[t * point.length + index] = (fPlus[t] - fMinus[t]) / (2.0 * h);
            }
        }
        return scoreObs;
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

    public final StateSpaceModel snapshotModel(double[] params) {
        PreparedEvaluation prepared = prepareNoCopy(params, "params");
        return StateSpaceModel.copyOf(prepared.model);
    }

    private PreparedEvaluation prepare(double[] params) {
        return prepareNoCopy(copyParams(params, "params"), "params");
    }

    private PreparedEvaluation prepareNoCopy(double[] params, String name) {
        double[] resolvedParams = validateParamsNoCopy(params, name);
        resetWorkingModel();
        updateModel(resolvedParams, workingModel);
        StateInitialization resolvedInitialization = Objects.requireNonNull(
            initialization(resolvedParams, workingModel),
            "initialization");
        return new PreparedEvaluation(resolvedParams, workingModel, resolvedInitialization);
    }

    private static void zeroLeadingLogLikelihood(double[] logLikelihoodObs, int burn) {
        int limit = Math.min(Math.max(0, burn), logLikelihoodObs.length);
        Arrays.fill(logLikelihoodObs, 0, limit, 0.0);
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
        for (int row = 0; row < dimension; row++) {
            for (int col = 0; col < dimension; col++) {
                double sum = 0.0;
                for (int inner = 0; inner < dimension; inner++) {
                    sum += left[row * dimension + inner] * right[inner * dimension + col];
                }
                product[row * dimension + col] = sum;
            }
        }
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

    private void resetWorkingModel() {
        copyArray(templateModel.design, workingModel.design);
        copyArray(templateModel.obsIntercept, workingModel.obsIntercept);
        copyArray(templateModel.obsCov, workingModel.obsCov);
        copyArray(templateModel.transition, workingModel.transition);
        copyArray(templateModel.stateIntercept, workingModel.stateIntercept);
        copyArray(templateModel.selection, workingModel.selection);
        copyArray(templateModel.stateCov, workingModel.stateCov);
        copyArray(templateModel.endog, workingModel.endog);
        copyArray(templateModel.missing, workingModel.missing);
        copyArray(templateModel.nmissing, workingModel.nmissing);
    }

    private static StateSpaceModel cloneWithLayout(StateSpaceModel source) {
        int secondIndex = source.observationCount() > 1 ? 1 : 0;
        boolean[] missing = source.getMissing();
        int[] nmissing = source.getNmissing();
        StateSpaceModel.Builder builder = (source.complex()
            ? StateSpaceModel.complexBuilder(
                source.observationDimension(),
                source.stateCount(),
                source.stateDisturbanceCount(),
                source.observationCount())
            : StateSpaceModel.builder(
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

    private static void copyArray(double[] source, double[] target) {
        if (source == null || target == null) {
            if (source != target) {
                throw new IllegalStateException("state-space template and working arrays must agree on nullability");
            }
            return;
        }
        System.arraycopy(source, 0, target, 0, source.length);
    }

    private static void copyArray(boolean[] source, boolean[] target) {
        if (source == null || target == null) {
            if (source != target) {
                throw new IllegalStateException("state-space template and working missing flags must agree on nullability");
            }
            return;
        }
        System.arraycopy(source, 0, target, 0, source.length);
    }

    private static void copyArray(int[] source, int[] target) {
        if (source == null || target == null) {
            if (source != target) {
                throw new IllegalStateException("state-space template and working missing counts must agree on nullability");
            }
            return;
        }
        System.arraycopy(source, 0, target, 0, source.length);
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

    private double[] validateParamsNoCopy(double[] params, String name) {
        Objects.requireNonNull(params, name);
        return validateParams(params, name);
    }

    private double[] validateParams(double[] params, String name) {
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

    private record PreparedEvaluation(double[] params, StateSpaceModel model, StateInitialization initialization) {
    }
}