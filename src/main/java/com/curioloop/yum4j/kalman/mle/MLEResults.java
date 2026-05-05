package com.curioloop.yum4j.kalman.mle;

import com.curioloop.yum4j.optim.Optimization;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

public abstract class MLEResults {

    private static final double DEFAULT_CONFIDENCE_ALPHA = 0.05;
    private static final DateTimeFormatter SUMMARY_DATE_FORMAT = DateTimeFormatter.ofPattern("MM-dd-yyyy");
    private static final DateTimeFormatter SUMMARY_TIME_FORMAT = DateTimeFormatter.ofPattern("HH:mm:ss");
    private static final double STANDARD_NORMAL_CDF_P = 0.2316419;
    private static final double STANDARD_NORMAL_CDF_B1 = 0.319381530;
    private static final double STANDARD_NORMAL_CDF_B2 = -0.356563782;
    private static final double STANDARD_NORMAL_CDF_B3 = 1.781477937;
    private static final double STANDARD_NORMAL_CDF_B4 = -1.821255978;
    private static final double STANDARD_NORMAL_CDF_B5 = 1.330274429;
    private static final double[] INVERSE_NORMAL_A = {
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00
    };
    private static final double[] INVERSE_NORMAL_B = {
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01
    };
    private static final double[] INVERSE_NORMAL_C = {
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00
    };
    private static final double[] INVERSE_NORMAL_D = {
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00
    };
    private static final double INVERSE_NORMAL_TAIL_PROBABILITY = 0.02425;

    private final Optimization optimization;
    private final double[] params;
    private final double[] unconstrainedParams;
    private final String[] parameterNames;
    private final MLECovariance covType;
    private double[] covParamsDefault;
    private double[] covParamsOim;
    private double[] covParamsApprox;
    private double[] covParamsOpg;
    private double[] covParamsRobustOim;
    private double[] covParamsRobustApprox;
    private double[] bse;
    private double[] bseOim;
    private double[] bseApprox;
    private double[] bseOpg;
    private double[] bseRobustOim;
    private double[] bseRobustApprox;
    private double[] zvalues;
    private double[] pvalues;

    protected MLEResults(Optimization optimization,
                         double[] params,
                         double[] unconstrainedParams,
                         String[] parameterNames,
                         MLECovariance covType) {
        this.optimization = optimization;
        this.params = Arrays.copyOf(Objects.requireNonNull(params, "params"), params.length);
        this.unconstrainedParams = Arrays.copyOf(Objects.requireNonNull(unconstrainedParams, "unconstrainedParams"), unconstrainedParams.length);
        this.parameterNames = Arrays.copyOf(Objects.requireNonNull(parameterNames, "parameterNames"), parameterNames.length);
        this.covType = Objects.requireNonNull(covType, "covType");
    }

    public final Optimization optimization() {
        return optimization;
    }

    public final double[] params() {
        return Arrays.copyOf(params, params.length);
    }

    public final double[] unconstrainedParams() {
        return Arrays.copyOf(unconstrainedParams, unconstrainedParams.length);
    }

    public final String[] parameterNames() {
        return Arrays.copyOf(parameterNames, parameterNames.length);
    }

    public final String covType() {
        return covType.id();
    }

    public final double logLikelihood() {
        return logLikelihoodInternal();
    }

    public final double scale() {
        return scaleInternal();
    }

    public final int nobsEffective() {
        return nobsEffectiveInternal();
    }

    public final int dfModel() {
        return dfModelInternal();
    }

    public final int dfResid() {
        return nobsEffective() - dfModel();
    }

    public final double[] covParams() {
        return covParamsDefault();
    }

    public final double[] covParamsDefault() {
        if (covParamsDefault == null) {
            covParamsDefault = switch (covType) {
                case APPROX -> computeCovParamsApprox();
                case OIM -> computeCovParamsOim();
                case OPG -> computeCovParamsOpg();
                case ROBUST, ROBUST_OIM -> computeCovParamsRobustOim();
                case ROBUST_APPROX -> computeCovParamsRobustApprox();
            };
        }
        return Arrays.copyOf(covParamsDefault, covParamsDefault.length);
    }

    public final double[] covParamsOim() {
        if (covParamsOim == null) {
            covParamsOim = computeCovParamsOim();
        }
        return Arrays.copyOf(covParamsOim, covParamsOim.length);
    }

    public final double[] covParamsApprox() {
        if (covParamsApprox == null) {
            covParamsApprox = computeCovParamsApprox();
        }
        return Arrays.copyOf(covParamsApprox, covParamsApprox.length);
    }

    public final double[] covParamsOpg() {
        if (covParamsOpg == null) {
            covParamsOpg = computeCovParamsOpg();
        }
        return Arrays.copyOf(covParamsOpg, covParamsOpg.length);
    }

    public final double[] covParamsRobustOim() {
        if (covParamsRobustOim == null) {
            covParamsRobustOim = computeCovParamsRobustOim();
        }
        return Arrays.copyOf(covParamsRobustOim, covParamsRobustOim.length);
    }

    public final double[] covParamsRobust() {
        return covParamsRobustOim();
    }

    public final double[] covParamsRobustApprox() {
        if (covParamsRobustApprox == null) {
            covParamsRobustApprox = computeCovParamsRobustApprox();
        }
        return Arrays.copyOf(covParamsRobustApprox, covParamsRobustApprox.length);
    }

    public final double[] bse() {
        if (bse == null) {
            bse = standardErrors(covParams());
        }
        return Arrays.copyOf(bse, bse.length);
    }

    public final double[] bseDefault() {
        return bse();
    }

    public final double[] bseOim() {
        if (bseOim == null) {
            bseOim = standardErrors(covParamsOim());
        }
        return Arrays.copyOf(bseOim, bseOim.length);
    }

    public final double[] bseApprox() {
        if (bseApprox == null) {
            bseApprox = standardErrors(covParamsApprox());
        }
        return Arrays.copyOf(bseApprox, bseApprox.length);
    }

    public final double[] bseOpg() {
        if (bseOpg == null) {
            bseOpg = standardErrors(covParamsOpg());
        }
        return Arrays.copyOf(bseOpg, bseOpg.length);
    }

    public final double[] bseRobustOim() {
        if (bseRobustOim == null) {
            bseRobustOim = standardErrors(covParamsRobustOim());
        }
        return Arrays.copyOf(bseRobustOim, bseRobustOim.length);
    }

    public final double[] bseRobust() {
        return bseRobustOim();
    }

    public final double[] bseRobustApprox() {
        if (bseRobustApprox == null) {
            bseRobustApprox = standardErrors(covParamsRobustApprox());
        }
        return Arrays.copyOf(bseRobustApprox, bseRobustApprox.length);
    }

    public final double[] tvalues() {
        return zvalues();
    }

    public final double[] zvalues() {
        if (zvalues == null) {
            double[] standardErrors = bse();
            zvalues = new double[params.length];
            for (int i = 0; i < params.length; i++) {
                zvalues[i] = params[i] / standardErrors[i];
            }
        }
        return Arrays.copyOf(zvalues, zvalues.length);
    }

    public final double[] pvalues() {
        if (pvalues == null) {
            double[] statistics = zvalues();
            pvalues = new double[statistics.length];
            Arrays.fill(pvalues, Double.NaN);
            for (int i = 0; i < statistics.length; i++) {
                double statistic = statistics[i];
                if (Double.isNaN(statistic)) {
                    continue;
                }
                pvalues[i] = 2.0 * standardNormalSurvivalProbability(Math.abs(statistic));
            }
        }
        return Arrays.copyOf(pvalues, pvalues.length);
    }

    public final double[][] confInt() {
        return confInt(DEFAULT_CONFIDENCE_ALPHA);
    }

    public final double[][] confInt(double alpha) {
        if (!(alpha > 0.0 && alpha < 1.0)) {
            throw new IllegalArgumentException("alpha must be between 0 and 1");
        }

        double criticalValue = inverseStandardNormal(1.0 - alpha / 2.0);
        double[] standardErrors = bse();
        double[][] intervals = new double[params.length][2];
        for (int i = 0; i < params.length; i++) {
            double margin = criticalValue * standardErrors[i];
            intervals[i][0] = params[i] - margin;
            intervals[i][1] = params[i] + margin;
        }
        return intervals;
    }

    public String summary() {
        return summary(DEFAULT_CONFIDENCE_ALPHA);
    }

    public String summary(double alpha) {
        LocalDateTime now = LocalDateTime.now();
        double[][] intervals = confInt(alpha);
        StringBuilder builder = new StringBuilder();
        builder.append("Statespace Model Results\n");
        builder.append("========================\n");
        appendSummaryLine(builder, "Dep. Variable:", dependentVariableNameInternal());
        appendSummaryLine(builder, "Model:", modelNameInternal());
        appendSummaryLine(builder, "Date:", SUMMARY_DATE_FORMAT.format(now));
        appendSummaryLine(builder, "Time:", SUMMARY_TIME_FORMAT.format(now));
        appendSummaryLine(builder, "Sample:", "0 - " + observationCountInternal());
        appendSummaryLine(builder, "No. Observations:", Integer.toString(observationCountInternal()));
        appendSummaryLine(builder, "Log Likelihood:", formatSummaryDouble(logLikelihood()));
        appendSummaryLine(builder, "AIC:", formatSummaryDouble(aic()));
        appendSummaryLine(builder, "BIC:", formatSummaryDouble(bic()));
        appendSummaryLine(builder, "HQIC:", formatSummaryDouble(hqic()));
        appendSummaryLine(builder, "Covariance Type:", covType());
        if (summaryIncludesScaleInternal()) {
            appendSummaryLine(builder, "Scale:", formatSummaryDouble(scale()));
        }

        builder.append('\n');
        builder.append(String.format(Locale.ROOT,
            "%-20s %12s %12s %12s %12s %14s %14s%n",
            "coef", "estimate", "std err", "z", "P>|z|", "[0.025", "0.975]"));

        double[] standardErrors = bse();
        double[] statistics = zvalues();
        double[] probabilities = pvalues();
        for (int i = 0; i < params.length; i++) {
            String label = i < parameterNames.length ? parameterNames[i] : "param." + i;
            builder.append(String.format(Locale.ROOT,
                "%-20s %12.6f %12.6f %12.6f %12.6f %14.6f %14.6f%n",
                label,
                params[i],
                standardErrors[i],
                statistics[i],
                probabilities[i],
                intervals[i][0],
                intervals[i][1]));
        }

        builder.append('\n');
        builder.append(summaryDiagnosticsTextInternal());
        if (!summaryDiagnosticsTextInternal().endsWith("\n")) {
            builder.append('\n');
        }
        return builder.toString();
    }

    public final double aic() {
        return -2.0 * logLikelihood() + 2.0 * dfModel();
    }

    public final double aicc() {
        return aicc(logLikelihood(), nobsEffective(), dfModel());
    }

    public final double bic() {
        return bic(logLikelihood(), nobsEffective(), dfModel());
    }

    public final double hqic() {
        return hqic(logLikelihood(), nobsEffective(), dfModel());
    }

    protected final double[] paramsInternal() {
        return params;
    }

    protected final double[] unconstrainedParamsInternal() {
        return unconstrainedParams;
    }

    protected abstract double[] computeCovParamsOim();

    protected abstract double[] computeCovParamsApprox();

    protected abstract double[] computeCovParamsOpg();

    protected abstract double[] computeCovParamsRobustOim();

    protected abstract double[] computeCovParamsRobustApprox();

    protected abstract double logLikelihoodInternal();

    protected abstract double scaleInternal();

    protected abstract int nobsEffectiveInternal();

    protected abstract int dfModelInternal();

    protected abstract int observationCountInternal();

    protected abstract String modelNameInternal();

    protected String dependentVariableNameInternal() {
        return "endog";
    }

    protected boolean summaryIncludesScaleInternal() {
        return false;
    }

    protected String summaryDiagnosticsTextInternal() {
        return "Diagnostics: unavailable in current results surface";
    }

    protected static double[] standardErrors(double[] covariance) {
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

    private static double aicc(double logLikelihood, int nobs, int dfModel) {
        double dofEffective = nobs - dfModel - 1.0;
        if (dofEffective <= 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return -2.0 * logLikelihood + 2.0 * dfModel * nobs / dofEffective;
    }

    private static double bic(double logLikelihood, int nobs, int dfModel) {
        if (nobs <= 0) {
            return Double.POSITIVE_INFINITY;
        }
        return -2.0 * logLikelihood + Math.log(nobs) * dfModel;
    }

    private static double hqic(double logLikelihood, int nobs, int dfModel) {
        if (nobs <= 1) {
            return Double.POSITIVE_INFINITY;
        }
        return -2.0 * logLikelihood + 2.0 * Math.log(Math.log(nobs)) * dfModel;
    }

    private static void appendSummaryLine(StringBuilder builder, String label, String value) {
        builder.append(String.format(Locale.ROOT, "%-18s %s%n", label, value));
    }

    private static String formatSummaryDouble(double value) {
        return String.format(Locale.ROOT, "%#.3f", value);
    }

    private static double standardNormalSurvivalProbability(double value) {
        return 1.0 - standardNormalCdf(value);
    }

    private static double standardNormalCdf(double value) {
        if (Double.isNaN(value)) {
            return Double.NaN;
        }
        if (value == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        if (value == Double.NEGATIVE_INFINITY) {
            return 0.0;
        }

        double x = Math.abs(value);
        double t = 1.0 / (1.0 + STANDARD_NORMAL_CDF_P * x);
        double density = Math.exp(-0.5 * x * x) / Math.sqrt(2.0 * Math.PI);
        double polynomial = (((((STANDARD_NORMAL_CDF_B5 * t + STANDARD_NORMAL_CDF_B4) * t
            + STANDARD_NORMAL_CDF_B3) * t + STANDARD_NORMAL_CDF_B2) * t
            + STANDARD_NORMAL_CDF_B1) * t);
        double cdf = 1.0 - density * polynomial;
        return value >= 0.0 ? cdf : 1.0 - cdf;
    }

    private static double inverseStandardNormal(double probability) {
        if (!(probability > 0.0 && probability < 1.0)) {
            throw new IllegalArgumentException("probability must be between 0 and 1");
        }

        if (probability < INVERSE_NORMAL_TAIL_PROBABILITY) {
            double q = Math.sqrt(-2.0 * Math.log(probability));
            return (((((INVERSE_NORMAL_C[0] * q + INVERSE_NORMAL_C[1]) * q + INVERSE_NORMAL_C[2]) * q
                + INVERSE_NORMAL_C[3]) * q + INVERSE_NORMAL_C[4]) * q + INVERSE_NORMAL_C[5]) /
                ((((INVERSE_NORMAL_D[0] * q + INVERSE_NORMAL_D[1]) * q + INVERSE_NORMAL_D[2]) * q
                    + INVERSE_NORMAL_D[3]) * q + 1.0);
        }
        if (probability > 1.0 - INVERSE_NORMAL_TAIL_PROBABILITY) {
            double q = Math.sqrt(-2.0 * Math.log(1.0 - probability));
            return -(((((INVERSE_NORMAL_C[0] * q + INVERSE_NORMAL_C[1]) * q + INVERSE_NORMAL_C[2]) * q
                + INVERSE_NORMAL_C[3]) * q + INVERSE_NORMAL_C[4]) * q + INVERSE_NORMAL_C[5]) /
                ((((INVERSE_NORMAL_D[0] * q + INVERSE_NORMAL_D[1]) * q + INVERSE_NORMAL_D[2]) * q
                    + INVERSE_NORMAL_D[3]) * q + 1.0);
        }

        double q = probability - 0.5;
        double r = q * q;
        return (((((INVERSE_NORMAL_A[0] * r + INVERSE_NORMAL_A[1]) * r + INVERSE_NORMAL_A[2]) * r
            + INVERSE_NORMAL_A[3]) * r + INVERSE_NORMAL_A[4]) * r + INVERSE_NORMAL_A[5]) * q /
            (((((INVERSE_NORMAL_B[0] * r + INVERSE_NORMAL_B[1]) * r + INVERSE_NORMAL_B[2]) * r
                + INVERSE_NORMAL_B[3]) * r + INVERSE_NORMAL_B[4]) * r + 1.0);
    }
}