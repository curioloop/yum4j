package com.curioloop.yum4j.ssm.kalman.mle;

import com.curioloop.yum4j.math.Normal;
import com.curioloop.yum4j.optim.Optimization;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

public abstract class MLEResults implements com.curioloop.yum4j.stats.tool.InformationCriterion {

    public enum Covariance {
        OPG("opg"),
        OIM("oim"),
        APPROX("approx"),
        ROBUST("robust"),
        ROBUST_OIM("robust_oim"),
        ROBUST_APPROX("robust_approx");

        private final String id;

        Covariance(String id) {
            this.id = id;
        }

        public String id() {
            return id;
        }

        public static Covariance fromId(String id) {
            if (id == null || id.isBlank()) {
                throw new IllegalArgumentException("covariance type must not be blank");
            }
            String normalized = id.toLowerCase(Locale.ROOT);
            for (Covariance value : values()) {
                if (value.id.equals(normalized)) {
                    return value;
                }
            }
            throw new IllegalArgumentException("Unsupported covariance type: " + id);
        }
    }

    private static final double DEFAULT_CONFIDENCE_ALPHA = 0.05;
    private static final DateTimeFormatter SUMMARY_DATE_FORMAT = DateTimeFormatter.ofPattern("MM-dd-yyyy");
    private static final DateTimeFormatter SUMMARY_TIME_FORMAT = DateTimeFormatter.ofPattern("HH:mm:ss");
    private final Optimization optimization;
    private final double[] params;
    private final double[] unconstrainedParams;
    private final String[] parameterNames;
    private final Covariance covType;
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
                         Covariance covType) {
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

    // ── InformationCriterion interface ────────────────────────────────────────
    @Override public int observationCount() { return nobsEffective(); }
    @Override public int parameterCount()   { return dfModel(); }

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
                pvalues[i] = 2.0 * Normal.ccdf(Math.abs(statistic));
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

        double criticalValue = Normal.inv(1.0 - alpha / 2.0);
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

    private static void appendSummaryLine(StringBuilder builder, String label, String value) {
        builder.append(String.format(Locale.ROOT, "%-18s %s%n", label, value));
    }

    private static String formatSummaryDouble(double value) {
        return String.format(Locale.ROOT, "%#.3f", value);
    }

}