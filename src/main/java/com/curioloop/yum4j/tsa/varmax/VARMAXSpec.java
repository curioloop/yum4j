package com.curioloop.yum4j.tsa.varmax;

import com.curioloop.yum4j.stats.tool.TrendTerms;

import java.util.Objects;

public final class VARMAXSpec {

    private static final double DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE = 1e6;
    private static final int DEFAULT_TREND_OFFSET = 1;
    private static final int DEFAULT_OPTION_MASK = bit(VARMAXOption.ENFORCE_STATIONARITY)
        | bit(VARMAXOption.ENFORCE_INVERTIBILITY);

    private final VARMAXOrder order;
    private final double[][] endog;
    private final double[][] exog;
    private final double[] observationIntercept;
    private final int[] trendPowers;
    private final int trendOffset;
    private final VARMAXErrorCovariance errorCovariance;
    private final boolean measurementError;
    private final int logLikelihoodBurn;
    private final Double approximateDiffuseVariance;
    private final int optionMask;
    private final String[] endogNames;
    private final String[] exogNames;

    private VARMAXSpec(Builder builder) {
        this.order = Objects.requireNonNull(builder.order, "order");
        this.endog = copyMatrix(Objects.requireNonNull(builder.endog, "endog"), "endog");
        this.exog = copyOptionalMatrix(builder.exog, "exog");
        this.observationIntercept = normalizeObservationIntercept(builder.observationIntercept, endog[0].length);
        this.trendPowers = TrendTerms.normalizePowers(builder.trendPowers);
        this.trendOffset = builder.trendOffset;
        this.errorCovariance = builder.errorCovariance;
        this.measurementError = builder.measurementError;
        this.logLikelihoodBurn = builder.logLikelihoodBurn;
        this.approximateDiffuseVariance = builder.approximateDiffuseVariance;
        this.optionMask = builder.optionMask;
        this.endogNames = normalizeNames(builder.endogNames, endog[0].length, "y");
        this.exogNames = normalizeNames(builder.exogNames, exog == null ? 0 : exog[0].length, "x");
        validate();
    }

    public static Builder builder(VARMAXOrder order, double[][] endog) {
        return new Builder(order, endog);
    }

    public VARMAXOrder order() {
        return order;
    }

    public double[][] endog() {
        return copyMatrix(endog, "endog");
    }

    public double[][] exog() {
        return copyOptionalMatrix(exog, "exog");
    }

    public boolean hasExog() {
        return exog != null && exog.length > 0;
    }

    public int exogDimension() {
        return hasExog() ? exog[0].length : 0;
    }

    public double[] observationIntercept() {
        return observationIntercept.clone();
    }

    public int[] trendPowers() {
        return trendPowers.clone();
    }

    public boolean hasTrend() {
        return trendPowers.length > 0;
    }

    public int trendOffset() {
        return trendOffset;
    }

    public VARMAXErrorCovariance errorCovariance() {
        return errorCovariance;
    }

    public boolean measurementError() {
        return measurementError;
    }

    public int logLikelihoodBurn() {
        return logLikelihoodBurn;
    }

    public boolean hasApproximateDiffuseVariance() {
        return approximateDiffuseVariance != null;
    }

    public double approximateDiffuseVariance() {
        return approximateDiffuseVariance == null ? DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE : approximateDiffuseVariance;
    }

    public VARMAXOption[] options() {
        VARMAXOption[] values = VARMAXOption.values();
        VARMAXOption[] selected = new VARMAXOption[Integer.bitCount(optionMask)];
        int count = 0;
        for (VARMAXOption option : values) {
            if (hasOption(option)) {
                selected[count++] = option;
            }
        }
        return selected;
    }

    public boolean hasOption(VARMAXOption option) {
        return (optionMask & bit(Objects.requireNonNull(option, "option"))) != 0;
    }

    public String[] endogNames() {
        return endogNames.clone();
    }

    public String[] exogNames() {
        return exogNames.clone();
    }

    public int observationCount() {
        return endog.length;
    }

    public int observationDimension() {
        return endog[0].length;
    }

    public int parameterCount() {
        int kEndog = observationDimension();
        return kEndog * trendPowers.length
            + kEndog * kEndog * order.autoregressive()
            + kEndog * kEndog * order.movingAverage()
            + kEndog * exogDimension()
            + (errorCovariance == VARMAXErrorCovariance.DIAGONAL
                ? kEndog
                : kEndog * (kEndog + 1) / 2)
            + (measurementError ? kEndog : 0);
    }

    private void validate() {
        if (endog.length == 0) {
            throw new IllegalArgumentException("endog must not be empty");
        }
        if (endog[0].length < 2) {
            throw new IllegalArgumentException("VARMAX requires at least two endogenous variables");
        }
        if (hasExog() && exog.length != endog.length) {
            throw new IllegalArgumentException("exog row count must match endog length");
        }
        TrendTerms.validateOffset(trendOffset);
        if (logLikelihoodBurn < 0 || logLikelihoodBurn > endog.length) {
            throw new IllegalArgumentException("logLikelihoodBurn must be between 0 and observation count");
        }
        if (approximateDiffuseVariance != null
            && (!(approximateDiffuseVariance > 0.0) || !Double.isFinite(approximateDiffuseVariance))) {
            throw new IllegalArgumentException("approximate diffuse variance must be positive and finite");
        }
    }

    private static double[][] copyMatrix(double[][] matrix, String name) {
        Objects.requireNonNull(matrix, name);
        if (matrix.length == 0) {
            return new double[0][];
        }
        int width = -1;
        double[][] copy = new double[matrix.length][];
        for (int row = 0; row < matrix.length; row++) {
            double[] source = Objects.requireNonNull(matrix[row], name + " row");
            if (width < 0) {
                width = source.length;
                if (width == 0) {
                    throw new IllegalArgumentException(name + " rows must not be empty");
                }
            } else if (source.length != width) {
                throw new IllegalArgumentException(name + " rows must all have the same length");
            }
            copy[row] = source.clone();
        }
        return copy;
    }

    private static double[][] copyOptionalMatrix(double[][] matrix, String name) {
        return matrix == null ? null : copyMatrix(matrix, name);
    }

    private static double[] normalizeObservationIntercept(double[] values, int kEndog) {
        if (values == null) {
            return new double[kEndog];
        }
        if (values.length != kEndog) {
            throw new IllegalArgumentException("observationIntercept length=" + values.length + " but expected " + kEndog);
        }
        double[] copy = values.clone();
        for (int i = 0; i < copy.length; i++) {
            if (!Double.isFinite(copy[i])) {
                throw new IllegalArgumentException("observationIntercept[" + i + "] must be finite");
            }
        }
        return copy;
    }

    private static String[] normalizeNames(String[] names, int count, String prefix) {
        if (count == 0) {
            return new String[0];
        }
        if (names == null || names.length == 0) {
            String[] generated = new String[count];
            for (int i = 0; i < count; i++) {
                generated[i] = prefix + (i + 1);
            }
            return generated;
        }
        if (names.length != count) {
            throw new IllegalArgumentException(prefix + "Names length=" + names.length + " but expected " + count);
        }
        String[] copy = names.clone();
        for (int i = 0; i < copy.length; i++) {
            if (copy[i] == null || copy[i].isBlank()) {
                throw new IllegalArgumentException(prefix + "Names[" + i + "] must not be blank");
            }
        }
        return copy;
    }

    private static int bit(VARMAXOption option) {
        return 1 << option.ordinal();
    }

    public static final class Builder {
        private final VARMAXOrder order;
        private final double[][] endog;
        private double[][] exog;
        private double[] observationIntercept;
        private int[] trendPowers = new int[]{0};
        private int trendOffset = DEFAULT_TREND_OFFSET;
        private VARMAXErrorCovariance errorCovariance = VARMAXErrorCovariance.UNSTRUCTURED;
        private boolean measurementError;
        private int logLikelihoodBurn;
        private Double approximateDiffuseVariance;
        private int optionMask = DEFAULT_OPTION_MASK;
        private String[] endogNames;
        private String[] exogNames;

        private Builder(VARMAXOrder order, double[][] endog) {
            this.order = Objects.requireNonNull(order, "order");
            this.endog = Objects.requireNonNull(endog, "endog");
        }

        public Builder exog(double[][] exog) {
            this.exog = exog;
            return this;
        }

        public Builder observationIntercept(double... observationIntercept) {
            this.observationIntercept = observationIntercept;
            return this;
        }

        public Builder trend(TrendTerms trend) {
            this.trendPowers = Objects.requireNonNull(trend, "trend").powers();
            return this;
        }

        public Builder trend(String trend) {
            return trend(TrendTerms.fromString(trend));
        }

        public Builder trendPowers(int... trendPowers) {
            this.trendPowers = trendPowers;
            return this;
        }

        public Builder trendOffset(int trendOffset) {
            this.trendOffset = trendOffset;
            return this;
        }

        public Builder errorCovariance(VARMAXErrorCovariance errorCovariance) {
            this.errorCovariance = Objects.requireNonNull(errorCovariance, "errorCovariance");
            return this;
        }

        public Builder measurementError(boolean measurementError) {
            this.measurementError = measurementError;
            return this;
        }

        public Builder logLikelihoodBurn(int logLikelihoodBurn) {
            this.logLikelihoodBurn = logLikelihoodBurn;
            return this;
        }

        public Builder approximateDiffuseVariance(double approximateDiffuseVariance) {
            this.approximateDiffuseVariance = approximateDiffuseVariance;
            return this;
        }

        public Builder include(VARMAXOption option) {
            optionMask |= bit(Objects.requireNonNull(option, "option"));
            return this;
        }

        public Builder exclude(VARMAXOption option) {
            optionMask &= ~bit(Objects.requireNonNull(option, "option"));
            return this;
        }

        public Builder endogNames(String... endogNames) {
            this.endogNames = endogNames;
            return this;
        }

        public Builder exogNames(String... exogNames) {
            this.exogNames = exogNames;
            return this;
        }

        public VARMAXSpec build() {
            return new VARMAXSpec(this);
        }
    }
}
