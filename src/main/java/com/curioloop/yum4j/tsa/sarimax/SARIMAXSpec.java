package com.curioloop.yum4j.tsa.sarimax;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Objects;

/**
 * Immutable configuration for a univariate real-valued SARIMAX model.
 */
public final class SARIMAXSpec {

    private static final double DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE = 1e6;

    private final ARIMAOrder order;
    private final SeasonalOrder seasonalOrder;
    private final double[] endog;
    private final double[][] exog;
    private final int[] trendPowers;
    private final int[] autoregressiveLags;
    private final int[] movingAverageLags;
    private final int[] seasonalAutoregressiveLags;
    private final int[] seasonalMovingAverageLags;
    private final boolean measurementError;
    private final boolean concentrateScale;
    private final boolean mleRegression;
    private final boolean timeVaryingRegression;
    private final boolean simpleDifferencing;
    private final boolean hamiltonRepresentation;
    private final Double approximateDiffuseVariance;
    private final EnumSet<SARIMAXOption> options;

    private SARIMAXSpec(Builder builder) {
        this.order = Objects.requireNonNull(builder.order, "order");
        this.seasonalOrder = builder.seasonalOrder == null ? SeasonalOrder.none() : builder.seasonalOrder;
        this.endog = copyEndog(builder.endog);
        this.exog = copyExog(builder.exog);
        this.trendPowers = normalizeTrendPowers(builder.trendPowers);
        this.autoregressiveLags = normalizeLags(order.autoregressive(), builder.autoregressiveLags, "autoregressive");
        this.movingAverageLags = normalizeLags(order.movingAverage(), builder.movingAverageLags, "movingAverage");
        this.seasonalAutoregressiveLags = seasonalLags(seasonalOrder.autoregressive(), seasonalOrder.period());
        this.seasonalMovingAverageLags = seasonalLags(seasonalOrder.movingAverage(), seasonalOrder.period());
        this.measurementError = builder.measurementError;
        this.concentrateScale = builder.concentrateScale;
        this.mleRegression = builder.mleRegression;
        this.timeVaryingRegression = builder.timeVaryingRegression;
        this.simpleDifferencing = builder.simpleDifferencing;
        this.hamiltonRepresentation = builder.hamiltonRepresentation;
        this.approximateDiffuseVariance = builder.approximateDiffuseVariance;
        this.options = EnumSet.copyOf(builder.options);
        validate();
    }

    public static Builder builder(ARIMAOrder order, double[] endog) {
        return new Builder(order, endog);
    }

    public ARIMAOrder order() {
        return order;
    }

    public SeasonalOrder seasonalOrder() {
        return seasonalOrder;
    }

    public double[] endog() {
        return endog.clone();
    }

    public double[][] exog() {
        return copyExog(exog);
    }

    public boolean hasExog() {
        return exog != null && exog.length > 0;
    }

    public int exogDimension() {
        return hasExog() ? exog[0].length : 0;
    }

    public int[] trendPowers() {
        return trendPowers.clone();
    }

    public boolean hasTrend() {
        return trendPowers.length > 0;
    }

    public int[] autoregressiveLags() {
        return autoregressiveLags.clone();
    }

    public int[] movingAverageLags() {
        return movingAverageLags.clone();
    }

    public int[] seasonalAutoregressiveLags() {
        return seasonalAutoregressiveLags.clone();
    }

    public int[] seasonalMovingAverageLags() {
        return seasonalMovingAverageLags.clone();
    }

    public int autoregressiveParameterCount() {
        return autoregressiveLags.length;
    }

    public int movingAverageParameterCount() {
        return movingAverageLags.length;
    }

    public int seasonalAutoregressiveParameterCount() {
        return seasonalAutoregressiveLags.length;
    }

    public int seasonalMovingAverageParameterCount() {
        return seasonalMovingAverageLags.length;
    }

    public boolean measurementError() {
        return measurementError;
    }

    public boolean concentrateScale() {
        return concentrateScale;
    }

    public boolean mleRegression() {
        return mleRegression;
    }

    public boolean timeVaryingRegression() {
        return timeVaryingRegression;
    }

    public boolean simpleDifferencing() {
        return simpleDifferencing;
    }

    public boolean hamiltonRepresentation() {
        return hamiltonRepresentation;
    }

    public boolean hasApproximateDiffuseVariance() {
        return approximateDiffuseVariance != null;
    }

    public double approximateDiffuseVariance() {
        return approximateDiffuseVariance == null ? DEFAULT_APPROXIMATE_DIFFUSE_VARIANCE : approximateDiffuseVariance;
    }

    public EnumSet<SARIMAXOption> options() {
        return EnumSet.copyOf(options);
    }

    public boolean hasOption(SARIMAXOption option) {
        return options.contains(option);
    }

    public int observationCount() {
        return endog.length;
    }

    public int parameterCount() {
        return trendPowers.length
            + (mleRegression ? exogDimension() : 0)
            + autoregressiveLags.length
            + movingAverageLags.length
            + seasonalAutoregressiveLags.length
            + seasonalMovingAverageLags.length
            + ((!mleRegression && timeVaryingRegression) ? exogDimension() : 0)
            + (measurementError ? 1 : 0)
            + (concentrateScale ? 0 : 1);
    }

    private void validate() {
        if (endog.length == 0) {
            throw new IllegalArgumentException("endog must not be empty");
        }
        if (hasExog() && exog.length != endog.length) {
            throw new IllegalArgumentException("exog row count must match endog length");
        }
        if (hasExog()) {
            int width = exog[0].length;
            for (double[] row : exog) {
                if (row == null || row.length != width) {
                    throw new IllegalArgumentException("all exog rows must have the same length");
                }
            }
        }
        if (!seasonalOrder.isZero() && seasonalOrder.period() <= 1) {
            throw new IllegalArgumentException("seasonal period must be greater than 1");
        }
        if (!seasonalOrder.isZero()) {
            if (hasOverlap(autoregressiveLags, seasonalAutoregressiveLags)) {
                throw new IllegalArgumentException("seasonal AR lags overlap with non-seasonal AR lags");
            }
            if (hasOverlap(movingAverageLags, seasonalMovingAverageLags)) {
                throw new IllegalArgumentException("seasonal MA lags overlap with non-seasonal MA lags");
            }
        }
        if (hamiltonRepresentation && !simpleDifferencing
            && (order.integration() > 0 || seasonalOrder.integration() > 0)) {
            throw new IllegalArgumentException("Hamilton representation requires simple differencing when differencing is present");
        }
        if (timeVaryingRegression && mleRegression) {
            throw new IllegalArgumentException("time-varying regression requires mleRegression(false)");
        }
        if (approximateDiffuseVariance != null
            && (!(approximateDiffuseVariance > 0.0) || !Double.isFinite(approximateDiffuseVariance))) {
            throw new IllegalArgumentException("approximate diffuse variance must be positive and finite");
        }
    }

    private static double[] copyEndog(double[] endog) {
        Objects.requireNonNull(endog, "endog");
        return endog.clone();
    }

    private static double[][] copyExog(double[][] exog) {
        if (exog == null) {
            return null;
        }
        double[][] copy = new double[exog.length][];
        for (int row = 0; row < exog.length; row++) {
            copy[row] = exog[row] == null ? null : exog[row].clone();
        }
        return copy;
    }

    private static int[] normalizeTrendPowers(int[] trendPowers) {
        if (trendPowers == null || trendPowers.length == 0) {
            return new int[0];
        }
        int[] copy = trendPowers.clone();
        Arrays.sort(copy);
        int uniqueCount = 0;
        for (int power : copy) {
            if (power < 0) {
                throw new IllegalArgumentException("trend powers must be non-negative");
            }
            if (uniqueCount == 0 || copy[uniqueCount - 1] != power) {
                copy[uniqueCount++] = power;
            }
        }
        return Arrays.copyOf(copy, uniqueCount);
    }

    private static int[] normalizeLags(int maxLag, int[] lags, String label) {
        if (lags == null || lags.length == 0) {
            return contiguousLags(maxLag);
        }
        int[] copy = lags.clone();
        Arrays.sort(copy);
        int uniqueCount = 0;
        for (int lag : copy) {
            if (lag <= 0) {
                throw new IllegalArgumentException(label + " lags must be positive");
            }
            if (lag > maxLag) {
                throw new IllegalArgumentException(label + " lag " + lag + " exceeds order " + maxLag);
            }
            if (uniqueCount == 0 || copy[uniqueCount - 1] != lag) {
                copy[uniqueCount++] = lag;
            }
        }
        if (uniqueCount == 0) {
            throw new IllegalArgumentException(label + " lags must not be empty");
        }
        return Arrays.copyOf(copy, uniqueCount);
    }

    private static int[] contiguousLags(int maxLag) {
        int[] lags = new int[Math.max(0, maxLag)];
        for (int i = 0; i < lags.length; i++) {
            lags[i] = i + 1;
        }
        return lags;
    }

    private static int[] seasonalLags(int count, int period) {
        int[] lags = new int[Math.max(0, count)];
        for (int i = 0; i < lags.length; i++) {
            lags[i] = (i + 1) * period;
        }
        return lags;
    }

    private static boolean hasOverlap(int[] left, int[] right) {
        int i = 0;
        int j = 0;
        while (i < left.length && j < right.length) {
            if (left[i] == right[j]) {
                return true;
            }
            if (left[i] < right[j]) {
                i++;
            } else {
                j++;
            }
        }
        return false;
    }

    public static final class Builder {
        private final ARIMAOrder order;
        private final double[] endog;
        private SeasonalOrder seasonalOrder;
        private double[][] exog;
        private int[] trendPowers;
        private int[] autoregressiveLags;
        private int[] movingAverageLags;
        private boolean measurementError;
        private boolean concentrateScale;
        private boolean mleRegression = true;
        private boolean timeVaryingRegression;
        private boolean simpleDifferencing;
        private boolean hamiltonRepresentation;
        private Double approximateDiffuseVariance;
        private final EnumSet<SARIMAXOption> options = EnumSet.of(
            SARIMAXOption.ENFORCE_STATIONARITY,
            SARIMAXOption.ENFORCE_INVERTIBILITY);

        private Builder(ARIMAOrder order, double[] endog) {
            this.order = Objects.requireNonNull(order, "order");
            this.endog = Objects.requireNonNull(endog, "endog");
        }

        public Builder seasonalOrder(SeasonalOrder seasonalOrder) {
            this.seasonalOrder = seasonalOrder;
            return this;
        }

        public Builder exog(double[][] exog) {
            this.exog = exog;
            return this;
        }

        public Builder trendPowers(int... trendPowers) {
            this.trendPowers = trendPowers;
            return this;
        }

        public Builder autoregressiveLags(int... autoregressiveLags) {
            this.autoregressiveLags = autoregressiveLags;
            return this;
        }

        public Builder movingAverageLags(int... movingAverageLags) {
            this.movingAverageLags = movingAverageLags;
            return this;
        }

        public Builder measurementError(boolean measurementError) {
            this.measurementError = measurementError;
            return this;
        }

        public Builder concentrateScale(boolean concentrateScale) {
            this.concentrateScale = concentrateScale;
            return this;
        }

        public Builder mleRegression(boolean mleRegression) {
            this.mleRegression = mleRegression;
            return this;
        }

        public Builder timeVaryingRegression(boolean timeVaryingRegression) {
            this.timeVaryingRegression = timeVaryingRegression;
            return this;
        }

        public Builder simpleDifferencing(boolean simpleDifferencing) {
            this.simpleDifferencing = simpleDifferencing;
            return this;
        }

        public Builder hamiltonRepresentation(boolean hamiltonRepresentation) {
            this.hamiltonRepresentation = hamiltonRepresentation;
            return this;
        }

        public Builder approximateDiffuseVariance(double approximateDiffuseVariance) {
            this.approximateDiffuseVariance = approximateDiffuseVariance;
            return this;
        }

        public Builder include(SARIMAXOption option) {
            options.add(option);
            return this;
        }

        public Builder exclude(SARIMAXOption option) {
            options.remove(option);
            return this;
        }

        public SARIMAXSpec build() {
            return new SARIMAXSpec(this);
        }
    }
}