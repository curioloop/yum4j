package com.curioloop.yum4j.tsa.prediction;

import com.curioloop.yum4j.math.Normal;
import com.curioloop.yum4j.stats.StudentsTDistribution;

import java.util.Arrays;
import java.util.OptionalDouble;

public final class UnivariatePredictionValues {

    private final double[] mean;
    private final double[] variance;
    private final double residualVariance;
    private final double degreesOfFreedom;

    public UnivariatePredictionValues(double[] mean, double[] variance) {
        this(mean, variance, Double.NaN, Double.NaN);
    }

    public UnivariatePredictionValues(double[] mean,
                                      double[] variance,
                                      double residualVariance,
                                      double degreesOfFreedom) {
        if (mean == null || variance == null) {
            throw new IllegalArgumentException("mean and variance must not be null");
        }
        if (mean.length != variance.length) {
            throw new IllegalArgumentException("mean and variance lengths must match");
        }
        if (Double.isFinite(residualVariance) && residualVariance < 0.0) {
            throw new IllegalArgumentException("residualVariance must be non-negative");
        }
        if (Double.isFinite(degreesOfFreedom) && degreesOfFreedom <= 0.0) {
            throw new IllegalArgumentException("degreesOfFreedom must be positive");
        }
        this.mean = Arrays.copyOf(mean, mean.length);
        this.variance = Arrays.copyOf(variance, variance.length);
        this.residualVariance = residualVariance;
        this.degreesOfFreedom = degreesOfFreedom;
    }

    public int length() {
        return mean.length;
    }

    public double[] mean() {
        return Arrays.copyOf(mean, mean.length);
    }

    public double[] variance() {
        return Arrays.copyOf(variance, variance.length);
    }

    public double[] variance(boolean observation) {
        if (!observation || !Double.isFinite(residualVariance)) {
            return variance();
        }
        double[] values = new double[variance.length];
        for (int i = 0; i < variance.length; i++) {
            values[i] = variance[i] + residualVariance;
        }
        return values;
    }

    public OptionalDouble residualVariance() {
        return Double.isFinite(residualVariance) ? OptionalDouble.of(residualVariance) : OptionalDouble.empty();
    }

    public OptionalDouble degreesOfFreedom() {
        return Double.isFinite(degreesOfFreedom) ? OptionalDouble.of(degreesOfFreedom) : OptionalDouble.empty();
    }

    public double[] seMean() {
        return se(false);
    }

    public double[] se(boolean observation) {
        double[] selectedVariance = variance(observation);
        double[] standardError = new double[selectedVariance.length];
        for (int i = 0; i < selectedVariance.length; i++) {
            standardError[i] = standardError(selectedVariance[i]);
        }
        return standardError;
    }

    public double[][] confInt(double alpha) {
        return confInt(alpha, false, false);
    }

    public double[][] confInt(double alpha, boolean useT, boolean observation) {
        double criticalValue = criticalValue(alpha, useT);
        double[] standardError = se(observation);
        double[][] intervals = new double[mean.length][2];
        for (int i = 0; i < mean.length; i++) {
            double margin = criticalValue * standardError[i];
            intervals[i][0] = mean[i] - margin;
            intervals[i][1] = mean[i] + margin;
        }
        return intervals;
    }

    private double criticalValue(double alpha, boolean useT) {
        validateAlpha(alpha);
        if (!useT) {
            return Normal.inv(1.0 - alpha / 2.0);
        }
        if (!(Double.isFinite(degreesOfFreedom) && degreesOfFreedom > 0.0)) {
            throw new IllegalArgumentException("degreesOfFreedom is required for t intervals");
        }
        return new StudentsTDistribution(degreesOfFreedom).quantileUpperTail(alpha / 2.0);
    }

    static void validateAlpha(double alpha) {
        if (!(alpha > 0.0 && alpha < 1.0)) {
            throw new IllegalArgumentException("alpha must be between 0 and 1");
        }
    }

    static double standardError(double variance) {
        return variance >= 0.0 && Double.isFinite(variance) ? Math.sqrt(variance) : Double.NaN;
    }
}