package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Normal;

/**
 * Boost-style normal distribution object with unified PDF/CDF/quantile access.
 */
public value record NormalDistribution(double mean, double standardDeviation) implements ContinuousDistribution {

    public NormalDistribution() {
        this(0.0, 1.0);
    }

    public NormalDistribution {
        if (!(standardDeviation > 0.0) || Double.isNaN(standardDeviation)) {
            throw new IllegalArgumentException(
                "standardDeviation must be greater than 0.0: " + standardDeviation
            );
        }
    }

    public double location() {
        return mean;
    }

    public double scale() {
        return standardDeviation;
    }

    @Override
    public Double2 range() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    public double pdf(double x) {
        return Normal.pdf(standardize(x)) / standardDeviation;
    }

    public double cdf(double x) {
        return Normal.cdf(standardize(x));
    }

    public double ccdf(double x) {
        return Normal.ccdf(standardize(x));
    }

    public double quantile(double probability) {
        return mean + standardDeviation * standardQuantile(probability);
    }

    public double quantileUpperTail(double probability) {
        return mean + standardDeviation * standardUpperTailQuantile(probability);
    }

    public double mode() {
        return mean;
    }

    public double median() {
        return mean;
    }

    public double variance() {
        return standardDeviation * standardDeviation;
    }

    public double skewness() {
        return 0.0;
    }

    public double kurtosis() {
        return 3.0;
    }

    public double kurtosisExcess() {
        return 0.0;
    }

    private double standardize(double x) {
        return (x - mean) / standardDeviation;
    }

    private static double standardQuantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (probability == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        return Normal.inv(probability);
    }

    private static double standardUpperTailQuantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (probability == 1.0) {
            return Double.NEGATIVE_INFINITY;
        }
        return Normal.invUpperTail(probability);
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException(
                "probability must be in [0, 1]: " + probability
            );
        }
    }
}