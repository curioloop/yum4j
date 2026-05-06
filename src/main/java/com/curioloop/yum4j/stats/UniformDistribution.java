package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style continuous uniform distribution on [lower, upper]. */
public value record UniformDistribution(double lower, double upper) implements ContinuousDistribution {

    public UniformDistribution() {
        this(0.0, 1.0);
    }

    public UniformDistribution {
        if (!Double.isFinite(lower)) {
            throw new IllegalArgumentException("lower must be finite: " + lower);
        }
        if (!Double.isFinite(upper)) {
            throw new IllegalArgumentException("upper must be finite: " + upper);
        }
        if (!(lower < upper)) {
            throw new IllegalArgumentException("lower must be less than upper: lower=" + lower + ", upper=" + upper);
        }
    }

    @Override
    public Double2 range() {
        return Double2.bound(-Double.MAX_VALUE, Double.MAX_VALUE);
    }

    @Override
    public Double2 support() {
        return Double2.bound(lower, upper);
    }

    @Override
    public double pdf(double x) {
        validateFiniteX(x);
        if (x < lower || x > upper) {
            return 0.0;
        }
        return 1.0 / (upper - lower);
    }

    @Override
    public double cdf(double x) {
        validateFiniteX(x);
        if (x < lower) {
            return 0.0;
        }
        if (x > upper) {
            return 1.0;
        }
        return (x - lower) / (upper - lower);
    }

    @Override
    public double ccdf(double x) {
        validateFiniteX(x);
        if (x < lower) {
            return 1.0;
        }
        if (x > upper) {
            return 0.0;
        }
        return (upper - x) / (upper - lower);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return lower;
        }
        if (probability == 1.0) {
            return upper;
        }
        return lower + probability * (upper - lower);
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return upper;
        }
        if (probability == 1.0) {
            return lower;
        }
        return upper - probability * (upper - lower);
    }

    public double mode() {
        return lower;
    }

    public double median() {
        return 0.5 * (lower + upper);
    }

    @Override
    public double mean() {
        return 0.5 * (lower + upper);
    }

    @Override
    public double variance() {
        double width = upper - lower;
        return width * width / 12.0;
    }

    public double skewness() {
        return 0.0;
    }

    public double kurtosis() {
        return 1.8;
    }

    public double kurtosisExcess() {
        return -1.2;
    }

    private static void validateFiniteX(double x) {
        if (!Double.isFinite(x)) {
            throw new IllegalArgumentException("x must be finite: " + x);
        }
    }

    private static void validateProbability(double probability) {
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}