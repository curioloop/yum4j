package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style triangular distribution with lower, mode, and upper parameters. */
public value record TriangularDistribution(double lower, double mode, double upper) implements ContinuousDistribution {

    private static final double ROOT_TWO = Math.sqrt(2.0);

    public TriangularDistribution() {
        this(-1.0, 0.0, 1.0);
    }

    public TriangularDistribution {
        if (!Double.isFinite(lower)) {
            throw new IllegalArgumentException("lower must be finite: " + lower);
        }
        if (!Double.isFinite(mode)) {
            throw new IllegalArgumentException("mode must be finite: " + mode);
        }
        if (!Double.isFinite(upper)) {
            throw new IllegalArgumentException("upper must be finite: " + upper);
        }
        if (!(lower < upper)) {
            throw new IllegalArgumentException("lower must be less than upper: lower=" + lower + ", upper=" + upper);
        }
        if (mode < lower || mode > upper) {
            throw new IllegalArgumentException("mode must be in [lower, upper]: mode=" + mode);
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
        if (x == lower) {
            return mode == lower ? 2.0 / (upper - lower) : 0.0;
        }
        if (x == upper) {
            return mode == upper ? 2.0 / (upper - lower) : 0.0;
        }
        if (x <= mode) {
            return 2.0 * (x - lower) / ((upper - lower) * (mode - lower));
        }
        return 2.0 * (upper - x) / ((upper - lower) * (upper - mode));
    }

    @Override
    public double cdf(double x) {
        validateFiniteX(x);
        if (x <= lower) {
            return 0.0;
        }
        if (x >= upper) {
            return 1.0;
        }
        if (x <= mode) {
            return ((x - lower) * (x - lower)) / ((upper - lower) * (mode - lower));
        }
        return 1.0 - ((upper - x) * (upper - x)) / ((upper - lower) * (upper - mode));
    }

    @Override
    public double ccdf(double x) {
        validateFiniteX(x);
        if (x <= lower) {
            return 1.0;
        }
        if (x >= upper) {
            return 0.0;
        }
        if (x <= mode) {
            return 1.0 - ((x - lower) * (x - lower)) / ((upper - lower) * (mode - lower));
        }
        return ((upper - x) * (upper - x)) / ((upper - lower) * (upper - mode));
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
        double pivot = (mode - lower) / (upper - lower);
        if (probability < pivot) {
            return lower + Math.sqrt((upper - lower) * (mode - lower) * probability);
        }
        if (probability == pivot) {
            return mode;
        }
        return upper - Math.sqrt((upper - lower) * (upper - mode) * (1.0 - probability));
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
        double pivot = (mode - lower) / (upper - lower);
        double probabilityLowerTail = 1.0 - probability;
        if (probabilityLowerTail < pivot) {
            return lower + Math.sqrt((upper - lower) * (mode - lower) * probabilityLowerTail);
        }
        if (probabilityLowerTail == pivot) {
            return mode;
        }
        return upper - Math.sqrt((upper - lower) * (upper - mode) * probability);
    }

    public double median() {
        if (mode >= 0.5 * (upper + lower)) {
            return lower + Math.sqrt((upper - lower) * (mode - lower)) / ROOT_TWO;
        }
        return upper - Math.sqrt((upper - lower) * (upper - mode)) / ROOT_TWO;
    }

    @Override
    public double mean() {
        return (lower + mode + upper) / 3.0;
    }

    @Override
    public double variance() {
        return (lower * lower + upper * upper + mode * mode - lower * upper - lower * mode - upper * mode) / 18.0;
    }

    public double skewness() {
        double numerator = ROOT_TWO * (lower + upper - 2.0 * mode) * (2.0 * lower - upper - mode) * (lower - 2.0 * upper + mode);
        double denominatorBase = lower * lower + upper * upper + mode * mode - lower * upper - lower * mode - upper * mode;
        return numerator / (5.0 * Math.pow(denominatorBase, 1.5));
    }

    public double kurtosis() {
        return 2.4;
    }

    public double kurtosisExcess() {
        return -0.6;
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