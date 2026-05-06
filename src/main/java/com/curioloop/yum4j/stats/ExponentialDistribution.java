package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style exponential distribution parameterized by lambda. */
public value record ExponentialDistribution(double lambda) implements ContinuousDistribution {

    private static final double LN_TWO = Math.log(2.0);

    public ExponentialDistribution() {
        this(1.0);
    }

    public ExponentialDistribution {
        validateLambda(lambda);
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(Double.MIN_NORMAL, Double.MAX_VALUE);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        return lambda * Math.exp(-lambda * x);
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        return -Math.expm1(-lambda * x);
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        return Math.exp(-lambda * x);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            throw quantileOverflow("quantile", probability);
        }
        return -Math.log1p(-probability) / lambda;
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            throw quantileOverflow("quantileUpperTail", probability);
        }
        return -Math.log(probability) / lambda;
    }

    public double mode() {
        return 0.0;
    }

    public double median() {
        return LN_TWO / lambda;
    }

    @Override
    public double mean() {
        return 1.0 / lambda;
    }

    @Override
    public double variance() {
        return 1.0 / (lambda * lambda);
    }

    public double skewness() {
        return 2.0;
    }

    public double kurtosis() {
        return 9.0;
    }

    public double kurtosisExcess() {
        return 6.0;
    }

    private static void validateLambda(double lambda) {
        if (!(lambda > 0.0) || !Double.isFinite(lambda)) {
            throw new IllegalArgumentException("lambda must be finite and greater than 0.0: " + lambda);
        }
    }

    private static void validateX(double x) {
        if (Double.isNaN(x) || x < 0.0) {
            throw new IllegalArgumentException("x must be in [0, +infinity): " + x);
        }
    }

    private static void validateProbability(double probability) {
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }

    private static ArithmeticException quantileOverflow(String method, double probability) {
        return new ArithmeticException("ExponentialDistribution." + method + " overflow: probability=" + probability);
    }
}