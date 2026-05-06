package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style logistic distribution with location and scale parameters. */
public value record LogisticDistribution(double location, double scale) implements ContinuousDistribution {

    public LogisticDistribution() {
        this(0.0, 1.0);
    }

    public LogisticDistribution {
        validateLocation(location);
        validateScale(scale);
    }

    @Override
    public Double2 range() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(-Double.MAX_VALUE, Double.MAX_VALUE);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        double expTerm = Math.exp(-Math.abs((x - location) / scale));
        double denominator = 1.0 + expTerm;
        return expTerm / (scale * denominator * denominator);
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x == Double.NEGATIVE_INFINITY) {
            return 0.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        return logisticSigmoid((x - location) / scale);
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x == Double.NEGATIVE_INFINITY) {
            return 1.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        return logisticSigmoid((location - x) / scale);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (probability == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        return location + scale * logit(probability);
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (probability == 1.0) {
            return Double.NEGATIVE_INFINITY;
        }
        return location - scale * logit(probability);
    }

    public double mode() {
        return location;
    }

    public double median() {
        return location;
    }

    @Override
    public double mean() {
        return location;
    }

    @Override
    public double variance() {
        return Math.PI * Math.PI * scale * scale / 3.0;
    }

    public double skewness() {
        return 0.0;
    }

    public double kurtosisExcess() {
        return 6.0 / 5.0;
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    private static double logisticSigmoid(double value) {
        if (value >= 0.0) {
            double expTerm = Math.exp(-value);
            return 1.0 / (1.0 + expTerm);
        }
        double expTerm = Math.exp(value);
        return expTerm / (1.0 + expTerm);
    }

    private static double logit(double probability) {
        return Math.log(probability) - Math.log1p(-probability);
    }

    private static void validateLocation(double location) {
        if (!Double.isFinite(location)) {
            throw new IllegalArgumentException("location must be finite: " + location);
        }
    }

    private static void validateScale(double scale) {
        if (!(scale > 0.0) || !Double.isFinite(scale)) {
            throw new IllegalArgumentException("scale must be finite and greater than 0.0: " + scale);
        }
    }

    private static void validateX(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
    }

    private static void validateProbability(double probability) {
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}