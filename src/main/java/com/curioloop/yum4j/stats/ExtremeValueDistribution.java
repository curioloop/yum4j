package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style extreme-value (Gumbel) distribution with location and scale parameters. */
public value record ExtremeValueDistribution(double location, double scale) implements ContinuousDistribution {

    private static final double EULER_MASCHERONI = 0.57721566490153286061;
    private static final double LN_LN_TWO = Math.log(Math.log(2.0));
    private static final double SKEWNESS = 1.1395470994046486574927930193898461120875997958366;

    public ExtremeValueDistribution() {
        this(0.0, 1.0);
    }

    public ExtremeValueDistribution {
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
        double exponent = (location - x) / scale;
        double expExponent = Math.exp(exponent);
        return Math.exp(exponent - expExponent) / scale;
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
        return Math.exp(-Math.exp((location - x) / scale));
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
        return -Math.expm1(-Math.exp((location - x) / scale));
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
        return location - scale * Math.log(-Math.log(probability));
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
        return location - scale * Math.log(-Math.log1p(-probability));
    }

    public double mode() {
        return location;
    }

    public double median() {
        return location - scale * LN_LN_TWO;
    }

    @Override
    public double mean() {
        return location + EULER_MASCHERONI * scale;
    }

    @Override
    public double variance() {
        return Math.PI * Math.PI * scale * scale / 6.0;
    }

    public double skewness() {
        return SKEWNESS;
    }

    public double kurtosis() {
        return 27.0 / 5.0;
    }

    public double kurtosisExcess() {
        return 12.0 / 5.0;
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