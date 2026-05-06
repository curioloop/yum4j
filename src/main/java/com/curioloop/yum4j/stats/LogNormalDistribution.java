package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Normal;

/** Boost-style lognormal distribution with location and scale parameters. */
public value record LogNormalDistribution(double location, double scale) implements ContinuousDistribution {

    public LogNormalDistribution() {
        this(0.0, 1.0);
    }

    public LogNormalDistribution {
        validateLocation(location);
        validateScale(scale);
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (x == 0.0 || x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        double z = (Math.log(x) - location) / scale;
        return Normal.pdf(z) / (scale * x);
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x == 0.0) {
            return 0.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        return Normal.cdf((Math.log(x) - location) / scale);
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x == 0.0) {
            return 1.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        return Normal.ccdf((Math.log(x) - location) / scale);
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
        return Math.exp(location + scale * Normal.inv(probability));
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
        return Math.exp(location + scale * Normal.invUpperTail(probability));
    }

    public double mode() {
        return Math.exp(location - scale * scale);
    }

    public double median() {
        return Math.exp(location);
    }

    @Override
    public double mean() {
        return Math.exp(location + 0.5 * scale * scale);
    }

    @Override
    public double variance() {
        double sigmaSquared = scale * scale;
        return Math.expm1(sigmaSquared) * Math.exp(2.0 * location + sigmaSquared);
    }

    public double skewness() {
        double sigmaSquared = scale * scale;
        double expSigmaSquared = Math.exp(sigmaSquared);
        return (expSigmaSquared + 2.0) * Math.sqrt(Math.expm1(sigmaSquared));
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        double sigmaSquared = scale * scale;
        return Math.exp(4.0 * sigmaSquared)
            + 2.0 * Math.exp(3.0 * sigmaSquared)
            + 3.0 * Math.exp(2.0 * sigmaSquared)
            - 6.0;
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
        return new ArithmeticException("LogNormalDistribution." + method + " overflow: probability=" + probability);
    }
}