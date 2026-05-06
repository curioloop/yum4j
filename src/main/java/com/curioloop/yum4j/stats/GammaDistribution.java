package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Gamma;

/**
 * Boost-style gamma distribution object with unified PDF/CDF/quantile access.
 */
public value record GammaDistribution(double shape, double scale) implements ContinuousDistribution {

    public GammaDistribution(double shape) {
        this(shape, 1.0);
    }

    public GammaDistribution {
        if (!(shape > 0.0) || Double.isNaN(shape)) {
            throw new IllegalArgumentException("shape must be greater than 0.0: " + shape);
        }
        if (!(scale > 0.0) || Double.isNaN(scale)) {
            throw new IllegalArgumentException("scale must be greater than 0.0: " + scale);
        }
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public Double2 support() {
        return Double2.bound(Double.MIN_NORMAL, Double.MAX_VALUE);
    }

    public double pdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        return Gamma.gammaPDerivative(shape, x / scale) / scale;
    }

    public double cdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 1.0;
        }
        return Gamma.gammaP(shape, x / scale);
    }

    public double ccdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        return Gamma.gammaQ(shape, x / scale);
    }

    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            throw quantileOverflow("quantile", probability);
        }
        return scale * Gamma.gammaPInv(shape, probability);
    }

    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            throw quantileOverflow("quantileUpperTail", probability);
        }
        return scale * Gamma.gammaQInv(shape, probability);
    }

    public double mode() {
        if (shape < 1.0) {
            throw new ArithmeticException("Mode is undefined for shape < 1: " + shape);
        }
        return (shape - 1.0) * scale;
    }

    public double median() {
        return quantile(0.5);
    }

    public double mean() {
        return shape * scale;
    }

    public double variance() {
        return shape * scale * scale;
    }

    public double skewness() {
        return 2.0 / Math.sqrt(shape);
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        return 6.0 / shape;
    }

    private static void validateX(double x) {
        if (Double.isNaN(x) || x < 0.0) {
            throw new IllegalArgumentException("x must be in [0, +infinity): " + x);
        }
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }

    private static ArithmeticException quantileOverflow(String method, double probability) {
        return new ArithmeticException("GammaDistribution." + method + " overflow: probability=" + probability);
    }
}