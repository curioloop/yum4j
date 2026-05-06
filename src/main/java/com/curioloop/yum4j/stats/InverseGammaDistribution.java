package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Gamma;

/** Boost-style inverse gamma distribution with shape and scale parameters. */
public value record InverseGammaDistribution(double shape, double scale) implements ContinuousDistribution {

    public InverseGammaDistribution() {
        this(1.0, 1.0);
    }

    public InverseGammaDistribution {
        validateShape(shape);
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
        double scaled = scale / x;
        if (scaled < Double.MIN_NORMAL) {
            return 0.0;
        }
        return Gamma.gammaPDerivative(shape, scaled) * scale / (x * x);
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
        return Gamma.gammaQ(shape, scale / x);
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
        return Gamma.gammaP(shape, scale / x);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        return scale / Gamma.gammaQInv(shape, probability);
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return scale / Gamma.gammaPInv(shape, probability);
    }

    public double mode() {
        return scale / (shape + 1.0);
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        if (shape <= 1.0) {
            throw new ArithmeticException("Mean is undefined for shape <= 1: " + shape);
        }
        return scale / (shape - 1.0);
    }

    @Override
    public double variance() {
        if (shape <= 2.0) {
            throw new ArithmeticException("Variance is undefined for shape <= 2: " + shape);
        }
        return (scale * scale) / ((shape - 1.0) * (shape - 1.0) * (shape - 2.0));
    }

    public double skewness() {
        if (shape <= 3.0) {
            throw new ArithmeticException("Skewness is undefined for shape <= 3: " + shape);
        }
        return 4.0 * Math.sqrt(shape - 2.0) / (shape - 3.0);
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        if (shape <= 4.0) {
            throw new ArithmeticException("Kurtosis excess is undefined for shape <= 4: " + shape);
        }
        return (30.0 * shape - 66.0) / ((shape - 3.0) * (shape - 4.0));
    }

    private static void validateShape(double shape) {
        if (!(shape > 0.0) || !Double.isFinite(shape)) {
            throw new IllegalArgumentException("shape must be finite and greater than 0.0: " + shape);
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
}