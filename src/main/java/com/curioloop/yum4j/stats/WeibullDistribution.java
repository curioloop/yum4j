package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Gamma;

/**
 * Boost-style Weibull distribution with shape and scale parameters.
 */
public value record WeibullDistribution(double shape, double scale) implements ContinuousDistribution {

    public WeibullDistribution {
        validateShape(shape);
        validateScale(scale);
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.MAX_VALUE);
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
        if (x == 0.0) {
            if (shape == 1.0) {
                return 1.0 / scale;
            }
            if (shape > 1.0) {
                return 0.0;
            }
            throw pdfOverflow(x);
        }
        double scaledX = x / scale;
        return Math.exp(-Math.pow(scaledX, shape)) * Math.pow(scaledX, shape - 1.0) * shape / scale;
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        return -Math.expm1(-Math.pow(x / scale, shape));
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        return Math.exp(-Math.pow(x / scale, shape));
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            throw quantileOverflow("quantile", probability);
        }
        return scale * Math.pow(-Math.log1p(-probability), 1.0 / shape);
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
        return scale * Math.pow(-Math.log(probability), 1.0 / shape);
    }

    public double mode() {
        if (shape <= 1.0) {
            return 0.0;
        }
        return scale * Math.pow((shape - 1.0) / shape, 1.0 / shape);
    }

    public double median() {
        return scale * Math.pow(Math.log(2.0), 1.0 / shape);
    }

    @Override
    public double mean() {
        return scale * Gamma.tgamma(1.0 + 1.0 / shape);
    }

    @Override
    public double variance() {
        double gamma1 = Gamma.tgamma(1.0 + 1.0 / shape);
        double gamma2 = Gamma.tgamma(1.0 + 2.0 / shape);
        return scale * scale * (gamma2 - gamma1 * gamma1);
    }

    public double skewness() {
        double gamma1 = Gamma.tgamma(1.0 + 1.0 / shape);
        double gamma2 = Gamma.tgamma(1.0 + 2.0 / shape);
        double gamma3 = Gamma.tgamma(1.0 + 3.0 / shape);
        double denominator = Math.pow(gamma2 - gamma1 * gamma1, 1.5);
        return (2.0 * gamma1 * gamma1 * gamma1 - 3.0 * gamma1 * gamma2 + gamma3) / denominator;
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        double gamma1 = Gamma.tgamma(1.0 + 1.0 / shape);
        double gamma2 = Gamma.tgamma(1.0 + 2.0 / shape);
        double gamma3 = Gamma.tgamma(1.0 + 3.0 / shape);
        double gamma4 = Gamma.tgamma(1.0 + 4.0 / shape);
        double gamma1Squared = gamma1 * gamma1;
        double denominator = gamma2 - gamma1Squared;
        denominator *= denominator;
        return (-6.0 * gamma1Squared * gamma1Squared
                + 12.0 * gamma1Squared * gamma2
                - 3.0 * gamma2 * gamma2
                - 4.0 * gamma1 * gamma3
                + gamma4) / denominator;
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

    private ArithmeticException pdfOverflow(double x) {
        return new ArithmeticException("WeibullDistribution.pdf overflow: x=" + x + ", shape=" + shape);
    }

    private static ArithmeticException quantileOverflow(String method, double probability) {
        return new ArithmeticException("WeibullDistribution." + method + " overflow: probability=" + probability);
    }
}