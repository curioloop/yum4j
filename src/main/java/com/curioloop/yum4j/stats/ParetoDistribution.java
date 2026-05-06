package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style Pareto distribution with scale and shape parameters. */
public value record ParetoDistribution(double scale, double shape) implements ContinuousDistribution {

    public ParetoDistribution {
        validateScale(scale);
        validateShape(shape);
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public Double2 support() {
        return Double2.bound(scale, Double.MAX_VALUE);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY || x < scale) {
            return 0.0;
        }
        double exponent = shape * Math.log(scale / x);
        return shape * Math.exp(exponent) / x;
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x <= scale) {
            return 0.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        return -Math.expm1(shape * Math.log(scale / x));
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x <= scale) {
            return 1.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        return Math.exp(shape * Math.log(scale / x));
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return scale;
        }
        if (probability == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        return scale * Math.exp(-Math.log1p(-probability) / shape);
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return scale;
        }
        if (probability == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return scale * Math.exp(-Math.log(probability) / shape);
    }

    public double mode() {
        return scale;
    }

    public double median() {
        return scale * Math.pow(2.0, 1.0 / shape);
    }

    @Override
    public double mean() {
        if (shape <= 1.0) {
            return Double.MAX_VALUE;
        }
        return shape * scale / (shape - 1.0);
    }

    @Override
    public double variance() {
        if (shape <= 2.0) {
            throw new ArithmeticException("Variance is undefined for shape <= 2: " + shape);
        }
        return scale * scale * shape / ((shape - 1.0) * (shape - 1.0) * (shape - 2.0));
    }

    public double skewness() {
        if (shape <= 3.0) {
            throw new ArithmeticException("Skewness is undefined for shape <= 3: " + shape);
        }
        return 2.0 * (shape + 1.0) * Math.sqrt((shape - 2.0) / shape) / (shape - 3.0);
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        if (shape <= 4.0) {
            throw new ArithmeticException("Kurtosis excess is undefined for shape <= 4: " + shape);
        }
        return 6.0 * (shape * shape * shape + shape * shape - 6.0 * shape - 2.0)
            / (shape * (shape - 3.0) * (shape - 4.0));
    }

    private static void validateScale(double scale) {
        if (!(scale > 0.0) || !Double.isFinite(scale)) {
            throw new IllegalArgumentException("scale must be finite and greater than 0.0: " + scale);
        }
    }

    private static void validateShape(double shape) {
        if (!(shape > 0.0) || !Double.isFinite(shape)) {
            throw new IllegalArgumentException("shape must be finite and greater than 0.0: " + shape);
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