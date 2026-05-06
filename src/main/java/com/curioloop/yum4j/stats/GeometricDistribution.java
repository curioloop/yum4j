package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style geometric distribution for the number of failures before the first success. */
public value record GeometricDistribution(double successFraction) implements DiscreteDistribution {

    public GeometricDistribution {
        validateSuccessFraction(successFraction);
    }

    public double successes() {
        return 1.0;
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
        if (Double.isInfinite(x) || x < 0.0 || x != Math.rint(x)) {
            return 0.0;
        }
        long k = (long) x;
        if (successFraction == 0.0) {
            return 0.0;
        }
        if (successFraction == 1.0) {
            return k == 0L ? 1.0 : 0.0;
        }
        if (k == 0L) {
            return successFraction;
        }
        return successFraction * Math.exp(k * Math.log1p(-successFraction));
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x < 0.0) {
            return 0.0;
        }
        if (Double.isInfinite(x)) {
            return successFraction == 0.0 ? 0.0 : 1.0;
        }
        long k = (long) Math.floor(x);
        if (successFraction == 0.0) {
            return 0.0;
        }
        if (k == 0L) {
            return successFraction;
        }
        return -Math.expm1(Math.log1p(-successFraction) * (k + 1.0));
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x < 0.0) {
            return 1.0;
        }
        if (Double.isInfinite(x)) {
            return successFraction == 0.0 ? 1.0 : 0.0;
        }
        long k = (long) Math.floor(x);
        if (successFraction == 0.0) {
            return 1.0;
        }
        return Math.exp(Math.log1p(-successFraction) * (k + 1.0));
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            throw quantileOverflow("quantile", probability);
        }
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability <= successFraction) {
            return 0.0;
        }
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return StatisticsQuantileSupport.discreteQuantile(this, probability);
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
        if (probability >= 1.0 - successFraction) {
            return 0.0;
        }
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return StatisticsQuantileSupport.discreteUpperTailQuantile(this, probability);
    }

    public double mode() {
        return 0.0;
    }

    public double median() {
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (successFraction >= 0.5) {
            return 0.0;
        }
        return Math.log1p(-0.5) / Math.log1p(-successFraction) - 1.0;
    }

    @Override
    public double mean() {
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return (1.0 - successFraction) / successFraction;
    }

    @Override
    public double variance() {
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return (1.0 - successFraction) / (successFraction * successFraction);
    }

    public double skewness() {
        return (2.0 - successFraction) / Math.sqrt(1.0 - successFraction);
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        return (successFraction * successFraction - 6.0 * successFraction + 6.0) / (1.0 - successFraction);
    }

    private static void validateSuccessFraction(double successFraction) {
        if (!Double.isFinite(successFraction) || successFraction < 0.0 || successFraction > 1.0) {
            throw new IllegalArgumentException(
                "successFraction must be finite and in [0, 1]: " + successFraction
            );
        }
    }

    private static void validateX(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }

    private static ArithmeticException quantileOverflow(String method, double probability) {
        return new ArithmeticException("GeometricDistribution." + method + " overflow: probability=" + probability);
    }
}