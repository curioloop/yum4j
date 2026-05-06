package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Gamma;

/**
 * Boost-style Poisson distribution object with unified PDF/CDF/quantile access.
 */
public value record PoissonDistribution(double lambda, double logLambda) implements DiscreteDistribution {

    public PoissonDistribution(double lambda) {
        if (!(lambda > 0.0) || !Double.isFinite(lambda)) {
            throw new IllegalArgumentException("lambda must be finite and greater than 0.0: " + lambda);
        }
        this(lambda, Math.log(lambda));
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    public double pdf(double x) {
        validateX(x);
        if (Double.isInfinite(x) || x < 0.0) {
            return 0.0;
        }
        if (!isIntegralPoint(x)) {
            return 0.0;
        }

        long k = (long) x;
        return Math.exp(k * logLambda - Gamma.lgamma(k + 1.0) - lambda);
    }

    public double cdf(double x) {
        validateX(x);
        if (x < 0.0) {
            return 0.0;
        }
        if (Double.isInfinite(x)) {
            return 1.0;
        }
        return cdfAt((long) Math.floor(x));
    }

    public double ccdf(double x) {
        validateX(x);
        if (x < 0.0) {
            return 1.0;
        }
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        return ccdfAt((long) Math.floor(x));
    }

    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            throw quantileOverflow("quantile", probability);
        }
        return quantileCount(probability);
    }

    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            throw quantileOverflow("quantileUpperTail", probability);
        }
        if (probability == 1.0) {
            return 0.0;
        }
        return quantile(1.0 - probability);
    }

    public double mode() {
        return Math.floor(lambda);
    }

    public double median() {
        return quantile(0.5);
    }

    public double mean() {
        return lambda;
    }

    public double variance() {
        return lambda;
    }

    public double skewness() {
        return 1.0 / Math.sqrt(lambda);
    }

    public double kurtosis() {
        return 3.0 + 1.0 / lambda;
    }

    public double kurtosisExcess() {
        return 1.0 / lambda;
    }

    private long quantileCount(double probability) {
        long lower = 0L;
        long upper = Math.max(1L, (long) Math.ceil(lambda));
        while (cdfAt(upper) < probability) {
            if (upper >= Long.MAX_VALUE / 2L) {
                return Long.MAX_VALUE;
            }
            upper *= 2L;
        }

        while (lower < upper) {
            long midpoint = lower + (upper - lower) / 2L;
            if (cdfAt(midpoint) >= probability) {
                upper = midpoint;
            } else {
                lower = midpoint + 1L;
            }
        }
        return lower;
    }

    private double cdfAt(long k) {
        if (k < 0L) {
            return 0.0;
        }
        return Gamma.gammaQ(k + 1.0, lambda);
    }

    private double ccdfAt(long k) {
        if (k < 0L) {
            return 1.0;
        }
        return Gamma.gammaP(k + 1.0, lambda);
    }

    private static boolean isIntegralPoint(double x) {
        return x >= Long.MIN_VALUE && x <= Long.MAX_VALUE && x == Math.rint(x);
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
        return new ArithmeticException("PoissonDistribution." + method + " overflow: probability=" + probability);
    }
}