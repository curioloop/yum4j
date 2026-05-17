package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Normal;

/** Boost-style inverse Gaussian distribution with mean and scale parameters. */
public value record InverseGaussianDistribution(double mean, double scale) implements ContinuousDistribution {

    private static final GammaDistribution HALF_SHAPE_GAMMA = new GammaDistribution(0.5, 1.0);

    public InverseGaussianDistribution() {
        this(1.0, 1.0);
    }

    public InverseGaussianDistribution {
        validateMean(mean);
        validateScale(scale);
    }

    public double meanParameter() {
        return mean;
    }

    public double location() {
        return mean;
    }

    public double shape() {
        return scale / mean;
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
        return Math.sqrt(scale / (2.0 * Math.PI * x * x * x))
            * Math.exp(-scale * (x - mean) * (x - mean) / (2.0 * x * mean * mean));
    }

    /**
     * Direct-formula log-pdf:
     * {@code 0.5·log(λ/(2π)) - 1.5·log(x) - λ·(x-μ)²/(2·x·μ²)}
     * where {@code λ = scale, μ = mean}.
     *
     * <p>Both tails underflow in the {@code pdf}-based default; the
     * direct form stays accurate throughout.
     */
    @Override
    public double logPdf(double x) {
        validateX(x);
        if (x == 0.0 || x == Double.POSITIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }
        double diff = x - mean;
        return 0.5 * (Math.log(scale) - Math.log(2.0 * Math.PI))
            - 1.5 * Math.log(x)
            - scale * diff * diff / (2.0 * x * mean * mean);
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
        double root = Math.sqrt(scale / x);
        double first = root * (x / mean - 1.0);
        double second = -root * (x / mean + 1.0);
        return Normal.cdf(first) + Math.exp(2.0 * scale / mean) * Normal.cdf(second);
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
        double root = Math.sqrt(scale / x);
        double first = root * (x / mean - 1.0);
        double second = root * (x / mean + 1.0);
        return Normal.ccdf(first) - Math.exp(2.0 * scale / mean) * Normal.ccdf(second);
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
        return StatisticsQuantileSupport.continuousQuantile(this, probability, initialGuess(probability));
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            throw quantileOverflow("quantileUpperTail", probability);
        }
        return StatisticsQuantileSupport.continuousUpperTailQuantile(this, probability, initialGuess(1.0 - probability));
    }

    public double mode() {
        double ratio = mean / scale;
        return mean * (Math.sqrt(1.0 + 2.25 * ratio * ratio) - 1.5 * ratio);
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double variance() {
        return mean * mean * mean / scale;
    }

    public double skewness() {
        return 3.0 * Math.sqrt(mean / scale);
    }

    public double kurtosis() {
        return 15.0 * mean / scale - 3.0;
    }

    public double kurtosisExcess() {
        return 15.0 * mean / scale;
    }

    private double initialGuess(double probability) {
        double phi = scale / mean;
        if (phi > 2.0) {
            return mean * Math.exp(Normal.inv(probability) / Math.sqrt(phi) - 0.5 / phi);
        }

        double complementGammaQuantile = HALF_SHAPE_GAMMA.quantileUpperTail(probability);
        double guess = scale / (2.0 * complementGammaQuantile);
        if (guess > mean / 2.0) {
            double gammaQuantile = HALF_SHAPE_GAMMA.quantile(probability);
            guess = mean * Math.exp(gammaQuantile / Math.sqrt(phi) - 0.5 / phi);
        }
        return guess;
    }

    private static void validateMean(double mean) {
        if (!(mean > 0.0) || !Double.isFinite(mean)) {
            throw new IllegalArgumentException("mean must be finite and greater than 0.0: " + mean);
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
        return new ArithmeticException("InverseGaussianDistribution." + method + " overflow: probability=" + probability);
    }
}