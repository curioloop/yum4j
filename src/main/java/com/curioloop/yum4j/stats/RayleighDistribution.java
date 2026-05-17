package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style Rayleigh distribution parameterized by sigma. */
public value record RayleighDistribution(double sigma) implements ContinuousDistribution {

    private static final double ROOT_HALF_PI = Math.sqrt(Math.PI / 2.0);
    private static final double ROOT_LN_FOUR = Math.sqrt(Math.log(4.0));
    private static final double SKEWNESS = 0.63111065781893713819189935154422777984404221106391;
    private static final double KURTOSIS_EXCESS = 0.2450893006876380628486604106197544154170667057995;

    public RayleighDistribution() {
        this(1.0);
    }

    public RayleighDistribution {
        validateSigma(sigma);
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        double sigmaSquared = sigma * sigma;
        return x * Math.exp(-(x * x) / (2.0 * sigmaSquared)) / sigmaSquared;
    }

    /**
     * Direct-formula log-pdf: {@code log(x) - x²/(2σ²) - 2·log(σ)}.
     *
     * <p>Preserves accuracy in the far right tail where
     * {@code exp(-x²/(2σ²))} underflows to zero.
     */
    @Override
    public double logPdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }
        if (x == 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        double sigmaSquared = sigma * sigma;
        return Math.log(x) - (x * x) / (2.0 * sigmaSquared) - 2.0 * Math.log(sigma);
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        double sigmaSquared = sigma * sigma;
        return -Math.expm1(-(x * x) / (2.0 * sigmaSquared));
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        double sigmaSquared = sigma * sigma;
        return Math.exp(-(x * x) / (2.0 * sigmaSquared));
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
        return sigma * Math.sqrt(-2.0 * Math.log1p(-probability));
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
        return sigma * Math.sqrt(-2.0 * Math.log(probability));
    }

    public double mode() {
        return sigma;
    }

    public double median() {
        return ROOT_LN_FOUR * sigma;
    }

    @Override
    public double mean() {
        return sigma * ROOT_HALF_PI;
    }

    @Override
    public double variance() {
        return (4.0 - Math.PI) * sigma * sigma / 2.0;
    }

    public double skewness() {
        return SKEWNESS;
    }

    public double kurtosis() {
        return 3.0 + KURTOSIS_EXCESS;
    }

    public double kurtosisExcess() {
        return KURTOSIS_EXCESS;
    }

    private static void validateSigma(double sigma) {
        if (!(sigma > 0.0) || !Double.isFinite(sigma)) {
            throw new IllegalArgumentException("sigma must be finite and greater than 0.0: " + sigma);
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