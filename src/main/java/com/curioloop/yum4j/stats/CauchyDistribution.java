package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style Cauchy distribution with location and scale parameters. */
public value record CauchyDistribution(double location, double scale) implements ContinuousDistribution {

    public CauchyDistribution() {
        this(0.0, 1.0);
    }

    public CauchyDistribution {
        validateLocation(location);
        validateScale(scale);
    }

    @Override
    public Double2 range() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        double standardized = (x - location) / scale;
        return 1.0 / (Math.PI * scale * (1.0 + standardized * standardized));
    }

    /**
     * Direct-formula log-pdf: {@code -log(π·σ) - log1p(z²)} with
     * {@code z = (x - μ)/σ}.
     *
     * <p>Uses {@link Math#log1p} to avoid precision loss for small
     * {@code z} compared to the default {@code log(pdf(x))}.
     */
    @Override
    public double logPdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        double z = (x - location) / scale;
        return -Math.log(Math.PI) - Math.log(scale) - Math.log1p(z * z);
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
        return Math.atan2(1.0, (location - x) / scale) / Math.PI;
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
        return Math.atan2(1.0, (x - location) / scale) / Math.PI;
    }

    @Override
    public double quantile(double probability) {
        return quantile(probability, false);
    }

    @Override
    public double quantileUpperTail(double probability) {
        return quantile(probability, true);
    }

    public double mode() {
        return location;
    }

    public double median() {
        return location;
    }

    @Override
    public double mean() {
        throw new ArithmeticException("Mean is undefined for the Cauchy distribution");
    }

    @Override
    public double variance() {
        throw new ArithmeticException("Variance is undefined for the Cauchy distribution");
    }

    public double skewness() {
        throw new ArithmeticException("Skewness is undefined for the Cauchy distribution");
    }

    public double kurtosis() {
        throw new ArithmeticException("Kurtosis is undefined for the Cauchy distribution");
    }

    public double kurtosisExcess() {
        throw new ArithmeticException("Kurtosis excess is undefined for the Cauchy distribution");
    }

    private double quantile(double probability, boolean complement) {
        validateProbability(probability);
        if (probability == 0.0) {
            return complement ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        }
        if (probability == 1.0) {
            return complement ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
        }
        double centeredProbability = probability > 0.5 ? probability - 1.0 : probability;
        if (centeredProbability == 0.5) {
            return location;
        }
        double displacement = -scale / Math.tan(Math.PI * centeredProbability);
        return complement ? location - displacement : location + displacement;
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