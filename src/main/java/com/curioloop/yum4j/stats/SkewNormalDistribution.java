package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Normal;
import com.curioloop.yum4j.math.OwensT;

/** Boost-style skew-normal distribution with location, scale, and shape parameters. */
public value record SkewNormalDistribution(double location, double scale, double shape) implements ContinuousDistribution {

    private static final double TWO_DIV_PI = 2.0 / Math.PI;

    public SkewNormalDistribution() {
        this(0.0, 1.0, 0.0);
    }

    public SkewNormalDistribution {
        validateLocation(location);
        validateScale(scale);
        validateShape(shape);
    }

    @Override
    public Double2 range() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(-Double.MAX_VALUE, Double.MAX_VALUE);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        double z = (x - location) / scale;
        return 2.0 * Normal.pdf(z) * Normal.cdf(shape * z) / scale;
    }

    /**
     * Direct-formula log-pdf:
     * {@code log(2) + logφ(z) + logΦ(α·z) - log(σ)}
     * with {@code z = (x - μ)/σ}.
     *
     * <p>Uses {@link Normal#logPdf(double)} and {@link Normal#logCdf(double)}
     * to preserve accuracy in the far left tail (where {@code Φ(αz)}
     * underflows for {@code αz ≪ 0}) and the far right tail
     * (where {@code φ(z)} underflows for {@code |z|} large).
     */
    @Override
    public double logPdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        double z = (x - location) / scale;
        return Math.log(2.0) + Normal.logPdf(z) + Normal.logCdf(shape * z) - Math.log(scale);
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
        double z = (x - location) / scale;
        return Normal.cdf(z) - 2.0 * OwensT.owensT(z, shape);
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
        double z = (x - location) / scale;
        return Normal.ccdf(z) + 2.0 * OwensT.owensT(z, shape);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (probability == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (shape == 0.0) {
            return location + scale * Normal.inv(probability);
        }
        return StatisticsQuantileSupport.continuousQuantile(this, probability, cornishFisherGuess(probability));
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (probability == 1.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (shape == 0.0) {
            return location + scale * Normal.invUpperTail(probability);
        }
        return StatisticsQuantileSupport.continuousUpperTailQuantile(this, probability, cornishFisherGuess(1.0 - probability));
    }

    public double mode() {
        if (shape == 0.0) {
            return location;
        }
        return maximizePdf(location - modeBracketWidth(), location + modeBracketWidth());
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        return location + scale * delta() * Math.sqrt(TWO_DIV_PI);
    }

    @Override
    public double variance() {
        double delta = delta();
        return scale * scale * (1.0 - TWO_DIV_PI * delta * delta);
    }

    public double skewness() {
        double delta = delta();
        double numerator = 0.5 * (4.0 - Math.PI) * Math.pow(Math.sqrt(TWO_DIV_PI) * delta, 3.0);
        double denominator = Math.pow(1.0 - TWO_DIV_PI * delta * delta, 1.5);
        return numerator / denominator;
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        double deltaSquared = deltaSquared();
        double x = 1.0 - TWO_DIV_PI * deltaSquared;
        double y = TWO_DIV_PI * deltaSquared;
        return 2.0 * (Math.PI - 3.0) * y * y / (x * x);
    }

    private double cornishFisherGuess(double probability) {
        double x = Normal.inv(probability);
        double skew = skewness();
        double kurtosisExcess = kurtosisExcess();
        x += (x * x - 1.0) * skew / 6.0;
        x += x * (x * x - 3.0) * kurtosisExcess / 24.0;
        x -= x * (2.0 * x * x - 5.0) * skew * skew / 36.0;
        return standardDeviation() * x + mean();
    }

    private double maximizePdf(double lower, double upper) {
        double left = lower;
        double right = upper;
        double invPhi = (Math.sqrt(5.0) - 1.0) / 2.0;
        double c = right - invPhi * (right - left);
        double d = left + invPhi * (right - left);
        double fc = -pdf(c);
        double fd = -pdf(d);
        for (int index = 0; index < 128; index++) {
            if (fc < fd) {
                right = d;
                d = c;
                fd = fc;
                c = right - invPhi * (right - left);
                fc = -pdf(c);
            } else {
                left = c;
                c = d;
                fc = fd;
                d = left + invPhi * (right - left);
                fd = -pdf(d);
            }
        }
        return 0.5 * (left + right);
    }

    private double modeBracketWidth() {
        return scale * Math.max(8.0, 4.0 * Math.abs(shape) + 4.0);
    }

    private double delta() {
        return shape / Math.sqrt(1.0 + shape * shape);
    }

    private double deltaSquared() {
        return shape == 0.0 ? 0.0 : 1.0 / (1.0 + 1.0 / (shape * shape));
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

    private static void validateShape(double shape) {
        if (!Double.isFinite(shape)) {
            throw new IllegalArgumentException("shape must be finite: " + shape);
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