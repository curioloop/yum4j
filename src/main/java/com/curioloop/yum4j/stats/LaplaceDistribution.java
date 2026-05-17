package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.random.RandomGenerator;

/** Boost-style Laplace distribution with location and scale parameters. */
public value record LaplaceDistribution(double location, double scale) implements ContinuousDistribution {

    public LaplaceDistribution() {
        this(0.0, 1.0);
    }

    public LaplaceDistribution {
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
        return Math.exp(-Math.abs(x - location) / scale) / (2.0 * scale);
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
        if (x < location) {
            return 0.5 * Math.exp((x - location) / scale);
        }
        return 1.0 - 0.5 * Math.exp((location - x) / scale);
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
        if (x > location) {
            return 0.5 * Math.exp((location - x) / scale);
        }
        return 1.0 - 0.5 * Math.exp((x - location) / scale);
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
        if (probability < 0.5) {
            return location + scale * Math.log(2.0 * probability);
        }
        return location - scale * Math.log(2.0 - 2.0 * probability);
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
        if (probability > 0.5) {
            return location + scale * Math.log(2.0 - 2.0 * probability);
        }
        return location - scale * Math.log(2.0 * probability);
    }

    public double mode() {
        return location;
    }

    public double median() {
        return location;
    }

    @Override
    public double mean() {
        return location;
    }

    @Override
    public double variance() {
        return 2.0 * scale * scale;
    }

    public double skewness() {
        return 0.0;
    }

    public double kurtosis() {
        return 6.0;
    }

    public double kurtosisExcess() {
        return 3.0;
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

    // ---------------------------------------------------------------
    // Batch / Sampling specialisations (particle-filter hot path)
    // ---------------------------------------------------------------

    /**
     * {@code logPdf(x) = -|x - loc| / scale - log(2 * scale)}.
     */
    @Override
    public double logPdf(double x) {
        if (Double.isInfinite(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        double s = scale;
        return -Math.abs(x - location) / s - Math.log(2.0 * s);
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n,
                            double[] out, int outOff) {
        if (n == 0) return;
        switch (metric) {
            case LOG_PDF -> {
                final double mu = location;
                final double s = scale;
                final double invScale = 1.0 / s;
                final double logNormaliser = -Math.log(2.0 * s);
                for (int i = 0; i < n; i++) {
                    double xi = x[xOff + i * xStride];
                    if (Double.isInfinite(xi)) {
                        out[outOff + i] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    out[outOff + i] = logNormaliser - Math.abs(xi - mu) * invScale;
                }
            }
            case PDF -> {
                final double mu = location;
                final double b = scale;
                final double normaliser = 0.5 / b;
                final double invB = 1.0 / b;
                for (int i = 0; i < n; i++) {
                    double xi = x[xOff + i * xStride];
                    out[outOff + i] = normaliser * Math.exp(-Math.abs(xi - mu) * invB);
                }
            }
            default -> ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    /**
     * Inverse-CDF sampling: {@code u = U(0,1) - 0.5;
     * x = loc - scale * sign(u) * log(1 - 2|u|)}.
     */
    @Override
    public double sample(RandomGenerator g) {
        double u = g.nextDouble() - 0.5;
        double sign = u < 0.0 ? -1.0 : 1.0;
        return location - scale * sign * Math.log1p(-2.0 * Math.abs(u));
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        final double mu = location;
        final double s = scale;
        for (int i = 0; i < n; i++) {
            double u = g.nextDouble() - 0.5;
            double sign = u < 0.0 ? -1.0 : 1.0;
            out[off + i * stride] = mu - s * sign * Math.log1p(-2.0 * Math.abs(u));
        }
    }
}