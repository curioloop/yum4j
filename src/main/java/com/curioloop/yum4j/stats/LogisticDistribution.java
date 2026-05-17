package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.random.RandomGenerator;

/** Boost-style logistic distribution with location and scale parameters. */
public value record LogisticDistribution(double location, double scale) implements ContinuousDistribution {

    public LogisticDistribution() {
        this(0.0, 1.0);
    }

    public LogisticDistribution {
        validateLocation(location);
        validateScale(scale);
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
        double expTerm = Math.exp(-Math.abs((x - location) / scale));
        double denominator = 1.0 + expTerm;
        return expTerm / (scale * denominator * denominator);
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
        return logisticSigmoid((x - location) / scale);
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
        return logisticSigmoid((location - x) / scale);
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
        return location + scale * logit(probability);
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
        return location - scale * logit(probability);
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
        return Math.PI * Math.PI * scale * scale / 3.0;
    }

    public double skewness() {
        return 0.0;
    }

    public double kurtosisExcess() {
        return 6.0 / 5.0;
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    private static double logisticSigmoid(double value) {
        if (value >= 0.0) {
            double expTerm = Math.exp(-value);
            return 1.0 / (1.0 + expTerm);
        }
        double expTerm = Math.exp(value);
        return expTerm / (1.0 + expTerm);
    }

    private static double logit(double probability) {
        return Math.log(probability) - Math.log1p(-probability);
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
     * Numerically-stable logistic log-pdf using the
     * {@code log1p(exp(-|z|))} trick to avoid overflow for large |z|.
     *
     * {@code logPdf(x) = -|z| - log(scale) - 2 * log(1 + exp(-|z|))}
     * where {@code z = (x - loc) / scale}.
     */
    @Override
    public double logPdf(double x) {
        if (Double.isInfinite(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        double z = (x - location) / scale;
        double absZ = Math.abs(z);
        return -absZ - Math.log(scale) - 2.0 * Math.log1p(Math.exp(-absZ));
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
                final double negLogScale = -Math.log(s);
                for (int i = 0; i < n; i++) {
                    double xi = x[xOff + i * xStride];
                    if (Double.isInfinite(xi)) {
                        out[outOff + i] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double z = (xi - mu) * invScale;
                    double absZ = Math.abs(z);
                    out[outOff + i] = negLogScale - absZ - 2.0 * Math.log1p(Math.exp(-absZ));
                }
            }
            case PDF -> {
                final double mu = location;
                final double s = scale;
                final double invS = 1.0 / s;
                for (int i = 0; i < n; i++) {
                    double z = (x[xOff + i * xStride] - mu) * invS;
                    double ez = Math.exp(-z);
                    double denom = 1.0 + ez;
                    out[outOff + i] = invS * ez / (denom * denom);
                }
            }
            default -> ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    /**
     * Inverse-CDF sampling: {@code loc + scale * log(u / (1 - u))}
     * with {@code u = U(0, 1)}.
     */
    @Override
    public double sample(RandomGenerator g) {
        double u = g.nextDouble();
        return location + scale * (Math.log(u) - Math.log1p(-u));
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        final double mu = location;
        final double s = scale;
        for (int i = 0; i < n; i++) {
            double u = g.nextDouble();
            out[off + i * stride] = mu + s * (Math.log(u) - Math.log1p(-u));
        }
    }
}