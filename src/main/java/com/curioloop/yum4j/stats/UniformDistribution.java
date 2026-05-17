package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.random.RandomGenerator;

/** Boost-style continuous uniform distribution on [lower, upper]. */
public value record UniformDistribution(double lower, double upper) implements ContinuousDistribution {

    public UniformDistribution() {
        this(0.0, 1.0);
    }

    public UniformDistribution {
        if (!Double.isFinite(lower)) {
            throw new IllegalArgumentException("lower must be finite: " + lower);
        }
        if (!Double.isFinite(upper)) {
            throw new IllegalArgumentException("upper must be finite: " + upper);
        }
        if (!(lower < upper)) {
            throw new IllegalArgumentException("lower must be less than upper: lower=" + lower + ", upper=" + upper);
        }
    }

    @Override
    public Double2 range() {
        return Double2.bound(-Double.MAX_VALUE, Double.MAX_VALUE);
    }

    @Override
    public Double2 support() {
        return Double2.bound(lower, upper);
    }

    @Override
    public double pdf(double x) {
        validateFiniteX(x);
        if (x < lower || x > upper) {
            return 0.0;
        }
        return 1.0 / (upper - lower);
    }

    @Override
    public double cdf(double x) {
        validateFiniteX(x);
        if (x < lower) {
            return 0.0;
        }
        if (x > upper) {
            return 1.0;
        }
        return (x - lower) / (upper - lower);
    }

    @Override
    public double ccdf(double x) {
        validateFiniteX(x);
        if (x < lower) {
            return 1.0;
        }
        if (x > upper) {
            return 0.0;
        }
        return (upper - x) / (upper - lower);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return lower;
        }
        if (probability == 1.0) {
            return upper;
        }
        return lower + probability * (upper - lower);
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return upper;
        }
        if (probability == 1.0) {
            return lower;
        }
        return upper - probability * (upper - lower);
    }

    public double mode() {
        return lower;
    }

    public double median() {
        return 0.5 * (lower + upper);
    }

    @Override
    public double mean() {
        return 0.5 * (lower + upper);
    }

    @Override
    public double variance() {
        double width = upper - lower;
        return width * width / 12.0;
    }

    public double skewness() {
        return 0.0;
    }

    public double kurtosis() {
        return 1.8;
    }

    public double kurtosisExcess() {
        return -1.2;
    }

    private static void validateFiniteX(double x) {
        if (!Double.isFinite(x)) {
            throw new IllegalArgumentException("x must be finite: " + x);
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
     * {@code logPdf(x) = -log(upper - lower)} for {@code lower <= x <= upper};
     * {@link Double#NEGATIVE_INFINITY} otherwise.
     */
    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x) || x < lower || x > upper) {
            return Double.NEGATIVE_INFINITY;
        }
        return -Math.log(upper - lower);
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n,
                            double[] out, int outOff) {
        if (n == 0) return;
        switch (metric) {
            case LOG_PDF -> {
                final double lo = lower;
                final double hi = upper;
                final double logDensity = -Math.log(hi - lo);
                for (int i = 0; i < n; i++) {
                    double xi = x[xOff + i * xStride];
                    if (Double.isNaN(xi) || xi < lo || xi > hi) {
                        out[outOff + i] = Double.NEGATIVE_INFINITY;
                    } else {
                        out[outOff + i] = logDensity;
                    }
                }
            }
            case PDF -> {
                final double lo = lower;
                final double hi = upper;
                final double density = 1.0 / (hi - lo);
                for (int i = 0; i < n; i++) {
                    double xi = x[xOff + i * xStride];
                    out[outOff + i] = (Double.isNaN(xi) || xi < lo || xi > hi) ? 0.0 : density;
                }
            }
            default -> ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    /**
     * {@code lower + (upper - lower) * U(0, 1)}.
     */
    @Override
    public double sample(RandomGenerator g) {
        return lower + (upper - lower) * g.nextDouble();
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        final double lo = lower;
        final double width = upper - lower;
        for (int i = 0; i < n; i++) {
            out[off + i * stride] = lo + width * g.nextDouble();
        }
    }
}