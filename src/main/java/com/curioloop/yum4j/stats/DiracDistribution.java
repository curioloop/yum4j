package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Point mass at {@link #location()}.
 *
 * <p>This is a degenerate distribution: the "density" is
 * {@link Double#POSITIVE_INFINITY} at the location and zero elsewhere
 * (matching the reference {@code particles.distributions.Dirac}
 * convention), while the log-density is {@code 0} at the location and
 * {@link Double#NEGATIVE_INFINITY} elsewhere. Sampling always returns
 * the fixed location.
 *
 * <p>The {@link #range()} / {@link #support()} collapse to the single
 * point; CDF is the Heaviside step function at the location.
 */
public value record DiracDistribution(double location) implements ContinuousDistribution {

    public DiracDistribution {
        if (Double.isNaN(location)) {
            throw new IllegalArgumentException("location must not be NaN");
        }
    }

    public DiracDistribution() {
        this(0.0);
    }

    @Override
    public Double2 range() {
        return Double2.bound(location, location);
    }

    @Override
    public Double2 support() {
        return Double2.bound(location, location);
    }

    @Override
    public double pdf(double x) {
        if (Double.isNaN(x)) {
            return 0.0;
        }
        return x == location ? Double.POSITIVE_INFINITY : 0.0;
    }

    @Override
    public double cdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        return x < location ? 0.0 : 1.0;
    }

    @Override
    public double ccdf(double x) {
        return 1.0 - cdf(x);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        return location;
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        return location;
    }

    @Override
    public double mean() {
        return location;
    }

    @Override
    public double variance() {
        return 0.0;
    }

    // ---------------------------------------------------------------
    // Batch / Sampling specialisations
    // ---------------------------------------------------------------

    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        return x == location ? 0.0 : Double.NEGATIVE_INFINITY;
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n,
                            double[] out, int outOff) {
        if (metric == Metric.LOG_PDF) {
            if (n == 0) return;
            final double loc = location;
            for (int i = 0; i < n; i++) {
                double v = x[xOff + i * xStride];
                out[outOff + i] = (!Double.isNaN(v) && v == loc) ? 0.0 : Double.NEGATIVE_INFINITY;
            }
        } else {
            ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    @Override
    public double sample(RandomGenerator g) {
        return location;
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        final double loc = location;
        if (stride == 1) {
            Arrays.fill(out, off, off + n, loc);
            return;
        }
        for (int i = 0; i < n; i++) {
            out[off + i * stride] = loc;
        }
    }

    // ---------------------------------------------------------------

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}
