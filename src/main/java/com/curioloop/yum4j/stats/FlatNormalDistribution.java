package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.random.RandomGenerator;

/**
 * Improper "Normal" with infinite variance, used to represent missing
 * observations in a hidden Markov / state-space model. Every finite
 * point contributes zero log-density (no penalty); non-finite points
 * contribute {@link Double#NEGATIVE_INFINITY}. The distribution is NOT
 * a proper probability measure; it is a semantic placeholder matching
 * the reference {@code particles.distributions.FlatNormal}.
 *
 * <p>Because the density is improper, {@link #sample(RandomGenerator)},
 * the batch sample, {@link #cdf(double)}, and {@link #quantile(double)}
 * are not supported and raise {@link UnsupportedOperationException}
 * with a message pointing callers to {@link NormalDistribution} when
 * they need actual samples.
 */
public value record FlatNormalDistribution(double location) implements ContinuousDistribution {

    private static final String UNSUPPORTED_MSG =
        "FlatNormalDistribution is an improper distribution used to represent "
            + "missing observations; sampling and quantile/cdf are undefined. "
            + "Use NormalDistribution for an actual proper Normal.";

    public FlatNormalDistribution {
        if (Double.isNaN(location)) {
            throw new IllegalArgumentException("location must not be NaN");
        }
    }

    public FlatNormalDistribution() {
        this(0.0);
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
        // Improper density; we follow the reference convention of
        // reporting 0.0 (the limit of Normal(loc, sigma^2) as sigma ->
        // infinity). The logPdf is the semantically meaningful
        // accessor for likelihood computations.
        return 0.0;
    }

    @Override
    public double cdf(double x) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    @Override
    public double ccdf(double x) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    @Override
    public double quantile(double probability) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    @Override
    public double quantileUpperTail(double probability) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    @Override
    public double mean() {
        return location;
    }

    @Override
    public double variance() {
        return Double.POSITIVE_INFINITY;
    }

    // ---------------------------------------------------------------
    // Batch / Sampling specialisations
    // ---------------------------------------------------------------

    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x) || Double.isInfinite(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        return 0.0;
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n,
                            double[] out, int outOff) {
        if (metric == Metric.LOG_PDF) {
            if (n == 0) return;
            for (int i = 0; i < n; i++) {
                double v = x[xOff + i * xStride];
                out[outOff + i] = (Double.isNaN(v) || Double.isInfinite(v))
                    ? Double.NEGATIVE_INFINITY
                    : 0.0;
            }
        } else {
            ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    @Override
    public double sample(RandomGenerator g) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }
}
