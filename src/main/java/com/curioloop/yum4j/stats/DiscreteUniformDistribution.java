package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.random.RandomGenerator;

/**
 * Uniform distribution over the integer interval {@code {lo, lo+1, ..., hi - 1}}
 * ({@code lo} inclusive, {@code hi} exclusive). Matches the reference
 * {@code particles.distributions.DiscreteUniform} contract.
 */
public value record DiscreteUniformDistribution(long lo, long hi) implements DiscreteDistribution {

    public DiscreteUniformDistribution {
        if (!(hi > lo)) {
            throw new IllegalArgumentException(
                "require lo < hi: lo=" + lo + ", hi=" + hi
            );
        }
    }

    /** Cardinality of the support {@code (hi - lo)}. */
    public long span() {
        return hi - lo;
    }

    @Override
    public Double2 range() {
        return Double2.bound((double) lo, (double) (hi - 1L));
    }

    @Override
    public Double2 support() {
        return Double2.bound((double) lo, (double) (hi - 1L));
    }

    @Override
    public double pdf(double x) {
        if (Double.isNaN(x) || !isIntegralPoint(x)) {
            return 0.0;
        }
        long k = (long) x;
        if (k < lo || k >= hi) {
            return 0.0;
        }
        return 1.0 / (hi - lo);
    }

    @Override
    public double cdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        if (x < lo) {
            return 0.0;
        }
        if (x >= hi - 1L) {
            return 1.0;
        }
        long floor = (long) Math.floor(x);
        return (double) (floor - lo + 1L) / (double) (hi - lo);
    }

    @Override
    public double ccdf(double x) {
        return 1.0 - cdf(x);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return lo;
        }
        if (probability >= 1.0) {
            return hi - 1L;
        }
        long span = hi - lo;
        long k = (long) Math.ceil(probability * span) - 1L;
        if (k < 0L) k = 0L;
        if (k >= span) k = span - 1L;
        return lo + k;
    }

    @Override
    public double quantileUpperTail(double probability) {
        return quantile(1.0 - probability);
    }

    @Override
    public double mean() {
        return (lo + hi - 1L) / 2.0;
    }

    @Override
    public double variance() {
        double span = hi - lo;
        return (span * span - 1.0) / 12.0;
    }

    // ---------------------------------------------------------------
    // Batch / Sampling specialisations
    // ---------------------------------------------------------------

    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x) || !isIntegralPoint(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        long k = (long) x;
        if (k < lo || k >= hi) {
            return Double.NEGATIVE_INFINITY;
        }
        return -Math.log((double) (hi - lo));
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n,
                            double[] out, int outOff) {
        if (metric == Metric.LOG_PDF) {
            if (n == 0) return;
            final long low = lo;
            final long high = hi;
            final double logDensity = -Math.log((double) (high - low));
            for (int i = 0; i < n; i++) {
                double v = x[xOff + i * xStride];
                double r;
                if (Double.isNaN(v) || !isIntegralPoint(v)) {
                    r = Double.NEGATIVE_INFINITY;
                } else {
                    long k = (long) v;
                    r = (k < low || k >= high) ? Double.NEGATIVE_INFINITY : logDensity;
                }
                out[outOff + i] = r;
            }
        } else {
            DiscreteDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    @Override
    public double sample(RandomGenerator g) {
        return (double) (lo + g.nextLong(hi - lo));
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        final long low = lo;
        final long span = hi - lo;
        for (int i = 0; i < n; i++) {
            out[off + i * stride] = (double) (low + g.nextLong(span));
        }
    }

    // ---------------------------------------------------------------

    private static boolean isIntegralPoint(double x) {
        return !Double.isInfinite(x) && x == Math.rint(x);
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}
