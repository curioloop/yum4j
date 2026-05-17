package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Normal;

import java.util.random.RandomGenerator;

/**
 * Boost-style normal distribution object with unified PDF/CDF/quantile access.
 */
public value record NormalDistribution(double mean, double standardDeviation) implements ContinuousDistribution {

    /** -0.5 * log(2π) */
    private static final double LOG_INV_SQRT_2PI = -0.5 * Math.log(2.0 * Math.PI);

    /** 1 / √(2π) */
    private static final double INV_SQRT_2PI = 1.0 / Math.sqrt(2.0 * Math.PI);

    public NormalDistribution() {
        this(0.0, 1.0);
    }

    public NormalDistribution {
        if (!(standardDeviation > 0.0)) {
            throw new IllegalArgumentException(
                "standardDeviation must be greater than 0.0: " + standardDeviation
            );
        }
    }

    public double location() {
        return mean;
    }

    public double scale() {
        return standardDeviation;
    }

    @Override
    public Double2 range() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    public double pdf(double x) {
        return Normal.pdf(standardize(x)) / standardDeviation;
    }

    public double cdf(double x) {
        return Normal.cdf(standardize(x));
    }

    public double ccdf(double x) {
        return Normal.ccdf(standardize(x));
    }

    public double quantile(double probability) {
        return mean + standardDeviation * standardQuantile(probability);
    }

    public double quantileUpperTail(double probability) {
        return mean + standardDeviation * standardUpperTailQuantile(probability);
    }

    public double mode() {
        return mean;
    }

    public double median() {
        return mean;
    }

    public double variance() {
        return standardDeviation * standardDeviation;
    }

    public double skewness() {
        return 0.0;
    }

    public double kurtosis() {
        return 3.0;
    }

    public double kurtosisExcess() {
        return 0.0;
    }

    private double standardize(double x) {
        return (x - mean) / standardDeviation;
    }

    // ---------------------------------------------------------------
    // Batch / Sampling specialisations (particle-filter hot path)
    // ---------------------------------------------------------------

    /**
     * Numerically-stable log-pdf using the cached
     * {@code -0.5 * log(2π) - log σ} normaliser and a single
     * {@code z = (x - μ) / σ} computation.
     */
    @Override
    public double logPdf(double x) {
        double sigma = standardDeviation;
        double z = (x - mean) / sigma;
        return LOG_INV_SQRT_2PI - Math.log(sigma) - 0.5 * z * z;
    }

    @Override
    public void batch(Metric metric,
                        double[] x, int xOff, int xStride, int n,
                        double[] out, int outOff) {
        if (n == 0) return;
        final double mu = mean;
        final double sigma = standardDeviation;
        final double invSigma = 1.0 / sigma;
        switch (metric) {
            case LOG_PDF -> {
                final double logNormaliser = LOG_INV_SQRT_2PI - Math.log(sigma);
                for (int i = 0; i < n; i++) {
                    double z = (x[xOff + i * xStride] - mu) * invSigma;
                    out[outOff + i] = logNormaliser - 0.5 * z * z;
                }
            }
            case PDF -> {
                final double normaliser = INV_SQRT_2PI / sigma;
                for (int i = 0; i < n; i++) {
                    double z = (x[xOff + i * xStride] - mu) * invSigma;
                    out[outOff + i] = normaliser * Math.exp(-0.5 * z * z);
                }
            }
            default -> ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    @Override
    public double sample(RandomGenerator g) {
        return mean + standardDeviation * g.nextGaussian();
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        final double mu = mean;
        final double sigma = standardDeviation;
        for (int i = 0; i < n; i++) {
            out[off + i * stride] = mu + sigma * g.nextGaussian();
        }
    }

    private static double standardQuantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (probability == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        return Normal.inv(probability);
    }

    private static double standardUpperTailQuantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (probability == 1.0) {
            return Double.NEGATIVE_INFINITY;
        }
        return Normal.invUpperTail(probability);
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException(
                "probability must be in [0, 1]: " + probability
            );
        }
    }
}