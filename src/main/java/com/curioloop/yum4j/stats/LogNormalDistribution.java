package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Normal;

import java.util.random.RandomGenerator;

/** Boost-style lognormal distribution with location and scale parameters. */
public value record LogNormalDistribution(double location, double scale) implements ContinuousDistribution {

    /** -0.5 * log(2π) */
    private static final double LOG_INV_SQRT_2PI = -0.5 * Math.log(2.0 * Math.PI);

    public LogNormalDistribution() {
        this(0.0, 1.0);
    }

    public LogNormalDistribution {
        validateLocation(location);
        validateScale(scale);
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (x == 0.0 || x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        double z = (Math.log(x) - location) / scale;
        return Normal.pdf(z) / (scale * x);
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x == 0.0) {
            return 0.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        return Normal.cdf((Math.log(x) - location) / scale);
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x == 0.0) {
            return 1.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        return Normal.ccdf((Math.log(x) - location) / scale);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            throw quantileOverflow("quantile", probability);
        }
        return Math.exp(location + scale * Normal.inv(probability));
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            throw quantileOverflow("quantileUpperTail", probability);
        }
        return Math.exp(location + scale * Normal.invUpperTail(probability));
    }

    public double mode() {
        return Math.exp(location - scale * scale);
    }

    public double median() {
        return Math.exp(location);
    }

    @Override
    public double mean() {
        return Math.exp(location + 0.5 * scale * scale);
    }

    @Override
    public double variance() {
        double sigmaSquared = scale * scale;
        return Math.expm1(sigmaSquared) * Math.exp(2.0 * location + sigmaSquared);
    }

    public double skewness() {
        double sigmaSquared = scale * scale;
        double expSigmaSquared = Math.exp(sigmaSquared);
        return (expSigmaSquared + 2.0) * Math.sqrt(Math.expm1(sigmaSquared));
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        double sigmaSquared = scale * scale;
        return Math.exp(4.0 * sigmaSquared)
            + 2.0 * Math.exp(3.0 * sigmaSquared)
            + 3.0 * Math.exp(2.0 * sigmaSquared)
            - 6.0;
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
        if (Double.isNaN(x) || x < 0.0) {
            throw new IllegalArgumentException("x must be in [0, +infinity): " + x);
        }
    }

    private static void validateProbability(double probability) {
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }

    private static ArithmeticException quantileOverflow(String method, double probability) {
        return new ArithmeticException("LogNormalDistribution." + method + " overflow: probability=" + probability);
    }

    // ---------------------------------------------------------------
    // Batch / Sampling specialisations (particle-filter hot path)
    // ---------------------------------------------------------------

    /**
     * Numerically-stable log-pdf:
     * {@code -log(x) - log(scale) + logφ((log x - location) / scale)}.
     * Returns {@link Double#NEGATIVE_INFINITY} for {@code x <= 0}.
     */
    @Override
    public double logPdf(double x) {
        if (!(x > 0.0)) {
            return Double.NEGATIVE_INFINITY;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }
        double s = scale;
        double logX = Math.log(x);
        double z = (logX - location) / s;
        return LOG_INV_SQRT_2PI - Math.log(s) - 0.5 * z * z - logX;
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
                final double logNormaliser = LOG_INV_SQRT_2PI - Math.log(s);
                for (int i = 0; i < n; i++) {
                    double xi = x[xOff + i * xStride];
                    if (!(xi > 0.0) || xi == Double.POSITIVE_INFINITY) {
                        out[outOff + i] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double logX = Math.log(xi);
                    double z = (logX - mu) * invScale;
                    out[outOff + i] = logNormaliser - 0.5 * z * z - logX;
                }
            }
            case PDF -> {
                final double mu = location;
                final double sigma = scale;
                final double invSigma = 1.0 / sigma;
                final double normaliser = 1.0 / (sigma * Math.sqrt(2.0 * Math.PI));
                for (int i = 0; i < n; i++) {
                    double xi = x[xOff + i * xStride];
                    if (xi > 0.0) {
                        double z = (Math.log(xi) - mu) * invSigma;
                        out[outOff + i] = normaliser / xi * Math.exp(-0.5 * z * z);
                    } else {
                        out[outOff + i] = 0.0;
                    }
                }
            }
            default -> ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    @Override
    public double sample(RandomGenerator g) {
        return Math.exp(location + scale * g.nextGaussian());
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        final double mu = location;
        final double s = scale;
        for (int i = 0; i < n; i++) {
            out[off + i * stride] = Math.exp(mu + s * g.nextGaussian());
        }
    }
}