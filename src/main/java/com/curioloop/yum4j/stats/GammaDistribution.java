package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Gamma;

import java.util.random.RandomGenerator;

/**
 * Boost-style gamma distribution object with unified PDF/CDF/quantile access.
 */
public value record GammaDistribution(double shape, double scale) implements ContinuousDistribution {

    public GammaDistribution(double shape) {
        this(shape, 1.0);
    }

    public GammaDistribution {
        if (!(shape > 0.0) || Double.isNaN(shape)) {
            throw new IllegalArgumentException("shape must be greater than 0.0: " + shape);
        }
        if (!(scale > 0.0) || Double.isNaN(scale)) {
            throw new IllegalArgumentException("scale must be greater than 0.0: " + scale);
        }
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public Double2 support() {
        return Double2.bound(Double.MIN_NORMAL, Double.MAX_VALUE);
    }

    public double pdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        return Gamma.gammaPDerivative(shape, x / scale) / scale;
    }

    public double cdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 1.0;
        }
        return Gamma.gammaP(shape, x / scale);
    }

    public double ccdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        return Gamma.gammaQ(shape, x / scale);
    }

    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            throw quantileOverflow("quantile", probability);
        }
        return scale * Gamma.gammaPInv(shape, probability);
    }

    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            throw quantileOverflow("quantileUpperTail", probability);
        }
        return scale * Gamma.gammaQInv(shape, probability);
    }

    public double mode() {
        if (shape < 1.0) {
            throw new ArithmeticException("Mode is undefined for shape < 1: " + shape);
        }
        return (shape - 1.0) * scale;
    }

    public double median() {
        return quantile(0.5);
    }

    public double mean() {
        return shape * scale;
    }

    public double variance() {
        return shape * scale * scale;
    }

    public double skewness() {
        return 2.0 / Math.sqrt(shape);
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        return 6.0 / shape;
    }

    // ---- Batch overrides --------------------------------------

    /**
     * Direct-formula log-pdf. Returns {@link Double#NEGATIVE_INFINITY}
     * for {@code x <= 0}; caches the invariant
     * {@code -lgamma(shape) - shape*log(scale)} locally so the batch
     * form hoists it.
     */
    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        if (!(x > 0.0)) {
            return Double.NEGATIVE_INFINITY;
        }
        double logConst = -Gamma.lgamma(shape) - shape * Math.log(scale);
        return (shape - 1.0) * Math.log(x) - x / scale + logConst;
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n,
                            double[] out, int outOff) {
        if (n == 0) return;
        final double shapeM1 = shape - 1.0;
        final double invScale = 1.0 / scale;
        switch (metric) {
            case LOG_PDF -> {
                double logConst = -Gamma.lgamma(shape) - shape * Math.log(scale);
                for (int i = 0; i < n; i++) {
                    double v = x[xOff + i * xStride];
                    if (v > 0.0) {
                        out[outOff + i] = shapeM1 * Math.log(v) - v * invScale + logConst;
                    } else {
                        out[outOff + i] = Double.NEGATIVE_INFINITY;
                    }
                }
            }
            case PDF -> {
                double pdfConst = Math.exp(-Gamma.lgamma(shape)) / Math.pow(scale, shape);
                for (int i = 0; i < n; i++) {
                    double v = x[xOff + i * xStride];
                    if (v > 0.0) {
                        out[outOff + i] = pdfConst * Math.pow(v, shapeM1) * Math.exp(-v * invScale);
                    } else {
                        out[outOff + i] = 0.0;
                    }
                }
            }
            default -> ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    // ---- Sampling overrides ---------------------------------------

    /**
     * Scalar Gamma draw via Marsaglia-Tsang (shape &gt;= 1) with the
     * Johnk boost for shape &lt; 1. Scaled by {@link #scale()}.
     */
    @Override
    public double sample(RandomGenerator g) {
        return scale * Distribution.nextGamma(g, shape);
    }

    /**
     * Batch Gamma sampler. Inlines the Marsaglia-Tsang kernel so the
     * inner loop only touches primitives.
     */
    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        if (shape >= 1.0) {
            double d = shape - 1.0 / 3.0;
            double c = 1.0 / Math.sqrt(9.0 * d);
            for (int i = 0; i < n; i++) {
                out[off + i * stride] = scale * marsagliaTsangCore(g, d, c);
            }
        } else {
            double shiftedShape = shape + 1.0;
            double d = shiftedShape - 1.0 / 3.0;
            double c = 1.0 / Math.sqrt(9.0 * d);
            double invShape = 1.0 / shape;
            for (int i = 0; i < n; i++) {
                double y = marsagliaTsangCore(g, d, c);
                out[off + i * stride] = scale * y * Math.pow(g.nextDouble(), invShape);
            }
        }
    }

    /** Inlined Marsaglia-Tsang core. */
    private static double marsagliaTsangCore(RandomGenerator g, double d, double c) {
        while (true) {
            double x = g.nextGaussian();
            double base = 1.0 + c * x;
            double v = base * base * base;
            if (v > 0.0) {
                double u = g.nextDouble();
                if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
                    return d * v;
                }
            }
        }
    }

    private static void validateX(double x) {
        if (Double.isNaN(x) || x < 0.0) {
            throw new IllegalArgumentException("x must be in [0, +infinity): " + x);
        }
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }

    private static ArithmeticException quantileOverflow(String method, double probability) {
        return new ArithmeticException("GammaDistribution." + method + " overflow: probability=" + probability);
    }
}