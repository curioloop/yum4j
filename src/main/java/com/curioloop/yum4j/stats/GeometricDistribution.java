package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.random.RandomGenerator;

/** Boost-style geometric distribution for the number of failures before the first success. */
public value record GeometricDistribution(double successFraction) implements DiscreteDistribution {

    public GeometricDistribution {
        validateSuccessFraction(successFraction);
    }

    public double successes() {
        return 1.0;
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
        if (Double.isInfinite(x) || x < 0.0 || x != Math.rint(x)) {
            return 0.0;
        }
        long k = (long) x;
        if (successFraction == 0.0) {
            return 0.0;
        }
        if (successFraction == 1.0) {
            return k == 0L ? 1.0 : 0.0;
        }
        if (k == 0L) {
            return successFraction;
        }
        return successFraction * Math.exp(k * Math.log1p(-successFraction));
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x < 0.0) {
            return 0.0;
        }
        if (Double.isInfinite(x)) {
            return successFraction == 0.0 ? 0.0 : 1.0;
        }
        long k = (long) Math.floor(x);
        if (successFraction == 0.0) {
            return 0.0;
        }
        if (k == 0L) {
            return successFraction;
        }
        return -Math.expm1(Math.log1p(-successFraction) * (k + 1.0));
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x < 0.0) {
            return 1.0;
        }
        if (Double.isInfinite(x)) {
            return successFraction == 0.0 ? 1.0 : 0.0;
        }
        long k = (long) Math.floor(x);
        if (successFraction == 0.0) {
            return 1.0;
        }
        return Math.exp(Math.log1p(-successFraction) * (k + 1.0));
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            throw quantileOverflow("quantile", probability);
        }
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability <= successFraction) {
            return 0.0;
        }
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return StatisticsQuantileSupport.discreteQuantile(this, probability);
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
        if (probability >= 1.0 - successFraction) {
            return 0.0;
        }
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return StatisticsQuantileSupport.discreteUpperTailQuantile(this, probability);
    }

    public double mode() {
        return 0.0;
    }

    public double median() {
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (successFraction >= 0.5) {
            return 0.0;
        }
        return Math.log1p(-0.5) / Math.log1p(-successFraction) - 1.0;
    }

    @Override
    public double mean() {
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return (1.0 - successFraction) / successFraction;
    }

    @Override
    public double variance() {
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return (1.0 - successFraction) / (successFraction * successFraction);
    }

    public double skewness() {
        return (2.0 - successFraction) / Math.sqrt(1.0 - successFraction);
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        return (successFraction * successFraction - 6.0 * successFraction + 6.0) / (1.0 - successFraction);
    }

    private static void validateSuccessFraction(double successFraction) {
        if (!Double.isFinite(successFraction) || successFraction < 0.0 || successFraction > 1.0) {
            throw new IllegalArgumentException(
                "successFraction must be finite and in [0, 1]: " + successFraction
            );
        }
    }

    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x) || Double.isInfinite(x) || x < 0.0 || x != Math.rint(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        long k = (long) x;
        if (successFraction == 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (successFraction == 1.0) {
            return k == 0L ? 0.0 : Double.NEGATIVE_INFINITY;
        }
        // k * log(1-p) + log(p)
        return k * Math.log1p(-successFraction) + Math.log(successFraction);
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n, double[] out, int outOff) {
        if (metric == Metric.LOG_PDF) {
            if (n == 0) return;
            final double logP = successFraction == 0.0 ? Double.NEGATIVE_INFINITY : Math.log(successFraction);
            final double logQ = successFraction == 1.0 ? Double.NEGATIVE_INFINITY : Math.log1p(-successFraction);
            for (int i = 0; i < n; i++) {
                double v = x[xOff + i * xStride];
                double r;
                if (Double.isNaN(v) || Double.isInfinite(v) || v < 0.0 || v != Math.rint(v)) {
                    r = Double.NEGATIVE_INFINITY;
                } else if (successFraction == 0.0) {
                    r = Double.NEGATIVE_INFINITY;
                } else if (successFraction == 1.0) {
                    r = v == 0.0 ? 0.0 : Double.NEGATIVE_INFINITY;
                } else {
                    long k = (long) v;
                    r = k * logQ + logP;
                }
                out[outOff + i] = r;
            }
        } else {
            DiscreteDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    @Override
    public double sample(RandomGenerator g) {
        if (successFraction == 1.0) {
            return 0.0;
        }
        if (successFraction == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        // k = floor(log(1 - U) / log(1 - p));   U ~ U[0,1).
        double u = g.nextDouble();
        double k = Math.floor(Math.log1p(-u) / Math.log1p(-successFraction));
        return k;
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        if (successFraction == 1.0) {
            for (int i = 0; i < n; i++) out[off + i * stride] = 0.0;
            return;
        }
        if (successFraction == 0.0) {
            for (int i = 0; i < n; i++) out[off + i * stride] = Double.POSITIVE_INFINITY;
            return;
        }
        final double invLogQ = 1.0 / Math.log1p(-successFraction);
        for (int i = 0; i < n; i++) {
            double u = g.nextDouble();
            out[off + i * stride] = Math.floor(Math.log1p(-u) * invLogQ);
        }
    }

    private static void validateX(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }

    private static ArithmeticException quantileOverflow(String method, double probability) {
        return new ArithmeticException("GeometricDistribution." + method + " overflow: probability=" + probability);
    }
}