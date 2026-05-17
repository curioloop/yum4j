package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Gamma;

import java.util.random.RandomGenerator;

/**
 * Boost-style Poisson distribution object with unified PDF/CDF/quantile access.
 */
public value record PoissonDistribution(double lambda, double logLambda) implements DiscreteDistribution {

    public PoissonDistribution(double lambda) {
        if (!(lambda > 0.0) || !Double.isFinite(lambda)) {
            throw new IllegalArgumentException("lambda must be finite and greater than 0.0: " + lambda);
        }
        this(lambda, Math.log(lambda));
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    public double pdf(double x) {
        validateX(x);
        if (Double.isInfinite(x) || x < 0.0) {
            return 0.0;
        }
        if (!isIntegralPoint(x)) {
            return 0.0;
        }

        long k = (long) x;
        return Math.exp(k * logLambda - Gamma.lgamma(k + 1.0) - lambda);
    }

    public double cdf(double x) {
        validateX(x);
        if (x < 0.0) {
            return 0.0;
        }
        if (Double.isInfinite(x)) {
            return 1.0;
        }
        return cdfAt((long) Math.floor(x));
    }

    public double ccdf(double x) {
        validateX(x);
        if (x < 0.0) {
            return 1.0;
        }
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        return ccdfAt((long) Math.floor(x));
    }

    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            throw quantileOverflow("quantile", probability);
        }
        return quantileCount(probability);
    }

    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            throw quantileOverflow("quantileUpperTail", probability);
        }
        if (probability == 1.0) {
            return 0.0;
        }
        return quantile(1.0 - probability);
    }

    public double mode() {
        return Math.floor(lambda);
    }

    public double median() {
        return quantile(0.5);
    }

    public double mean() {
        return lambda;
    }

    public double variance() {
        return lambda;
    }

    public double skewness() {
        return 1.0 / Math.sqrt(lambda);
    }

    public double kurtosis() {
        return 3.0 + 1.0 / lambda;
    }

    public double kurtosisExcess() {
        return 1.0 / lambda;
    }

    private long quantileCount(double probability) {
        long lower = 0L;
        long upper = Math.max(1L, (long) Math.ceil(lambda));
        while (cdfAt(upper) < probability) {
            if (upper >= Long.MAX_VALUE / 2L) {
                return Long.MAX_VALUE;
            }
            upper *= 2L;
        }

        while (lower < upper) {
            long midpoint = lower + (upper - lower) / 2L;
            if (cdfAt(midpoint) >= probability) {
                upper = midpoint;
            } else {
                lower = midpoint + 1L;
            }
        }
        return lower;
    }

    private double cdfAt(long k) {
        if (k < 0L) {
            return 0.0;
        }
        return Gamma.gammaQ(k + 1.0, lambda);
    }

    private double ccdfAt(long k) {
        if (k < 0L) {
            return 1.0;
        }
        return Gamma.gammaP(k + 1.0, lambda);
    }

    private static boolean isIntegralPoint(double x) {
        return x >= Long.MIN_VALUE && x <= Long.MAX_VALUE && x == Math.rint(x);
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
        return new ArithmeticException("PoissonDistribution." + method + " overflow: probability=" + probability);
    }

    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x) || Double.isInfinite(x) || x < 0.0 || !isIntegralPoint(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        long k = (long) x;
        return k * logLambda - lambda - Gamma.lgamma(k + 1.0);
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n, double[] out, int outOff) {
        if (metric == Metric.LOG_PDF) {
            if (n == 0) return;
            final double negLambda = -lambda;
            for (int i = 0; i < n; i++) {
                double v = x[xOff + i * xStride];
                double r;
                if (Double.isNaN(v) || Double.isInfinite(v) || v < 0.0 || !isIntegralPoint(v)) {
                    r = Double.NEGATIVE_INFINITY;
                } else {
                    long k = (long) v;
                    r = k * logLambda + negLambda - Gamma.lgamma(k + 1.0);
                }
                out[outOff + i] = r;
            }
        } else {
            DiscreteDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    @Override
    public double sample(RandomGenerator g) {
        if (lambda <= 30.0) {
            return knuthSample(g, lambda);
        }
        return atkinsonSample(g, lambda);
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        if (lambda <= 30.0) {
            final double L = Math.exp(-lambda);
            for (int i = 0; i < n; i++) {
                int k = 0;
                double p = 1.0;
                do {
                    k += 1;
                    p *= g.nextDouble();
                } while (p > L);
                out[off + i * stride] = k - 1;
            }
            return;
        }
        // Atkinson PA rejection path: hoist constants that depend on lambda only.
        final double lam = lambda;
        final double c = 0.767 - 3.36 / lam;
        final double beta = Math.PI / Math.sqrt(3.0 * lam);
        final double alpha = beta * lam;
        final double k0 = Math.log(c) - lam - Math.log(beta);
        for (int i = 0; i < n; i++) {
            out[off + i * stride] = atkinsonInner(g, lam, c, beta, alpha, k0, logLambda);
        }
    }

    private static double knuthSample(RandomGenerator g, double lambda) {
        double L = Math.exp(-lambda);
        int k = 0;
        double p = 1.0;
        do {
            k += 1;
            p *= g.nextDouble();
        } while (p > L);
        return k - 1;
    }

    private static double atkinsonSample(RandomGenerator g, double lam) {
        double c = 0.767 - 3.36 / lam;
        double beta = Math.PI / Math.sqrt(3.0 * lam);
        double alpha = beta * lam;
        double k0 = Math.log(c) - lam - Math.log(beta);
        return atkinsonInner(g, lam, c, beta, alpha, k0, Math.log(lam));
    }

    /** Atkinson PA rejection loop; returns a non-negative integer sample. */
    private static double atkinsonInner(RandomGenerator g,
                                        double lam,
                                        double c,
                                        double beta,
                                        double alpha,
                                        double k0,
                                        double logLam) {
        while (true) {
            double u = g.nextDouble();
            double xv = (alpha - Math.log((1.0 - u) / u)) / beta;
            double nn = Math.floor(xv + 0.5);
            if (nn < 0.0) continue;
            double v = g.nextDouble();
            double y = alpha - beta * xv;
            double oneExpY = 1.0 + Math.exp(y);
            double lhs = y + Math.log(v / (oneExpY * oneExpY));
            double rhs = k0 + nn * logLam - Gamma.lgamma(nn + 1.0);
            if (lhs <= rhs) {
                return nn;
            }
        }
    }
}