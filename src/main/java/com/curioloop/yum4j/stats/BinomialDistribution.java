package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Beta;
import com.curioloop.yum4j.math.Gamma;

import java.util.random.RandomGenerator;

/**
 * Boost-style binomial distribution object with unified PDF/CDF/quantile access.
 */
public final class BinomialDistribution implements DiscreteDistribution {

    private final int trials;
    private final double successProbability;
    private final double logSuccessProbability;
    private final double logFailureProbability;
    private final double lgammaTrialsPlus1;

    public BinomialDistribution(int trials, double successProbability) {
        if (trials < 0) {
            throw new IllegalArgumentException("trials must be non-negative: " + trials);
        }
        if (Double.isNaN(successProbability) || successProbability < 0.0 || successProbability > 1.0) {
            throw new IllegalArgumentException(
                "successProbability must be in [0, 1]: " + successProbability
            );
        }
        this.trials = trials;
        this.successProbability = successProbability;
        this.logSuccessProbability = successProbability == 0.0 ? 0.0 : Math.log(successProbability);
        this.logFailureProbability = successProbability == 1.0 ? 0.0 : Math.log1p(-successProbability);
        this.lgammaTrialsPlus1 = Gamma.lgamma(trials + 1.0);
    }

    public int trials() {
        return trials;
    }

    public double successProbability() {
        return successProbability;
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, trials);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, trials);
    }

    public double pdf(double x) {
        validateX(x);
        if (Double.isInfinite(x) || x < 0.0 || x > trials) {
            return 0.0;
        }
        if (!isIntegralPoint(x)) {
            return 0.0;
        }

        int k = (int) x;
        if (successProbability == 0.0) {
            return k == 0 ? 1.0 : 0.0;
        }
        if (successProbability == 1.0) {
            return k == trials ? 1.0 : 0.0;
        }

        double logCoefficient = Gamma.lgamma(trials + 1.0)
            - Gamma.lgamma(k + 1.0)
            - Gamma.lgamma(trials - k + 1.0);
        return Math.exp(logCoefficient + k * logSuccessProbability + (trials - k) * logFailureProbability);
    }

    public double cdf(double x) {
        validateX(x);
        if (x < 0.0) {
            return 0.0;
        }
        if (Double.isInfinite(x) || x >= trials) {
            return 1.0;
        }
        return cdfAt((int) Math.floor(x));
    }

    public double ccdf(double x) {
        validateX(x);
        if (x < 0.0) {
            return 1.0;
        }
        if (Double.isInfinite(x) || x >= trials) {
            return 0.0;
        }
        return ccdfAt((int) Math.floor(x));
    }

    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0 || successProbability == 0.0 || trials == 0) {
            return 0.0;
        }
        if (probability == 1.0 || successProbability == 1.0) {
            return trials;
        }

        int lower = 0;
        int upper = trials;
        while (lower < upper) {
            int midpoint = lower + (upper - lower) / 2;
            if (cdfAt(midpoint) >= probability) {
                upper = midpoint;
            } else {
                lower = midpoint + 1;
            }
        }
        return lower;
    }

    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        return quantile(1.0 - probability);
    }

    public double mode() {
        return Math.floor((trials + 1.0) * successProbability);
    }

    public double median() {
        return Math.floor(trials * successProbability);
    }

    public double mean() {
        return trials * successProbability;
    }

    public double variance() {
        return trials * successProbability * (1.0 - successProbability);
    }

    public double skewness() {
        return (1.0 - 2.0 * successProbability) / Math.sqrt(variance());
    }

    public double kurtosis() {
        return 3.0 - 6.0 / trials + 1.0 / variance();
    }

    public double kurtosisExcess() {
        return (1.0 - 6.0 * successProbability * (1.0 - successProbability)) / variance();
    }

    private double cdfAt(int k) {
        if (k < 0) {
            return 0.0;
        }
        if (k >= trials) {
            return 1.0;
        }
        if (successProbability == 0.0) {
            return 1.0;
        }
        if (successProbability == 1.0) {
            return 0.0;
        }
        return Beta.ibetac(k + 1.0, trials - k, successProbability);
    }

    private double ccdfAt(int k) {
        if (k < 0) {
            return 1.0;
        }
        if (k >= trials) {
            return 0.0;
        }
        if (successProbability == 0.0) {
            return 0.0;
        }
        if (successProbability == 1.0) {
            return 1.0;
        }
        return Beta.ibeta(k + 1.0, trials - k, successProbability);
    }

    private static boolean isIntegralPoint(double x) {
        return x >= Integer.MIN_VALUE && x <= Integer.MAX_VALUE && x == Math.rint(x);
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

    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x) || Double.isInfinite(x) || x < 0.0 || x > trials || !isIntegralPoint(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        int k = (int) x;
        if (successProbability == 0.0) {
            return k == 0 ? 0.0 : Double.NEGATIVE_INFINITY;
        }
        if (successProbability == 1.0) {
            return k == trials ? 0.0 : Double.NEGATIVE_INFINITY;
        }
        double logBinom = lgammaTrialsPlus1 - Gamma.lgamma(k + 1.0) - Gamma.lgamma(trials - k + 1.0);
        return logBinom + k * logSuccessProbability + (trials - k) * logFailureProbability;
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n, double[] out, int outOff) {
        if (metric == Metric.LOG_PDF) {
            if (n == 0) return;
            final double logP = logSuccessProbability;
            final double logQ = logFailureProbability;
            final double lgN1 = lgammaTrialsPlus1;
            final int N = trials;
            final double p = successProbability;
            for (int i = 0; i < n; i++) {
                double v = x[xOff + i * xStride];
                double r;
                if (Double.isNaN(v) || Double.isInfinite(v) || v < 0.0 || v > N || !isIntegralPoint(v)) {
                    r = Double.NEGATIVE_INFINITY;
                } else {
                    int k = (int) v;
                    if (p == 0.0) {
                        r = k == 0 ? 0.0 : Double.NEGATIVE_INFINITY;
                    } else if (p == 1.0) {
                        r = k == N ? 0.0 : Double.NEGATIVE_INFINITY;
                    } else {
                        double logBinom = lgN1 - Gamma.lgamma(k + 1.0) - Gamma.lgamma(N - k + 1.0);
                        r = logBinom + k * logP + (N - k) * logQ;
                    }
                }
                out[outOff + i] = r;
            }
        } else {
            DiscreteDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    /*
     * Sampling strategy note:
     * We use the inverse-CDF (BINV) recurrence from Kemp (1981) on the smaller of
     * p and 1 - p, returning `trials - k` when we worked on 1 - p. This keeps the
     * expected number of iterations at O(n * min(p, 1 - p)), which is acceptable
     * for Phase 1. The design document mentions BTPE for large n; that remains
     * a follow-up optimisation. The parity fixture emits exactly the same algorithm
     * on the Python side so byte-equal output is guaranteed.
     */
    @Override
    public double sample(RandomGenerator g) {
        return binvSample(g, trials, successProbability);
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        final int N = trials;
        final double p = successProbability;
        if (N == 0 || p == 0.0) {
            for (int i = 0; i < n; i++) out[off + i * stride] = 0.0;
            return;
        }
        if (p == 1.0) {
            for (int i = 0; i < n; i++) out[off + i * stride] = N;
            return;
        }
        // BINV over min(p, 1 - p) with symmetric reflection.
        final boolean flip = p > 0.5;
        final double q = flip ? 1.0 - p : p;
        final double logOneMinusQ = Math.log1p(-q);
        final double ratio = q / (1.0 - q);
        for (int i = 0; i < n; i++) {
            int k = binvInner(g, N, logOneMinusQ, ratio);
            out[off + i * stride] = flip ? (N - k) : k;
        }
    }

    private static int binvSample(RandomGenerator g, int N, double p) {
        if (N == 0 || p == 0.0) return 0;
        if (p == 1.0) return N;
        boolean flip = p > 0.5;
        double q = flip ? 1.0 - p : p;
        double logOneMinusQ = Math.log1p(-q);
        double ratio = q / (1.0 - q);
        int k = binvInner(g, N, logOneMinusQ, ratio);
        return flip ? (N - k) : k;
    }

    /**
     * Kemp (1981) BINV recurrence. We accumulate CDF terms f(0), f(1), ...
     * using f(k+1) = f(k) * (N - k) / (k + 1) * q / (1 - q) starting from
     * f(0) = (1 - q)^N. Consumes exactly one uniform.
     */
    private static int binvInner(RandomGenerator g, int N, double logOneMinusQ, double ratio) {
        double u = g.nextDouble();
        double cdf = Math.exp(N * logOneMinusQ);
        double term = cdf;
        int k = 0;
        while (u > cdf && k < N) {
            term *= ratio * (N - k) / (k + 1.0);
            cdf += term;
            k++;
        }
        return k;
    }
}