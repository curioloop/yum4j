package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Beta;
import com.curioloop.yum4j.math.Gamma;

import java.util.random.RandomGenerator;

/**
 * Boost-style beta distribution object with unified PDF/CDF/quantile access.
 */
public value record BetaDistribution(double alpha, double beta) implements ContinuousDistribution {

    public BetaDistribution {
        if (!(alpha > 0.0) || Double.isNaN(alpha)) {
            throw new IllegalArgumentException("alpha must be greater than 0.0: " + alpha);
        }
        if (!(beta > 0.0) || Double.isNaN(beta)) {
            throw new IllegalArgumentException("beta must be greater than 0.0: " + beta);
        }
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, 1.0);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, 1.0);
    }

    public double pdf(double x) {
        validateX(x);
        if (x == 0.0) {
            if (alpha > 1.0) {
                return 0.0;
            }
            if (alpha == 1.0) {
                return 1.0 / Beta.beta(alpha, beta);
            }
            return Double.POSITIVE_INFINITY;
        }
        if (x == 1.0) {
            if (beta > 1.0) {
                return 0.0;
            }
            if (beta == 1.0) {
                return 1.0 / Beta.beta(alpha, beta);
            }
            return Double.POSITIVE_INFINITY;
        }
        return Beta.ibetaDerivative(alpha, beta, x);
    }

    public double cdf(double x) {
        validateX(x);
        if (x == 0.0) {
            return 0.0;
        }
        if (x == 1.0) {
            return 1.0;
        }
        return Beta.ibeta(alpha, beta, x);
    }

    public double ccdf(double x) {
        validateX(x);
        if (x == 0.0) {
            return 1.0;
        }
        if (x == 1.0) {
            return 0.0;
        }
        return Beta.ibetac(alpha, beta, x);
    }

    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            return 1.0;
        }
        return Beta.ibetaInv(alpha, beta, probability);
    }

    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            return 1.0;
        }
        return Beta.ibetacInv(alpha, beta, probability);
    }

    public double mode() {
        if (alpha <= 1.0 || beta <= 1.0) {
            throw new ArithmeticException(
                "Mode is undefined for alpha <= 1 or beta <= 1: alpha=" + alpha + ", beta=" + beta
            );
        }
        return (alpha - 1.0) / (alpha + beta - 2.0);
    }

    public double median() {
        return quantile(0.5);
    }

    public double mean() {
        return alpha / (alpha + beta);
    }

    public double variance() {
        double sum = alpha + beta;
        return alpha * beta / (sum * sum * (sum + 1.0));
    }

    public double skewness() {
        double sum = alpha + beta;
        return 2.0 * (beta - alpha) * Math.sqrt(sum + 1.0)
            / ((sum + 2.0) * Math.sqrt(alpha * beta));
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        double sum = alpha + beta;
        double numerator = 6.0 * ((alpha - beta) * (alpha - beta) * (sum + 1.0) - alpha * beta * (sum + 2.0));
        double denominator = alpha * beta * (sum + 2.0) * (sum + 3.0);
        return numerator / denominator;
    }

    // ---- Batch overrides --------------------------------------

    /**
     * Direct-formula log-pdf for Beta(alpha, beta):
     * {@code (alpha-1)*log(x) + (beta-1)*log(1-x) - logBeta(alpha, beta)}.
     * Returns {@link Double#NEGATIVE_INFINITY} outside {@code (0, 1)}
     * unless the distribution has finite density there.
     */
    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        if (x < 0.0 || x > 1.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (x == 0.0) {
            if (alpha > 1.0) return Double.NEGATIVE_INFINITY;
            if (alpha == 1.0) return -logBeta();
            return Double.POSITIVE_INFINITY;
        }
        if (x == 1.0) {
            if (beta > 1.0) return Double.NEGATIVE_INFINITY;
            if (beta == 1.0) return -logBeta();
            return Double.POSITIVE_INFINITY;
        }
        return (alpha - 1.0) * Math.log(x) + (beta - 1.0) * Math.log1p(-x) - logBeta();
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n,
                            double[] out, int outOff) {
        if (metric == Metric.LOG_PDF) {
            if (n == 0) return;
            double minusLogBeta = -logBeta();
            double aM1 = alpha - 1.0;
            double bM1 = beta - 1.0;
            for (int i = 0; i < n; i++) {
                double v = x[xOff + i * xStride];
                out[outOff + i] = betaLogPdf(v, aM1, bM1, minusLogBeta);
            }
        } else {
            ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    private double betaLogPdf(double v, double aM1, double bM1, double minusLogBeta) {
        if (v < 0.0 || v > 1.0 || Double.isNaN(v)) {
            return Double.NEGATIVE_INFINITY;
        }
        if (v == 0.0) {
            if (alpha > 1.0) return Double.NEGATIVE_INFINITY;
            if (alpha == 1.0) return minusLogBeta;
            return Double.POSITIVE_INFINITY;
        }
        if (v == 1.0) {
            if (beta > 1.0) return Double.NEGATIVE_INFINITY;
            if (beta == 1.0) return minusLogBeta;
            return Double.POSITIVE_INFINITY;
        }
        return aM1 * Math.log(v) + bM1 * Math.log1p(-v) + minusLogBeta;
    }

    /** {@code log B(alpha, beta) = lgamma(a) + lgamma(b) - lgamma(a+b)}. */
    private double logBeta() {
        return Gamma.lgamma(alpha) + Gamma.lgamma(beta) - Gamma.lgamma(alpha + beta);
    }

    // ---- Sampling overrides ---------------------------------------

    /**
     * If {@code X ~ Gamma(alpha, 1)} and {@code Y ~ Gamma(beta, 1)} are
     * independent then {@code X / (X + Y) ~ Beta(alpha, beta)}.
     */
    @Override
    public double sample(RandomGenerator g) {
        double x = Distribution.nextGamma(g, alpha);
        double y = Distribution.nextGamma(g, beta);
        return x / (x + y);
    }

    /**
     * Batch ratio sampler. Draws two Gamma streams per element in
     * place: the first Gamma lands in {@code out}, the second is a
     * per-element scalar. This keeps the allocation profile at zero
     * (callers that need a scratch can still pass disjoint storage).
     */
    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        // Draw X ~ Gamma(alpha) directly into the output buffer.
        Distribution.nextGamma(g, alpha, n, out, off, stride);
        // Overwrite each slot with X / (X + Y), where Y is a fresh
        // Gamma(beta) draw consumed inline.
        for (int i = 0; i < n; i++) {
            int idx = off + i * stride;
            double x = out[idx];
            double y = Distribution.nextGamma(g, beta);
            out[idx] = x / (x + y);
        }
    }

    private static void validateX(double x) {
        if (Double.isNaN(x) || x < 0.0 || x > 1.0) {
            throw new IllegalArgumentException("x must be in [0, 1]: " + x);
        }
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}