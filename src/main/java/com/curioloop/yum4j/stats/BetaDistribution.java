package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Beta;

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