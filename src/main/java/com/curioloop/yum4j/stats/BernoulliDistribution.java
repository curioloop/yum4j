package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style Bernoulli distribution. */
public value record BernoulliDistribution(double successProbability) implements DiscreteDistribution {

    public BernoulliDistribution() {
        this(0.5);
    }

    public BernoulliDistribution {
        if (!Double.isFinite(successProbability) || successProbability < 0.0 || successProbability > 1.0) {
            throw new IllegalArgumentException(
                "successProbability must be in [0, 1]: " + successProbability
            );
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

    @Override
    public double pdf(double x) {
        validateOutcome(x);
        return x == 0.0 ? 1.0 - successProbability : successProbability;
    }

    @Override
    public double cdf(double x) {
        validateOutcome(x);
        return x == 0.0 ? 1.0 - successProbability : 1.0;
    }

    @Override
    public double ccdf(double x) {
        validateOutcome(x);
        return x == 0.0 ? successProbability : 0.0;
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        return probability <= 1.0 - successProbability ? 0.0 : 1.0;
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        return probability <= 1.0 - successProbability ? 1.0 : 0.0;
    }

    public double mode() {
        return successProbability <= 0.5 ? 0.0 : 1.0;
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        return successProbability;
    }

    @Override
    public double variance() {
        return successProbability * (1.0 - successProbability);
    }

    public double skewness() {
        double variance = variance();
        if (variance == 0.0) {
            return successProbability == 0.0 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        }
        return (1.0 - 2.0 * successProbability) / Math.sqrt(variance);
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        double variance = variance();
        if (variance == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return 1.0 / (1.0 - successProbability) + 1.0 / successProbability - 6.0;
    }

    private static void validateOutcome(double x) {
        if (!Double.isFinite(x) || !(x == 0.0 || x == 1.0)) {
            throw new IllegalArgumentException("x must be 0 or 1: " + x);
        }
    }

    private static void validateProbability(double probability) {
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}