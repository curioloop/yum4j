package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.Arrays;
import java.util.Objects;

/** Boost-style hyperexponential distribution object with unified PDF/CDF/quantile access. */
public value record HyperexponentialDistribution(double[] probabilities, double[] rates) implements ContinuousDistribution {

    private static final double[] SINGLE_PHASE_DEFAULT_PROBABILITIES = {1.0};
    private static final double[] SINGLE_PHASE_DEFAULT_RATES = {1.0};

    public HyperexponentialDistribution() {
        this(SINGLE_PHASE_DEFAULT_PROBABILITIES, SINGLE_PHASE_DEFAULT_RATES);
    }

    public HyperexponentialDistribution(double[] rates) {
        this(equalWeights(Objects.requireNonNull(rates, "rates cannot be null").length), rates);
    }

    public HyperexponentialDistribution(double[] probabilities, double[] rates) {
        Objects.requireNonNull(probabilities, "probabilities cannot be null");
        Objects.requireNonNull(rates, "rates cannot be null");
        if (probabilities.length == 0) {
            throw new IllegalArgumentException("probabilities must not be empty");
        }
        if (probabilities.length != rates.length) {
            throw new IllegalArgumentException(
                "probabilities and rates must have the same length: probabilities=" + probabilities.length
                    + ", rates=" + rates.length
            );
        }
        this.probabilities = normalize(probabilities.clone());
        this.rates = rates.clone();
        validateRates(this.rates);
    }

    public int numPhases() {
        return rates.length;
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(Double.MIN_NORMAL, Double.MAX_VALUE);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        double result = 0.0;
        for (int index = 0; index < rates.length; index++) {
            result += probabilities[index] * rates[index] * Math.exp(-rates[index] * x);
        }
        return result;
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        double result = 0.0;
        for (int index = 0; index < rates.length; index++) {
            result += probabilities[index] * -Math.expm1(-rates[index] * x);
        }
        return result;
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        double result = 0.0;
        for (int index = 0; index < rates.length; index++) {
            result += probabilities[index] * Math.exp(-rates[index] * x);
        }
        return result;
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (rates.length == 1) {
            return exponentialQuantile(rates[0], probability);
        }
        return StatisticsQuantileSupport.continuousQuantile(this, probability, initialQuantileGuess(probability, false));
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (rates.length == 1) {
            return exponentialQuantileUpperTail(rates[0], probability);
        }
        return StatisticsQuantileSupport.continuousUpperTailQuantile(this, probability, initialQuantileGuess(probability, true));
    }

    public double mode() {
        return 0.0;
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        double result = 0.0;
        for (int index = 0; index < rates.length; index++) {
            result += probabilities[index] / rates[index];
        }
        return result;
    }

    @Override
    public double variance() {
        double secondMomentHalf = 0.0;
        for (int index = 0; index < rates.length; index++) {
            secondMomentHalf += probabilities[index] / (rates[index] * rates[index]);
        }
        double mean = mean();
        return 2.0 * secondMomentHalf - mean * mean;
    }

    public double skewness() {
        double s1 = 0.0;
        double s2 = 0.0;
        double s3 = 0.0;
        for (int index = 0; index < rates.length; index++) {
            double probability = probabilities[index];
            double rate = rates[index];
            double rate2 = rate * rate;
            s1 += probability / rate;
            s2 += probability / rate2;
            s3 += probability / (rate2 * rate);
        }
        double s1Squared = s1 * s1;
        double numerator = 6.0 * s3 - (3.0 * (2.0 * s2 - s1Squared) + s1Squared) * s1;
        double denominator = 2.0 * s2 - s1Squared;
        return numerator / Math.pow(denominator, 1.5);
    }

    public double kurtosis() {
        double s1 = 0.0;
        double s2 = 0.0;
        double s3 = 0.0;
        double s4 = 0.0;
        for (int index = 0; index < rates.length; index++) {
            double probability = probabilities[index];
            double rate = rates[index];
            double rate2 = rate * rate;
            double rate3 = rate2 * rate;
            s1 += probability / rate;
            s2 += probability / rate2;
            s3 += probability / rate3;
            s4 += probability / (rate3 * rate);
        }
        double s1Squared = s1 * s1;
        double numerator = 24.0 * s4 - 24.0 * s3 * s1 + 3.0 * (2.0 * (2.0 * s2 - s1Squared) + s1Squared) * s1Squared;
        double denominator = 2.0 * s2 - s1Squared;
        return numerator / (denominator * denominator);
    }

    public double kurtosisExcess() {
        return kurtosis() - 3.0;
    }

    private double initialQuantileGuess(double probability, boolean complement) {
        double guess = 0.0;
        for (int index = 0; index < rates.length; index++) {
            guess += probabilities[index] * (complement
                ? exponentialQuantileUpperTail(rates[index], probability)
                : exponentialQuantile(rates[index], probability));
        }
        return guess;
    }

    private static double exponentialQuantile(double rate, double probability) {
        return -Math.log1p(-probability) / rate;
    }

    private static double exponentialQuantileUpperTail(double rate, double probability) {
        return -Math.log(probability) / rate;
    }

    private static double[] equalWeights(int length) {
        if (length <= 0) {
            throw new IllegalArgumentException("rates must not be empty");
        }
        double[] weights = new double[length];
        Arrays.fill(weights, 1.0);
        return weights;
    }

    private static double[] normalize(double[] weights) {
        double sum = 0.0;
        for (double weight : weights) {
            if (!Double.isFinite(weight) || weight < 0.0) {
                throw new IllegalArgumentException("probabilities must be finite and non-negative: " + weight);
            }
            sum += weight;
        }
        if (!(sum > 0.0) || !Double.isFinite(sum)) {
            throw new IllegalArgumentException("probability weights must have a finite positive sum");
        }

        double normalizedSum = 0.0;
        for (int index = 0; index < weights.length - 1; index++) {
            weights[index] /= sum;
            normalizedSum += weights[index];
        }
        weights[weights.length - 1] = 1.0 - normalizedSum;
        return weights;
    }

    private static void validateRates(double[] rates) {
        for (double rate : rates) {
            if (!(rate > 0.0) || !Double.isFinite(rate)) {
                throw new IllegalArgumentException("rates must be finite and greater than 0.0: " + rate);
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
}