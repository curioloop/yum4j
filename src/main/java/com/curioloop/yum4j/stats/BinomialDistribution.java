package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Beta;
import com.curioloop.yum4j.math.Gamma;

/**
 * Boost-style binomial distribution object with unified PDF/CDF/quantile access.
 */
public final class BinomialDistribution implements DiscreteDistribution {

    private final int trials;
    private final double successProbability;
    private final double logSuccessProbability;
    private final double logFailureProbability;

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
}