package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Beta;
import com.curioloop.yum4j.math.Gamma;

/** Boost-style negative binomial distribution for failures before a target number of successes. */
public value record NegativeBinomialDistribution(double successes, double successFraction)
    implements DiscreteDistribution {

    public NegativeBinomialDistribution {
        validateSuccesses(successes);
        validateSuccessFraction(successFraction);
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
        return (successFraction / (successes + k)) * Beta.ibetaDerivative(successes, k + 1.0, successFraction);
    }

    /**
     * Direct-formula log-pmf:
     * {@code lgamma(k+r) - lgamma(k+1) - lgamma(r) + r·log(p) + k·log(1-p)}
     * where {@code r = successes, p = successFraction, k = x}.
     *
     * <p>Avoids underflow-through-zero that plagues
     * {@link #pdf(double)} for large {@code k} when {@code p < 1}.
     */
    @Override
    public double logPdf(double x) {
        validateX(x);
        if (Double.isInfinite(x) || x < 0.0 || x != Math.rint(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        long k = (long) x;
        if (successFraction == 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (successFraction == 1.0) {
            return k == 0L ? 0.0 : Double.NEGATIVE_INFINITY;
        }
        return Gamma.lgamma(successes + k)
            - Gamma.lgamma(k + 1.0)
            - Gamma.lgamma(successes)
            + successes * Math.log(successFraction)
            + k * Math.log1p(-successFraction);
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
        if (successFraction == 1.0) {
            return 1.0;
        }
        return Beta.ibeta(successes, k + 1.0, successFraction);
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
        if (successFraction == 1.0) {
            return 0.0;
        }
        return Beta.ibetac(successes, k + 1.0, successFraction);
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
        if (probability <= Math.pow(successFraction, successes)) {
            return 0.0;
        }
        if (successFraction == 0.0) {
            throw quantileOverflow("quantile", probability);
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
        if (probability >= zeroUpperTailProbability()) {
            return 0.0;
        }
        if (successFraction == 0.0) {
            throw quantileOverflow("quantileUpperTail", probability);
        }
        return StatisticsQuantileSupport.discreteUpperTailQuantile(this, probability);
    }

    public double mode() {
        return Math.floor((successes - 1.0) * (1.0 - successFraction) / successFraction);
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        return successes * (1.0 - successFraction) / successFraction;
    }

    @Override
    public double variance() {
        return successes * (1.0 - successFraction) / (successFraction * successFraction);
    }

    public double skewness() {
        return (2.0 - successFraction) / Math.sqrt(successes * (1.0 - successFraction));
    }

    public double kurtosis() {
        return 3.0 + 6.0 / successes + (successFraction * successFraction) / (successes * (1.0 - successFraction));
    }

    public double kurtosisExcess() {
        return (6.0 - successFraction * (6.0 - successFraction)) / (successes * (1.0 - successFraction));
    }

    private static void validateSuccesses(double successes) {
        if (!(successes > 0.0) || !Double.isFinite(successes)) {
            throw new IllegalArgumentException("successes must be finite and greater than 0.0: " + successes);
        }
    }

    private static void validateSuccessFraction(double successFraction) {
        if (!Double.isFinite(successFraction) || successFraction < 0.0 || successFraction > 1.0) {
            throw new IllegalArgumentException(
                "successFraction must be finite and in [0, 1]: " + successFraction
            );
        }
    }

    private static void validateX(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
    }

    private static void validateProbability(double probability) {
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }

    private double zeroUpperTailProbability() {
        if (successFraction == 0.0) {
            return 1.0;
        }
        if (successFraction == 1.0) {
            return 0.0;
        }
        return -Math.expm1(successes * Math.log(successFraction));
    }

    private static ArithmeticException quantileOverflow(String method, double probability) {
        return new ArithmeticException("NegativeBinomialDistribution." + method + " overflow: probability=" + probability);
    }
}