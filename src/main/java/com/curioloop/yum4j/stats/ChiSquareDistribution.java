package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Gamma;

/**
 * Boost-style central chi-square distribution object with unified PDF/CDF/quantile access.
 */
public value record ChiSquareDistribution(double degreesOfFreedom) implements ContinuousDistribution {

    public ChiSquareDistribution {
        if (!(degreesOfFreedom > 0.0) || Double.isNaN(degreesOfFreedom)) {
            throw new IllegalArgumentException(
                "degreesOfFreedom must be greater than 0.0: " + degreesOfFreedom
            );
        }
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    public double pdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        return 0.5 * Gamma.gammaPDerivative(0.5 * degreesOfFreedom, 0.5 * x);
    }

    /**
     * Direct-formula log-pdf:
     * {@code -(k/2)·log(2) - lgamma(k/2) + (k/2 - 1)·log(x) - x/2}
     * where {@code k = degreesOfFreedom}.
     *
     * <p>Preserves accuracy in the far right tail where the
     * {@code exp(-x/2)} factor in {@code pdf} underflows to zero.
     */
    @Override
    public double logPdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        if (x == 0.0) {
            if (degreesOfFreedom == 2.0) return -Math.log(2.0);
            if (degreesOfFreedom > 2.0) return Double.NEGATIVE_INFINITY;
            return Double.POSITIVE_INFINITY;
        }
        double halfDf = 0.5 * degreesOfFreedom;
        return -halfDf * Math.log(2.0) - Gamma.lgamma(halfDf)
            + (halfDf - 1.0) * Math.log(x)
            - 0.5 * x;
    }

    public double cdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 1.0;
        }
        return Gamma.gammaP(0.5 * degreesOfFreedom, 0.5 * x);
    }

    public double ccdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        return Gamma.gammaQ(0.5 * degreesOfFreedom, 0.5 * x);
    }

    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            throw quantileOverflow("quantile", probability);
        }
        return 2.0 * Gamma.gammaPInv(0.5 * degreesOfFreedom, probability);
    }

    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            throw quantileOverflow("quantileUpperTail", probability);
        }
        return 2.0 * Gamma.gammaQInv(0.5 * degreesOfFreedom, probability);
    }

    public double mode() {
        if (degreesOfFreedom < 2.0) {
            throw new ArithmeticException(
                "Mode is undefined for degrees of freedom < 2: " + degreesOfFreedom
            );
        }
        return degreesOfFreedom - 2.0;
    }

    public double median() {
        return quantile(0.5);
    }

    public double mean() {
        return degreesOfFreedom;
    }

    public double variance() {
        return 2.0 * degreesOfFreedom;
    }

    public double skewness() {
        return Math.sqrt(8.0 / degreesOfFreedom);
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        return 12.0 / degreesOfFreedom;
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
        return new ArithmeticException("ChiSquareDistribution." + method + " overflow: probability=" + probability);
    }
}