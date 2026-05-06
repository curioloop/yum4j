package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Beta;

/**
 * Boost-style Fisher F distribution object with unified PDF/CDF/quantile access.
 */
public value record FisherFDistribution(double degreesOfFreedom1, double degreesOfFreedom2)
    implements ContinuousDistribution {

    public FisherFDistribution {
        if (!(degreesOfFreedom1 > 0.0) || Double.isNaN(degreesOfFreedom1)) {
            throw new IllegalArgumentException(
                "degreesOfFreedom1 must be greater than 0.0: " + degreesOfFreedom1
            );
        }
        if (!(degreesOfFreedom2 > 0.0) || Double.isNaN(degreesOfFreedom2)) {
            throw new IllegalArgumentException(
                "degreesOfFreedom2 must be greater than 0.0: " + degreesOfFreedom2
            );
        }
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
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        if (x == 0.0) {
            if (degreesOfFreedom1 > 2.0) {
                return 0.0;
            }
            if (degreesOfFreedom1 == 2.0) {
                return 1.0;
            }
            return Double.POSITIVE_INFINITY;
        }

        double y = betaTransform(x);
        return Beta.ibetaDerivative(0.5 * degreesOfFreedom1, 0.5 * degreesOfFreedom2, y)
            * betaTransformDerivative(x);
    }

    public double cdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 1.0;
        }
        return Beta.ibeta(0.5 * degreesOfFreedom1, 0.5 * degreesOfFreedom2, betaTransform(x));
    }

    public double ccdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        return Beta.ibetac(0.5 * degreesOfFreedom1, 0.5 * degreesOfFreedom2, betaTransform(x));
    }

    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        return inverseBetaTransform(Beta.ibetaInv(0.5 * degreesOfFreedom1, 0.5 * degreesOfFreedom2, probability));
    }

    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return inverseBetaTransform(Beta.ibetacInv(0.5 * degreesOfFreedom1, 0.5 * degreesOfFreedom2, probability));
    }

    public double mode() {
        if (degreesOfFreedom1 <= 2.0) {
            throw new ArithmeticException(
                "Mode is undefined for degreesOfFreedom1 <= 2: " + degreesOfFreedom1
            );
        }
        return degreesOfFreedom2 * (degreesOfFreedom1 - 2.0)
            / (degreesOfFreedom1 * (degreesOfFreedom2 + 2.0));
    }

    public double median() {
        return quantile(0.5);
    }

    public double mean() {
        if (degreesOfFreedom2 <= 2.0) {
            throw new ArithmeticException(
                "Mean is undefined for degreesOfFreedom2 <= 2: " + degreesOfFreedom2
            );
        }
        return degreesOfFreedom2 / (degreesOfFreedom2 - 2.0);
    }

    public double variance() {
        if (degreesOfFreedom2 <= 4.0) {
            throw new ArithmeticException(
                "Variance is undefined for degreesOfFreedom2 <= 4: " + degreesOfFreedom2
            );
        }
        return 2.0 * degreesOfFreedom2 * degreesOfFreedom2 * (degreesOfFreedom1 + degreesOfFreedom2 - 2.0)
            / (degreesOfFreedom1
            * (degreesOfFreedom2 - 2.0)
            * (degreesOfFreedom2 - 2.0)
            * (degreesOfFreedom2 - 4.0));
    }

    public double skewness() {
        if (degreesOfFreedom2 <= 6.0) {
            throw new ArithmeticException(
                "Skewness is undefined for degreesOfFreedom2 <= 6: " + degreesOfFreedom2
            );
        }
        return (2.0 * degreesOfFreedom1 + degreesOfFreedom2 - 2.0) * Math.sqrt(8.0 * (degreesOfFreedom2 - 4.0))
            / ((degreesOfFreedom2 - 6.0) * Math.sqrt(degreesOfFreedom1 * (degreesOfFreedom1 + degreesOfFreedom2 - 2.0)));
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        if (degreesOfFreedom2 <= 8.0) {
            throw new ArithmeticException(
                "Kurtosis excess is undefined for degreesOfFreedom2 <= 8: " + degreesOfFreedom2
            );
        }
        double numerator = 12.0 * (degreesOfFreedom1 * (5.0 * degreesOfFreedom2 - 22.0)
            * (degreesOfFreedom1 + degreesOfFreedom2 - 2.0)
            + (degreesOfFreedom2 - 4.0) * (degreesOfFreedom2 - 2.0) * (degreesOfFreedom2 - 2.0));
        double denominator = degreesOfFreedom1
            * (degreesOfFreedom2 - 6.0)
            * (degreesOfFreedom2 - 8.0)
            * (degreesOfFreedom1 + degreesOfFreedom2 - 2.0);
        return numerator / denominator;
    }

    private double betaTransform(double x) {
        double ratio = degreesOfFreedom2 / degreesOfFreedom1;
        return x / (x + ratio);
    }

    private double betaTransformDerivative(double x) {
        double ratio = degreesOfFreedom2 / degreesOfFreedom1;
        double denominator = x + ratio;
        return ratio / (denominator * denominator);
    }

    private double inverseBetaTransform(double y) {
        if (y == 0.0) {
            return 0.0;
        }
        if (y == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        return (degreesOfFreedom2 / degreesOfFreedom1) * y / (1.0 - y);
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