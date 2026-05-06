package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style non-central Fisher F distribution built as a non-central beta transform. */
public value record NonCentralFDistribution(double degreesOfFreedom1, double degreesOfFreedom2, double nonCentrality)
    implements ContinuousDistribution {

    public NonCentralFDistribution {
        validateDegreesOfFreedom(degreesOfFreedom1, "degreesOfFreedom1");
        validateDegreesOfFreedom(degreesOfFreedom2, "degreesOfFreedom2");
        validateNonCentrality(nonCentrality);
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
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        double y = scaledRatio(x);
        return NonCentralBetaDistribution.pdf(alpha(), beta(), nonCentrality, y / (1.0 + y))
            * betaTransformDerivative(y);
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 1.0;
        }
        double y = scaledRatio(x);
        return NonCentralBetaDistribution.cdf(alpha(), beta(), nonCentrality, y / (1.0 + y));
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        double y = scaledRatio(x);
        return NonCentralBetaDistribution.ccdf(alpha(), beta(), nonCentrality, y / (1.0 + y));
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
        return inverseBetaTransform(NonCentralBetaDistribution.quantile(alpha(), beta(), nonCentrality, probability));
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
        return inverseBetaTransform(
            NonCentralBetaDistribution.quantileUpperTail(alpha(), beta(), nonCentrality, probability)
        );
    }

    public double mode() {
        double pdfAtZero = pdf(0.0);
        if (Double.isInfinite(pdfAtZero)) {
            return 0.0;
        }
        double guess = degreesOfFreedom2 > 2.0
            ? degreesOfFreedom2 * (degreesOfFreedom1 + nonCentrality) / (degreesOfFreedom1 * (degreesOfFreedom2 - 2.0))
            : 1.0;
        return StatisticsModeSupport.positiveSupportMode(this, guess);
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        if (degreesOfFreedom2 <= 2.0) {
            throw new ArithmeticException(
                "Mean is undefined for degreesOfFreedom2 <= 2: " + degreesOfFreedom2
            );
        }
        return degreesOfFreedom2 * (degreesOfFreedom1 + nonCentrality)
            / (degreesOfFreedom1 * (degreesOfFreedom2 - 2.0));
    }

    @Override
    public double variance() {
        if (degreesOfFreedom2 <= 4.0) {
            throw new ArithmeticException(
                "Variance is undefined for degreesOfFreedom2 <= 4: " + degreesOfFreedom2
            );
        }
        double sum = degreesOfFreedom1 + nonCentrality;
        return 2.0 * degreesOfFreedom2 * degreesOfFreedom2
            * (sum * sum + (degreesOfFreedom2 - 2.0) * (degreesOfFreedom1 + 2.0 * nonCentrality))
            / ((degreesOfFreedom2 - 4.0)
            * (degreesOfFreedom2 - 2.0)
            * (degreesOfFreedom2 - 2.0)
            * degreesOfFreedom1
            * degreesOfFreedom1);
    }

    public double skewness() {
        if (degreesOfFreedom2 <= 6.0) {
            throw new ArithmeticException(
                "Skewness is undefined for degreesOfFreedom2 <= 6: " + degreesOfFreedom2
            );
        }
        double numerator = 2.0 * Math.sqrt(2.0)
            * Math.sqrt(degreesOfFreedom2 - 4.0)
            * (degreesOfFreedom1
            * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0)
            * (degreesOfFreedom2 + 2.0 * degreesOfFreedom1 - 2.0)
            + 3.0 * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0)
            * (degreesOfFreedom2 + 2.0 * degreesOfFreedom1 - 2.0)
            * nonCentrality
            + 6.0 * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0) * nonCentrality * nonCentrality
            + 2.0 * nonCentrality * nonCentrality * nonCentrality);
        double denominator = (degreesOfFreedom2 - 6.0)
            * Math.pow(degreesOfFreedom1 * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0)
            + 2.0 * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0) * nonCentrality
            + nonCentrality * nonCentrality, 1.5);
        return numerator / denominator;
    }

    public double kurtosis() {
        return kurtosisExcess() + 3.0;
    }

    public double kurtosisExcess() {
        if (degreesOfFreedom2 <= 8.0) {
            throw new ArithmeticException(
                "Kurtosis excess is undefined for degreesOfFreedom2 <= 8: " + degreesOfFreedom2
            );
        }
        double lambda2 = nonCentrality * nonCentrality;
        double lambda3 = lambda2 * nonCentrality;
        double lambda4 = lambda2 * lambda2;
        double numerator = 3.0 * (degreesOfFreedom2 - 4.0)
            * (degreesOfFreedom1
            * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0)
            * (4.0 * (degreesOfFreedom2 - 2.0) * (degreesOfFreedom2 - 2.0)
            + (degreesOfFreedom2 - 2.0) * (degreesOfFreedom2 + 10.0) * degreesOfFreedom1
            + (degreesOfFreedom2 + 10.0) * degreesOfFreedom1 * degreesOfFreedom1)
            + 4.0 * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0)
            * (4.0 * (degreesOfFreedom2 - 2.0) * (degreesOfFreedom2 - 2.0)
            + (degreesOfFreedom2 - 2.0) * (degreesOfFreedom2 + 10.0) * degreesOfFreedom1
            + (degreesOfFreedom2 + 10.0) * degreesOfFreedom1 * degreesOfFreedom1)
            * nonCentrality
            + 2.0 * (degreesOfFreedom2 + 10.0)
            * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0)
            * (2.0 * degreesOfFreedom2 + 3.0 * degreesOfFreedom1 - 4.0)
            * lambda2
            + 4.0 * (degreesOfFreedom2 + 10.0)
            * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0)
            * lambda3
            + (degreesOfFreedom2 + 10.0) * lambda4);
        double denominator = (degreesOfFreedom2 - 8.0)
            * (degreesOfFreedom2 - 6.0)
            * Math.pow(degreesOfFreedom1 * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0)
            + 2.0 * (degreesOfFreedom2 + degreesOfFreedom1 - 2.0) * nonCentrality
            + lambda2, 2.0);
        return numerator / denominator;
    }

    private double alpha() {
        return degreesOfFreedom1 / 2.0;
    }

    private double beta() {
        return degreesOfFreedom2 / 2.0;
    }

    private double scaledRatio(double x) {
        return x * degreesOfFreedom1 / degreesOfFreedom2;
    }

    private double betaTransformDerivative(double scaledRatio) {
        double denominator = 1.0 + scaledRatio;
        return (degreesOfFreedom1 / degreesOfFreedom2) / (denominator * denominator);
    }

    private double inverseBetaTransform(double x) {
        if (x == 0.0) {
            return 0.0;
        }
        if (x == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        return (x / (1.0 - x)) * (degreesOfFreedom2 / degreesOfFreedom1);
    }

    private static void validateDegreesOfFreedom(double degreesOfFreedom, String label) {
        if (!(degreesOfFreedom > 0.0) || !Double.isFinite(degreesOfFreedom)) {
            throw new IllegalArgumentException(label + " must be finite and greater than 0.0: " + degreesOfFreedom);
        }
    }

    private static void validateNonCentrality(double nonCentrality) {
        if (Double.isNaN(nonCentrality) || nonCentrality < 0.0 || Double.isInfinite(nonCentrality)) {
            throw new IllegalArgumentException(
                "nonCentrality must be finite and in [0, +infinity): " + nonCentrality
            );
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