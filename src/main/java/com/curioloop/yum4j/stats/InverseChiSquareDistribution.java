package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style inverse chi-square distribution with degrees of freedom and scale. */
public final class InverseChiSquareDistribution implements ContinuousDistribution {

    private final double degreesOfFreedom;
    private final double scale;
    private final InverseGammaDistribution inverseGammaEquivalent;

    public InverseChiSquareDistribution() {
        this(1.0);
    }

    public InverseChiSquareDistribution(double degreesOfFreedom) {
        this(degreesOfFreedom, 1.0 / degreesOfFreedom);
    }

    public InverseChiSquareDistribution(double degreesOfFreedom, double scale) {
        validateDegreesOfFreedom(degreesOfFreedom);
        validateScale(scale);
        this.degreesOfFreedom = degreesOfFreedom;
        this.scale = scale;
        this.inverseGammaEquivalent = new InverseGammaDistribution(
            0.5 * degreesOfFreedom,
            0.5 * degreesOfFreedom * scale
        );
    }

    public double degreesOfFreedom() {
        return degreesOfFreedom;
    }

    public double scale() {
        return scale;
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
        return inverseGammaEquivalent.pdf(x);
    }

    /** Delegate to the equivalent inverse-gamma: both forms are identical up to re-parameterisation. */
    @Override
    public double logPdf(double x) {
        return inverseGammaEquivalent.logPdf(x);
    }

    @Override
    public double cdf(double x) {
        return inverseGammaEquivalent.cdf(x);
    }

    @Override
    public double ccdf(double x) {
        return inverseGammaEquivalent.ccdf(x);
    }

    @Override
    public double quantile(double probability) {
        return inverseGammaEquivalent.quantile(probability);
    }

    @Override
    public double quantileUpperTail(double probability) {
        return inverseGammaEquivalent.quantileUpperTail(probability);
    }

    public double mode() {
        return inverseGammaEquivalent.mode();
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        return inverseGammaEquivalent.mean();
    }

    @Override
    public double variance() {
        return inverseGammaEquivalent.variance();
    }

    public double skewness() {
        return inverseGammaEquivalent.skewness();
    }

    public double kurtosis() {
        return inverseGammaEquivalent.kurtosis();
    }

    public double kurtosisExcess() {
        return inverseGammaEquivalent.kurtosisExcess();
    }

    private static void validateDegreesOfFreedom(double degreesOfFreedom) {
        if (!(degreesOfFreedom > 0.0) || !Double.isFinite(degreesOfFreedom)) {
            throw new IllegalArgumentException(
                "degreesOfFreedom must be finite and greater than 0.0: " + degreesOfFreedom
            );
        }
    }

    private static void validateScale(double scale) {
        if (!(scale > 0.0) || !Double.isFinite(scale)) {
            throw new IllegalArgumentException("scale must be finite and greater than 0.0: " + scale);
        }
    }
}