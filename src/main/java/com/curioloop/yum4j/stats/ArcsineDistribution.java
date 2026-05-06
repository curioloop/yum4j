package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/** Boost-style arcsine distribution over a bounded support. */
public value record ArcsineDistribution(double xMin, double xMax) implements ContinuousDistribution {

    public ArcsineDistribution() {
        this(0.0, 1.0);
    }

    public ArcsineDistribution {
        if (!Double.isFinite(xMin)) {
            throw new IllegalArgumentException("xMin must be finite: " + xMin);
        }
        if (!Double.isFinite(xMax)) {
            throw new IllegalArgumentException("xMax must be finite: " + xMax);
        }
        if (!(xMin < xMax)) {
            throw new IllegalArgumentException("xMax must be greater than xMin: xMin=" + xMin + ", xMax=" + xMax);
        }
    }

    @Override
    public Double2 range() {
        return Double2.bound(xMin, xMax);
    }

    @Override
    public Double2 support() {
        return Double2.bound(xMin, xMax);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (x == xMin || x == xMax) {
            return Double.POSITIVE_INFINITY;
        }
        return 1.0 / (Math.PI * Math.sqrt((x - xMin) * (xMax - x)));
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x == xMin) {
            return 0.0;
        }
        if (x == xMax) {
            return 1.0;
        }
        return 2.0 * Math.asin(Math.sqrt((x - xMin) / (xMax - xMin))) / Math.PI;
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x == xMin) {
            return 1.0;
        }
        if (x == xMax) {
            return 0.0;
        }
        return 2.0 * Math.acos(Math.sqrt((x - xMin) / (xMax - xMin))) / Math.PI;
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return xMin;
        }
        if (probability == 1.0) {
            return xMax;
        }
        double sinHalfPi = Math.sin(0.5 * Math.PI * probability);
        double weight = sinHalfPi * sinHalfPi;
        return xMin + (xMax - xMin) * weight;
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return xMin;
        }
        if (probability == 0.0) {
            return xMax;
        }
        double cosHalfPi = Math.cos(0.5 * Math.PI * probability);
        double weight = cosHalfPi * cosHalfPi;
        return xMin + (xMax - xMin) * weight;
    }

    public double mode() {
        throw new ArithmeticException("The arcsine distribution has two modes at xMin and xMax");
    }

    public double median() {
        return 0.5 * (xMin + xMax);
    }

    @Override
    public double mean() {
        return 0.5 * (xMin + xMax);
    }

    @Override
    public double variance() {
        double width = xMax - xMin;
        return width * width / 8.0;
    }

    public double skewness() {
        return 0.0;
    }

    public double kurtosis() {
        return 1.5;
    }

    public double kurtosisExcess() {
        return -1.5;
    }

    private void validateX(double x) {
        if (!Double.isFinite(x) || x < xMin || x > xMax) {
            throw new IllegalArgumentException("x must be finite and in [xMin, xMax]: " + x);
        }
    }

    private static void validateProbability(double probability) {
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}