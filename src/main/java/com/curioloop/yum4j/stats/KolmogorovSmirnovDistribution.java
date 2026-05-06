package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.JacobiTheta;

/** Boost-style asymptotic Kolmogorov-Smirnov distribution parameterized by sample size. */
public value record KolmogorovSmirnovDistribution(double numberOfObservations) implements ContinuousDistribution {

    private static final double PI = Math.PI;
    private static final double PI_SQUARED = Math.PI * Math.PI;
    private static final double ROOT_HALF_PI = Math.sqrt(Math.PI / 2.0);
    private static final double APERY = 1.2020569031595942854;
    private static final double SERIES_EPSILON = 1.0e-15;

    public KolmogorovSmirnovDistribution {
        validateNumberOfObservations(numberOfObservations);
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
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        if (2.0 * x * x * numberOfObservations < PI) {
            return pdfSmallX(x, numberOfObservations);
        }
        return pdfLargeX(x, numberOfObservations);
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (x == 0.0) {
            return 0.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        return JacobiTheta.theta4Tau(0.0, 2.0 * x * x * numberOfObservations / PI);
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (x == 0.0) {
            return 1.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        double tau = 2.0 * x * x * numberOfObservations / PI;
        if (tau > 1.0) {
            return -JacobiTheta.theta4MinusOneTau(0.0, tau);
        }
        return 1.0 - JacobiTheta.theta4Tau(0.0, tau);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            return 1.0;
        }
        return StatisticsQuantileSupport.continuousQuantile(this, probability, quantileGuess(probability));
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            return 1.0;
        }
        return StatisticsQuantileSupport.continuousUpperTailQuantile(this, probability, quantileGuess(1.0 - probability));
    }

    public double mode() {
        double standardized = maximizePdf(0.0, 1.0);
        return standardized / Math.sqrt(numberOfObservations);
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        return ROOT_HALF_PI * Math.log(2.0) / Math.sqrt(numberOfObservations);
    }

    @Override
    public double variance() {
        return ((PI * PI / 6.0) - PI * Math.log(2.0) * Math.log(2.0)) / (2.0 * numberOfObservations);
    }

    public double skewness() {
        double mean = mean();
        double variance = variance();
        double ex3 = 0.5625 * ROOT_HALF_PI * APERY / (numberOfObservations * Math.sqrt(numberOfObservations));
        return (ex3 - 3.0 * mean * variance - mean * mean * mean) / (variance * Math.sqrt(variance));
    }

    public double kurtosis() {
        double mean = mean();
        double variance = variance();
        double ex4 = 7.0 * Math.pow(PI * PI / 6.0, 2.0) / (20.0 * numberOfObservations * numberOfObservations);
        double skewness = skewness();
        return (ex4 - 4.0 * mean * skewness * variance * Math.sqrt(variance)
            - 6.0 * mean * mean * variance - Math.pow(mean, 4.0)) / (variance * variance);
    }

    public double kurtosisExcess() {
        return kurtosis() - 3.0;
    }

    private double quantileGuess(double probability) {
        double scaledGuess;
        if (probability > 0.9) {
            scaledGuess = 1.8 - 5.0 * (1.0 - probability);
        } else if (probability < 0.3) {
            scaledGuess = probability + 0.45;
        } else {
            scaledGuess = probability + 0.3;
        }
        return scaledGuess / Math.sqrt(numberOfObservations);
    }

    private double maximizePdf(double lower, double upper) {
        double left = lower;
        double right = upper;
        double invPhi = (Math.sqrt(5.0) - 1.0) / 2.0;
        double c = right - invPhi * (right - left);
        double d = left + invPhi * (right - left);
        double fc = -standardizedPdf(c);
        double fd = -standardizedPdf(d);
        for (int index = 0; index < 128; index++) {
            if (fc < fd) {
                right = d;
                d = c;
                fd = fc;
                c = right - invPhi * (right - left);
                fc = -standardizedPdf(c);
            } else {
                left = c;
                c = d;
                fc = fd;
                d = left + invPhi * (right - left);
                fd = -standardizedPdf(d);
            }
        }
        return 0.5 * (left + right);
    }

    private static double standardizedPdf(double x) {
        if (2.0 * x * x < PI) {
            return pdfSmallX(x, 1.0);
        }
        return pdfLargeX(x, 1.0);
    }

    private static double pdfSmallX(double x, double n) {
        double x2n = x * x * n;
        if (x2n * x2n == 0.0) {
            return 0.0;
        }
        double value = 0.0;
        double lastDelta = 0.0;
        for (int index = 0; ; index++) {
            double iPlusHalf = index + 0.5;
            double delta = Math.exp(-iPlusHalf * iPlusHalf * PI_SQUARED / (2.0 * x2n))
                * (iPlusHalf * iPlusHalf * PI_SQUARED - x2n);
            if (delta == 0.0) {
                break;
            }
            if (lastDelta != 0.0 && Math.abs(delta / lastDelta) < SERIES_EPSILON) {
                break;
            }
            value += 2.0 * delta;
            lastDelta = delta;
        }
        return value * Math.sqrt(n) * ROOT_HALF_PI / (x2n * x2n);
    }

    private static double pdfLargeX(double x, double n) {
        double value = 0.0;
        double lastDelta = 0.0;
        for (int index = 1; ; index++) {
            double delta = 8.0 * x * index * index * Math.exp(-2.0 * index * index * x * x * n);
            if (delta == 0.0) {
                break;
            }
            if (lastDelta != 0.0 && Math.abs(delta / lastDelta) < SERIES_EPSILON) {
                break;
            }
            if ((index & 1) == 0) {
                delta = -delta;
            }
            value += delta;
            lastDelta = delta;
        }
        return value * n;
    }

    private static void validateNumberOfObservations(double numberOfObservations) {
        if (!(numberOfObservations > 0.0) || !Double.isFinite(numberOfObservations)) {
            throw new IllegalArgumentException(
                "numberOfObservations must be finite and greater than 0.0: " + numberOfObservations
            );
        }
    }

    private static void validateX(double x) {
        if (Double.isNaN(x) || x < 0.0) {
            throw new IllegalArgumentException("x must be in [0, +infinity): " + x);
        }
    }

    private static void validateProbability(double probability) {
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}