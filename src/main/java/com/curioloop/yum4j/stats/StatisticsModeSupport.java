package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Minima;

final class StatisticsModeSupport {

    private static final int MODE_MINIMIZER_BITS = 48;
    private static final int MODE_MINIMIZER_ITERATIONS = 256;

    private StatisticsModeSupport() {
    }

    static double finiteSupportMode(ContinuousDistribution distribution, double lowerBound, double upperBound) {
        if (!Double.isFinite(lowerBound) || !Double.isFinite(upperBound) || !(lowerBound < upperBound)) {
            throw new IllegalArgumentException(
                "finiteSupportMode requires finite lower < upper bounds: lower=" + lowerBound + ", upper=" + upperBound
            );
        }
        double lower = Math.nextUp(lowerBound);
        double upper = Math.nextDown(upperBound);
        if (!(lower < upper)) {
            throw new IllegalArgumentException("finiteSupportMode requires a non-empty open interval");
        }

        return minimizePdf(distribution, lower, upper, "Finite-support mode search did not converge");
    }

    static double positiveSupportMode(ContinuousDistribution distribution, double guess) {
        if (!(guess > 0.0) || !Double.isFinite(guess)) {
            throw new IllegalArgumentException("positiveSupportMode requires a finite positive guess: " + guess);
        }

        double upperBound = guess;
        double value = distribution.pdf(upperBound);
        double maxValue;
        do {
            maxValue = value;
            double nextUpper = upperBound * 2.0;
            if (!Double.isFinite(nextUpper) || nextUpper <= upperBound) {
                break;
            }
            upperBound = nextUpper;
            value = distribution.pdf(upperBound);
        } while (maxValue < value);

        double lowerBound = upperBound;
        do {
            maxValue = value;
            lowerBound /= 2.0;
            if (lowerBound < Double.MIN_NORMAL) {
                return 0.0;
            }
            value = distribution.pdf(lowerBound);
        } while (maxValue < value);

        return minimizePdf(distribution, lowerBound, upperBound, "Positive-support mode search did not converge");
    }

    static double unitIntervalMode(ContinuousDistribution distribution, double guess) {
        if (!Double.isFinite(guess)) {
            throw new IllegalArgumentException("unitIntervalMode requires a finite guess: " + guess);
        }

        double upperBound = Math.min(Math.max(guess, Double.MIN_NORMAL), Math.nextDown(1.0));
        double value = distribution.pdf(upperBound);
        double maxValue;
        do {
            maxValue = value;
            double nextUpper = 1.0 - (1.0 - upperBound) / 2.0;
            if (!(nextUpper < 1.0) || nextUpper == upperBound) {
                return 1.0;
            }
            upperBound = nextUpper;
            value = distribution.pdf(upperBound);
        } while (maxValue < value);

        double lowerBound = upperBound;
        do {
            maxValue = value;
            lowerBound /= 2.0;
            if (lowerBound < Double.MIN_NORMAL) {
                return 0.0;
            }
            value = distribution.pdf(lowerBound);
        } while (maxValue < value);

        return minimizePdf(distribution, lowerBound, upperBound, "Unit-interval mode search did not converge");
    }

    private static double minimizePdf(ContinuousDistribution distribution,
                                      double lowerBound,
                                      double upperBound,
                                      String failureMessage) {
        Minima result = Minima.brentFindMinima(
            x -> -distribution.pdf(x),
            lowerBound,
            upperBound,
            MODE_MINIMIZER_BITS,
            MODE_MINIMIZER_ITERATIONS
        );
        if (!result.converged()) {
            throw new ArithmeticException(failureMessage);
        }
        return result.point();
    }
}