package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

/**
 * Common distribution contract used by Boost-style free-function accessors.
 *
 * <p>Concrete distribution classes keep their direct instance methods; this interface provides the
 * shared shape needed for complement-style accessors and generic distribution utilities.
 */
public interface Distribution {

    double pdf(double x);

    double cdf(double x);

    double ccdf(double x);

    double quantile(double probability);

    double quantileUpperTail(double probability);

    double mean();

    double variance();

    Double2 range();

    Double2 support();

    default double standardDeviation() {
        double variance = variance();
        if (Double.isNaN(variance) || variance < 0.0) {
            throw new ArithmeticException("Standard deviation is undefined for variance: " + variance);
        }
        return Math.sqrt(variance);
    }
}