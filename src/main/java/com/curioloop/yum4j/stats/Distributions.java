package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.Objects;

/** Boost-style free-function accessors for distribution objects. */
public final class Distributions {

    private Distributions() {
    }

    public value record Complement<D extends Distribution>(D distribution, double value) {
        public Complement {
            Objects.requireNonNull(distribution, "distribution cannot be null");
        }
    }

    public static <D extends Distribution> Complement<D> complement(D distribution, double value) {
        return new Complement<>(distribution, value);
    }

    public static <D extends Distribution> double pdf(D distribution, double x) {
        return requireDistribution(distribution).pdf(x);
    }

    public static <D extends Distribution> double cdf(D distribution, double x) {
        return requireDistribution(distribution).cdf(x);
    }

    public static <D extends Distribution> double cdf(Complement<D> complement) {
        return requireComplement(complement).distribution().ccdf(complement.value());
    }

    public static <D extends Distribution> double quantile(D distribution, double probability) {
        D resolved = requireDistribution(distribution);
        validateProbability(probability);
        return resolved.quantile(probability);
    }

    public static <D extends Distribution> double quantile(Complement<D> complement) {
        Complement<D> resolved = requireComplement(complement);
        validateProbability(resolved.value());
        return resolved.distribution().quantileUpperTail(resolved.value());
    }

    public static Double2 range(Distribution distribution) {
        return requireDistribution(distribution).range();
    }

    public static Double2 support(Distribution distribution) {
        return requireDistribution(distribution).support();
    }

    public static double mean(Distribution distribution) {
        return requireDistribution(distribution).mean();
    }

    public static double variance(Distribution distribution) {
        return requireDistribution(distribution).variance();
    }

    public static double standardDeviation(Distribution distribution) {
        return requireDistribution(distribution).standardDeviation();
    }

    public static double hazard(Distribution distribution, double x) {
        Distribution resolved = requireDistribution(distribution);
        double survivor = resolved.ccdf(x);
        double density = resolved.pdf(x);
        if (density > survivor * Double.MAX_VALUE) {
            throw new ArithmeticException("Hazard overflow for x=" + x);
        }
        if (density == 0.0) {
            return 0.0;
        }
        return density / survivor;
    }

    public static double chf(Distribution distribution, double x) {
        return -Math.log(cdf(complement(requireDistribution(distribution), x)));
    }

    public static double coefficientOfVariation(Distribution distribution) {
        Distribution resolved = requireDistribution(distribution);
        double mean = resolved.mean();
        double standardDeviation = resolved.standardDeviation();
        if (mean == 0.0 || (!Double.isInfinite(standardDeviation) && Math.abs(mean) < 1.0
            && standardDeviation > Math.abs(mean) * Double.MAX_VALUE)) {
            throw new ArithmeticException("Coefficient of variation is undefined for mean=" + mean);
        }
        return standardDeviation / mean;
    }

    private static <D extends Distribution> D requireDistribution(D distribution) {
        return Objects.requireNonNull(distribution, "distribution cannot be null");
    }

    private static <D extends Distribution> Complement<D> requireComplement(Complement<D> complement) {
        return Objects.requireNonNull(complement, "complement cannot be null");
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}