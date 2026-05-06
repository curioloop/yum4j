package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;
import com.curioloop.yum4j.math.RootIterators;

final class StatisticsQuantileSupport {

    private static final int ROOT_BITS = 53;
    private static final int ROOT_ITERATIONS = 128;
    private static final int BRACKET_ITERATIONS = 256;

    private StatisticsQuantileSupport() {
    }

    static double continuousQuantile(ContinuousDistribution distribution, double probability, double guess) {
        validateProbability(probability);
        Double2 range = distribution.range();
        if (probability == 0.0) {
            return lowerQuantileEndpoint(range.lower());
        }
        if (probability == 1.0) {
            return upperQuantileEndpoint(range.upper());
        }
        return solveContinuousQuantile(distribution, probability, guess, range);
    }

    static double continuousUpperTailQuantile(ContinuousDistribution distribution, double probability, double guess) {
        validateProbability(probability);
        Double2 range = distribution.range();
        if (probability == 0.0) {
            return upperQuantileEndpoint(range.upper());
        }
        if (probability == 1.0) {
            return lowerQuantileEndpoint(range.lower());
        }
        return solveContinuousUpperTailQuantile(distribution, probability, guess, range);
    }

    static double discreteQuantile(DiscreteDistribution distribution, double probability) {
        validateProbability(probability);
        Double2 range = distribution.range();
        Double2 support = distribution.support();
        if (probability == 0.0) {
            return lowerQuantileEndpoint(range.lower());
        }
        if (probability == 1.0) {
            return upperQuantileEndpoint(range.upper());
        }

        long lower = ceilToLong(support.lower());
        long upper = initialDiscreteUpperBound(distribution, support);
        while (!cdfAtLeast(distribution.cdf(upper), probability)) {
            if (upper == Long.MAX_VALUE) {
                return Double.POSITIVE_INFINITY;
            }
            long nextUpper = upper >= Long.MAX_VALUE / 2L ? Long.MAX_VALUE : upper * 2L;
            if (nextUpper <= upper) {
                nextUpper = Long.MAX_VALUE;
            }
            upper = nextUpper;
        }

        while (lower < upper) {
            long midpoint = lower + (upper - lower) / 2L;
            if (cdfAtLeast(distribution.cdf(midpoint), probability)) {
                upper = midpoint;
            } else {
                lower = midpoint + 1L;
            }
        }
        return lower;
    }

    static double discreteUpperTailQuantile(DiscreteDistribution distribution, double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return upperQuantileEndpoint(distribution.range().upper());
        }
        if (probability == 1.0) {
            return lowerQuantileEndpoint(distribution.range().lower());
        }
        return discreteQuantile(distribution, 1.0 - probability);
    }

    private static double solveContinuousQuantile(ContinuousDistribution distribution,
                                                  double probability,
                                                  double guess,
                                                  Double2 range) {
        double seed = initialContinuousGuess(distribution, guess, range);
        double valueAtSeed = quantileValue(distribution, probability, seed);
        if (valueAtSeed == 0.0) {
            return seed;
        }

        double step = initialStep(distribution, seed, range);
        return valueAtSeed < 0.0
            ? solveContinuousQuantileFromBelow(distribution, probability, seed, step, range.upper())
            : solveContinuousQuantileFromAbove(distribution, probability, seed, step, range.lower());
    }

    private static double solveContinuousUpperTailQuantile(ContinuousDistribution distribution,
                                                           double probability,
                                                           double guess,
                                                           Double2 range) {
        double seed = initialContinuousGuess(distribution, guess, range);
        double valueAtSeed = upperTailQuantileValue(distribution, probability, seed);
        if (valueAtSeed == 0.0) {
            return seed;
        }

        double step = initialStep(distribution, seed, range);
        return valueAtSeed < 0.0
            ? solveContinuousUpperTailQuantileFromBelow(distribution, probability, seed, step, range.upper())
            : solveContinuousUpperTailQuantileFromAbove(distribution, probability, seed, step, range.lower());
    }

    private static double solveContinuousQuantileFromBelow(ContinuousDistribution distribution,
                                                           double probability,
                                                           double seed,
                                                           double step,
                                                           double upperBound) {
        double lower = seed;
        double upper = seed;
        double nextStep = step;
        for (int index = 0; index < BRACKET_ITERATIONS; index++) {
            upper = nextUpper(lower, nextStep, upperBound);
            double value = quantileValue(distribution, probability, upper);
            if (value >= 0.0) {
                return solveBracketedContinuousQuantile(
                    new ContinuousQuantileFunction(distribution, probability),
                    seed,
                    lower,
                    upper,
                    "continuousQuantile"
                );
            }
            lower = upper;
            nextStep *= 2.0;
        }
        throw new ArithmeticException("Unable to bracket continuous quantile root");
    }

    private static double solveContinuousQuantileFromAbove(ContinuousDistribution distribution,
                                                           double probability,
                                                           double seed,
                                                           double step,
                                                           double lowerBound) {
        double lower = seed;
        double upper = seed;
        double nextStep = step;
        for (int index = 0; index < BRACKET_ITERATIONS; index++) {
            lower = nextLower(upper, nextStep, lowerBound);
            double value = quantileValue(distribution, probability, lower);
            if (value <= 0.0) {
                return solveBracketedContinuousQuantile(
                    new ContinuousQuantileFunction(distribution, probability),
                    seed,
                    lower,
                    upper,
                    "continuousQuantile"
                );
            }
            upper = lower;
            nextStep *= 2.0;
        }
        throw new ArithmeticException("Unable to bracket continuous quantile root");
    }

    private static double solveContinuousUpperTailQuantileFromBelow(ContinuousDistribution distribution,
                                                                    double probability,
                                                                    double seed,
                                                                    double step,
                                                                    double upperBound) {
        double lower = seed;
        double upper = seed;
        double nextStep = step;
        for (int index = 0; index < BRACKET_ITERATIONS; index++) {
            upper = nextUpper(lower, nextStep, upperBound);
            double value = upperTailQuantileValue(distribution, probability, upper);
            if (value >= 0.0) {
                return solveBracketedContinuousQuantile(
                    new ContinuousUpperTailQuantileFunction(distribution, probability),
                    seed,
                    lower,
                    upper,
                    "continuousUpperTailQuantile"
                );
            }
            lower = upper;
            nextStep *= 2.0;
        }
        throw new ArithmeticException("Unable to bracket continuous quantile root");
    }

    private static double solveContinuousUpperTailQuantileFromAbove(ContinuousDistribution distribution,
                                                                    double probability,
                                                                    double seed,
                                                                    double step,
                                                                    double lowerBound) {
        double lower = seed;
        double upper = seed;
        double nextStep = step;
        for (int index = 0; index < BRACKET_ITERATIONS; index++) {
            lower = nextLower(upper, nextStep, lowerBound);
            double value = upperTailQuantileValue(distribution, probability, lower);
            if (value <= 0.0) {
                return solveBracketedContinuousQuantile(
                    new ContinuousUpperTailQuantileFunction(distribution, probability),
                    seed,
                    lower,
                    upper,
                    "continuousUpperTailQuantile"
                );
            }
            upper = lower;
            nextStep *= 2.0;
        }
        throw new ArithmeticException("Unable to bracket continuous quantile root");
    }

    private static double solveBracketedContinuousQuantile(RootIterators.FirstOrderFunction function,
                                                           double seed,
                                                           double lower,
                                                           double upper,
                                                           String functionName) {
        if (lower == upper) {
            return lower;
        }

        double clampedGuess = Math.min(Math.max(seed, lower), upper);
        return RootIterators.newtonRaphsonIterate(
            function,
            clampedGuess,
            lower,
            upper,
            ROOT_BITS,
            ROOT_ITERATIONS,
            functionName
        );
    }

    private static double quantileValue(ContinuousDistribution distribution,
                                        double probability,
                                        double x) {
        return distribution.cdf(x) - probability;
    }

    private static double upperTailQuantileValue(ContinuousDistribution distribution,
                                                 double probability,
                                                 double x) {
        return probability - distribution.ccdf(x);
    }

    private static double initialContinuousGuess(ContinuousDistribution distribution,
                                                 double guess,
                                                 Double2 range) {
        if (Double.isFinite(guess) && guess >= range.lower() && guess <= range.upper()) {
            return guess;
        }
        try {
            double mean = distribution.mean();
            if (Double.isFinite(mean) && mean >= range.lower() && mean <= range.upper()) {
                return mean;
            }
        } catch (ArithmeticException ignored) {
        }
        if (Double.isFinite(range.lower()) && Double.isFinite(range.upper())) {
            return 0.5 * (range.lower() + range.upper());
        }
        if (Double.isFinite(range.lower()) && range.lower() >= 0.0) {
            return Math.max(1.0, range.lower());
        }
        if (Double.isFinite(range.upper()) && range.upper() <= 0.0) {
            return Math.min(-1.0, range.upper());
        }
        return 0.0;
    }

    private static double initialStep(ContinuousDistribution distribution,
                                      double seed,
                                      Double2 range) {
        double step = 1.0;
        try {
            double standardDeviation = distribution.standardDeviation();
            if (Double.isFinite(standardDeviation) && standardDeviation > 0.0) {
                step = standardDeviation;
            }
        } catch (ArithmeticException ignored) {
        }
        if (Double.isFinite(seed)) {
            step = Math.max(step, Math.max(Math.ulp(seed) * 16.0, Math.abs(seed) * 0.25));
        }
        if (Double.isFinite(range.lower()) && Double.isFinite(range.upper())) {
            step = Math.min(step, 0.25 * (range.upper() - range.lower()));
        }
        return Math.max(step, Double.MIN_NORMAL * 16.0);
    }

    private static double nextUpper(double current, double step, double upperBound) {
        if (Double.isFinite(upperBound)) {
            return Math.min(upperBound, current + step);
        }
        double next = current + step;
        if (!Double.isFinite(next)) {
            return Double.MAX_VALUE / 4.0;
        }
        return next;
    }

    private static double nextLower(double current, double step, double lowerBound) {
        if (Double.isFinite(lowerBound)) {
            return Math.max(lowerBound, current - step);
        }
        double next = current - step;
        if (!Double.isFinite(next)) {
            return -Double.MAX_VALUE / 4.0;
        }
        return next;
    }

    private static long initialDiscreteUpperBound(DiscreteDistribution distribution, Double2 support) {
        if (Double.isFinite(support.upper())) {
            return floorToLong(support.upper());
        }
        long lower = ceilToLong(support.lower());
        long upper = Math.max(1L, lower);
        try {
            double mean = distribution.mean();
            if (Double.isFinite(mean)) {
                upper = Math.max(upper, ceilToLong(mean));
            }
        } catch (ArithmeticException ignored) {
        }
        return upper;
    }

    private static long ceilToLong(double value) {
        if (value <= Long.MIN_VALUE) {
            return Long.MIN_VALUE;
        }
        if (value >= Long.MAX_VALUE) {
            return Long.MAX_VALUE;
        }
        return (long) Math.ceil(value);
    }

    private static long floorToLong(double value) {
        if (value <= Long.MIN_VALUE) {
            return Long.MIN_VALUE;
        }
        if (value >= Long.MAX_VALUE) {
            return Long.MAX_VALUE;
        }
        return (long) Math.floor(value);
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }

    private static final class ContinuousQuantileFunction implements RootIterators.FirstOrderFunction {

        private final ContinuousDistribution distribution;
        private final double probability;

        private ContinuousQuantileFunction(ContinuousDistribution distribution, double probability) {
            this.distribution = distribution;
            this.probability = probability;
        }

        @Override
        public void evaluate(double x, double[] evaluation) {
            fillQuantileEvaluation(distribution, probability, x, evaluation);
        }
    }

    private static final class ContinuousUpperTailQuantileFunction implements RootIterators.FirstOrderFunction {

        private final ContinuousDistribution distribution;
        private final double probability;

        private ContinuousUpperTailQuantileFunction(ContinuousDistribution distribution, double probability) {
            this.distribution = distribution;
            this.probability = probability;
        }

        @Override
        public void evaluate(double x, double[] evaluation) {
            fillUpperTailQuantileEvaluation(distribution, probability, x, evaluation);
        }
    }

    private static void fillQuantileEvaluation(ContinuousDistribution distribution,
                                               double probability,
                                               double x,
                                               double[] evaluation) {
        double value = quantileValue(distribution, probability, x);
        double density = distribution.pdf(x);
        evaluation[0] = value;
        evaluation[1] = density;
    }

    private static void fillUpperTailQuantileEvaluation(ContinuousDistribution distribution,
                                                        double probability,
                                                        double x,
                                                        double[] evaluation) {
        double value = upperTailQuantileValue(distribution, probability, x);
        double density = distribution.pdf(x);
        evaluation[0] = value;
        evaluation[1] = density;
    }

    private static boolean cdfAtLeast(double cdf, double probability) {
        double tolerance = Math.max(1.0e-12, Math.max(Math.ulp(probability), Math.ulp(cdf)) * 8.0);
        return cdf + tolerance >= probability;
    }

    private static double lowerQuantileEndpoint(double lower) {
        return lower == -Double.MAX_VALUE ? Double.NEGATIVE_INFINITY : lower;
    }

    private static double upperQuantileEndpoint(double upper) {
        return upper == Double.MAX_VALUE ? Double.POSITIVE_INFINITY : upper;
    }
}