package com.curioloop.yum4j.math;

import java.util.function.DoubleUnaryOperator;

/**
 * Shared low-level continued-fraction evaluators used by special-function kernels.
 */
public value record ContinuedFraction(double value, int iterations, boolean converged) {

    private static final double DEFAULT_TINY = 16.0 * Double.MIN_NORMAL;

    @FunctionalInterface
    public interface FractionTermGenerator {
        Double2 apply(int index);
    }

    public static ContinuedFraction evaluateFractionA(FractionTermGenerator generator,
                                                      double tolerance,
                                                      int maxIterations) {
        return evaluateFractionA(generator, ContinuedFraction::safeTiny, tolerance, maxIterations);
    }

    public static ContinuedFraction evaluateFractionA(FractionTermGenerator generator,
                                                      DoubleUnaryOperator denominatorGuard,
                                                      double tolerance,
                                                      int maxIterations) {
        Double2 first = generator.apply(0);
        double f = denominatorGuard.applyAsDouble(first.denominator());
        double a0 = first.numerator();
        double c = f;
        double d = 0.0;

        for (int index = 1; index <= maxIterations; index++) {
            Double2 term = generator.apply(index);
            d = denominatorGuard.applyAsDouble(term.denominator() + term.numerator() * d);
            c = denominatorGuard.applyAsDouble(term.denominator() + term.numerator() / c);
            d = 1.0 / d;
            double delta = c * d;
            f *= delta;
            if (Math.abs(delta - 1.0) <= Math.abs(tolerance)) {
                return new ContinuedFraction(a0 / f, index, true);
            }
        }

        return new ContinuedFraction(a0 / f, maxIterations, false);
    }

    public static ContinuedFraction evaluateFractionB(FractionTermGenerator generator,
                                                      double tolerance,
                                                      int maxIterations) {
        return evaluateFractionB(generator, ContinuedFraction::safeTiny, tolerance, maxIterations);
    }

    public static ContinuedFraction evaluateFractionB(FractionTermGenerator generator,
                                                      DoubleUnaryOperator denominatorGuard,
                                                      double tolerance,
                                                      int maxIterations) {
        Double2 first = generator.apply(0);
        double f = denominatorGuard.applyAsDouble(first.denominator());
        double c = f;
        double d = 0.0;

        for (int index = 1; index <= maxIterations; index++) {
            Double2 term = generator.apply(index);
            d = denominatorGuard.applyAsDouble(term.denominator() + term.numerator() * d);
            c = denominatorGuard.applyAsDouble(term.denominator() + term.numerator() / c);
            d = 1.0 / d;
            double delta = c * d;
            f *= delta;
            if (Math.abs(delta - 1.0) <= Math.abs(tolerance)) {
                return new ContinuedFraction(f, index, true);
            }
        }

        return new ContinuedFraction(f, maxIterations, false);
    }

    public static ContinuedFraction evaluateReciprocalSeededFractionB(FractionTermGenerator generator,
                                                                      double tolerance,
                                                                      int maxIterations) {
        return evaluateReciprocalSeededFractionB(generator, ContinuedFraction::safeTiny, tolerance, maxIterations);
    }

    public static ContinuedFraction evaluateReciprocalSeededFractionB(FractionTermGenerator generator,
                                                                      DoubleUnaryOperator denominatorGuard,
                                                                      double tolerance,
                                                                      int maxIterations) {
        Double2 first = generator.apply(0);
        double f = 1.0 / denominatorGuard.applyAsDouble(first.denominator());
        double c = 1.0;
        double d = f;

        for (int index = 1; index <= maxIterations; index++) {
            Double2 term = generator.apply(index);
            d = 1.0 / denominatorGuard.applyAsDouble(term.denominator() + term.numerator() * d);
            c = denominatorGuard.applyAsDouble(term.denominator() + term.numerator() / c);
            double delta = c * d;
            f *= delta;
            if (Math.abs(delta - 1.0) <= Math.abs(tolerance)) {
                return new ContinuedFraction(f, index, true);
            }
        }

        return new ContinuedFraction(f, maxIterations, false);
    }

    static double safeTiny(double value) {
        return Math.abs(value) < DEFAULT_TINY ? DEFAULT_TINY : value;
    }
}