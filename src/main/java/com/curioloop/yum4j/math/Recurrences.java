package com.curioloop.yum4j.math;

final class Recurrences {

    private static final double BESSEL_SEED_EPSILON = 2.0 * Math.ulp(1.0);
    private static final int BESSEL_MAX_SEED_ITERATIONS = 100000;
    private static final double RECURRENCE_RESCALE_MARGIN = 2048.0;

    private Recurrences() {
    }

    static double besselJyBackwardRatio(double order,
                                        double x,
                                        double factor,
                                        int maxTerms,
                                        String context) {
        return evaluateFractionA(
            index -> backwardRatioTerm(besselJyCoefficient(order, x, index)),
            factor,
            maxTerms,
            context);
    }

    static double besselIkBackwardRatio(double order,
                                        double x,
                                        double factor,
                                        int maxTerms,
                                        String context) {
        return evaluateFractionA(
            index -> backwardRatioTerm(besselIkCoefficient(order, x, index)),
            factor,
            maxTerms,
            context);
    }

    static double besselIkForwardRatio(double order,
                                       double x,
                                       double factor,
                                       int maxTerms,
                                       String context) {
        return evaluateFractionA(
            index -> forwardRatioTerm(besselIkCoefficient(order, x, index == 0 ? 0 : -index)),
            factor,
            maxTerms,
            context);
    }

    static void validateBesselJBackwardOrder(double order) {
        if (order < 0.0) {
            throw new IllegalArgumentException("Bessel J backward recurrence requires order >= 0: " + order);
        }
    }

    static void validateBesselJBackwardPairOrder(double order) {
        if (order < -1.0) {
            throw new IllegalArgumentException(
                "Bessel J backward recurrence requires order >= -1 for adjacent-pair initialization: " + order);
        }
    }

    static void validateBesselIBackwardOrder(double order) {
        if (order < -1.0) {
            throw new IllegalArgumentException("Bessel I backward recurrence requires order >= -1: " + order);
        }
    }

    static void validateBesselIForwardOrder(double order) {
        if (order > 1.0) {
            throw new IllegalArgumentException("Bessel I forward recurrence requires order <= 1: " + order);
        }
    }

    static double besselJBackwardSeedNext(double order, double x, double current) {
        validateBesselJBackwardOrder(order);
        return current * besselJyBackwardRatio(
            order + 1.0,
            x,
            BESSEL_SEED_EPSILON,
            BESSEL_MAX_SEED_ITERATIONS,
            "bessel_j_backwards_iterator seed ratio");
    }

    static double besselIBackwardSeedNext(double order, double x, double current) {
        validateBesselIBackwardOrder(order);
        return current * besselIkBackwardRatio(
            order + 1.0,
            x,
            BESSEL_SEED_EPSILON,
            BESSEL_MAX_SEED_ITERATIONS,
            "bessel_i_backwards_iterator seed ratio");
    }

    static double besselIForwardSeedPrevious(double order, double x, double current) {
        validateBesselIForwardOrder(order);
        return current * besselIkForwardRatio(
            order - 1.0,
            x,
            BESSEL_SEED_EPSILON,
            BESSEL_MAX_SEED_ITERATIONS,
            "bessel_i_forwards_iterator seed ratio");
    }

    static double besselJBackwardPrevious(double order, double x, double current, double next) {
        return 2.0 * order * current / x - next;
    }

    static double besselIBackwardPrevious(double order, double x, double current, double next) {
        return 2.0 * order * current / x + next;
    }

    static double besselIForwardNext(double order, double x, double previous, double current) {
        return previous - 2.0 * order * current / x;
    }

    static double oneFOneBackwardRatioOnB(double a,
                                          double b,
                                          double z,
                                          double factor,
                                          int maxTerms,
                                          String context) {
        return evaluateFractionA(
            index -> backwardRatioTerm(oneFOneCoefficientOnB(a, b, z, index)),
            factor,
            maxTerms,
            context);
    }

    static double oneFOneBackwardRatioOnAAndB(double a,
                                              double b,
                                              double z,
                                              int offset,
                                              double factor,
                                              int maxTerms,
                                              String context) {
        return evaluateFractionA(
            index -> backwardRatioTerm(oneFOneCoefficientOnAAndB(a, b, z, offset, index)),
            factor,
            maxTerms,
            context);
    }

    static double oneFOneForwardRatioOnB(double a,
                                         double b,
                                         double z,
                                         double factor,
                                         int maxTerms,
                                         String context) {
        return evaluateFractionA(
            index -> forwardRatioTerm(oneFOneCoefficientOnB(a, b, z, index == 0 ? 0 : -index)),
            factor,
            maxTerms,
            context);
    }

    static double oneFOneForwardRatioOnAAndB(double a,
                                             double b,
                                             double z,
                                             int offset,
                                             double factor,
                                             int maxTerms,
                                             String context) {
        return evaluateFractionA(
            index -> forwardRatioTerm(oneFOneCoefficientOnAAndB(a, b, z, offset, index == 0 ? 0 : -index)),
            factor,
            maxTerms,
            context);
    }

    private static double evaluateFractionA(ContinuedFraction.FractionTermGenerator generator,
                                            double tolerance,
                                            int maxTerms,
                                            String context) {
        ContinuedFraction fraction = ContinuedFraction.evaluateFractionA(generator, tolerance, maxTerms);
        if (fraction.converged()) {
            return fraction.value();
        }
        throw new IllegalStateException("Continued fraction did not converge for " + context);
    }

    static double oneFOneApplyForwardOnA(double a,
                                         double b,
                                         double z,
                                         int steps,
                                         double first,
                                         double second,
                                         long[] logScalingHolder,
                                         double[] previousHolder) {
        for (int k = 0; k < steps; k++) {
            Double3 coefficient = oneFOneCoefficientOnA(a, b, z, k);
            if (logScalingHolder != null
                && (shouldRescaleRecurrenceTerm(first, coefficient._1() / coefficient._3())
                || shouldRescaleRecurrenceTerm(second, coefficient._2() / coefficient._3()))) {
                long logScale = (long) Math.log(Math.abs(second));
                if (logScale != 0L) {
                    double scale = Math.exp(-logScale);
                    second *= scale;
                    first *= scale;
                    logScalingHolder[0] += logScale;
                }
            }
            double third = (coefficient._1() / -coefficient._3()) * first
                + (coefficient._2() / -coefficient._3()) * second;
            first = second;
            second = third;
        }
        if (previousHolder != null) {
            previousHolder[0] = first;
        }
        return second;
    }

    static double oneFOneApplyForwardOnB(double a,
                                         double b,
                                         double z,
                                         int steps,
                                         double first,
                                         double second,
                                         long[] logScalingHolder,
                                         double[] previousHolder) {
        for (int k = 0; k < steps; k++) {
            Double3 coefficient = oneFOneCoefficientOnB(a, b, z, k);
            if (logScalingHolder != null
                && (shouldRescaleRecurrenceTerm(first, coefficient._1() / coefficient._3())
                || shouldRescaleRecurrenceTerm(second, coefficient._2() / coefficient._3()))) {
                long logScale = (long) Math.log(Math.abs(second));
                if (logScale != 0L) {
                    double scale = Math.exp(-logScale);
                    second *= scale;
                    first *= scale;
                    logScalingHolder[0] += logScale;
                }
            }
            double third = (coefficient._1() / -coefficient._3()) * first
                + (coefficient._2() / -coefficient._3()) * second;
            first = second;
            second = third;
        }
        if (previousHolder != null) {
            previousHolder[0] = first;
        }
        return second;
    }

    static double oneFOneApplyForwardOnAAndB(double a,
                                             double b,
                                             double z,
                                             int offset,
                                             int steps,
                                             double first,
                                             double second,
                                             long[] logScalingHolder,
                                             double[] previousHolder) {
        for (int k = 0; k < steps; k++) {
            Double3 coefficient = oneFOneCoefficientOnAAndB(a, b, z, offset, k);
            if (logScalingHolder != null
                && (shouldRescaleRecurrenceTerm(first, coefficient._1() / coefficient._3())
                || shouldRescaleRecurrenceTerm(second, coefficient._2() / coefficient._3()))) {
                long logScale = (long) Math.log(Math.abs(second));
                if (logScale != 0L) {
                    double scale = Math.exp(-logScale);
                    second *= scale;
                    first *= scale;
                    logScalingHolder[0] += logScale;
                }
            }
            double third = (coefficient._1() / -coefficient._3()) * first
                + (coefficient._2() / -coefficient._3()) * second;
            first = second;
            second = third;
        }
        if (previousHolder != null) {
            previousHolder[0] = first;
        }
        return second;
    }

    static double oneFOneApplyBackwardOnA(double a,
                                          double b,
                                          double z,
                                          int steps,
                                          double first,
                                          double second,
                                          long[] logScalingHolder,
                                          double[] previousHolder) {
        for (int k = 0; k < steps; k++) {
            Double3 coefficient = oneFOneCoefficientOnA(a, b, z, -k);
            if ((logScalingHolder != null && second != 0.0
                && shouldRescaleRecurrenceTerm(second, coefficient._2() / coefficient._1()))
                || (logScalingHolder != null
                && shouldRescaleRecurrenceTerm(first, coefficient._3() / coefficient._1()))) {
                long logScale = (long) Math.log(Math.abs(second));
                if (logScale != 0L) {
                    double scale = Math.exp(-logScale);
                    second *= scale;
                    first *= scale;
                    logScalingHolder[0] += logScale;
                }
            }
            double next = (coefficient._2() / -coefficient._1()) * second
                + (coefficient._3() / -coefficient._1()) * first;
            first = second;
            second = next;
        }
        if (previousHolder != null) {
            previousHolder[0] = first;
        }
        return second;
    }

    static double oneFOneApplyBackwardOnB(double a,
                                          double b,
                                          double z,
                                          int steps,
                                          double first,
                                          double second,
                                          long[] logScalingHolder,
                                          double[] previousHolder) {
        for (int k = 0; k < steps; k++) {
            Double3 coefficient = oneFOneCoefficientOnB(a, b, z, -k);
            if ((logScalingHolder != null && second != 0.0
                && shouldRescaleRecurrenceTerm(second, coefficient._2() / coefficient._1()))
                || (logScalingHolder != null
                && shouldRescaleRecurrenceTerm(first, coefficient._3() / coefficient._1()))) {
                long logScale = (long) Math.log(Math.abs(second));
                if (logScale != 0L) {
                    double scale = Math.exp(-logScale);
                    second *= scale;
                    first *= scale;
                    logScalingHolder[0] += logScale;
                }
            }
            double next = (coefficient._2() / -coefficient._1()) * second
                + (coefficient._3() / -coefficient._1()) * first;
            first = second;
            second = next;
        }
        if (previousHolder != null) {
            previousHolder[0] = first;
        }
        return second;
    }

    static double oneFOneApplyBackwardOnAAndB(double a,
                                              double b,
                                              double z,
                                              int offset,
                                              int steps,
                                              double first,
                                              double second,
                                              long[] logScalingHolder,
                                              double[] previousHolder) {
        for (int k = 0; k < steps; k++) {
            Double3 coefficient = oneFOneCoefficientOnAAndB(a, b, z, offset, -k);
            if ((logScalingHolder != null && second != 0.0
                && shouldRescaleRecurrenceTerm(second, coefficient._2() / coefficient._1()))
                || (logScalingHolder != null
                && shouldRescaleRecurrenceTerm(first, coefficient._3() / coefficient._1()))) {
                long logScale = (long) Math.log(Math.abs(second));
                if (logScale != 0L) {
                    double scale = Math.exp(-logScale);
                    second *= scale;
                    first *= scale;
                    logScalingHolder[0] += logScale;
                }
            }
            double next = (coefficient._2() / -coefficient._1()) * second
                + (coefficient._3() / -coefficient._1()) * first;
            first = second;
            second = next;
        }
        if (previousHolder != null) {
            previousHolder[0] = first;
        }
        return second;
    }

    static double oneFOneApplyBackwardOnSmallB(double a,
                                               double b,
                                               double z,
                                               int steps,
                                               double first,
                                               double second,
                                               long[] logScalingHolder,
                                               double[] previousHolder) {
        for (int k = 0; k < steps; k++) {
            Double3 coefficient = oneFOneCoefficientOnSmallB(a, b, z, steps, -k);
            if ((logScalingHolder != null && second != 0.0
                && shouldRescaleRecurrenceTerm(second, coefficient._2() / coefficient._1()))
                || (logScalingHolder != null
                && shouldRescaleRecurrenceTerm(first, coefficient._3() / coefficient._1()))) {
                long logScale = (long) Math.log(Math.abs(second));
                if (logScale != 0L) {
                    double scale = Math.exp(-logScale);
                    second *= scale;
                    first *= scale;
                    logScalingHolder[0] += logScale;
                }
            }
            double next = (coefficient._2() / -coefficient._1()) * second
                + (coefficient._3() / -coefficient._1()) * first;
            first = second;
            second = next;
        }
        if (previousHolder != null) {
            previousHolder[0] = first;
        }
        return second;
    }

    private static Double2 backwardRatioTerm(Double3 coefficient) {
        double bn = coefficient._1() / coefficient._3();
        double an = coefficient._2() / coefficient._3();
        return Double2.fraction(-bn, an);
    }

    private static Double2 forwardRatioTerm(Double3 coefficient) {
        double bn = coefficient._3() / coefficient._1();
        double an = coefficient._2() / coefficient._1();
        return Double2.fraction(-bn, an);
    }

    private static Double3 besselJyCoefficient(double order, double x, int index) {
        return new Double3(1.0, -2.0 * (order + index) / x, 1.0);
    }

    private static Double3 besselIkCoefficient(double order, double x, int index) {
        return new Double3(1.0, -2.0 * (order + index) / x, -1.0);
    }

    private static Double3 oneFOneCoefficientOnA(double a, double b, double z, int index) {
        double shiftedA = a + index;
        return new Double3(b - shiftedA, 2.0 * shiftedA - b + z, -shiftedA);
    }

    private static Double3 oneFOneCoefficientOnB(double a, double b, double z, int index) {
        return oneFOneCoefficientOnSmallB(a, b, z, 0, index);
    }

    private static Double3 oneFOneCoefficientOnSmallB(double a, double b, double z, int steps, int index) {
        double shiftedB = b + index + steps;
        double previousB = b + index + steps - 1.0;
        return new Double3(shiftedB * previousB, shiftedB * (-previousB - z), z * (shiftedB - a));
    }

    private static Double3 oneFOneCoefficientOnAAndB(double a, double b, double z, int offset, int index) {
        double shiftedA = a + offset + index;
        double shiftedB = b + offset + index;
        return new Double3(shiftedB * (b + offset + index - 1.0), shiftedB * (z - (b + offset + index - 1.0)), -shiftedA * z);
    }

    private static boolean shouldRescaleRecurrenceTerm(double value, double multiplier) {
        double scaledMagnitude = Math.abs(value * multiplier);
        if (!Double.isFinite(scaledMagnitude)) {
            return true;
        }
        if (scaledMagnitude == 0.0) {
            return false;
        }
        return scaledMagnitude > Double.MAX_VALUE / RECURRENCE_RESCALE_MARGIN
            || scaledMagnitude < Double.MIN_NORMAL * RECURRENCE_RESCALE_MARGIN;
    }
}