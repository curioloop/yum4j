package com.curioloop.yum4j.math;

import java.util.function.DoubleUnaryOperator;

/**
 * Boost-style bounded root iterators.
 *
 * <p>This utility exposes the bounded refinement and bracket-adaptation modes needed by the
 * special-function layer: Newton-Raphson and Halley refinement for already bracketed roots,
 * plus domain-aware bracket expansion around the internal {@link Toms748} core solver.
 */
public final class RootIterators {

    @FunctionalInterface
    public interface FirstOrderFunction {
        void evaluate(double x, double[] evaluation);
    }

    @FunctionalInterface
    public interface SecondOrderFunction {
        void evaluate(double x, double[] evaluation);
    }

    private RootIterators() {
    }

    public static double newtonRaphsonIterate(FirstOrderFunction function,
                                              double guess,
                                              double lower,
                                              double upper,
                                              int bits,
                                              int maxIterations,
                                              String functionName) {
        return newtonRaphsonIterateInternal(function, guess, lower, upper, bits, maxIterations, functionName);
    }

    private static double newtonRaphsonIterateInternal(FirstOrderFunction function,
                                                       double guess,
                                                       double lower,
                                                       double upper,
                                                       int bits,
                                                       int maxIterations,
                                                       String functionName) {
        validateArguments(lower, upper, bits, maxIterations, functionName);

        double result = clampGuess(guess, lower, upper);
        return newtonRaphsonIterateLoop(
            function,
            result,
            lower,
            upper,
            toleranceFactor(bits),
            initialDelta(result),
            maxIterations,
            functionName
        );
    }

    private static double newtonRaphsonIterateLoop(FirstOrderFunction function,
                                                   double result,
                                                   double lower,
                                                   double upper,
                                                   double factor,
                                                   double delta,
                                                   int maxIterations,
                                                   String functionName) {
        double delta1 = delta;
        double delta2 = delta;
        double lastF0 = 0.0;
        double maxRangeF = 0.0;
        double minRangeF = 0.0;
        int count = maxIterations;
        double[] evaluation = new double[2];

        do {
            delta2 = delta1;
            delta1 = delta;

            function.evaluate(result, evaluation);
            double f0 = evaluation[0];
            if (f0 == 0.0) {
                break;
            }

            delta = resolveNewtonDelta(function, evaluation, lastF0, f0, delta, result, lower, upper);

            delta = finiteDelta(delta, result);

            lastF0 = f0;

            if (shouldBackoff(delta, delta2)) {
                delta = backoffDelta(delta, result, lower, upper);
                delta1 = delta * 3.0;
                delta2 = delta * 3.0;
            }

            double previousResult = result;
            double steppedResult = result - delta;
            result = keepStepInBracket(previousResult, steppedResult, lower, upper);
            if (result != steppedResult) {
                if (Double.isNaN(result)) {
                    result = midpoint(lower, upper);
                    break;
                }
                delta = previousResult - result;
                continue;
            }

            boolean positiveDelta = delta > 0.0;
            upper = positiveDelta ? previousResult : upper;
            lower = positiveDelta ? lower : previousResult;
            maxRangeF = positiveDelta ? f0 : maxRangeF;
            minRangeF = positiveDelta ? minRangeF : f0;
            ensureSignChangingBracket(maxRangeF, minRangeF, functionName);
        } while (hasIterationsRemaining(--count, result, factor, delta));

        ensureConverged(count, result, factor, delta, functionName);
        return result;
    }

    public static double halleyIterate(SecondOrderFunction function,
                                       double guess,
                                       double lower,
                                       double upper,
                                       int bits,
                                       int maxIterations,
                                       String functionName) {
        validateArguments(lower, upper, bits, maxIterations, functionName);

        double result = clampGuess(guess, lower, upper);
        return halleyIterateLoop(
            function,
            result,
            lower,
            upper,
            toleranceFactor(bits),
            initialDelta(result),
            maxIterations,
            functionName
        );
    }

    private static double halleyIterateLoop(SecondOrderFunction function,
                                            double result,
                                            double lower,
                                            double upper,
                                            double factor,
                                            double delta,
                                            int maxIterations,
                                            String functionName) {
        double delta1 = delta;
        double delta2 = delta;
        double lastF0 = 0.0;
        double maxRangeF = 0.0;
        double minRangeF = 0.0;
        int count = maxIterations;
        double[] evaluation = new double[3];

        do {
            delta2 = delta1;
            delta1 = delta;

            function.evaluate(result, evaluation);
            double f0 = evaluation[0];
            if (f0 == 0.0) {
                break;
            }

            delta = resolveHalleyDelta(function, evaluation, lastF0, f0, delta, result, lower, upper);

            delta = finiteDelta(delta, result);

            lastF0 = f0;

            if (shouldBackoff(delta, delta2)) {
                delta = backoffDelta(delta, result, lower, upper);
                delta1 = delta * 3.0;
                delta2 = delta * 3.0;
            }

            double previousResult = result;
            double steppedResult = result - delta;
            result = keepStepInBracket(previousResult, steppedResult, lower, upper);
            if (result != steppedResult) {
                if (Double.isNaN(result)) {
                    result = midpoint(lower, upper);
                    break;
                }
                delta = previousResult - result;
                continue;
            }

            boolean positiveDelta = delta > 0.0;
            upper = positiveDelta ? previousResult : upper;
            lower = positiveDelta ? lower : previousResult;
            maxRangeF = positiveDelta ? f0 : maxRangeF;
            minRangeF = positiveDelta ? minRangeF : f0;
            ensureSignChangingBracket(maxRangeF, minRangeF, functionName);
        } while (hasIterationsRemaining(--count, result, factor, delta));

        ensureConverged(count, result, factor, delta, functionName);
        return result;
    }

    static double toms748Solve(DoubleUnaryOperator function,
                               double guess,
                               double factor,
                               boolean rising,
                               int bits,
                               int maxIterations,
                               String functionName) {
        double tolerance = Math.max(Math.scalb(1.0, 1 - bits), 4.0 * Math.ulp(1.0));
        return midpoint(toms748SolveInternal(
            function,
            guess,
            factor,
            rising,
            tolerance,
            Math.max(maxIterations, 0),
            functionName
        ));
    }

    public static double toms748SolveBounded(DoubleUnaryOperator function,
                                             double guess,
                                             double factor,
                                             boolean rising,
                                             double lowerBound,
                                             int bits,
                                             int maxIterations,
                                             String functionName) {
        double tolerance = Math.max(Math.scalb(1.0, 1 - bits), 4.0 * Math.ulp(1.0));
        return midpoint(toms748SolveBoundedInternal(
            function,
            guess,
            factor,
            rising,
            lowerBound,
            tolerance,
            Math.max(maxIterations, 0),
            functionName
        ));
    }

    public static double toms748Solve01(DoubleUnaryOperator function,
                                        double guess,
                                        double factor,
                                        boolean rising,
                                        int bits,
                                        int maxIterations,
                                        String functionName) {
        double tolerance = Math.max(Math.scalb(1.0, 1 - bits), 4.0 * Math.ulp(1.0));
        return midpoint(toms748Solve01Internal(
            function,
            guess,
            factor,
            rising,
            tolerance,
            Math.max(maxIterations, 0),
            functionName
        ));
    }

    private static double midpoint(Double2 interval) {
        return midpoint(interval.lower(), interval.upper());
    }

    private static double midpoint(double lower, double upper) {
        return 0.5 * (lower + upper);
    }

    private static double clampGuess(double guess, double lower, double upper) {
        return Math.max(Math.min(guess, upper), lower);
    }

    private static double toleranceFactor(int bits) {
        return Math.scalb(1.0, 1 - bits);
    }

    private static double initialDelta(double result) {
        return Math.max(1.0e7 * Math.abs(result), 1.0e7);
    }

    private static boolean hasCollapsedBracket(double lower, double upper) {
        return Math.nextUp(lower) >= upper;
    }

    private static boolean hasIterationsRemaining(int count, double result, double factor, double delta) {
        return (count > 0) && (Math.abs(result * factor) < Math.abs(delta));
    }

    private static void ensureConverged(int count,
                                        double result,
                                        double factor,
                                        double delta,
                                        String functionName) {
        if ((count <= 0) && (Math.abs(result * factor) < Math.abs(delta))) {
            throw new ArithmeticException(functionName + " did not converge within the configured iterations");
        }
    }

    private static void validateArguments(double lower,
                                          double upper,
                                          int bits,
                                          int maxIterations,
                                          String functionName) {
        if (!(lower < upper)) {
            throw new IllegalArgumentException(functionName + " requires lower < upper");
        }
        if (bits <= 0) {
            throw new IllegalArgumentException(functionName + " requires positive bit precision: " + bits);
        }
        if (maxIterations <= 0) {
            throw new IllegalArgumentException(functionName + " requires positive iteration count: " + maxIterations);
        }
    }

    private static double halleyStep(double x, double f0, double f1, double f2) {
        if (f2 == 0.0) {
            return f0 / f1;
        }
        double denominator = 2.0 * f1 - f0 * (f2 / f1);
        if ((Math.abs(denominator) < 1.0) && (Math.abs(2.0 * f0) >= Math.abs(denominator) * Double.MAX_VALUE)) {
            return f0 / f1;
        }
        double delta = (2.0 * f0) / denominator;
        if (((delta * f1) / f0) < 0.0) {
            delta = f0 / f1;
            if ((x != 0.0) && (Math.abs(delta) > 2.0 * Math.abs(x))) {
                delta = Math.copySign(2.0 * Math.abs(x), delta);
            }
        }
        return delta;
    }

    private static double finiteDelta(double delta, double result) {
        if (Double.isFinite(delta)) {
            return delta;
        }
        return Math.copySign(Math.abs(result == 0.0 ? 1.0 : result), delta >= 0.0 ? 1.0 : -1.0);
    }

    private static double resolveNewtonDelta(FirstOrderFunction function,
                                             double[] evaluation,
                                             double lastF0,
                                             double f0,
                                             double delta,
                                             double result,
                                             double lower,
                                             double upper) {
        double f1 = evaluation[1];
        return f1 == 0.0
            ? handleZeroDerivative(function, evaluation, lastF0, f0, delta, result, lower, upper)
            : f0 / f1;
    }

    private static double resolveHalleyDelta(SecondOrderFunction function,
                                             double[] evaluation,
                                             double lastF0,
                                             double f0,
                                             double delta,
                                             double result,
                                             double lower,
                                             double upper) {
        double f1 = evaluation[1];
        return f1 == 0.0
            ? handleZeroDerivative(function, evaluation, lastF0, f0, delta, result, lower, upper)
            : halleyStep(result, f0, f1, evaluation[2]);
    }

    private static boolean shouldBackoff(double delta, double delta2) {
        double convergence = (Math.abs(delta2) > 1.0
            || (Math.abs(Double.MAX_VALUE * delta2) > Math.abs(delta)))
            ? Math.abs(delta / delta2)
            : Double.MAX_VALUE;
        return (convergence > 0.8) && (convergence < 2.0);
    }

    private static double backoffDelta(double delta, double result, double lower, double upper) {
        double adjusted = delta > 0.0 ? (result - lower) / 2.0 : (result - upper) / 2.0;
        if ((result != 0.0) && (Math.abs(adjusted) > Math.abs(result))) {
            adjusted = Math.copySign(Math.abs(result) * 0.9, adjusted);
        }
        return adjusted;
    }

    private static double reboundInsideBracket(double previousResult,
                                               double lower,
                                               double upper,
                                               boolean belowLower) {
        double boundary = belowLower ? lower : upper;
        double rebound = previousResult - ((previousResult - boundary) / 2.0);
        if (belowLower) {
            if (rebound <= lower) {
                rebound = Math.nextUp(lower);
            }
            if (rebound >= upper) {
                rebound = Math.nextDown(upper);
            }
            return rebound;
        }
        if (rebound >= upper) {
            rebound = Math.nextDown(upper);
        }
        if (rebound <= lower) {
            rebound = Math.nextUp(lower);
        }
        return rebound;
    }

    private static double keepStepInBracket(double previousResult,
                                            double steppedResult,
                                            double lower,
                                            double upper) {
        if (steppedResult < lower) {
            return hasCollapsedBracket(lower, upper)
                ? Double.NaN
                : reboundInsideBracket(previousResult, lower, upper, true);
        }
        if (steppedResult > upper) {
            return hasCollapsedBracket(lower, upper)
                ? Double.NaN
                : reboundInsideBracket(previousResult, lower, upper, false);
        }
        return steppedResult;
    }

    private static void ensureSignChangingBracket(double maxRangeF,
                                                  double minRangeF,
                                                  String functionName) {
        if (maxRangeF * minRangeF > 0.0) {
            throw new ArithmeticException(functionName + " failed to preserve a sign-changing bracket");
        }
    }

    private static Double2 toms748SolveInternal(DoubleUnaryOperator function,
                                                double guess,
                                                double factor,
                                                boolean rising,
                                                double tolerance,
                                                int maxIterations,
                                                String functionName) {
        int requestedIterations = Math.max(maxIterations, 0);
        if (requestedIterations <= 0) {
            return Double2.bound(guess, guess);
        }

        String name = normalizeFunctionName("toms748Solve", functionName);
        double a = guess;
        double b = a;
        double fa = function.applyAsDouble(a);
        double fb = fa;
        int count = requestedIterations - 1;
        int step = 32;

        if ((fa < 0.0) == (guess < 0.0 ? !rising : rising)) {
            while (sameSign(fb, fa)) {
                if (count == 0) {
                    throw new ArithmeticException(name + " failed to bracket root");
                }
                if (((requestedIterations - count) % step) == 0) {
                    factor *= 2.0;
                    if (step > 1) {
                        step /= 2;
                    }
                }
                a = b;
                fa = fb;
                b *= factor;
                fb = function.applyAsDouble(b);
                --count;
            }
        } else {
            while (sameSign(fb, fa)) {
                if (Math.abs(a) < Double.MIN_NORMAL) {
                    return a > 0.0 ? Double2.bound(0.0, a) : Double2.bound(a, 0.0);
                }
                if (count == 0) {
                    throw new ArithmeticException(name + " failed to bracket root");
                }
                if (((requestedIterations - count) % step) == 0) {
                    factor *= 2.0;
                    if (step > 1) {
                        step /= 2;
                    }
                }
                b = a;
                fb = fa;
                a /= factor;
                fa = function.applyAsDouble(a);
                --count;
            }
        }

        return finishToms748Solve(
            function,
            a,
            b,
            fa,
            fb,
            tolerance,
            count,
            name
        );
    }

    private static Double2 toms748SolveBoundedInternal(DoubleUnaryOperator function,
                                                       double guess,
                                                       double factor,
                                                       boolean rising,
                                                       double lowerBound,
                                                       double tolerance,
                                                       int maxIterations,
                                                       String functionName) {
        double lower = Math.max(guess, lowerBound);
        int requestedIterations = Math.max(maxIterations, 0);
        if (requestedIterations <= 0) {
            return Double2.bound(lower, lower);
        }

        String name = normalizeFunctionName("toms748SolveBounded", functionName);
        double upper = lower;
        double fLower = function.applyAsDouble(lower);
        if (fLower == 0.0) {
            return Double2.bound(lower, lower);
        }

        double fUpper = fLower;
        int count = requestedIterations - 1;
        int step = 32;

        if ((fLower < 0.0) == rising) {
            while (sameSign(fLower, fUpper)) {
                if (count == 0) {
                    throw new ArithmeticException(name + " failed to bracket root");
                }
                if (((requestedIterations - count) % step) == 0) {
                    factor *= 2.0;
                    if (step > 1) {
                        step /= 2;
                    }
                }
                lower = upper;
                fLower = fUpper;
                upper *= factor;
                if (!Double.isFinite(upper) || upper >= Double.MAX_VALUE / 4.0) {
                    throw new ArithmeticException(name + " overflow");
                }
                fUpper = function.applyAsDouble(upper);
                --count;
            }
        } else {
            while (sameSign(fLower, fUpper)) {
                if (count == 0) {
                    throw new ArithmeticException(name + " failed to bracket root");
                }
                if (((requestedIterations - count) % step) == 0) {
                    factor *= 2.0;
                    if (step > 1) {
                        step /= 2;
                    }
                }
                upper = lower;
                fUpper = fLower;
                lower = Math.max(lowerBound, lower / factor);
                fLower = function.applyAsDouble(lower);
                --count;
                if ((lower == lowerBound) && sameSign(fLower, fUpper)) {
                    throw new ArithmeticException(name + " failed to bracket root");
                }
            }
        }

        return finishToms748Solve(
            function,
            lower,
            upper,
            fLower,
            fUpper,
            tolerance,
            count,
            name
        );
    }

    private static Double2 toms748Solve01Internal(DoubleUnaryOperator function,
                                                  double guess,
                                                  double factor,
                                                  boolean rising,
                                                  double tolerance,
                                                  int maxIterations,
                                                  String functionName) {
        double lower = Math.min(Math.max(guess, 0.0), 1.0);
        int requestedIterations = Math.max(maxIterations, 0);
        if (requestedIterations <= 0) {
            return Double2.bound(lower, lower);
        }

        String name = normalizeFunctionName("toms748Solve01", functionName);
        double upper = lower;
        double fLower = function.applyAsDouble(lower);
        if (fLower == 0.0) {
            return Double2.bound(lower, lower);
        }

        double fUpper = fLower;
        int count = requestedIterations - 1;

        if ((fLower < 0.0) == rising) {
            while (sameSign(fLower, fUpper)) {
                if (count == 0) {
                    throw new ArithmeticException(name + " failed to bracket root on [0, 1]");
                }
                if (((requestedIterations - count) % 20) == 0) {
                    factor *= 2.0;
                }
                lower = upper;
                fLower = fUpper;
                upper = 1.0 - ((1.0 - upper) / factor);
                fUpper = function.applyAsDouble(upper);
                --count;
            }
        } else {
            while (sameSign(fLower, fUpper)) {
                if (Math.abs(lower) < Double.MIN_NORMAL) {
                    return Double2.bound(0.0, upper == 0.0 ? Double.MIN_NORMAL : upper);
                }
                if (count == 0) {
                    throw new ArithmeticException(name + " failed to bracket root on [0, 1]");
                }
                if (((requestedIterations - count) % 20) == 0) {
                    factor *= 2.0;
                }
                upper = lower;
                fUpper = fLower;
                lower /= factor;
                fLower = function.applyAsDouble(lower);
                --count;
            }
        }

        return finishToms748Solve(
            function,
            lower,
            upper,
            fLower,
            fUpper,
            tolerance,
            count,
            name
        );
    }

    private static Double2 finishToms748Solve(DoubleUnaryOperator function,
                                              double lower,
                                              double upper,
                                              double fLower,
                                              double fUpper,
                                              double tolerance,
                                              int remainingIterations,
                                              String functionName) {
        if (upper < lower) {
            double tmp = lower;
            lower = upper;
            upper = tmp;
            tmp = fLower;
            fLower = fUpper;
            fUpper = tmp;
        }

        return Toms748.solve(
            function,
            lower,
            upper,
            fLower,
            fUpper,
            tolerance,
            Math.max(remainingIterations, 0),
            functionName
        );
    }

    private static String normalizeFunctionName(String defaultName, String functionName) {
        return (functionName == null || functionName.isEmpty()) ? defaultName : functionName;
    }

    private static boolean sameSign(double a, double b) {
        return Math.signum(a) == Math.signum(b);
    }

    private static double handleZeroDerivative(FirstOrderFunction function,
                                               double[] evaluation,
                                               double lastF0,
                                               double f0,
                                               double delta,
                                               double result,
                                               double lower,
                                               double upper) {
        double previousF0 = lastF0;
        double adjustedDelta = delta;
        if (previousF0 == 0.0) {
            double guess = result == lower ? upper : lower;
            function.evaluate(guess, evaluation);
            previousF0 = evaluation[0];
            adjustedDelta = guess - result;
        }
        if ((Math.signum(previousF0) * Math.signum(f0)) < 0.0) {
            adjustedDelta = adjustedDelta < 0.0 ? (result - lower) / 2.0 : (result - upper) / 2.0;
        } else {
            adjustedDelta = adjustedDelta < 0.0 ? (result - upper) / 2.0 : (result - lower) / 2.0;
        }
        return adjustedDelta;
    }

    private static double handleZeroDerivative(SecondOrderFunction function,
                                               double[] evaluation,
                                               double lastF0,
                                               double f0,
                                               double delta,
                                               double result,
                                               double lower,
                                               double upper) {
        double previousF0 = lastF0;
        double adjustedDelta = delta;
        if (previousF0 == 0.0) {
            double guess = result == lower ? upper : lower;
            function.evaluate(guess, evaluation);
            previousF0 = evaluation[0];
            adjustedDelta = guess - result;
        }
        if ((Math.signum(previousF0) * Math.signum(f0)) < 0.0) {
            adjustedDelta = adjustedDelta < 0.0 ? (result - lower) / 2.0 : (result - upper) / 2.0;
        } else {
            adjustedDelta = adjustedDelta < 0.0 ? (result - upper) / 2.0 : (result - lower) / 2.0;
        }
        return adjustedDelta;
    }
}