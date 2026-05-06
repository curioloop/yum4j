package com.curioloop.yum4j.math;

import java.util.function.DoubleUnaryOperator;

/**
 * Internal double-precision Java port of Boost's TOMS748 core solver.
 *
 * <p>This class keeps only the sign-changing-bracket refinement algorithm and its
 * tolerance policies. Bracket expansion and domain-specific adaptation live in
 * {@link RootIterators}.
 */
final class Toms748 {

    private static final double EPSILON = Math.ulp(1.0);
    private static final double MU = 0.5;

    enum Tolerance {
        EQUAL_FLOOR,
        EQUAL_CEIL,
        EQUAL_NEAREST_INTEGER
    }

    private Toms748() {
    }

    static Double2 solve(DoubleUnaryOperator function,
                         double lower,
                         double upper,
                         double fLower,
                         double fUpper,
                         double epsilon,
                         int maxIterations,
                         String functionName) {
        return solveInternal(function, lower, upper, fLower, fUpper, null, epsilon, maxIterations, functionName);
    }

    static Double2 solve(DoubleUnaryOperator function,
                         double lower,
                         double upper,
                         double fLower,
                         double fUpper,
                         Tolerance tolerance,
                         int maxIterations,
                         String functionName) {
        return solveInternal(function, lower, upper, fLower, fUpper, tolerance, Double.NaN, maxIterations, functionName);
    }

    private static boolean hasConverged(Tolerance tolerance,
                                        double a,
                                        double b,
                                        double epsilon) {
        if (tolerance == null) {
            return Math.abs(a - b) <= epsilon * Math.min(Math.abs(a), Math.abs(b));
        }
        switch (tolerance) {
            case EQUAL_FLOOR:
                return (Math.floor(a) == Math.floor(b)) || (Math.abs((b - a) / b) < 2.0 * EPSILON);
            case EQUAL_CEIL:
                return (Math.ceil(a) == Math.ceil(b)) || (Math.abs((b - a) / b) < 2.0 * EPSILON);
            case EQUAL_NEAREST_INTEGER:
                return (Math.floor(a + 0.5) == Math.floor(b + 0.5)) || (Math.abs((b - a) / b) < 2.0 * EPSILON);
            default:
                return Math.abs(a - b) <= epsilon * Math.min(Math.abs(a), Math.abs(b));
        }
    }

    private static void validateEpsilon(double epsilon,
                                        String functionName) {
        if (!Double.isFinite(epsilon) || epsilon <= 0.0) {
            throw new IllegalArgumentException(functionName + " requires finite positive epsilon: " + epsilon);
        }
    }

    private static void validateDiscreteTolerance(Tolerance tolerance,
                                                  String functionName) {
        if (tolerance == null) {
            throw new IllegalArgumentException(functionName + " requires tolerance");
        }
    }

    private static Double2 solveInternal(DoubleUnaryOperator function,
                                         double lower,
                                         double upper,
                                         double fLower,
                                         double fUpper,
                                         Tolerance tolerance,
                                         double epsilon,
                                         int maxIterations,
                                         String functionName) {
        String name = normalizeFunctionName("toms748", functionName);
        int requestedIterations = Math.max(maxIterations, 0);
        if (requestedIterations == 0) {
            return Double2.bound(lower, upper);
        }
        if (tolerance == null) {
            validateEpsilon(epsilon, name);
        } else {
            validateDiscreteTolerance(tolerance, name);
        }

        double a = lower;
        double b = upper;
        double fa = fLower;
        double fb = fUpper;

        if (!(a < b)) {
            throw new IllegalArgumentException(name + " requires lower < upper");
        }
        if (hasConverged(tolerance, a, b, epsilon) || fa == 0.0 || fb == 0.0) {
            if (fa == 0.0) {
                b = a;
            } else if (fb == 0.0) {
                a = b;
            }
            return Double2.bound(a, b);
        }
        if (Math.signum(fa) * Math.signum(fb) > 0.0) {
            throw new IllegalArgumentException(name + " requires a sign-changing bracket");
        }

        int count = requestedIterations;
        SolveState state = new SolveState(a, b, fa, fb);
        state.e = 1.0e5;
        state.fe = 1.0e5;

        if (state.fa != 0.0) {
            double candidate = secantInterpolate(state.a, state.b, state.fa, state.fb);
            bracket(function, state, candidate);
            --count;
            if ((count > 0) && (state.fa != 0.0) && !hasConverged(tolerance, state.a, state.b, epsilon)) {
                candidate = quadraticInterpolate(state.a, state.b, state.d, state.fa, state.fb, state.fd, 2);
                state.e = state.d;
                state.fe = state.fd;
                bracket(function, state, candidate);
                --count;
            }
        }

        while ((count > 0) && (state.fa != 0.0) && !hasConverged(tolerance, state.a, state.b, epsilon)) {
            double a0 = state.a;
            double b0 = state.b;
            double minDiff = Double.MIN_NORMAL * 32.0;
            boolean useQuadratic = (Math.abs(state.fa - state.fb) < minDiff)
                || (Math.abs(state.fa - state.fd) < minDiff)
                || (Math.abs(state.fa - state.fe) < minDiff)
                || (Math.abs(state.fb - state.fd) < minDiff)
                || (Math.abs(state.fb - state.fe) < minDiff)
                || (Math.abs(state.fd - state.fe) < minDiff);

            double candidate = useQuadratic
                ? quadraticInterpolate(state.a, state.b, state.d, state.fa, state.fb, state.fd, 2)
                : cubicInterpolate(state.a, state.b, state.d, state.e, state.fa, state.fb, state.fd, state.fe);
            state.e = state.d;
            state.fe = state.fd;
            bracket(function, state, candidate);
            if ((--count == 0) || (state.fa == 0.0) || hasConverged(tolerance, state.a, state.b, epsilon)) {
                break;
            }

            useQuadratic = (Math.abs(state.fa - state.fb) < minDiff)
                || (Math.abs(state.fa - state.fd) < minDiff)
                || (Math.abs(state.fa - state.fe) < minDiff)
                || (Math.abs(state.fb - state.fd) < minDiff)
                || (Math.abs(state.fb - state.fe) < minDiff)
                || (Math.abs(state.fd - state.fe) < minDiff);
            candidate = useQuadratic
                ? quadraticInterpolate(state.a, state.b, state.d, state.fa, state.fb, state.fd, 3)
                : cubicInterpolate(state.a, state.b, state.d, state.e, state.fa, state.fb, state.fd, state.fe);
            bracket(function, state, candidate);
            if ((--count == 0) || (state.fa == 0.0) || hasConverged(tolerance, state.a, state.b, epsilon)) {
                break;
            }

            double u = Math.abs(state.fa) < Math.abs(state.fb) ? state.a : state.b;
            double fu = Math.abs(state.fa) < Math.abs(state.fb) ? state.fa : state.fb;
            candidate = u - 2.0 * (fu / (state.fb - state.fa)) * (state.b - state.a);
            if (Math.abs(candidate - u) > (state.b - state.a) / 2.0) {
                candidate = state.a + (state.b - state.a) / 2.0;
            }

            state.e = state.d;
            state.fe = state.fd;
            bracket(function, state, candidate);
            if ((--count == 0) || (state.fa == 0.0) || hasConverged(tolerance, state.a, state.b, epsilon)) {
                break;
            }

            if ((state.b - state.a) < MU * (b0 - a0)) {
                continue;
            }

            state.e = state.d;
            state.fe = state.fd;
            bracket(function, state, state.a + (state.b - state.a) / 2.0);
            --count;
        }

        if (state.fa == 0.0) {
            state.b = state.a;
        } else if (state.fb == 0.0) {
            state.a = state.b;
        }
        return Double2.bound(state.a, state.b);
    }

    private static void bracket(DoubleUnaryOperator function, SolveState state, double candidate) {
        double tolerance = EPSILON * 2.0;
        if ((state.b - state.a) < 2.0 * tolerance * state.a) {
            candidate = state.a + (state.b - state.a) / 2.0;
        } else if (candidate <= state.a + Math.abs(state.a) * tolerance) {
            candidate = state.a + Math.abs(state.a) * tolerance;
        } else if (candidate >= state.b - Math.abs(state.b) * tolerance) {
            candidate = state.b - Math.abs(state.b) * tolerance;
        }

        double fCandidate = function.applyAsDouble(candidate);
        if (fCandidate == 0.0) {
            state.a = candidate;
            state.fa = 0.0;
            state.d = 0.0;
            state.fd = 0.0;
            return;
        }

        if (Math.signum(state.fa) * Math.signum(fCandidate) < 0.0) {
            state.d = state.b;
            state.fd = state.fb;
            state.b = candidate;
            state.fb = fCandidate;
        } else {
            state.d = state.a;
            state.fd = state.fa;
            state.a = candidate;
            state.fa = fCandidate;
        }
    }

    private static double safeDivide(double numerator, double denominator, double fallback) {
        if (Math.abs(denominator) < 1.0 && Math.abs(denominator * Double.MAX_VALUE) <= Math.abs(numerator)) {
            return fallback;
        }
        return numerator / denominator;
    }

    private static double secantInterpolate(double lower, double upper, double fLower, double fUpper) {
        double candidate = lower - (fLower / (fUpper - fLower)) * (upper - lower);
        double tolerance = EPSILON * 5.0;
        if ((candidate <= lower + Math.abs(lower) * tolerance) || (candidate >= upper - Math.abs(upper) * tolerance)) {
            return (lower + upper) / 2.0;
        }
        return candidate;
    }

    private static double quadraticInterpolate(double a,
                                               double b,
                                               double d,
                                               double fa,
                                               double fb,
                                               double fd,
                                               int count) {
        double linear = safeDivide(fb - fa, b - a, Double.MAX_VALUE);
        double delta = safeDivide(fd - fb, d - b, Double.MAX_VALUE);
        double quadratic = safeDivide(delta - linear, d - a, 0.0);
        if (quadratic == 0.0) {
            return secantInterpolate(a, b, fa, fb);
        }

        double candidate = (Math.signum(quadratic) * Math.signum(fa) > 0.0) ? a : b;
        for (int iteration = 1; iteration <= count; iteration++) {
            double numerator = fa + (linear + quadratic * (candidate - b)) * (candidate - a);
            double denominator = linear + quadratic * (2.0 * candidate - a - b);
            candidate -= safeDivide(numerator, denominator, 1.0 + candidate - a);
        }
        if ((candidate <= a) || (candidate >= b)) {
            return secantInterpolate(a, b, fa, fb);
        }
        return candidate;
    }

    private static double cubicInterpolate(double a,
                                           double b,
                                           double d,
                                           double e,
                                           double fa,
                                           double fb,
                                           double fd,
                                           double fe) {
        double q11 = (d - e) * fd / (fe - fd);
        double q21 = (b - d) * fb / (fd - fb);
        double q31 = (a - b) * fa / (fb - fa);
        double d21 = (b - d) * fd / (fd - fb);
        double d31 = (a - b) * fb / (fb - fa);
        double q22 = (d21 - q11) * fb / (fe - fb);
        double q32 = (d31 - q21) * fa / (fd - fa);
        double d32 = (d31 - q21) * fd / (fd - fa);
        double q33 = (d32 - q22) * fa / (fe - fa);
        double candidate = q31 + q32 + q33 + a;
        if ((candidate <= a) || (candidate >= b)) {
            return quadraticInterpolate(a, b, d, fa, fb, fd, 3);
        }
        return candidate;
    }

    private static String normalizeFunctionName(String defaultName, String functionName) {
        return (functionName == null || functionName.isEmpty()) ? defaultName : functionName;
    }

    private static final class SolveState {
        private double a;
        private double b;
        private double fa;
        private double fb;
        private double d;
        private double fd;
        private double e;
        private double fe;

        private SolveState(double a, double b, double fa, double fb) {
            this.a = a;
            this.b = b;
            this.fa = fa;
            this.fb = fb;
        }
    }
}