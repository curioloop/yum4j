package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.Commons.validateFinite;

/**
 * Real scalar Jacobi theta functions.
 *
 * <p>This tranche exposes the four classical theta functions in both nome-q and
 * imaginary-tau parameterizations, together with the Boost-style minus-one
 * variants for theta3 and theta4 that preserve accuracy when q is small.
 *
 * <pre>
 * theta1(z, q) = 2 sum((-1)^n q^((n + 1/2)^2) sin((2n + 1) z), n = 0..infinity)
 * theta2(z, q) = 2 sum(q^((n + 1/2)^2) cos((2n + 1) z), n = 0..infinity)
 * theta3(z, q) = 1 + 2 sum(q^(n^2) cos(2 n z), n = 1..infinity)
 * theta4(z, q) = 1 + 2 sum((-1)^n q^(n^2) cos(2 n z), n = 1..infinity)
 *
 * q = exp(-pi tau), tau > 0
 * </pre>
 *
 * <p>The implementation is real-valued only. Source provenance follows Boost.Math's
 * {@code jacobi_theta.hpp}: tau-parameterized evaluation uses the direct Fourier
 * series when tau >= 1 and a modular-transform rewrite when tau < 1, and the
 * public minus-one variants stay split out explicitly so downstream callers can
 * reuse the same small-q accuracy boundary as Boost's real API.
 */
public final class JacobiTheta {

    private static final double PI = Math.PI;
    private static final double HALF_PI = 0.5 * PI;
    private static final double TWO_PI = 2.0 * PI;
    private static final double SERIES_EPSILON = Math.ulp(1.0);

    private JacobiTheta() {
    }

    public static double theta1(double z, double q) {
        validateFinite(z, "theta1: z");
        return theta1Tau(z, tauFromQ(q, "theta1"));
    }

    public static double theta2(double z, double q) {
        validateFinite(z, "theta2: z");
        return theta2Tau(z, tauFromQ(q, "theta2"));
    }

    public static double theta3(double z, double q) {
        validateFinite(z, "theta3: z");
        return theta3Tau(z, tauFromQ(q, "theta3"));
    }

    public static double theta4(double z, double q) {
        validateFinite(z, "theta4: z");
        return theta4Tau(z, tauFromQ(q, "theta4"));
    }

    public static double theta1Tau(double z, double tau) {
        validateFinite(z, "theta1Tau: z");
        validatePositiveTau(tau, "theta1Tau");

        if (z == 0.0) {
            return 0.0;
        }
        if (tau < 1.0) {
            return imaginaryTheta1(reduceAngle(z, TWO_PI, PI), 1.0 / tau);
        }

        double result = 0.0;
        double previous = 0.0;
        for (int n = 0; ; n++) {
            double shifted = n + 0.5;
            double termScale = Math.exp(-tau * PI * shifted * shifted);
            double term = termScale * Math.sin((2.0 * n + 1.0) * z);
            if ((n & 1) == 1) {
                term = -term;
            }
            result += 2.0 * term;
            if (converged(previous, termScale)) {
                return result;
            }
            previous = termScale;
        }
    }

    public static double theta2Tau(double z, double tau) {
        validateFinite(z, "theta2Tau: z");
        validatePositiveTau(tau, "theta2Tau");

        if (tau < 1.0 && z == 0.0) {
            return theta4Tau(0.0, 1.0 / tau) / Math.sqrt(tau);
        }
        if (tau < 1.0) {
            return imaginaryTheta2(reduceAngle(z, TWO_PI, PI), 1.0 / tau);
        }

        double result = 0.0;
        double previous = 0.0;
        for (int n = 0; ; n++) {
            double shifted = n + 0.5;
            double termScale = Math.exp(-tau * PI * shifted * shifted);
            result += 2.0 * termScale * Math.cos((2.0 * n + 1.0) * z);
            if (converged(previous, termScale)) {
                return result;
            }
            previous = termScale;
        }
    }

    public static double theta3(double z, double q, boolean minusOne) {
        return minusOne ? theta3MinusOne(z, q) : theta3(z, q);
    }

    public static double theta3Tau(double z, double tau) {
        validateFinite(z, "theta3Tau: z");
        validatePositiveTau(tau, "theta3Tau");

        if (tau < 1.0 && z == 0.0) {
            return theta3Tau(0.0, 1.0 / tau) / Math.sqrt(tau);
        }
        if (tau < 1.0) {
            return imaginaryTheta3(reduceAngle(z, PI, HALF_PI), 1.0 / tau);
        }
        return 1.0 + theta3MinusOneTau(z, tau);
    }

    public static double theta4Tau(double z, double tau) {
        validateFinite(z, "theta4Tau: z");
        validatePositiveTau(tau, "theta4Tau");

        if (tau < 1.0 && z == 0.0) {
            return theta2Tau(0.0, 1.0 / tau) / Math.sqrt(tau);
        }
        if (tau < 1.0) {
            return imaginaryTheta4(reduceAngle(z, PI, HALF_PI), 1.0 / tau);
        }
        return 1.0 + theta4MinusOneTau(z, tau);
    }

    public static double theta3MinusOne(double z, double q) {
        validateFinite(z, "theta3MinusOne: z");
        return theta3MinusOneTau(z, tauFromQ(q, "theta3MinusOne"));
    }

    public static double theta4MinusOne(double z, double q) {
        validateFinite(z, "theta4MinusOne: z");
        return theta4MinusOneTau(z, tauFromQ(q, "theta4MinusOne"));
    }

    public static double theta3MinusOneTau(double z, double tau) {
        validateFinite(z, "theta3MinusOneTau: z");
        validatePositiveTau(tau, "theta3MinusOneTau");

        if (tau < 1.0) {
            return theta3Tau(z, tau) - 1.0;
        }

        double result = 0.0;
        double previous = 0.0;
        for (int n = 1; ; n++) {
            double termScale = Math.exp(-tau * PI * n * n);
            result += 2.0 * termScale * Math.cos(2.0 * n * z);
            if (converged(previous, termScale)) {
                return result;
            }
            previous = termScale;
        }
    }

    public static double theta4MinusOneTau(double z, double tau) {
        validateFinite(z, "theta4MinusOneTau: z");
        validatePositiveTau(tau, "theta4MinusOneTau");

        if (tau < 1.0) {
            return theta4Tau(z, tau) - 1.0;
        }

        double result = 0.0;
        double previous = 0.0;
        for (int n = 1; ; n++) {
            double termScale = Math.exp(-tau * PI * n * n);
            double term = termScale * Math.cos(2.0 * n * z);
            if ((n & 1) == 1) {
                term = -term;
            }
            result += 2.0 * term;
            if (converged(previous, termScale)) {
                return result;
            }
            previous = termScale;
        }
    }

    private static double imaginaryTheta1(double z, double tau) {
        double result = 0.0;
        result -= gaussianSum(tau, z + HALF_PI, TWO_PI);
        result += gaussianSum(tau, z + HALF_PI + PI, TWO_PI);
        result += gaussianSum(tau, z - HALF_PI, -TWO_PI);
        result -= gaussianSum(tau, z - HALF_PI - PI, -TWO_PI);
        return result * Math.sqrt(tau);
    }

    private static double imaginaryTheta2(double z, double tau) {
        double result = 0.0;
        result += Math.exp(-z * z * tau / PI);
        result -= gaussianSum(tau, z + PI, TWO_PI);
        result -= gaussianSum(tau, z - PI, -TWO_PI);
        result += gaussianSum(tau, z + TWO_PI, TWO_PI);
        result += gaussianSum(tau, z - TWO_PI, -TWO_PI);
        return result * Math.sqrt(tau);
    }

    private static double imaginaryTheta3(double z, double tau) {
        double result = Math.exp(-z * z * tau / PI);
        result += gaussianSum(tau, z + PI, PI);
        result += gaussianSum(tau, z - PI, -PI);
        return result * Math.sqrt(tau);
    }

    private static double imaginaryTheta4(double z, double tau) {
        double result = 0.0;
        result += gaussianSum(tau, z + HALF_PI, PI);
        result += gaussianSum(tau, z - HALF_PI, -PI);
        return result * Math.sqrt(tau);
    }

    private static double gaussianSum(double tau, double start, double increment) {
        double result = 0.0;
        double previous = 0.0;
        double position = start;
        while (true) {
            double term = Math.exp(-tau * position * position / PI);
            result += term;
            if (converged(previous, term)) {
                return result;
            }
            previous = term;
            position += increment;
        }
    }

    private static boolean converged(double previous, double current) {
        return current == 0.0 || (previous > 0.0 && current < SERIES_EPSILON * previous);
    }

    private static double reduceAngle(double z, double period, double upperBound) {
        double reduced = Math.IEEEremainder(z, period);
        if (reduced > upperBound) {
            reduced -= period;
        }
        if (reduced < -upperBound) {
            reduced += period;
        }
        return reduced;
    }

    private static double tauFromQ(double q, String functionName) {
        if (!Double.isFinite(q) || !(q > 0.0) || !(q < 1.0)) {
            throw new IllegalArgumentException(functionName + ": q must be finite with 0 < q < 1: " + q);
        }
        return -Math.log(q) / PI;
    }

    private static void validatePositiveTau(double tau, String functionName) {
        if (!Double.isFinite(tau) || !(tau > 0.0)) {
            throw new IllegalArgumentException(functionName + ": tau must be finite and positive: " + tau);
        }
    }

}