package com.curioloop.yum4j.math;

import java.util.function.DoubleSupplier;
import java.util.function.DoubleUnaryOperator;

/**
 * Real elliptic integrals of the first, second, and third kinds.
 *
 * <p>This class exposes the complete and incomplete Legendre-form integrals:
 *
 * <pre>
 * K(k)      = integral(dtheta / sqrt(1 - k^2 sin^2(theta)), theta = 0..pi/2)
 * F(k, phi) = integral(dtheta / sqrt(1 - k^2 sin^2(theta)), theta = 0..phi)
 *
 * E(k)      = integral(sqrt(1 - k^2 sin^2(theta)) dtheta, theta = 0..pi/2)
 * E(k, phi) = integral(sqrt(1 - k^2 sin^2(theta)) dtheta, theta = 0..phi)
 *
 * Pi(k, nu)      = integral(dtheta / ((1 - nu sin^2(theta)) sqrt(1 - k^2 sin^2(theta))), theta = 0..pi/2)
 * Pi(k, nu, phi) = integral(dtheta / ((1 - nu sin^2(theta)) sqrt(1 - k^2 sin^2(theta))), theta = 0..phi)
 * </pre>
 *
 * <p>The current implementation stays in the real-valued domain and evaluates the defining
 * integrals with an adaptive Simpson scheme after reducing amplitudes into the principal
 * interval {@code [-pi/2, pi/2]}.
 */
final class EllipticIntegrals {

    private static final double HALF_PI = 0.5 * Math.PI;
    private static final double MODULUS_LIMIT = 1.0;
    private static final double ABSOLUTE_TOLERANCE = 1.0e-13;
    private static final double RELATIVE_TOLERANCE = 1.0e-13;
    private static final int MAX_ADAPTIVE_DEPTH = 24;

    private EllipticIntegrals() {
    }

    /**
     * Returns the complete elliptic integral of the first kind.
     *
     * @param k modulus with {@code |k| < 1}
     * @return {@code K(k)}
     */
    public static double completeFirstKind(double k) {
        validateModulus(k, "completeFirstKind");
        return integrateFirst(k, HALF_PI);
    }

    /**
     * Returns the incomplete elliptic integral of the first kind.
     *
     * @param k modulus with {@code |k| < 1}
     * @param phi amplitude
     * @return {@code F(k, phi)}
     */
    public static double incompleteFirstKind(double k, double phi) {
        validateModulus(k, "incompleteFirstKind");
        validateAmplitude(phi, "incompleteFirstKind");
        return reduceAmplitude(phi, () -> completeFirstKind(k), reduced -> integrateFirst(k, reduced));
    }

    /**
     * Returns the complete elliptic integral of the second kind.
     *
     * @param k modulus with {@code |k| < 1}
     * @return {@code E(k)}
     */
    public static double completeSecondKind(double k) {
        validateModulus(k, "completeSecondKind");
        return integrateSecond(k, HALF_PI);
    }

    /**
     * Returns the incomplete elliptic integral of the second kind.
     *
     * @param k modulus with {@code |k| < 1}
     * @param phi amplitude
     * @return {@code E(k, phi)}
     */
    public static double incompleteSecondKind(double k, double phi) {
        validateModulus(k, "incompleteSecondKind");
        validateAmplitude(phi, "incompleteSecondKind");
        return reduceAmplitude(phi, () -> completeSecondKind(k), reduced -> integrateSecond(k, reduced));
    }

    /**
     * Returns the complete elliptic integral of the third kind.
     *
     * @param k modulus with {@code |k| < 1}
     * @param nu characteristic with {@code nu < 1}
     * @return {@code Pi(k, nu)}
     */
    public static double completeThirdKind(double k, double nu) {
        validateModulus(k, "completeThirdKind");
        validateCharacteristic(nu, "completeThirdKind");
        return integrateThird(k, nu, HALF_PI);
    }

    /**
     * Returns the incomplete elliptic integral of the third kind.
     *
     * @param k modulus with {@code |k| < 1}
     * @param nu characteristic
     * @param phi amplitude
     * @return {@code Pi(k, nu, phi)}
     */
    public static double incompleteThirdKind(double k, double nu, double phi) {
        validateModulus(k, "incompleteThirdKind");
        validateAmplitude(phi, "incompleteThirdKind");

        double reduced = Math.IEEEremainder(phi, Math.PI);
        long periods = (long) Math.rint((phi - reduced) / Math.PI);
        validateIncompleteThirdDomain(nu, reduced, periods, "incompleteThirdKind");

        double principal = integrateThird(k, nu, reduced);
        if (periods == 0L) {
            return principal;
        }
        return 2.0 * periods * completeThirdKind(k, nu) + principal;
    }

    private static double integrateFirst(double k, double phi) {
        double modulusSquared = k * k;
        validateFirstAndSecondDomain(modulusSquared, phi, "first kind");
        return integrate(phi, theta -> 1.0 / Math.sqrt(1.0 - modulusSquared * squaredSine(theta)));
    }

    private static double integrateSecond(double k, double phi) {
        double modulusSquared = k * k;
        validateFirstAndSecondDomain(modulusSquared, phi, "second kind");
        return integrate(phi, theta -> Math.sqrt(1.0 - modulusSquared * squaredSine(theta)));
    }

    private static double integrateThird(double k, double nu, double phi) {
        double modulusSquared = k * k;
        validateFirstAndSecondDomain(modulusSquared, phi, "third kind");
        validateThirdIntegrand(nu, phi, "third kind");
        return integrate(phi, theta -> {
            double sineSquared = squaredSine(theta);
            return 1.0 / ((1.0 - nu * sineSquared) * Math.sqrt(1.0 - modulusSquared * sineSquared));
        });
    }

    private static double reduceAmplitude(double phi,
                                          DoubleSupplier completeSupplier,
                                          DoubleUnaryOperator principalEvaluator) {
        double reduced = Math.IEEEremainder(phi, Math.PI);
        long periods = (long) Math.rint((phi - reduced) / Math.PI);
        double principal = principalEvaluator.applyAsDouble(reduced);
        if (periods == 0L) {
            return principal;
        }
        return 2.0 * periods * completeSupplier.getAsDouble() + principal;
    }

    private static double integrate(double phi, DoubleUnaryOperator integrand) {
        if (phi == 0.0) {
            return 0.0;
        }
        if (phi < 0.0) {
            return -integrate(-phi, integrand);
        }

        double a = 0.0;
        double b = phi;
        double fa = integrand.applyAsDouble(a);
        double fm = integrand.applyAsDouble(0.5 * (a + b));
        double fb = integrand.applyAsDouble(b);
        double whole = simpson(a, b, fa, fm, fb);
        double tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * Math.abs(whole);
        return adaptiveSimpson(integrand, a, b, fa, fm, fb, whole, tolerance, MAX_ADAPTIVE_DEPTH);
    }

    private static double adaptiveSimpson(DoubleUnaryOperator integrand,
                                          double a,
                                          double b,
                                          double fa,
                                          double fm,
                                          double fb,
                                          double whole,
                                          double tolerance,
                                          int depth) {
        double midpoint = 0.5 * (a + b);
        double leftMidpoint = 0.5 * (a + midpoint);
        double rightMidpoint = 0.5 * (midpoint + b);

        double fLeftMidpoint = integrand.applyAsDouble(leftMidpoint);
        double fRightMidpoint = integrand.applyAsDouble(rightMidpoint);

        double left = simpson(a, midpoint, fa, fLeftMidpoint, fm);
        double right = simpson(midpoint, b, fm, fRightMidpoint, fb);
        double delta = left + right - whole;

        if (depth == 0 || Math.abs(delta) <= 15.0 * tolerance) {
            return left + right + delta / 15.0;
        }

        double halfTolerance = 0.5 * tolerance;
        return adaptiveSimpson(integrand, a, midpoint, fa, fLeftMidpoint, fm, left, halfTolerance, depth - 1)
            + adaptiveSimpson(integrand, midpoint, b, fm, fRightMidpoint, fb, right, halfTolerance, depth - 1);
    }

    private static double simpson(double a, double b, double fa, double fm, double fb) {
        return (b - a) * (fa + 4.0 * fm + fb) / 6.0;
    }

    private static double squaredSine(double theta) {
        double sine = Math.sin(theta);
        return sine * sine;
    }

    private static void validateModulus(double k, String functionName) {
        if (!Double.isFinite(k) || Math.abs(k) >= MODULUS_LIMIT) {
            throw new IllegalArgumentException(functionName + ": modulus must be finite with |k| < 1: " + k);
        }
    }

    private static void validateAmplitude(double phi, String functionName) {
        if (!Double.isFinite(phi)) {
            throw new IllegalArgumentException(functionName + ": amplitude must be finite: " + phi);
        }
    }

    private static void validateCharacteristic(double nu, String functionName) {
        if (!Double.isFinite(nu) || !(nu < 1.0)) {
            throw new IllegalArgumentException(functionName + ": characteristic must be finite with nu < 1: " + nu);
        }
    }

    private static void validateFirstAndSecondDomain(double modulusSquared, double phi, String functionName) {
        double radicand = 1.0 - modulusSquared * squaredSine(phi);
        if (!(radicand > 0.0)) {
            throw new IllegalArgumentException(functionName + ": real-valued domain requires 1 - k^2 sin^2(phi) > 0");
        }
    }

    private static void validateThirdIntegrand(double nu, double phi, String functionName) {
        double poleDistance = 1.0 - nu * squaredSine(phi);
        if (!(poleDistance > 0.0)) {
            throw new IllegalArgumentException(functionName + ": real-valued domain requires 1 - nu sin^2(phi) > 0");
        }
    }

    private static void validateIncompleteThirdDomain(double nu, double reducedPhi, long periods, String functionName) {
        if (!Double.isFinite(nu)) {
            throw new IllegalArgumentException(functionName + ": characteristic must be finite: " + nu);
        }
        if (periods != 0L && !(nu < 1.0)) {
            throw new IllegalArgumentException(functionName + ": characteristic must satisfy nu < 1 when amplitude spans full periods: " + nu);
        }
        validateThirdIntegrand(nu, reducedPhi, functionName);
    }

}