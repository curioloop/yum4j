package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.Commons.validateFinite;

/**
 * Real scalar Heuman lambda function aligned with Boost.Math's public surface.
 *
 * <p>The public surface exposed here is
 *
 * <pre>
 * Lambda(k, phi)
 * </pre>
 *
 * for real modulus {@code |k| <= 1}. The implementation mirrors Boost's two-branch
 * structure: the principal-amplitude branch {@code |phi| <= pi/2} is evaluated through
 * Carlson symmetric integrals, while larger amplitudes reuse the existing elliptic-integral
 * and Jacobi-zeta surfaces without smoothing over Boost's singular boundary cases.
 */
public final class HeumanLambda {

    private static final double HALF_PI = 0.5 * Math.PI;

    private HeumanLambda() {
    }

    /**
     * Returns the real scalar Heuman lambda function.
     *
     * @param k modulus with {@code |k| <= 1}
     * @param phi finite amplitude
     * @return {@code Lambda(k, phi)}
     */
    public static double lambda(double k, double phi) {
        validateFinite(phi, "lambda: phi");
        if (!Double.isFinite(k) || Math.abs(k) > 1.0) {
            throw new IllegalArgumentException("lambda: modulus must be finite with |k| <= 1: " + k);
        }

        double sinPhi = Math.sin(phi);
        double cosPhi = Math.cos(phi);
        double sinSquared = sinPhi * sinPhi;
        double kSquared = k * k;
        double complementarySquared = 1.0 - kSquared;
        double delta = Math.sqrt(1.0 - complementarySquared * sinSquared);

        if (Math.abs(phi) <= HALF_PI) {
            if (k == 0.0 && Math.abs(phi) == HALF_PI) {
                return Double.NaN;
            }

            double deltaSquared = delta * delta;
            double factor = complementarySquared * sinPhi * cosPhi / (delta * HALF_PI);
            double rf = CarlsonIntegrals.rf(0.0, complementarySquared, 1.0);
            double rj = CarlsonIntegrals.rj(
                0.0,
                complementarySquared,
                1.0,
                1.0 - kSquared / deltaSquared
            );
            return factor * (rf + (kSquared * rj) / (3.0 * deltaSquared));
        }

        double complementaryModulus = Math.sqrt(complementarySquared);
        if (complementaryModulus == 0.0) {
            throw new ArithmeticException("lambda overflow at |k| = 1 for |phi| > pi/2: " + k);
        }
        if (complementaryModulus == 1.0) {
            throw new IllegalArgumentException("lambda: when 1 - k^2 == 1, |phi| must be < pi/2: " + phi);
        }

        double ratio = EllipticIntegrals.incompleteFirstKind(complementaryModulus, phi)
            / EllipticIntegrals.completeFirstKind(complementaryModulus);
        return ratio + EllipticIntegrals.completeFirstKind(k)
            * JacobiZeta.zeta(complementaryModulus, phi) / HALF_PI;
    }
}