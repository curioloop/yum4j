package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.Commons.validateFinite;

/**
 * Real scalar Jacobi zeta function.
 *
 * <p>The Jacobi zeta function is defined by
 *
 * <pre>
 * Z(k, phi) = E(k, phi) - E(k) / K(k) * F(k, phi)
 * </pre>
 *
 * <p>where {@code F} and {@code E} are the incomplete and complete elliptic integrals
 * of the first and second kinds. The current implementation stays in the real-valued
 * domain for {@code |k| <= 1} and mirrors Boost.Math's {@code jacobi_zeta.hpp}
 * routing: negative amplitudes are reflected through oddness, {@code k == 1}
 * uses Boost's explicit special case, and the general path evaluates the
 * equivalent Carlson {@code R_J} representation over the existing
 * {@link CarlsonIntegrals} and {@link EllipticIntegrals} kernels.
 */
public final class JacobiZeta {

    private JacobiZeta() {
    }

    /**
     * Returns the real-valued Jacobi zeta function {@code Z(k, phi)}.
     *
     * @param k modulus with {@code |k| <= 1}
     * @param phi amplitude
     * @return {@code Z(k, phi)}
     */
    public static double zeta(double k, double phi) {
        validateFinite(phi, "zeta: phi");
        if (!Double.isFinite(k) || Math.abs(k) > 1.0) {
            throw new IllegalArgumentException("zeta: modulus must be finite with |k| <= 1: " + k);
        }

        if (phi == 0.0 || k == 0.0) {
            return 0.0;
        }

        boolean invert = false;
        if (phi < 0.0) {
            phi = -phi;
            invert = true;
        }

        double sine = Math.sin(phi);
        double cosine = Math.cos(phi);
        if (k == 1.0) {
            double value = sine * Math.signum(cosine);
            return invert ? -value : value;
        }

        double cosineSquared = cosine * cosine;
        double complementaryParameter = 1.0 - k * k;
        double oneMinusKSineSquared = complementaryParameter + cosineSquared - complementaryParameter * cosineSquared;
        double symmetricThird = CarlsonIntegrals.rj(0.0, complementaryParameter, 1.0, oneMinusKSineSquared);
        double completeFirst = EllipticIntegrals.completeFirstKind(k);
        double value = k * k * sine * cosine * Math.sqrt(oneMinusKSineSquared)
            * symmetricThird
            / (3.0 * completeFirst);
        return invert ? -value : value;
    }

}
