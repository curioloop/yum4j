package com.curioloop.yum4j.math;

/**
 * Real scalar Jacobi elliptic functions aligned with Boost.Math 1.90.
 *
 * <p>The core surface is the triple {@code sn(k, u)}, {@code cn(k, u)}, and {@code dn(k, u)}
 * parameterized by the real modulus {@code k >= 0} and real argument {@code u}. The quotient
 * families exposed by Boost are derived directly from that triple.
 *
 * <p>Implementation notes:
 *
 * <ul>
 *   <li>The core evaluation follows Boost's AGM-style recursion from
 *       {@code jacobi_elliptic.hpp}.</li>
 *   <li>{@code k > 1} uses the same reciprocal-modulus transform as Boost.</li>
 *   <li>The public Java adaptation exposes the shared core as {@link #snCnDn(double, double)}
 *       returning {@link Double3} in {@code (sn, cn, dn)} order.</li>
 *   <li>Domain errors raise {@link IllegalArgumentException}.</li>
 * </ul>
 */
public final class JacobiElliptic {

    private static final double FORTH_ROOT_EPSILON = Math.sqrt(Math.sqrt(Math.ulp(1.0)));

    private JacobiElliptic() {
    }

    /**
     * Returns the core Jacobi elliptic triple in {@code (sn, cn, dn)} order.
     *
     * @param k real modulus with {@code k >= 0}
     * @param theta real argument
     * @return {@link Double3} containing {@code (sn, cn, dn)}
     */
    public static Double3 snCnDn(double k, double theta) {
        validateInputs(k, theta);
        return evaluate(theta, k);
    }

    public static double sn(double k, double theta) {
        return snCnDn(k, theta)._1();
    }

    public static double cn(double k, double theta) {
        return snCnDn(k, theta)._2();
    }

    public static double dn(double k, double theta) {
        return snCnDn(k, theta)._3();
    }

    public static double cd(double k, double theta) {
        Double3 values = snCnDn(k, theta);
        return values._2() / values._3();
    }

    public static double dc(double k, double theta) {
        Double3 values = snCnDn(k, theta);
        return values._3() / values._2();
    }

    public static double ns(double k, double theta) {
        return 1.0 / sn(k, theta);
    }

    public static double sd(double k, double theta) {
        Double3 values = snCnDn(k, theta);
        return values._1() / values._3();
    }

    public static double ds(double k, double theta) {
        Double3 values = snCnDn(k, theta);
        return values._3() / values._1();
    }

    public static double nc(double k, double theta) {
        return 1.0 / cn(k, theta);
    }

    public static double nd(double k, double theta) {
        return 1.0 / dn(k, theta);
    }

    public static double sc(double k, double theta) {
        Double3 values = snCnDn(k, theta);
        return values._1() / values._2();
    }

    public static double cs(double k, double theta) {
        Double3 values = snCnDn(k, theta);
        return values._2() / values._1();
    }

    private static void validateInputs(double k, double theta) {
        if (!Double.isFinite(k)) {
            throw new IllegalArgumentException("JacobiElliptic: modulus must be finite: " + k);
        }
        if (!Double.isFinite(theta)) {
            throw new IllegalArgumentException("JacobiElliptic: theta must be finite: " + theta);
        }
        if (k < 0.0) {
            throw new IllegalArgumentException("JacobiElliptic: modulus k must be non-negative: " + k);
        }
    }

    private static Double3 evaluate(double theta, double k) {
        if (k > 1.0) {
            double scaledTheta = theta * k;
            double reciprocalModulus = 1.0 / k;
            Double3 transformed = evaluate(scaledTheta, reciprocalModulus);
            return new Double3(transformed._1() * reciprocalModulus, transformed._3(), transformed._2());
        }

        if (theta == 0.0) {
            return new Double3(0.0, 1.0, 1.0);
        }
        if (k == 0.0) {
            return new Double3(Math.sin(theta), Math.cos(theta), 1.0);
        }
        if (k == 1.0) {
            double sech = 1.0 / Math.cosh(theta);
            return new Double3(Math.tanh(theta), sech, sech);
        }
        if (k < FORTH_ROOT_EPSILON) {
            return smallModulusApproximation(theta, k);
        }

        double kc = 1.0 - k;
        double kPrime = k < 0.5 ? Math.sqrt(1.0 - k * k) : Math.sqrt(2.0 * kc - kc * kc);
        double[] t1 = new double[1];
        double t0 = recurse(theta, 1.0, kPrime, 0, t1);
        double cn = Math.cos(t0);
        double dn = cn / Math.cos(t1[0] - t0);
        double sn = Math.sin(t0);
        return new Double3(sn, cn, dn);
    }

    private static Double3 smallModulusApproximation(double theta, double k) {
        double sine = Math.sin(theta);
        double cosine = Math.cos(theta);
        double modulusSquared = k * k;
        double thetaCorrection = theta - sine * cosine;
        double dn = 1.0 - modulusSquared * sine * sine / 2.0;
        double cn = cosine + modulusSquared * thetaCorrection * sine / 4.0;
        double sn = sine - modulusSquared * thetaCorrection * cosine / 4.0;
        return new Double3(sn, cn, dn);
    }

    private static double recurse(double x, double anm1, double bnm1, int depth, double[] tN) {
        int nextDepth = depth + 1;
        double currentC = (anm1 - bnm1) / 2.0;
        double currentA = (anm1 + bnm1) / 2.0;

        double inner;
        if (currentC < Math.ulp(1.0)) {
            inner = Math.scalb(1.0, nextDepth) * x * currentA;
        } else {
            inner = recurse(x, currentA, Math.sqrt(anm1 * bnm1), nextDepth, null);
        }

        if (tN != null) {
            tN[0] = inner;
        }

        return (inner + Math.asin((currentC / currentA) * Math.sin(inner))) / 2.0;
    }
}