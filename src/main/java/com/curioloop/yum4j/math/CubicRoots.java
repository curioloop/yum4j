package com.curioloop.yum4j.math;

/**
 * Real-root solver for cubic polynomials.
 *
 * <p>The solver first normalizes
 *
 * <pre>
 * c1 * x^3 + c2 * x^2 + c3 * x + c4 = 0
 * </pre>
 *
 * to the monic form {@code x^3 + a x^2 + b x + c = 0}, then evaluates the
 * Boost-aligned {@code Q/R} classification and returns a fixed three-slot result.
 *
 * <p>The returned {@link Double3} is always sorted in ascending order. Missing
 * real roots are represented by {@code Double.NaN}, and repeated roots are preserved.
 */
public final class CubicRoots {

    private static final double SQRT_3 = Math.sqrt(3.0);
    private static final Double2 NAN_PAIR = new Double2(Double.NaN, Double.NaN);
    private static final Double3 NAN_TRIPLE = new Double3(Double.NaN, Double.NaN, Double.NaN);

    private CubicRoots() {}

    /**
     * Returns the real roots in ascending order using a fixed three-slot result.
     *
     * <p>Missing real roots are represented by {@code Double.NaN}. If the leading
     * coefficient vanishes, the method falls back to the quadratic or linear equation
     * implied by the remaining coefficients.
     */
    public static Double3 solveReal(double c1, double c2, double c3, double c4) {
        validateFinite(c1, c2, c3, c4);

        if (c1 == 0.0) {
            if (c2 == 0.0) {
                if (c3 == 0.0) {
                    return c4 != 0.0 ? NAN_TRIPLE : new Double3(0.0, 0.0, 0.0);
                }
                return new Double3(canonicalizeZero(-c4 / c3), Double.NaN, Double.NaN);
            }
            Double2 roots = solveQuadratic(c2, c3, c4);
            return sortRoots(roots._1(), roots._2(), Double.NaN);
        }

        if (c4 == 0.0) {
            Double2 roots = solveQuadratic(c1, c2, c3);
            return sortRoots(roots._1(), roots._2(), 0.0);
        }

        double p = c2 / c1;
        double q = c3 / c1;
        double r = c4 / c1;
        double qTerm = (p * p - 3.0 * q) / 9.0;
        double rTerm = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0;

        double root0 = Double.NaN;
        double root1 = Double.NaN;
        double root2 = Double.NaN;
        if (rTerm * rTerm < qTerm * qTerm * qTerm) {
            double rootQ = Math.sqrt(qTerm);
            double theta = Math.acos(rTerm / (qTerm * rootQ)) / 3.0;
            double sinTheta = Math.sin(theta);
            double cosTheta = Math.cos(theta);
            root0 = -2.0 * rootQ * cosTheta - p / 3.0;
            root1 = -rootQ * (-cosTheta + SQRT_3 * sinTheta) - p / 3.0;
            root2 = rootQ * (cosTheta + SQRT_3 * sinTheta) - p / 3.0;
        } else {
            double arg = rTerm * rTerm - qTerm * qTerm * qTerm;
            double aTerm = (rTerm >= 0.0 ? -1.0 : 1.0) * Math.cbrt(Math.abs(rTerm) + Math.sqrt(arg));
            double bTerm = aTerm != 0.0 ? qTerm / aTerm : 0.0;
            root0 = aTerm + bTerm - p / 3.0;
            if (aTerm == bTerm || arg == 0.0) {
                root1 = -aTerm - p / 3.0;
                root2 = -aTerm - p / 3.0;
            }
        }

        root0 = polishRoot(c1, c2, c3, c4, root0);
        root1 = polishRoot(c1, c2, c3, c4, root1);
        root2 = polishRoot(c1, c2, c3, c4, root2);
        return sortRoots(root0, root1, root2);
    }

    /**
     * Returns the empirical residual {@code p(r)} and expected residual
     * {@code eps * |r p'(r)|} for a candidate cubic root.
     *
     * <p>The returned {@link Double2} stores the empirical residual in {@code _1}
     * and the expected residual bound in {@code _2}.
     */
    public static Double2 cubicRootResidual(double c1, double c2, double c3, double c4, double root) {
        validateFinite(c1, c2, c3, c4);

        double residual = Math.fma(c1, root, c2);
        residual = Math.fma(residual, root, c3);
        residual = Math.fma(residual, root, c4);

        double absRoot = Math.abs(root);
        double expectedResidual = Math.fma(4.0 * Math.abs(c1), absRoot, 3.0 * Math.abs(c2));
        expectedResidual = Math.fma(expectedResidual, absRoot, 2.0 * Math.abs(c3));
        expectedResidual = Math.fma(expectedResidual, absRoot, Math.abs(c4));
        return new Double2(residual, expectedResidual * Math.ulp(1.0));
    }

    /**
     * Returns the cubic rootfinding condition number used by Boost for a
     * candidate root.
     */
    public static double cubicRootConditionNumber(double c1, double c2, double c3, double c4, double root) {
        validateFinite(c1, c2, c3, c4);

        if (root == 0.0) {
            return Double.POSITIVE_INFINITY;
        }

        double absRoot = Math.abs(root);
        double numerator = Math.fma(Math.abs(c1), absRoot, Math.abs(c2));
        numerator = Math.fma(numerator, absRoot, Math.abs(c3));
        numerator = Math.fma(numerator, absRoot, Math.abs(c4));

        double denominator = Math.fma(3.0 * c1, root, 2.0 * c2);
        denominator = Math.fma(denominator, root, c3);
        if (denominator == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return numerator / Math.abs(denominator * root);
    }

    private static Double2 solveQuadratic(double c1, double c2, double c3) {
        if (c1 == 0.0) {
            if (c2 == 0.0 && c3 != 0.0) {
                return NAN_PAIR;
            }
            if (c2 == 0.0) {
                return new Double2(0.0, 0.0);
            }
            double root = canonicalizeZero(-c3 / c2);
            return new Double2(root, root);
        }

        if (c2 == 0.0) {
            double rootSquared = -c3 / c1;
            if (rootSquared < 0.0) {
                return NAN_PAIR;
            }
            double root = Math.sqrt(rootSquared);
            return new Double2(canonicalizeZero(-root), canonicalizeZero(root));
        }

        double discriminant = Math.fma(c2, c2, -4.0 * c1 * c3);
        if (discriminant < 0.0) {
            return NAN_PAIR;
        }
        double q = -(c2 + Math.copySign(Math.sqrt(discriminant), c2)) / 2.0;
        double root0 = q / c1;
        double root1 = c3 / q;
        if (root0 < root1) {
            return new Double2(canonicalizeZero(root0), canonicalizeZero(root1));
        }
        return new Double2(canonicalizeZero(root1), canonicalizeZero(root0));
    }

    private static double polishRoot(double c1, double c2, double c3, double c4, double root) {
        if (!Double.isFinite(root)) {
            return root;
        }
        double functionValue = Math.fma(c1, root, c2);
        functionValue = Math.fma(functionValue, root, c3);
        functionValue = Math.fma(functionValue, root, c4);
        double firstDerivative = Math.fma(3.0 * c1, root, 2.0 * c2);
        firstDerivative = Math.fma(firstDerivative, root, c3);
        if (firstDerivative != 0.0) {
            double secondDerivative = Math.fma(6.0 * c1, root, 2.0 * c2);
            double denominator = 2.0 * firstDerivative * firstDerivative - functionValue * secondDerivative;
            if (denominator != 0.0) {
                root -= 2.0 * functionValue * firstDerivative / denominator;
            } else {
                root -= functionValue / firstDerivative;
            }
        }
        return canonicalizeZero(root);
    }

    private static Double3 sortRoots(double root0, double root1, double root2) {
        double first = canonicalizeZero(root0);
        double second = canonicalizeZero(root1);
        double third = canonicalizeZero(root2);

        if (Double.compare(first, second) > 0) {
            double swap = first;
            first = second;
            second = swap;
        }
        if (Double.compare(second, third) > 0) {
            double swap = second;
            second = third;
            third = swap;
        }
        if (Double.compare(first, second) > 0) {
            double swap = first;
            first = second;
            second = swap;
        }

        return new Double3(first, second, third);
    }

    private static double canonicalizeZero(double value) {
        return value == 0.0 ? 0.0 : value;
    }

    private static void validateFinite(double c1, double c2, double c3, double c4) {
        if (!Double.isFinite(c1)
            || !Double.isFinite(c2)
            || !Double.isFinite(c3)
            || !Double.isFinite(c4)) {
            throw new IllegalArgumentException("Polynomial coefficients must be finite");
        }
    }
}