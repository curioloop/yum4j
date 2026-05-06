package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.HornerPoly.evaluatePolynomial;

/**
 * Real scalar Lambert W branches aligned with Boost.Math 1.90.
 *
 * <p>The Lambert W function is the inverse of {@code w -> w * exp(w)} on the real branches
 * retained by Boost's public surface:
 *
 * <pre>
 * w0(z)  on [-exp(-1), +infinity)
 * wm1(z) on [-exp(-1), 0)
 * </pre>
 *
 * <p>Implementation notes:
 *
 * <ul>
 *   <li>The public surface mirrors Boost's real branch naming: {@code w0}, {@code wm1},
 *       plus their first derivatives.</li>
 *   <li>Seeds follow the same qualitative regions as Boost's implementation: small-z series,
 *       branch-point series near {@code -exp(-1)}, and logarithmic asymptotics for larger
 *       magnitudes.</li>
 *   <li>Refinement uses a guarded Halley iteration on a branch-preserving bracket, so the
 *       result stays on the requested real branch even when the first seed is rough.</li>
 *   <li>Domain errors raise {@link IllegalArgumentException}; Boost overflow points such as
 *       {@code wm1(0)} and the derivative singularities raise {@link ArithmeticException}.</li>
 * </ul>
 */
public final class LambertW {

    private static final double BRANCH_POINT = -1.0 / Math.E;
    private static final double SMALL_Z_THRESHOLD = 0.05;
    private static final double W0_SINGULARITY_THRESHOLD = -0.3;
    private static final double WM1_SINGULARITY_THRESHOLD = -0.35;
    private static final double RELATIVE_TOLERANCE = 8.0 * Math.ulp(1.0);
    private static final int MAX_ITERATIONS = 64;

    private static final double[] SMALL_Z_COEFFICIENTS = {
        0.0,
        1.0,
        -1.0,
        1.5,
        -2.6666666666666666667,
        5.2083333333333333333,
        -10.8,
        23.343055555555555556,
        -52.012698412698412698,
        118.62522321428571429,
        -275.57319223985890653,
        649.78717234347442681,
        -1551.1605194805194805,
        3741.4497029592385495,
        -9104.5002411580189358,
        22324.308512706601434,
        -55103.621972903835338,
        136808.86090394293563
    };

    private static final double[] SINGULARITY_COEFFICIENTS = {
        -1.0,
        1.0,
        -1.0 / 3.0,
        11.0 / 72.0,
        -43.0 / 540.0,
        769.0 / 17280.0,
        -221.0 / 8505.0,
        680863.0 / 43545600.0,
        -1963.0 / 204120.0,
        226287557.0 / 37623398400.0,
        -5776369.0 / 1515591000.0,
        169709463197.0 / 69528040243200.0,
        -1118511313.0 / 709296588000.0,
        667874164916771.0 / 650782456676352000.0,
        -500525573.0 / 744761417400.0,
        0.000442473061814620910,
        -0.000292677224729627445,
        0.000194387276054539318,
        -0.000129574266852748819,
        0.0000866503580520812717,
        -0.00005811360750441381677220546477882818
    };

    private LambertW() {
    }

    /**
     * Returns the principal real branch {@code W0(z)}.
     *
     * @param z real argument on {@code [-exp(-1), +infinity)}
     * @return {@code W0(z)}
     */
    public static double w0(double z) {
        validateW0Argument(z);
        if (z == 0.0) {
            return z;
        }
        if (z == BRANCH_POINT) {
            return -1.0;
        }
        if (Math.abs(z) < SMALL_Z_THRESHOLD) {
            return smallZSeries(z);
        }
        if (z < W0_SINGULARITY_THRESHOLD) {
            return singularitySeries(Math.sqrt(2.0 * (Math.E * z + 1.0)));
        }

        double guess = initialGuessW0(z);
        return refine(z, guess, true);
    }

    /**
     * Returns the lower real branch {@code W-1(z)}.
     *
     * @param z real argument on {@code [-exp(-1), 0)}
     * @return {@code W-1(z)}
     */
    public static double wm1(double z) {
        validateWm1Argument(z);
        if (z == BRANCH_POINT) {
            return -1.0;
        }
        if (z < WM1_SINGULARITY_THRESHOLD) {
            return singularitySeries(-Math.sqrt(2.0 * (Math.E * z + 1.0)));
        }

        double guess = initialGuessWm1(z);
        return refine(z, guess, false);
    }

    /**
     * Returns the first derivative of the principal real branch.
     *
     * @param z real argument on {@code [-exp(-1), +infinity)}
     * @return {@code dW0(z) / dz}
     */
    public static double w0Prime(double z) {
        validateW0Argument(z);
        if (z == 0.0) {
            return 1.0;
        }
        if (z == BRANCH_POINT) {
            throw new ArithmeticException("w0Prime overflow at z = " + z);
        }

        double w = w0(z);
        return w / (z * (1.0 + w));
    }

    /**
     * Returns the first derivative of the lower real branch.
     *
     * @param z real argument on {@code [-exp(-1), 0)}
     * @return {@code dW-1(z) / dz}
     */
    public static double wm1Prime(double z) {
        validateWm1Argument(z);
        if (z == BRANCH_POINT || z == 0.0) {
            throw new ArithmeticException("wm1Prime overflow at z = " + z);
        }

        double w = wm1(z);
        return w / (z * (1.0 + w));
    }

    private static void validateW0Argument(double z) {
        if (Double.isNaN(z)) {
            throw new IllegalArgumentException("w0 requires z >= -exp(-1), but got z = " + z);
        }
        if (Double.isInfinite(z)) {
            throw new ArithmeticException("w0 requires a finite argument, but got z = " + z);
        }
        if (z < BRANCH_POINT) {
            throw new IllegalArgumentException("w0 requires z >= -exp(-1), but got z = " + z);
        }
    }

    private static void validateWm1Argument(double z) {
        if (!Double.isFinite(z)) {
            throw new IllegalArgumentException("wm1 requires -exp(-1) <= z <= 0, but got z = " + z);
        }
        if (z == 0.0) {
            throw new ArithmeticException("wm1 overflow at z = 0.0");
        }
        if (Math.abs(z) < Double.MIN_NORMAL) {
            throw new ArithmeticException("wm1 overflow for subnormal z = " + z);
        }
        if (z > 0.0 || z < BRANCH_POINT) {
            throw new IllegalArgumentException("wm1 requires -exp(-1) <= z < 0, but got z = " + z);
        }
    }

    private static double initialGuessW0(double z) {
        if (z < 8.0) {
            return Math.log1p(z);
        }
        return asymptoticGuessW0(z);
    }

    private static double initialGuessWm1(double z) {
        return asymptoticGuessWm1(z);
    }

    private static double refine(double z, double guess, boolean principalBranch) {
        double lower = principalBranch ? -1.0 : lowerBracketWm1(z, guess);
        double upper = principalBranch ? upperBracketW0(z) : -1.0;
        double w = clamp(guess, lower, upper);

        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            double expW = Math.exp(w);
            double residual = w * expW - z;
            if (residual == 0.0) {
                return w;
            }

            if (principalBranch) {
                if (residual < 0.0) {
                    lower = w;
                } else {
                    upper = w;
                }
            } else {
                if (residual > 0.0) {
                    lower = w;
                } else {
                    upper = w;
                }
            }

            double candidate = halleyCandidate(w, expW, residual, z);
            if (!Double.isFinite(candidate) || candidate <= lower || candidate >= upper) {
                candidate = 0.5 * (lower + upper);
            }

            if (Math.abs(candidate - w) <= RELATIVE_TOLERANCE * Math.max(1.0, Math.abs(candidate))) {
                return candidate;
            }
            w = candidate;
        }
        return w;
    }

    private static double halleyCandidate(double w, double expW, double residual, double z) {
        double denominator = z * (w + 2.0) + expW * (w * (w + 2.0) + 2.0);
        if (denominator == 0.0 || !Double.isFinite(denominator)) {
            return Double.NaN;
        }
        return w - 2.0 * (w + 1.0) * residual / denominator;
    }

    private static double upperBracketW0(double z) {
        if (z < 0.0) {
            return 0.0;
        }
        if (z < Math.E) {
            return 1.0;
        }
        return Math.log(z);
    }

    private static double lowerBracketWm1(double z, double guess) {
        double lower = Math.min(-2.0, guess - 1.0);
        while (lower * Math.exp(lower) - z <= 0.0) {
            lower *= 2.0;
            if (!Double.isFinite(lower)) {
                return -Double.MAX_VALUE;
            }
        }
        return lower;
    }

    private static double asymptoticGuessW0(double z) {
        double logZ = Math.log(z);
        double logLogZ = Math.log(logZ);
        return logZ - logLogZ + logLogZ / logZ;
    }

    private static double asymptoticGuessWm1(double z) {
        double logMinusZ = Math.log(-z);
        double logLogMinusZ = Math.log(-logMinusZ);
        return logMinusZ - logLogMinusZ + logLogMinusZ / logMinusZ;
    }

    private static double smallZSeries(double z) {
        return evaluatePolynomial(SMALL_Z_COEFFICIENTS, z);
    }

    private static double singularitySeries(double p) {
        return evaluatePolynomial(SINGULARITY_COEFFICIENTS, p);
    }

    private static double clamp(double value, double lower, double upper) {
        return Math.max(lower, Math.min(upper, value));
    }
}