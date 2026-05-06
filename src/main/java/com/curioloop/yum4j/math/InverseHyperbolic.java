package com.curioloop.yum4j.math;

/**
 * Inverse hyperbolic functions aligned with Boost.Math real-scalar semantics.
 *
 * <p>The public surface intentionally mirrors the three real elementary functions
 * used by Boost's inverse-hyperbolic family:
 *
 * <pre>
 * asinh(x) = log(x + sqrt(x^2 + 1))
 * acosh(x) = log(x + sqrt(x^2 - 1)), x >= 1
 * atanh(x) = 0.5 * log((1 + x) / (1 - x)), |x| < 1
 * </pre>
 *
 * <p>Implementation notes:
 *
 * <ul>
 *   <li>Branch structure follows Boost.Math's double-precision implementation.</li>
 *   <li>Small-argument regions use short Taylor expansions to preserve digits.</li>
 *   <li>{@code asinh}/{@code acosh} switch to large-argument logarithmic asymptotics to
 *       avoid unnecessary cancellation.</li>
 *   <li>Domain errors raise {@link IllegalArgumentException}; the real overflow points
 *       of {@code atanh} at {@code x = +/-1} raise {@link ArithmeticException}.</li>
 * </ul>
 */
public final class InverseHyperbolic {

    private static final double EPSILON = Math.ulp(1.0);
    private static final double ROOT_EPSILON = Math.sqrt(EPSILON);
    private static final double FORTH_ROOT_EPSILON = Math.sqrt(ROOT_EPSILON);
    private static final double LARGE_ARGUMENT_THRESHOLD = 1.0 / ROOT_EPSILON;
    private static final double LOG_TWO = Math.log(2.0);

    private InverseHyperbolic() {
    }

    /**
     * Returns the inverse hyperbolic sine.
     *
     * @param x finite real argument, with {@code +/-infinity} accepted
     * @return {@code asinh(x)}
     */
    public static double asinh(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("asinh requires a finite argument, but got x = " + x);
        }
        if (x >= FORTH_ROOT_EPSILON) {
            if (x > LARGE_ARGUMENT_THRESHOLD) {
                return LOG_TWO + Math.log(x) + 1.0 / (4.0 * x * x);
            }
            if (x < 0.5) {
                return Math.log1p(x + sqrt1pm1(x * x));
            }
            return Math.log(x + Math.sqrt(x * x + 1.0));
        }
        if (x <= -FORTH_ROOT_EPSILON) {
            return -asinh(-x);
        }

        double result = x;
        if (Math.abs(x) >= ROOT_EPSILON) {
            double x3 = x * x * x;
            result -= x3 / 6.0;
        }
        return result;
    }

    /**
     * Returns the inverse hyperbolic cosine.
     *
     * @param x real argument on {@code [1, +infinity]}
     * @return {@code acosh(x)}
     */
    public static double acosh(double x) {
        if (x < 1.0 || Double.isNaN(x)) {
            throw new IllegalArgumentException("acosh requires x >= 1, but got x = " + x);
        }
        if (x - 1.0 >= ROOT_EPSILON) {
            if (x > LARGE_ARGUMENT_THRESHOLD) {
                return Math.log(x) + LOG_TWO;
            }
            if (x < 1.5) {
                double y = x - 1.0;
                return Math.log1p(y + Math.sqrt(y * y + 2.0 * y));
            }
            return Math.log(x + Math.sqrt(x * x - 1.0));
        }

        double y = x - 1.0;
        return Math.sqrt(2.0 * y) * (1.0 - y / 12.0 + 3.0 * y * y / 160.0);
    }

    /**
     * Returns the inverse hyperbolic tangent.
     *
     * @param x real argument in {@code [-1, 1]}
     * @return {@code atanh(x)}
     */
    public static double atanh(double x) {
        if (x < -1.0) {
            throw new IllegalArgumentException("atanh requires x >= -1, but got x = " + x);
        }
        if (x > 1.0) {
            throw new IllegalArgumentException("atanh requires x <= 1, but got x = " + x);
        }
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("atanh requires -1 <= x <= 1, but got x = " + x);
        }
        if (x < -1.0 + EPSILON || x > 1.0 - EPSILON) {
            throw new ArithmeticException("atanh overflow at x = " + x);
        }

        double absoluteX = Math.abs(x);
        if (absoluteX >= FORTH_ROOT_EPSILON) {
            if (absoluteX < 0.5) {
                return (Math.log1p(x) - Math.log1p(-x)) / 2.0;
            }
            return Math.log((1.0 + x) / (1.0 - x)) / 2.0;
        }

        double result = x;
        if (absoluteX >= ROOT_EPSILON) {
            double x3 = x * x * x;
            result += x3 / 3.0;
        }
        return result;
    }

    private static double sqrt1pm1(double x) {
        if (Math.abs(x) > 0.75) {
            return Math.sqrt(1.0 + x) - 1.0;
        }
        return x / (Math.sqrt(1.0 + x) + 1.0);
    }
}