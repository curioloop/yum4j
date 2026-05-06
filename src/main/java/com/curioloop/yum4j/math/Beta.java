package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.Commons.uncheckedFactorial;
import static com.curioloop.yum4j.math.Commons.validateProbability;
import static com.curioloop.yum4j.math.Lanczos13m53.*;
import static com.curioloop.yum4j.math.HornerPoly.evaluateEvenPolynomial;
import static com.curioloop.yum4j.math.HornerPoly.evaluateOddPolynomial;
import static com.curioloop.yum4j.math.HornerPoly.evaluatePolynomial;

/**
 * Beta-family special functions aligned with Boost.Math semantics.
 *
 * <p>The public surface covers the complete beta function, the lower and upper
 * incomplete beta integrals, their regularized ratios, the derivative of the
 * regularized lower ratio with respect to {@code x}, and the inverse problems in
 * {@code x}, {@code a}, and {@code b}.
 *
 * <p>Definitions:
 *
 * <pre>
 * Complete beta:
 *     B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)
 *
 * Lower incomplete beta:
 *     B_x(a, b) = integral(t^(a - 1) * (1 - t)^(b - 1), t = 0..x)
 *
 * Upper incomplete beta complement:
 *     B̄_x(a, b) = integral(t^(a - 1) * (1 - t)^(b - 1), t = x..1)
 *
 * Lower regularized incomplete beta:
 *     I_x(a, b) = B_x(a, b) / B(a, b)
 *
 * Upper regularized incomplete beta complement:
 *     Ī_x(a, b) = B̄_x(a, b) / B(a, b) = 1 - I_x(a, b)
 *
 * Derivative of the regularized lower ratio:
 *     dI_x(a, b)/dx = x^(a - 1) * (1 - x)^(b - 1) / B(a, b)
 * </pre>
 *
 * <p>Implementation notes:
 *
 * <ul>
 *   <li>The complete beta uses a Lanczos-style gamma ratio, following the same
 *       broad structure used by Boost.Math for stable double-precision evaluation.</li>
 *   <li>The regularized functions switch among symmetry transforms, lower-tail
 *       series expansions, and continued fractions according to the input region.</li>
 *   <li>{@code ibetaInv}/{@code ibetacInv} use Boost-style special-case initializers
 *       and a digits-driven Halley refinement on a bracketed interval.</li>
 *   <li>{@code ibetaInva}/{@code ibetacInva}/{@code ibetaInvb}/{@code ibetacInvb}
 *       solve the inverse-on-parameter problems by Boost-style bracketing followed by
 *       TOMS748 refinement.</li>
 * </ul>
 */
public final class Beta {

    private static final double EPSILON = Math.ulp(1.0);
    private static final double SERIES_EPSILON = 1.0e-15;
    private static final double CONTINUED_FRACTION_EPSILON = 1.0e-15;
    private static final double LOG_MAX_VALUE = Math.log(Double.MAX_VALUE);
    private static final double LOG_MIN_VALUE = Math.log(Double.MIN_VALUE);
    private static final double MIN_PARAMETER = Double.MIN_NORMAL;
    private static final double PARAMETER_SEARCH_FLOOR = Double.MIN_NORMAL;
    private static final double HALF_PI = 0.5 * Math.PI;
    private static final double SQRT_TWO = Math.sqrt(2.0);
    private static final int PN_SIZE = 30;
    private static final int DOUBLE_PRECISION_BITS = 53;
    private static final int MAX_ITERATIONS = 256;
    private static final double[] TEMME_METHOD2_COEFFS_1 = {-1.0, -5.0, 5.0};
    private static final double[] TEMME_METHOD2_COEFFS_2 = {1.0, 21.0, -69.0, 46.0};
    private static final double[] TEMME_METHOD2_COEFFS_3 = {7.0, -2.0, 33.0, -62.0, 31.0};
    private static final double[] TEMME_METHOD2_COEFFS_4 = {25.0, -52.0, -17.0, 88.0, -115.0, 46.0};
    private static final double[] TEMME_METHOD2_COEFFS_5 = {7.0, 12.0, -78.0, 52.0};
    private static final double[] TEMME_METHOD2_COEFFS_6 = {-7.0, 2.0, 183.0, -370.0, 185.0};
    private static final double[] TEMME_METHOD2_COEFFS_7 = {-533.0, 776.0, -1835.0, 10240.0, -13525.0, 5410.0};
    private static final double[] TEMME_METHOD2_COEFFS_8 = {-1579.0, 3747.0, -3372.0, -15821.0, 45588.0, -45213.0, 15071.0};
    private static final double[] TEMME_METHOD2_COEFFS_9 = {449.0, -1259.0, -769.0, 6686.0, -9260.0, 3704.0};
    private static final double[] TEMME_METHOD2_COEFFS_10 = {63149.0, -151557.0, 140052.0, -727469.0, 2239932.0, -2251437.0, 750479.0};
    private static final double[] TEMME_METHOD2_COEFFS_11 = {29233.0, -78755.0, 105222.0, 146879.0, -1602610.0, 3195183.0, -2554139.0, 729754.0};
    private static final double[] TEMME_METHOD2_SMALL_ETA_COEFFS = {1.0, -13.0, 13.0};

    private Beta() {
    }

    /**
     * Returns the complete beta function.
     *
     * <pre>
     * beta(a, b) = B(a, b)
     *            = Gamma(a) * Gamma(b) / Gamma(a + b)
     * </pre>
     *
     * @param a first shape parameter, must be strictly positive
     * @param b second shape parameter, must be strictly positive
     * @return complete beta value {@code B(a, b)}
     */
    public static double beta(double a, double b) {
        validatePositiveParameters(a, b, "beta");
        return completeBeta(a, b);
    }

    /**
     * Returns the lower incomplete beta function.
     *
     * <pre>
     * beta(a, b, x) = B_x(a, b)
     *               = integral(t^(a - 1) * (1 - t)^(b - 1), t = 0..x)
     * </pre>
     *
     * @param a first shape parameter, must be strictly positive
     * @param b second shape parameter, must be strictly positive
     * @param x upper integration bound in {@code [0, 1]}
     * @return lower incomplete beta value {@code B_x(a, b)}
     */
    public static double beta(double a, double b, double x) {
        validateIncompleteParameters(a, b, x, "beta");
        return ibetaImp(a, b, x, false, false);
    }

    /**
     * Returns the upper incomplete beta complement.
     *
     * <pre>
     * betac(a, b, x) = B̄_x(a, b)
     *                = integral(t^(a - 1) * (1 - t)^(b - 1), t = x..1)
     *                = B(a, b) - B_x(a, b)
     * </pre>
     *
     * @param a first shape parameter, must be strictly positive
     * @param b second shape parameter, must be strictly positive
     * @param x split point in {@code [0, 1]}
     * @return upper incomplete beta complement {@code B̄_x(a, b)}
     */
    public static double betac(double a, double b, double x) {
        validateIncompleteParameters(a, b, x, "betac");
        return ibetaImp(a, b, x, true, false);
    }

    /**
     * Returns the regularized lower incomplete beta ratio.
     *
     * <pre>
     * ibeta(a, b, x) = I_x(a, b)
     *                = B_x(a, b) / B(a, b)
     * </pre>
     *
     * <p>Boost-compatible domain rules are used: {@code a >= 0}, {@code b >= 0},
     * not both zero, and {@code x in [0, 1]}.
     *
     * @param a first shape parameter, must be non-negative
     * @param b second shape parameter, must be non-negative
     * @param x upper integration bound in {@code [0, 1]}
     * @return regularized lower ratio {@code I_x(a, b)}
     */
    public static double ibeta(double a, double b, double x) {
        validateRegularizedParameters(a, b, x, "ibeta");
        return ibetaImp(a, b, x, false, true);
    }

    /**
     * Returns the regularized upper incomplete beta complement.
     *
     * <pre>
     * ibetac(a, b, x) = Ī_x(a, b)
     *                 = B̄_x(a, b) / B(a, b)
     *                 = 1 - I_x(a, b)
     * </pre>
     *
     * @param a first shape parameter, must be non-negative
     * @param b second shape parameter, must be non-negative
     * @param x split point in {@code [0, 1]}
     * @return regularized upper complement {@code Ī_x(a, b)}
     */
    public static double ibetac(double a, double b, double x) {
        validateRegularizedParameters(a, b, x, "ibetac");
        return ibetaImp(a, b, x, true, true);
    }

    /**
     * Returns the derivative of the regularized lower incomplete beta ratio with respect to {@code x}.
     *
     * <pre>
     * ibetaDerivative(a, b, x) = dI_x(a, b)/dx
     *                          = x^(a - 1) * (1 - x)^(b - 1) / B(a, b)
     * </pre>
     *
     * @param a first shape parameter, must be strictly positive
     * @param b second shape parameter, must be strictly positive
     * @param x evaluation point in {@code [0, 1]}
     * @return derivative of {@code I_x(a, b)} with respect to {@code x}
     */
    public static double ibetaDerivative(double a, double b, double x) {
        validatePositiveParameters(a, b, "ibetaDerivative");
        validateUnitInterval(x, "x");
        return ibetaDerivativeImp(a, b, x);
    }

    /**
     * Solves for {@code x} in the regularized lower incomplete beta equation.
     *
     * <pre>
     * Find x such that I_x(a, b) = p
     * </pre>
     *
     * @param a first shape parameter, must be strictly positive
     * @param b second shape parameter, must be strictly positive
     * @param p target lower-tail probability in {@code [0, 1]}
     * @return solution {@code x in [0, 1]}
     */
    public static double ibetaInv(double a, double b, double p) {
        validatePositiveParameters(a, b, "ibetaInv");
        validateProbability(p, "p");
        if (p == 0.0) {
            return 0.0;
        }
        if (p == 1.0) {
            return 1.0;
        }
        if (a == 1.0 && b == 1.0) {
            return p;
        }
        if (a == 1.0) {
            return -Math.expm1(Math.log1p(-p) / b);
        }
        if (b == 1.0) {
            return Math.exp(Math.log(p) / a);
        }
        if (a == 0.5 && b == 0.5) {
            double s = Math.sin(0.5 * Math.PI * p);
            return s * s;
        }
        return ibetaInvImp(a, b, p, 1.0 - p, false)._1();
    }

    /**
     * Solves for {@code x} in the regularized lower incomplete beta equation and returns
     * the pair {@code (x, y)} with {@code y = 1 - x}.
     *
     * @param a first shape parameter, must be strictly positive
     * @param b second shape parameter, must be strictly positive
     * @param p target lower-tail probability in {@code [0, 1]}
     * @return {@link Double2} containing {@code (x, 1 - x)}
     */
    public static Double2 ibetaInvComplement(double a, double b, double p) {
        validatePositiveParameters(a, b, "ibetaInv");
        validateProbability(p, "p");
        if (p == 0.0) {
            return new Double2(0.0, 1.0);
        }
        if (p == 1.0) {
            return new Double2(1.0, 0.0);
        }
        if (a == 1.0 && b == 1.0) {
            return new Double2(p, 1.0 - p);
        }
        if (a == 1.0) {
            double logY = Math.log1p(-p) / b;
            double y = Math.exp(logY);
            return new Double2(-Math.expm1(logY), y);
        }
        if (b == 1.0) {
            double logX = Math.log(p) / a;
            double x = Math.exp(logX);
            return new Double2(x, -Math.expm1(logX));
        }
        if (a == 0.5 && b == 0.5) {
            double angle = 0.5 * Math.PI * p;
            double s = Math.sin(angle);
            double c = Math.cos(angle);
            return new Double2(s * s, c * c);
        }
        return ibetaInvImp(a, b, p, 1.0 - p, true);
    }

    /**
     * Solves for {@code x} in the regularized upper incomplete beta complement equation.
     *
     * <pre>
     * Find x such that Ī_x(a, b) = q
     * </pre>
     *
     * @param a first shape parameter, must be strictly positive
     * @param b second shape parameter, must be strictly positive
     * @param q target upper-tail probability in {@code [0, 1]}
     * @return solution {@code x in [0, 1]}
     */
    public static double ibetacInv(double a, double b, double q) {
        validatePositiveParameters(a, b, "ibetacInv");
        validateProbability(q, "q");
        if (q == 0.0) {
            return 1.0;
        }
        if (q == 1.0) {
            return 0.0;
        }
        return ibetaInvImp(a, b, 1.0 - q, q, false)._1();
    }

    /**
     * Solves for {@code x} in the regularized upper incomplete beta complement equation and returns
     * the pair {@code (x, y)} with {@code y = 1 - x}.
     *
     * @param a first shape parameter, must be strictly positive
     * @param b second shape parameter, must be strictly positive
     * @param q target upper-tail probability in {@code [0, 1]}
     * @return {@link Double2} containing {@code (x, 1 - x)}
     */
    public static Double2 ibetacInvComplement(double a, double b, double q) {
        validatePositiveParameters(a, b, "ibetacInv");
        validateProbability(q, "q");
        if (q == 0.0) {
            return new Double2(1.0, 0.0);
        }
        if (q == 1.0) {
            return new Double2(0.0, 1.0);
        }
        return ibetaInvImp(a, b, 1.0 - q, q, true);
    }

    /**
     * Solves for {@code a} in the regularized lower incomplete beta equation.
     *
     * <pre>
     * Find a such that I_x(a, b) = p
     * </pre>
     *
     * @param b second shape parameter, must be strictly positive
     * @param x split point in {@code (0, 1)}
     * @param p target lower-tail probability in {@code [0, 1]}
     * @return shape parameter {@code a}
     */
    public static double ibetaInva(double b, double x, double p) {
        validatePositiveParameter(b, "b", "ibetaInva");
        validateOpenUnitInterval(x, "x");
        validateProbability(p, "p");
        if (p == 0.0) {
            throw new ArithmeticException("ibetaInva overflow: p must be greater than zero");
        }
        if (p == 1.0) {
            return MIN_PARAMETER;
        }
        return ibetaInvAbImp(b, x, p, 1.0 - p, false, "ibetaInva");
    }

    /**
     * Solves for {@code a} in the regularized upper incomplete beta complement equation.
     *
     * <pre>
     * Find a such that Ī_x(a, b) = q
     * </pre>
     *
     * @param b second shape parameter, must be strictly positive
     * @param x split point in {@code (0, 1)}
     * @param q target upper-tail probability in {@code [0, 1]}
     * @return shape parameter {@code a}
     */
    public static double ibetacInva(double b, double x, double q) {
        validatePositiveParameter(b, "b", "ibetacInva");
        validateOpenUnitInterval(x, "x");
        validateProbability(q, "q");
        if (q == 0.0) {
            return MIN_PARAMETER;
        }
        if (q == 1.0) {
            throw new ArithmeticException("ibetacInva overflow: q must be less than one");
        }
        return ibetaInvAbImp(b, x, 1.0 - q, q, false, "ibetacInva");
    }

    /**
     * Solves for {@code b} in the regularized lower incomplete beta equation.
     *
     * <pre>
     * Find b such that I_x(a, b) = p
     * </pre>
     *
     * @param a first shape parameter, must be strictly positive
     * @param x split point in {@code (0, 1)}
     * @param p target lower-tail probability in {@code [0, 1]}
     * @return shape parameter {@code b}
     */
    public static double ibetaInvb(double a, double x, double p) {
        validatePositiveParameter(a, "a", "ibetaInvb");
        validateOpenUnitInterval(x, "x");
        validateProbability(p, "p");
        if (p == 0.0) {
            return MIN_PARAMETER;
        }
        if (p == 1.0) {
            throw new ArithmeticException("ibetaInvb overflow: p must be less than one");
        }
        return ibetaInvAbImp(a, x, p, 1.0 - p, true, "ibetaInvb");
    }

    /**
     * Solves for {@code b} in the regularized upper incomplete beta complement equation.
     *
     * <pre>
     * Find b such that Ī_x(a, b) = q
     * </pre>
     *
     * @param a first shape parameter, must be strictly positive
     * @param x split point in {@code (0, 1)}
     * @param q target upper-tail probability in {@code [0, 1]}
     * @return shape parameter {@code b}
     */
    public static double ibetacInvb(double a, double x, double q) {
        validatePositiveParameter(a, "a", "ibetacInvb");
        validateOpenUnitInterval(x, "x");
        validateProbability(q, "q");
        if (q == 0.0) {
            throw new ArithmeticException("ibetacInvb overflow: q must be greater than zero");
        }
        if (q == 1.0) {
            return MIN_PARAMETER;
        }
        return ibetaInvAbImp(a, x, 1.0 - q, q, true, "ibetacInvb");
    }

    private static double completeBeta(double a, double b) {
        double c = a + b;
        if ((c == a) && (b < EPSILON)) {
            return 1.0 / b;
        }
        if ((c == b) && (a < EPSILON)) {
            return 1.0 / a;
        }
        if (b == 1.0) {
            return 1.0 / a;
        }
        if (a == 1.0) {
            return 1.0 / b;
        }
        if (c < EPSILON) {
            return c / a / b;
        }

        if (a < b) {
            double tmp = a;
            a = b;
            b = tmp;
        }

        double g = g();
        double agh = a + g - 0.5;
        double bgh = b + g - 0.5;
        double cgh = c + g - 0.5;
        double result = lanczosSumExpGScaled(a)
            * (lanczosSumExpGScaled(b) / lanczosSumExpGScaled(c));
        double ambh = a - 0.5 - b;
        if ((Math.abs(b * ambh) < (cgh * 100.0)) && (a > 100.0)) {
            result *= Math.exp(ambh * Math.log1p(-b / cgh));
        } else {
            result *= Math.pow(agh / cgh, ambh);
        }
        if (cgh > 1.0e10) {
            result *= Math.pow((agh / cgh) * (bgh / cgh), b);
        } else {
            result *= Math.pow((agh * bgh) / (cgh * cgh), b);
        }
        result *= Math.sqrt(Math.E / bgh);
        return result;
    }

    private static double ibetaImp(double a, double b, double x, boolean inv, boolean normalised) {
        return ibetaImpWithDerivative(a, b, x, inv, normalised, false)._1();
    }

    private static Double2 ibetaImpWithDerivative(double a,
                                                  double b,
                                                  double x,
                                                  boolean inv,
                                                  boolean normalised) {
        return ibetaImpWithDerivative(a, b, x, inv, normalised, true);
    }

    private static Double2 ibetaImpWithDerivative(double a,
                                                  double b,
                                                  double x,
                                                  boolean inv,
                                                  boolean normalised,
                                                  boolean withDerivative) {
        boolean invert = inv;
        double y = 1.0 - x;

        if (normalised) {
            if (a == 0.0) {
                if (b == 0.0) {
                    throw new IllegalArgumentException("ibeta: a and b must not both be zero");
                }
                return betaResult(invert ? 0.0 : 1.0, 0.0, withDerivative);
            }
            if (b == 0.0) {
                return betaResult(invert ? 1.0 : 0.0, 0.0, withDerivative);
            }
        }

        if (x == 0.0) {
            return betaResult(invert ? wholeValue(normalised, a, b) : 0.0,
                normalised ? ibetaDerivativeImp(a, b, x) : Double.NaN,
                withDerivative);
        }
        if (x == 1.0) {
            return betaResult(invert ? 0.0 : wholeValue(normalised, a, b),
                normalised ? ibetaDerivativeImp(a, b, x) : Double.NaN,
                withDerivative);
        }
        if ((a == 0.5) && (b == 0.5)) {
            double p = invert ? Math.asin(Math.sqrt(y)) / HALF_PI : Math.asin(Math.sqrt(x)) / HALF_PI;
            return betaResult(normalised ? p : p * Math.PI,
                normalised ? ibetaDerivativeImp(a, b, x) : Double.NaN,
                withDerivative);
        }
        if (a == 1.0) {
            double tmp = a;
            a = b;
            b = tmp;
            tmp = x;
            x = y;
            y = tmp;
            invert = !invert;
        }
        if (b == 1.0) {
            if (a == 1.0) {
                return betaResult(invert ? y : x,
                    normalised ? ibetaDerivativeImp(a, b, x) : Double.NaN,
                    withDerivative);
            }
            double p;
            if (y < 0.5) {
                p = invert ? -Math.expm1(a * Math.log1p(-y)) : Math.exp(a * Math.log1p(-y));
            } else {
                p = invert ? -powm1(x, a) : Math.pow(x, a);
            }
            return betaResult(normalised ? p : p / a,
                normalised ? ibetaDerivativeImp(a, b, x) : Double.NaN,
                withDerivative);
        }

        double fract;
        if (Math.min(a, b) <= 1.0) {
            if (x > 0.5) {
                double tmp = a;
                a = b;
                b = tmp;
                tmp = x;
                x = y;
                y = tmp;
                invert = !invert;
            }
            if (Math.max(a, b) <= 1.0) {
                if ((a >= Math.min(0.2, b)) || (Math.pow(x, a) <= 0.9)) {
                    fract = evaluateSeriesBranch(a, b, x, y, invert, normalised);
                    invert = false;
                } else {
                    double tmp = a;
                    a = b;
                    b = tmp;
                    tmp = x;
                    x = y;
                    y = tmp;
                    invert = !invert;
                    if (y >= 0.3) {
                        fract = evaluateSeriesBranch(a, b, x, y, invert, normalised);
                        invert = false;
                    } else {
                        double prefix = normalised ? 1.0 : risingFactorialRatio(a + b, a, 20);
                        fract = ibetaAStep(a, b, x, y, 20, normalised);
                        if (!invert) {
                            fract = betaSmallBLargeASeries(a + 20.0, b, x, y, fract, prefix, normalised);
                        } else {
                            fract -= wholeValue(normalised, a, b);
                            fract = -betaSmallBLargeASeries(a + 20.0, b, x, y, fract, prefix, normalised);
                        }
                        invert = false;
                    }
                }
            } else {
                if ((b <= 1.0) || ((x < 0.1) && (Math.pow(b * x, a) <= 0.7))) {
                    fract = evaluateSeriesBranch(a, b, x, y, invert, normalised);
                    invert = false;
                } else {
                    double tmp = a;
                    a = b;
                    b = tmp;
                    tmp = x;
                    x = y;
                    y = tmp;
                    invert = !invert;
                    if (y >= 0.3) {
                        fract = evaluateSeriesBranch(a, b, x, y, invert, normalised);
                    } else if (a >= 15.0) {
                        fract = !invert
                            ? betaSmallBLargeASeries(a, b, x, y, 0.0, 1.0, normalised)
                            : -betaSmallBLargeASeries(a, b, x, y, -wholeValue(normalised, a, b), 1.0, normalised);
                    } else {
                        double prefix = normalised ? 1.0 : risingFactorialRatio(a + b, a, 20);
                        fract = ibetaAStep(a, b, x, y, 20, normalised);
                        if (!invert) {
                            fract = betaSmallBLargeASeries(a + 20.0, b, x, y, fract, prefix, normalised);
                        } else {
                            fract -= wholeValue(normalised, a, b);
                            fract = -betaSmallBLargeASeries(a + 20.0, b, x, y, fract, prefix, normalised);
                        }
                    }
                    invert = false;
                }
            }
        } else {
            double lambda = a < b ? a - (a + b) * x : (a + b) * y - b;
            if (lambda < 0.0) {
                double tmp = a;
                a = b;
                b = tmp;
                tmp = x;
                x = y;
                y = tmp;
                invert = !invert;
            }

            if (b < 40.0) {
                if ((Math.floor(a) == a) && (Math.floor(b) == b) && (a < (Integer.MAX_VALUE - 100)) && (y != 1.0)) {
                    double k = a - 1.0;
                    double n = b + k;
                    fract = binomialCcdf(n, k, x, y);
                    if (!normalised) {
                        fract *= completeBeta(a, b);
                    }
                } else if (b * x <= 0.7) {
                    fract = evaluateSeriesBranch(a, b, x, y, invert, normalised);
                    invert = false;
                } else if (a > 15.0) {
                    int n = (int) Math.floor(b);
                    if (n == b) {
                        --n;
                    }
                    double bBar = b - n;
                    double prefix = normalised ? 1.0 : risingFactorialRatio(a + bBar, bBar, n);
                    fract = ibetaAStep(bBar, a, y, x, n, normalised);
                    fract = betaSmallBLargeASeries(a, bBar, x, y, fract, 1.0, normalised);
                    fract /= prefix;
                } else if (normalised) {
                    int n = (int) Math.floor(b);
                    double bBar = b - n;
                    if (bBar <= 0.0) {
                        --n;
                        bBar += 1.0;
                    }
                    fract = ibetaAStep(bBar, a, y, x, n, true);
                    fract += ibetaAStep(a, bBar, x, y, 20, true);
                    if (invert) {
                        fract -= 1.0;
                    }
                    fract = betaSmallBLargeASeries(a + 20.0, bBar, x, y, fract, 1.0, true);
                    if (invert) {
                        fract = -fract;
                        invert = false;
                    }
                } else {
                    fract = ibetaFraction2(a, b, x, y, false);
                }
            } else {
                boolean useAsym = false;
                double ma = Math.max(a, b);
                double xa = ma == a ? x : y;
                double saddle = ma / (a + b);
                double powers = 0.0;
                if ((ma > 1.0e-5 / EPSILON) && (ma / Math.min(a, b) < (xa < saddle ? 2.0 : 15.0))) {
                    if (a == b) {
                        useAsym = true;
                    } else {
                        powers = Math.exp(Math.log(x / (a / (a + b))) * a + Math.log(y / (b / (a + b))) * b);
                        if (powers < EPSILON) {
                            useAsym = true;
                        }
                    }
                }
                if (useAsym) {
                    fract = ibetaLargeAB(a, b, x, y, invert, normalised);
                    if (fract * EPSILON < powers) {
                        fract = ibetaFraction2(a, b, x, y, normalised);
                    } else {
                        invert = false;
                    }
                } else {
                    fract = ibetaFraction2(a, b, x, y, normalised);
                }
            }
        }

        return betaResult(invert ? wholeValue(normalised, a, b) - fract : fract,
            normalised ? ibetaDerivativeFromTransformedInputs(a, b, x, y) : Double.NaN,
            withDerivative);
    }

    private static double evaluateSeriesBranch(double a,
                                               double b,
                                               double x,
                                               double y,
                                               boolean invert,
                                               boolean normalised) {
        if (!invert) {
            return ibetaSeries(a, b, x, 0.0, normalised, y);
        }
        double whole = wholeValue(normalised, a, b);
        return -ibetaSeries(a, b, x, -whole, normalised, y);
    }

    private static double ibetaDerivativeImp(double a, double b, double x) {
        if (x == 0.0) {
            if (a > 1.0) {
                return 0.0;
            }
            if (a == 1.0) {
                return 1.0 / completeBeta(a, b);
            }
            return Double.MAX_VALUE / 2.0;
        }
        if (x == 1.0) {
            if (b > 1.0) {
                return 0.0;
            }
            if (b == 1.0) {
                return 1.0 / completeBeta(a, b);
            }
            return Double.MAX_VALUE / 2.0;
        }

        return ibetaDerivativeFromTransformedInputs(a, b, x, 1.0 - x);
    }

    private static double wholeValue(boolean normalised, double a, double b) {
        return normalised ? 1.0 : completeBeta(a, b);
    }

    private static Double2 betaResult(double value, double derivative, boolean withDerivative) {
        return new Double2(value, withDerivative ? derivative : Double.NaN);
    }

    private static double ibetaDerivativeFromTransformedInputs(double a, double b, double x, double y) {
        double derivative = ibetaPowerTerms(a, b, x, y, true);
        double divisor = x * y;
        if (derivative != 0.0) {
            if (Double.MAX_VALUE * divisor < derivative) {
                return Double.MAX_VALUE / 2.0;
            }
            derivative /= divisor;
        }
        return derivative;
    }

    private static double ibetaPowerTerms(double a, double b, double x, double y, boolean normalised) {
        return ibetaPowerTerms(a, b, x, y, normalised, 1.0);
    }

    private static double ibetaPowerTerms(double a,
                                          double b,
                                          double x,
                                          double y,
                                          boolean normalised,
                                          double prefix) {
        if (!normalised) {
            return prefix * Math.pow(x, a) * Math.pow(y, b);
        }

        double c = a + b;
        double gh = g() - 0.5;
        double agh = a + gh;
        double bgh = b + gh;
        double cgh = c + gh;
        double result;
        if ((a < Double.MIN_NORMAL) || (b < Double.MIN_NORMAL)) {
            result = 0.0;
        } else {
            result = lanczosSumExpGScaled(c)
                / (lanczosSumExpGScaled(a) * lanczosSumExpGScaled(b));
        }
        result *= prefix;
        result *= Math.sqrt(bgh / Math.E);
        result *= Math.sqrt(agh / cgh);

        double l1 = ((x * b - y * a) - y * gh) / agh;
        double l2 = ((y * a - x * b) - x * gh) / bgh;
        if (Math.min(Math.abs(l1), Math.abs(l2)) < 0.2) {
            if ((l1 * l2 > 0.0) || (Math.min(a, b) < 1.0)) {
                if (Math.abs(l1) < 0.1) {
                    result *= Math.exp(a * Math.log1p(l1));
                } else {
                    result *= Math.pow((x * cgh) / agh, a);
                }
                if (Math.abs(l2) < 0.1) {
                    result *= Math.exp(b * Math.log1p(l2));
                } else {
                    result *= Math.pow((y * cgh) / bgh, b);
                }
            } else if (Math.max(Math.abs(l1), Math.abs(l2)) < 0.5) {
                boolean smallA = a < b;
                double ratio = b / a;
                if ((smallA && (ratio * l2 < 0.1)) || (!smallA && (l1 / ratio > 0.1))) {
                    double l3 = Math.expm1(ratio * Math.log1p(l2));
                    l3 = l1 + l3 + l3 * l1;
                    result *= Math.exp(a * Math.log1p(l3));
                } else {
                    double l3 = Math.expm1(Math.log1p(l1) / ratio);
                    l3 = l2 + l3 + l3 * l2;
                    result *= Math.exp(b * Math.log1p(l3));
                }
            } else if (Math.abs(l1) < Math.abs(l2)) {
                double l = a * Math.log1p(l1) + b * Math.log((y * cgh) / bgh);
                result = multiplyExpLog(result, l);
            } else {
                double l = b * Math.log1p(l2) + a * Math.log((x * cgh) / agh);
                result = multiplyExpLog(result, l);
            }
        } else {
            double b1 = (x * cgh) / agh;
            double b2 = (y * cgh) / bgh;
            double ll1 = a * Math.log(b1);
            double ll2 = b * Math.log(b2);
            if ((ll1 >= LOG_MAX_VALUE)
                || (ll1 <= LOG_MIN_VALUE)
                || (ll2 >= LOG_MAX_VALUE)
                || (ll2 <= LOG_MIN_VALUE)) {
                if (a < b) {
                    double p1 = Math.pow(b2, b / a);
                    double l3 = (b1 != 0.0) && (p1 != 0.0)
                        ? a * (Math.log(b1) + Math.log(p1))
                        : Double.POSITIVE_INFINITY;
                    if ((l3 < LOG_MAX_VALUE) && (l3 > LOG_MIN_VALUE)) {
                        result *= Math.pow(p1 * b1, a);
                    } else {
                        result = multiplyExpLog(result, ll1 + ll2);
                    }
                } else {
                    double p1 = (b1 < 1.0) && (b < 1.0) && (Double.MAX_VALUE * b < a)
                        ? 0.0
                        : Math.pow(b1, a / b);
                    double l3 = (p1 != 0.0) && (b2 != 0.0)
                        ? (Math.log(p1) + Math.log(b2)) * b
                        : Double.POSITIVE_INFINITY;
                    if ((l3 < LOG_MAX_VALUE) && (l3 > LOG_MIN_VALUE)) {
                        result *= Math.pow(p1 * b2, b);
                    } else if (result != 0.0) {
                        result = multiplyExpLog(result, ll1 + ll2);
                    }
                }
            } else {
                result *= Math.pow(b1, a) * Math.pow(b2, b);
            }
        }

        if (result == 0.0) {
            return 0.0;
        }
        return result;
    }

    private static double multiplyExpLog(double result, double exponentLog) {
        if (result == 0.0) {
            return 0.0;
        }
        if ((exponentLog <= LOG_MIN_VALUE) || (exponentLog >= LOG_MAX_VALUE)) {
            double total = Math.log(result) + exponentLog;
            if (total <= LOG_MIN_VALUE) {
                return 0.0;
            }
            if (total >= LOG_MAX_VALUE) {
                return Double.POSITIVE_INFINITY;
            }
            return Math.exp(total);
        }
        return result * Math.exp(exponentLog);
    }

    private static double ibetaSeries(double a,
                                      double b,
                                      double x,
                                      double s0,
                                      boolean normalised,
                                      double y) {
        double result;
        if (normalised) {
            double c = a + b;
            double gh = g() - 0.5;
            double agh = a + gh;
            double bgh = b + gh;
            double cgh = c + gh;

            if ((a < Double.MIN_NORMAL) || (b < Double.MIN_NORMAL)) {
                result = 0.0;
            } else {
                double l1 = lanczosSumExpGScaled(c);
                double l2 = lanczosSumExpGScaled(a);
                double l3 = lanczosSumExpGScaled(b);
                result = ((l2 > 1.0) && (l3 > 1.0) && (Double.MAX_VALUE / l2 < l3))
                    ? (l1 / l2) / l3
                    : l1 / (l2 * l3);
            }
            if (!Double.isFinite(result)) {
                result = 0.0;
            }

            double logTerm1 = Math.log(cgh / bgh) * (b - 0.5);
            double logTerm2 = Math.log(x * cgh / agh) * a;
            if ((logTerm1 > LOG_MIN_VALUE)
                && (logTerm1 < LOG_MAX_VALUE)
                && (logTerm2 > LOG_MIN_VALUE)
                && (logTerm2 < LOG_MAX_VALUE)) {
                if (a * b < bgh * 10.0) {
                    result *= Math.exp((b - 0.5) * Math.log1p(a / bgh));
                } else {
                    result *= Math.pow(cgh / bgh, b - 0.5);
                }
                result *= Math.pow(x * cgh / agh, a);
                result *= Math.sqrt(agh / Math.E);
            } else if (result != 0.0) {
                result = Math.exp(Math.log(result) + logTerm1 + logTerm2 + (Math.log(agh) - 1.0) / 2.0);
            }
        } else {
            result = Math.pow(x, a);
        }

        if (result < Double.MIN_NORMAL) {
            return s0;
        }

        double sum = s0;
        double term = result;
        double apn = a;
        double poch = 1.0 - b;
        for (int n = 1; n <= MAX_ITERATIONS * 8; n++) {
            double r = term / apn;
            sum += r;
            if (Math.abs(r) <= Math.abs(sum) * SERIES_EPSILON) {
                return sum;
            }
            apn += 1.0;
            term *= poch * x / n;
            poch += 1.0;
        }
        throw new IllegalStateException("ibeta series did not converge");
    }

    private static double ibetaFraction2(double a, double b, double x, double y, boolean normalised) {
        double result = ibetaPowerTerms(a, b, x, y, normalised);
        if (result == 0.0) {
            return 0.0;
        }
        ContinuedFraction fraction = ContinuedFraction.evaluateFractionB(
            i -> ibetaFraction(a, b, x, y, i),
            CONTINUED_FRACTION_EPSILON,
            MAX_ITERATIONS * 4);
        if (!fraction.converged()) {
            throw new IllegalStateException("Continued fraction did not converge for ibeta fraction2");
        }
        return result / fraction.value();
    }

    private static double ibetaAStep(double a, double b, double x, double y, int k, boolean normalised) {
        double prefix = ibetaPowerTerms(a, b, x, y, normalised);
        prefix /= a;
        if (prefix == 0.0) {
            return 0.0;
        }
        double sum = 1.0;
        double term = 1.0;
        for (int i = 0; i < k - 1; i++) {
            term *= (a + b + i) * x / (a + i + 1.0);
            sum += term;
        }
        return prefix * sum;
    }

    private static double betaSmallBLargeASeries(double a,
                                                 final double b,
                                                 double x,
                                                 double y,
                                                 double s0,
                                                 double mult,
                                                 boolean normalised) {
        double bm1 = b - 1.0;
        double t = a + bm1 / 2.0;
        double lx = y < 0.35 ? Math.log1p(-y) : Math.log(x);
        double u = -t * lx;
        double h = Gamma.regularizedIncompletePrefix(b, u);
        if (h <= Double.MIN_NORMAL) {
            return s0;
        }
        double prefix = normalised
            ? h / Gamma.tgammaDeltaRatio(a, b) / Math.pow(t, b)
            : Gamma.fullIncompletePrefix(b, u) / Math.pow(t, b);
        prefix *= mult;

        double j = Gamma.gammaQ(b, u) / h;
        double sum = s0 + prefix * j;
        int tnp1 = 1;
        double lx2 = lx / 2.0;
        lx2 *= lx2;
        double lxp = 1.0;
        double t4 = 4.0 * t * t;
        double b2n = b;

        double p0 = 1.0, p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0;
        double p5 = 0.0, p6 = 0.0, p7 = 0.0, p8 = 0.0, p9 = 0.0;
        double p10 = 0.0, p11 = 0.0, p12 = 0.0, p13 = 0.0, p14 = 0.0;
        double p15 = 0.0, p16 = 0.0, p17 = 0.0, p18 = 0.0, p19 = 0.0;
        double p20 = 0.0, p21 = 0.0, p22 = 0.0, p23 = 0.0, p24 = 0.0;
        double p25 = 0.0, p26 = 0.0, p27 = 0.0, p28 = 0.0, p29 = 0.0;
        for (int n = 1; n < PN_SIZE; n++) {
            tnp1 += 2;
            double pn = 0.0;
            int tmp1 = 3;
            for (int m = 1; m < n; m++) {
                double mbn = m * b - n;
                double previous = switch (n - m) {
                    case 0 -> p0; case 1 -> p1; case 2 -> p2; case 3 -> p3; case 4 -> p4;
                    case 5 -> p5; case 6 -> p6; case 7 -> p7; case 8 -> p8; case 9 -> p9;
                    case 10 -> p10; case 11 -> p11; case 12 -> p12; case 13 -> p13; case 14 -> p14;
                    case 15 -> p15; case 16 -> p16; case 17 -> p17; case 18 -> p18; case 19 -> p19;
                    case 20 -> p20; case 21 -> p21; case 22 -> p22; case 23 -> p23; case 24 -> p24;
                    case 25 -> p25; case 26 -> p26; case 27 -> p27; case 28 -> p28; case 29 -> p29;
                    default -> throw new AssertionError("Unexpected PN history index: " + (n - m));
                };
                pn += mbn * previous / uncheckedFactorial(tmp1);
                tmp1 += 2;
            }
            pn /= n;
            pn += bm1 / uncheckedFactorial(tnp1);
            switch (n) {
                case 1 -> p1 = pn; case 2 -> p2 = pn; case 3 -> p3 = pn; case 4 -> p4 = pn; case 5 -> p5 = pn;
                case 6 -> p6 = pn; case 7 -> p7 = pn; case 8 -> p8 = pn; case 9 -> p9 = pn; case 10 -> p10 = pn;
                case 11 -> p11 = pn; case 12 -> p12 = pn; case 13 -> p13 = pn; case 14 -> p14 = pn; case 15 -> p15 = pn;
                case 16 -> p16 = pn; case 17 -> p17 = pn; case 18 -> p18 = pn; case 19 -> p19 = pn; case 20 -> p20 = pn;
                case 21 -> p21 = pn; case 22 -> p22 = pn; case 23 -> p23 = pn; case 24 -> p24 = pn; case 25 -> p25 = pn;
                case 26 -> p26 = pn; case 27 -> p27 = pn; case 28 -> p28 = pn; case 29 -> p29 = pn;
                default -> throw new AssertionError("Unexpected PN slot index: " + n);
            }

            j = (b2n * (b2n + 1.0) * j + (u + b2n + 1.0) * lxp) / t4;
            lxp *= lx2;
            b2n += 2.0;

            double r = prefix * pn * j;
            sum += r;
            if (Math.abs(r / EPSILON) < Math.abs(sum)) {
                break;
            }
        }
        return sum;
    }

    private static double binomialCcdf(double n, double k, double x, double y) {
        int nInt = (int) Math.rint(n);
        int kInt = (int) Math.rint(k);
        double result = Math.pow(x, n);

        if (result > Double.MIN_NORMAL) {
            double term = result;
            for (int i = nInt - 1; i > kInt; --i) {
                term *= ((i + 1.0) * y) / ((nInt - i) * x);
                result += term;
            }
        } else {
            int start = (int) (n * x);
            if (start <= kInt + 1) {
                start = kInt + 2;
            }
            result = Math.exp(logBinomialCoefficient(nInt, start) + start * Math.log(x) + (nInt - start) * Math.log(y));
            if (result == 0.0) {
                for (int i = start - 1; i > kInt; --i) {
                    result += Math.exp(logBinomialCoefficient(nInt, i) + i * Math.log(x) + (nInt - i) * Math.log(y));
                }
            } else {
                double term = result;
                double startTerm = result;
                for (int i = start - 1; i > kInt; --i) {
                    term *= ((i + 1.0) * y) / ((nInt - i) * x);
                    result += term;
                }
                term = startTerm;
                for (int i = start + 1; i <= nInt; ++i) {
                    term *= (nInt - i + 1.0) * x / (i * y);
                    result += term;
                }
            }
        }

        return result;
    }

    private static double ibetaLargeAB(double a,
                                       double b,
                                       double x,
                                       double y,
                                       boolean invert,
                                       boolean normalised) {
        double x0 = a / (a + b);
        double y0 = b / (a + b);
        double nu = x0 * Math.log(x / x0) + y0 * Math.log(y / y0);
        if ((nu > 0.0) || (x == x0) || (y == y0)) {
            nu = 0.0;
        }
        nu = Math.sqrt(-2.0 * nu);
        if ((nu != 0.0) && (nu / (x - x0) < 0.0)) {
            nu = -nu;
        }
        double multiplier = normalised ? 1.0 : completeBeta(a, b);
        double erfArg = -nu * Math.sqrt((a + b) / 2.0);
        return multiplier * (invert ? (1.0 + Normal.erf(erfArg)) / 2.0 : Normal.erfc(erfArg) / 2.0);
    }

    private static Double2 ibetaInvImp(double a, double b, double p, double q, boolean withComplement) {
        boolean invert = false;
        if (q == 0.0) {
            return new Double2(1.0, 0.0);
        }
        if (p == 0.0) {
            return new Double2(0.0, 1.0);
        }
        if (a == 1.0) {
            if (b == 1.0) {
                return new Double2(p, 1.0 - p);
            }
            double tmp = a;
            a = b;
            b = tmp;
            tmp = p;
            p = q;
            q = tmp;
            invert = true;
        }

        double x = 0.0;
        double y = Double.NaN;
        double lower = 0.0;
        double upper = 1.0;

        if (a == 0.5) {
            if (b == 0.5) {
                x = Math.sin(p * HALF_PI);
                x *= x;
                y = Math.sin(q * HALF_PI);
                y *= y;
                return new Double2(x, y);
            }
            if (b > 0.5) {
                double tmp = a;
                a = b;
                b = tmp;
                tmp = p;
                p = q;
                q = tmp;
                invert = !invert;
            }
        }

        if ((b == 0.5) && (a >= 0.5) && (p != 1.0)) {
            Double2 tDistInitializer = findIbetaInvFromTDist(a, p);
            x = tDistInitializer._1();
            y = tDistInitializer._2();
        } else if (b == 1.0) {
            if (p < q) {
                x = Math.pow(p, 1.0 / a);
                y = a > 1.0 ? -Math.expm1(Math.log(p) / a) : 1.0 - x;
            } else {
                x = Math.exp(Math.log1p(-q) / a);
                y = -Math.expm1(Math.log1p(-q) / a);
            }
            if (invert) {
                double tmp = x;
                x = y;
                y = tmp;
            }
            return new Double2(x, y);
        } else if (a + b > 5.0) {
            if (p > 0.5) {
                double tmp = a;
                a = b;
                b = tmp;
                tmp = p;
                p = q;
                q = tmp;
                invert = !invert;
            }
            double minv = Math.min(a, b);
            double maxv = Math.max(a, b);
            if ((Math.sqrt(minv) > (maxv - minv)) && (minv > 5.0)) {
                x = temmeMethod1IbetaInverse(a, b, p);
                y = 1.0 - x;
            } else {
                double r = a + b;
                double theta = Math.asin(Math.sqrt(a / r));
                double lambda = minv / r;
                if ((lambda >= 0.2) && (lambda <= 0.8) && (r >= 10.0)) {
                    double ppa = Math.pow(p, 1.0 / a);
                    if ((ppa < 0.0025) && (a + b < 200.0)) {
                        x = ppa * Math.pow(a * completeBeta(a, b), 1.0 / a);
                    } else {
                        x = temmeMethod2IbetaInverse(a, b, p, r, theta);
                    }
                    y = 1.0 - x;
                } else {
                    if (a < b) {
                        double tmp = a;
                        a = b;
                        b = tmp;
                        tmp = p;
                        p = q;
                        q = tmp;
                        invert = !invert;
                    }
                    double bet = b < 2.0 ? completeBeta(a, b) : 0.0;
                    if (bet != 0.0 && Double.isFinite(bet)) {
                        y = Math.pow(b * q * bet, 1.0 / b);
                        x = 1.0 - y;
                    } else {
                        y = 1.0;
                    }
                    if ((y > 1.0e-5) && (Math.min(a, b) < 1000.0)) {
                        x = temmeMethod3IbetaInverse(a, b, p, q);
                        y = 1.0 - x;
                    } else if ((y > 1.0e-5) && (Math.min(a, b) > 1000.0)) {
                        x = Math.max(a, b) / (a + b);
                        y = Math.min(a, b) / (a + b);
                    }
                }
            }
        } else if ((a < 1.0) && (b < 1.0)) {
            double xs = (1.0 - a) / (2.0 - a - b);
            double fs = ibetaImp(a, b, xs, false, true) - p;
            if (Math.abs(fs) / p < EPSILON * 3.0) {
                return invert ? new Double2(1.0 - xs, xs) : new Double2(xs, 1.0 - xs);
            }
            if (fs < 0.0) {
                double tmp = a;
                a = b;
                b = tmp;
                tmp = p;
                p = q;
                q = tmp;
                invert = !invert;
                xs = 1.0 - xs;
            }
            if ((a < Double.MIN_NORMAL) && (b > Double.MIN_NORMAL)) {
                return invert ? new Double2(1.0, 0.0) : new Double2(0.0, 1.0);
            }
            double bet = completeBeta(a, b);
            double xg;
            if (!Double.isFinite(bet)) {
                xg = Math.exp((Gamma.lgamma(a + 1.0) + Gamma.lgamma(b) - Gamma.lgamma(a + b) + Math.log(p)) / a);
                if (xg > 2.0 / EPSILON) {
                    xg = 2.0 / EPSILON;
                }
            } else {
                xg = Math.pow(a * p * bet, 1.0 / a);
            }
            x = xg / (1.0 + xg);
            y = 1.0 / (1.0 + xg);
            if (x > xs) {
                x = xs;
            }
            upper = xs;
        } else if ((a > 1.0) && (b > 1.0)) {
            double xs = (a - 1.0) / (a + b - 2.0);
            double xs2 = (b - 1.0) / (a + b - 2.0);
            double ps = ibetaImp(a, b, xs, false, true) - p;
            if (ps < 0.0) {
                double tmp = a;
                a = b;
                b = tmp;
                tmp = p;
                p = q;
                q = tmp;
                tmp = xs;
                xs = xs2;
                xs2 = tmp;
                invert = !invert;
            }
            double lx = Math.log(p * a * completeBeta(a, b)) / a;
            x = Math.exp(lx);
            y = x < 0.9 ? 1.0 - x : -Math.expm1(lx);
            if ((b < a) && (x < 0.2)) {
                double ap1 = a - 1.0;
                double bm1 = b - 1.0;
                double a2 = a * a;
                double a3 = a2 * a;
                double b2 = b * b;
                double[] terms = new double[5];
                terms[0] = 0.0;
                terms[1] = 1.0;
                terms[2] = bm1 / ap1;
                ap1 *= ap1;
                terms[3] = bm1 * (3.0 * a * b + 5.0 * b + a2 - a - 4.0) / (2.0 * (a + 2.0) * ap1);
                ap1 *= (a + 1.0);
                terms[4] = bm1 * (33.0 * a * b2 + 31.0 * b2 + 8.0 * a2 * b2 - 30.0 * a * b - 47.0 * b
                    + 11.0 * a2 * b + 6.0 * a3 * b + 18.0 + 4.0 * a - a3 + a2 * a2 - 10.0 * a2)
                    / (3.0 * (a + 3.0) * (a + 2.0) * ap1);
                x = evaluatePolynomial(terms, x);
            }
            if (x > xs) {
                x = xs;
            }
            upper = xs;
        } else {
            if (b < a) {
                double tmp = a;
                a = b;
                b = tmp;
                tmp = p;
                p = q;
                q = tmp;
                invert = !invert;
            }
            if (a < Double.MIN_NORMAL) {
                if (p < 1.0) {
                    x = 1.0;
                    y = 0.0;
                } else {
                    x = 0.0;
                    y = 1.0;
                }
            } else if (Math.pow(p, 1.0 / a) < 0.5) {
                x = Math.pow(p * a * completeBeta(a, b), 1.0 / a);
                if ((x > 1.0) || !Double.isFinite(x)) {
                    x = 1.0;
                }
                if (x == 0.0) {
                    x = Double.MIN_NORMAL;
                }
                y = 1.0 - x;
            } else {
                y = Math.pow(1.0 - Math.pow(p, b * completeBeta(a, b)), 1.0 / b);
                if ((y > 1.0) || !Double.isFinite(y)) {
                    y = 1.0;
                }
                if (y == 0.0) {
                    y = Double.MIN_NORMAL;
                }
                x = 1.0 - y;
            }
        }

        if (x > 0.5) {
            double tmp = a;
            a = b;
            b = tmp;
            tmp = p;
            p = q;
            q = tmp;
            tmp = x;
            x = y;
            y = tmp;
            invert = !invert;
            double l = 1.0 - upper;
            double u = 1.0 - lower;
            lower = l;
            upper = u;
        }
        if (lower == 0.0) {
            lower = (invert && !withComplement) ? EPSILON : Double.MIN_NORMAL;
            if (x < lower) {
                x = lower;
            }
        }

        boolean rootInvert = p >= q;
        double target = rootInvert ? q : p;
        double rootA = a;
        double rootB = b;
        int digits = DOUBLE_PRECISION_BITS / 2;
        if ((x < 1.0e-50) && ((a < 1.0) || (b < 1.0))) {
            digits = (digits * 3) / 2;
        }
        x = RootIterators.halleyIterate((root, evaluation) -> ibetaRoot(rootA, rootB, target, rootInvert, root, evaluation),
            x,
            lower,
            upper,
            digits,
            MAX_ITERATIONS,
            "ibetaInv"
        );
        if (x == lower) {
            x = 0.0;
        }
        return invert ? new Double2(1.0 - x, x) : new Double2(x, 1.0 - x);
    }

    private static double ibetaInvAbImp(double b,
                                        double z,
                                        double p,
                                        double q,
                                        boolean swapAb,
                                        String functionName) {
        double target = Math.min(p, q);
        boolean invert = p >= q;
        double guess = 0.0;
        double factor = 5.0;
        double n = b;
        double sf = swapAb ? z : 1.0 - z;
        double sfc = swapAb ? 1.0 - z : z;
        double u = swapAb ? p : q;
        double v = swapAb ? q : p;
        if (u <= Math.pow(sf, n)) {
            guess = ((p < q) != swapAb) ? Math.min(b * 2.0, 1.0) : Math.min(b / 2.0, 1.0);
        }
        if (n * n * n * u * sf > 0.005) {
            guess = 1.0 + inverseNegativeBinomialCornishFisher(n, sf, sfc, u, v);
        }
        if (guess < 10.0) {
            guess = ((p < q) != swapAb) ? Math.min(b * 2.0, 10.0) : Math.min(b / 2.0, 10.0);
        } else {
            factor = (v < Math.sqrt(EPSILON)) ? 2.0 : (guess < 20.0 ? 1.2 : 1.1);
        }
        return RootIterators.toms748SolveBounded(
            alpha -> betaInvAbValue(b, z, target, invert, swapAb, alpha),
            guess,
            factor,
            swapAb,
            PARAMETER_SEARCH_FLOOR,
            DOUBLE_PRECISION_BITS,
            MAX_ITERATIONS,
            functionName
        );
    }

    private static double inverseNegativeBinomialCornishFisher(double n,
                                                               double sf,
                                                               double sfc,
                                                               double p,
                                                               double q) {
        double m = n * sfc / sf;
        double t = Math.sqrt(n * sfc);
        double sigma = t / sf;
        double sk = (1.0 + sfc) / t;
        double kurtosis = (6.0 - sf * (5.0 + sfc)) / (n * sfc);
        double x = Normal.erfcInv((p > q ? 2.0 * q : 2.0 * p)) * SQRT_TWO;
        if (p < 0.5) {
            x = -x;
        }
        double x2 = x * x;
        double w = x + sk * (x2 - 1.0) / 6.0;
        if (n >= 10.0) {
            w += kurtosis * x * (x2 - 3.0) / 24.0 + sk * sk * x * (2.0 * x2 - 5.0) / -36.0;
        }
        w = m + sigma * w;
        return Math.max(w, MIN_PARAMETER);
    }

    private static double temmeMethod1IbetaInverse(double a, double b, double z) {
        double eta0 = Normal.erfcInv(2.0 * z);
        eta0 /= -Math.sqrt(a / 2.0);

        double[] terms = {eta0, 0.0, 0.0, 0.0};
        double[] workspace = new double[7];
        double B = b - a;
        double B2 = B * B;
        double B3 = B2 * B;

        workspace[0] = -B * SQRT_TWO / 2.0;
        workspace[1] = (1.0 - 2.0 * B) / 8.0;
        workspace[2] = -(B * SQRT_TWO / 48.0);
        workspace[3] = -1.0 / 192.0;
        workspace[4] = -B * SQRT_TWO / 3840.0;
        terms[1] = evaluatePolynomial(workspace, eta0, 5);

        workspace[0] = B * SQRT_TWO * (3.0 * B - 2.0) / 12.0;
        workspace[1] = (20.0 * B2 - 12.0 * B + 1.0) / 128.0;
        workspace[2] = B * SQRT_TWO * (20.0 * B - 1.0) / 960.0;
        workspace[3] = (16.0 * B2 + 30.0 * B - 15.0) / 4608.0;
        workspace[4] = B * SQRT_TWO * (21.0 * B + 32.0) / 53760.0;
        workspace[5] = (-32.0 * B2 + 63.0) / 368640.0;
        workspace[6] = -B * SQRT_TWO * (120.0 * B + 17.0) / 25804480.0;
        terms[2] = evaluatePolynomial(workspace, eta0, 7);

        workspace[0] = B * SQRT_TWO * (-75.0 * B2 + 80.0 * B - 16.0) / 480.0;
        workspace[1] = (-1080.0 * B3 + 868.0 * B2 - 90.0 * B - 45.0) / 9216.0;
        workspace[2] = B * SQRT_TWO * (-1190.0 * B2 + 84.0 * B + 373.0) / 53760.0;
        workspace[3] = (-2240.0 * B3 - 2508.0 * B2 + 2100.0 * B - 165.0) / 368640.0;
        terms[3] = evaluatePolynomial(workspace, eta0, 4);

        double eta = evaluatePolynomial(terms, 1.0 / a);
        double eta2 = eta * eta;
        double c = -Math.exp(-eta2 / 2.0);
        double x = eta2 == 0.0 ? 0.5 : (1.0 + eta * Math.sqrt((1.0 + c) / eta2)) / 2.0;
        if (x < 0.0) {
            x = 0.0;
        } else if (x > 1.0) {
            x = 1.0;
        }
        return x;
    }

    private static double temmeMethod2IbetaInverse(double a, double b, double z, double r, double theta) {
        double eta0 = Normal.erfcInv(2.0 * z);
        eta0 /= -Math.sqrt(r / 2.0);

        double s = Math.sin(theta);
        double c = Math.cos(theta);
        double[] terms = {eta0, 0.0, 0.0, 0.0};
        double[] workspace = new double[6];
        double sc = s * c;
        double sc2 = sc * sc;
        double sc3 = sc2 * sc;
        double sc4 = sc2 * sc2;
        double sc5 = sc2 * sc3;
        double sc6 = sc3 * sc3;
        double sc7 = sc4 * sc3;

        workspace[0] = (2.0 * s * s - 1.0) / (3.0 * s * c);
        workspace[1] = -evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_1, s) / (36.0 * sc2);
        workspace[2] = evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_2, s) / (1620.0 * sc3);
        workspace[3] = -evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_3, s) / (6480.0 * sc4);
        workspace[4] = evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_4, s) / (90720.0 * sc5);
        terms[1] = evaluatePolynomial(workspace, eta0, 5);

        workspace[0] = -evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_5, s) / (405.0 * sc3);
        workspace[1] = evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_6, s) / (2592.0 * sc4);
        workspace[2] = -evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_7, s) / (204120.0 * sc5);
        workspace[3] = -evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_8, s) / (2099520.0 * sc6);
        terms[2] = evaluatePolynomial(workspace, eta0, 4);

        workspace[0] = evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_9, s) / (102060.0 * sc5);
        workspace[1] = -evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_10, s) / (20995200.0 * sc6);
        workspace[2] = evaluateEvenPolynomial(TEMME_METHOD2_COEFFS_11, s) / (36741600.0 * sc7);
        terms[3] = evaluatePolynomial(workspace, eta0, 3);

        double eta = evaluatePolynomial(terms, 1.0 / r);
        double s2 = s * s;
        double c2 = c * c;
        double alpha = (c / s) * (c / s);
        double lu = (-(eta * eta) / (2.0 * s2) + Math.log(s2) + c2 * Math.log(c2) / s2);

        double x;
        if (Math.abs(eta) < 0.7) {
            workspace[0] = s2;
            workspace[1] = s * c;
            workspace[2] = (1.0 - 2.0 * workspace[0]) / 3.0;
            workspace[3] = evaluatePolynomial(TEMME_METHOD2_SMALL_ETA_COEFFS, workspace[0]) / (36.0 * s * c);
            workspace[4] = evaluatePolynomial(TEMME_METHOD2_COEFFS_2, workspace[0]) / (270.0 * workspace[0] * c * c);
            x = evaluatePolynomial(workspace, eta, 5);
        } else {
            double u = Math.exp(lu);
            workspace[0] = u;
            workspace[1] = alpha;
            workspace[2] = 0.0;
            workspace[3] = 3.0 * alpha * (3.0 * alpha + 1.0) / 6.0;
            workspace[4] = 4.0 * alpha * (4.0 * alpha + 1.0) * (4.0 * alpha + 2.0) / 24.0;
            workspace[5] = 5.0 * alpha * (5.0 * alpha + 1.0) * (5.0 * alpha + 2.0) * (5.0 * alpha + 3.0) / 120.0;
            x = evaluatePolynomial(workspace, u, 6);
            if ((x - s2) * eta < 0.0) {
                x = 1.0 - x;
            }
        }

        double lower = eta < 0.0 ? 0.0 : s2;
        double upper = eta < 0.0 ? s2 : 1.0;
        if ((x < lower) || (x > upper)) {
            x = (lower + upper) / 2.0;
        }
        try {
            return solveTemmeRoot(-lu, alpha, x, lower, upper);
        } catch (ArithmeticException error) {
            return x;
        }
    }

    private static double temmeMethod3IbetaInverse(double a, double b, double p, double q) {
        double eta0 = p < q ? Gamma.gammaQInv(b, p) : Gamma.gammaPInv(b, q);
        eta0 /= a;

        double mu = b / a;
        double w = Math.sqrt(1.0 + mu);
        double w2 = w * w;
        double w3 = w2 * w;
        double w4 = w2 * w2;
        double w5 = w3 * w2;
        double w6 = w3 * w3;
        double w7 = w4 * w3;
        double w8 = w4 * w4;
        double w9 = w5 * w4;
        double w10 = w5 * w5;
        double d = eta0 - mu;
        double d2 = d * d;
        double d3 = d2 * d;
        double d4 = d2 * d2;
        double w1 = w + 1.0;
        double w12 = w1 * w1;
        double w13 = w1 * w12;
        double w14 = w12 * w12;

        double e1 = (w + 2.0) * (w - 1.0) / (3.0 * w);
        e1 += (w3 + 9.0 * w2 + 21.0 * w + 5.0) * d / (36.0 * w2 * w1);
        e1 -= (w4 - 13.0 * w3 + 69.0 * w2 + 167.0 * w + 46.0) * d2 / (1620.0 * w12 * w3);
        e1 -= (7.0 * w5 + 21.0 * w4 + 70.0 * w3 + 26.0 * w2 - 93.0 * w - 31.0) * d3 / (6480.0 * w13 * w4);
        e1 -= (75.0 * w6 + 202.0 * w5 + 188.0 * w4 - 888.0 * w3 - 1345.0 * w2 + 118.0 * w + 138.0) * d4 / (272160.0 * w14 * w5);

        double e2 = (28.0 * w4 + 131.0 * w3 + 402.0 * w2 + 581.0 * w + 208.0) * (w - 1.0) / (1620.0 * w1 * w3);
        e2 -= (35.0 * w6 - 154.0 * w5 - 623.0 * w4 - 1636.0 * w3 - 3983.0 * w2 - 3514.0 * w - 925.0) * d / (12960.0 * w12 * w4);
        e2 -= (2132.0 * w7 + 7915.0 * w6 + 16821.0 * w5 + 35066.0 * w4 + 87490.0 * w3 + 141183.0 * w2 + 95993.0 * w + 21640.0) * d2 / (816480.0 * w5 * w13);
        e2 -= (11053.0 * w8 + 53308.0 * w7 + 117010.0 * w6 + 163924.0 * w5 + 116188.0 * w4 - 258428.0 * w3 - 677042.0 * w2 - 481940.0 * w - 105497.0) * d3 / (14696640.0 * w14 * w6);

        double e3 = -((3592.0 * w7 + 8375.0 * w6 - 1323.0 * w5 - 29198.0 * w4 - 89578.0 * w3 - 154413.0 * w2 - 116063.0 * w - 29632.0) * (w - 1.0)) / (816480.0 * w5 * w12);
        e3 -= (442043.0 * w9 + 2054169.0 * w8 + 3803094.0 * w7 + 3470754.0 * w6 + 2141568.0 * w5 - 2393568.0 * w4 - 19904934.0 * w3 - 34714674.0 * w2 - 23128299.0 * w - 5253353.0) * d / (146966400.0 * w6 * w13);
        e3 -= (116932.0 * w10 + 819281.0 * w9 + 2378172.0 * w8 + 4341330.0 * w7 + 6806004.0 * w6 + 10622748.0 * w5 + 18739500.0 * w4 + 30651894.0 * w3 + 30869976.0 * w2 + 15431867.0 * w + 2919016.0) * d2 / (146966400.0 * w14 * w7);

        double eta = eta0 + e1 / a + e2 / (a * a) + e3 / (a * a * a);
        if (eta <= 0.0) {
            eta = Double.MIN_NORMAL;
        }
        double u = eta - mu * Math.log(eta) + (1.0 + mu) * Math.log(1.0 + mu) - mu;
        double cross = 1.0 / (1.0 + mu);
        if ((cross == 0.0) || (cross == 1.0)) {
            return cross;
        }
        double lower = eta < mu ? cross : 0.0;
        double upper = eta < mu ? 1.0 : cross;
        double x = (lower + upper) / 2.0;
        return solveTemmeRoot(u, mu, x, lower, upper);
    }

    private static double solveTemmeRoot(final double t, final double a, double guess, double lower, double upper) {
        return RootIterators.newtonRaphsonIterate(
            (x, evaluation) -> temmeRoot(t, a, x, evaluation),
            guess,
            lower,
            upper,
            DOUBLE_PRECISION_BITS / 2,
            MAX_ITERATIONS,
            "Temme root refinement"
        );
    }

    private static void temmeRoot(double t, double a, double x, double[] evaluation) {
        double y = 1.0 - x;
        evaluation[0] = Math.log(x) + a * Math.log(y) + t;
        evaluation[1] = (1.0 / x) - (a / y);
    }

    private static void ibetaRoot(double a,
                                  double b,
                                  double target,
                                  boolean invert,
                                  double x,
                                  double[] evaluation) {
        double y = 1.0 - x;
        Double2 valueAndDerivative = ibetaImpWithDerivative(a, b, x, invert, true);
        double functionValue = valueAndDerivative._1() - target;
        double firstDerivative = valueAndDerivative._2();
        if (invert) {
            firstDerivative = -firstDerivative;
        }
        double xSafe = x == 0.0 ? Double.MIN_NORMAL * 64.0 : x;
        double ySafe = y == 0.0 ? Double.MIN_NORMAL * 64.0 : y;
        double secondDerivative = firstDerivative * (-ySafe * a + (b - 2.0) * xSafe + 1.0);
        if (Math.abs(secondDerivative) < ySafe * xSafe * Double.MAX_VALUE) {
            secondDerivative /= (ySafe * xSafe);
        }
        if (invert) {
            secondDerivative = -secondDerivative;
        }
        if (firstDerivative == 0.0) {
            firstDerivative = (invert ? -1.0 : 1.0) * Double.MIN_NORMAL * 64.0;
        }
        evaluation[0] = functionValue;
        evaluation[1] = firstDerivative;
        evaluation[2] = secondDerivative;
    }

    private static double betaInvAbValue(double b,
                                         double z,
                                         double target,
                                         boolean invert,
                                         boolean swapAb,
                                         double alphaCandidate) {
        double alpha = swapAb ? b : alphaCandidate;
        double beta = swapAb ? alphaCandidate : b;
        return invert
            ? target - ibetaImp(alpha, beta, z, true, true)
            : ibetaImp(alpha, beta, z, false, true) - target;
    }

    private static Double2 findIbetaInvFromTDist(double a, double p) {
        double u = p / 2.0;
        double v = 1.0 - u;
        double df = a * 2.0;
        double t = inverseStudentsT(df, u, v);
        double t2 = t * t;
        double y = t2 / (df + t2);
        return new Double2(df / (df + t2), y);
    }

    private static double inverseStudentsTHill(double degreesOfFreedom, double u) {
        if (degreesOfFreedom > 1.0e20) {
            return -Normal.erfcInv(2.0 * u) * SQRT_TWO;
        }

        double a = 1.0 / (degreesOfFreedom - 0.5);
        double b = 48.0 / (a * a);
        double c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36;
        double d = ((94.5 / (b + c) - 3.0) / b + 1.0) * Math.sqrt(a * Math.PI / 2.0) * degreesOfFreedom;
        double y = Math.pow(d * 2.0 * u, 2.0 / degreesOfFreedom);

        if (y > (0.05 + a)) {
            double x = -Normal.erfcInv(2.0 * u) * SQRT_TWO;
            y = x * x;
            if (degreesOfFreedom < 5.0) {
                c += 0.3 * (degreesOfFreedom - 4.5) * (x + 0.6);
            }
            c += (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b;
            y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x;
            y = Math.expm1(a * y * y);
        } else {
            y = ((1.0 / ((((degreesOfFreedom + 6.0) / (degreesOfFreedom * y)) - 0.089 * d - 0.822)
                * (degreesOfFreedom + 2.0) * 3.0) + 0.5 / (degreesOfFreedom + 4.0)) * y - 1.0)
                * (degreesOfFreedom + 1.0) / (degreesOfFreedom + 2.0) + 1.0 / y;
        }

        return -Math.sqrt(degreesOfFreedom * y);
    }

    private static double inverseStudentsTTailSeries(double degreesOfFreedom, double v) {
        double w = Gamma.tgammaDeltaRatio(degreesOfFreedom / 2.0, 0.5)
            * Math.sqrt(degreesOfFreedom * Math.PI) * v;
        double np2 = degreesOfFreedom + 2.0;
        double np4 = degreesOfFreedom + 4.0;
        double np6 = degreesOfFreedom + 6.0;
        double[] d = new double[7];
        d[0] = 1.0;
        d[1] = -(degreesOfFreedom + 1.0) / (2.0 * np2);
        np2 *= degreesOfFreedom + 2.0;
        d[2] = -degreesOfFreedom * (degreesOfFreedom + 1.0) * (degreesOfFreedom + 3.0) / (8.0 * np2 * np4);
        np2 *= degreesOfFreedom + 2.0;
        d[3] = -degreesOfFreedom * (degreesOfFreedom + 1.0) * (degreesOfFreedom + 5.0)
            * ((((3.0 * degreesOfFreedom) + 7.0) * degreesOfFreedom) - 2.0) / (48.0 * np2 * np4 * np6);
        np2 *= degreesOfFreedom + 2.0;
        np4 *= degreesOfFreedom + 4.0;
        double poly4 = ((((((15.0 * degreesOfFreedom) + 154.0) * degreesOfFreedom + 465.0) * degreesOfFreedom + 286.0)
            * degreesOfFreedom - 336.0) * degreesOfFreedom) + 64.0;
        d[4] = -degreesOfFreedom * (degreesOfFreedom + 1.0) * (degreesOfFreedom + 7.0)
            * poly4
            / (384.0 * np2 * np4 * np6 * (degreesOfFreedom + 8.0));
        np2 *= degreesOfFreedom + 2.0;
        double poly5 = (((((((35.0 * degreesOfFreedom) + 452.0) * degreesOfFreedom + 1573.0) * degreesOfFreedom + 600.0)
            * degreesOfFreedom - 2020.0) * degreesOfFreedom) + 928.0) * degreesOfFreedom - 128.0;
        d[5] = -degreesOfFreedom * (degreesOfFreedom + 1.0) * (degreesOfFreedom + 3.0) * (degreesOfFreedom + 9.0)
            * poly5
            / (1280.0 * np2 * np4 * np6 * (degreesOfFreedom + 8.0) * (degreesOfFreedom + 10.0));
        np2 *= degreesOfFreedom + 2.0;
        np4 *= degreesOfFreedom + 4.0;
        np6 *= degreesOfFreedom + 6.0;
        double poly6 = 945.0 * degreesOfFreedom + 31506.0;
        poly6 = poly6 * degreesOfFreedom + 425858.0;
        poly6 = poly6 * degreesOfFreedom + 2980236.0;
        poly6 = poly6 * degreesOfFreedom + 11266745.0;
        poly6 = poly6 * degreesOfFreedom + 20675018.0;
        poly6 = poly6 * degreesOfFreedom + 7747124.0;
        poly6 = poly6 * degreesOfFreedom - 22574632.0;
        poly6 = poly6 * degreesOfFreedom - 8565600.0;
        poly6 = poly6 * degreesOfFreedom + 18108416.0;
        poly6 = poly6 * degreesOfFreedom - 7099392.0;
        poly6 = poly6 * degreesOfFreedom + 884736.0;
        d[6] = -degreesOfFreedom * (degreesOfFreedom + 1.0) * (degreesOfFreedom + 11.0)
            * poly6
            / (46080.0 * np2 * np4 * np6 * (degreesOfFreedom + 8.0) * (degreesOfFreedom + 10.0) * (degreesOfFreedom + 12.0));

        double rootN = Math.sqrt(degreesOfFreedom);
        double div = Math.pow(rootN * w, 1.0 / degreesOfFreedom);
        double power = div * div;
        double result = evaluatePolynomial(d, power);
        result *= rootN;
        result /= div;
        return -result;
    }

    private static double inverseStudentsTBodySeries(double degreesOfFreedom, double u) {
        double v = Gamma.tgammaDeltaRatio(degreesOfFreedom / 2.0, 0.5)
            * Math.sqrt(degreesOfFreedom * Math.PI) * (u - 0.5);
        double[] c = new double[11];
        c[1] = 1.0;
        double in = 1.0 / degreesOfFreedom;
        c[2] = 0.16666666666666666 + 0.16666666666666666 * in;
        c[3] = (0.008333333333333333 * in + 0.06666666666666667) * in + 0.058333333333333334;
        c[4] = ((0.0001984126984126984 * in + 0.0017857142857142857) * in + 0.026785714285714284) * in + 0.0251984126984127;
        c[5] = (((2.755731922398589E-6 * in + 0.0003747795414462081) * in - 0.0011078042328042327) * in
            + 0.010559964726631393) * in + 0.012039792768959435;
        c[6] = ((((2.5052108385441717E-8 * in - 6.270542728876062E-5) * in + 0.0005945867404200738) * in
            - 0.0016095979637646305) * in + 0.006103921156004489) * in + 0.003837005972422639;
        c[7] = (((((1.6059043836821614E-10 * in + 1.54012654012654E-5) * in - 0.00016376804137220804) * in
            + 0.0006908420797309686) * in - 0.0012579159844784844) * in + 0.0010898206731540065) * in + 0.0032177478835464947;
        c[8] = ((((((7.647163731819816E-13 * in - 3.9851014346715405E-6) * in + 4.925574636636145E-5) * in
            - 0.000249472580470431) * in + 0.0006451304695145635) * in - 0.0007624513544032393) * in
            + 3.3530976880017886E-5) * in + 0.001743826229834001;
        c[9] = (((((((2.8114572543455207E-15 * in + 1.091417917349679E-6) * in - 1.5303004486655378E-5) * in
            + 9.08671079352199E-5) * in - 0.0002913341446693807) * in + 0.0005140660578834112) * in
            - 0.00036307660358786885) * in - 0.0003110108632631878) * in + 0.0009647274732138864;
        c[10] = ((((((((8.22063524662433E-18 * in - 3.123956959982987E-7) * in + 4.890304529197535E-6) * in
            - 3.320265239137206E-5) * in + 0.00012645437628698077) * in - 0.00028690924218514614) * in
            + 0.00035764655430568634) * in - 0.00010230378073700413) * in - 0.0003694266780000966) * in
            + 0.0005422926281312969;
        return evaluateOddPolynomial(c, v);
    }

    private static double inverseStudentsT(double degreesOfFreedom, double u, double v) {
        boolean invert = false;
        double result;
        if (u > v) {
            double tmp = u;
            u = v;
            v = tmp;
            invert = true;
        }

        if ((Math.floor(degreesOfFreedom) == degreesOfFreedom) && (degreesOfFreedom < 20.0)) {
            double tolerance = Math.scalb(1.0, (2 * DOUBLE_PRECISION_BITS) / 3);
            switch ((int) Math.rint(degreesOfFreedom)) {
                case 1:
                    result = u == 0.5 ? 0.0 : -Math.cos(Math.PI * u) / Math.sin(Math.PI * u);
                    return invert ? -result : result;
                case 2:
                    result = (2.0 * u - 1.0) / Math.sqrt(2.0 * u * v);
                    return invert ? -result : result;
                case 4:
                    double alpha = 4.0 * u * v;
                    double rootAlpha = Math.sqrt(alpha);
                    double r = 4.0 * Math.cos(Math.acos(rootAlpha) / 3.0) / rootAlpha;
                    double x = Math.sqrt(r - 4.0);
                    result = (u - 0.5) < 0.0 ? -x : x;
                    return invert ? -result : result;
                case 6:
                    if (u < 1.0e-150) {
                        result = inverseStudentsTHill(degreesOfFreedom, u);
                        return invert ? -result : result;
                    }
                    double a = 4.0 * (u - u * u);
                    double b = Math.cbrt(a);
                    double c = 0.8549879733383485;
                    double p = 6.0 * (1.0 + c * (1.0 / b - 1.0));
                    double previous;
                    do {
                        double p2 = p * p;
                        double p4 = p2 * p2;
                        double p5 = p * p4;
                        previous = p;
                        p = 2.0 * (8.0 * a * p5 - 270.0 * p2 + 2187.0) / (5.0 * (4.0 * a * p4 - 216.0 * p - 243.0));
                    } while (Math.abs((p - previous) / p) > tolerance);
                    p = Math.sqrt(p - degreesOfFreedom);
                    result = (u - 0.5) < 0.0 ? -p : p;
                    return invert ? -result : result;
                default:
                    break;
            }
        }

        if (degreesOfFreedom > 268435456.0) {
            result = -Normal.erfcInv(2.0 * u) * SQRT_TWO;
            return invert ? -result : result;
        }
        if (degreesOfFreedom < 3.0) {
            double crossover = 0.2742 - degreesOfFreedom * 0.0242143;
            result = u > crossover ? inverseStudentsTBodySeries(degreesOfFreedom, u) : inverseStudentsTTailSeries(degreesOfFreedom, u);
            return invert ? -result : result;
        }

        int uExponent = Math.getExponent(u) + 1;
        if (uExponent < degreesOfFreedom / 0.654) {
            result = inverseStudentsTHill(degreesOfFreedom, u);
        } else {
            result = inverseStudentsTTailSeries(degreesOfFreedom, u);
        }
        return invert ? -result : result;
    }

    private static double logBinomialCoefficient(int n, int k) {
        return Gamma.lgamma(n + 1.0) - Gamma.lgamma(k + 1.0) - Gamma.lgamma(n - k + 1.0);
    }

    private static double risingFactorialRatio(double a, double b, int k) {
        double result = 1.0;
        for (int i = 0; i < k; i++) {
            result *= (a + i) / (b + i);
        }
        return result;
    }

    private static double powm1(double x, double exponent) {
        return Math.expm1(exponent * Math.log(x));
    }

    private static void validatePositiveParameters(double a, double b, String functionName) {
        validatePositiveParameter(a, "a", functionName);
        validatePositiveParameter(b, "b", functionName);
    }

    private static void validatePositiveParameter(double value, String name, String functionName) {
        if (!Double.isFinite(value) || !(value > 0.0)) {
            throw new IllegalArgumentException(functionName + ": " + name + " must be finite and greater than zero");
        }
    }

    private static void validateIncompleteParameters(double a, double b, double x, String functionName) {
        validatePositiveParameters(a, b, functionName);
        validateUnitInterval(x, "x");
    }

    private static void validateRegularizedParameters(double a, double b, double x, String functionName) {
        if (!Double.isFinite(a) || a < 0.0) {
            throw new IllegalArgumentException(functionName + ": a must be finite and non-negative");
        }
        if (!Double.isFinite(b) || b < 0.0) {
            throw new IllegalArgumentException(functionName + ": b must be finite and non-negative");
        }
        if (a == 0.0 && b == 0.0) {
            throw new IllegalArgumentException(functionName + ": a and b must not both be zero");
        }
        validateUnitInterval(x, "x");
    }

    private static void validateUnitInterval(double x, String name) {
        if (Double.isNaN(x) || x < 0.0 || x > 1.0) {
            throw new IllegalArgumentException(name + " must be in [0, 1]");
        }
    }

    private static void validateOpenUnitInterval(double x, String name) {
        if (Double.isNaN(x) || x <= 0.0 || x >= 1.0) {
            throw new IllegalArgumentException(name + " must be in (0, 1)");
        }
    }

    private static Double2 ibetaFraction(double a, double b, double x, double y, double m) {
        double denominator = a + 2.0 * m - 1.0;
        double numerator = (m * (a + m - 1.0) / denominator)
                * ((a + b + m - 1.0) / denominator)
                * (b - m) * x * x;
        double bTerm = m;
        bTerm += (m * (b - m) * x) / (a + 2.0 * m - 1.0);
        bTerm += ((a + m) * (a * y - b * x + 1.0 + m * (2.0 - x))) / (a + 2.0 * m + 1.0);
        return Double2.fraction(numerator, bTerm);
    }

}