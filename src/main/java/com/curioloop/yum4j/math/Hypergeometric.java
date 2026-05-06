package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.Commons.factorial;
import static com.curioloop.yum4j.math.Gamma.completeExtended;
import static com.curioloop.yum4j.math.Gamma.logAbsExtended;
import static com.curioloop.yum4j.math.Gamma.signExtended;
import static com.curioloop.yum4j.math.Commons.isInteger;
import static com.curioloop.yum4j.math.Commons.sameParameter;
import static com.curioloop.yum4j.math.Commons.validateFinite;
import static com.curioloop.yum4j.math.Commons.validateHolder;

/**
 * Real-valued hypergeometric primitives.
 *
 * <p>This tranche covers a focused first cut of the classical hypergeometric family:
 *
 * <pre>
 * 1F0(a;;z)     = sum((a)_n z^n / n!, n = 0..infinity)
 * 0F1(;b;z)     = sum(z^n / ((b)_n n!), n = 0..infinity)
 * 1F1(a;b;z)    = sum((a)_n z^n / ((b)_n n!), n = 0..infinity)
 * 2F0(a,b;;z)   = sum((a)_n (b)_n z^n / n!, n = 0..infinity)
 * pFq(a;b;z)    = sum(prod((a_i)_n) z^n / (prod((b_j)_n) n!), n = 0..infinity)
 * </pre>
 *
 * <p>The implementation keeps the real scalar and generic public accept domains aligned to the
 * supported Boost product surface. Source provenance is intentionally split the same way as Boost:
 * public named-family boundaries follow the corresponding
 * {@code hypergeometric_1F0.hpp}, {@code hypergeometric_0F1.hpp},
 * {@code hypergeometric_1F1.hpp}, {@code hypergeometric_2F0.hpp}, and
 * {@code hypergeometric_pFq.hpp} entry points, while internal route precedence for the real scalar
 * {@code 1F1} core tracks Boost's live detail selector taxonomy rather than a repo-local abstraction.
 * The implementation uses the exact closed form for {@code 1F0}, a Bessel-based identity for positive-argument {@code 0F1}, the symmetric
 * {@code 1F1(a;2a;z)} modified-Bessel identity on the positive real branch, and compensated
 * power-series summation for the remaining convergent entire-series branches. The current {@code 2F0}
 * public surface is intentionally narrower: on the real-valued branch it
 * supports only terminating polynomial cases, which are finite on any real {@code z}. The
 * generic {@code pFq} public entry point keeps Boost's checked-series accept domain: it supports the
 * built-in {@code 1F0} inversion for {@code |z| > 1}, all-real evaluation when {@code p <= q}, and
 * the disk plus unit-circle convergence boundary when {@code p = q + 1}. Within that same accept
 * domain, the {@code 1F0}, {@code 0F1}, and {@code 1F1} parameter shapes dispatch to their direct
 * named-family APIs, while the {@code absErrorHolder} overload keeps the Boost-style checked-series
 * error estimate. This
 * keeps the public freeze aligned with Boost's public product surface rather than repo-local
 * widening.
 */
public final class Hypergeometric {

    private static final double INTEGER_EPSILON = 1.0e-12;
    private static final double SERIES_EPSILON = 1.0e-15;
    private static final double CONTINUED_FRACTION_EPSILON = 1.0e-15;
    private static final double ONE_F_ONE_ASYMPTOTIC_MIN_ABS_A = 1.0e-3;
    private static final double ONE_F_ONE_ASYMPTOTIC_RATIO_LIMIT = 0.7;
    private static final double ONE_F_ONE_ASYMPTOTIC_INITIAL_TERM_LIMIT = 0.5;
    private static final double ONE_F_ONE_ASYMPTOTIC_MIN_Z = 40.0;
    private static final int ONE_F_ONE_ASYMPTOTIC_HALF_DIGITS = 26;
    private static final double ONE_F_ONE_LARGE_ABZ_MIN_PRODUCT = 50.0;
    private static final int ONE_F_ONE_LARGE_ABZ_LOCAL_A = 5;
    private static final double ROOT_EPSILON = Math.sqrt(Math.ulp(1.0));
    private static final double LOG_MAX_VALUE = Math.log(Double.MAX_VALUE);
    private static final double LOG_MIN_NORMAL = Math.log(Double.MIN_NORMAL);
    private static final double LOG_TRUE_MIN = Math.log(Double.MIN_VALUE);
    private static final int ONE_F_ONE_SERIES_CACHE_SIZE = 64;
    private static final int MAX_SERIES_ITERATIONS = 100000;
    private static final int BOOST_PUBLIC_PFQ_MAX_SERIES_ITERATIONS = 1000000;
    private static final int BOOST_PUBLIC_PFQ_MAX_B_TERMS = 5;
    // Dedicated NaN payload used as a primitive "no route matched" sentinel on real helper paths.
    private static final long NO_ROUTE_BITS = 0x7ff8000000000001L;
    private static final double NO_REAL_ROUTE = Double.longBitsToDouble(NO_ROUTE_BITS);
    private static final int ONE_F_ONE_ROUTE_EXACT = 1;
    private static final int ONE_F_ONE_ROUTE_PRE_13_3_6 = 2;
    private static final int ONE_F_ONE_ROUTE_SMALL_A_NEGATIVE_B = 3;
    private static final int ONE_F_ONE_ROUTE_ASYMPTOTIC = 4;
    private static final int ONE_F_ONE_ROUTE_RATIONAL = 5;
    private static final int ONE_F_ONE_ROUTE_PADE = 6;
    private static final int ONE_F_ONE_ROUTE_KUMMER = 7;
    private static final int ONE_F_ONE_ROUTE_TRICOMI = 8;
    private static final int ONE_F_ONE_ROUTE_DIVERGENT = 9;
    private static final int ONE_F_ONE_ROUTE_ALT_13_3_6 = 10;
    private static final int ONE_F_ONE_ROUTE_LARGE_ABZ = 11;
    private static final int ONE_F_ONE_ROUTE_CHECKED_SERIES = 12;
    private static final int ONE_F_ONE_ROUTE_GENERIC_SERIES = 13;
    private static final String[] ONE_F_ONE_ROUTE_NAMES = {
        null,
        "exact-or-terminating",
        "pre-13.3.6",
        "small-a-negative-b",
        "large-z-asymptotic",
        "rational",
        "pade",
        "negative-z-kummer",
        "tricomi",
        "divergent-fallback",
        "alt-13.3.6",
        "large-abz",
        "checked-series",
        "generic-series"
    };
    private static final String ONE_F_ONE_RECURRENCE_RATIO_CONTEXT = "oneFOne recurrence ratio";
    private static final int PFQ_ROUTE_ONE_F_ZERO_EXTERIOR_SPECIAL_CASE = 1;
    private static final int PFQ_ROUTE_DELEGATE_ONE_F_ZERO = 2;
    private static final int PFQ_ROUTE_DELEGATE_ZERO_F_ONE = 3;
    private static final int PFQ_ROUTE_DELEGATE_ONE_F_ONE = 4;
    private static final int PFQ_ROUTE_CHECKED_SERIES = 5;
    private static final String[] PFQ_ROUTE_NAMES = {
        null,
        "one-f-zero-exterior-special-case",
        "delegate-one-f-zero",
        "delegate-zero-f-one",
        "delegate-one-f-one",
        "checked-series"
    };
    private static final double[] ONE_F_ONE_SMALL_A_NEGATIVE_B_MAX_B_TABLE = {
        0.0, -47.3046, -6.7275, -52.0351, -8.9543, -57.2386, -11.9182, -62.9625,
        -14.421, -69.2587, -19.1943, -76.1846, -23.2252, -83.803, -28.1024, -92.1833,
        -34.0039, -101.402, -37.4043, -111.542, -45.2593, -122.696, -54.7637, -134.966,
        -60.2401, -148.462, -72.8905, -163.308, -88.1975, -179.639, -88.1975, -197.603,
        -106.719, -217.363, -129.13, -239.1, -142.043, -263.01, -156.247, -289.311,
        -189.059, -318.242, -207.965, -350.066, -228.762, -385.073, -276.801, -423.58,
        -304.482, -465.938, -334.93, -512.532, -368.423, -563.785, -405.265, -620.163,
        -445.792, -682.18, -539.408, -750.398, -593.349, -825.437, -652.683, -907.981,
        -717.952, -998.779
    };

    private Hypergeometric() {
    }

    private static boolean hasRealRoute(double value) {
        return Double.doubleToRawLongBits(value) != NO_ROUTE_BITS;
    }

    private static String lookupRouteName(String[] routeNames, int routeId) {
        return routeId > 0 && routeId < routeNames.length ? routeNames[routeId] : null;
    }

    private static String decodeOneFOneRealRoute(int routeId) {
        return lookupRouteName(ONE_F_ONE_ROUTE_NAMES, routeId);
    }

    private static String decodePFqRoute(int routeId) {
        return lookupRouteName(PFQ_ROUTE_NAMES, routeId);
    }

    /**
     * Returns the real-valued hypergeometric function {@code 1F0(a;;z)}.
     *
     * <p>For general real {@code a}, the real-valued branch is restricted to {@code z < 1} and is
     * evaluated via the exact identity {@code (1 - z)^(-a)}. When {@code a} is a non-positive integer,
     * the series terminates and the resulting polynomial is evaluated for any finite real {@code z}.
     *
     * @param a upper parameter
     * @param z argument
     * @return {@code 1F0(a;;z)}
     */
    public static double oneFZero(double a, double z) {
        validateFinite(a, "oneFZero: a");
        validateFinite(z, "oneFZero: z");

        if (a == 0.0 || z == 0.0) {
            return 1.0;
        }
        if (z == 1.0) {
            throw new IllegalArgumentException("oneFZero: z = 1 is a pole on the real-valued branch");
        }
        if (isInteger(a)) {
            return Math.pow(1.0 - z, -Math.rint(a));
        }
        if (!(z < 1.0)) {
            throw new IllegalArgumentException("oneFZero: real-valued branch requires z < 1 unless a is integral: " + z);
        }
        return Math.exp(-a * Math.log1p(-z));
    }

    /**
     * Returns the confluent hypergeometric limit function {@code 0F1(;b;z)}.
     *
     * @param b lower parameter, must not be a non-positive integer
     * @param z argument
     * @return {@code 0F1(;b;z)}
     */
    public static double zeroFOne(double b, double z) {
        validateFinite(b, "zeroFOne: b");
        validateFinite(z, "zeroFOne: z");
        validateNonPoleLowerParameter(b);

        if (z == 0.0) {
            return 1.0;
        }
        if (z > 0.0) {
            double root = Math.sqrt(z);
            return completeExtended(b) * Math.pow(root, 1.0 - b) * Bessel.i(b - 1.0, 2.0 * root);
        }
        if (z < -5.0 && b > -5.0) {
            if (Math.abs(z / b) > 4.0) {
                return zeroFOneNegativeRealBessel(b, z);
            }
            return zeroFOneNegativeRealContinuedFraction(b, z);
        }
        return zeroFOneSeries(b, z);
    }

    /**
     * Returns the confluent hypergeometric function {@code 1F1(a;b;z)}.
     *
    * <p>In addition to the regular non-pole lower-parameter surface, this implementation supports
    * the finite polynomial cases with a non-positive integer lower parameter {@code b = -m} when
    * the upper parameter is a non-positive integer {@code a = -n} and {@code n <= m}, so the
    * terminating series ends before the lower Pochhammer factor reaches zero. Internally the real
    * core now follows Boost's live {@code 1F1} selector taxonomy more closely: exact identities,
    * A&S 13.3.6 argument-shift routes, the small-{@code a} / negative-{@code b} ratio route,
    * large-{@code z} asymptotic routing, Luke's rational approximation, Kummer reflection,
    * divergent-region fallbacks, and the large-{@code a z} shifted-series regime. This keeps the
    * public real contract unchanged while making route precedence match Boost's detail code much
    * more closely.
     *
     * @param a upper parameter
     * @param b lower parameter, must not be a non-positive integer
     * @param z argument
     * @return {@code 1F1(a;b;z)}
     */
    public static double oneFOne(double a, double b, double z) {
        validateFinite(a, "oneFOne: a");
        validateFinite(b, "oneFOne: b");
        validateFinite(z, "oneFOne: z");

        return oneFOneImpl(a, b, z);
    }

    /**
     * Returns the regularized confluent hypergeometric function {@code 1F1(a;b;z) / Gamma(b)}.
     *
     * @param a upper parameter
     * @param b lower parameter, must not be a non-positive integer unless the series terminates
     * @param z argument
     * @return regularized {@code 1F1(a;b;z)}
     */
    public static double oneFOneRegularized(double a, double b, double z) {
        validateFinite(a, "oneFOneRegularized: a");
        validateFinite(b, "oneFOneRegularized: b");
        validateFinite(z, "oneFOneRegularized: z");

        DoubleI signedLog = logOneFOneSignedImpl(a, b, z);
        return signedExp(signedLog.value() - logAbsExtended(b), signedLog.sign() * signExtended(b));
    }

    /**
     * Returns {@code log(abs(1F1(a;b;z)))}.
     *
     * @param a upper parameter
     * @param b lower parameter, must not be a non-positive integer unless the series terminates
     * @param z argument
     * @return {@code log(abs(1F1(a;b;z)))}
     */
    public static double logOneFOne(double a, double b, double z) {
        return logOneFOneSigned(a, b, z).value();
    }

    /**
     * Returns {@code log(abs(1F1(a;b;z)))} together with the sign of the real result.
     *
     * @param a upper parameter
     * @param b lower parameter, must not be a non-positive integer unless the series terminates
     * @param z argument
     * @return {@link DoubleI} containing {@code (log(abs(1F1(a;b;z))), sign(1F1(a;b;z)))}
     */
    public static DoubleI logOneFOneSigned(double a, double b, double z) {
        validateFinite(a, "logOneFOne: a");
        validateFinite(b, "logOneFOne: b");
        validateFinite(z, "logOneFOne: z");

        return logOneFOneSignedImpl(a, b, z);
    }

    private static double oneFOneImpl(double a, double b, double z) {
        return oneFOneRouted(a, b, z).value();
    }

    static String debugOneFOneRealRoute(double a, double b, double z) {
        return decodeOneFOneRealRoute(oneFOneRouted(a, b, z)._2());
    }

    private static DoubleI oneFOneRouted(double a, double b, double z) {
        validateOneFOneParameters(a, b);

        double routed = oneFOneExactOrTerminating(a, b, z);
        if (hasRealRoute(routed)) {
            return new DoubleI(routed, ONE_F_ONE_ROUTE_EXACT);
        }

        routed = oneFOneAs1336NegativeZRoute(a, b, z);
        if (hasRealRoute(routed)) {
            return new DoubleI(routed, ONE_F_ONE_ROUTE_PRE_13_3_6);
        }

        routed = oneFOneSmallANegativeBByRatioRoute(a, b, z);
        if (hasRealRoute(routed)) {
            return new DoubleI(routed, ONE_F_ONE_ROUTE_SMALL_A_NEGATIVE_B);
        }

        routed = oneFOneLargeZAsymptoticRoute(a, b, z);
        if (hasRealRoute(routed)) {
            return new DoubleI(routed, ONE_F_ONE_ROUTE_ASYMPTOTIC);
        }

        routed = oneFOneSmallRatioRoute(a, b, z);
        if (hasRealRoute(routed)) {
            return new DoubleI(routed, ONE_F_ONE_ROUTE_RATIONAL);
        }

        DoubleI negativeZRoute = oneFOneNegativeZRoute(a, b, z);
        if (hasRealRoute(negativeZRoute.value())) {
            return negativeZRoute;
        }

        if (isOneFOneSeriesDivergent(a, b, z) && isNegativeInteger(a) && -a < BOOST_PUBLIC_PFQ_MAX_SERIES_ITERATIONS) {
            return new DoubleI(oneFOneBackwardRecurrenceForNegativeA(a, b, z), ONE_F_ONE_ROUTE_DIVERGENT);
        }

        routed = oneFOneTricomiRoute(a, b, z);
        if (hasRealRoute(routed)) {
            return new DoubleI(routed, ONE_F_ONE_ROUTE_TRICOMI);
        }

        routed = oneFOneDivergentFallbackRoute(a, b, z);
        if (hasRealRoute(routed)) {
            return new DoubleI(routed, ONE_F_ONE_ROUTE_DIVERGENT);
        }

        routed = oneFOneAs1336PositiveZRoute(a, b, z);
        if (hasRealRoute(routed)) {
            return new DoubleI(routed, ONE_F_ONE_ROUTE_ALT_13_3_6);
        }

        routed = oneFOneLargeAbzRoute(a, b, z);
        if (hasRealRoute(routed)) {
            return new DoubleI(routed, ONE_F_ONE_ROUTE_LARGE_ABZ);
        }

        routed = oneFOneCheckedSeriesRoute(a, b, z);
        if (hasRealRoute(routed)) {
            return new DoubleI(routed, ONE_F_ONE_ROUTE_CHECKED_SERIES);
        }

        return new DoubleI(oneFOneSeries(a, b, z), ONE_F_ONE_ROUTE_GENERIC_SERIES);
    }

    private static double oneFOneExactOrTerminating(double a, double b, double z) {
        if (a == 0.0 || z == 0.0) {
            return 1.0;
        }
        double bMinusA = b - a;
        if (sameParameter(a, -1.0)) {
            double result = 1.0 - z / b;
            if (Math.abs(result) < 0.5) {
                result = (b - z) / b;
            }
            return result;
        }
        if (sameParameter(a, b)) {
            if (isNegativeInteger(a)) {
                if (a < -20.0 && z > 0.0 && z < 1.0) {
                    return scaleByExp(Gamma.gammaQ(1.0 - a, z), z);
                }
                return oneFOneCheckedSeries(a, b, z);
            }
            return Math.exp(z);
        }
        if (sameParameter(bMinusA, -1.0) && Math.abs(a) > 0.5) {
            if (isNegativeInteger(a) && a > -50.0) {
                return oneFOneSeries(a, b, z);
            }
            return scaleByExp((b + z) / b, z);
        }
        if (sameParameter(a, 1.0) && sameParameter(b, 2.0)) {
            return Math.expm1(z) / z;
        }
        if (z > 0.0 && sameParameter(b, 2.0 * a) && !isOneFOneLargeAbzRegion(a, b, z)) {
            return oneFOneDoubleUpper(a, z);
        }
        if (bMinusA == b && Math.abs(z / b) < Math.ulp(1.0)) {
            return 1.0;
        }
        return NO_REAL_ROUTE;
    }

    private static DoubleI oneFOneNegativeZRoute(double a, double b, double z) {
        if (!needKummerReflection(a, b, z)) {
            return new DoubleI(NO_REAL_ROUTE, 0);
        }

        if (sameParameter(a, 1.0)) {
            return new DoubleI(oneFOnePade(b, z), ONE_F_ONE_ROUTE_PADE);
        }

        if (isNegativeInteger(b)) {
            if (a > -50.0) {
                return new DoubleI(oneFOneSeries(a, b, z), ONE_F_ONE_ROUTE_GENERIC_SERIES);
            }
            return oneFOneRouted(a, oneFOneFloatNextLowerParameter(b), z);
        }

        return new DoubleI(scaleByExp(oneFOneImpl(b - a, b, -z), z), ONE_F_ONE_ROUTE_KUMMER);
    }

    private static double oneFOneSmallRatioRoute(double a, double b, double z) {
        if (canUseOneFOneRational(a, b, z)) {
            return oneFOneRational(a, b, z);
        }
        return NO_REAL_ROUTE;
    }

    private static double oneFOneAs1336NegativeZRoute(double a, double b, double z) {
        if (z >= 0.0 || !isOneFOneAs1336Region(a, b, z)) {
            return NO_REAL_ROUTE;
        }

        try {
            return oneFOneAs1336(b - a, b, -z, a, z);
        } catch (ArithmeticException | IllegalStateException ignored) {
            return NO_REAL_ROUTE;
        }
    }

    private static double oneFOneSmallANegativeBByRatioRoute(double a, double b, double z) {
        if (!(z < 0.0) || !(b < 0.0) || !(Math.abs(a) < 1.0e-2)) {
            return NO_REAL_ROUTE;
        }

        if (sameParameter(b - Math.ceil(b), a)) {
            return scaleByExp(oneFOneImpl(b - a, b, -z), z);
        }
        if (b < -1.0 && maxBForOneFOneSmallANegativeBByRatio(z) < b) {
            try {
                return oneFOneSmallANegativeBByRatio(a, b, z);
            } catch (IllegalStateException ignored) {
                return NO_REAL_ROUTE;
            }
        }
        if (b > -1.0 && b < -0.5) {
            try {
                double first = oneFOneImpl(a, b + 2.0, z);
                double second = oneFOneImpl(a, b + 1.0, z);
                return Recurrences.oneFOneApplyBackwardOnSmallB(a, b, z, 1, first, second, null, null);
            } catch (IllegalStateException ignored) {
                return NO_REAL_ROUTE;
            }
        }

        try {
            return oneFOneAs1336(b - a, b, -z, a, z);
        } catch (ArithmeticException | IllegalStateException ignored) {
            return NO_REAL_ROUTE;
        }
    }

    private static double oneFOneLargeZAsymptoticRoute(double a, double b, double z) {
        return tryOneFOneAsymptotic(a, b, z);
    }

    private static double oneFOneTricomiRoute(double a, double b, double z) {
        if (!isOneFOneSeriesDivergent(a, b, z)) {
            return NO_REAL_ROUTE;
        }
        if (b > 0.0) {
            double zLimit = Math.abs((2.0 * a - b) / Math.sqrt(Math.abs(a)));
            if (z < zLimit && isOneFOneTricomiViablePositiveB(a, b, z)) {
                try {
                    return oneFOneTricomi(a, b, z);
                } catch (IllegalStateException ignored) {
                    return NO_REAL_ROUTE;
                }
            }
            return NO_REAL_ROUTE;
        }
        if (a < 0.0) {
            double zLimit = Math.abs((2.0 * a - b) / Math.sqrt(Math.abs(a)));
            boolean inOuterWindow = ((z < zLimit) || (a > -500.0))
                && ((b > -500.0) || (b - 2.0 * a > 0.0))
                && (z < -a);
            if (inOuterWindow) {
                boolean preferFallback = ((a < b) && (z > -b)) || (z > zLimit);
                if (!preferFallback || b > -1.0) {
                    try {
                        return oneFOneTricomi(a, b, z);
                    } catch (IllegalStateException ignored) {
                        return NO_REAL_ROUTE;
                    }
                }
            }
        }
        return NO_REAL_ROUTE;
    }

    private static double oneFOneDivergentFallbackRoute(double a, double b, double z) {
        if (!isOneFOneSeriesDivergent(a, b, z)) {
            return NO_REAL_ROUTE;
        }
        return oneFOneDifficultRegionRoute(a, b, z);
    }

    private static double oneFOneAs1336PositiveZRoute(double a, double b, double z) {
        if (!isOneFOneAs1336Region(b - a, b, -z)) {
            return NO_REAL_ROUTE;
        }

        try {
            return oneFOneAs1336(a, b, z, b - a, 0.0);
        } catch (ArithmeticException | IllegalStateException ignored) {
            return NO_REAL_ROUTE;
        }
    }

    private static double oneFOneDifficultRegionRoute(double a, double b, double z) {
        if (a > 0.0 && b < 0.0 && z > 0.0) {
            int domain = oneFOneNegativeBRecurrenceRegion(a, b, z);
            if (domain < 0) {
                try {
                    return oneFOneFromFunctionRatioNegativeB(a, b, z);
                } catch (IllegalStateException ignored) {
                }
            } else if (domain > 0) {
                try {
                    return oneFOneFromFunctionRatioNegativeBForwards(a, b, z);
                } catch (IllegalStateException ignored) {
                }
            }
        }
        if (canUseOneFOneNegativeBPositiveZRecurrence(a, b, z)) {
            try {
                return oneFOneBackwardsRecursionOnBForNegativeA(a, b, z);
            } catch (IllegalStateException ignored) {
            }
        }
        if (a < 0.0 || b < 0.0) {
            return oneFOneDivergentFallback(a, b, z);
        }
        return NO_REAL_ROUTE;
    }

    private static double oneFOneLargeAbzRoute(double a, double b, double z) {
        if (!isOneFOneLargeAbzRegion(a, b, z)) {
            return NO_REAL_ROUTE;
        }
        return tryOneFOneLargeSeries(a, b, z);
    }

    private static double oneFOneCheckedSeriesRoute(double a, double b, double z) {
        if (b >= 0.0) {
            return NO_REAL_ROUTE;
        }
        return oneFOneCheckedSeries(a, b, z);
    }

    private static double tryOneFOneAsymptotic(double a, double b, double z) {
        if (!canUseOneFOneAsymptotic(a, b, z)) {
            return NO_REAL_ROUTE;
        }
        try {
            return oneFOneAsymptoticLargeZ(a, b, z);
        } catch (IllegalStateException ignored) {
            return NO_REAL_ROUTE;
        }
    }

    private static boolean isOneFOneLargeAbzRegion(double a, double b, double z) {
        return a > 0.0 && b > 0.0 && z > 0.0 && a * z > ONE_F_ONE_LARGE_ABZ_MIN_PRODUCT;
    }

    private static double tryOneFOneLargeSeries(double a, double b, double z) {
        double directSeriesCost = oneFOneLargeAbzDirectSeriesCost(a, b, z);
        double shiftedSeriesCost = a + (b < z ? z - b : 0.0);
        if (!(b > 1.0) || !(shiftedSeriesCost < directSeriesCost) || ((b <= z) && isNegativeInteger(b - a))) {
            return NO_REAL_ROUTE;
        }
        try {
            return oneFOneLargeSeries(a, b, z);
        } catch (IllegalStateException ignored) {
            return NO_REAL_ROUTE;
        }
    }

    private static double oneFOneLargeAbzDirectSeriesCost(double a, double b, double z) {
        double discriminant = 16.0 * z * (3.0 * a + z) + 9.0 * b * b - 24.0 * b * z;
        return (Math.sqrt(Math.max(0.0, discriminant)) - 3.0 * b + 4.0 * z) / 6.0;
    }

    private static double oneFOneLargeSeries(double a, double b, double z) {
        int aShift = 0;
        int bShift = 0;
        if (a * z > b) {
            aShift = (int) a - ONE_F_ONE_LARGE_ABZ_LOCAL_A;
            bShift = b < z ? (int) (b - z - 1.0) : 0;
        }
        if (aShift < ONE_F_ONE_LARGE_ABZ_LOCAL_A) {
            aShift = 0;
        }
        if (aShift < 0 || aShift > MAX_SERIES_ITERATIONS || Math.abs(bShift) > MAX_SERIES_ITERATIONS) {
            throw new IllegalStateException("oneFOne large_abz shifted series exceeded iteration budget for a=" + a + ", b=" + b + ", z=" + z);
        }

        double aLocal = a - aShift;
        double bLocal = b - bShift;
        double value = oneFOneSeries(aLocal, bLocal, z);
        if (aShift > 0 && sameParameter(aLocal, 0.0)) {
            double next = oneFOneSeries(1.0, bLocal, z);
            if (aShift == 1) {
                value = next;
            } else {
                long[] logScaling = new long[1];
                value = Recurrences.oneFOneApplyForwardOnA(1.0, bLocal, z, aShift - 1, value, next, logScaling, null);
                value = scaleByExp(value, logScaling[0]);
            }
            return oneFOneShiftOnB(value, a, bLocal, z, bShift);
        }

        value = oneFOneShiftOnA(value, aLocal, bLocal, z, aShift);
        return oneFOneShiftOnB(value, a, bLocal, z, bShift);
    }

    private static double oneFOneShiftOnA(double value, double aLocal, double bLocal, double z, int aShift) {
        if (aShift == 0) {
            return value;
        }
        if (!(aLocal > 0.0)) {
            throw new IllegalStateException("oneFOne large_abz shift on a requires aLocal > 0: aLocal=" + aLocal);
        }

        double bRatio = Recurrences.oneFOneBackwardRatioOnB(
            aLocal,
            bLocal,
            z,
            Math.ulp(1.0),
            MAX_SERIES_ITERATIONS,
            ONE_F_ONE_RECURRENCE_RATIO_CONTEXT);
        double next = ((1.0 + aLocal - bLocal) / aLocal) * value + ((bLocal - 1.0) / aLocal) * value / bRatio;
        if (aShift == 1) {
            return next;
        }

        long[] logScaling = new long[1];
        double shifted = Recurrences.oneFOneApplyForwardOnA(aLocal + 1.0, bLocal, z, aShift - 1, value, next, logScaling, null);
        return scaleByExp(shifted, logScaling[0]);
    }

    private static double oneFOneShiftOnB(double value, double a, double bLocal, double z, int bShift) {
        if (bShift == 0) {
            return value;
        }
        if (bShift > 0) {
            throw new IllegalStateException("oneFOne large_abz positive b shifts are unsupported: bShift=" + bShift);
        }

        double previous;
        if (sameParameter(a, bLocal)) {
            previous = -(1.0 - bLocal - z) * value / (bLocal - 1.0);
        } else {
            double bRatio = Recurrences.oneFOneBackwardRatioOnB(
                a,
                bLocal,
                z,
                Math.ulp(1.0),
                MAX_SERIES_ITERATIONS,
                ONE_F_ONE_RECURRENCE_RATIO_CONTEXT);
            previous = value / bRatio;
        }
        if (bShift == -1) {
            return previous;
        }

        long[] logScaling = new long[1];
        double shifted = Recurrences.oneFOneApplyBackwardOnB(a, bLocal - 1.0, z, -(bShift + 1), value, previous, logScaling, null);
        return scaleByExp(shifted, logScaling[0]);
    }

    private static double oneFOneDivergentFallback(double a, double b, double z) {
        if (b > 0.0) {
            if (z < b) {
                return oneFOneBackwardRecurrenceForNegativeA(a, b, z);
            }
            return oneFOneBackwardsRecursionOnBForNegativeA(a, b, z);
        }
        if (b < 0.0) {
            if (a < 0.0) {
                if (canUseOneFOneFromFunctionRatioNegativeAb(a, b, z)) {
                    try {
                        return oneFOneFromFunctionRatioNegativeAb(a, b, z);
                    } catch (IllegalStateException ignored) {
                    }
                }

                boolean canUseRecursion = (z - b + 100.0 < MAX_SERIES_ITERATIONS) && (100.0 - a < MAX_SERIES_ITERATIONS);
                if (canUseRecursion) {
                    double square = 4.0 * a * z + b * b - 2.0 * b * z + z * z;
                    double iterationsToConvergence = square > 0.0 ? 0.5 * (-Math.sqrt(square) - b + z) : -a - b;
                    if (Math.max(a, b) + iterationsToConvergence > -300.0 || a < b) {
                        return oneFOneBackwardsRecursionOnBForNegativeA(a, b, z);
                    }
                }
            }
        }
        return oneFOneSeries(a, b, z);
    }

    private static double oneFOneTricomi(double a, double b, double z) {
        if (sameParameter(b, 2.0 * a)) {
            throw new IllegalStateException("oneFOne Tricomi is unavailable when b = 2a: a=" + a + ", b=" + b + ", z=" + z);
        }

        boolean useLogPrefix = false;
        double prefix = 0.0;
        double prefixLogRemainder = 0.0;
        double prefixSign = 1.0;
        double totalScale = 0.0;

        try {
            prefix = completeExtended(b) * Math.exp(0.5 * z);
            if (prefix == 0.0 || !Double.isFinite(prefix)) {
                useLogPrefix = true;
            }
        } catch (ArithmeticException ex) {
            useLogPrefix = true;
        }

        if (useLogPrefix) {
            prefixLogRemainder = logAbsExtended(b) + 0.5 * z;
            double integralScale = Math.copySign(Math.floor(Math.abs(prefixLogRemainder)), prefixLogRemainder);
            totalScale += integralScale;
            prefixLogRemainder -= integralScale;
            prefixSign = signExtended(b);
        }

        double[] besselCache = new double[ONE_F_ONE_SERIES_CACHE_SIZE];
        double aMinus2 = 1.0;
        double aMinus1 = 0.0;
        double aTerm = 0.5 * b;
        double bMinus1PlusN = b - 1.0;
        double besselArg = (0.5 * b - a) * z;
        double besselRoot = Math.sqrt(Math.abs(besselArg));
        double besselX = 2.0 * besselRoot;
        double mult = 0.5 * z / besselRoot;
        double twoAMinusB = 2.0 * a - b;
        double term = 1.0;
        int seriesN = 2;
        int cacheOffset = -ONE_F_ONE_SERIES_CACHE_SIZE;
        double logScale = 0.0;

        term /= Math.pow(Math.abs(besselArg), 0.5 * bMinus1PlusN);

        double order = bMinus1PlusN - 1.0;
        if (besselArg > 0.0) {
            besselCache[ONE_F_ONE_SERIES_CACHE_SIZE - 1] = Bessel.j(order, besselX);
        } else {
            besselCache[ONE_F_ONE_SERIES_CACHE_SIZE - 1] = Bessel.iExponentiallyWeighted(order, besselX);
            logScale += besselX;
        }
        if (!Double.isFinite(besselCache[ONE_F_ONE_SERIES_CACHE_SIZE - 1])) {
            throw new IllegalStateException("oneFOne Tricomi Bessel seed was non-finite for a=" + a + ", b=" + b + ", z=" + z);
        }
        if (Math.abs(besselCache[ONE_F_ONE_SERIES_CACHE_SIZE - 1]) < Double.MIN_NORMAL / Math.ulp(1.0)) {
            throw new IllegalStateException("oneFOne Tricomi Bessel seed underflowed for a=" + a + ", b=" + b + ", z=" + z);
        }

        double product = Math.abs(term * besselCache[ONE_F_ONE_SERIES_CACHE_SIZE - 1]);
        if (!(Double.isFinite(term)) || product < Double.MIN_NORMAL / (Math.ulp(1.0) * Math.ulp(1.0))) {
            double termLog = -0.5 * bMinus1PlusN * Math.log(Math.abs(besselArg));
            double integralScale = Math.copySign(Math.floor(Math.abs(termLog)), termLog);
            logScale += integralScale;
            term = Math.exp(termLog - integralScale);
        }
        cacheOffset = refillOneFOneTricomiCache(
            besselCache,
            bMinus1PlusN,
            besselArg,
            besselRoot,
            besselX,
            cacheOffset
        );
        totalScale += logScale;

        double sum = 0.0;
        double compensation = 0.0;
        double norm = 0.0;
        boolean converged = false;

        for (int iteration = 0; iteration < MAX_SERIES_ITERATIONS; iteration++) {
            if (seriesN - 2 - cacheOffset >= ONE_F_ONE_SERIES_CACHE_SIZE) {
                cacheOffset = refillOneFOneTricomiCache(
                    besselCache,
                    bMinus1PlusN,
                    besselArg,
                    besselRoot,
                    besselX,
                    cacheOffset
                );
            }

            double contribution = aMinus2 * term * besselCache[seriesN - 2 - cacheOffset];
            term *= mult;
            seriesN += 1;
            double nextA = ((bMinus1PlusN + 2.0) * aMinus1 + twoAMinusB * aMinus2) / seriesN;
            bMinus1PlusN += 1.0;
            aMinus2 = aMinus1;
            aMinus1 = aTerm;
            aTerm = nextA;

            if (aMinus2 != 0.0) {
                if (seriesN - 2 - cacheOffset >= ONE_F_ONE_SERIES_CACHE_SIZE) {
                    cacheOffset = refillOneFOneTricomiCache(
                        besselCache,
                        bMinus1PlusN,
                        besselArg,
                        besselRoot,
                        besselX,
                        cacheOffset
                    );
                }
                contribution += aMinus2 * term * besselCache[seriesN - 2 - cacheOffset];
            }

            term *= mult;
            seriesN += 1;
            nextA = ((bMinus1PlusN + 2.0) * aMinus1 + twoAMinusB * aMinus2) / seriesN;
            bMinus1PlusN += 1.0;
            aMinus2 = aMinus1;
            aMinus1 = aTerm;
            aTerm = nextA;

            if (!Double.isFinite(contribution)) {
                throw new IllegalStateException("oneFOne Tricomi series overflowed for a=" + a + ", b=" + b + ", z=" + z);
            }

            double updated = addCompensated(sum, compensation, contribution);
            compensation = updated - sum - (contribution - compensation);
            sum = updated;
            norm += Math.abs(contribution);

            if (sum == 0.0 || !Double.isFinite(sum)) {
                throw new IllegalStateException("oneFOne Tricomi series produced a non-finite partial sum for a=" + a + ", b=" + b + ", z=" + z);
            }
            if (Math.abs(contribution) <= Math.max(1.0, Math.abs(sum)) * SERIES_EPSILON) {
                converged = true;
                break;
            }
            if (a < 0.0 && b < 0.0 && norm / Math.abs(sum) > 1.0 / ROOT_EPSILON) {
                throw new IllegalStateException("oneFOne Tricomi series lost all digits to cancellation for a=" + a + ", b=" + b + ", z=" + z);
            }
        }

        if (!converged) {
            throw new IllegalStateException("oneFOne Tricomi series did not converge for a=" + a + ", b=" + b + ", z=" + z);
        }

        double sign = Math.signum(sum);
        double logAbs = Math.log(Math.abs(sum)) + totalScale;
        if (useLogPrefix) {
            return signedExp(logAbs + prefixLogRemainder, sign * prefixSign);
        }
        return signedExp(logAbs + Math.log(Math.abs(prefix)), sign * Math.signum(prefix));
    }

    /**
     * Returns the generalized hypergeometric function {@code 2F0(a,b;;z)} on the real scalar branch
     * supported by Boost's public implementation.
     *
     * <p>The generic {@code 2F0} series has zero radius of convergence, so the public real-valued
     * surface still requires at least one upper parameter to be a non-positive integer. Within that
     * terminating domain, this implementation follows Boost's stable branch selection: a Hermite
     * reduction on the half-shifted negative-{@code z} line, a Laguerre reduction when both upper
     * parameters are non-positive integers, a continued fraction on alternating cancellation-heavy
     * cases, and direct terminated summation otherwise.
     *
     * @param a first upper parameter
     * @param b second upper parameter
     * @param z argument
     * @return {@code 2F0(a,b;;z)} on the supported real scalar branch
     */
    public static double twoFZero(double a, double b, double z) {
        validateFinite(a, "twoFZero: a");
        validateFinite(b, "twoFZero: b");
        validateFinite(z, "twoFZero: z");

        if (a == 0.0 || b == 0.0 || z == 0.0) {
            return 1.0;
        }

        boolean aInteger = isInteger(a);
        boolean bInteger = isInteger(b);
        double first = a;
        double second = b;
        boolean firstInteger = aInteger;
        boolean secondInteger = bInteger;

        if (!firstInteger || first > 0.0) {
            first = b;
            second = a;
            firstInteger = bInteger;
            secondInteger = aInteger;
        }
        if (!firstInteger || first > 0.0) {
            throw new IllegalArgumentException("twoFZero: real-valued scalar surface requires at least one upper parameter to be a non-positive integer");
        }

        if (sameParameter(first, second - 0.5) && z < 0.0) {
            return twoFZeroHermite(first, z);
        }
        if (firstInteger && secondInteger) {
            if (first < 1.0 && second <= first) {
                int degree = (int) Math.rint(-first);
                int order = (int) Math.rint(-second) - degree;
                return twoFZeroLaguerre(degree, order, z);
            }
            if (second < 1.0 && first <= second) {
                int degree = (int) Math.rint(-second);
                int order = (int) Math.rint(-first) - degree;
                return twoFZeroLaguerre(degree, order, z);
            }
        }
        if (first * second * z < 0.0 && second < -5.0 && Math.abs(first * second * z) > 0.5) {
            return twoFZeroContinuedFraction(first, second, z);
        }

        return terminatedTwoFZero(first, second, z, terminatingDegree(first));
    }

    /**
     * Returns the real-valued generalized hypergeometric function {@code pFq(a;b;z)}.
     *
    * <p>This generic entry point follows Boost's public checked-series accept domain. It supports
     * the built-in {@code 1F0} inversion for {@code |z| > 1}; checked-series acceptance for
    * {@code p <= q}; and the disk plus convergent unit-circle boundary when {@code p = q + 1}.
    * Within that same domain, the Boost-grounded {@code 1F0}, {@code 0F1}, and {@code 1F1}
    * parameter shapes route
     * to the direct named-family APIs. For terminating cases with non-positive integer lower
     * parameters, evaluation remains valid only when the upper-parameter termination occurs before
     * any lower-parameter pole is reached. No stronger named-family continuation is exposed through
     * this public generic contract.
     *
     * @param upperParameters upper parameters {@code a_i}
     * @param lowerParameters lower parameters {@code b_j}
     * @param z argument
     * @return {@code pFq(a;b;z)}
     */
    public static double pFq(double[] upperParameters, double[] lowerParameters, double z) {
        validateParameterArray(upperParameters, "pFq: upperParameters");
        validateParameterArray(lowerParameters, "pFq: lowerParameters");
        validateFinite(z, "pFq: z");

        PFQ oneFZeroSpecialCase = tryPFqOneFZeroSpecialCase(upperParameters, lowerParameters, z);
        if (oneFZeroSpecialCase != null) {
            return oneFZeroSpecialCase.value();
        }

        double delegated = tryPFqNamedFamilyDelegation(upperParameters, lowerParameters, z);
        if (hasRealRoute(delegated)) {
            return delegated;
        }

        validatePFqCheckedSeriesPublicSurface(upperParameters, lowerParameters, z);
        return pFqSeries(upperParameters, lowerParameters, z, terminatingDegree(upperParameters));
    }

    /**
    * Returns {@code pFq(a;b;z)} and writes a Boost-style checked-series absolute error estimate to
    * {@code absErrorHolder[0]} when provided.

    * <p>When the generic parameter shape matches Boost-grounded direct named families
    * ({@code 1F0}, {@code 0F1}, or {@code 1F1}) inside the existing accept domain, the returned
    * value follows the direct named-family API, while the absolute-error estimate remains anchored
    * to the shared generic checked-series kernel.
     *
     * @param upperParameters upper parameters {@code a_i}
     * @param lowerParameters lower parameters {@code b_j}
     * @param z argument
     * @param absErrorHolder optional single-element output buffer for the absolute-error estimate
     * @return {@code pFq(a;b;z)}
     */
    public static double pFq(double[] upperParameters, double[] lowerParameters, double z, double[] absErrorHolder) {
        validateParameterArray(upperParameters, "pFq: upperParameters");
        validateParameterArray(lowerParameters, "pFq: lowerParameters");
        validateFinite(z, "pFq: z");
        validateHolder(absErrorHolder, "pFq: absErrorHolder");

        PFQ oneFZeroSpecialCase = tryPFqOneFZeroSpecialCase(upperParameters, lowerParameters, z);
        if (oneFZeroSpecialCase != null) {
            if (absErrorHolder != null) {
                absErrorHolder[0] = oneFZeroSpecialCase.absErrorEstimate();
            }
            return oneFZeroSpecialCase.value();
        }

        validatePFqCheckedSeriesPublicSurface(upperParameters, lowerParameters, z);
        double delegated = tryPFqNamedFamilyDelegation(upperParameters, lowerParameters, z);
        PFQ result = pFqCheckedSeries(upperParameters, lowerParameters, z);
        if (absErrorHolder != null) {
            absErrorHolder[0] = result.absErrorEstimate();
        }
        if (hasRealRoute(delegated)) {
            return delegated;
        }
        return result.value();
    }

    static String debugPFqRoute(double[] upperParameters, double[] lowerParameters, double z) {
        validateParameterArray(upperParameters, "debugPFqRoute: upperParameters");
        validateParameterArray(lowerParameters, "debugPFqRoute: lowerParameters");
        validateFinite(z, "debugPFqRoute: z");

        if (tryPFqOneFZeroSpecialCase(upperParameters, lowerParameters, z) != null) {
            return decodePFqRoute(PFQ_ROUTE_ONE_F_ZERO_EXTERIOR_SPECIAL_CASE);
        }

        int namedFamilyRouteId = tryPFqNamedFamilyRouteId(upperParameters, lowerParameters, z);
        if (namedFamilyRouteId != 0) {
            return decodePFqRoute(namedFamilyRouteId);
        }

        validatePFqCheckedSeriesPublicSurface(upperParameters, lowerParameters, z);
        return decodePFqRoute(PFQ_ROUTE_CHECKED_SERIES);
    }

    private static double zeroFOneSeries(double b, double z) {
        double sum = 1.0;
        double compensation = 0.0;
        double term = 1.0;

        for (int n = 0; n < MAX_SERIES_ITERATIONS; n++) {
            term *= z / ((b + n) * (n + 1.0));
            double updated = addCompensated(sum, compensation, term);
            compensation = updated - sum - (term - compensation);
            sum = updated;
            if (Math.abs(term) <= Math.max(1.0, Math.abs(sum)) * SERIES_EPSILON) {
                return sum;
            }
        }

        throw new IllegalStateException("zeroFOne series did not converge for b=" + b + ", z=" + z);
    }

    private static double oneFOneSeries(double a, double b, double z) {
        double sum = 0.0;
        double term = 1.0;
        double upperLimit = Math.sqrt(Double.MAX_VALUE);
        double lowerLimit = 1.0 / upperLimit;
        long logScaling = 0L;
        int seriesIndex = 0;
        long logScalingFactor = (long) LOG_MAX_VALUE - 2L;
        double scalingFactor = Math.exp(logScalingFactor);
        double previousTerm = 0.0;
        long localScaling = 0L;
        boolean smallA = Math.abs(a) < 0.25;
        int convergenceFloor = oneFOneSeriesConvergenceFloor(a, b, z);

        int summitLocation = 0;
        boolean haveMinima = false;
        double discriminant = 4.0 * a * z + b * b - 2.0 * b * z + z * z;
        if (discriminant >= 0.0) {
            double sqrtDiscriminant = Math.sqrt(discriminant);
            double minimaLocation = (-sqrtDiscriminant - b + z) / 2.0;
            if (minimaLocation > 1.0) {
                haveMinima = true;
            }
            double peakLocation = (sqrtDiscriminant - b + z) / 2.0;
            if (peakLocation > 0.0) {
                summitLocation = (int) Math.floor(peakLocation);
            }
        }

        if (summitLocation > MAX_SERIES_ITERATIONS / 4) {
            DoubleI upperPochhammer = logPochhammerSigned(a, summitLocation);
            DoubleI lowerPochhammer = logPochhammerSigned(b, summitLocation);
            double termLog = upperPochhammer.value()
                + summitLocation * Math.log(Math.abs(z))
                - lowerPochhammer.value()
                - Gamma.lgamma(summitLocation + 1.0);
            localScaling = (long) termLog;
            logScaling += localScaling;
            int sign = upperPochhammer.sign() * lowerPochhammer.sign();
            if (z < 0.0 && (summitLocation & 1) == 1) {
                sign = -sign;
            }
            term = sign * Math.exp(termLog - localScaling);
            seriesIndex = summitLocation;
        } else {
            summitLocation = 0;
        }

        double savedTerm = term;
        long savedScale = localScaling;
        double diff;
        boolean terminated = false;

        do {
            sum += term;
            if (Math.abs(sum) >= upperLimit) {
                sum /= scalingFactor;
                term /= scalingFactor;
                logScaling += logScalingFactor;
                localScaling += logScalingFactor;
            }
            if (Math.abs(sum) < lowerLimit) {
                sum *= scalingFactor;
                term *= scalingFactor;
                logScaling -= logScalingFactor;
                localScaling -= logScalingFactor;
            }

            previousTerm = term;
            term *= ((a + seriesIndex) * z) / ((b + seriesIndex) * (seriesIndex + 1.0));
            if (seriesIndex - summitLocation > MAX_SERIES_ITERATIONS) {
                throw new IllegalStateException("oneFOne series did not converge for a=" + a + ", b=" + b + ", z=" + z);
            }
            seriesIndex += 1;
            if (term == 0.0) {
                terminated = true;
                break;
            }
            diff = sum == 0.0 ? Double.POSITIVE_INFINITY : Math.abs(term / sum);
        } while ((diff > SERIES_EPSILON)
            || (Math.abs(previousTerm) < Math.abs(term))
            || (smallA && seriesIndex < 10)
            || (seriesIndex < convergenceFloor));

        if (summitLocation != 0 && !terminated) {
            term = savedTerm * Math.exp(localScaling - savedScale);
            seriesIndex = summitLocation;
            term *= (b + (seriesIndex - 1.0)) * seriesIndex / ((a + (seriesIndex - 1.0)) * z);
            seriesIndex -= 1;

            do {
                sum += term;
                if (seriesIndex == 0) {
                    break;
                }
                if (Math.abs(sum) >= upperLimit) {
                    sum /= scalingFactor;
                    term /= scalingFactor;
                    logScaling += logScalingFactor;
                    localScaling += logScalingFactor;
                }
                if (Math.abs(sum) < lowerLimit) {
                    sum *= scalingFactor;
                    term *= scalingFactor;
                    logScaling -= logScalingFactor;
                    localScaling -= logScalingFactor;
                }

                previousTerm = term;
                term *= (b + (seriesIndex - 1.0)) * seriesIndex / ((a + (seriesIndex - 1.0)) * z);
                if (summitLocation - seriesIndex > MAX_SERIES_ITERATIONS) {
                    throw new IllegalStateException("oneFOne series did not converge for a=" + a + ", b=" + b + ", z=" + z);
                }
                seriesIndex -= 1;
                diff = sum == 0.0 ? Double.POSITIVE_INFINITY : Math.abs(term / sum);
            } while ((diff > SERIES_EPSILON) || (Math.abs(previousTerm) < Math.abs(term)));
        }

        if (haveMinima && seriesIndex != 0 && summitLocation != 0) {
            int backstop = seriesIndex;
            seriesIndex = 0;
            term = Math.exp(-localScaling);
            do {
                sum += term;
                if (Math.abs(sum) >= upperLimit) {
                    sum /= scalingFactor;
                    term /= scalingFactor;
                    logScaling += logScalingFactor;
                }
                if (Math.abs(sum) < lowerLimit) {
                    sum *= scalingFactor;
                    term *= scalingFactor;
                    logScaling -= logScalingFactor;
                }

                term *= ((a + seriesIndex) * z) / ((b + seriesIndex) * (seriesIndex + 1.0));
                if (seriesIndex > MAX_SERIES_ITERATIONS) {
                    throw new IllegalStateException("oneFOne series did not converge for a=" + a + ", b=" + b + ", z=" + z);
                }
                seriesIndex += 1;
                if (term == 0.0) {
                    break;
                }
                if (seriesIndex == backstop) {
                    break;
                }
                diff = sum == 0.0 ? Double.POSITIVE_INFINITY : Math.abs(term / sum);
            } while (diff > SERIES_EPSILON);
        }

        return scaleByExp(sum, logScaling);
    }

    private static int oneFOneSeriesConvergenceFloor(double a, double b, double z) {
        if (b < 0.0 && z > 0.0) {
            return (int) Math.max(0.0, 3.0 - Math.floor(b));
        }
        return 0;
    }

    private static double oneFOneCheckedSeries(double a, double b, double z) {
        PFQ result = pFqCheckedSeries(new double[] {a}, new double[] {b}, z);
        if (result.rawAbsResult() * ROOT_EPSILON > Math.abs(result.rawValue())) {
            throw new IllegalStateException(
                "oneFOne checked series lost more than half the bits to cancellation for a=" + a + ", b=" + b + ", z=" + z);
        }
        return result.value();
    }

    private static double oneFOnePade(double cp, double zp) {
        double z = -zp;
        double zz = z * z;
        double b0 = 1.0;
        double a0 = 1.0;
        double xi1 = 1.0;
        double ct1 = cp + 1.0;
        double cp1 = cp - 1.0;
        double b1 = 1.0 + (z / ct1);
        double a1 = b1 - (z / cp);
        double b2 = 0.0;
        double a2 = 0.0;
        double result = 0.0;

        for (int k = 1; k < MAX_SERIES_ITERATIONS; k++) {
            double ct2 = ct1 * ct1;
            double g1 = 1.0 + ((cp1 / (ct2 + ct1 + ct1)) * z);
            double g2 = ((xi1 / (ct2 - 1.0)) * ((xi1 + cp1) / ct2)) * zz;

            b2 = (g1 * b1) + (g2 * b0);
            a2 = (g1 * a1) + (g2 * a0);

            double previous = result;
            result = a2 / b2;
            if ((Math.abs(result) * Math.ulp(1.0)) > Math.abs(result - previous)) {
                break;
            }

            b0 = b1;
            b1 = b2;
            a0 = a1;
            a1 = a2;
            ct1 += 2.0;
            xi1 += 1.0;
        }

        return a2 / b2;
    }

    private static double oneFOneRational(double a, double b, double z) {
        double internalZ = -z;
        double halfInternalZ = 0.5 * internalZ;

        double ct1 = a * (internalZ / b);
        double ct2 = halfInternalZ / (1.0 + b);
        double xn3 = 0.0;
        double xn2 = 1.0;
        double xn1 = 2.0;
        double xn0 = 3.0;

        double b1 = 1.0;
        double a1 = 1.0;
        double b2 = 1.0 + ((1.0 + a) * (halfInternalZ / b));
        double a2 = b2 - ct1;
        double b3 = 1.0 + ((2.0 + b2) * (((2.0 + a) / 3.0) * ct2));
        double a3 = b3 - ((1.0 + ct2) * ct1);
        ct1 = 3.0;

        double result = 0.0;
        double previousResult = a3 / b3;
        for (int k = 2; k < MAX_SERIES_ITERATIONS; k++) {
            ct2 = (halfInternalZ / ct1) / (b + xn1);
            double g1 = 1.0 + (ct2 * (xn2 - a));
            ct2 *= ((a + xn1) / (b + xn2));
            double g2 = ct2 * ((b - xn1) + (((a + xn0) / (ct1 + 2.0)) * halfInternalZ));
            double g3 = ((ct2 * halfInternalZ) * (((halfInternalZ / ct1) / (ct1 - 2.0)) * (a + xn2) / (b + xn3))) * (a - xn2);

            double b4 = (g1 * b3) + (g2 * b2) + (g3 * b1);
            double a4 = (g1 * a3) + (g2 * a2) + (g3 * a1);

            previousResult = result;
            result = a4 / b4;
            if (result != 0.0 && Math.abs(result) * Math.ulp(1.0) > Math.abs(result - previousResult) / Math.abs(result)) {
                break;
            }

            b1 = b2;
            b2 = b3;
            b3 = b4;
            a1 = a2;
            a2 = a3;
            a3 = a4;

            xn3 = xn2;
            xn2 = xn1;
            xn1 = xn0;
            xn0 += 1.0;
            ct1 += 2.0;
        }

        return result;
    }

    private static double oneFOneAs1336(double a, double b, double z, double bMinusA, double exponentShift) {
        if (!(z > 0.0)) {
            throw new IllegalStateException("oneFOne A&S 13.3.6 requires positive z: " + z);
        }
        if (sameParameter(bMinusA, 0.0)) {
            return scaleByExp(1.0, z + exponentShift);
        }

        double[] besselCache = new double[ONE_F_ONE_SERIES_CACHE_SIZE];
        double halfZ = 0.5 * z;
        double poch1 = 2.0 * bMinusA - 1.0;
        double poch2 = bMinusA - a;
        double bPoch = b;
        double term = 1.0;
        double lastResult = 1.0;
        int sign = 1;
        int seriesN = 0;
        int cacheOffset = -ONE_F_ONE_SERIES_CACHE_SIZE;

        besselCache[ONE_F_ONE_SERIES_CACHE_SIZE - 1] = Bessel.iExponentiallyWeighted(bMinusA - 1.5, halfZ);
        if (!Double.isFinite(besselCache[ONE_F_ONE_SERIES_CACHE_SIZE - 1])) {
            throw new IllegalStateException("oneFOne A&S 13.3.6 Bessel seed overflowed for a=" + a + ", b=" + b + ", z=" + z);
        }
        cacheOffset = refillOneFOneAs1336Cache(besselCache, bMinusA, halfZ, cacheOffset);

        double sum = 0.0;
        double compensation = 0.0;
        boolean converged = false;

        for (int iteration = 0; iteration < MAX_SERIES_ITERATIONS; iteration++) {
            if (seriesN - cacheOffset >= ONE_F_ONE_SERIES_CACHE_SIZE) {
                cacheOffset = refillOneFOneAs1336Cache(besselCache, bMinusA, halfZ, cacheOffset);
            }

            double contribution = term * (bMinusA - 0.5 + seriesN) * sign * besselCache[seriesN - cacheOffset];
            seriesN += 1;
            term *= poch1;
            poch1 = seriesN == 1 ? 2.0 * bMinusA : poch1 + 1.0;
            term *= poch2;
            poch2 += 1.0;
            term /= seriesN;
            term /= bPoch;
            bPoch += 1.0;
            sign = -sign;

            if (seriesN > 100 && Math.abs(contribution) > Math.abs(lastResult)) {
                contribution = 0.0;
            } else {
                lastResult = contribution;
            }

            if (!Double.isFinite(contribution)) {
                throw new IllegalStateException("oneFOne A&S 13.3.6 overflowed for a=" + a + ", b=" + b + ", z=" + z);
            }

            if (contribution == 0.0) {
                converged = true;
                break;
            }

            double updated = addCompensated(sum, compensation, contribution);
            compensation = updated - sum - (contribution - compensation);
            sum = updated;

            if (iteration >= 4 && Math.abs(contribution) <= Math.max(1.0, Math.abs(sum)) * SERIES_EPSILON) {
                converged = true;
                break;
            }
        }

        if (!converged) {
            throw new IllegalStateException("oneFOne A&S 13.3.6 did not converge for a=" + a + ", b=" + b + ", z=" + z);
        }
        if (sum == 0.0) {
            return 0.0;
        }

        double logAbs = logAbsExtended(bMinusA - 0.5)
            + (0.5 - bMinusA) * Math.log(0.25 * z)
            + Math.log(Math.abs(sum))
            + z
            + exponentShift;
        double signFactor = signExtended(bMinusA - 0.5) * Math.signum(sum);
        return signedExp(logAbs, signFactor);
    }

    private static double oneFOneBackwardRecurrenceForNegativeA(double a, double b, double z) {
        long integerPart = (long) a;
        double ak = a - integerPart;
        if (!sameParameter(ak, 0.0)) {
            ak += 2.0;
            integerPart -= 2L;
        }
        if (sameParameter(ak - 1.0, b)) {
            ak -= 1.0;
            integerPart += 1L;
        }
        if (-integerPart > MAX_SERIES_ITERATIONS) {
            throw new IllegalStateException("oneFOne backward recurrence exceeded iteration budget for a=" + a + ", b=" + b + ", z=" + z);
        }

        double first;
        double second;
        long[] logScaling = new long[1];
        if (sameParameter(ak, 0.0)) {
            first = 1.0;
            ak -= 1.0;
            second = 1.0 - z / b;
            if (Math.abs(second) < 0.5) {
                second = (b - z) / b;
            }
        } else {
            first = oneFOneImpl(ak, b, z);
            ak -= 1.0;
            second = oneFOneImpl(ak, b, z);
        }
        integerPart += 1L;

        double recurrenceA = ak;
        int recurrenceSteps = (int) Math.abs(integerPart);
        double result = Recurrences.oneFOneApplyBackwardOnA(recurrenceA, b, z, recurrenceSteps, first, second, logScaling, null);
        return scaleByExp(result, logScaling[0]);
    }

    private static double oneFOneBackwardsRecursionOnBForNegativeA(double a, double b, double z) {
        int bShift = (int) (z - b) + 2;
        int aShift = (int) (-a);
        if (!sameParameter(a + aShift, 0.0)) {
            aShift += 2;
        }
        if (bShift > MAX_SERIES_ITERATIONS || aShift > MAX_SERIES_ITERATIONS) {
            return oneFOneSeries(a, b, z);
        }

        int aBShift = b < 0.0 ? (int) (b + bShift) : bShift;
        int leadingAShift = Math.min(3, aShift);
        if (aBShift > aShift - 3) {
            aBShift = aShift < 3 ? 0 : aShift - 3;
        } else {
            leadingAShift = aShift - aBShift;
        }
        int trailingBShift = bShift - aBShift;
        if (aBShift < 5) {
            if (aBShift > 0) {
                leadingAShift += aBShift;
                trailingBShift += aBShift;
            }
            aBShift = 0;
            leadingAShift -= 1;
        }
        if (trailingBShift == 0 && Math.abs(b) < 0.5 && aBShift > 0) {
            int diff = Math.min(aBShift, 3);
            aBShift -= diff;
            leadingAShift += diff;
            trailingBShift += diff;
        }

        double first = oneFOneImpl(a + aShift, b + bShift, z);
        double second = oneFOneImpl(a + aShift - 1.0, b + bShift, z);
        double[] previous = new double[1];
        long[] logScaling = new long[1];

        double leadingRecurrenceA = a + aShift - 1.0;
        double shiftedB = b + bShift;
        int leadingSteps = leadingAShift;

        second = Recurrences.oneFOneApplyBackwardOnA(leadingRecurrenceA, shiftedB, z, leadingSteps, first, second, logScaling, previous);
        first = previous[0];

        if (aBShift > 0) {
            double localA = a + aShift - leadingAShift - 1.0;
            double localB = b + bShift;
            second = ((1.0 + localA - localB) * second - localA * first) / (1.0 - localB);
            double combinedShiftedB = b + bShift - aBShift;
            int combinedOffset = aBShift - 1;
            second = Recurrences.oneFOneApplyBackwardOnAAndB(a, combinedShiftedB, z, combinedOffset,
                combinedOffset, first, second, logScaling, previous);
            first = previous[0];

            double trailingShiftedB = b + trailingBShift + 1.0;
            first = (second * (trailingShiftedB - 1.0) - a * first) / -(1.0 + a - trailingShiftedB);
        } else {
            double third = -(second * (1.0 + a - b - bShift) - first * a) / (b + bShift - 1.0);
            first = second;
            second = third;
            trailingBShift -= 1;
        }

        if (trailingBShift > 0) {
            second = Recurrences.oneFOneApplyBackwardOnSmallB(a, b, z, trailingBShift,
                first, second, logScaling, null);
        }
        return scaleByExp(second, logScaling[0]);
    }

    private static double oneFOneAsymptoticLargeZ(double a, double b, double z) {
        double logPrefix = z + (a - b) * Math.log(z) + logAbsExtended(b) - logAbsExtended(a);
        double prefixSign = signExtended(b) * signExtended(a);
        if (logPrefix >= LOG_MAX_VALUE) {
            return Math.copySign(Double.POSITIVE_INFINITY, prefixSign);
        }
        if (logPrefix <= LOG_TRUE_MIN) {
            return Math.copySign(0.0, prefixSign);
        }

        double prefix = Math.copySign(Math.exp(logPrefix), prefixSign);
        double sum = 0.0;
        double compensation = 0.0;
        double absSum = 0.0;
        double term = 1.0;
        double oneMinusA = 1.0 - a;
        double bMinusA = b - a;
        double inverseZ = 1.0 / z;
        double lastTerm = 0.0;

        for (int iteration = 0; iteration < MAX_SERIES_ITERATIONS; iteration++) {
            double updated = addCompensated(sum, compensation, term);
            compensation = updated - sum - (term - compensation);
            sum = updated;
            absSum += Math.abs(sum);
            if (Math.abs(sum) * SERIES_EPSILON > Math.abs(term)) {
                return prefix * sum;
            }
            if (absSum > 0.0 && Math.abs(sum) / absSum < SERIES_EPSILON) {
                throw new IllegalStateException("oneFOne asymptotic branch lost all digits to cancellation for a=" + a + ", b=" + b + ", z=" + z);
            }

            lastTerm = term;
            term *= oneMinusA * bMinusA * inverseZ / (iteration + 1.0);
            oneMinusA += 1.0;
            bMinusA += 1.0;
            if (!Double.isFinite(term)) {
                throw new IllegalStateException("oneFOne asymptotic branch overflowed for a=" + a + ", b=" + b + ", z=" + z);
            }
            if (iteration >= 10 && Math.abs(term) > Math.abs(lastTerm)) {
                throw new IllegalStateException("oneFOne asymptotic branch diverged for a=" + a + ", b=" + b + ", z=" + z);
            }
        }

        throw new IllegalStateException("oneFOne asymptotic branch did not converge for a=" + a + ", b=" + b + ", z=" + z);
    }

    private static double pFqSeries(double[] upperParameters, double[] lowerParameters, double z, int terminatingDegree) {
        return pFqCheckedSeries(upperParameters, lowerParameters, z).value();
    }

    private static PFQ tryPFqOneFZeroSpecialCase(double[] upperParameters, double[] lowerParameters, double z) {
        if (upperParameters.length != 1 || lowerParameters.length != 0 || Math.abs(z) <= 1.0) {
            return null;
        }
        if (z > 0.0 && !isInteger(upperParameters[0])) {
            throw new IllegalArgumentException(
                "pFq: 1F0 public generic branch is complex when a is non-integral and z > 1: " + z);
        }

        PFQ transformed = pFqCheckedSeries(upperParameters, lowerParameters, 1.0 / z);
        double multiplier = Math.pow(-z, -upperParameters[0]);
        return new PFQ(transformed.value() * multiplier, transformed.absResult() * multiplier, 0L);
    }

    private static double tryPFqNamedFamilyDelegation(double[] upperParameters, double[] lowerParameters, double z) {
        int routeId = tryPFqNamedFamilyRouteId(upperParameters, lowerParameters, z);
        if (routeId == 0) {
            return NO_REAL_ROUTE;
        }
        if (routeId == PFQ_ROUTE_DELEGATE_ONE_F_ZERO) {
            return oneFZero(upperParameters[0], z);
        }
        if (routeId == PFQ_ROUTE_DELEGATE_ZERO_F_ONE) {
            return zeroFOne(lowerParameters[0], z);
        }
        return oneFOne(upperParameters[0], lowerParameters[0], z);
    }

    private static int tryPFqNamedFamilyRouteId(double[] upperParameters, double[] lowerParameters, double z) {
        if (upperParameters.length == 1 && lowerParameters.length == 0) {
            if (z == 1.0) {
                return 0;
            }
            return PFQ_ROUTE_DELEGATE_ONE_F_ZERO;
        }
        if (upperParameters.length == 0 && lowerParameters.length == 1) {
            return PFQ_ROUTE_DELEGATE_ZERO_F_ONE;
        }
        if (upperParameters.length == 1 && lowerParameters.length == 1) {
            return PFQ_ROUTE_DELEGATE_ONE_F_ONE;
        }
        return 0;
    }

    private static PFQ pFqCheckedSeries(double[] upperParameters, double[] lowerParameters, double z) {
        double result = 1.0;
        double absResult = 1.0;
        double term = 1.0;
        double previousTerm = 0.0;
        double tolerance = SERIES_EPSILON;
        int seriesIndex = 0;
        double upperLimit = Math.sqrt(Double.MAX_VALUE);
        double lowerLimit = 1.0 / upperLimit;
        long logScaling = 0L;
        long localScaling = 0L;
        long logScalingFactor = (long) LOG_MAX_VALUE - 2L;
        double scalingFactor = Math.exp(logScalingFactor);
        boolean haveNoCorrectBits = false;

        while (seriesIndex < BOOST_PUBLIC_PFQ_MAX_SERIES_ITERATIONS) {
            for (double upperParameter : upperParameters) {
                term *= upperParameter + seriesIndex;
            }
            if (term == 0.0) {
                return new PFQ(result, absResult, logScaling);
            }

            for (double lowerParameter : lowerParameters) {
                if (lowerParameter + seriesIndex == 0.0) {
                    throw new IllegalArgumentException(
                        "pFq: one of the lower parameters hits the negative integer pole " + lowerParameter);
                }
                term /= lowerParameter + seriesIndex;
            }

            term *= z;
            seriesIndex += 1;
            term /= seriesIndex;
            result += term;
            absResult += Math.abs(term);

            if (absResult >= upperLimit) {
                result /= scalingFactor;
                absResult /= scalingFactor;
                term /= scalingFactor;
                logScaling += logScalingFactor;
                localScaling += logScalingFactor;
            }
            if (absResult < lowerLimit) {
                result *= scalingFactor;
                absResult *= scalingFactor;
                term *= scalingFactor;
                logScaling -= logScalingFactor;
                localScaling -= logScalingFactor;
            }

            if ((Math.abs(result) * tolerance > Math.abs(term)) && (Math.abs(previousTerm) > Math.abs(term))) {
                break;
            }
            if (absResult * tolerance > Math.abs(result)) {
                if (haveNoCorrectBits) {
                    throw new IllegalStateException(
                        "pFq checked series lost all digits to cancellation for p=" + upperParameters.length
                            + ", q=" + lowerParameters.length + ", z=" + z);
                }
                haveNoCorrectBits = true;
            } else {
                haveNoCorrectBits = false;
            }
            previousTerm = term;
        }

        if (lowerParameters.length > BOOST_PUBLIC_PFQ_MAX_B_TERMS) {
            throw new IllegalArgumentException(
                "pFq: the number of lower parameters must be <= " + BOOST_PUBLIC_PFQ_MAX_B_TERMS + "; got "
                    + lowerParameters.length);
        }

        int[] crossoverLocations = new int[BOOST_PUBLIC_PFQ_MAX_B_TERMS];
        int crossoverCount = setPFqCrossoverLocations(upperParameters, lowerParameters, z, crossoverLocations);
        boolean terminated = false;

        for (int crossoverIndex = 0; crossoverIndex < crossoverCount; crossoverIndex++) {
            int summitLocation = crossoverLocations[crossoverIndex];
            if (seriesIndex >= summitLocation) {
                continue;
            }

            double loopResult = 0.0;
            double loopAbsResult = 0.0;
            long loopScale = 0L;
            double loopErrorScale = 0.0;
            int backstop = seriesIndex;
            int upperSign = 1;
            int lowerSign = 1;
            double termLog = 0.0;

            for (double upperParameter : upperParameters) {
                if (upperParameter < 0.0 && isInteger(upperParameter) && -upperParameter <= summitLocation) {
                    terminated = true;
                    break;
                }

                DoubleI pochhammer = logPochhammerSigned(upperParameter, summitLocation);
                upperSign *= pochhammer.sign();
                termLog += pochhammer.value();
                loopErrorScale = Math.max(loopErrorScale, pochhammer.value());
            }
            if (terminated) {
                break;
            }

            for (double lowerParameter : lowerParameters) {
                DoubleI pochhammer = logPochhammerSigned(lowerParameter, summitLocation);
                lowerSign *= pochhammer.sign();
                termLog -= pochhammer.value();
                loopErrorScale = Math.max(loopErrorScale, pochhammer.value());
            }

            double gammaLog = Gamma.lgamma(summitLocation + 1.0);
            termLog -= gammaLog;
            loopErrorScale = Math.max(loopErrorScale, gammaLog);

            double zLog = summitLocation * Math.log(Math.abs(z));
            termLog += zLog;
            loopErrorScale = Math.max(loopErrorScale, zLog);

            loopErrorScale *= Math.ulp(1.0);
            loopErrorScale = Math.abs(Math.expm1(loopErrorScale));
            loopErrorScale /= Math.ulp(1.0);

            if (z < 0.0 && (summitLocation & 1) == 1) {
                upperSign = -upperSign;
            }

            if (termLog <= LOG_MIN_NORMAL) {
                long scale = (long) Math.floor(termLog - LOG_MIN_NORMAL) - 2L;
                termLog -= scale;
                loopScale += scale;
            }
            if (termLog > 10.0) {
                long scale = (long) Math.floor(termLog);
                termLog -= scale;
                loopScale += scale;
            }

            term = upperSign * lowerSign * Math.exp(termLog);
            seriesIndex = summitLocation;
            double summitTerm = term;
            long savedLoopScale = loopScale;
            double diff = Double.POSITIVE_INFINITY;
            boolean termsAreGrowing = true;
            boolean trivialSmallSeriesCheck = false;

            do {
                loopResult += term;
                loopAbsResult += Math.abs(term);

                if (Math.abs(loopResult) >= upperLimit) {
                    loopResult /= scalingFactor;
                    loopAbsResult /= scalingFactor;
                    term /= scalingFactor;
                    loopScale += logScalingFactor;
                }
                if (Math.abs(loopResult) < lowerLimit) {
                    loopResult *= scalingFactor;
                    loopAbsResult *= scalingFactor;
                    term *= scalingFactor;
                    loopScale -= logScalingFactor;
                }

                double priorTerm = term;
                for (double upperParameter : upperParameters) {
                    term *= upperParameter + seriesIndex;
                }
                if (term == 0.0) {
                    return new PFQ(result, absResult, logScaling);
                }
                for (double lowerParameter : lowerParameters) {
                    if (lowerParameter + seriesIndex == 0.0) {
                        throw new IllegalArgumentException(
                            "pFq: one of the lower parameters hits the negative integer pole " + lowerParameter);
                    }
                    term /= lowerParameter + seriesIndex;
                }
                term *= z / (seriesIndex + 1.0);
                seriesIndex += 1;

                diff = Math.abs(term / loopResult);
                termsAreGrowing = Math.abs(term) > Math.abs(priorTerm);
                if (!trivialSmallSeriesCheck && !termsAreGrowing) {
                    trivialSmallSeriesCheck = true;
                    double relativeTerm;
                    if (loopScale > localScaling) {
                        long rescale = localScaling - loopScale;
                        if (rescale < LOG_MIN_NORMAL) {
                            relativeTerm = 1.0;
                        } else {
                            relativeTerm = Math.abs(term / (result * Math.exp(rescale)));
                        }
                    } else {
                        long rescale = loopScale - localScaling;
                        if (rescale < LOG_MIN_NORMAL) {
                            relativeTerm = 0.0;
                        } else {
                            relativeTerm = Math.abs(term * Math.exp(rescale) / result);
                        }
                    }
                    if (relativeTerm < tolerance) {
                        break;
                    }
                }
            } while ((seriesIndex - summitLocation) < BOOST_PUBLIC_PFQ_MAX_SERIES_ITERATIONS
                && (diff > tolerance || termsAreGrowing));

            int nextBackstop = seriesIndex;
            loopAbsResult += loopErrorScale * Math.abs(loopResult);
            if (loopScale > localScaling) {
                long rescale = localScaling - loopScale;
                localScaling = loopScale;
                logScaling -= rescale;
                double scale = Math.exp(rescale);
                result *= scale;
                absResult *= scale;
                result += loopResult;
                absResult += loopAbsResult;
            } else if (localScaling > loopScale) {
                long rescale = loopScale - localScaling;
                double scale = Math.exp(rescale);
                loopResult *= scale;
                loopAbsResult *= scale;
                result += loopResult;
                absResult += loopAbsResult;
            } else {
                result += loopResult;
                absResult += loopAbsResult;
            }

            seriesIndex = summitLocation;
            term = summitTerm;
            loopResult = 0.0;
            loopAbsResult = 0.0;
            loopScale = savedLoopScale;
            trivialSmallSeriesCheck = false;

            diff = Double.POSITIVE_INFINITY;
            double priorTerm = Double.POSITIVE_INFINITY;
            do {
                seriesIndex -= 1;
                if (seriesIndex == backstop) {
                    break;
                }

                priorTerm = term;
                for (double upperParameter : upperParameters) {
                    term /= upperParameter + seriesIndex;
                }
                for (double lowerParameter : lowerParameters) {
                    if (lowerParameter + seriesIndex == 0.0) {
                        throw new IllegalArgumentException(
                            "pFq: one of the lower parameters hits the negative integer pole " + lowerParameter);
                    }
                    term *= lowerParameter + seriesIndex;
                }
                term *= (seriesIndex + 1.0) / z;
                loopResult += term;
                loopAbsResult += Math.abs(term);

                if (!trivialSmallSeriesCheck && Math.abs(term) < Math.abs(priorTerm)) {
                    trivialSmallSeriesCheck = true;
                    double relativeTerm;
                    if (loopScale > localScaling) {
                        long rescale = localScaling - loopScale;
                        if (rescale < LOG_MIN_NORMAL) {
                            relativeTerm = 1.0;
                        } else {
                            relativeTerm = Math.abs(term / (result * Math.exp(rescale)));
                        }
                    } else {
                        long rescale = loopScale - localScaling;
                        if (rescale < LOG_MIN_NORMAL) {
                            relativeTerm = 0.0;
                        } else {
                            relativeTerm = Math.abs(term * Math.exp(rescale) / result);
                        }
                    }
                    if (relativeTerm < tolerance) {
                        break;
                    }
                }

                if (Math.abs(loopResult) >= upperLimit) {
                    loopResult /= scalingFactor;
                    loopAbsResult /= scalingFactor;
                    term /= scalingFactor;
                    loopScale += logScalingFactor;
                }
                if (Math.abs(loopResult) < lowerLimit) {
                    loopResult *= scalingFactor;
                    loopAbsResult *= scalingFactor;
                    term *= scalingFactor;
                    loopScale -= logScalingFactor;
                }

                diff = Math.abs(term / loopResult);
            } while ((summitLocation - seriesIndex) < BOOST_PUBLIC_PFQ_MAX_SERIES_ITERATIONS
                && (diff > tolerance || Math.abs(term) > Math.abs(priorTerm)));

            loopAbsResult += loopErrorScale * Math.abs(loopResult);
            if (loopScale > localScaling) {
                long rescale = localScaling - loopScale;
                localScaling = loopScale;
                logScaling -= rescale;
                double scale = Math.exp(rescale);
                result *= scale;
                absResult *= scale;
                result += loopResult;
                absResult += loopAbsResult;
            } else if (localScaling > loopScale) {
                long rescale = loopScale - localScaling;
                double scale = Math.exp(rescale);
                loopResult *= scale;
                loopAbsResult *= scale;
                result += loopResult;
                absResult += loopAbsResult;
            } else {
                result += loopResult;
                absResult += loopAbsResult;
            }

            seriesIndex = nextBackstop;
        }

        return new PFQ(result, absResult, logScaling);
    }

    private static int setPFqCrossoverLocations(double[] upperParameters,
                                                double[] lowerParameters,
                                                double z,
                                                int[] crossoverLocations) {
        int locationCount = 0;
        if (upperParameters.length == 1 && lowerParameters.length == 1) {
            double a = upperParameters[0];
            double b = lowerParameters[0];

            double discriminant = 4.0 * a * z + b * b - 2.0 * b * z + z * z;
            if (discriminant >= 0.0) {
                double root = Math.sqrt(discriminant);
                double first = (-root - b + z) / 2.0;
                if (first >= 0.0) {
                    crossoverLocations[locationCount++] = (int) first;
                }
                double second = (root - b + z) / 2.0;
                if (second >= 0.0) {
                    crossoverLocations[locationCount++] = (int) second;
                }
            }

            discriminant = -4.0 * a * z + b * b + 2.0 * b * z + z * z;
            if (discriminant >= 0.0) {
                double root = Math.sqrt(discriminant);
                double first = (-root - b - z) / 2.0;
                if (first >= 0.0) {
                    crossoverLocations[locationCount++] = (int) first;
                }
                double second = (root - b - z) / 2.0;
                if (second >= 0.0) {
                    crossoverLocations[locationCount++] = (int) second;
                }
            }

            java.util.Arrays.sort(crossoverLocations, 0, locationCount);
            switch (locationCount) {
                case 2 -> {
                    crossoverLocations[0] = crossoverLocations[1];
                    locationCount -= 1;
                }
                case 3 -> {
                    crossoverLocations[1] = crossoverLocations[2];
                    locationCount -= 1;
                }
                case 4 -> {
                    crossoverLocations[0] = crossoverLocations[1];
                    crossoverLocations[1] = crossoverLocations[3];
                    locationCount -= 2;
                }
                default -> {
                }
            }
            return locationCount;
        }

        for (int index = 0; index < lowerParameters.length; index++) {
            double lowerParameter = lowerParameters[index];
            crossoverLocations[index] = lowerParameter >= 0.0 ? 0 : (int) (-lowerParameter) + 1;
        }
        java.util.Arrays.sort(crossoverLocations, 0, lowerParameters.length);
        return lowerParameters.length;
    }

    private static DoubleI logPochhammerSigned(double z, int n) {
        if (z + n < 0.0) {
            DoubleI reflected = logPochhammerSigned(-z - n + 1.0, n);
            return DoubleI.signed(reflected.value(), reflected.sign() * (((n & 1) == 1) ? -1 : 1));
        }

        double result = logAbsExtended(z + n) - logAbsExtended(z);
        return DoubleI.signed(result, signExtended(z + n) * signExtended(z));
    }

    private static double oneFOneDoubleUpper(double a, double z) {
        double halfZ = 0.5 * z;
        return completeExtended(a + 0.5)
            * Math.exp(halfZ)
            * Math.pow(0.25 * z, 0.5 - a)
            * Bessel.i(a - 0.5, halfZ);
    }

    private static double terminatedTwoFZero(double a, double b, double z, int degree) {
        double sum = 1.0;
        double compensation = 0.0;
        double term = 1.0;

        for (int n = 0; n < degree; n++) {
            term *= ((a + n) * (b + n) * z) / (n + 1.0);
            double updated = addCompensated(sum, compensation, term);
            compensation = updated - sum - (term - compensation);
            sum = updated;
        }
        return sum;
    }

    private static double twoFZeroHermite(double a, double z) {
        double root = Math.sqrt(-z);
        int degree = (int) Math.rint(-2.0 * a);
        return Math.pow(0.5 * root, degree) * hermitePolynomial(degree, 1.0 / root);
    }

    private static double twoFZeroLaguerre(int degree, int order, double z) {
        return Math.pow(z, degree) * factorial(degree) * generalizedLaguerrePolynomial(degree, order, -1.0 / z);
    }

    private static double twoFZeroContinuedFraction(double a, double b, double z) {
        ContinuedFraction fraction = ContinuedFraction.evaluateFractionB(
            index -> {
                if (index <= 1) {
                    return Double2.fraction(z * a * b, 1.0);
                }
                double shifted = index - 1.0;
                double ratio = z * (a + shifted) * (b + shifted) / (shifted + 1.0);
                return Double2.fraction(-ratio, 1.0 + ratio);
            },
            CONTINUED_FRACTION_EPSILON,
            MAX_SERIES_ITERATIONS);
        if (!fraction.converged()) {
            throw new IllegalStateException(
                "Continued fraction did not converge for twoFZero continued fraction for a=" + a + ", b=" + b + ", z=" + z);
        }
        return fraction.value();
    }

    private static double hermitePolynomial(int degree, double x) {
        if (degree == 0) {
            return 1.0;
        }
        if (degree == 1) {
            return 2.0 * x;
        }

        double previous = 1.0;
        double current = 2.0 * x;
        for (int n = 1; n < degree; n++) {
            double next = 2.0 * x * current - 2.0 * n * previous;
            previous = current;
            current = next;
        }
        return current;
    }

    private static double generalizedLaguerrePolynomial(int degree, int order, double x) {
        if (degree == 0) {
            return 1.0;
        }
        if (degree == 1) {
            return 1.0 + order - x;
        }

        double previous = 1.0;
        double current = 1.0 + order - x;
        for (int n = 1; n < degree; n++) {
            double next = ((2.0 * n + 1.0 + order - x) * current - (n + order) * previous) / (n + 1.0);
            previous = current;
            current = next;
        }
        return current;
    }

    private static void validatePFqCheckedSeriesPublicSurface(double[] upperParameters,
                                                              double[] lowerParameters,
                                                              double z) {
        int p = upperParameters.length;
        int q = lowerParameters.length;
        if (p > q + 1) {
            throw new IllegalArgumentException(
                "pFq: public checked-series surface supports only p <= q + 1; got p=" + p + ", q=" + q);
        }
        if (p != q + 1) {
            return;
        }

        double absZ = Math.abs(z);
        if (absZ < 1.0) {
            return;
        }
        if (absZ > 1.0) {
            throw new IllegalArgumentException(
                "pFq: public checked-series surface requires |z| <= 1 when p = q + 1: " + z);
        }

        double parameterGap = 0.0;
        for (double lowerParameter : lowerParameters) {
            parameterGap += lowerParameter;
        }
        for (double upperParameter : upperParameters) {
            parameterGap -= upperParameter;
        }
        if ((z == 1.0 && !(parameterGap > 0.0)) || (z == -1.0 && !(parameterGap > -1.0))) {
            throw new IllegalArgumentException(
                "pFq: public checked-series surface does not converge on the unit-circle boundary for z=" + z);
        }
    }

    private static int terminatingDegree(double value) {
        if (!isNonPositiveInteger(value)) {
            return -1;
        }
        return (int) Math.rint(-value);
    }

    private static int terminatingDegree(double[] upperParameters) {
        int degree = -1;
        for (double upperParameter : upperParameters) {
            int parameterDegree = terminatingDegree(upperParameter);
            if (parameterDegree >= 0) {
                degree = degree < 0 ? parameterDegree : Math.min(degree, parameterDegree);
            }
        }
        return degree;
    }

    private static double addCompensated(double sum, double compensation, double term) {
        double adjusted = term - compensation;
        return sum + adjusted;
    }

    private static boolean isNonPositiveInteger(double value) {
        return value <= 0.0 && isInteger(value);
    }

    private static boolean isNegativeInteger(double value) {
        return value < 0.0 && isInteger(value);
    }

    private static boolean canUseOneFOneAsymptotic(double a, double b, double z) {
        if (z < ONE_F_ONE_ASYMPTOTIC_MIN_Z || Math.abs(a) < ONE_F_ONE_ASYMPTOTIC_MIN_ABS_A) {
            return false;
        }
        if (isNonPositiveInteger(a) || isNonPositiveInteger(b)) {
            return false;
        }

        double oneMinusA = 1.0 - a;
        double bMinusA = b - a;
        double halfDigits = ONE_F_ONE_ASYMPTOTIC_HALF_DIGITS;
        double ratio = Math.abs((oneMinusA + halfDigits) * (bMinusA + halfDigits) / (halfDigits * z));
        if (ratio >= ONE_F_ONE_ASYMPTOTIC_RATIO_LIMIT) {
            return false;
        }
        if ((oneMinusA < 0.0 || bMinusA < 0.0)
            && Math.abs(oneMinusA * bMinusA / z) > ONE_F_ONE_ASYMPTOTIC_INITIAL_TERM_LIMIT) {
            return false;
        }
        return true;
    }

    private static boolean isOneFOneAs1336Region(double a, double b, double z) {
        if (Math.abs(a) == 0.5) {
            return false;
        }
        if (z < 0.0 && Math.abs(10.0 * a / b) < 1.0 && Math.abs(a) < 50.0) {
            double shrinkage = cylBesselIShrinkageRate(z);
            return Math.abs((2.0 * a - 1.0) * (2.0 * a - b) / b) < 2.0
                && Math.abs(shrinkage * (2.0 * a + 9.0) * (2.0 * a - b + 10.0) / (10.0 * (b + 10.0))) < 0.75;
        }
        return false;
    }

    private static double cylBesselIShrinkageRate(double z) {
        return switch (z) {
            case double v when v < -160.0 -> 1.0;
            case double v when v < -40.0 -> 0.75;
            case double v when v < -20.0 -> 0.5;
            case double v when v < -7.0 -> 0.25;
            case double v when v < -2.0 -> 0.1;
            default -> 0.05;
        };
    }

    private static boolean canUseOneFOneRational(double a, double b, double z) {
        return Math.abs(a * z / b) < 3.5
            && Math.abs(100.0 * z) < Math.abs(b)
            && (Math.abs(a) > 1.0e-2 || b < -5.0);
    }

    private static double maxBForOneFOneSmallANegativeBByRatio(double z) {
        if (z < -998.0) {
            return (2.0 * z) / 3.0;
        }

        double maxB = 0.0;
        for (int index = 0; index < ONE_F_ONE_SMALL_A_NEGATIVE_B_MAX_B_TABLE.length; index += 2) {
            double boundary = ONE_F_ONE_SMALL_A_NEGATIVE_B_MAX_B_TABLE[index + 1];
            if (boundary <= z) {
                break;
            }
            maxB = ONE_F_ONE_SMALL_A_NEGATIVE_B_MAX_B_TABLE[index];
        }
        return maxB;
    }

    private static boolean needKummerReflection(double a, double b, double z) {
        if (z > 0.0) {
            return false;
        }
        if (z < -1.0) {
            return true;
        }
        if (a > 0.0) {
            if (b > 0.0) {
                return Math.abs((a + 10.0) * z / (10.0 * (b + 10.0))) < 1.0;
            }
            return true;
        }
        if (b > 0.0) {
            return false;
        }
        return true;
    }

    private static double oneFOneFloatNextLowerParameter(double b) {
        double adjusted = b + Math.max(Math.ulp(b), 4.0 * INTEGER_EPSILON * Math.max(1.0, Math.abs(b)));
        while (isInteger(adjusted)) {
            adjusted = Math.nextUp(adjusted);
        }
        return adjusted;
    }

    private static boolean isOneFOneSeriesDivergent(double a, double b, double z) {
        boolean seriesIsDivergent = (a + 1.0) * z / (b + 1.0) < -1.0;
        if (seriesIsDivergent && a < 0.0 && b < 0.0 && a > -1.0) {
            seriesIsDivergent = false;
        }
        if (!seriesIsDivergent && a < 0.0 && b < 0.0 && b > a) {
            double convergencePoint = Math.sqrt((a - 1.0) * (a - b)) - a;
            if (-b < convergencePoint) {
                double n = -Math.floor(b);
                seriesIsDivergent = (a + n) * z / ((b + n) * n) < -1.0;
            }
        } else if (!seriesIsDivergent && b < 0.0 && a > 0.0) {
            seriesIsDivergent = true;
        }
        if (seriesIsDivergent && b < -1.0 && b > -5.0 && a > b) {
            seriesIsDivergent = false;
        }
        return seriesIsDivergent;
    }

    private static void refillBackwardBesselCache(double[] cache,
                                                  double orderBase,
                                                  double x,
                                                  double lastValue,
                                                  boolean modifiedBesselI) {
        double order = orderBase + ONE_F_ONE_SERIES_CACHE_SIZE - 1.0;
        double current = arbitrarySmallValue(lastValue);
        double next = modifiedBesselI
            ? Recurrences.besselIBackwardSeedNext(order, x, current)
            : Recurrences.besselJBackwardSeedNext(order, x, current);

        for (int j = ONE_F_ONE_SERIES_CACHE_SIZE - 1; j >= 0; j--) {
            cache[j] = current;
            if (shouldRescaleBesselCache(cache, j)) {
                scaleBesselCache(cache, j, besselCacheRescaleFactor(cache, j));
                order = orderBase + j;
                if (modifiedBesselI) {
                    Recurrences.validateBesselIBackwardOrder(order);
                } else {
                    Recurrences.validateBesselJBackwardPairOrder(order);
                }
                next = cache[j + 1];
                current = cache[j];
            }

            double previous = modifiedBesselI
                ? Recurrences.besselIBackwardPrevious(order, x, current, next)
                : Recurrences.besselJBackwardPrevious(order, x, current, next);
            next = current;
            current = previous;
            order -= 1.0;
        }

        scaleBesselCache(cache, 0, lastValue / current);
    }

    private static boolean shouldRescaleBesselCache(double[] cache, int index) {
        return index < ONE_F_ONE_SERIES_CACHE_SIZE - 2
            && cache[index + 1] != 0.0
            && Double.MAX_VALUE / Math.abs(64.0 * cache[index] / cache[index + 1]) < Math.abs(cache[index]);
    }

    private static double besselCacheRescaleFactor(double[] cache, int index) {
        double rescale = Math.pow(Math.abs(cache[index] / cache[index + 1]), index + 1.0) * 2.0;
        if (!Double.isFinite(rescale)) {
            rescale = Double.MAX_VALUE;
        }
        return 1.0 / rescale;
    }

    private static void scaleBesselCache(double[] cache, int startInclusive, double factor) {
        if (factor == 1.0) {
            return;
        }
        for (int index = startInclusive; index < ONE_F_ONE_SERIES_CACHE_SIZE; index++) {
            cache[index] *= factor;
        }
    }

    private static int refillOneFOneAs1336Cache(double[] besselCache,
                                                double bMinusA,
                                                double halfZ,
                                                int cacheOffset) {
        int nextCacheOffset = cacheOffset + ONE_F_ONE_SERIES_CACHE_SIZE;
        double lastValue = besselCache[ONE_F_ONE_SERIES_CACHE_SIZE - 1];
        refillBackwardBesselCache(besselCache, bMinusA + nextCacheOffset - 0.5, halfZ, lastValue, true);
        return nextCacheOffset;
    }

    private static int refillOneFOneTricomiCache(double[] besselCache,
                                                 double bMinus1PlusN,
                                                 double besselArg,
                                                 double besselRoot,
                                                 double besselX,
                                                 int cacheOffset) {
        int nextCacheOffset = cacheOffset + ONE_F_ONE_SERIES_CACHE_SIZE;
        double lastValue = besselCache[ONE_F_ONE_SERIES_CACHE_SIZE - 1];

        if (besselArg > 0.0) {
            if (bMinus1PlusN > 0.0) {
                refillBackwardBesselCache(besselCache, bMinus1PlusN, besselX, lastValue, false);
            } else {
                int pos;
                if (Math.abs(bMinus1PlusN) <= SERIES_EPSILON) {
                    besselCache[0] = Bessel.j(0.0, besselX);
                    besselCache[1] = Bessel.j(1.0, besselX);
                    pos = 2;
                } else {
                    besselCache[1] = Bessel.j(bMinus1PlusN + 1.0, besselX);
                    besselCache[0] = (lastValue + besselCache[1]) / (bMinus1PlusN / besselRoot);
                    pos = 2;
                }
                while (pos < ONE_F_ONE_SERIES_CACHE_SIZE - 1 && bMinus1PlusN + pos < 0.0) {
                    besselCache[pos + 1] = Bessel.j(bMinus1PlusN + pos + 1.0, besselX);
                    besselCache[pos] = (besselCache[pos - 1] + besselCache[pos + 1])
                        / ((bMinus1PlusN + pos) / besselRoot);
                    pos += 2;
                }
                if (pos < ONE_F_ONE_SERIES_CACHE_SIZE) {
                    double order = bMinus1PlusN + ONE_F_ONE_SERIES_CACHE_SIZE - 1.0;
                    double current = arbitrarySmallValue(besselCache[pos - 1]);
                    double next = Recurrences.besselJBackwardSeedNext(order, besselX, current);
                    for (int loc = ONE_F_ONE_SERIES_CACHE_SIZE - 1; loc >= pos; loc--) {
                        besselCache[loc] = current;
                        double previous = Recurrences.besselJBackwardPrevious(order, besselX, current, next);
                        next = current;
                        current = previous;
                        order -= 1.0;
                    }
                    double ratio = besselCache[pos - 1] / current;
                    if (Math.abs(besselCache[pos] * ratio / besselCache[pos - 1]) > 5.0) {
                        ratio = Bessel.j(bMinus1PlusN + pos, besselX) / besselCache[pos];
                    }
                    scaleBesselCache(besselCache, pos, ratio);
                }
            }
        } else {
            if (bMinus1PlusN > 0.0) {
                refillBackwardBesselCache(besselCache, bMinus1PlusN, besselX, lastValue, true);
            } else {
                double order = bMinus1PlusN;
                double current = Bessel.iExponentiallyWeighted(order, besselX);
                double previous = Recurrences.besselIForwardSeedPrevious(order, besselX, current);
                int pos = 0;
                while (pos < ONE_F_ONE_SERIES_CACHE_SIZE && bMinus1PlusN + pos < 0.5) {
                    besselCache[pos++] = current;
                    double next = Recurrences.besselIForwardNext(order, besselX, previous, current);
                    previous = current;
                    current = next;
                    order += 1.0;
                }
                if (pos < ONE_F_ONE_SERIES_CACHE_SIZE) {
                    double backwardOrder = bMinus1PlusN + ONE_F_ONE_SERIES_CACHE_SIZE - 1.0;
                    double backwardCurrent = arbitrarySmallValue(lastValue);
                    double backwardNext = Recurrences.besselIBackwardSeedNext(backwardOrder, besselX, backwardCurrent);
                    for (int loc = ONE_F_ONE_SERIES_CACHE_SIZE - 1; loc >= pos; loc--) {
                        besselCache[loc] = backwardCurrent;
                        double prior = Recurrences.besselIBackwardPrevious(backwardOrder, besselX, backwardCurrent, backwardNext);
                        backwardNext = backwardCurrent;
                        backwardCurrent = prior;
                        backwardOrder -= 1.0;
                    }
                    scaleBesselCache(besselCache, pos, besselCache[pos - 1] / backwardCurrent);
                }
            }
        }

        if (Math.abs(besselCache[0] / lastValue) > 5.0) {
            double renormalization = (besselArg < 0.0)
                ? Bessel.iExponentiallyWeighted(bMinus1PlusN, besselX) / besselCache[0]
                : Bessel.j(bMinus1PlusN, besselX) / besselCache[0];
            scaleBesselCache(besselCache, 0, renormalization);
        }
        if (!Double.isFinite(besselCache[0])) {
            throw new IllegalStateException("oneFOne Tricomi cache renormalization produced a non-finite Bessel value");
        }
        return nextCacheOffset;
    }

    private static double oneFOneFromFunctionRatioNegativeAb(double a, double b, double z) {
        double ratio = Recurrences.oneFOneBackwardRatioOnB(
            a,
            b + 1.0,
            z,
            Math.ulp(1.0),
            MAX_SERIES_ITERATIONS,
            ONE_F_ONE_RECURRENCE_RATIO_CONTEXT);
        ratio = ((a - b) * ratio + b) / a;

        double m2 = oneFOneImpl(1.0 + a - b, 2.0 - b, z);
        double m3 = oneFOneImpl(2.0 + a - b, 3.0 - b, z);
        double rhs = scaleByExp(1.0 - b, z);
        double lhs = (a - b + 1.0) * z * m3 / (2.0 - b) + (1.0 - b) * m2 - a * z * ratio * m2 / b;
        return rhs / lhs;
    }

    private static boolean canUseOneFOneFromFunctionRatioNegativeAb(double a, double b, double z) {
        return a < 0.0
            && b < 0.0
            && z > 0.0
            && b < a
            && z < -0.25 * b
            && Math.abs(a) < Math.abs(5.0 * b);
    }

    private static double oneFOneFromFunctionRatioNegativeB(double a, double b, double z) {
        double ratio = Recurrences.oneFOneBackwardRatioOnAAndB(
            a + 1.0,
            b + 1.0,
            z,
            0,
            Math.ulp(1.0),
            MAX_SERIES_ITERATIONS,
            ONE_F_ONE_RECURRENCE_RATIO_CONTEXT);
        double m2 = oneFOneImpl(1.0 + a - b, 2.0 - b, z);
        double m3Ratio = Recurrences.oneFOneBackwardRatioOnAAndB(
            2.0 + a - b,
            3.0 - b,
            z,
            0,
            Math.ulp(1.0),
            MAX_SERIES_ITERATIONS,
            ONE_F_ONE_RECURRENCE_RATIO_CONTEXT);
        double m3 = m3Ratio * m2;
        double rhs = scaleByExp(1.0 - b, z);
        double lhs = (a - b + 1.0) * z * m3 / (2.0 - b) + (1.0 - b) * m2 - a * z * ratio * m2 / b;
        return rhs / lhs;
    }

    private static double oneFOneFromFunctionRatioNegativeBForwards(double a, double b, double z) {
        int steps = (int) Math.ceil(-b);
        if (steps <= 0 || steps > MAX_SERIES_ITERATIONS) {
            throw new IllegalStateException("oneFOne forward negative-b ratio exceeded iteration budget for a=" + a + ", b=" + b + ", z=" + z);
        }

        double ratio = 1.0 / Recurrences.oneFOneForwardRatioOnAAndB(
            a,
            b,
            z,
            0,
            Math.ulp(1.0),
            MAX_SERIES_ITERATIONS,
            ONE_F_ONE_RECURRENCE_RATIO_CONTEXT);
        double referenceValue = oneFOneImpl(a + steps, b + steps, z);
        long[] logScaling = new long[1];
        double found = Recurrences.oneFOneApplyForwardOnAAndB(a + 1.0, b + 1.0, z, 0,
            steps - 1, 1.0, ratio, logScaling, null);
        return divideByScaled(referenceValue, found, logScaling[0]);
    }

    private static double oneFOneSmallANegativeBByRatio(double a, double b, double z) {
        int steps = (int) (-b);
        if (steps <= 0 || steps > MAX_SERIES_ITERATIONS) {
            throw new IllegalStateException("oneFOne small-a negative-b ratio exceeded iteration budget for a=" + a + ", b=" + b + ", z=" + z);
        }

        double ratio = Recurrences.oneFOneForwardRatioOnB(
            a,
            b,
            z,
            Math.ulp(1.0),
            MAX_SERIES_ITERATIONS,
            ONE_F_ONE_RECURRENCE_RATIO_CONTEXT);
        double first = 1.0;
        double second = 1.0 / ratio;
        long[] logScaling = new long[1];
        second = Recurrences.oneFOneApplyForwardOnB(a, b + 1.0, z, steps, first, second, logScaling, null);
        double referenceValue = oneFOneImpl(a, b + steps + 1.0, z);
        return divideByScaled(referenceValue, second, logScaling[0]);
    }

    private static boolean isInOneFOneFromFunctionRatioNegativeBRegion(double a, double b, double z) {
        if (!(a > 0.0) || !(b < 0.0) || !(z > 0.0)) {
            return false;
        }
        double logA = Math.log(a);
        if (!(logA >= 0.0)) {
            return false;
        }
        double correction = a < 100.0 ? 5.0 * logA * a / b : 5.0 * Math.sqrt(logA) * a / b;
        double denominator = 4.0 - correction;
        if (!(denominator > 0.0)) {
            return false;
        }
        return z < -b / denominator;
    }

    private static int oneFOneNegativeBRecurrenceRegion(double a, double b, double z) {
        if (!(a > 0.0) || !(b < 0.0) || !(z > 0.0)) {
            return 0;
        }
        if (a <= 1.0) {
            if (-b < 40.0) {
                return 0;
            }
            return oneFOneNegativeBSmallARecurrenceRegion(a, b, z);
        }
        if (isInOneFOneFromFunctionRatioNegativeBRegion(a, b, z)) {
            return -1;
        }
        if (z >= Math.max(1.25, -0.25 * b - 0.5) && Math.ceil(-b) < MAX_SERIES_ITERATIONS) {
            return 1;
        }
        return 0;
    }

    private static int oneFOneNegativeBSmallARecurrenceRegion(double a, double b, double z) {
        double lowerLimit = -0.2 * b + 10.0 * (1.0 - Math.min(a, 1.0));
        if (z <= lowerLimit) {
            return -1;
        }

        double upperLimit = lowerLimit + 19.0;
        if (z > upperLimit && Math.ceil(-b) < MAX_SERIES_ITERATIONS) {
            return 1;
        }
        return 0;
    }

    private static boolean canUseOneFOneNegativeBPositiveZRecurrence(double a, double b, double z) {
        return a <= -10.0
            && b <= -40.0
            && b >= -80.0
            && z >= 20.0
            && z <= -0.6 * b;
    }

    private static boolean isOneFOneTricomiViablePositiveB(double a, double b, double z) {
        if (z < b && a > -50.0) {
            return false;
        }
        if (b <= 100.0) {
            return true;
        }
        double x = Math.sqrt(Math.abs(2.0 * z * b - 4.0 * a * z));
        double v = b - 1.0;
        return Math.log(Math.E * x / (2.0 * v)) * v > LOG_MIN_NORMAL;
    }

    private static double divideByScaled(double numerator, double denominator, long logScale) {
        if (numerator == 0.0) {
            return 0.0;
        }
        double sign = Math.signum(numerator) * Math.signum(denominator);
        double logAbs = Math.log(Math.abs(numerator)) - Math.log(Math.abs(denominator)) - logScale;
        return signedExp(logAbs, sign);
    }

    private static double scaleByExp(double value, double exponent) {
        if (value == 0.0) {
            return value;
        }
        return signedExp(Math.log(Math.abs(value)) + exponent, Math.signum(value));
    }

    private static double signedExp(double logAbs, double sign) {
        if (sign == 0.0) {
            return 0.0;
        }
        if (logAbs >= LOG_MAX_VALUE) {
            return Math.copySign(Double.POSITIVE_INFINITY, sign);
        }
        if (logAbs <= LOG_TRUE_MIN) {
            return Math.copySign(0.0, sign);
        }
        return Math.copySign(Math.exp(logAbs), sign);
    }

    private static double zeroFOneNegativeRealBessel(double b, double z) {
        double sqrtMinusZ = Math.sqrt(-z);
        double bessel = Bessel.j(b - 1.0, 2.0 * sqrtMinusZ);
        if (bessel == 0.0) {
            return 0.0;
        }
        double logAbs = logAbsExtended(b)
            - (b - 1.0) * Math.log(sqrtMinusZ)
            + Math.log(Math.abs(bessel));
        double sign = signExtended(b) * Math.signum(bessel);
        return signedExp(logAbs, sign);
    }

    private static double zeroFOneNegativeRealContinuedFraction(double b, double z) {
        ContinuedFraction fraction = ContinuedFraction.evaluateFractionB(
            index -> {
                if (index <= 1) {
                    return Double2.fraction(z / b, 1.0);
                }
                double shifted = index - 1.0;
                double ratio = z / ((shifted + 1.0) * (b + shifted));
                return Double2.fraction(-ratio, 1.0 + ratio);
            },
            CONTINUED_FRACTION_EPSILON,
            MAX_SERIES_ITERATIONS * 4);
        if (!fraction.converged()) {
            throw new IllegalStateException(
                "Continued fraction did not converge for zeroFOne continued fraction for b=" + b + ", z=" + z);
        }
        return fraction.value();
    }

    private static DoubleI logOneFOneSignedImpl(double a, double b, double z) {
        DoubleI routed = oneFOneRouted(a, b, z);
        double value = routed.value();
        if (Double.isFinite(value)) {
            return signedLogValue(value);
        }

        int routeId = routed._2();
        if (routeId == ONE_F_ONE_ROUTE_KUMMER) {
            DoubleI local = logOneFOneSignedImpl(b - a, b, -z);
            return DoubleI.signed(z + local.value(), local.sign());
        }
        if (routeId == ONE_F_ONE_ROUTE_ASYMPTOTIC) {
            return oneFOneAsymptoticLargeZLogSigned(a, b, z);
        }
        if (routeId == ONE_F_ONE_ROUTE_EXACT) {
            return logOneFOneExactRouteSigned(a, b, z);
        }

        return DoubleI.signed(Math.log(Math.abs(value)), value < 0.0 ? -1 : 1);
    }

    private static DoubleI logOneFOneExactRouteSigned(double a, double b, double z) {
        if (a == 0.0 || z == 0.0) {
            return DoubleI.signed(0.0, 1);
        }
        if (sameParameter(a, b) && !isNegativeInteger(a)) {
            return DoubleI.signed(z, 1);
        }
        if (sameParameter(b - a, -1.0) && Math.abs(a) > 0.5 && !(isNegativeInteger(a) && a > -50.0)) {
            double ratio = (b + z) / b;
            int sign = ratio < 0.0 ? -1 : 1;
            if (ratio == 0.0) {
                return DoubleI.signed(Double.NEGATIVE_INFINITY, sign);
            }
            return DoubleI.signed(z + Math.log(Math.abs(ratio)), sign);
        }
        double value = oneFOneExactOrTerminating(a, b, z);
        return signedLogValue(value);
    }

    private static DoubleI oneFOneAsymptoticLargeZLogSigned(double a, double b, double z) {
        double logPrefix = z + (a - b) * Math.log(z) + logAbsExtended(b) - logAbsExtended(a);
        double prefixSign = signExtended(b) * signExtended(a);
        double sum = 0.0;
        double compensation = 0.0;
        double absSum = 0.0;
        double term = 1.0;
        double oneMinusA = 1.0 - a;
        double bMinusA = b - a;
        double inverseZ = 1.0 / z;
        double lastTerm = 0.0;

        for (int iteration = 0; iteration < MAX_SERIES_ITERATIONS; iteration++) {
            double updated = addCompensated(sum, compensation, term);
            compensation = updated - sum - (term - compensation);
            sum = updated;
            absSum += Math.abs(sum);
            if (Math.abs(sum) * SERIES_EPSILON > Math.abs(term)) {
                int sign = (prefixSign * Math.signum(sum)) < 0.0 ? -1 : 1;
                if (sum == 0.0) {
                    return DoubleI.signed(Double.NEGATIVE_INFINITY, sign);
                }
                return DoubleI.signed(logPrefix + Math.log(Math.abs(sum)), sign);
            }
            if (absSum > 0.0 && Math.abs(sum) / absSum < SERIES_EPSILON) {
                throw new IllegalStateException("oneFOne asymptotic branch lost all digits to cancellation for a=" + a + ", b=" + b + ", z=" + z);
            }

            lastTerm = term;
            term *= oneMinusA * bMinusA * inverseZ / (iteration + 1.0);
            oneMinusA += 1.0;
            bMinusA += 1.0;
            if (!Double.isFinite(term)) {
                throw new IllegalStateException("oneFOne asymptotic branch overflowed for a=" + a + ", b=" + b + ", z=" + z);
            }
            if (iteration >= 10 && Math.abs(term) > Math.abs(lastTerm)) {
                throw new IllegalStateException("oneFOne asymptotic branch diverged for a=" + a + ", b=" + b + ", z=" + z);
            }
        }

        throw new IllegalStateException("oneFOne asymptotic branch did not converge for a=" + a + ", b=" + b + ", z=" + z);
    }

    private static DoubleI signedLogValue(double value) {
        int sign = value < 0.0 ? -1 : 1;
        if (value == 0.0) {
            return DoubleI.signed(Double.NEGATIVE_INFINITY, sign);
        }
        return DoubleI.signed(Math.log(Math.abs(value)), sign);
    }

    private value record PFQ(double rawValue, double rawAbsResult, long logScale) {
        private double value() {
            return scaleByExp(rawValue, logScale);
        }

        private double absResult() {
            return scaleByExp(rawAbsResult, logScale);
        }

        private double absErrorEstimate() {
            return absResult() * Math.ulp(1.0);
        }
    }

    private static double arbitrarySmallValue(double target) {
        return (Double.MIN_NORMAL / Math.ulp(1.0)) * (Math.abs(target) > 1.0 ? target : 1.0);
    }

    private static void validateParameterArray(double[] parameters, String label) {
        if (parameters == null) {
            throw new IllegalArgumentException(label + " must not be null");
        }
        for (double parameter : parameters) {
            validateFinite(parameter, label);
        }
    }

    private static void validateOneFOneParameters(double a, double b) {
        if (isNonPositiveInteger(b) && (a >= 0.0 || a < b || !isInteger(a))) {
            throw new IllegalArgumentException(
                "oneFOne: function is indeterminate for negative integer b unless a is an integral terminating branch with a >= b: a="
                    + a + ", b=" + b);
        }
    }

    private static void validateNonPoleLowerParameter(double b) {
        if (isNonPositiveInteger(b)) {
            throw new IllegalArgumentException("zeroFOne: lower parameter b must not be a non-positive integer: " + b);
        }
    }

}