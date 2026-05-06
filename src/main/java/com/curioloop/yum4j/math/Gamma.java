package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.Commons.factorial;
import static com.curioloop.yum4j.math.Commons.uncheckedFactorial;
import static com.curioloop.yum4j.math.Lanczos13m53.*;
import static com.curioloop.yum4j.math.Commons.cosPi;
import static com.curioloop.yum4j.math.Commons.sinPi;
import static com.curioloop.yum4j.math.Commons.validateProbability;
import static com.curioloop.yum4j.math.HornerPoly.evaluateEvenPolynomial;
import static com.curioloop.yum4j.math.HornerPoly.evaluatePolynomial;

/**
 * Gamma-family special functions used by distribution utilities and analytic pricers.
 *
 * <p>This class provides the complete Gamma function, its logarithm, the lower and
 * upper incomplete gamma functions, the lower and upper regularized incomplete gamma
 * ratios, their inverse problems in {@code x}, and the derivative of the lower
 * regularized ratio with respect to {@code x}.
 *
 * <p>The primary definitions are:
 *
 * <pre>
 * Complete Gamma:
 *     Gamma(a) = integral(t^(a - 1) * exp(-t), t = 0..infinity),  a > 0
 *
 * Log Gamma:
 *     logGamma(a) = log(Gamma(a))
 *
 * Lower incomplete gamma:
 *     gamma(a, x) = integral(t^(a - 1) * exp(-t), t = 0..x)
 *
 * Upper incomplete gamma:
 *     Gamma(a, x) = integral(t^(a - 1) * exp(-t), t = x..infinity)
 *
 * Lower regularized ratio:
 *     P(a, x) = gamma(a, x) / Gamma(a)
 *
 * Upper regularized ratio:
 *     Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)
 *
 * Inverse lower regularized ratio:
 *     gamma_p_inv(a, p) = x such that P(a, x) = p
 *
 * Inverse upper regularized ratio:
 *     gamma_q_inv(a, q) = x such that Q(a, x) = q
 *
 * Derivative of P with respect to x:
 *     dP(a, x)/dx = x^(a - 1) * exp(-x) / Gamma(a)
 *
 * Digamma:
 *     digamma(x) = psi(x) = d(logGamma(x)) / dx
 *
 * Trigamma:
 *     trigamma(x) = psi'(x) = d^2(logGamma(x)) / dx^2
 *
 * Polygamma:
 *     polygamma(n, x) = psi^(n)(x) = d^(n + 1)(logGamma(x)) / dx^(n + 1),  n >= 0
 * </pre>
 *
 * <p>Implementation notes:
 *
 * <ul>
 *   <li>{@link #tgamma(double)} and {@link #lgamma(double)} reuse a Lanczos approximation
 *       for the complete Gamma family.</li>
 *   <li>{@link #gammaP(double, double)} and {@link #gammaQ(double, double)}
 *       follow Boost.Math's double-precision branch strategy for the incomplete gamma ratios.</li>
 *   <li>{@link #gammaPInv(double, double)} and {@link #gammaQInv(double, double)}
 *       solve the inverse-on-{@code x} problems with Boost-style DiDonato-Morris
 *       initial guesses followed by a digits-driven second-order root iteration.</li>
 *   <li>{@link #digamma(double)} and {@link #trigamma(double)} reuse the Lanczos
 *       backbone through analytic differentiation of the same scalar approximation.</li>
 *   <li>{@link #polygamma(int, double)} reuses digamma/trigamma for the first two
 *       orders and evaluates higher orders by combining the exact recurrence with a
 *       Hurwitz-zeta-style Euler-Maclaurin tail.</li>
 *   <li>{@link #tgamma(double, double)} and {@link #tgammaLower(double, double)} expose the
 *       unregularized incomplete-gamma surface while reusing the same Boost-style
 *       branch partition already used for the regularized ratios.</li>
 *   <li>The incomplete-gamma evaluation switches among finite sums, power series,
 *       continued fractions, Temme's large-parameter expansion, and large-{@code x}
 *       asymptotics according to the input region.</li>
 *   <li>{@link #gammaPDerivative(double, double)} returns the exact analytic derivative of
 *       {@code P(a, x)} and preserves Boost-style boundary behavior at {@code x = 0}.</li>
 * </ul>
 *
 * <p>The public API is intentionally scalar and allocation-free so it can be used in
 * tight numerical loops such as non-central chi-square evaluation, CEV pricing, and
 * hybrid analytic kernels.
 */
public final class Gamma {

    private static final double TWO_PI = 2.0 * Math.PI;
    private static final double HALF_LOG_TWO_PI = 0.91893853320467274178;
    private static final double EULER_MASCHERONI = 0.57721566490153286061;
    private static final double SERIES_EPSILON = 1.0e-15;
    private static final double CONTINUED_FRACTION_EPSILON = 1.0e-15;
    private static final double FPMIN = 1.0e-300;
    private static final double EPSILON = Math.ulp(1.0);
    private static final double ROOT_EPSILON = Math.sqrt(Math.ulp(1.0));
    private static final double LOG_MAX_VALUE = Math.log(Double.MAX_VALUE);
    private static final double LOG_TRUE_MIN = Math.log(Double.MIN_VALUE);
    private static final double MIN_SHAPE = Double.MIN_NORMAL;
    private static final double POLYGAMMA_ASYMPTOTIC_THRESHOLD = 10.0;
    private static final int MAX_ITERATIONS = 100000;
    private static final int MAX_INVERSE_ITERATIONS = 256;
    private static final int MAX_DOUBLE_FACTORIAL = 170;
    private static final int DOUBLE_PRECISION_BITS = 53;
    private static final double DOUBLE_DIGITS10_FACTOR = 6.0;
    private static final double[][] POLY_COT_PI_FIXED_COEFFICIENTS = {
        null,
        null,
        null,
        {-2L, -4L},
        {16L, 8L},
        {-16L, -88L, -16L},
        {272L, 416L, 32L},
        {-272L, -2880L, -1824L, -64L},
        {7936L, 24576L, 7680L, 128L},
        {-7936L, -137216L, -185856L, -31616L, -256L},
        {353792L, 1841152L, 1304832L, 128512L, 512L},
        {-353792L, -9061376L, -21253376L, -8728576L, -518656L, -1024L},
        {22368256L, 175627264L, 222398464L, 56520704L, 2084864L, 2048L},
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        {-22368256L, -795300864L, -2868264960L, -2174832640L, -357888000L, -8361984L, -4096L},
        {1903757312L, 21016670208L, 41731645440L, 20261765120L, 2230947840L, 33497088L, 8192L},
        {-1903757312L, -89702612992L, -460858269696L, -559148810240L, -182172651520L, -13754155008L, -134094848L, -16384L},
        {209865342976L, 3099269660672L, 8885192097792L, 7048869314560L, 1594922762240L, 84134068224L, 536608768L, 32768L},
        {-209865342976L, -12655654469632L, -87815735738368L, -155964390375424L, -84842998005760L, -13684856848384L, -511780323328L, -2146926592L, -65536L},
        {29088885112832L, 553753414467584L, 2165206642589696L, 2550316668551168L, 985278548541440L, 115620218667008L, 3100738912256L, 8588754944L, 131072L},
        {-29088885112832L, -2184860175433728L, -19686087844429824L, -48165109676113920L, -39471306959486976L, -11124607890751488L, -965271355195392L, -18733264797696L, -34357248000L, -262144L},
        {4951498053124096L, 118071834535526400L, 603968063567560704L, 990081991141490688L, 584901762421358592L, 122829335169859584L, 7984436548730880L, 112949304754176L, 137433710592L, 524288L}
    };
    private static final Object POLY_COT_PI_TABLE_LOCK = new Object();
    private static volatile double[][] polyCotPiTable = new double[][]{{-1.0}};

    private static final double DIGAMMA_ROOT1 = 1569415565.0 / 1073741824.0;
    private static final double DIGAMMA_ROOT2 = (381566830.0 / 1073741824.0) / 1073741824.0;
    private static final double DIGAMMA_ROOT3 = 0.9016312093258696e-19;
    private static final double DIGAMMA_Y = 0.99558162689208984;
    private static final double[] DIGAMMA_P = {
        0.25479851061131551,
        -0.32555031186804491,
        -0.65031853770896507,
        -0.28919126444774784,
        -0.045251321448739056,
        -0.0020713321167745952
    };
    private static final double[] DIGAMMA_Q = {
        1.0,
        2.0767117023730469,
        1.4606242909763515,
        0.43593529692665969,
        0.054151797245674225,
        0.0021284987017821144,
        -0.55789841321675513e-6
    };
    private static final double[] DIGAMMA_ASYMPTOTIC_P = {
        0.08333333333333333,
        -0.008333333333333333,
        0.003968253968253968,
        -0.004166666666666667,
        0.007575757575757576,
        -0.021092796092796094,
        0.08333333333333333,
        -0.4432598039215686
    };

    private static final double TRIGAMMA_1_2_OFFSET = 2.1093254089355469;
    private static final double[] TRIGAMMA_1_2_P = {
        -1.1093280605946045,
        -3.8310674472619321,
        -3.3703848401898283,
        0.28080574467981213,
        1.6638069578676164,
        0.64468386819102836
    };
    private static final double[] TRIGAMMA_1_2_Q = {
        1.0,
        3.4535389668541151,
        4.5208926987851437,
        2.7012734178351534,
        0.64468798399785611,
        -0.20314516859987728e-6
    };
    private static final double[] TRIGAMMA_2_4_P = {
        -0.13803835004508849e-7,
        0.50000049158540261,
        1.6077979838469348,
        2.5645435828098254,
        2.0534873203680393,
        0.74566981111565923
    };
    private static final double[] TRIGAMMA_2_4_Q = {
        1.0,
        2.8822787662376169,
        4.1681660554090917,
        2.7853527819234466,
        0.74967671848044792,
        -0.00057069112416246805
    };
    private static final double[] TRIGAMMA_4_INF_P = {
        0.68947581948701249e-17,
        0.49999999999998975,
        1.0177274392923795,
        2.498208511343429,
        2.1921221359427595,
        1.5897035272532764,
        0.40154388356961734
    };
    private static final double[] TRIGAMMA_4_INF_Q = {
        1.0,
        1.7021215452463932,
        4.4290431747556469,
        2.9745631894384922,
        2.3013614809773616,
        0.28360399799075752,
        0.022892987908906897
    };

    private static final double[] BERNOULLI_OVER_EVEN_FACTORIAL = {
        1.0 / 12.0,
        -1.0 / 720.0,
        1.0 / 30240.0,
        -1.0 / 1209600.0,
        5.0 / 239500800.0,
        -691.0 / 1307674368000.0,
        7.0 / 523069747200.0,
        -3617.0 / 10670622842880000.0
    };

    private static final double[] TEMME_C0 = {
        -0.33333333333333333,
        0.083333333333333333,
        -0.014814814814814815,
        0.0011574074074074074,
        0.0003527336860670194,
        -0.00017875514403292181,
        0.39192631785224378e-4,
        -0.21854485106799922e-5,
        -0.185406221071516e-5,
        0.8296711340953086e-6,
        -0.17665952736826079e-6,
        0.67078535434014986e-8,
        0.10261809784240308e-7,
        -0.43820360184533532e-8,
        0.91476995822367902e-9
    };

    private static final double[] TEMME_C1 = {
        -0.0018518518518518519,
        -0.0034722222222222222,
        0.0026455026455026455,
        -0.00099022633744855967,
        0.00020576131687242798,
        -0.40187757201646091e-6,
        -0.18098550334489978e-4,
        0.76491609160811101e-5,
        -0.16120900894563446e-5,
        0.46471278028074343e-8,
        0.1378633446915721e-6,
        -0.5752545603517705e-7,
        0.11951628599778147e-7
    };

    private static final double[] TEMME_C2 = {
        0.0041335978835978836,
        -0.0026813271604938272,
        0.00077160493827160494,
        0.20093878600823045e-5,
        -0.00010736653226365161,
        0.52923448829120125e-4,
        -0.12760635188618728e-4,
        0.34235787340961381e-7,
        0.13721957309062933e-5,
        -0.6298992138380055e-6,
        0.14280614206064242e-6
    };

    private static final double[] TEMME_C3 = {
        0.00064943415637860082,
        0.00022947209362139918,
        -0.00046918949439525571,
        0.00026772063206283885,
        -0.75618016718839764e-4,
        -0.23965051138672967e-6,
        0.11082654115347302e-4,
        -0.56749528269915966e-5,
        0.14230900732435884e-5
    };

    private static final double[] TEMME_C4 = {
        -0.0008618882909167117,
        0.00078403922172006663,
        -0.00029907248030319018,
        -0.14638452578843418e-5,
        0.66414982154651222e-4,
        -0.39683650471794347e-4,
        0.11375726970678419e-4
    };

    private static final double[] TEMME_C5 = {
        -0.00033679855336635815,
        -0.69728137583658578e-4,
        0.00027727532449593921,
        -0.00019932570516188848,
        0.67977804779372078e-4,
        0.1419062920643967e-6,
        -0.13594048189768693e-4,
        0.80184702563342015e-5,
        -0.22914811765080952e-5
    };

    private static final double[] TEMME_C6 = {
        0.00053130793646399222,
        -0.00059216643735369388,
        0.00027087820967180448,
        0.79023532326603279e-6,
        -0.81539693675619688e-4,
        0.56116827531062497e-4,
        -0.18329116582843376e-4
    };

    private static final double[] TEMME_C7 = {
        0.00034436760689237767,
        0.51717909082605922e-4,
        -0.00033493161081142236,
        0.0002812695154763237,
        -0.00010976582244684731
    };

    private static final double[] TEMME_C8 = {
        -0.00065262391859530942,
        0.00083949872067208728,
        -0.00043829709854172101
    };

    private static final double[] INVERSE_S_P = {
        3.31125922108741,
        11.6616720288968,
        4.28342155967104,
        0.213623493715853
    };

    private static final double[] INVERSE_S_Q = {
        1.0,
        6.61053765625462,
        6.40691597760039,
        1.27364489782223,
        0.03611708101884203
    };

    private static final double TEMME_WORKSPACE_BASE = -0.00059676129019274625;

    private Gamma() {
    }

    private static double lgammaPositive(double x) {
        if (!(x > 0.0)) {
            throw new IllegalArgumentException("x must be positive: " + x);
        }

        if (isSmallPositiveInteger(x)) {
            return Math.log(smallPositiveIntegerGamma(x));
        }

        if (x < 0.5) {
            return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * x)) - lgammaPositive(1.0 - x);
        }

        double t = x + g() - 0.5;
        return (x - 0.5) * (Math.log(t) - 1.0) + Math.log(lanczosSumExpGScaled(x));
    }

    private static double tgammaPositive(double x) {
        if (isSmallPositiveInteger(x)) {
            return smallPositiveIntegerGamma(x);
        }
        return Math.exp(lgammaPositive(x));
    }

    private static boolean isSmallPositiveInteger(double x) {
        return x > 0.0 && x == Math.rint(x) && x <= MAX_DOUBLE_FACTORIAL + 1.0;
    }

    private static double smallPositiveIntegerGamma(double x) {
        return uncheckedFactorial((int) Math.rint(x) - 1);
    }

    /**
     * Returns the complete Gamma function.
     *
     * <pre>
     * tgamma(x) = Gamma(x)
     *           = integral(t^(x - 1) * exp(-t), t = 0..infinity)
     * </pre>
     *
     * <p>Matches Boost.Math's real scalar surface for finite arguments excluding gamma poles.
     *
     * @param x finite real argument excluding gamma poles
     * @return {@code Gamma(x)}
     */
    public static double tgamma(double x) {
        validateFiniteReal(x, "x");
        ensureNotGammaPole(x, "tgamma");
        return x > 0.0 ? tgammaPositive(x) : completeExtended(x);
    }

    /**
     * Returns the natural logarithm of the absolute Gamma value.
     *
     * <pre>
     * lgamma(x) = log(|Gamma(x)|)
     * </pre>
     *
     * @param x finite argument excluding gamma poles
     * @return {@code log(|Gamma(x)|)}
     */
    public static double lgamma(double x) {
        validateFiniteReal(x, "x");
        ensureNotGammaPole(x, "lgamma");

        double value = x > 0.0
            ? lgammaPositive(x)
            : Math.log(Math.PI) - Math.log(Math.abs(sinPi(x))) - lgammaPositive(1.0 - x);
        return value;
    }

    /**
     * Returns the natural logarithm of the absolute Gamma value together with the sign of {@code Gamma(x)}.
     *
     * @param x finite argument excluding gamma poles
     * @return {@link DoubleI} containing {@code (log(|Gamma(x)|), sign(Gamma(x)))}
     */
    public static DoubleI lgammaSigned(double x) {
        validateFiniteReal(x, "x");
        ensureNotGammaPole(x, "lgamma");

        double sp = sinPi(x);
        int sign = x > 0.0 ? 1 : (sp < 0.0 ? -1 : 1);
        double value = x > 0.0
            ? lgammaPositive(x)
            : Math.log(Math.PI) - Math.log(Math.abs(sp)) - lgammaPositive(1.0 - x);
        return DoubleI.signed(value, sign);
    }

    /**
     * Returns {@code Gamma(1 + z) - 1} with cancellation-safe behavior near {@code z = 0}.
     *
     * @param z finite argument such that {@code 1 + z} is not a gamma pole
     * @return {@code Gamma(1 + z) - 1}
     */
    public static double tgamma1pm1(double z) {
        validateFiniteReal(z, "z");
        double shifted = 1.0 + z;
        ensureNotGammaPole(shifted, "tgamma1pm1");

        if (isSmallPositiveInteger(shifted)) {
            return smallPositiveIntegerGamma(shifted) - 1.0;
        }

        if (shifted > 0.0 && z > -1.0 && z < 2.0) {
            return Math.expm1(lgammaPositive(shifted));
        }
        return completeExtended(shifted) - 1.0;
    }

    /**
     * Returns the safe ratio {@code Gamma(z) / Gamma(z + delta)} for positive arguments.
     *
     * @param z numerator gamma argument, must be strictly positive
     * @param delta offset applied to the denominator argument, must keep {@code z + delta > 0}
     * @return {@code Gamma(z) / Gamma(z + delta)}
     */
    public static double tgammaDeltaRatio(double z, double delta) {
        validateFiniteReal(z, "z");
        validateFiniteReal(delta, "delta");
        if (!(z > 0.0) || !((z + delta) > 0.0)) {
            throw new IllegalArgumentException("tgammaDeltaRatio requires z > 0 and z + delta > 0");
        }
        return tgammaDeltaRatioUnchecked(z, delta);
    }

    /**
     * Returns the safe ratio {@code Gamma(a) / Gamma(b)} for positive arguments.
     *
     * @param a numerator gamma argument, must be strictly positive
     * @param b denominator gamma argument, must be strictly positive
     * @return {@code Gamma(a) / Gamma(b)}
     */
    public static double tgammaRatio(double a, double b) {
        validateFiniteReal(a, "a");
        validateFiniteReal(b, "b");
        if (!(a > 0.0) || !(b > 0.0)) {
            throw new IllegalArgumentException("tgammaRatio requires a > 0 and b > 0");
        }
        return tgammaDeltaRatioUnchecked(a, b - a);
    }

    /**
     * Returns the lower incomplete gamma function.
     *
     * <pre>
     * tgammaLower(a, x) = gamma(a, x)
     *                   = integral(t^(a - 1) * exp(-t), t = 0..x)
     * </pre>
     *
     * @param a shape parameter, must be positive
     * @param x evaluation point, must be non-negative
     * @return lower incomplete gamma value {@code gamma(a, x)}
     */
    public static double tgammaLower(double a, double x) {
        validateArguments(a, x);

        if (x == 0.0) {
            return 0.0;
        }
        if (Double.isInfinite(x)) {
            return tgammaPositive(a);
        }

        return incomplete(a, x, false, false);
    }

    /**
     * Returns the upper incomplete gamma complement.
     *
     * <pre>
     * tgamma(a, x) = Gamma(a, x)
     *              = integral(t^(a - 1) * exp(-t), t = x..infinity)
     * </pre>
     *
     * @param a shape parameter, must be positive
     * @param x evaluation point, must be non-negative
     * @return upper incomplete gamma complement {@code Gamma(a, x)}
     */
    public static double tgamma(double a, double x) {
        validateArguments(a, x);

        if (x == 0.0) {
            return tgammaPositive(a);
        }
        if (Double.isInfinite(x)) {
            return 0.0;
        }

        return incomplete(a, x, true, false);
    }

    /**
     * Returns the lower regularized incomplete gamma ratio.
     *
     * <pre>
     * gammaP(a, x) = P(a, x)
     *              = gamma(a, x) / Gamma(a)
     *              = (1 / Gamma(a)) * integral(t^(a - 1) * exp(-t), t = 0..x)
     * </pre>
     *
     * @param a shape parameter, must be positive
     * @param x evaluation point, must be non-negative
     * @return {@code P(a, x)}
     */
    public static double gammaP(double a, double x) {
        validateArguments(a, x);

        if (x == 0.0) {
            return 0.0;
        }
        if (Double.isInfinite(x)) {
            return 1.0;
        }

        return incomplete(a, x, false, true);
    }

    /**
     * Returns the upper regularized incomplete gamma ratio.
     *
     * <pre>
     * gammaQ(a, x) = Q(a, x)
     *              = Gamma(a, x) / Gamma(a)
     *              = (1 / Gamma(a)) * integral(t^(a - 1) * exp(-t), t = x..infinity)
     *              = 1 - P(a, x)
     * </pre>
     *
     * @param a shape parameter, must be positive
     * @param x evaluation point, must be non-negative
     * @return {@code Q(a, x)}
     */
    public static double gammaQ(double a, double x) {
        validateArguments(a, x);

        if (x == 0.0) {
            return 1.0;
        }
        if (Double.isInfinite(x)) {
            return 0.0;
        }

        return incomplete(a, x, true, true);
    }

    /**
     * Solves for {@code x} in the lower regularized incomplete gamma equation.
     *
     * <pre>
     * Find x such that P(a, x) = p
     * </pre>
     *
     * @param a shape parameter, must be positive
     * @param p target lower-tail probability in {@code [0, 1]}
     * @return solution {@code x in [0, +infinity]}
     */
    public static double gammaPInv(double a, double p) {
        validateShape(a);
        validateProbability(p, "p");

        if (p == 0.0) {
            return 0.0;
        }
        if (p == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (a == 1.0) {
            return -Math.log1p(-p);
        }

        return inverseRegularizedBoost(a, p, false);
    }

    /**
     * Solves for {@code x} in the upper regularized incomplete gamma equation.
     *
     * <pre>
     * Find x such that Q(a, x) = q
     * </pre>
     *
     * @param a shape parameter, must be positive
     * @param q target upper-tail probability in {@code [0, 1]}
     * @return solution {@code x in [0, +infinity]}
     */
    public static double gammaQInv(double a, double q) {
        validateShape(a);
        validateProbability(q, "q");

        if (q == 1.0) {
            return 0.0;
        }
        if (q == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (a == 1.0) {
            return -Math.log(q);
        }

        return inverseRegularizedBoost(a, q, true);
    }

    /**
     * Solves for {@code a} in the lower regularized incomplete gamma equation.
     *
     * @param x evaluation point, must be strictly positive
     * @param p target lower-tail probability in {@code [0, 1]}
     * @return shape parameter {@code a}
     */
    public static double gammaPInva(double x, double p) {
        validateInverseShapeArgument(x, "gammaPInva");
        validateProbability(p, "p");

        if (p == 0.0) {
            throw new ArithmeticException("gammaPInva overflow: p must be greater than zero");
        }
        if (p == 1.0) {
            return MIN_SHAPE;
        }
        return inverseRegularizedShape(x, p, 1.0 - p, "gammaPInva");
    }

    /**
     * Solves for {@code a} in the upper regularized incomplete gamma equation.
     *
     * @param x evaluation point, must be strictly positive
     * @param q target upper-tail probability in {@code [0, 1]}
     * @return shape parameter {@code a}
     */
    public static double gammaQInva(double x, double q) {
        validateInverseShapeArgument(x, "gammaQInva");
        validateProbability(q, "q");

        if (q == 1.0) {
            throw new ArithmeticException("gammaQInva overflow: q must be less than one");
        }
        if (q == 0.0) {
            return MIN_SHAPE;
        }
        return inverseRegularizedShape(x, 1.0 - q, q, "gammaQInva");
    }

    /**
     * Returns the derivative of the lower regularized incomplete gamma ratio with
     * respect to {@code x}.
     *
     * <pre>
     * gammaPDerivative(a, x) = dP(a, x)/dx
     *                        = x^(a - 1) * exp(-x) / Gamma(a)
     * </pre>
     *
     * <p>At {@code x = 0}, the limiting behavior is:
     *
     * <pre>
     * a > 1  -> 0
     * a = 1  -> 1
     * a < 1  -> +infinity
     * </pre>
     *
     * @param a shape parameter, must be positive
     * @param x evaluation point, must be non-negative
     * @return {@code dP(a, x)/dx}
     */
    public static double gammaPDerivative(double a, double x) {
        validateArguments(a, x);

        return gammaPDerivativeUnchecked(a, x);
    }

    private static double gammaPDerivativeUnchecked(double a, double x) {
        if (x == 0.0) {
            if (a > 1.0) {
                return 0.0;
            }
            if (a == 1.0) {
                return 1.0;
            }
            return Double.POSITIVE_INFINITY;
        }

        double prefix = regularizedIncompletePrefix(a, x);
        if ((x < 1.0) && (Double.MAX_VALUE * x < prefix)) {
            return Double.POSITIVE_INFINITY;
        }
        if (prefix == 0.0) {
            double logDerivative = (a - 1.0) * Math.log(x) - x - lgammaPositive(a);
            if (logDerivative <= LOG_TRUE_MIN) {
                return 0.0;
            }
            return Math.exp(logDerivative);
        }
        return prefix / x;
    }

    /**
     * Returns the digamma function, the first derivative of {@code log(Gamma(x))}.
     *
     * <pre>
     * digamma(x) = psi(x) = d(logGamma(x)) / dx
     * </pre>
     *
     * @param x strictly positive argument
     * @return {@code psi(x)}
     */
    public static double digamma(double x) {
        validateFiniteReal(x, "x");
        ensureNotGammaPole(x, "digamma");

        double result = 0.0;
        if (x <= -1.0) {
            x = 1.0 - x;
            double remainder = x - Math.floor(x);
            if (remainder > 0.5) {
                remainder -= 1.0;
            }
            if (remainder == 0.0) {
                throw new ArithmeticException("digamma pole at non-positive integer");
            }
            result = Math.PI / Math.tan(Math.PI * remainder);
        }
        if (x == 0.0) {
            throw new ArithmeticException("digamma pole at zero");
        }
        if (x >= POLYGAMMA_ASYMPTOTIC_THRESHOLD) {
            return result + digammaLarge(x);
        }
        while (x > 2.0) {
            x -= 1.0;
            result += 1.0 / x;
        }
        while (x < 1.0) {
            result -= 1.0 / x;
            x += 1.0;
        }
        return result + digammaOneTwo(x);
    }

    /**
     * Returns the trigamma function, the derivative of {@link #digamma(double)}.
     *
     * <pre>
     * trigamma(x) = psi'(x) = d^2(logGamma(x)) / dx^2
     * </pre>
     *
     * @param x strictly positive argument
     * @return {@code psi'(x)}
     */
    public static double trigamma(double x) {
        validateFiniteReal(x, "x");
        ensureNotGammaPole(x, "trigamma");

        if (x <= 0.0) {
            double shifted = 1.0 - x;
            double sine = Math.sin(Math.PI * x);
            if (sine == 0.0) {
                throw new ArithmeticException("trigamma pole at non-positive integer");
            }
            return -trigammaPositive(shifted) + (Math.PI * Math.PI) / (sine * sine);
        }
        return trigammaPositive(x);
    }

    /**
     * Returns the order-{@code n} polygamma function.
     *
     * <pre>
     * polygamma(n, x) = psi^(n)(x) = d^(n + 1)(logGamma(x)) / dx^(n + 1)
     * </pre>
     *
     * <p>By convention, {@code polygamma(0, x)} is the digamma function and
     * {@code polygamma(1, x)} is the trigamma function.
     *
     * @param order derivative order, must be non-negative
     * @param x finite argument, excluding integer singularities for orders {@code >= 2}
     * @return order-{@code n} polygamma value
     */
    public static double polygamma(int order, double x) {
        validatePolygammaOrder(order);
        validateFiniteReal(x, "x");

        if (order == 0) {
            return digamma(x);
        }
        if (order == 1) {
            return trigamma(x);
        }

        if (x < 0.0) {
            if (x == Math.rint(x)) {
                double half = x * 0.5;
                if (half != Math.rint(half)) {
                    throw new ArithmeticException("polygamma overflow at negative odd integer: " + x);
                }
                throw new ArithmeticException("polygamma pole at negative integer: " + x);
            }
            double z = 1.0 - x;
            double result = polygammaPositive(order, z) + Math.PI * polyCotPi(order, z, x);
            return ((order & 1) != 0) ? -result : result;
        }
        if (x == 0.0) {
            throw new ArithmeticException("polygamma pole at zero");
        }

        return polygammaPositive(order, x);
    }

    private static double polygammaPositive(int order, double x) {

        double smallXLimit = Math.min(5.0 / order, 0.25);
        if (x < smallXLimit) {
            return polygammaNearZero(order, x);
        }
        if (x > DOUBLE_DIGITS10_FACTOR + 4.0 * order) {
            return polygammaAtInfinity(order, x);
        }

        double sign = ((order & 1) == 0) ? -1.0 : 1.0;
        double factorial = factorial(order);
        if (x == 1.0) {
            return sign * factorial * riemannZeta(order + 1);
        }
        if (x == 0.5) {
            double result = sign * factorial * riemannZeta(order + 1);
            if (Math.abs(result) >= Math.scalb(Double.MAX_VALUE, -order - 1)) {
                throw new ArithmeticException("polygamma overflow at x = 0.5");
            }
            return result * (Math.scalb(1.0, order + 1) - 1.0);
        }
        return polygammaAtTransition(order, x);
    }

    private static double polyCotPi(int order, double x, double xc) {
        double s = Math.abs(x) < Math.abs(xc) ? sinPi(x) : sinPi(xc);
        double c = cosPi(x);
        double pi = Math.PI;

        switch (order) {
            case 1:
                return -pi / (s * s);
            case 2:
                return 2.0 * pi * pi * c / Math.pow(s, 3.0);
            default:
                break;
        }

        if (order <= 20) {
            return polyCotPiFixed(order, s, c, evaluateEvenPolynomial(POLY_COT_PI_FIXED_COEFFICIENTS[order], c));
        }

        int index = order - 1;
        double sum = HornerPoly.evaluateEvenPolynomial(polyCotPiCoefficients(index), c);
        if ((index & 1) != 0) {
            sum *= c;
        }
        if (sum == 0.0) {
            return 0.0;
        }
        if (s == 0.0) {
            throw new ArithmeticException("polygamma overflow at reflection singularity");
        }

        double powerTerms = order * Math.log(pi);
        powerTerms -= Math.log(Math.abs(s)) * (order + 1.0);
        powerTerms += lgamma(order);
        powerTerms += Math.log(Math.abs(sum));
        if (powerTerms > LOG_MAX_VALUE) {
            throw new ArithmeticException("polygamma overflow at reflection singularity");
        }

        double sign = Math.signum(sum);
        if ((s < 0.0) && (((order + 1) & 1) != 0)) {
            sign = -sign;
        }
        return Math.exp(powerTerms) * sign;
    }

    private static double polyCotPiFixed(int order, double s, double c, double sum) {
        double numerator = ((order & 1) == 0) ? sum * c : sum;
        return Math.pow(Math.PI, order) * numerator / Math.pow(s, order + 1.0);
    }

    private static double[] polyCotPiCoefficients(int index) {
        double[][] table = polyCotPiTable;
        if (index < table.length) {
            return table[index];
        }

        synchronized (POLY_COT_PI_TABLE_LOCK) {
            table = polyCotPiTable;
            if (index >= table.length) {
                double[][] expanded = new double[index + 1][];
                System.arraycopy(table, 0, expanded, 0, table.length);
                for (int i = table.length - 1; i < index; i++) {
                    int offset = i & 1;
                    int sinOrder = i + 2;
                    int maxCosOrder = sinOrder - 1;
                    int maxColumns = (maxCosOrder - offset) / 2;
                    int nextOffset = offset != 0 ? 0 : 1;
                    int nextMaxColumns = (maxCosOrder + 1 - nextOffset) / 2;
                    double[] next = new double[nextMaxColumns + 1];
                    double[] current = expanded[i];
                    for (int column = 0; column <= maxColumns; column++) {
                        int cosOrder = 2 * column + offset;
                        next[(cosOrder + 1) / 2] += ((double) (cosOrder - sinOrder) * current[column]) / (sinOrder - 1.0);
                        if (cosOrder != 0) {
                            next[(cosOrder - 1) / 2] += (-(double) cosOrder * current[column]) / (sinOrder - 1.0);
                        }
                    }
                    expanded[i + 1] = next;
                }
                polyCotPiTable = expanded;
                table = expanded;
            }
            return table[index];
        }
    }

    private static double inverseRegularizedBoost(double a, double probability, boolean complement) {
        double p = complement ? 1.0 - probability : probability;
        double q = complement ? probability : 1.0 - probability;
        // Boost threads a has_10_digits flag out of find_inverse_gamma and uses it only
        // for low-precision types (digits <= 36). This port is double-only, so that
        // early-return branch is unreachable and we keep just the initial guess value.
        double guess = findInverseGamma(a, p, q);
        double normalizedTarget = probability;
        boolean normalizedInvert = complement;
        if (normalizedTarget > 0.9) {
            normalizedTarget = 1.0 - normalizedTarget;
            normalizedInvert = !normalizedInvert;
        }

        double lower = Double.MIN_NORMAL;
        if (guess <= lower) {
            guess = lower;
        }
        int digits = gammaInverseDigits(a, guess, complement);
        double target = normalizedTarget;
        boolean invert = normalizedInvert;
        return RootIterators.halleyIterate((root, evaluation) -> gammaInverseRoot(a, target, invert, root, evaluation),
            guess,
            lower,
            Double.MAX_VALUE,
            digits,
            MAX_INVERSE_ITERATIONS,
            "gammaInv"
        );
    }

    private static int gammaInverseDigits(double a, double guess, boolean complement) {
        int digits = DOUBLE_PRECISION_BITS;
        if (digits < 30) {
            digits *= 2;
            digits /= 3;
        } else {
            digits /= 2;
            digits -= 1;
        }
        if ((a < 0.125) && (Math.abs(gammaPDerivativeUnchecked(a, guess)) > 1.0 / Math.sqrt(EPSILON))) {
            return complement ? DOUBLE_PRECISION_BITS : DOUBLE_PRECISION_BITS - 2;
        }
        return digits;
    }

    private static double findInverseGamma(double a, double p, double q) {
        if (a == 1.0) {
            return -Math.log(q);
        }

        if (a < 1.0) {
            double g = tgamma(a);
            double b = q * g;
            if ((b > 0.6) || ((b >= 0.45) && (a >= 0.3))) {
                double u;
                if ((b * q > 1.0e-8) && (q > 1.0e-5)) {
                    u = Math.pow(p * g * a, 1.0 / a);
                } else {
                    u = Math.exp((-q / a) - EULER_MASCHERONI);
                }
                return u / (1.0 - (u / (a + 1.0)));
            }
            if ((a < 0.3) && (b >= 0.35)) {
                double t = Math.exp(-EULER_MASCHERONI - b);
                double u = t * Math.exp(t);
                return t * Math.exp(u);
            }
            if ((b > 0.15) || (a >= 0.3)) {
                double y = -Math.log(b);
                double u = y - (1.0 - a) * Math.log(y);
                return y - (1.0 - a) * Math.log(u) - Math.log(1.0 + (1.0 - a) / (1.0 + u));
            }
            if (b > 0.1) {
                double y = -Math.log(b);
                double u = y - (1.0 - a) * Math.log(y);
                return y - (1.0 - a) * Math.log(u)
                    - Math.log((u * u + 2.0 * (3.0 - a) * u + (2.0 - a) * (3.0 - a)) / (u * u + (5.0 - a) * u + 2.0));
            }

            double y = -Math.log(b);
            double c1 = (a - 1.0) * Math.log(y);
            double c1_2 = c1 * c1;
            double c1_3 = c1_2 * c1;
            double c1_4 = c1_2 * c1_2;
            double a2 = a * a;
            double a3 = a2 * a;

            double c2 = (a - 1.0) * (1.0 + c1);
            double c3 = (a - 1.0) * (-(c1_2 / 2.0) + (a - 2.0) * c1 + (3.0 * a - 5.0) / 2.0);
            double c4 = (a - 1.0) * ((c1_3 / 3.0) - (3.0 * a - 5.0) * c1_2 / 2.0 + (a2 - 6.0 * a + 7.0) * c1 + (11.0 * a2 - 46.0 * a + 47.0) / 6.0);
            double c5 = (a - 1.0) * (-(c1_4 / 4.0)
                + (11.0 * a - 17.0) * c1_3 / 6.0
                + (-3.0 * a2 + 13.0 * a - 13.0) * c1_2
                + (2.0 * a3 - 25.0 * a2 + 72.0 * a - 61.0) * c1 / 2.0
                + (25.0 * a3 - 195.0 * a2 + 477.0 * a - 379.0) / 12.0);

            double y2 = y * y;
            double y3 = y2 * y;
            double y4 = y2 * y2;
            return y + c1 + (c2 / y) + (c3 / y2) + (c4 / y3) + (c5 / y4);
        }

        double s = findInverseS(p, q);
        double s2 = s * s;
        double s3 = s2 * s;
        double s4 = s2 * s2;
        double s5 = s4 * s;
        double ra = Math.sqrt(a);
        double w = a + s * ra + (s2 - 1.0) / 3.0;
        w += (s3 - 7.0 * s) / (36.0 * ra);
        w -= (3.0 * s4 + 7.0 * s2 - 16.0) / (810.0 * a);
        w += (9.0 * s5 + 256.0 * s3 - 433.0 * s) / (38880.0 * a * ra);

        if ((a >= 500.0) && (Math.abs(1.0 - w / a) < 1.0e-6)) {
            return w;
        }
        if (p > 0.5) {
            if (w < 3.0 * a) {
                return w;
            }

            double d = Math.max(2.0, a * (a - 1.0));
            double lb = Math.log(q) + lgamma(a);
            if (lb < -d * 2.3) {
                double y = -lb;
                double c1 = (a - 1.0) * Math.log(y);
                double c1_2 = c1 * c1;
                double c1_3 = c1_2 * c1;
                double c1_4 = c1_2 * c1_2;
                double a2 = a * a;
                double a3 = a2 * a;

                double c2 = (a - 1.0) * (1.0 + c1);
                double c3 = (a - 1.0) * (-(c1_2 / 2.0) + (a - 2.0) * c1 + (3.0 * a - 5.0) / 2.0);
                double c4 = (a - 1.0) * ((c1_3 / 3.0) - (3.0 * a - 5.0) * c1_2 / 2.0 + (a2 - 6.0 * a + 7.0) * c1 + (11.0 * a2 - 46.0 * a + 47.0) / 6.0);
                double c5 = (a - 1.0) * (-(c1_4 / 4.0)
                    + (11.0 * a - 17.0) * c1_3 / 6.0
                    + (-3.0 * a2 + 13.0 * a - 13.0) * c1_2
                    + (2.0 * a3 - 25.0 * a2 + 72.0 * a - 61.0) * c1 / 2.0
                    + (25.0 * a3 - 195.0 * a2 + 477.0 * a - 379.0) / 12.0);

                double y2 = y * y;
                double y3 = y2 * y;
                double y4 = y2 * y2;
                return y + c1 + (c2 / y) + (c3 / y2) + (c4 / y3) + (c5 / y4);
            }

            double u = -lb + (a - 1.0) * Math.log(w) - Math.log(1.0 + (1.0 - a) / (1.0 + w));
            return -lb + (a - 1.0) * Math.log(u) - Math.log(1.0 + (1.0 - a) / (1.0 + u));
        }

        double z = w;
        double ap1 = a + 1.0;
        double ap2 = a + 2.0;
        if (w < 0.15 * ap1) {
            double v = Math.log(p) + lgamma(ap1);
            z = Math.exp((v + w) / a);
            double series = Math.log1p(z / ap1 * (1.0 + z / ap2));
            z = Math.exp((v + z - series) / a);
            series = Math.log1p(z / ap1 * (1.0 + z / ap2));
            z = Math.exp((v + z - series) / a);
            series = Math.log1p(z / ap1 * (1.0 + z / ap2 * (1.0 + z / (a + 3.0))));
            z = Math.exp((v + z - series) / a);
        }

        if ((z <= 0.01 * ap1) || (z > 0.7 * ap1)) {
            return z;
        }

        double logSeries = Math.log(didonatoSN(a, z, 100, 1.0e-4));
        double v = Math.log(p) + lgamma(ap1);
        z = Math.exp((v + z - logSeries) / a);
        return z * (1.0 - (a * Math.log(z) - z - v + logSeries) / (a - z));
    }

    private static double findInverseS(double p, double q) {
        double t = p < 0.5 ? Math.sqrt(-2.0 * Math.log(p)) : Math.sqrt(-2.0 * Math.log(q));
        double s = t - evaluatePolynomial(INVERSE_S_P, t) / evaluatePolynomial(INVERSE_S_Q, t);
        return p < 0.5 ? -s : s;
    }
    private static double didonatoSN(double a, double x, int n, double tolerance) {
        double sum = 1.0;
        if (n >= 1) {
            double partial = x / (a + 1.0);
            sum += partial;
            for (int i = 2; i <= n; i++) {
                partial *= x / (a + i);
                sum += partial;
                if (partial < tolerance) {
                    break;
                }
            }
        }
        return sum;
    }

    private static void gammaInverseRoot(double a, double target, boolean invert, double x, double[] evaluation) {
        Double2 valueAndDerivative = incompleteWithDerivative(a, x, invert, true);
        double functionValue = valueAndDerivative._1() - target;
        double firstDerivative = valueAndDerivative._2();
        double secondDerivative = firstDerivative;
        double div = (a - x - 1.0) / x;
        if (Math.abs(div) > 1.0) {
            if ((Double.MAX_VALUE / Math.abs(div)) < secondDerivative) {
                secondDerivative = -Double.MAX_VALUE / 2.0;
            } else {
                secondDerivative *= div;
            }
        } else {
            secondDerivative *= div;
        }
        if (invert) {
            firstDerivative = -firstDerivative;
            secondDerivative = -secondDerivative;
        }
        evaluation[0] = functionValue;
        evaluation[1] = firstDerivative;
        evaluation[2] = secondDerivative;
    }

    private static double gammaShapeInverseValue(double x, double target, boolean invert, double a) {
        return invert ? target - gammaQ(a, x) : gammaP(a, x) - target;
    }

    private static double incomplete(double a, double x, boolean complement, boolean regularized) {
        return incompleteImp(a, x, complement, regularized, false)._1();
    }

    private static Double2 incompleteWithDerivative(double a, double x, boolean complement, boolean regularized) {
        return incompleteImp(a, x, complement, regularized, true);
    }

    private static Double2 incompleteImp(double a, double x, boolean complement, boolean regularized, boolean withDerivative) {
        boolean invert = complement;

        boolean isInteger = false;
        boolean isHalfInteger = false;
        boolean isSmallA = (a < 30.0) && (a <= x + 1.0) && (x < LOG_MAX_VALUE);
        if (isSmallA) {
            double floorA = Math.floor(a);
            isInteger = floorA == a;
            isHalfInteger = !isInteger && Math.abs(floorA - a) == 0.5;
        }

        int evaluationMethod;
        if (isInteger && (x > 0.6)) {
            invert = !invert;
            evaluationMethod = 0;
        } else if (isHalfInteger && (x > 0.2)) {
            invert = !invert;
            evaluationMethod = 1;
        } else if ((x < ROOT_EPSILON) && (a > 1.0)) {
            evaluationMethod = 6;
        } else if ((x > 1000.0) && ((a < x) || (Math.abs(a - 50.0) / x < 1.0))) {
            invert = !invert;
            evaluationMethod = 7;
        } else if (x < 0.5) {
            evaluationMethod = (-0.4 / Math.log(x) < a) ? 2 : 3;
        } else if (x < 1.1) {
            evaluationMethod = (x * 0.75 < a) ? 2 : 3;
        } else {
            boolean useTemme = false;
            if (a > 20.0) {
                double sigma = Math.abs((x - a) / a);
                if (a > 200.0) {
                    if (20.0 / a > sigma * sigma) {
                        useTemme = true;
                    }
                } else if (sigma < 0.4) {
                    useTemme = true;
                }
            }

            if (useTemme) {
                evaluationMethod = 5;
            } else if (x - (1.0 / (3.0 * x)) < a) {
                evaluationMethod = 2;
            } else {
                evaluationMethod = 4;
                invert = !invert;
            }
        }

        double result;
        switch (evaluationMethod) {
            case 0:
                result = regularized ? finiteQInteger((int) Math.rint(a), x)
                    : multiplyByCompleteGamma(finiteQInteger((int) Math.rint(a), x), a);
                break;
            case 1:
                result = regularized ? finiteHalfQ(a, x)
                    : multiplyByCompleteGamma(finiteHalfQ(a, x), a);
                break;
            case 2:
                result = regularized ? regularizedSeriesP(a, x) : lowerSeriesValue(a, x);
                break;
            case 3:
                result = regularized
                    ? 1.0 - regularizedSeriesP(a, x)
                    : subtractFromCompleteGamma(a, lowerSeriesValue(a, x));
                invert = !invert;
                break;
            case 4:
                result = regularized
                    ? regularizedIncompletePrefix(a, x) * upperFraction(a, x)
                    : upperFractionValue(a, x);
                break;
            case 5:
                result = regularized ? temmeLarge(a, x) : multiplyByCompleteGamma(temmeLarge(a, x), a);
                if (x >= a) {
                    invert = !invert;
                }
                break;
            case 6:
                result = regularized ? smallXRegularizedP(a, x) : smallXLower(a, x);
                break;
            case 7:
                result = regularized
                    ? (regularizedIncompletePrefix(a, x) / x) * incompleteLargeX(a, x)
                    : largeXUpper(a, x);
                break;
            default:
                throw new IllegalStateException("Unsupported gamma evaluation method: " + evaluationMethod);
        }

        double derivative = withDerivative ? (regularized ? gammaPDerivativeUnchecked(a, x) : Double.NaN) : Double.NaN;

        if (regularized) {
            if (result > 1.0 && !Double.isNaN(result)) {
                result = 1.0;
            }
            if (invert) {
                result = 1.0 - result;
            }
            return new Double2(clampProbability(result), derivative);
        }
        if (invert) {
            result = subtractFromCompleteGamma(a, result);
        }
        return new Double2(clampNonNegative(result), derivative);
    }

    private static double finiteQInteger(int a, double x) {
        double e = Math.exp(-x);
        double sum = e;
        if (sum == 0.0) {
            return 0.0;
        }

        double term = sum;
        for (int n = 1; n < a; n++) {
            term *= x / n;
            sum += term;
        }
        return sum;
    }

    private static double finiteHalfQ(double a, double x) {
        double q = 2.0 * Normal.ccdf(Math.sqrt(2.0 * x));
        if ((q != 0.0) && (a > 1.0)) {
            double term = Math.exp(-x) / Math.sqrt(Math.PI * x);
            term *= 2.0 * x;
            double sum = term;
            int limit = (int) Math.floor(a);
            for (int n = 2; n <= limit; n++) {
                term *= x / (n - 0.5);
                sum += term;
            }
            q += sum;
        }
        return q;
    }

    private static double regularizedSeriesP(double a, double x) {
        double prefix = regularizedIncompletePrefix(a, x);
        if (prefix == 0.0) {
            return 0.0;
        }
        return prefix * lowerSeries(a, x) / a;
    }

    private static double lowerSeriesValue(double a, double x) {
        double prefix = fullIncompletePrefix(a, x);
        if (prefix == 0.0) {
            return 0.0;
        }
        return prefix * lowerSeries(a, x) / a;
    }

    private static double lowerSeries(double a, double x) {
        double sum = 1.0;
        double term = 1.0;
        for (int iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
            term *= x / (a + iteration);
            sum += term;
            if (Math.abs(term) <= Math.abs(sum) * SERIES_EPSILON) {
                return sum;
            }
        }

        throw new IllegalStateException("Lower incomplete gamma series did not converge for a=" + a + ", x=" + x);
    }

    private static double upperFraction(double a, double x) {
        double initial = x - a + 1.0;
        ContinuedFraction fraction = ContinuedFraction.evaluateFractionA(
            iteration -> {
                if (iteration == 0) {
                    return Double2.fraction(1.0, initial);
                }
                double term = iteration;
                return Double2.fraction(term * (a - term), initial + 2.0 * term);
            },
            Gamma::safeDenominator,
            CONTINUED_FRACTION_EPSILON,
            MAX_ITERATIONS);
        if (!fraction.converged()) {
            throw new IllegalStateException(
                "Continued fraction did not converge for upper incomplete gamma fraction for a=" + a + ", x=" + x);
        }
        return fraction.value();
    }

    private static double smallXRegularizedP(double a, double x) {
        double logResult = a * Math.log(x) - lgammaPositive(a + 1.0);
        if (logResult <= LOG_TRUE_MIN) {
            return 0.0;
        }
        double result = Math.exp(logResult);
        return result * (1.0 - a * x / (a + 1.0));
    }

    private static double smallXLower(double a, double x) {
        double logResult = a * Math.log(x) - Math.log(a);
        if (logResult <= LOG_TRUE_MIN) {
            return 0.0;
        }
        if (logResult >= LOG_MAX_VALUE) {
            return Double.POSITIVE_INFINITY;
        }
        double result = Math.exp(logResult);
        return clampNonNegative(result * (1.0 - a * x / (a + 1.0)));
    }

    private static double incompleteLargeX(double a, double x) {
        double aPochhammer = a - 1.0;
        double term = 1.0;
        double sum = 0.0;

        for (int iteration = 0; iteration <= MAX_ITERATIONS; iteration++) {
            sum += term;
            double nextTerm = term * aPochhammer / x;
            if (Math.abs(nextTerm) <= Math.abs(sum) * SERIES_EPSILON) {
                return sum;
            }
            term = nextTerm;
            aPochhammer -= 1.0;
        }

        throw new IllegalStateException("Large-x incomplete gamma series did not converge for a=" + a + ", x=" + x);
    }

    private static double temmeLarge(double a, double x) {
        double sigma = (x - a) / a;
        double phi = -log1pmx(sigma);
        double y = a * phi;
        double z = Math.sqrt(2.0 * phi);
        if (x < a) {
            z = -z;
        }

        double[] workspace = new double[10];
        workspace[0] = evaluatePolynomial(TEMME_C0, z);
        workspace[1] = evaluatePolynomial(TEMME_C1, z);
        workspace[2] = evaluatePolynomial(TEMME_C2, z);
        workspace[3] = evaluatePolynomial(TEMME_C3, z);
        workspace[4] = evaluatePolynomial(TEMME_C4, z);
        workspace[5] = evaluatePolynomial(TEMME_C5, z);
        workspace[6] = evaluatePolynomial(TEMME_C6, z);
        workspace[7] = evaluatePolynomial(TEMME_C7, z);
        workspace[8] = evaluatePolynomial(TEMME_C8, z);
        workspace[9] = TEMME_WORKSPACE_BASE;

        double result = evaluatePolynomial(workspace, 1.0 / a);
        result *= Math.exp(-y) / Math.sqrt(TWO_PI * a);
        if (x < a) {
            result = -result;
        }

        return result + Normal.ccdf(Math.sqrt(2.0 * y));
    }

    static double regularizedIncompletePrefix(double a, double x) {
        if (x >= Double.MAX_VALUE) {
            return 0.0;
        }

        double logPrefix;
        if ((a > 150.0) && (Math.abs((x - a) / a) < 0.5)) {
            double sigma = (x - a) / a;
            logPrefix = a * log1pmx(sigma)
                + 0.5 * Math.log(a)
                - HALF_LOG_TWO_PI
                - stirlingCorrection(a);
        } else {
            logPrefix = a * Math.log(x) - x - lgammaPositive(a);
        }

        if (logPrefix <= LOG_TRUE_MIN) {
            return 0.0;
        }
        if (logPrefix >= LOG_MAX_VALUE) {
            return Double.POSITIVE_INFINITY;
        }
        return Math.exp(logPrefix);
    }

    static double fullIncompletePrefix(double a, double x) {
        double logPrefix = a * Math.log(x) - x;
        if (logPrefix <= LOG_TRUE_MIN) {
            return 0.0;
        }
        if (logPrefix >= LOG_MAX_VALUE) {
            return Double.POSITIVE_INFINITY;
        }
        return Math.exp(logPrefix);
    }

    private static double stirlingCorrection(double a) {
        return lgammaPositive(a) - ((a - 0.5) * Math.log(a) - a + HALF_LOG_TWO_PI);
    }

    private static double log1pmx(double x) {
        if (Math.abs(x) > 0.5) {
            return Math.log1p(x) - x;
        }

        double power = x * x;
        double sum = -0.5 * power;
        for (int n = 3; n <= MAX_ITERATIONS; n++) {
            power *= x;
            double term = ((n & 1) == 0 ? -power : power) / n;
            sum += term;
            if (Math.abs(term) <= Math.abs(sum) * SERIES_EPSILON) {
                return sum;
            }
        }

        return Math.log1p(x) - x;
    }

    private static double upperFractionValue(double a, double x) {
        double prefix = fullIncompletePrefix(a, x);
        if (prefix == 0.0) {
            return 0.0;
        }
        return prefix * upperFraction(a, x);
    }

    private static double largeXUpper(double a, double x) {
        double logResult = (a - 1.0) * Math.log(x) - x;
        if (logResult <= LOG_TRUE_MIN) {
            return 0.0;
        }
        if (logResult >= LOG_MAX_VALUE) {
            return Double.POSITIVE_INFINITY;
        }
        return Math.exp(logResult) * incompleteLargeX(a, x);
    }

    private static double multiplyByCompleteGamma(double regularizedValue, double a) {
        if (regularizedValue == 0.0) {
            return 0.0;
        }
        double completeGamma = tgammaPositive(a);
        if (!Double.isFinite(completeGamma)) {
            return regularizedValue > 0.0 ? completeGamma : 0.0;
        }
        return regularizedValue * completeGamma;
    }

    private static double subtractFromCompleteGamma(double a, double valueToSubtract) {
        if (valueToSubtract == 0.0) {
            return tgammaPositive(a);
        }
        double completeGamma = tgammaPositive(a);
        if (!Double.isFinite(completeGamma)) {
            return completeGamma;
        }
        return clampNonNegative(completeGamma - valueToSubtract);
    }

    private static double clampNonNegative(double value) {
        if (Double.isNaN(value) || value >= 0.0) {
            return value;
        }
        return value > -1.0e-12 ? 0.0 : value;
    }

    private static double digammaLarge(double x) {
        double shifted = x - 1.0;
        double result = Math.log(shifted);
        result += 1.0 / (2.0 * shifted);
        double z = 1.0 / (shifted * shifted);
        return result - z * evaluatePolynomial(DIGAMMA_ASYMPTOTIC_P, z);
    }

    private static double digammaOneTwo(double x) {
        double g = x - DIGAMMA_ROOT1;
        g -= DIGAMMA_ROOT2;
        g -= DIGAMMA_ROOT3;
        double r = evaluatePolynomial(DIGAMMA_P, x - 1.0) / evaluatePolynomial(DIGAMMA_Q, x - 1.0);
        return g * DIGAMMA_Y + g * r;
    }

    private static double trigammaPositive(double x) {
        double result = 0.0;
        double shifted = x;
        if (shifted < 1.0) {
            result = 1.0 / (shifted * shifted);
            shifted += 1.0;
        }
        return result + trigammaPrec(shifted);
    }

    private static double trigammaPrec(double x) {
        if (x <= 2.0) {
            return (TRIGAMMA_1_2_OFFSET
                + evaluatePolynomial(TRIGAMMA_1_2_P, x) / evaluatePolynomial(TRIGAMMA_1_2_Q, x)) / (x * x);
        }
        if (x <= 4.0) {
            double y = 1.0 / x;
            return (1.0 + evaluatePolynomial(TRIGAMMA_2_4_P, y) / evaluatePolynomial(TRIGAMMA_2_4_Q, y)) / x;
        }
        double y = 1.0 / x;
        return (1.0 + evaluatePolynomial(TRIGAMMA_4_INF_P, y) / evaluatePolynomial(TRIGAMMA_4_INF_Q, y)) / x;
    }

    private static double polygammaNearZero(int order, double x) {
        double scale = factorial(order);
        double prefix = 1.0 / Math.pow(x, order + 1.0);
        if (!Double.isFinite(prefix)) {
            throw new ArithmeticException("polygamma overflow near zero");
        }
        double sum = prefix;
        double factorialPart = 1.0;
        for (int k = 0; k < MAX_ITERATIONS; k++) {
            double term = factorialPart * riemannZeta(order + k + 1);
            sum += term;
            if (Math.abs(term) <= Math.abs(sum) * SERIES_EPSILON) {
                double signed = ((order & 1) == 1) ? sum : -sum;
                double result = signed * scale;
                if (!Double.isFinite(result)) {
                    throw new ArithmeticException("polygamma overflow near zero");
                }
                return result;
            }
            factorialPart *= (-x * (order + k + 1.0)) / (k + 1.0);
        }
        throw new IllegalStateException("Polygamma near-zero series did not converge for order=" + order + ", x=" + x);
    }

    private static double polygammaAtTransition(int order, double x) {
        int target = (int) DOUBLE_DIGITS10_FACTOR + 4 * order;
        int iterations = Math.max(0, target - (int) x);
        double z = x;
        double sum = 0.0;
        for (int k = 0; k < iterations; k++) {
            sum += Math.pow(z, -order - 1.0);
            z += 1.0;
        }
        sum *= factorial(order);
        if (((order - 1) & 1) != 0) {
            sum = -sum;
        }
        return sum + polygammaAtInfinity(order, z);
    }

    private static double polygammaAtInfinity(int order, double x) {
        double sign = ((order & 1) == 0) ? -1.0 : 1.0;
        return polygammaAsymptotic(order, x, sign, factorial(order));
    }

    private static double polygammaAsymptotic(int order, double x, double sign, double factorial) {
        double xPower = Math.pow(x, order);
        double sum = 1.0 / (order * xPower);
        sum += 0.5 / (xPower * x);

        double xSquared = x * x;
        double xTermPower = xPower * xSquared;
        double rising = order + 1.0;
        for (int index = 0; index < BERNOULLI_OVER_EVEN_FACTORIAL.length; index++) {
            if (index > 0) {
                double next = order + 2.0 * index;
                rising *= next;
                rising *= next + 1.0;
                xTermPower *= xSquared;
            }
            sum += BERNOULLI_OVER_EVEN_FACTORIAL[index] * rising / xTermPower;
        }
        return sign * factorial * sum;
    }

    private static double riemannZeta(int s) {
        if (s <= 1) {
            throw new IllegalArgumentException("zeta argument must be greater than 1: " + s);
        }
        if (s <= 10) {
            return switch (s) {
                case 2 -> 1.6449340668482264;
                case 3 -> 1.2020569031595942;
                case 4 -> 1.0823232337111381;
                case 5 -> 1.03692775514337;
                case 6 -> 1.017343061984449;
                case 7 -> 1.008349277381923;
                case 8 -> 1.0040773561979444;
                case 9 -> 1.0020083928260821;
                default -> 1.000994575127818;
            };
        }
        double denominator = 1.0 - Math.scalb(1.0, 1 - s);
        double sum = 0.0;
        for (int k = 1; k < MAX_ITERATIONS; k++) {
            double term = Math.pow(k, -s);
            sum += ((k & 1) == 1) ? term : -term;
            if (term <= Math.abs(sum) * SERIES_EPSILON) {
                return sum / denominator;
            }
        }
        throw new IllegalStateException("Zeta series did not converge for s=" + s);
    }

    private static double tgammaDeltaRatioUnchecked(double z, double delta) {
        if ((z <= 0.0) || (z + delta <= 0.0)) {
            return tgamma(z) / tgamma(z + delta);
        }

        if (Math.rint(delta) == delta) {
            if ((Math.rint(z) == z)
                && (z <= MAX_DOUBLE_FACTORIAL)
                && (z + delta <= MAX_DOUBLE_FACTORIAL)) {
                return uncheckedFactorial((int) Math.rint(z) - 1)
                    / uncheckedFactorial((int) Math.rint(z + delta) - 1);
            }
            if (Math.abs(delta) < 20.0) {
                if (delta == 0.0) {
                    return 1.0;
                }
                if (delta < 0.0) {
                    z -= 1.0;
                    double result = z;
                    while ((delta += 1.0) != 0.0) {
                        z -= 1.0;
                        result *= z;
                    }
                    return result;
                }

                double result = 1.0 / z;
                while ((delta -= 1.0) != 0.0) {
                    z += 1.0;
                    result /= z;
                }
                return result;
            }
        }

        return tgammaDeltaRatioLanczos(z, delta);
    }

    private static double tgammaDeltaRatioLanczos(double z, double delta) {
        if (z < EPSILON) {
            if (MAX_DOUBLE_FACTORIAL < delta) {
                double ratio = tgammaDeltaRatioLanczosFinal(delta, MAX_DOUBLE_FACTORIAL - delta);
                ratio *= z;
                ratio *= uncheckedFactorial(MAX_DOUBLE_FACTORIAL - 1);
                return 1.0 / ratio;
            }
            return 1.0 / (z * tgamma(z + delta));
        }
        return tgammaDeltaRatioLanczosFinal(z, delta);
    }

    private static double tgammaDeltaRatioLanczosFinal(double z, double delta) {
        double zgh = z + g() - 0.5;
        double result;
        if (z + delta == z) {
            result = Math.exp(-delta);
        } else {
            if (Math.abs(delta) < 10.0) {
                result = Math.exp((0.5 - z) * Math.log1p(delta / zgh));
            } else {
                result = Math.pow(zgh / (zgh + delta), z - 0.5);
            }
            result *= lanczosSum(z) / lanczosSum(z + delta);
        }
        result *= Math.pow(Math.E / (zgh + delta), delta);
        return result;
    }

    private static double inverseRegularizedShape(double x, double p, double q, String functionName) {
        double target = Math.min(p, q);
        boolean invert = p >= q;
        double guess;
        double factor = 8.0;
        if (x >= 1.0) {
            guess = 1.0 + inversePoissonCornishFisher(x, q, p);
            if (x > 5.0) {
                if (x > 1000.0) {
                    factor = 1.01;
                } else if (x > 50.0) {
                    factor = 1.1;
                } else if (guess > 10.0) {
                    factor = 1.25;
                } else {
                    factor = 2.0;
                }
                if (guess < 1.1) {
                    factor = 8.0;
                }
            }
        } else if (x > 0.5) {
            guess = x * 1.2;
        } else {
            guess = -0.4 / Math.log(x);
        }

        return RootIterators.toms748SolveBounded(
            a -> gammaShapeInverseValue(x, target, invert, a),
            guess,
            factor,
            false,
            MIN_SHAPE,
            DOUBLE_PRECISION_BITS,
            MAX_INVERSE_ITERATIONS,
            functionName
        );
    }

    private static double inversePoissonCornishFisher(double lambda, double p, double q) {
        double mean = lambda;
        double sigma = Math.sqrt(lambda);
        double skewness = 1.0 / sigma;
        double x = Normal.erfcInv(p > q ? 2.0 * q : 2.0 * p) * Math.sqrt(2.0);
        if (p < 0.5) {
            x = -x;
        }
        double x2 = x * x;
        double correction = x + skewness * (x2 - 1.0) / 6.0;
        double guess = mean + sigma * correction;
        return Math.max(guess, MIN_SHAPE);
    }

    static double logAbsExtended(double x) {
        if (x > 0.0) {
            return lgammaPositive(x);
        }
        ensureNotGammaPole(x, "lgamma");
        return Math.log(Math.PI) - Math.log(Math.abs(sinPi(x))) - lgammaPositive(1.0 - x);
    }

    static int signExtended(double x) {
        if (x > 0.0) {
            return 1;
        }
        ensureNotGammaPole(x, "lgamma");
        return sinPi(x) < 0.0 ? -1 : 1;
    }

    static double completeExtended(double x) {
        if (x > 0.0) {
            return tgammaPositive(x);
        }
        int sign = signExtended(x);
        double logAbs = logAbsExtended(x);
        if (logAbs > LOG_MAX_VALUE) {
            return Math.copySign(Double.POSITIVE_INFINITY, sign);
        }
        if (logAbs <= LOG_TRUE_MIN) {
            return Math.copySign(0.0, sign);
        }
        return Math.copySign(Math.exp(logAbs), sign);
    }

    private static void validateArguments(double a, double x) {
        if (!(a > 0.0)) {
            throw new IllegalArgumentException("a must be positive: " + a);
        }
        if (Double.isNaN(x) || x < 0.0) {
            throw new IllegalArgumentException("x must be non-negative: " + x);
        }
    }

    private static void validateShape(double a) {
        if (!(a > 0.0)) {
            throw new IllegalArgumentException("a must be positive: " + a);
        }
    }

    private static void validateInverseShapeArgument(double x, String functionName) {
        if (!Double.isFinite(x) || !(x > 0.0)) {
            throw new IllegalArgumentException(functionName + ": x must be finite and greater than zero");
        }
    }

    private static double safeDenominator(double value) {
        if (Math.abs(value) < FPMIN) {
            return value < 0.0 ? -FPMIN : FPMIN;
        }
        return value;
    }

    private static double clampProbability(double value) {
        if (value <= 0.0) {
            return 0.0;
        }
        if (value >= 1.0) {
            return 1.0;
        }
        return value;
    }

    private static void validateFiniteReal(double x, String name) {
        if (!Double.isFinite(x)) {
            throw new IllegalArgumentException(name + " must be finite: " + x);
        }
    }

    private static void ensureNotGammaPole(double x, String functionName) {
        if (x == 0.0 || (x < 0.0 && x == Math.rint(x))) {
            throw new ArithmeticException(functionName + " pole at non-positive integer: " + x);
        }
    }

    private static void validatePolygammaOrder(int order) {
        if (order < 0) {
            throw new IllegalArgumentException("order must be non-negative: " + order);
        }
    }
}
