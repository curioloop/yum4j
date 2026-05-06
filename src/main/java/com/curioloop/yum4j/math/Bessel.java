package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.Gamma.completeExtended;
import static com.curioloop.yum4j.math.Gamma.logAbsExtended;
import static com.curioloop.yum4j.math.Gamma.signExtended;
import static com.curioloop.yum4j.math.Commons.cosPi;
import static com.curioloop.yum4j.math.Commons.factorial;
import static com.curioloop.yum4j.math.Commons.isInteger;
import static com.curioloop.yum4j.math.Commons.sameParameter;
import static com.curioloop.yum4j.math.Commons.sinPi;
import static com.curioloop.yum4j.math.Commons.validateFinite;
import static com.curioloop.yum4j.math.HornerPoly.evaluatePolynomial;
import static com.curioloop.yum4j.math.HornerPoly.evaluateRational;

import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;

/**
 * Real-valued cylindrical and modified Bessel functions.
 *
 * <p>This class is the single Boost-aligned implementation used both by public callers and by the
 * internal hypergeometric kernels. The supported real surface intentionally covers the Boost-aligned
 * real scalar cylindrical and spherical Bessel family, plus the retained quant4j scaled
 * convenience helpers used by the local hypergeometric and iterator layers:
 *
 * <pre>
 * J_nu(x), Y_nu(x), I_nu(x), K_nu(x)
 * J'_nu(x), Y'_nu(x), I'_nu(x), K'_nu(x)
 * j_n(x), y_n(x), j'_n(x), y'_n(x)
 * J_nu zero m, Y_nu zero m
 * J_nu zero sequence, Y_nu zero sequence
 * exp(-|x|) I_nu(x), exp(-x) K_nu(x)
 * </pre>
 */
public final class Bessel {

    private static final double MACHINE_EPSILON = Math.ulp(1.0);
    private static final double SERIES_EPSILON = 2.0 * MACHINE_EPSILON;
    private static final double ROOT_MIN_NORMAL = Math.sqrt(Double.MIN_NORMAL);
    private static final double FORTH_ROOT_EPSILON = Math.sqrt(Math.sqrt(MACHINE_EPSILON));
    private static final double EULER = 0.57721566490153286060;
    private static final double LOG_MAX_VALUE = Math.log(Double.MAX_VALUE);
    private static final double MAX_FACTORIAL = 170.0;
    private static final int MAX_ITERATIONS = 100000;
    private static final int ROOT_MAX_ITERATIONS = 200;
    private static final int NEED_J = 1;
    private static final int NEED_Y = 2;
    private static final int NEED_I = 1;
    private static final int NEED_K = 2;
    private static final double[] BESSEL_I0_SMALL_COEFFICIENTS = {
        1.00000000000000000e+00,
        2.49999999999999909e-01,
        2.77777777777782257e-02,
        1.73611111111023792e-03,
        6.94444444453352521e-05,
        1.92901234513219920e-06,
        3.93675991102510739e-08,
        6.15118672704439289e-10,
        7.59407002058973446e-12,
        7.59389793369836367e-14,
        6.27767773636292611e-16,
        4.34709704153272287e-18,
        2.63417742690109154e-20,
        1.13943037744822825e-22,
        9.07926920085624812e-25
    };
    private static final double[] BESSEL_I0_MEDIUM_COEFFICIENTS = {
        3.98942280401425088e-01,
        4.98677850604961985e-02,
        2.80506233928312623e-02,
        2.92211225166047873e-02,
        4.44207299493659561e-02,
        1.30970574605856719e-01,
        -3.35052280231727022e+00,
        2.33025711583514727e+02,
        -1.13366350697172355e+04,
        4.24057674317867331e+05,
        -1.23157028595698731e+07,
        2.80231938155267516e+08,
        -5.01883999713777929e+09,
        7.08029243015109113e+10,
        -7.84261082124811106e+11,
        6.76825737854096565e+12,
        -4.49034849696138065e+13,
        2.24155239966958995e+14,
        -8.13426467865659318e+14,
        2.02391097391687777e+15,
        -3.08675715295370878e+15,
        2.17587543863819074e+15
    };
    private static final double[] BESSEL_I0_LARGE_COEFFICIENTS = {
        3.98942280401432905e-01,
        4.98677850491434560e-02,
        2.80506308916506102e-02,
        2.92179096853915176e-02,
        4.53371208762579442e-02
    };
    private static final double[] BESSEL_I1_SMALL_COEFFICIENTS = {
        8.333333333333333803e-02,
        6.944444444444341983e-03,
        3.472222222225921045e-04,
        1.157407407354987232e-05,
        2.755731926254790268e-07,
        4.920949692800671435e-09,
        6.834657311305621830e-11,
        7.593969849687574339e-13,
        6.904822652741917551e-15,
        5.220157095351373194e-17,
        3.410720494727771276e-19,
        1.625212890947171108e-21,
        1.332898928162290861e-23
    };
    private static final double[] BESSEL_I1_MEDIUM_COEFFICIENTS = {
        3.989422804014406054e-01,
        -1.496033551613111533e-01,
        -4.675104253598537322e-02,
        -4.090895951581637791e-02,
        -5.719036414430205390e-02,
        -1.528189554374492735e-01,
        3.458284470977172076e+00,
        -2.426181371595021021e+02,
        1.178785865993440669e+04,
        -4.404655582443487334e+05,
        1.277677779341446497e+07,
        -2.903390398236656519e+08,
        5.192386898222206474e+09,
        -7.313784438967834057e+10,
        8.087824484994859552e+11,
        -6.967602516005787001e+12,
        4.614040809616582764e+13,
        -2.298849639457172489e+14,
        8.325554073334618015e+14,
        -2.067285045778906105e+15,
        3.146401654361325073e+15,
        -2.213318202179221945e+15
    };
    private static final double[] BESSEL_I1_LARGE_COEFFICIENTS = {
        3.989422804014314820e-01,
        -1.496033551467584157e-01,
        -4.675105322571775911e-02,
        -4.090421597376992892e-02,
        -5.843630344778927582e-02
    };
    private static final double[] BESSEL_K0_SMALL_P = {
        -1.372509002685546267e-01,
        2.574916117833312855e-01,
        1.395474602146869316e-02,
        5.445476986653926759e-04,
        7.125159422136622118e-06
    };
    private static final double[] BESSEL_K0_SMALL_Q = {
        1.000000000000000000e+00,
        -5.458333438017788530e-02,
        1.291052816975251298e-03,
        -1.367653946978586591e-05
    };
    private static final double[] BESSEL_K0_SMALL_LOG_P = {
        1.159315156584124484e-01,
        2.789828789146031732e-01,
        2.524892993216121934e-02,
        8.460350907213637784e-04,
        1.491471924309617534e-05,
        1.627106892422088488e-07,
        1.208266102392756055e-09,
        6.611686391749704310e-12
    };
    private static final double[] BESSEL_K0_LARGE_P = {
        2.533141373155002416e-01,
        3.628342133984595192e+00,
        1.868441889406606057e+01,
        4.306243981063412784e+01,
        4.424116209627428189e+01,
        1.562095339356220468e+01,
        -1.810138978229410898e+00,
        -1.414237994269995877e+00,
        -9.369168119754924625e-02
    };
    private static final double[] BESSEL_K0_LARGE_Q = {
        1.000000000000000000e+00,
        1.494194694879908328e+01,
        8.265296455388554217e+01,
        2.162779506621866970e+02,
        2.845145155184222157e+02,
        1.851714491916334995e+02,
        5.486540717439723515e+01,
        6.118075837628957015e+00,
        1.586261269326235053e-01
    };
    private static final double[] BESSEL_K1_SMALL_P = {
        -3.62137953440350228e-03,
        7.11842087490330300e-03,
        1.00302560256614306e-05,
        1.77231085381040811e-06
    };
    private static final double[] BESSEL_K1_SMALL_Q = {
        1.00000000000000000e+00,
        -4.80414794429043831e-02,
        9.85972641934416525e-04,
        -8.91196859397070326e-06
    };
    private static final double[] BESSEL_K1_SMALL_LOG_P = {
        -3.07965757829206184e-01,
        -7.80929703673074907e-02,
        -2.70619343754051620e-03,
        -2.49549522229072008e-05
    };
    private static final double[] BESSEL_K1_SMALL_LOG_Q = {
        1.00000000000000000e+00,
        -2.36316836412163098e-02,
        2.64524577525962719e-04,
        -1.49749618004162787e-06
    };
    private static final double[] BESSEL_K1_LARGE_P = {
        -1.97028041029226295e-01,
        -2.32408961548087617e+00,
        -7.98269784507699938e+00,
        -2.39968410774221632e+00,
        3.28314043780858713e+01,
        5.67713761158496058e+01,
        3.30907788466509823e+01,
        6.62582288933739787e+00,
        3.08851840645286691e-01
    };
    private static final double[] BESSEL_K1_LARGE_Q = {
        1.00000000000000000e+00,
        1.41811409298826118e+01,
        7.35979466317556420e+01,
        1.77821793937080859e+02,
        2.11014501598705982e+02,
        1.19425262951064454e+02,
        2.88448064302447607e+01,
        2.27912927104139732e+00,
        2.50358186953478678e-02
    };

    private Bessel() {
    }

    /**
     * Returns the cylindrical Bessel function of the first kind {@code J_nu(x)}.
     */
    public static double j(double nu, double x) {
        validateFinite(nu, "j: order");
        validateFinite(x, "j: x");

        if (x < 0.0) {
            if (!isInteger(nu)) {
                throw new IllegalArgumentException("j: x must be non-negative for non-integer order: " + x);
            }
            long integerOrder = Math.round(Math.rint(nu));
            double value = j(nu, -x);
            return (integerOrder & 1L) == 0L ? value : -value;
        }
        return besselJY(nu, x, NEED_J)._1();
    }

    /**
     * Returns the cylindrical Bessel function of the second kind {@code Y_nu(x)}.
     */
    public static double y(double nu, double x) {
        validateFinite(nu, "y: order");
        validateFinite(x, "y: x");
        if (!(x > 0.0)) {
            throw new IllegalArgumentException("y: x must be positive: " + x);
        }
        return besselJY(nu, x, NEED_Y)._2();
    }

    static Double2 cylJY(double nu, double x) {
        validateFinite(nu, "cylJY: order");
        validateFinite(x, "cylJY: x");
        if (!(x > 0.0)) {
            throw new IllegalArgumentException("cylJY: x must be positive: " + x);
        }
        return besselJY(nu, x, NEED_J | NEED_Y);
    }

    /**
     * Returns the modified Bessel function of the first kind {@code I_nu(x)}.
     */
    public static double i(double nu, double x) {
        validateFinite(nu, "i: order");
        validateFinite(x, "i: x");
        return cylBesselI(nu, x, false);
    }

    /**
     * Returns the modified Bessel function of the second kind {@code K_nu(x)}.
     */
    public static double k(double nu, double x) {
        validateFinite(nu, "k: order");
        validateFinite(x, "k: x");
        return cylBesselK(nu, x, false);
    }

    /**
     * Returns the exponentially weighted modified Bessel function {@code exp(-|x|) I_nu(x)}.
     */
    public static double iExponentiallyWeighted(double nu, double x) {
        validateFinite(nu, "iExponentiallyWeighted: order");
        validateFinite(x, "iExponentiallyWeighted: x");
        return cylBesselI(nu, x, true);
    }

    /**
     * Returns the exponentially weighted modified Bessel function {@code exp(-x) K_nu(x)}.
     */
    public static double kExponentiallyWeighted(double nu, double x) {
        validateFinite(nu, "kExponentiallyWeighted: order");
        validateFinite(x, "kExponentiallyWeighted: x");
        return cylBesselK(nu, x, true);
    }

    /**
     * Returns the derivative of the cylindrical Bessel function of the first kind {@code J'_nu(x)}.
     */
    public static double jPrime(double nu, double x) {
        validateFinite(nu, "jPrime: order");
        validateFinite(x, "jPrime: x");

        if (x < 0.0) {
            if (!isInteger(nu)) {
                throw new IllegalArgumentException("jPrime: x must be non-negative for non-integer order: " + x);
            }
            long integerOrder = Math.round(Math.rint(nu));
            double value = jPrime(nu, -x);
            return (integerOrder & 1L) == 0L ? -value : value;
        }
        return cylBesselJPrimePositiveX(nu, x);
    }

    /**
     * Returns the derivative of the cylindrical Bessel function of the second kind {@code Y'_nu(x)}.
     */
    public static double yPrime(double nu, double x) {
        validateFinite(nu, "yPrime: order");
        validateFinite(x, "yPrime: x");
        if (!(x > 0.0)) {
            throw new IllegalArgumentException("yPrime: x must be positive: " + x);
        }
        return cylNeumannPrimePositiveX(nu, x);
    }

    /**
     * Returns the derivative of the modified Bessel function of the first kind {@code I'_nu(x)}.
     */
    public static double iPrime(double nu, double x) {
        validateFinite(nu, "iPrime: order");
        validateFinite(x, "iPrime: x");

        if (x < 0.0) {
            if (!isInteger(nu)) {
                throw new IllegalArgumentException("iPrime: x must be non-negative for non-integer order: " + x);
            }
            long integerOrder = Math.round(Math.rint(nu));
            double value = iPrime(nu, -x);
            return (integerOrder & 1L) == 0L ? -value : value;
        }
        if (x == 0.0) {
            if (sameParameter(nu, 1.0) || sameParameter(nu, -1.0)) {
                return 0.5;
            }
            if (isInteger(nu) || nu > 1.0) {
                return 0.0;
            }
            throw new IllegalArgumentException("iPrime: derivative is indeterminate at x=0 for order=" + nu);
        }
        if (sameParameter(nu, 0.0)) {
            return i(1.0, x);
        }
        return besselIDerivativeLinear(nu, x);
    }

    /**
     * Returns the derivative of the modified Bessel function of the second kind {@code K'_nu(x)}.
     */
    public static double kPrime(double nu, double x) {
        validateFinite(nu, "kPrime: order");
        validateFinite(x, "kPrime: x");
        if (!(x > 0.0)) {
            throw new IllegalArgumentException("kPrime: x must be positive: " + x);
        }
        if (sameParameter(nu, 0.0)) {
            return -k(1.0, x);
        }
        return besselKDerivativeLinear(nu, x);
    }

    /**
     * Returns the spherical Bessel function {@code j_n(x)}.
     */
    public static double sphBessel(int n, double x) {
        validateNonNegativeOrder(n, "sphBessel: order");
        validateFinite(x, "sphBessel: x");
        if (x < 0.0) {
            throw new IllegalArgumentException("sphBessel: x must be non-negative: " + x);
        }
        if (n == 0) {
            return x == 0.0 ? 1.0 : Math.sin(x) / x;
        }
        if (x == 0.0) {
            return 0.0;
        }
        if (x < 1.0) {
            return sphBesselSmallZSeries(n, x);
        }
        return Math.sqrt(Math.PI / (2.0 * x)) * j(n + 0.5, x);
    }

    /**
     * Returns the spherical Neumann function {@code y_n(x)}.
     */
    public static double sphNeumann(int n, double x) {
        validateNonNegativeOrder(n, "sphNeumann: order");
        validateFinite(x, "sphNeumann: x");
        if (x < 0.0) {
            throw new IllegalArgumentException("sphNeumann: x must be non-negative: " + x);
        }
        if (x < 2.0 * Double.MIN_NORMAL) {
            return Double.NEGATIVE_INFINITY;
        }
        double value = y(n + 0.5, x);
        double scale = Math.sqrt(Math.PI / (2.0 * x));
        if ((scale > 1.0) && (Double.MAX_VALUE / scale < Math.abs(value))) {
            return Math.copySign(Double.POSITIVE_INFINITY, value);
        }
        return value * scale;
    }

    /**
     * Returns the derivative of the spherical Bessel function {@code j'_n(x)}.
     */
    public static double sphBesselPrime(int n, double x) {
        validateNonNegativeOrder(n, "sphBesselPrime: order");
        validateFinite(x, "sphBesselPrime: x");
        if (x < 0.0) {
            throw new IllegalArgumentException("sphBesselPrime: x must be non-negative: " + x);
        }
        if (n == 0) {
            return x == 0.0 ? Double.POSITIVE_INFINITY : -sphBessel(1, x);
        }
        if (x == 0.0) {
            throw new IllegalArgumentException("sphBesselPrime: derivative is indeterminate at x=0 for order=" + n);
        }
        return (n / x) * sphBessel(n, x) - sphBessel(n + 1, x);
    }

    /**
     * Returns the derivative of the spherical Neumann function {@code y'_n(x)}.
     */
    public static double sphNeumannPrime(int n, double x) {
        validateNonNegativeOrder(n, "sphNeumannPrime: order");
        validateFinite(x, "sphNeumannPrime: x");
        if (!(x > 0.0)) {
            throw new IllegalArgumentException("sphNeumannPrime: x must be positive: " + x);
        }
        if (n == 0) {
            return -sphNeumann(1, x);
        }
        return (n / x) * sphNeumann(n, x) - sphNeumann(n + 1, x);
    }

    /**
     * Returns the rank-th positive real zero of {@code J_nu(x)}.
     */
    public static double jZero(double nu, int rank) {
        validateFinite(nu, "jZero: order");
        if (rank < 0) {
            throw new IllegalArgumentException("jZero: rank must be non-negative: " + rank);
        }
        return cylBesselJZero(nu, rank);
    }

    /**
     * Returns the rank-th positive real zero of {@code Y_nu(x)}.
     */
    public static double yZero(double nu, int rank) {
        validateFinite(nu, "yZero: order");
        if (rank < 0) {
            throw new IllegalArgumentException("yZero: rank must be non-negative: " + rank);
        }
        return cylNeumannZero(nu, rank);
    }

    /**
     * Returns a rank-to-zero function for {@code J_nu(x)}.
     */
    public static IntToDoubleFunction jZeros(double nu) {
        validateFinite(nu, "jZeros: order");
        return rank -> jZero(nu, rank);
    }

    /**
     * Returns a rank-to-zero function for {@code Y_nu(x)}.
     */
    public static IntToDoubleFunction yZeros(double nu) {
        validateFinite(nu, "yZeros: order");
        return rank -> yZero(nu, rank);
    }

    static String debugJRoute(double nu, double x) {
        return classifyJRoute(nu, x);
    }

    static String debugYRoute(double nu, double x) {
        return classifyYRoute(nu, x);
    }

    static String debugIRoute(double nu, double x) {
        return classifyIRoute(nu, x);
    }

    static String debugKRoute(double nu, double x) {
        return classifyKRoute(nu, x);
    }

    private static double cylBesselJPrimePositiveX(double v, double x) {
        if (x == 0.0) {
            if (sameParameter(v, 1.0)) {
                return 0.5;
            }
            if (sameParameter(v, -1.0)) {
                return -0.5;
            }
            if (isInteger(v) || v > 1.0) {
                return 0.0;
            }
            throw new IllegalArgumentException("jPrime: derivative is indeterminate at x=0 for order=" + v);
        }
        if (asymptoticBesselDerivativeLargeXLimit(v, x)) {
            return asymptoticBesselJDerivativeLargeX(v, x);
        }
        if ((Math.abs(x) < 5.0) || (Math.abs(v) > x * x / 4.0)) {
            boolean reflectedInteger = false;
            if (isInteger(v) && v < 0.0) {
                v = -v;
                reflectedInteger = (Math.round(Math.rint(v)) & 1L) != 0L;
            }
            double result = besselJDerivativeSmallZSeries(v, x);
            return reflectedInteger ? -result : result;
        }
        if (sameParameter(v, 0.0)) {
            return -j(1.0, x);
        }
        return besselJDerivativeLinear(v, x);
    }

    private static double cylNeumannPrimePositiveX(double v, double x) {
        if (asymptoticBesselDerivativeLargeXLimit(v, x)) {
            return asymptoticBesselYDerivativeLargeX(v, x);
        }
        if ((v > 0.0) && !isInteger(v)
            && (Math.log(MACHINE_EPSILON / 2.0) > v * Math.log((x * x) / (v * 4.0)))) {
            return besselYDerivativeSmallZSeries(v, x);
        }
        if (sameParameter(v, 0.0)) {
            return -y(1.0, x);
        }
        return besselYDerivativeLinear(v, x);
    }

    private static String classifyJRoute(double order, double x) {
        if (x < 0.0) {
            if (!isInteger(order)) {
                return "domain-error";
            }
            return "negative-x-reflection/" + classifyJRoute(order, -x);
        }
        double v = order;
        String prefix = "";
        boolean onlyJ = true;
        if (v < 0.0) {
            prefix = "negative-order-reflection/";
            v = -v;
        }
        int n = (int) Math.rint(v);
        double u = isInteger(v) ? 0.0 : v - n;
        if (!prefix.isEmpty() && (u != 0.0)) {
            onlyJ = false;
        }
        if (x == 0.0) {
            return prefix + "origin";
        }
        if (onlyJ && ((x < 1.0) || (v > x * x / 4.0) || (x < 5.0))) {
            return prefix + "small-z-series-j-only";
        }
        if ((x < 1.0) && (u != 0.0) && (Math.log(MACHINE_EPSILON / 2.0) > v * Math.log((x / 2.0) * (x / 2.0) / v))) {
            return prefix + "small-z-series-jy";
        }
        if ((u == 0.0) && (x < MACHINE_EPSILON)) {
            return prefix + "small-z-integer-y";
        }
        if (asymptoticBesselLargeXLimit(v, x)) {
            return prefix + "large-x-asymptotic";
        }
        if (x > 8.0) {
            Double2 pq = hankelPQ(v, x);
            if (!Double.isNaN(pq._1())) {
                return prefix + "hankel-pq";
            }
        }
        if (x <= 2.0) {
            return prefix + "temme-cf1";
        }
        return prefix + "cf1-cf2";
    }

    private static String classifyYRoute(double order, double x) {
        if (!(x > 0.0)) {
            return "domain-error";
        }
        double v = order;
        String prefix = "";
        if (v < 0.0) {
            prefix = "negative-order-reflection/";
            v = -v;
        }
        int n = (int) Math.rint(v);
        double u = isInteger(v) ? 0.0 : v - n;
        if ((x < 1.0) && (u != 0.0) && (Math.log(MACHINE_EPSILON / 2.0) > v * Math.log((x / 2.0) * (x / 2.0) / v))) {
            return prefix + "small-z-series-jy";
        }
        if ((u == 0.0) && (x < MACHINE_EPSILON)) {
            return prefix + "small-z-integer-y";
        }
        if (asymptoticBesselLargeXLimit(v, x)) {
            return prefix + "large-x-asymptotic";
        }
        if (x > 8.0) {
            Double2 pq = hankelPQ(v, x);
            if (!Double.isNaN(pq._1())) {
                return prefix + "hankel-pq";
            }
        }
        if (x <= 2.0) {
            return prefix + "temme-cf1";
        }
        return prefix + "cf1-cf2";
    }

    private static String classifyIRoute(double order, double x) {
        if (x < 0.0) {
            if (!isInteger(order)) {
                return "domain-error";
            }
            return "negative-x-reflection/" + classifyIRoute(order, -x);
        }
        double v = order;
        String prefix = "";
        if (x == 0.0) {
            return "origin";
        }
        if (sameParameter(v, 0.5)) {
            return "half-integer-closed-form";
        }
        if (sameParameter(v, 0.0)) {
            return "i0-fast-path";
        }
        if (sameParameter(v, 1.0)) {
            return "i1-fast-path";
        }
        if (v > 0.0 && x / v < 0.25) {
            return "small-z-series";
        }
        if (v < 0.0) {
            prefix = "negative-order-reflection/";
            v = -v;
        }
        if (isAsymptoticBesselILargeX(v, x)) {
            return prefix + "asymptotic-large-x";
        }
        return prefix + (x <= 2.0 ? "temme-ik" : "cf2-ik");
    }

    private static String classifyKRoute(double order, double x) {
        if (x < 0.0) {
            return "domain-error";
        }
        if (x == 0.0) {
            return "origin";
        }
        if (isInteger(order)) {
            return "integer-order-kn";
        }
        double v = order;
        String prefix = "";
        if (v < 0.0) {
            prefix = "negative-order-reflection/";
            v = -v;
        }
        return prefix + (x <= 2.0 ? "temme-ik" : "cf2-ik");
    }

    private static Double2 besselJY(double order, double x, int kind) {
        if (x < 0.0) {
            throw new IllegalArgumentException("besselJY requires x >= 0: " + x);
        }

        double v = order;
        boolean reflect = false;
        if (v < 0.0) {
            reflect = true;
            v = -v;
        }
        if (v > Integer.MAX_VALUE) {
            throw new IllegalStateException("Bessel order is too large to evaluate: " + v);
        }

        int n = (int) Math.rint(v);
        double u = isInteger(v) ? 0.0 : v - n;
        double cp = 0.0;
        double sp = 0.0;
        int originalKind = kind;

        if (reflect) {
            double z = u + (n & 1);
            cp = cosPi(z);
            sp = sinPi(z);
            if (u != 0.0) {
                kind = NEED_J | NEED_Y;
            }
        }

        if (x == 0.0) {
            double j;
            if (v == 0.0) {
                j = 1.0;
            } else if (u == 0.0 || !reflect) {
                j = 0.0;
            } else {
                throw new IllegalStateException("Bessel J is complex-infinite at x=0 for reflected non-integer order: " + order);
            }
            if ((kind & NEED_Y) != 0) {
                throw new IllegalStateException("Bessel Y is singular at x=0 for order=" + order);
            }
            return new Double2(j, Double.NaN);
        }

        double wronskian = 2.0 / (x * Math.PI);
        double yScale = 1.0;
        double jv;
        double yv;
        double yv1;

        boolean onlyJ = (kind & NEED_Y) == 0;
        boolean reflectionNeedsY = ((originalKind & NEED_Y) != 0 && (!reflect || cp != 0.0))
            || ((originalKind & NEED_J) != 0 && reflect && sp != 0.0);

        if (onlyJ && ((x < 1.0) || (v > x * x / 4.0) || (x < 5.0))) {
            jv = besselJSmallZSeries(v, x);
            yv = Double.NaN;
            yv1 = Double.NaN;
        } else if ((x < 1.0) && (u != 0.0) && (Math.log(MACHINE_EPSILON / 2.0) > v * Math.log((x / 2.0) * (x / 2.0) / v))) {
            jv = (kind & NEED_J) != 0 ? besselJSmallZSeries(v, x) : Double.NaN;
            if (reflectionNeedsY) {
                Double2 smallZSeriesResult = besselYSmallZSeries(v, x);
                yv = smallZSeriesResult._1();
                yScale = smallZSeriesResult._2();
            } else {
                yv = Double.NaN;
            }
            yv1 = Double.NaN;
        } else if ((u == 0.0) && (x < MACHINE_EPSILON)) {
            jv = (kind & NEED_J) != 0 ? besselJSmallZSeries(v, x) : Double.NaN;
            if (reflectionNeedsY) {
                Double2 smallZResult = besselYnSmallZ(n, x);
                yv = smallZResult._1();
                yScale = smallZResult._2();
            } else {
                yv = Double.NaN;
            }
            yv1 = Double.NaN;
        } else if (asymptoticBesselLargeXLimit(v, x)) {
            yv = (kind & NEED_Y) != 0 ? asymptoticBesselYLargeX(v, x) : Double.NaN;
            jv = (kind & NEED_J) != 0 ? asymptoticBesselJLargeX(v, x) : Double.NaN;
            yv1 = Double.NaN;
        } else {
            Double2 pq = x > 8.0 ? hankelPQ(v, x) : null;
            if (pq != null && !Double.isNaN(pq._1())) {
                double modV = Math.IEEEremainder(v / 2.0 + 0.25, 2.0);
                if (modV < 0.0) {
                    modV += 2.0;
                }
                double sx = Math.sin(x);
                double cx = Math.cos(x);
                double sv = sinPi(modV);
                double cv = cosPi(modV);

                double sc = sx * cv - sv * cx;
                double cc = cx * cv + sx * sv;
                double chi = Math.sqrt(2.0) / (Math.sqrt(Math.PI) * Math.sqrt(x));
                yv = chi * (pq._1() * sc + pq._2() * cc);
                jv = chi * (pq._1() * cc - pq._2() * sc);
                yv1 = Double.NaN;
            } else if (x <= 2.0) {
                Double2 temme = temmeJY(u, x);
                double prev = temme._1();
                double current = temme._2();
                double scale = 1.0;
                for (int k = 1; k <= n; k++) {
                    double factor = 2.0 * (u + k) / x;
                    if ((Double.MAX_VALUE - Math.abs(prev)) / factor < Math.abs(current)) {
                        scale /= current;
                        prev /= current;
                        current = 1.0;
                    }
                    double next = factor * current - prev;
                    prev = current;
                    current = next;
                }
                yv = prev;
                yv1 = current;
                if ((kind & NEED_J) != 0) {
                    DoubleI cf1 = cf1JY(v, x);
                    jv = scale * wronskian / (yv * cf1.value() - yv1);
                } else {
                    jv = Double.NaN;
                }
                yScale = scale;
            } else {
                DoubleI cf1 = cf1JY(v, x);
                int cf1Sign = cf1.sign();
                double init = ROOT_MIN_NORMAL;
                double prev = cf1.value() * cf1Sign * init;
                double current = cf1Sign * init;
                double ratio;
                double fu;

                if (v < MAX_FACTORIAL) {
                    for (int k = n; k > 0; k--) {
                        double next = 2.0 * (u + k) * current / x - prev;
                        if (next == 0.0) {
                            next = prev * MACHINE_EPSILON / 2.0;
                        }
                        prev = current;
                        current = next;
                    }
                    ratio = (cf1Sign * init) / current;
                    fu = prev / current;
                } else {
                    boolean overflow = false;
                    for (int k = n; k > 0; k--) {
                        double t = 2.0 * (u + k) / x;
                        if ((t > 1.0) && (Double.MAX_VALUE / t < current)) {
                            overflow = true;
                            break;
                        }
                        double next = t * current - prev;
                        prev = current;
                        current = next;
                    }
                    if (!overflow) {
                        ratio = (cf1Sign * init) / current;
                        fu = prev / current;
                    } else {
                        ratio = 0.0;
                        fu = 1.0;
                    }
                }

                Double2 cf2 = cf2JY(u, x);
                double cf2P = cf2._1();
                double cf2Q = cf2._2();
                double t = u / x - fu;
                double gamma = (cf2P - t) / cf2Q;
                if (gamma == 0.0) {
                    gamma = u * MACHINE_EPSILON / x;
                }
                double ju = Math.signum(current) * Math.sqrt(wronskian / (cf2Q + gamma * (cf2P - t)));
                jv = ju * ratio;

                double yu = gamma * ju;
                double yu1 = yu * (u / x - cf2P - cf2Q / gamma);

                if ((kind & NEED_Y) != 0) {
                    prev = yu;
                    current = yu1;
                    for (int k = 1; k <= n; k++) {
                        double factor = 2.0 * (u + k) / x;
                        if ((Double.MAX_VALUE - Math.abs(prev)) / factor < Math.abs(current)) {
                            prev /= current;
                            yScale /= current;
                            current = 1.0;
                        }
                        double next = factor * current - prev;
                        prev = current;
                        current = next;
                    }
                    yv = prev;
                    yv1 = current;
                } else {
                    yv = Double.NaN;
                    yv1 = Double.NaN;
                }
            }
        }

        if (reflect) {
            if (sp != 0.0 && Double.MAX_VALUE * Math.abs(yScale) < Math.abs(sp * yv)) {
                throw new IllegalStateException("Reflected Bessel J overflowed for order=" + order + ", x=" + x);
            }
            double reflectedJ = cp * jv - (sp == 0.0 ? 0.0 : (sp * yv) / yScale);

            double reflectedY;
            if (cp != 0.0 && Double.MAX_VALUE * Math.abs(yScale) < Math.abs(cp * yv)) {
                reflectedY = Math.copySign(Double.POSITIVE_INFINITY, cp * yv * yScale);
            } else {
                reflectedY = (sp != 0.0 ? sp * jv : 0.0) + (cp == 0.0 ? 0.0 : (cp * yv) / yScale);
            }
            return new Double2(reflectedJ, reflectedY);
        }

        double scaledY = Double.NaN;
        if ((kind & NEED_Y) != 0) {
            if (Double.MAX_VALUE * Math.abs(yScale) < Math.abs(yv)) {
                throw new IllegalStateException("Bessel Y overflowed for order=" + order + ", x=" + x);
            }
            scaledY = yv / yScale;
        }
        return new Double2(jv, scaledY);
    }

    private static Double2 hankelPQ(double v, double x) {
        double tolerance = SERIES_EPSILON;
        double p = 1.0;
        double q = 0.0;
        double k = 1.0;
        double z8 = 8.0 * x;
        double sq = 1.0;
        double mu = 4.0 * v * v;
        double term = 1.0;
        boolean ok = true;
        do {
            term *= (mu - sq * sq) / (k * z8);
            q += term;
            k += 1.0;
            sq += 2.0;
            double mult = (sq * sq - mu) / (k * z8);
            ok = Math.abs(mult) < 0.5;
            term *= mult;
            p += term;
            k += 1.0;
            sq += 2.0;
        } while ((Math.abs(term) > tolerance * Math.abs(p)) && ok);
        return ok ? new Double2(p, q) : new Double2(Double.NaN, Double.NaN);
    }

    private static Double2 temmeJY(double v, double x) {
        if (Math.abs(v) > 0.5) {
            throw new IllegalStateException("Temme JY requires |v| <= 0.5: " + v);
        }

        double gp = gamma1pm1(v);
        double gm = gamma1pm1(-v);
        double spv = sinPi(v);
        double spv2 = sinPi(v / 2.0);
        double xp = Math.pow(x / 2.0, v);
        double a = Math.log(x / 2.0);
        double sigma = -a * v;
        double d = Math.abs(sigma) < MACHINE_EPSILON ? 1.0 : Math.sinh(sigma) / sigma;
        double e = Math.abs(v) < MACHINE_EPSILON ? v * Math.PI * Math.PI / 2.0 : 2.0 * spv2 * spv2 / v;

        double g1 = (v == 0.0) ? -EULER : (gp - gm) / ((1.0 + gp) * (1.0 + gm) * 2.0 * v);
        double g2 = (2.0 + gp + gm) / ((1.0 + gp) * (1.0 + gm) * 2.0);
        double vspv = Math.abs(v) < MACHINE_EPSILON ? 1.0 / Math.PI : v / spv;
        double f = (g1 * Math.cosh(sigma) - g2 * a * d) * 2.0 * vspv;

        double p = vspv / (xp * (1.0 + gm));
        double q = vspv * xp / (1.0 + gp);
        double g = f + e * q;
        double h = p;
        double coef = 1.0;
        double sum = coef * g;
        double sum1 = coef * h;
        double v2 = v * v;
        double coefMult = -x * x / 4.0;

        for (int k = 1; k < MAX_ITERATIONS; k++) {
            f = (k * f + p + q) / (k * (double) k - v2);
            p /= k - v;
            q /= k + v;
            g = f + e * q;
            h = p - k * g;
            coef *= coefMult / k;
            sum += coef * g;
            sum1 += coef * h;
            if (Math.abs(coef * g) < Math.abs(sum) * MACHINE_EPSILON) {
                return new Double2(-sum, -2.0 * sum1 / x);
            }
        }

        throw new IllegalStateException("Temme JY did not converge for v=" + v + ", x=" + x);
    }

    private static DoubleI cf1JY(double v, double x) {
        double tolerance = SERIES_EPSILON;
        double tiny = ROOT_MIN_NORMAL;
        double c = tiny;
        double d = 0.0;
        double f = tiny;
        int sign = 1;

        for (int k = 1; k < MAX_ITERATIONS * 100; k++) {
            double a = -1.0;
            double b = 2.0 * (v + k) / x;
            c = b + a / c;
            d = b + a * d;
            if (c == 0.0) {
                c = tiny;
            }
            if (d == 0.0) {
                d = tiny;
            }
            d = 1.0 / d;
            double delta = c * d;
            f *= delta;
            if (d < 0.0) {
                sign = -sign;
            }
            if (Math.abs(delta - 1.0) < tolerance) {
                return new DoubleI(-f, sign);
            }
        }

        throw new IllegalStateException("Bessel CF1 did not converge for v=" + v + ", x=" + x);
    }

    private static Double2 cf2JY(double v, double x) {
        if (!(Math.abs(x) > 1.0)) {
            throw new IllegalStateException("Bessel CF2 requires |x| > 1: " + x);
        }

        double tolerance = SERIES_EPSILON;
        double tiny = ROOT_MIN_NORMAL;
        double fr = -0.5 / x;
        double fi = 1.0;
        double cr = fr;
        double ci = fi;
        double dr;
        double di;
        double v2 = v * v;
        double a = (0.25 - v2) / x;
        double br = 2.0 * x;
        double bi = 2.0;
        double temp = cr * cr + 1.0;
        ci = bi + a * cr / temp;
        cr = br + a / temp;
        dr = br;
        di = bi;
        if (Math.abs(cr) + Math.abs(ci) < tiny) {
            cr = tiny;
        }
        if (Math.abs(dr) + Math.abs(di) < tiny) {
            dr = tiny;
        }
        temp = dr * dr + di * di;
        dr /= temp;
        di = -di / temp;
        double deltaR = cr * dr - ci * di;
        double deltaI = ci * dr + cr * di;
        temp = fr;
        fr = temp * deltaR - fi * deltaI;
        fi = temp * deltaI + fi * deltaR;

        for (int k = 2; k < MAX_ITERATIONS; k++) {
            a = k - 0.5;
            a = a * a - v2;
            bi += 2.0;
            temp = cr * cr + ci * ci;
            cr = br + a * cr / temp;
            ci = bi - a * ci / temp;
            dr = br + a * dr;
            di = bi + a * di;
            if (Math.abs(cr) + Math.abs(ci) < tiny) {
                cr = tiny;
            }
            if (Math.abs(dr) + Math.abs(di) < tiny) {
                dr = tiny;
            }
            temp = dr * dr + di * di;
            dr /= temp;
            di = -di / temp;
            deltaR = cr * dr - ci * di;
            deltaI = ci * dr + cr * di;
            temp = fr;
            fr = temp * deltaR - fi * deltaI;
            fi = temp * deltaI + fi * deltaR;
            if (Math.abs(deltaR - 1.0) + Math.abs(deltaI) < tolerance) {
                return new Double2(fr, fi);
            }
        }

        throw new IllegalStateException("Bessel CF2 did not converge for v=" + v + ", x=" + x);
    }

    private static double besselJSmallZSeries(double v, double x) {
        double prefix;
        if (v < MAX_FACTORIAL) {
            prefix = Math.pow(x / 2.0, v) / Gamma.tgamma(v + 1.0);
        } else {
            prefix = Math.exp(v * Math.log(x / 2.0) - Gamma.lgamma(v + 1.0));
        }
        if (prefix == 0.0) {
            return 0.0;
        }

        double mult = -0.25 * x * x;
        double term = 1.0;
        double sum = 1.0;
        for (int n = 1; n < MAX_ITERATIONS; n++) {
            term *= mult / (n * (n + v));
            sum += term;
            if (Math.abs(term) <= Math.abs(sum) * MACHINE_EPSILON) {
                return prefix * sum;
            }
        }

        throw new IllegalStateException("Bessel J small-z series did not converge for v=" + v + ", x=" + x);
    }

    private static Double2 besselYSmallZSeries(double v, double x) {
        double p = Math.log(x / 2.0);
        double scale = 1.0;
        boolean needLogs = (v >= MAX_FACTORIAL) || (LOG_MAX_VALUE / v < Math.abs(p));

        double prefix;
        if (!needLogs) {
            double gam = Gamma.tgamma(v);
            double power = Math.pow(x / 2.0, v);
            if (Double.MAX_VALUE * power < gam) {
                scale /= gam;
                gam = 1.0;
            }
            prefix = -gam / (Math.PI * power);
        } else {
            double logPrefix = Gamma.lgamma(v) - Math.log(Math.PI) - v * p;
            if (LOG_MAX_VALUE < logPrefix) {
                double divisor = Double.MAX_VALUE / 4.0;
                logPrefix -= Math.log(divisor);
                scale /= divisor;
                if (LOG_MAX_VALUE < logPrefix) {
                    throw new IllegalStateException("Bessel Y small-z series overflowed for v=" + v + ", x=" + x);
                }
            }
            prefix = -Math.exp(logPrefix);
        }

        double mult = -0.25 * x * x;
        double term = 1.0;
        double result = 1.0;
        for (int n = 1; n < MAX_ITERATIONS; n++) {
            term *= mult / (n * (n - v));
            result += term;
            if (Math.abs(term) <= Math.abs(result) * MACHINE_EPSILON) {
                break;
            }
            if (n + 1 == MAX_ITERATIONS) {
                throw new IllegalStateException("Bessel Y small-z primary series did not converge for v=" + v + ", x=" + x);
            }
        }
        result *= prefix;

        double secondPrefix;
        if (!needLogs) {
            secondPrefix = completeExtended(-v) * cosPi(v) * Math.pow(x / 2.0, v) / Math.PI;
        } else {
            secondPrefix = signExtended(-v)
                * Math.exp(logAbsExtended(-v) + v * p)
                * cosPi(v)
                / Math.PI;
        }

        term = 1.0;
        double second = 1.0;
        for (int n = 1; n < MAX_ITERATIONS; n++) {
            term *= mult / (n * (n + v));
            second += term;
            if (Math.abs(term) <= Math.abs(second) * MACHINE_EPSILON) {
                return new Double2(result - scale * secondPrefix * second, scale);
            }
        }

        throw new IllegalStateException("Bessel Y small-z secondary series did not converge for v=" + v + ", x=" + x);
    }

    private static Double2 besselYnSmallZ(int n, double z) {
        if (n < 0) {
            throw new IllegalArgumentException("Bessel Y_n small-z requires n >= 0: " + n);
        }
        if (!(z < MACHINE_EPSILON)) {
            throw new IllegalArgumentException("Bessel Y_n small-z requires z < eps: " + z);
        }

        if (n == 0) {
            return new Double2((2.0 / Math.PI) * (Math.log(z / 2.0) + EULER), 1.0);
        }
        if (n == 1) {
            return new Double2(
                (z / Math.PI) * Math.log(z / 2.0)
                    - 2.0 / (Math.PI * z)
                    - (z / (2.0 * Math.PI)) * (1.0 - 2.0 * EULER),
                1.0
            );
        }
        if (n == 2) {
            return new Double2(
                (z * z) / (4.0 * Math.PI) * Math.log(z / 2.0)
                    - 4.0 / (Math.PI * z * z)
                    - (z * z) / (8.0 * Math.PI) * (1.5 - 2.0 * EULER),
                1.0
            );
        }

        double p = Math.pow(z / 2.0, n);
        double result = -factorial(n - 1) / Math.PI;
        double scale = 1.0;
        if (p * Double.MAX_VALUE < Math.abs(result)) {
            double divisor = Double.MAX_VALUE / 8.0;
            result /= divisor;
            scale /= divisor;
            if (p * Double.MAX_VALUE < Math.abs(result)) {
                throw new IllegalStateException("Bessel Y_n small-z overflowed for n=" + n + ", z=" + z);
            }
        }
        return new Double2(result / p, scale);
    }

    private static double asymptoticBesselAmplitude(double v, double x) {
        double mu = 4.0 * v * v;
        double txq = 4.0 * x * x;
        double s = 1.0;
        s += (mu - 1.0) / (2.0 * txq);
        s += 3.0 * (mu - 1.0) * (mu - 9.0) / (txq * txq * 8.0);
        s += 15.0 * (mu - 1.0) * (mu - 9.0) * (mu - 25.0) / (txq * txq * txq * 48.0);
        return Math.sqrt(s * 2.0 / (Math.PI * x));
    }

    private static double asymptoticBesselPhaseMx(double v, double x) {
        double mu = 4.0 * v * v;
        double denom = 4.0 * x;
        double denomMult = denom * denom;
        double s = 0.0;
        s += (mu - 1.0) / (2.0 * denom);
        denom *= denomMult;
        s += (mu - 1.0) * (mu - 25.0) / (6.0 * denom);
        denom *= denomMult;
        s += (mu - 1.0) * (mu * mu - 114.0 * mu + 1073.0) / (5.0 * denom);
        denom *= denomMult;
        s += (mu - 1.0) * (5.0 * mu * mu * mu - 1535.0 * mu * mu + 54703.0 * mu - 375733.0) / (14.0 * denom);
        return s;
    }

    private static double asymptoticBesselYLargeX(double v, double x) {
        double amplitude = asymptoticBesselAmplitude(v, x);
        if (amplitude == 0.0) {
            return 0.0;
        }
        double phase = asymptoticBesselPhaseMx(v, x);
        double cx = Math.cos(x);
        double sx = Math.sin(x);
        double ci = cosPi(v / 2.0 + 0.25);
        double si = sinPi(v / 2.0 + 0.25);
        double sinPhase = Math.sin(phase) * (cx * ci + sx * si) + Math.cos(phase) * (sx * ci - cx * si);
        return sinPhase * amplitude;
    }

    private static double asymptoticBesselJLargeX(double v, double x) {
        double amplitude = asymptoticBesselAmplitude(v, x);
        if (amplitude == 0.0) {
            return 0.0;
        }
        double phase = asymptoticBesselPhaseMx(v, x);
        double cx = Math.cos(x);
        double sx = Math.sin(x);
        double ci = cosPi(v / 2.0 + 0.25);
        double si = sinPi(v / 2.0 + 0.25);
        double sinPhase = Math.cos(phase) * (cx * ci + sx * si) - Math.sin(phase) * (sx * ci - cx * si);
        return sinPhase * amplitude;
    }

    private static double asymptoticBesselDerivativeAmplitude(double v, double x) {
        double mu = 4.0 * v * v;
        double txq = 4.0 * x * x;
        double s = 1.0;
        s -= (mu - 3.0) / (2.0 * txq);
        s -= ((mu - 1.0) * (mu - 45.0)) / (txq * txq * 8.0);
        return Math.sqrt(s * 2.0 / (Math.PI * x));
    }

    private static double asymptoticBesselDerivativePhaseMx(double v, double x) {
        double mu = 4.0 * v * v;
        double mu2 = mu * mu;
        double mu3 = mu2 * mu;
        double denom = 4.0 * x;
        double denomMult = denom * denom;
        double s = 0.0;
        s += (mu + 3.0) / (2.0 * denom);
        denom *= denomMult;
        s += (mu2 + 46.0 * mu - 63.0) / (6.0 * denom);
        denom *= denomMult;
        s += (mu3 + 185.0 * mu2 - 2053.0 * mu + 1899.0) / (5.0 * denom);
        return s;
    }

    private static double asymptoticBesselYDerivativeLargeX(double v, double x) {
        double amplitude = asymptoticBesselDerivativeAmplitude(v, x);
        if (amplitude == 0.0) {
            return 0.0;
        }
        double phase = asymptoticBesselDerivativePhaseMx(v, x);
        double cx = Math.cos(x);
        double sx = Math.sin(x);
        double ci = cosPi(v / 2.0 - 0.25);
        double si = sinPi(v / 2.0 - 0.25);
        double sinPhase = Math.sin(phase) * (cx * ci + sx * si) + Math.cos(phase) * (sx * ci - cx * si);
        return sinPhase * amplitude;
    }

    private static double asymptoticBesselJDerivativeLargeX(double v, double x) {
        double amplitude = asymptoticBesselDerivativeAmplitude(v, x);
        if (amplitude == 0.0) {
            return 0.0;
        }
        double phase = asymptoticBesselDerivativePhaseMx(v, x);
        double cx = Math.cos(x);
        double sx = Math.sin(x);
        double ci = cosPi(v / 2.0 - 0.25);
        double si = sinPi(v / 2.0 - 0.25);
        double sinPhase = Math.cos(phase) * (cx * ci + sx * si) - Math.sin(phase) * (sx * ci - cx * si);
        return sinPhase * amplitude;
    }

    private static boolean asymptoticBesselLargeXLimit(double v, double x) {
        return Math.max(Math.abs(v), 1.0) < x * Math.sqrt(FORTH_ROOT_EPSILON);
    }

    private static boolean asymptoticBesselDerivativeLargeXLimit(double v, double x) {
        return asymptoticBesselLargeXLimit(v, x);
    }

    private static boolean isAsymptoticBesselILargeX(double v, double x) {
        double lim = (4.0 * v * v + 10.0) / (8.0 * x);
        lim *= lim;
        lim *= lim;
        lim /= 24.0;
        return (lim < MACHINE_EPSILON * 10.0) && (x > 100.0);
    }

    private static double gamma1pm1(double x) {
        return Math.expm1(Gamma.lgamma(1.0 + x));
    }

    private static double besselJDerivativeLinear(double v, double x) {
        return 0.5 * (j(v - 1.0, x) - j(v + 1.0, x));
    }

    private static double besselYDerivativeLinear(double v, double x) {
        return 0.5 * (y(v - 1.0, x) - y(v + 1.0, x));
    }

    private static double besselIDerivativeLinear(double v, double x) {
        double result = i(v - 1.0, x);
        if (result >= Double.MAX_VALUE) {
            return result;
        }
        double result2 = i(v + 1.0, x);
        return 0.5 * result + 0.5 * result2;
    }

    private static double besselKDerivativeLinear(double v, double x) {
        double result = k(v - 1.0, x);
        if (result >= Double.MAX_VALUE) {
            return -result;
        }
        double result2 = k(v + 1.0, x);
        if (result2 >= Double.MAX_VALUE - result) {
            return Double.NEGATIVE_INFINITY;
        }
        return -0.5 * result - 0.5 * result2;
    }

    private static double besselJDerivativeSmallZSeries(double v, double x) {
        double prefix;
        if (v < MAX_FACTORIAL) {
            prefix = Math.pow(x / 2.0, v - 1.0) / (2.0 * Gamma.tgamma(v + 1.0));
        } else {
            prefix = Math.exp((v - 1.0) * Math.log(x / 2.0) - Math.log(2.0) - Gamma.lgamma(v + 1.0));
        }
        if (prefix == 0.0) {
            return 0.0;
        }
        double mult = -0.25 * x * x;
        double term = 1.0;
        double sum = 0.0;
        int start = 1;
        if (sameParameter(v, 0.0)) {
            term *= mult;
            sum = 2.0;
            start = 2;
        }
        for (int n = start; n < MAX_ITERATIONS; n++) {
            sum += term * (v + 2.0 * (n - 1));
            term *= mult / (n * (n + v));
            if (Math.abs(term * (v + 2.0 * n)) <= Math.abs(sum) * MACHINE_EPSILON) {
                return prefix * sum;
            }
        }
        throw new IllegalStateException("Bessel J derivative small-z series did not converge for v=" + v + ", x=" + x);
    }

    private static double besselYDerivativeSmallZSeries(double v, double x) {
        double p = Math.log(x / 2.0);
        double scale = 1.0;
        boolean needLogs = (v >= MAX_FACTORIAL) || (LOG_MAX_VALUE / v < Math.abs(p));

        double prefix;
        if (!needLogs) {
            double gammaV = Gamma.tgamma(v);
            double power = 2.0 * Math.pow(x / 2.0, v + 1.0);
            if (Double.MAX_VALUE * power < gammaV) {
                scale /= gammaV;
                gammaV = 1.0;
                if (Double.MAX_VALUE * power < gammaV) {
                    return Double.POSITIVE_INFINITY;
                }
            }
            prefix = -gammaV / (Math.PI * power);
        } else {
            double logPrefix = Gamma.lgamma(v) - Math.log(Math.PI) - (v + 1.0) * p - Math.log(2.0);
            if (LOG_MAX_VALUE < logPrefix) {
                double divisor = Double.MAX_VALUE / 4.0;
                logPrefix -= Math.log(divisor);
                scale /= divisor;
                if (LOG_MAX_VALUE < logPrefix) {
                    return Double.POSITIVE_INFINITY;
                }
            }
            prefix = -Math.exp(logPrefix);
        }

        double mult = -0.25 * x * x;
        double term = 1.0;
        double result = 0.0;
        for (int n = 0; n < MAX_ITERATIONS; n++) {
            result += term * (-v + 2.0 * n);
            int next = n + 1;
            term *= mult / (next * (next - v));
            if (Math.abs(term * (-v + 2.0 * next)) <= Math.abs(result) * MACHINE_EPSILON) {
                break;
            }
            if (next + 1 == MAX_ITERATIONS) {
                throw new IllegalStateException("Bessel Y derivative small-z primary series did not converge for v=" + v + ", x=" + x);
            }
        }
        result *= prefix;

        double secondPrefix;
        if (!needLogs) {
            secondPrefix = completeExtended(-v) * cosPi(v) * Math.pow(x / 2.0, v - 1.0) / (2.0 * Math.PI);
        } else {
            secondPrefix = signExtended(-v)
                * Math.exp(logAbsExtended(-v) + (v - 1.0) * p - Math.log(2.0))
                * cosPi(v)
                / Math.PI;
        }

        term = 1.0;
        double second = 0.0;
        for (int n = 0; n < MAX_ITERATIONS; n++) {
            second += term * (v + 2.0 * n);
            int next = n + 1;
            term *= mult / (next * (next + v));
            if (Math.abs(term * (v + 2.0 * next)) <= Math.abs(second) * MACHINE_EPSILON) {
                double combined = result + scale * secondPrefix * second;
                if (scale * Double.MAX_VALUE < Math.abs(combined)) {
                    return Double.POSITIVE_INFINITY;
                }
                return combined / scale;
            }
        }
        throw new IllegalStateException("Bessel Y derivative small-z secondary series did not converge for v=" + v + ", x=" + x);
    }

    private static double sphBesselSmallZSeries(int n, double x) {
        double prefix;
        if (n + 1.5 < MAX_FACTORIAL) {
            prefix = Math.pow(x / 2.0, n) / Gamma.tgamma(n + 1.5);
        } else {
            prefix = Math.exp(n * Math.log(x / 2.0) - Gamma.lgamma(n + 1.5));
        }
        if (prefix == 0.0) {
            return 0.0;
        }
        double mult = -0.25 * x * x;
        double term = 1.0;
        double sum = 1.0;
        for (int k = 1; k < MAX_ITERATIONS; k++) {
            term *= mult / (k * (k + n + 0.5));
            sum += term;
            if (Math.abs(term) <= Math.abs(sum) * MACHINE_EPSILON) {
                return prefix * sum * Math.sqrt(Math.PI / 4.0);
            }
        }
        throw new IllegalStateException("Spherical Bessel small-z series did not converge for n=" + n + ", x=" + x);
    }

    private static double cylBesselJZero(double v, int m) {
        double halfEpsilon = MACHINE_EPSILON / 2.0;
        boolean orderIsNegative = v < 0.0;
        double absOrder = orderIsNegative ? -v : v;
        boolean orderIsZero = absOrder < halfEpsilon;
        boolean orderIsInteger = (absOrder - Math.floor(absOrder)) < halfEpsilon;

        if (m == 0) {
            if (orderIsZero) {
                throw new IllegalArgumentException("jZero: rank 0 is undefined for order 0");
            }
            if (orderIsNegative && !orderIsInteger) {
                throw new IllegalArgumentException("jZero: rank 0 is undefined for negative non-integer order");
            }
            return 0.0;
        }

        double evalOrder = orderIsInteger ? absOrder : v;
        double guess = initialGuessJZero(evalOrder, m);
        double deltaLo = guess > 0.2 ? 0.2 : guess / 2.0;
        return solveBesselRoot(
            x -> j(evalOrder, x),
            x -> orderIsZero ? -j(1.0, x) : jPrime(evalOrder, x),
            guess,
            Math.max(guess - deltaLo, Double.MIN_NORMAL),
            guess + 0.2,
            "jZero"
        );
    }

    private static double cylNeumannZero(double v, int m) {
        double halfEpsilon = MACHINE_EPSILON / 2.0;
        boolean orderIsNegative = v < 0.0;
        double absOrder = orderIsNegative ? -v : v;
        boolean orderIsInteger = (absOrder - Math.floor(absOrder)) < halfEpsilon;

        if (orderIsNegative && orderIsInteger) {
            return cylNeumannZero(absOrder, m);
        }

        double deltaHalfInteger = absOrder - (Math.floor(absOrder) + 0.5);
        boolean orderIsNegativeHalfInteger = orderIsNegative && (deltaHalfInteger > -halfEpsilon) && (deltaHalfInteger < halfEpsilon);

        if ((m == 0) && !orderIsNegativeHalfInteger) {
            throw new IllegalArgumentException("yZero: rank 0 is undefined for this order");
        }
        if (orderIsNegativeHalfInteger) {
            return cylBesselJZero(absOrder, m);
        }

        double guess = initialGuessYZero(v, m);
        double deltaLo = guess > 0.2 ? 0.2 : guess / 2.0;
        return solveBesselRoot(
            x -> y(v, x),
            x -> sameParameter(v, 0.0) ? -y(1.0, x) : yPrime(v, x),
            guess,
            Math.max(guess - deltaLo, Double.MIN_NORMAL),
            guess + 0.2,
            "yZero"
        );
    }

    private static double initialGuessJZero(double v, int m) {
        if (v < 0.0) {
            if ((m == 1) && (v > -0.5)) {
                return ((((((-0.2321156900729 * v - 0.1493247777488) * v - 0.15205419167239)
                    * v + 0.07814930561249) * v - 0.17757573537688) * v + 1.542805677045663) * v + 2.40482555769577277);
            }

            double absOrder = -v;
            double absFloor = Math.floor(absOrder);
            double rootHi = initialGuessJZero(absFloor, m);
            double rootLo;
            if (m == 1) {
                rootLo = rootHi - 0.1;
                boolean hiNegative = j(v, rootHi) < 0.0;
                while (rootLo > MACHINE_EPSILON) {
                    boolean loNegative = j(v, rootLo) < 0.0;
                    if (hiNegative != loNegative) {
                        break;
                    }
                    rootHi = rootLo;
                    rootLo = rootLo > 0.5 ? rootLo - 0.5 : rootLo * 0.75;
                }
            } else {
                rootLo = initialGuessJZero(absFloor, m - 1);
            }
            return bisectRoot(x -> j(v, x), rootLo, rootHi, 12, "jZero-initial");
        }

        if (m == 1) {
            if (v < 2.2) {
                return ((((((-0.0008342379046010 * v + 0.007590035637410) * v - 0.030640914772013)
                    * v + 0.078232088020106) * v - 0.169668712590620) * v + 1.542187960073750) * v + 2.4048359915254634);
            }
            return equationNist102140a(v);
        }

        if (v < 2.2) {
            double a = ((v + 2.0 * m) - 0.5) * (Math.PI / 2.0);
            return equationNist102119(v, a);
        }
        return equationAs9526(v, Airy.initialAiZero(m));
    }

    private static double initialGuessYZero(double v, int m) {
        if (v < 0.0) {
            double absOrder = -v;
            double absFloor = Math.floor(absOrder);
            double rootHi;
            double rootLo;

            if (m == 1) {
                if ((absOrder - absFloor) < 0.5) {
                    rootHi = initialGuessYZero(absFloor, m);
                } else {
                    rootHi = initialGuessJZero(absFloor + 0.5, m);
                }
                rootLo = rootHi - 0.1;
                boolean hiNegative = y(v, rootHi) < 0.0;
                while (rootLo > MACHINE_EPSILON) {
                    boolean loNegative = y(v, rootLo) < 0.0;
                    if (hiNegative != loNegative) {
                        break;
                    }
                    rootHi = rootLo;
                    rootLo = rootLo > 0.5 ? rootLo - 0.5 : rootLo * 0.75;
                }
            } else {
                if ((absOrder - absFloor) < 0.5) {
                    rootLo = initialGuessYZero(absFloor, m - 1) + 0.01;
                    rootHi = initialGuessYZero(absFloor, m) + 0.01;
                } else {
                    rootLo = initialGuessJZero(absFloor + 0.5, m - 1) + 0.01;
                    rootHi = initialGuessJZero(absFloor + 0.5, m) + 0.01;
                }
            }
            return bisectRoot(x -> y(v, x), rootLo, rootHi, 12, "yZero-initial");
        }

        if (m == 1) {
            if (v < 2.2) {
                return ((((((-0.0025095909235652 * v + 0.021291887049053) * v - 0.076487785486526)
                    * v + 0.159110268115362) * v - 0.241681668765196) * v + 1.4437846310885244) * v + 0.89362115190200490);
            }
            return equationNist102140b(v);
        }

        if (v < 2.2) {
            double a = ((v + 2.0 * m) - 1.5) * (Math.PI / 2.0);
            return equationNist102119(v, a);
        }
        return equationAs9526(v, Airy.initialBiZero(m));
    }

    private static double equationNist102119(double v, double a) {
        double mu = 4.0 * v * v;
        double muMinusOne = mu - 1.0;
        double eightAInv = 1.0 / (8.0 * a);
        double eightAInvSquared = eightAInv * eightAInv;

        double term3 = ((muMinusOne * 4.0) * (mu * 7.0 - 31.0)) / 3.0;
        double term5 = ((muMinusOne * 32.0) * (((mu * 83.0) - 982.0) * mu + 3779.0)) / 15.0;
        double term7 = ((muMinusOne * 64.0) * (((((mu * 6949.0) - 153855.0) * mu) + 1585743.0) * mu - 6277237.0)) / 105.0;

        return a + (((( -term7 * eightAInvSquared - term5) * eightAInvSquared - term3) * eightAInvSquared - muMinusOne) * eightAInv);
    }

    private static double equationNist102140a(double v) {
        double vPowThird = Math.cbrt(v);
        double vPowMinusTwoThirds = 1.0 / (vPowThird * vPowThird);
        return v * (((((0.043 * vPowMinusTwoThirds - 0.0908) * vPowMinusTwoThirds - 0.00397)
            * vPowMinusTwoThirds + 1.033150) * vPowMinusTwoThirds + 1.8557571) * vPowMinusTwoThirds + 1.0);
    }

    private static double equationNist102140b(double v) {
        double vPowThird = Math.cbrt(v);
        double vPowMinusTwoThirds = 1.0 / (vPowThird * vPowThird);
        return v * ((((( -0.001 * vPowMinusTwoThirds - 0.0060) * vPowMinusTwoThirds + 0.01198)
            * vPowMinusTwoThirds + 0.260351) * vPowMinusTwoThirds + 0.9315768) * vPowMinusTwoThirds + 1.0);
    }

    private static double equationAs9526(double v, double airyRoot) {
        double vPowThird = Math.cbrt(v);
        double vPowMinusTwoThirds = 1.0 / (vPowThird * vPowThird);
        double zeta = vPowMinusTwoThirds * (-airyRoot);
        double zetaSqrt = Math.sqrt(zeta);

        double b = -((((zeta * zetaSqrt) * 2.0) / 3.0) + (Math.PI / 2.0));
        double zEstimate = (-b + Math.sqrt(b * b - 2.0)) / 2.0;
        double z = solveEquationAs93139(zeta, zEstimate);

        double zsqMinusOne = z * z - 1.0;
        double zsqMinusOneSqrt = Math.sqrt(zsqMinusOne);
        double b0Term524 = 5.0 / ((zsqMinusOne * zsqMinusOneSqrt) * 24.0);
        double b0Term18 = 1.0 / (zsqMinusOneSqrt * 8.0);
        double b0Term548 = 5.0 / ((zeta * zeta) * 48.0);
        double b0 = -b0Term548 + ((b0Term524 + b0Term18) / zetaSqrt);
        double f1 = ((z * zetaSqrt) * b0) / zsqMinusOneSqrt;
        return (v * z) + (f1 / v);
    }

    private static double solveEquationAs93139(double zeta, double initialGuess) {
        Double2 bracket = bracketEquationAs93139Root(zeta, initialGuess);
        double lower = bracket.lower();
        double upper = bracket.upper();
        if (lower == upper) {
            return lower;
        }

        return RootIterators.newtonRaphsonIterate(
            (z, evaluation) -> equationAs93139(z, zeta, evaluation),
            initialGuess,
            lower,
            upper,
            53,
            ROOT_MAX_ITERATIONS,
            "Bessel zero auxiliary Newton iteration"
        );
    }

    private static Double2 bracketEquationAs93139Root(double zeta, double initialGuess) {
        double lower = Math.max(initialGuess - 1.0, 1.0);
        double upper = initialGuess + 1.0;
        Double2 lowerEvaluation = equationAs93139(lower, zeta);
        if (lowerEvaluation._1() == 0.0) {
            return Double2.bound(lower, lower);
        }
        if (lowerEvaluation._1() > 0.0) {
            lower = 1.0;
            lowerEvaluation = equationAs93139(lower, zeta);
            if (lowerEvaluation._1() == 0.0) {
                return Double2.bound(lower, lower);
            }
            if (lowerEvaluation._1() > 0.0) {
                throw new IllegalStateException("Bessel zero auxiliary Newton iteration failed to bracket root for zeta=" + zeta);
            }
        }

        Double2 upperEvaluation = equationAs93139(upper, zeta);
        if (upperEvaluation._1() == 0.0) {
            return Double2.bound(upper, upper);
        }

        double step = Math.max(1.0, upper - lower);
        for (int iteration = 0; iteration < ROOT_MAX_ITERATIONS; iteration++) {
            if (Math.signum(lowerEvaluation._1()) != Math.signum(upperEvaluation._1())) {
                return Double2.bound(lower, upper);
            }
            lower = upper;
            lowerEvaluation = upperEvaluation;
            upper += step;
            upperEvaluation = equationAs93139(upper, zeta);
            if (upperEvaluation._1() == 0.0) {
                return Double2.bound(upper, upper);
            }
            step *= 2.0;
        }
        throw new IllegalStateException("Bessel zero auxiliary Newton iteration failed to bracket root for zeta=" + zeta);
    }

    private static Double2 equationAs93139(double z, double zeta) {
        double zsqMinusOneSqrt = Math.sqrt(z * z - 1.0);
        double function = zsqMinusOneSqrt - (Math.acos(1.0 / z) + ((2.0 / 3.0) * (zeta * Math.sqrt(zeta))));
        double derivative = zsqMinusOneSqrt / z;
        return new Double2(function, derivative);
    }

    private static void equationAs93139(double z, double zeta, double[] evaluation) {
        double zsqMinusOneSqrt = Math.sqrt(z * z - 1.0);
        evaluation[0] = zsqMinusOneSqrt - (Math.acos(1.0 / z) + ((2.0 / 3.0) * (zeta * Math.sqrt(zeta))));
        evaluation[1] = zsqMinusOneSqrt / z;
    }

    private static double bisectRoot(DoubleUnaryOperator function, double lower, double upper, int iterations, String label) {
        double fLower = function.applyAsDouble(lower);
        double fUpper = function.applyAsDouble(upper);
        if (sameSign(fLower, fUpper)) {
            throw new IllegalStateException(label + " requires a sign-changing bracket");
        }
        double lo = lower;
        double hi = upper;
        for (int iteration = 0; iteration < iterations; iteration++) {
            double mid = 0.5 * (lo + hi);
            double fMid = function.applyAsDouble(mid);
            if (fMid == 0.0) {
                return mid;
            }
            if (sameSign(fLower, fMid)) {
                lo = mid;
                fLower = fMid;
            } else {
                hi = mid;
                fUpper = fMid;
            }
        }
        return 0.5 * (lo + hi);
    }

    private static double solveBesselRoot(DoubleUnaryOperator function,
                                          DoubleUnaryOperator derivative,
                                          double guess,
                                          double lower,
                                          double upper,
                                          String label) {
        Double2 bracket = bracketRootAroundGuess(function, lower, upper, label);
        double lo = bracket.lower();
        double hi = bracket.upper();
        double fLo = function.applyAsDouble(lo);
        double fHi = function.applyAsDouble(hi);
        if (fLo == 0.0) {
            return lo;
        }
        if (fHi == 0.0) {
            return hi;
        }
        return RootIterators.newtonRaphsonIterate(
            (x, evaluation) -> {
                evaluation[0] = function.applyAsDouble(x);
                evaluation[1] = derivative.applyAsDouble(x);
            },
            Math.max(Math.min(guess, hi), lo),
            lo,
            hi,
            53,
            ROOT_MAX_ITERATIONS,
            label
        );
    }

    private static Double2 bracketRootAroundGuess(DoubleUnaryOperator function, double lower, double upper, String label) {
        double lo = Math.max(lower, Double.MIN_NORMAL);
        double hi = upper;
        double fLo = function.applyAsDouble(lo);
        if (fLo == 0.0) {
            return Double2.bound(lo, lo);
        }
        double fHi = function.applyAsDouble(hi);
        if (fHi == 0.0) {
            return Double2.bound(hi, hi);
        }
        double step = 0.2;
        for (int iteration = 0; iteration < ROOT_MAX_ITERATIONS; iteration++) {
            if (!sameSign(fLo, fHi)) {
                return Double2.bound(lo, hi);
            }
            lo = Math.max(Double.MIN_NORMAL, lo - step);
            hi += step;
            fLo = function.applyAsDouble(lo);
            if (fLo == 0.0) {
                return Double2.bound(lo, lo);
            }
            fHi = function.applyAsDouble(hi);
            if (fHi == 0.0) {
                return Double2.bound(hi, hi);
            }
            step *= 1.5;
        }
        throw new IllegalStateException(label + " failed to bracket root");
    }

    private static double cylBesselI(double v, double x, boolean weighted) {
        if (x < 0.0) {
            if (!isInteger(v)) {
                throw new IllegalArgumentException("i: x must be non-negative for non-integer order: " + x);
            }
            long integerOrder = Math.round(Math.rint(v));
            double value = cylBesselI(v, -x, weighted);
            return (integerOrder & 1L) == 0L ? value : -value;
        }
        return cylBesselIPositiveX(v, x, weighted);
    }

    private static double cylBesselIPositiveX(double v, double x, boolean weighted) {
        if (x == 0.0) {
            if (v < 0.0 && !isInteger(v)) {
                return Double.POSITIVE_INFINITY;
            }
            return v == 0.0 ? 1.0 : 0.0;
        }
        if (sameParameter(v, 0.5)) {
            if (weighted) {
                return -Math.expm1(-2.0 * x) / Math.sqrt(2.0 * x * Math.PI);
            }
            if (x >= LOG_MAX_VALUE) {
                double e = Math.exp(x / 2.0);
                return e * (e / Math.sqrt(2.0 * x * Math.PI));
            }
            return Math.sqrt(2.0 / (x * Math.PI)) * Math.sinh(x);
        }
        if (sameParameter(v, 0.0)) {
            return besselI0(x, weighted);
        }
        if (sameParameter(v, 1.0)) {
            return besselI1(x, weighted);
        }
        if (v > 0.0 && x / v < 0.25) {
            double result = besselISmallZSeries(v, x);
            return weighted ? result * Math.exp(-x) : result;
        }

        Double2 pair = besselIK(v, x, NEED_I);
        double iv = pair._1();
        if (!weighted) {
            return iv;
        }
        if (Double.isFinite(iv)) {
            return iv * Math.exp(-x);
        }
        return asymptoticBesselILargeX(v, x, true);
    }

    private static double cylBesselK(double v, double x, boolean weighted) {
        if (x < 0.0) {
            throw new IllegalArgumentException("k: x must be non-negative: " + x);
        }
        if (x == 0.0) {
            if (sameParameter(v, 0.0)) {
                return Double.POSITIVE_INFINITY;
            }
            throw new IllegalArgumentException("k: x must be positive: " + x);
        }

        double raw;
        if (isInteger(v)) {
            raw = besselKn((int) Math.rint(v), x);
        } else {
            Double2 pair = besselIK(v, x, NEED_K);
            raw = pair._2();
        }
        return weighted ? raw * Math.exp(-x) : raw;
    }

    private static double besselISmallZSeries(double v, double x) {
        double prefix;
        if (v < MAX_FACTORIAL) {
            prefix = Math.pow(x / 2.0, v) / Gamma.tgamma(v + 1.0);
        } else {
            prefix = Math.exp(v * Math.log(x / 2.0) - Gamma.lgamma(v + 1.0));
        }
        if (prefix == 0.0) {
            return 0.0;
        }

        double mult = x * x / 4.0;
        double term = 1.0;
        double sum = 1.0;
        for (int k = 1; k < MAX_ITERATIONS; k++) {
            term *= mult / k;
            term /= k + v;
            sum += term;
            if (Math.abs(term) <= Math.abs(sum) * MACHINE_EPSILON) {
                return prefix * sum;
            }
        }
        throw new IllegalStateException("Bessel I small-z series did not converge for v=" + v + ", x=" + x);
    }

    private static Double2 besselIK(double v, double x, int kind) {
        boolean reflect = false;
        if (v < 0.0) {
            reflect = true;
            v = -v;
            kind |= NEED_K;
        }
        if (v > Integer.MAX_VALUE) {
            throw new IllegalStateException("Bessel order is too large to evaluate: " + v);
        }

        int n = (int) Math.rint(v);
        double u = v - n;
        double iv;
        double kv;
        double kv1;

        if (x <= 2.0) {
            Double2 seeds = temmeIK(u, x);
            kv = seeds._1();
            kv1 = seeds._2();
        } else {
            Double2 seeds = cf2IK(u, x);
            kv = seeds._1();
            kv1 = seeds._2();
        }

        double prev = kv;
        double current = kv1;
        double scale = 1.0;
        for (int k = 1; k <= n; k++) {
            double factor = 2.0 * (u + k) / x;
            if ((factor < 1.0) && ((Double.MAX_VALUE - Math.abs(prev)) / factor < Math.abs(current))) {
                prev /= current;
                scale /= current;
                current = 1.0;
            }
            double next = factor * current + prev;
            prev = current;
            current = next;
        }
        kv = prev;
        kv1 = current;

        if ((kind & NEED_I) != 0) {
            double lim = (4.0 * v * v + 10.0) / (8.0 * x);
            lim *= lim;
            lim *= lim;
            lim /= 24.0;
            if ((lim < MACHINE_EPSILON * 10.0) && (x > 100.0)) {
                iv = asymptoticBesselILargeX(v, x, false);
            } else if ((v > 0.0) && (x / v < 0.25)) {
                iv = besselISmallZSeries(v, x);
            } else {
                double fv = cf1IK(v, x);
                iv = scale * (1.0 / x) / (kv * fv + kv1);
            }
        } else {
            iv = Double.NaN;
        }

        if (reflect && (kind & NEED_I) != 0) {
            double z = u + (n & 1);
            double fact = (2.0 / Math.PI) * sinPi(z) * kv;
            if (fact != 0.0) {
                iv += fact / scale;
            }
        }
        return new Double2(iv, kv / scale);
    }

    private static Double2 temmeIK(double v, double x) {
        if (Math.abs(x) > 2.0 || Math.abs(v) > 0.5) {
            throw new IllegalStateException("Temme IK requires |x| <= 2 and |v| <= 0.5: v=" + v + ", x=" + x);
        }

        double gp = Gamma.tgamma1pm1(v);
        double gm = Gamma.tgamma1pm1(-v);
        double a = Math.log(x / 2.0);
        double b = Math.exp(v * a);
        double sigma = -a * v;
        double c = Math.abs(v) < MACHINE_EPSILON ? 1.0 : sinPi(v) / (v * Math.PI);
        double d = Math.abs(sigma) < MACHINE_EPSILON ? 1.0 : Math.sinh(sigma) / sigma;
        double gamma1 = Math.abs(v) < MACHINE_EPSILON ? -EULER : 0.5 * (gp - gm) * c / v;
        double gamma2 = (2.0 + gp + gm) * c / 2.0;

        double p = (gp + 1.0) / (2.0 * b);
        double q = (1.0 + gm) * b / 2.0;
        double f = (Math.cosh(sigma) * gamma1 + d * (-a) * gamma2) / c;
        double h = p;
        double coef = 1.0;
        double sum = f;
        double sum1 = h;

        for (int k = 1; k < MAX_ITERATIONS; k++) {
            f = (k * f + p + q) / (k * (double) k - v * v);
            p /= k - v;
            q /= k + v;
            h = p - k * f;
            coef *= x * x / (4.0 * k);
            sum += coef * f;
            sum1 += coef * h;
            if (Math.abs(coef * f) < Math.abs(sum) * MACHINE_EPSILON) {
                return new Double2(sum, 2.0 * sum1 / x);
            }
        }
        throw new IllegalStateException("Temme IK did not converge for v=" + v + ", x=" + x);
    }

    private static double cf1IK(double v, double x) {
        double tolerance = 2.0 * MACHINE_EPSILON;
        double tiny = ROOT_MIN_NORMAL;
        double c = tiny;
        double d = 0.0;
        double f = tiny;

        for (int k = 1; k < MAX_ITERATIONS; k++) {
            double b = 2.0 * (v + k) / x;
            c = b + 1.0 / c;
            d = b + d;
            if (c == 0.0) {
                c = tiny;
            }
            if (d == 0.0) {
                d = tiny;
            }
            d = 1.0 / d;
            double delta = c * d;
            f *= delta;
            if (Math.abs(delta - 1.0) <= tolerance) {
                return f;
            }
        }
        throw new IllegalStateException("Bessel CF1_ik did not converge for v=" + v + ", x=" + x);
    }

    private static Double2 cf2IK(double v, double x) {
        if (!(Math.abs(x) > 1.0)) {
            throw new IllegalStateException("Bessel CF2_ik requires |x| > 1: " + x);
        }

        double tolerance = MACHINE_EPSILON;
        double a = v * v - 0.25;
        double b = 2.0 * (x + 1.0);
        double d = 1.0 / b;
        double f = d;
        double delta = d;
        double prev = 0.0;
        double current = 1.0;
        double q = 0.0;
        double c = -a;
        double bigQ = c;
        double s = 1.0 + bigQ * delta;

        for (int k = 2; k < MAX_ITERATIONS; k++) {
            a -= 2.0 * (k - 1.0);
            b += 2.0;
            d = 1.0 / (b + a * d);
            delta *= b * d - 1.0;
            f += delta;

            q = (prev - (b - 2.0) * current) / a;
            prev = current;
            current = q;
            c *= -a / k;
            bigQ += c * q;
            s += bigQ * delta;

            if (Math.abs(q) < MACHINE_EPSILON) {
                c *= q;
                prev /= q;
                current /= q;
                q = 1.0;
            }
            if (Math.abs(bigQ * delta) < Math.abs(s) * tolerance) {
                double kv = Math.sqrt(Math.PI / (2.0 * x)) * Math.exp(-x) / s;
                double kv1 = kv * (0.5 + v + x + (v * v - 0.25) * f) / x;
                return new Double2(kv, kv1);
            }
        }
        throw new IllegalStateException("Bessel CF2_ik did not converge for v=" + v + ", x=" + x);
    }

    private static double asymptoticBesselILargeX(double v, double x, boolean weighted) {
        double s = 1.0;
        double mu = 4.0 * v * v;
        double ex = 8.0 * x;
        double num = mu - 1.0;
        double denom = ex;

        s -= num / denom;
        num *= mu - 9.0;
        denom *= ex * 2.0;
        s += num / denom;
        num *= mu - 25.0;
        denom *= ex * 3.0;
        s -= num / denom;

        if (weighted) {
            return s / Math.sqrt(2.0 * x * Math.PI);
        }
        double e = Math.exp(x / 2.0);
        return e * (e * s / Math.sqrt(2.0 * x * Math.PI));
    }

    private static double besselI0(double x, boolean weighted) {
        if (x < 7.75) {
            double a = x * x / 4.0;
            double result = a * evaluatePolynomial(BESSEL_I0_SMALL_COEFFICIENTS, a) + 1.0;
            return weighted ? result * Math.exp(-x) : result;
        }
        if (x < 500.0) {
            double core = evaluatePolynomial(BESSEL_I0_MEDIUM_COEFFICIENTS, 1.0 / x) / Math.sqrt(x);
            return weighted ? core : Math.exp(x) * core;
        }
        double core = evaluatePolynomial(BESSEL_I0_LARGE_COEFFICIENTS, 1.0 / x) / Math.sqrt(x);
        if (weighted) {
            return core;
        }
        double e = Math.exp(x / 2.0);
        return e * core * e;
    }

    private static double besselI1(double x, boolean weighted) {
        if (x < 7.75) {
            double a = x * x / 4.0;
            double[] q = {1.0, 0.5, evaluatePolynomial(BESSEL_I1_SMALL_COEFFICIENTS, a)};
            double result = x * evaluatePolynomial(q, a) / 2.0;
            return weighted ? result * Math.exp(-x) : result;
        }
        if (x < 500.0) {
            double core = evaluatePolynomial(BESSEL_I1_MEDIUM_COEFFICIENTS, 1.0 / x) / Math.sqrt(x);
            return weighted ? core : Math.exp(x) * core;
        }
        double core = evaluatePolynomial(BESSEL_I1_LARGE_COEFFICIENTS, 1.0 / x) / Math.sqrt(x);
        if (weighted) {
            return core;
        }
        double e = Math.exp(x / 2.0);
        return e * core * e;
    }

    private static double besselK0(double x) {
        if (x <= 1.0) {
            double a = x * x / 4.0;
            a = (evaluatePolynomial(BESSEL_K0_SMALL_P, a) / evaluatePolynomial(BESSEL_K0_SMALL_Q, a) + 1.137250900268554688) * a + 1.0;

            return evaluatePolynomial(BESSEL_K0_SMALL_LOG_P, x * x) - Math.log(x) * a;
        }
        double core = (evaluateRational(BESSEL_K0_LARGE_P, BESSEL_K0_LARGE_Q, 1.0 / x) + 1.0) / Math.sqrt(x);
        if (x < LOG_MAX_VALUE) {
            return core * Math.exp(-x);
        }
        double e = Math.exp(-x / 2.0);
        return core * e * e;
    }

    private static double besselK1(double x) {
        if (x <= 1.0) {
            double a = x * x / 4.0;
            a = ((evaluateRational(BESSEL_K1_SMALL_P, BESSEL_K1_SMALL_Q, a) + 8.69547128677368164e-02) * a * a + a / 2.0 + 1.0) * x / 2.0;

            return evaluateRational(BESSEL_K1_SMALL_LOG_P, BESSEL_K1_SMALL_LOG_Q, x * x) * x + 1.0 / x + Math.log(x) * a;
        }
        double core = (evaluateRational(BESSEL_K1_LARGE_P, BESSEL_K1_LARGE_Q, 1.0 / x) + 1.45034217834472656) / Math.sqrt(x);
        if (x < LOG_MAX_VALUE) {
            return core * Math.exp(-x);
        }
        double e = Math.exp(-x / 2.0);
        return core * e * e;
    }

    private static double besselKn(int n, double x) {
        if (n < 0) {
            n = -n;
        }
        if (n == 0) {
            return besselK0(x);
        }
        if (n == 1) {
            return besselK1(x);
        }
        double prev = besselK0(x);
        double current = besselK1(x);
        double scale = 1.0;
        for (int k = 1; k < n; k++) {
            double factor = 2.0 * k / x;
            if ((Double.MAX_VALUE - Math.abs(prev)) / factor < Math.abs(current)) {
                scale /= current;
                prev /= current;
                current = 1.0;
            }
            double value = factor * current + prev;
            prev = current;
            current = value;
        }
        return current / scale;
    }

    private static boolean sameSign(double left, double right) {
        return (left < 0.0) == (right < 0.0);
    }

    private static void validateNonNegativeOrder(int order, String label) {
        if (order < 0) {
            throw new IllegalArgumentException(label + " must be non-negative: " + order);
        }
    }

}