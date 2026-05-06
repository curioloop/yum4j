package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.Commons.sinPi;
import static com.curioloop.yum4j.math.Commons.validateFinite;
import static com.curioloop.yum4j.math.HornerPoly.evaluatePolynomial;

/**
 * Real scalar Riemann zeta function aligned with Boost.Math's double-precision surface.
 *
 * <p>The public surface exposed here is the real-valued scalar function
 *
 * <pre>
 * zeta(s) = sum(n^(-s), n = 1..infinity), Re(s) > 1
 * </pre>
 *
 * analytically continued to the real line except for the pole at {@code s = 1}.
 *
 * <p>Implementation notes:
 *
 * <ul>
 *   <li>The positive half-line reuses Boost.Math's double-precision piecewise rational fits.</li>
 *   <li>Negative arguments use the functional equation with a log-gamma fallback when the
 *       reflected gamma factor would overflow a direct evaluation.</li>
 *   <li>Exact negative even integers return the trivial zeros directly.</li>
 *   <li>The singular point {@code s = 1} raises {@link ArithmeticException}.</li>
 * </ul>
 */
public final class Zeta {

    private static final double ROOT_EPSILON = Math.sqrt(Math.ulp(1.0));
    private static final double LOG_ROOT_TWO_PI = 0.91893853320467274178;
    private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);
    private static final double LOG_MAX_VALUE = Math.log(Double.MAX_VALUE);
    private static final double LOG_MIN_NORMAL = Math.log(Double.MIN_NORMAL);
    private static final double DOUBLE_DIGITS = 53.0;
    private static final double ZETA_2_4_OFFSET = 0.6986598968505859375;

    private static final double[] REGION_LT_ONE_P = {
        0.24339294433593750202,
        -0.49092470516353571651,
        0.0557616214776046784287,
        -0.00320912498879085894856,
        0.000451534528645796438704,
        -0.933241270357061460782e-5
    };
    private static final double[] REGION_LT_ONE_Q = {
        1.0,
        -0.279960334310344432495,
        0.0419676223309986037706,
        -0.00413421406552171059003,
        0.00024978985622317935355,
        -0.101855788418564031874e-4
    };
    private static final double[] REGION_LE_TWO_P = {
        0.577215664901532860516,
        0.243210646940107164097,
        0.0417364673988216497593,
        0.00390252087072843288378,
        0.000249606367151877175456,
        0.110108440976732897969e-4
    };
    private static final double[] REGION_LE_TWO_Q = {
        1.0,
        0.295201277126631761737,
        0.043460910607305495864,
        0.00434930582085826330659,
        0.000255784226140488490982,
        0.10991819782396112081e-4
    };
    private static final double[] REGION_LE_FOUR_P = {
        -0.0537258300023595030676,
        0.0445163473292365591906,
        0.0128677673534519952905,
        0.00097541770457391752726,
        0.769875101573654070925e-4,
        0.328032510000383084155e-5
    };
    private static final double[] REGION_LE_FOUR_Q = {
        1.0,
        0.33383194553034051422,
        0.0487798431291407621462,
        0.00479039708573558490716,
        0.000270776703956336357707,
        0.106951867532057341359e-4,
        0.236276623974978646399e-7
    };
    private static final double[] REGION_LE_SEVEN_P = {
        -2.49710190602259410021,
        -2.60013301809475665334,
        -0.939260435377109939261,
        -0.138448617995741530935,
        -0.00701721240549802377623,
        -0.229257310594893932383e-4
    };
    private static final double[] REGION_LE_SEVEN_Q = {
        1.0,
        0.706039025937745133628,
        0.15739599649558626358,
        0.0106117950976845084417,
        -0.36910273311764618902e-4,
        0.493409563927590008943e-5,
        -0.234055487025287216506e-6,
        0.718833729365459760664e-8,
        -0.1129200113474947419e-9
    };
    private static final double[] REGION_LT_FIFTEEN_P = {
        -4.78558028495135619286,
        -1.89197364881972536382,
        -0.211407134874412820099,
        -0.000189204758260076688518,
        0.00115140923889178742086,
        0.639949204213164496988e-4,
        0.139348932445324888343e-5
    };
    private static final double[] REGION_LT_FIFTEEN_Q = {
        1.0,
        0.244345337378188557777,
        0.00873370754492288653669,
        -0.00117592765334434471562,
        -0.743743682899933180415e-4,
        -0.21750464515767984778e-5,
        0.471001264003076486547e-8,
        -0.833378440625385520576e-10,
        0.699841545204845636531e-12
    };
    private static final double[] REGION_LT_THIRTY_SIX_P = {
        -10.3948950573308896825,
        -2.85827219671106697179,
        -0.347728266539245787271,
        -0.0251156064655346341766,
        -0.00119459173416968685689,
        -0.382529323507967522614e-4,
        -0.785523633796723466968e-6,
        -0.821465709095465524192e-8
    };
    private static final double[] REGION_LT_THIRTY_SIX_Q = {
        1.0,
        0.208196333572671890965,
        0.0195687657317205033485,
        0.00111079638102485921877,
        0.408507746266039256231e-4,
        0.955561123065693483991e-6,
        0.118507153474022900583e-7,
        0.222609483627352615142e-14
    };

    private Zeta() {
    }

    /**
     * Returns the real scalar Riemann zeta function.
     *
     * @param s finite real argument, excluding the pole at {@code s = 1}
     * @return {@code zeta(s)}
     */
    public static double zeta(double s) {
        validateFinite(s, "zeta: s");

        double sc = 1.0 - s;
        if (sc == 0.0) {
            throw new ArithmeticException("zeta pole at s = 1.0");
        }
        if (s > DOUBLE_DIGITS) {
            return 1.0;
        }

        double rounded = Math.rint(s);
        if (rounded == s && Math.abs(rounded) <= Integer.MAX_VALUE) {
            return zetaInteger((int) rounded);
        }

        if (Math.abs(s) < ROOT_EPSILON) {
            return -0.5 - LOG_ROOT_TWO_PI * s;
        }
        if (s < 0.0) {
            return reflect(s);
        }
        return zetaPositive(s, sc);
    }

    static double zetaPositiveInteger(int s) {
        if (s <= 1) {
            throw new IllegalArgumentException("zeta integer argument must be > 1: " + s);
        }
        return zetaPositive(s, 1.0 - s);
    }

    private static double zetaInteger(int s) {
        if (s == 1) {
            throw new ArithmeticException("zeta pole at s = 1.0");
        }
        if (s == 0) {
            return -0.5;
        }
        if (s < 0 && ((-s) & 1) == 0) {
            return 0.0;
        }
        if (s < 0) {
            return reflect(s);
        }
        return zetaPositive(s, 1.0 - s);
    }

    private static double reflect(double s) {
        double reflectedArgument = 1.0 - s;
        double reflected = zetaPositive(reflectedArgument, s);
        double multiplier = 2.0 * sinPi(0.5 * s) * reflected;
        if (multiplier == 0.0) {
            return 0.0;
        }

        if (reflectedArgument > 170.0) {
            double logScale = Gamma.lgamma(reflectedArgument) - reflectedArgument * LOG_TWO_PI;
            double logResult = Math.log(Math.abs(multiplier)) + logScale;
            if (logResult > LOG_MAX_VALUE) {
                throw new ArithmeticException("zeta overflow at s = " + s);
            }
            if (logResult < LOG_MIN_NORMAL) {
                return Math.copySign(0.0, multiplier);
            }
            return Math.copySign(Math.exp(logResult), multiplier);
        }

        return multiplier * Math.pow(2.0 * Math.PI, -reflectedArgument) * Gamma.tgamma(reflectedArgument);
    }

    private static double zetaPositive(double s, double sc) {
        if (s < 1.0) {
            double result = evaluatePolynomial(REGION_LT_ONE_P, sc) / evaluatePolynomial(REGION_LT_ONE_Q, sc);
            result -= 1.2433929443359375;
            result += sc;
            return result / sc;
        }
        if (s <= 2.0) {
            double negSc = -sc;
            return evaluatePolynomial(REGION_LE_TWO_P, negSc) / evaluatePolynomial(REGION_LE_TWO_Q, negSc) + 1.0 / negSc;
        }
        if (s <= 4.0) {
            double shifted = s - 2.0;
            return evaluatePolynomial(REGION_LE_FOUR_P, shifted) / evaluatePolynomial(REGION_LE_FOUR_Q, shifted)
                + ZETA_2_4_OFFSET + 1.0 / (-sc);
        }
        if (s <= 7.0) {
            double shifted = s - 4.0;
            return 1.0 + Math.exp(evaluatePolynomial(REGION_LE_SEVEN_P, shifted)
                / evaluatePolynomial(REGION_LE_SEVEN_Q, shifted));
        }
        if (s < 15.0) {
            double shifted = s - 7.0;
            return 1.0 + Math.exp(evaluatePolynomial(REGION_LT_FIFTEEN_P, shifted)
                / evaluatePolynomial(REGION_LT_FIFTEEN_Q, shifted));
        }
        if (s < 36.0) {
            double shifted = s - 15.0;
            return 1.0 + Math.exp(evaluatePolynomial(REGION_LT_THIRTY_SIX_P, shifted)
                / evaluatePolynomial(REGION_LT_THIRTY_SIX_Q, shifted));
        }
        return 1.0 + Math.pow(2.0, -s);
    }
}