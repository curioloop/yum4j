package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.HornerPoly.evaluatePolynomial;

/**
 * Real scalar exponential integrals aligned with Boost.Math 1.90.
 *
 * <p>The public real surface mirrors Boost's overload set:
 *
 * <pre>
 * expint(z)      = Ei(z)
 * expint(n, z)   = E_n(z), n >= 0, z >= 0
 * </pre>
 *
 * <p>Implementation notes:
 *
 * <ul>
 *   <li>{@code Ei(z)} follows Boost's double-precision rational regions for positive real
 *       arguments and reuses {@code E1(-z)} on the negative half-line.</li>
 *   <li>{@code E_n(z)} uses the same Boost branch split between the small-{@code z} series and
 *       the continued fraction, with the dedicated {@code E1} rational kept as its own fast path.</li>
 *   <li>Domain errors raise {@link IllegalArgumentException}; indexed overflow singularities such as
 *       {@code expint(1, 0)} raise {@link ArithmeticException}. Unary {@code Ei} retains the Boost
 *       real return values at its logarithmic/overflow limits, including {@code Ei(0) = -infinity}.</li>
 * </ul>
 */
public final class Expint {

    private static final double EPSILON = 8.0 * Math.ulp(1.0);
    private static final int MAX_ITERATIONS = 100000;
    private static final double LOG_MAX_VALUE = Math.log(Double.MAX_VALUE);
    private static final double E1_UNDERFLOW_THRESHOLD = -Math.log(Double.MIN_NORMAL);
    private static final double EXP_40 = 2.35385266837019985407899910749034804508871617254555467236651e17;
    private static final double EI_ROOT_R1 = 1677624236387711.0 / 4503599627370496.0;
    private static final double EI_ROOT_R2 = 0.131401834143860282009280387409357165515556574352422001206362e-16;
    private static final double EI_ROOT_R = 0.372507410781366634461991866580119133535689497771654051555657435242200120636201854384926049951548942392;

    private static final double E1_Y = 0.66373538970947265625;
    private static final double[] E1_SMALL_P = {
        0.0865197248079397976498,
        0.0320913665303559189999,
        -0.245088216639761496153,
        -0.0368031736257943745142,
        -0.00399167106081113256961,
        -0.000111507792921197858394
    };
    private static final double[] E1_SMALL_Q = {
        1.0,
        0.37091387659397013215,
        0.056770677104207528384,
        0.00427347600017103698101,
        0.000131049900798434683324,
        -0.528611029520217142048e-6
    };
    private static final double[] E1_LARGE_P = {
        -0.121013190657725568138e-18,
        -0.999999999999998811143,
        -43.3058660811817946037,
        -724.581482791462469795,
        -6046.8250112711035463,
        -27182.6254466733970467,
        -66598.2652345418633509,
        -86273.1567711649528784,
        -54844.4587226402067411,
        -14751.4895786128450662,
        -1185.45720315201027667
    };
    private static final double[] E1_LARGE_Q = {
        1.0,
        45.3058660811801465927,
        809.193214954550328455,
        7417.37624454689546708,
        38129.5594484818471461,
        113057.05869159631492,
        192104.047790227984431,
        180329.498380501819718,
        86722.3403467334749201,
        18455.4124737722049515,
        1229.20784182403048905,
        -0.776491285282330997549
    };

    private static final double[] EI_FIRST_P = {
        2.98677224343598593013,
        0.356343618769377415068,
        0.780836076283730801839,
        0.114670926327032002811,
        0.0499434773576515260534,
        0.00726224593341228159561,
        0.00115478237227804306827,
        0.000116419523609765200999,
        0.798296365679269702435e-5,
        0.2777056254402008721e-6
    };
    private static final double[] EI_FIRST_Q = {
        1.0,
        -1.17090412365413911947,
        0.62215109846016746276,
        -0.195114782069495403315,
        0.0391523431392967238166,
        -0.00504800158663705747345,
        0.000389034007436065401822,
        -0.138972589601781706598e-4
    };
    private static final double EI_SECOND_Y = 1.158985137939453125;
    private static final double[] EI_SECOND_P = {
        0.00139324086199402804173,
        -0.0349921221823888744966,
        -0.0264095520754134848538,
        -0.00761224003005476438412,
        -0.00247496209592143627977,
        -0.000374885917942100256775,
        -0.554086272024881826253e-4,
        -0.396487648924804510056e-5
    };
    private static final double[] EI_SECOND_Q = {
        1.0,
        0.744625566823272107711,
        0.329061095011767059236,
        0.100128624977313872323,
        0.0223851099128506347278,
        0.00365334190742316650106,
        0.000402453408512476836472,
        0.263649630720255691787e-4
    };
    private static final double EI_THIRD_Y = 1.0869731903076171875;
    private static final double[] EI_THIRD_P = {
        -0.00893891094356945667451,
        -0.0484607730127134045806,
        -0.0652810444222236895772,
        -0.0478447572647309671455,
        -0.0226059218923777094596,
        -0.00720603636917482065907,
        -0.00155941947035972031334,
        -0.000209750022660200888349,
        -0.138652200349182596186e-4
    };
    private static final double[] EI_THIRD_Q = {
        1.0,
        1.97017214039061194971,
        1.86232465043073157508,
        1.09601437090337519977,
        0.438873285773088870812,
        0.122537731979686102756,
        0.0233458478275769288159,
        0.00278170769163303669021,
        0.000159150281166108755531
    };
    private static final double EI_FOURTH_Y = 1.03937530517578125;
    private static final double[] EI_FOURTH_P = {
        -0.00356165148914447597995,
        -0.0229930320357982333406,
        -0.0449814350482277917716,
        -0.0453759383048193402336,
        -0.0272050837209380717069,
        -0.00994403059883350813295,
        -0.00207592267812291726961,
        -0.000192178045857733706044,
        -0.113161784705911400295e-9
    };
    private static final double[] EI_FOURTH_Q = {
        1.0,
        2.84354408840148561131,
        3.6599610090072393012,
        2.75088464344293083595,
        1.2985244073998398643,
        0.383213198510794507409,
        0.0651165455496281337831,
        0.00488071077519227853585,
        0.0
    };
    private static final double EI_FIFTH_Y = 1.013065338134765625;
    private static final double[] EI_FIFTH_P = {
        -0.0130653381347656243849,
        0.19029710559486576682,
        94.7365094537197236011,
        -2516.35323679844256203,
        18932.0850014925993025,
        -38703.1431362056714134
    };
    private static final double[] EI_FIFTH_Q = {
        1.0,
        61.9733592849439884145,
        -2354.56211323420194283,
        22329.1459489893079041,
        -70126.245140396567133,
        54738.2833147775537106,
        8297.16296356518409347
    };

    private Expint() {
    }

    /**
     * Returns the real exponential integral {@code Ei(z)}.
     *
     * @param z real argument
     * @return {@code Ei(z)}
     */
    public static double expint(double z) {
        if (Double.isNaN(z)) {
            throw new IllegalArgumentException("expint(z) requires a real argument: " + z);
        }
        if (z < 0.0) {
            return -expint(1, -z);
        }
        if (z == 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (z == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }
        return expintEiPositive(z);
    }

    /**
     * Returns the real generalized exponential integral {@code E_n(z)}.
     *
     * @param n non-negative order
     * @param z real argument with {@code z >= 0}
     * @return {@code E_n(z)}
     */
    public static double expint(int n, double z) {
        validateIndexedArguments(n, z);
        if (z == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        if (z == 0.0) {
            if (n <= 1) {
                throw new ArithmeticException("expint(n, z) overflows at z = 0.0 for n <= 1: n = " + n);
            }
            return 1.0 / (n - 1.0);
        }
        if (n == 0) {
            return Math.exp(-z) / z;
        }
        if (n == 1) {
            return expintE1(z);
        }

        double seriesThreshold = n < 3 ? 0.5 : (double) (n - 2) / (n - 1);
        return z < seriesThreshold ? expintEnSeries(n, z) : expintEnFraction(n, z);
    }

    private static void validateIndexedArguments(int n, double z) {
        if (n < 0) {
            throw new IllegalArgumentException("expint(n, z) requires n >= 0, but got n = " + n);
        }
        if (Double.isNaN(z) || z < 0.0 || z == Double.NEGATIVE_INFINITY) {
            throw new IllegalArgumentException("expint(n, z) requires z >= 0, but got z = " + z);
        }
    }

    private static double expintE1(double z) {
        if (z <= 1.0) {
            return evaluatePolynomial(E1_SMALL_P, z) / evaluatePolynomial(E1_SMALL_Q, z)
                + z - Math.log(z) - E1_Y;
        }
        if (z >= E1_UNDERFLOW_THRESHOLD) {
            return 0.0;
        }
        double reciprocal = 1.0 / z;
        double result = 1.0 + evaluatePolynomial(E1_LARGE_P, reciprocal) / evaluatePolynomial(E1_LARGE_Q, reciprocal);
        return result * Math.exp(-z) * reciprocal;
    }

    private static double expintEiPositive(double z) {
        if (z <= 6.0) {
            double t = z / 3.0 - 1.0;
            double result = evaluatePolynomial(EI_FIRST_P, t) / evaluatePolynomial(EI_FIRST_Q, t);
            double delta = (z - EI_ROOT_R1) - EI_ROOT_R2;
            result *= delta;
            if (Math.abs(delta) < 0.1) {
                return result + Math.log1p(delta / EI_ROOT_R);
            }
            return result + Math.log(z / EI_ROOT_R);
        }
        if (z <= 10.0) {
            double t = z / 2.0 - 4.0;
            double result = EI_SECOND_Y + evaluatePolynomial(EI_SECOND_P, t) / evaluatePolynomial(EI_SECOND_Q, t);
            return result * Math.exp(z) / z + z;
        }
        if (z <= 20.0) {
            double t = z / 5.0 - 3.0;
            double result = EI_THIRD_Y + evaluatePolynomial(EI_THIRD_P, t) / evaluatePolynomial(EI_THIRD_Q, t);
            return result * Math.exp(z) / z + z;
        }
        if (z <= 40.0) {
            double t = z / 10.0 - 3.0;
            double result = EI_FOURTH_Y + evaluatePolynomial(EI_FOURTH_P, t) / evaluatePolynomial(EI_FOURTH_Q, t);
            return result * Math.exp(z) / z + z;
        }

        double reciprocal = 1.0 / z;
        double result = EI_FIFTH_Y + evaluatePolynomial(EI_FIFTH_P, reciprocal) / evaluatePolynomial(EI_FIFTH_Q, reciprocal);
        if (z < 41.0) {
            return result * Math.exp(z) / z + z;
        }

        double shifted = z - 40.0;
        if (shifted > LOG_MAX_VALUE) {
            return Double.POSITIVE_INFINITY;
        }
        result *= Math.exp(shifted) / z;
        if (result > Double.MAX_VALUE / EXP_40) {
            return Double.POSITIVE_INFINITY;
        }
        return result * EXP_40 + z;
    }

    private static double expintEnSeries(int n, double z) {
        double result = 0.0;
        double xk = -1.0;
        double denominator = 1.0 - n;
        double factorial = 1.0;
        int k = 0;

        while (k < n - 1) {
            result += xk / (denominator * factorial);
            denominator += 1.0;
            xk *= -z;
            factorial *= ++k;
        }

        result += Math.pow(-z, n - 1) * (Gamma.digamma(n) - Math.log(z)) / factorial;

        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            xk *= -z;
            denominator += 1.0;
            factorial *= ++k;
            double term = xk / (denominator * factorial);
            double next = result + term;
            if (next == result || Math.abs(term) <= EPSILON * Math.max(1.0, Math.abs(next))) {
                return next;
            }
            result = next;
        }

        throw new IllegalStateException("expint series did not converge for n=" + n + ", z=" + z);
    }

    private static double expintEnFraction(int n, double z) {
        ContinuedFraction fraction = ContinuedFraction.evaluateFractionB(index -> {
            if (index == 0) {
                return Double2.fraction(0.0, n + z);
            }
            double i = index;
            return Double2.fraction(-i * (n + i - 1.0), n + z + 2.0 * i);
        }, EPSILON, MAX_ITERATIONS);

        if (!fraction.converged()) {
            throw new IllegalStateException("expint continued fraction did not converge for n=" + n + ", z=" + z);
        }
        return Math.exp(-z) / fraction.value();
    }
}