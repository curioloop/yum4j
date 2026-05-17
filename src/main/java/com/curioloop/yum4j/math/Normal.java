package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.HornerPoly.evaluatePolynomial;

/**
 * Boost.Math special-function utility surface for the standard normal distribution.
 *
 * <p>This class provides direct Java ports of the Boost.Math double-precision
 * implementations in {@code special_functions/erf.hpp} and
 * {@code special_functions/detail/erf_inv.hpp}. The public surface covers:
 *
 * <ul>
 *   <li>Probability density function (PDF)</li>
 *   <li>Log-PDF</li>
 *   <li>Cumulative distribution function (CDF)</li>
 *   <li>Complementary cumulative distribution function (CCDF)</li>
 *   <li>Error function and complementary error function</li>
 *   <li>Inverse error function and inverse complementary error function</li>
 *   <li>Inverse CDF (quantile function)</li>
 *   <li>Inverse CCDF (upper-tail quantile function)</li>
 * </ul>
 *
 * <p>The standard normal distribution is defined as:
 *
 * <pre>
 * Z ~ N(0,1)
 *
 * PDF:
 *     φ(x) = exp(-x² / 2) / √(2π)
 *
 * CDF:
 *     Φ(x) = P(Z ≤ x) = ∫_{-∞}^{x} φ(t) dt
 *
 * CCDF:
 *     Q(x) = P(Z > x) = 1 - Φ(x)
 *
 * Quantile:
 *     Φ⁻¹(p)
 *
 * Upper-tail quantile:
 *     Q⁻¹(p)
 * </pre>
 *
 * <p>Implementation notes:
 *
 * <ul>
 *   <li>The PDF uses the direct analytical expression.</li>
 *   <li>{@link #logPdf(double)} uses the same rounded normalizing constant as
 *       {@link #pdf(double)} so that the center-point value matches the Boost
 *       fixture exactly.</li>
 *   <li>{@link #cdf(double)} now mirrors Boost.Math's
 *       {@code normal_distribution<double>::cdf} surface:
 *       {@code Phi(x) = erfc(-x / sqrt(2)) / 2}.</li>
 *   <li>{@link #inv(double)} now mirrors Boost.Math's
 *       {@code normal_distribution<double>::quantile} surface:
 *       {@code Phi^{-1}(p) = -sqrt(2) * erfc_inv(2p)}.</li>
 *   <li>{@link #erf(double)} and {@link #erfc(double)} are both evaluated by a
 *       single Boost-style piecewise backend, including the same reflection and
 *       scaled-exponential branches used by the 64-bit implementation.</li>
 *   <li>{@link #erfInv(double)} and {@link #erfcInv(double)} use the same
 *       Boost 64-bit rational approximations for the initial guess and then
 *       apply a short Halley correction so Java double evaluation matches the
 *       Boost fixture generated through the default policy path.</li>
 *   <li>No memory allocation occurs during evaluation.</li>
 * </ul>
 *
 * <p>Typical numerical accuracy:
 *
 * <ul>
 *   <li>PDF:      ~1e-16 (machine precision)</li>
 *   <li>CDF:      machine precision for the Boost normal surface, including far tails</li>
 *   <li>Inverse:  near machine precision for double, following Boost's 64-bit backend</li>
 * </ul>
 *
 * <p>References:
 *
 * <ol>
 *   <li>Boost.Math {@code special_functions/erf.hpp} — source of the 64-bit
 *       {@code erf}/{@code erfc} piecewise approximations and reflection rules.</li>
 *   <li>Boost.Math {@code special_functions/detail/erf_inv.hpp} — source of the
 *       inverse-erf and inverse-erfc rational approximations.</li>
 * </ol>
 *
 * <p>This implementation is suitable for high-performance numerical
 * applications such as Monte-Carlo simulation, option pricing,
 * risk analytics, and numerical optimization.
 */
public final class Normal {

    private Normal() {}

    /** 1 / sqrt(2π) */
    private static final double INV_SQRT_2PI = 0.39894228040143267794;
    private static final double LOG_INV_SQRT_2PI = Math.log(INV_SQRT_2PI);

    /** 1 / sqrt(2) */
    private static final double INV_SQRT2 = 0.70710678118654752440;

    /** log(2·sqrt(π)) */
    private static final double LOG_2_SQRT_PI = Math.log(2.0 * Math.sqrt(Math.PI));

    /** sqrt(2) */
    private static final double SQRT2 = 1.41421356237309504880;
    private static final double TWO_DIV_SQRT_PI = 1.128379167095512573896158903121545171688;
    private static final double SMALL_ERF_COEFFICIENT = 0.003379167095512573896158903121545171688;
    private static final int INVERSE_REFINEMENT_STEPS = 2;

    private static final double ERF_SMALL_Y = 1.044948577880859375;
    private static final double[] ERF_SMALL_P = {
        0.0834305892146531988966,
        -0.338097283075565413695,
        -0.0509602734406067204596,
        -0.00904906346158537794396,
        -0.000489468651464798669181,
        -0.0000200305626366151877759
    };
    private static final double[] ERF_SMALL_Q = {
        1.0,
        0.455817300515875172439,
        0.0916537354356241792007,
        0.0102722652675910031202,
        0.000650511752687851548735,
        0.0000189532519105655496778
    };

    private static final double ERFC_0P5_TO_1P5_Y = 0.405935764312744140625;
    private static final double[] ERFC_0P5_TO_1P5_P = {
        -0.0980905922162812031672,
        0.159989089922969141329,
        0.222359821619935712378,
        0.127303921703577362312,
        0.0384057530342762400273,
        0.00628431160851156719325,
        0.000441266654514391746428,
        0.0000000266689068336295642561
    };
    private static final double[] ERFC_0P5_TO_1P5_Q = {
        1.0,
        2.03237474985469469291,
        1.78355454954969405222,
        0.867940326293760578231,
        0.248025606990021698392,
        0.0396649631833002269861,
        0.00279220237309449026796
    };

    private static final double ERFC_1P5_TO_2P5_Y = 0.50672817230224609375;
    private static final double[] ERFC_1P5_TO_2P5_P = {
        -0.024350047620769840217,
        0.0343522687935671451309,
        0.0505420824305544949541,
        0.0257479325917757388209,
        0.00669349844190354356118,
        0.00090807914416099524444,
        0.0000515917266698050027934
    };
    private static final double[] ERFC_1P5_TO_2P5_Q = {
        1.0,
        1.71657861671930336344,
        1.26409634824280366218,
        0.512371437838969015941,
        0.120902623051120950935,
        0.0158027197831887485261,
        0.000897871370778031611439
    };

    private static final double ERFC_2P5_TO_4P5_Y = 0.5405750274658203125;
    private static final double[] ERFC_2P5_TO_4P5_P = {
        0.0029527671653097284033,
        0.0141853245895495604051,
        0.0104959584626432293901,
        0.00343963795976100077626,
        0.00059065441194877637899,
        0.0000523435380636174008685,
        0.00000189896043050331257262
    };
    private static final double[] ERFC_2P5_TO_4P5_Q = {
        1.0,
        1.19352160185285642574,
        0.603256964363454392857,
        0.165411142458540585835,
        0.0259729870946203166468,
        0.00221657568292893699158,
        0.0000804149464190309799804
    };

    private static final double ERFC_4P5_TO_6P6_Y = 0.55825519561767578125;
    private static final double[] ERFC_4P5_TO_6P6_P = {
        0.00593438793008050214106,
        0.0280666231009089713937,
        -0.141597835204583050043,
        -0.978088201154300548842,
        -5.47351527796012049443,
        -13.8677304660245326627,
        -27.1274948720539821722,
        -29.2545152747009461519,
        -16.8865774499799676937
    };
    private static final double[] ERFC_4P5_TO_6P6_Q = {
        1.0,
        4.72948911186645394541,
        23.6750543147695749212,
        60.0021517335693186785,
        131.766251645149522868,
        178.167924971283482513,
        182.499390505915222699,
        104.365251479578577989,
        30.8365511891224291717
    };

    // ── Boost.Math inverse-erfc coefficients for 64-bit doubles ─────────────

    private static final double ERF_INV_P_LE_HALF_Y = 0.08913147449493408;
    private static final double[] ERF_INV_P_LE_HALF_P = {
        -0.000508781949658280665617,
        -0.00836874819741736770379,
        0.0334806625409744615033,
        -0.0126926147662974029034,
        -0.0365637971411762664006,
        0.0219878681111168899165,
        0.00822687874676915743155,
        -0.00538772965071242932965
    };
    private static final double[] ERF_INV_P_LE_HALF_Q = {
        1.0,
        -0.970005043303290640362,
        -1.56574558234175846809,
        1.56221558398423026363,
        0.662328840472002992063,
        -0.71228902341542847553,
        -0.0527396382340099713954,
        0.0795283687341571680018,
        -0.00233393759374190016776,
        0.000886216390456424707504
    };

    private static final double ERF_INV_Q_GE_QUARTER_Y = 2.249481201171875;
    private static final double[] ERF_INV_Q_GE_QUARTER_P = {
        -0.202433508355938759655,
        0.105264680699391713268,
        8.37050328343119927838,
        17.6447298408374015486,
        -18.8510648058714251895,
        -44.6382324441786960818,
        17.445385985570866523,
        21.1294655448340526258,
        -3.67192254707729348546
    };
    private static final double[] ERF_INV_Q_GE_QUARTER_Q = {
        1.0,
        6.24264124854247537712,
        3.9713437953343869095,
        -28.6608180499800029974,
        -20.1432634680485188801,
        48.5609213108739935468,
        10.8268667355460159008,
        -22.6436933413139721736,
        1.72114765761200282724
    };

    private static final double ERF_INV_X_LT_3_Y = 0.807220458984375;
    private static final double[] ERF_INV_X_LT_3_P = {
        -0.131102781679951906451,
        -0.163794047193317060787,
        0.117030156341995252019,
        0.387079738972604337464,
        0.337785538912035898924,
        0.142869534408157156766,
        0.0290157910005329060432,
        0.00214558995388805277169,
        -0.000000679465575181126350155,
        0.0000000285225331782217055858,
        -0.000000000681149956853776992068
    };
    private static final double[] ERF_INV_X_LT_3_Q = {
        1.0,
        3.46625407242567245975,
        5.38168345707006855425,
        4.77846592945843778382,
        2.59301921623620271374,
        0.848854343457902036425,
        0.152264338295331783612,
        0.01105924229346489121
    };

    private static final double ERF_INV_X_LT_6_Y = 0.9399557113647461;
    private static final double[] ERF_INV_X_LT_6_P = {
        -0.0350353787183177984712,
        -0.00222426529213447927281,
        0.0185573306514231072324,
        0.00950804701325919603619,
        0.00187123492819559223345,
        0.000157544617424960554631,
        0.00000460469890584317994083,
        -0.000000000230404776911882601748,
        0.00000000000266339227425782031962
    };
    private static final double[] ERF_INV_X_LT_6_Q = {
        1.0,
        1.3653349817554063097,
        0.762059164553623404043,
        0.220091105764131249824,
        0.0341589143670947727934,
        0.00263861676657015992959,
        0.0000764675292302794483503
    };

    private static final double ERF_INV_X_LT_18_Y = 0.9836282730102539;
    private static final double[] ERF_INV_X_LT_18_P = {
        -0.0167431005076633737133,
        -0.00112951438745580278863,
        0.00105628862152492910091,
        0.000209386317487588078668,
        0.0000149624783758342370182,
        0.000000449696789927706453732,
        0.00000000462596163522878599135,
        -0.000000000000281128735628831791805,
        0.000000000000000099055709973310326855
    };
    private static final double[] ERF_INV_X_LT_18_Q = {
        1.0,
        0.591429344886417493481,
        0.138151865749083321638,
        0.0160746087093676504695,
        0.000964011807005165528527,
        0.0000275335474764726041141,
        0.000000282243172016108031869
    };

    private static final double ERF_INV_X_LT_44_Y = 0.9971456527709961;
    private static final double[] ERF_INV_X_LT_44_P = {
        -0.0024978212791898131227,
        -0.00000779190719229053954292,
        0.0000254723037413027451751,
        0.00000162397777342510920873,
        0.0000000396341011304801168516,
        0.000000000411632831190944208473,
        0.00000000000145596286718675035587,
        -0.0000000000000000116765012397184275695
    };
    private static final double[] ERF_INV_X_LT_44_Q = {
        1.0,
        0.207123112214422517181,
        0.0169410838120975906478,
        0.000690538265622684595676,
        0.0000145007359818232637924,
        0.000000144437756628144157666,
        0.000000000509761276599778486139
    };

    private static final double ERF_INV_X_GE_44_Y = 0.9994134902954102;
    private static final double[] ERF_INV_X_GE_44_P = {
        -0.000539042911019078575891,
        -0.00000028398759004727721098,
        0.000000899465114892291446442,
        0.0000000229345859265920864296,
        0.000000000225561444863500149219,
        0.000000000000947846627503022684216,
        0.00000000000000135880130108924861008,
        -0.000000000000000000348890393399948882918
    };
    private static final double[] ERF_INV_X_GE_44_Q = {
        1.0,
        0.0845746234001899436914,
        0.00282092984726264681981,
        0.0000468292921940894236786,
        0.000000399968812193862100054,
        0.00000000161809290887904476097,
        0.00000000000231558608310259605225
    };

    /**
     * Standard normal probability density function.
     *
     * <pre>
     * φ(x) = exp(-x² / 2) / √(2π)
     * </pre>
     *
     * @param x evaluation point
     * @return φ(x)
     */
    public static double pdf(double x) {
        return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
    }

    /**
     * Logarithm of the standard normal PDF.
     *
     * <pre>
     * log φ(x) = -x²/2 − log(√(2π))
     * </pre>
     *
     * This form avoids numerical underflow for large |x|.
     *
     * @param x evaluation point
     * @return log φ(x)
     */
    public static double logPdf(double x) {
        return -0.5 * x * x + LOG_INV_SQRT_2PI;
    }

    /**
     * Standard normal cumulative distribution function.
     *
     * <pre>
     * Φ(x) = erfc(-x / √2) / 2
     * </pre>
     *
     * <p>This method is the Boost-parity surface used by local code that wants
     * the same backend shape as {@code boost::math::cdf(normal_distribution, x)}.
     * It intentionally does not route through QuantLib's error-function-based
     * cumulative normal implementation.
     *
     * @param x evaluation point
     * @return Φ(x)
     */
    public static double cdf(double x) {
        return 0.5 * erfc(-x * INV_SQRT2);
    }

    /**
     * Standard normal complementary cumulative distribution function.
     *
     * <pre>
     * Q(x) = P(Z > x) = erfc(x / √2) / 2
     * </pre>
     *
     * <p>This method is the Boost-parity surface used by local code that wants
     * the same backend shape as
     * {@code boost::math::cdf(complement(normal_distribution, x))}.
     *
     * @param x evaluation point
     * @return Q(x)
     */
    public static double ccdf(double x) {
        return 0.5 * erfc(x * INV_SQRT2);
    }

    /**
     * Logarithm of the standard normal CDF.
     *
     * <pre>
     * log Φ(x)
     * </pre>
     *
     * <p>Numerically stable in the far left tail where {@link #cdf(double)}
     * underflows to zero: prefers {@code log(cdf(x))} when that is finite
     * (the Boost {@code erfc} implementation stays non-zero down to
     * roughly {@code x ≈ -37}), and switches to an asymptotic expansion
     * for deeper inputs.
     *
     * @param x evaluation point
     * @return log Φ(x)
     */
    public static double logCdf(double x) {
        double c = cdf(x);
        if (c > 0.0) {
            return Math.log(c);
        }
        // cdf(x) underflowed to zero. Use the asymptotic expansion of erfc
        // for large positive argument t = -x/√2:
        //   erfc(t) ≈ exp(-t²) / (t·√π) · [1 − 1/(2t²) + 3/(4t⁴) − ...]
        // Pull the exp(-x²/2) factor into logspace; truncate the alternating
        // asymptotic series when the next term stops shrinking in magnitude.
        double t = -x * INV_SQRT2;
        double t2 = t * t;
        double leading = -0.5 * x * x - Math.log(t) - LOG_2_SQRT_PI;
        double term = 1.0;
        double sum = 1.0;
        double prev = Double.POSITIVE_INFINITY;
        for (int k = 1; k < 32; k++) {
            term *= -(2 * k - 1) / (2.0 * t2);
            double abs = Math.abs(term);
            if (abs >= prev) break;
            sum += term;
            prev = abs;
            if (abs < 1e-18) break;
        }
        return leading + Math.log(sum);
    }

    /**
     * Error function.
     *
     * @param x evaluation point
     * @return erf(x)
     */
    public static double erf(double x) {
        return erfImp(x, false);
    }

    /**
     * Inverse of the standard normal CDF (quantile function).
     *
     * <pre>
     * Φ⁻¹(p)
     * </pre>
     *
     * <p>This method is the Boost-parity surface used by local code that wants
     * the same backend shape as {@code boost::math::quantile(normal_distribution, p)}.
     * The implementation is a direct Java port of Boost.Math's 64-bit
     * {@code erfc_inv} rational approximation tables.
     *
     * @param p probability in {@code [0, 1]}
     * @return z such that Φ(z) = p
     * @throws IllegalArgumentException if {@code p} is outside {@code [0, 1]} or NaN
     * @throws ArithmeticException if {@code p} is exactly {@code 0} or {@code 1}
     */
    public static double inv(double p) {
        return -SQRT2 * erfcInv(2.0 * p);
    }

    /**
     * Inverse of the standard normal complementary CDF.
     *
     * <pre>
     * Q⁻¹(p), where Q(x) = P(Z > x) = 1 - Φ(x)
     * </pre>
     *
     * <p>This method mirrors the Boost.Math upper-tail quantile surface and
     * keeps the same domain and overflow semantics as {@link #inv(double)}.
     *
     * @param p upper-tail probability in {@code [0, 1]}
     * @return z such that Q(z) = p
     * @throws IllegalArgumentException if {@code p} is outside {@code [0, 1]} or NaN
     * @throws ArithmeticException if {@code p} is exactly {@code 0} or {@code 1}
     */
    public static double invUpperTail(double p) {
        return SQRT2 * erfcInv(2.0 * p);
    }

    /**
     * Complementary error function.
     *
     * <pre>
     * erfc(x) = 1 - erf(x)
     * </pre>
     *
     * <p>This is the public Boost-style complementary-error-function surface used
     * by {@link #cdf(double)}, {@link #ccdf(double)}, and the inverse-normal APIs.
     *
     * @param x evaluation point
     * @return {@code erfc(x)}
     */
    public static double erfc(double x) {
        return erfImp(x, true);
    }

    /**
     * Inverse error function.
     *
     * <pre>
     * erfInv(z) = x such that erf(x) = z
     * </pre>
     *
     * @param z target value in {@code [-1, 1]}
     * @return {@code x} such that {@code erf(x) = z}
     * @throws IllegalArgumentException if {@code z} is outside {@code [-1, 1]} or NaN
     * @throws ArithmeticException if {@code z} is exactly {@code -1} or {@code 1}
     */
    public static double erfInv(double z) {
        if (Double.isNaN(z) || z < -1.0 || z > 1.0) {
            throw new IllegalArgumentException("Normal.erfInv undefined: argument must be in [-1,1]");
        }
        if (z == 1.0 || z == -1.0) {
            throw new ArithmeticException("Normal.erfInv overflow: argument must satisfy -1 < z < 1");
        }
        if (z == 0.0) {
            return 0.0;
        }

        double p;
        double q;
        double sign;
        if (z < 0.0) {
            p = -z;
            q = 1.0 - p;
            sign = -1.0;
        } else {
            p = z;
            q = 1.0 - z;
            sign = 1.0;
        }

        double guess = sign * erfInvImp(p, q);
        return refineInverseErf(guess, z);
    }

    /**
     * Inverse complementary error function.
     *
     * <pre>
     * erfcInv(z) = x such that erfc(x) = z
     * </pre>
     *
     * @param z target value in {@code [0, 2]}
     * @return {@code x} such that {@code erfc(x) = z}
     * @throws IllegalArgumentException if {@code z} is outside {@code [0, 2]} or NaN
     * @throws ArithmeticException if {@code z} is exactly {@code 0} or {@code 2}
     */
    public static double erfcInv(double z) {
        if (Double.isNaN(z) || z < 0.0 || z > 2.0) {
            throw new IllegalArgumentException("Normal.erfcInv undefined: argument must be in [0,2]");
        }
        if (z == 0.0 || z == 2.0) {
            throw new ArithmeticException("Normal.erfcInv overflow: argument must satisfy 0 < z < 2");
        }

        double p;
        double q;
        double sign;
        if (z > 1.0) {
            q = 2.0 - z;
            p = 1.0 - q;
            sign = -1.0;
        } else {
            p = 1.0 - z;
            q = z;
            sign = 1.0;
        }
        double guess = sign * erfInvImp(p, q);
        return refineInverseErfc(guess, z);
    }

    private static double erfImp(double z, boolean invert) {
        if (Double.isNaN(z)) {
            return Double.NaN;
        }

        if (z < 0.0) {
            if (!invert) {
                return -erfImp(-z, false);
            }
            if (z < -0.5) {
                return 2.0 - erfImp(-z, true);
            }
            return 1.0 + erfImp(-z, false);
        }

        double result;
        if (z < 0.5) {
            if (z == 0.0) {
                result = 0.0;
            } else if (z < 1.0e-10) {
                result = z * 1.125 + z * SMALL_ERF_COEFFICIENT;
            } else {
                double zz = z * z;
                result = z * (
                    ERF_SMALL_Y
                        + evaluatePolynomial(ERF_SMALL_P, zz) / evaluatePolynomial(ERF_SMALL_Q, zz)
                );
            }
        } else if (invert ? (z < 110.0) : (z < 6.6)) {
            invert = !invert;
            if (z < 1.5) {
                result = ERFC_0P5_TO_1P5_Y
                    + evaluatePolynomial(ERFC_0P5_TO_1P5_P, z - 0.5)
                    / evaluatePolynomial(ERFC_0P5_TO_1P5_Q, z - 0.5);
                result = scaledErfcTerm(z, result);
            } else if (z < 2.5) {
                result = ERFC_1P5_TO_2P5_Y
                    + evaluatePolynomial(ERFC_1P5_TO_2P5_P, z - 1.5)
                    / evaluatePolynomial(ERFC_1P5_TO_2P5_Q, z - 1.5);
                result = scaledErfcTerm(z, result);
            } else if (z < 4.5) {
                result = ERFC_2P5_TO_4P5_Y
                    + evaluatePolynomial(ERFC_2P5_TO_4P5_P, z - 3.5)
                    / evaluatePolynomial(ERFC_2P5_TO_4P5_Q, z - 3.5);
                result = scaledErfcTerm(z, result);
            } else {
                double inverse = 1.0 / z;
                result = ERFC_4P5_TO_6P6_Y
                    + evaluatePolynomial(ERFC_4P5_TO_6P6_P, inverse)
                    / evaluatePolynomial(ERFC_4P5_TO_6P6_Q, inverse);
                result = scaledErfcTerm(z, result);
            }
        } else {
            result = 0.0;
            invert = !invert;
        }

        return invert ? 1.0 - result : result;
    }

    private static double scaledErfcTerm(double z, double approximation) {
        int exponent = Math.getExponent(z) + 1;
        double mantissa = Math.scalb(z, -exponent);
        double high = Math.floor(Math.scalb(mantissa, 32));
        high = Math.scalb(high, exponent - 32);
        double low = z - high;
        double square = z * z;
        double errorSquare = ((high * high - square) + 2.0 * high * low) + low * low;
        return approximation * Math.exp(-square) * Math.exp(-errorSquare) / z;
    }

    private static double erfInvImp(double p, double q) {
        if (p <= 0.5) {
            double g = p * (p + 10.0);
            double r = evaluatePolynomial(ERF_INV_P_LE_HALF_P, p) / evaluatePolynomial(ERF_INV_P_LE_HALF_Q, p);
            return g * ERF_INV_P_LE_HALF_Y + g * r;
        }
        if (q >= 0.25) {
            double g = Math.sqrt(-2.0 * Math.log(q));
            double xs = q - 0.25;
            double r = evaluatePolynomial(ERF_INV_Q_GE_QUARTER_P, xs) / evaluatePolynomial(ERF_INV_Q_GE_QUARTER_Q, xs);
            return g / (ERF_INV_Q_GE_QUARTER_Y + r);
        }

        double x = Math.sqrt(-Math.log(q));
        if (x < 3.0) {
            double xs = x - 1.125;
            double r = evaluatePolynomial(ERF_INV_X_LT_3_P, xs) / evaluatePolynomial(ERF_INV_X_LT_3_Q, xs);
            return ERF_INV_X_LT_3_Y * x + r * x;
        }
        if (x < 6.0) {
            double xs = x - 3.0;
            double r = evaluatePolynomial(ERF_INV_X_LT_6_P, xs) / evaluatePolynomial(ERF_INV_X_LT_6_Q, xs);
            return ERF_INV_X_LT_6_Y * x + r * x;
        }
        if (x < 18.0) {
            double xs = x - 6.0;
            double r = evaluatePolynomial(ERF_INV_X_LT_18_P, xs) / evaluatePolynomial(ERF_INV_X_LT_18_Q, xs);
            return ERF_INV_X_LT_18_Y * x + r * x;
        }
        if (x < 44.0) {
            double xs = x - 18.0;
            double r = evaluatePolynomial(ERF_INV_X_LT_44_P, xs) / evaluatePolynomial(ERF_INV_X_LT_44_Q, xs);
            return ERF_INV_X_LT_44_Y * x + r * x;
        }

        double xs = x - 44.0;
        double r = evaluatePolynomial(ERF_INV_X_GE_44_P, xs) / evaluatePolynomial(ERF_INV_X_GE_44_Q, xs);
        return ERF_INV_X_GE_44_Y * x + r * x;
    }

    private static double refineInverseErf(double guess, double target) {
        return refineInverse(guess, target, true);
    }

    private static double refineInverseErfc(double guess, double target) {
        return refineInverse(guess, target, false);
    }

    private static double refineInverse(double guess, double target, boolean solveErf) {
        double x = guess;
        for (int i = 0; i < INVERSE_REFINEMENT_STEPS; i++) {
            double value = solveErf ? erfImp(x, false) : erfImp(x, true);
            double fx = value - target;
            if (fx == 0.0) {
                return x;
            }

            double exponential = Math.exp(-x * x);
            if (exponential == 0.0) {
                return x;
            }

            double derivative = (solveErf ? TWO_DIV_SQRT_PI : -TWO_DIV_SQRT_PI) * exponential;
            double secondDerivative = -2.0 * x * derivative;
            double denominator = 2.0 * derivative * derivative - fx * secondDerivative;

            double step;
            if (Double.isFinite(denominator) && denominator != 0.0) {
                step = 2.0 * fx * derivative / denominator;
            } else {
                step = fx / derivative;
            }

            if (!Double.isFinite(step)) {
                return x;
            }

            x -= step;
            if (!Double.isFinite(x)) {
                return guess;
            }
            if (Math.abs(step) <= Math.ulp(Math.max(1.0, Math.abs(x)))) {
                return x;
            }
        }
        return x;
    }

}