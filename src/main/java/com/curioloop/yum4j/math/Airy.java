package com.curioloop.yum4j.math;

import java.util.function.IntToDoubleFunction;

import static com.curioloop.yum4j.math.Commons.validateFinite;

/**
 * Real scalar Airy functions aligned with Boost.Math 1.90.
 *
 * <p>The public real surface mirrors Boost's retained Airy family:
 *
 * <pre>
 * Ai(x), Bi(x), Ai'(x), Bi'(x)
 * Ai zero m, Bi zero m
 * </pre>
 *
 * <p>Implementation notes:
 *
 * <ul>
 *   <li>The value and derivative formulas follow Boost's real Bessel reductions on the
 *       negative and positive half-lines.</li>
 *   <li>Near the origin, the exact gamma-constant limits are returned directly using the same
 *       small-argument tests as Boost's real implementation.</li>
 *   <li>Zero locators stay 1-based, matching Boost's public {@code airy_ai_zero} and
 *       {@code airy_bi_zero} rank semantics.</li>
 * </ul>
 */
public final class Airy {

    private static final double THIRD = 1.0 / 3.0;
    private static final double TWO_THIRDS = 2.0 / 3.0;
    private static final double ROOT_THREE = Math.sqrt(3.0);
    private static final double CBRT_THREE = Math.cbrt(3.0);
    private static final double SQRT_CBRT_THREE = Math.sqrt(CBRT_THREE);
    private static final double PI = Math.PI;
    private static final double MACHINE_EPSILON = Math.ulp(1.0);
    private static final double VALUE_NEAR_ZERO_THRESHOLD = Math.cbrt(6.0 * MACHINE_EPSILON);
    private static final double DERIVATIVE_NEAR_ZERO_THRESHOLD = Math.sqrt(2.0 * MACHINE_EPSILON);
    private static final int ROOT_BITS = 53;
    private static final int ROOT_MAX_ITERATIONS = 128;

    private static final double AI_AT_ZERO = 1.0 / (Math.pow(3.0, TWO_THIRDS) * Gamma.tgamma(TWO_THIRDS));
    private static final double BI_AT_ZERO = 1.0 / (SQRT_CBRT_THREE * Gamma.tgamma(TWO_THIRDS));
    private static final double AI_PRIME_AT_ZERO = -1.0 / (CBRT_THREE * Gamma.tgamma(THIRD));
    private static final double BI_PRIME_AT_ZERO = SQRT_CBRT_THREE / Gamma.tgamma(THIRD);

    private Airy() {
    }

    /**
     * Returns the Airy Ai function.
     */
    public static double ai(double x) {
        validateFinite(x, "ai: x");
        if (Math.abs(x) < VALUE_NEAR_ZERO_THRESHOLD) {
            return AI_AT_ZERO;
        }
        if (x < 0.0) {
            double p = airyPhaseNegative(x);
            double jPositive = Bessel.j(THIRD, p);
            double jNegative = Bessel.j(-THIRD, p);
            return Math.sqrt(-x) * (jPositive + jNegative) / 3.0;
        }
        double p = airyPhasePositive(x);
        return Bessel.k(THIRD, p) * Math.sqrt(x / 3.0) / PI;
    }

    /**
     * Returns the Airy Bi function.
     */
    public static double bi(double x) {
        validateFinite(x, "bi: x");
        if (Math.abs(x) < VALUE_NEAR_ZERO_THRESHOLD) {
            return BI_AT_ZERO;
        }
        if (x < 0.0) {
            double p = airyPhaseNegative(x);
            double jPositive = Bessel.j(THIRD, p);
            double jNegative = Bessel.j(-THIRD, p);
            return Math.sqrt(-x / 3.0) * (jNegative - jPositive);
        }
        double p = airyPhasePositive(x);
        return Math.sqrt(x / 3.0) * (Bessel.i(-THIRD, p) + Bessel.i(THIRD, p));
    }

    /**
     * Returns the derivative of the Airy Ai function.
     */
    public static double aiPrime(double x) {
        validateFinite(x, "aiPrime: x");
        if (Math.abs(x) < DERIVATIVE_NEAR_ZERO_THRESHOLD) {
            return AI_PRIME_AT_ZERO;
        }
        if (x < 0.0) {
            double p = airyPhaseNegative(x);
            double jPositive = Bessel.j(TWO_THIRDS, p);
            double jNegative = Bessel.j(-TWO_THIRDS, p);
            return -x * (jPositive - jNegative) / 3.0;
        }
        double p = airyPhasePositive(x);
        return -Bessel.k(TWO_THIRDS, p) * x / (ROOT_THREE * PI);
    }

    /**
     * Returns the derivative of the Airy Bi function.
     */
    public static double biPrime(double x) {
        validateFinite(x, "biPrime: x");
        if (Math.abs(x) < DERIVATIVE_NEAR_ZERO_THRESHOLD) {
            return BI_PRIME_AT_ZERO;
        }
        if (x < 0.0) {
            double p = airyPhaseNegative(x);
            double jPositive = Bessel.j(TWO_THIRDS, p);
            double jNegative = Bessel.j(-TWO_THIRDS, p);
            return -x * (jPositive + jNegative) / ROOT_THREE;
        }
        double p = airyPhasePositive(x);
        return x * (Bessel.i(-TWO_THIRDS, p) + Bessel.i(TWO_THIRDS, p)) / ROOT_THREE;
    }

    /**
     * Returns the rank-th negative real zero of Ai(x), with 1-based rank semantics.
     */
    public static double aiZero(int rank) {
        validateRank(rank, "aiZero");
        return solveAiryZero(rank, true);
    }

    /**
     * Returns the rank-th negative real zero of Bi(x), with 1-based rank semantics.
     */
    public static double biZero(int rank) {
        validateRank(rank, "biZero");
        return solveAiryZero(rank, false);
    }

    /**
     * Returns a rank-to-zero function for Ai(x), preserving the 1-based scalar rank contract.
     */
    public static IntToDoubleFunction aiZeros() {
        return Airy::aiZero;
    }

    /**
     * Returns a rank-to-zero function for Bi(x), preserving the 1-based scalar rank contract.
     */
    public static IntToDoubleFunction biZeros() {
        return Airy::biZero;
    }

    private static double solveAiryZero(int rank, boolean aiFamily) {
        double guess = aiFamily ? initialAiZero(rank) : initialBiZero(rank);
        double tolerance = zeroBracketTolerance(rank);
        return RootIterators.newtonRaphsonIterate(
            (x, evaluation) -> {
                if (aiFamily) {
                    evaluation[0] = ai(x);
                    evaluation[1] = aiPrime(x);
                } else {
                    evaluation[0] = bi(x);
                    evaluation[1] = biPrime(x);
                }
            },
            guess,
            guess - tolerance,
            guess + tolerance,
            ROOT_BITS,
            ROOT_MAX_ITERATIONS,
            aiFamily ? "aiZero" : "biZero"
        );
    }

    private static void validateRank(int rank, String functionName) {
        if (rank <= 0) {
            throw new IllegalArgumentException(functionName + ": rank must be >= 1: " + rank);
        }
    }

    private static double airyPhaseNegative(double x) {
        return (-x * Math.sqrt(-x) * 2.0) / 3.0;
    }

    private static double airyPhasePositive(double x) {
        return (2.0 * x * Math.sqrt(x)) / 3.0;
    }

    private static double zeroBracketTolerance(int rank) {
        if (rank <= 10) {
            return 0.3;
        }
        if (rank <= 100) {
            return 0.1;
        }
        if (rank <= 1000) {
            return 0.05;
        }
        return 1.0 / Math.sqrt(rank);
    }

    static double initialAiZero(int rank) {
        return switch (rank) {
            case 0 -> 0.0;
            case 1 -> -2.33810741045976703849;
            case 2 -> -4.08794944413097061664;
            case 3 -> -5.52055982809555105913;
            case 4 -> -6.78670809007175899878;
            case 5 -> -7.94413358712085312314;
            case 6 -> -9.02265085334098038016;
            case 7 -> -10.0401743415580859306;
            case 8 -> -11.0085243037332628932;
            case 9 -> -11.9360155632362625170;
            case 10 -> -12.8287767528657572004;
            default -> -airyZeroAsymptotic(((3.0 * PI) * ((4.0 * rank) - 1.0)) / 8.0);
        };
    }

    static double initialBiZero(int rank) {
        return switch (rank) {
            case 0 -> 0.0;
            case 1 -> -1.17371322270912792492;
            case 2 -> -3.27109330283635271568;
            case 3 -> -4.83073784166201593267;
            case 4 -> -6.16985212831025125983;
            case 5 -> -7.37676207936776371360;
            case 6 -> -8.49194884650938801345;
            case 7 -> -9.53819437934623888663;
            case 8 -> -10.5299135067053579244;
            case 9 -> -11.4769535512787794379;
            case 10 -> -12.3864171385827387456;
            default -> -airyZeroAsymptotic(((3.0 * PI) * ((4.0 * rank) - 3.0)) / 8.0);
        };
    }

    private static double airyZeroAsymptotic(double z) {
        double inverse = 1.0 / z;
        double inverseSquare = inverse * inverse;
        double zPowThird = Math.cbrt(z);
        double zPowTwoThirds = zPowThird * zPowThird;
        return zPowTwoThirds * ((((((162375596875.0 / 334430208.0) * inverseSquare - (108056875.0 / 6967296.0))
            * inverseSquare + (77125.0 / 82944.0)) * inverseSquare - (5.0 / 36.0))
            * inverseSquare + (5.0 / 48.0)) * inverseSquare + 1.0);
    }
}