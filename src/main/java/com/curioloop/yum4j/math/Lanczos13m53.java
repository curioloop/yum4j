package com.curioloop.yum4j.math;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector-API port of Boost.Math's {@code lanczos13m53} SSE2 evaluator.
 *
 * <p>This helper mirrors the functionality of {@code lanczos_sse2.hpp} for the
 * double-precision {@code lanczos13m53} tables, including the large-{@code x}
 * reciprocal Horner fallback used by Boost when the packed evaluation loses
 * accuracy.
 */
final class Lanczos13m53 {

    private static final VectorSpecies<Double> PAIR_SPECIES = DoubleVector.SPECIES_128;

    private static final double LANCZOS_G = 6.024680040776729583740234375;
    private static final double LANCZOS_SUM_LIMIT = 4.31965e+25;
    private static final double LANCZOS_SUM_EXP_G_SCALED_LIMIT = 4.76886e+25;

    private static final double[] LANCZOS_SUM_COEFFICIENTS = {
        2.506628274631000270164908177133837338626,
        1.0,
        210.8242777515793458725097339207133627117,
        66.0,
        8071.672002365816210638002902272250613822,
        1925.0,
        186056.2653952234950402949897160456992822,
        32670.0,
        2876370.628935372441225409051620849613599,
        357423.0,
        31426415.58540019438061423162831820536287,
        2637558.0,
        248874557.8620541565114603864132294232163,
        13339535.0,
        1439720407.311721673663223072794912393972,
        45995730.0,
        6039542586.35202800506429164430729792107,
        105258076.0,
        17921034426.03720969991975575445893111267,
        150917976.0,
        35711959237.35566804944018545154716670596,
        120543840.0,
        42919803642.64909876895789904700198885093,
        39916800.0,
        23531376880.41075968857200767445163675473,
        0.0
    };

    private static final double[] LANCZOS_SUM_EXP_G_SCALED_COEFFICIENTS = {
        0.006061842346248906525783753964555936883222,
        1.0,
        0.5098416655656676188125178644804694509993,
        66.0,
        19.51992788247617482847860966235652136208,
        1925.0,
        449.9445569063168119446858607650988409623,
        32670.0,
        6955.999602515376140356310115515198987526,
        357423.0,
        75999.29304014542649875303443598909137092,
        2637558.0,
        601859.6171681098786670226533699352302507,
        13339535.0,
        3481712.15498064590882071018964774556468,
        45995730.0,
        14605578.08768506808414169982791359218571,
        105258076.0,
        43338889.32467613834773723740590533316085,
        150917976.0,
        86363131.28813859145546927288977868422342,
        120543840.0,
        103794043.1163445451906271053616070238554,
        39916800.0,
        56906521.91347156388090791033559122686859,
        0.0
    };

    private Lanczos13m53() {
    }

    static double g() {
        return LANCZOS_G;
    }

    static double lanczosSum(double x) {
        return evaluateVector(LANCZOS_SUM_COEFFICIENTS, x, LANCZOS_SUM_LIMIT);
    }

    static double lanczosSumScalar(double x) {
        return evaluateScalar(LANCZOS_SUM_COEFFICIENTS, x, LANCZOS_SUM_LIMIT);
    }

    static double lanczosSumExpGScaled(double x) {
        return evaluateVector(LANCZOS_SUM_EXP_G_SCALED_COEFFICIENTS, x, LANCZOS_SUM_EXP_G_SCALED_LIMIT);
    }

    static double lanczosSumExpGScaledScalar(double x) {
        return evaluateScalar(LANCZOS_SUM_EXP_G_SCALED_COEFFICIENTS, x, LANCZOS_SUM_EXP_G_SCALED_LIMIT);
    }

    private static double evaluateVector(double[] coefficients, double x, double reciprocalLimit) {
        if (x > reciprocalLimit) {
            return evaluateReciprocal(coefficients, x);
        }

        DoubleVector vx = DoubleVector.broadcast(PAIR_SPECIES, x);
        DoubleVector vx2 = vx.mul(vx);
        DoubleVector sumEven = DoubleVector.fromArray(PAIR_SPECIES, coefficients, 0);
        DoubleVector sumOdd = DoubleVector.fromArray(PAIR_SPECIES, coefficients, 2);

        for (int offset = 4; offset < 24; offset += 4) {
            sumEven = sumEven.mul(vx2).add(DoubleVector.fromArray(PAIR_SPECIES, coefficients, offset));
            sumOdd = sumOdd.mul(vx2).add(DoubleVector.fromArray(PAIR_SPECIES, coefficients, offset + 2));
        }

        sumEven = sumEven.mul(vx2).add(DoubleVector.fromArray(PAIR_SPECIES, coefficients, 24));
        DoubleVector total = sumEven.add(sumOdd.mul(vx));
        return total.lane(0) / total.lane(1);
    }

    private static double evaluateScalar(double[] coefficients, double x, double reciprocalLimit) {
        if (x > reciprocalLimit) {
            return evaluateReciprocal(coefficients, x);
        }

        double x2 = x * x;
        double numeratorEven = coefficients[0];
        double denominatorEven = coefficients[1];
        double numeratorOdd = coefficients[2];
        double denominatorOdd = coefficients[3];

        for (int offset = 4; offset < 24; offset += 4) {
            numeratorEven = numeratorEven * x2 + coefficients[offset];
            denominatorEven = denominatorEven * x2 + coefficients[offset + 1];
            numeratorOdd = numeratorOdd * x2 + coefficients[offset + 2];
            denominatorOdd = denominatorOdd * x2 + coefficients[offset + 3];
        }

        numeratorEven = numeratorEven * x2 + coefficients[24];
        denominatorEven = denominatorEven * x2 + coefficients[25];
        return (numeratorEven + numeratorOdd * x) / (denominatorEven + denominatorOdd * x);
    }

    private static double evaluateReciprocal(double[] coefficients, double x) {
        double z = 1.0 / x;
        double numerator = coefficients[24];
        double denominator = coefficients[25];
        for (int offset = 22; offset >= 0; offset -= 2) {
            numerator = numerator * z + coefficients[offset];
            denominator = denominator * z + coefficients[offset + 1];
        }
        return numerator / denominator;
    }
}