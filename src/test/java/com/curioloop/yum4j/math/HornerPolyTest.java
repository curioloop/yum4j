package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.assertTrue;

class HornerPolyTest {

    private static final double[] SAMPLE_X = {-3.0, -1.5, -0.75, -0.125, 0.0, 0.125, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0};
    private static final double RELATIVE_TOLERANCE = 2.0e-13;
    private static final double ABSOLUTE_TOLERANCE = 2.0e-15;

    @Test
    void polynomialTracksReferenceForShortAndLongCases() {
        for (int length = 0; length <= 16; length++) {
            double[] coefficients = syntheticCoefficients(length);
            for (double x : SAMPLE_X) {
                assertClose(
                    referenceEvaluatePolynomial(coefficients, x),
                    HornerPoly.evaluatePolynomial(coefficients, x),
                    "full length=" + length + ", x=" + x);
            }
        }
    }

    @Test
    void publicHelpersTrackReference() {
        double[] evenCoefficients = {1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125};
        double[] oddCoefficients = {0.75, -0.2, 0.1, -0.04, 0.02};
        double[] numerator = {1.0, -0.2, 0.05, -0.01, 0.005, -0.001};
        double[] denominator = {1.0, 0.25, 0.125, 0.0625, 0.03125, 0.015625};

        for (double x : SAMPLE_X) {
            assertClose(
                referenceEvaluateEvenPolynomial(evenCoefficients, x),
                HornerPoly.evaluateEvenPolynomial(evenCoefficients, x),
                "even x=" + x);
            assertClose(
                referenceEvaluateOddPolynomial(oddCoefficients, x),
                HornerPoly.evaluateOddPolynomial(oddCoefficients, x),
                "odd x=" + x);
            assertClose(
                referenceEvaluateRational(numerator, denominator, x),
                HornerPoly.evaluateRational(numerator, denominator, x),
                "rational x=" + x);
        }
    }

    @Test
    void representativeBoostTablesTrackReference() throws ReflectiveOperationException {
        double[] normalP = staticDoubleArray(Normal.class, "ERF_INV_P_LE_HALF_P");
        double[] normalQ = staticDoubleArray(Normal.class, "ERF_INV_P_LE_HALF_Q");
        double[] expintP = staticDoubleArray(Expint.class, "E1_SMALL_P");
        double[] expintQ = staticDoubleArray(Expint.class, "E1_SMALL_Q");
        double[] betaEven = staticDoubleArray(Beta.class, "TEMME_METHOD2_COEFFS_10");
        double[] gammaP = staticDoubleArray(Gamma.class, "DIGAMMA_P");
        double[] gammaQ = staticDoubleArray(Gamma.class, "DIGAMMA_Q");
        double[] zetaP = staticDoubleArray(Zeta.class, "REGION_LE_FOUR_P");
        double[] zetaQ = staticDoubleArray(Zeta.class, "REGION_LE_FOUR_Q");

        double[] normalInputs = {1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 0.05, 0.1, 0.2, 0.35, 0.5};
        for (double input : normalInputs) {
            assertClose(
                referenceEvaluateRational(normalP, normalQ, input),
                HornerPoly.evaluateRational(normalP, normalQ, input),
                "normal rational input=" + input);
        }

        double[] expintInputs = {1.0e-8, 1.0e-6, 1.0e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0};
        for (double input : expintInputs) {
            assertClose(
                referenceEvaluateRational(expintP, expintQ, input),
                HornerPoly.evaluateRational(expintP, expintQ, input),
                "expint rational input=" + input);
        }

        double[] betaInputs = {0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95};
        for (double input : betaInputs) {
            assertClose(
                referenceEvaluateEvenPolynomial(betaEven, input),
                HornerPoly.evaluateEvenPolynomial(betaEven, input),
                "beta even input=" + input);
        }

        double[] gammaInputs = {0.0, 0.1, 0.25, 0.5, 0.75, 1.0};
        for (double input : gammaInputs) {
            assertClose(
                referenceEvaluateRational(gammaP, gammaQ, input),
                HornerPoly.evaluateRational(gammaP, gammaQ, input),
                "gamma rational input=" + input);
        }

        double[] zetaInputs = {-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0};
        for (double input : zetaInputs) {
            assertClose(
                referenceEvaluateRational(zetaP, zetaQ, input),
                HornerPoly.evaluateRational(zetaP, zetaQ, input),
                "zeta rational input=" + input);
        }
    }

    @Test
    void gammaFixedDoubleTablesTrackReference() throws ReflectiveOperationException {
        double[][] gammaFixed = staticDoubleMatrix(Gamma.class, "POLY_COT_PI_FIXED_COEFFICIENTS");
        double[] gammaInputs = {0.0, 0.1, 0.25, 0.5, 0.75, 1.0};

        for (int order = 0; order < gammaFixed.length; order++) {
            double[] coefficients = gammaFixed[order];
            if (coefficients == null) {
                continue;
            }
            for (double input : gammaInputs) {
                assertClose(
                    referenceEvaluateEvenPolynomial(coefficients, input),
                    HornerPoly.evaluateEvenPolynomial(coefficients, input),
                    "gamma fixed double order=" + order + ", x=" + input);
            }
        }
    }

    private static void assertClose(double expected, double actual, String label) {
        double scale = Math.max(1.0, Math.abs(expected));
        double tolerance = Math.max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * scale);
        double error = Math.abs(expected - actual);
        assertTrue(error <= tolerance,
            label + ": expected=" + expected + ", actual=" + actual + ", error=" + error + ", tolerance=" + tolerance);
    }

    private static double referenceEvaluatePolynomial(double[] coefficients, double x) {
        return referenceEvaluatePolynomial(coefficients, x, coefficients.length);
    }

    private static double referenceEvaluatePolynomial(double[] coefficients, double x, int length) {
        if (length == 0) {
            return 0.0;
        }

        double value = coefficients[length - 1];
        for (int index = length - 2; index >= 0; index--) {
            value = Math.fma(value, x, coefficients[index]);
        }
        return value;
    }

    private static double referenceEvaluateEvenPolynomial(double[] coefficients, double x) {
        return referenceEvaluatePolynomial(coefficients, x * x);
    }

    private static double referenceEvaluateOddPolynomial(double[] coefficients, double x) {
        return x * referenceEvaluatePolynomial(coefficients, x * x);
    }

    private static double referenceEvaluateRational(double[] numerator, double[] denominator, double x) {
        return referenceEvaluatePolynomial(numerator, x) / referenceEvaluatePolynomial(denominator, x);
    }

    private static double[] syntheticCoefficients(int length) {
        double[] coefficients = new double[length];
        for (int index = 0; index < length; index++) {
            double sign = (index & 1) == 0 ? 1.0 : -1.0;
            coefficients[index] = sign * (index + 1.0) / (length + index + 1.0);
        }
        return coefficients;
    }

    private static double[] staticDoubleArray(Class<?> type, String fieldName) throws ReflectiveOperationException {
        Field field = type.getDeclaredField(fieldName);
        field.setAccessible(true);
        return ((double[]) field.get(null)).clone();
    }

    private static double[][] staticDoubleMatrix(Class<?> type, String fieldName) throws ReflectiveOperationException {
        Field field = type.getDeclaredField(fieldName);
        field.setAccessible(true);
        double[][] matrix = (double[][]) field.get(null);
        double[][] copy = new double[matrix.length][];
        for (int index = 0; index < matrix.length; index++) {
            copy[index] = matrix[index] == null ? null : matrix[index].clone();
        }
        return copy;
    }
}