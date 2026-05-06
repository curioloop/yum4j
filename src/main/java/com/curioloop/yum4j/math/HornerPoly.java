package com.curioloop.yum4j.math;

/**
 * Experimental polynomial evaluator with short-case unrolled Horner variants.
 */
public final class HornerPoly {

    private HornerPoly() {
    }

    public static double evaluatePolynomial(double[] coefficients, double x) {
        return evaluatePolynomial(coefficients, x, coefficients.length);
    }

    public static double evaluatePolynomial(double[] coefficients, double x, int length) {
        return switch (length) {
            case 0 -> 0.0;
            case 1 -> coefficients[0];
            case 2 -> poly2(coefficients, x);
            case 3 -> poly3(coefficients, x);
            case 4 -> poly4(coefficients, x);
            case 5 -> poly5(coefficients, x);
            case 6 -> poly6(coefficients, x);
            case 7 -> poly7(coefficients, x);
            case 8 -> poly8(coefficients, x);
            case 9 -> poly9(coefficients, x);
            case 10 -> poly10(coefficients, x);
            case 11 -> poly11(coefficients, x);
            case 12 -> poly12(coefficients, x);
            default -> polyGeneric(coefficients, x, length);
        };
    }

    public static double evaluateEvenPolynomial(double[] coefficients, double x) {
        return evaluatePolynomial(coefficients, x * x);
    }

    public static double evaluateOddPolynomial(double[] coefficients, double x) {
        return Math.fma(evaluatePolynomial(coefficients, x * x), x, 0.0);
    }

    public static double evaluateRational(double[] numerator, double[] denominator, double x) {
        return evaluatePolynomial(numerator, x) / evaluatePolynomial(denominator, x);
    }

    private static double poly2(double[] coefficients, double x) {
        return Math.fma(coefficients[1], x, coefficients[0]);
    }

    private static double poly3(double[] coefficients, double x) {
        return Math.fma(Math.fma(coefficients[2], x, coefficients[1]), x, coefficients[0]);
    }

    private static double poly4(double[] coefficients, double x) {
        return Math.fma(Math.fma(Math.fma(coefficients[3], x, coefficients[2]), x, coefficients[1]), x, coefficients[0]);
    }

    private static double poly5(double[] coefficients, double x) {
        double x2 = x * x;
        double even = Math.fma(Math.fma(coefficients[4], x2, coefficients[2]), x2, coefficients[0]);
        double odd = Math.fma(coefficients[3], x2, coefficients[1]);
        return Math.fma(odd, x, even);
    }

    private static double poly6(double[] coefficients, double x) {
        double x2 = x * x;
        double even = Math.fma(Math.fma(coefficients[4], x2, coefficients[2]), x2, coefficients[0]);
        double odd = Math.fma(Math.fma(coefficients[5], x2, coefficients[3]), x2, coefficients[1]);
        return Math.fma(odd, x, even);
    }

    private static double poly7(double[] coefficients, double x) {
        double x2 = x * x;
        double even = Math.fma(Math.fma(Math.fma(coefficients[6], x2, coefficients[4]), x2, coefficients[2]), x2, coefficients[0]);
        double odd = Math.fma(Math.fma(coefficients[5], x2, coefficients[3]), x2, coefficients[1]);
        return Math.fma(odd, x, even);
    }

    private static double poly8(double[] coefficients, double x) {
        double x2 = x * x;
        double even = Math.fma(Math.fma(Math.fma(coefficients[6], x2, coefficients[4]), x2, coefficients[2]), x2, coefficients[0]);
        double odd = Math.fma(Math.fma(Math.fma(coefficients[7], x2, coefficients[5]), x2, coefficients[3]), x2, coefficients[1]);
        return Math.fma(odd, x, even);
    }

    private static double poly9(double[] coefficients, double x) {
        double x2 = x * x;
        double even = Math.fma(Math.fma(Math.fma(Math.fma(coefficients[8], x2, coefficients[6]), x2, coefficients[4]), x2, coefficients[2]), x2, coefficients[0]);
        double odd = Math.fma(Math.fma(Math.fma(coefficients[7], x2, coefficients[5]), x2, coefficients[3]), x2, coefficients[1]);
        return Math.fma(odd, x, even);
    }

    private static double poly10(double[] coefficients, double x) {
        double x2 = x * x;
        double even = Math.fma(Math.fma(Math.fma(Math.fma(coefficients[8], x2, coefficients[6]), x2, coefficients[4]), x2, coefficients[2]), x2, coefficients[0]);
        double odd = Math.fma(Math.fma(Math.fma(Math.fma(coefficients[9], x2, coefficients[7]), x2, coefficients[5]), x2, coefficients[3]), x2, coefficients[1]);
        return Math.fma(odd, x, even);
    }

    private static double poly11(double[] coefficients, double x) {
        double x2 = x * x;
        double even = Math.fma(Math.fma(Math.fma(Math.fma(Math.fma(coefficients[10], x2, coefficients[8]), x2, coefficients[6]), x2, coefficients[4]), x2, coefficients[2]), x2, coefficients[0]);
        double odd = Math.fma(Math.fma(Math.fma(Math.fma(coefficients[9], x2, coefficients[7]), x2, coefficients[5]), x2, coefficients[3]), x2, coefficients[1]);
        return Math.fma(odd, x, even);
    }

    private static double poly12(double[] coefficients, double x) {
        double x2 = x * x;
        double even = Math.fma(Math.fma(Math.fma(Math.fma(Math.fma(coefficients[10], x2, coefficients[8]), x2, coefficients[6]), x2, coefficients[4]), x2, coefficients[2]), x2, coefficients[0]);
        double odd = Math.fma(Math.fma(Math.fma(Math.fma(Math.fma(coefficients[11], x2, coefficients[9]), x2, coefficients[7]), x2, coefficients[5]), x2, coefficients[3]), x2, coefficients[1]);
        return Math.fma(odd, x, even);
    }

    private static double polyGeneric(double[] coefficients, double x, int length) {
        if (length == 0) {
            return 0.0;
        }

        double value = coefficients[length - 1];
        for (int index = length - 2; index >= 0; index--) {
            value = Math.fma(value, x, coefficients[index]);
        }
        return value;
    }

}