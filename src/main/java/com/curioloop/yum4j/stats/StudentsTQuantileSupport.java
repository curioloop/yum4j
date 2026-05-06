package com.curioloop.yum4j.stats;

import static com.curioloop.yum4j.math.HornerPoly.evaluateOddPolynomial;
import static com.curioloop.yum4j.math.HornerPoly.evaluatePolynomial;

import com.curioloop.yum4j.math.Beta;
import com.curioloop.yum4j.math.Double2;
import com.curioloop.yum4j.math.DoubleI;
import com.curioloop.yum4j.math.Gamma;
import com.curioloop.yum4j.math.Normal;

final class StudentsTQuantileSupport {

    private static final double HALF = 0.5;
    private static final double PI = Math.PI;
    private static final double SQRT_TWO = Math.sqrt(2.0);
    private static final int DOUBLE_DIGITS = 53;
    private static final double LARGE_DF_NORMAL_THRESHOLD = 0x10000000L;
    private static final double EXACT_NORMAL_THRESHOLD = 1.0e20;
    private static final double DF_LT_THREE_CROSSOVER_INTERCEPT = 0.2742;
    private static final double DF_LT_THREE_CROSSOVER_SLOPE = 0.0242143;
    private static final double HILL_TAIL_CROSSOVER_DENOMINATOR = 0.654;
    private static final double DF_SIX_UNDERFLOW_THRESHOLD = 1.0e-150;
    private static final double DF_SIX_SEED = 0.85498797333834849467655443627193;

    private StudentsTQuantileSupport() {
    }

    static double fastQuantile(double degreesOfFreedom, double probability) {
        if ((degreesOfFreedom < 2.0) && (Math.floor(degreesOfFreedom) != degreesOfFreedom)) {
            return inverseBetaQuantile(degreesOfFreedom, probability);
        }

        boolean invert = false;
        double lowerTailProbability = probability;
        if (lowerTailProbability > HALF) {
            lowerTailProbability = 1.0 - lowerTailProbability;
            invert = true;
        }

        DoubleI initial = inverseStudentsT(degreesOfFreedom, lowerTailProbability, 1.0 - lowerTailProbability);
        double t = initial.value();
        if ((t == 0.0) || initial.flag()) {
            return invert ? -t : t;
        }

        double tSquared = t * t;
        double xb = degreesOfFreedom / (degreesOfFreedom + tSquared);
        double y = tSquared / (degreesOfFreedom + tSquared);
        double a = degreesOfFreedom * HALF;
        if (xb == 0.0) {
            return t;
        }

        double incompleteBeta;
        double incompleteBetaDerivative;
        if (xb < y) {
            incompleteBeta = Beta.ibeta(a, HALF, xb);
            incompleteBetaDerivative = Beta.ibetaDerivative(a, HALF, xb);
        } else {
            incompleteBeta = Beta.ibetac(HALF, a, y);
            incompleteBetaDerivative = Beta.ibetaDerivative(HALF, a, y);
        }

        double cdfResidual = incompleteBeta * HALF - lowerTailProbability;
        double pdfAtT = incompleteBetaDerivative * Math.sqrt(y * xb * xb * xb / degreesOfFreedom);
        double secondDerivativeRatio = t * (degreesOfFreedom + 1.0) / (tSquared + degreesOfFreedom);

        t = Math.abs(t);
        t += cdfResidual / (pdfAtT + cdfResidual * secondDerivativeRatio * HALF);
        return invert ? t : -t;
    }

    private static double inverseBetaQuantile(double degreesOfFreedom, double probability) {
        double tailProbability = probability > HALF ? 1.0 - probability : probability;
        Double2 xy = Beta.ibetaInvComplement(degreesOfFreedom * HALF, HALF, 2.0 * tailProbability);
        double x = xy._1();
        double y = xy._2();
        if (degreesOfFreedom * y > Double.MAX_VALUE * x) {
            throw new ArithmeticException("StudentsTDistribution.quantile overflow");
        }
        double quantile = Math.sqrt(degreesOfFreedom * y / x);
        return probability < HALF ? -quantile : quantile;
    }

    private static DoubleI inverseStudentsT(double degreesOfFreedom, double u, double v) {
        boolean invert = false;
        if (u > v) {
            double tmp = u;
            u = v;
            v = tmp;
            invert = true;
        }

        boolean exact = false;
        double result;
        if ((Math.floor(degreesOfFreedom) == degreesOfFreedom) && (degreesOfFreedom < 20.0)) {
            double tolerance = Math.scalb(1.0, (2 * DOUBLE_DIGITS) / 3);
            switch ((int) degreesOfFreedom) {
                case 1 -> {
                    result = (u == HALF) ? 0.0 : -Math.cos(PI * u) / Math.sin(PI * u);
                    exact = true;
                }
                case 2 -> {
                    result = (2.0 * u - 1.0) / Math.sqrt(2.0 * u * v);
                    exact = true;
                }
                case 4 -> {
                    double alpha = 4.0 * u * v;
                    double rootAlpha = Math.sqrt(alpha);
                    double r = 4.0 * Math.cos(Math.acos(rootAlpha) / 3.0) / rootAlpha;
                    double x = Math.sqrt(r - 4.0);
                    result = (u - HALF) < 0.0 ? -x : x;
                    exact = true;
                }
                case 6 -> {
                    if (u < DF_SIX_UNDERFLOW_THRESHOLD) {
                        result = (invert ? -1.0 : 1.0) * inverseStudentsTHill(degreesOfFreedom, u);
                        return DoubleI.flagged(result, false);
                    }

                    double a = 4.0 * (u - u * u);
                    double b = Math.cbrt(a);
                    double p = 6.0 * (1.0 + DF_SIX_SEED * (1.0 / b - 1.0));
                    double previous;
                    do {
                        double pSquared = p * p;
                        double pFourth = pSquared * pSquared;
                        double pFifth = p * pFourth;
                        previous = p;
                        p = 2.0 * (8.0 * a * pFifth - 270.0 * pSquared + 2187.0)
                            / (5.0 * (4.0 * a * pFourth - 216.0 * p - 243.0));
                    } while (Math.abs((p - previous) / p) > tolerance);
                    p = Math.sqrt(p - degreesOfFreedom);
                    result = (u - HALF) < 0.0 ? -p : p;
                }
                default -> result = inverseStudentsTGeneral(degreesOfFreedom, u);
            }
        } else {
            result = inverseStudentsTGeneral(degreesOfFreedom, u);
            exact = degreesOfFreedom >= EXACT_NORMAL_THRESHOLD;
        }

        return DoubleI.flagged(invert ? -result : result, exact);
    }

    private static double inverseStudentsTGeneral(double degreesOfFreedom, double u) {
        if (degreesOfFreedom > LARGE_DF_NORMAL_THRESHOLD) {
            return normalLimitQuantile(u);
        }
        if (degreesOfFreedom < 3.0) {
            double crossover = DF_LT_THREE_CROSSOVER_INTERCEPT - degreesOfFreedom * DF_LT_THREE_CROSSOVER_SLOPE;
            return u > crossover
                ? inverseStudentsTBodySeries(degreesOfFreedom, u)
                : inverseStudentsTTailSeries(degreesOfFreedom, u);
        }

        int exponent = frexpExponent(u);
        if ((u > 0.0) && (exponent < degreesOfFreedom / HILL_TAIL_CROSSOVER_DENOMINATOR)) {
            return inverseStudentsTHill(degreesOfFreedom, u);
        }
        return inverseStudentsTTailSeries(degreesOfFreedom, u);
    }

    private static double inverseStudentsTHill(double degreesOfFreedom, double u) {
        if (degreesOfFreedom > EXACT_NORMAL_THRESHOLD) {
            return normalLimitQuantile(u);
        }

        double a = 1.0 / (degreesOfFreedom - 0.5);
        double b = 48.0 / (a * a);
        double c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36;
        double d = ((94.5 / (b + c) - 3.0) / b + 1.0) * Math.sqrt(a * PI / 2.0) * degreesOfFreedom;
        double y = Math.pow(d * 2.0 * u, 2.0 / degreesOfFreedom);

        if (y > (0.05 + a)) {
            double x = normalLimitQuantile(u);
            y = x * x;
            if (degreesOfFreedom < 5.0) {
                c += 0.3 * (degreesOfFreedom - 4.5) * (x + 0.6);
            }
            c += (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b;
            y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x;
            y = Math.expm1(a * y * y);
        } else {
            y = (((1.0 / (((degreesOfFreedom + 6.0) / (degreesOfFreedom * y) - 0.089 * d - 0.822)
                * (degreesOfFreedom + 2.0) * 3.0) + 0.5 / (degreesOfFreedom + 4.0)) * y - 1.0)
                * (degreesOfFreedom + 1.0) / (degreesOfFreedom + 2.0) + 1.0 / y);
        }

        return -Math.sqrt(degreesOfFreedom * y);
    }

    private static double inverseStudentsTTailSeries(double degreesOfFreedom, double probability) {
        double w = Gamma.tgammaDeltaRatio(degreesOfFreedom * HALF, HALF)
            * Math.sqrt(degreesOfFreedom * PI) * probability;

        double np2 = degreesOfFreedom + 2.0;
        double np4 = degreesOfFreedom + 4.0;
        double np6 = degreesOfFreedom + 6.0;

        double[] coefficients = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        coefficients[1] = -(degreesOfFreedom + 1.0) / (2.0 * np2);
        np2 *= degreesOfFreedom + 2.0;
        coefficients[2] = -degreesOfFreedom * (degreesOfFreedom + 1.0) * (degreesOfFreedom + 3.0)
            / (8.0 * np2 * np4);
        np2 *= degreesOfFreedom + 2.0;
        coefficients[3] = -degreesOfFreedom * (degreesOfFreedom + 1.0) * (degreesOfFreedom + 5.0)
            * (((3.0 * degreesOfFreedom) + 7.0) * degreesOfFreedom - 2.0)
            / (48.0 * np2 * np4 * np6);
        np2 *= degreesOfFreedom + 2.0;
        np4 *= degreesOfFreedom + 4.0;
        double poly4 = 15.0 * degreesOfFreedom + 154.0;
        poly4 = poly4 * degreesOfFreedom + 465.0;
        poly4 = poly4 * degreesOfFreedom + 286.0;
        poly4 = poly4 * degreesOfFreedom - 336.0;
        poly4 = poly4 * degreesOfFreedom + 64.0;
        coefficients[4] = -degreesOfFreedom * (degreesOfFreedom + 1.0) * (degreesOfFreedom + 7.0)
            * poly4
            / (384.0 * np2 * np4 * np6 * (degreesOfFreedom + 8.0));
        np2 *= degreesOfFreedom + 2.0;
        double poly5 = 35.0 * degreesOfFreedom + 452.0;
        poly5 = poly5 * degreesOfFreedom + 1573.0;
        poly5 = poly5 * degreesOfFreedom + 600.0;
        poly5 = poly5 * degreesOfFreedom - 2020.0;
        poly5 = poly5 * degreesOfFreedom + 928.0;
        poly5 = poly5 * degreesOfFreedom - 128.0;
        coefficients[5] = -degreesOfFreedom * (degreesOfFreedom + 1.0) * (degreesOfFreedom + 3.0)
            * (degreesOfFreedom + 9.0)
            * poly5
            / (1280.0 * np2 * np4 * np6 * (degreesOfFreedom + 8.0) * (degreesOfFreedom + 10.0));
        np2 *= degreesOfFreedom + 2.0;
        np4 *= degreesOfFreedom + 4.0;
        np6 *= degreesOfFreedom + 6.0;
        double poly6 = 945.0 * degreesOfFreedom + 31506.0;
        poly6 = poly6 * degreesOfFreedom + 425858.0;
        poly6 = poly6 * degreesOfFreedom + 2980236.0;
        poly6 = poly6 * degreesOfFreedom + 11266745.0;
        poly6 = poly6 * degreesOfFreedom + 20675018.0;
        poly6 = poly6 * degreesOfFreedom + 7747124.0;
        poly6 = poly6 * degreesOfFreedom - 22574632.0;
        poly6 = poly6 * degreesOfFreedom - 8565600.0;
        poly6 = poly6 * degreesOfFreedom + 18108416.0;
        poly6 = poly6 * degreesOfFreedom - 7099392.0;
        poly6 = poly6 * degreesOfFreedom + 884736.0;
        coefficients[6] = -degreesOfFreedom * (degreesOfFreedom + 1.0) * (degreesOfFreedom + 11.0)
            * poly6
            / (46080.0 * np2 * np4 * np6 * (degreesOfFreedom + 8.0) * (degreesOfFreedom + 10.0)
            * (degreesOfFreedom + 12.0));

        double rootDegrees = Math.sqrt(degreesOfFreedom);
        double div = Math.pow(rootDegrees * w, 1.0 / degreesOfFreedom);
        double power = div * div;
        double result = evaluatePolynomial(coefficients, power);
        result *= rootDegrees;
        result /= div;
        return -result;
    }

    private static double inverseStudentsTBodySeries(double degreesOfFreedom, double probability) {
        double v = Gamma.tgammaDeltaRatio(degreesOfFreedom * HALF, HALF)
            * Math.sqrt(degreesOfFreedom * PI) * (probability - HALF);

        double[] coefficients = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double inverseDf = 1.0 / degreesOfFreedom;
        coefficients[1] = 0.16666666666666666 + 0.16666666666666666 * inverseDf;
        coefficients[2] = (0.008333333333333333 * inverseDf + 0.06666666666666667) * inverseDf
            + 0.058333333333333334;
        coefficients[3] = ((0.0001984126984126984 * inverseDf + 0.0017857142857142857) * inverseDf
            + 0.026785714285714284) * inverseDf + 0.0251984126984127;
        coefficients[4] = (((2.755731922398589E-6 * inverseDf + 0.00037477954144620813) * inverseDf
            - 0.0011078042328042327) * inverseDf + 0.010559964726631393) * inverseDf + 0.012039792768959436;
        coefficients[5] = ((((2.505210838544172E-8 * inverseDf - 0.00006270542728876063) * inverseDf
            + 0.0005945867404200737) * inverseDf - 0.0016095979637646305) * inverseDf
            + 0.006103921156004489) * inverseDf + 0.003837005972422639;
        coefficients[6] = (((((1.6059043836821614E-10 * inverseDf + 0.0000154012654012654) * inverseDf
            - 0.00016376804137220805) * inverseDf + 0.0006908420797309686) * inverseDf
            - 0.0012579159844784845) * inverseDf + 0.0010898206731540064) * inverseDf
            + 0.0032177478835464947;
        coefficients[7] = ((((((7.647163731819817E-13 * inverseDf - 3.9851014346715406E-6) * inverseDf
            + 0.000049255746366361445) * inverseDf - 0.000249472580470431) * inverseDf
            + 0.0006451304695145634) * inverseDf - 0.0007624513544032393) * inverseDf
            + 0.000033530976880017886) * inverseDf + 0.001743826229834001;
        coefficients[8] = (((((((2.8114572543455208E-15 * inverseDf + 1.091417917349679E-6) * inverseDf
            - 0.000015303004486655378) * inverseDf + 0.0000908671079352199) * inverseDf
            - 0.0002913341446693807) * inverseDf + 0.0005140660578834112) * inverseDf
            - 0.00036307660358786887) * inverseDf - 0.0003110108632631878) * inverseDf
            + 0.0009647274732138864;
        coefficients[9] = ((((((((8.22063524662433E-18 * inverseDf - 3.123956959982987E-7) * inverseDf
            + 4.890304529197535E-6) * inverseDf - 0.00003320265239137206) * inverseDf
            + 0.00012645437628698076) * inverseDf - 0.00028690924218514614) * inverseDf
            + 0.00035764655430568635) * inverseDf - 0.00010230378073700412) * inverseDf
            - 0.00036942667800009663) * inverseDf + 0.0005422926281312969;

        return evaluateOddPolynomial(coefficients, v);
    }

    private static int frexpExponent(double value) {
        return Math.getExponent(value) + 1;
    }

    private static double normalLimitQuantile(double probability) {
        return -Normal.erfcInv(2.0 * probability) * SQRT_TWO;
    }
}