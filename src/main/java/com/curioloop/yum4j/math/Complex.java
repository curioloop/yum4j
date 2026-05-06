package com.curioloop.yum4j.math;

/**
 * Immutable complex-number helper with principal-branch elementary functions.
 */
public value record Complex(double real, double imag) {

    private static final double HALF_PI = 0.5 * Math.PI;

    public static final Complex ZERO = new Complex(0.0, 0.0);
    public static final Complex ONE = new Complex(1.0, 0.0);
    public static final Complex I = new Complex(0.0, 1.0);
    public static final Complex NAN = new Complex(Double.NaN, Double.NaN);
    public static final Complex INF = new Complex(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);

    public static Complex polar(double radius, double angle) {
        return new Complex(radius * Math.cos(angle), radius * Math.sin(angle));
    }

    public Complex add(double re, double im) {
        return new Complex(real + re, imag + im);
    }

    public Complex add(Complex other) {
        return new Complex(real + other.real, imag + other.imag);
    }

    public Complex sub(double re, double im) {
        return new Complex(real - re, imag - im);
    }

    public Complex sub(Complex other) {
        return new Complex(real - other.real, imag - other.imag);
    }

    public Complex mul(Complex other) {
        return new Complex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }

    public Complex mul(double re, double im) {
        return new Complex(real * re, imag * im);
    }

    public Complex mul(double factor) {
        return new Complex(real * factor, imag * factor);
    }

    public Complex div(Complex other) {
        return quotient(real, imag, other.real, other.imag);
    }

    public Complex div(double re, double im) {
        return quotient(real, imag, re, im);
    }

    public Complex div(double divisor) {
        return new Complex(real / divisor, imag / divisor);
    }

    public Complex recip() {
        return quotient(1.0, 0.0, real, imag);
    }

    public Complex negate() {
        return new Complex(-real, -imag);
    }

    public Complex pow(double exponent) {
        if (exponent == 0.0) {
            return Complex.ONE;
        }
        if (exponent == 1.0) {
            return this;
        }
        if (isZero()) {
            if (exponent > 0.0) {
                return Complex.ZERO;
            }
            throw new ArithmeticException("Complex zero cannot be raised to a non-positive exponent");
        }
        double logAbs = Math.log(abs());
        double angle = arg();
        return expOfCartesian(exponent * logAbs, exponent * angle);
    }

    public Complex pow(Complex exponent) {
        if (exponent.isZero()) {
            return Complex.ONE;
        }
        if (exponent.imag == 0.0 && exponent.real == 1.0) {
            return this;
        }
        if (isZero()) {
            if (exponent.imag == 0.0 && exponent.real > 0.0) {
                return Complex.ZERO;
            }
            throw new ArithmeticException("Complex zero cannot be raised to a non-positive or non-real exponent");
        }
        double logAbs = Math.log(abs());
        double angle = arg();
        double exponentReal = exponent.real;
        double exponentImag = exponent.imag;
        double transformedReal = exponentReal * logAbs - exponentImag * angle;
        double transformedImag = exponentReal * angle + exponentImag * logAbs;
        return expOfCartesian(transformedReal, transformedImag);
    }

    public Complex exp() {
        return expOfCartesian(real, imag);
    }

    public Complex sin() {
        if (isFinite()) {
            return new Complex(
                Math.sin(real) * Math.cosh(imag),
                Math.cos(real) * Math.sinh(imag)
            );
        }
        Complex value = sinhOfCartesian(-imag, real);
        return new Complex(value.imag, -value.real);
    }

    public Complex cos() {
        if (isFinite()) {
            return new Complex(
                Math.cos(real) * Math.cosh(imag),
                -Math.sin(real) * Math.sinh(imag)
            );
        }
        Complex value = coshOfCartesian(-imag, real);
        return new Complex(value.real, value.imag);
    }

    public Complex tan() {
        if (imag == 0.0) {
            return new Complex(Math.tan(real), imag);
        }
        double doubleReal = 2.0 * real;
        double doubleImag = 2.0 * imag;
        double denominator = Math.cos(doubleReal) + Math.cosh(doubleImag);
        double sinhDoubleImag = Math.sinh(doubleImag);
        if (Double.isInfinite(sinhDoubleImag) && Double.isInfinite(denominator)) {
            return new Complex(Math.copySign(0.0, Math.sin(doubleReal)), Math.copySign(1.0, imag));
        }
        return new Complex(Math.sin(doubleReal) / denominator, sinhDoubleImag / denominator);
    }

    public Complex sinh() {
        return sinhOfCartesian(real, imag);
    }

    public Complex cosh() {
        return coshOfCartesian(real, imag);
    }

    public Complex tanh() {
        if (imag == 0.0) {
            return new Complex(Math.tanh(real), imag);
        }
        if (Double.isInfinite(real)) {
            if (!Double.isFinite(imag)) {
                return new Complex(Math.copySign(1.0, real), 0.0);
            }
            return new Complex(Math.copySign(1.0, real), Math.copySign(0.0, Math.sin(2.0 * imag)));
        }
        double doubleReal = 2.0 * real;
        double doubleImag = 2.0 * imag;
        double denominator = Math.cosh(doubleReal) + Math.cos(doubleImag);
        double sinhDoubleReal = Math.sinh(doubleReal);
        if (Double.isInfinite(sinhDoubleReal) && Double.isInfinite(denominator)) {
            return new Complex(Math.copySign(1.0, real), Math.copySign(0.0, Math.sin(doubleImag)));
        }
        return new Complex(sinhDoubleReal / denominator, Math.sin(doubleImag) / denominator);
    }

    public Complex log() {
        if (Double.isFinite(real) && Double.isFinite(imag)) {
            return finiteLog(real, imag);
        }
        return new Complex(Math.log(abs()), arg());
    }

    public Complex sqrt() {
        if (Double.isFinite(real) && Double.isFinite(imag)) {
            return finiteSqrt(real, imag);
        }
        return exceptionalSqrt(real, imag);
    }

    public Complex asin() {
        Complex result = new Complex(-imag, real).asinh();
        return new Complex(result.imag, -result.real);
    }

    public Complex acos() {
        Complex result = asin();
        return new Complex(HALF_PI - result.real, -result.imag);
    }

    public Complex atan() {
        Complex result = new Complex(-imag, real).atanh();
        return new Complex(result.imag, -result.real);
    }

    public Complex asinh() {
        if (Double.isInfinite(real)) {
            if (Double.isNaN(imag)) {
                return this;
            }
            if (Double.isInfinite(imag)) {
                return new Complex(real, Math.copySign(0.25 * Math.PI, imag));
            }
            return new Complex(real, Math.copySign(0.0, imag));
        }
        if (Double.isNaN(real)) {
            if (Double.isInfinite(imag)) {
                return new Complex(imag, real);
            }
            if (imag == 0.0) {
                return new Complex(real, imag);
            }
            return new Complex(real, real);
        }
        if (Double.isInfinite(imag)) {
            return new Complex(Math.copySign(imag, real), Math.copySign(0.5 * Math.PI, imag));
        }

        Complex result = add(mul(this).add(ONE).sqrt()).log();
        return new Complex(
            Math.copySign(Math.abs(result.real), real),
            Math.copySign(Math.abs(result.imag), imag)
        );
    }

    public Complex acosh() {
        if (Double.isInfinite(real)) {
            if (Double.isNaN(imag)) {
                return new Complex(Math.abs(real), imag);
            }
            if (Double.isInfinite(imag)) {
                if (real > 0.0) {
                    return new Complex(real, Math.copySign(0.25 * Math.PI, imag));
                }
                return new Complex(-real, Math.copySign(0.75 * Math.PI, imag));
            }
            if (real < 0.0) {
                return new Complex(-real, Math.copySign(Math.PI, imag));
            }
            return new Complex(real, Math.copySign(0.0, imag));
        }
        if (Double.isNaN(real)) {
            if (Double.isInfinite(imag)) {
                return new Complex(Math.abs(imag), real);
            }
            return new Complex(real, real);
        }
        if (Double.isInfinite(imag)) {
            return new Complex(Math.abs(imag), Math.copySign(0.5 * Math.PI, imag));
        }

        Complex result = add(add(ONE).sqrt().mul(sub(ONE).sqrt())).log();
        return new Complex(
            Math.abs(result.real),
            Math.copySign(Math.abs(result.imag), imag)
        );
    }

    public Complex atanh() {
        if (Double.isInfinite(imag)) {
            return new Complex(Math.copySign(0.0, real), Math.copySign(0.5 * Math.PI, imag));
        }
        if (Double.isNaN(imag)) {
            if (Double.isInfinite(real) || real == 0.0) {
                return new Complex(Math.copySign(0.0, real), imag);
            }
            return new Complex(imag, imag);
        }
        if (Double.isNaN(real)) {
            return new Complex(real, real);
        }
        if (Double.isInfinite(real)) {
            return new Complex(Math.copySign(0.0, real), Math.copySign(0.5 * Math.PI, imag));
        }
        if (Math.abs(real) == 1.0 && imag == 0.0) {
            return new Complex(Math.copySign(Double.POSITIVE_INFINITY, real), Math.copySign(0.0, imag));
        }

        double imagSquared = imag * imag;
        double x = 1.0 - imagSquared - real * real;
        double numerator = 1.0 + real;
        double denominator = 1.0 - real;
        numerator = imagSquared + numerator * numerator;
        denominator = imagSquared + denominator * denominator;
        double realPart = 0.25 * (Math.log(numerator) - Math.log(denominator));
        double imagPart = 0.5 * Math.atan2(2.0 * imag, x);
        return new Complex(
            Math.copySign(Math.abs(realPart), real),
            Math.copySign(Math.abs(imagPart), imag)
        );
    }

    public Complex conj() {
        return new Complex(real, -imag);
    }

    public double abs() {
        if (Double.isFinite(real) && Double.isFinite(imag)) {
            return finiteMagnitude(real, imag);
        }
        return Math.hypot(real, imag);
    }

    public double norm() {
        double scale = Math.max(Math.abs(real), Math.abs(imag));
        if (scale == 0.0) {
            return 0.0;
        }
        double scaledReal = real / scale;
        double scaledImag = imag / scale;
        return scale * scale * (scaledReal * scaledReal + scaledImag * scaledImag);
    }

    public double arg() {
        return Math.atan2(imag, real);
    }

    public boolean isNaN() {
        return Double.isNaN(real) || Double.isNaN(imag);
    }

    public boolean isFinite() {
        return Double.isFinite(real) && Double.isFinite(imag);
    }

    public boolean isInfinite() {
        return Double.isInfinite(real) || Double.isInfinite(imag);
    }

    public boolean isZero() {
        return real == 0.0 && imag == 0.0;
    }

    private static Complex expOfCartesian(double real, double imag) {
        if (Double.isFinite(real) && Double.isFinite(imag)) {
            return finiteExpOfCartesian(real, imag);
        }
        return exceptionalExpOfCartesian(real, imag);
    }

    private static Complex sinhOfCartesian(double real, double imag) {
        if (Double.isInfinite(real) && !Double.isFinite(imag)) {
            return new Complex(real, Double.NaN);
        }
        if (real == 0.0 && !Double.isFinite(imag)) {
            return new Complex(real, Double.NaN);
        }
        if (imag == 0.0 && !Double.isFinite(real)) {
            return new Complex(real, imag);
        }
        return new Complex(
            Math.sinh(real) * Math.cos(imag),
            Math.cosh(real) * Math.sin(imag)
        );
    }

    private static Complex coshOfCartesian(double real, double imag) {
        if (Double.isInfinite(real) && !Double.isFinite(imag)) {
            return new Complex(Math.abs(real), Double.NaN);
        }
        if (real == 0.0 && !Double.isFinite(imag)) {
            return new Complex(Double.NaN, real);
        }
        if (real == 0.0 && imag == 0.0) {
            return new Complex(1.0, imag);
        }
        if (imag == 0.0 && !Double.isFinite(real)) {
            return new Complex(Math.abs(real), imag);
        }
        return new Complex(
            Math.cosh(real) * Math.cos(imag),
            Math.sinh(real) * Math.sin(imag)
        );
    }

    private static Complex quotient(double a, double b, double c, double d) {
        if (Double.isFinite(a)
            && Double.isFinite(b)
            && Double.isFinite(c)
            && Double.isFinite(d)
            && (c != 0.0 || d != 0.0)) {
            return finiteQuotient(a, b, c, d);
        }
        return exceptionalQuotient(a, b, c, d);
    }

    private static Complex finiteExpOfCartesian(double real, double imag) {
        if (imag == 0.0) {
            return new Complex(Math.exp(real), Math.copySign(0.0, imag));
        }
        if (real == 0.0) {
            return new Complex(Math.cos(imag), Math.sin(imag));
        }
        double magnitude = Math.exp(real);
        return new Complex(magnitude * Math.cos(imag), magnitude * Math.sin(imag));
    }

    private static Complex exceptionalExpOfCartesian(double real, double imag) {
        double imaginary = imag;
        if (Double.isInfinite(real)) {
            if (real < 0.0) {
                if (!Double.isFinite(imaginary)) {
                    imaginary = 1.0;
                }
            } else if (imaginary == 0.0 || !Double.isFinite(imaginary)) {
                if (Double.isInfinite(imaginary)) {
                    imaginary = Double.NaN;
                }
                return new Complex(real, imaginary);
            }
        } else if (Double.isNaN(real) && imag == 0.0) {
            return new Complex(real, imag);
        }
        double magnitude = Math.exp(real);
        return new Complex(magnitude * Math.cos(imaginary), magnitude * Math.sin(imaginary));
    }

    private static Complex finiteLog(double real, double imag) {
        return new Complex(Math.log(finiteMagnitude(real, imag)), Math.atan2(imag, real));
    }

    private static double finiteMagnitude(double real, double imag) {
        double scale = Math.max(Math.abs(real), Math.abs(imag));
        if (scale == 0.0) {
            return 0.0;
        }
        double scaledReal = real / scale;
        double scaledImag = imag / scale;
        return scale * Math.sqrt(scaledReal * scaledReal + scaledImag * scaledImag);
    }

    private static Complex finiteSqrt(double real, double imag) {
        if (real == 0.0 && imag == 0.0) {
            return new Complex(0.0, imag);
        }
        double magnitude = finiteMagnitude(real, imag);
        if (real >= 0.0) {
            double realPart = Math.sqrt(0.5 * (magnitude + real));
            return new Complex(realPart, imag / (2.0 * realPart));
        }

        double imagPart = Math.sqrt(0.5 * (magnitude - real));
        double realPart = Math.abs(imag) / (2.0 * imagPart);
        return new Complex(realPart, Math.copySign(imagPart, imag));
    }

    private static Complex exceptionalSqrt(double real, double imag) {
        if (real == 0.0 && imag == 0.0) {
            return new Complex(0.0, imag);
        }
        double magnitude = Math.hypot(real, imag);
        if (real >= 0.0) {
            double realPart = Math.sqrt(0.5 * (magnitude + real));
            return new Complex(realPart, imag / (2.0 * realPart));
        }

        double imagPart = Math.sqrt(0.5 * (magnitude - real));
        double realPart = Math.abs(imag) / (2.0 * imagPart);
        return new Complex(realPart, Math.copySign(imagPart, imag));
    }

    private static Complex finiteQuotient(double a, double b, double c, double d) {
        double scale = Math.max(Math.abs(c), Math.abs(d));
        int exponent = scalingExponent(scale);
        double scaledC = scaleCoordinate(c, exponent);
        double scaledD = scaleCoordinate(d, exponent);
        double denominator = scaledC * scaledC + scaledD * scaledD;
        double realPart = scaleResult((a * scaledC + b * scaledD) / denominator, exponent);
        double imagPart = scaleResult((b * scaledC - a * scaledD) / denominator, exponent);
        return new Complex(realPart, imagPart);
    }

    private static Complex exceptionalQuotient(double a, double b, double c, double d) {
        double scale = Math.max(Math.abs(c), Math.abs(d));
        int exponent = scalingExponent(scale);
        double scaledC = scaleCoordinate(c, exponent);
        double scaledD = scaleCoordinate(d, exponent);
        double denominator = scaledC * scaledC + scaledD * scaledD;
        double realPart = scaleResult((a * scaledC + b * scaledD) / denominator, exponent);
        double imagPart = scaleResult((b * scaledC - a * scaledD) / denominator, exponent);

        double originalC = c;
        double originalD = d;

        if (Double.isNaN(realPart) && Double.isNaN(imagPart)) {
            if (denominator == 0.0 && (!Double.isNaN(a) || !Double.isNaN(b))) {
                double infinity = Math.copySign(Double.POSITIVE_INFINITY, originalC);
                return new Complex(infinity * a, infinity * b);
            }

            if ((Double.isInfinite(a) || Double.isInfinite(b))
                && Double.isFinite(originalC)
                && Double.isFinite(originalD)) {
                double scaledA = Math.copySign(Double.isInfinite(a) ? 1.0 : 0.0, a);
                double scaledB = Math.copySign(Double.isInfinite(b) ? 1.0 : 0.0, b);
                return new Complex(
                    Double.POSITIVE_INFINITY * (scaledA * originalC + scaledB * originalD),
                    Double.POSITIVE_INFINITY * (scaledB * originalC - scaledA * originalD)
                );
            }

            if (Double.isInfinite(scale) && Double.isFinite(a) && Double.isFinite(b)) {
                double signC = Math.copySign(Double.isInfinite(originalC) ? 1.0 : 0.0, originalC);
                double signD = Math.copySign(Double.isInfinite(originalD) ? 1.0 : 0.0, originalD);
                return new Complex(
                    0.0 * (a * signC + b * signD),
                    0.0 * (b * signC - a * signD)
                );
            }
        }

        return new Complex(realPart, imagPart);
    }

    private static int scalingExponent(double scale) {
        if (!Double.isFinite(scale) || scale == 0.0) {
            return 0;
        }
        return Math.getExponent(scale);
    }

    private static double scaleCoordinate(double value, int exponent) {
        if (exponent == 0) {
            return value;
        }
        return Math.scalb(value, -exponent);
    }

    private static double scaleResult(double value, int exponent) {
        if (exponent == 0) {
            return value;
        }
        return Math.scalb(value, -exponent);
    }

    private static boolean hasNegativeSign(double value) {
        return (Double.doubleToRawLongBits(value) & Long.MIN_VALUE) != 0L;
    }
}
