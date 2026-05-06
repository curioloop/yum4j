package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.Commons.isInteger;
import static com.curioloop.yum4j.math.Commons.validateFinite;

/**
 * Boost-aligned Hankel functions with real scalar inputs and complex outputs.
 */
public final class Hankel {

    private static final double ROOT_HALF_PI = Math.sqrt(Math.PI / 2.0);

    private Hankel() {
    }

    /**
     * Returns the cylindrical Hankel function of the first kind {@code H^(1)_nu(x)}.
     */
    public static Complex cylHankel1(double nu, double x) {
        validateFinite(nu, "cylHankel1: order");
        validateFinite(x, "cylHankel1: x");
        return cylindricalHankel(nu, x, 1, "cylHankel1");
    }

    /**
     * Returns the cylindrical Hankel function of the second kind {@code H^(2)_nu(x)}.
     */
    public static Complex cylHankel2(double nu, double x) {
        validateFinite(nu, "cylHankel2: order");
        validateFinite(x, "cylHankel2: x");
        return cylindricalHankel(nu, x, -1, "cylHankel2");
    }

    /**
     * Returns the spherical Hankel function of the first kind {@code h^(1)_nu(x)}.
     */
    public static Complex sphHankel1(double nu, double x) {
        validateFinite(nu, "sphHankel1: order");
        validateFinite(x, "sphHankel1: x");
        return sphericalHankel(nu, x, 1, "sphHankel1");
    }

    /**
     * Returns the spherical Hankel function of the second kind {@code h^(2)_nu(x)}.
     */
    public static Complex sphHankel2(double nu, double x) {
        validateFinite(nu, "sphHankel2: order");
        validateFinite(x, "sphHankel2: x");
        return sphericalHankel(nu, x, -1, "sphHankel2");
    }

    static String debugCylHankel1Route(double nu, double x) {
        return classifyCylindricalRoute(nu, x);
    }

    static String debugCylHankel2Route(double nu, double x) {
        return classifyCylindricalRoute(nu, x);
    }

    static String debugSphHankel1Route(double nu, double x) {
        return "spherical-reduction/" + classifyCylindricalRoute(nu + 0.5, x);
    }

    static String debugSphHankel2Route(double nu, double x) {
        return "spherical-reduction/" + classifyCylindricalRoute(nu + 0.5, x);
    }

    private static Complex sphericalHankel(double nu, double x, int sign, String functionName) {
        if (x == 0.0) {
            throw new ArithmeticException(functionName + ": singular at x=0");
        }

        Complex z = new Complex(x, 0.0);
        return cylindricalHankel(nu + 0.5, x, sign, functionName)
            .mul(ROOT_HALF_PI)
            .div(z.sqrt());
    }

    private static Complex cylindricalHankel(double nu, double x, int sign, String functionName) {
        if (x == 0.0) {
            throw new ArithmeticException(functionName + ": singular at x=0");
        }
        if (x > 0.0) {
            Double2 jy = Bessel.cylJY(nu, x);
            return new Complex(jy._1(), sign * jy._2());
        }

        double reflectedX = -x;
        Double2 jy = Bessel.cylJY(nu, reflectedX);
        double j = jy._1();
        double y = jy._2();
        Complex cx = new Complex(x, 0.0);
        if (isInteger(nu)) {
            return negativeIntegerOrderContinuation(nu, j, y, cx, sign);
        }
        return negativeNonIntegerOrderContinuation(nu, j, y, reflectedX, cx, sign);
    }

    private static Complex negativeIntegerOrderContinuation(double nu, double j, double y, Complex cx, int sign) {
        long integerOrder = Math.round(Math.rint(nu));
        int paritySign = (integerOrder & 1L) == 0L ? 1 : -1;
        Complex logDifference = new Complex(Math.log(-cx.real()), 0.0).sub(cx.log());
        Complex yContinuation = new Complex(y, 0.0)
            .sub(logDifference.mul((2.0 / Math.PI) * j))
            .mul(paritySign);
        Complex jContinuation = new Complex(paritySign * j, 0.0);
        return jContinuation.add(multiplyByImaginarySign(yContinuation, sign));
    }

    private static Complex negativeNonIntegerOrderContinuation(
        double nu,
        double j,
        double y,
        double reflectedX,
        Complex cx,
        int sign
    ) {
        Complex p2 = cx.pow(nu);
        double p1 = Math.pow(reflectedX, nu);
        Complex p1Complex = new Complex(p1, 0.0);
        Complex yContinuation = p1Complex.mul(y).div(p2)
            .add(
                p2.div(p1)
                    .sub(p1Complex.div(p2))
                    .mul(j / Math.tan(Math.PI * nu))
            );
        Complex jContinuation = p2.mul(Math.pow(reflectedX, -nu) * j);
        return jContinuation.add(multiplyByImaginarySign(yContinuation, sign));
    }

    private static Complex multiplyByImaginarySign(Complex value, int sign) {
        return new Complex(-sign * value.imag(), sign * value.real());
    }

    private static String classifyCylindricalRoute(double nu, double x) {
        if (!Double.isFinite(nu) || !Double.isFinite(x)) {
            return "domain-error";
        }
        if (x == 0.0) {
            return "origin";
        }
        if (x > 0.0) {
            return "positive-x";
        }
        return isInteger(nu) ? "negative-x-integer-order" : "negative-x-non-integer-order";
    }
}