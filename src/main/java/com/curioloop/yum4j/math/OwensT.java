package com.curioloop.yum4j.math;

/**
 * Owen's T function aligned with the Boost.Math double-precision surface.
 *
 * <p>The public entry point mirrors {@code boost::math::owens_t(h, a)} on the
 * real scalar domain and keeps the same argument transforms, symmetry rules,
 * and T1-T6 region dispatch from the Patefield-Tandy algorithm used by Boost.
 */
public final class OwensT {

    private static final double HALF = 0.5;
    private static final double ONE_DIV_TWO_PI = 0.15915494309189533577;
    private static final double ONE_DIV_ROOT_TWO = 0.70710678118654752440;
    private static final double ONE_DIV_ROOT_TWO_PI = 0.39894228040143267794;
    private static final double A_TRANSFORM_THRESHOLD = 0.67;

    private static final double[] H_RANGE = {
        0.02, 0.06, 0.09, 0.125, 0.26, 0.4, 0.6,
        1.6, 1.7, 2.33, 2.4, 3.36, 3.4, 4.8
    };

    private static final double[] A_RANGE = {0.025, 0.09, 0.15, 0.36, 0.5, 0.9, 0.99999};

    private static final int[] SELECT = {
        0, 0, 1, 12, 12, 12, 12, 12, 12, 12, 12, 15, 15, 15, 8,
        0, 1, 1, 2, 2, 4, 4, 13, 13, 14, 14, 15, 15, 15, 8,
        1, 1, 2, 2, 2, 4, 4, 14, 14, 14, 14, 15, 15, 15, 9,
        1, 1, 2, 4, 4, 4, 4, 6, 6, 15, 15, 15, 15, 15, 9,
        1, 2, 2, 4, 4, 5, 5, 7, 7, 16, 16, 16, 11, 11, 10,
        1, 2, 4, 4, 4, 5, 5, 7, 7, 16, 16, 16, 11, 11, 11,
        1, 2, 3, 3, 5, 5, 7, 7, 16, 16, 16, 16, 16, 11, 11,
        1, 2, 3, 3, 5, 5, 17, 17, 17, 17, 16, 16, 16, 11, 11
    };

    private static final int[] METHOD = {
        1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2,
        3,
        4, 4, 4, 4,
        5,
        6
    };

    private static final int[] ORDER = {
        2, 3, 4, 5, 7, 10, 12, 18, 10, 20, 30, 0, 4, 7, 8, 20, 0, 0
    };

    private static final double[] T3_COEFFICIENTS = {
        0.99999999999999987510,
        -0.99999999999988796462,
        0.99999999998290743652,
        -0.99999999896282500134,
        0.99999996660459362918,
        -0.99999933986272476760,
        0.99999125611136965852,
        -0.99991777624463387686,
        0.99942835555870132569,
        -0.99697311720723000295,
        0.98751448037275303682,
        -0.95915857980572882813,
        0.89246305511006708555,
        -0.76893425990463999675,
        0.58893528468484693250,
        -0.38380345160440256652,
        0.20317601701045299653,
        -0.082813631607004984866,
        0.024167984735759576523,
        -0.0044676566663971825242,
        0.00039141169402373836468
    };

    private static final double[] T5_POINTS = {
        0.0035082039676451715489,
        0.031279042338030753740,
        0.085266826283219451090,
        0.16245071730812277011,
        0.25851196049125434828,
        0.36807553840697533536,
        0.48501092905604697475,
        0.60277514152618576821,
        0.71477884217753226516,
        0.81475510988760098605,
        0.89711029755948965867,
        0.95723808085944261843,
        0.99178832974629703586
    };

    private static final double[] T5_WEIGHTS = {
        0.018831438115323502887,
        0.018567086243977649478,
        0.018042093461223385584,
        0.017263829606398753364,
        0.016243219975989856730,
        0.014994592034116704829,
        0.013535474469662088392,
        0.011886351605820165233,
        0.010070377242777431897,
        0.0081130545742299586629,
        0.0060419009528470238773,
        0.0038862217010742057883,
        0.0016793031084546090448
    };

    private OwensT() {
    }

    public static double value(double h, double a) {
        if (Double.isNaN(h) || Double.isNaN(a)) {
            return Double.NaN;
        }

        h = Math.abs(h);
        if (h == 0.0) {
            return Math.atan(a) * ONE_DIV_TWO_PI;
        }
        if (a == 0.0 || Double.isInfinite(h)) {
            return Math.copySign(0.0, a);
        }
        if (Double.isInfinite(a)) {
            return Math.copySign(HALF * Normal.ccdf(h), a);
        }

        double absA = Math.abs(a);
        double absAh = absA * h;
        double value;
        if (absA <= 1.0) {
            value = dispatch(h, absA, absAh);
        } else if (h <= A_TRANSFORM_THRESHOLD) {
            value = 0.25 - znorm1(h) * znorm1(absAh) - dispatch(absAh, 1.0 / absA, h);
        } else {
            double normH = Normal.ccdf(h);
            double normAh = Normal.ccdf(absAh);
            value = HALF * (normH + normAh) - normH * normAh - dispatch(absAh, 1.0 / absA, h);
        }
        return a < 0.0 ? -value : value;
    }

    public static double owensT(double h, double a) {
        return value(h, a);
    }

    private static double dispatch(double h, double a, double ah) {
        if (h == 0.0) {
            return Math.atan(a) * ONE_DIV_TWO_PI;
        }
        if (a == 0.0) {
            return 0.0;
        }
        if (a == 1.0) {
            return HALF * Normal.ccdf(-h) * Normal.ccdf(h);
        }

        int code = computeCode(h, a);
        int order = ORDER[code];
        return switch (METHOD[code]) {
            case 1 -> t1(h, a, order);
            case 2 -> t2(h, a, order, ah);
            case 3 -> t3(h, a, ah);
            case 4 -> t4(h, a, order);
            case 5 -> t5(h, a);
            case 6 -> t6(h, a);
            default -> throw new IllegalStateException("Unsupported Owen's T dispatch method: " + METHOD[code]);
        };
    }

    private static int computeCode(double h, double a) {
        int hIndex = 14;
        int aIndex = 7;
        for (int index = 0; index < H_RANGE.length; index++) {
            if (h <= H_RANGE[index]) {
                hIndex = index;
                break;
            }
        }
        for (int index = 0; index < A_RANGE.length; index++) {
            if (a <= A_RANGE[index]) {
                aIndex = index;
                break;
            }
        }
        return SELECT[aIndex * 15 + hIndex];
    }

    private static double t1(double h, double a, int order) {
        double hs = -0.5 * h * h;
        double expHs = Math.exp(hs);
        double as = a * a;

        int j = 1;
        double jj = 1.0;
        double aj = a * ONE_DIV_TWO_PI;
        double dj = Math.expm1(hs);
        double gj = hs * expHs;
        double value = Math.atan(a) * ONE_DIV_TWO_PI;

        while (true) {
            value += dj * aj / jj;
            if (order <= j) {
                return value;
            }
            j++;
            jj += 2.0;
            aj *= as;
            dj = gj - dj;
            gj *= hs / j;
        }
    }

    private static double t2(double h, double a, int order, double ah) {
        int maxIndex = order + order + 1;
        double hs = h * h;
        double as = -a * a;
        double y = 1.0 / hs;

        int ii = 1;
        double value = 0.0;
        double vi = a * Math.exp(-0.5 * ah * ah) * ONE_DIV_ROOT_TWO_PI;
        double z = znorm1(ah) / h;

        while (true) {
            value += z;
            if (maxIndex <= ii) {
                return value * Math.exp(-0.5 * hs) * ONE_DIV_ROOT_TWO_PI;
            }
            z = y * (vi - ii * z);
            vi *= as;
            ii += 2;
        }
    }

    private static double t3(double h, double a, double ah) {
        double as = a * a;
        double hs = h * h;
        double y = 1.0 / hs;

        double ii = 1.0;
        int index = 0;
        double vi = a * Math.exp(-0.5 * ah * ah) * ONE_DIV_ROOT_TWO_PI;
        double zi = znorm1(ah) / h;
        double value = 0.0;

        while (true) {
            value += zi * T3_COEFFICIENTS[index];
            if (index >= 20) {
                return value * Math.exp(-0.5 * hs) * ONE_DIV_ROOT_TWO_PI;
            }
            zi = y * (ii * zi - vi);
            vi *= as;
            ii += 2.0;
            index++;
        }
    }

    private static double t4(double h, double a, int order) {
        int maxIndex = order + order + 1;
        double hs = h * h;
        double as = -a * a;

        int ii = 1;
        double ai = a * Math.exp(-0.5 * hs * (1.0 - as)) * ONE_DIV_TWO_PI;
        double yi = 1.0;
        double value = 0.0;

        while (true) {
            value += ai * yi;
            if (maxIndex <= ii) {
                return value;
            }
            ii += 2;
            yi = (1.0 - hs * yi) / ii;
            ai *= as;
        }
    }

    private static double t5(double h, double a) {
        double as = a * a;
        double hs = -0.5 * h * h;
        double value = 0.0;
        for (int index = 0; index < T5_POINTS.length; index++) {
            double r = 1.0 + as * T5_POINTS[index];
            value += T5_WEIGHTS[index] * Math.exp(hs * r) / r;
        }
        return value * a;
    }

    private static double t6(double h, double a) {
        double normH = Normal.ccdf(h);
        double y = 1.0 - a;
        double r = Math.atan2(y, 1.0 + a);

        double value = HALF * normH * (1.0 - normH);
        if (r != 0.0) {
            value -= r * Math.exp(-0.5 * y * h * h / r) * ONE_DIV_TWO_PI;
        }
        return value;
    }

    private static double znorm1(double x) {
        return HALF * Normal.erf(x * ONE_DIV_ROOT_TWO);
    }
}