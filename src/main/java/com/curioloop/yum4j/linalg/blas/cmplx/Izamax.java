package com.curioloop.yum4j.linalg.blas.cmplx;

interface Izamax {

    public static int izamax(int n, double[] x, int xOff, int incX) {
        if (n < 1 || incX <= 0) return -1;
        if (n == 1) return 0;

        int idmax = 0;
        if (incX == 1) {
            double dmax = x[xOff] * x[xOff] + x[xOff + 1] * x[xOff + 1];
            int i = 1;
            int limit = n - 3;
            for (; i < limit; i += 4) {
                int p1 = xOff + i * 2;
                int p2 = p1 + 2;
                int p3 = p2 + 2;
                int p4 = p3 + 2;
                double a1 = x[p1] * x[p1] + x[p1 + 1] * x[p1 + 1];
                double a2 = x[p2] * x[p2] + x[p2 + 1] * x[p2 + 1];
                double a3 = x[p3] * x[p3] + x[p3 + 1] * x[p3 + 1];
                double a4 = x[p4] * x[p4] + x[p4 + 1] * x[p4 + 1];
                if (a1 > dmax) { dmax = a1; idmax = i; }
                if (a2 > dmax) { dmax = a2; idmax = i + 1; }
                if (a3 > dmax) { dmax = a3; idmax = i + 2; }
                if (a4 > dmax) { dmax = a4; idmax = i + 3; }
            }
            for (; i < n; i++) {
                int p = xOff + i * 2;
                double a = x[p] * x[p] + x[p + 1] * x[p + 1];
                if (a > dmax) { dmax = a; idmax = i; }
            }
        } else {
            int ix = xOff;
            double dmax = x[ix] * x[ix] + x[ix + 1] * x[ix + 1];
            ix += incX * 2;
            for (int i = 1; i < n; i++) {
                double a = x[ix] * x[ix] + x[ix + 1] * x[ix + 1];
                if (a > dmax) { dmax = a; idmax = i; }
                ix += incX * 2;
            }
        }
        return idmax;
    }

    public static double izamaxAbs(int n, double[] x, int xOff, int incX) {
        int ix = izamax(n, x, xOff, incX);
        if (ix < 0) return 0.0;
        int p = xOff + ix * incX * 2;
        return Math.hypot(x[p], x[p + 1]);
    }

    public static int izamax(int n, double[] x) {
        return izamax(n, x, 0, 1);
    }
}
