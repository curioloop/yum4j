/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

import static java.lang.Math.abs;

interface Zdrscl {

    static void zdrscl(int n, double a, double[] x, int xOff, int incX) {
        if (n <= 0) return;

        double cden = a;
        double cnum = 1.0;
        double smlnum = BLAS.dlamch('S');
        double bignum = 1.0 / smlnum;

        while (true) {
            double cden1 = cden * smlnum;
            double cnum1 = cnum / bignum;
            double mul;
            boolean done;

            if (cnum != 0 && abs(cden1) > abs(cnum)) {
                mul = smlnum;
                done = false;
                cden = cden1;
            } else if (abs(cnum1) > abs(cden)) {
                mul = bignum;
                done = false;
                cnum = cnum1;
            } else {
                mul = cnum / cden;
                done = true;
            }

            Zdscal.zdscal(n, mul, x, xOff, incX);
            if (done) break;
        }
    }
}
