/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;

interface Zlacn2 {

    static double zlacn2(int n, double[] v, int vOff,
                         double[] x, int xOff,
                         int[] isgn, int isgnOff,
                         double est, int[] kase, int kaseOff,
                         int[] isave, int isaveOff) {
        final int itmax = 5;
        double safmin = BLAS.safmin();

        if (kase[kaseOff] == 0) {
            for (int i = 0; i < n; i++) {
                x[xOff + 2 * i] = 1.0 / n;
                x[xOff + 2 * i + 1] = 0;
            }
            kase[kaseOff] = 1;
            isave[isaveOff] = 1;
            return est;
        }

        switch (isave[isaveOff]) {
            case 1: {
                if (n == 1) {
                    v[vOff] = x[xOff];
                    v[vOff + 1] = x[xOff + 1];
                    est = Math.hypot(v[vOff], v[vOff + 1]);
                    kase[kaseOff] = 0;
                    return est;
                }
                est = ZLAS.dzasum(n, x, xOff, 1);
                for (int i = 0; i < n; i++) {
                    int idx = xOff + 2 * i;
                    double re = x[idx];
                    double im = x[idx + 1];
                    double absxi = Math.hypot(re, im);
                    if (absxi > safmin) {
                        x[idx] = re / absxi;
                        x[idx + 1] = im / absxi;
                    } else {
                        x[idx] = 1.0;
                        x[idx + 1] = 0;
                    }
                }
                kase[kaseOff] = 2;
                isave[isaveOff] = 2;
                return est;
            }
            case 2: {
                isave[isaveOff + 1] = ZLAS.izamax(n, x, xOff, 1);
                isave[isaveOff + 2] = 2;
                for (int i = 0; i < n; i++) {
                    x[xOff + 2 * i] = 0;
                    x[xOff + 2 * i + 1] = 0;
                }
                int j = isave[isaveOff + 1];
                x[xOff + 2 * j] = 1;
                x[xOff + 2 * j + 1] = 0;
                kase[kaseOff] = 1;
                isave[isaveOff] = 3;
                return est;
            }
            case 3: {
                ZLAS.zcopy(n, x, xOff, 1, v, vOff, 1);
                double estold = est;
                est = ZLAS.dzasum(n, v, vOff, 1);
                if (est <= estold) {
                    break;
                }
                for (int i = 0; i < n; i++) {
                    int idx = xOff + 2 * i;
                    double re = x[idx];
                    double im = x[idx + 1];
                    double absxi = Math.hypot(re, im);
                    if (absxi > safmin) {
                        x[idx] = re / absxi;
                        x[idx + 1] = im / absxi;
                    } else {
                        x[idx] = 1.0;
                        x[idx + 1] = 0;
                    }
                }
                kase[kaseOff] = 2;
                isave[isaveOff] = 4;
                return est;
            }
            case 4: {
                int jlast = isave[isaveOff + 1];
                isave[isaveOff + 1] = ZLAS.izamax(n, x, xOff, 1);
                int jnew = isave[isaveOff + 1];
                double absJlast = Math.hypot(x[xOff + 2 * jlast], x[xOff + 2 * jlast + 1]);
                double absJnew = Math.hypot(x[xOff + 2 * jnew], x[xOff + 2 * jnew + 1]);
                if (absJlast != absJnew && isave[isaveOff + 2] < itmax) {
                    isave[isaveOff + 2]++;
                    for (int i = 0; i < n; i++) {
                        x[xOff + 2 * i] = 0;
                        x[xOff + 2 * i + 1] = 0;
                    }
                    x[xOff + 2 * jnew] = 1;
                    x[xOff + 2 * jnew + 1] = 0;
                    kase[kaseOff] = 1;
                    isave[isaveOff] = 3;
                    return est;
                }
                break;
            }
            case 5: {
                double temp = 2.0 * ZLAS.dzasum(n, x, xOff, 1) / (3.0 * n);
                if (temp > est) {
                    ZLAS.zcopy(n, x, xOff, 1, v, vOff, 1);
                    est = temp;
                }
                kase[kaseOff] = 0;
                return est;
            }
        }

        double altsgn = 1.0;
        for (int i = 0; i < n; i++) {
            x[xOff + 2 * i] = altsgn * (1.0 + (double) i / (n - 1));
            x[xOff + 2 * i + 1] = 0;
            altsgn *= -1;
        }
        kase[kaseOff] = 1;
        isave[isaveOff] = 5;
        return est;
    }
}
