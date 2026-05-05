/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.kalman.mle;

import com.curioloop.yum4j.optim.Univariate;

import java.util.Arrays;
import java.util.Objects;

/**
 * Numerical Hessian computation methods for approximating second derivatives when
 * analytical Hessians are unavailable.
 */
final class NumericalHessian {

    /** Machine epsilon (ε ≈ 2.22e-16). */
    static final double EPSILON = Math.ulp(1.0);

    /** Default step size for second-order central differences (ε^(1/4) ≈ 1.2e-4). */
    static final double FOURTH_ROOT_EPSILON = Math.pow(EPSILON, 0.25);

    private NumericalHessian() {
    }

    /**
     * Computes a row-major Hessian approximation using second-order central differences.
     *
     * @param func objective function
     * @param point evaluation point
     * @return row-major Hessian matrix with shape {@code point.length x point.length}
     */
    static double[] central(Univariate.Objective func, double[] point) {
        Objects.requireNonNull(point, "point");
        double[] x = Arrays.copyOf(point, point.length);
        double[] hessian = new double[x.length * x.length];
        central(func, x, x.length, hessian);
        return hessian;
    }

    /**
     * Computes a row-major Hessian approximation using second-order central differences.
     *
     * <p>The input vector {@code x} is restored to its original values before returning.</p>
     *
     * @param func objective function
     * @param x mutable evaluation point buffer
     * @param n active dimension
     * @param hessian output row-major Hessian matrix with length at least {@code n * n}
     */
    static void central(Univariate.Objective func, double[] x, int n, double[] hessian) {
        Objects.requireNonNull(func, "func");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(hessian, "hessian");
        if (n < 0 || n > x.length) {
            throw new IllegalArgumentException("n out of range");
        }
        if (hessian.length < n * n) {
            throw new IllegalArgumentException("hessian output too small");
        }
        if (n == 0) {
            return;
        }

        double f0 = func.evaluate(x, n);
        Arrays.fill(hessian, 0, n * n, 0.0);

        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double hi = stepSize(xi);

            x[i] = xi + hi;
            double fPlus = func.evaluate(x, n);

            x[i] = xi - hi;
            double fMinus = func.evaluate(x, n);

            x[i] = xi;
            hessian[i * n + i] = (fPlus - 2.0 * f0 + fMinus) / (hi * hi);

            for (int j = i + 1; j < n; j++) {
                double xj = x[j];
                double hj = stepSize(xj);

                x[i] = xi + hi;
                x[j] = xj + hj;
                double fPlusPlus = func.evaluate(x, n);

                x[j] = xj - hj;
                double fPlusMinus = func.evaluate(x, n);

                x[i] = xi - hi;
                x[j] = xj + hj;
                double fMinusPlus = func.evaluate(x, n);

                x[j] = xj - hj;
                double fMinusMinus = func.evaluate(x, n);

                x[i] = xi;
                x[j] = xj;

                double mixed = (fPlusPlus - fPlusMinus - fMinusPlus + fMinusMinus) / (4.0 * hi * hj);
                hessian[i * n + j] = mixed;
                hessian[j * n + i] = mixed;
            }
        }
    }

    private static double stepSize(double value) {
        double step = FOURTH_ROOT_EPSILON * Math.max(1.0, Math.abs(value));
        double adjusted = (value + step) - value;
        return adjusted == 0.0 ? step : adjusted;
    }
}