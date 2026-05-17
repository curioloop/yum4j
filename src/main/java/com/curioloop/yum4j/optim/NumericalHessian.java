package com.curioloop.yum4j.optim;

import java.util.Objects;

/**
 * Numerical Hessian approximation methods for scalar functions R^n -> R.
 *
 * <p>Analogous to {@link NumericalGradient}, this enum provides finite-difference
 * methods for approximating the Hessian matrix H where
 * H[i,j] = d2f / dx_i dx_j, when analytical second derivatives are not available.</p>
 *
 * <h2>Hessian storage format</h2>
 * <p>Hessians are stored row-major:</p>
 * <pre>
 *   hessian[i*n + j] = d2f / dx_i dx_j
 * </pre>
 * <p>The matrix is symmetric for smooth scalar objectives. Implementations compute the
 * upper triangle and mirror it into the lower triangle before returning.</p>
 *
 * <h2>Workspace usage</h2>
 * <p>The output {@code hessian} array is also used as temporary workspace while the
 * finite-difference stencil is evaluated. No separate workspace array is required.</p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * double[] hessian = new double[n * n];
 * NumericalHessian.CENTRAL.compute(objective, point, n, hessian);
 * }</pre>
 *
 * @see Univariate
 * @see NumericalGradient
 * @see NumericalJacobian
 */
public enum NumericalHessian {

    /**
     * Forward finite-difference Hessian.
     *
     * <p>Uses forward perturbations around {@code x}. Requires one base evaluation,
     * one single-step evaluation per dimension, and one paired evaluation per upper
     * triangle entry. This is the cheapest method but has the largest truncation error.</p>
     */
    FORWARD(3) {
        @Override
        void computeCore(Univariate.Objective func, double[] x, int n, double[] hessian, double defaultFactor) {
            double f0 = func.evaluate(x, n);
            int diagonalStride = n + 1;
            for (int i = 0, diag = 0; i < n; i++, diag += diagonalStride) {
                double stepI = step(x, i, defaultFactor);
                hessian[diag] = evaluateShift(func, x, n, i, stepI, i, 0.0);
            }

            for (int i = 0, row = 0, diag = 0; i < n; i++, row += n, diag += diagonalStride) {
                double stepI = step(x, i, defaultFactor);
                double invStepI = 1.0 / stepI;
                double forwardI = hessian[diag];
                for (int j = i + 1, diagJ = diag + diagonalStride; j < n; j++, diagJ += diagonalStride) {
                    double stepJ = step(x, j, defaultFactor);
                    double value = (evaluateShift(func, x, n, i, stepI, j, stepJ)
                            - forwardI - hessian[diagJ] + f0) * invStepI / stepJ;
                    hessian[row + j] = value;
                }
                double diagonal = (evaluateShift(func, x, n, i, stepI, i, stepI)
                        - 2.0 * forwardI + f0) * invStepI * invStepI;
                hessian[diag] = diagonal;
            }
            mirrorUpperTriangle(hessian, n);
        }
    },

    /**
     * Average of forward and backward finite-difference Hessians.
     *
     * <p>Combines matching forward and backward stencils. It costs more function
     * evaluations than {@link #FORWARD}, but cancels part of the one-sided error.</p>
     */
    AVERAGE(3) {
        @Override
        void computeCore(Univariate.Objective func, double[] x, int n, double[] hessian, double defaultFactor) {
            if (n == 1) {
                computeOneDimensional(func, x, n, hessian, defaultFactor);
                return;
            }
            if (n == 2) {
                computeTwoDimensional(func, x, n, hessian, defaultFactor);
                return;
            }

            double f0 = func.evaluate(x, n);
            int diagonalStride = n + 1;
            for (int i = 0, diag = 0; i < n; i++, diag += diagonalStride) {
                double stepI = step(x, i, defaultFactor);
                hessian[diag] = evaluateShift(func, x, n, i, stepI, i, 0.0);
                hessian[backwardSlot(i, n)] = evaluateShift(func, x, n, i, -stepI, i, 0.0);
            }

            for (int i = 0, row = 0, diag = 0; i < n; i++, row += n, diag += diagonalStride) {
                double stepI = step(x, i, defaultFactor);
                double invStepI = 1.0 / stepI;
                double forwardI = hessian[diag];
                double backwardI = hessian[backwardSlot(i, n)];
                for (int j = i + 1, diagJ = diag + diagonalStride; j < n; j++, diagJ += diagonalStride) {
                    double stepJ = step(x, j, defaultFactor);
                    double numerator = evaluateShift(func, x, n, i, stepI, j, stepJ)
                            - forwardI - hessian[diagJ] + f0
                            + evaluateShift(func, x, n, i, -stepI, j, -stepJ)
                            - backwardI - hessian[backwardSlot(j, n)] + f0;
                    hessian[row + j] = 0.5 * numerator * invStepI / stepJ;
                }
                double diagonalNumerator = evaluateShift(func, x, n, i, stepI, i, stepI)
                        - 2.0 * forwardI + f0
                        + evaluateShift(func, x, n, i, -stepI, i, -stepI)
                        - 2.0 * backwardI + f0;
                hessian[diag] = 0.5 * diagonalNumerator * invStepI * invStepI;
            }
            mirrorUpperTriangle(hessian, n);
        }
    },

    /**
     * Centered finite-difference Hessian.
     *
     * <p>Uses a four-point centered stencil for each upper-triangle entry. This is the
     * most accurate method here and the usual choice when function evaluations are not
     * prohibitively expensive.</p>
     */
    CENTRAL(4) {
        @Override
        void computeCore(Univariate.Objective func, double[] x, int n, double[] hessian, double defaultFactor) {
            for (int i = 0, row = 0; i < n; i++, row += n) {
                double stepI = step(x, i, defaultFactor);
                double invStepI = 1.0 / stepI;
                for (int j = i; j < n; j++) {
                    double stepJ = step(x, j, defaultFactor);
                    double numerator = evaluateShift(func, x, n, i, stepI, j, stepJ)
                            - evaluateShift(func, x, n, i, stepI, j, -stepJ)
                            - evaluateShift(func, x, n, i, -stepI, j, stepJ)
                            + evaluateShift(func, x, n, i, -stepI, j, -stepJ);
                    hessian[row + j] = 0.25 * numerator * invStepI / stepJ;
                }
            }
            mirrorUpperTriangle(hessian, n);
        }
    };

    /** Machine epsilon used by the default finite-difference step-size rule. */
    public static final double EPSILON = Math.ulp(1.0);

    private final double defaultFactor;

    NumericalHessian(int scale) {
        this.defaultFactor = Math.pow(Math.ulp(1.0), 1.0 / scale);
    }

    /**
     * Computes a row-major Hessian in {@code hessian} using the default step-size rule.
     *
     * <p>The mutable input buffer {@code x} is restored before this method returns.</p>
     *
     * @param func objective function
     * @param x mutable evaluation point
     * @param n active dimension
     * @param hessian output row-major matrix with length at least {@code n * n}
     */
    public void compute(Univariate.Objective func, double[] x, int n, double[] hessian) {
        validateInputs(func, x, n, hessian);
        computeCore(func, x, n, hessian, defaultFactor);
    }

    abstract void computeCore(Univariate.Objective func, double[] x, int n, double[] hessian, double defaultFactor);

    private static double step(double[] x, int index, double defaultFactor) {
        return defaultFactor * Math.max(Math.abs(x[index]), 0.1);
    }

    private static void computeOneDimensional(Univariate.Objective func, double[] x, int n, double[] hessian,
                                              double defaultFactor) {
        double f0 = func.evaluate(x, n);
        double step0 = step(x, 0, defaultFactor);
        double invStep0 = 1.0 / step0;
        double forward = evaluateShift(func, x, n, 0, step0, 0, 0.0);
        double backward = evaluateShift(func, x, n, 0, -step0, 0, 0.0);
        double numerator = evaluateShift(func, x, n, 0, step0, 0, step0)
                - 2.0 * forward + f0
                + evaluateShift(func, x, n, 0, -step0, 0, -step0)
                - 2.0 * backward + f0;
        hessian[0] = 0.5 * numerator * invStep0 * invStep0;
    }

    private static void computeTwoDimensional(Univariate.Objective func, double[] x, int n, double[] hessian,
                                              double defaultFactor) {
        double f0 = func.evaluate(x, n);
        double step0 = step(x, 0, defaultFactor);
        double step1 = step(x, 1, defaultFactor);
        double invStep0 = 1.0 / step0;
        double invStep1 = 1.0 / step1;
        double forward0 = evaluateShift(func, x, n, 0, step0, 0, 0.0);
        double forward1 = evaluateShift(func, x, n, 1, step1, 1, 0.0);
        double backward0 = evaluateShift(func, x, n, 0, -step0, 0, 0.0);
        double backward1 = evaluateShift(func, x, n, 1, -step1, 1, 0.0);

        hessian[1] = (evaluateShift(func, x, n, 0, step0, 1, step1)
                - forward0 - forward1 + f0
                + evaluateShift(func, x, n, 0, -step0, 1, -step1)
                - backward0 - backward1 + f0) * 0.5 * invStep0 * invStep1;
        hessian[0] = (evaluateShift(func, x, n, 0, step0, 0, step0)
                - 2.0 * forward0 + f0
                + evaluateShift(func, x, n, 0, -step0, 0, -step0)
                - 2.0 * backward0 + f0) * 0.5 * invStep0 * invStep0;
        hessian[3] = (evaluateShift(func, x, n, 1, step1, 1, step1)
                - 2.0 * forward1 + f0
                + evaluateShift(func, x, n, 1, -step1, 1, -step1)
                - 2.0 * backward1 + f0) * 0.5 * invStep1 * invStep1;
        hessian[2] = hessian[1];
    }

    private static int backwardSlot(int index, int n) {
        if (index < n - 1) {
            return (n - 1) * n + index;
        }
        return (n - 2) * n;
    }

    private static void mirrorUpperTriangle(double[] hessian, int n) {
        for (int row = 0, rowOffset = 0; row < n - 1; row++, rowOffset += n) {
            for (int col = row + 1, colOffset = col * n; col < n; col++, colOffset += n) {
                hessian[colOffset + row] = hessian[rowOffset + col];
            }
        }
    }

    private static double evaluateShift(Univariate.Objective func, double[] x, int n,
                                        int i, double iStep, int j, double jStep) {
        double xi = x[i];
        if (i == j) {
            x[i] = xi + iStep + jStep;
            try {
                return func.evaluate(x, n);
            } finally {
                x[i] = xi;
            }
        }

        double xj = x[j];
        x[i] = xi + iStep;
        x[j] = xj + jStep;
        try {
            return func.evaluate(x, n);
        } finally {
            x[i] = xi;
            x[j] = xj;
        }
    }

    private static void validateInputs(Univariate.Objective func, double[] x, int n, double[] hessian) {
        Objects.requireNonNull(func, "func");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(hessian, "hessian");
        validateDimension(x, n);
        if (hessian.length < matrixSize(n)) {
            throw new IllegalArgumentException("hessian output too small");
        }
    }

    private static void validateDimension(double[] x, int n) {
        validateNonNegativeDimension(n);
        if (n > x.length) {
            throw new IllegalArgumentException("n out of range");
        }
    }

    private static void validateNonNegativeDimension(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("n out of range");
        }
    }

    private static int matrixSize(int n) {
        long size = (long) n * n;
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("n is too large");
        }
        return (int) size;
    }
}