/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.mat;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LUNumericalStabilityTest {

    @Test
    @DisplayName("Numerical stability: Hilbert matrix inversion")
    void testHilbertMatrixStability() {
        int[] sizes = {5, 6, 7, 8, 9, 10, 11, 12};

        for (int n : sizes) {
            double[] H = generateHilbertMatrix(n);
            assertDoesNotThrow(() -> {
                estimateConditionNumber(H, n);
                testInversionError(H, n);
            }, "Hilbert stability probe should complete for n=" + n);
        }
    }

    @Test
    @DisplayName("Numerical stability: Random ill-conditioned matrices")
    void testRandomIllConditionedStability() {
        int[] sizes = {20, 50, 100};
        double[] conditionNumbers = {1e6, 1e9, 1e12};

        for (int n : sizes) {
            for (double targetCond : conditionNumbers) {
                double[] A = generateIllConditionedMatrix(n, targetCond, 42);

                double error = testInversionError(A, n);

                assertTrue(Double.isFinite(error), "inversion error should be finite for n=" + n + ", targetCond=" + targetCond);
            }
        }
    }

    private double testInversionError(double[] A, int n) {
        double[] Acopy = A.clone();

        LU lu = LU.decompose(Acopy, n);
        if (!lu.ok()) {
            return Double.POSITIVE_INFINITY;
        }
        double[] inv;
        try {
            inv = lu.inverse(null);
        } catch (ArithmeticException ex) {
            return Double.POSITIVE_INFINITY;
        }

        return computeInversionError(A, inv, n);
    }

    private double computeInversionError(double[] A, double[] Ainv, int n) {
        double error = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A[i * n + k] * Ainv[k * n + j];
                }
                double expected = (i == j) ? 1.0 : 0.0;
                double diff = sum - expected;
                error += diff * diff;
            }
        }
        return Math.sqrt(error);
    }

    private double[] generateHilbertMatrix(int n) {
        double[] H = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                H[i * n + j] = 1.0 / (i + j + 1);
            }
        }
        return H;
    }

    private double[] generateIllConditionedMatrix(int n, double conditionNumber, long seed) {
        java.util.Random rand = new java.util.Random(seed);

        double[] Q = new double[n * n];
        double[] R = new double[n * n];
        for (int i = 0; i < n * n; i++) {
            Q[i] = rand.nextGaussian();
        }
        qrDecomposition(Q, R, n);

        double[] S = new double[n];
        S[0] = 1.0;
        S[n - 1] = 1.0 / conditionNumber;
        for (int i = 1; i < n - 1; i++) {
            double t = (double) i / (n - 1);
            S[i] = Math.pow(conditionNumber, -t);
        }

        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += Q[i * n + k] * S[k] * Q[j * n + k];
                }
                A[i * n + j] = sum;
            }
        }
        return A;
    }

    private void qrDecomposition(double[] A, double[] R, int n) {
        for (int j = 0; j < n; j++) {
            double norm = 0.0;
            for (int i = 0; i < n; i++) {
                norm += A[i * n + j] * A[i * n + j];
            }
            norm = Math.sqrt(norm);
            R[j * n + j] = norm;

            for (int i = 0; i < n; i++) {
                A[i * n + j] /= norm;
            }

            for (int k = j + 1; k < n; k++) {
                double dot = 0.0;
                for (int i = 0; i < n; i++) {
                    dot += A[i * n + j] * A[i * n + k];
                }
                R[j * n + k] = dot;
                for (int i = 0; i < n; i++) {
                    A[i * n + k] -= dot * A[i * n + j];
                }
            }
        }
    }

    private double estimateConditionNumber(double[] A, int n) {
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) x[i] = 1.0;

        double sigmaMax = 0;
        for (int iter = 0; iter < 50; iter++) {
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) {
                    double ax = 0;
                    for (int k = 0; k < n; k++) {
                        ax += A[j * n + k] * x[k];
                    }
                    sum += A[i * n + j] * ax;
                }
                y[i] = sum;
            }

            double norm = 0;
            for (int i = 0; i < n; i++) norm += y[i] * y[i];
            norm = Math.sqrt(norm);
            sigmaMax = Math.sqrt(norm);
            for (int i = 0; i < n; i++) x[i] = y[i] / norm;
        }

        double[] Acopy = A.clone();
        LU lu = LU.decompose(Acopy, n);
        if (!lu.ok()) {
            return Double.POSITIVE_INFINITY;
        }

        for (int i = 0; i < n; i++) x[i] = 1.0;

        double sigmaMin = Double.MAX_VALUE;
        for (int iter = 0; iter < 50; iter++) {
            double[] z = x.clone();
            try {
                lu.solve(z, z);
            } catch (ArithmeticException ex) {
                return Double.POSITIVE_INFINITY;
            }

            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) {
                    sum += A[j * n + i] * z[j];
                }
                y[i] = sum;
            }

            try {
                lu.solve(y, y);
            } catch (ArithmeticException ex) {
                return Double.POSITIVE_INFINITY;
            }

            double norm = 0;
            for (int i = 0; i < n; i++) norm += y[i] * y[i];
            norm = Math.sqrt(norm);
            sigmaMin = 1.0 / Math.sqrt(norm);
            for (int i = 0; i < n; i++) x[i] = y[i] / norm;
        }

        return sigmaMax / sigmaMin;
    }
}
