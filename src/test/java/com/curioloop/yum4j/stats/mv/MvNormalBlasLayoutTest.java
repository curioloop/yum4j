package com.curioloop.yum4j.ssm.particle.dist;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Sanity test for the BLAS layout used in {@link MvNormalDistribution}.
 *
 * <p>Exercises the claim that a column-major {@code (d, N)} buffer with
 * stride {@code N} between dimensions is bit-identical in memory to a
 * row-major {@code (d, N)} matrix with leading dimension {@code N}, so
 * a row-major BLAS {@code dtrmm(Left, Lower, NoTrans, NonUnit, d, N, 1,
 * L, 0, d, X, 0, N)} computes {@code X := L · X} as a mathematician
 * would expect.
 *
 * <p>The test builds a known lower-triangular {@code L}, a hand-written
 * {@code Z} matrix in column-major layout, runs both the scalar kernel
 * and the BLAS kernel, and compares the results element-by-element.
 */
class MvNormalBlasLayoutTest {

    private static final double ABS_TOL = 1e-14;

    @Test
    void dtrmmAgreesWithScalarKernelOnKnownLowerTriangle() {
        int d = 4;
        int N = 5;

        // Lower-triangular L, row-major d × d. Distinct positive diagonals keep
        // the triangular solve well-conditioned.
        double[] L = {
            1.0,  0.0,  0.0,  0.0,
            0.5,  2.0,  0.0,  0.0,
            0.1, -0.3,  1.5,  0.0,
            0.2,  0.4, -0.1,  0.8,
        };

        // Z[j, n] = (j + 1) * 10 + (n + 1), column-major (d, N) so
        // Z[j * N + n] = 10*j + n + 11.
        double[] z0 = new double[d * N];
        for (int j = 0; j < d; j++) {
            for (int n = 0; n < N; n++) {
                z0[j * N + n] = 10.0 * j + n + 11.0;
            }
        }

        // BLAS path: apply L via dtrmm.
        double[] blasOut = z0.clone();
        MvNormalDistribution.applyLowerCholeskyBlas(L, d, blasOut, 0, N);

        // Verify against manual computation: out[j, n] = Σ_k L[j,k] * z[k, n]
        for (int j = 0; j < d; j++) {
            for (int n = 0; n < N; n++) {
                double expected = 0.0;
                for (int k = 0; k <= j; k++) {
                    expected += L[j * d + k] * z0[k * N + n];
                }
                assertThat(blasOut[j * N + n])
                    .as("BLAS L*Z at (j=%d, n=%d)", j, n)
                    .isCloseTo(expected, org.assertj.core.data.Offset.offset(ABS_TOL));
            }
        }
    }

    @Test
    void endToEndBlasWithMeanAddition() {
        // Verify that applyLowerCholeskyBlas + addMean produces correct results.
        int d = 4;
        int N = 3;
        double[] L = {
            1.2,  0.0,  0.0,  0.0,
            0.3,  0.9,  0.0,  0.0,
            -0.1, 0.4,  1.1,  0.0,
            0.2,  -0.5, 0.2,  0.7,
        };
        double[] mean = {0.1, -0.2, 0.3, -0.4};

        double[] z = {
            // j = 0 row: 0.1, 0.2, 0.3
            0.1, 0.2, 0.3,
            // j = 1 row: -0.1, 0.5, -0.4
            -0.1, 0.5, -0.4,
            // j = 2 row: 0.7, -0.2, 0.1
            0.7, -0.2, 0.1,
            // j = 3 row: 0.0, 0.3, -0.6
            0.0, 0.3, -0.6,
        };

        // Compute expected: L * z + mean
        double[] expected = new double[d * N];
        for (int j = 0; j < d; j++) {
            for (int n = 0; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k <= j; k++) {
                    acc += L[j * d + k] * z[k * N + n];
                }
                expected[j * N + n] = acc + mean[j];
            }
        }

        double[] blasOut = z.clone();
        MvNormalDistribution.applyLowerCholeskyBlas(L, d, blasOut, 0, N);
        MvNormalDistribution.addMean(mean, d, blasOut, 0, N);

        for (int i = 0; i < d * N; i++) {
            assertThat(blasOut[i]).isCloseTo(expected[i], org.assertj.core.data.Offset.offset(ABS_TOL));
        }
    }
}
