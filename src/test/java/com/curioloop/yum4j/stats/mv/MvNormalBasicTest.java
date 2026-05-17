package com.curioloop.yum4j.ssm.particle.dist;

import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Construction, accessor, and zero-alloc invariants for
 * {@link MvNormalDistribution}.
 */
class MvNormalBasicTest {

    @Test
    void constructorRejectsNonPositiveDefiniteCovariance() {
        double[] mean = {0.0, 0.0};
        double[] cov = {1.0, 2.0, 2.0, 1.0}; // eigenvalues {3, -1}
        assertThatThrownBy(() -> MvNormalDistribution.of(mean, cov))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("not positive definite");
    }

    @Test
    void constructorRejectsMismatchedCovSize() {
        double[] mean = {0.0, 0.0};
        double[] cov = {1.0, 0.0, 0.0}; // wrong length
        assertThatThrownBy(() -> MvNormalDistribution.of(mean, cov))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("cov length");
    }

    @Test
    void lowerCholeskyIsTriangularWithPositiveDiagonal() {
        double[] mean = {0.0, 0.0, 0.0};
        double[] cov = {
            2.0, 0.3, 0.1,
            0.3, 1.5, -0.2,
            0.1, -0.2, 1.8,
        };
        MvNormalDistribution dist = MvNormalDistribution.of(mean, cov);
        double[] L = dist.lowerCholesky();
        int d = dist.dim();
        for (int j = 0; j < d; j++) {
            assertThat(L[j * d + j]).as("diag %d > 0", j).isPositive();
            for (int k = j + 1; k < d; k++) {
                assertThat(L[j * d + k]).as("upper (%d,%d) must be 0", j, k).isZero();
            }
        }
    }

    @Test
    void sampleWithCallerScratchUsesScratchBuffer() {
        double[] mean = {1.0, 2.0};
        double[] cov = {1.0, 0.0, 0.0, 1.0};
        MvNormalDistribution dist = MvNormalDistribution.of(mean, cov);

        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(42L);
        int n = 100;
        double[] scratch = new double[2 * n]; // d*N
        double[] x = new double[2 * n];
        dist.sample(g, n, x, 0, scratch);
        // sanity: results are finite and roughly centred around the mean
        for (int i = 0; i < 2 * n; i++) {
            assertThat(Double.isFinite(x[i])).isTrue();
        }
    }

    @Test
    void scalarAndBatchLogPdfAgree() {
        double[] mean = {1.0, -0.5, 2.0};
        double[] cov = {
            2.0, 0.3, 0.1,
            0.3, 1.5, -0.2,
            0.1, -0.2, 1.8,
        };
        MvNormalDistribution dist = MvNormalDistribution.of(mean, cov);
        double[] point = {0.5, -1.0, 1.5};

        double scalar = dist.logPdf(point);

        // Build a single-particle column-major buffer and run the batch path.
        double[] x = {0.5, -1.0, 1.5}; // d=3, N=1 so column-major is the same as row.
        double[] out = new double[1];
        dist.logPdfBatch(x, 0, 1, out, 0, null);

        assertThat(scalar).isCloseTo(out[0], org.assertj.core.data.Offset.offset(1e-14));
    }

    @Test
    void logPdfBatchZeroN_isNoop() {
        double[] mean = {0.0, 0.0};
        double[] cov = {1.0, 0.0, 0.0, 1.0};
        MvNormalDistribution dist = MvNormalDistribution.of(mean, cov);
        double[] out = {42.0, 43.0};
        dist.logPdfBatch(new double[0], 0, 0, out, 0, null);
        assertThat(out[0]).isEqualTo(42.0);
    }
}
