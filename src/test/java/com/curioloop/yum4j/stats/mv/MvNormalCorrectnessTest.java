package com.curioloop.yum4j.ssm.particle.dist;

import com.curioloop.yum4j.stats.NormalDistribution;
import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

/**
 * Mathematical correctness checks for {@link MvNormalDistribution}.
 *
 * <p>No external reference fixtures: every assertion is grounded in
 * either a closed-form value, a translation/scale invariant, or
 * agreement with the already-tested univariate
 * {@link NormalDistribution} (which has its own boost parity test
 * upstream).
 */
class MvNormalCorrectnessTest {

    private static final double TOL = 1e-12;
    private static final double LOG_2PI = Math.log(2.0 * Math.PI);

    /**
     * At {@code x = μ}, the quadratic form is zero, so
     * {@code logPdf(μ) = -0.5 · (d·log(2π) + log|Σ|)}. The 3×3 case
     * uses a non-trivial PD matrix; we compare against {@code log|Σ|}
     * computed from a fresh Cholesky.
     */
    @Test
    void logPdfAtMeanMatchesClosedForm() {
        int d = 3;
        double[] mean = {1.0, -2.0, 0.5};
        double[] cov = {
            2.0, 0.5, 0.1,
            0.5, 1.5, 0.2,
            0.1, 0.2, 1.0
        };
        double logDet = logDeterminant(cov, d);
        double expected = -0.5 * (d * LOG_2PI + logDet);

        MvNormalDistribution dist = MvNormalDistribution.of(mean, cov);
        assertThat(dist.logPdf(mean)).isCloseTo(expected, offset(TOL));
    }

    /**
     * For diagonal covariance the joint logPdf equals the sum of the
     * marginal univariate Normal logPdfs. This propagates the upstream
     * boost-verified univariate {@link NormalDistribution} correctness
     * to the multivariate case.
     */
    @Test
    void logPdfDiagonalCovEqualsSumOfMarginalNormals() {
        int d = 4;
        double[] mean = {0.0, 1.0, -1.0, 2.5};
        double[] sigmas = {0.5, 1.0, 2.0, 0.7};
        double[] cov = new double[d * d];
        for (int j = 0; j < d; j++) cov[j * d + j] = sigmas[j] * sigmas[j];

        MvNormalDistribution dist = MvNormalDistribution.of(mean, cov);

        double[] x = {0.3, 0.7, -0.2, 2.0};
        double expected = 0.0;
        for (int j = 0; j < d; j++) {
            expected += new NormalDistribution(mean[j], sigmas[j]).logPdf(x[j]);
        }
        assertThat(dist.logPdf(x)).isCloseTo(expected, offset(TOL));
    }

    /**
     * Translation invariance: shifting both {@code x} and {@code μ}
     * by the same vector leaves logPdf unchanged.
     */
    @Test
    void logPdfIsTranslationInvariant() {
        int d = 3;
        double[] mean = {1.0, -2.0, 0.5};
        double[] cov = {
            2.0, 0.5, 0.1,
            0.5, 1.5, 0.2,
            0.1, 0.2, 1.0
        };
        double[] x = {2.0, -1.0, 0.0};
        double[] shift = {10.0, -3.5, 7.2};

        MvNormalDistribution base = MvNormalDistribution.of(mean, cov);
        double[] shiftedMean = new double[d];
        double[] shiftedX = new double[d];
        for (int j = 0; j < d; j++) {
            shiftedMean[j] = mean[j] + shift[j];
            shiftedX[j] = x[j] + shift[j];
        }
        MvNormalDistribution shifted = MvNormalDistribution.of(shiftedMean, cov);

        assertThat(shifted.logPdf(shiftedX))
            .isCloseTo(base.logPdf(x), offset(TOL));
    }

    /**
     * Batch and scalar paths must agree for every particle.
     */
    @Test
    void batchMatchesScalarOnEveryParticle() {
        int d = 3;
        int N = 64;
        double[] mean = {0.5, -1.0, 2.0};
        double[] cov = {
            1.0, 0.3, 0.0,
            0.3, 1.0, 0.1,
            0.0, 0.1, 1.5
        };
        MvNormalDistribution dist = MvNormalDistribution.of(mean, cov);

        // Generate column-major (d, N) test points.
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(20251201L);
        double[] X = new double[d * N];
        dist.sample(g, N, X, 0, null);

        double[] batchOut = new double[N];
        dist.logPdfBatch(X, 0, N, batchOut, 0, null);

        double[] point = new double[d];
        for (int n = 0; n < N; n++) {
            for (int j = 0; j < d; j++) {
                point[j] = X[j * N + n];
            }
            assertThat(dist.logPdf(point))
                .as("scalar logPdf at particle %d", n)
                .isCloseTo(batchOut[n], offset(TOL));
        }
    }

    /**
     * Sample mean and covariance converge to {@code (μ, Σ)} at the
     * Monte Carlo rate. The bound is generous to absorb realistic
     * sampling fluctuations at N=20000 while still detecting algorithm
     * regressions (e.g. wrong cholesky direction).
     */
    @Test
    void sampleMomentsConverge() {
        int d = 3;
        int N = 20_000;
        double[] mean = {1.0, -1.0, 0.5};
        double[] cov = {
            2.0, 0.5, 0.0,
            0.5, 1.5, 0.2,
            0.0, 0.2, 1.0
        };
        MvNormalDistribution dist = MvNormalDistribution.of(mean, cov);
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(20251202L);
        double[] X = new double[d * N];
        dist.sample(g, N, X, 0, null);

        double[] sampleMean = new double[d];
        for (int j = 0; j < d; j++) {
            double s = 0.0;
            for (int n = 0; n < N; n++) s += X[j * N + n];
            sampleMean[j] = s / N;
        }
        for (int j = 0; j < d; j++) {
            // Standard error ~ sqrt(Σ_jj / N); 4 σ tolerance.
            double tol = 4.0 * Math.sqrt(cov[j * d + j] / N);
            assertThat(sampleMean[j])
                .as("sampleMean[%d]", j)
                .isCloseTo(mean[j], offset(tol));
        }

        // Off-diagonal sample covariance for entry (0, 1).
        double s01 = 0.0;
        for (int n = 0; n < N; n++) {
            s01 += (X[0 * N + n] - sampleMean[0]) * (X[1 * N + n] - sampleMean[1]);
        }
        double sampleCov01 = s01 / (N - 1);
        // Standard error of cov estimator ~ sqrt((Σ_00·Σ_11 + Σ_01²) / N).
        double s00 = cov[0 * d + 0], s11 = cov[1 * d + 1], s01true = cov[0 * d + 1];
        double seCov01 = Math.sqrt((s00 * s11 + s01true * s01true) / N);
        assertThat(sampleCov01)
            .as("sampleCov[0,1]")
            .isCloseTo(s01true, offset(4.0 * seCov01));
    }

    // ────────────────────────────────────────────────────────────────────

    /** Computes {@code log|A|} via a fresh Cholesky for the test oracle. */
    private static double logDeterminant(double[] sigma, int d) {
        double[] L = sigma.clone();
        cholesky(L, d);
        double s = 0.0;
        for (int j = 0; j < d; j++) s += Math.log(L[j * d + j]);
        return 2.0 * s;
    }

    /** In-place lower-triangular Cholesky used only as a test oracle. */
    private static void cholesky(double[] A, int d) {
        for (int j = 0; j < d; j++) {
            double diag = A[j * d + j];
            for (int k = 0; k < j; k++) {
                diag -= A[j * d + k] * A[j * d + k];
            }
            if (!(diag > 0.0)) {
                throw new IllegalArgumentException("matrix not PD");
            }
            A[j * d + j] = Math.sqrt(diag);
            double Ljj = A[j * d + j];
            for (int i = j + 1; i < d; i++) {
                double s = A[i * d + j];
                for (int k = 0; k < j; k++) {
                    s -= A[i * d + k] * A[j * d + k];
                }
                A[i * d + j] = s / Ljj;
            }
            // zero strict upper triangle
            for (int i = 0; i < j; i++) A[i * d + j] = 0.0;
        }
    }
}
