package com.curioloop.yum4j.ssm.particle.dist;

import com.curioloop.yum4j.math.Gamma;
import com.curioloop.yum4j.stats.StudentsTDistribution;
import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

/**
 * Mathematical correctness checks for {@link MvStudentDistribution}.
 *
 * <p>No external reference fixtures — relies on closed-form values at
 * the mode, agreement with the boost-verified
 * {@link StudentsTDistribution} in {@code d=1}, and Monte Carlo
 * sample-moment convergence (using {@code Cov[X] = (ν / (ν−2)) · Σ}
 * which holds for {@code ν > 2}).
 */
class MvStudentCorrectnessTest {

    private static final double TOL = 1e-12;

    /**
     * At {@code x = μ} the quadratic form vanishes so
     * {@code logPdf(μ) = logΓ((ν+d)/2) − logΓ(ν/2) − 0.5·d·log(νπ) − 0.5·log|Σ|}.
     */
    @Test
    void logPdfAtMeanMatchesClosedForm() {
        int d = 3;
        double df = 5.0;
        double[] mean = {0.5, -1.5, 2.0};
        double[] cov = {
            1.5, 0.3, 0.0,
            0.3, 1.0, 0.2,
            0.0, 0.2, 0.8
        };
        double logDet = logDeterminant(cov, d);
        double expected =
            Gamma.lgamma(0.5 * (df + d))
                - Gamma.lgamma(0.5 * df)
                - 0.5 * d * Math.log(df * Math.PI)
                - 0.5 * logDet;

        MvStudentDistribution dist = MvStudentDistribution.of(mean, cov, df);
        assertThat(dist.logPdf(mean)).isCloseTo(expected, offset(TOL));
    }

    /**
     * The 1-D multivariate-t collapses to a location-scale Student's t.
     * Since {@link StudentsTDistribution} is a standard (centered, unit
     * scale) t, the oracle here applies the location-scale change of
     * variables explicitly:
     * {@code logPdf(x; μ, σ², ν) = −log σ + standardT_ν.logPdf((x − μ) / σ)}.
     */
    @Test
    void logPdfD1MatchesUnivariateStudentsT() {
        double mu = 1.5;
        double sigma2 = 2.25;
        double sigma = Math.sqrt(sigma2);
        double df = 7.0;

        MvStudentDistribution mv = MvStudentDistribution.of(
            new double[]{mu}, new double[]{sigma2}, df);
        StudentsTDistribution standard = new StudentsTDistribution(df);

        double[] points = {-2.0, -0.5, 0.0, 1.5, 3.7};
        for (double x : points) {
            double expected = -Math.log(sigma) + standard.logPdf((x - mu) / sigma);
            assertThat(mv.logPdf(new double[]{x}))
                .as("logPdf at %s", x)
                .isCloseTo(expected, offset(TOL));
        }
    }

    /** Batch and scalar paths must agree for every particle. */
    @Test
    void batchMatchesScalarOnEveryParticle() {
        int d = 2;
        int N = 32;
        double df = 4.5;
        double[] mean = {0.0, 1.0};
        double[] cov = {
            1.0, 0.4,
            0.4, 1.5
        };
        MvStudentDistribution dist = MvStudentDistribution.of(mean, cov, df);

        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(20251203L);
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
     * For ν > 2, {@code E[X] = μ} and {@code Cov[X] = (ν/(ν−2)) · Σ}.
     * Checks both first and second moments converge at the Monte Carlo
     * rate.
     */
    @Test
    void sampleMomentsConvergeForFiniteCovariance() {
        int d = 2;
        int N = 30_000;
        double df = 8.0; // > 4 so cov of sample cov is finite-ish
        double[] mean = {1.0, -1.0};
        double[] cov = {
            1.5, 0.3,
            0.3, 1.0
        };
        MvStudentDistribution dist = MvStudentDistribution.of(mean, cov, df);
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(20251204L);
        double[] X = new double[d * N];
        dist.sample(g, N, X, 0, null);

        // sample mean
        double[] sampleMean = new double[d];
        for (int j = 0; j < d; j++) {
            double s = 0.0;
            for (int n = 0; n < N; n++) s += X[j * N + n];
            sampleMean[j] = s / N;
        }
        double covScale = df / (df - 2.0);
        for (int j = 0; j < d; j++) {
            // SE ~ sqrt((covScale·Σ_jj) / N); 5 σ tolerance.
            double tol = 5.0 * Math.sqrt(covScale * cov[j * d + j] / N);
            assertThat(sampleMean[j])
                .as("sampleMean[%d]", j)
                .isCloseTo(mean[j], offset(tol));
        }

        // sample covariance, diagonal entry
        double s00 = 0.0;
        for (int n = 0; n < N; n++) {
            double dx = X[n] - sampleMean[0];
            s00 += dx * dx;
        }
        double sampleCov00 = s00 / (N - 1);
        double expectedCov00 = covScale * cov[0];
        double tolCov = 0.10 * expectedCov00; // 10% loose bound; sample-cov SE for t-distributions widens with smaller df
        assertThat(sampleCov00)
            .as("sampleCov[0,0]")
            .isCloseTo(expectedCov00, offset(tolCov));
    }

    // ────────────────────────────────────────────────────────────────────

    /** Computes {@code log|A|} via a fresh Cholesky for the test oracle. */
    private static double logDeterminant(double[] sigma, int d) {
        double[] L = sigma.clone();
        for (int j = 0; j < d; j++) {
            double diag = L[j * d + j];
            for (int k = 0; k < j; k++) diag -= L[j * d + k] * L[j * d + k];
            if (!(diag > 0.0)) throw new IllegalArgumentException("matrix not PD");
            L[j * d + j] = Math.sqrt(diag);
            double Ljj = L[j * d + j];
            for (int i = j + 1; i < d; i++) {
                double s = L[i * d + j];
                for (int k = 0; k < j; k++) s -= L[i * d + k] * L[j * d + k];
                L[i * d + j] = s / Ljj;
            }
            for (int i = 0; i < j; i++) L[i * d + j] = 0.0;
        }
        double s = 0.0;
        for (int j = 0; j < d; j++) s += Math.log(L[j * d + j]);
        return 2.0 * s;
    }
}
