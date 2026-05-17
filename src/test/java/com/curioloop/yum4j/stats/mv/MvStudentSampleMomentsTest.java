package com.curioloop.yum4j.ssm.particle.dist;

import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Basic statistical sanity test for {@link MvStudentDistribution#sample}.
 *
 * <p>For {@code df > 2}, the covariance of a multivariate t is
 * {@code (df / (df - 2)) · cov}. We draw {@code N = 20000} samples and
 * verify the empirical mean and empirical covariance agree with the
 * theoretical values to within a reasonable Monte-Carlo tolerance.
 * The finite-df increases variance materially so the bounds here are
 * looser than the MvNormal version.
 */
class MvStudentSampleMomentsTest {

    @Test
    void sampleMomentsApproachTheoreticalValues() {
        int d = 3;
        int n = 20_000;
        double df = 5.0; // finite covariance
        double[] mean = {1.0, -0.5, 2.0};
        double[] cov = {
            2.0, 0.3, 0.1,
            0.3, 1.5, -0.2,
            0.1, -0.2, 1.8,
        };
        double covScale = df / (df - 2.0); // E[(X-μ)(X-μ)'] = covScale · cov

        MvStudentDistribution dist = MvStudentDistribution.of(mean, cov, df);
        RandomGenerator g = RandomGeneratorFactory.of("L64X128MixRandom").create(20250117L);

        double[] x = new double[d * n];
        dist.sample(g, n, x, 0, null);

        // Empirical mean.
        double[] mhat = new double[d];
        for (int j = 0; j < d; j++) {
            double s = 0.0;
            int rowOff = j * n;
            for (int i = 0; i < n; i++) s += x[rowOff + i];
            mhat[j] = s / n;
        }
        for (int j = 0; j < d; j++) {
            assertThat(Math.abs(mhat[j] - mean[j]))
                .as("mean coord %d", j)
                .isLessThanOrEqualTo(0.15);
        }

        // Empirical covariance (1/N form).
        double[] cHat = new double[d * d];
        for (int a = 0; a < d; a++) {
            for (int b = a; b < d; b++) {
                double s = 0.0;
                for (int i = 0; i < n; i++) {
                    s += (x[a * n + i] - mhat[a]) * (x[b * n + i] - mhat[b]);
                }
                double v = s / n;
                cHat[a * d + b] = v;
                cHat[b * d + a] = v;
            }
        }

        double fro2 = 0.0;
        for (int i = 0; i < d * d; i++) {
            double diff = cHat[i] - covScale * cov[i];
            fro2 += diff * diff;
        }
        double fro = Math.sqrt(fro2);
        assertThat(fro).as("empirical cov Frobenius deviation").isLessThan(1.0);
    }
}
