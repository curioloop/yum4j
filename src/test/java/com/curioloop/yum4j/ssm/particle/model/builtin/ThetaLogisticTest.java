package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * Behavioural-invariant tests for {@link ThetaLogistic}.
 */
class ThetaLogisticTest {

    private static final double LOG_INV_SQRT_2PI = -0.5 * Math.log(2.0 * Math.PI);

    // ------------------------------------------------------------------
    // Constructor / defaults
    // ------------------------------------------------------------------

    @Test
    void defaultConstructor_usesDefaultParameters() {
        ThetaLogistic m = new ThetaLogistic();
        assertThat(m.r()).isCloseTo(0.3, within(1e-15));
        assertThat(m.K()).isCloseTo(1.0, within(1e-15));
        assertThat(m.theta()).isCloseTo(1.0, within(1e-15));
        assertThat(m.sigmaX()).isCloseTo(0.5, within(1e-15));
        assertThat(m.sigmaY()).isCloseTo(0.2, within(1e-15));
        assertThat(m.dim()).isEqualTo(1);
    }

    @Test
    void constructor_rejectsInvalidParameters() {
        assertThatThrownBy(() -> new ThetaLogistic(0.3, 0.0, 1.0, 0.5, 0.2))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("K");
        assertThatThrownBy(() -> new ThetaLogistic(0.3, 1.0, 1.0, 0.0, 0.2))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("sigmaX");
        assertThatThrownBy(() -> new ThetaLogistic(0.3, 1.0, 1.0, 0.5, -0.1))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("sigmaY");
    }

    // ------------------------------------------------------------------
    // sampleM0 moments
    // ------------------------------------------------------------------

    @Test
    void sampleM0_normalAroundLogK() {
        double r = 0.3, K = 2.0, theta = 1.0, sigmaX = 0.5, sigmaY = 0.2;
        ThetaLogistic m = new ThetaLogistic(r, K, theta, sigmaX, sigmaY);
        int N = 100_000;
        double[] X = new double[N];
        RandomBatch rng = RandomBatch.of(0x55L);
        StepContext<Double> ctx = new StepContext<>(0, 0.0, null, 0, X, 0,
                new double[N], 0, N, 1, rng, new double[N], 0);
        m.sampleM0(ctx);

        double sum = 0.0, sumSq = 0.0;
        double logK = Math.log(K);
        for (int n = 0; n < N; n++) {
            double d = X[n] - logK;
            sum += d;
            sumSq += d * d;
        }
        double mean = sum / N;
        double variance = sumSq / N - mean * mean;
        double se = sigmaX / Math.sqrt(N);
        assertThat(mean).isBetween(-4.0 * se, 4.0 * se);
        assertThat(variance).isCloseTo(sigmaX * sigmaX, within(0.05));
    }

    // ------------------------------------------------------------------
    // sampleM conditional mean
    // ------------------------------------------------------------------

    @Test
    void sampleM_conditionalMeanMatchesTransitionFormula() {
        double r = 0.3, K = 2.0, theta = 1.5, sigmaX = 0.5, sigmaY = 0.2;
        ThetaLogistic m = new ThetaLogistic(r, K, theta, sigmaX, sigmaY);
        int N = 100_000;
        double xp = 0.8;
        double[] Xprev = new double[N];
        java.util.Arrays.fill(Xprev, xp);
        double[] X = new double[N];
        RandomBatch rng = RandomBatch.of(0x66L);
        StepContext<Double> ctx = new StepContext<>(1, 0.0, Xprev, 0, X, 0,
                new double[N], 0, N, 1, rng, new double[N], 0);
        m.sampleM(ctx);

        double sum = 0.0;
        for (int n = 0; n < N; n++) sum += X[n];
        double mean = sum / N;

        double expected = xp + r * (1.0 - Math.pow(Math.exp(xp) / K, theta));
        double se = sigmaX / Math.sqrt(N);
        assertThat(mean).isCloseTo(expected, within(4.0 * se));
    }

    // ------------------------------------------------------------------
    // logG hand-check
    // ------------------------------------------------------------------

    @Test
    void logG_matchesGaussianLogPdf() {
        double r = 0.3, K = 2.0, theta = 1.0, sigmaX = 0.5, sigmaY = 0.2;
        ThetaLogistic m = new ThetaLogistic(r, K, theta, sigmaX, sigmaY);
        double[] X = {0.1, -0.5, 1.7};
        double[] logW = new double[3];
        double y = 0.3;
        StepContext<Double> ctx = new StepContext<>(4, y, null, 0, X, 0,
                logW, 0, 3, 1, RandomBatch.of(1L), new double[3], 0);
        m.logG(ctx);

        double logN = LOG_INV_SQRT_2PI - Math.log(sigmaY);
        for (int i = 0; i < 3; i++) {
            double z = (X[i] - y) / sigmaY;
            double expected = logN - 0.5 * z * z;
            assertThat(logW[i]).isCloseTo(expected, within(1e-12));
        }
    }

    // ------------------------------------------------------------------
    // logPt0 hand-check
    // ------------------------------------------------------------------

    @Test
    void logPt0_matchesGaussianAtLogK() {
        double r = 0.3, K = 2.0, theta = 1.0, sigmaX = 0.5, sigmaY = 0.2;
        ThetaLogistic m = new ThetaLogistic(r, K, theta, sigmaX, sigmaY);
        double logK = Math.log(K);
        double[] X = {0.0, 0.5, -1.5};
        double[] out = new double[3];
        StepContext<Double> ctx = new StepContext<>(0, 0.0, null, 0, X, 0,
                new double[3], 0, 3, 1, RandomBatch.of(1L), new double[3], 0);
        m.logPt0(ctx, out, 0);

        double logN = LOG_INV_SQRT_2PI - Math.log(sigmaX);
        for (int i = 0; i < 3; i++) {
            double z = (X[i] - logK) / sigmaX;
            double expected = logN - 0.5 * z * z;
            assertThat(out[i]).isCloseTo(expected, within(1e-12));
        }
    }

    // ------------------------------------------------------------------
    // logPt hand-check
    // ------------------------------------------------------------------

    @Test
    void logPt_matchesNonlinearTransitionDensity() {
        double r = 0.3, K = 2.0, theta = 1.5, sigmaX = 0.5, sigmaY = 0.2;
        ThetaLogistic m = new ThetaLogistic(r, K, theta, sigmaX, sigmaY);
        double[] Xprev = {0.2, -0.4};
        double[] X = {0.3, -0.3};
        double[] out = new double[2];
        StepContext<Double> ctx = new StepContext<>(5, 0.0, Xprev, 0, X, 0,
                new double[2], 0, 2, 1, RandomBatch.of(1L), new double[2], 0);
        m.logPt(ctx, out, 0);

        double logN = LOG_INV_SQRT_2PI - Math.log(sigmaX);
        for (int i = 0; i < 2; i++) {
            double xp = Xprev[i];
            double loc = xp + r * (1.0 - Math.pow(Math.exp(xp) / K, theta));
            double z = (X[i] - loc) / sigmaX;
            double expected = logN - 0.5 * z * z;
            assertThat(out[i]).isCloseTo(expected, within(1e-12));
        }
    }

    // ------------------------------------------------------------------
    // Proposal consistency
    // ------------------------------------------------------------------

    @Test
    void proposal0_sampleAndLogProposalAreConsistent() {
        double r = 0.3, K = 2.0, theta = 1.0, sigmaX = 0.5, sigmaY = 0.2;
        ThetaLogistic m = new ThetaLogistic(r, K, theta, sigmaX, sigmaY);
        int N = 64;
        double y0 = 0.25;
        double[] X = new double[N];
        RandomBatch rng = RandomBatch.of(17L);
        StepContext<Double> ctx = new StepContext<>(0, y0, null, 0, X, 0,
                new double[N], 0, N, 1, rng, new double[N], 0);
        m.sampleQ0(ctx);

        double[] logq = new double[N];
        StepContext<Double> ctx2 = new StepContext<>(0, y0, null, 0, X, 0,
                new double[N], 0, N, 1, RandomBatch.of(1L), new double[N], 0);
        m.logQ0(ctx2, logq, 0);

        double logK = Math.log(K);
        double sigmaXSq = sigmaX * sigmaX;
        double sigmaYSq = sigmaY * sigmaY;
        double varStar = 1.0 / (1.0 / sigmaXSq + 1.0 / sigmaYSq);
        double sigmaStar = Math.sqrt(varStar);
        double muStar = varStar * (logK / sigmaXSq + y0 / sigmaYSq);
        double logN = LOG_INV_SQRT_2PI - Math.log(sigmaStar);
        for (int i = 0; i < N; i++) {
            double z = (X[i] - muStar) / sigmaStar;
            double expected = logN - 0.5 * z * z;
            assertThat(logq[i]).isCloseTo(expected, within(1e-12));
        }
    }

    @Test
    void proposal_sampleAndLogProposalAreConsistent() {
        double r = 0.3, K = 2.0, theta = 1.5, sigmaX = 0.5, sigmaY = 0.2;
        ThetaLogistic m = new ThetaLogistic(r, K, theta, sigmaX, sigmaY);
        int N = 32;
        double yt = 0.75;
        double[] Xprev = new double[N];
        for (int n = 0; n < N; n++) Xprev[n] = -1.0 + 2.0 * n / (N - 1);
        double[] X = new double[N];
        RandomBatch rng = RandomBatch.of(19L);
        StepContext<Double> ctx = new StepContext<>(6, yt, Xprev, 0, X, 0,
                new double[N], 0, N, 1, rng, new double[N], 0);
        m.sampleQ(ctx);

        double[] logq = new double[N];
        StepContext<Double> ctx2 = new StepContext<>(6, yt, Xprev, 0, X, 0,
                new double[N], 0, N, 1, RandomBatch.of(1L), new double[N], 0);
        m.logQ(ctx2, logq, 0);

        double sigmaXSq = sigmaX * sigmaX;
        double sigmaYSq = sigmaY * sigmaY;
        double varStar = 1.0 / (1.0 / sigmaXSq + 1.0 / sigmaYSq);
        double sigmaStar = Math.sqrt(varStar);
        double logN = LOG_INV_SQRT_2PI - Math.log(sigmaStar);
        for (int i = 0; i < N; i++) {
            double xp = Xprev[i];
            double loc = xp + r * (1.0 - Math.pow(Math.exp(xp) / K, theta));
            double muStar = varStar * (loc / sigmaXSq + yt / sigmaYSq);
            double z = (X[i] - muStar) / sigmaStar;
            double expected = logN - 0.5 * z * z;
            assertThat(logq[i]).isCloseTo(expected, within(1e-12));
        }
    }

    // ------------------------------------------------------------------
    // Upper bound
    // ------------------------------------------------------------------

    @Test
    void upperBound_returnsGaussianNormaliser() {
        double r = 0.3, K = 2.0, theta = 1.0, sigmaX = 0.5, sigmaY = 0.2;
        ThetaLogistic m = new ThetaLogistic(r, K, theta, sigmaX, sigmaY);
        double expected = -Math.log(sigmaX) - 0.5 * Math.log(2.0 * Math.PI);
        assertThat(m.upperBound(0)).isCloseTo(expected, within(1e-12));
        assertThat(m.upperBound(3)).isCloseTo(expected, within(1e-12));
    }
}
