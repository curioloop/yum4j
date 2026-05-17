package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * Unit tests for {@link MVStochVol} — multivariate stochastic volatility
 * with d independent log-volatility processes.
 */
class MVStochVolTest {

    private static final double LOG_2PI = Math.log(2.0 * Math.PI);

    /** Build a 3-d model for testing. */
    private static MVStochVol threeD() {
        double[] mu = {-1.0, 0.0, 0.5};
        double[] rho = {0.95, 0.9, 0.85};
        double[] sigma = {0.2, 0.15, 0.25};
        return new MVStochVol(mu, rho, sigma);
    }

    // ------------------------------------------------------------------
    // Constructor validation
    // ------------------------------------------------------------------

    @Test
    void constructor_rejectsNull() {
        double[] ok = {0.0};
        assertThatThrownBy(() -> new MVStochVol(null, ok, ok))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new MVStochVol(ok, null, ok))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new MVStochVol(ok, ok, null))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void constructor_rejectsDimensionMismatch() {
        double[] mu = {0.0, 0.0};
        double[] rho = {0.9};
        double[] sigma = {0.1, 0.2};
        assertThatThrownBy(() -> new MVStochVol(mu, rho, sigma))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("same length");
    }

    @Test
    void constructor_rejectsInvalidRho() {
        double[] mu = {0.0};
        double[] rho = {1.0}; // |rho| must be < 1
        double[] sigma = {0.1};
        assertThatThrownBy(() -> new MVStochVol(mu, rho, sigma))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("rho");
    }

    @Test
    void constructor_rejectsNonPositiveSigma() {
        double[] mu = {0.0};
        double[] rho = {0.9};
        double[] sigma = {0.0};
        assertThatThrownBy(() -> new MVStochVol(mu, rho, sigma))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("sigma");
    }

    @Test
    void dim_returnsCorrectValue() {
        MVStochVol m = threeD();
        assertThat(m.dim()).isEqualTo(3);
    }

    // ------------------------------------------------------------------
    // sampleM0 — marginal moments
    // ------------------------------------------------------------------

    @Test
    void sampleM0_marginalMomentsMatchMuAndSigma0() {
        MVStochVol m = threeD();
        int N = 50_000;
        int d = m.dim();
        double[] X = new double[N * d];
        double[] logW = new double[N];
        double[] scratch = new double[2 * N];

        RandomBatch rng = RandomBatch.of(0x1234L);
        StepContext<double[]> ctx = new StepContext<>(
                0, null, null, 0, X, 0, logW, 0, N, d, rng, scratch, 0);

        m.sampleM0(ctx);

        double[] mu = m.mu();
        double[] rho = m.rho();
        double[] sigma = m.sigma();

        for (int k = 0; k < d; k++) {
            double sigma0k = sigma[k] / Math.sqrt(1.0 - rho[k] * rho[k]);
            double sum = 0.0, sumSq = 0.0;
            for (int n = 0; n < N; n++) {
                double val = X[n * d + k];
                sum += val;
                sumSq += val * val;
            }
            double mean = sum / N;
            double var = sumSq / N - mean * mean;
            double se = sigma0k / Math.sqrt(N);

            assertThat(mean).isCloseTo(mu[k], within(5.0 * se));
            assertThat(var).isCloseTo(sigma0k * sigma0k, within(0.05 * sigma0k * sigma0k));
        }
    }

    // ------------------------------------------------------------------
    // sampleM — conditional mean
    // ------------------------------------------------------------------

    @Test
    void sampleM_conditionalMeanMatchesFormula() {
        MVStochVol m = threeD();
        int N = 50_000;
        int d = m.dim();
        double[] Xprev = new double[N * d];
        double[] X = new double[N * d];
        double[] logW = new double[N];
        double[] scratch = new double[2 * N];

        // Fix ancestor to a constant
        double[] xpVals = {0.5, -0.3, 1.0};
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < d; k++) {
                Xprev[n * d + k] = xpVals[k];
            }
        }

        RandomBatch rng = RandomBatch.of(0x5678L);
        StepContext<double[]> ctx = new StepContext<>(
                1, null, Xprev, 0, X, 0, logW, 0, N, d, rng, scratch, 0);

        m.sampleM(ctx);

        double[] mu = m.mu();
        double[] rho = m.rho();
        double[] sigma = m.sigma();

        for (int k = 0; k < d; k++) {
            // E[X_t[k] | x_prev] = (1-rho[k])*mu[k] + rho[k]*x_prev[k]
            //                     = mu[k] + rho[k]*(x_prev[k] - mu[k])
            double expectedMean = mu[k] + rho[k] * (xpVals[k] - mu[k]);
            double sum = 0.0;
            for (int n = 0; n < N; n++) {
                sum += X[n * d + k];
            }
            double mean = sum / N;
            double se = sigma[k] / Math.sqrt(N);
            assertThat(mean).isCloseTo(expectedMean, within(5.0 * se));
        }
    }

    // ------------------------------------------------------------------
    // logG — hand-check against closed-form
    // ------------------------------------------------------------------

    @Test
    void logG_matchesClosedForm() {
        MVStochVol m = threeD();
        int N = 4;
        int d = m.dim();
        double[] X = new double[N * d];
        double[] logW = new double[N];
        double[] scratch = new double[2 * N];

        // Set up known state values
        double[][] xVals = {
                {0.1, -0.2, 0.3},
                {-0.5, 0.4, -0.1},
                {0.0, 0.0, 0.0},
                {1.0, -1.0, 0.5}
        };
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < d; k++) {
                X[n * d + k] = xVals[n][k];
            }
        }

        double[] yt = {0.05, -0.1, 0.2};

        RandomBatch rng = RandomBatch.of(42L);
        StepContext<double[]> ctx = new StepContext<>(
                0, yt, null, 0, X, 0, logW, 0, N, d, rng, scratch, 0);

        m.logG(ctx);

        // Reference: logG[n] = sum_k [-0.5*log(2π) - 0.5*X[n,k] - 0.5*yt[k]²*exp(-X[n,k])]
        for (int n = 0; n < N; n++) {
            double expected = 0.0;
            for (int k = 0; k < d; k++) {
                double xnk = xVals[n][k];
                expected += -0.5 * LOG_2PI - 0.5 * xnk
                        - 0.5 * yt[k] * yt[k] * Math.exp(-xnk);
            }
            assertThat(logW[n]).isCloseTo(expected, within(1e-12));
        }
    }

    @Test
    void logG_singleDimensionMatchesUnivariateStochVol() {
        // A 1-d MVStochVol should produce the same logG as StochVol
        double mu = -1.02;
        double rho = 0.9702;
        double sigma = 0.178;
        MVStochVol mv = new MVStochVol(new double[]{mu}, new double[]{rho}, new double[]{sigma});
        StochVol sv = new StochVol(mu, rho, sigma);

        int N = 8;
        double[] X_mv = new double[N]; // dim=1, so layout is same
        double[] X_sv = new double[N];
        double[] logW_mv = new double[N];
        double[] logW_sv = new double[N];
        double[] scratch = new double[2 * N];

        // Same state values
        for (int n = 0; n < N; n++) {
            double val = -1.0 + 0.3 * n;
            X_mv[n] = val;
            X_sv[n] = val;
        }

        double y = 0.05;
        double[] yt = {y};

        RandomBatch rng1 = RandomBatch.of(99L);
        RandomBatch rng2 = RandomBatch.of(99L);

        StepContext<double[]> ctxMv = new StepContext<>(
                0, yt, null, 0, X_mv, 0, logW_mv, 0, N, 1, rng1, scratch, 0);
        StepContext<Double> ctxSv = new StepContext<>(
                0, y, null, 0, X_sv, 0, logW_sv, 0, N, 1, rng2, scratch, 0);

        mv.logG(ctxMv);
        sv.logG(ctxSv);

        for (int n = 0; n < N; n++) {
            assertThat(logW_mv[n]).isCloseTo(logW_sv[n], within(1e-14));
        }
    }
}
