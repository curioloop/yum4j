package com.curioloop.yum4j.ssm.particle.filter;

import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.engine.*;
import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.builtin.LinearGauss;
import com.curioloop.yum4j.ssm.particle.model.builtin.StochVol;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import com.curioloop.yum4j.ssm.particle.smooth.Full;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Bootstrap parity tests for LinearGauss and StochVol models.
 *
 * <p>Validates Requirements R13.1–R13.4:
 * <ul>
 *   <li><b>Self-consistency (determinism):</b> Same seed + same config = bitwise-identical
 *       logLt and ESS across two runs.</li>
 *   <li><b>Cross-scheme consistency:</b> Different schemes produce similar (but not identical)
 *       logLt estimates within Monte Carlo variance.</li>
 *   <li><b>History mode invariance:</b> FULL vs NONE produce identical filter output
 *       (logLt, ESS, X).</li>
 *   <li><b>Kalman cross-check (LinearGauss only):</b> Particle filter weighted mean tracks
 *       Kalman filter mean with RMSE &lt; 0.1 at N=5000.</li>
 * </ul>
 *
 * <p>Test grid: 5 seeds × {SYSTEMATIC, SSP} schemes × {NONE, FULL} history modes.
 * N=512, T=30 for speed (except Kalman cross-check: N=5000, T=100).
 */
class BootstrapFilterTest {

    // ── Test grid parameters ────────────────────────────────────────────

    private static final long[] SEEDS = {1L, 42L, 123L, 2024L, 0xCAFEL};
    private static final Scheme[] SCHEMES = {Scheme.SYSTEMATIC, Scheme.SSP};
    private static final HistoryMode[] HISTORY_MODES = {HistoryMode.NONE, HistoryMode.FULL};

    private static final int N = 512;
    private static final int T = 30;

    private static final double LOG_LT_RTOL = 1e-10;
    private static final double ESS_ATOL = 1e-12;

    // ── Models ──────────────────────────────────────────────────────────

    private static final LinearGauss LINEAR_GAUSS = new LinearGauss(0.9, 1.0, 0.5, Double.NaN);
    private static final StochVol STOCH_VOL = new StochVol();

    // ── Data generators ─────────────────────────────────────────────────

    private static RandomGenerator rng(long seed) {
        return RandomGeneratorFactory.of("L64X128MixRandom").create(seed);
    }

    private static List<Double> syntheticLinearGauss(int T, long seed) {
        RandomGenerator g = rng(seed);
        List<Double> data = new ArrayList<>(T);
        double x = LINEAR_GAUSS.sigma0() * g.nextGaussian();
        data.add(x + LINEAR_GAUSS.sigmaY() * g.nextGaussian());
        for (int t = 1; t < T; t++) {
            x = LINEAR_GAUSS.rho() * x + LINEAR_GAUSS.sigmaX() * g.nextGaussian();
            data.add(x + LINEAR_GAUSS.sigmaY() * g.nextGaussian());
        }
        return data;
    }

    private static List<Double> syntheticStochVol(int T, long seed) {
        RandomGenerator g = rng(seed);
        List<Double> data = new ArrayList<>(T);
        double x = STOCH_VOL.mu() + STOCH_VOL.sigma0() * g.nextGaussian();
        data.add(Math.exp(0.5 * x) * g.nextGaussian());
        for (int t = 1; t < T; t++) {
            x = (1.0 - STOCH_VOL.rho()) * STOCH_VOL.mu() + STOCH_VOL.rho() * x
                    + STOCH_VOL.sigma() * g.nextGaussian();
            data.add(Math.exp(0.5 * x) * g.nextGaussian());
        }
        return data;
    }

    // ── Method sources for parameterised tests ──────────────────────────

    static Stream<Arguments> linearGaussGrid() {
        List<Arguments> args = new ArrayList<>();
        for (long seed : SEEDS) {
            for (Scheme scheme : SCHEMES) {
                for (HistoryMode mode : HISTORY_MODES) {
                    args.add(Arguments.of(seed, scheme, mode));
                }
            }
        }
        return args.stream();
    }

    static Stream<Arguments> stochVolGrid() {
        return linearGaussGrid(); // same grid
    }

    static Stream<Arguments> crossSchemeLinearGauss() {
        List<Arguments> args = new ArrayList<>();
        for (long seed : SEEDS) {
            for (HistoryMode mode : HISTORY_MODES) {
                args.add(Arguments.of(seed, mode));
            }
        }
        return args.stream();
    }

    static Stream<Arguments> crossSchemeStochVol() {
        return crossSchemeLinearGauss();
    }

    static Stream<Arguments> historyModeLinearGauss() {
        List<Arguments> args = new ArrayList<>();
        for (long seed : SEEDS) {
            for (Scheme scheme : SCHEMES) {
                args.add(Arguments.of(seed, scheme));
            }
        }
        return args.stream();
    }

    static Stream<Arguments> historyModeStochVol() {
        return historyModeLinearGauss();
    }

    // ══════════════════════════════════════════════════════════════════════
    // 1. Self-consistency (determinism): same seed → bitwise-identical
    // ══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "LinearGauss determinism: seed={0}, scheme={1}, history={2}")
    @MethodSource("linearGaussGrid")
    void linearGauss_determinism(long seed, Scheme scheme, HistoryMode mode) {
        List<Double> data = syntheticLinearGauss(T, seed);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(LINEAR_GAUSS);

        RunResult r1 = runFilter(fk, N, T, seed, scheme, mode, data);
        RunResult r2 = runFilter(fk, N, T, seed, scheme, mode, data);

        assertDeterministic(r1, r2, "LinearGauss");
    }

    @ParameterizedTest(name = "StochVol determinism: seed={0}, scheme={1}, history={2}")
    @MethodSource("stochVolGrid")
    void stochVol_determinism(long seed, Scheme scheme, HistoryMode mode) {
        List<Double> data = syntheticStochVol(T, seed);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(STOCH_VOL);

        RunResult r1 = runFilter(fk, N, T, seed, scheme, mode, data);
        RunResult r2 = runFilter(fk, N, T, seed, scheme, mode, data);

        assertDeterministic(r1, r2, "StochVol");
    }

    private void assertDeterministic(RunResult r1, RunResult r2, String model) {
        for (int t = 0; t < T; t++) {
            // logLt: within 1e-10 relative tolerance
            double logLt1 = r1.logLtSeries[t];
            double logLt2 = r2.logLtSeries[t];
            if (logLt1 == 0.0) {
                assertThat(logLt2).as("%s logLt[%d]", model, t).isEqualTo(0.0);
            } else {
                double relErr = Math.abs((logLt2 - logLt1) / logLt1);
                assertThat(relErr).as("%s logLt[%d] relative error", model, t)
                        .isLessThanOrEqualTo(LOG_LT_RTOL);
            }

            // ESS: within 1e-12 absolute tolerance
            assertThat(r2.essSeries[t]).as("%s ESS[%d]", model, t)
                    .isCloseTo(r1.essSeries[t], within(ESS_ATOL));
        }

        // Final X buffer should be bitwise identical
        assertThat(r2.finalX).as("%s final X", model).containsExactly(r1.finalX);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 2. Cross-scheme consistency: SYSTEMATIC vs SSP produce similar logLt
    // ══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "LinearGauss cross-scheme: seed={0}, history={1}")
    @MethodSource("crossSchemeLinearGauss")
    void linearGauss_crossSchemeConsistency(long seed, HistoryMode mode) {
        List<Double> data = syntheticLinearGauss(T, seed);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(LINEAR_GAUSS);

        RunResult systematic = runFilter(fk, N, T, seed, Scheme.SYSTEMATIC, mode, data);
        RunResult ssp = runFilter(fk, N, T, seed, Scheme.SSP, mode, data);

        // Final logLt should be similar but not necessarily identical
        double logLtSys = systematic.logLtSeries[T - 1];
        double logLtSsp = ssp.logLtSeries[T - 1];

        // Both should be finite
        assertThat(Double.isFinite(logLtSys)).as("SYSTEMATIC logLt finite").isTrue();
        assertThat(Double.isFinite(logLtSsp)).as("SSP logLt finite").isTrue();

        // Within Monte Carlo variance: |diff| < 5.0 (generous bound for N=512, T=30)
        assertThat(Math.abs(logLtSys - logLtSsp))
                .as("cross-scheme logLt difference (seed=%d)", seed)
                .isLessThan(5.0);
    }

    @ParameterizedTest(name = "StochVol cross-scheme: seed={0}, history={1}")
    @MethodSource("crossSchemeStochVol")
    void stochVol_crossSchemeConsistency(long seed, HistoryMode mode) {
        List<Double> data = syntheticStochVol(T, seed);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(STOCH_VOL);

        RunResult systematic = runFilter(fk, N, T, seed, Scheme.SYSTEMATIC, mode, data);
        RunResult ssp = runFilter(fk, N, T, seed, Scheme.SSP, mode, data);

        double logLtSys = systematic.logLtSeries[T - 1];
        double logLtSsp = ssp.logLtSeries[T - 1];

        assertThat(Double.isFinite(logLtSys)).as("SYSTEMATIC logLt finite").isTrue();
        assertThat(Double.isFinite(logLtSsp)).as("SSP logLt finite").isTrue();

        // StochVol has higher variance; use a generous bound
        assertThat(Math.abs(logLtSys - logLtSsp))
                .as("cross-scheme logLt difference (seed=%d)", seed)
                .isLessThan(10.0);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 3. History mode invariance: FULL vs NONE → identical filter output
    // ══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "LinearGauss history invariance: seed={0}, scheme={1}")
    @MethodSource("historyModeLinearGauss")
    void linearGauss_historyModeInvariance(long seed, Scheme scheme) {
        List<Double> data = syntheticLinearGauss(T, seed);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(LINEAR_GAUSS);

        RunResult none = runFilter(fk, N, T, seed, scheme, HistoryMode.NONE, data);
        RunResult full = runFilter(fk, N, T, seed, scheme, HistoryMode.FULL, data);

        for (int t = 0; t < T; t++) {
            assertThat(full.logLtSeries[t]).as("logLt[%d]", t)
                    .isEqualTo(none.logLtSeries[t]);
            assertThat(full.essSeries[t]).as("ESS[%d]", t)
                    .isEqualTo(none.essSeries[t]);
        }
        assertThat(full.finalX).as("final X").containsExactly(none.finalX);
        assertThat(full.finalLogW).as("final logW").containsExactly(none.finalLogW);
    }

    @ParameterizedTest(name = "StochVol history invariance: seed={0}, scheme={1}")
    @MethodSource("historyModeStochVol")
    void stochVol_historyModeInvariance(long seed, Scheme scheme) {
        List<Double> data = syntheticStochVol(T, seed);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(STOCH_VOL);

        RunResult none = runFilter(fk, N, T, seed, scheme, HistoryMode.NONE, data);
        RunResult full = runFilter(fk, N, T, seed, scheme, HistoryMode.FULL, data);

        for (int t = 0; t < T; t++) {
            assertThat(full.logLtSeries[t]).as("logLt[%d]", t)
                    .isEqualTo(none.logLtSeries[t]);
            assertThat(full.essSeries[t]).as("ESS[%d]", t)
                    .isEqualTo(none.essSeries[t]);
        }
        assertThat(full.finalX).as("final X").containsExactly(none.finalX);
        assertThat(full.finalLogW).as("final logW").containsExactly(none.finalLogW);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 4. Kalman cross-check (LinearGauss only)
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @Tag("slow")
    void linearGauss_kalmanCrossCheck() {
        int kalmanN = 5000;
        int kalmanT = 100;
        long seed = 2024L;

        double rho = LINEAR_GAUSS.rho();
        double sigmaX = LINEAR_GAUSS.sigmaX();
        double sigmaY = LINEAR_GAUSS.sigmaY();
        double sigma0 = LINEAR_GAUSS.sigma0();

        List<Double> data = syntheticLinearGauss(kalmanT, seed);

        // Kalman filter
        double[] kalmanMean = new double[kalmanT];
        {
            double mPrev = 0.0, vPrev = sigma0 * sigma0;
            for (int t = 0; t < kalmanT; t++) {
                double mPred = (t == 0) ? 0.0 : rho * mPrev;
                double vPred = (t == 0) ? sigma0 * sigma0 : rho * rho * vPrev + sigmaX * sigmaX;
                double S = vPred + sigmaY * sigmaY;
                double K = vPred / S;
                double y = data.get(t);
                double m = mPred + K * (y - mPred);
                double v = (1.0 - K) * vPred;
                kalmanMean[t] = m;
                mPrev = m;
                vPrev = v;
            }
        }

        // Particle filter with FULL history
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(LINEAR_GAUSS);
        Workspace ws = Workspace.allocate(kalmanN, fk.dim(), HistoryMode.FULL, kalmanT);
        // Override the default-rate Full (0.6) with rate=1.0 for the
        // long-horizon Kalman cross-check; the noisy LinearGauss process
        // resamples more often than the conservative default permits.
        ws.history = new Full(kalmanN, fk.dim(), kalmanT, 1.0);
        ws.rng = RandomBatch.of(seed);
        RunState rs = RunState.allocate(kalmanT, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), seed);
        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);

        // Compute RMSE of weighted mean vs Kalman mean
        double sse = 0.0;
        int kalmanDim = fk.dim();
        double[] Xt = new double[kalmanDim * kalmanN];
        double[] lwt = new double[kalmanN];
        for (int t = 0; t < kalmanT; t++) {
            ws.history.viewX(t, Xt, 0);
            ws.history.viewLogW(t, lwt, 0);
            double maxLw = Double.NEGATIVE_INFINITY;
            for (int n = 0; n < kalmanN; n++) {
                if (lwt[n] > maxLw) maxLw = lwt[n];
            }
            double sumW = 0.0, sumWX = 0.0;
            for (int n = 0; n < kalmanN; n++) {
                double w = Math.exp(lwt[n] - maxLw);
                sumW += w;
                sumWX += w * Xt[n];
            }
            double pfMean = sumWX / sumW;
            double err = pfMean - kalmanMean[t];
            sse += err * err;
        }
        double rmse = Math.sqrt(sse / kalmanT);

        assertThat(rmse).as("RMSE PF vs Kalman (N=%d, T=%d)", kalmanN, kalmanT)
                .isLessThan(0.1);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Helper: run a filter and capture results
    // ══════════════════════════════════════════════════════════════════════

    private record RunResult(
            double[] logLtSeries,
            double[] essSeries,
            boolean[] resampledFlags,
            double[] finalX,
            double[] finalLogW
    ) {}

    private static RunResult runFilter(FeynmanKac<Double> fk, int N, int T,
                                       long seed, Scheme scheme, HistoryMode mode,
                                       List<Double> data) {
        int histArg = (mode == HistoryMode.FULL) ? T : 0;
        Workspace ws = Workspace.allocate(N, fk.dim(), mode, histArg);
        // High-noise LinearGauss / StochVol parity tests resample more often
        // than the conservative default maxResampleRate of 0.6 — replace the
        // workspace's default-rate Full with a rate=1.0 variant so the
        // ancestor slot pool can hold every step's resample permutation.
        if (mode == HistoryMode.FULL) {
            ws.history = new Full(N, fk.dim(), T, 1.0);
        }
        ws.rng = RandomBatch.of(seed);
        RunState rs = RunState.allocate(T, 0.5, scheme, Collections.emptyList(), seed);

        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);

        return new RunResult(
                rs.logLtSeries.clone(),
                rs.essSeries.clone(),
                rs.resampledFlags.clone(),
                ws.X.clone(),
                ws.logW.clone()
        );
    }
}
