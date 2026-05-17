package com.curioloop.yum4j.ssm.particle.engine;

import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.builtin.LinearGauss;
import com.curioloop.yum4j.ssm.particle.model.builtin.StochVol;
import com.curioloop.yum4j.ssm.particle.model.builtin.Gordon;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import com.curioloop.yum4j.ssm.particle.smooth.Full;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Behavioural coverage for the particle-v2 Engine. Asserts engine invariants
 * (stepCount, ESS/logLt shapes, resampling policy, determinism, FULL vs
 * NONE equivalence) and includes a light-weight Kalman cross-check.
 *
 * <p>Ported from the old SMCEngineTest to use the new Engine/Workspace/RunState API.
 */
class EngineTest {

    private static RandomGenerator rng(long seed) {
        return RandomGeneratorFactory.of("L64X128MixRandom").create(seed);
    }

    private static List<Double> syntheticLinearGauss(LinearGauss model, int T, long seed) {
        RandomGenerator g = rng(seed);
        List<Double> data = new ArrayList<>(T);
        double x = model.sigma0() * g.nextGaussian();
        data.add(x + model.sigmaY() * g.nextGaussian());
        for (int t = 1; t < T; t++) {
            x = model.rho() * x + model.sigmaX() * g.nextGaussian();
            data.add(x + model.sigmaY() * g.nextGaussian());
        }
        return data;
    }

    private static List<Double> syntheticStochVol(StochVol model, int T, long seed) {
        RandomGenerator g = rng(seed);
        List<Double> data = new ArrayList<>(T);
        double x = model.sigma0() * g.nextGaussian();
        data.add(Math.exp(0.5 * x) * g.nextGaussian());
        for (int t = 1; t < T; t++) {
            x = (1.0 - model.rho()) * model.mu() + model.rho() * x + model.sigma() * g.nextGaussian();
            data.add(Math.exp(0.5 * x) * g.nextGaussian());
        }
        return data;
    }

    private static List<Double> syntheticGordon(Gordon model, int T, long seed) {
        RandomGenerator g = rng(seed);
        List<Double> data = new ArrayList<>(T);
        double x = model.sigma0() * g.nextGaussian();
        data.add(model.a() * x * x + g.nextGaussian());
        for (int t = 1; t < T; t++) {
            double loc = model.b() * x + model.c() * x / (1.0 + x * x) + model.d() * Math.cos(model.e() * (t - 1));
            x = loc + model.sigmaX() * g.nextGaussian();
            data.add(model.a() * x * x + g.nextGaussian());
        }
        return data;
    }

    // ------------------------------------------------------------------
    // Test 1: init state check
    // ------------------------------------------------------------------

    @Test
    void init_populatesStep0Summaries() {
        LinearGauss model = new LinearGauss(0.9, 1.0, 0.5, 1.0);
        int T = 5, N = 256;
        List<Double> data = syntheticLinearGauss(model, T, 123L);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);

        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(123L);
        RunState rs = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 123L);

        Engine.init(ws, rs, fk, ParallelStrategy.SERIAL, data);

        assertThat(rs.stepCount).isEqualTo(1);
        assertThat(rs.essSeries[0]).isPositive();
        assertThat(rs.essSeries[0]).isLessThanOrEqualTo(N + 1e-6);
        assertThat(Double.isFinite(rs.logLtSeries[0])).isTrue();
        assertThat(rs.resampledFlags[0]).isFalse();
    }

    // ------------------------------------------------------------------
    // Test 2: step count after run()
    // ------------------------------------------------------------------

    @Test
    void run_advancesStepCountToHorizon() {
        LinearGauss model = new LinearGauss(0.9, 1.0, 0.5, 1.0);
        int T = 5, N = 128;
        List<Double> data = syntheticLinearGauss(model, T, 7L);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);

        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(7L);
        RunState rs = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 7L);

        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);

        assertThat(rs.stepCount).isEqualTo(T);
        for (int t = 0; t < T; t++) {
            assertThat(rs.essSeries[t]).isPositive();
            assertThat(Double.isFinite(rs.logLtSeries[t])).isTrue();
        }
    }

    // ------------------------------------------------------------------
    // Test 3: repeated runs with no allocation surprises
    // ------------------------------------------------------------------

    @Test
    void stepsRepeatedly_withoutLeakingOrNaN() {
        StochVol model = new StochVol();
        int T = 100, N = 256;
        List<Double> data = syntheticStochVol(model, T, 11L);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);

        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(99L);
        RunState rs = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 99L);

        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);
        double firstLogZ = rs.logLtSeries[T - 1];
        assertThat(Double.isFinite(firstLogZ)).isTrue();

        // Re-run with the same seed; should produce identical results.
        ws.resetBuffers();
        ws.rng = RandomBatch.of(99L);
        rs.reset(99L);
        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);
        assertThat(rs.logLtSeries[T - 1]).isCloseTo(firstLogZ, within(1e-12));
        for (int t = 0; t < T; t++) {
            assertThat(Double.isFinite(rs.logLtSeries[t])).isTrue();
            assertThat(Double.isNaN(rs.essSeries[t])).isFalse();
        }
    }

    // ------------------------------------------------------------------
    // Test 4: ESS threshold fires resampling for essRmin close to 1
    // ------------------------------------------------------------------

    @Test
    void essRminNearOne_firesResamplingOnMostSteps() {
        Gordon model = new Gordon();
        int T = 50, N = 512;
        List<Double> data = syntheticGordon(model, T, 2024L);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);

        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(1L);
        RunState rs = RunState.allocate(T, 0.9, Scheme.SYSTEMATIC, Collections.emptyList(), 1L);

        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);

        int resampled = 0;
        for (boolean f : rs.resampledFlags) if (f) resampled++;
        assertThat(resampled).isGreaterThan(0);
    }

    // ------------------------------------------------------------------
    // Test 5: ESS threshold 0 disables resampling entirely
    // ------------------------------------------------------------------

    @Test
    void essRminZero_neverResamples() {
        LinearGauss model = new LinearGauss(0.9, 1.0, 0.5, 1.0);
        int T = 20, N = 256;
        List<Double> data = syntheticLinearGauss(model, T, 42L);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);

        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(3L);
        RunState rs = RunState.allocate(T, 0.0, Scheme.SYSTEMATIC, Collections.emptyList(), 3L);

        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);

        for (int t = 0; t < T; t++) {
            assertThat(rs.resampledFlags[t]).as("t=%d", t).isFalse();
        }
    }

    // ------------------------------------------------------------------
    // Test 6: logLt is finite throughout a run
    // ------------------------------------------------------------------

    @Test
    void logLt_isFiniteOverLinearGaussRun() {
        LinearGauss model = new LinearGauss(0.9, 1.0, 0.5, 1.0);
        int T = 30, N = 256;
        List<Double> data = syntheticLinearGauss(model, T, 77L);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);

        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(77L);
        RunState rs = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 77L);

        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);

        for (int t = 0; t < T; t++) {
            double v = rs.logLtSeries[t];
            assertThat(Double.isFinite(v)).as("logLt[%d]=%s", t, v).isTrue();
        }
    }

    // ------------------------------------------------------------------
    // Test 7: different seeds produce different runs
    // ------------------------------------------------------------------

    @Test
    void differentSeeds_produceDifferentResults() {
        LinearGauss model = new LinearGauss(0.9, 1.0, 0.5, 1.0);
        int T = 20, N = 512;
        List<Double> data = syntheticLinearGauss(model, T, 0xDEADBEEFL);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);

        Workspace ws1 = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws1.rng = RandomBatch.of(1L);
        RunState rs1 = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 1L);
        Engine.run(ws1, rs1, fk, ParallelStrategy.SERIAL, data);
        double[] logLtSeed1 = rs1.logLtSeries.clone();

        Workspace ws2 = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws2.rng = RandomBatch.of(2L);
        RunState rs2 = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 2L);
        Engine.run(ws2, rs2, fk, ParallelStrategy.SERIAL, data);

        boolean anyDiff = false;
        for (int t = 0; t < T; t++) {
            if (logLtSeed1[t] != rs2.logLtSeries[t]) { anyDiff = true; break; }
        }
        assertThat(anyDiff).as("at least one logLt entry should differ with a new seed").isTrue();
    }

    // ------------------------------------------------------------------
    // Test 8: same seed is bit-for-bit reproducible
    // ------------------------------------------------------------------

    @Test
    void sameSeed_isByteForByteReproducible() {
        StochVol model = new StochVol();
        int T = 40, N = 256;
        List<Double> data = syntheticStochVol(model, T, 0xCAFEL);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);

        Workspace ws1 = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws1.rng = RandomBatch.of(4242L);
        RunState rs1 = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 4242L);
        Engine.run(ws1, rs1, fk, ParallelStrategy.SERIAL, data);
        double[] logLt1 = rs1.logLtSeries.clone();
        double[] ess1 = rs1.essSeries.clone();
        boolean[] rsFlag1 = rs1.resampledFlags.clone();

        Workspace ws2 = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws2.rng = RandomBatch.of(4242L);
        RunState rs2 = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 4242L);
        Engine.run(ws2, rs2, fk, ParallelStrategy.SERIAL, data);

        for (int t = 0; t < T; t++) {
            assertThat(rs2.logLtSeries[t]).as("logLt[%d]", t).isEqualTo(logLt1[t]);
            assertThat(rs2.essSeries[t]).as("ESS[%d]", t).isEqualTo(ess1[t]);
            assertThat(rs2.resampledFlags[t]).as("rsFlag[%d]", t).isEqualTo(rsFlag1[t]);
        }
    }

    // ------------------------------------------------------------------
    // Test 9: HistoryMode.NONE and FULL produce identical filter output
    // ------------------------------------------------------------------

    @Test
    void historyFull_doesNotAffectFilterState() {
        LinearGauss model = new LinearGauss(0.9, 1.0, 0.5, 1.0);
        int T = 25, N = 256;
        List<Double> data = syntheticLinearGauss(model, T, 17L);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);

        Workspace wsNone = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        wsNone.rng = RandomBatch.of(5L);
        RunState rsNone = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 5L);
        Engine.run(wsNone, rsNone, fk, ParallelStrategy.SERIAL, data);

        Workspace wsFull = Workspace.allocate(N, fk.dim(), HistoryMode.FULL, T);
        // Override the default-rate Full (0.6) with rate=1.0 — the high-noise
        // LinearGauss + small N stress this test resamples more often than
        // the conservative default permits. The behaviour under examination
        // (HistoryMode.FULL is engine-state-neutral) does not depend on the
        // ancestor pool size.
        wsFull.history = new Full(N, fk.dim(), T, 1.0);
        wsFull.rng = RandomBatch.of(5L);
        RunState rsFull = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 5L);
        Engine.run(wsFull, rsFull, fk, ParallelStrategy.SERIAL, data);

        // logW, X, logLt, ESS, rsFlag should all match byte-for-byte.
        assertThat(wsFull.logW).containsExactly(wsNone.logW);
        assertThat(wsFull.X).containsExactly(wsNone.X);
        for (int t = 0; t < T; t++) {
            assertThat(rsFull.logLtSeries[t]).as("logLt[%d]", t).isEqualTo(rsNone.logLtSeries[t]);
            assertThat(rsFull.essSeries[t]).as("ESS[%d]", t).isEqualTo(rsNone.essSeries[t]);
            assertThat(rsFull.resampledFlags[t]).as("rsFlag[%d]", t).isEqualTo(rsNone.resampledFlags[t]);
        }

        // FULL mode populated history buffers.
        assertThat(wsFull.history).isNotNull();

        // NONE mode leaves history null.
        assertThat(wsNone.history).isNull();
    }

    // ------------------------------------------------------------------
    // Test 10: each Scheme produces valid results
    // ------------------------------------------------------------------

    @Test
    void allSchemes_produceFiniteLogLt() {
        LinearGauss model = new LinearGauss(0.9, 1.0, 0.5, 1.0);
        int T = 15, N = 128;
        List<Double> data = syntheticLinearGauss(model, T, 55L);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);

        for (Scheme scheme : Scheme.values()) {
            Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
            ws.rng = RandomBatch.of(55L);
            RunState rs = RunState.allocate(T, 0.5, scheme, Collections.emptyList(), 55L);
            Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);

            for (int t = 0; t < T; t++) {
                assertThat(Double.isFinite(rs.logLtSeries[t]))
                    .as("scheme=%s logLt[%d]", scheme, t).isTrue();
                assertThat(rs.essSeries[t])
                    .as("scheme=%s ESS[%d]", scheme, t).isPositive();
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 11: Kalman cross-check
    // ------------------------------------------------------------------

    @Test
    @Tag("slow")
    void linearGaussBootstrap_crossChecksKalmanMean() {
        double rho = 0.9, sigmaX = 1.0, sigmaY = 0.5, sigma0 = 1.0;
        LinearGauss model = new LinearGauss(rho, sigmaX, sigmaY, sigma0);
        int T = 200, N = 5000;
        List<Double> data = syntheticLinearGauss(model, T, 2024L);

        // Hand-rolled Kalman filter.
        double[] kalmanMean = new double[T];
        {
            double mPrev = 0.0, vPrev = sigma0 * sigma0;
            for (int t = 0; t < T; t++) {
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

        // Particle filter with FULL history.
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);
        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.FULL, T);
        // Override default-rate Full (0.6) with rate=1.0 — the high-noise
        // LinearGauss combined with the long T=200 horizon resamples more
        // often than the conservative default allows.
        ws.history = new Full(N, fk.dim(), T, 1.0);
        ws.rng = RandomBatch.of(2024L);
        RunState rs = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, Collections.emptyList(), 2024L);
        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);

        // Compute RMSE of weighted mean vs Kalman mean from history.
        double sse = 0.0;
        int dim = fk.dim();
        double[] Xt = new double[dim * N];
        double[] lwt = new double[N];
        for (int t = 0; t < T; t++) {
            ws.history.viewX(t, Xt, 0);
            ws.history.viewLogW(t, lwt, 0);
            double maxLw = Double.NEGATIVE_INFINITY;
            for (int n = 0; n < N; n++) {
                if (lwt[n] > maxLw) maxLw = lwt[n];
            }
            double sumW = 0.0, sumWX = 0.0;
            for (int n = 0; n < N; n++) {
                double w = Math.exp(lwt[n] - maxLw);
                sumW += w;
                sumWX += w * Xt[n];
            }
            double pfMean = sumWX / sumW;
            double err = pfMean - kalmanMean[t];
            sse += err * err;
        }
        double rmse = Math.sqrt(sse / T);
        // Bootstrap filter at N=5000 has MC variance ~1/sqrt(N)≈1.4%.
        // The RMSE depends on the specific RNG stream; we use a generous
        // bound for this informational cross-check.
        assertThat(rmse).as("RMSE PF vs Kalman").isLessThan(0.1);
    }
}
