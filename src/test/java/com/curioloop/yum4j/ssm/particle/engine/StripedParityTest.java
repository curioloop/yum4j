package com.curioloop.yum4j.ssm.particle.engine;

import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.builtin.LinearGauss;
import com.curioloop.yum4j.ssm.particle.model.builtin.StochVol;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Parity test verifying that STRIPED parallelism is deterministic:
 * same seed + same slab count produces bitwise-identical results
 * across repeated runs.
 *
 * <p>Validates: Requirements R10.6
 */
class StripedParityTest {

    private static final int N = 256;
    private static final int T = 20;
    private static final long[] SEEDS = {1L, 2L, 3L, 4L, 5L};
    private static final int[] CHUNKS = {2, 4, 8};
    private static final double ESS_RMIN = 0.5;

    private static ExecutorService executor;

    @BeforeAll
    static void setUp() {
        executor = Executors.newFixedThreadPool(8);
        // Warmup: run a few iterations to stabilise JIT compilation
        // so that both runs in each test use the same compiled code.
        warmup();
    }

    private static void warmup() {
        LinearGauss model = new LinearGauss(0.9, 1.0, 0.5, Double.NaN);
        List<Double> data = new ArrayList<>(T);
        RandomBatch rng = RandomBatch.of(999L);
        double[] buf = new double[1];
        rng.nextGaussians(buf, 0, 1);
        double x = buf[0] * model.sigma0();
        for (int t = 0; t < T; t++) {
            rng.nextGaussians(buf, 0, 1);
            data.add(x + buf[0] * model.sigmaY());
            rng.nextGaussians(buf, 0, 1);
            x = model.rho() * x + buf[0] * model.sigmaX();
        }
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);
        ParallelStrategy par = ParallelStrategy.striped(executor, 4);
        // Run several times to trigger JIT compilation of hot paths
        for (int i = 0; i < 10; i++) {
            Workspace ws = Workspace.allocate(N, 1, HistoryMode.NONE, 0);
            ws.rng = RandomBatch.of(42L);
            RunState rs = RunState.allocate(T, ESS_RMIN, Scheme.SYSTEMATIC, List.of(), 42L);
            Engine.run(ws, rs, fk, par, data);
        }
    }

    @AfterAll
    static void tearDown() {
        executor.shutdownNow();
    }

    // ------------------------------------------------------------------
    // Test parameter sources
    // ------------------------------------------------------------------

    record Config(long seed, int chunks) {
        @Override
        public String toString() {
            return "seed=" + seed + ",chunks=" + chunks;
        }
    }

    static Stream<Config> seedChunkCombinations() {
        List<Config> configs = new ArrayList<>();
        for (long seed : SEEDS) {
            for (int chunk : CHUNKS) {
                configs.add(new Config(seed, chunk));
            }
        }
        return configs.stream();
    }

    // ------------------------------------------------------------------
    // Determinism tests: two STRIPED runs with same config must match
    // ------------------------------------------------------------------

    @ParameterizedTest(name = "LinearGauss determinism: {0}")
    @MethodSource("seedChunkCombinations")
    void stripedDeterminism_linearGauss(Config cfg) {
        LinearGauss model = new LinearGauss(0.9, 1.0, 0.5, Double.NaN);
        List<Double> data = generateData(model, cfg.seed, T);

        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);
        ParallelStrategy par = ParallelStrategy.striped(executor, cfg.chunks);

        RunResult r1 = runFilter(fk, cfg.seed, par, data);
        RunResult r2 = runFilter(fk, cfg.seed, par, data);

        assertThat(Arrays.equals(r1.logLtSeries, r2.logLtSeries))
                .as("logLtSeries must be bitwise-identical across runs (seed=%d, chunks=%d)",
                        cfg.seed, cfg.chunks)
                .isTrue();
        assertThat(Arrays.equals(r1.essSeries, r2.essSeries))
                .as("essSeries must be bitwise-identical across runs (seed=%d, chunks=%d)",
                        cfg.seed, cfg.chunks)
                .isTrue();
        assertThat(Arrays.equals(r1.finalX, r2.finalX))
                .as("final X must be bitwise-identical across runs (seed=%d, chunks=%d)",
                        cfg.seed, cfg.chunks)
                .isTrue();
    }

    @ParameterizedTest(name = "StochVol determinism: {0}")
    @MethodSource("seedChunkCombinations")
    void stripedDeterminism_stochVol(Config cfg) {
        StochVol model = new StochVol();
        List<Double> data = generateData(model, cfg.seed, T);

        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);
        ParallelStrategy par = ParallelStrategy.striped(executor, cfg.chunks);

        RunResult r1 = runFilter(fk, cfg.seed, par, data);
        RunResult r2 = runFilter(fk, cfg.seed, par, data);

        assertThat(Arrays.equals(r1.logLtSeries, r2.logLtSeries))
                .as("logLtSeries must be bitwise-identical across runs (seed=%d, chunks=%d)",
                        cfg.seed, cfg.chunks)
                .isTrue();
        assertThat(Arrays.equals(r1.essSeries, r2.essSeries))
                .as("essSeries must be bitwise-identical across runs (seed=%d, chunks=%d)",
                        cfg.seed, cfg.chunks)
                .isTrue();
        assertThat(Arrays.equals(r1.finalX, r2.finalX))
                .as("final X must be bitwise-identical across runs (seed=%d, chunks=%d)",
                        cfg.seed, cfg.chunks)
                .isTrue();
    }

    // ------------------------------------------------------------------
    // Finite results test: logLt is finite, ESS is positive
    // ------------------------------------------------------------------

    @ParameterizedTest(name = "Finite results: {0}")
    @MethodSource("seedChunkCombinations")
    void stripedProducesFiniteResults(Config cfg) {
        // Test with both models
        LinearGauss lgModel = new LinearGauss(0.9, 1.0, 0.5, Double.NaN);
        List<Double> lgData = generateData(lgModel, cfg.seed, T);
        FeynmanKac<Double> lgFk = FeynmanKac.bootstrap(lgModel);

        StochVol svModel = new StochVol();
        List<Double> svData = generateData(svModel, cfg.seed, T);
        FeynmanKac<Double> svFk = FeynmanKac.bootstrap(svModel);

        ParallelStrategy par = ParallelStrategy.striped(executor, cfg.chunks);

        RunResult lgResult = runFilter(lgFk, cfg.seed, par, lgData);
        RunResult svResult = runFilter(svFk, cfg.seed, par, svData);

        // LinearGauss: logLt finite, ESS positive
        for (int t = 0; t < T; t++) {
            assertThat(Double.isFinite(lgResult.logLtSeries[t]))
                    .as("LinearGauss logLt[%d] must be finite (seed=%d, chunks=%d)", t, cfg.seed, cfg.chunks)
                    .isTrue();
            assertThat(lgResult.essSeries[t])
                    .as("LinearGauss ESS[%d] must be positive (seed=%d, chunks=%d)", t, cfg.seed, cfg.chunks)
                    .isGreaterThan(0.0);
        }

        // StochVol: logLt finite, ESS positive
        for (int t = 0; t < T; t++) {
            assertThat(Double.isFinite(svResult.logLtSeries[t]))
                    .as("StochVol logLt[%d] must be finite (seed=%d, chunks=%d)", t, cfg.seed, cfg.chunks)
                    .isTrue();
            assertThat(svResult.essSeries[t])
                    .as("StochVol ESS[%d] must be positive (seed=%d, chunks=%d)", t, cfg.seed, cfg.chunks)
                    .isGreaterThan(0.0);
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private record RunResult(double[] logLtSeries, double[] essSeries, double[] finalX) {}

    private RunResult runFilter(FeynmanKac<Double> fk, long seed, ParallelStrategy par, List<Double> data) {
        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(seed);

        RunState rs = RunState.allocate(T, ESS_RMIN, Scheme.SYSTEMATIC, List.of(), seed);

        Engine.run(ws, rs, fk, par, data);

        return new RunResult(
                Arrays.copyOf(rs.logLtSeries, T),
                Arrays.copyOf(rs.essSeries, T),
                Arrays.copyOf(ws.X, N * fk.dim())
        );
    }

    /**
     * Generates synthetic observation data by simulating from the model.
     * Uses a separate RNG seeded deterministically so data is reproducible.
     */
    private static List<Double> generateData(LinearGauss model, long seed, int T) {
        RandomBatch rng = RandomBatch.of(seed + 1000);
        double[] buf = new double[1];
        List<Double> data = new ArrayList<>(T);

        // x_0 ~ N(0, sigma0)
        rng.nextGaussians(buf, 0, 1);
        double x = buf[0] * model.sigma0();

        for (int t = 0; t < T; t++) {
            // y_t ~ N(x_t, sigmaY)
            rng.nextGaussians(buf, 0, 1);
            double y = x + buf[0] * model.sigmaY();
            data.add(y);

            // x_{t+1} ~ N(rho * x_t, sigmaX)
            rng.nextGaussians(buf, 0, 1);
            x = model.rho() * x + buf[0] * model.sigmaX();
        }
        return data;
    }

    /**
     * Generates synthetic observation data for StochVol model.
     */
    private static List<Double> generateData(StochVol model, long seed, int T) {
        RandomBatch rng = RandomBatch.of(seed + 2000);
        double[] buf = new double[1];
        List<Double> data = new ArrayList<>(T);

        // x_0 ~ N(mu, sigma0)
        rng.nextGaussians(buf, 0, 1);
        double x = model.mu() + buf[0] * model.sigma0();

        for (int t = 0; t < T; t++) {
            // y_t ~ N(0, exp(0.5 * x_t))
            rng.nextGaussians(buf, 0, 1);
            double y = buf[0] * Math.exp(0.5 * x);
            data.add(y);

            // x_{t+1} ~ N((1-rho)*mu + rho*x_t, sigma)
            rng.nextGaussians(buf, 0, 1);
            x = (1.0 - model.rho()) * model.mu() + model.rho() * x + buf[0] * model.sigma();
        }
        return data;
    }
}
