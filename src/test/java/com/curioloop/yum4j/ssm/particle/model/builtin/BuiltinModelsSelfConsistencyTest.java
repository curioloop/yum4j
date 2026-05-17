package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.engine.*;
import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.builtin.*;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Self-consistency (determinism) parity tests for the six remaining built-in
 * models: StochVolLeverage, MVStochVol, Gordon, BearingsOnly, DiscreteCox,
 * and ThetaLogistic.
 *
 * <p>Since the old engine was deleted, these verify:
 * <ol>
 *   <li><b>Determinism:</b> Same seed + same config produces logLt within 1e-10
 *       relative and ESS within 1e-12 absolute across two runs</li>
 *   <li><b>Finite results:</b> logLt is finite, ESS is positive for all steps</li>
 *   <li><b>Scheme invariance:</b> SYSTEMATIC and SSP both produce finite, reasonable results</li>
 * </ol>
 *
 * <p>Validates: Requirements R13.1–R13.3
 */
class BuiltinModelsSelfConsistencyTest {

    private static final int N = 256;
    private static final int T = 20;
    private static final long[] SEEDS = {1L, 2L, 3L, 4L, 5L};
    private static final Scheme[] SCHEMES = {Scheme.SYSTEMATIC, Scheme.SSP};
    private static final double ESS_RMIN = 0.5;

    /** ESS absolute tolerance per R13.3. */
    private static final double ESS_ABS_TOL = 1e-12;

    /** logLt relative tolerance per R13.1–R13.2. */
    private static final double LOG_LT_REL_TOL = 1e-10;

    // ------------------------------------------------------------------
    // Test parameter source
    // ------------------------------------------------------------------

    enum ModelId {
        STOCH_VOL_LEVERAGE,
        MV_STOCH_VOL,
        GORDON,
        BEARINGS_ONLY,
        DISCRETE_COX,
        THETA_LOGISTIC
    }

    record Config(ModelId modelId, long seed, Scheme scheme) {
        @Override
        public String toString() {
            return modelId + " seed=" + seed + " " + scheme;
        }
    }

    static Stream<Config> modelSeedSchemeGrid() {
        List<Config> configs = new ArrayList<>();
        for (ModelId m : ModelId.values()) {
            for (long seed : SEEDS) {
                for (Scheme scheme : SCHEMES) {
                    configs.add(new Config(m, seed, scheme));
                }
            }
        }
        return configs.stream();
    }

    // ------------------------------------------------------------------
    // Test 1: Determinism — two runs with same seed match within tolerance
    // ------------------------------------------------------------------

    @ParameterizedTest(name = "determinism: {0}")
    @MethodSource("modelSeedSchemeGrid")
    void sameSeedProducesConsistentResults(Config cfg) {
        // Run the filter twice with fresh model instances (required for
        // StochVolLeverage which has mutable prevObservation state).
        // Assert results match within R13 tolerances. JIT compilation of
        // SIMD exp() paths can cause sub-ULP differences between runs.
        RunResult r1 = runModelFresh(cfg);
        RunResult r2 = runModelFresh(cfg);

        for (int t = 0; t < T; t++) {
            double logLt1 = r1.logLtSeries[t];
            double logLt2 = r2.logLtSeries[t];
            double denom = Math.abs(logLt1);
            double relDiff = denom > 1e-15
                    ? Math.abs(logLt1 - logLt2) / denom
                    : Math.abs(logLt1 - logLt2);
            assertThat(relDiff)
                    .as("logLt[%d] relative diff (%s): %s vs %s", t, cfg, logLt1, logLt2)
                    .isLessThanOrEqualTo(LOG_LT_REL_TOL);

            double essDiff = Math.abs(r1.essSeries[t] - r2.essSeries[t]);
            assertThat(essDiff)
                    .as("ESS[%d] absolute diff (%s): %s vs %s", t, cfg,
                            r1.essSeries[t], r2.essSeries[t])
                    .isLessThanOrEqualTo(ESS_ABS_TOL);
        }
    }

    // ------------------------------------------------------------------
    // Test 2: Finite results — logLt is finite, ESS is positive
    // ------------------------------------------------------------------

    @ParameterizedTest(name = "finite results: {0}")
    @MethodSource("modelSeedSchemeGrid")
    void logLtFiniteAndEssPositive(Config cfg) {
        RunResult r = runModelFresh(cfg);

        for (int t = 0; t < T; t++) {
            assertThat(Double.isFinite(r.logLtSeries[t]))
                    .as("logLt[%d] must be finite (%s), was %s", t, cfg, r.logLtSeries[t])
                    .isTrue();
            assertThat(r.essSeries[t])
                    .as("ESS[%d] must be positive (%s)", t, cfg)
                    .isGreaterThan(0.0);
        }
    }

    // ------------------------------------------------------------------
    // Test 3: Scheme invariance — both SYSTEMATIC and SSP produce finite results
    //         (just checks both are finite and reasonable; not comparing values)
    // ------------------------------------------------------------------

    @ParameterizedTest(name = "scheme invariance: {0}")
    @MethodSource("modelSeedSchemeGrid")
    void bothSchemesProduceReasonableResults(Config cfg) {
        RunResult r = runModelFresh(cfg);

        // logLt should be finite and not degenerate (not all zeros)
        boolean anyNonZero = false;
        for (int t = 0; t < T; t++) {
            assertThat(Double.isFinite(r.logLtSeries[t]))
                    .as("logLt[%d] finite (%s)", t, cfg).isTrue();
            if (r.logLtSeries[t] != 0.0) anyNonZero = true;
        }
        assertThat(anyNonZero)
                .as("logLt should have non-zero entries (%s)", cfg).isTrue();

        // ESS should be bounded: 0 < ESS <= N
        for (int t = 0; t < T; t++) {
            assertThat(r.essSeries[t])
                    .as("ESS[%d] in (0, N] (%s)", t, cfg)
                    .isGreaterThan(0.0)
                    .isLessThanOrEqualTo(N + 1e-6);
        }
    }

    // ------------------------------------------------------------------
    // Run helper
    // ------------------------------------------------------------------

    private record RunResult(double[] logLtSeries, double[] essSeries) {}

    /**
     * Creates a fresh model instance and runs the filter. This ensures no
     * mutable state leaks between runs (e.g. StochVolLeverage.prevObservation).
     */
    private RunResult runModelFresh(Config cfg) {
        return switch (cfg.modelId) {
            case STOCH_VOL_LEVERAGE -> runStochVolLeverage(cfg.seed, cfg.scheme);
            case MV_STOCH_VOL -> runMVStochVol(cfg.seed, cfg.scheme);
            case GORDON -> runGordon(cfg.seed, cfg.scheme);
            case BEARINGS_ONLY -> runBearingsOnly(cfg.seed, cfg.scheme);
            case DISCRETE_COX -> runDiscreteCox(cfg.seed, cfg.scheme);
            case THETA_LOGISTIC -> runThetaLogistic(cfg.seed, cfg.scheme);
        };
    }

    // ------------------------------------------------------------------
    // StochVolLeverage — observation type Double, dim=1
    // ------------------------------------------------------------------

    private RunResult runStochVolLeverage(long seed, Scheme scheme) {
        StochVolLeverage model = new StochVolLeverage();
        List<Double> data = generateStochVolLeverageData(model, seed);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);
        return runFilter(fk, seed, scheme, data);
    }

    private static List<Double> generateStochVolLeverageData(StochVolLeverage model, long seed) {
        RandomGenerator g = rng(seed + 3000);
        List<Double> data = new ArrayList<>(T);
        double x = model.mu() + model.sigma0() * g.nextGaussian();
        for (int t = 0; t < T; t++) {
            // Y_t ~ N(0, exp(X_t/2))
            double y = Math.exp(0.5 * x) * g.nextGaussian();
            data.add(y);
            // X_{t+1}: AR(1) around mu (simplified, no leverage in data gen)
            x = model.mu() + model.rho() * (x - model.mu()) + model.sigma() * g.nextGaussian();
        }
        return data;
    }

    // ------------------------------------------------------------------
    // MVStochVol — observation type double[], dim=2
    // ------------------------------------------------------------------

    private RunResult runMVStochVol(long seed, Scheme scheme) {
        MVStochVol model = new MVStochVol(
                new double[]{-1.0, 0.0},
                new double[]{0.95, 0.9},
                new double[]{0.2, 0.15});
        List<double[]> data = generateMVStochVolData(model, seed);
        FeynmanKac<double[]> fk = FeynmanKac.bootstrap(model);
        return runFilter(fk, seed, scheme, data);
    }

    private static List<double[]> generateMVStochVolData(MVStochVol model, long seed) {
        RandomGenerator g = rng(seed + 4000);
        int d = model.dim();
        double[] mu = model.mu();
        double[] rho = model.rho();
        double[] sigma = model.sigma();
        double[] x = new double[d];

        // X_0 ~ N(mu[k], sigma0[k])
        for (int k = 0; k < d; k++) {
            double sigma0k = sigma[k] / Math.sqrt(1.0 - rho[k] * rho[k]);
            x[k] = mu[k] + sigma0k * g.nextGaussian();
        }

        List<double[]> data = new ArrayList<>(T);
        for (int t = 0; t < T; t++) {
            // Y_t[k] ~ N(0, exp(X_t[k]/2))
            double[] yt = new double[d];
            for (int k = 0; k < d; k++) {
                yt[k] = Math.exp(0.5 * x[k]) * g.nextGaussian();
            }
            data.add(yt);
            // X_{t+1}[k] = (1-rho[k])*mu[k] + rho[k]*X_t[k] + sigma[k]*N(0,1)
            for (int k = 0; k < d; k++) {
                x[k] = (1.0 - rho[k]) * mu[k] + rho[k] * x[k] + sigma[k] * g.nextGaussian();
            }
        }
        return data;
    }

    // ------------------------------------------------------------------
    // Gordon — observation type Double, dim=1
    // ------------------------------------------------------------------

    private RunResult runGordon(long seed, Scheme scheme) {
        Gordon model = new Gordon();
        List<Double> data = generateGordonData(model, seed);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);
        return runFilter(fk, seed, scheme, data);
    }

    private static List<Double> generateGordonData(Gordon model, long seed) {
        RandomGenerator g = rng(seed + 5000);
        List<Double> data = new ArrayList<>(T);
        double x = model.sigma0() * g.nextGaussian();
        for (int t = 0; t < T; t++) {
            // Y_t ~ N(a * X_t², sigmaY)
            double y = model.a() * x * x + model.sigmaY() * g.nextGaussian();
            data.add(y);
            // X_{t+1} = b*X_t + c*X_t/(1+X_t²) + d*cos(e*t) + sigmaX*N(0,1)
            double loc = model.b() * x + model.c() * x / (1.0 + x * x)
                    + model.d() * Math.cos(model.e() * t);
            x = loc + model.sigmaX() * g.nextGaussian();
        }
        return data;
    }

    // ------------------------------------------------------------------
    // BearingsOnly — observation type Double, dim=4
    // ------------------------------------------------------------------

    private RunResult runBearingsOnly(long seed, Scheme scheme) {
        BearingsOnly model = new BearingsOnly();
        List<Double> data = generateBearingsOnlyData(model, seed);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);
        return runFilter(fk, seed, scheme, data);
    }

    private static List<Double> generateBearingsOnlyData(BearingsOnly model, long seed) {
        RandomGenerator g = rng(seed + 6000);
        double[] x0 = model.x0();
        double vx = x0[0] + model.sigmaX() * g.nextGaussian();
        double vy = x0[1] + model.sigmaX() * g.nextGaussian();
        double px = x0[2];
        double py = x0[3];

        List<Double> data = new ArrayList<>(T);
        for (int t = 0; t < T; t++) {
            // Y_t ~ N(angle(px, py), sigmaY)
            double angle = BearingsOnly.bearingAngle(px, py);
            double y = angle + model.sigmaY() * g.nextGaussian();
            data.add(y);
            // Advance state
            double newVx = vx + model.sigmaX() * g.nextGaussian();
            double newVy = vy + model.sigmaX() * g.nextGaussian();
            px = px + vx;
            py = py + vy;
            vx = newVx;
            vy = newVy;
        }
        return data;
    }

    // ------------------------------------------------------------------
    // DiscreteCox — observation type Integer, dim=1
    // ------------------------------------------------------------------

    private RunResult runDiscreteCox(long seed, Scheme scheme) {
        DiscreteCox model = new DiscreteCox();
        List<Integer> data = generateDiscreteCoxData(model, seed);
        FeynmanKac<Integer> fk = FeynmanKac.bootstrap(model);
        return runFilter(fk, seed, scheme, data);
    }

    private static List<Integer> generateDiscreteCoxData(DiscreteCox model, long seed) {
        RandomGenerator g = rng(seed + 7000);
        List<Integer> data = new ArrayList<>(T);
        double x = model.mu() + model.sigma0() * g.nextGaussian();
        for (int t = 0; t < T; t++) {
            // Y_t ~ Poisson(exp(X_t))
            double lambda = Math.exp(x);
            int y = poissonSample(g, lambda);
            data.add(y);
            // X_{t+1} = mu + rho*(X_t - mu) + sigma*N(0,1)
            x = model.mu() + model.rho() * (x - model.mu()) + model.sigma() * g.nextGaussian();
        }
        return data;
    }

    /**
     * Simple Poisson sampling via inverse-CDF (Knuth's algorithm).
     * Adequate for moderate lambda values used in tests.
     */
    private static int poissonSample(RandomGenerator g, double lambda) {
        double L = Math.exp(-lambda);
        int k = 0;
        double p = 1.0;
        do {
            k++;
            p *= g.nextDouble();
        } while (p > L);
        return k - 1;
    }

    // ------------------------------------------------------------------
    // ThetaLogistic — observation type Double, dim=1
    // ------------------------------------------------------------------

    private RunResult runThetaLogistic(long seed, Scheme scheme) {
        ThetaLogistic model = new ThetaLogistic();
        List<Double> data = generateThetaLogisticData(model, seed);
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(model);
        return runFilter(fk, seed, scheme, data);
    }

    private static List<Double> generateThetaLogisticData(ThetaLogistic model, long seed) {
        RandomGenerator g = rng(seed + 8000);
        List<Double> data = new ArrayList<>(T);
        double logK = Math.log(model.K());
        double x = logK + model.sigmaX() * g.nextGaussian();
        for (int t = 0; t < T; t++) {
            // Y_t ~ N(X_t, sigmaY)
            double y = x + model.sigmaY() * g.nextGaussian();
            data.add(y);
            // X_{t+1} = X_t + r*(1 - (exp(X_t)/K)^theta) + sigmaX*N(0,1)
            double ratio = Math.pow(Math.exp(x) / model.K(), model.theta());
            x = x + model.r() * (1.0 - ratio) + model.sigmaX() * g.nextGaussian();
        }
        return data;
    }

    // ------------------------------------------------------------------
    // Generic filter runner
    // ------------------------------------------------------------------

    private <Y> RunResult runFilter(FeynmanKac<Y> fk, long seed, Scheme scheme, List<Y> data) {
        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(seed);
        RunState rs = RunState.allocate(T, ESS_RMIN, scheme, List.of(), seed);

        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, data);

        return new RunResult(
                Arrays.copyOf(rs.logLtSeries, T),
                Arrays.copyOf(rs.essSeries, T)
        );
    }

    private static RandomGenerator rng(long seed) {
        return RandomGeneratorFactory.of("L64X128MixRandom").create(seed);
    }
}
