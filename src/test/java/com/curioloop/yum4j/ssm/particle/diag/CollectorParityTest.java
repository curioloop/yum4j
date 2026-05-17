package com.curioloop.yum4j.ssm.particle.diag;

import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.engine.*;
import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.builtin.LinearGauss;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Self-consistency parity tests for built-in collectors.
 *
 * <p>Since the old engine was deleted, these verify:
 * <ol>
 *   <li><b>Determinism:</b> same seed + same collectors → identical output across two runs</li>
 *   <li><b>Moments correctness:</b> weighted mean matches hand-computed value from final particles</li>
 *   <li><b>VarLogLt:</b> variance estimate is non-negative and finite</li>
 *   <li><b>VarEstimator:</b> output is finite</li>
 *   <li><b>Collector lifecycle:</b> attach is called before afterReweight, correct time index</li>
 * </ol>
 *
 * <p>Validates: Requirements R21.3
 */
class CollectorParityTest {

    // Fixed parameters
    private static final int N = 256;
    private static final int T = 20;
    private static final long SEED = 42L;
    private static final long DATA_SEED = 99L;

    private static final LinearGauss MODEL = new LinearGauss(0.9, 1.0, 0.5, 1.0);
    private static List<Double> DATA;

    @BeforeAll
    static void generateData() {
        DATA = syntheticLinearGauss(MODEL, T, DATA_SEED);
    }

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

    // ------------------------------------------------------------------
    // Test 1: Determinism — Moments collector produces identical output
    // ------------------------------------------------------------------

    @Test
    void moments_deterministic_sameSeedSameOutput() {
        Moments m1 = new Moments();
        Moments m2 = new Moments();

        runFilter(SEED, m1);
        runFilter(SEED, m2);

        assertThat(m1.means()).containsExactly(m2.means());
        assertThat(m1.variances()).containsExactly(m2.variances());
    }

    // ------------------------------------------------------------------
    // Test 2: Determinism — VarLogLt collector produces identical output
    // ------------------------------------------------------------------

    @Test
    void varLogLt_deterministic_sameSeedSameOutput() {
        VarLogLt v1 = new VarLogLt();
        VarLogLt v2 = new VarLogLt();

        runFilter(SEED, v1);
        runFilter(SEED, v2);

        assertThat(v1.estimates()).containsExactly(v2.estimates());
    }

    // ------------------------------------------------------------------
    // Test 3: Determinism — VarEstimator collector produces identical output
    // ------------------------------------------------------------------

    @Test
    void varEstimator_deterministic_sameSeedSameOutput() {
        VarEstimator ve1 = new VarEstimator();
        VarEstimator ve2 = new VarEstimator();

        runFilter(SEED, ve1);
        runFilter(SEED, ve2);

        assertThat(ve1.estimates()).containsExactly(ve2.estimates());
    }

    // ------------------------------------------------------------------
    // Test 4: Moments correctness — weighted mean matches hand-computed
    // ------------------------------------------------------------------

    @Test
    void moments_weightedMean_matchesHandComputed() {
        Moments moments = new Moments();

        // Run filter and capture final workspace state
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(MODEL);
        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(SEED);
        RunState rs = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, List.of(moments), SEED);
        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, DATA);

        // Hand-compute weighted mean from final particles (ws.X) and log-weights (ws.logW)
        double maxLw = Double.NEGATIVE_INFINITY;
        for (int n = 0; n < N; n++) {
            if (ws.logW[n] > maxLw) maxLw = ws.logW[n];
        }
        double sumW = 0.0, sumWX = 0.0;
        for (int n = 0; n < N; n++) {
            double w = Math.exp(ws.logW[n] - maxLw);
            sumW += w;
            sumWX += w * ws.X[n]; // dim=1, so X[n] is the state
        }
        double handMean = sumWX / sumW;

        // The Moments collector stores means at index [t * dim + j]
        // For the final step (T-1), dim=1, j=0:
        double collectorMean = moments.means()[(T - 1) * moments.dim()];

        assertThat(collectorMean).isCloseTo(handMean, within(1e-12));
    }

    // ------------------------------------------------------------------
    // Test 5: VarLogLt — variance is non-negative and finite
    // ------------------------------------------------------------------

    @Test
    void varLogLt_nonNegativeAndFinite() {
        VarLogLt varLogLt = new VarLogLt();
        runFilter(SEED, varLogLt);

        double[] estimates = varLogLt.estimates();
        assertThat(estimates).hasSize(T);
        for (int t = 0; t < T; t++) {
            assertThat(estimates[t])
                    .as("VarLogLt[%d]", t)
                    .isGreaterThanOrEqualTo(0.0);
            assertThat(Double.isFinite(estimates[t]))
                    .as("VarLogLt[%d] finite", t)
                    .isTrue();
        }
    }

    // ------------------------------------------------------------------
    // Test 6: VarEstimator — output is finite
    // ------------------------------------------------------------------

    @Test
    void varEstimator_outputIsFinite() {
        VarEstimator varEst = new VarEstimator();
        runFilter(SEED, varEst);

        double[] estimates = varEst.estimates();
        assertThat(estimates).hasSize(T);
        for (int t = 0; t < T; t++) {
            assertThat(Double.isFinite(estimates[t]))
                    .as("VarEstimator[%d] finite", t)
                    .isTrue();
            assertThat(estimates[t])
                    .as("VarEstimator[%d] non-negative", t)
                    .isGreaterThanOrEqualTo(0.0);
        }
    }

    // ------------------------------------------------------------------
    // Test 7: Collector lifecycle — attach called before afterReweight,
    //         correct time index received
    // ------------------------------------------------------------------

    @Test
    void collectorLifecycle_attachBeforeReweight_correctTimeIndex() {
        LifecycleTracker tracker = new LifecycleTracker();
        runFilter(SEED, tracker);

        assertThat(tracker.attachCalled).isTrue();
        assertThat(tracker.firstReweightBeforeAttach).isFalse();
        // afterReweight should have been called for every step 0..T-1
        assertThat(tracker.reweightTimeIndices).hasSize(T);
        for (int t = 0; t < T; t++) {
            assertThat(tracker.reweightTimeIndices.get(t)).isEqualTo(t);
        }
        // afterMutation should also have been called for every step
        assertThat(tracker.mutationTimeIndices).hasSize(T);
        for (int t = 0; t < T; t++) {
            assertThat(tracker.mutationTimeIndices.get(t)).isEqualTo(t);
        }
    }

    // ------------------------------------------------------------------
    // Test 8: VarLogLt is monotonically non-decreasing (cumulative)
    // ------------------------------------------------------------------

    @Test
    void varLogLt_monotonicallyNonDecreasing() {
        VarLogLt varLogLt = new VarLogLt();
        runFilter(SEED, varLogLt);

        double[] estimates = varLogLt.estimates();
        for (int t = 1; t < T; t++) {
            assertThat(estimates[t])
                    .as("VarLogLt[%d] >= VarLogLt[%d]", t, t - 1)
                    .isGreaterThanOrEqualTo(estimates[t - 1]);
        }
    }

    // ------------------------------------------------------------------
    // Test 9: Moments variance is non-negative
    // ------------------------------------------------------------------

    @Test
    void moments_varianceNonNegative() {
        Moments moments = new Moments();
        runFilter(SEED, moments);

        double[] vars = moments.variances();
        for (int t = 0; t < T; t++) {
            assertThat(vars[t])
                    .as("Moments.var[%d]", t)
                    .isGreaterThanOrEqualTo(0.0);
            assertThat(Double.isFinite(vars[t]))
                    .as("Moments.var[%d] finite", t)
                    .isTrue();
        }
    }

    // ------------------------------------------------------------------
    // Test 10: All three collectors together produce same results as
    //          running them individually (no interference)
    // ------------------------------------------------------------------

    @Test
    void multipleCollectors_noInterference() {
        // Run all together
        Moments mAll = new Moments();
        VarLogLt vlAll = new VarLogLt();
        VarEstimator veAll = new VarEstimator();
        runFilter(SEED, mAll, vlAll, veAll);

        // Run individually
        Moments mSolo = new Moments();
        VarLogLt vlSolo = new VarLogLt();
        VarEstimator veSolo = new VarEstimator();
        runFilter(SEED, mSolo);
        runFilter(SEED, vlSolo);
        runFilter(SEED, veSolo);

        assertThat(mAll.means()).containsExactly(mSolo.means());
        assertThat(mAll.variances()).containsExactly(mSolo.variances());
        assertThat(vlAll.estimates()).containsExactly(vlSolo.estimates());
        assertThat(veAll.estimates()).containsExactly(veSolo.estimates());
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private void runFilter(long seed, Collector... collectors) {
        FeynmanKac<Double> fk = FeynmanKac.bootstrap(MODEL);
        Workspace ws = Workspace.allocate(N, fk.dim(), HistoryMode.NONE, 0);
        ws.rng = RandomBatch.of(seed);
        RunState rs = RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, List.of(collectors), seed);
        Engine.run(ws, rs, fk, ParallelStrategy.SERIAL, DATA);
    }

    /**
     * A test collector that tracks lifecycle events.
     */
    private static class LifecycleTracker implements Collector {
        boolean attachCalled = false;
        boolean firstReweightBeforeAttach = false;
        final List<Integer> reweightTimeIndices = new ArrayList<>();
        final List<Integer> mutationTimeIndices = new ArrayList<>();

        @Override
        public void attach(Workspace ws, RunState rs) {
            attachCalled = true;
        }

        @Override
        public void afterReweight(Workspace ws, RunState rs, int t) {
            if (!attachCalled) {
                firstReweightBeforeAttach = true;
            }
            reweightTimeIndices.add(t);
        }

        @Override
        public void afterMutation(Workspace ws, RunState rs, int t) {
            mutationTimeIndices.add(t);
        }
    }
}
