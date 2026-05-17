package com.curioloop.yum4j.ssm.particle.smooth;

import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Unit tests for {@link FixedLag} fixed-lag smoother.
 */
class FixedLagTest {

    private static final int N = 8;
    private static final int DIM = 1;

    /**
     * Creates a workspace with deterministic data for step t.
     * Particles: X[n] = t * 10.0 + n
     * LogW: uniform (all zeros)
     * Ancestors: identity (a[n] = n)
     */
    private Workspace makeWorkspace(int t, int N, int dim) {
        Workspace ws = Workspace.allocate(N, dim, HistoryMode.NONE, 0);
        for (int i = 0; i < N * dim; i++) {
            ws.X[i] = t * 10.0 + i;
        }
        for (int i = 0; i < N; i++) {
            ws.logW[i] = 0.0; // uniform weights
        }
        for (int i = 0; i < N; i++) {
            ws.a[i] = i; // identity ancestors
        }
        return ws;
    }

    @Test
    void zeroLag_returnsFilterMean() {
        int L = 5;
        Rolling history = new Rolling(L, N, DIM);

        // Save 5 steps with identity ancestors and uniform weights
        for (int t = 0; t < L; t++) {
            Workspace ws = makeWorkspace(t, N, DIM);
            history.save(t, ws, false);
        }

        // With lag=0, the smoothed estimate should equal the filter mean
        // at the newest step (t=4). Particles are [40, 41, ..., 47].
        // Uniform weights → mean = (40+41+...+47)/8 = 43.5
        double result = FixedLag.compute(history, 0, FixedLag.identity1D());
        assertThat(result).isCloseTo(43.5, org.assertj.core.data.Offset.offset(1e-12));
    }

    @Test
    void zeroLag_matchesSmoothMean1D() {
        int L = 5;
        Rolling history = new Rolling(L, N, DIM);
        for (int t = 0; t < L; t++) {
            history.save(t, makeWorkspace(t, N, DIM), false);
        }
        double result = FixedLag.smoothedMean1D(history, 0);
        assertThat(result).isCloseTo(43.5, org.assertj.core.data.Offset.offset(1e-12));
    }

    @Test
    void identityAncestors_lagEqualsDirectMean() {
        // With identity ancestors, tracing back any lag should give
        // the particles at the target step directly.
        int L = 5;
        Rolling history = new Rolling(L, N, DIM);
        for (int t = 0; t < L; t++) {
            history.save(t, makeWorkspace(t, N, DIM), false);
        }

        // lag=2 → target step = 4-2 = 2. Particles at step 2: [20..27].
        // Mean = 23.5
        double result = FixedLag.compute(history, 2, FixedLag.identity1D());
        assertThat(result).isCloseTo(23.5, org.assertj.core.data.Offset.offset(1e-12));
    }

    @Test
    void nonTrivialAncestors_tracesCorrectly() {
        // Set up a scenario where ancestors are NOT identity.
        // Step 0: particles [0, 1, 2, 3]
        // Step 1: ancestors [0, 0, 1, 1] → particles at step 0 traced
        //         through step 1 ancestors give [0, 0, 1, 1]
        int N = 4;
        int L = 3;
        Rolling history = new Rolling(L, N, DIM);

        // Step 0: particles = [10, 11, 12, 13], uniform weights, identity ancestors
        Workspace ws0 = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        ws0.X = new double[]{10.0, 11.0, 12.0, 13.0};
        ws0.logW = new double[]{0.0, 0.0, 0.0, 0.0};
        ws0.a = new int[]{0, 1, 2, 3};
        history.save(0, ws0, false);

        // Step 1: particles = [20, 21, 22, 23], uniform weights,
        //         ancestors = [0, 0, 2, 2] (particles 0,1 came from 0; 2,3 from 2)
        Workspace ws1 = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        ws1.X = new double[]{20.0, 21.0, 22.0, 23.0};
        ws1.logW = new double[]{0.0, 0.0, 0.0, 0.0};
        ws1.a = new int[]{0, 0, 2, 2};
        history.save(1, ws1, true);

        // Step 2: particles = [30, 31, 32, 33], uniform weights,
        //         ancestors = [1, 1, 3, 3] (particles 0,1 came from 1; 2,3 from 3)
        Workspace ws2 = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        ws2.X = new double[]{30.0, 31.0, 32.0, 33.0};
        ws2.logW = new double[]{0.0, 0.0, 0.0, 0.0};
        ws2.a = new int[]{1, 1, 3, 3};
        history.save(2, ws2, true);

        // Lag=1: trace from step 2 back to step 1.
        // B starts as [0,1,2,3]. ancestors(2) = [1,1,3,3].
        // B becomes [1,1,3,3].
        // Particles at step 1: [20,21,22,23].
        // Gathered: [21, 21, 23, 23]. Mean = (21+21+23+23)/4 = 22.0
        double lag1 = FixedLag.compute(history, 1, FixedLag.identity1D());
        assertThat(lag1).isCloseTo(22.0, org.assertj.core.data.Offset.offset(1e-12));

        // Lag=2: trace from step 2 back to step 0.
        // B starts as [0,1,2,3].
        // After ancestors(2): B = [1,1,3,3].
        // After ancestors(1): ancestors(1) = [0,0,2,2].
        //   B[0] = ancestors(1)[1] = 0
        //   B[1] = ancestors(1)[1] = 0
        //   B[2] = ancestors(1)[3] = 2
        //   B[3] = ancestors(1)[3] = 2
        // Particles at step 0: [10,11,12,13].
        // Gathered: [10, 10, 12, 12]. Mean = (10+10+12+12)/4 = 11.0
        double lag2 = FixedLag.compute(history, 2, FixedLag.identity1D());
        assertThat(lag2).isCloseTo(11.0, org.assertj.core.data.Offset.offset(1e-12));
    }

    @Test
    void nonUniformWeights_weightedAverage() {
        int N = 4;
        int L = 2;
        Rolling history = new Rolling(L, N, DIM);

        // Step 0: particles = [1, 2, 3, 4], identity ancestors
        Workspace ws0 = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        ws0.X = new double[]{1.0, 2.0, 3.0, 4.0};
        ws0.logW = new double[]{0.0, 0.0, 0.0, 0.0};
        ws0.a = new int[]{0, 1, 2, 3};
        history.save(0, ws0, false);

        // Step 1: particles = [10, 20, 30, 40], identity ancestors,
        //         non-uniform weights: log(1), log(2), log(3), log(4)
        Workspace ws1 = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        ws1.X = new double[]{10.0, 20.0, 30.0, 40.0};
        ws1.logW = new double[]{Math.log(1), Math.log(2), Math.log(3), Math.log(4)};
        ws1.a = new int[]{0, 1, 2, 3};
        history.save(1, ws1, false);

        // Lag=1: trace back to step 0 with identity ancestors.
        // Gathered particles at step 0: [1, 2, 3, 4].
        // Weights at step 1: [1, 2, 3, 4] (after exp).
        // Weighted mean = (1*1 + 2*2 + 3*3 + 4*4) / (1+2+3+4) = 30/10 = 3.0
        double result = FixedLag.compute(history, 1, FixedLag.identity1D());
        assertThat(result).isCloseTo(3.0, org.assertj.core.data.Offset.offset(1e-12));
    }

    @Test
    void traceAncestors_returnsCorrectIndices() {
        int N = 4;
        int L = 3;
        Rolling history = new Rolling(L, N, DIM);

        Workspace ws0 = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        ws0.X = new double[]{10, 11, 12, 13};
        ws0.logW = new double[]{0, 0, 0, 0};
        ws0.a = new int[]{0, 1, 2, 3};
        history.save(0, ws0, false);

        Workspace ws1 = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        ws1.X = new double[]{20, 21, 22, 23};
        ws1.logW = new double[]{0, 0, 0, 0};
        ws1.a = new int[]{0, 0, 2, 2};
        history.save(1, ws1, true);

        Workspace ws2 = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        ws2.X = new double[]{30, 31, 32, 33};
        ws2.logW = new double[]{0, 0, 0, 0};
        ws2.a = new int[]{1, 1, 3, 3};
        history.save(2, ws2, true);

        // Lag=2: trace from step 2 → step 0
        int[] ancestors = FixedLag.traceAncestors(history, 2);
        // B starts [0,1,2,3]
        // After ancestors(2)=[1,1,3,3]: B=[1,1,3,3]
        // After ancestors(1)=[0,0,2,2]: B=[0,0,2,2]
        assertThat(ancestors).containsExactly(0, 0, 2, 2);
    }

    @Test
    void nullHistory_throws() {
        assertThatThrownBy(() -> FixedLag.compute(null, 0, FixedLag.identity1D()))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void negativeLag_throws() {
        Rolling history = new Rolling(3, N, DIM);
        history.save(0, makeWorkspace(0, N, DIM), false);
        assertThatThrownBy(() -> FixedLag.compute(history, -1, FixedLag.identity1D()))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void emptyHistory_throws() {
        Rolling history = new Rolling(3, N, DIM);
        assertThatThrownBy(() -> FixedLag.compute(history, 0, FixedLag.identity1D()))
            .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void lagExceedsWindow_throws() {
        int L = 3;
        Rolling history = new Rolling(L, N, DIM);
        // Save steps 5, 6, 7 (window = [5, 7])
        for (int t = 0; t < 8; t++) {
            history.save(t, makeWorkspace(t, N, DIM), false);
        }
        // newest=7, oldest=5. lag=3 → target=4, which is outside window.
        assertThatThrownBy(() -> FixedLag.compute(history, 3, FixedLag.identity1D()))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void traceAncestors_nullHistory_throws() {
        assertThatThrownBy(() -> FixedLag.traceAncestors(null, 0))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void traceAncestors_negativeLag_throws() {
        Rolling history = new Rolling(3, N, DIM);
        history.save(0, makeWorkspace(0, N, DIM), false);
        assertThatThrownBy(() -> FixedLag.traceAncestors(history, -1))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void multiDimensional_gatherCorrectly() {
        int N = 4;
        int dim = 2;
        int L = 2;
        Rolling history = new Rolling(L, N, dim);

        // Step 0: particles laid out as (dim, N) = [[1,2,3,4], [5,6,7,8]]
        Workspace ws0 = Workspace.allocate(N, dim, HistoryMode.NONE, 0);
        ws0.X = new double[]{1, 2, 3, 4, 5, 6, 7, 8};
        ws0.logW = new double[]{0, 0, 0, 0};
        ws0.a = new int[]{0, 1, 2, 3};
        history.save(0, ws0, false);

        // Step 1: ancestors = [2, 2, 0, 0], uniform weights
        Workspace ws1 = Workspace.allocate(N, dim, HistoryMode.NONE, 0);
        ws1.X = new double[]{10, 20, 30, 40, 50, 60, 70, 80};
        ws1.logW = new double[]{0, 0, 0, 0};
        ws1.a = new int[]{2, 2, 0, 0};
        history.save(1, ws1, true);

        // Lag=1: trace from step 1 → step 0.
        // B starts [0,1,2,3]. ancestors(1)=[2,2,0,0]. B=[2,2,0,0].
        // Particles at step 0: dim0=[1,2,3,4], dim1=[5,6,7,8].
        // Gathered: dim0=[3,3,1,1], dim1=[7,7,5,5].
        // Use a custom phi that sums both dimensions.
        FixedLag.Phi sumDims = (particles, d, n, out) -> {
            for (int i = 0; i < n; i++) {
                out[i] = 0;
                for (int j = 0; j < d; j++) {
                    out[i] += particles[j * n + i];
                }
            }
        };
        // Gathered sums: [3+7, 3+7, 1+5, 1+5] = [10, 10, 6, 6]
        // Mean = (10+10+6+6)/4 = 8.0
        double result = FixedLag.compute(history, 1, sumDims);
        assertThat(result).isCloseTo(8.0, org.assertj.core.data.Offset.offset(1e-12));
    }
}
