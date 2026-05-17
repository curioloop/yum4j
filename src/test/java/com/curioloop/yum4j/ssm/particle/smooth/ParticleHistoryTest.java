package com.curioloop.yum4j.ssm.particle.smooth;

import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Unit tests for the new {@link ParticleHistory} sealed interface and its
 * implementations: {@link Full}, {@link Rolling}, {@link Partial}.
 */
class ParticleHistoryTest {

    private static final int N = 8;
    private static final int DIM = 2;

    /** Creates a workspace and fills it with deterministic test data for step t. */
    private Workspace makeWorkspace(int t) {
        Workspace ws = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        // Fill X with t-based pattern
        for (int i = 0; i < N * DIM; i++) {
            ws.X[i] = t * 100.0 + i;
        }
        // Fill logW with t-based pattern
        for (int i = 0; i < N; i++) {
            ws.logW[i] = -(t + 1) * 0.1 * (i + 1);
        }
        // Fill ancestors with t-based pattern
        for (int i = 0; i < N; i++) {
            ws.a[i] = (t + i) % N;
        }
        return ws;
    }

    // ── Full ────────────────────────────────────────────────────────────

    @Test
    void full_constructionValidation() {
        assertThatThrownBy(() -> new Full(0, DIM, 10))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Full(N, 0, 10))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Full(N, DIM, 0))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void full_capacityAndDimensions() {
        Full h = new Full(N, DIM, 5);
        assertThat(h.N()).isEqualTo(N);
        assertThat(h.dim()).isEqualTo(DIM);
        assertThat(h.capacity()).isEqualTo(5);
        assertThat(h.T()).isEqualTo(5);
    }

    @Test
    void full_saveAndRetrieve() {
        int T = 5;
        // maxResampleRate=1.0 so the slot pool can hold all-resampled saves.
        Full h = new Full(N, DIM, T, 1.0);

        for (int t = 0; t < T; t++) {
            // After the post-task-3.2 rewrite hasStep tracks savedCount, not T.
            assertThat(h.hasStep(t)).isFalse();
            Workspace ws = makeWorkspace(t);
            // Pass resampled=true so the non-identity ws.a permutation actually
            // gets stored in the slot pool; this exercises the resample-store
            // path that the assertions below depend on.
            h.save(t, ws, true);
            assertThat(h.hasStep(t)).isTrue();
        }

        // Verify retrieval
        double[] x = new double[N * DIM];
        double[] lw = new double[N];
        int[] anc = new int[N];
        for (int t = 0; t < T; t++) {
            h.viewX(t, x, 0);
            for (int i = 0; i < N * DIM; i++) {
                assertThat(x[i]).isEqualTo(t * 100.0 + i);
            }

            h.viewLogW(t, lw, 0);
            for (int i = 0; i < N; i++) {
                assertThat(lw[i]).isEqualTo(-(t + 1) * 0.1 * (i + 1));
            }

            h.viewAncestors(t, anc, 0);
            for (int i = 0; i < N; i++) {
                assertThat(anc[i]).isEqualTo((t + i) % N);
            }
        }
    }

    @Test
    void full_outOfRangeThrows() {
        Full h = new Full(N, DIM, 5);
        double[] xBuf = new double[N * DIM];
        assertThatThrownBy(() -> h.save(-1, makeWorkspace(0), false))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> h.save(5, makeWorkspace(0), false))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> h.viewX(-1, xBuf, 0))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> h.viewX(5, xBuf, 0))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void full_hasStepCoversRange() {
        // Post-task-3.2: hasStep tracks savedCount, not T.
        Full h = new Full(N, DIM, 5);
        assertThat(h.hasStep(-1)).isFalse();
        assertThat(h.hasStep(0)).isFalse();
        assertThat(h.hasStep(4)).isFalse();
        assertThat(h.hasStep(5)).isFalse();

        // After saving step 0 only that step is available.
        h.save(0, makeWorkspace(0), false);
        assertThat(h.hasStep(0)).isTrue();
        assertThat(h.hasStep(1)).isFalse();

        // After saving step 4 (skipping 1..3) all steps up to 4 are reachable
        // since savedCount is the high-water mark.
        h.save(4, makeWorkspace(4), false);
        assertThat(h.hasStep(4)).isTrue();
        assertThat(h.hasStep(5)).isFalse();
    }

    @Test
    void full_rawAccessors() {
        Full h = new Full(N, DIM, 3);
        assertThat(h.rawX()).hasSize(3 * DIM * N);
        assertThat(h.rawLw()).hasSize(3 * N);
        // rawA() was removed in task 3.2 — ancestor backing is now a slot pool
        // (char[] or int[]) sized to maxResampledSlots * N, with non-resampled
        // steps synthesised on read. Use viewAncestors with a scratch buffer.
    }

    /**
     * Canonical mixed-step round-trip: T=5, N=3, with the engine resampling
     * exactly once at t=2 (permutation [2, 0, 1]) and identity ancestors at
     * t in {0, 1, 3, 4}. Catches off-by-one slot-indexing errors in the
     * compressed ancestor pool.
     */
    @Test
    void full_canonicalMixedStepRoundTrip() {
        int N3 = 3;
        int T = 5;
        Full h = new Full(N3, DIM, T, 0.6); // slots = ceil(0.6 * 5) = 3, only 1 used

        Workspace ws = Workspace.allocate(N3, DIM, HistoryMode.NONE, 0);

        // t=0..1, 3..4: identity ancestors (resampled=false)
        // t=2: resample with permutation [2, 0, 1]
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < N3 * DIM; i++) ws.X[i] = t * 100.0 + i;
            for (int i = 0; i < N3; i++) ws.logW[i] = -t - 0.5 * i;
            if (t == 2) {
                ws.a[0] = 2;
                ws.a[1] = 0;
                ws.a[2] = 1;
                h.save(t, ws, true);
            } else {
                // Ancestors written to ws.a are ignored on identity-skip steps
                // — the compressed Full synthesises identity at view time.
                ws.a[0] = 99;
                ws.a[1] = 99;
                ws.a[2] = 99;
                h.save(t, ws, false);
            }
        }

        // Verify ancestors round-trip.
        int[] anc = new int[N3];
        h.viewAncestors(0, anc, 0);
        assertThat(anc).containsExactly(0, 1, 2); // identity
        h.viewAncestors(1, anc, 0);
        assertThat(anc).containsExactly(0, 1, 2); // identity
        h.viewAncestors(2, anc, 0);
        assertThat(anc).containsExactly(2, 0, 1); // resample permutation
        h.viewAncestors(3, anc, 0);
        assertThat(anc).containsExactly(0, 1, 2); // identity
        h.viewAncestors(4, anc, 0);
        assertThat(anc).containsExactly(0, 1, 2); // identity

        // Verify resampledAt agrees with the resample bitset.
        assertThat(h.resampledAt(0)).isFalse();
        assertThat(h.resampledAt(1)).isFalse();
        assertThat(h.resampledAt(2)).isTrue();
        assertThat(h.resampledAt(3)).isFalse();
        assertThat(h.resampledAt(4)).isFalse();

        // viewX and viewLogW round-trip unchanged.
        double[] xBuf = new double[N3 * DIM];
        double[] lwBuf = new double[N3];
        for (int t = 0; t < T; t++) {
            h.viewX(t, xBuf, 0);
            for (int i = 0; i < N3 * DIM; i++) {
                assertThat(xBuf[i]).isEqualTo(t * 100.0 + i);
            }
            h.viewLogW(t, lwBuf, 0);
            for (int i = 0; i < N3; i++) {
                assertThat(lwBuf[i]).isEqualTo(-t - 0.5 * i);
            }
        }
    }

    @Test
    void full_overflowingResamplesThrows() {
        // maxResampleRate=0.1, T=10 -> maxResampledSlots = ceil(0.1 * 10) = 1.
        Full h = new Full(N, DIM, 10, 0.1);
        h.save(0, makeWorkspace(0), true);
        // Second resampled save must trip the budget.
        assertThatThrownBy(() -> h.save(1, makeWorkspace(1), true))
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("maxResampleRate");
    }

    @Test
    void full_constructorRejectsBadResampleRate() {
        assertThatThrownBy(() -> new Full(N, DIM, 10, 0.0))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Full(N, DIM, 10, -0.1))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Full(N, DIM, 10, 1.5))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void full_charBackedAtSmallN() {
        // N=4 ≤ 65535 → ancestor backing is char[].
        Full h = new Full(4, DIM, 3, 1.0);
        assertThat(h.charBacked()).isTrue();
    }

    @Test
    void full_intBackedAtLargeN() {
        // N=70_000 > 65535 → ancestor backing is int[]. T=2 keeps the buffer
        // size manageable (slots=2, N=70_000 → 2 * 70_000 * 4 = 560 KB).
        Full h = new Full(70_000, 1, 2, 1.0);
        assertThat(h.charBacked()).isFalse();
    }

    // ── Rolling ─────────────────────────────────────────────────────────

    @Test
    void rolling_constructionValidation() {
        assertThatThrownBy(() -> new Rolling(0, N, DIM))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Rolling(3, 0, DIM))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Rolling(3, N, 0))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rolling_capacityAndDimensions() {
        Rolling h = new Rolling(3, N, DIM);
        assertThat(h.N()).isEqualTo(N);
        assertThat(h.dim()).isEqualTo(DIM);
        assertThat(h.capacity()).isEqualTo(3);
        assertThat(h.lag()).isEqualTo(3);
    }

    @Test
    void rolling_saveAndRetrieveWithinWindow() {
        int L = 3;
        // maxResampleRate=1.0 so all-resampled saves fit the slot pool.
        Rolling h = new Rolling(L, N, DIM, 1.0);

        // Save steps 0, 1, 2 with resampled=true so the non-identity
        // ws.a permutation set by makeWorkspace is actually retained
        // by the post-task-3.3 identity-skip storage.
        for (int t = 0; t < L; t++) {
            h.save(t, makeWorkspace(t), true);
        }

        assertThat(h.newest()).isEqualTo(2);
        assertThat(h.oldest()).isEqualTo(0);

        // All three steps should be available
        double[] xBuf = new double[N * DIM];
        int[] ancBuf = new int[N];
        for (int t = 0; t < L; t++) {
            assertThat(h.hasStep(t)).isTrue();
            h.viewX(t, xBuf, 0);
            assertThat(xBuf[0]).isEqualTo(t * 100.0);
            h.viewAncestors(t, ancBuf, 0);
            assertThat(ancBuf[0]).isEqualTo(t % N);
        }
    }

    @Test
    void rolling_evictsOldestOnOverflow() {
        int L = 3;
        Rolling h = new Rolling(L, N, DIM);

        // Save steps 0..4
        for (int t = 0; t < 5; t++) {
            h.save(t, makeWorkspace(t), false);
        }

        assertThat(h.newest()).isEqualTo(4);
        assertThat(h.oldest()).isEqualTo(2);

        // Steps 0, 1 should be evicted
        assertThat(h.hasStep(0)).isFalse();
        assertThat(h.hasStep(1)).isFalse();
        assertThat(h.hasStep(2)).isTrue();
        assertThat(h.hasStep(3)).isTrue();
        assertThat(h.hasStep(4)).isTrue();

        // Verify data integrity for retained steps
        double[] xBuf = new double[N * DIM];
        for (int t = 2; t <= 4; t++) {
            h.viewX(t, xBuf, 0);
            assertThat(xBuf[0]).isEqualTo(t * 100.0);
        }
    }

    @Test
    void rolling_outOfWindowThrows() {
        Rolling h = new Rolling(3, N, DIM);
        for (int t = 0; t < 5; t++) {
            h.save(t, makeWorkspace(t), false);
        }
        double[] xBuf = new double[N * DIM];
        double[] lwBuf = new double[N];
        int[] ancBuf = new int[N];
        // Step 0 is evicted
        assertThatThrownBy(() -> h.viewX(0, xBuf, 0))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> h.viewLogW(1, lwBuf, 0))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> h.viewAncestors(1, ancBuf, 0))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rolling_nonIncreasingTThrows() {
        Rolling h = new Rolling(3, N, DIM);
        h.save(0, makeWorkspace(0), false);
        h.save(1, makeWorkspace(1), false);
        assertThatThrownBy(() -> h.save(1, makeWorkspace(1), false))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> h.save(0, makeWorkspace(0), false))
            .isInstanceOf(IllegalArgumentException.class);
    }

    // ── Partial ─────────────────────────────────────────────────────────

    @Test
    void partial_constructionValidation() {
        assertThatThrownBy(() -> new Partial(0, DIM, 10))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Partial(N, 0, 10))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Partial(N, DIM, 0))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void partial_capacityAndDimensions() {
        Partial h = new Partial(N, DIM, 10);
        assertThat(h.N()).isEqualTo(N);
        assertThat(h.dim()).isEqualTo(DIM);
        assertThat(h.capacity()).isEqualTo(10);
        assertThat(h.T()).isEqualTo(10);
    }

    @Test
    void partial_savesOnlyAncestors() {
        int T = 5;
        Partial h = new Partial(N, DIM, T, 1.0);

        // Save with resampled=true so the ancestor permutation is retained
        // (identity-skip would otherwise discard ws.a on resampled=false).
        for (int t = 0; t < T; t++) {
            h.save(t, makeWorkspace(t), true);
        }

        // Ancestors should be retrievable
        int[] anc = new int[N];
        for (int t = 0; t < T; t++) {
            h.viewAncestors(t, anc, 0);
            for (int i = 0; i < N; i++) {
                assertThat(anc[i]).isEqualTo((t + i) % N);
            }
        }
    }

    @Test
    void partial_identityStepReturnsIdentity() {
        int T = 5;
        Partial h = new Partial(N, DIM, T);

        // Save every step as non-resampled — viewAncestors must yield identity.
        for (int t = 0; t < T; t++) {
            h.save(t, makeWorkspace(t), false);
        }

        int[] anc = new int[N];
        for (int t = 0; t < T; t++) {
            h.viewAncestors(t, anc, 0);
            for (int i = 0; i < N; i++) {
                assertThat(anc[i]).isEqualTo(i);
            }
            assertThat(h.resampledAt(t)).isFalse();
        }
    }

    @Test
    void partial_mixedIdentityAndResampleSteps() {
        int T = 10;
        Partial h = new Partial(N, DIM, T);

        // Resample at t = 3 and t = 7; identity at all other steps.
        for (int t = 0; t < T; t++) {
            boolean resampled = (t == 3 || t == 7);
            h.save(t, makeWorkspace(t), resampled);
        }

        int[] anc = new int[N];
        for (int t = 0; t < T; t++) {
            h.viewAncestors(t, anc, 0);
            if (t == 3 || t == 7) {
                assertThat(h.resampledAt(t)).isTrue();
                for (int i = 0; i < N; i++) {
                    assertThat(anc[i]).isEqualTo((t + i) % N);
                }
            } else {
                assertThat(h.resampledAt(t)).isFalse();
                for (int i = 0; i < N; i++) {
                    assertThat(anc[i]).isEqualTo(i);
                }
            }
        }
    }

    @Test
    void partial_charBackedNarrowingRoundTrips() {
        int N4 = 4;
        int T = 3;
        Partial h = new Partial(N4, DIM, T, 1.0);

        // Hand-craft a workspace with N=4 particles so we can verify
        // char-narrowing returns int values that equal the original.
        Workspace ws = Workspace.allocate(N4, DIM, HistoryMode.NONE, 0);
        for (int i = 0; i < N4; i++) ws.a[i] = (i * 3 + 1) % N4;
        h.save(0, ws, true);
        for (int i = 0; i < N4; i++) ws.a[i] = (N4 - 1 - i);
        h.save(1, ws, true);
        for (int i = 0; i < N4; i++) ws.a[i] = i;
        h.save(2, ws, true);

        int[] anc = new int[N4];
        h.viewAncestors(0, anc, 0);
        assertThat(anc).containsExactly(1, 0, 3, 2);
        h.viewAncestors(1, anc, 0);
        assertThat(anc).containsExactly(3, 2, 1, 0);
        h.viewAncestors(2, anc, 0);
        assertThat(anc).containsExactly(0, 1, 2, 3);
    }

    @Test
    void partial_overflowingResamplesThrows() {
        // maxResampleRate=0.1, T=10 -> maxResampledSlots = ceil(0.1 * 10) = 1.
        Partial h = new Partial(N, DIM, 10, 0.1);

        // First resampled save fits.
        h.save(0, makeWorkspace(0), true);
        // Second resampled save must trip the budget.
        assertThatThrownBy(() -> h.save(1, makeWorkspace(1), true))
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("maxResampledSlots");
    }

    @Test
    void partial_constructorRejectsBadResampleRate() {
        assertThatThrownBy(() -> new Partial(N, DIM, 10, 0.0))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Partial(N, DIM, 10, -0.1))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Partial(N, DIM, 10, 1.5))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void partial_xThrowsUnsupported() {
        Partial h = new Partial(N, DIM, 5);
        h.save(0, makeWorkspace(0), false);
        double[] xBuf = new double[N * DIM];
        assertThatThrownBy(() -> h.viewX(0, xBuf, 0))
            .isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void partial_logWThrowsUnsupported() {
        Partial h = new Partial(N, DIM, 5);
        h.save(0, makeWorkspace(0), false);
        double[] lwBuf = new double[N];
        assertThatThrownBy(() -> h.viewLogW(0, lwBuf, 0))
            .isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void partial_outOfRangeThrows() {
        Partial h = new Partial(N, DIM, 5);
        int[] ancBuf = new int[N];
        assertThatThrownBy(() -> h.save(-1, makeWorkspace(0), false))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> h.save(5, makeWorkspace(0), false))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> h.viewAncestors(-1, ancBuf, 0))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> h.viewAncestors(5, ancBuf, 0))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void partial_hasStepCoversRange() {
        Partial h = new Partial(N, DIM, 5);
        // Pre-save: nothing is stored.
        assertThat(h.hasStep(-1)).isFalse();
        assertThat(h.hasStep(0)).isFalse();

        // After saving up to t=2, only [0, 2] are available.
        h.save(0, makeWorkspace(0), false);
        h.save(1, makeWorkspace(1), false);
        h.save(2, makeWorkspace(2), false);
        assertThat(h.hasStep(-1)).isFalse();
        assertThat(h.hasStep(0)).isTrue();
        assertThat(h.hasStep(2)).isTrue();
        assertThat(h.hasStep(3)).isFalse();
        assertThat(h.hasStep(5)).isFalse();
    }

    // ── Workspace integration ───────────────────────────────────────────

    @Test
    void workspace_allocatesFull() {
        Workspace ws = Workspace.allocate(N, DIM, HistoryMode.FULL, 10);
        assertThat(ws.history).isNotNull();
        assertThat(ws.history).isInstanceOf(Full.class);
        assertThat(ws.history.capacity()).isEqualTo(10);
    }

    @Test
    void workspace_allocatesRolling() {
        Workspace ws = Workspace.allocate(N, DIM, HistoryMode.ROLLING, 5);
        assertThat(ws.history).isNotNull();
        assertThat(ws.history).isInstanceOf(Rolling.class);
        assertThat(ws.history.capacity()).isEqualTo(5);
    }

    @Test
    void workspace_allocatesPartial() {
        Workspace ws = Workspace.allocate(N, DIM, HistoryMode.PARTIAL, 10);
        assertThat(ws.history).isNotNull();
        assertThat(ws.history).isInstanceOf(Partial.class);
        assertThat(ws.history.capacity()).isEqualTo(10);
    }

    @Test
    void workspace_noneHistoryIsNull() {
        Workspace ws = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        assertThat(ws.history).isNull();
    }

    @Test
    void workspace_fullHistorySaveAndRetrieve() {
        int T = 3;
        Workspace ws = Workspace.allocate(N, DIM, HistoryMode.FULL, T);
        Full history = (Full) ws.history;

        // Simulate engine saving steps
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < N * DIM; i++) ws.X[i] = t * 10.0 + i;
            for (int i = 0; i < N; i++) ws.logW[i] = -t * 0.5;
            for (int i = 0; i < N; i++) ws.a[i] = (t + i) % N;
            // resampled=false — the engine's identity-skip path. ws.a is
            // discarded; viewAncestors synthesises identity at view time.
            history.save(t, ws, false);
        }

        // Verify
        double[] x0 = new double[DIM * N];
        history.viewX(0, x0, 0);
        assertThat(x0[0]).isEqualTo(0.0);
        assertThat(x0[1]).isEqualTo(1.0);

        double[] x2 = new double[DIM * N];
        history.viewX(2, x2, 0);
        assertThat(x2[0]).isEqualTo(20.0);

        double[] lw1 = new double[N];
        history.viewLogW(1, lw1, 0);
        assertThat(lw1[0]).isEqualTo(-0.5);

        int[] a1 = new int[N];
        history.viewAncestors(1, a1, 0);
        // Identity-skip: a non-resampled save yields the identity permutation
        // (0, 1, ..., N-1) regardless of what was in ws.a.
        for (int i = 0; i < N; i++) {
            assertThat(a1[i]).isEqualTo(i);
        }
    }
}
