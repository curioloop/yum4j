package com.curioloop.yum4j.ssm.particle.smooth;

import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Self-validation tests for {@link Rolling}'s post-task-3.3 ancestor
 * compression and circular slot pool, covering:
 *
 * <ul>
 *   <li>Pure identity sequence — every {@code viewAncestors} returns the
 *       identity permutation regardless of how far the rolling window has
 *       shifted past slot 0.</li>
 *   <li>Single resample at the middle of a window — the resampled step's
 *       ancestor permutation round-trips while it's in the window, then
 *       falls outside on subsequent saves.</li>
 *   <li>Wrap-invariant violation — when the configured
 *       {@code maxResampleRate} under-estimates the actual rate, the
 *       circular pool wraps and would silently corrupt a still-queryable
 *       slot. With assertions enabled this surfaces as
 *       {@link AssertionError}.</li>
 * </ul>
 */
class RollingCompressionTest {

    private static final int N = 4;
    private static final int DIM = 1;

    /** Workspace primed for step {@code t} with deterministic data. */
    private Workspace makeWorkspace(int t, int[] ancestors) {
        Workspace ws = Workspace.allocate(N, DIM, HistoryMode.NONE, 0);
        for (int i = 0; i < N; i++) ws.X[i] = t * 100.0 + i;
        for (int i = 0; i < N; i++) ws.logW[i] = -t * 0.1 - i * 0.01;
        System.arraycopy(ancestors, 0, ws.a, 0, N);
        return ws;
    }

    private static int[] identity(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = i;
        return out;
    }

    private static int[] reverse(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = n - 1 - i;
        return out;
    }

    /**
     * With L=4 and 6 saves all marked {@code resampled=false},
     * {@link Rolling#viewAncestors} must always synthesise the identity
     * permutation, even after the window has shifted past slot 0.
     */
    @Test
    void identitySequence_alwaysReturnsIdentity() {
        int L = 4;
        int T = 6;
        Rolling h = new Rolling(L, N, DIM);

        int[] anc = new int[N];
        for (int t = 0; t < T; t++) {
            // Workspace ancestors carry a non-identity permutation so the
            // test would fail if Rolling stored ws.a despite resampled=false.
            h.save(t, makeWorkspace(t, reverse(N)), false);

            for (int s = h.oldest(); s <= h.newest(); s++) {
                h.viewAncestors(s, anc, 0);
                for (int n = 0; n < N; n++) {
                    assertThat(anc[n])
                        .as("step %d ancestor[%d]", s, n)
                        .isEqualTo(n);
                }
                assertThat(h.resampledAt(s)).isFalse();
            }
        }

        // After the last save the window is [2, 5]; step 0 must throw.
        assertThat(h.oldest()).isEqualTo(2);
        assertThat(h.newest()).isEqualTo(5);
        int[] buf = new int[N];
        assertThatThrownBy(() -> h.viewAncestors(0, buf, 0))
            .isInstanceOf(IllegalArgumentException.class);
    }

    /**
     * One resampled step at t=2, identity at all other steps, L=4, T=6.
     * While t=2 is in the window the stored permutation round-trips; once
     * the window has moved past it (by t=5's save) viewAncestors(2) must
     * throw with a window-bounds error.
     */
    @Test
    void singleResample_roundTripsThenFallsOutOfWindow() {
        int L = 4;
        int T = 6;
        Rolling h = new Rolling(L, N, DIM);
        int[] perm = reverse(N);

        for (int t = 0; t < T; t++) {
            boolean resampled = (t == 2);
            int[] anc = resampled ? perm : identity(N);
            h.save(t, makeWorkspace(t, anc), resampled);

            // While t=2 is still in the window, its permutation round-trips.
            if (h.hasStep(2)) {
                int[] back = new int[N];
                h.viewAncestors(2, back, 0);
                assertThat(back).containsExactly(perm);
                assertThat(h.resampledAt(2)).isTrue();
            }
        }

        // After saves through t=5 the window is [2, 5] — t=2 is the oldest.
        assertThat(h.oldest()).isEqualTo(2);
        assertThat(h.newest()).isEqualTo(5);

        // Save one more step (t=6) so the window becomes [3, 6] and t=2 falls out.
        h.save(6, makeWorkspace(6, identity(N)), false);
        assertThat(h.oldest()).isEqualTo(3);
        int[] buf = new int[N];
        assertThatThrownBy(() -> h.viewAncestors(2, buf, 0))
            .isInstanceOf(IllegalArgumentException.class);
    }

    /**
     * With L=4 and {@code maxResampleRate=0.5}, the slot pool holds 2
     * resample slots. Saving four resamples in a row exceeds the
     * configured rate (actual rate is 1.0) — when nextSlot wraps from 1
     * back to 0 (third save), it overwrites slot 0 which was last
     * written by t=0, but t=0 is still inside the window [0, 2]. With
     * assertions enabled this surfaces as {@link AssertionError}; the
     * production-mode behavior (silent corruption) is documented in
     * the class Javadoc.
     */
    @Test
    void wrapInvariant_violationFiresAssertion() {
        // Pre-flight: only run this assertion check when the JVM has -ea on.
        // mvn surefire enables assertions by default, but skip cleanly otherwise
        // so the test does not produce a false positive on production launches.
        boolean ea = false;
        assert ea = true;
        if (!ea) return;

        int L = 4;
        Rolling h = new Rolling(L, N, DIM, 0.5);
        // maxResampledSlots = ceil(0.5 * 4) = 2.
        assertThat(h.maxResampledSlots()).isEqualTo(2);

        // First two saves consume slots 0 and 1 — both are still in window
        // [0, 1] but no wrap has happened yet.
        h.save(0, makeWorkspace(0, reverse(N)), true);
        h.save(1, makeWorkspace(1, reverse(N)), true);

        // Third save tries to wrap nextSlot 2 → ancestorSlot 0. Slot 0 was
        // last written by t=0, and t=0 is still in window [0, 2] (oldest=0).
        // → wrap invariant violated → AssertionError under -ea.
        assertThatThrownBy(() -> h.save(2, makeWorkspace(2, reverse(N)), true))
            .isInstanceOf(AssertionError.class)
            .hasMessageContaining("ancestor pool wrap");
    }

    /**
     * When the configured rate matches the actual rate, the wrap is
     * legitimate: slot 0 is overwritten only after t=0 has fallen out
     * of the window. This is the intended steady-state behavior.
     */
    @Test
    void wrapInvariant_legitimateWrapDoesNotFire() {
        // L = 4, rate = 0.5 → 2 slots. Resample every other step so
        // between two consecutive resamples to the same slot the window
        // has shifted past the previous holder.
        int L = 4;
        Rolling h = new Rolling(L, N, DIM, 0.5);

        // t = 0 (resample, slot 0), t = 1 (identity), t = 2 (resample, slot 1),
        // t = 3 (identity), t = 4 (resample, wrap to slot 0). At t=4 the
        // window is [1, 4], so t=0 has fallen out — wrap is legitimate.
        h.save(0, makeWorkspace(0, reverse(N)), true);
        h.save(1, makeWorkspace(1, identity(N)), false);
        h.save(2, makeWorkspace(2, reverse(N)), true);
        h.save(3, makeWorkspace(3, identity(N)), false);
        h.save(4, makeWorkspace(4, reverse(N)), true);  // wraps to slot 0

        assertThat(h.oldest()).isEqualTo(1);
        assertThat(h.newest()).isEqualTo(4);
        // viewAncestors(0) should throw because t=0 is outside the window;
        // the slot reuse is therefore correct.
        int[] buf = new int[N];
        assertThatThrownBy(() -> h.viewAncestors(0, buf, 0))
            .isInstanceOf(IllegalArgumentException.class);

        // The new resample at t=4 lives in slot 0; viewAncestors(4) must
        // round-trip the permutation we just stored.
        h.viewAncestors(4, buf, 0);
        assertThat(buf).containsExactly(reverse(N));
    }

    /** Construction validates the {@code maxResampleRate} bound. */
    @Test
    void constructorRejectsBadResampleRate() {
        assertThatThrownBy(() -> new Rolling(4, N, DIM, 0.0))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Rolling(4, N, DIM, -0.5))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new Rolling(4, N, DIM, 1.5))
            .isInstanceOf(IllegalArgumentException.class);
        // Boundary values are accepted.
        new Rolling(4, N, DIM, 1.0);
    }

    /**
     * char-backed path: with N <= 65535 the ancestor pool uses {@code char[]},
     * so values must round-trip through the widening cast at view time.
     */
    @Test
    void charBackedNarrowing_roundTrips() {
        int L = 3;
        Rolling h = new Rolling(L, N, DIM, 1.0);
        int[] perm = {3, 1, 0, 2};
        h.save(0, makeWorkspace(0, perm), true);
        int[] buf = new int[N];
        h.viewAncestors(0, buf, 0);
        assertThat(buf).containsExactly(perm);
    }

    /**
     * int-backed path: with {@code N > 65535} the ancestor pool falls
     * back to {@code int[]}. Verify a single resample round-trips.
     */
    @Test
    void intBackedPath_roundTrips() {
        int bigN = 70_000;
        int L = 3;
        Rolling h = new Rolling(L, bigN, DIM, 1.0);
        Workspace ws = Workspace.allocate(bigN, DIM, HistoryMode.NONE, 0);
        for (int n = 0; n < bigN; n++) {
            ws.X[n] = n;
            ws.logW[n] = -n * 1e-3;
            ws.a[n] = (bigN - 1 - n);
        }
        h.save(0, ws, true);
        int[] buf = new int[bigN];
        h.viewAncestors(0, buf, 0);
        for (int n = 0; n < bigN; n++) {
            assertThat(buf[n]).isEqualTo(bigN - 1 - n);
        }
    }
}
