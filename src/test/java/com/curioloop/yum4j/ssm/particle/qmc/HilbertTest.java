package com.curioloop.yum4j.ssm.particle.qmc;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Unit tests for {@link Hilbert} primitives.
 * Ported from old smc/qmc/HilbertTest — the Hilbert class moved to filter.particle.qmc.
 */
class HilbertTest {

    @Test
    void invLogitSymmetry() {
        assertThat(Hilbert.invLogit(0.0)).isEqualTo(0.5);
        assertThat(Hilbert.invLogit(1.0) + Hilbert.invLogit(-1.0))
            .isCloseTo(1.0, within(1e-15));
        assertThat(Hilbert.invLogit(Double.POSITIVE_INFINITY)).isEqualTo(1.0);
        assertThat(Hilbert.invLogit(Double.NEGATIVE_INFINITY)).isEqualTo(0.0);
    }

    @Test
    void hilbertIndexD1IsIdentity() {
        for (long v : new long[]{0L, 1L, 7L, 42L, 1_000_000L}) {
            assertThat(Hilbert.hilbertIndex(new long[]{v}, 1)).isEqualTo(v);
        }
    }

    @Test
    void hilbertIndexD2OriginAndFirstStep() {
        assertThat(Hilbert.hilbertIndex(new long[]{0, 0}, 2)).isEqualTo(0L);
        assertThat(Hilbert.hilbertIndex(new long[]{1, 0}, 2)).isEqualTo(1L);
    }

    @Test
    void hilbertIndexIsInjectiveOnSmallGrid2D() {
        java.util.Set<Long> seen = new java.util.HashSet<>();
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) {
                long h = Hilbert.hilbertIndex(new long[]{x, y}, 2);
                assertThat(h).as("(%d,%d)", x, y).isBetween(0L, 15L);
                seen.add(h);
            }
        }
        assertThat(seen).hasSize(16);
    }

    @Test
    void hilbertSortD1MatchesArgsort() {
        double[] X = {3.0, 1.0, 4.0, 1.5, 2.0};
        int N = X.length;
        int[] perm = new int[N];
        double[] scratchD = new double[N];
        long[] scratchL = new long[N];
        Hilbert.hilbertSort(X, 0, 1, N, perm, scratchD, scratchL);
        assertThat(perm).containsExactly(1, 3, 4, 0, 2);
    }

    @Test
    void hilbertSortD2Deterministic() {
        double[] X = {0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0};
        int[] perm = new int[4];
        double[] scratchD = new double[2 * 4];
        long[] scratchL = new long[2 * 4];
        Hilbert.hilbertSort(X, 0, 2, 4, perm, scratchD, scratchL);
        int[] sorted = perm.clone();
        java.util.Arrays.sort(sorted);
        assertThat(sorted).containsExactly(0, 1, 2, 3);
    }
}
