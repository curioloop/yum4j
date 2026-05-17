package com.curioloop.yum4j.ssm.particle.resample;

import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import net.jqwik.api.ForAll;
import net.jqwik.api.Label;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.IntRange;
import net.jqwik.api.constraints.LongRange;

import java.util.HashSet;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Property-based tests for the resampling kernels.
 *
 * <p>Ported from the old smc.ResampleProperties to use the new
 * resample.Resample API with RandomBatch.
 *
 * <p>
 * Ancestors in {@code [0, N)}; unique count ≤ {@code min(N, M)}
 * single-particle degeneracy must not underflow).
 */
class ResampleProperties {

    // ---- Single-particle degeneracy ---------------------------------

    /**
     * Scratch size that covers every scheme's post-R7.4 per-scheme
     * contract: {@code max(3*N, N + M + 1)}. SSP at {@code M == N} needs
     * {@code 3*N - 1} (rounded up to {@code 3*N}); MULTINOMIAL needs
     * {@code N + M + 1}; all other schemes need {@code 2*N} or less.
     */
    private static double[] universalScratch(int N, int M) {
        return new double[Math.max(3 * N, N + M + 1)];
    }

    @Property(tries = 80)
    @Label("R6: systematic collapses to k when only lw[k] is finite")
    void systematicSingleParticleCollapsesToK(
            @ForAll @IntRange(min = 1, max = 512) int N,
            @ForAll @IntRange(min = 0, max = 511) int k,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        int kk = k % N;
        int M = N;
        double[] lw = singleFiniteLw(N, kk);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.SYSTEMATIC, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertAllEqualTo(a, kk);
    }

    @Property(tries = 80)
    @Label("R6: stratified collapses to k when only lw[k] is finite")
    void stratifiedSingleParticleCollapsesToK(
            @ForAll @IntRange(min = 1, max = 512) int N,
            @ForAll @IntRange(min = 0, max = 511) int k,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        int kk = k % N;
        int M = N;
        double[] lw = singleFiniteLw(N, kk);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.STRATIFIED, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertAllEqualTo(a, kk);
    }

    @Property(tries = 80)
    @Label("R6: multinomial collapses to k when only lw[k] is finite")
    void multinomialSingleParticleCollapsesToK(
            @ForAll @IntRange(min = 1, max = 512) int N,
            @ForAll @IntRange(min = 0, max = 511) int k,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        int kk = k % N;
        int M = N;
        double[] lw = singleFiniteLw(N, kk);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.MULTINOMIAL, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertAllEqualTo(a, kk);
    }

    @Property(tries = 80)
    @Label("R6: residual collapses to k when only lw[k] is finite")
    void residualSingleParticleCollapsesToK(
            @ForAll @IntRange(min = 1, max = 512) int N,
            @ForAll @IntRange(min = 0, max = 511) int k,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        int kk = k % N;
        int M = N;
        double[] lw = singleFiniteLw(N, kk);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.RESIDUAL, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertAllEqualTo(a, kk);
    }

    @Property(tries = 80)
    @Label("R6: ssp collapses to k when only lw[k] is finite")
    void sspSingleParticleCollapsesToK(
            @ForAll @IntRange(min = 1, max = 512) int N,
            @ForAll @IntRange(min = 0, max = 511) int k,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        int kk = k % N;
        int M = N;
        double[] lw = singleFiniteLw(N, kk);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.SSP, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertAllEqualTo(a, kk);
    }

    @Property(tries = 80)
    @Label("R6: killing collapses to k when only lw[k] is finite")
    void killingSingleParticleCollapsesToK(
            @ForAll @IntRange(min = 1, max = 512) int N,
            @ForAll @IntRange(min = 0, max = 511) int k,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        int kk = k % N;
        int M = N;
        double[] lw = singleFiniteLw(N, kk);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.KILLING, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertAllEqualTo(a, kk);
    }

    // ---- Range and uniqueness ---------------------------------------

    @Property(tries = 60)
    @Label("R6: systematic ancestors are in [0, N) and unique ≤ min(N, M)")
    void systematicRangeAndUniqueness(
            @ForAll @IntRange(min = 1, max = 300) int N,
            @ForAll @IntRange(min = 1, max = 300) int M,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        double[] lw = diverseLw(N, seed);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.SYSTEMATIC, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertInRangeAndBoundedUnique(a, N, M);
    }

    @Property(tries = 60)
    @Label("R6: stratified ancestors are in [0, N) and unique ≤ min(N, M)")
    void stratifiedRangeAndUniqueness(
            @ForAll @IntRange(min = 1, max = 300) int N,
            @ForAll @IntRange(min = 1, max = 300) int M,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        double[] lw = diverseLw(N, seed);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.STRATIFIED, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertInRangeAndBoundedUnique(a, N, M);
    }

    @Property(tries = 60)
    @Label("R6: multinomial ancestors are in [0, N) and unique ≤ min(N, M)")
    void multinomialRangeAndUniqueness(
            @ForAll @IntRange(min = 1, max = 300) int N,
            @ForAll @IntRange(min = 1, max = 300) int M,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        double[] lw = diverseLw(N, seed);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.MULTINOMIAL, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertInRangeAndBoundedUnique(a, N, M);
    }

    @Property(tries = 60)
    @Label("R6: residual ancestors are in [0, N) and unique ≤ min(N, M)")
    void residualRangeAndUniqueness(
            @ForAll @IntRange(min = 1, max = 300) int N,
            @ForAll @IntRange(min = 1, max = 300) int M,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        double[] lw = diverseLw(N, seed);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.RESIDUAL, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertInRangeAndBoundedUnique(a, N, M);
    }

    @Property(tries = 60)
    @Label("R6: ssp ancestors are in [0, N) and unique ≤ N (M == N)")
    void sspRangeAndUniqueness(
            @ForAll @IntRange(min = 1, max = 300) int N,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        int M = N;
        double[] lw = diverseLw(N, seed);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.SSP, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertInRangeAndBoundedUnique(a, N, M);
    }

    @Property(tries = 60)
    @Label("R6: killing ancestors are in [0, N) and unique ≤ N (M == N)")
    void killingRangeAndUniqueness(
            @ForAll @IntRange(min = 1, max = 300) int N,
            @ForAll @LongRange(min = 1L, max = Long.MAX_VALUE) long seed
    ) {
        int M = N;
        double[] lw = diverseLw(N, seed);
        int[] a = new int[M];
        double[] scratch = universalScratch(N, M);
        Resample.apply(Scheme.KILLING, lw, N, M, a, RandomBatch.of(seed), scratch);
        assertInRangeAndBoundedUnique(a, N, M);
    }

    // ---- helpers ----------------------------------------------------

    private static double[] singleFiniteLw(int N, int k) {
        double[] lw = new double[N];
        for (int i = 0; i < N; i++) lw[i] = Double.NEGATIVE_INFINITY;
        lw[k] = 0.0;
        return lw;
    }

    private static double[] diverseLw(int N, long seed) {
        java.util.Random g = new java.util.Random(seed ^ 0xABCDEF01L);
        double[] lw = new double[N];
        for (int i = 0; i < N; i++) lw[i] = (g.nextDouble() - 0.5) * 10.0;
        if (N > 4) {
            lw[0] = 50.0;
            lw[1] = -50.0;
        }
        return lw;
    }

    private static void assertAllEqualTo(int[] a, int expected) {
        for (int i = 0; i < a.length; i++) {
            assertThat(a[i]).as("a[%d]", i).isEqualTo(expected);
        }
    }

    private static void assertInRangeAndBoundedUnique(int[] a, int N, int M) {
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < M; i++) {
            assertThat(a[i]).as("a[%d] range", i).isGreaterThanOrEqualTo(0).isLessThan(N);
            seen.add(a[i]);
        }
        assertThat(seen.size()).as("unique count").isLessThanOrEqualTo(Math.min(N, M));
    }
}
