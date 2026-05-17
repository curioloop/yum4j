package com.curioloop.yum4j.ssm.particle.kernel;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Tests for {@link LogWeight} — ported from the old WeightMathTest and LogSumExpTest.
 * Covers logSumExp, essFromLog, shiftIfOverflowing, and the fused logSumEssMax reduction.
 */
class LogWeightTest {

    private static final double TOL = 1e-10;

    // ---- logSumExp --------------------------------------------------

    @Test
    void logSumExp_emptyReturnsNegativeInfinity() {
        assertThat(LogWeight.logSumExp(new double[0], 0, 0)).isEqualTo(Double.NEGATIVE_INFINITY);
    }

    @Test
    void logSumExp_singleElementReturnsThatElement() {
        assertThat(LogWeight.logSumExp(new double[]{3.14}, 0, 1)).isCloseTo(3.14, within(1e-12));
        assertThat(LogWeight.logSumExp(new double[]{-1e9}, 0, 1)).isCloseTo(-1e9, within(1e-12));
    }

    @Test
    void logSumExp_allEqualReturnsVPlusLogN() {
        double v = 2.5;
        int n = 17;
        double[] lw = new double[n];
        java.util.Arrays.fill(lw, v);
        assertThat(LogWeight.logSumExp(lw, 0, n)).isCloseTo(v + Math.log(n), within(1e-12));
    }

    @Test
    void logSumExp_uniformWeightsAtZeroGivesLogN() {
        double[] lw = new double[512];
        assertThat(LogWeight.logSumExp(lw, 0, 512)).isCloseTo(Math.log(512), within(1e-12));
    }

    @Test
    void logSumExp_mixedMagnitudesMatchesHandComputed() {
        double[] lw = {0.0, 1.0, 2.0, 3.0};
        double m = 3.0;
        double expected = m + Math.log(Math.exp(-3) + Math.exp(-2) + Math.exp(-1) + 1.0);
        assertThat(LogWeight.logSumExp(lw, 0, 4)).isCloseTo(expected, within(1e-12));
    }

    @Test
    void logSumExp_positiveInfinityDominates() {
        double[] lw = {0.0, 1.0, Double.POSITIVE_INFINITY, 3.0};
        assertThat(LogWeight.logSumExp(lw, 0, 4)).isEqualTo(Double.POSITIVE_INFINITY);
    }

    @Test
    void logSumExp_allNegativeInfinityReturnsNegativeInfinity() {
        double[] lw = {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};
        assertThat(LogWeight.logSumExp(lw, 0, 3)).isEqualTo(Double.NEGATIVE_INFINITY);
    }

    @Test
    void logSumExp_singleFinitePlusNegInfsReducesToFiniteValue() {
        double[] lw = {Double.NEGATIVE_INFINITY, 2.0, Double.NEGATIVE_INFINITY};
        assertThat(LogWeight.logSumExp(lw, 0, 3)).isCloseTo(2.0, within(1e-12));
    }

    @Test
    void logSumExp_simdPathMatchesScalarPathAcrossLengths() {
        java.util.Random rng = new java.util.Random(42);
        for (int n : new int[]{1, 2, 3, 7, 8, 9, 16, 31, 32, 33, 64, 100, 257}) {
            double[] lw = new double[n];
            for (int i = 0; i < n; i++) lw[i] = rng.nextGaussian() * 3.0;
            double simd = LogWeight.logSumExp(lw, 0, n);
            // scalar reference
            double max = Double.NEGATIVE_INFINITY;
            for (double v : lw) if (v > max) max = v;
            double sum = 0;
            for (double v : lw) sum += Math.exp(v - max);
            double scalar = max + Math.log(sum);
            assertThat(simd).isCloseTo(scalar, within(1e-12));
        }
    }

    // ---- essFromLog -------------------------------------------------

    @Test
    void essForUniformWeightsEqualsN() {
        int n = 512;
        double[] lw = new double[n];
        double logSum = LogWeight.logSumExp(lw, 0, n);
        assertThat(LogWeight.essFromLog(lw, 0, n, logSum)).isCloseTo(n, within(TOL));
    }

    @Test
    void essForSingleFiniteEntryEqualsOne() {
        double[] lw = {Double.NEGATIVE_INFINITY, 0.0, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};
        double logSum = LogWeight.logSumExp(lw, 0, 4);
        assertThat(LogWeight.essFromLog(lw, 0, 4, logSum)).isCloseTo(1.0, within(TOL));
    }

    @Test
    void essForAllNegInfIsZero() {
        double[] lw = {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};
        assertThat(LogWeight.essFromLog(lw, 0, 2, Double.NEGATIVE_INFINITY)).isEqualTo(0.0);
    }

    @Test
    void essExtremeMagnitudesMatchNormalisedFormula() {
        double[] lw = {1000.0, 1001.0, 999.0};
        double logSum = LogWeight.logSumExp(lw, 0, 3);
        double ess = LogWeight.essFromLog(lw, 0, 3, logSum);
        double[] w = {Math.exp(0), Math.exp(1), Math.exp(-1)};
        double sw = w[0] + w[1] + w[2];
        double sw2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        double expected = (sw * sw) / sw2;
        assertThat(ess).isCloseTo(expected, within(1e-10));
    }

    // ---- shiftIfOverflowing -----------------------------------------

    @Test
    void shiftNoOpWhenMaxInSafeBand() {
        double[] lw = {-5.0, -100.0, -2.0};
        double[] copy = lw.clone();
        LogWeight.shiftIfOverflowing(lw, 0, 3);
        assertThat(lw).containsExactly(copy);
    }

    @Test
    void shiftNoOpOnEmpty() {
        double[] lw = new double[0];
        LogWeight.shiftIfOverflowing(lw, 0, 0);
        assertThat(lw).hasSize(0);
    }

    @Test
    void shiftNoOpOnAllNegInf() {
        double[] lw = {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};
        LogWeight.shiftIfOverflowing(lw, 0, 2);
        assertThat(lw).containsExactly(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
    }

    // ---- logSumEssMax (fused reduction) -----------------------------

    @Test
    void logSumEssMax_uniformWeights() {
        int n = 64;
        double[] lw = new double[n];
        LogWeight.Triple r = LogWeight.logSumEssMax(lw, 0, n);
        assertThat(r.logSum()).isCloseTo(Math.log(n), within(1e-12));
        assertThat(r.ess()).isCloseTo(n, within(TOL));
        assertThat(r.max()).isEqualTo(0.0);
    }

    @Test
    void logSumEssMax_singleFiniteEntry() {
        double[] lw = {Double.NEGATIVE_INFINITY, 5.0, Double.NEGATIVE_INFINITY};
        LogWeight.Triple r = LogWeight.logSumEssMax(lw, 0, 3);
        assertThat(r.logSum()).isCloseTo(5.0, within(1e-12));
        assertThat(r.ess()).isCloseTo(1.0, within(TOL));
        assertThat(r.max()).isEqualTo(5.0);
    }

    @Test
    void logSumEssMax_emptyReturnsDefaults() {
        LogWeight.Triple r = LogWeight.logSumEssMax(new double[0], 0, 0);
        assertThat(r.logSum()).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(r.ess()).isEqualTo(0.0);
        assertThat(r.max()).isEqualTo(Double.NEGATIVE_INFINITY);
    }
}
