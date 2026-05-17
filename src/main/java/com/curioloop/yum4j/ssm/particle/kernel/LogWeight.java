package com.curioloop.yum4j.ssm.particle.kernel;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Fused log-weight reductions for the particle-filter engine.
 *
 * <p>Computes {@code (logSumExp, ESS, max)} in a single pass over a
 * log-weight array, using a SIMD two-accumulator pattern with
 * {@link DoubleVector#SPECIES_256} to keep both FP ports busy.
 *
 * <p>All methods are allocation-free on the hot path (the {@link Triple}
 * record is scalarised by the JIT). Ranges are expressed as
 * {@code (off, n)} to avoid slicing.
 *
 * <p><b>Replaces the outgoing {@code WeightMath}.</b>
 */
public final class LogWeight {

    private LogWeight() {}

    /**
     * Value carrier for the fused reduction triple.
     * The JIT scalarises this on the hot path.
     */
    public record Triple(double logSum, double ess, double max) {}

    // ------------------------------------------------------------------
    // SIMD configuration
    // ------------------------------------------------------------------

    private static final String SIMD_PROPERTY = "yum4j.vector";
    private static final boolean SIMD_ENABLED =
        !"false".equalsIgnoreCase(System.getProperty(SIMD_PROPERTY, "true"));

    private static final VectorSpecies<Double> SPECIES = preferredSpecies256Cap();
    private static final int LANES = SPECIES.length();

    /** Minimum length below which scalar loops win over SIMD setup cost. */
    static final int SIMD_MIN_N = 32;

    static final boolean SIMD_SUPPORTED = detectSimd();

    private static VectorSpecies<Double> preferredSpecies256Cap() {
        VectorSpecies<Double> preferred = DoubleVector.SPECIES_PREFERRED;
        if (preferred.length() >= DoubleVector.SPECIES_256.length()) {
            return DoubleVector.SPECIES_256;
        }
        return preferred;
    }

    private static boolean detectSimd() {
        if (!SIMD_ENABLED) return false;
        if (LANES <= 1) return false;
        try {
            double[] probe = new double[LANES * 2];
            for (int i = 0; i < probe.length; i++) probe[i] = (i % 4) * 0.25;
            simdReduce(probe, 0, probe.length);
            double[] incr = probe.clone();
            simdAddIntoAndReduce(probe, 0, incr, 0, probe.length);
            return true;
        } catch (LinkageError e) {
            return false;
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Fused logSumExp / ESS / max reduction over {@code lw[off..off+n)}.
     *
     * <p>Two passes: one for the max, one for {@code Σ exp(lw-m)} and
     * {@code Σ exp(2(lw-m))}. Equivalent to calling logSumExp + essFromLog
     * + max separately, but halves the memory traffic.
     *
     * <p>Semantics: NaN propagates, ±Inf collapses to same ±Inf with
     * ess=0, empty input returns {@code (NEGATIVE_INFINITY, 0, NEGATIVE_INFINITY)}.
     *
     * @param lw  log-weight buffer
     * @param off offset into {@code lw}
     * @param n   number of entries
     * @return {@code Triple(logSumExp, ess, max)}
     */
    public static Triple logSumEssMax(double[] lw, int off, int n) {
        if (n == 0) {
            return new Triple(Double.NEGATIVE_INFINITY, 0.0, Double.NEGATIVE_INFINITY);
        }
        if (SIMD_SUPPORTED && n >= SIMD_MIN_N) {
            return simdReduce(lw, off, n);
        }
        return scalarReduce(lw, off, n);
    }

    /**
     * Fused {@code lw += incr} and logSumExp / ESS / max reduction.
     *
     * <p>Two passes: the first accumulates the increment while tracking
     * the running max; the second computes the exponent sums.
     *
     * @param lw      log-weight buffer (mutated in place)
     * @param lwOff   offset into {@code lw}
     * @param incr    increment buffer (read only)
     * @param incrOff offset into {@code incr}
     * @param n       number of entries
     * @return {@code Triple(logSumExp, ess, max)}
     */
    public static Triple addIntoAndReduce(double[] lw, int lwOff,
                                          double[] incr, int incrOff, int n) {
        if (n == 0) {
            return new Triple(Double.NEGATIVE_INFINITY, 0.0, Double.NEGATIVE_INFINITY);
        }
        if (SIMD_SUPPORTED && n >= SIMD_MIN_N) {
            return simdAddIntoAndReduce(lw, lwOff, incr, incrOff, n);
        }
        return scalarAddIntoAndReduce(lw, lwOff, incr, incrOff, n);
    }

    /**
     * Effective sample size from log-weights, using a pre-computed
     * {@code logSum = logSumExp(lw)}.
     *
     * <p>ESS = (Σ exp(lw-m))² / Σ exp(2(lw-m)). Returns 0 when every
     * weight is {@code -Infinity}.
     *
     * @param lw     log-weight buffer
     * @param off    offset into {@code lw}
     * @param n      number of entries
     * @param logSum pre-computed logSumExp value
     * @return effective sample size
     */
    public static double essFromLog(double[] lw, int off, int n, double logSum) {
        if (n == 0) return 0.0;
        if (logSum == Double.NEGATIVE_INFINITY) return 0.0;
        double m = scalarMax(lw, off, n);
        if (m == Double.NEGATIVE_INFINITY) return 0.0;
        if (Double.isNaN(m)) return Double.NaN;
        double s1 = 0.0;
        double s2 = 0.0;
        for (int i = 0; i < n; i++) {
            double e = Math.exp(lw[off + i] - m);
            s1 += e;
            s2 += e * e;
        }
        return (s1 * s1) / s2;
    }

    /**
     * Shift {@code lw[off..off+n)} by {@code -max(lw)} when the max is
     * outside the numerically safe band {@code [-700, 0]}. Otherwise a no-op.
     *
     * <p>Keeps the input range where {@code Math.exp(lw[i])} is well-defined.
     * The shift is additive on the log scale so normalised weights are unchanged.
     *
     * @param lw  log-weight buffer (mutated in place)
     * @param off offset into {@code lw}
     * @param n   number of entries
     */
    public static void shiftIfOverflowing(double[] lw, int off, int n) {
        if (n == 0) return;
        double m = scalarMax(lw, off, n);
        if (m == Double.NEGATIVE_INFINITY || Double.isNaN(m)) return;
        if (m > 0.0 || m < -700.0) {
            for (int i = off, end = off + n; i < end; i++) lw[i] -= m;
        }
    }

    /**
     * Convenience log-sum-exp over {@code lw[off..off+n)}.
     *
     * @param lw  log-weight buffer
     * @param off offset into {@code lw}
     * @param n   number of entries
     * @return logSumExp value
     */
    public static double logSumExp(double[] lw, int off, int n) {
        return logSumEssMax(lw, off, n).logSum();
    }

    // ------------------------------------------------------------------
    // Scalar kernels
    // ------------------------------------------------------------------

    private static double scalarMax(double[] lw, int off, int n) {
        double m = lw[off];
        for (int i = off + 1, end = off + n; i < end; i++) {
            m = Math.max(m, lw[i]);
        }
        return m;
    }

    private static Triple scalarReduce(double[] lw, int off, int n) {
        double m = scalarMax(lw, off, n);
        if (Double.isNaN(m) || Double.isInfinite(m)) return new Triple(m, 0.0, m);
        double s1 = 0.0, s2 = 0.0;
        for (int i = 0; i < n; i++) {
            double e = Math.exp(lw[off + i] - m);
            s1 += e;
            s2 += e * e;
        }
        return new Triple(m + Math.log(s1), (s1 * s1) / s2, m);
    }

    private static Triple scalarAddIntoAndReduce(double[] lw, int lwOff,
                                                 double[] incr, int incOff, int n) {
        // Pass 1: accumulate increment and find max
        double m = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            double v = lw[lwOff + i] + incr[incOff + i];
            lw[lwOff + i] = v;
            m = Math.max(m, v);
        }
        if (Double.isNaN(m) || Double.isInfinite(m)) return new Triple(m, 0.0, m);
        // Pass 2: exp sums
        double s1 = 0.0, s2 = 0.0;
        for (int i = 0; i < n; i++) {
            double e = Math.exp(lw[lwOff + i] - m);
            s1 += e;
            s2 += e * e;
        }
        return new Triple(m + Math.log(s1), (s1 * s1) / s2, m);
    }

    // ------------------------------------------------------------------
    // SIMD kernels — species width capped at 256 bits to avoid AVX-512
    // frequency throttling. Two independent accumulators per reduction
    // keep both FP ports busy.
    //
    // Correctness notes:
    //   - Max reduction uses DoubleVector.max which propagates NaN per
    //     lane the same way Math.max does.
    //   - Exp sum uses VectorOperators.EXP intrinsic.
    //   - Tail (n mod LANES != 0) is handled scalar to maintain exact
    //     NaN / ±Inf propagation from the scalar kernels.
    // ------------------------------------------------------------------

    private static Triple simdReduce(double[] lw, int off, int n) {
        final int step2 = LANES * 2;
        final int bound2 = (n / step2) * step2;

        // Pass 1: max
        DoubleVector mx0 = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);
        DoubleVector mx1 = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < bound2; i += step2) {
            mx0 = mx0.max(DoubleVector.fromArray(SPECIES, lw, off + i));
            mx1 = mx1.max(DoubleVector.fromArray(SPECIES, lw, off + i + LANES));
        }
        int bound = SPECIES.loopBound(n);
        for (; i < bound; i += LANES) {
            mx0 = mx0.max(DoubleVector.fromArray(SPECIES, lw, off + i));
        }
        double m = mx0.max(mx1).reduceLanes(VectorOperators.MAX);
        for (; i < n; i++) {
            double v = lw[off + i];
            m = Math.max(m, v);
        }
        if (Double.isNaN(m) || Double.isInfinite(m)) return new Triple(m, 0.0, m);

        // Pass 2: s1 = Σ exp(lw-m), s2 = Σ exp(2(lw-m))
        final DoubleVector mv = DoubleVector.broadcast(SPECIES, m);
        DoubleVector s1a = DoubleVector.zero(SPECIES);
        DoubleVector s1b = DoubleVector.zero(SPECIES);
        DoubleVector s2a = DoubleVector.zero(SPECIES);
        DoubleVector s2b = DoubleVector.zero(SPECIES);
        i = 0;
        for (; i < bound2; i += step2) {
            DoubleVector e0 = DoubleVector.fromArray(SPECIES, lw, off + i).sub(mv)
                .lanewise(VectorOperators.EXP);
            DoubleVector e1 = DoubleVector.fromArray(SPECIES, lw, off + i + LANES).sub(mv)
                .lanewise(VectorOperators.EXP);
            s1a = s1a.add(e0);
            s1b = s1b.add(e1);
            s2a = e0.fma(e0, s2a);
            s2b = e1.fma(e1, s2b);
        }
        for (; i < bound; i += LANES) {
            DoubleVector e = DoubleVector.fromArray(SPECIES, lw, off + i).sub(mv)
                .lanewise(VectorOperators.EXP);
            s1a = s1a.add(e);
            s2a = e.fma(e, s2a);
        }
        double s1 = s1a.add(s1b).reduceLanes(VectorOperators.ADD);
        double s2 = s2a.add(s2b).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            double e = Math.exp(lw[off + i] - m);
            s1 += e;
            s2 += e * e;
        }
        return new Triple(m + Math.log(s1), (s1 * s1) / s2, m);
    }

    private static Triple simdAddIntoAndReduce(double[] lw, int lwOff,
                                               double[] incr, int incOff, int n) {
        final int step2 = LANES * 2;
        final int bound2 = (n / step2) * step2;

        // Pass 1: lw += incr; track max.
        DoubleVector mx0 = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);
        DoubleVector mx1 = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < bound2; i += step2) {
            DoubleVector v0 = DoubleVector.fromArray(SPECIES, lw, lwOff + i)
                .add(DoubleVector.fromArray(SPECIES, incr, incOff + i));
            DoubleVector v1 = DoubleVector.fromArray(SPECIES, lw, lwOff + i + LANES)
                .add(DoubleVector.fromArray(SPECIES, incr, incOff + i + LANES));
            v0.intoArray(lw, lwOff + i);
            v1.intoArray(lw, lwOff + i + LANES);
            mx0 = mx0.max(v0);
            mx1 = mx1.max(v1);
        }
        int bound = SPECIES.loopBound(n);
        for (; i < bound; i += LANES) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, lw, lwOff + i)
                .add(DoubleVector.fromArray(SPECIES, incr, incOff + i));
            v.intoArray(lw, lwOff + i);
            mx0 = mx0.max(v);
        }
        double m = mx0.max(mx1).reduceLanes(VectorOperators.MAX);
        for (; i < n; i++) {
            double v = lw[lwOff + i] + incr[incOff + i];
            lw[lwOff + i] = v;
            m = Math.max(m, v);
        }
        if (Double.isNaN(m) || Double.isInfinite(m)) return new Triple(m, 0.0, m);

        // Pass 2: same as simdReduce pass 2 over the mutated lw buffer.
        final DoubleVector mv = DoubleVector.broadcast(SPECIES, m);
        DoubleVector s1a = DoubleVector.zero(SPECIES);
        DoubleVector s1b = DoubleVector.zero(SPECIES);
        DoubleVector s2a = DoubleVector.zero(SPECIES);
        DoubleVector s2b = DoubleVector.zero(SPECIES);
        i = 0;
        for (; i < bound2; i += step2) {
            DoubleVector e0 = DoubleVector.fromArray(SPECIES, lw, lwOff + i).sub(mv)
                .lanewise(VectorOperators.EXP);
            DoubleVector e1 = DoubleVector.fromArray(SPECIES, lw, lwOff + i + LANES).sub(mv)
                .lanewise(VectorOperators.EXP);
            s1a = s1a.add(e0);
            s1b = s1b.add(e1);
            s2a = e0.fma(e0, s2a);
            s2b = e1.fma(e1, s2b);
        }
        for (; i < bound; i += LANES) {
            DoubleVector e = DoubleVector.fromArray(SPECIES, lw, lwOff + i).sub(mv)
                .lanewise(VectorOperators.EXP);
            s1a = s1a.add(e);
            s2a = e.fma(e, s2a);
        }
        double s1 = s1a.add(s1b).reduceLanes(VectorOperators.ADD);
        double s2 = s2a.add(s2b).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            double e = Math.exp(lw[lwOff + i] - m);
            s1 += e;
            s2 += e * e;
        }
        return new Triple(m + Math.log(s1), (s1 * s1) / s2, m);
    }
}
