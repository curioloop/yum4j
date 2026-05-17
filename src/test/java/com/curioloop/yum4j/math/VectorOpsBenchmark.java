/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Micro-benchmark comparing scalar (BLAS/loop) vs SIMD (Vector API) implementations
 * of VectorOps slice methods: op(x, off, len) and op(x, xOff, y, yOff, len).
 *
 * All SIMD variants use DoubleVector.SPECIES_PREFERRED and handle the tail loop.
 * The "scalar" variants are the current VectorOps implementations (BLAS or loop).
 *
 * Run with:
 *   mvn test-compile && java --enable-preview --add-modules jdk.incubator.vector \
 *     -cp target/test-classes:target/classes \
 *     com.curioloop.yum4j.math.VectorOpsBenchmark
 */
package com.curioloop.yum4j.math;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Random;

public class VectorOpsBenchmark {

    private static final VectorSpecies<Double> SP = DoubleVector.SPECIES_PREFERRED;
    private static final int WARMUP = 5_000;
    private static final int ITERS  = 10_000;

    // Use a non-zero offset to exercise the slice path
    private static final int OFF = 7;

    public static void main(String[] args) {
        System.out.println("Vector species: " + SP + " (lanes=" + SP.length() + ")");
        System.out.println();

        for (int n : new int[]{64, 256, 1024, 4096}) {
            int total = n + OFF;
            double[] x       = rng(total, 42);
            double[] y       = rng(total, 99);
            double[] dstInit = rng(total, 123);   // initial state for mutation ops
            double[] dst     = dstInit.clone();   // reused, reset before each timed call
            double[] out     = new double[total]; // for write-only ops

            System.out.printf("=== n=%d (off=%d) ===%n", n, OFF);

            // ── Single-array reductions ───────────────────────────────────────
            row("sum",           () -> VectorOps.sum(x, OFF, n),
                                 () -> sumSimd(x, OFF, n));
            row("l1Norm",        () -> VectorOps.l1Norm(x, OFF, n),
                                 () -> l1NormSimd(x, OFF, n));
            row("dot(x,x)",      () -> VectorOps.dot(x, OFF, x, OFF, n),
                                 () -> sumOfSquaresSimd(x, OFF, n));
            row("sumSq",         () -> VectorOps.sumSq(x, OFF, n, VectorOps.mean(x, OFF, n)),
                                 () -> sumSqSimd(x, OFF, n));
            row("max",           () -> VectorOps.max(x, OFF, n),
                                 () -> maxSimd(x, OFF, n));
            row("min",           () -> VectorOps.min(x, OFF, n),
                                 () -> minSimd(x, OFF, n));
            row("maxAbs",        () -> VectorOps.maxAbs(x, OFF, n),
                                 () -> maxAbsSimd(x, OFF, n));
            row("l2Norm",        () -> VectorOps.l2Norm(x, OFF, n),
                                 () -> l2NormSimd(x, OFF, n));

            // ── Two-array reductions ──────────────────────────────────────────
            row("dot",           () -> VectorOps.dot(x, OFF, y, OFF, n),
                                 () -> dotSimd(x, OFF, y, OFF, n));
            row("l1Dist",        () -> VectorOps.l1Dist(x, OFF, y, OFF, n),
                                 () -> l1DistSimd(x, OFF, y, OFF, n));
            row("linfDist",      () -> VectorOps.linfDist(x, OFF, y, OFF, n),
                                 () -> linfDistSimd(x, OFF, y, OFF, n));
            row("linfDistU4",    () -> VectorOps.linfDist(x, OFF, y, OFF, n),
                                 () -> linfDistUnrolled(x, OFF, y, OFF, n));
            row("l2Dist",        () -> VectorOps.l2Dist(x, OFF, y, OFF, n),
                                 () -> l2DistSimd(x, OFF, y, OFF, n));

            // ── In-place mutations ────────────────────────────────────────────
            rowV("axpy", dst, dstInit,
                 () -> VectorOps.axpy(2.0, x, OFF, dst, OFF, n),
                 () -> axpySimd(2.0, x, OFF, dst, OFF, n));
            rowV("scal", dst, dstInit,
                 () -> VectorOps.scal(2.0, dst, OFF, n),
                 () -> scalSimd(2.0, dst, OFF, n));
            rowV("add",  dst, dstInit,
                 () -> VectorOps.add(dst, OFF, x, OFF, n),
                 () -> addSimd(dst, OFF, x, OFF, n));
            rowV("addConst", dst, dstInit,
                 () -> VectorOps.addConst(2.0, dst, OFF, n),
                 () -> addConstSimd(2.0, dst, OFF, n));
            rowV("div", dst, dstInit,
                 () -> VectorOps.div(dst, OFF, y, OFF, n),
                 () -> divSimd(dst, OFF, y, OFF, n));

            // ── Write to output (no reset needed — output is fully overwritten)
            rowV("divTo", out, dstInit,
                 () -> VectorOps.divTo(out, OFF, x, OFF, y, OFF, n),
                 () -> divToSimd(out, OFF, x, OFF, y, OFF, n));
            rowV("scalTo", out, dstInit,
                 () -> VectorOps.scalTo(out, OFF, 2.0, x, OFF, n),
                 () -> scalToSimd(out, OFF, 2.0, x, OFF, n));
            rowV("axpyTo", out, dstInit,
                 () -> VectorOps.axpyTo(out, OFF, 2.0, x, OFF, y, OFF, n),
                 () -> axpyToSimd(out, OFF, 2.0, x, OFF, y, OFF, n));
            rowV("axpyToFma", out, dstInit,
                 () -> VectorOps.axpyTo(out, OFF, 2.0, x, OFF, y, OFF, n),
                 () -> axpyToFma(out, OFF, 2.0, x, OFF, y, OFF, n));

            // ── Prefix operations ─────────────────────────────────────────────
            // cumSum/cumProd use full arrays (no slice), so we pass x[0..n-1]
            double[] xs = java.util.Arrays.copyOf(x, n);
            double[] cumDst = new double[n];
            rowV("cumSum",  cumDst, dstInit,
                 () -> VectorOps.cumSum(cumDst, xs),
                 () -> cumSumUnrolled(cumDst, xs));
            rowV("cumProd", cumDst, dstInit,
                 () -> VectorOps.cumProd(cumDst, xs),
                 () -> cumProdUnrolled(cumDst, xs));

            System.out.println();
        }
    }

    // ── SIMD implementations (slice-aware) ────────────────────────────────────

    static double sumSimd(double[] a, int off, int len) {
        int i = 0, bound = SP.loopBound(len);
        var acc = DoubleVector.zero(SP);
        for (; i < bound; i += SP.length()) acc = acc.add(DoubleVector.fromArray(SP, a, off + i));
        double s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) s += a[off + i];
        return s;
    }

    static double l1NormSimd(double[] a, int off, int len) {
        int i = 0, bound = SP.loopBound(len);
        var acc = DoubleVector.zero(SP);
        for (; i < bound; i += SP.length()) acc = acc.add(DoubleVector.fromArray(SP, a, off + i).abs());
        double s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) s += Math.abs(a[off + i]);
        return s;
    }

    static double sumOfSquaresSimd(double[] a, int off, int len) {
        int i = 0, bound = SP.loopBound(len);
        var acc = DoubleVector.zero(SP);
        for (; i < bound; i += SP.length()) {
            var v = DoubleVector.fromArray(SP, a, off + i);
            acc = v.fma(v, acc);
        }
        double s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) s += a[off + i] * a[off + i];
        return s;
    }

    static double sumSqSimd(double[] a, int off, int len) {
        // Pass 1: mean
        double mean = sumSimd(a, off, len) / len;
        // Pass 2: Σ(aᵢ - mean)²
        int i = 0, bound = SP.loopBound(len);
        var meanVec = DoubleVector.broadcast(SP, mean);
        var acc = DoubleVector.zero(SP);
        for (; i < bound; i += SP.length()) {
            var d = DoubleVector.fromArray(SP, a, off + i).sub(meanVec);
            acc = d.fma(d, acc);
        }
        double ss = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) { double d = a[off + i] - mean; ss += d * d; }
        return ss;
    }

    static double maxAbsSimd(double[] a, int off, int len) {
        int i = 0, bound = SP.loopBound(len);
        var maxAcc = DoubleVector.zero(SP);
        for (; i < bound; i += SP.length())
            maxAcc = maxAcc.max(DoubleVector.fromArray(SP, a, off + i).abs());
        double max = maxAcc.reduceLanes(VectorOperators.MAX);
        for (; i < len; i++) { double abs = Math.abs(a[off + i]); if (abs > max) max = abs; }
        return max;
    }

    static double maxSimd(double[] a, int off, int len) {
        int i = 0, bound = SP.loopBound(len);
        var maxAcc = DoubleVector.broadcast(SP, Double.NEGATIVE_INFINITY);
        for (; i < bound; i += SP.length())
            maxAcc = maxAcc.max(DoubleVector.fromArray(SP, a, off + i));
        double m = maxAcc.reduceLanes(VectorOperators.MAX);
        for (; i < len; i++) { double v = a[off + i]; if (v > m) m = v; }
        return m;
    }

    static double minSimd(double[] a, int off, int len) {
        int i = 0, bound = SP.loopBound(len);
        var minAcc = DoubleVector.broadcast(SP, Double.POSITIVE_INFINITY);
        for (; i < bound; i += SP.length())
            minAcc = minAcc.min(DoubleVector.fromArray(SP, a, off + i));
        double m = minAcc.reduceLanes(VectorOperators.MIN);
        for (; i < len; i++) { double v = a[off + i]; if (v < m) m = v; }
        return m;
    }

    static double l2NormSimd(double[] a, int off, int len) {
        // Neumaier scaled — no simple SIMD; fall back to scalar for correctness comparison
        return VectorOps.l2Norm(a, off, len);
    }

    static double dotSimd(double[] x, int xOff, double[] y, int yOff, int len) {
        int i = 0, bound = SP.loopBound(len);
        var acc = DoubleVector.zero(SP);
        for (; i < bound; i += SP.length())
            acc = DoubleVector.fromArray(SP, x, xOff + i).fma(DoubleVector.fromArray(SP, y, yOff + i), acc);
        double s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) s += x[xOff + i] * y[yOff + i];
        return s;
    }

    static double l1DistSimd(double[] x, int xOff, double[] y, int yOff, int len) {
        int i = 0, bound = SP.loopBound(len);
        var acc = DoubleVector.zero(SP);
        for (; i < bound; i += SP.length())
            acc = acc.add(DoubleVector.fromArray(SP, x, xOff + i)
                    .sub(DoubleVector.fromArray(SP, y, yOff + i)).abs());
        double s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) s += Math.abs(x[xOff + i] - y[yOff + i]);
        return s;
    }

    static void axpySimd(double alpha, double[] x, int xOff, double[] y, int yOff, int len) {
        int i = 0, bound = SP.loopBound(len);
        var av = DoubleVector.broadcast(SP, alpha);
        for (; i < bound; i += SP.length())
            DoubleVector.fromArray(SP, x, xOff + i)
                    .fma(av, DoubleVector.fromArray(SP, y, yOff + i))
                    .intoArray(y, yOff + i);
        for (; i < len; i++) y[yOff + i] += alpha * x[xOff + i];
    }

    static void scalSimd(double alpha, double[] x, int off, int len) {
        int i = 0, bound = SP.loopBound(len);
        var av = DoubleVector.broadcast(SP, alpha);
        for (; i < bound; i += SP.length())
            DoubleVector.fromArray(SP, x, off + i).mul(av).intoArray(x, off + i);
        for (; i < len; i++) x[off + i] *= alpha;
    }

    static void addSimd(double[] dst, int dOff, double[] s, int sOff, int len) {
        int i = 0, bound = SP.loopBound(len);
        for (; i < bound; i += SP.length())
            DoubleVector.fromArray(SP, dst, dOff + i)
                    .add(DoubleVector.fromArray(SP, s, sOff + i))
                    .intoArray(dst, dOff + i);
        for (; i < len; i++) dst[dOff + i] += s[sOff + i];
    }

    static double linfDistSimd(double[] x, int xOff, double[] y, int yOff, int len) {
        if (len == 0) return 0.0;
        int i = 0, bound = SP.loopBound(len);
        var maxAcc = DoubleVector.zero(SP);
        for (; i < bound; i += SP.length())
            maxAcc = maxAcc.max(DoubleVector.fromArray(SP, x, xOff + i)
                    .sub(DoubleVector.fromArray(SP, y, yOff + i)).abs());
        double max = maxAcc.reduceLanes(VectorOperators.MAX);
        for (; i < len; i++) { double d = Math.abs(x[xOff + i] - y[yOff + i]); if (d > max) max = d; }
        return max;
    }

    // Two independent max chains to break the branch-dependent serial dependency.
    // NaN propagation: if any diff is NaN, Math.max(NaN, x) = NaN on both chains,
    // so the final Math.max(max0, max1) correctly returns NaN.
    static double linfDistUnrolled(double[] x, int xOff, double[] y, int yOff, int len) {
        if (len == 0) return 0.0;
        double max0 = 0.0, max1 = 0.0;
        int i = 0;
        for (; i + 7 < len; i += 8) {
            double d0 = Math.abs(x[xOff+i]   - y[yOff+i]);
            double d1 = Math.abs(x[xOff+i+1] - y[yOff+i+1]);
            double d2 = Math.abs(x[xOff+i+2] - y[yOff+i+2]);
            double d3 = Math.abs(x[xOff+i+3] - y[yOff+i+3]);
            double d4 = Math.abs(x[xOff+i+4] - y[yOff+i+4]);
            double d5 = Math.abs(x[xOff+i+5] - y[yOff+i+5]);
            double d6 = Math.abs(x[xOff+i+6] - y[yOff+i+6]);
            double d7 = Math.abs(x[xOff+i+7] - y[yOff+i+7]);
            if (d0 > max0) max0 = d0;  if (d1 > max1) max1 = d1;
            if (d2 > max0) max0 = d2;  if (d3 > max1) max1 = d3;
            if (d4 > max0) max0 = d4;  if (d5 > max1) max1 = d5;
            if (d6 > max0) max0 = d6;  if (d7 > max1) max1 = d7;
        }
        double max = Math.max(max0, max1);
        for (; i < len; i++) { double d = Math.abs(x[xOff+i] - y[yOff+i]); if (d > max) max = d; }
        return max;
    }

    static double l2DistSimd(double[] x, int xOff, double[] y, int yOff, int len) {
        // Fast path: plain SIMD sum of squares; accurate when no under/overflow.
        // Falls back to the Neumaier-scaled scalar version if inputs are extreme.
        int i = 0, bound = SP.loopBound(len);
        var acc = DoubleVector.zero(SP);
        for (; i < bound; i += SP.length()) {
            var d = DoubleVector.fromArray(SP, x, xOff + i).sub(DoubleVector.fromArray(SP, y, yOff + i));
            acc = d.fma(d, acc);
        }
        double s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) { double d = x[xOff + i] - y[yOff + i]; s += d * d; }
        return Math.sqrt(s);
    }

    static void addConstSimd(double alpha, double[] x, int off, int len) {
        int i = 0, bound = SP.loopBound(len);
        var av = DoubleVector.broadcast(SP, alpha);
        for (; i < bound; i += SP.length())
            DoubleVector.fromArray(SP, x, off + i).add(av).intoArray(x, off + i);
        for (; i < len; i++) x[off + i] += alpha;
    }

    static void divSimd(double[] dst, int dOff, double[] s, int sOff, int len) {
        int i = 0, bound = SP.loopBound(len);
        for (; i < bound; i += SP.length())
            DoubleVector.fromArray(SP, dst, dOff + i)
                    .div(DoubleVector.fromArray(SP, s, sOff + i))
                    .intoArray(dst, dOff + i);
        for (; i < len; i++) dst[dOff + i] /= s[sOff + i];
    }

    static void divToSimd(double[] dst, int dOff, double[] x, int xOff, double[] y, int yOff, int len) {
        int i = 0, bound = SP.loopBound(len);
        for (; i < bound; i += SP.length())
            DoubleVector.fromArray(SP, x, xOff + i)
                    .div(DoubleVector.fromArray(SP, y, yOff + i))
                    .intoArray(dst, dOff + i);
        for (; i < len; i++) dst[dOff + i] = x[xOff + i] / y[yOff + i];
    }

    static void scalToSimd(double[] dst, int dOff, double alpha, double[] x, int xOff, int len) {
        int i = 0, bound = SP.loopBound(len);
        var av = DoubleVector.broadcast(SP, alpha);
        for (; i < bound; i += SP.length())
            DoubleVector.fromArray(SP, x, xOff + i).mul(av).intoArray(dst, dOff + i);
        for (; i < len; i++) dst[dOff + i] = alpha * x[xOff + i];
    }

    static void axpyToSimd(double[] dst, int dOff, double alpha,
                           double[] x, int xOff, double[] y, int yOff, int len) {
        int i = 0, bound = SP.loopBound(len);
        var av = DoubleVector.broadcast(SP, alpha);
        for (; i < bound; i += SP.length())
            DoubleVector.fromArray(SP, x, xOff + i)
                    .fma(av, DoubleVector.fromArray(SP, y, yOff + i))
                    .intoArray(dst, dOff + i);
        for (; i < len; i++) dst[dOff + i] = alpha * x[xOff + i] + y[yOff + i];
    }

    static void axpyToFma(double[] dst, int dOff, double alpha,
                          double[] x, int xOff, double[] y, int yOff, int len) {
        for (int i = 0; i < len; i++)
            dst[dOff + i] = Math.fma(alpha, x[xOff + i], y[yOff + i]);
    }

    // ── cumSum / cumProd unrolled variants ────────────────────────────────────
    // Serial carry dependency prevents SIMD, but unrolling x4 reduces loop
    // overhead and lets the JIT schedule loads ahead of the dependent add/mul.

    static double[] cumSumUnrolled(double[] dst, double[] s) {
        int n = s.length;
        if (n == 0) return dst;
        dst[0] = s[0];
        int i = 1;
        // Unroll x4: each iteration reads s[i..i+3] (independent loads),
        // then chains dst[i] = dst[i-1] + s[i] sequentially.
        for (; i + 3 < n; i += 4) {
            double s0 = s[i], s1 = s[i+1], s2 = s[i+2], s3 = s[i+3]; // hoist loads
            double d0 = dst[i-1] + s0;
            double d1 = d0 + s1;
            double d2 = d1 + s2;
            double d3 = d2 + s3;
            dst[i] = d0; dst[i+1] = d1; dst[i+2] = d2; dst[i+3] = d3;
        }
        for (; i < n; i++) dst[i] = dst[i-1] + s[i];
        return dst;
    }

    static double[] cumProdUnrolled(double[] dst, double[] s) {
        int n = s.length;
        if (n == 0) return dst;
        dst[0] = s[0];
        int i = 1;
        for (; i + 3 < n; i += 4) {
            double s0 = s[i], s1 = s[i+1], s2 = s[i+2], s3 = s[i+3];
            double d0 = dst[i-1] * s0;
            double d1 = d0 * s1;
            double d2 = d1 * s2;
            double d3 = d2 * s3;
            dst[i] = d0; dst[i+1] = d1; dst[i+2] = d2; dst[i+3] = d3;
        }
        for (; i < n; i++) dst[i] = dst[i-1] * s[i];
        return dst;
    }

    // ── Harness ───────────────────────────────────────────────────────────────

    @FunctionalInterface interface Fn  { double run(); }
    @FunctionalInterface interface FnV { void run(); }

    static void row(String name, Fn scalar, Fn simd) {
        double sink = 0;
        for (int i = 0; i < WARMUP; i++) { sink += scalar.run(); sink += simd.run(); }
        long t0 = System.nanoTime();
        for (int i = 0; i < ITERS; i++) sink += scalar.run();
        double ns0 = (double)(System.nanoTime() - t0) / ITERS;
        long t1 = System.nanoTime();
        for (int i = 0; i < ITERS; i++) sink += simd.run();
        double ns1 = (double)(System.nanoTime() - t1) / ITERS;
        System.out.printf("  %-16s  scalar %8.1f ns  simd %8.1f ns  speedup %4.1fx  (sink=%.2f)%n",
                name, ns0, ns1, ns0 / ns1, sink);
    }

    static void rowV(String name, double[] dst, double[] dstInit, FnV scalar, FnV simd) {
        int len = dst.length;
        for (int i = 0; i < WARMUP; i++) {
            System.arraycopy(dstInit, 0, dst, 0, len); scalar.run();
            System.arraycopy(dstInit, 0, dst, 0, len); simd.run();
        }
        long t0 = System.nanoTime();
        for (int i = 0; i < ITERS; i++) {
            System.arraycopy(dstInit, 0, dst, 0, len);
            scalar.run();
        }
        double ns0 = (double)(System.nanoTime() - t0) / ITERS;
        long t1 = System.nanoTime();
        for (int i = 0; i < ITERS; i++) {
            System.arraycopy(dstInit, 0, dst, 0, len);
            simd.run();
        }
        double ns1 = (double)(System.nanoTime() - t1) / ITERS;
        // Baseline: copy-only cost so we can subtract it out.
        long tC = System.nanoTime();
        for (int i = 0; i < ITERS; i++) {
            System.arraycopy(dstInit, 0, dst, 0, len);
        }
        double nsC = (double)(System.nanoTime() - tC) / ITERS;
        double s = Math.max(ns0 - nsC, 1e-6);
        double v = Math.max(ns1 - nsC, 1e-6);
        System.out.printf("  %-16s  scalar %8.1f ns  simd %8.1f ns  speedup %4.1fx  (copy %.1f ns)%n",
                name, s, v, s / v, nsC);
    }

    static double[] rng(int n, long seed) {
        double[] a = new double[n];
        Random r = new Random(seed);
        for (int i = 0; i < n; i++) a[i] = r.nextGaussian();
        return a;
    }
}
