/*
 * Copyright (c) 2026 curioloop. All rights reserved.
 *
 * Micro-benchmark comparing the packed complex ZGEMM kernels (zgemmNNPacked /
 * zgemmTNPacked) against the reference blocked kernels on large square matrices.
 *
 * Run with:
 *   mvn -Pjmh -o test-compile exec:exec@jmh-benchmark -Dbenchmarks=PackedZgemmBenchmark
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.math.ArrayPool;
import org.openjdk.jmh.annotations.*;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgs = {"--enable-preview", "--add-modules", "jdk.incubator.vector"})
@State(Scope.Benchmark)
public class PackedZgemmBenchmark {

    // Representative sizes bracketing the ZPACK_MIN=256 gate (below/at/above).
    // For a crossover sweep, temporarily widen to {48,64,80,96,128,160,192,256,384}.
    @Param({"192", "256", "512", "1024"})
    int n;

    double[] A, B, C0;
    ArrayPool.OfDouble pool;

    @Setup(Level.Trial)
    public void setup() {
        Random rng = new Random(42);
        A = new double[n * n * 2];
        B = new double[n * n * 2];
        C0 = new double[n * n * 2];
        for (int i = 0; i < A.length; i++) {
            A[i] = rng.nextDouble();
            B[i] = rng.nextDouble();
            C0[i] = rng.nextDouble();
        }
        pool = new ArrayPool.OfDouble();
    }

    @Benchmark
    public double[] packedNN() {
        double[] c = C0.clone();
        Zgemm.zgemmNNPacked(n, n, n, 1.0, 0.0, A, 0, n, B, 0, n, c, 0, n, pool);
        return c;
    }

    @Benchmark
    public double[] blockedNN() {
        double[] c = C0.clone();
        Zgemm.zgemmNNBlocked(n, n, n, 1.0, 0.0, A, 0, n, B, 0, n, c, 0, n);
        return c;
    }

    @Benchmark
    public double[] packedTN() {
        double[] c = C0.clone();
        Zgemm.zgemmTNPacked(n, n, n, 1.0, 0.0, A, 0, n, B, 0, n, c, 0, n, false, pool);
        return c;
    }

    @Benchmark
    public double[] blockedTN() {
        double[] c = C0.clone();
        Zgemm.zgemmTNBlocked(n, n, n, 1.0, 0.0, A, 0, n, B, 0, n, c, 0, n, false);
        return c;
    }
}
