/*
 * Copyright (c) 2026 curioloop. All rights reserved.
 *
 * Micro-benchmark comparing the packed NN DGEMM kernel (dgemmNNPacked) against the
 * reference blocked kernel (dgemmNNBlocked) on large square matrices.
 *
 * Run with:
 *   mvn -Pjmh -o test-compile exec:exec@jmh-benchmark -Dbenchmarks=PackedDgemmBenchmark
 */
package com.curioloop.yum4j.linalg.blas;

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
public class PackedDgemmBenchmark {

    // Representative sizes at/above the PACK_MIN=128 gate. The 128 threshold aligns
    // with the direct/blocked boundary (maxBlock=128) and was validated by a
    // crossover sweep {96,128,160,192,256,384,512}: packing wins from 128 upward.
    @Param({"128", "256", "512", "1024"})
    int n;

    double[] A, B, C0;
    ArrayPool.OfDouble pool;

    @Setup(Level.Trial)
    public void setup() {
        Random rng = new Random(42);
        A = new double[n * n];
        B = new double[n * n];
        C0 = new double[n * n];
        for (int i = 0; i < n * n; i++) {
            A[i] = rng.nextDouble();
            B[i] = rng.nextDouble();
            C0[i] = rng.nextDouble();
        }
        pool = new ArrayPool.OfDouble();
    }

    @Benchmark
    public double[] packed() {
        double[] c = C0.clone();
        Dgemm.dgemmNNPacked(n, n, n, 1.0, A, 0, n, B, 0, n, c, 0, n, pool);
        return c;
    }

    @Benchmark
    public double[] blocked() {
        double[] c = C0.clone();
        Dgemm.dgemmNNBlocked(n, n, n, 1.0, A, 0, n, B, 0, n, c, 0, n);
        return c;
    }

    @Benchmark
    public double[] packedTN() {
        double[] c = C0.clone();
        Dgemm.dgemmTNPacked(n, n, n, 1.0, A, 0, n, B, 0, n, c, 0, n, pool);
        return c;
    }

    @Benchmark
    public double[] blockedTN() {
        double[] c = C0.clone();
        Dgemm.dgemmTNBlocked(n, n, n, 1.0, A, 0, n, B, 0, n, c, 0, n);
        return c;
    }
}
