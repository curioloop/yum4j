package com.curioloop.yum4j.stats;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.TimeUnit;
import java.util.random.RandomGenerator;

/**
 * Benchmark comparing {@code collect(Metric.LOG_PDF, ...)} (specialised
 * batch path with hoisted constants) against a naive per-element
 * {@code logPdf(x[i])} loop.
 *
 * <p>Tests Normal, Gamma, and Exponential distributions at N=1024.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 2, jvmArgsAppend = {"--enable-preview", "--add-modules=jdk.incubator.vector"})
@State(Scope.Thread)
public class CollectBenchmark {

    @Param({"1024"})
    int N;

    private double[] x;
    private double[] out;

    private NormalDistribution normal;
    private GammaDistribution gamma;
    private ExponentialDistribution exponential;

    @Setup(Level.Trial)
    public void setup() {
        RandomGenerator rng = RandomGenerator.of("L64X128MixRandom");
        x = new double[N];
        out = new double[N];
        for (int i = 0; i < N; i++) {
            x[i] = 0.5 + rng.nextGaussian() * 2.0;
        }
        // Ensure gamma/exponential inputs are positive
        for (int i = 0; i < N; i++) {
            if (x[i] <= 0.0) x[i] = 0.01 + rng.nextDouble();
        }
        normal = new NormalDistribution(1.0, 2.0);
        gamma = new GammaDistribution(2.5, 1.5);
        exponential = new ExponentialDistribution(3.0);
    }

    // ── Normal ───────────────────────────────────────────────────────

    @Benchmark
    public double normalCollect() {
        normal.batch(Distribution.Metric.LOG_PDF, x, 0, 1, N, out, 0);
        return out[0] + out[N - 1];
    }

    @Benchmark
    public double normalLoop() {
        double sink = 0.0;
        for (int i = 0; i < N; i++) {
            sink += normal.logPdf(x[i]);
        }
        return sink;
    }

    // ── Gamma ────────────────────────────────────────────────────────

    @Benchmark
    public double gammaCollect() {
        gamma.batch(Distribution.Metric.LOG_PDF, x, 0, 1, N, out, 0);
        return out[0] + out[N - 1];
    }

    @Benchmark
    public double gammaLoop() {
        double sink = 0.0;
        for (int i = 0; i < N; i++) {
            sink += gamma.logPdf(x[i]);
        }
        return sink;
    }

    // ── Exponential ──────────────────────────────────────────────────

    @Benchmark
    public double exponentialCollect() {
        exponential.batch(Distribution.Metric.LOG_PDF, x, 0, 1, N, out, 0);
        return out[0] + out[N - 1];
    }

    @Benchmark
    public double exponentialLoop() {
        double sink = 0.0;
        for (int i = 0; i < N; i++) {
            sink += exponential.logPdf(x[i]);
        }
        return sink;
    }

    public static void main(String[] args) throws RunnerException {
        Options opts = new OptionsBuilder()
            .include(CollectBenchmark.class.getSimpleName())
            .jvmArgsAppend("--enable-preview", "--add-modules", "jdk.incubator.vector")
            .build();
        new Runner(opts).run();
    }
}
