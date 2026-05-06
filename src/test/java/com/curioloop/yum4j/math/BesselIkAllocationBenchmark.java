package com.curioloop.yum4j.math;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview", "--add-modules=jdk.incubator.vector"})
public class BesselIkAllocationBenchmark {

    private static final int INVOCATIONS = 256;

    @State(Scope.Thread)
    public static class TemmeState {
        double order;
        double x;

        @Setup(Level.Trial)
        public void setup() {
            order = 0.75;
            x = 0.25;
            validateFinite("Bessel.i", Bessel.i(order, x));
            validateFinite("Bessel.k", Bessel.k(order, x));
            validateFinite("Bessel.iPrime", Bessel.iPrime(order, x));
            validateFinite("Bessel.kPrime", Bessel.kPrime(order, x));
        }
    }

    @State(Scope.Thread)
    public static class Cf2State {
        double order;
        double x;

        @Setup(Level.Trial)
        public void setup() {
            order = 0.75;
            x = 20.0;
            validateFinite("Bessel.i", Bessel.i(order, x));
            validateFinite("Bessel.k", Bessel.k(order, x));
            validateFinite("Bessel.iPrime", Bessel.iPrime(order, x));
            validateFinite("Bessel.kPrime", Bessel.kPrime(order, x));
        }
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double iTemmeRoute(TemmeState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Bessel.i(state.order, state.x);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double kTemmeRoute(TemmeState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Bessel.k(state.order, state.x);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double iPrimeTemmeRoute(TemmeState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Bessel.iPrime(state.order, state.x);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double kPrimeTemmeRoute(TemmeState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Bessel.kPrime(state.order, state.x);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double iCf2Route(Cf2State state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Bessel.i(state.order, state.x);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double kCf2Route(Cf2State state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Bessel.k(state.order, state.x);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double iPrimeCf2Route(Cf2State state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Bessel.iPrime(state.order, state.x);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double kPrimeCf2Route(Cf2State state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Bessel.kPrime(state.order, state.x);
        }
        return sink;
    }

    private static void validateFinite(String name, double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalStateException("non-finite " + name + " benchmark seed: " + value);
        }
    }
}