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
public class HypergeometricBesselSeriesAllocationBenchmark {

    private static final int INVOCATIONS = 256;

    @State(Scope.Thread)
    public static class As1336AltState {
        double a;
        double b;
        double z;

        @Setup(Level.Trial)
        public void setup() {
            a = -40.5;
            b = -40.25;
            z = 0.5;
            verifyRoute(a, b, z, "alt-13.3.6");
        }
    }

    @State(Scope.Thread)
    public static class As1336PreState {
        double a;
        double b;
        double z;

        @Setup(Level.Trial)
        public void setup() {
            a = 0.009;
            b = -120.5;
            z = -40.0;
            verifyRoute(a, b, z, "pre-13.3.6");
        }
    }

    @State(Scope.Thread)
    public static class TricomiState {
        double a;
        double b;
        double z;

        @Setup(Level.Trial)
        public void setup() {
            a = -80.5;
            b = -55.25;
            z = 2.5;
            verifyRoute(a, b, z, "tricomi");
        }
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double oneFOneAltAs1336Route(As1336AltState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Hypergeometric.oneFOne(state.a, state.b, state.z);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double oneFOnePreAs1336Route(As1336PreState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Hypergeometric.oneFOne(state.a, state.b, state.z);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double oneFOneTricomiRoute(TricomiState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Hypergeometric.oneFOne(state.a, state.b, state.z);
        }
        return sink;
    }

    private static void verifyRoute(double a, double b, double z, String expectedRoute) {
        String actualRoute = Hypergeometric.debugOneFOneRealRoute(a, b, z);
        if (!expectedRoute.equals(actualRoute)) {
            throw new IllegalStateException(
                "unexpected route for benchmark parameters: expected=" + expectedRoute
                    + ", actual=" + actualRoute
                    + ", a=" + a + ", b=" + b + ", z=" + z);
        }
    }
}