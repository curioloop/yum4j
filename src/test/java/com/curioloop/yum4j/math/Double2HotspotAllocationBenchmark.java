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
public class Double2HotspotAllocationBenchmark {

    private static final int INVOCATIONS = 256;

    @State(Scope.Thread)
    public static class BesselJState {
        double order;
        double x;

        @Setup(Level.Trial)
        public void setup() {
            double a = -80.5;
            double b = -55.25;
            double z = 2.5;
            double besselArg = (0.5 * b - a) * z;
            order = b - 2.0;
            x = 2.0 * Math.sqrt(Math.abs(besselArg));

            double value = Bessel.j(order, x);
            if (!Double.isFinite(value)) {
                throw new IllegalStateException("non-finite Bessel.j benchmark seed for order=" + order + ", x=" + x);
            }
        }
    }

    @State(Scope.Thread)
    public static class TwoFZeroCfState {
        double a;
        double b;
        double z;

        @Setup(Level.Trial)
        public void setup() {
            a = -4.0;
            b = -7.25;
            z = -0.75;

            double value = Hypergeometric.twoFZero(a, b, z);
            if (!Double.isFinite(value)) {
                throw new IllegalStateException("non-finite twoFZero benchmark seed for a=" + a + ", b=" + b + ", z=" + z);
            }
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
            String route = Hypergeometric.debugOneFOneRealRoute(a, b, z);
            if (!"tricomi".equals(route)) {
                throw new IllegalStateException("unexpected route for tricomi benchmark: " + route);
            }
        }
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double besselJNegativeOrder(BesselJState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Bessel.j(state.order, state.x);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double twoFZeroContinuedFraction(TwoFZeroCfState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Hypergeometric.twoFZero(state.a, state.b, state.z);
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
}