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
public class InverseRootIteratorAllocationBenchmark {

    private static final int INVOCATIONS = 32;

    @State(Scope.Thread)
    public static class BetaInverseState {
        double a;
        double b;
        double x;
        double p;
        double q;

        @Setup(Level.Trial)
        public void setup() {
            a = 2.75;
            b = 4.5;
            x = 0.37;
            p = Beta.ibeta(a, b, x);
            q = 1.0 - p;

            assertClose("ibetaInv", Beta.ibetaInv(a, b, p), x, 1.0e-10);
            assertClose("ibetacInv", Beta.ibetacInv(a, b, q), x, 1.0e-10);
        }
    }

    @State(Scope.Thread)
    public static class GammaInverseState {
        double a;
        double x;
        double p;
        double q;

        @Setup(Level.Trial)
        public void setup() {
            a = 3.5;
            x = 2.0;
            p = Gamma.gammaP(a, x);
            q = Gamma.gammaQ(a, x);

            assertClose("gammaPInv", Gamma.gammaPInv(a, p), x, 1.0e-10);
            assertClose("gammaQInv", Gamma.gammaQInv(a, q), x, 1.0e-10);
        }
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double ibetaInv(BetaInverseState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Beta.ibetaInv(state.a, state.b, state.p);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double ibetacInv(BetaInverseState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Beta.ibetacInv(state.a, state.b, state.q);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double gammaPInv(GammaInverseState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Gamma.gammaPInv(state.a, state.p);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double gammaQInv(GammaInverseState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += Gamma.gammaQInv(state.a, state.q);
        }
        return sink;
    }

    private static void assertClose(String label, double actual, double expected, double tolerance) {
        if (!Double.isFinite(actual) || Math.abs(actual - expected) > tolerance) {
            throw new IllegalStateException(label + " seed mismatch: actual=" + actual + ", expected=" + expected);
        }
    }
}