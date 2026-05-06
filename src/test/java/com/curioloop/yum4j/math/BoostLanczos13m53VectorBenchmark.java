package com.curioloop.yum4j.math;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview", "--add-modules=jdk.incubator.vector"})
public class BoostLanczos13m53VectorBenchmark {

    @State(Scope.Thread)
    public static class InputState {
        static final int LENGTH = 16;

        final double[] interior = {
            0.125, 0.25, 0.5, 0.75,
            1.0, 1.25, 1.5, 2.0,
            2.5, 3.0, 4.0, 6.0,
            8.0, 12.0, 16.0, 24.0
        };

        final double[] reciprocal = {
            5.0e25, 1.0e26, 2.0e26, 5.0e26,
            1.0e27, 1.0e30, 1.0e40, 1.0e60,
            1.0e80, 1.0e100, 1.0e120, 1.0e140,
            1.0e160, 1.0e180, 1.0e200, 1.0e250
        };
    }

    @Benchmark
    @OperationsPerInvocation(InputState.LENGTH)
    public double scalarLanczosSumInterior(InputState state) {
        return sumLanczosInterior(state.interior, true, false);
    }

    @Benchmark
    @OperationsPerInvocation(InputState.LENGTH)
    public double vectorLanczosSumInterior(InputState state) {
        return sumLanczosInterior(state.interior, false, false);
    }

    @Benchmark
    @OperationsPerInvocation(InputState.LENGTH)
    public double scalarLanczosSumReciprocal(InputState state) {
        return sumLanczosInterior(state.reciprocal, true, false);
    }

    @Benchmark
    @OperationsPerInvocation(InputState.LENGTH)
    public double vectorLanczosSumReciprocal(InputState state) {
        return sumLanczosInterior(state.reciprocal, false, false);
    }

    @Benchmark
    @OperationsPerInvocation(InputState.LENGTH)
    public double scalarLanczosSumExpGScaledInterior(InputState state) {
        return sumLanczosInterior(state.interior, true, true);
    }

    @Benchmark
    @OperationsPerInvocation(InputState.LENGTH)
    public double vectorLanczosSumExpGScaledInterior(InputState state) {
        return sumLanczosInterior(state.interior, false, true);
    }

    @Benchmark
    @OperationsPerInvocation(InputState.LENGTH)
    public double scalarLanczosSumExpGScaledReciprocal(InputState state) {
        return sumLanczosInterior(state.reciprocal, true, true);
    }

    @Benchmark
    @OperationsPerInvocation(InputState.LENGTH)
    public double vectorLanczosSumExpGScaledReciprocal(InputState state) {
        return sumLanczosInterior(state.reciprocal, false, true);
    }

    private static double sumLanczosInterior(double[] inputs, boolean scalar, boolean scaled) {
        double sink = 0.0;
        for (double x : inputs) {
            if (scaled) {
                sink += scalar
                    ? Lanczos13m53.lanczosSumExpGScaledScalar(x)
                    : Lanczos13m53.lanczosSumExpGScaled(x);
            } else {
                sink += scalar
                    ? Lanczos13m53.lanczosSumScalar(x)
                    : Lanczos13m53.lanczosSum(x);
            }
        }
        return sink;
    }
}