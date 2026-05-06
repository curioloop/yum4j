package com.curioloop.yum4j.stats;

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
public class StatisticsQuantileAllocationBenchmark {

    private static final int INVOCATIONS = 16;

    @State(Scope.Thread)
    public static class SolverControlState {
        NormalDistribution distribution;
        double probability;
        double upperTailProbability;
        double initialGuess;

        @Setup(Level.Trial)
        public void setup() {
            distribution = new NormalDistribution(0.5, 1.25);
            probability = 0.84;
            upperTailProbability = 0.16;
            initialGuess = distribution.mean();

            assertClose(
                StatisticsQuantileSupport.continuousQuantile(distribution, probability, initialGuess),
                distribution.quantile(probability),
                1.0e-12,
                "solver control quantile"
            );
            assertClose(
                StatisticsQuantileSupport.continuousUpperTailQuantile(distribution, upperTailProbability, initialGuess),
                distribution.quantileUpperTail(upperTailProbability),
                1.0e-12,
                "solver control upper-tail quantile"
            );
        }
    }

    @State(Scope.Thread)
    public static class NonCentralChiSquareState {
        NonCentralChiSquareDistribution distribution;
        double probability;
        double upperTailProbability;

        @Setup(Level.Trial)
        public void setup() {
            distribution = new NonCentralChiSquareDistribution(3.5, 2.0);
            probability = 0.42424000248778682;
            upperTailProbability = 1.0 - probability;

            assertClose(distribution.quantile(probability), 4.0, 1.0e-6, "non-central chi-square quantile");
            assertClose(distribution.quantileUpperTail(upperTailProbability), 4.0, 1.0e-6, "non-central chi-square upper-tail quantile");
        }
    }

    @State(Scope.Thread)
    public static class NonCentralTState {
        NonCentralTDistribution distribution;
        double probability;
        double upperTailProbability;

        @Setup(Level.Trial)
        public void setup() {
            distribution = new NonCentralTDistribution(7.0, 1.5);
            probability = 0.3716903605396073;
            upperTailProbability = 0.62830963946039264;

            assertClose(distribution.quantile(probability), 1.2, 1.0e-6, "non-central t quantile");
            assertClose(distribution.quantileUpperTail(upperTailProbability), 1.2, 1.0e-6, "non-central t upper-tail quantile");
        }
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double solverControlQuantile(SolverControlState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += StatisticsQuantileSupport.continuousQuantile(state.distribution, state.probability, state.initialGuess);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double solverControlUpperTailQuantile(SolverControlState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += StatisticsQuantileSupport.continuousUpperTailQuantile(
                state.distribution,
                state.upperTailProbability,
                state.initialGuess
            );
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double nonCentralChiSquareQuantile(NonCentralChiSquareState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += state.distribution.quantile(state.probability);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double nonCentralChiSquareUpperTailQuantile(NonCentralChiSquareState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += state.distribution.quantileUpperTail(state.upperTailProbability);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double nonCentralTQuantile(NonCentralTState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += state.distribution.quantile(state.probability);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double nonCentralTUpperTailQuantile(NonCentralTState state) {
        double sink = 0.0;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            sink += state.distribution.quantileUpperTail(state.upperTailProbability);
        }
        return sink;
    }

    private static void assertClose(double actual, double expected, double tolerance, String label) {
        if (!Double.isFinite(actual) || Math.abs(actual - expected) > tolerance) {
            throw new IllegalStateException(label + " seed mismatch: actual=" + actual + ", expected=" + expected);
        }
    }
}