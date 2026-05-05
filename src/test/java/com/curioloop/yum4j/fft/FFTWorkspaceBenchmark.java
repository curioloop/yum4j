package com.curioloop.yum4j.fft;

import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class FFTWorkspaceBenchmark {
    @State(Scope.Thread)
    public static class PreparedState {
        Transform.PreparedComplex complex;
        FftWorkspace complexWorkspace;
        double[] complexInput;
        double[] complexOutput;

        Transform.PreparedReal real;
        FftWorkspace realWorkspace;
        double[] realInput;
        double[] realOutput;

        @Setup(Level.Trial)
        public void setup() {
            complex = Transform.prepareComplex(256);
            complexWorkspace = complex.createWorkspace();
            complexInput = complexSample(256);
            complexOutput = new double[complexInput.length];

            real = Transform.prepareReal(256);
            realWorkspace = real.createWorkspace();
            realInput = realSample(256);
            realOutput = new double[2 * (256 / 2 + 1)];
        }
    }

    @Benchmark
    public void preparedC2c(PreparedState state, Blackhole blackhole) {
        state.complex.c2c(state.complexInput, true, 1.0, state.complexWorkspace);
        blackhole.consume(state.complexOutput);
    }

    @Benchmark
    public void preparedR2c(PreparedState state, Blackhole blackhole) {
        state.real.r2c(state.realInput, state.realOutput, true, 1.0, state.realWorkspace);
        blackhole.consume(state.realOutput);
    }

    private static double[] complexSample(int size) {
        double[] data = new double[2 * size];
        for (int index = 0; index < size; index++) {
            data[2 * index] = 0.25 + 0.125 * ((11 * index + 3) % 19);
            data[2 * index + 1] = -0.5 + 0.0625 * ((7 * index + 5) % 23);
        }
        return data;
    }

    private static double[] realSample(int size) {
        double[] data = new double[size];
        for (int index = 0; index < size; index++) {
            data[index] = 0.25 + 0.125 * ((13 * index + 7) % 17) - 0.03125 * (index % 5);
        }
        return data;
    }
}