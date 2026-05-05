package com.curioloop.yum4j.fft;

import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
@Threads(1)
public class FFT1DBenchmark {
    @State(Scope.Thread)
    public static class ComplexState {
        @Param({"64", "96", "120", "257", "509", "1024", "4096"})
        int length;

        Transform.PreparedComplex fft;
        FftWorkspace workspace;
        double[] template;
        double[] work;

        @Setup(Level.Trial)
        public void setupTrial() {
            fft = Transform.prepareComplex(length);
            workspace = fft.createWorkspace();
            template = complexSample(length);
            work = new double[template.length];
        }

        @Setup(Level.Invocation)
        public void setupInvocation() {
            System.arraycopy(template, 0, work, 0, template.length);
        }
    }

    @State(Scope.Thread)
    public static class RealState {
        @Param({"64", "96", "120", "257", "509", "1024", "4096"})
        int length;

        Transform.PreparedReal fft;
        FftWorkspace workspace;
        double[] inputTemplate;
        double[] input;
        double[] spectrum;
        double[] inverseSpectrumTemplate;
        double[] inverseSpectrum;
        double[] restored;

        @Setup(Level.Trial)
        public void setupTrial() {
            fft = Transform.prepareReal(length);
            workspace = fft.createWorkspace();
            inputTemplate = realSample(length);
            input = new double[length];
            spectrum = new double[2 * (length / 2 + 1)];
            inverseSpectrumTemplate = new double[spectrum.length];
            fft.r2c(inputTemplate, inverseSpectrumTemplate, true, 1.0, workspace);
            inverseSpectrum = new double[inverseSpectrumTemplate.length];
            restored = new double[length];
        }

        @Setup(Level.Invocation)
        public void setupInvocation() {
            System.arraycopy(inputTemplate, 0, input, 0, length);
            System.arraycopy(inverseSpectrumTemplate, 0, inverseSpectrum, 0, inverseSpectrumTemplate.length);
        }
    }

    @Benchmark
    public void fftC2cForward(ComplexState state, Blackhole blackhole) {
        state.fft.c2c(state.work, true, 1.0, state.workspace);
        blackhole.consume(state.work);
    }

    @Benchmark
    public void fftR2cForward(RealState state, Blackhole blackhole) {
        state.fft.r2c(state.input, state.spectrum, true, 1.0, state.workspace);
        blackhole.consume(state.spectrum);
    }

    @Benchmark
    public void fftC2rInverse(RealState state, Blackhole blackhole) {
        state.fft.c2r(state.inverseSpectrum, state.restored, false, 1.0 / state.length, state.workspace);
        blackhole.consume(state.restored);
    }

    private static double[] complexSample(int length) {
        double[] data = new double[2 * length];
        for (int index = 0; index < length; index++) {
            data[2 * index] = 0.25 + 0.125 * ((11 * index + 3) % 19);
            data[2 * index + 1] = -0.5 + 0.0625 * ((7 * index + 5) % 23);
        }
        return data;
    }

    private static double[] realSample(int length) {
        double[] data = new double[length];
        for (int index = 0; index < length; index++) {
            data[index] = 0.25 + 0.125 * ((13 * index + 7) % 17) - 0.03125 * (index % 5);
        }
        return data;
    }
}