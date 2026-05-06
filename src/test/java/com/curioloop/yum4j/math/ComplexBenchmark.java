package com.curioloop.yum4j.math;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview"})
public class ComplexBenchmark {

    @State(Scope.Thread)
    public static class InputState {
        final Complex finite = new Complex(1.25, -0.75);
        final Complex finiteOther = new Complex(-0.5, 1.5);
        final Complex divisor = new Complex(2.0, -0.25);
        final Complex complexExponent = new Complex(0.6, -0.8);
        final Complex largeFinite = new Complex(1.0e300, -1.0e300);
        final Complex largeFiniteDivisor = new Complex(1.0e300, 1.0e300);
        final Complex realAxis = new Complex(0.75, 0.0);
        final Complex pureImaginary = new Complex(0.0, 0.75);
    }

    @Benchmark
    public Complex powDouble(InputState state) {
        return state.finite.pow(2.25);
    }

    @Benchmark
    public Complex powComplex(InputState state) {
        return state.finite.pow(state.complexExponent);
    }

    @Benchmark
    public Complex div(InputState state) {
        return state.finite.div(state.divisor);
    }

    @Benchmark
    public Complex recip(InputState state) {
        return state.divisor.recip();
    }

    @Benchmark
    public Complex divLargeFinite(InputState state) {
        return state.largeFinite.div(state.largeFiniteDivisor);
    }

    @Benchmark
    public Complex recipLargeFinite(InputState state) {
        return state.largeFiniteDivisor.recip();
    }

    @Benchmark
    public Complex sin(InputState state) {
        return state.finite.sin();
    }

    @Benchmark
    public Complex cos(InputState state) {
        return state.finite.cos();
    }

    @Benchmark
    public Complex tanRealAxis(InputState state) {
        return state.realAxis.tan();
    }

    @Benchmark
    public Complex tanhRealAxis(InputState state) {
        return state.realAxis.tanh();
    }

    @Benchmark
    public Complex expFinite(InputState state) {
        return state.finiteOther.exp();
    }

    @Benchmark
    public Complex expPureImaginary(InputState state) {
        return state.pureImaginary.exp();
    }

    @Benchmark
    public double argFinite(InputState state) {
        return state.finiteOther.arg();
    }

    @Benchmark
    public Complex logFinite(InputState state) {
        return state.finiteOther.log();
    }

    @Benchmark
    public Complex sqrtFinite(InputState state) {
        return state.finiteOther.sqrt();
    }

    @Benchmark
    public Complex asin(InputState state) {
        return state.finite.asin();
    }

    @Benchmark
    public Complex acos(InputState state) {
        return state.finite.acos();
    }

    @Benchmark
    public Complex atan(InputState state) {
        return state.finite.atan();
    }

    @Benchmark
    public Complex asinh(InputState state) {
        return state.finite.asinh();
    }

    @Benchmark
    public Complex acosh(InputState state) {
        return state.finiteOther.acosh();
    }

    @Benchmark
    public Complex atanh(InputState state) {
        return state.finite.atanh();
    }
}