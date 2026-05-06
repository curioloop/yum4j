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
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview", "--add-modules=jdk.incubator.vector"})
public class ComplexAllocationBenchmark {

    private static final int INVOCATIONS = 256;
    private static final int STATE_SIZE = 64;
    private static final int MASK = STATE_SIZE - 1;

    @State(Scope.Thread)
    public static class ArithmeticChainState {
        final Complex[] left = new Complex[STATE_SIZE];
        final Complex[] right = new Complex[STATE_SIZE];
        final Complex[] divisor = new Complex[STATE_SIZE];
        final double[] scale = new double[STATE_SIZE];
        int cursor;

        @Setup(Level.Trial)
        public void setup() {
            for (int index = 0; index < STATE_SIZE; index++) {
                left[index] = new Complex(0.75 + 0.015625 * index, -0.45 + 0.0078125 * index);
                right[index] = new Complex(-0.25 + 0.01171875 * index, 1.10 - 0.009765625 * index);
                divisor[index] = new Complex(1.50 + 0.01953125 * index, -0.35 - 0.00390625 * index);
                scale[index] = 0.80 + 0.0025 * index;

                Complex result = arithmeticChain(this, index);
                if (!result.isFinite()) {
                    throw new IllegalStateException("non-finite arithmetic chain seed at index=" + index);
                }
            }
        }
    }

    @State(Scope.Thread)
    public static class HestonChainState {
        final double[] t0 = new double[STATE_SIZE];
        final double[] rpsig = new double[STATE_SIZE];
        final double[] phi = new double[STATE_SIZE];
        final double[] sigma2 = new double[STATE_SIZE];
        final double[] sign = new double[STATE_SIZE];
        final double[] maturity = new double[STATE_SIZE];
        final double[] v0 = new double[STATE_SIZE];
        final double[] kappaThetaOverSigma2 = new double[STATE_SIZE];
        final double[] imagShift = new double[STATE_SIZE];
        int cursor;

        @Setup(Level.Trial)
        public void setup() {
            for (int index = 0; index < STATE_SIZE; index++) {
                double sigma = 0.28 + 0.00075 * index;
                double rho = -0.70 + 0.0025 * index;
                double kappa = 1.10 + 0.01 * index;
                double theta = 0.035 + 0.0002 * index;
                double localV0 = 0.04 + 0.00015 * index;
                double localPhi = 0.15 + 0.01 * index;
                int j = (index & 1) == 0 ? 1 : 2;

                phi[index] = localPhi;
                sigma2[index] = sigma * sigma;
                sign[index] = j == 1 ? 1.0 : -1.0;
                t0[index] = kappa - (j == 1 ? rho * sigma : 0.0);
                rpsig[index] = rho * sigma * localPhi;
                maturity[index] = 0.25 + 0.01 * index;
                v0[index] = localV0;
                kappaThetaOverSigma2[index] = kappa * theta / sigma2[index];
                imagShift[index] = localPhi * (-0.20 + 0.01 * index);

                Complex result = hestonKernelChain(this, index);
                if (!result.isFinite()) {
                    throw new IllegalStateException("non-finite Heston chain seed at index=" + index);
                }

                Complex optimizedResult = hestonKernelChainReducedTemps(this, index);
                if (!optimizedResult.isFinite()) {
                    throw new IllegalStateException("non-finite optimized Heston chain seed at index=" + index);
                }
                assertComplexClose(result, optimizedResult, 1.0e-12, index);
            }
        }
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double arithmeticChainScalarized(ArithmeticChainState state) {
        double sink = 0.0;
        int cursor = state.cursor;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            int index = (cursor + invocation) & MASK;
            Complex result = arithmeticChain(state, index);
            sink += result.norm();
        }
        state.cursor = (cursor + INVOCATIONS) & MASK;
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public void arithmeticChainEscaping(ArithmeticChainState state, Blackhole blackhole) {
        int cursor = state.cursor;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            int index = (cursor + invocation) & MASK;
            blackhole.consume(arithmeticChain(state, index));
        }
        state.cursor = (cursor + INVOCATIONS) & MASK;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double hestonKernelChainScalarized(HestonChainState state) {
        double sink = 0.0;
        int cursor = state.cursor;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            int index = (cursor + invocation) & MASK;
            sink += hestonKernelChain(state, index).imag();
        }
        state.cursor = (cursor + INVOCATIONS) & MASK;
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public void hestonKernelChainEscaping(HestonChainState state, Blackhole blackhole) {
        int cursor = state.cursor;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            int index = (cursor + invocation) & MASK;
            blackhole.consume(hestonKernelChain(state, index));
        }
        state.cursor = (cursor + INVOCATIONS) & MASK;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public double hestonKernelChainReducedTempsScalarized(HestonChainState state) {
        double sink = 0.0;
        int cursor = state.cursor;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            int index = (cursor + invocation) & MASK;
            sink += hestonKernelChainReducedTemps(state, index).imag();
        }
        state.cursor = (cursor + INVOCATIONS) & MASK;
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(INVOCATIONS)
    public void hestonKernelChainReducedTempsEscaping(HestonChainState state, Blackhole blackhole) {
        int cursor = state.cursor;
        for (int invocation = 0; invocation < INVOCATIONS; invocation++) {
            int index = (cursor + invocation) & MASK;
            blackhole.consume(hestonKernelChainReducedTemps(state, index));
        }
        state.cursor = (cursor + INVOCATIONS) & MASK;
    }

    private static Complex arithmeticChain(ArithmeticChainState state, int index) {
        Complex numerator = state.left[index]
            .mul(state.right[index])
            .add(Complex.ONE)
            .sub(state.divisor[index]);
        Complex denominator = state.right[index].add(Complex.ONE);
        return numerator
            .div(denominator)
            .mul(state.scale[index])
            .sqrt();
    }

    private static Complex hestonKernelChain(HestonChainState state, int index) {
        Complex t1 = new Complex(state.t0[index], -state.rpsig[index]);
        Complex d = t1.mul(t1)
            .sub(new Complex(-state.phi[index], state.sign[index]).mul(state.sigma2[index] * state.phi[index]))
            .sqrt();
        Complex ex = d.mul(-state.maturity[index]).exp();
        Complex pTerm = t1.sub(d).div(t1.add(d));
        Complex g = Complex.ONE.sub(pTerm.mul(ex))
            .div(Complex.ONE.sub(pTerm))
            .log();

        Complex dPart = new Complex(state.v0[index], 0.0)
            .mul(t1.sub(d))
            .mul(Complex.ONE.sub(ex))
            .div(new Complex(state.sigma2[index], 0.0)
                .mul(Complex.ONE.sub(ex.mul(pTerm))));
        Complex cPart = new Complex(state.kappaThetaOverSigma2[index], 0.0)
            .mul(t1.sub(d).mul(state.maturity[index]).sub(g.mul(2.0)));

        return dPart
            .add(cPart)
            .add(new Complex(0.0, state.imagShift[index]))
            .exp();
    }

    private static Complex hestonKernelChainReducedTemps(HestonChainState state, int index) {
        double phi = state.phi[index];
        double sigma2 = state.sigma2[index];
        double maturity = state.maturity[index];

        Complex t1 = new Complex(state.t0[index], -state.rpsig[index]);
        Complex d = t1.mul(t1)
            .sub(new Complex(-phi, state.sign[index]).mul(sigma2 * phi))
            .sqrt();
        Complex t1MinusD = t1.sub(d);
        Complex t1PlusD = t1.add(d);
        Complex ex = d.mul(-maturity).exp();
        Complex oneMinusEx = Complex.ONE.sub(ex);
        Complex pTerm = t1MinusD.div(t1PlusD);
        Complex oneMinusPTerm = Complex.ONE.sub(pTerm);
        Complex exMulPTerm = ex.mul(pTerm);
        Complex oneMinusExMulPTerm = Complex.ONE.sub(exMulPTerm);
        Complex g = oneMinusExMulPTerm.div(oneMinusPTerm).log();

        Complex dPart = t1MinusD
            .mul(state.v0[index])
            .mul(oneMinusEx)
            .div(oneMinusExMulPTerm)
            .div(sigma2);
        Complex cPart = t1MinusD
            .mul(maturity)
            .sub(g.mul(2.0))
            .mul(state.kappaThetaOverSigma2[index]);

        return dPart
            .add(cPart)
            .add(new Complex(0.0, state.imagShift[index]))
            .exp();
    }

    private static void assertComplexClose(Complex expected, Complex actual, double tolerance, int index) {
        if (Math.abs(expected.real() - actual.real()) > tolerance
            || Math.abs(expected.imag() - actual.imag()) > tolerance) {
            throw new IllegalStateException(
                "optimized Heston chain deviated at index=" + index
                    + ", expected=" + expected
                    + ", actual=" + actual);
        }
    }
}