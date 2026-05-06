package com.curioloop.yum4j.math;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.lang.reflect.Field;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview", "--add-modules=jdk.incubator.vector"})
public class HornerPolyBenchmark {

    @State(Scope.Thread)
    public static class PolynomialState {
        static final int INPUT_COUNT = 16;

        @Param({"3", "4", "5", "6", "8", "10", "12", "16"})
        int length;

        double[] coefficients;

        final double[] inputs = {
            -3.0, -2.0, -1.5, -1.0,
            -0.75, -0.25, 0.0, 0.125,
            0.25, 0.5, 0.75, 1.0,
            1.5, 2.0, 3.0, 4.0
        };

        @Setup(Level.Trial)
        public void setup() {
            coefficients = syntheticCoefficients(length);
        }
    }

    @State(Scope.Thread)
    public static class WorkloadState {
        static final int INPUT_COUNT = 8;

        double[] normalP;
        double[] normalQ;
        double[] expintP;
        double[] expintQ;
        double[] betaEven;

        final double[] normalInputs = {1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 0.05, 0.1, 0.2, 0.35};
        final double[] expintInputs = {1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 0.1, 0.25, 0.5, 1.0};
        final double[] betaInputs = {0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95};

        @Setup(Level.Trial)
        public void setup() throws ReflectiveOperationException {
            normalP = staticDoubleArray(Normal.class, "ERF_INV_P_LE_HALF_P");
            normalQ = staticDoubleArray(Normal.class, "ERF_INV_P_LE_HALF_Q");
            expintP = staticDoubleArray(Expint.class, "E1_SMALL_P");
            expintQ = staticDoubleArray(Expint.class, "E1_SMALL_Q");
            betaEven = staticDoubleArray(Beta.class, "TEMME_METHOD2_COEFFS_10");
        }
    }

    @Benchmark
    @OperationsPerInvocation(PolynomialState.INPUT_COUNT)
    public double referencePolynomial(PolynomialState state) {
        double sink = 0.0;
        for (double input : state.inputs) {
            sink += referenceEvaluatePolynomial(state.coefficients, input, state.length);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(PolynomialState.INPUT_COUNT)
    public double hornerPolyPolynomial(PolynomialState state) {
        double sink = 0.0;
        for (double input : state.inputs) {
            sink += HornerPoly.evaluatePolynomial(state.coefficients, input);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(WorkloadState.INPUT_COUNT)
    public double referenceNormalRational(WorkloadState state) {
        return sumReferenceRational(state.normalP, state.normalQ, state.normalInputs);
    }

    @Benchmark
    @OperationsPerInvocation(WorkloadState.INPUT_COUNT)
    public double hornerPolyNormalRational(WorkloadState state) {
        return sumHornerPolyRational(state.normalP, state.normalQ, state.normalInputs);
    }

    @Benchmark
    @OperationsPerInvocation(WorkloadState.INPUT_COUNT)
    public double referenceExpintRational(WorkloadState state) {
        return sumReferenceRational(state.expintP, state.expintQ, state.expintInputs);
    }

    @Benchmark
    @OperationsPerInvocation(WorkloadState.INPUT_COUNT)
    public double hornerPolyExpintRational(WorkloadState state) {
        return sumHornerPolyRational(state.expintP, state.expintQ, state.expintInputs);
    }

    @Benchmark
    @OperationsPerInvocation(WorkloadState.INPUT_COUNT)
    public double referenceBetaEvenPolynomial(WorkloadState state) {
        double sink = 0.0;
        for (double input : state.betaInputs) {
            sink += referenceEvaluateEvenPolynomial(state.betaEven, input);
        }
        return sink;
    }

    @Benchmark
    @OperationsPerInvocation(WorkloadState.INPUT_COUNT)
    public double hornerPolyBetaEvenPolynomial(WorkloadState state) {
        double sink = 0.0;
        for (double input : state.betaInputs) {
            sink += HornerPoly.evaluateEvenPolynomial(state.betaEven, input);
        }
        return sink;
    }

    private static double sumReferenceRational(double[] numerator, double[] denominator, double[] inputs) {
        double sink = 0.0;
        for (double input : inputs) {
            sink += referenceEvaluateRational(numerator, denominator, input);
        }
        return sink;
    }

    private static double sumHornerPolyRational(double[] numerator, double[] denominator, double[] inputs) {
        double sink = 0.0;
        for (double input : inputs) {
            sink += HornerPoly.evaluateRational(numerator, denominator, input);
        }
        return sink;
    }

    private static double referenceEvaluatePolynomial(double[] coefficients, double x, int length) {
        double value = coefficients[length - 1];
        for (int index = length - 2; index >= 0; index--) {
            value = Math.fma(value, x, coefficients[index]);
        }
        return value;
    }

    private static double referenceEvaluateEvenPolynomial(double[] coefficients, double x) {
        return referenceEvaluatePolynomial(coefficients, x * x, coefficients.length);
    }

    private static double referenceEvaluateRational(double[] numerator, double[] denominator, double x) {
        return referenceEvaluatePolynomial(numerator, x, numerator.length)
            / referenceEvaluatePolynomial(denominator, x, denominator.length);
    }

    private static double[] syntheticCoefficients(int length) {
        double[] coefficients = new double[length];
        for (int index = 0; index < length; index++) {
            double sign = (index & 1) == 0 ? 1.0 : -1.0;
            coefficients[index] = sign * (index + 1.0) / (length + index + 1.0);
        }
        return coefficients;
    }

    private static double[] staticDoubleArray(Class<?> type, String fieldName) throws ReflectiveOperationException {
        Field field = type.getDeclaredField(fieldName);
        field.setAccessible(true);
        return ((double[]) field.get(null)).clone();
    }
}