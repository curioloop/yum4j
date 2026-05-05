package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.kalman.filter.FilterResult;
import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.mle.MLECovariance;
import com.curioloop.yum4j.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
import com.curioloop.yum4j.optim.Optimization;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Allocation-focused benchmark for the SARIMAX/MLE outer layer.
 *
 * <p>Run with JDK 25 and the GC profiler, then compare {@code gc.alloc.rate.norm}:</p>
 * <pre>
 * mvn -q -DskipTests test-compile dependency:build-classpath -Dmdep.outputFile=tmp/jmh-test-classpath.txt
 * /Library/Java/JavaVirtualMachines/jdk-25.jdk/Contents/Home/bin/java --add-modules jdk.incubator.vector \
 *   -cp target/test-classes:target/classes:$(cat tmp/jmh-test-classpath.txt) \
 *   org.openjdk.jmh.Main SARIMAXAllocationBenchmark -prof gc -f 1 -wi 3 -i 5 -tu us -bm avgt \
 *   -jvm /Library/Java/JavaVirtualMachines/jdk-25.jdk/Contents/Home/bin/java \
 *   -jvmArgsAppend=--add-modules=jdk.incubator.vector
 * </pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(0)
@State(Scope.Thread)
public class SARIMAXAllocationBenchmark {

    @Param({"256", "1024"})
    int nobs;

    private SARIMAX ar1;
    private SARIMAX integratedMa1;
    private SARIMAX airline;
    private SARIMAXResults ar1Results;
    private double[] ar1Params;
    private double[] integratedMa1Params;
    private double[] airlineParams;
    private SmootherSpec stateOnlySmootherSpec;

    @Setup
    public void setup() {
        ar1Params = new double[]{0.62, 0.35};
        integratedMa1Params = new double[]{0.41, 0.24};
        airlineParams = new double[]{0.38, 0.27, 0.18};

        ar1 = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(1, 0, 0),
            generateAr1(ar1Params[0], Math.sqrt(ar1Params[1]), nobs, 2026050301L)).build());
        integratedMa1 = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1),
            generateIntegratedMa1(integratedMa1Params[0], Math.sqrt(integratedMa1Params[1]), nobs, 2026050302L)).build());
        airline = new SARIMAX(SARIMAXSpec.builder(ARIMAOrder.of(0, 1, 1),
                generateAirlineLike(airlineParams[0], airlineParams[1], Math.sqrt(airlineParams[2]), nobs, 2026050303L))
            .seasonalOrder(SeasonalOrder.of(0, 1, 1, 12))
            .build());
        stateOnlySmootherSpec = SmootherSpec.conventional().stateOnly();

        FilterResult filterResult = ar1.filter(ar1Params, FilterSpec.full());
        Optimization optimization = new Optimization(
            Double.NaN,
            ar1Params.clone(),
            -filterResult.logLikelihood(),
            Optimization.Status.GRADIENT_TOLERANCE_REACHED,
            0,
            0);
        ar1Results = new SARIMAXResults(
            ar1,
            optimization,
            ar1Params.clone(),
            ar1.untransformParams(ar1Params),
            filterResult,
            MLECovariance.OPG);
    }

    @Benchmark
    public double ar1LogLikelihood() {
        return ar1.logLikelihood(ar1Params);
    }

    @Benchmark
    public FilterResult ar1FilterFull() {
        return ar1.filter(ar1Params, FilterSpec.full());
    }

    @Benchmark
    public SmootherResult ar1SmoothStateOnly() {
        return ar1.smooth(ar1Params, stateOnlySmootherSpec);
    }

    @Benchmark
    public double integratedMa1LogLikelihood() {
        return integratedMa1.logLikelihood(integratedMa1Params);
    }

    @Benchmark
    public double airlineLogLikelihood() {
        return airline.logLikelihood(airlineParams);
    }

    @Benchmark
    public SARIMAXPrediction predictInSample() {
        int end = Math.min(nobs - 1, Math.max(0, nobs / 2));
        return ar1Results.predict(0, end);
    }

    @Benchmark
    public SARIMAXPrediction predictOutOfSampleDynamic() {
        int start = Math.max(0, nobs - 16);
        int end = nobs + 12;
        return ar1Results.predict(start, end, true, null);
    }

    private static double[] generateAr1(double phi, double sigma, int length, long seed) {
        Random rng = new Random(seed);
        double[] values = new double[length];
        double state = 0.0;
        for (int i = 0; i < length; i++) {
            state = phi * state + sigma * rng.nextGaussian();
            values[i] = state;
        }
        return values;
    }

    private static double[] generateIntegratedMa1(double theta, double sigma, int length, long seed) {
        Random rng = new Random(seed);
        double[] values = new double[length];
        double level = 0.0;
        double previousShock = 0.0;
        for (int i = 0; i < length; i++) {
            double shock = sigma * rng.nextGaussian();
            level += shock + theta * previousShock;
            values[i] = level;
            previousShock = shock;
        }
        return values;
    }

    private static double[] generateAirlineLike(double theta,
                                                double seasonalTheta,
                                                double sigma,
                                                int length,
                                                long seed) {
        Random rng = new Random(seed);
        double[] values = new double[length];
        double[] shocks = new double[Math.max(length, 12)];
        double level = 0.0;
        for (int i = 0; i < length; i++) {
            double shock = sigma * rng.nextGaussian();
            double previous = i > 0 ? shocks[i - 1] : 0.0;
            double previousSeasonal = i >= 12 ? shocks[i - 12] : 0.0;
            double innovation = shock + theta * previous + seasonalTheta * previousSeasonal;
            double seasonalBase = i >= 12 ? values[i - 12] : 0.0;
            level += innovation;
            values[i] = seasonalBase + level;
            shocks[i] = shock;
        }
        return values;
    }
}