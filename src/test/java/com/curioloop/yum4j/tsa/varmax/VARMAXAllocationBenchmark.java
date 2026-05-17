package com.curioloop.yum4j.tsa.varmax;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.mle.MLEResults;
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

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(0)
@State(Scope.Thread)
public class VARMAXAllocationBenchmark {
    @Param({"128", "512"})
    int nobs;

    @Param({"2", "4"})
    int kEndog;

    @Param({"1", "3"})
    int kExog;

    private VARMAXSpec spec;
    private VARMAX model;
    private VARMAXResults results;
    private double[] params;
    private double[][] futureExog;

    @Setup
    public void setup() {
        spec = buildSpec(nobs, kEndog, kExog);
        model = new VARMAX(spec);
        params = model.startParams();
        futureExog = exog(8, kExog, nobs);
        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        Optimization optimization = new Optimization(
            Double.NaN,
            params.clone(),
            -filterResult.logLikelihood(),
            Optimization.Status.GRADIENT_TOLERANCE_REACHED,
            0,
            0);
        results = new VARMAXResults(model, optimization, params.clone(), model.untransformParams(params),
            filterResult, MLEResults.Covariance.OPG);
    }

    @Benchmark
    public VARMAX constructModel() {
        return new VARMAX(spec);
    }

    @Benchmark
    public VARMAXSupport.Meta analyzeSupport() {
        return VARMAXSupport.analyze(spec);
    }

    @Benchmark
    public double[] startParamsCopy() {
        return model.startParams();
    }

    @Benchmark
    public double logLikelihood() {
        return model.logLikelihood(params);
    }

    @Benchmark
    public FilterResult filterFull() {
        return model.filter(params, FilterOptions.defaults());
    }

    @Benchmark
    public VARMAXResults resultsCopyFilterResult() {
        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        return new VARMAXResults(model, optimization(filterResult), params.clone(), model.untransformParams(params),
            filterResult, MLEResults.Covariance.OPG);
    }

    @Benchmark
    public VARMAXResults resultsAdoptFilterResult() {
        FilterResult filterResult = model.filter(params, FilterOptions.defaults());
        return VARMAXResults.adoptFilterResult(model, optimization(filterResult), params.clone(),
            model.untransformParams(params), filterResult, MLEResults.Covariance.OPG, null);
    }

    @Benchmark
    public VARMAXPrediction predictInSample() {
        int end = Math.min(nobs - 1, Math.max(0, nobs / 2));
        return results.predict(0, end);
    }

    @Benchmark
    public VARMAXPrediction forecastOutOfSample() {
        return results.forecast(futureExog.length, futureExog);
    }

    private static VARMAXSpec buildSpec(int nobs, int kEndog, int kExog) {
        return VARMAXSpec.builder(VARMAXOrder.of(1, 0), endog(nobs, kEndog))
            .trend("ct")
            .exog(exog(nobs, kExog))
            .errorCovariance(VARMAXErrorCovariance.DIAGONAL)
            .build();
    }

    private Optimization optimization(FilterResult filterResult) {
        return new Optimization(
            Double.NaN,
            params.clone(),
            -filterResult.logLikelihood(),
            Optimization.Status.GRADIENT_TOLERANCE_REACHED,
            0,
            0);
    }

    private static double[][] endog(int nobs, int kEndog) {
        Random rng = new Random(2026051701L + 31L * nobs + kEndog);
        double[][] values = new double[nobs][kEndog];
        double[] state = new double[kEndog];
        for (int t = 0; t < nobs; t++) {
            for (int j = 0; j < kEndog; j++) {
                state[j] = 0.35 * state[j] + 0.08 * (j + 1) + rng.nextGaussian() * 0.2;
                values[t][j] = state[j] + 0.03 * t;
            }
        }
        return values;
    }

    private static double[][] exog(int nobs, int kExog) {
        return exog(nobs, kExog, 0);
    }

    private static double[][] exog(int nobs, int kExog, int offset) {
        double[][] values = new double[nobs][kExog];
        for (int t = 0; t < nobs; t++) {
            for (int j = 0; j < kExog; j++) {
                int index = offset + t;
                values[t][j] = Math.sin(0.05 * (index + 1) * (j + 1)) + 0.01 * index;
            }
        }
        return values;
    }
}
