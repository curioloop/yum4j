package com.curioloop.yum4j.linalg.reg;

import com.curioloop.yum4j.linalg.Regressor;
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
import org.openjdk.jmh.annotations.Warmup;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 300, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 300, timeUnit = TimeUnit.MILLISECONDS)
@Fork(0)
@State(Scope.Thread)
public class OLSVectorOpsBenchmark {

    @Param({"128", "1024"})
    int n;

    @Param({"5"})
    int k;

    private double[] y;
    private double[] sourceX;
    private double[] x;
    private double[] scalarSigma;
    private OLS.Pool qrPool;
    private OLS.Pool pinvPool;
    private GLS.Pool glsQrPool;

    @Setup(Level.Trial)
    public void setupTrial() {
        y = new double[n];
        sourceX = new double[n * k];
        x = new double[n * k];
        fillDesign(y, sourceX, n, k, 0x4f4c53L);
        scalarSigma = new double[]{2.5};
        qrPool = new OLS.Pool().ensure(n, k,
                Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);
        pinvPool = new OLS.Pool().ensure(n, k,
                Regressor.Opts.PINV, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);
        glsQrPool = new GLS.Pool();
    }

    @Setup(Level.Invocation)
    public void setupInvocation() {
        System.arraycopy(sourceX, 0, x, 0, sourceX.length);
    }

    @Benchmark
    public double qrFitness() {
        OLS fit = Regressor.ols(y, x, n, k, qrPool,
                Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);
        double[] residual = fit.residual(false);
        return residual[0] + residual[n - 1];
    }

    @Benchmark
    public double qrR2() {
        OLS fit = Regressor.ols(y, x, n, k, qrPool,
                Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);
        return fit.r2(false);
    }

    @Benchmark
    public double pinvFitness() {
        OLS fit = Regressor.ols(y, x, n, k, pinvPool,
                Regressor.Opts.PINV, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);
        double[] residual = fit.residual(false);
        return residual[0] + residual[n - 1];
    }

    @Benchmark
    public double pinvR2() {
        OLS fit = Regressor.ols(y, x, n, k, pinvPool,
                Regressor.Opts.PINV, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);
        return fit.r2(false);
    }

    @Benchmark
    public double glsScalarQrFitness() {
        GLS fit = Regressor.gls(y, x, scalarSigma, n, k, glsQrPool,
                Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);
        double[] residual = fit.residual(false);
        return residual[0] + residual[n - 1];
    }

    private static void fillDesign(double[] y, double[] x, int n, int k, long seed) {
        Random random = new Random(seed);
        double[] beta = new double[k];
        for (int j = 0; j < k; j++) beta[j] = 0.8 - 0.13 * j;
        for (int i = 0; i < n; i++) {
            double signal = 0.0;
            for (int j = 0; j < k; j++) {
                double value = j == 0
                        ? 1.0
                        : Math.sin((i + 1) * (j + 1) * 0.021) + 0.03 * random.nextGaussian();
                x[i * k + j] = value;
                signal += value * beta[j];
            }
            y[i] = signal + 0.01 * random.nextGaussian();
        }
    }
}