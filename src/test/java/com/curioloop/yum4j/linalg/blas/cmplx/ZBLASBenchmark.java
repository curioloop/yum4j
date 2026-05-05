package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(0)
@State(Scope.Benchmark)
public class ZBLASBenchmark {

    @Param({"100", "1000"})
    int n1;

    @Param({"50", "100", "200", "500"})
    int n2;

    @Param({"50", "100", "200", "500"})
    int n3;

    double[] x1, y1;
    double[] x2, y2, A2, S2;
    double[] A3, B3, C3;

    @Setup(Level.Invocation)
    public void setup() {
        Random rng = new Random(42);

        x1 = new double[n1 * 2];
        y1 = new double[n1 * 2];
        for (int i = 0; i < n1 * 2; i++) {
            x1[i] = rng.nextDouble();
            y1[i] = rng.nextDouble();
        }

        x2 = new double[n2 * 2];
        y2 = new double[n2 * 2];
        A2 = new double[n2 * n2 * 2];
        for (int i = 0; i < n2 * 2; i++) {
            x2[i] = rng.nextDouble();
            y2[i] = rng.nextDouble();
        }
        for (int i = 0; i < n2 * n2 * 2; i++) {
            A2[i] = rng.nextDouble();
        }
        makeHermitian(A2, n2);

        S2 = new double[n2 * n2 * 2];
        for (int i = 0; i < n2 * n2 * 2; i++) {
            S2[i] = rng.nextDouble();
        }
        makeSymmetric(S2, n2);

        int m = n3, k = n3;
        A3 = new double[m * k * 2];
        B3 = new double[k * n3 * 2];
        C3 = new double[m * n3 * 2];
        for (int i = 0; i < m * k * 2; i++) A3[i] = rng.nextDouble();
        for (int i = 0; i < k * n3 * 2; i++) B3[i] = rng.nextDouble();
        for (int i = 0; i < m * n3 * 2; i++) C3[i] = rng.nextDouble();
    }

    private static void makeHermitian(double[] A, int n) {
        int lda = n;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int ij = (i * lda + j) * 2;
                int ji = (j * lda + i) * 2;
                double re = (A[ij] + A[ji]) / 2.0;
                double im = (A[ij + 1] - A[ji + 1]) / 2.0;
                A[ij] = re;
                A[ij + 1] = im;
                A[ji] = re;
                A[ji + 1] = -im;
            }
        }
    }

    private static void makeSymmetric(double[] A, int n) {
        int lda = n;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int ij = (i * lda + j) * 2;
                int ji = (j * lda + i) * 2;
                double re = (A[ij] + A[ji]) / 2.0;
                double im = (A[ij + 1] + A[ji + 1]) / 2.0;
                A[ij] = re;
                A[ij + 1] = im;
                A[ji] = re;
                A[ji + 1] = im;
            }
        }
    }

    @Benchmark
    public double[] zdotu() {
        double[] result = new double[2];
        Zdot.zdotu(n1, x1, 0, 1, y1, 0, 1, result, 0);
        return result;
    }

    @Benchmark
    public double[] zaxpy() {
        double[] y = y1.clone();
        Zaxpy.zaxpy(n1, 1.0, 0.0, x1, 0, 1, y, 0, 1);
        return y;
    }

    @Benchmark
    public double[] zdscal() {
        double[] x = x1.clone();
        Zdscal.zdscal(n1, 2.0, x, 0, 1);
        return x;
    }

    @Benchmark
    public double[] zgemv_notrans() {
        double[] y = y2.clone();
        Zgemv.zgemv(BLAS.Trans.NoTrans, n2, n2, 1.0, 0.0, A2, 0, n2, x2, 0, 1, 0.0, 0.0, y, 0, 1);
        return y;
    }

    @Benchmark
    public double[] zgemv_conjtrans() {
        double[] y = y2.clone();
        Zgemv.zgemv(BLAS.Trans.Conj, n2, n2, 1.0, 0.0, A2, 0, n2, x2, 0, 1, 0.0, 0.0, y, 0, 1);
        return y;
    }

    @Benchmark
    public double[] zhemv() {
        double[] y = y2.clone();
        Zhemv.zhemv(BLAS.Uplo.Upper, n2, 1.0, 0.0, A2, 0, n2, x2, 0, 1, 0.0, 0.0, y, 0, 1);
        return y;
    }

    @Benchmark
    public double[] zgemm_nn() {
        double[] c = C3.clone();
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, n3, n3, n3, 1.0, 0.0, A3, 0, n3, B3, 0, n3, 0.0, 0.0, c, 0, n3);
        return c;
    }

    @Benchmark
    public double[] zgemm_cn() {
        double[] c = C3.clone();
        Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n3, n3, n3, 1.0, 0.0, A3, 0, n3, B3, 0, n3, 0.0, 0.0, c, 0, n3);
        return c;
    }

    @Benchmark
    public double[] zgemm_tn() {
        double[] c = C3.clone();
        Zgemm.zgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, n3, n3, n3, 1.0, 0.0, A3, 0, n3, B3, 0, n3, 0.0, 0.0, c, 0, n3);
        return c;
    }

    @Benchmark
    public double[] zgemm_nt() {
        double[] c = C3.clone();
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, n3, n3, n3, 1.0, 0.0, A3, 0, n3, B3, 0, n3, 0.0, 0.0, c, 0, n3);
        return c;
    }

    @Benchmark
    public double[] zgemm_nc() {
        double[] c = C3.clone();
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, n3, n3, n3, 1.0, 0.0, A3, 0, n3, B3, 0, n3, 0.0, 0.0, c, 0, n3);
        return c;
    }

    @Benchmark
    public double[] zgemm_tt() {
        double[] c = C3.clone();
        Zgemm.zgemm(BLAS.Trans.Trans, BLAS.Trans.Trans, n3, n3, n3, 1.0, 0.0, A3, 0, n3, B3, 0, n3, 0.0, 0.0, c, 0, n3);
        return c;
    }

    @Benchmark
    public double[] zgemm_cc() {
        double[] c = C3.clone();
        Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.Conj, n3, n3, n3, 1.0, 0.0, A3, 0, n3, B3, 0, n3, 0.0, 0.0, c, 0, n3);
        return c;
    }

    @Benchmark
    public double[] zdotc() {
        double[] result = new double[2];
        Zdot.zdotc(n1, x1, 0, 1, y1, 0, 1, result, 0);
        return result;
    }

    @Benchmark
    public double[] zsymv() {
        double[] y = y2.clone();
        Zsymv.zsymv(BLAS.Uplo.Upper, n2, 1.0, 0.0, S2, 0, n2, x2, 0, 1, 0.0, 0.0, y, 0, 1);
        return y;
    }

    @Benchmark
    public double[] zher() {
        double[] a = A2.clone();
        Zher.zher(BLAS.Uplo.Upper, n2, 1.0, x2, 0, 1, a, 0, n2);
        return a;
    }

    @Benchmark
    public double[] zher2() {
        double[] a = A2.clone();
        Zher2.zher2(BLAS.Uplo.Upper, n2, 1.0, 0.0, x2, 0, 1, y2, 0, 1, a, 0, n2);
        return a;
    }

    @Benchmark
    public double[] zsyr() {
        double[] s = S2.clone();
        Zsyr.zsyr(BLAS.Uplo.Upper, n2, 1.0, 0.0, x2, 0, 1, s, 0, n2);
        return s;
    }

    @Benchmark
    public double[] zsyr2() {
        double[] s = S2.clone();
        Zsyr2.zsyr2(BLAS.Uplo.Upper, n2, 1.0, 0.0, x2, 0, 1, y2, 0, 1, s, 0, n2);
        return s;
    }

    @Benchmark
    public double[] ztrmv_notrans() {
        double[] x = x2.clone();
        Ztrmv.ztrmv(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n2, A2, 0, n2, x, 0, 1);
        return x;
    }

    @Benchmark
    public double[] ztrmv_conjtrans() {
        double[] x = x2.clone();
        Ztrmv.ztrmv(BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit, n2, A2, 0, n2, x, 0, 1);
        return x;
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
            .include(ZBLASBenchmark.class.getSimpleName())
            .forks(0).warmupIterations(5).measurementIterations(10).build();
        new Runner(opt).run();
    }
}
