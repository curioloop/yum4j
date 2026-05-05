/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas;

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
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(0)
public class LapackFrontierRealBenchmark {

    private static final long SEED = 42L;

    @Benchmark
    public void dlabrd_direct(DlabrdDirectState state, Blackhole blackhole) {
        Dlabrd.dlabrd(state.m, state.n, state.nb, state.a, 0, state.lda,
                state.d, 0, state.e, 0, state.tauQ, 0, state.tauP, 0,
                state.x, 0, state.ldx, state.y, 0, state.ldy);
        blackhole.consume(state.d[0]);
        blackhole.consume(state.e[0]);
        blackhole.consume(state.a[state.a.length - 1]);
    }

    @Benchmark
    public void dgebrd_replay(DgebrdReplayState state, Blackhole blackhole) {
        requireZero("Dgebrd", Dgebrd.dgebrd(state.m, state.n, state.a, 0, state.lda,
                state.d, 0, state.e, 0, state.tauQ, 0, state.tauP, 0,
                state.work, 0, state.lwork));
        blackhole.consume(state.d[state.d.length - 1]);
        blackhole.consume(state.a[state.a.length - 1]);
    }

    @Benchmark
    public void dgesvd_replay(DgesvdReplayState state, Blackhole blackhole) {
        requireZero("Dgesvd", Dgesvd.dgesvd('S', 'S', state.m, state.n,
                state.a, 0, state.lda,
                state.s, 0,
                state.u, 0, state.ldu,
                state.vt, 0, state.ldvt,
                state.work, 0, state.lwork));
        blackhole.consume(state.s[0]);
        blackhole.consume(state.u[state.u.length - 1]);
        blackhole.consume(state.vt[state.vt.length - 1]);
    }

    @Benchmark
    public void dormqr_direct(DormqrDirectState state, Blackhole blackhole) {
        requireZero("Dormqr", Dormqr.dormqr(BLAS.Side.Left, BLAS.Trans.NoTrans,
                state.m, state.n, state.k,
                state.a, 0, state.lda,
                state.tau, 0,
                state.c, 0, state.ldc,
                state.work, 0, state.lwork));
        blackhole.consume(state.c[0]);
        blackhole.consume(state.c[state.c.length - 1]);
    }

    @Benchmark
    public void dormbr_q_replay(DormbrQReplayState state, Blackhole blackhole) {
        requireZero("Dormbr(Q)", Dormbr.dormbr('Q', BLAS.Side.Left, BLAS.Trans.NoTrans,
                state.m, state.n, state.k,
                state.a, 0, state.lda,
                state.tauQ, 0,
                state.c, 0, state.ldc,
                state.work, 0, state.lwork));
        blackhole.consume(state.c[0]);
        blackhole.consume(state.c[state.c.length - 1]);
    }

    @Benchmark
    public void dlatrd_direct(DlatrdDirectState state, Blackhole blackhole) {
        Dlatrd.dlatrd(state.uplo(), state.n, state.nb, state.a, 0, state.lda,
                state.e, 0, state.tau, 0, state.w, 0, state.ldw);
        blackhole.consume(state.e[0]);
        blackhole.consume(state.a[state.a.length - 1]);
    }

    @Benchmark
    public void dsyev_replay(DsyevReplayState state, Blackhole blackhole) {
        requireZero("Dsyev", Dsyev.dsyev('N', state.uploChar(), state.n,
                state.a, state.lda,
                state.wEig, 0,
                state.work, 0, state.lwork));
        blackhole.consume(state.wEig[0]);
        blackhole.consume(state.wEig[state.wEig.length - 1]);
    }

    @State(Scope.Thread)
    public static class DlabrdDirectState {

        @Param({"96", "160"})
        int n;

        int m;
        int nb;
        int lda;
        int ldx;
        int ldy;
        double[] baseA;

        double[] a;
        double[] d;
        double[] e;
        double[] tauQ;
        double[] tauP;
        double[] x;
        double[] y;

        @Setup(Level.Trial)
        public void trialSetup() {
            m = n + n / 2;
            nb = Math.min(32, Math.max(16, n / 4));
            lda = n;
            ldx = nb;
            ldy = nb;
            baseA = randomRealMatrix(new Random(SEED + n), m, lda);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            d = new double[nb];
            e = new double[nb];
            tauQ = new double[nb];
            tauP = new double[nb];
            x = new double[m * ldx];
            y = new double[n * ldy];
        }
    }

    @State(Scope.Thread)
    public static class DgebrdReplayState {

        @Param({"96", "160"})
        int n;

        int m;
        int lda;
        int minmn;
        int lwork;
        double[] baseA;

        double[] a;
        double[] d;
        double[] e;
        double[] tauQ;
        double[] tauP;
        double[] work;

        @Setup(Level.Trial)
        public void trialSetup() {
            m = n + n / 2;
            lda = n;
            minmn = Math.min(m, n);
            baseA = randomRealMatrix(new Random(SEED + 100 + n), m, lda);
            lwork = queryDgebrdWork(m, n, lda, minmn, baseA);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            d = new double[minmn];
            e = new double[minmn];
            tauQ = new double[minmn];
            tauP = new double[minmn];
            work = new double[lwork];
        }
    }

    @State(Scope.Thread)
    public static class DgesvdReplayState {

        @Param({"96", "128"})
        int n;

        int m;
        int lda;
        int ldu;
        int ldvt;
        int minmn;
        int lwork;
        double[] baseA;

        double[] a;
        double[] s;
        double[] u;
        double[] vt;
        double[] work;

        @Setup(Level.Trial)
        public void trialSetup() {
            m = n + n / 2;
            lda = n;
            minmn = Math.min(m, n);
            ldu = minmn;
            ldvt = n;
            baseA = randomRealMatrix(new Random(SEED + 200 + n), m, lda);
            lwork = queryDgesvdWork(m, n, lda, ldu, ldvt, minmn, baseA);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            s = new double[minmn];
            u = new double[m * ldu];
            vt = new double[minmn * ldvt];
            work = new double[lwork];
        }
    }

    @State(Scope.Thread)
    public static class DormqrDirectState {

        @Param({"96", "128"})
        int k;

        int m;
        int n;
        int lda;
        int ldc;
        int lwork;
        double[] baseA;
        double[] baseTau;
        double[] baseC;

        double[] a;
        double[] tau;
        double[] c;
        double[] work;

        @Setup(Level.Trial)
        public void trialSetup() {
            m = 2 * k;
            n = k + k / 2;
            lda = k;
            ldc = n;
            double[] qrInput = randomRealMatrix(new Random(SEED + 300 + k), m, lda);
            baseTau = new double[k];
            int qrWork = queryDgeqrfWork(m, k, lda, qrInput, baseTau.length);
            double[] workQrf = new double[qrWork];
            baseA = qrInput.clone();
            requireZero("Dgeqrf", Dgeqr.dgeqrf(m, k, baseA, 0, lda, baseTau, 0, workQrf, 0, qrWork));
            baseC = randomRealMatrix(new Random(SEED + 400 + k), m, ldc);
            lwork = queryDormqrWork(m, n, k, lda, ldc, baseA, baseTau, baseC);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            tau = baseTau.clone();
            c = baseC.clone();
            work = new double[lwork];
        }
    }

    @State(Scope.Thread)
    public static class DormbrQReplayState {

        @Param({"96", "128"})
        int k;

        int m;
        int n;
        int lda;
        int ldc;
        int lwork;
        double[] baseA;
        double[] baseTauQ;
        double[] baseC;

        double[] a;
        double[] tauQ;
        double[] c;
        double[] work;

        @Setup(Level.Trial)
        public void trialSetup() {
            m = 2 * k;
            n = k + k / 2;
            lda = k;
            ldc = n;
            double[] factor = randomRealMatrix(new Random(SEED + 500 + k), m, lda);
            int minmn = Math.min(m, k);
            double[] d = new double[minmn];
            double[] e = new double[minmn];
            baseTauQ = new double[minmn];
            double[] tauP = new double[minmn];
            int geWork = queryDgebrdWork(m, k, lda, minmn, factor);
            double[] workGe = new double[geWork];
            baseA = factor.clone();
            requireZero("Dgebrd", Dgebrd.dgebrd(m, k, baseA, 0, lda,
                    d, 0, e, 0, baseTauQ, 0, tauP, 0, workGe, 0, geWork));
            baseC = randomRealMatrix(new Random(SEED + 600 + k), m, ldc);
            lwork = queryDormbrQWork(m, n, k, lda, ldc, baseA, baseTauQ, baseC);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            tauQ = baseTauQ.clone();
            c = baseC.clone();
            work = new double[lwork];
        }
    }

    @State(Scope.Thread)
    public static class DlatrdDirectState {

        @Param({"128", "192"})
        int n;

        @Param({"L", "U"})
        String triangle;

        int nb;
        int lda;
        int ldw;
        double[] baseA;

        double[] a;
        double[] e;
        double[] tau;
        double[] w;

        @Setup(Level.Trial)
        public void trialSetup() {
            nb = Math.min(32, Math.max(16, n / 4));
            lda = n;
            ldw = nb;
            baseA = randomSymmetricMatrix(new Random(SEED + 700 + n + triangle.hashCode()), n);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            e = new double[n];
            tau = new double[n];
            w = new double[n * ldw];
        }

        BLAS.Uplo uplo() {
            return "U".equals(triangle) ? BLAS.Uplo.Upper : BLAS.Uplo.Lower;
        }
    }

    @State(Scope.Thread)
    public static class DsyevReplayState {

        @Param({"128", "192"})
        int n;

        @Param({"L", "U"})
        String triangle;

        int lda;
        int lwork;
        double[] baseA;

        double[] a;
        double[] wEig;
        double[] work;

        @Setup(Level.Trial)
        public void trialSetup() {
            lda = n;
            baseA = randomSymmetricMatrix(new Random(SEED + 800 + n + triangle.hashCode()), n);
            lwork = queryDsyevWork(n, lda, triangle.charAt(0), baseA);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            wEig = new double[n];
            work = new double[lwork];
        }

        char uploChar() {
            return triangle.charAt(0);
        }
    }

    public static void main(String[] args) throws RunnerException {
        Options options = new OptionsBuilder()
                .include(LapackFrontierRealBenchmark.class.getSimpleName())
                .forks(0)
                .warmupIterations(3)
                .measurementIterations(5)
                .build();
        new Runner(options).run();
    }

    private static int queryDgeqrfWork(int m, int n, int lda, double[] a, int tauLen) {
        double[] tau = new double[tauLen];
        double[] query = new double[1];
        requireZero("Dgeqrf query", Dgeqr.dgeqrf(m, n, a.clone(), 0, lda, tau, 0, query, 0, -1));
        return workspace(query[0]);
    }

    private static int queryDormqrWork(int m, int n, int k, int lda, int ldc,
                                       double[] a, double[] tau, double[] c) {
        double[] query = new double[1];
        requireZero("Dormqr query", Dormqr.dormqr(BLAS.Side.Left, BLAS.Trans.NoTrans,
                m, n, k, a.clone(), 0, lda, tau.clone(), 0, c.clone(), 0, ldc, query, 0, -1));
        return workspace(query[0]);
    }

    private static int queryDormbrQWork(int m, int n, int k, int lda, int ldc,
                                        double[] a, double[] tauQ, double[] c) {
        double[] query = new double[1];
        requireZero("Dormbr query", Dormbr.dormbr('Q', BLAS.Side.Left, BLAS.Trans.NoTrans,
                m, n, k, a.clone(), 0, lda, tauQ.clone(), 0, c.clone(), 0, ldc, query, 0, -1));
        return workspace(query[0]);
    }

    private static int queryDgebrdWork(int m, int n, int lda, int minmn, double[] a) {
        double[] d = new double[minmn];
        double[] e = new double[minmn];
        double[] tauQ = new double[minmn];
        double[] tauP = new double[minmn];
        double[] query = new double[1];
        requireZero("Dgebrd query", Dgebrd.dgebrd(m, n, a.clone(), 0, lda,
                d, 0, e, 0, tauQ, 0, tauP, 0, query, 0, -1));
        return workspace(query[0]);
    }

    private static int queryDgesvdWork(int m, int n, int lda, int ldu, int ldvt, int minmn, double[] a) {
        double[] s = new double[minmn];
        double[] u = new double[m * ldu];
        double[] vt = new double[minmn * ldvt];
        double[] query = new double[1];
        requireZero("Dgesvd query", Dgesvd.dgesvd('S', 'S', m, n, a.clone(), 0, lda,
                s, 0, u, 0, ldu, vt, 0, ldvt, query, 0, -1));
        return workspace(query[0]);
    }

    private static int queryDsyevWork(int n, int lda, char uplo, double[] a) {
        double[] w = new double[n];
        double[] query = new double[1];
        requireZero("Dsyev query", Dsyev.dsyev('N', uplo, n, a.clone(), lda, w, 0, query, 0, -1));
        return workspace(query[0]);
    }

    private static double[] randomRealMatrix(Random random, int rows, int lda) {
        double[] data = new double[rows * lda];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return data;
    }

    private static double[] randomSymmetricMatrix(Random random, int n) {
        double[] data = new double[n * n];
        for (int row = 0; row < n; row++) {
            for (int col = row; col < n; col++) {
                double value = random.nextDouble() * 2.0 - 1.0;
                data[row * n + col] = value;
                data[col * n + row] = value;
            }
        }
        return data;
    }

    private static int workspace(double query) {
        return Math.max(1, (int) Math.ceil(query));
    }

    private static void requireZero(String op, int info) {
        if (info != 0) {
            throw new IllegalStateException(op + " returned info=" + info);
        }
    }
}