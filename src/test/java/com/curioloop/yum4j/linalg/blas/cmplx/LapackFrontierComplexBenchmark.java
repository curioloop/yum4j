/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

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
public class LapackFrontierComplexBenchmark {

    private static final long SEED = 42L;

    @Benchmark
    public void zormqr_direct(ZormqrDirectState state, Blackhole blackhole) {
        requireZero("Zormqr", Zormqr.zormqr('L', 'N',
                state.m, state.n, state.k,
                state.a, 0, state.lda,
                state.tau, 0,
                state.c, 0, state.ldc,
                state.work, 0, state.lwork));
        blackhole.consume(state.c[0]);
        blackhole.consume(state.c[state.c.length - 1]);
    }

    @Benchmark
    public void zormbr_q_replay(ZormbrQReplayState state, Blackhole blackhole) {
        requireZero("Zormbr(Q)", Zormbr.zormbr('Q', 'L', 'N',
                state.m, state.n, state.k,
                state.a, 0, state.lda,
                state.tauQ, 0,
                state.c, 0, state.ldc,
                state.work, 0, state.lwork));
        blackhole.consume(state.c[0]);
        blackhole.consume(state.c[state.c.length - 1]);
    }

    @Benchmark
    public void zormlq_direct(ZormlqDirectState state, Blackhole blackhole) {
        requireZero("Zormlq", Zgelq.zormlq('R', 'C',
                state.m, state.n, state.k,
                state.a, 0, state.lda,
                state.tau, 0,
                state.c, 0, state.ldc,
                state.work, 0, state.lwork));
        blackhole.consume(state.c[0]);
        blackhole.consume(state.c[state.c.length - 1]);
    }

    @Benchmark
    public void zormbr_p_replay(ZormbrPReplayState state, Blackhole blackhole) {
        requireZero("Zormbr(P)", Zormbr.zormbr('P', 'R', 'N',
                state.m, state.n, state.k,
                state.a, 0, state.lda,
                state.tauP, 0,
                state.c, 0, state.ldc,
                state.work, 0, state.lwork));
        blackhole.consume(state.c[0]);
        blackhole.consume(state.c[state.c.length - 1]);
    }

    @Benchmark
    public void zlabrd_direct(ZlabrdDirectState state, Blackhole blackhole) {
        Zlabrd.zlabrd(state.m, state.n, state.k,
                state.a, 0, state.lda,
                state.d, 0, state.e, 0,
                state.tauQ, 0, state.tauP, 0,
                state.x, 0, state.ldx,
                state.y, 0, state.ldy);
        blackhole.consume(state.d[0]);
        blackhole.consume(state.e[0]);
        blackhole.consume(state.a[state.a.length - 1]);
    }

    @Benchmark
    public void zgebrd_replay(ZgebrdReplayState state, Blackhole blackhole) {
        requireZero("Zgebrd", Zgebrd.zgebrd(state.m, state.n,
                state.a, 0, state.lda,
                state.d, 0, state.e, 0,
                state.tauQ, 0, state.tauP, 0,
                state.work, 0, state.lwork));
        blackhole.consume(state.d[state.d.length - 1]);
        blackhole.consume(state.a[state.a.length - 1]);
    }

    @Benchmark
    public void zgesvd_replay(ZgesvdReplayState state, Blackhole blackhole) {
        requireZero("Zgesvd", Zgesvd.zgesvd('A', 'A',
                state.m, state.n,
                state.a, 0, state.lda,
                state.s, 0,
                state.u, 0, state.ldu,
                state.vt, 0, state.ldvt,
                state.work, 0, state.lwork));
        blackhole.consume(state.s[0]);
        blackhole.consume(state.u[state.u.length - 1]);
        blackhole.consume(state.vt[state.vt.length - 1]);
    }

    @State(Scope.Thread)
    public static class ZormqrDirectState {

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
            double[] qrInput = randomComplexMatrix(new Random(SEED + k), m, lda);
            baseTau = new double[2 * k];
            int qrfWork = queryZgeqrfWork(m, k, lda, qrInput, k);
            double[] workQrf = new double[complexArrayLength(qrfWork)];
            baseA = qrInput.clone();
            requireZero("Zgeqrf", Zgeqr.zgeqrf(m, k, baseA, 0, lda, baseTau, 0, workQrf, 0, qrfWork));
            baseC = randomComplexMatrix(new Random(SEED + 100 + k), m, ldc);
            lwork = queryZormqrWork(m, n, k, lda, ldc, baseA, baseTau, baseC);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            tau = baseTau.clone();
            c = baseC.clone();
            work = new double[complexArrayLength(lwork)];
        }
    }

    @State(Scope.Thread)
    public static class ZormbrQReplayState {

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
            m = k;
            n = k + k / 2;
            lda = k;
            ldc = n;
            double[] factor = randomComplexMatrix(new Random(SEED + 200 + k), m, lda);
            int minmn = Math.min(m, k);
            double[] d = new double[minmn];
            double[] e = new double[minmn];
            baseTauQ = new double[2 * minmn];
            double[] tauP = new double[2 * minmn];
            int geWork = queryZgebrdWork(m, k, lda, minmn, factor);
            double[] workGe = new double[complexArrayLength(geWork)];
            baseA = factor.clone();
            requireZero("Zgebrd", Zgebrd.zgebrd(m, k, baseA, 0, lda,
                    d, 0, e, 0, baseTauQ, 0, tauP, 0, workGe, 0, geWork));
            baseC = randomComplexMatrix(new Random(SEED + 300 + k), m, ldc);
            lwork = queryZormbrWork('Q', 'L', 'N', m, n, k, lda, ldc, baseA, baseTauQ, baseC);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            tauQ = baseTauQ.clone();
            c = baseC.clone();
            work = new double[complexArrayLength(lwork)];
        }
    }

    @State(Scope.Thread)
    public static class ZormlqDirectState {

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
            m = k + k / 2;
            n = 2 * k;
            lda = n;
            ldc = n;
            double[] lqInput = randomComplexMatrix(new Random(SEED + 400 + k), k, lda);
            baseTau = new double[2 * k];
            int lqWork = queryZgelqfWork(k, n, lda, lqInput, k);
            double[] workLq = new double[complexArrayLength(lqWork)];
            baseA = lqInput.clone();
            requireZero("Zgelqf", Zgelq.zgelqf(k, n, baseA, 0, lda, baseTau, 0, workLq, 0, lqWork));
            baseC = randomComplexMatrix(new Random(SEED + 500 + k), m, ldc);
            lwork = queryZormlqWork(m, n, k, lda, ldc, baseA, baseTau, baseC);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            tau = baseTau.clone();
            c = baseC.clone();
            work = new double[complexArrayLength(lwork)];
        }
    }

    @State(Scope.Thread)
    public static class ZormbrPReplayState {

        @Param({"96", "128"})
        int k;

        int m;
        int n;
        int lda;
        int ldc;
        int lwork;
        double[] baseA;
        double[] baseTauP;
        double[] baseC;

        double[] a;
        double[] tauP;
        double[] c;
        double[] work;

        @Setup(Level.Trial)
        public void trialSetup() {
            m = k + k / 2;
            n = 2 * k;
            lda = n;
            ldc = n;
            double[] factor = randomComplexMatrix(new Random(SEED + 600 + k), k, lda);
            int minmn = k;
            double[] d = new double[minmn];
            double[] e = new double[minmn];
            double[] tauQ = new double[2 * minmn];
            baseTauP = new double[2 * minmn];
            int geWork = queryZgebrdWork(k, n, lda, minmn, factor);
            double[] workGe = new double[complexArrayLength(geWork)];
            baseA = factor.clone();
            requireZero("Zgebrd", Zgebrd.zgebrd(k, n, baseA, 0, lda,
                    d, 0, e, 0, tauQ, 0, baseTauP, 0, workGe, 0, geWork));
            baseC = randomComplexMatrix(new Random(SEED + 700 + k), m, ldc);
            lwork = queryZormbrWork('P', 'R', 'N', m, n, k, lda, ldc, baseA, baseTauP, baseC);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            tauP = baseTauP.clone();
            c = baseC.clone();
            work = new double[complexArrayLength(lwork)];
        }
    }

    @State(Scope.Thread)
    public static class ZlabrdDirectState {

        @Param({"96", "128"})
        int n;

        int m;
        int k;
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
            k = n;
            lda = n;
            ldx = k;
            ldy = n;
            baseA = randomComplexMatrix(new Random(SEED + 800 + n), m, lda);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            d = new double[k];
            e = new double[k];
            tauQ = new double[2 * k];
            tauP = new double[2 * k];
            x = new double[m * ldx * 2];
            y = new double[(k + 1) * ldy * 2];
        }
    }

    @State(Scope.Thread)
    public static class ZgebrdReplayState {

        @Param({"96", "128"})
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
            baseA = randomComplexMatrix(new Random(SEED + 900 + n), m, lda);
            lwork = queryZgebrdWork(m, n, lda, minmn, baseA);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            d = new double[minmn];
            e = new double[minmn];
            tauQ = new double[2 * minmn];
            tauP = new double[2 * minmn];
            work = new double[complexArrayLength(lwork)];
        }
    }

    @State(Scope.Thread)
    public static class ZgesvdReplayState {

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
            ldu = m;
            ldvt = n;
            baseA = randomComplexMatrix(new Random(SEED + 1000 + n), m, lda);
            lwork = queryZgesvdWork(m, n, lda, ldu, ldvt, minmn, baseA);
        }

        @Setup(Level.Invocation)
        public void invocationSetup() {
            a = baseA.clone();
            s = new double[minmn];
            u = new double[m * ldu * 2];
            vt = new double[n * ldvt * 2];
            work = new double[complexArrayLength(lwork)];
        }
    }

    public static void main(String[] args) throws RunnerException {
        Options options = new OptionsBuilder()
                .include(LapackFrontierComplexBenchmark.class.getSimpleName())
                .forks(0)
                .warmupIterations(3)
                .measurementIterations(5)
                .build();
        new Runner(options).run();
    }

    private static int queryZgeqrfWork(int m, int n, int lda, double[] a, int k) {
        double[] tau = new double[2 * k];
        double[] query = new double[2];
        requireZero("Zgeqrf query", Zgeqr.zgeqrf(m, n, a.clone(), 0, lda, tau, 0, query, 0, -1));
        return complexWorkspace(query[0]);
    }

    private static int queryZgelqfWork(int m, int n, int lda, double[] a, int k) {
        double[] tau = new double[2 * k];
        double[] query = new double[2];
        requireZero("Zgelqf query", Zgelq.zgelqf(m, n, a.clone(), 0, lda, tau, 0, query, 0, -1));
        return complexWorkspace(query[0]);
    }

    private static int queryZormqrWork(int m, int n, int k, int lda, int ldc,
                                       double[] a, double[] tau, double[] c) {
        double[] query = new double[2];
        requireZero("Zormqr query", Zormqr.zormqr('L', 'N', m, n, k,
                a.clone(), 0, lda, tau.clone(), 0, c.clone(), 0, ldc, query, 0, -1));
        return complexWorkspace(query[0]);
    }

    private static int queryZormlqWork(int m, int n, int k, int lda, int ldc,
                                       double[] a, double[] tau, double[] c) {
        double[] query = new double[2];
        requireZero("Zormlq query", Zgelq.zormlq('R', 'C', m, n, k,
                a.clone(), 0, lda, tau.clone(), 0, c.clone(), 0, ldc, query, 0, -1));
        return complexWorkspace(query[0]);
    }

    private static int queryZormbrWork(char vect, char side, char trans,
                                       int m, int n, int k, int lda, int ldc,
                                       double[] a, double[] tau, double[] c) {
        double[] query = new double[2];
        requireZero("Zormbr query", Zormbr.zormbr(vect, side, trans, m, n, k,
                a.clone(), 0, lda, tau.clone(), 0, c.clone(), 0, ldc, query, 0, -1));
        return complexWorkspace(query[0]);
    }

    private static int queryZgebrdWork(int m, int n, int lda, int minmn, double[] a) {
        double[] d = new double[minmn];
        double[] e = new double[minmn];
        double[] tauQ = new double[2 * minmn];
        double[] tauP = new double[2 * minmn];
        double[] query = new double[2];
        requireZero("Zgebrd query", Zgebrd.zgebrd(m, n, a.clone(), 0, lda,
                d, 0, e, 0, tauQ, 0, tauP, 0, query, 0, -1));
        return complexWorkspace(query[0]);
    }

    private static int queryZgesvdWork(int m, int n, int lda, int ldu, int ldvt, int minmn, double[] a) {
        double[] s = new double[minmn];
        double[] u = new double[m * ldu * 2];
        double[] vt = new double[n * ldvt * 2];
        double[] query = new double[2];
        requireZero("Zgesvd query", Zgesvd.zgesvd('A', 'A', m, n,
                a.clone(), 0, lda,
                s, 0,
                u, 0, ldu,
                vt, 0, ldvt,
                query, 0, -1));
        return complexWorkspace(query[0]);
    }

    private static double[] randomComplexMatrix(Random random, int rows, int lda) {
        double[] data = new double[rows * lda * 2];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return data;
    }

    private static int complexWorkspace(double query) {
        return Math.max(1, (int) Math.ceil(query));
    }

    private static int complexArrayLength(int lwork) {
        return Math.max(2, 2 * lwork);
    }

    private static void requireZero(String op, int info) {
        if (info != 0) {
            throw new IllegalStateException(op + " returned info=" + info);
        }
    }
}