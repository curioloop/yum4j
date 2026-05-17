/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.trf;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(0)
public class RobustLossBenchmark {

    @Param({"SOFT_L1", "HUBER", "CAUCHY", "ARCTAN"})
    public RobustLoss loss;

    @Param({"250:8", "1000:30", "2000:50"})
    public String shape;

    private int m;
    private int n;
    private double lossScale;
    private double[] baseF;
    private double[] baseJ;
    private double[] f;
    private double[] j;

    @Setup(Level.Trial)
    public void setupTrial() {
        String[] parts = shape.split(":");
        m = Integer.parseInt(parts[0]);
        n = Integer.parseInt(parts[1]);
        lossScale = 2.0;
        Random random = new Random(0x7A_FEL + m * 31L + n);
        baseF = new double[m];
        baseJ = new double[m * n];
        f = new double[m];
        j = new double[m * n];
        for (int i = 0; i < m; i++) {
            double outlier = (i % 17 == 0) ? 8.0 : 1.0;
            baseF[i] = outlier * (Math.sin(i * 0.17) + 0.1 * random.nextGaussian());
        }
        for (int i = 0; i < baseJ.length; i++) {
            baseJ[i] = Math.cos(i * 0.013) + 0.05 * random.nextGaussian();
        }
    }

    @Setup(Level.Invocation)
    public void setupInvocation() {
        System.arraycopy(baseF, 0, f, 0, baseF.length);
        System.arraycopy(baseJ, 0, j, 0, baseJ.length);
    }

    @Benchmark
    public double separateCostThenScale() {
        double cost = loss.cost(f, m, lossScale);
        loss.scaleJF(f, j, m, n, lossScale);
        return cost + f[m - 1] + j[j.length - 1];
    }

    @Benchmark
    public double fusedCostAndScale() {
        double cost = loss.costAndScaleJF(f, j, m, n, lossScale);
        return cost + f[m - 1] + j[j.length - 1];
    }

    public static void main(String[] args) throws RunnerException {
        Options options = new OptionsBuilder()
                .include(".*RobustLossBenchmark.*")
                .param("loss", "SOFT_L1", "HUBER", "CAUCHY", "ARCTAN")
                .param("shape", "250:8", "1000:30", "2000:50")
                .warmupIterations(3)
                .measurementIterations(5)
                .forks(0)
                .build();
        new Runner(options).run();
    }
}
