package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.model.builtin.LinearGauss;
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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(0)
@State(Scope.Thread)
public class ParticleAllocationBenchmark {
    private static final LinearGauss MODEL = new LinearGauss(0.9, 1.0, 0.5, 1.0);

    @Param({"256", "1024"})
    int particles;

    @Param({"64", "256"})
    int observations;

    @Param({"NONE", "FULL"})
    HistoryMode historyMode;

    private List<Double> data;
    private ParticleWorkspace publicWorkspace;
    private ParticleWorkspace borrowedWorkspace;

    @Setup
    public void setup() {
        data = data(observations);
        publicWorkspace = Particle.workspace();
        borrowedWorkspace = Particle.workspace();
        filterTask().run(publicWorkspace);
        filterTask().runBorrowedUnsafe(borrowedWorkspace);
    }

    @Benchmark
    public FilterResult<Double> filterPublicFreshWorkspace() {
        return filterTask().run();
    }

    @Benchmark
    public FilterResult<Double> filterPublicReusedWorkspace() {
        return filterTask().run(publicWorkspace);
    }

    @Benchmark
    public BorrowedFilterResult<Double> filterBorrowedReusedWorkspace() {
        return filterTask().runBorrowedUnsafe(borrowedWorkspace);
    }

    private ParticleFiltering<Double> filterTask() {
        ParticleFiltering<Double> task = Particle.filter(MODEL)
            .particles(particles)
            .observations(data)
            .seed(20260517L);
        return switch (historyMode) {
            case NONE -> task.noHistory();
            case FULL -> task.fullHistory();
            case ROLLING -> task.rollingHistory(Math.min(16, Math.max(1, observations / 4)));
            case PARTIAL -> task.partialHistory();
        };
    }

    private static List<Double> data(int length) {
        RandomGenerator rng = RandomGeneratorFactory.of("L64X128MixRandom").create(4242L);
        List<Double> out = new ArrayList<>(length);
        double x = MODEL.sigma0() * rng.nextGaussian();
        out.add(x + MODEL.sigmaY() * rng.nextGaussian());
        for (int t = 1; t < length; t++) {
            x = MODEL.rho() * x + MODEL.sigmaX() * rng.nextGaussian();
            out.add(x + MODEL.sigmaY() * rng.nextGaussian());
        }
        return out;
    }
}
