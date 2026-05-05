package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.kalman.filter.FilterResult;
import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.filter.KalmanFilter;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.model.ModelFixture;
import com.curioloop.yum4j.kalman.smooth.KalmanSmoother;
import com.curioloop.yum4j.kalman.smooth.SimulationSmoother;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherSpec;
import com.curioloop.yum4j.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
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
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Run with:
 * mvn test-compile exec:java -Dexec.mainClass=org.openjdk.jmh.Main \
 *   -Dexec.args="KalmanMemoryBenchmark -prof gc -f 0 -wi 3 -i 5"
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(0)
@State(Scope.Thread)
public class KalmanMemoryBenchmark {

    @Param({"256", "1024"})
    int nobs;

    private ModelFixture rep;
    private ModelFixture diffuseRep;
    private StateInitialization init;
    private StateInitialization diffuseInit;
    private FilterSpec fullFilterSpec;
    private FilterSpec compactFilterSpec;
    private FilterSpec smootherFilterSpec;
    private SmootherSpec fullSmootherSpec;
    private SmootherSpec stateOnlySmootherSpec;
    private SmootherSpec classicalFullSmootherSpec;
    private SmootherSpec alternativeFullSmootherSpec;
    private FilterResult smootherFilterInput;
    private double[] measurementVariates;
    private double[] stateVariates;
    private double[] initialVariates;
    private KalmanFilter.Pool borrowedFullPool;
    private KalmanFilter.Pool reservedBorrowedFullPool;
    private KalmanFilter.Pool borrowedCompactPool;
    private KalmanFilter.Pool reservedBorrowedCompactPool;
    private KalmanFilter.Pool borrowedDiffuseFullPool;
    private KalmanFilter.Pool reservedBorrowedDiffuseFullPool;
    private KalmanSmoother.Pool borrowedFullSmootherPool;
    private KalmanSmoother.Pool borrowedStateOnlySmootherPool;
    private KalmanSmoother.Pool borrowedClassicalFullSmootherPool;
    private KalmanSmoother.Pool borrowedAlternativeFullSmootherPool;
    private KalmanSmoother.Pool reservedBorrowedFullSmootherPool;
    private KalmanSmoother.Pool reservedBorrowedStateOnlySmootherPool;
    private KalmanSmoother.Pool reservedBorrowedClassicalFullSmootherPool;
    private KalmanSmoother.Pool reservedBorrowedAlternativeFullSmootherPool;
    private SimulationSmoother.Pool borrowedSimulationPool;
    private SimulationSmoother.Pool reservedBorrowedSimulationPool;
    private Random simulationRandomFullRng;
    private Random simulationRandomBorrowedRng;
    private Random simulationRandomReservedBorrowedRng;

    @Setup(Level.Trial)
    public void setup() {
        rep = buildBenchmarkModel(nobs);
        diffuseRep = buildDiffuseBenchmarkModel(nobs);
        init = StateInitialization.approximateDiffuse();
        diffuseInit = StateInitialization.diffuse();
        fullFilterSpec = FilterSpec.full();
        compactFilterSpec = FilterSpec.compact();
        smootherFilterSpec = FilterSpec.conventionalSmoothing();
        fullSmootherSpec = SmootherSpec.conventional();
        stateOnlySmootherSpec = SmootherSpec.conventional().stateOnly();
        classicalFullSmootherSpec = SmootherSpec.classical();
        alternativeFullSmootherSpec = SmootherSpec.alternative();

        borrowedFullPool = new KalmanFilter.Pool();
        reservedBorrowedFullPool = new KalmanFilter.Pool();
        borrowedCompactPool = new KalmanFilter.Pool();
        reservedBorrowedCompactPool = new KalmanFilter.Pool();
        borrowedDiffuseFullPool = new KalmanFilter.Pool();
        reservedBorrowedDiffuseFullPool = new KalmanFilter.Pool();
        borrowedFullSmootherPool = new KalmanSmoother.Pool();
        borrowedStateOnlySmootherPool = new KalmanSmoother.Pool();
        borrowedClassicalFullSmootherPool = new KalmanSmoother.Pool();
        borrowedAlternativeFullSmootherPool = new KalmanSmoother.Pool();
        reservedBorrowedFullSmootherPool = new KalmanSmoother.Pool();
        reservedBorrowedStateOnlySmootherPool = new KalmanSmoother.Pool();
        reservedBorrowedClassicalFullSmootherPool = new KalmanSmoother.Pool();
        reservedBorrowedAlternativeFullSmootherPool = new KalmanSmoother.Pool();
        borrowedSimulationPool = new SimulationSmoother.Pool();
        reservedBorrowedSimulationPool = new SimulationSmoother.Pool();
        simulationRandomFullRng = new Random(17L);
        simulationRandomBorrowedRng = new Random(17L);
        simulationRandomReservedBorrowedRng = new Random(17L);

        KalmanFilter.filter(rep, init, borrowedFullPool, fullFilterSpec);
        KalmanFilter.filter(rep, init, borrowedCompactPool, compactFilterSpec);
        KalmanFilter.filter(diffuseRep, diffuseInit, borrowedDiffuseFullPool, fullFilterSpec);
        reservedBorrowedFullPool.reserve(rep, fullFilterSpec);
        reservedBorrowedCompactPool.reserve(rep, compactFilterSpec);
        reservedBorrowedDiffuseFullPool.reserve(diffuseRep, fullFilterSpec);
        KalmanFilter.filter(rep, init, reservedBorrowedFullPool, fullFilterSpec);
        KalmanFilter.filter(rep, init, reservedBorrowedCompactPool, compactFilterSpec);
        KalmanFilter.filter(diffuseRep, diffuseInit, reservedBorrowedDiffuseFullPool, fullFilterSpec);
        smootherFilterInput = KalmanFilter.filter(rep, init, null, smootherFilterSpec);
        KalmanSmoother.smooth(rep, smootherFilterInput, borrowedFullSmootherPool, fullSmootherSpec);
        KalmanSmoother.smooth(rep, smootherFilterInput, borrowedStateOnlySmootherPool, stateOnlySmootherSpec);
        KalmanSmoother.smooth(rep, smootherFilterInput, borrowedClassicalFullSmootherPool, classicalFullSmootherSpec);
        KalmanSmoother.smooth(rep, smootherFilterInput, borrowedAlternativeFullSmootherPool, alternativeFullSmootherSpec);
        reservedBorrowedFullSmootherPool.reserve(rep, fullSmootherSpec);
        reservedBorrowedStateOnlySmootherPool.reserve(rep, stateOnlySmootherSpec);
        reservedBorrowedClassicalFullSmootherPool.reserve(rep, classicalFullSmootherSpec);
        reservedBorrowedAlternativeFullSmootherPool.reserve(rep, alternativeFullSmootherSpec);
        KalmanSmoother.smooth(rep, smootherFilterInput, reservedBorrowedFullSmootherPool, fullSmootherSpec);
        KalmanSmoother.smooth(rep, smootherFilterInput, reservedBorrowedStateOnlySmootherPool, stateOnlySmootherSpec);
        KalmanSmoother.smooth(rep, smootherFilterInput, reservedBorrowedClassicalFullSmootherPool, classicalFullSmootherSpec);
        KalmanSmoother.smooth(rep, smootherFilterInput, reservedBorrowedAlternativeFullSmootherPool, alternativeFullSmootherSpec);
        SimulationSmoother.simulate(rep, init,
            measurementVariatesFor(rep, 7L), stateVariatesFor(rep, 11L), initialVariatesFor(rep, 13L),
            borrowedSimulationPool, SimulationSmootherSpec.all());
        reservedBorrowedSimulationPool.reserve(rep, init, SimulationSmootherSpec.all());

        Random rng = new Random(7L);
        measurementVariates = new double[rep.nobs * rep.kEndog];
        stateVariates = new double[rep.nobs * rep.kPosdef];
        initialVariates = new double[rep.kStates];
        fillGaussian(measurementVariates, rng);
        fillGaussian(stateVariates, rng);
        fillGaussian(initialVariates, rng);

        SimulationSmoother.simulate(rep, init,
            measurementVariates, stateVariates, initialVariates,
            reservedBorrowedSimulationPool, SimulationSmootherSpec.all());
    }

    @Benchmark
    public FilterResult filterFull() {
        return KalmanFilter.filter(rep, init);
    }

    @Benchmark
    public FilterResult filterCompact() {
        return KalmanFilter.filter(rep, init, null, compactFilterSpec);
    }

    @Benchmark
    public FilterResult filterBorrowedResultFull() {
        return KalmanFilter.filter(rep, init, borrowedFullPool, fullFilterSpec);
    }

    @Benchmark
    public FilterResult filterReservedBorrowedResultFull() {
        return KalmanFilter.filter(rep, init, reservedBorrowedFullPool, fullFilterSpec);
    }

    @Benchmark
    public FilterResult filterBorrowedResultCompact() {
        return KalmanFilter.filter(rep, init, borrowedCompactPool, compactFilterSpec);
    }

    @Benchmark
    public FilterResult filterReservedBorrowedResultCompact() {
        return KalmanFilter.filter(rep, init, reservedBorrowedCompactPool, compactFilterSpec);
    }

    @Benchmark
    public FilterResult filterExactDiffuseFull() {
        return KalmanFilter.filter(diffuseRep, diffuseInit);
    }

    @Benchmark
    public FilterResult filterExactDiffuseBorrowedResultFull() {
        return KalmanFilter.filter(diffuseRep, diffuseInit, borrowedDiffuseFullPool, fullFilterSpec);
    }

    @Benchmark
    public FilterResult filterExactDiffuseReservedBorrowedResultFull() {
        return KalmanFilter.filter(diffuseRep, diffuseInit, reservedBorrowedDiffuseFullPool, fullFilterSpec);
    }

    @Benchmark
    public SmootherResult smoothFull() {
        FilterResult fr = KalmanFilter.filter(rep, init, null,
                FilterSpec.conventionalSmoothing());
        return KalmanSmoother.smooth(rep, fr);
    }

    @Benchmark
    public SmootherResult smoothStateOnly() {
        FilterResult fr = KalmanFilter.filter(rep, init, null,
                FilterSpec.conventionalSmoothing());
        return KalmanSmoother.smooth(rep, fr, null, stateOnlySmootherSpec);
    }

    @Benchmark
    public SmootherResult smoothBorrowedResultFull() {
        return KalmanSmoother.smooth(rep, smootherFilterInput, borrowedFullSmootherPool, fullSmootherSpec);
    }

    @Benchmark
    public SmootherResult smoothBorrowedResultStateOnly() {
        return KalmanSmoother.smooth(rep, smootherFilterInput, borrowedStateOnlySmootherPool, stateOnlySmootherSpec);
    }

    @Benchmark
    public SmootherResult smoothReservedBorrowedResultFull() {
        return KalmanSmoother.smooth(rep, smootherFilterInput, reservedBorrowedFullSmootherPool, fullSmootherSpec);
    }

    @Benchmark
    public SmootherResult smoothReservedBorrowedResultStateOnly() {
        return KalmanSmoother.smooth(rep, smootherFilterInput, reservedBorrowedStateOnlySmootherPool, stateOnlySmootherSpec);
    }

    @Benchmark
    public SmootherResult smoothClassicalBorrowedResultFull() {
        return KalmanSmoother.smooth(rep, smootherFilterInput, borrowedClassicalFullSmootherPool, classicalFullSmootherSpec);
    }

    @Benchmark
    public SmootherResult smoothClassicalReservedBorrowedResultFull() {
        return KalmanSmoother.smooth(rep, smootherFilterInput, reservedBorrowedClassicalFullSmootherPool, classicalFullSmootherSpec);
    }

    @Benchmark
    public SmootherResult smoothAlternativeBorrowedResultFull() {
        return KalmanSmoother.smooth(rep, smootherFilterInput, borrowedAlternativeFullSmootherPool, alternativeFullSmootherSpec);
    }

    @Benchmark
    public SmootherResult smoothAlternativeReservedBorrowedResultFull() {
        return KalmanSmoother.smooth(rep, smootherFilterInput, reservedBorrowedAlternativeFullSmootherPool, alternativeFullSmootherSpec);
    }

    @Benchmark
    public SimulationSmootherResult simulationFull() {
        return SimulationSmoother.simulate(rep, init,
                measurementVariates, stateVariates, initialVariates,
                SimulationSmootherSpec.all());
    }

    @Benchmark
    public SimulationSmootherResult simulationStateOnly() {
        return SimulationSmoother.simulate(rep, init,
                measurementVariates, stateVariates, initialVariates,
                SimulationSmootherSpec.stateOnly());
    }

    @Benchmark
    public SimulationSmootherResult simulationBorrowedResultFull() {
        return SimulationSmoother.simulate(rep, init,
                measurementVariates, stateVariates, initialVariates,
                borrowedSimulationPool, SimulationSmootherSpec.all());
    }

    @Benchmark
    public SimulationSmootherResult simulationReservedBorrowedResultFull() {
        return SimulationSmoother.simulate(rep, init,
                measurementVariates, stateVariates, initialVariates,
                reservedBorrowedSimulationPool, SimulationSmootherSpec.all());
    }

    @Benchmark
    public SimulationSmootherResult simulationRandomFull() {
        return SimulationSmoother.simulate(rep, init,
                simulationRandomFullRng, SimulationSmootherSpec.all());
    }

    @Benchmark
    public SimulationSmootherResult simulationRandomBorrowedResultFull() {
        return SimulationSmoother.simulate(rep, init,
                simulationRandomBorrowedRng, borrowedSimulationPool, SimulationSmootherSpec.all());
    }

    @Benchmark
    public SimulationSmootherResult simulationRandomReservedBorrowedResultFull() {
        return SimulationSmoother.simulate(rep, init,
                simulationRandomReservedBorrowedRng, reservedBorrowedSimulationPool, SimulationSmootherSpec.all());
    }

    public static void main(String[] args) throws RunnerException {
        Options options = new OptionsBuilder()
                .include(KalmanMemoryBenchmark.class.getSimpleName())
                .build();
        new Runner(options).run();
    }

    private static ModelFixture buildBenchmarkModel(int nobs) {
        int kEndog = 4;
        int kStates = 4;
        int kPosdef = 4;
        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] design = {
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
        };
        double[] transition = {
                0.85, 0.05, 0.00, 0.00,
                0.02, 0.80, 0.03, 0.00,
                0.00, 0.04, 0.78, 0.06,
                0.00, 0.00, 0.02, 0.75
        };
        double[] selection = {
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
        };
        double[] obsCov = {
                0.4, 0.0, 0.0, 0.0,
                0.0, 0.5, 0.0, 0.0,
                0.0, 0.0, 0.6, 0.0,
                0.0, 0.0, 0.0, 0.7
        };
        double[] stateCov = {
                0.2, 0.0, 0.0, 0.0,
                0.0, 0.2, 0.0, 0.0,
                0.0, 0.0, 0.2, 0.0,
                0.0, 0.0, 0.0, 0.2
        };

        Random rng = new Random(42L);
        double[] latent = new double[kStates];
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(design, t);
            rep.setObsIntercept(new double[kEndog], t);
            rep.setObsCov(obsCov, t);
            rep.setTransition(transition, t);
            rep.setStateIntercept(new double[kStates], t);
            rep.setSelection(selection, t);
            rep.setStateCov(stateCov, t);

            double[] obs = new double[kEndog];
            for (int i = 0; i < kStates; i++) {
                double next = 0.0;
                for (int j = 0; j < kStates; j++) {
                    next += transition[i * kStates + j] * latent[j];
                }
                next += rng.nextGaussian() * Math.sqrt(stateCov[i * kStates + i]);
                latent[i] = next;
                obs[i] = next + rng.nextGaussian() * Math.sqrt(obsCov[i * kEndog + i]);
            }
            rep.setEndog(obs, t);
        }
        return rep;
    }

    private static ModelFixture buildDiffuseBenchmarkModel(int nobs) {
        ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
        double[] design = {1.0};
        double[] transition = {1.0};
        double[] selection = {1.0};
        double[] obsCov = {1.993};
        double[] stateCov = {8.253};

        Random rng = new Random(99L);
        double latent = 0.0;
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(design, t);
            rep.setObsIntercept(new double[]{0.0}, t);
            rep.setObsCov(obsCov, t);
            rep.setTransition(transition, t);
            rep.setStateIntercept(new double[]{0.0}, t);
            rep.setSelection(selection, t);
            rep.setStateCov(stateCov, t);

            latent += rng.nextGaussian() * Math.sqrt(stateCov[0]);
            rep.setEndog(new double[]{latent + rng.nextGaussian() * Math.sqrt(obsCov[0])}, t);
        }
        return rep;
    }

    private static double[] measurementVariatesFor(ModelFixture rep, long seed) {
        double[] values = new double[rep.nobs * rep.kEndog];
        fillGaussian(values, new Random(seed));
        return values;
    }

    private static double[] stateVariatesFor(ModelFixture rep, long seed) {
        double[] values = new double[rep.nobs * rep.kPosdef];
        fillGaussian(values, new Random(seed));
        return values;
    }

    private static double[] initialVariatesFor(ModelFixture rep, long seed) {
        double[] values = new double[rep.kStates];
        fillGaussian(values, new Random(seed));
        return values;
    }

    private static void fillGaussian(double[] target, Random rng) {
        for (int i = 0; i < target.length; i++) {
            target[i] = rng.nextGaussian();
        }
    }
}