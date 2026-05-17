package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherEngine;

import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
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
    private InitialState init;
    private InitialState diffuseInit;
    private FilterOptions fullFilterOptions;
    private FilterOptions compactFilterOptions;
    private FilterOptions predictedStateOnlyFilterOptions;
    private FilterOptions predictedCovarianceOnlyFilterOptions;
    private FilterOptions filteredStateOnlyFilterOptions;
    private FilterOptions filteredCovarianceOnlyFilterOptions;
    private FilterOptions initFilteredCompactFilterOptions;
    private FilterOptions smootherFilterOptions;
    private SmootherOptions fullSmootherOptions;
    private SmootherOptions stateOnlySmootherOptions;
    private SmootherOptions classicalFullSmootherOptions;
    private SmootherOptions alternativeFullSmootherOptions;
    private FilterResult smootherFilterInput;
    private FilterResult classicalSmootherFilterInput;
    private double[] measurementVariates;
    private double[] stateVariates;
    private double[] initialVariates;
    private KalmanEngine.Workspace borrowedFullWorkspace;
    private KalmanEngine.Workspace reservedBorrowedFullWorkspace;
    private KalmanEngine.Workspace borrowedCompactWorkspace;
    private KalmanEngine.Workspace reservedBorrowedCompactWorkspace;
    private KalmanEngine.Workspace borrowedPredictedStateOnlyWorkspace;
    private KalmanEngine.Workspace borrowedPredictedCovarianceOnlyWorkspace;
    private KalmanEngine.Workspace borrowedFilteredStateOnlyWorkspace;
    private KalmanEngine.Workspace borrowedFilteredCovarianceOnlyWorkspace;
    private KalmanEngine.Workspace borrowedInitFilteredCompactWorkspace;
    private KalmanEngine.Workspace borrowedDiffuseFullWorkspace;
    private KalmanEngine.Workspace reservedBorrowedDiffuseFullWorkspace;
    private SmootherEngine.Workspace borrowedFullSmootherWorkspace;
    private SmootherEngine.Workspace borrowedStateOnlySmootherWorkspace;
    private SmootherEngine.Workspace borrowedClassicalFullSmootherWorkspace;
    private SmootherEngine.Workspace borrowedAlternativeFullSmootherWorkspace;
    private SmootherEngine.Workspace reservedBorrowedFullSmootherWorkspace;
    private SmootherEngine.Workspace reservedBorrowedStateOnlySmootherWorkspace;
    private SmootherEngine.Workspace reservedBorrowedClassicalFullSmootherWorkspace;
    private SmootherEngine.Workspace reservedBorrowedAlternativeFullSmootherWorkspace;
    private SimulationSmootherEngine.Workspace borrowedSimulationWorkspace;
    private SimulationSmootherEngine.Workspace reservedBorrowedSimulationWorkspace;
    private KalmanWorkspace facadeWorkspace;
    private Random simulationRandomFullRng;
    private Random simulationRandomBorrowedRng;
    private Random simulationRandomReservedBorrowedRng;

    @Setup(Level.Trial)
    public void setup() {
        rep = buildBenchmarkModel(nobs);
        diffuseRep = buildDiffuseBenchmarkModel(nobs);
        init = InitialState.approximateDiffuse();
        diffuseInit = InitialState.diffuse();
        fullFilterOptions = FilterOptions.defaults();
        compactFilterOptions = FilterOptions.compact();
        predictedStateOnlyFilterOptions = FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.PREDICTED_STATE)
            .build();
        predictedCovarianceOnlyFilterOptions = FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.PREDICTED_STATE_COVARIANCE)
            .build();
        filteredStateOnlyFilterOptions = FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.FILTERED_STATE)
            .build();
        filteredCovarianceOnlyFilterOptions = FilterOptions.builder()
            .retainOnly(FilterOptions.Surface.FILTERED_STATE_COVARIANCE)
            .build();
        initFilteredCompactFilterOptions = FilterOptions.compact().toBuilder()
            .timing(FilterOptions.Timing.INIT_FILTERED)
            .build();
        smootherFilterOptions = SmootherOptions.conventional().requiredFilterOptions();
        fullSmootherOptions = SmootherOptions.conventional();
        stateOnlySmootherOptions = SmootherOptions.conventional().only(SmootherOptions.Surface.STATE);
        classicalFullSmootherOptions = SmootherOptions.classical();
        alternativeFullSmootherOptions = SmootherOptions.alternative();

        borrowedFullWorkspace = KalmanEngine.workspace();
        reservedBorrowedFullWorkspace = KalmanEngine.workspace();
        borrowedCompactWorkspace = KalmanEngine.workspace();
        reservedBorrowedCompactWorkspace = KalmanEngine.workspace();
        borrowedPredictedStateOnlyWorkspace = KalmanEngine.workspace();
        borrowedPredictedCovarianceOnlyWorkspace = KalmanEngine.workspace();
        borrowedFilteredStateOnlyWorkspace = KalmanEngine.workspace();
        borrowedFilteredCovarianceOnlyWorkspace = KalmanEngine.workspace();
        borrowedInitFilteredCompactWorkspace = KalmanEngine.workspace();
        borrowedDiffuseFullWorkspace = KalmanEngine.workspace();
        reservedBorrowedDiffuseFullWorkspace = KalmanEngine.workspace();
        borrowedFullSmootherWorkspace = SmootherEngine.workspace();
        borrowedStateOnlySmootherWorkspace = SmootherEngine.workspace();
        borrowedClassicalFullSmootherWorkspace = SmootherEngine.workspace();
        borrowedAlternativeFullSmootherWorkspace = SmootherEngine.workspace();
        reservedBorrowedFullSmootherWorkspace = SmootherEngine.workspace();
        reservedBorrowedStateOnlySmootherWorkspace = SmootherEngine.workspace();
        reservedBorrowedClassicalFullSmootherWorkspace = SmootherEngine.workspace();
        reservedBorrowedAlternativeFullSmootherWorkspace = SmootherEngine.workspace();
        borrowedSimulationWorkspace = SimulationSmootherEngine.workspace();
        reservedBorrowedSimulationWorkspace = SimulationSmootherEngine.workspace();
        facadeWorkspace = Kalman.workspace();
        simulationRandomFullRng = new Random(17L);
        simulationRandomBorrowedRng = new Random(17L);
        simulationRandomReservedBorrowedRng = new Random(17L);

        KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedFullWorkspace, fullFilterOptions);
        KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedCompactWorkspace, compactFilterOptions);
        KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedPredictedStateOnlyWorkspace, predictedStateOnlyFilterOptions);
        KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedPredictedCovarianceOnlyWorkspace, predictedCovarianceOnlyFilterOptions);
        KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedFilteredStateOnlyWorkspace, filteredStateOnlyFilterOptions);
        KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedFilteredCovarianceOnlyWorkspace, filteredCovarianceOnlyFilterOptions);
        KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedInitFilteredCompactWorkspace, initFilteredCompactFilterOptions);
        KalmanEngine.filterBorrowedUnsafe(diffuseRep, diffuseInit, borrowedDiffuseFullWorkspace, fullFilterOptions);
        reservedBorrowedFullWorkspace.reserveFilter(rep, fullFilterOptions);
        reservedBorrowedCompactWorkspace.reserveFilter(rep, compactFilterOptions);
        reservedBorrowedDiffuseFullWorkspace.reserveFilter(diffuseRep, fullFilterOptions);
        KalmanEngine.filterBorrowedUnsafe(rep, init, reservedBorrowedFullWorkspace, fullFilterOptions);
        KalmanEngine.filterBorrowedUnsafe(rep, init, reservedBorrowedCompactWorkspace, compactFilterOptions);
        KalmanEngine.filterBorrowedUnsafe(diffuseRep, diffuseInit, reservedBorrowedDiffuseFullWorkspace, fullFilterOptions);
        smootherFilterInput = KalmanEngine.filter(rep, init, smootherFilterOptions);
        classicalSmootherFilterInput = KalmanEngine.filter(rep, init, SmootherOptions.classical().requiredFilterOptions());
        SmootherEngine.smoothBorrowedUnsafe(rep, smootherFilterInput, borrowedFullSmootherWorkspace, fullSmootherOptions);
        SmootherEngine.smoothBorrowedUnsafe(rep, smootherFilterInput, borrowedStateOnlySmootherWorkspace, stateOnlySmootherOptions);
        SmootherEngine.smoothBorrowedUnsafe(rep, classicalSmootherFilterInput, borrowedClassicalFullSmootherWorkspace, classicalFullSmootherOptions);
        SmootherEngine.smoothBorrowedUnsafe(rep, classicalSmootherFilterInput, borrowedAlternativeFullSmootherWorkspace, alternativeFullSmootherOptions);
        reservedBorrowedFullSmootherWorkspace.reserveSmoother(rep, fullSmootherOptions);
        reservedBorrowedStateOnlySmootherWorkspace.reserveSmoother(rep, stateOnlySmootherOptions);
        reservedBorrowedClassicalFullSmootherWorkspace.reserveSmoother(rep, classicalFullSmootherOptions);
        reservedBorrowedAlternativeFullSmootherWorkspace.reserveSmoother(rep, alternativeFullSmootherOptions);
        SmootherEngine.smoothBorrowedUnsafe(rep, smootherFilterInput, reservedBorrowedFullSmootherWorkspace, fullSmootherOptions);
        SmootherEngine.smoothBorrowedUnsafe(rep, smootherFilterInput, reservedBorrowedStateOnlySmootherWorkspace, stateOnlySmootherOptions);
        SmootherEngine.smoothBorrowedUnsafe(rep, classicalSmootherFilterInput, reservedBorrowedClassicalFullSmootherWorkspace, classicalFullSmootherOptions);
        SmootherEngine.smoothBorrowedUnsafe(rep, classicalSmootherFilterInput, reservedBorrowedAlternativeFullSmootherWorkspace, alternativeFullSmootherOptions);
        SimulationSmootherEngine.simulate(rep, init,
            measurementVariatesFor(rep, 7L), stateVariatesFor(rep, 11L), initialVariatesFor(rep, 13L),
            borrowedSimulationWorkspace, SimulationSmootherOptions.defaults());
        reservedBorrowedSimulationWorkspace.reserve(rep, init, SimulationSmootherOptions.defaults());

        Random rng = new Random(7L);
        measurementVariates = new double[rep.nobs * rep.kEndog];
        stateVariates = new double[rep.nobs * rep.kPosdef];
        initialVariates = new double[rep.kStates];
        fillGaussian(measurementVariates, rng);
        fillGaussian(stateVariates, rng);
        fillGaussian(initialVariates, rng);

        SimulationSmootherEngine.simulate(rep, init,
            measurementVariates, stateVariates, initialVariates,
            reservedBorrowedSimulationWorkspace, SimulationSmootherOptions.defaults());
    }

    @Benchmark
    public FilterResult filterFull() {
        return KalmanEngine.filter(rep, init, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
    }

    @Benchmark
    public FilterResult filterCompact() {
        return KalmanEngine.filter(rep, init, compactFilterOptions);
    }

    @Benchmark
    public FilterResult filterBorrowedResultFull() {
        return KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedFullWorkspace, fullFilterOptions);
    }

    @Benchmark
    public FilterResult filterReservedBorrowedResultFull() {
        return KalmanEngine.filterBorrowedUnsafe(rep, init, reservedBorrowedFullWorkspace, fullFilterOptions);
    }

    @Benchmark
    public FilterResult filterBorrowedResultCompact() {
        return KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedCompactWorkspace, compactFilterOptions);
    }

    @Benchmark
    public FilterResult filterReservedBorrowedResultCompact() {
        return KalmanEngine.filterBorrowedUnsafe(rep, init, reservedBorrowedCompactWorkspace, compactFilterOptions);
    }

    @Benchmark
    public FilterResult filterBorrowedResultPredictedStateOnly() {
        return KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedPredictedStateOnlyWorkspace, predictedStateOnlyFilterOptions);
    }

    @Benchmark
    public FilterResult filterBorrowedResultPredictedCovarianceOnly() {
        return KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedPredictedCovarianceOnlyWorkspace, predictedCovarianceOnlyFilterOptions);
    }

    @Benchmark
    public FilterResult filterBorrowedResultFilteredStateOnly() {
        return KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedFilteredStateOnlyWorkspace, filteredStateOnlyFilterOptions);
    }

    @Benchmark
    public FilterResult filterBorrowedResultFilteredCovarianceOnly() {
        return KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedFilteredCovarianceOnlyWorkspace, filteredCovarianceOnlyFilterOptions);
    }

    @Benchmark
    public FilterResult filterBorrowedInitFilteredCompact() {
        return KalmanEngine.filterBorrowedUnsafe(rep, init, borrowedInitFilteredCompactWorkspace, initFilteredCompactFilterOptions);
    }

    @Benchmark
    public FilterResult facadeFilterSafeWorkspaceLikelihoodOnly() {
        return Kalman.filter()
            .model(rep)
            .initialState(init)
            .likelihoodOnly()
            .run(facadeWorkspace);
    }

    @Benchmark
    public FilterResult filterExactDiffuseFull() {
        return KalmanEngine.filter(diffuseRep, diffuseInit, com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
    }

    @Benchmark
    public FilterResult filterExactDiffuseBorrowedResultFull() {
        return KalmanEngine.filterBorrowedUnsafe(diffuseRep, diffuseInit, borrowedDiffuseFullWorkspace, fullFilterOptions);
    }

    @Benchmark
    public FilterResult filterExactDiffuseReservedBorrowedResultFull() {
        return KalmanEngine.filterBorrowedUnsafe(diffuseRep, diffuseInit, reservedBorrowedDiffuseFullWorkspace, fullFilterOptions);
    }

    @Benchmark
    public SmootherResult smoothFull() {
        FilterResult fr = KalmanEngine.filter(rep, init,
                SmootherOptions.conventional().requiredFilterOptions());
        return SmootherEngine.smooth(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
    }

    @Benchmark
    public SmootherResult smoothStateOnly() {
        FilterResult fr = KalmanEngine.filter(rep, init,
                SmootherOptions.conventional().requiredFilterOptions());
        return SmootherEngine.smooth(rep, fr, stateOnlySmootherOptions);
    }

    @Benchmark
    public SmootherResult smoothBorrowedResultFull() {
        return SmootherEngine.smoothBorrowedUnsafe(rep, smootherFilterInput, borrowedFullSmootherWorkspace, fullSmootherOptions);
    }

    @Benchmark
    public SmootherResult smoothBorrowedResultStateOnly() {
        return SmootherEngine.smoothBorrowedUnsafe(rep, smootherFilterInput, borrowedStateOnlySmootherWorkspace, stateOnlySmootherOptions);
    }

    @Benchmark
    public SmootherResult smoothReservedBorrowedResultFull() {
        return SmootherEngine.smoothBorrowedUnsafe(rep, smootherFilterInput, reservedBorrowedFullSmootherWorkspace, fullSmootherOptions);
    }

    @Benchmark
    public SmootherResult smoothReservedBorrowedResultStateOnly() {
        return SmootherEngine.smoothBorrowedUnsafe(rep, smootherFilterInput, reservedBorrowedStateOnlySmootherWorkspace, stateOnlySmootherOptions);
    }

    @Benchmark
    public SmootherResult smoothClassicalBorrowedResultFull() {
        return SmootherEngine.smoothBorrowedUnsafe(rep, classicalSmootherFilterInput, borrowedClassicalFullSmootherWorkspace, classicalFullSmootherOptions);
    }

    @Benchmark
    public SmootherResult smoothClassicalReservedBorrowedResultFull() {
        return SmootherEngine.smoothBorrowedUnsafe(rep, classicalSmootherFilterInput, reservedBorrowedClassicalFullSmootherWorkspace, classicalFullSmootherOptions);
    }

    @Benchmark
    public SmootherResult smoothAlternativeBorrowedResultFull() {
        return SmootherEngine.smoothBorrowedUnsafe(rep, classicalSmootherFilterInput, borrowedAlternativeFullSmootherWorkspace, alternativeFullSmootherOptions);
    }

    @Benchmark
    public SmootherResult smoothAlternativeReservedBorrowedResultFull() {
        return SmootherEngine.smoothBorrowedUnsafe(rep, classicalSmootherFilterInput, reservedBorrowedAlternativeFullSmootherWorkspace, alternativeFullSmootherOptions);
    }

    @Benchmark
    public SimulationSmootherResult simulationFull() {
        return SimulationSmootherEngine.simulate(rep, init,
                measurementVariates, stateVariates, initialVariates,
                SimulationSmootherOptions.defaults());
    }

    @Benchmark
    public SimulationSmootherResult simulationStateOnly() {
        return SimulationSmootherEngine.simulate(rep, init,
                measurementVariates, stateVariates, initialVariates,
                SimulationSmootherOptions.stateOnly());
    }

    @Benchmark
    public SimulationSmootherResult simulationBorrowedResultFull() {
        return SimulationSmootherEngine.simulate(rep, init,
                measurementVariates, stateVariates, initialVariates,
            borrowedSimulationWorkspace, SimulationSmootherOptions.defaults());
    }

    @Benchmark
    public SimulationSmootherResult simulationReservedBorrowedResultFull() {
        return SimulationSmootherEngine.simulate(rep, init,
                measurementVariates, stateVariates, initialVariates,
            reservedBorrowedSimulationWorkspace, SimulationSmootherOptions.defaults());
    }

    @Benchmark
    public SimulationSmootherResult simulationRandomFull() {
        return SimulationSmootherEngine.simulate(rep, init,
                simulationRandomFullRng, SimulationSmootherOptions.defaults());
    }

    @Benchmark
    public SimulationSmootherResult simulationRandomBorrowedResultFull() {
        return SimulationSmootherEngine.simulateBorrowedUnsafe(rep, init,
            simulationRandomBorrowedRng, borrowedSimulationWorkspace, SimulationSmootherOptions.defaults());
    }

    @Benchmark
    public SimulationSmootherResult simulationRandomReservedBorrowedResultFull() {
        return SimulationSmootherEngine.simulateBorrowedUnsafe(rep, init,
            simulationRandomReservedBorrowedRng, reservedBorrowedSimulationWorkspace, SimulationSmootherOptions.defaults());
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