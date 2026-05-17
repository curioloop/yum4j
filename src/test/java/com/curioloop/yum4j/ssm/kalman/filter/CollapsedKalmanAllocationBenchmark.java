package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
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

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(0)
@State(Scope.Thread)
public class CollapsedKalmanAllocationBenchmark {
    @Param({"256", "1024"})
    int nobs;

    private ModelFixture model;
    private InitialState initialState;
    private FilterOptions collapsedFull;
    private FilterOptions collapsedLikelihoodOnly;
    private KalmanEngine.Workspace borrowedFullWorkspace;
    private KalmanEngine.Workspace borrowedLikelihoodWorkspace;

    @Setup
    public void setup() {
        model = buildCollapsedModel(nobs);
        initialState = InitialState.known(new double[]{0.2, -0.1}, new double[]{1.0, 0.1, 0.1, 1.4});
        collapsedFull = FilterOptions.builder()
            .method(FilterMethod.COLLAPSED)
            .retainAll()
            .build();
        collapsedLikelihoodOnly = FilterOptions.builder()
            .method(FilterMethod.COLLAPSED)
            .retainOnly(FilterOptions.Surface.LIKELIHOOD)
            .build();
        borrowedFullWorkspace = KalmanEngine.workspace();
        borrowedLikelihoodWorkspace = KalmanEngine.workspace();
        KalmanEngine.filterBorrowedUnsafe(model, initialState, borrowedFullWorkspace, collapsedFull);
        KalmanEngine.filterBorrowedUnsafe(model, initialState, borrowedLikelihoodWorkspace, collapsedLikelihoodOnly);
    }

    @Benchmark
    public FilterResult collapsedFull() {
        return KalmanEngine.filter(model, initialState, collapsedFull);
    }

    @Benchmark
    public FilterResult collapsedLikelihoodOnly() {
        return KalmanEngine.filter(model, initialState, collapsedLikelihoodOnly);
    }

    @Benchmark
    public FilterResult collapsedBorrowedFull() {
        return KalmanEngine.filterBorrowedUnsafe(model, initialState, borrowedFullWorkspace, collapsedFull);
    }

    @Benchmark
    public FilterResult collapsedBorrowedLikelihoodOnly() {
        return KalmanEngine.filterBorrowedUnsafe(model, initialState, borrowedLikelihoodWorkspace, collapsedLikelihoodOnly);
    }

    private static ModelFixture buildCollapsedModel(int nobs) {
        ModelFixture model = new ModelFixture(3, 2, 2, nobs);
        double[] design = {0.5, 0.2, 0.0, 0.8, 1.0, -0.5};
        double[] obsCov = {0.2, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.5};
        double[] transition = {0.4, 0.5, 1.0, 0.0};
        double[] selection = {1.0, 0.0, 0.0, 1.0};
        double[] stateCov = {2.0, 0.0, 0.0, 1.0};
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsIntercept(new double[3], t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setStateIntercept(new double[]{0.03, -0.01}, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{1.0 + 0.2 * t, -0.3 + 0.1 * t, 0.6 - 0.15 * t}, t);
        }
        return model;
    }
}
