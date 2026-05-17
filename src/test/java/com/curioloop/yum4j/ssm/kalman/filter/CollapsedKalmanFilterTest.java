package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CollapsedKalmanFilterTest {

    private static final double TOL = 1e-9;

    @Test
    void collapsedRetainedScratchUsesOverlayLayout() {
        ModelFixture model = buildCollapsedModel(32);
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();

        KalmanEngine.filterBorrowedUnsafe(model, knownInitialState(), workspace, collapsedOptions());

        assertEquals(262L, workspace.retainedCollapsedScratchDoubleCount());
        assertEquals(0L, workspace.retainedFilterScratchDoubleCount());
        assertEquals(0L, workspace.retainedFilterResultDoubleCount());
    }

    @Test
    void releaseRetainedScratchClearsCollapsedScratchButKeepsBorrowedResult() {
        ModelFixture model = buildCollapsedModel(8);
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();

        KalmanEngine.filterBorrowedUnsafe(model, knownInitialState(), workspace, collapsedOptions());
        long retainedResult = workspace.retainedCollapsedResultDoubleCount();
        assertTrue(workspace.retainedCollapsedScratchDoubleCount() > 0);
        assertTrue(retainedResult > 0);

        workspace.releaseCollapsedRetainedScratch();

        assertEquals(0L, workspace.retainedCollapsedScratchDoubleCount());
        assertEquals(retainedResult, workspace.retainedCollapsedResultDoubleCount());
    }

    @Test
    void borrowedCollapsedResultReusesBacking() {
        ModelFixture model = buildCollapsedModel(8);
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();

        FilterResult first = KalmanEngine.filterBorrowedUnsafe(model, knownInitialState(), workspace, collapsedOptions());
        double[] forecastBacking = first.forecast;
        double[] forecastErrorCovBacking = first.forecastErrorCov;
        double[] kalmanGainBacking = first.kalmanGain;
        long retainedResult = workspace.retainedCollapsedResultDoubleCount();

        FilterResult second = KalmanEngine.filterBorrowedUnsafe(model, knownInitialState(), workspace, collapsedOptions());

        assertSame(first, second);
        assertSame(forecastBacking, second.forecast);
        assertSame(forecastErrorCovBacking, second.forecastErrorCov);
        assertSame(kalmanGainBacking, second.kalmanGain);
        assertEquals(retainedResult, workspace.retainedCollapsedResultDoubleCount());
    }

    @Test
    void exactDiffuseReplayScratchIsReleasedWithCollapsedScratch() {
        ModelFixture model = buildCollapsedModel(8);
        KalmanEngine.Workspace workspace = KalmanEngine.workspace();

        KalmanEngine.filterBorrowedUnsafe(model, InitialState.diffuse(), workspace, collapsedOptions());
        assertTrue(workspace.retainedCollapsedScratchDoubleCount() > 0);
        assertTrue(workspace.retainedCollapsedResultDoubleCount() > 0);
        assertEquals(0L, workspace.retainedFilterScratchDoubleCount());
        assertEquals(0L, workspace.retainedFilterResultDoubleCount());

        workspace.releaseCollapsedRetainedScratch();
        assertEquals(0L, workspace.retainedCollapsedScratchDoubleCount());

        workspace.releaseCollapsedRetainedResults();
        assertEquals(0L, workspace.retainedCollapsedResultDoubleCount());
    }

    @Test
    void collapsedObservationSurfacesMatchConventionalFilter() {
        ModelFixture model = buildCollapsedModel(12);
        InitialState initialState = knownInitialState();
        FilterOptions conventionalOptions = FilterOptions.builder()
            .method(FilterMethod.CONVENTIONAL)
            .retainAll()
            .build();
        FilterOptions collapsedOptions = FilterOptions.builder()
            .method(FilterMethod.COLLAPSED)
            .retainAll()
            .build();

        FilterResult conventional = KalmanEngine.filter(model, initialState, conventionalOptions);
        FilterResult collapsed = KalmanEngine.filter(model, initialState, collapsedOptions);

        assertFilterOutputsMatch(conventional, collapsed);
    }

    @Test
    void collapsedLikelihoodOnlyDoesNotRetainStateOrObservationSurfaces() {
        ModelFixture model = buildCollapsedModel(16);
        InitialState initialState = knownInitialState();
        FilterOptions conventionalOptions = FilterOptions.builder()
            .method(FilterMethod.CONVENTIONAL)
            .retainOnly(FilterOptions.Surface.LIKELIHOOD)
            .build();
        FilterOptions collapsedOptions = FilterOptions.builder()
            .method(FilterMethod.COLLAPSED)
            .retainOnly(FilterOptions.Surface.LIKELIHOOD)
            .build();

        FilterResult conventional = KalmanEngine.filter(model, initialState, conventionalOptions);
        FilterResult collapsed = KalmanEngine.filter(model, initialState, collapsedOptions);

        assertArrayEquals(activeSlice(conventional.logLikelihoodObs, conventional.logLikelihoodObsBase(), conventional.logLikelihoodObsLength()),
            activeSlice(collapsed.logLikelihoodObs, collapsed.logLikelihoodObsBase(), collapsed.logLikelihoodObsLength()), TOL);
        assertEquals(0, collapsed.predictedStateLength());
        assertEquals(0, collapsed.predictedStateCovLength());
        assertEquals(0, collapsed.filteredStateLength());
        assertEquals(0, collapsed.filteredStateCovLength());
        assertEquals(0, collapsed.forecastLength());
        assertEquals(0, collapsed.forecastErrorLength());
        assertEquals(0, collapsed.forecastErrorCovLength());
        assertEquals(0, collapsed.kalmanGainLength());
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

    private static InitialState knownInitialState() {
        return InitialState.known(new double[]{0.2, -0.1}, new double[]{1.0, 0.1, 0.1, 1.4});
    }

    private static FilterOptions collapsedOptions() {
        return FilterOptions.builder()
            .method(FilterMethod.COLLAPSED)
            .build();
    }

    private static void assertFilterOutputsMatch(FilterResult expected, FilterResult actual) {
        assertArrayEquals(activeSlice(expected.predictedState, expected.predictedStateBase(), expected.predictedStateLength()),
            activeSlice(actual.predictedState, actual.predictedStateBase(), actual.predictedStateLength()), TOL);
        assertArrayEquals(activeSlice(expected.predictedStateCov, expected.predictedStateCovBase(), expected.predictedStateCovLength()),
            activeSlice(actual.predictedStateCov, actual.predictedStateCovBase(), actual.predictedStateCovLength()), TOL);
        assertArrayEquals(activeSlice(expected.filteredState, expected.filteredStateBase(), expected.filteredStateLength()),
            activeSlice(actual.filteredState, actual.filteredStateBase(), actual.filteredStateLength()), TOL);
        assertArrayEquals(activeSlice(expected.filteredStateCov, expected.filteredStateCovBase(), expected.filteredStateCovLength()),
            activeSlice(actual.filteredStateCov, actual.filteredStateCovBase(), actual.filteredStateCovLength()), TOL);
        assertArrayEquals(activeSlice(expected.forecast, expected.forecastBase(), expected.forecastLength()),
            activeSlice(actual.forecast, actual.forecastBase(), actual.forecastLength()), TOL);
        assertArrayEquals(activeSlice(expected.forecastError, expected.forecastErrorBase(), expected.forecastErrorLength()),
            activeSlice(actual.forecastError, actual.forecastErrorBase(), actual.forecastErrorLength()), TOL);
        assertArrayEquals(activeSlice(expected.forecastErrorCov, expected.forecastErrorCovBase(), expected.forecastErrorCovLength()),
            activeSlice(actual.forecastErrorCov, actual.forecastErrorCovBase(), actual.forecastErrorCovLength()), TOL);
        assertArrayEquals(activeSlice(expected.standardizedForecastError, expected.standardizedForecastErrorBase(), expected.standardizedForecastErrorLength()),
            activeSlice(actual.standardizedForecastError, actual.standardizedForecastErrorBase(), actual.standardizedForecastErrorLength()), TOL);
        assertArrayEquals(activeSlice(expected.kalmanGain, expected.kalmanGainBase(), expected.kalmanGainLength()),
            activeSlice(actual.kalmanGain, actual.kalmanGainBase(), actual.kalmanGainLength()), TOL);
        assertArrayEquals(activeSlice(expected.logLikelihoodObs, expected.logLikelihoodObsBase(), expected.logLikelihoodObsLength()),
            activeSlice(actual.logLikelihoodObs, actual.logLikelihoodObsBase(), actual.logLikelihoodObsLength()), TOL);
        assertEquals(expected.nobsDiffuse, actual.nobsDiffuse);
    }

    private static double[] activeSlice(double[] values, int base, int length) {
        return values == null ? null : Arrays.copyOfRange(values, base, base + length);
    }
}
