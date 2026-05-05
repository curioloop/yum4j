package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.kalman.filter.FilterResult;
import com.curioloop.yum4j.kalman.filter.KalmanFilter;
import com.curioloop.yum4j.kalman.filter.UnivariateFilter;
import com.curioloop.yum4j.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.kalman.filter.ZKalmanFilter;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.model.ModelFixture;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.kalman.smooth.KalmanSmoother;
import com.curioloop.yum4j.kalman.smooth.SimulationSmoother;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherSpec;
import com.curioloop.yum4j.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.kalman.smooth.ZKalmanSmoother;
import com.curioloop.yum4j.kalman.smooth.ZSimulationSmoother;
import com.curioloop.yum4j.kalman.smooth.ZSimulationSmootherResult;
import com.curioloop.yum4j.kalman.smooth.ZSmootherResult;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

class RepresentationViewTest {

    @Test
    void testRealCompositeModelEntryPointsMatchRepresentationOverloads() {
        ModelFixture stateSpace = realSampleRepresentation();
        ModelFixture observationView = ModelFixture.copyOf(stateSpace);
        StateSpaceModel model = combine(
                new DelegatingStateSpace(stateSpace),
                new DelegatingObservationModel(observationView));
        StateInitialization init = StateInitialization.approximateDiffuse();

        FilterResult expectedKalman = KalmanFilter.filter(observationView, init);
        FilterResult actualKalman = KalmanFilter.filter(model, init);
        assertFilterResultEquals(expectedKalman, actualKalman, 1e-12);

        FilterResult expectedUnivariate = UnivariateFilter.filter(observationView, init);
        FilterResult actualUnivariate = UnivariateFilter.filter(model, init);
        assertFilterResultEquals(expectedUnivariate, actualUnivariate, 1e-12);

        SmootherResult expectedSmooth = KalmanSmoother.smooth(observationView, expectedKalman);
        SmootherResult actualSmooth = KalmanSmoother.smooth(model, actualKalman);
        assertSmootherResultEquals(expectedSmooth, actualSmooth, 1e-12);

        double[] measurementVariates = {0.2, -0.4, 0.1, 0.3};
        double[] stateVariates = {0.3, -0.1, 0.2, 0.05};
        double[] initialVariates = {0.15};
        SimulationSmootherSpec spec = SimulationSmootherSpec.all().withGeneratedOutputs();

        SimulationSmootherResult expectedSimulation = SimulationSmoother.simulate(observationView, init, measurementVariates, stateVariates, initialVariates, spec);
        SimulationSmootherResult actualSimulation = SimulationSmoother.simulate(
            model,
            init,
            measurementVariates,
            stateVariates,
            initialVariates,
            spec);
        assertSimulationResultEquals(expectedSimulation, actualSimulation, 1e-12);
    }

    @Test
        void testComplexCompositeModelEntryPointsMatchRepresentationOverloads() {
        ModelFixture stateSpace = complexSampleRepresentation();
        ModelFixture observationView = ModelFixture.copyOf(stateSpace);
        StateSpaceModel model = combine(
            new DelegatingComplexStateSpace(stateSpace),
            new DelegatingComplexObservationModel(observationView));
        StateInitialization init = StateInitialization.approximateDiffuse();

        ZFilterResult expectedFilter = ZKalmanFilter.filter(observationView, init);
        ZFilterResult actualFilter = ZKalmanFilter.filter(model, init);
        assertFilterResultEquals(expectedFilter, actualFilter, 1e-12);

        ZSmootherResult expectedSmooth = ZKalmanSmoother.smooth(observationView, expectedFilter);
        ZSmootherResult actualSmooth = ZKalmanSmoother.smooth(model, actualFilter);
        assertSmootherResultEquals(expectedSmooth, actualSmooth, 1e-12);

        double[] measurementVariates = {0.2, -0.1, -0.3, 0.4, 0.1, 0.2, -0.2, 0.1};
        double[] stateVariates = {0.1, -0.2, 0.2, 0.05, -0.1, 0.15, 0.12, -0.08};
        double[] initialVariates = {0.25, -0.05};
        SimulationSmootherSpec spec = SimulationSmootherSpec.all().withGeneratedOutputs();

        ZSimulationSmootherResult expectedSimulation = ZSimulationSmoother.simulate(observationView, init, measurementVariates, stateVariates, initialVariates, spec);
        ZSimulationSmootherResult actualSimulation = ZSimulationSmoother.simulate(
            model,
            init,
            measurementVariates,
            stateVariates,
            initialVariates,
            spec);
        assertSimulationResultEquals(expectedSimulation, actualSimulation, 1e-12);
    }

    @Test
    void testRealStationaryInitSupportsPureStateWrapper() {
        ModelFixture rep = realSampleRepresentation();
        StateSpaceModel stateSpace = new DelegatingStateSpace(rep);

        StateInitialization expected = StateInitialization.stationary(rep, 2);
        StateInitialization actual = StateInitialization.stationary(stateSpace, 2);

        assertArrayEquals(expected.initialState(), actual.initialState(), 1e-12);
        assertArrayEquals(expected.initialStateCov(), actual.initialStateCov(), 1e-12);
    }

    @Test
    void testRealFilterSupportsPureWrapperViews() {
        ModelFixture stateSpaceRep = realSampleRepresentation();
        ModelFixture observationRep = ModelFixture.copyOf(stateSpaceRep);
        StateSpaceModel model = combine(
                new DelegatingStateSpace(stateSpaceRep),
                new DelegatingObservationModel(observationRep));
        StateInitialization init = StateInitialization.approximateDiffuse();

        FilterResult expected = KalmanFilter.filter(observationRep, init);
        FilterResult actual = KalmanFilter.filter(model, init);

        assertFilterResultEquals(expected, actual, 1e-12);
    }

    @Test
    void testRealDiffuseFilterSupportsPureWrapperViews() {
        ModelFixture stateSpaceRep = realSampleRepresentation();
        ModelFixture observationRep = ModelFixture.copyOf(stateSpaceRep);
        StateSpaceModel model = combine(
                new DelegatingStateSpace(stateSpaceRep),
                new DelegatingObservationModel(observationRep));
        StateInitialization init = StateInitialization.diffuse();

        FilterResult expected = KalmanFilter.filter(observationRep, init);
        FilterResult actual = KalmanFilter.filter(model, init);

        assertFilterResultEquals(expected, actual, 1e-12);
    }

    @Test
    void testRealUnivariateFilterSupportsPureWrapperViews() {
        ModelFixture stateSpaceRep = realSampleRepresentation();
        ModelFixture observationRep = ModelFixture.copyOf(stateSpaceRep);
        StateSpaceModel model = combine(
                new DelegatingStateSpace(stateSpaceRep),
                new DelegatingObservationModel(observationRep));
        StateInitialization init = StateInitialization.approximateDiffuse();

        FilterResult expected = UnivariateFilter.filter(observationRep, init);
        FilterResult actual = UnivariateFilter.filter(model, init);

        assertFilterResultEquals(expected, actual, 1e-12);
    }

    @Test
    void testRealSmootherSupportsPureWrapperViews() {
        ModelFixture stateSpaceRep = realSampleRepresentation();
        ModelFixture observationRep = ModelFixture.copyOf(stateSpaceRep);
        StateSpaceModel model = combine(
                new DelegatingStateSpace(stateSpaceRep),
                new DelegatingObservationModel(observationRep));
        StateInitialization init = StateInitialization.approximateDiffuse();

        FilterResult expectedFilter = KalmanFilter.filter(observationRep, init);
        FilterResult actualFilter = KalmanFilter.filter(model, init);
        SmootherResult expected = KalmanSmoother.smooth(observationRep, expectedFilter);
        SmootherResult actual = KalmanSmoother.smooth(model, actualFilter);

        assertSmootherResultEquals(expected, actual, 1e-12);
    }

    @Test
    void testRealDiffuseSmootherSupportsPureWrapperViews() {
        ModelFixture stateSpaceRep = realSampleRepresentation();
        ModelFixture observationRep = ModelFixture.copyOf(stateSpaceRep);
        StateSpaceModel model = combine(
                new DelegatingStateSpace(stateSpaceRep),
                new DelegatingObservationModel(observationRep));
        StateInitialization init = StateInitialization.diffuse();

        FilterResult expectedFilter = KalmanFilter.filter(observationRep, init);
        FilterResult actualFilter = KalmanFilter.filter(model, init);
        SmootherResult expected = KalmanSmoother.smooth(observationRep, expectedFilter);
        SmootherResult actual = KalmanSmoother.smooth(model, actualFilter);

        assertSmootherResultEquals(expected, actual, 1e-12);
    }

    @Test
    void testRealUnivariateSmootherSupportsPureWrapperViews() {
        ModelFixture stateSpaceRep = realSampleRepresentation();
        ModelFixture observationRep = ModelFixture.copyOf(stateSpaceRep);
        StateSpaceModel model = combine(
                new DelegatingStateSpace(stateSpaceRep),
                new DelegatingObservationModel(observationRep));
        StateInitialization init = StateInitialization.approximateDiffuse();

        FilterResult expectedFilter = UnivariateFilter.filter(observationRep, init);
        FilterResult actualFilter = UnivariateFilter.filter(model, init);
        SmootherResult expected = KalmanSmoother.smooth(observationRep, expectedFilter);
        SmootherResult actual = KalmanSmoother.smooth(model, actualFilter);

        assertSmootherResultEquals(expected, actual, 1e-12);
    }

    @Test
    void testComplexStationaryInitSupportsPureStateWrapper() {
        ModelFixture rep = complexSampleRepresentation();
        StateSpaceModel stateSpace = new DelegatingComplexStateSpace(rep);

        StateInitialization expected = StateInitialization.stationary(rep, 1);
        StateInitialization actual = StateInitialization.stationary(stateSpace, 1);

        assertArrayEquals(expected.initialState(), actual.initialState(), 1e-12);
        assertArrayEquals(expected.initialStateCov(), actual.initialStateCov(), 1e-12);
    }

    @Test
    void testComplexFilterSupportsPureWrapperViews() {
        ModelFixture stateSpaceRep = complexSampleRepresentation();
        ModelFixture observationRep = ModelFixture.copyOf(stateSpaceRep);
        StateSpaceModel model = combine(
                new DelegatingComplexStateSpace(stateSpaceRep),
                new DelegatingComplexObservationModel(observationRep));
        StateInitialization init = StateInitialization.approximateDiffuse();

        ZFilterResult expected = ZKalmanFilter.filter(observationRep, init);
        ZFilterResult actual = ZKalmanFilter.filter(model, init);

        assertFilterResultEquals(expected, actual, 1e-12);
    }

    @Test
    void testComplexSmootherSupportsPureWrapperViews() {
        ModelFixture stateSpaceRep = complexSampleRepresentation();
        ModelFixture observationRep = ModelFixture.copyOf(stateSpaceRep);
        StateSpaceModel model = combine(
                new DelegatingComplexStateSpace(stateSpaceRep),
                new DelegatingComplexObservationModel(observationRep));
        StateInitialization init = StateInitialization.approximateDiffuse();

        ZFilterResult expectedFilter = ZKalmanFilter.filter(observationRep, init);
        ZFilterResult actualFilter = ZKalmanFilter.filter(model, init);
        ZSmootherResult expected = ZKalmanSmoother.smooth(observationRep, expectedFilter);
        ZSmootherResult actual = ZKalmanSmoother.smooth(model, actualFilter);

        assertSmootherResultEquals(expected, actual, 1e-12);
    }

    @Test
    void testRealSimulationPoolSupportsPureWrapperViews() {
        ModelFixture stateSpaceRep = realSampleRepresentation();
        ModelFixture observationRep = ModelFixture.copyOf(stateSpaceRep);
        StateSpaceModel model = combine(
                new DelegatingStateSpace(stateSpaceRep),
                new DelegatingObservationModel(observationRep));
        StateInitialization init = StateInitialization.approximateDiffuse();
        SimulationSmootherSpec spec = SimulationSmootherSpec.all().withGeneratedOutputs();
        SimulationSmoother.Pool pool = new SimulationSmoother.Pool();
        double[] measurementVariates = {0.2, -0.4, 0.1, 0.3};
        double[] stateVariates = {0.3, -0.1, 0.2, 0.05};
        double[] initialVariates = {0.15};

        SimulationSmootherResult expected = SimulationSmoother.simulate(observationRep, init, measurementVariates, stateVariates, initialVariates, spec);
        SimulationSmootherResult first = SimulationSmoother.simulate(
                model,
                init,
                measurementVariates,
                stateVariates,
                initialVariates,
                pool,
                spec);

        assertSimulationResultEquals(expected, first, 1e-12);
        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());

        SimulationSmootherResult second = SimulationSmoother.simulate(
                model,
                init,
                measurementVariates,
                stateVariates,
                initialVariates,
                pool,
                spec);

        assertSame(first, second);
        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());
        assertSimulationResultEquals(expected, second, 1e-12);
    }

    @Test
    void testComplexSimulationPoolSupportsPureWrapperViews() {
        ModelFixture stateSpaceRep = complexSampleRepresentation();
        ModelFixture observationRep = ModelFixture.copyOf(stateSpaceRep);
        StateSpaceModel model = combine(
                new DelegatingComplexStateSpace(stateSpaceRep),
                new DelegatingComplexObservationModel(observationRep));
        StateInitialization init = StateInitialization.approximateDiffuse();
        SimulationSmootherSpec spec = SimulationSmootherSpec.all().withGeneratedOutputs();
        ZSimulationSmoother.Pool pool = new ZSimulationSmoother.Pool();
        double[] measurementVariates = {0.2, -0.1, -0.3, 0.4, 0.1, 0.2, -0.2, 0.1};
        double[] stateVariates = {0.1, -0.2, 0.2, 0.05, -0.1, 0.15, 0.12, -0.08};
        double[] initialVariates = {0.25, -0.05};

        ZSimulationSmootherResult expected = ZSimulationSmoother.simulate(observationRep, init, measurementVariates, stateVariates, initialVariates, spec);
        ZSimulationSmootherResult first = ZSimulationSmoother.simulate(
                model,
                init,
                measurementVariates,
                stateVariates,
                initialVariates,
                pool,
                spec);

        assertSimulationResultEquals(expected, first, 1e-12);
        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());

        ZSimulationSmootherResult second = ZSimulationSmoother.simulate(
                model,
                init,
                measurementVariates,
                stateVariates,
                initialVariates,
                pool,
                spec);

        assertSame(first, second);
        assertEquals(0L, pool.retainedNestedResultDoubleCount());
        assertEquals(0L, pool.retainedWorkspaceDoubleCount());
        assertEquals(0L, pool.retainedVariateDoubleCount());
        assertSimulationResultEquals(expected, second, 1e-12);
    }

    private static ModelFixture realSampleRepresentation() {
        ModelFixture rep = new ModelFixture(1, 1, 1, 4);
        rep.setTransition(new double[]{0.85}, 0);
        rep.setTransition(new double[]{0.80}, 1);
        rep.setTransition(new double[]{0.75}, 2);
        rep.setTransition(new double[]{0.70}, 3);
        rep.setStateIntercept(new double[]{0.10}, 0);
        rep.setStateIntercept(new double[]{0.05}, 1);
        rep.setStateIntercept(new double[]{-0.02}, 2);
        rep.setStateIntercept(new double[]{0.03}, 3);
        rep.setSelection(new double[]{1.0}, 0);
        rep.setSelection(new double[]{1.0}, 1);
        rep.setSelection(new double[]{1.0}, 2);
        rep.setSelection(new double[]{1.0}, 3);
        rep.setStateCov(new double[]{0.30}, 0);
        rep.setStateCov(new double[]{0.25}, 1);
        rep.setStateCov(new double[]{0.20}, 2);
        rep.setStateCov(new double[]{0.22}, 3);
        rep.setDesign(new double[]{1.0}, 0);
        rep.setDesign(new double[]{1.0}, 1);
        rep.setDesign(new double[]{1.0}, 2);
        rep.setDesign(new double[]{1.0}, 3);
        rep.setObsIntercept(new double[]{0.0}, 0);
        rep.setObsIntercept(new double[]{0.0}, 1);
        rep.setObsIntercept(new double[]{0.0}, 2);
        rep.setObsIntercept(new double[]{0.0}, 3);
        rep.setObsCov(new double[]{0.6}, 0);
        rep.setObsCov(new double[]{0.5}, 1);
        rep.setObsCov(new double[]{0.4}, 2);
        rep.setObsCov(new double[]{0.7}, 3);
        rep.setEndog(new double[]{1.2}, 0);
        rep.setEndog(new double[]{0.7}, 1);
        rep.setEndog(new double[]{1.0}, 2);
        rep.setEndog(new double[]{0.2}, 3);
        return rep;
    }

    private static ModelFixture complexSampleRepresentation() {
        ModelFixture rep = ModelFixture.complex(1, 1, 1, 4);
        rep.setTransition(new double[]{0.80, 0.05}, 0);
        rep.setTransition(new double[]{0.75, -0.03}, 1);
        rep.setTransition(new double[]{0.70, 0.02}, 2);
        rep.setTransition(new double[]{0.72, -0.04}, 3);
        rep.setStateIntercept(new double[]{0.05, -0.01}, 0);
        rep.setStateIntercept(new double[]{0.02, 0.03}, 1);
        rep.setStateIntercept(new double[]{-0.04, 0.01}, 2);
        rep.setStateIntercept(new double[]{0.01, -0.02}, 3);
        rep.setSelection(new double[]{1.0, 0.0}, 0);
        rep.setSelection(new double[]{1.0, 0.0}, 1);
        rep.setSelection(new double[]{1.0, 0.0}, 2);
        rep.setSelection(new double[]{1.0, 0.0}, 3);
        rep.setStateCov(new double[]{0.35, 0.0}, 0);
        rep.setStateCov(new double[]{0.30, 0.0}, 1);
        rep.setStateCov(new double[]{0.25, 0.0}, 2);
        rep.setStateCov(new double[]{0.28, 0.0}, 3);
        rep.setDesign(new double[]{1.0, 0.0}, 0);
        rep.setDesign(new double[]{1.0, 0.0}, 1);
        rep.setDesign(new double[]{1.0, 0.0}, 2);
        rep.setDesign(new double[]{1.0, 0.0}, 3);
        rep.setObsIntercept(new double[]{0.0, 0.0}, 0);
        rep.setObsIntercept(new double[]{0.0, 0.0}, 1);
        rep.setObsIntercept(new double[]{0.0, 0.0}, 2);
        rep.setObsIntercept(new double[]{0.0, 0.0}, 3);
        rep.setObsCov(new double[]{0.6, 0.0}, 0);
        rep.setObsCov(new double[]{0.5, 0.0}, 1);
        rep.setObsCov(new double[]{0.4, 0.0}, 2);
        rep.setObsCov(new double[]{0.7, 0.0}, 3);
        rep.setEndog(new double[]{1.0, -0.2}, 0);
        rep.setEndog(new double[]{0.8, 0.1}, 1);
        rep.setEndog(new double[]{1.1, -0.1}, 2);
        rep.setEndog(new double[]{0.4, 0.3}, 3);
        return rep;
    }

    private static StateSpaceModel combine(StateSpaceModel stateSpace, StateSpaceModel observationModel) {
        return new CombinedModelView(stateSpace, observationModel);
    }

    private static void assertFilterResultEquals(FilterResult expected, FilterResult actual, double tolerance) {
        assertEquals(expected.kEndog, actual.kEndog);
        assertEquals(expected.kStates, actual.kStates);
        assertEquals(expected.nobs, actual.nobs);
        assertArrayEquals(expected.predictedState, actual.predictedState, tolerance);
        assertArrayEquals(expected.predictedStateCov, actual.predictedStateCov, tolerance);
        assertArrayEquals(expected.filteredState, actual.filteredState, tolerance);
        assertArrayEquals(expected.filteredStateCov, actual.filteredStateCov, tolerance);
        assertArrayEquals(expected.forecast, actual.forecast, tolerance);
        assertArrayEquals(expected.forecastError, actual.forecastError, tolerance);
        assertArrayEquals(expected.forecastErrorCov, actual.forecastErrorCov, tolerance);
        assertArrayEquals(expected.kalmanGain, actual.kalmanGain, tolerance);
        assertArrayEquals(expected.logLikelihoodObs, actual.logLikelihoodObs, tolerance);
        assertNullableArrayEquals(expected.predictedDiffuseStateCov, actual.predictedDiffuseStateCov, tolerance);
        assertNullableArrayEquals(expected.forecastErrorDiffuseCov, actual.forecastErrorDiffuseCov, tolerance);
        assertEquals(expected.converged, actual.converged);
        assertEquals(expected.periodConverged, actual.periodConverged);
        assertEquals(expected.nobsDiffuse, actual.nobsDiffuse);
        assertEquals(expected.perObsKalmanGain, actual.perObsKalmanGain);
    }

    private static void assertFilterResultEquals(ZFilterResult expected, ZFilterResult actual, double tolerance) {
        assertEquals(expected.kEndog, actual.kEndog);
        assertEquals(expected.kStates, actual.kStates);
        assertEquals(expected.nobs, actual.nobs);
        assertArrayEquals(expected.predictedState, actual.predictedState, tolerance);
        assertArrayEquals(expected.predictedStateCov, actual.predictedStateCov, tolerance);
        assertArrayEquals(expected.filteredState, actual.filteredState, tolerance);
        assertArrayEquals(expected.filteredStateCov, actual.filteredStateCov, tolerance);
        assertArrayEquals(expected.forecast, actual.forecast, tolerance);
        assertArrayEquals(expected.forecastError, actual.forecastError, tolerance);
        assertArrayEquals(expected.forecastErrorCov, actual.forecastErrorCov, tolerance);
        assertArrayEquals(expected.kalmanGain, actual.kalmanGain, tolerance);
        assertArrayEquals(expected.logLikelihoodObs, actual.logLikelihoodObs, tolerance);
        assertNullableArrayEquals(expected.predictedDiffuseStateCov, actual.predictedDiffuseStateCov, tolerance);
        assertNullableArrayEquals(expected.forecastErrorDiffuseCov, actual.forecastErrorDiffuseCov, tolerance);
        assertEquals(expected.converged, actual.converged);
        assertEquals(expected.periodConverged, actual.periodConverged);
        assertEquals(expected.nobsDiffuse, actual.nobsDiffuse);
        assertEquals(expected.perObsKalmanGain, actual.perObsKalmanGain);
    }

    private static void assertSmootherResultEquals(SmootherResult expected, SmootherResult actual, double tolerance) {
        assertEquals(expected.kEndog, actual.kEndog);
        assertEquals(expected.kStates, actual.kStates);
        assertEquals(expected.kPosdef, actual.kPosdef);
        assertEquals(expected.nobs, actual.nobs);
        assertArrayEquals(expected.smoothedState, actual.smoothedState, tolerance);
        assertArrayEquals(expected.smoothedStateCov, actual.smoothedStateCov, tolerance);
        assertArrayEquals(expected.smoothedObsDisturbance, actual.smoothedObsDisturbance, tolerance);
        assertArrayEquals(expected.smoothedStateDisturbance, actual.smoothedStateDisturbance, tolerance);
        assertArrayEquals(expected.smoothedObsDisturbanceCov, actual.smoothedObsDisturbanceCov, tolerance);
        assertArrayEquals(expected.smoothedStateDisturbanceCov, actual.smoothedStateDisturbanceCov, tolerance);
        assertArrayEquals(expected.scaledSmoothedEstimator, actual.scaledSmoothedEstimator, tolerance);
        assertArrayEquals(expected.scaledSmoothedEstimatorCov, actual.scaledSmoothedEstimatorCov, tolerance);
        assertArrayEquals(expected.smoothingError, actual.smoothingError, tolerance);
        assertArrayEquals(expected.innovationsTransition, actual.innovationsTransition, tolerance);
        assertNullableArrayEquals(expected.scaledSmoothedDiffuseEstimator, actual.scaledSmoothedDiffuseEstimator, tolerance);
        assertNullableArrayEquals(expected.scaledSmoothedDiffuse1EstimatorCov, actual.scaledSmoothedDiffuse1EstimatorCov, tolerance);
        assertNullableArrayEquals(expected.scaledSmoothedDiffuse2EstimatorCov, actual.scaledSmoothedDiffuse2EstimatorCov, tolerance);
    }

    private static void assertSmootherResultEquals(ZSmootherResult expected, ZSmootherResult actual, double tolerance) {
        assertEquals(expected.kEndog, actual.kEndog);
        assertEquals(expected.kStates, actual.kStates);
        assertEquals(expected.kPosdef, actual.kPosdef);
        assertEquals(expected.nobs, actual.nobs);
        assertArrayEquals(expected.smoothedState, actual.smoothedState, tolerance);
        assertArrayEquals(expected.smoothedStateCov, actual.smoothedStateCov, tolerance);
        assertArrayEquals(expected.smoothedObsDisturbance, actual.smoothedObsDisturbance, tolerance);
        assertArrayEquals(expected.smoothedStateDisturbance, actual.smoothedStateDisturbance, tolerance);
        assertArrayEquals(expected.smoothedObsDisturbanceCov, actual.smoothedObsDisturbanceCov, tolerance);
        assertArrayEquals(expected.smoothedStateDisturbanceCov, actual.smoothedStateDisturbanceCov, tolerance);
        assertArrayEquals(expected.scaledSmoothedEstimator, actual.scaledSmoothedEstimator, tolerance);
        assertArrayEquals(expected.scaledSmoothedEstimatorCov, actual.scaledSmoothedEstimatorCov, tolerance);
        assertArrayEquals(expected.smoothingError, actual.smoothingError, tolerance);
        assertArrayEquals(expected.innovationsTransition, actual.innovationsTransition, tolerance);
        assertNullableArrayEquals(expected.scaledSmoothedDiffuseEstimator, actual.scaledSmoothedDiffuseEstimator, tolerance);
        assertNullableArrayEquals(expected.scaledSmoothedDiffuse1EstimatorCov, actual.scaledSmoothedDiffuse1EstimatorCov, tolerance);
        assertNullableArrayEquals(expected.scaledSmoothedDiffuse2EstimatorCov, actual.scaledSmoothedDiffuse2EstimatorCov, tolerance);
    }

    private static void assertSimulationResultEquals(SimulationSmootherResult expected,
                                                     SimulationSmootherResult actual,
                                                     double tolerance) {
        assertEquals(expected.kEndog, actual.kEndog);
        assertEquals(expected.kStates, actual.kStates);
        assertEquals(expected.kPosdef, actual.kPosdef);
        assertEquals(expected.nobs, actual.nobs);
        assertArrayEquals(activeSlice(expected.generatedMeasurementDisturbance,
            expected.generatedMeasurementDisturbanceBase(), expected.generatedMeasurementDisturbanceLength()),
            activeSlice(actual.generatedMeasurementDisturbance,
                actual.generatedMeasurementDisturbanceBase(), actual.generatedMeasurementDisturbanceLength()), tolerance);
        assertArrayEquals(activeSlice(expected.generatedStateDisturbance,
            expected.generatedStateDisturbanceBase(), expected.generatedStateDisturbanceLength()),
            activeSlice(actual.generatedStateDisturbance,
                actual.generatedStateDisturbanceBase(), actual.generatedStateDisturbanceLength()), tolerance);
        assertArrayEquals(activeSlice(expected.generatedObs, expected.generatedObsBase(), expected.generatedObsLength()),
            activeSlice(actual.generatedObs, actual.generatedObsBase(), actual.generatedObsLength()), tolerance);
        assertArrayEquals(activeSlice(expected.generatedState, expected.generatedStateBase(), expected.generatedStateLength()),
            activeSlice(actual.generatedState, actual.generatedStateBase(), actual.generatedStateLength()), tolerance);
        assertArrayEquals(activeSlice(expected.simulatedMeasurementDisturbance,
            expected.simulatedMeasurementDisturbanceBase(), expected.simulatedMeasurementDisturbanceLength()),
            activeSlice(actual.simulatedMeasurementDisturbance,
                actual.simulatedMeasurementDisturbanceBase(), actual.simulatedMeasurementDisturbanceLength()), tolerance);
        assertArrayEquals(activeSlice(expected.simulatedStateDisturbance,
            expected.simulatedStateDisturbanceBase(), expected.simulatedStateDisturbanceLength()),
            activeSlice(actual.simulatedStateDisturbance,
                actual.simulatedStateDisturbanceBase(), actual.simulatedStateDisturbanceLength()), tolerance);
        assertArrayEquals(activeSlice(expected.simulatedState, expected.simulatedStateBase(), expected.simulatedStateLength()),
            activeSlice(actual.simulatedState, actual.simulatedStateBase(), actual.simulatedStateLength()), tolerance);
    }

    private static void assertSimulationResultEquals(ZSimulationSmootherResult expected,
                                                     ZSimulationSmootherResult actual,
                                                     double tolerance) {
        assertEquals(expected.kEndog, actual.kEndog);
        assertEquals(expected.kStates, actual.kStates);
        assertEquals(expected.kPosdef, actual.kPosdef);
        assertEquals(expected.nobs, actual.nobs);
        assertArrayEquals(activeSlice(expected.generatedMeasurementDisturbance,
            expected.generatedMeasurementDisturbanceBase(), expected.generatedMeasurementDisturbanceLength()),
            activeSlice(actual.generatedMeasurementDisturbance,
                actual.generatedMeasurementDisturbanceBase(), actual.generatedMeasurementDisturbanceLength()), tolerance);
        assertArrayEquals(activeSlice(expected.generatedStateDisturbance,
            expected.generatedStateDisturbanceBase(), expected.generatedStateDisturbanceLength()),
            activeSlice(actual.generatedStateDisturbance,
                actual.generatedStateDisturbanceBase(), actual.generatedStateDisturbanceLength()), tolerance);
        assertArrayEquals(activeSlice(expected.generatedObs, expected.generatedObsBase(), expected.generatedObsLength()),
            activeSlice(actual.generatedObs, actual.generatedObsBase(), actual.generatedObsLength()), tolerance);
        assertArrayEquals(activeSlice(expected.generatedState, expected.generatedStateBase(), expected.generatedStateLength()),
            activeSlice(actual.generatedState, actual.generatedStateBase(), actual.generatedStateLength()), tolerance);
        assertArrayEquals(activeSlice(expected.simulatedMeasurementDisturbance,
            expected.simulatedMeasurementDisturbanceBase(), expected.simulatedMeasurementDisturbanceLength()),
            activeSlice(actual.simulatedMeasurementDisturbance,
                actual.simulatedMeasurementDisturbanceBase(), actual.simulatedMeasurementDisturbanceLength()), tolerance);
        assertArrayEquals(activeSlice(expected.simulatedStateDisturbance,
            expected.simulatedStateDisturbanceBase(), expected.simulatedStateDisturbanceLength()),
            activeSlice(actual.simulatedStateDisturbance,
                actual.simulatedStateDisturbanceBase(), actual.simulatedStateDisturbanceLength()), tolerance);
        assertArrayEquals(activeSlice(expected.simulatedState, expected.simulatedStateBase(), expected.simulatedStateLength()),
            activeSlice(actual.simulatedState, actual.simulatedStateBase(), actual.simulatedStateLength()), tolerance);
    }

    private static double[] activeSlice(double[] values, int base, int length) {
        return Arrays.copyOfRange(values, base, base + length);
    }

    private static void assertNullableArrayEquals(double[] expected, double[] actual, double tolerance) {
        if (expected == null || actual == null) {
            assertSame(expected, actual);
            return;
        }
        assertArrayEquals(expected, actual, tolerance);
    }

    private static class DelegatingModel extends StateSpaceModel {
        private final StateSpaceModel delegate;

        private DelegatingModel(StateSpaceModel delegate) {
            this.delegate = Objects.requireNonNull(delegate, "delegate");
        }

        @Override
        public int stateCount() {
            return delegate.stateCount();
        }

        @Override
        public int stateDisturbanceCount() {
            return delegate.stateDisturbanceCount();
        }

        @Override
        public int observationDimension() {
            return delegate.observationDimension();
        }

        @Override
        public int observationCount() {
            return delegate.observationCount();
        }

        @Override
        public boolean complex() {
            return delegate.complex();
        }

        @Override
        public double[] transitionData() {
            return delegate.transitionData();
        }

        @Override
        public int transitionOffset(int t) {
            return delegate.transitionOffset(t);
        }

        @Override
        public int transitionLeadingDimension() {
            return delegate.transitionLeadingDimension();
        }

        @Override
        public double[] stateInterceptData() {
            return delegate.stateInterceptData();
        }

        @Override
        public int stateInterceptOffset(int t) {
            return delegate.stateInterceptOffset(t);
        }

        @Override
        public double[] selectionData() {
            return delegate.selectionData();
        }

        @Override
        public int selectionOffset(int t) {
            return delegate.selectionOffset(t);
        }

        @Override
        public int selectionLeadingDimension() {
            return delegate.selectionLeadingDimension();
        }

        @Override
        public double[] stateCovarianceData() {
            return delegate.stateCovarianceData();
        }

        @Override
        public int stateCovarianceOffset(int t) {
            return delegate.stateCovarianceOffset(t);
        }

        @Override
        public int stateCovarianceLeadingDimension() {
            return delegate.stateCovarianceLeadingDimension();
        }

        @Override
        public double[] designData() {
            return delegate.designData();
        }

        @Override
        public int designOffset(int t) {
            return delegate.designOffset(t);
        }

        @Override
        public int designLeadingDimension() {
            return delegate.designLeadingDimension();
        }

        @Override
        public double[] obsInterceptData() {
            return delegate.obsInterceptData();
        }

        @Override
        public int obsInterceptOffset(int t) {
            return delegate.obsInterceptOffset(t);
        }

        @Override
        public double[] obsCovData() {
            return delegate.obsCovData();
        }

        @Override
        public int obsCovOffset(int t) {
            return delegate.obsCovOffset(t);
        }

        @Override
        public int obsCovLeadingDimension() {
            return delegate.obsCovLeadingDimension();
        }

        @Override
        public double[] endogData() {
            return delegate.endogData();
        }

        @Override
        public int endogOffset(int t) {
            return delegate.endogOffset(t);
        }

        @Override
        public boolean isMissing(int obsIndex, int t) {
            return delegate.isMissing(obsIndex, t);
        }

        @Override
        public int missingCount(int t) {
            return delegate.missingCount(t);
        }
    }

    private static class DelegatingStateSpace extends DelegatingModel {
        protected DelegatingStateSpace(StateSpaceModel delegate) {
            super(delegate);
        }
    }

    private static class DelegatingObservationModel extends DelegatingModel {
        protected DelegatingObservationModel(StateSpaceModel delegate) {
            super(delegate);
        }
    }

    private static final class CombinedModelView extends StateSpaceModel {
        private final StateSpaceModel stateDelegate;
        private final StateSpaceModel observationDelegate;

        private CombinedModelView(StateSpaceModel stateDelegate, StateSpaceModel observationDelegate) {
            this.stateDelegate = Objects.requireNonNull(stateDelegate, "stateDelegate");
            this.observationDelegate = Objects.requireNonNull(observationDelegate, "observationDelegate");
        }

        @Override
        public int stateCount() {
            return stateDelegate.stateCount();
        }

        @Override
        public int stateDisturbanceCount() {
            return stateDelegate.stateDisturbanceCount();
        }

        @Override
        public int observationDimension() {
            return observationDelegate.observationDimension();
        }

        @Override
        public int observationCount() {
            return observationDelegate.observationCount();
        }

        @Override
        public boolean complex() {
            return observationDelegate.complex();
        }

        @Override
        public double[] transitionData() {
            return stateDelegate.transitionData();
        }

        @Override
        public int transitionOffset(int t) {
            return stateDelegate.transitionOffset(t);
        }

        @Override
        public int transitionLeadingDimension() {
            return stateDelegate.transitionLeadingDimension();
        }

        @Override
        public double[] stateInterceptData() {
            return stateDelegate.stateInterceptData();
        }

        @Override
        public int stateInterceptOffset(int t) {
            return stateDelegate.stateInterceptOffset(t);
        }

        @Override
        public double[] selectionData() {
            return stateDelegate.selectionData();
        }

        @Override
        public int selectionOffset(int t) {
            return stateDelegate.selectionOffset(t);
        }

        @Override
        public int selectionLeadingDimension() {
            return stateDelegate.selectionLeadingDimension();
        }

        @Override
        public double[] stateCovarianceData() {
            return stateDelegate.stateCovarianceData();
        }

        @Override
        public int stateCovarianceOffset(int t) {
            return stateDelegate.stateCovarianceOffset(t);
        }

        @Override
        public int stateCovarianceLeadingDimension() {
            return stateDelegate.stateCovarianceLeadingDimension();
        }

        @Override
        public double[] designData() {
            return observationDelegate.designData();
        }

        @Override
        public int designOffset(int t) {
            return observationDelegate.designOffset(t);
        }

        @Override
        public int designLeadingDimension() {
            return observationDelegate.designLeadingDimension();
        }

        @Override
        public double[] obsInterceptData() {
            return observationDelegate.obsInterceptData();
        }

        @Override
        public int obsInterceptOffset(int t) {
            return observationDelegate.obsInterceptOffset(t);
        }

        @Override
        public double[] obsCovData() {
            return observationDelegate.obsCovData();
        }

        @Override
        public int obsCovOffset(int t) {
            return observationDelegate.obsCovOffset(t);
        }

        @Override
        public int obsCovLeadingDimension() {
            return observationDelegate.obsCovLeadingDimension();
        }

        @Override
        public double[] endogData() {
            return observationDelegate.endogData();
        }

        @Override
        public int endogOffset(int t) {
            return observationDelegate.endogOffset(t);
        }

        @Override
        public boolean isMissing(int obsIndex, int t) {
            return observationDelegate.isMissing(obsIndex, t);
        }

        @Override
        public int missingCount(int t) {
            return observationDelegate.missingCount(t);
        }
    }

    private static final class DelegatingComplexStateSpace extends DelegatingStateSpace {
        private DelegatingComplexStateSpace(StateSpaceModel delegate) {
            super(delegate);
        }
    }

    private static final class DelegatingComplexObservationModel extends DelegatingObservationModel {
        private DelegatingComplexObservationModel(StateSpaceModel delegate) {
            super(delegate);
        }
    }
}