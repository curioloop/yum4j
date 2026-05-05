package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.kalman.filter.FilterResult;
import com.curioloop.yum4j.kalman.filter.FilterResultShape;
import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.filter.KalmanFilter;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.model.ModelFixture;
import com.curioloop.yum4j.kalman.smooth.SimulationSmoother;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherSpec;
import com.curioloop.yum4j.kalman.smooth.SmootherResultShape;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KalmanProfileContractTest {

    private static final double TOL = 1e-12;

    private static ModelFixture buildScalarModel() {
        ModelFixture rep = new ModelFixture(1, 1, 1, 3);
        for (int t = 0; t < rep.nobs; t++) {
            rep.setDesign(new double[]{1.0}, t);
            rep.setObsIntercept(new double[]{0.0}, t);
            rep.setObsCov(new double[]{1.0}, t);
            rep.setTransition(new double[]{1.0}, t);
            rep.setStateIntercept(new double[]{0.0}, t);
            rep.setSelection(new double[]{1.0}, t);
            rep.setStateCov(new double[]{0.5}, t);
            rep.setEndog(new double[]{t + 1.0}, t);
        }
        return rep;
    }

    @Test
    void testNamedFilterProfilesMatchExpectedStorageContracts() {
        FilterSpec compact = FilterSpec.compact();
        FilterSpec conventional = FilterSpec.conventionalSmoothing();
        FilterSpec classical = FilterSpec.classicalSmoothing();

        assertFalse(compact.stores(FilterSpec.Storage.FORECAST));
        assertFalse(compact.stores(FilterSpec.Storage.PREDICTED_STATE));
        assertFalse(compact.stores(FilterSpec.Storage.FILTERED_STATE));
        assertFalse(compact.stores(FilterSpec.Storage.KALMAN_GAIN));
        assertFalse(compact.stores(FilterSpec.Storage.LIKELIHOOD));

        assertTrue(conventional.stores(FilterSpec.Storage.FORECAST));
        assertTrue(conventional.stores(FilterSpec.Storage.PREDICTED_STATE));
        assertFalse(conventional.stores(FilterSpec.Storage.FILTERED_STATE));
        assertTrue(conventional.stores(FilterSpec.Storage.KALMAN_GAIN));
        assertFalse(conventional.stores(FilterSpec.Storage.LIKELIHOOD));

        assertTrue(classical.stores(FilterSpec.Storage.FILTERED_STATE));
        assertTrue(classical.stores(FilterSpec.Storage.KALMAN_GAIN));
        assertFalse(classical.stores(FilterSpec.Storage.LIKELIHOOD));
    }

    @Test
    void testSmootherProfilesExposeRequiredFilterContracts() {
        assertSame(FilterSpec.conventionalSmoothing(), SmootherSpec.conventional().requiredFilterSpec());
        assertSame(FilterSpec.classicalSmoothing(), SmootherSpec.classical().requiredFilterSpec());
        assertSame(FilterSpec.classicalSmoothing(), SmootherSpec.alternative().requiredFilterSpec());
        assertFalse(SmootherSpec.conventional().requiresFilteredStateHistory());
        assertTrue(SmootherSpec.classical().requiresFilteredStateHistory());
        assertTrue(SmootherSpec.alternative().requiresFilteredStateHistory());
    }

    @Test
    void testProfileResultShapesMatchRetainedStorageNeeds() {
        FilterResultShape conventionalShape = FilterSpec.conventionalSmoothing().resultShape(false);
        assertTrue(conventionalShape.storesForecast());
        assertTrue(conventionalShape.storesPredictedState());
        assertFalse(conventionalShape.storesFilteredState());
        assertTrue(conventionalShape.storesKalmanGain());
        assertFalse(conventionalShape.storesLikelihood());

        FilterSpec forecastOnly = FilterSpec.defaults().without(
                FilterSpec.Storage.PREDICTED_STATE,
                FilterSpec.Storage.FILTERED_STATE,
                FilterSpec.Storage.KALMAN_GAIN,
                FilterSpec.Storage.LIKELIHOOD);
        assertFalse(forecastOnly.resultShape(false).storesPredictedState());
        assertTrue(forecastOnly.resultShape(true).storesPredictedState());

        SmootherResultShape conventionalOutput = SmootherSpec.conventional().resultShape();
        assertTrue(conventionalOutput.storesState());
        assertTrue(conventionalOutput.storesStateCovariance());
        assertTrue(conventionalOutput.storesDisturbance());
        assertTrue(conventionalOutput.storesDisturbanceCovariance());
        assertTrue(conventionalOutput.storesAuxiliary());

        SmootherResultShape leanOutput = SmootherSpec.conventional().withoutCovariances().resultShape();
        assertTrue(leanOutput.storesState());
        assertFalse(leanOutput.storesStateCovariance());
        assertTrue(leanOutput.storesDisturbance());
        assertFalse(leanOutput.storesDisturbanceCovariance());
        assertFalse(leanOutput.storesAuxiliary());
    }

    @Test
    void testSimulationProfilesMapToExplicitFilterAndSmootherContracts() {
        assertSame(FilterSpec.conventionalSmoothing(), SimulationSmootherSpec.all().requiredFilterSpec());
        assertSame(SmootherSpec.conventional().withoutCovariances(), SimulationSmootherSpec.all().requiredSmootherSpec());
        assertSame(SmootherSpec.conventional().stateOnly(), SimulationSmootherSpec.stateOnly().requiredSmootherSpec());
        assertSame(SmootherSpec.conventional().disturbanceOnly(), SimulationSmootherSpec.disturbanceOnly().requiredSmootherSpec());
        assertFalse(SimulationSmootherSpec.all().storesGeneratedOutputs());
        assertTrue(SimulationSmootherSpec.all().withGeneratedOutputs().storesGeneratedOutputs());
        assertTrue(SimulationSmootherSpec.stateOnly().withGeneratedOutputs().storesGeneratedOutputs());
        assertTrue(SimulationSmootherSpec.disturbanceOnly().withGeneratedOutputs().storesGeneratedOutputs());
    }

    @Test
    void testExplicitFilterSpecsDriveRetainedStorage() {
        ModelFixture rep = buildScalarModel();
        StateInitialization init = StateInitialization.approximateDiffuse();

        FilterResult compact = KalmanFilter.filter(rep, init, null, FilterSpec.compact());
        FilterResult conventional = KalmanFilter.filter(rep, init, null, FilterSpec.conventionalSmoothing());

        assertEquals(0, compact.predictedStateLength());
        assertEquals(0, compact.forecastLength());
        assertEquals(0, compact.kalmanGainLength());
        assertTrue(conventional.predictedStateLength() > 0);
        assertTrue(conventional.forecastErrorLength() > 0);
        assertTrue(conventional.kalmanGainLength() > 0);
    }

    @Test
    void testExplicitSimulationSpecsControlGeneratedOutputs() {
        ModelFixture rep = buildScalarModel();
        StateInitialization init = StateInitialization.approximateDiffuse();
        double[] measurement = new double[rep.nobs * rep.kEndog];
        double[] state = new double[rep.nobs * rep.kPosdef];
        double[] initial = new double[rep.kStates];

        SimulationSmootherResult lean = SimulationSmoother.simulate(rep, init,
                measurement, state, initial, SimulationSmootherSpec.all());
        SimulationSmootherResult generated = SimulationSmoother.simulate(rep, init,
                measurement, state, initial, SimulationSmootherSpec.all().withGeneratedOutputs());

        assertEquals(0, lean.generatedObs.length);
        assertArrayEquals(lean.simulatedState, generated.simulatedState, TOL);
        assertArrayEquals(lean.simulatedMeasurementDisturbance, generated.simulatedMeasurementDisturbance, TOL);
        assertTrue(generated.generatedObs.length > 0);
        assertTrue(generated.generatedState.length > 0);
    }
}