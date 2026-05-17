package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KalmanApiUsabilityTest {
    private static final double TOL = 1e-12;

    @Test
    void filterFacadeMatchesEngineAndSupportsWorkspaceReuse() {
        KalmanSSM model = KalmanModelFixtures.defaultScalarAr1();
        InitialState initialState = InitialState.known(new double[]{0.0}, new double[]{1.0});
        FilterOptions options = FilterOptions.likelihoodOnly();

        FilterResult expected = KalmanEngine.filter(model, initialState, options);
        FilterResult actual = Kalman.filter()
            .model(model)
            .initialState(initialState)
            .likelihoodOnly()
            .run();

        assertEquals(expected.logLikelihood(), actual.logLikelihood(), TOL);
        assertEquals(model.observationCount(), actual.logLikelihoodObsLength());

        try (KalmanWorkspace workspace = Kalman.workspace()) {
            FilterResult first = Kalman.filter()
                .model(model)
                .initialState(initialState)
                .options(options)
                .run(workspace);
            FilterResult second = Kalman.filter()
                .model(model)
                .initialState(initialState)
                .options(options)
                .run(workspace);

            assertEquals(expected.logLikelihood(), first.logLikelihood(), TOL);
            assertEquals(first.logLikelihood(), second.logLikelihood(), TOL);
        }
    }

    @Test
    void smootherFacadeSupportsModelAndFilterResultInputs() {
        KalmanSSM model = KalmanModelFixtures.defaultScalarAr1();
        InitialState initialState = InitialState.known(new double[]{0.0}, new double[]{1.0});

        try (KalmanWorkspace workspace = Kalman.workspace()) {
            SmootherResult fromInitialState = Kalman.smooth()
                .model(model)
                .initialState(initialState)
                .stateOnly()
                .run(workspace);
            FilterResult filterResult = Kalman.filter()
                .model(model)
                .initialState(initialState)
                .run(workspace);
            SmootherResult fromFilterResult = Kalman.smooth()
                .model(model)
                .filterResult(filterResult)
                .stateOnly()
                .run(workspace);

            assertEquals(model.stateCount() * model.observationCount(), fromInitialState.smoothedStateLength());
            assertEquals(fromInitialState.smoothedStateLength(), fromFilterResult.smoothedStateLength());
        }
    }

    @Test
    void simulationSmootherFacadeSupportsRandomDraws() {
        KalmanSSM model = KalmanModelFixtures.defaultScalarAr1();
        InitialState initialState = InitialState.known(new double[]{0.0}, new double[]{1.0});

        SimulationSmootherResult result = Kalman.simulate()
            .model(model)
            .initialState(initialState)
            .stateOnly()
            .random(new Random(1234))
            .run();

        assertEquals(model.stateCount() * model.observationCount(), result.simulatedStateLength());
    }

    @Test
    void facadeReportsIncompleteAndMismatchedConfiguration() {
        KalmanSSM complexModel = KalmanModelFixtures.complexSignalOnlyAr1(0.7, 0.2, 0.5,
            new double[]{1.0, 0.4, -0.2, 0.6});
        InitialState initialState = InitialState.known(new double[]{0.0}, new double[]{1.0});

        IllegalArgumentException missingModel = assertThrows(IllegalArgumentException.class,
            () -> Kalman.filter().initialState(initialState).run());
        assertTrue(missingModel.getMessage().contains("model"));

        IllegalArgumentException wrongFilter = assertThrows(IllegalArgumentException.class,
            () -> Kalman.filter().model(complexModel).initialState(initialState).run());
        assertTrue(wrongFilter.getMessage().contains("real-valued"));
    }
}