package com.curioloop.yum4j.tsa.statespace;

import com.curioloop.yum4j.ssm.kalman.KalmanSimulation;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherResult;

import java.util.Random;

public final class SimulationSupport {

    private SimulationSupport() {
    }

    public static void requirePositiveSimulations(int nsimulations) {
        if (nsimulations < 1) {
            throw new IllegalArgumentException("nsimulations must be positive");
        }
    }

    public static void requirePositiveRepetitions(int repetitions) {
        if (repetitions < 1) {
            throw new IllegalArgumentException("repetitions must be positive");
        }
    }

    public static SimulationSmootherOptions resolvedOptions(SimulationSmootherOptions options) {
        return options == null ? SimulationSmootherOptions.defaults().withGeneratedOutputs() : options;
    }

    public static SimulationSmootherResult simulate(KalmanSSM model,
                                                    InitialState initialState,
                                                    Random random,
                                                    SimulationSmootherOptions options) {
        KalmanSimulation simulation = new KalmanSimulation()
            .model(model)
            .initialState(initialState)
            .options(resolvedOptions(options));
        if (random != null) {
            simulation.random(random);
        }
        return simulation.simulate();
    }

    public static SimulationSmootherResult simulate(KalmanSSM model,
                                                    InitialState initialState,
                                                    double[] measurementDisturbanceVariates,
                                                    double[] stateDisturbanceVariates,
                                                    double[] initialStateVariates,
                                                    SimulationSmootherOptions options) {
        return new KalmanSimulation()
            .model(model)
            .initialState(initialState)
            .variates(measurementDisturbanceVariates, stateDisturbanceVariates, initialStateVariates)
            .options(resolvedOptions(options))
            .simulate();
    }
}