package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherResult;

import java.util.Objects;
import java.util.Random;

public final class KalmanSimulation implements KalmanInference<SimulationSmootherResult, KalmanWorkspace> {
    private KalmanSSM model;
    private InitialState initialState;
    private Random rng;
    private boolean explicitVariates;
    private double[] measurementDisturbanceVariates;
    private double[] stateDisturbanceVariates;
    private double[] initialStateVariates;
    private SimulationSmootherOptions options = SimulationSmootherOptions.defaults();

    public KalmanSimulation model(KalmanSSM model) {
        this.model = Objects.requireNonNull(model, "model");
        return this;
    }

    public KalmanSimulation initialState(InitialState initialState) {
        this.initialState = Objects.requireNonNull(initialState, "initialState");
        return this;
    }

    public KalmanSimulation random(Random rng) {
        this.rng = Objects.requireNonNull(rng, "rng");
        this.explicitVariates = false;
        return this;
    }

    public KalmanSimulation variates(double[] measurementDisturbanceVariates,
                                        double[] stateDisturbanceVariates,
                                        double[] initialStateVariates) {
        this.measurementDisturbanceVariates = measurementDisturbanceVariates;
        this.stateDisturbanceVariates = stateDisturbanceVariates;
        this.initialStateVariates = initialStateVariates;
        this.explicitVariates = true;
        return this;
    }

    public KalmanSimulation options(SimulationSmootherOptions options) {
        this.options = Objects.requireNonNull(options, "options");
        return this;
    }

    public KalmanSimulation method(SimulationSmootherOptions.Method method) {
        this.options = options.toBuilder().method(method).build();
        return this;
    }

    public KalmanSimulation kfs() {
        return method(SimulationSmootherOptions.Method.KFS);
    }

    public KalmanSimulation cfa() {
        return method(SimulationSmootherOptions.Method.CFA);
    }

    public KalmanSimulation stateOnly() {
        this.options = options.toBuilder().retainOnly(SimulationSmootherOptions.Surface.STATE).build();
        return this;
    }

    public KalmanSimulation disturbanceOnly() {
        this.options = options.toBuilder().retainOnly(SimulationSmootherOptions.Surface.DISTURBANCE).build();
        return this;
    }

    public KalmanSimulation generatedOutputs() {
        this.options = options.withGeneratedOutputs();
        return this;
    }

    public KalmanSimulation withoutGeneratedOutputs() {
        this.options = options.withoutGeneratedOutputs();
        return this;
    }

    public KalmanSimulation univariateFilter() {
        this.options = options.withUnivariateFilter();
        return this;
    }

    public SimulationSmootherResult simulate() {
        return simulate(null);
    }

    public SimulationSmootherResult simulate(KalmanWorkspace workspace) {
        KalmanSSM resolvedModel = requireRealModel();
        InitialState resolvedInitialState = requireInitialState();
        if (explicitVariates) {
            if (workspace == null) {
                return SimulationSmootherEngine.simulate(resolvedModel,
                    resolvedInitialState,
                    measurementDisturbanceVariates,
                    stateDisturbanceVariates,
                    initialStateVariates,
                    options);
            }
            return SimulationSmootherEngine.simulate(resolvedModel,
                resolvedInitialState,
                measurementDisturbanceVariates,
                stateDisturbanceVariates,
                initialStateVariates,
                workspace.simulationWorkspace(),
                options);
        }
        Random resolvedRng = rng == null ? new Random() : rng;
        if (workspace == null) {
            return SimulationSmootherEngine.simulate(resolvedModel, resolvedInitialState, resolvedRng, options);
        }
        return SimulationSmootherEngine.simulate(resolvedModel,
            resolvedInitialState,
            resolvedRng,
            workspace.simulationWorkspace(),
            options);
    }

    @Override
    public SimulationSmootherResult run(KalmanWorkspace workspace) {
        return simulate(workspace);
    }

    private KalmanSSM requireRealModel() {
        if (model == null) {
            throw new IllegalArgumentException("model must be set before simulation smoothing");
        }
        if (model.complex()) {
            throw new IllegalArgumentException("model is complex; the high-level Kalman API currently supports real-valued models");
        }
        return model;
    }

    private InitialState requireInitialState() {
        if (initialState == null) {
            throw new IllegalArgumentException("initialState must be set before simulation smoothing");
        }
        return initialState;
    }
}