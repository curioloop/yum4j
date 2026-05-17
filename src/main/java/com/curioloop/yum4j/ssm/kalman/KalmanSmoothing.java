package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherMethod;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;

import java.util.Objects;

public final class KalmanSmoothing implements KalmanInference<SmootherResult, KalmanWorkspace> {
    private KalmanSSM model;
    private InitialState initialState;
    private FilterResult filterResult;
    private SmootherOptions options = SmootherOptions.defaults();

    public KalmanSmoothing model(KalmanSSM model) {
        this.model = Objects.requireNonNull(model, "model");
        return this;
    }

    public KalmanSmoothing initialState(InitialState initialState) {
        this.initialState = Objects.requireNonNull(initialState, "initialState");
        return this;
    }

    public KalmanSmoothing filterResult(FilterResult filterResult) {
        this.filterResult = Objects.requireNonNull(filterResult, "filterResult");
        return this;
    }

    public KalmanSmoothing options(SmootherOptions options) {
        this.options = Objects.requireNonNull(options, "options");
        return this;
    }

    public KalmanSmoothing method(SmootherMethod method) {
        this.options = options.toBuilder().method(method).build();
        return this;
    }

    public KalmanSmoothing conventional() {
        return method(SmootherMethod.CONVENTIONAL);
    }

    public KalmanSmoothing classical() {
        return method(SmootherMethod.CLASSICAL);
    }

    public KalmanSmoothing alternative() {
        return method(SmootherMethod.ALTERNATIVE);
    }

    public KalmanSmoothing univariate() {
        return method(SmootherMethod.UNIVARIATE);
    }

    public KalmanSmoothing surfaces(SmootherOptions.Surface surface, SmootherOptions.Surface... more) {
        this.options = options.toBuilder().retainOnly(surface, more).build();
        return this;
    }

    public KalmanSmoothing retain(SmootherOptions.Surface surface, SmootherOptions.Surface... more) {
        this.options = options.toBuilder().retain(surface, more).build();
        return this;
    }

    public KalmanSmoothing drop(SmootherOptions.Surface surface, SmootherOptions.Surface... more) {
        this.options = options.toBuilder().drop(surface, more).build();
        return this;
    }

    public KalmanSmoothing stateOnly() {
        return surfaces(SmootherOptions.Surface.STATE);
    }

    public KalmanSmoothing disturbanceOnly() {
        return surfaces(SmootherOptions.Surface.DISTURBANCE);
    }

    public KalmanSmoothing all() {
        this.options = options.toBuilder().retainAll().build();
        return this;
    }

    public SmootherResult smooth() {
        return smooth(null);
    }

    public SmootherResult smooth(KalmanWorkspace workspace) {
        KalmanSSM resolvedModel = requireRealModel();
        if (filterResult != null) {
            if (workspace == null) {
                return SmootherEngine.smooth(resolvedModel, filterResult, options);
            }
            SmootherResult result = SmootherEngine.smoothBorrowedUnsafe(
                resolvedModel,
                filterResult,
                workspace.smootherWorkspace(),
                options).copy();
            workspace.smootherWorkspace().releaseSmootherRetainedResults();
            return result;
        }
        InitialState resolvedInitialState = requireInitialState();
        if (workspace == null) {
            return SmootherEngine.smooth(resolvedModel, resolvedInitialState, options);
        }
        return SmootherEngine.smooth(resolvedModel, resolvedInitialState, workspace.smootherWorkspace(), options);
    }

    @Override
    public SmootherResult run(KalmanWorkspace workspace) {
        return smooth(workspace);
    }

    private KalmanSSM requireRealModel() {
        if (model == null) {
            throw new IllegalArgumentException("model must be set before smoothing");
        }
        if (model.complex()) {
            throw new IllegalArgumentException("model is complex; the high-level Kalman API currently supports real-valued models");
        }
        return model;
    }

    private InitialState requireInitialState() {
        if (initialState == null) {
            throw new IllegalArgumentException("initialState or filterResult must be set before smoothing");
        }
        return initialState;
    }
}