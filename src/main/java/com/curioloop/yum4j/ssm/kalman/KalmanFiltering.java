package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.filter.FilterMethod;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;

import java.util.Objects;

public final class KalmanFiltering implements KalmanInference<FilterResult, KalmanWorkspace> {
    private KalmanSSM model;
    private InitialState initialState;
    private FilterOptions options = FilterOptions.defaults();

    public KalmanFiltering model(KalmanSSM model) {
        this.model = Objects.requireNonNull(model, "model");
        return this;
    }

    public KalmanFiltering initialState(InitialState initialState) {
        this.initialState = Objects.requireNonNull(initialState, "initialState");
        return this;
    }

    public KalmanFiltering options(FilterOptions options) {
        this.options = Objects.requireNonNull(options, "options");
        return this;
    }

    public KalmanFiltering method(FilterMethod method) {
        this.options = options.toBuilder().method(method).build();
        return this;
    }

    public KalmanFiltering auto() {
        return method(FilterMethod.AUTO);
    }

    public KalmanFiltering conventional() {
        return method(FilterMethod.CONVENTIONAL);
    }

    public KalmanFiltering univariate() {
        return method(FilterMethod.UNIVARIATE);
    }

    public KalmanFiltering collapsed() {
        return method(FilterMethod.COLLAPSED);
    }

    public KalmanFiltering chandrasekhar() {
        return method(FilterMethod.CHANDRASEKHAR);
    }

    public KalmanFiltering surfaces(FilterOptions.Surface surface, FilterOptions.Surface... more) {
        this.options = options.toBuilder().retainOnly(surface, more).build();
        return this;
    }

    public KalmanFiltering retain(FilterOptions.Surface surface, FilterOptions.Surface... more) {
        this.options = options.toBuilder().retain(surface, more).build();
        return this;
    }

    public KalmanFiltering drop(FilterOptions.Surface surface, FilterOptions.Surface... more) {
        this.options = options.toBuilder().drop(surface, more).build();
        return this;
    }

    public KalmanFiltering likelihoodOnly() {
        return surfaces(FilterOptions.Surface.LIKELIHOOD);
    }

    public KalmanFiltering prediction() {
        this.options = options.toBuilder()
            .drop(FilterOptions.Surface.FILTERED_STATE,
                FilterOptions.Surface.FILTERED_STATE_COVARIANCE,
                FilterOptions.Surface.KALMAN_GAIN,
                FilterOptions.Surface.LIKELIHOOD)
            .build();
        return this;
    }

    public KalmanFiltering compact() {
        this.options = options.toBuilder().retainNone().build();
        return this;
    }

    public KalmanFiltering storeAll() {
        this.options = options.toBuilder().retainAll().build();
        return this;
    }

    public FilterResult filter() {
        return filter(null);
    }

    public FilterResult filter(KalmanWorkspace workspace) {
        KalmanSSM resolvedModel = requireRealModel();
        InitialState resolvedInitialState = requireInitialState();
        if (workspace == null) {
            return KalmanEngine.filter(resolvedModel, resolvedInitialState, options);
        }
        FilterResult result = KalmanEngine.filterBorrowedUnsafe(
            resolvedModel,
            resolvedInitialState,
            workspace.filterWorkspace(),
            options).copy();
        workspace.filterWorkspace().releaseRetainedResults();
        return result;
    }

    @Override
    public FilterResult run(KalmanWorkspace workspace) {
        return filter(workspace);
    }

    private KalmanSSM requireRealModel() {
        if (model == null) {
            throw new IllegalArgumentException("model must be set before filtering");
        }
        if (model.complex()) {
            throw new IllegalArgumentException("model is complex; the high-level Kalman API currently supports real-valued models");
        }
        return model;
    }

    private InitialState requireInitialState() {
        if (initialState == null) {
            throw new IllegalArgumentException("initialState must be set before filtering");
        }
        return initialState;
    }
}