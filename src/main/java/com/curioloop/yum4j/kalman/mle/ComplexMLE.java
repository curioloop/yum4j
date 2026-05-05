package com.curioloop.yum4j.kalman.mle;

import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.kalman.filter.ZKalmanFilter;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.internal.StateSpaceModelSupport;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
import com.curioloop.yum4j.kalman.smooth.ZKalmanSmoother;
import com.curioloop.yum4j.kalman.smooth.ZSmootherResult;

import java.util.Arrays;

public abstract class ComplexMLE extends MLEModel<ZFilterResult, ZSmootherResult> {

    private final ZKalmanFilter.Pool filterPool = new ZKalmanFilter.Pool();
    private final ZKalmanSmoother.Pool smootherPool = new ZKalmanSmoother.Pool();

    private FilterSpec reservedFilterSpec;
    private int reservedFilterKEndog = -1;
    private int reservedFilterKStates = -1;
    private int reservedFilterKPosdef = -1;
    private int reservedFilterNobs = -1;
    private boolean reservedFilterHasMissing;

    private SmootherSpec reservedSmootherSpec;
    private int reservedSmootherKEndog = -1;
    private int reservedSmootherKStates = -1;
    private int reservedSmootherKPosdef = -1;
    private int reservedSmootherNobs = -1;

    protected ComplexMLE(StateSpaceModel templateModel,
                         double[] startParams,
                         String... parameterNames) {
        super(templateModel, startParams, parameterNames);
    }

    @Override
    public final void requireSupported(StateSpaceModel templateModel) {
        StateSpaceModelSupport.requireComplexStorage(templateModel, "templateModel");
    }

    @Override
    public final ZFilterResult filter(StateSpaceModel model,
                                      StateInitialization initialization,
                                      FilterSpec spec) {
        return filterBorrowed(model, initialization, spec).copy();
    }

    @Override
    public final ZFilterResult filterBorrowed(StateSpaceModel model,
                                              StateInitialization initialization,
                                              FilterSpec spec) {
        FilterSpec resolvedSpec = spec == null ? FilterSpec.defaults() : spec;
        ensureFilterReservation(model, resolvedSpec);
        return ZKalmanFilter.filter(model, initialization, filterPool, resolvedSpec);
    }

    @Override
    public final ZSmootherResult smooth(StateSpaceModel model,
                                        StateInitialization initialization,
                                        SmootherSpec spec) {
        SmootherSpec resolvedSpec = spec == null ? SmootherSpec.conventional() : spec;
        FilterSpec requiredFilterSpec = resolvedSpec.requiredFilterSpec();
        ensureFilterReservation(model, requiredFilterSpec);
        ensureSmootherReservation(model, resolvedSpec);
        ZFilterResult filterResult = ZKalmanFilter.filter(model, initialization, filterPool, requiredFilterSpec);
        return ZKalmanSmoother.smooth(model, filterResult, smootherPool, resolvedSpec).copy();
    }

    @Override
    public final double logLikelihood(ZFilterResult filterResult) {
        return filterResult.logLikelihood();
    }

    @Override
    public final double logLikelihood(ZFilterResult filterResult, int burn) {
        int start = Math.max(0, burn);
        if (start == 0) {
            return filterResult.logLikelihood();
        }
        double sum = 0.0;
        for (int t = Math.min(start, filterResult.nobs); t < filterResult.nobs; t++) {
            sum += filterResult.logLikelihoodObs[filterResult.logLikelihoodObsOffset(t)];
        }
        return sum;
    }

    @Override
    public final double[] logLikelihoodObs(ZFilterResult filterResult) {
        int base = filterResult.logLikelihoodObsBase();
        return Arrays.copyOfRange(filterResult.logLikelihoodObs, base, base + filterResult.nobs);
    }

    @Override
    public final int nobsDiffuse(ZFilterResult filterResult) {
        return filterResult.nobsDiffuse;
    }

    private void ensureFilterReservation(StateSpaceModel model, FilterSpec spec) {
        boolean hasMissing = StateSpaceModelSupport.hasMissingObservations(model);
        if (spec == reservedFilterSpec
                && reservedFilterKEndog == model.observationDimension()
                && reservedFilterKStates == model.stateCount()
                && reservedFilterKPosdef == model.stateDisturbanceCount()
                && reservedFilterNobs == model.observationCount()
                && reservedFilterHasMissing == hasMissing) {
            return;
        }
        filterPool.reserve(model, spec);
        reservedFilterSpec = spec;
        reservedFilterKEndog = model.observationDimension();
        reservedFilterKStates = model.stateCount();
        reservedFilterKPosdef = model.stateDisturbanceCount();
        reservedFilterNobs = model.observationCount();
        reservedFilterHasMissing = hasMissing;
    }

    private void ensureSmootherReservation(StateSpaceModel model, SmootherSpec spec) {
        if (spec == reservedSmootherSpec
                && reservedSmootherKEndog == model.observationDimension()
                && reservedSmootherKStates == model.stateCount()
                && reservedSmootherKPosdef == model.stateDisturbanceCount()
                && reservedSmootherNobs == model.observationCount()) {
            return;
        }
        smootherPool.reserve(model, spec);
        reservedSmootherSpec = spec;
        reservedSmootherKEndog = model.observationDimension();
        reservedSmootherKStates = model.stateCount();
        reservedSmootherKPosdef = model.stateDisturbanceCount();
        reservedSmootherNobs = model.observationCount();
    }
}
