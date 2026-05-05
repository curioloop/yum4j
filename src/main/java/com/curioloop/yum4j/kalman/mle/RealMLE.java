package com.curioloop.yum4j.kalman.mle;

import com.curioloop.yum4j.kalman.filter.FilterResult;
import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.filter.KalmanFilter;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.internal.StateSpaceModelSupport;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.kalman.smooth.KalmanSmoother;
import com.curioloop.yum4j.kalman.smooth.SmootherResult;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;

import java.util.Arrays;

public abstract class RealMLE extends MLEModel<FilterResult, SmootherResult> {

    private final KalmanFilter.Pool filterPool = new KalmanFilter.Pool();
    private final KalmanSmoother.Pool smootherPool = new KalmanSmoother.Pool();

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

    protected RealMLE(StateSpaceModel templateModel,
                      double[] startParams,
                      String... parameterNames) {
        super(templateModel, startParams, parameterNames);
    }

    @Override
    public final void requireSupported(StateSpaceModel templateModel) {
        StateSpaceModelSupport.requireRealStorage(templateModel, "templateModel");
    }

    @Override
    public final FilterResult filter(StateSpaceModel model,
                                     StateInitialization initialization,
                                     FilterSpec spec) {
        return filterBorrowed(model, initialization, spec).copy();
    }

    @Override
    public final FilterResult filterBorrowed(StateSpaceModel model,
                                             StateInitialization initialization,
                                             FilterSpec spec) {
        FilterSpec resolvedSpec = spec == null ? FilterSpec.defaults() : spec;
        ensureFilterReservation(model, resolvedSpec);
        return KalmanFilter.filter(model, initialization, filterPool, resolvedSpec);
    }

    @Override
    public final SmootherResult smooth(StateSpaceModel model,
                                       StateInitialization initialization,
                                       SmootherSpec spec) {
        SmootherSpec resolvedSpec = spec == null ? SmootherSpec.conventional() : spec;
        FilterSpec requiredFilterSpec = resolvedSpec.requiredFilterSpec();
        ensureFilterReservation(model, requiredFilterSpec);
        ensureSmootherReservation(model, resolvedSpec);
        FilterResult filterResult = KalmanFilter.filter(model, initialization, filterPool, requiredFilterSpec);
        return KalmanSmoother.smooth(model, filterResult, smootherPool, resolvedSpec).copy();
    }

    @Override
    public final double logLikelihood(FilterResult filterResult) {
        return filterResult.logLikelihood();
    }

    @Override
    public final double logLikelihood(FilterResult filterResult, int burn) {
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
    public final double[] logLikelihoodObs(FilterResult filterResult) {
        int base = filterResult.logLikelihoodObsBase();
        return Arrays.copyOfRange(filterResult.logLikelihoodObs, base, base + filterResult.nobs);
    }

    @Override
    public final int nobsDiffuse(FilterResult filterResult) {
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
