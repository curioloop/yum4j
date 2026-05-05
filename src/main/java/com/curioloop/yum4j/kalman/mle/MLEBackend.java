package com.curioloop.yum4j.kalman.mle;

import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.model.StateSpaceModel;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;

interface MLEBackend<F, S> {

    void requireSupported(StateSpaceModel templateModel);

    F filter(StateSpaceModel model, StateInitialization initialization, FilterSpec spec);

    F filterBorrowed(StateSpaceModel model, StateInitialization initialization, FilterSpec spec);

    S smooth(StateSpaceModel model, StateInitialization initialization, SmootherSpec spec);

    double logLikelihood(F filterResult);

    double logLikelihood(F filterResult, int burn);

    double[] logLikelihoodObs(F filterResult);

    int nobsDiffuse(F filterResult);
}