package com.curioloop.yum4j.tsa.prediction;

public record StatePredictionSurface(double[] state,
                                     int stateOffset,
                                     double[] covariance,
                                     int covarianceOffset) {
}