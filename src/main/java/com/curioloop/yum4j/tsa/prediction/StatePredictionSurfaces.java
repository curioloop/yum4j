package com.curioloop.yum4j.tsa.prediction;

import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherResult;

public final class StatePredictionSurfaces {

    private StatePredictionSurfaces() {
    }

    public static StatePredictionSurface resolve(FilterResult filterResult,
                                                 SmootherResult smootherResult,
                                                 PredictionInformationSet informationSet,
                                                 int time) {
        return switch (PredictionToolkit.resolveInformationSet(informationSet)) {
            case PREDICTED -> predicted(filterResult, time);
            case FILTERED -> filtered(filterResult, time);
            case SMOOTHED -> smoothed(smootherResult, time);
        };
    }

    private static StatePredictionSurface predicted(FilterResult filterResult, int time) {
        if (filterResult.predictedStateLength() == 0) {
            throw new IllegalArgumentException("predicted-state predictions are not available when predicted states were not retained");
        }
        return new StatePredictionSurface(filterResult.predictedState,
            filterResult.predictedStateOffset(time),
            filterResult.predictedStateCov,
            filterResult.predictedStateCovLength() == 0 ? -1 : filterResult.predictedStateCovOffset(time));
    }

    private static StatePredictionSurface filtered(FilterResult filterResult, int time) {
        if (filterResult.filteredStateLength() == 0) {
            throw new IllegalArgumentException("filtered predictions are not available when filtered states were not retained");
        }
        return new StatePredictionSurface(filterResult.filteredState,
            filterResult.filteredStateOffset(time),
            filterResult.filteredStateCov,
            filterResult.filteredStateCovLength() == 0 ? -1 : filterResult.filteredStateCovOffset(time));
    }

    private static StatePredictionSurface smoothed(SmootherResult smootherResult, int time) {
        if (smootherResult == null || smootherResult.smoothedStateLength() == 0) {
            throw new IllegalArgumentException("smoothed predictions are not available when smoothed states were not retained");
        }
        return new StatePredictionSurface(smootherResult.smoothedState,
            smootherResult.smoothedStateOffset(time),
            smootherResult.smoothedStateCov,
            smootherResult.smoothedStateCovLength() == 0 ? -1 : smootherResult.smoothedStateCovOffset(time));
    }
}