package com.curioloop.yum4j.tsa.prediction;

public final class PredictionToolkit {

    private PredictionToolkit() {
    }

    public static PredictionInformationSet resolveInformationSet(PredictionInformationSet informationSet) {
        return informationSet == null ? PredictionInformationSet.PREDICTED : informationSet;
    }

    public static PredictionKind predictionKind(int start,
                                                int end,
                                                int dynamicStart,
                                                int nobs,
                                                boolean forecastOnly) {
        validatePredictionBounds(start, end, dynamicStart);
        if (nobs < 0) {
            throw new IllegalArgumentException("nobs must be non-negative");
        }
        if (forecastOnly) {
            return PredictionKind.FORECAST;
        }
        if (start >= nobs) {
            return PredictionKind.OUT_OF_SAMPLE;
        }
        if (end >= nobs) {
            return PredictionKind.MIXED;
        }
        return dynamicStart >= 0 ? PredictionKind.DYNAMIC_IN_SAMPLE : PredictionKind.IN_SAMPLE;
    }

    public static void validatePredictionBounds(int start, int end, int dynamicStart) {
        if (start < 0) {
            throw new IllegalArgumentException("start must be non-negative");
        }
        if (end < start) {
            throw new IllegalArgumentException("end must be greater than or equal to start");
        }
        if (dynamicStart < -1) {
            throw new IllegalArgumentException("dynamicStart must be non-negative or -1");
        }
    }

    public static int effectiveDynamicStart(int start, int end, int dynamicStart) {
        validatePredictionBounds(start, end, dynamicStart);
        return dynamicStart >= 0 && dynamicStart <= end ? Math.max(start, dynamicStart) : -1;
    }

    public static int metadataDynamicStart(int start, int end, int dynamicStart) {
        validatePredictionBounds(start, end, dynamicStart);
        return dynamicStart >= start && dynamicStart <= end ? dynamicStart : -1;
    }
}