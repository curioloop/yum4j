package com.curioloop.yum4j.tsa.prediction;

import java.util.Objects;

public record PredictionMetadata(int start,
                                 int end,
                                 int dynamicStart,
                                 PredictionKind kind,
                                 PredictionInformationSet informationSet,
                                 boolean signalOnly) {

    public PredictionMetadata {
        PredictionToolkit.validatePredictionBounds(start, end, dynamicStart);
        dynamicStart = PredictionToolkit.metadataDynamicStart(start, end, dynamicStart);
        kind = Objects.requireNonNull(kind, "kind");
        informationSet = PredictionToolkit.resolveInformationSet(informationSet);
    }

    public boolean dynamic() {
        return dynamicStart >= 0;
    }

    public int length() {
        return end - start + 1;
    }
}