package com.curioloop.yum4j.kalman.internal;

import com.curioloop.yum4j.kalman.model.StateSpaceModel;

import java.util.Objects;

public final class StateSpaceModelSupport {

    private StateSpaceModelSupport() {
    }

    public static void requireRealStorage(StateSpaceModel model, String name) {
        Objects.requireNonNull(model, name);
        if (model.complex()) {
            throw new IllegalArgumentException(name + " must provide real-valued storage");
        }
    }

    public static void requireComplexStorage(StateSpaceModel model, String name) {
        Objects.requireNonNull(model, name);
        if (!model.complex()) {
            throw new IllegalArgumentException(name + " must provide interleaved complex storage");
        }
    }

    public static boolean hasMissingObservations(StateSpaceModel model) {
        Objects.requireNonNull(model, "model");
        for (int t = 0; t < model.observationCount(); t++) {
            if (model.missingCount(t) > 0) {
                return true;
            }
        }
        return false;
    }
}