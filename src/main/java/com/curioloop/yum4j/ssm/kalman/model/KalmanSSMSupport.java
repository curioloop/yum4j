package com.curioloop.yum4j.ssm.kalman.model;

import java.util.Objects;

public final class KalmanSSMSupport {

    private KalmanSSMSupport() {
    }

    public static void requireRealStorage(KalmanSSM model, String name) {
        Objects.requireNonNull(model, name);
        if (model.complex()) {
            throw new IllegalArgumentException(name + " must provide real-valued storage");
        }
    }

    public static void requireComplexStorage(KalmanSSM model, String name) {
        Objects.requireNonNull(model, name);
        if (!model.complex()) {
            throw new IllegalArgumentException(name + " must provide interleaved complex storage");
        }
    }

    public static boolean hasMissingObservations(KalmanSSM model) {
        Objects.requireNonNull(model, "model");
        for (int t = 0; t < model.observationCount(); t++) {
            if (model.missingCount(t) > 0) {
                return true;
            }
        }
        return false;
    }

    public static boolean hasNonDiagonalObservationCovariance(KalmanSSM model) {
        Objects.requireNonNull(model, "model");
        int kEndog = model.observationDimension();
        if (kEndog <= 1) {
            return false;
        }
        double[] obsCov = model.obsCovData();
        int obsCovLd = model.obsCovLeadingDimension();
        int scalarWidth = model.complex() ? 2 : 1;
        for (int t = 0; t < model.observationCount(); t++) {
            int offset = model.obsCovOffset(t);
            for (int row = 0; row < kEndog; row++) {
                int rowOffset = offset + row * scalarWidth * obsCovLd;
                for (int col = 0; col < kEndog; col++) {
                    if (row == col) {
                        continue;
                    }
                    int cell = rowOffset + col * scalarWidth;
                    if (obsCov[cell] != 0.0 || (scalarWidth == 2 && obsCov[cell + 1] != 0.0)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    public static void requireDiagonalObservationCovariance(KalmanSSM model, String context) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(context, "context");
        int kEndog = model.observationDimension();
        if (kEndog <= 1) {
            return;
        }
        double[] obsCov = model.obsCovData();
        int obsCovLd = model.obsCovLeadingDimension();
        int scalarWidth = model.complex() ? 2 : 1;
        for (int t = 0; t < model.observationCount(); t++) {
            int offset = model.obsCovOffset(t);
            for (int row = 0; row < kEndog; row++) {
                int rowOffset = offset + row * scalarWidth * obsCovLd;
                for (int col = 0; col < kEndog; col++) {
                    if (row == col) {
                        continue;
                    }
                    int cell = rowOffset + col * scalarWidth;
                    if (obsCov[cell] != 0.0 || (scalarWidth == 2 && obsCov[cell + 1] != 0.0)) {
                        throw new UnsupportedOperationException(context
                            + " requires diagonal observation covariance; found nonzero H["
                            + row + "," + col + "] at t=" + t);
                    }
                }
            }
        }
    }
}