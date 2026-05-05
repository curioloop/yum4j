package com.curioloop.yum4j.kalman.mle;

import java.util.Locale;

public enum MLECovariance {
    OPG("opg"),
    OIM("oim"),
    APPROX("approx"),
    ROBUST("robust"),
    ROBUST_OIM("robust_oim"),
    ROBUST_APPROX("robust_approx");

    private final String id;

    MLECovariance(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }

    public static MLECovariance fromId(String id) {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("covariance type must not be blank");
        }
        String normalized = id.toLowerCase(Locale.ROOT);
        for (MLECovariance value : values()) {
            if (value.id.equals(normalized)) {
                return value;
            }
        }
        throw new IllegalArgumentException("Unsupported covariance type: " + id);
    }
}