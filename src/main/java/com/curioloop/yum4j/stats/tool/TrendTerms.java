package com.curioloop.yum4j.stats.tool;

import java.util.Arrays;

/**
 * Statsmodels-style deterministic time-trend terms.
 */
public enum TrendTerms {

    NONE("n"),
    CONSTANT("c", 0),
    LINEAR("t", 1),
    CONSTANT_TREND("ct", 0, 1),
    CONSTANT_TREND_QUADRATIC("ctt", 0, 1, 2);

    private final String code;
    private final int[] powers;

    TrendTerms(String code, int... powers) {
        this.code = code;
        this.powers = normalizePowers(powers);
    }

    public String code() {
        return code;
    }

    public int term() {
        switch (this) {
            case NONE:
                return 0;
            case CONSTANT:
                return 1;
            case CONSTANT_TREND:
                return 2;
            case CONSTANT_TREND_QUADRATIC:
                return 3;
            default:
                throw new IllegalArgumentException("trend is not supported by statsmodels stattools regression: " + code);
        }
    }

    public int[] powers() {
        return powers.clone();
    }

    public static TrendTerms fromString(String trend) {
        if (trend == null) {
            return NONE;
        }
        for (TrendTerms terms : values()) {
            if (terms.code.equals(trend)) {
                return terms;
            }
        }
        throw new IllegalArgumentException("Unsupported trend: " + trend);
    }

    public static int[] normalizePowers(int[] powers) {
        if (powers == null || powers.length == 0) {
            return new int[0];
        }
        int[] copy = powers.clone();
        Arrays.sort(copy);
        int uniqueCount = 0;
        for (int power : copy) {
            if (power < 0) {
                throw new IllegalArgumentException("trend powers must be non-negative");
            }
            if (uniqueCount == 0 || copy[uniqueCount - 1] != power) {
                copy[uniqueCount++] = power;
            }
        }
        return Arrays.copyOf(copy, uniqueCount);
    }

    public static void validateOffset(int offset) {
        if (offset < 0) {
            throw new IllegalArgumentException("trendOffset must be non-negative");
        }
    }
}