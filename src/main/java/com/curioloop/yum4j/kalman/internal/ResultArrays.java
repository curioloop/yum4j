package com.curioloop.yum4j.kalman.internal;

/**
 * Shared array helpers for Kalman result surfaces.
 */
public final class ResultArrays {

    private static final double[] EMPTY = new double[0];

    private ResultArrays() {
    }

    public static double[] emptyArray() {
        return EMPTY;
    }

    public static double[] allocate(int length) {
        return length == 0 ? EMPTY : new double[length];
    }

    public static void copySurface(double[] source, double[] target) {
        copySurface(source, 0, source == null ? 0 : source.length, target, 0);
    }

    public static void copySurface(double[] source, int sourceOffset, int length, double[] target, int targetOffset) {
        if (source == null || target == null || source == EMPTY || target == EMPTY || length == 0) {
            return;
        }
        System.arraycopy(source, sourceOffset, target, targetOffset, length);
    }
}