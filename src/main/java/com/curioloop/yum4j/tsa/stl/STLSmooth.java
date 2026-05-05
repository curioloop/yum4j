package com.curioloop.yum4j.tsa.stl;

/**
 * LOESS smoothing controls used by STL.
 *
 * <p>{@code length} is the odd neighborhood size {@code q}; {@code degree} selects local-constant
 * ({@code 0}) or local-linear ({@code 1}) regression; {@code jump} evaluates every {@code jump}-th
 * point and fills skipped points by linear interpolation.</p>
 */
public value record STLSmooth(int length, int degree, int jump) {

    public STLSmooth {
        if (length < 3 || (length & 1) == 0) {
            throw new IllegalArgumentException("length must be an odd number greater than or equal to 3");
        }
        if (degree < 0 || degree > 1) {
            throw new IllegalArgumentException("degree must be 0 or 1");
        }
        if (jump < 1) {
            throw new IllegalArgumentException("jump must be positive");
        }
    }

    public static STLSmooth of(int length, int degree, int jump) {
        return new STLSmooth(length, degree, jump);
    }
}
