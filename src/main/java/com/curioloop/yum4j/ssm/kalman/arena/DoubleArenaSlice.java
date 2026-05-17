package com.curioloop.yum4j.ssm.kalman.arena;

import java.util.Objects;

public record DoubleArenaSlice(double[] backing, int offset, int length) {

    public DoubleArenaSlice {
        Objects.requireNonNull(backing, "backing");
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be non-negative");
        }
        if (length < 0) {
            throw new IllegalArgumentException("length must be non-negative");
        }
        if (offset + length < offset || offset + length > backing.length) {
            throw new IllegalArgumentException("slice exceeds backing length");
        }
    }

    public int end() {
        return offset + length;
    }

    public boolean isEmpty() {
        return length == 0;
    }
}
