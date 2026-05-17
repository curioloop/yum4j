package com.curioloop.yum4j.ssm.kalman.arena;

public record ArenaSegment(int offset, int length) {

    public ArenaSegment {
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be non-negative");
        }
        if (length < 0) {
            throw new IllegalArgumentException("length must be non-negative");
        }
        if (offset + length < offset) {
            throw new IllegalArgumentException("segment end overflows int range");
        }
    }

    public int end() {
        return offset + length;
    }

    public boolean isEmpty() {
        return length == 0;
    }
}
