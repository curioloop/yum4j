package com.curioloop.yum4j.ssm.kalman.arena;

public final class DoubleArena {

    private static final double[] EMPTY = new double[0];

    private double[] backing = EMPTY;
    private int peakLength;
    private long growthCount;

    public double[] ensureCapacity(int requiredLength) {
        if (requiredLength < 0) {
            throw new IllegalArgumentException("requiredLength must be non-negative");
        }
        if (backing.length < requiredLength) {
            backing = new double[requiredLength];
            peakLength = Math.max(peakLength, requiredLength);
            growthCount++;
        }
        return backing;
    }

    public double[] borrow(int requiredLength) {
        return ensureCapacity(requiredLength);
    }

    public DoubleArenaSlice bind(ArenaSegment segment) {
        double[] current = ensureCapacity(segment.end());
        return new DoubleArenaSlice(current, segment.offset(), segment.length());
    }

    public DoubleArenaSlice bind(int offset, int length) {
        return bind(new ArenaSegment(offset, length));
    }

    public double[] backing() {
        return backing;
    }

    public long retainedDoubleCount() {
        return backing.length;
    }

    public long retainedByteCount() {
        return retainedDoubleCount() * Double.BYTES;
    }

    public long peakDoubleCount() {
        return peakLength;
    }

    public long peakByteCount() {
        return peakDoubleCount() * Double.BYTES;
    }

    public long growthCount() {
        return growthCount;
    }

    public void resetPeak() {
        peakLength = backing.length;
        growthCount = 0L;
    }

    public boolean isEmpty() {
        return backing.length == 0;
    }

    public void release() {
        backing = EMPTY;
    }
}
