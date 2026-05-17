package com.curioloop.yum4j.ssm.kalman.arena;

public final class IntArena {

    private static final int[] EMPTY = new int[0];

    private int[] backing = EMPTY;
    private int peakLength;
    private long growthCount;

    public int[] ensureCapacity(int requiredLength) {
        if (requiredLength < 0) {
            throw new IllegalArgumentException("requiredLength must be non-negative");
        }
        if (backing.length < requiredLength) {
            backing = new int[requiredLength];
            peakLength = Math.max(peakLength, requiredLength);
            growthCount++;
        }
        return backing;
    }

    public int[] borrow(int requiredLength) {
        return ensureCapacity(requiredLength);
    }

    public IntArenaSlice bind(ArenaSegment segment) {
        int[] current = ensureCapacity(segment.end());
        return new IntArenaSlice(current, segment.offset(), segment.length());
    }

    public IntArenaSlice bind(int offset, int length) {
        return bind(new ArenaSegment(offset, length));
    }

    public int[] backing() {
        return backing;
    }

    public long retainedIntCount() {
        return backing.length;
    }

    public long retainedByteCount() {
        return retainedIntCount() * Integer.BYTES;
    }

    public long peakIntCount() {
        return peakLength;
    }

    public long peakByteCount() {
        return peakIntCount() * Integer.BYTES;
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
