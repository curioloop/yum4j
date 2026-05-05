package com.curioloop.yum4j.fft;

import java.util.Objects;

public final class FftWorkspace {

    /**
     * Describes how much scratch memory an FFT plan needs.
     *
     * <p>Instances are composed with {@link #plus} (both regions needed
     * simultaneously) and {@link #max} (sequential phases sharing space).
     * Create via the static factories {@link #empty()} and {@link #doubles}.
     */
    public value record Requirement(int totalDoubles, int intSlots) {

        private static final Requirement EMPTY = new Requirement(0, 0);

        /** Compact constructor — validates invariants. */
        public Requirement {
            if (totalDoubles < 0) {
                throw new IllegalArgumentException("totalDoubles must be >= 0");
            }
            if (intSlots < 0) {
                throw new IllegalArgumentException("intSlots must be >= 0");
            }
        }

        /** Returns a requirement with zero doubles and zero int slots. */
        public static Requirement empty() {
            return EMPTY;
        }

        /**
         * Returns a requirement whose {@code totalDoubles} equals the sum of
         * all arguments.  Backward-compatible varargs: {@code doubles(a, b)}
         * produces {@code totalDoubles = a + b}.
         */
        public static Requirement doubles(int... sizes) {
            int total = 0;
            for (int size : sizes) {
                if (size < 0) {
                    throw new IllegalArgumentException("workspace sizes must be >= 0");
                }
                total = Math.addExact(total, size);
            }
            return new Requirement(total, 0);
        }

        /** Additive composition: both regions needed simultaneously. */
        public Requirement plus(Requirement other) {
            return new Requirement(
                Math.addExact(totalDoubles, other.totalDoubles),
                Math.addExact(intSlots, other.intSlots)
            );
        }

        /** Maximum composition: sequential phases sharing space. */
        public Requirement max(Requirement other) {
            return new Requirement(
                Math.max(totalDoubles, other.totalDoubles),
                Math.max(intSlots, other.intSlots)
            );
        }

        /** Returns a copy with the given int-slot count. */
        public Requirement withIntSlots(int slots) {
            return new Requirement(totalDoubles, slots);
        }

        /** Allocate a workspace satisfying this requirement. */
        public FftWorkspace allocate() {
            return new FftWorkspace(this);
        }
    }

    // ---- workspace state ----

    private final double[] doubleArena;
    private final int[] intLane;
    private int stackPointer;

    public FftWorkspace(Requirement requirement) {
        Objects.requireNonNull(requirement, "requirement");
        this.doubleArena = new double[requirement.totalDoubles()];
        this.intLane = new int[requirement.intSlots()];
        this.stackPointer = 0;
    }

    /** Reset the stack pointer to zero. */
    public void reset() {
        stackPointer = 0;
    }

    /** Save the current stack pointer position. */
    int mark() {
        return stackPointer;
    }

    /** Restore the stack pointer to a previously saved mark. */
    void rewind(int mark) {
        if (mark < 0 || mark > stackPointer) {
            throw new IllegalArgumentException("invalid workspace mark: " + mark
                + " (current pointer: " + stackPointer + ")");
        }
        stackPointer = mark;
    }

    /** Return the backing double array. */
    public double[] doubles() {
        return doubleArena;
    }

    /**
     * Allocate n doubles from the workspace.
     * Returns the offset into doubles() where the allocation starts.
     */
    int alloc(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("workspace alloc size must be >= 0");
        }
        int offset = stackPointer;
        int newPointer = offset + n;
        if (newPointer > doubleArena.length) {
            throw new IllegalArgumentException("workspace capacity exceeded: requested "
                + n + " doubles at offset " + offset + ", capacity " + doubleArena.length);
        }
        stackPointer = newPointer;
        return offset;
    }

    /** Return the int scratch array, validating minimum length. */
    int[] ints(int requiredLength) {
        if (requiredLength < 0) {
            throw new IllegalArgumentException("workspace length must be >= 0");
        }
        if (intLane.length < requiredLength) {
            throw new IllegalArgumentException("workspace int lane is too small: required "
                + requiredLength + ", actual " + intLane.length);
        }
        return intLane;
    }
}
