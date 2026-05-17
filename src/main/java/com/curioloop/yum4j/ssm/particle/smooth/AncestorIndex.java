package com.curioloop.yum4j.ssm.particle.smooth;

import java.util.Arrays;

/**
 * Compressed ancestor storage shared by {@link Full}, {@link Rolling},
 * and {@link Partial}.
 *
 * <p>Implements the two lossless ancestor transforms introduced by
 * Stage B of {@code particle-v2-mem}:
 *
 * <ul>
 *   <li><b>Identity-step skip</b> — non-resampled steps consume zero
 *       slots in the ancestor backing array; the implicit identity
 *       vector {@code (0, 1, ..., N-1)} is synthesised at read time
 *       when {@code stepToSlot[idx] == -1}. Drives the {@code resampledAt}
 *       fast-path on the smoother side.</li>
 *   <li><b>Char vs int narrowing</b> — when {@code N <= 65535} the
 *       backing is a {@code char[]} (16-bit unsigned), halving the
 *       per-element width vs {@code int[]}. The widening cast at read
 *       time is lossless on every JIT.</li>
 * </ul>
 *
 * <p>The pool exposes two flavours:
 * <ul>
 *   <li><b>Linear, fail-fast</b> ({@code circular = false}) — used by
 *       {@link Full} and {@link Partial}. {@code nextSlot} advances
 *       monotonically; on overflow {@link #recordResample(int, int[], int)}
 *       throws {@link IllegalStateException} naming both the
 *       {@code maxResampledSlots} cap and the underlying
 *       {@code maxResampleRate} parameter, so callers can decide
 *       whether to rebuild with a higher rate or switch to
 *       {@link Rolling}.</li>
 *   <li><b>Circular</b> ({@code circular = true}) — used by
 *       {@link Rolling}. {@code nextSlot} wraps modulo
 *       {@code maxResampledSlots}; correctness rests on the wrap
 *       invariant (the slot being overwritten must already be outside
 *       the rolling window). Asserted in dev builds via
 *       {@link #ASSERTS_ENABLED} and the {@code slotToStep} shadow.
 *       Production paths skip the check to keep the hot-path branch
 *       count bounded.</li>
 * </ul>
 *
 * <p>Package-private. The three history variants own one
 * {@code AncestorIndex} each and delegate every ancestor-side operation
 * (save / view / {@code resampledAt}) to it. {@code X} and {@code lw}
 * remain managed directly by the variants since their footprint and
 * indexing semantics differ structurally between {@link Full} (linear,
 * length {@code T}), {@link Rolling} (circular, length {@code L}), and
 * {@link Partial} (no {@code X} / {@code lw}).
 *
 * <p>Final and stateful: the class holds direct references to the
 * pre-allocated backing arrays. {@link #recordResample(int, int[], int)},
 * {@link #recordIdentity(int)}, and {@link #read(int, int[], int)} all
 * run with no heap allocation, preserving the {@code Engine.step}
 * 0.0 B/op gate inherited from {@code particle-v2-perf}.
 *
 */
final class AncestorIndex {

    /** Sentinel value in {@link #stepToSlot} indicating identity ancestors. */
    static final int IDENTITY_SLOT = -1;

    /** Default upper bound on the engine's observed resample rate. */
    static final double DEFAULT_MAX_RESAMPLE_RATE = 0.6;

    /**
     * {@code true} iff the JVM was started with assertions enabled
     * ({@code -ea}). Gates the wrap-invariant check and the allocation
     * of {@link #slotToStep}.
     */
    private static final boolean ASSERTS_ENABLED;
    static {
        boolean a = false;
        assert a = true;  // intentional side-effecting assert
        ASSERTS_ENABLED = a;
    }

    private final int N;
    /** Index space: {@code T} for linear pools, {@code L} for circular pools. */
    private final int indexSpace;
    private final boolean circular;
    private final int maxResampledSlots;
    private final boolean charBacked;

    // ── Compressed ancestor state ────────────────────────────────────
    /** Bitset over {@code [0, indexSpace)}: {@code true} iff the engine resampled at that slot. */
    private final long[] resampledBits;
    /** Bitset over {@code [0, indexSpace)}: {@code true} iff the slot has a pool entry assigned. */
    private final long[] hasAncestorSlot;
    /** Length {@code indexSpace}; {@link #IDENTITY_SLOT} or a pool index in {@code [0, maxResampledSlots)}. */
    private final int[] stepToSlot;
    /** Length {@code maxResampledSlots * N} when {@link #charBacked}, else {@code null}. */
    final char[] aChar;
    /** Length {@code maxResampledSlots * N} when {@code !charBacked}, else {@code null}. */
    final int[] aInt;

    private int nextSlot;
    /**
     * Dev-only: tracks the absolute step that last wrote each pool slot
     * (circular pools only). {@code null} when assertions are disabled
     * or {@link #circular} is false. Used solely to validate the wrap
     * invariant for {@link Rolling}.
     */
    private final int[] slotToStep;

    /**
     * Allocates a compressed ancestor pool.
     *
     * @param N                particle count (must be {@code > 0})
     * @param indexSpace       the variant's index space — {@code T} for
     *                         {@link Full}/{@link Partial}, {@code L}
     *                         for {@link Rolling}; must be {@code > 0}
     * @param maxResampleRate  upper bound on the observed resample rate
     *                         in {@code (0, 1]}
     * @param circular         {@code true} for circular wrap-on-overflow
     *                         (Rolling); {@code false} for linear
     *                         fail-fast (Full / Partial)
     * @throws IllegalArgumentException if any parameter is out of range
     *         or the pool would exceed {@code Integer.MAX_VALUE} elements
     */
    AncestorIndex(int N, int indexSpace, double maxResampleRate, boolean circular) {
        if (N <= 0) throw new IllegalArgumentException("N must be > 0: " + N);
        if (indexSpace <= 0) throw new IllegalArgumentException("indexSpace must be > 0: " + indexSpace);
        if (!(maxResampleRate > 0.0 && maxResampleRate <= 1.0)) {
            throw new IllegalArgumentException(
                "maxResampleRate must be in (0, 1]: " + maxResampleRate);
        }
        int slots = Math.max(1, (int) Math.ceil(maxResampleRate * (double) indexSpace));
        long ancPool = (long) slots * N;
        if (ancPool > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                "ancestor slot pool exceeds addressable limit: maxResampledSlots="
                    + slots + " N=" + N + " (needs " + ancPool + " elements)");
        }
        this.N = N;
        this.indexSpace = indexSpace;
        this.circular = circular;
        this.maxResampledSlots = slots;
        this.charBacked = N <= 0xFFFF;
        this.resampledBits = new long[(indexSpace + 63) >>> 6];
        this.hasAncestorSlot = new long[(indexSpace + 63) >>> 6];
        this.stepToSlot = new int[indexSpace];
        Arrays.fill(this.stepToSlot, IDENTITY_SLOT);
        if (this.charBacked) {
            this.aChar = new char[(int) ancPool];
            this.aInt = null;
        } else {
            this.aChar = null;
            this.aInt = new int[(int) ancPool];
        }
        this.nextSlot = 0;
        if (circular && ASSERTS_ENABLED) {
            this.slotToStep = new int[slots];
            Arrays.fill(this.slotToStep, Integer.MIN_VALUE);
        } else {
            this.slotToStep = null;
        }
    }

    /** Pre-sized number of resampled-step slots in the pool. */
    int maxResampledSlots() { return maxResampledSlots; }

    /** {@code true} iff the ancestor backing is {@code char[]} ({@code N <= 65535}). */
    boolean charBacked() { return charBacked; }

    /**
     * Records a resample at index {@code idx} (i.e. {@code t} for
     * {@link Full}/{@link Partial}, {@code t mod L} for {@link Rolling}),
     * copying ancestor indices from {@code src[0..N)} into the pool.
     *
     * <p>{@code absoluteStep} is the engine's absolute step number; only
     * read in dev builds for the wrap-invariant assertion on circular
     * pools, and ignored otherwise.
     *
     * @throws IllegalStateException for linear pools when
     *         {@code nextSlot >= maxResampledSlots}; the message names
     *         both {@code maxResampledSlots} (the cap) and
     *         {@code maxResampleRate} (the user-facing parameter)
     * @throws AssertionError for circular pools when the wrap invariant
     *         is violated (development builds only)
     */
    void recordResample(int idx, int[] src, int absoluteStep) {
        resampledBits[idx >>> 6] |= 1L << (idx & 63);
        int slot;
        if (circular) {
            slot = nextSlot % maxResampledSlots;
            assert validateAndRecordWrap(slot, absoluteStep)
                : wrapViolationMessage(slot, absoluteStep);
            nextSlot++;
        } else {
            if (nextSlot >= maxResampledSlots) {
                throw new IllegalStateException(
                    "ancestor slot pool exhausted at idx=" + idx
                        + " (slots used=" + nextSlot
                        + ", maxResampledSlots=" + maxResampledSlots + ");"
                        + " observed resample rate exceeds configured"
                        + " maxResampleRate; rebuild with a higher"
                        + " maxResampleRate or use Rolling");
            }
            slot = nextSlot++;
        }
        stepToSlot[idx] = slot;
        hasAncestorSlot[idx >>> 6] |= 1L << (idx & 63);
        int base = slot * N;
        if (charBacked) {
            for (int n = 0; n < N; n++) {
                int an = src[n];
                assert an >= 0 && an <= 0xFFFF
                    : "ancestor index out of char range at n=" + n + ": " + an;
                aChar[base + n] = (char) an;
            }
        } else {
            System.arraycopy(src, 0, aInt, base, N);
        }
    }

    /**
     * Records that index {@code idx} did not resample; the read path
     * synthesises identity ancestors {@code (0, 1, ..., N-1)} at view
     * time. No pool slot is consumed.
     */
    void recordIdentity(int idx) {
        long mask = ~(1L << (idx & 63));
        resampledBits[idx >>> 6] &= mask;
        hasAncestorSlot[idx >>> 6] &= mask;
        stepToSlot[idx] = IDENTITY_SLOT;
    }

    /** Returns whether index {@code idx} was a resample step. */
    boolean resampledAt(int idx) {
        return (resampledBits[idx >>> 6] & (1L << (idx & 63))) != 0;
    }

    /**
     * Reads the ancestor permutation at index {@code idx} into
     * {@code out[outOff .. outOff + N)}. Synthesises identity for
     * non-resampled steps; widens {@code char[]} backing to {@code int[]}
     * losslessly.
     */
    void read(int idx, int[] out, int outOff) {
        int slot = stepToSlot[idx];
        if (slot == IDENTITY_SLOT) {
            for (int n = 0; n < N; n++) {
                out[outOff + n] = n;
            }
        } else if (charBacked) {
            int base = slot * N;
            for (int n = 0; n < N; n++) {
                out[outOff + n] = aChar[base + n];
            }
        } else {
            System.arraycopy(aInt, slot * N, out, outOff, N);
        }
    }

    private boolean validateAndRecordWrap(int ancestorSlot, int absoluteStep) {
        if (slotToStep == null) return true;
        int prev = slotToStep[ancestorSlot];
        int oldestForT = Math.max(0, absoluteStep - indexSpace + 1);
        if (prev != Integer.MIN_VALUE && prev >= oldestForT) {
            return false;
        }
        slotToStep[ancestorSlot] = absoluteStep;
        return true;
    }

    private String wrapViolationMessage(int ancestorSlot, int absoluteStep) {
        int prev = slotToStep != null ? slotToStep[ancestorSlot] : -1;
        return "Rolling: ancestor pool wrap overwrote slot " + ancestorSlot
            + " (held by step " + prev + ") while saving step " + absoluteStep
            + "; previous holder is still within window [oldest="
            + Math.max(0, absoluteStep - indexSpace + 1)
            + ", newest=" + absoluteStep + "]. "
            + "Actual resample rate exceeds configured maxResampleRate. "
            + "maxResampledSlots=" + maxResampledSlots + " L=" + indexSpace;
    }
}
