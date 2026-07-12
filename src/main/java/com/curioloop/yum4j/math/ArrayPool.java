/*
 * Copyright (c) 2026 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.math;

import java.lang.invoke.*;
import java.lang.ref.SoftReference;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;

/**
 * Power-of-two bucketed array pool.
 *
 * <h2>Virtual-thread friendly</h2>
 * Slot dispatch is pure CAS over a shared {@link SoftReference} array.
 * No thread-locals, no carrier-thread pinning, no per-task setup —
 * works equally well on platform threads, ForkJoin workers, and
 * virtual threads.
 *
 * <h2>Bucketing</h2>
 * Powers of two from 2^13 (8K) up to 2^22 (4M). Below 8K elements the
 * pool dispatch (slot scan + CAS) costs more than the JVM TLAB allocator
 * saves, so small caps fall through to a plain uninit-alloc; above 4M is
 * rare enough that a pool miss isn't worth the per-bucket footprint.
 *
 * <h2>Memory footprint</h2>
 * Each pool retains at most {@code CAP_PER_BUCKET} (= 8) idle arrays per
 * bucket; extra {@link #release}d arrays are dropped when a bucket is full.
 * The retained (idle) memory is therefore bounded by:
 * <pre>{@code
 *   max = CAP_PER_BUCKET * Σ(s=13..22) 2^s * sizeof(element)
 *       = 8 * (2^23 - 2^13) * sizeof(element)
 *       = 67,043,328 elements * sizeof(element)
 * }</pre>
 * Per element type (one pool holds a single type):
 * <ul>
 *   <li>{@link OfDouble} / {@link OfLong} (8 B): ~511.5 MiB</li>
 *   <li>{@link OfFloat} / {@link OfInt} (4 B): ~255.75 MiB</li>
 *   <li>{@link OfShort} (2 B): ~127.9 MiB</li>
 *   <li>{@link OfByte} (1 B): ~63.9 MiB</li>
 * </ul>
 * Caveats: (1) this is an <em>upper bound on idle memory</em>, not a
 * reservation — arrays are held via {@link SoftReference} and the GC may
 * reclaim them under pressure; (2) arrays currently borrowed (between
 * {@link #acquire(int)} and {@link #release}) are owned by the caller and
 * counted separately; (3) the bound is per pool instance and per type. In
 * practice a workload only touches a few buckets, so steady-state usage is
 * far below the cap.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * static final ArrayPool.OfDouble POOL = new ArrayPool.OfDouble();  // owned by the subsystem
 * double[] a = POOL.acquire(n);   // length >= n, contents undefined
 * try {
 *     // ... use a[0 .. n) as scratch ...
 * } finally {
 *     POOL.release(a);            // hand back for reuse
 * }
 * }</pre>
 *
 * {@link #acquire(int)} always returns a usable array whose length is
 * {@code >= minLength} (up to roughly 2&times; larger, rounded to the
 * enclosing bucket). Requests outside the pooled range are served by a
 * plain exact-size allocation and are silently dropped on {@link #release}.
 *
 * <h2>Contents contract</h2>
 * {@link #acquire(int)} returns <em>dirty</em> memory: a recycled array
 * still holds the previous borrower's data, and a freshly allocated one is
 * uninitialized. Callers that need a zeroed region must either overwrite
 * every element they read or pass {@code clear = true} to
 * {@link #acquire(int, boolean)}.
 *
 * <h2>Ownership</h2>
 * Instantiate a concrete subclass ({@link OfDouble}, {@link OfInt}, &hellip;) and
 * keep it as a field owned by the subsystem that needs scratch buffers. A single
 * instance is safe to share across threads (slot dispatch is pure CAS), so one
 * pool per subsystem is the intended granularity — this keeps unrelated subsystems
 * from contending for the same bucket slots.
 *
 * @param <T> the pooled array type, e.g. {@code double[]}
 */
public abstract class ArrayPool<T> {

    private static final int MIN_SHIFT       = 13;  // 8 * 1024
    private static final int MAX_SHIFT       = 22;  // 4 * 1024 * 1024
    private static final int N_BUCKETS       = MAX_SHIFT - MIN_SHIFT + 1;
    private static final int CAP_PER_BUCKET  = 8;
    private static final int N_SLOTS         = N_BUCKETS * CAP_PER_BUCKET;

    private static final VarHandle SLOT_VH =
            MethodHandles.arrayElementVarHandle(SoftReference[].class);

    @SuppressWarnings("unchecked")
    final SoftReference<T>[] slots = new SoftReference[N_SLOTS];

    private static int bucketIndexForRequest(int n) {
        if (n < (1 << MIN_SHIFT) || n > (1 << MAX_SHIFT)) return -1;
        int k = 32 - Integer.numberOfLeadingZeros(n - 1);
        return k - MIN_SHIFT;
    }

    private static int bucketIndexForLength(int len) {
        if (Integer.bitCount(len) != 1) return -1;
        if (len < (1 << MIN_SHIFT) || len > (1 << MAX_SHIFT)) return -1;
        return Integer.numberOfTrailingZeros(len) - MIN_SHIFT;
    }

    private static int slotBase(int bucketIdx) { return bucketIdx * CAP_PER_BUCKET; }

    private static int bucketSize(int idx) { return 1 << (MIN_SHIFT + idx); }

    /** Smallest pow-2 &ge; {@code n} within the bucket range, or {@code -1}
     *  if {@code n} is outside the pooled range. */
    public static int bucketSizeForRequest(int n) {
        int idx = bucketIndexForRequest(n);
        return idx < 0 ? -1 : bucketSize(idx);
    }

    private static <T> T tryAcquire(SoftReference<T>[] slots, int idx) {
        int base = slotBase(idx);
        for (int s = 0; s < CAP_PER_BUCKET; s++) {
            int slot = base + s;
            SoftReference<T> ref = (SoftReference<T>) SLOT_VH.getVolatile(slots, slot);
            if (ref == null) continue;
            T arr = ref.get();
            if (arr == null) {
                SLOT_VH.compareAndSet(slots, slot, ref, null);
                continue;
            }
            if (SLOT_VH.compareAndSet(slots, slot, ref, null)) return arr;
        }
        return null;
    }

    private static <T> void tryRelease(SoftReference<T>[] slots, int idx, T arr) {
        int base = slotBase(idx);
        SoftReference<T> wrap = new SoftReference<>(arr);
        for (int s = 0; s < CAP_PER_BUCKET; s++) {
            int slot = base + s;
            if (SLOT_VH.getVolatile(slots, slot) == null
                    && SLOT_VH.compareAndSet(slots, slot, (SoftReference<T>) null, wrap)) {
                return;
            }
        }
    }

    // ------------------------------------------------------------------
    // Per-type primitives supplied by concrete subclasses.
    // ------------------------------------------------------------------

    /** Allocate an uninitialized array of exactly {@code n} elements. */
    protected abstract T allocate(int n);

    /** Number of elements in {@code array}. */
    protected abstract int lengthOf(T array);

    /** Zero the first {@code len} elements of {@code array}. */
    protected abstract void clear(T array, int len);

    // ------------------------------------------------------------------
    // Public pool API.
    // ------------------------------------------------------------------

    /**
     * Borrow an array whose length is at least {@code minLength}, with
     * <em>undefined</em> contents. Shorthand for {@code acquire(minLength, false)}.
     *
     * @param minLength minimum required capacity, must be {@code >= 0}
     * @return an array with {@code length >= minLength}, never {@code null}
     * @throws IllegalArgumentException if {@code minLength < 0}
     */
    public final T acquire(int minLength) {
        return acquire(minLength, false);
    }

    /**
     * Borrow an array whose length is at least {@code minLength}.
     * <p>
     * Within the pooled range this returns a recycled array when one is
     * available, otherwise a fresh bucket-sized (power-of-two) array that
     * can later be handed back via {@link #release}. Requests outside the
     * pooled range get an exact-size allocation that {@link #release} will
     * ignore.
     * <p>
     * When {@code clear} is {@code false} the returned contents are
     * <em>undefined</em> (recycled or uninitialized memory). When
     * {@code clear} is {@code true} the {@code [0, minLength)} range is
     * zeroed; elements beyond {@code minLength} (bucket slack) are left
     * undefined either way.
     *
     * @param minLength minimum required capacity, must be {@code >= 0}
     * @param clear     whether to zero the {@code [0, minLength)} range
     * @return an array with {@code length >= minLength}, never {@code null}
     * @throws IllegalArgumentException if {@code minLength < 0}
     */
    public final T acquire(int minLength, boolean clear) {
        if (minLength < 0) throw new IllegalArgumentException("minLength < 0: " + minLength);
        int idx = bucketIndexForRequest(minLength);
        T arr;
        if (idx < 0) {
            arr = allocate(minLength);              // out of range: exact size, non-poolable
        } else {
            T pooled = tryAcquire(slots, idx);
            arr = pooled != null ? pooled : allocate(bucketSize(idx));
        }
        if (clear) clear(arr, minLength);
        return arr;
    }

    /**
     * Return an array to the pool for reuse.
     * <p>
     * Only power-of-two arrays within the pooled range are retained; any
     * other array (including {@code null} or an already-full bucket) is
     * silently dropped and left to the garbage collector. It is the
     * caller's responsibility not to touch the array after releasing it.
     *
     * @param array the array to recycle, may be {@code null}
     */
    public final void release(T array) {
        if (array == null) return;
        int idx = bucketIndexForLength(lengthOf(array));
        if (idx < 0) return;
        tryRelease(slots, idx, array);
    }

    // ------------------------------------------------------------------
    // Concrete primitive pools.
    // ------------------------------------------------------------------

    public static final class OfByte extends ArrayPool<byte[]> {
        protected byte[] allocate(int n) { return newByte(n, false); }
        protected int lengthOf(byte[] a) { return a.length; }
        protected void clear(byte[] a, int len) { Arrays.fill(a, 0, len, (byte) 0); }
    }

    public static final class OfShort extends ArrayPool<short[]> {
        protected short[] allocate(int n) { return newShort(n, false); }
        protected int lengthOf(short[] a) { return a.length; }
        protected void clear(short[] a, int len) { Arrays.fill(a, 0, len, (short) 0); }
    }

    public static final class OfInt extends ArrayPool<int[]> {
        protected int[] allocate(int n) { return newInt(n, false); }
        protected int lengthOf(int[] a) { return a.length; }
        protected void clear(int[] a, int len) { Arrays.fill(a, 0, len, 0); }
    }

    public static final class OfLong extends ArrayPool<long[]> {
        protected long[] allocate(int n) { return newLong(n, false); }
        protected int lengthOf(long[] a) { return a.length; }
        protected void clear(long[] a, int len) { Arrays.fill(a, 0, len, 0L); }
    }

    public static final class OfFloat extends ArrayPool<float[]> {
        protected float[] allocate(int n) { return newFloat(n, false); }
        protected int lengthOf(float[] a) { return a.length; }
        protected void clear(float[] a, int len) { Arrays.fill(a, 0, len, 0f); }
    }

    public static final class OfDouble extends ArrayPool<double[]> {
        protected double[] allocate(int n) { return newDouble(n, false); }
        protected int lengthOf(double[] a) { return a.length; }
        protected void clear(double[] a, int len) { Arrays.fill(a, 0, len, 0d); }
    }

    // ------------------------------------------------------------------
    // Primitive-array allocation. With init=false the array is uninitialized
    // (fast path); with init=true it is zeroed when the underlying allocator
    // does not already do so (see INIT).
    // ------------------------------------------------------------------

    public static byte[] newByte(int n, boolean init) {
        if (init || ALLOC == null) return new byte[n];
        return (byte[]) ALLOC.allocate(byte.class, n);
    }

    public static short[] newShort(int n, boolean init) {
        if (init || ALLOC == null) return new short[n];
        return (short[]) ALLOC.allocate(short.class, n);
    }

    public static int[] newInt(int n, boolean init) {
        if (init || ALLOC == null) return new int[n];
        return (int[]) ALLOC.allocate(int.class, n);
    }

    public static long[] newLong(int n, boolean init) {
        if (init || ALLOC == null) return new long[n];
        return (long[]) ALLOC.allocate(long.class, n);
    }

    public static float[] newFloat(int n, boolean init) {
        if (init || ALLOC == null) return new float[n];
        return (float[]) ALLOC.allocate(float.class, n);
    }

    public static double[] newDouble(int n, boolean init) {
        if (init || ALLOC == null) return new double[n];
        return (double[]) ALLOC.allocate(double.class, n);
    }

    @FunctionalInterface
    private interface Allocator { Object allocate(Class<?> elemType, int n); }

    /** The uninitialized-array allocator (Unsafe when available, else a zeroing fallback). */
    private static final Allocator ALLOC;
    static {
        Allocator alloc;
        try {
            Class<?> unsafeCls = Class.forName("jdk.internal.misc.Unsafe");
            Field theUnsafeField = unsafeCls.getDeclaredField("theUnsafe");
            theUnsafeField.setAccessible(true);
            Object theUnsafe = theUnsafeField.get(null);
            Method m = unsafeCls.getDeclaredMethod("allocateUninitializedArray", Class.class, int.class);
            m.setAccessible(true);
            MethodHandles.Lookup lookup = MethodHandles.lookup();
            MethodHandle base = lookup.unreflect(m);
            MethodType samType = MethodType.methodType(Object.class, Class.class, int.class);
            CallSite site = LambdaMetafactory.metafactory(
                    lookup,
                    "allocate",
                    MethodType.methodType(Allocator.class, unsafeCls),
                    samType,
                    base,
                    samType
            );
            alloc = (Allocator) site.getTarget().invoke(theUnsafe);
        } catch (Throwable t) {
            alloc = null;
        }
        ALLOC = alloc;
    }

}
