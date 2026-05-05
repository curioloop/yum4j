package com.curioloop.yum4j.fft;

import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Thread-safe plan cache for FFT transforms.
 *
 * <p>Design rationale: FFT plan objects are expensive to create (twiddle factor
 * computation, factorization) but immutable once built, and the working set is
 * typically tiny (a handful of distinct lengths).  The cache therefore optimises
 * for the <em>hit</em> path:
 *
 * <ul>
 *   <li>A single {@link ConcurrentHashMap} with {@code long} keys encoded as
 *       {@link Long} provides lock-free reads with minimal per-lookup allocation
 *       (only the unavoidable {@code Long} box, which the JVM may cache for
 *       small values and will scalar-replace in hot loops).</li>
 *   <li>No LRU tracking on the read path — the cache is effectively unbounded
 *       while {@code maxEntries > 0}.  Eviction only happens when the user
 *       explicitly calls {@link #setMaxEntries} or {@link #clear}.</li>
 *   <li>Plan creation uses {@code putIfAbsent} rather than
 *       {@code computeIfAbsent} to avoid {@code ConcurrentHashMap}'s recursive-
 *       update restriction (plan constructors call back into the cache for
 *       sub-plans).</li>
 * </ul>
 *
 * <h3>Key encoding</h3>
 * All plan types are packed into a single {@code long}:
 * <pre>
 *   bits 63..32  tag (plan type + dcst parameters)
 *   bits 31..0   FFT length
 *
 *   Tag values:
 *     0       complex transform
 *     1       real transform
 *     2       split-radix plan
 *     4..7    DCST cosine types 1-4   (DCST_BASE + type)
 *     12..15  DCST sine   types 1-4   (DCST_BASE + type + 8)
 * </pre>
 */
public final class FftPlanCache {

    // -- key encoding --------------------------------------------------------

    private static final long TAG_COMPLEX     = 0L;
    private static final long TAG_REAL        = 1L;
    private static final long TAG_SPLIT_RADIX = 2L;
    private static final long DCST_BASE       = 3L;

    private static long dcstTag(int type, boolean cosine) {
        return DCST_BASE + type + (cosine ? 0 : 8);
    }

    private static long key(long tag, int length) {
        return (tag << 32) | (length & 0xFFFF_FFFFL);
    }

    // -- cache state ---------------------------------------------------------

    private static final ConcurrentHashMap<Long, Object> CACHE = new ConcurrentHashMap<>(32);

    private static volatile int maxEntries = 32;

    private FftPlanCache() {
    }

    // -- public API ----------------------------------------------------------

    static FftComplexTransform complexTransform(int length) {
        if (maxEntries == 0) {
            return new FftComplexTransform(length);
        }
        return getOrCreate(key(TAG_COMPLEX, length), () -> new FftComplexTransform(length));
    }

    static FftRealTransform realTransform(int length) {
        if (maxEntries == 0) {
            return new FftRealTransform(length);
        }
        return getOrCreate(key(TAG_REAL, length), () -> new FftRealTransform(length));
    }

    static FftSplitRadixPlan splitRadixPlan(int length) {
        if (maxEntries == 0) {
            return new FftSplitRadixPlan(length);
        }
        return getOrCreate(key(TAG_SPLIT_RADIX, length), () -> new FftSplitRadixPlan(length));
    }

    static FftDcstPlan dcstPlan(int length, int type, boolean cosine) {
        if (maxEntries == 0) {
            return new FftDcstPlan(length, type, cosine);
        }
        return getOrCreate(key(dcstTag(type, cosine), length), () -> new FftDcstPlan(length, type, cosine));
    }

    /** Returns the maximum number of cached plans. */
    public static int maxEntries() {
        return maxEntries;
    }

    /**
     * Set the maximum number of cached plans.  If the new limit is smaller than
     * the current cache size the cache is cleared entirely (simple and safe;
     * entries will be re-created on demand).  Setting {@code entries} to 0
     * disables caching — every request creates a fresh plan.
     */
    public static void setMaxEntries(int entries) {
        if (entries < 0) {
            throw new IllegalArgumentException("plan cache size must be >= 0");
        }
        maxEntries = entries;
        if (entries == 0 || CACHE.size() > entries) {
            CACHE.clear();
        }
    }

    /** Returns the current number of cached plans. */
    public static int size() {
        return CACHE.size();
    }

    /** Removes all cached plans. */
    public static void clear() {
        CACHE.clear();
    }

    // -- internals -----------------------------------------------------------

    @SuppressWarnings("unchecked")
    private static <T> T getOrCreate(long key, Supplier<T> factory) {
        Long boxedKey = key;
        // Fast path: lock-free lookup
        Object cached = CACHE.get(boxedKey);
        if (cached != null) {
            return (T) cached;
        }
        // Slow path: create outside the map to avoid recursive-update errors
        // (plan constructors may call back into the cache for sub-plans).
        T created = factory.get();
        Object winner = CACHE.putIfAbsent(boxedKey, created);
        if (winner == null) {
            trimIfNeeded();
        }
        return winner != null ? (T) winner : created;
    }

    /**
     * If the cache has grown beyond {@code maxEntries}, remove arbitrary
     * entries until it fits.  No LRU tracking — for a cache this small
     * (typically ≤32 entries) the simplicity is worth more than optimal
     * eviction order.
     */
    private static void trimIfNeeded() {
        int max = maxEntries;
        if (max > 0 && CACHE.size() > max) {
            for (Long k : CACHE.keySet()) {
                if (CACHE.size() <= max) break;
                CACHE.remove(k);
            }
        }
    }
}
