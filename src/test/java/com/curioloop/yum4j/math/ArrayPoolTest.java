package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.IdentityHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ArrayPoolTest {

    private static final int MIN_POOLED = 1 << 13;   // 8K
    private static final int MAX_POOLED = 1 << 22;   // 4M
    private static final int CAP_PER_BUCKET = 8;

    // ------------------------------------------------------------------
    // Bucket sizing
    // ------------------------------------------------------------------

    @ParameterizedTest
    @CsvSource({
            "8191,        -1",        // just below the pooled range
            "8192,        8192",      // exact lower boundary (pow2)
            "8193,        16384",     // rounds up to next pow2
            "9000,        16384",
            "16384,       16384",     // exact pow2
            "16385,       32768",
            "4194304,     4194304",   // exact upper boundary (4M)
            "4194305,     -1",        // just above the pooled range
            "0,           -1",
    })
    void bucketSizeForRequestRoundsToEnclosingPowerOfTwo(int request, int expected) {
        assertEquals(expected, ArrayPool.bucketSizeForRequest(request));
    }

    // ------------------------------------------------------------------
    // acquire capacity contract
    // ------------------------------------------------------------------

    @Test
    void acquireInPooledRangeReturnsBucketSizedArray() {
        ArrayPool.OfDouble pool = new ArrayPool.OfDouble();
        for (int req : new int[]{8192, 8193, 9000, 16384, 100_000}) {
            double[] a = pool.acquire(req);
            assertEquals(ArrayPool.bucketSizeForRequest(req), a.length, "req=" + req);
            assertTrue(a.length >= req);
        }
    }

    @Test
    void acquireBelowRangeReturnsExactSize() {
        ArrayPool.OfDouble pool = new ArrayPool.OfDouble();
        // even a power-of-two below the floor is not bucketed
        for (int req : new int[]{0, 1, 100, 128, 8191}) {
            assertEquals(req, pool.acquire(req).length, "req=" + req);
        }
    }

    @Test
    void acquireAboveRangeReturnsExactSize() {
        ArrayPool.OfDouble pool = new ArrayPool.OfDouble();
        int req = MAX_POOLED + 1;
        assertEquals(req, pool.acquire(req).length);
    }

    @Test
    void acquireNegativeThrows() {
        ArrayPool.OfDouble pool = new ArrayPool.OfDouble();
        assertThrows(IllegalArgumentException.class, () -> pool.acquire(-1));
    }

    // ------------------------------------------------------------------
    // recycling
    // ------------------------------------------------------------------

    @Test
    void releaseThenAcquireRecyclesSameInstance() {
        ArrayPool.OfDouble pool = new ArrayPool.OfDouble();
        double[] first = pool.acquire(9000);          // bucket 16384
        pool.release(first);
        double[] second = pool.acquire(9000);
        assertSame(first, second, "in-range array should be recycled");
    }

    @Test
    void releaseOfNonBucketArrayIsDropped() {
        ArrayPool.OfDouble pool = new ArrayPool.OfDouble();
        double[] belowRange = pool.acquire(100);       // exact length 100, not pooled
        pool.release(belowRange);
        assertNotSame(belowRange, pool.acquire(100));
    }

    @Test
    void releaseNullIsNoOp() {
        ArrayPool.OfDouble pool = new ArrayPool.OfDouble();
        assertDoesNotThrow(() -> pool.release(null));
    }

    @Test
    void bucketRetainsAtMostCapPerBucketArrays() {
        ArrayPool.OfDouble pool = new ArrayPool.OfDouble();
        int overCap = CAP_PER_BUCKET + 2;

        // Acquire distinct arrays first (pool is empty, so each is freshly allocated),
        // then release them all: only CAP_PER_BUCKET should be retained.
        IdentityHashMap<double[], Boolean> released = new IdentityHashMap<>();
        double[][] borrowed = new double[overCap][];
        for (int i = 0; i < overCap; i++) {
            borrowed[i] = pool.acquire(9000);          // all land in the 16384 bucket
            released.put(borrowed[i], Boolean.TRUE);
        }
        for (double[] a : borrowed) pool.release(a);

        int recycled = 0;
        IdentityHashMap<double[], Boolean> seen = new IdentityHashMap<>();
        for (int i = 0; i < overCap; i++) {
            double[] a = pool.acquire(9000);
            assertTrue(seen.put(a, Boolean.TRUE) == null, "instance handed out twice");
            if (released.containsKey(a)) recycled++;
        }
        assertEquals(CAP_PER_BUCKET, recycled, "at most CAP_PER_BUCKET arrays should survive");
    }

    // ------------------------------------------------------------------
    // clear semantics
    // ------------------------------------------------------------------

    @Test
    void acquireDirtyKeepsRecycledContents() {
        ArrayPool.OfDouble pool = new ArrayPool.OfDouble();
        double[] a = pool.acquire(9000);
        java.util.Arrays.fill(a, 7.0);
        pool.release(a);

        double[] b = pool.acquire(9000, false);
        assertSame(a, b);
        assertEquals(7.0, b[0], 0.0);
        assertEquals(7.0, b[8999], 0.0);
    }

    @Test
    void acquireClearZeroesRequestedRangeOnly() {
        ArrayPool.OfDouble pool = new ArrayPool.OfDouble();
        double[] a = pool.acquire(16384);              // exact bucket size
        java.util.Arrays.fill(a, 7.0);
        pool.release(a);

        double[] b = pool.acquire(9000, true);         // in-range: recycles a, clears [0, 9000)
        assertSame(a, b);
        for (int i = 0; i < 9000; i++) assertEquals(0.0, b[i], 0.0, "i=" + i);
        assertEquals(7.0, b[9000], 0.0, "slack beyond minLength must stay untouched");
        assertEquals(7.0, b[16383], 0.0);
    }

    // ------------------------------------------------------------------
    // typed pools & shared singletons
    // ------------------------------------------------------------------

    @Test
    void typedPoolsProduceCorrectArrayTypes() {
        assertEquals(16384, new ArrayPool.OfByte().acquire(9000).length);
        assertEquals(16384, new ArrayPool.OfShort().acquire(9000).length);
        assertEquals(16384, new ArrayPool.OfInt().acquire(9000).length);
        assertEquals(16384, new ArrayPool.OfLong().acquire(9000).length);
        assertEquals(16384, new ArrayPool.OfFloat().acquire(9000).length);
        assertEquals(16384, new ArrayPool.OfDouble().acquire(9000).length);
    }

    @Test
    void intPoolClearAndRecycle() {
        ArrayPool.OfInt pool = new ArrayPool.OfInt();
        int[] a = pool.acquire(9000);
        java.util.Arrays.fill(a, 42);
        pool.release(a);
        int[] b = pool.acquire(9000, true);
        assertSame(a, b);
        assertEquals(0, b[0]);
        assertEquals(0, b[8999]);
    }

    // ------------------------------------------------------------------
    // concurrency: a borrowed array is exclusively owned
    // ------------------------------------------------------------------

    @Test
    void concurrentAcquireReleaseGivesExclusiveOwnership() throws InterruptedException {
        ArrayPool.OfLong pool = new ArrayPool.OfLong();
        int threads = 12;
        int iterations = 5_000;
        int len = 8192;                                // all threads hammer the same bucket

        ExecutorService exec = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        AtomicInteger corruptions = new AtomicInteger();
        AtomicInteger shortArrays = new AtomicInteger();

        for (int t = 0; t < threads; t++) {
            final long marker = (t + 1L) << 40;
            exec.submit(() -> {
                try {
                    start.await();
                    for (int it = 0; it < iterations; it++) {
                        long tag = marker | it;
                        long[] a = pool.acquire(len);
                        if (a.length < len) { shortArrays.incrementAndGet(); continue; }
                        for (int i = 0; i < len; i++) a[i] = tag;
                        for (int i = 0; i < len; i++) {
                            if (a[i] != tag) { corruptions.incrementAndGet(); break; }
                        }
                        pool.release(a);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        start.countDown();
        exec.shutdown();
        assertTrue(exec.awaitTermination(60, TimeUnit.SECONDS), "workers did not finish in time");
        assertEquals(0, shortArrays.get(), "acquire returned an undersized array");
        assertEquals(0, corruptions.get(), "a borrowed array was observed by two threads at once");
    }
}
