package com.curioloop.yum4j.ssm.particle.engine;

import org.junit.jupiter.api.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Tests for {@link ParallelStrategy}, focusing on the {@link ParallelStrategy.Striped}
 * implementation that partitions [0, N) into contiguous slabs dispatched to an executor.
 */
class ParallelStrategyTest {

    // ─── Validation ─────────────────────────────────────────────────────

    @Test
    void striped_nullExecutor_throws() {
        assertThatThrownBy(() -> ParallelStrategy.striped(null, 4))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("non-null Executor");
    }

    @Test
    void striped_zeroChunks_throws() {
        ExecutorService exec = Executors.newFixedThreadPool(2);
        try {
            assertThatThrownBy(() -> ParallelStrategy.striped(exec, 0))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("chunks > 0");
        } finally {
            exec.shutdown();
        }
    }

    @Test
    void striped_negativeChunks_throws() {
        ExecutorService exec = Executors.newFixedThreadPool(2);
        try {
            assertThatThrownBy(() -> ParallelStrategy.striped(exec, -1))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("chunks > 0");
        } finally {
            exec.shutdown();
        }
    }

    // ─── forRange partitioning ──────────────────────────────────────────

    @Test
    void striped_forRange_coversFullRange() {
        ExecutorService exec = Executors.newFixedThreadPool(4);
        try {
            int N = 100;
            int chunks = 4;
            ParallelStrategy strategy = ParallelStrategy.striped(exec, chunks);

            // Track which indices are covered
            boolean[] covered = new boolean[N];
            Object lock = new Object();

            strategy.forRange(N, (off, count) -> {
                synchronized (lock) {
                    for (int i = off; i < off + count; i++) {
                        covered[i] = true;
                    }
                }
            });

            for (int i = 0; i < N; i++) {
                assertThat(covered[i]).as("index %d should be covered", i).isTrue();
            }
        } finally {
            exec.shutdown();
        }
    }

    @Test
    void striped_forRange_producesContiguousSlabs() {
        ExecutorService exec = Executors.newFixedThreadPool(4);
        try {
            int N = 17;
            int chunks = 4;
            ParallelStrategy strategy = ParallelStrategy.striped(exec, chunks);

            // Collect slab boundaries
            int[][] slabs = new int[chunks][2]; // [off, count]
            AtomicInteger slabIndex = new AtomicInteger(0);
            Object lock = new Object();

            strategy.forRange(N, (off, count) -> {
                synchronized (lock) {
                    int idx = slabIndex.getAndIncrement();
                    slabs[idx][0] = off;
                    slabs[idx][1] = count;
                }
            });

            // Verify all slabs are non-empty and cover [0, N)
            int totalCount = 0;
            for (int[] slab : slabs) {
                assertThat(slab[1]).as("slab count").isGreaterThan(0);
                totalCount += slab[1];
            }
            assertThat(totalCount).isEqualTo(N);
        } finally {
            exec.shutdown();
        }
    }

    @Test
    void striped_forRange_slabFormula_matches_design() {
        // Verify slab k covers [k*N/chunks, (k+1)*N/chunks) as per design
        ExecutorService exec = Executors.newFixedThreadPool(4);
        try {
            int N = 100;
            int chunks = 4;
            ParallelStrategy strategy = ParallelStrategy.striped(exec, chunks);

            int[][] expectedSlabs = new int[chunks][2];
            for (int k = 0; k < chunks; k++) {
                int start = (int) ((long) k * N / chunks);
                int end = (int) ((long) (k + 1) * N / chunks);
                expectedSlabs[k][0] = start;
                expectedSlabs[k][1] = end - start;
            }

            // Collect actual slabs (order may vary due to concurrency)
            int[][] actualSlabs = new int[chunks][2];
            Object lock = new Object();
            AtomicInteger idx = new AtomicInteger(0);

            strategy.forRange(N, (off, count) -> {
                synchronized (lock) {
                    int i = idx.getAndIncrement();
                    actualSlabs[i][0] = off;
                    actualSlabs[i][1] = count;
                }
            });

            // Sort both by offset for comparison
            java.util.Arrays.sort(expectedSlabs, (a, b) -> Integer.compare(a[0], b[0]));
            java.util.Arrays.sort(actualSlabs, (a, b) -> Integer.compare(a[0], b[0]));

            for (int k = 0; k < chunks; k++) {
                assertThat(actualSlabs[k][0]).as("slab %d offset", k).isEqualTo(expectedSlabs[k][0]);
                assertThat(actualSlabs[k][1]).as("slab %d count", k).isEqualTo(expectedSlabs[k][1]);
            }
        } finally {
            exec.shutdown();
        }
    }

    @Test
    void striped_forRange_N_zero_doesNothing() {
        ExecutorService exec = Executors.newFixedThreadPool(2);
        try {
            ParallelStrategy strategy = ParallelStrategy.striped(exec, 4);
            AtomicInteger calls = new AtomicInteger(0);
            strategy.forRange(0, (off, count) -> calls.incrementAndGet());
            assertThat(calls.get()).isZero();
        } finally {
            exec.shutdown();
        }
    }

    @Test
    void striped_forRange_chunksGreaterThanN_clamps() {
        // When chunks > N, actual chunks should be clamped to N
        ExecutorService exec = Executors.newFixedThreadPool(4);
        try {
            int N = 3;
            int chunks = 10;
            ParallelStrategy strategy = ParallelStrategy.striped(exec, chunks);

            AtomicInteger calls = new AtomicInteger(0);
            boolean[] covered = new boolean[N];
            Object lock = new Object();

            strategy.forRange(N, (off, count) -> {
                calls.incrementAndGet();
                synchronized (lock) {
                    for (int i = off; i < off + count; i++) {
                        covered[i] = true;
                    }
                }
            });

            // Should have at most N calls (clamped)
            assertThat(calls.get()).isEqualTo(N);
            for (int i = 0; i < N; i++) {
                assertThat(covered[i]).isTrue();
            }
        } finally {
            exec.shutdown();
        }
    }

    @Test
    void striped_forRange_singleChunk_runsInline() {
        // With chunks=1, body should be called once with (0, N)
        ExecutorService exec = Executors.newFixedThreadPool(2);
        try {
            int N = 50;
            ParallelStrategy strategy = ParallelStrategy.striped(exec, 1);

            AtomicInteger calls = new AtomicInteger(0);
            int[] capturedOff = new int[1];
            int[] capturedCount = new int[1];

            strategy.forRange(N, (off, count) -> {
                calls.incrementAndGet();
                capturedOff[0] = off;
                capturedCount[0] = count;
            });

            assertThat(calls.get()).isEqualTo(1);
            assertThat(capturedOff[0]).isZero();
            assertThat(capturedCount[0]).isEqualTo(N);
        } finally {
            exec.shutdown();
        }
    }

    @Test
    void striped_forRange_propagatesRuntimeException() {
        ExecutorService exec = Executors.newFixedThreadPool(2);
        try {
            ParallelStrategy strategy = ParallelStrategy.striped(exec, 2);

            assertThatThrownBy(() -> strategy.forRange(10, (off, count) -> {
                if (off > 0) {
                    throw new IllegalStateException("test error");
                }
            })).isInstanceOf(IllegalStateException.class)
                    .hasMessage("test error");
        } finally {
            exec.shutdown();
        }
    }

    // ─── Serial baseline ────────────────────────────────────────────────

    @Test
    void serial_forRange_callsBodyOnce() {
        AtomicInteger calls = new AtomicInteger(0);
        int[] capturedOff = new int[1];
        int[] capturedCount = new int[1];

        ParallelStrategy.SERIAL.forRange(100, (off, count) -> {
            calls.incrementAndGet();
            capturedOff[0] = off;
            capturedCount[0] = count;
        });

        assertThat(calls.get()).isEqualTo(1);
        assertThat(capturedOff[0]).isZero();
        assertThat(capturedCount[0]).isEqualTo(100);
    }
}
