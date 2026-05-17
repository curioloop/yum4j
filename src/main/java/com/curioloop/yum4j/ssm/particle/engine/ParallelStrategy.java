package com.curioloop.yum4j.ssm.particle.engine;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;

/**
 * Sealed interface governing how the engine partitions the particle index
 * range {@code [0, N)} across execution units.
 *
 * <p>Two implementations are permitted:
 * <ul>
 *   <li>{@link Serial} — invokes the kernel body once with {@code (0, N)},
 *       no executor, no synchronisation.</li>
 *   <li>{@link Striped} — partitions into contiguous slabs and dispatches
 *       to a user-supplied {@link Executor}, joining before return.</li>
 * </ul>
 *
 * @see SlabConsumer
 */
public sealed interface ParallelStrategy permits ParallelStrategy.Serial, ParallelStrategy.Striped {

    /**
     * Per-slab kernel callback. Accepts the contiguous slab boundaries
     * {@code (off, count)} as primitive {@code int}s — a specialisation
     * of {@code BiConsumer<Integer, Integer>} that avoids boxing on the
     * hot path.
     */
    @FunctionalInterface
    interface SlabConsumer {
        /**
         * Performs this operation on the given slab.
         *
         * @param off   the starting offset of the slab
         * @param count the number of elements in the slab
         */
        void accept(int off, int count);
    }

    /**
     * Invokes {@code body} for contiguous slabs that tile {@code [0, N)}.
     *
     * <p>{@link Serial} calls body once with {@code (0, N)}.
     * {@link Striped} splits into {@code chunks} slabs and dispatches to
     * the executor, joining before return.
     *
     * @param N    the total number of particles
     * @param body the kernel to invoke per slab, receiving {@code (off, count)}
     */
    void forRange(int N, SlabConsumer body);

    /**
     * The default serial strategy. Invokes the kernel directly without any
     * {@code Executor.submit} or synchronisation.
     */
    ParallelStrategy SERIAL = new Serial();

    /**
     * Creates a striped parallel strategy that partitions the particle range
     * into {@code chunks} contiguous slabs dispatched to the given executor.
     *
     * @param executor the executor to submit slab tasks to; must not be null
     * @param chunks   the number of contiguous slabs; must be &gt; 0
     * @return a striped parallel strategy
     * @throws IllegalArgumentException if {@code executor} is null or
     *                                  {@code chunks <= 0}
     */
    static ParallelStrategy striped(Executor executor, int chunks) {
        return new Striped(executor, chunks);
    }

    // ─── Implementations ────────────────────────────────────────────────

    /**
     * Serial execution: invokes the body once with the full range
     * {@code (0, N)}. No executor, no synchronisation overhead.
     */
    record Serial() implements ParallelStrategy {
        @Override
        public void forRange(int N, SlabConsumer body) {
            body.accept(0, N);
        }
    }

    /**
     * Striped parallel execution: partitions {@code [0, N)} into
     * {@code chunks} contiguous slabs and dispatches each to the supplied
     * {@link Executor}.
     *
     * <p>Slab {@code k} covers the index range
     * {@code [k * N / chunks, (k+1) * N / chunks)}.
     * One task per slab is submitted to the executor; the calling thread
     * blocks until all tasks complete via a {@link CountDownLatch}.
     *
     * @param executor the executor for slab dispatch; must not be null
     * @param chunks   the number of slabs; must be &gt; 0
     */
    record Striped(Executor executor, int chunks) implements ParallelStrategy {
        public Striped {
            if (executor == null) {
                throw new IllegalArgumentException(
                        "ParallelStrategy.striped requires a non-null Executor");
            }
            if (chunks <= 0) {
                throw new IllegalArgumentException(
                        "ParallelStrategy.striped requires chunks > 0, got: " + chunks);
            }
        }

        @Override
        public void forRange(int N, SlabConsumer body) {
            if (N <= 0) {
                return;
            }
            int actualChunks = Math.min(chunks, N);
            if (actualChunks == 1) {
                // Single slab — run inline, no executor overhead.
                body.accept(0, N);
                return;
            }

            CountDownLatch latch = new CountDownLatch(actualChunks);
            Throwable[] error = new Throwable[1];

            for (int k = 0; k < actualChunks; k++) {
                int slabStart = (int) ((long) k * N / actualChunks);
                int slabEnd = (int) ((long) (k + 1) * N / actualChunks);
                int off = slabStart;
                int count = slabEnd - slabStart;

                executor.execute(() -> {
                    try {
                        body.accept(off, count);
                    } catch (Throwable t) {
                        synchronized (error) {
                            if (error[0] == null) {
                                error[0] = t;
                            }
                        }
                    } finally {
                        latch.countDown();
                    }
                });
            }

            try {
                latch.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Striped forRange interrupted", e);
            }

            if (error[0] != null) {
                if (error[0] instanceof RuntimeException re) {
                    throw re;
                }
                if (error[0] instanceof Error err) {
                    throw err;
                }
                throw new RuntimeException("Exception in striped slab task", error[0]);
            }
        }
    }
}
