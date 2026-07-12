/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.de;

import com.curioloop.yum4j.optim.Univariate;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/**
 * Evaluates a batch of differential evolution candidates.
 *
 * <p>Evaluators are used for the initial population and for deferred trial populations.
 * Supplying a custom evaluator forces deferred updating so that the candidate buffer is
 * stable for the whole batch. Implementations may evaluate serially, vectorize objective
 * calls, or distribute work across an executor.</p>
 */
@FunctionalInterface
public interface DEEvaluator {

    /**
     * Evaluates candidates in {@code [startInclusive, endExclusive)}.
     *
     * <p>{@code candidates} is flat row-major: candidate {@code i}, coordinate {@code j}
     * is stored at {@code candidates[i * dimension + j]}. Implementations must write
    * {@code energies[i]} for each evaluated candidate. {@code scratch} is explicit
    * caller-owned workspace for implementations that need standalone objective arrays.
    * Implementations must not retain references to {@code candidates} or {@code scratch}
    * after the call returns.</p>
     */
    void evaluate(Univariate.Objective objective,
                  double[] candidates,
                  int dimension,
                  int startInclusive,
                  int endExclusive,
                  double[] energies,
                  double[][] scratch);

    /** Serial candidate evaluator used as the identity/default implementation. */
    static DEEvaluator serial() {
        return SerialHolder.SERIAL;
    }

    /**
     * Executor-backed evaluator. The caller owns {@code executor} and is responsible for closing it.
     *
     * <p>Each worker gets a disjoint contiguous index range. Objective arguments are copied
     * into row-specific scratch arrays before evaluation so objective functions can safely
     * read a conventional {@code double[]} vector.</p>
     */
    static DEEvaluator parallel(ExecutorService executor, int workers) {
        Objects.requireNonNull(executor, "executor must not be null");
        if (workers <= 0) throw new IllegalArgumentException("workers must be positive, got " + workers);
        return (objective, candidates, dimension, startInclusive, endExclusive, energies, scratch) -> {
            int count = endExclusive - startInclusive;
            if (count <= 0) return;
            int tasks = Math.min(workers, count);
            int chunk = (count + tasks - 1) / tasks;
            List<Future<?>> futures = new ArrayList<>(tasks);
            for (int task = 0; task < tasks; task++) {
                int from = startInclusive + task * chunk;
                int to = Math.min(endExclusive, from + chunk);
                if (from >= to) break;
                futures.add(executor.submit(() -> SerialHolder.SERIAL.evaluate(objective, candidates, dimension, from, to, energies, scratch)));
            }
            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (InterruptedException ex) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("parallel differential evolution evaluation interrupted", ex);
                } catch (ExecutionException ex) {
                    Throwable cause = ex.getCause();
                    if (cause instanceof RuntimeException runtimeException) throw runtimeException;
                    if (cause instanceof Error error) throw error;
                    throw new RuntimeException("parallel differential evolution evaluation failed", cause);
                }
            }
        };
    }

    final class SerialHolder {
        private static final DEEvaluator SERIAL = (objective, candidates, dimension,
                                                   startInclusive, endExclusive,
                                                   energies, scratch) -> {
            for (int i = startInclusive; i < endExclusive; i++) {
                double[] x = scratch[i];
                System.arraycopy(candidates, i * dimension, x, 0, dimension);
                double value = objective.evaluate(x, dimension);
                energies[i] = Double.isFinite(value) ? value : Double.POSITIVE_INFINITY;
            }
        };

        private SerialHolder() {}
    }
}