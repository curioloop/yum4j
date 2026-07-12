/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.de;

import java.util.Random;

/**
 * Custom differential evolution strategy that writes complete trial vectors directly.
 *
 * <p>This is the extension point that corresponds to SciPy's callable {@code strategy}.
 * Unlike {@link EvolutionStrategy}, a custom strategy is responsible for the
 * whole trial vector, not just the mutant vector. The solver still repairs bounds,
 * projects integral coordinates, evaluates the objective, and applies greedy selection.</p>
 */
@FunctionalInterface
public interface TrialStrategy {

    /**
    * Produces a trial vector in real parameter space.
     *
     * <p>{@code population} is the live flat internal population array. Member {@code i},
     * coordinate {@code j} is stored at {@code population[i * dimension + j]}. Mutating it
    * can corrupt optimization state. Implementations must fill every coordinate in
    * {@code trialOut[trialOffset .. trialOffset + dimension)} with a finite value.
    * Returning out-of-bounds coordinates is allowed; the core repairs them before
    * evaluation.</p>
     */
    void trial(int candidate,
               double[] population,
               int dimension,
               int populationSize,
               Random rng,
               double[] trialOut,
               int trialOffset);
}