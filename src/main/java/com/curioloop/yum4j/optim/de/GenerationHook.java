/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.de;

/** Hook invoked after each completed differential evolution generation. */
@FunctionalInterface
public interface GenerationHook {

    /**
     * Called with live internal solver arrays. Mutating any array can corrupt optimization state.
     *
     * @return true to request early termination, false to continue
     */
    boolean onGeneration(int iteration,
                         int evaluations,
                         double[] best,
                         double cost,
                         double convergence,
                         double[] population,
                         double[] populationEnergies,
                         int dimension,
                         int populationSize);
}