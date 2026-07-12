/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.da;

/** Hook invoked when dual annealing finds a new global best. */
@FunctionalInterface
public interface SearchHook {

    /**
    * Called with the live internal best array. Mutating it can corrupt optimization state.
    *
    * <p>The {@code search} argument identifies whether the improvement came from the global
    * annealing proposal, the first local-search pass after a global improvement, or the
    * strategy-chain local-search pass.</p>
     *
     * @return true to request early termination, false to continue
     */
    boolean onMinimum(int iteration,
                      int evaluations,
                      double[] x,
                      double cost,
                      SearchType search);
}