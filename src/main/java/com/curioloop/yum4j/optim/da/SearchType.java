/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.da;

/** Source of a new global best reported to {@link SearchHook}. */
public enum SearchType {
    /** Improvement came directly from an accepted annealing proposal. */
    ANNEALING,

    /** Improvement came from the local search launched after a global-best improvement. */
    LOCAL_SEARCH,

    /** Improvement came from the chain-minimum local search used by dual annealing. */
    DUAL_ANNEALING
}