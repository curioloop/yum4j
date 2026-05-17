package com.curioloop.yum4j.ssm.particle;

/**
 * Selects the backward-smoothing algorithm dispatched by
 * {@link FilterResult#smooth(SmoothingMode, int)}.
 *
 * <p>Each mode corresponds to a concrete smoother in the
 * {@code com.curioloop.yum4j.ssm.particle.smooth} package.
 *
 */
public enum SmoothingMode {

    /**
     * Forward-Filtering Backward-Sampling (FFBS).
     * Requires {@link com.curioloop.yum4j.ssm.particle.HistoryMode#FULL}
     * history and a {@link com.curioloop.yum4j.ssm.particle.model.TransitionDensity}.
     */
    FFBS,

    /**
     * Two-filter smoother.
     * Requires {@link com.curioloop.yum4j.ssm.particle.HistoryMode#FULL}
     * history and a {@link com.curioloop.yum4j.ssm.particle.model.TransitionDensity}.
     */
    TWO_FILTER,

    /**
     * Fixed-lag smoother.
     * Requires {@link com.curioloop.yum4j.ssm.particle.HistoryMode#ROLLING}
     * history.
     */
    FIXED_LAG,

    /**
     * Paris-style online additive smoother.
     * Requires {@link com.curioloop.yum4j.ssm.particle.HistoryMode#FULL}
     * history and a {@link com.curioloop.yum4j.ssm.particle.model.TransitionDensity}
     * with an upper bound.
     */
    PARIS
}
