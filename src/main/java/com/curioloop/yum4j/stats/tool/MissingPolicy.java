package com.curioloop.yum4j.stats.tool;

/**
 * Strategy for handling {@code NaN} values in time-series input.
 *
 * <p>Used by {@link AutoCovariance}, {@link AutoCorrelation}, and other tools
 * that accept time-series data which may contain missing observations.</p>
 */
public enum MissingPolicy {
    /** No NaN checks are performed (fastest). */
    NONE,
    /** Throws {@link IllegalArgumentException} if any NaN is found. */
    RAISE,
    /**
     * Removes NaN observations and treats the remaining values as a
     * contiguous series before computing.
     */
    DROP,
    /**
     * Replaces NaN with 0 and adjusts each lag's denominator by the
     * number of non-missing observation pairs at that lag.
     */
    CONSERVATIVE
}
