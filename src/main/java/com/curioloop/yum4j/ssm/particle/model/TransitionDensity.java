package com.curioloop.yum4j.ssm.particle.model;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;

/**
 * Capability trait for evaluating the transition density in closed form.
 *
 * <p>Required by the guided particle filter, the auxiliary particle filter,
 * FFBS, and the Paris smoother. Provides log-density evaluation for both
 * the initial distribution and the transition kernel, plus an upper bound
 * used by rejection-based smoothers.
 *
 * @param <Y> observation type
 */
public interface TransitionDensity<Y> {

    /**
     * Evaluates the log-density of the initial distribution {@code log p(X_0)}
     * and writes the length-{@code ctx.N()} result into {@code out} starting at {@code outOff}.
     *
     * @param ctx    step context carrying particle states
     * @param out    output buffer for log-density values
     * @param outOff offset into {@code out}
     */
    void logPt0(StepContext<Y> ctx, double[] out, int outOff);

    /**
     * Evaluates the log-density of the transition kernel {@code log p(X_t | X_{t-1})}
     * and writes the length-{@code ctx.N()} result into {@code out} starting at {@code outOff}.
     *
     * @param ctx    step context carrying particle states (current and previous)
     * @param out    output buffer for log-density values
     * @param outOff offset into {@code out}
     */
    void logPt(StepContext<Y> ctx, double[] out, int outOff);

    /**
     * Returns an upper bound on the transition log-density at time {@code t},
     * used by rejection-based backward-sampling smoothers.
     *
     * @param t time index
     * @return upper bound on {@code log p(X_t | X_{t-1})} for all particle pairs
     */
    double upperBound(int t);
}
