package com.curioloop.yum4j.ssm.particle.model;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;

/**
 * Capability trait for a user-defined proposal distribution.
 *
 * <p>Used by the guided and auxiliary particle filters. Provides sampling
 * methods for the initial and transition proposals, as well as log-density
 * evaluation methods needed for importance-weight correction.
 *
 * @param <Y> observation type
 */
public interface Proposal<Y> {

    /**
     * Samples from the initial proposal {@code Q_0(·)} and writes the result
     * into {@code ctx.X()} starting at {@code ctx.xOff()} for {@code ctx.N()} particles.
     *
     * @param ctx step context carrying buffers and RNG
     */
    void sampleQ0(StepContext<Y> ctx);

    /**
     * Samples from the transition proposal {@code Q_t(X_{t-1}, ·)} and writes the result
     * into {@code ctx.X()} starting at {@code ctx.xOff()} for {@code ctx.N()} particles.
     *
     * @param ctx step context carrying buffers and RNG
     */
    void sampleQ(StepContext<Y> ctx);

    /**
     * Evaluates the log-density of the initial proposal {@code log q_0(X_0)}
     * and writes the length-{@code ctx.N()} result into {@code out} starting at {@code outOff}.
     *
     * @param ctx    step context carrying particle states
     * @param out    output buffer for log-density values
     * @param outOff offset into {@code out}
     */
    void logQ0(StepContext<Y> ctx, double[] out, int outOff);

    /**
     * Evaluates the log-density of the transition proposal {@code log q_t(X_{t-1}, X_t)}
     * and writes the length-{@code ctx.N()} result into {@code out} starting at {@code outOff}.
     *
     * @param ctx    step context carrying particle states
     * @param out    output buffer for log-density values
     * @param outOff offset into {@code out}
     */
    void logQ(StepContext<Y> ctx, double[] out, int outOff);
}
