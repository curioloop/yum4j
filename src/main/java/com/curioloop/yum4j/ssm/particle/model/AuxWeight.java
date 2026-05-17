package com.curioloop.yum4j.ssm.particle.model;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;

/**
 * Capability trait for auxiliary particle filter pre-weights.
 *
 * <p>Evaluates the auxiliary log-weight {@code log η(X_t, y_{t+1})} used by
 * {@code FeynmanKac.auxiliary} to adjust resampling weights before the
 * transition step.
 *
 * @param <Y> observation type
 */
public interface AuxWeight<Y> {

    /**
     * Evaluates the auxiliary log-weight {@code log η(X_t, y_{t+1})} and writes
     * the length-{@code ctx.N()} result into {@code out} starting at {@code outOff}.
     *
     * @param ctx    step context carrying current particle states
     * @param yNext  the next observation {@code y_{t+1}}
     * @param out    output buffer for auxiliary log-weight values
     * @param outOff offset into {@code out}
     */
    void logEta(StepContext<Y> ctx, Y yNext, double[] out, int outOff);
}
