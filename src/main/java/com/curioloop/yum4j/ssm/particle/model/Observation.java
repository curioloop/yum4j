package com.curioloop.yum4j.ssm.particle.model;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;

/**
 * Capability trait for the observation (measurement) model.
 *
 * <p>Evaluates the log-density {@code log p(y_t | X_t)} and writes the result
 * into the log-weight buffer carried by the {@link StepContext}.
 *
 * @param <Y> observation type
 */
public interface Observation<Y> {

    /**
     * Computes the log-density of the observation given the current particles
     * and writes the length-{@code ctx.N()} result into {@code ctx.logW()}
     * starting at {@code ctx.lwOff()}.
     *
     * @param ctx step context carrying buffers, observation, and particle states
     */
    void logG(StepContext<Y> ctx);
}
