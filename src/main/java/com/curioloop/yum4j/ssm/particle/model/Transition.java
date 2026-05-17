package com.curioloop.yum4j.ssm.particle.model;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;

/**
 * Capability trait for the state-transition model.
 *
 * <p>Provides the state dimension and two sampling methods: one for the initial
 * distribution {@code M_0} and one for the transition kernel {@code M_t(x_{t-1}, ·)}.
 * Both methods write directly into the particle buffer carried by the {@link StepContext}
 * without allocating on the hot path.
 *
 * @param <Y> observation type
 */
public interface Transition<Y> {

    /**
     * Returns the state dimension per particle.
     *
     * @return state dimension (≥ 1)
     */
    int dim();

    /**
     * Samples from the initial distribution {@code M_0} and writes the result
     * into {@code ctx.X()} starting at {@code ctx.xOff()} for {@code ctx.N()} particles.
     *
     * @param ctx step context carrying buffers and RNG
     */
    void sampleM0(StepContext<Y> ctx);

    /**
     * Samples from the transition kernel {@code M_t(X_{t-1}, ·)} and writes the result
     * into {@code ctx.X()} starting at {@code ctx.xOff()} for {@code ctx.N()} particles,
     * reading previous states from {@code ctx.Xprev()} at {@code ctx.xpOff()}.
     *
     * @param ctx step context carrying buffers and RNG
     */
    void sampleM(StepContext<Y> ctx);
}
