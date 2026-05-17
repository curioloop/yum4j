package com.curioloop.yum4j.ssm.particle.filter;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.model.Observation;
import com.curioloop.yum4j.ssm.particle.model.Transition;

/**
 * Bootstrap particle filter implementation of {@link FeynmanKac}.
 *
 * <p>Uses the transition kernel as the proposal and the observation density
 * as the importance weight. The fused advance kernel inlines
 * {@link Transition#sampleM(StepContext)} followed by
 * {@link Observation#logG(StepContext)} in one pass over {@code n ∈ [0, N)}.
 *
 * <p>Reads the current observation from {@link StepContext#observation()}; the
 * engine writes it in place before dispatching (R12a.6).
 *
 * @param <Y> observation type
 */
final class BootstrapFk<Y> implements FeynmanKac<Y> {

    private final Transition<Y> m;
    private final Observation<Y> g;

    BootstrapFk(Transition<Y> m, Observation<Y> g) {
        this.m = m;
        this.g = g;
    }

    @Override
    public int dim() {
        return m.dim();
    }

    @Override
    public void init(StepContext<Y> ctx) {
        m.sampleM0(ctx);
        g.logG(ctx);
    }

    @Override
    public void advance(StepContext<Y> ctx) {
        m.sampleM(ctx);
        g.logG(ctx);
    }
}
