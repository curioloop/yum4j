package com.curioloop.yum4j.ssm.particle.filter;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.model.Observation;
import com.curioloop.yum4j.ssm.particle.model.Proposal;
import com.curioloop.yum4j.ssm.particle.model.Transition;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;

/**
 * Guided particle filter implementation of {@link FeynmanKac}.
 *
 * <p>Uses a user-supplied {@link Proposal} for sampling and corrects the
 * importance weights using the transition density. The fused advance kernel
 * computes {@code logW = logG + logPt - logQ} in a single pass over
 * {@code n ∈ [0, N)}.
 *
 * <p>Requires the model to provide both {@link Proposal} and
 * {@link TransitionDensity} traits; validated at factory construction time.
 *
 * <p>Reads the current observation from {@link StepContext#observation()}; the
 * engine writes it in place before dispatching (R12a.6).
 *
 * @param <Y> observation type
 */
final class GuidedFk<Y> implements FeynmanKac<Y> {

    private final Transition<Y> m;
    private final Observation<Y> g;
    private final Proposal<Y> q;
    private final TransitionDensity<Y> pt;

    GuidedFk(Transition<Y> m, Observation<Y> g, Proposal<Y> q, TransitionDensity<Y> pt) {
        this.m = m;
        this.g = g;
        this.q = q;
        this.pt = pt;
    }

    @Override
    public int dim() {
        return m.dim();
    }

    @Override
    public void init(StepContext<Y> ctx) {
        int N = ctx.N();
        int lwOff = ctx.lwOff();
        int scrOff = ctx.scratchOff();
        double[] logW = ctx.logW();
        double[] scratch = ctx.scratch();

        // Sample from proposal Q_0
        q.sampleQ0(ctx);

        // Compute logG (observation log-density) — writes into logW
        g.logG(ctx);

        // Compute logPt0 (initial transition log-density) into scratch
        pt.logPt0(ctx, scratch, scrOff);

        // Compute logQ0 (proposal log-density) into scratch offset by N
        q.logQ0(ctx, scratch, scrOff + N);

        // Fused weight correction: logW[n] = logG[n] + logPt0[n] - logQ0[n]
        for (int n = 0; n < N; n++) {
            logW[lwOff + n] += scratch[scrOff + n] - scratch[scrOff + N + n];
        }
    }

    @Override
    public void advance(StepContext<Y> ctx) {
        int N = ctx.N();
        int lwOff = ctx.lwOff();
        int scrOff = ctx.scratchOff();
        double[] logW = ctx.logW();
        double[] scratch = ctx.scratch();

        // Sample from proposal Q_t
        q.sampleQ(ctx);

        // Compute logG (observation log-density) — writes into logW
        g.logG(ctx);

        // Compute logPt (transition log-density) into scratch
        pt.logPt(ctx, scratch, scrOff);

        // Compute logQ (proposal log-density) into scratch offset by N
        q.logQ(ctx, scratch, scrOff + N);

        // Fused weight correction: logW[n] = logG[n] + logPt[n] - logQ[n]
        for (int n = 0; n < N; n++) {
            logW[lwOff + n] += scratch[scrOff + n] - scratch[scrOff + N + n];
        }
    }
}
