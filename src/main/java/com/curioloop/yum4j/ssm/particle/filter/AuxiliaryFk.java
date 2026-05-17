package com.curioloop.yum4j.ssm.particle.filter;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.model.AuxWeight;
import com.curioloop.yum4j.ssm.particle.model.Observation;
import com.curioloop.yum4j.ssm.particle.model.Transition;

import java.util.List;

/**
 * Auxiliary particle filter implementation of {@link FeynmanKac}.
 *
 * <p>Uses {@link AuxWeight} pre-weights to adjust the resampling weights
 * before the transition step. The auxiliary filter adds {@code logEta(X_t, y_{t+1})}
 * to the current weights (for the next step's resampling) and subtracts the
 * previous step's {@code logEta} contribution after advancing.
 *
 * <p>The fused advance kernel writes both X and logW in a single pass over
 * {@code n ∈ [0, N)}.
 *
 * <p>Reads the current observation from {@link StepContext#observation()}. Unlike
 * {@link BootstrapFk} and {@link GuidedFk}, this variant retains the full
 * observation list as a constructor argument for the one-step lookahead
 * {@code y_{t+1}} needed to build the auxiliary pre-weight.
 *
 * @param <Y> observation type
 */
final class AuxiliaryFk<Y> implements FeynmanKac<Y> {

    private final Transition<Y> m;
    private final Observation<Y> g;
    private final AuxWeight<Y> aux;
    private final List<Y> data;

    AuxiliaryFk(Transition<Y> m, Observation<Y> g, AuxWeight<Y> aux, List<Y> data) {
        this.m = m;
        this.g = g;
        this.aux = aux;
        this.data = data;
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

        // Sample from initial distribution M_0
        m.sampleM0(ctx);

        // Compute logG (observation log-density) — writes into logW
        g.logG(ctx);

        // If there is a next observation, add auxiliary pre-weight logEta(X_0, y_1)
        if (data.size() > 1) {
            Y yNext = data.get(1);
            aux.logEta(ctx, yNext, scratch, scrOff);
            for (int n = 0; n < N; n++) {
                logW[lwOff + n] += scratch[scrOff + n];
            }
        }
    }

    @Override
    public void advance(StepContext<Y> ctx) {
        int t = ctx.t();
        int N = ctx.N();
        int lwOff = ctx.lwOff();
        int scrOff = ctx.scratchOff();
        double[] logW = ctx.logW();
        double[] scratch = ctx.scratch();

        // Compute logEta from previous step (on Xprev) to subtract.
        // This is the pre-weight that was added at step t-1 for y_t (the current obs).
        Y yCurrent = ctx.observation();
        aux.logEta(ctx, yCurrent, scratch, scrOff);

        // Sample from transition kernel M_t
        m.sampleM(ctx);

        // Compute logG (observation log-density) — writes into logW
        g.logG(ctx);

        // Subtract the previous step's auxiliary pre-weight (already in scratch[scrOff..])
        for (int n = 0; n < N; n++) {
            logW[lwOff + n] -= scratch[scrOff + n];
        }

        // If there is a next observation, add auxiliary pre-weight logEta(X_t, y_{t+1})
        if (t + 1 < data.size()) {
            Y yNext = data.get(t + 1);
            aux.logEta(ctx, yNext, scratch, scrOff);
            for (int n = 0; n < N; n++) {
                logW[lwOff + n] += scratch[scrOff + n];
            }
        }
    }
}
