package com.curioloop.yum4j.ssm.particle.filter;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.model.*;

import java.util.List;

/**
 * The single engine-facing contract for a particle filter run.
 *
 * <p>A {@code FeynmanKac} encapsulates the complete specification of a sequential
 * Monte Carlo algorithm: how to initialise particles at time 0 and how to advance
 * them at each subsequent time step. The engine calls {@link #init(StepContext)}
 * exactly once, then {@link #advance(StepContext)} for each step {@code t = 1, ..., T-1}.
 *
 * <p>Implementations are obtained via the static factory methods
 * {@link #bootstrap(ParticleSSM)}, {@link #guided(ParticleSSM)}, and
 * {@link #auxiliary(ParticleSSM, List)}, which compose capability traits from the
 * supplied {@link ParticleSSM} into a fused advance kernel.
 *
 * <p>As of particle-v2-perf (R12a, R8.3), the observation sequence is no longer
 * owned by the {@code FeynmanKac}. The engine threads a {@code List<Y> data} into
 * {@code Engine.run/init/step} and writes the current observation into
 * {@code ctx.observation} before dispatching; kernels read it via
 * {@link StepContext#observation()}. The horizon comes from {@code data.size()} at
 * the call site. The one exception is the auxiliary filter, which genuinely needs
 * a one-step lookahead {@code y_{t+1}}; it therefore retains the observation list
 * as a constructor argument (see {@link #auxiliary(ParticleSSM, List)}).
 *
 * @param <Y> observation type
 */
public interface FeynmanKac<Y> {

    /**
     * Returns the state dimension per particle.
     *
     * @return state dimension (≥ 1)
     */
    int dim();

    /**
     * Initialises particles at time 0. Writes both the state buffer {@code ctx.X()}
     * and the log-weight buffer {@code ctx.logW()} in a single fused pass over
     * {@code n ∈ [0, N)}.
     *
     * @param ctx step context carrying buffers, observation, and RNG
     */
    void init(StepContext<Y> ctx);

    /**
     * Advances particles from time {@code t-1} to time {@code t}. Writes both
     * the state buffer {@code ctx.X()} and the log-weight buffer {@code ctx.logW()}
     * in a single fused pass over {@code n ∈ [0, N)}.
     *
     * @param ctx step context carrying buffers, observation, and RNG
     */
    void advance(StepContext<Y> ctx);

    /**
     * Creates a bootstrap particle filter from the given state-space model.
     *
     * <p>The bootstrap filter uses the transition kernel as the proposal and the
     * observation density as the importance weight. This is the simplest and most
     * common particle filter variant.
     *
     * <p>The observation sequence is supplied at {@code Engine.run(ws, rs, fk, par, data)}
     * time; the returned FK is therefore reusable across runs with different data.
     *
     * @param ssm the state-space model (must implement {@link Transition} and {@link Observation})
     * @param <Y> observation type
     * @return a bootstrap FeynmanKac instance
     */
    static <Y> FeynmanKac<Y> bootstrap(ParticleSSM<Y> ssm) {
        return new BootstrapFk<>(ssm, ssm);
    }

    /**
     * Creates a guided particle filter from the given state-space model.
     *
     * <p>The guided filter uses a user-supplied {@link Proposal} for sampling and
     * corrects the importance weights using the {@link TransitionDensity}. This
     * requires the model to provide both traits.
     *
     * <p>The observation sequence is supplied at {@code Engine.run} time.
     *
     * @param ssm the state-space model
     * @param <Y> observation type
     * @return a guided FeynmanKac instance
     * @throws IllegalArgumentException if the model does not provide {@link Proposal} or {@link TransitionDensity}
     */
    static <Y> FeynmanKac<Y> guided(ParticleSSM<Y> ssm) {
        Proposal<Y> q = ssm.proposal();
        TransitionDensity<Y> pt = ssm.transitionDensity();
        if (q == null) {
            throw new IllegalArgumentException(
                    "guided(...) requires Proposal<Y>; " + ssm.getClass().getName() + " returns null");
        }
        if (pt == null) {
            throw new IllegalArgumentException(
                    "guided(...) requires TransitionDensity<Y>; " + ssm.getClass().getName() + " returns null");
        }
        return new GuidedFk<>(ssm, ssm, q, pt);
    }

    /**
     * Creates an auxiliary particle filter from the given state-space model and data.
     *
     * <p>The auxiliary filter uses {@link AuxWeight} pre-weights to adjust the
     * resampling step before the transition. This requires the model to provide
     * the {@link AuxWeight} trait.
     *
     * <p>Unlike {@link #bootstrap(ParticleSSM)} and {@link #guided(ParticleSSM)}, the auxiliary
     * filter genuinely needs a one-step lookahead: the pre-weight {@code logEta(X_t, y_{t+1})}
     * is added to the weights at step {@code t} for the next step's resampling.
     * The observation list is therefore retained on the FK for the lookahead;
     * the current observation is still read from {@code ctx.observation} (the
     * engine writes it before dispatching).
     *
     * @param ssm  the state-space model
     * @param data the observation sequence (used only for the {@code y_{t+1}} lookahead)
     * @param <Y>  observation type
     * @return an auxiliary FeynmanKac instance
     * @throws IllegalArgumentException if the model does not provide {@link AuxWeight}
     */
    static <Y> FeynmanKac<Y> auxiliary(ParticleSSM<Y> ssm, List<Y> data) {
        AuxWeight<Y> aux = ssm.auxWeight();
        if (aux == null) {
            throw new IllegalArgumentException(
                    "auxiliary(...) requires AuxWeight<Y>; " + ssm.getClass().getName() + " returns null");
        }
        return new AuxiliaryFk<>(ssm, ssm, aux, data);
    }
}
