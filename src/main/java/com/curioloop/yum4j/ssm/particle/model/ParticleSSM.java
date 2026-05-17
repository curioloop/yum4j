package com.curioloop.yum4j.ssm.particle.model;

/**
 * Aggregate base class for state-space models.
 *
 * <p>Composes the six capability traits ({@link Transition}, {@link Observation},
 * {@link Proposal}, {@link TransitionDensity}, {@link AuxWeight}, {@link QuasiMonteCarlo})
 * into a single model object. Concrete subclasses must implement {@link Transition}
 * and {@link Observation}; the remaining four traits are optional and return
 * {@code null} by default, indicating the model does not provide that capability.
 *
 * <p>Factory methods on {@code FeynmanKac} inspect these accessors at construction
 * time and throw {@link IllegalArgumentException} if a required trait is missing.
 *
 * @param <Y> observation type
 */
public abstract class ParticleSSM<Y> implements Transition<Y>, Observation<Y> {

    /**
     * Returns the proposal distribution, or {@code null} if this model does not
     * provide a custom proposal.
     *
     * @return proposal capability, or {@code null}
     */
    public Proposal<Y> proposal() {
        return null;
    }

    /**
     * Returns the transition density evaluator, or {@code null} if this model
     * does not provide closed-form transition density evaluation.
     *
     * @return transition density capability, or {@code null}
     */
    public TransitionDensity<Y> transitionDensity() {
        return null;
    }

    /**
     * Returns the auxiliary weight evaluator, or {@code null} if this model
     * does not provide auxiliary pre-weights.
     *
     * @return auxiliary weight capability, or {@code null}
     */
    public AuxWeight<Y> auxWeight() {
        return null;
    }

    /**
     * Returns the quasi-Monte Carlo transform, or {@code null} if this model
     * does not support QMC integration.
     *
     * @return QMC capability, or {@code null}
     */
    public QuasiMonteCarlo qmc() {
        return null;
    }
}
