package com.curioloop.yum4j.ssm.particle;

/**
 * Selects the importance-weighting strategy for the particle filter.
 *
 * <p>Each value corresponds to a different {@code FeynmanKac} factory:
 * <ul>
 *   <li>{@link #BOOTSTRAP} — proposal equals the transition kernel;
 *       requires only {@code Transition} and {@code Observation}.</li>
 *   <li>{@link #GUIDED} — user-supplied proposal with importance
 *       correction; requires {@code Proposal} and
 *       {@code TransitionDensity}.</li>
 *   <li>{@link #AUXILIARY} — auxiliary particle filter (Pitt–Shephard);
 *       requires {@code AuxWeight}.</li>
 * </ul>
 *
 */
public enum Weighting {

    /** Bootstrap particle filter: proposal = transition. */
    BOOTSTRAP,

    /** Guided particle filter: user-supplied proposal. */
    GUIDED,

    /** Auxiliary particle filter (Pitt–Shephard). */
    AUXILIARY
}
