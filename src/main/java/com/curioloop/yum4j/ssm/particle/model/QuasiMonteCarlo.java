package com.curioloop.yum4j.ssm.particle.model;

/**
 * Capability trait for quasi-Monte Carlo (QMC) integration.
 *
 * <p>Exposes the uniform-stream access required by SQMC. Provides the
 * dimension of the QMC uniform input and deterministic transforms
 * {@code γ_0} and {@code γ_t} that map uniform draws to particle states.
 */
public interface QuasiMonteCarlo {

    /**
     * Returns the dimension of the QMC uniform input per particle.
     *
     * @return uniform input dimension (≥ 1)
     */
    int du();

    /**
     * Transforms initial QMC uniform draws into particle states.
     *
     * <p>Reads {@code N * du()} uniforms from {@code u} starting at {@code uOff}
     * and writes {@code N * dim} state values into {@code X} starting at {@code xOff}.
     *
     * @param u    uniform input buffer
     * @param uOff offset into {@code u}
     * @param X    output particle state buffer
     * @param xOff offset into {@code X}
     * @param N    number of particles
     */
    void gamma0(double[] u, int uOff, double[] X, int xOff, int N);

    /**
     * Transforms QMC uniform draws into particle states at time {@code t},
     * conditioned on the previous states.
     *
     * <p>Reads previous states from {@code Xprev} at {@code xpOff},
     * reads {@code N * du()} uniforms from {@code u} at {@code uOff},
     * and writes {@code N * dim} state values into {@code X} at {@code xOff}.
     *
     * @param t     time index
     * @param Xprev previous-step particle buffer
     * @param xpOff offset into {@code Xprev}
     * @param u     uniform input buffer
     * @param uOff  offset into {@code u}
     * @param X     output particle state buffer
     * @param xOff  offset into {@code X}
     * @param N     number of particles
     */
    void gamma(int t, double[] Xprev, int xpOff, double[] u, int uOff, double[] X, int xOff, int N);
}
