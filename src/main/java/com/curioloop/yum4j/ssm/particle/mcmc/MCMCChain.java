package com.curioloop.yum4j.ssm.particle.mcmc;

/**
 * Result container for MCMC chains (PMMH, Particle Gibbs, etc.).
 *
 * <p>Stores the full chain of parameter samples and log-posterior values
 * in a single contiguous arena, plus the overall acceptance rate.
 *
 * <h3>Arena layout</h3>
 * <pre>
 *   arena: [theta: dim*niter | lpost: niter]  (total (dim+1)*niter)
 * </pre>
 * <p>Theta is column-major (dim, niter): {@code arena[j*niter + i]} = param j at iteration i.
 * Lpost occupies {@code arena[LPOST_OFF + i]}.
 */
public final class MCMCChain {

    /** Parameter dimension. */
    public final int dim;

    /** Number of iterations. */
    public final int niter;

    /** Unified arena: [theta: dim*niter | lpost: niter]. Length = (dim+1)*niter. */
    public final double[] arena;

    /** Offset of lpost within {@link #arena}. Equal to dim*niter. */
    public final int LPOST_OFF;

    /** Overall acceptance rate. */
    public double accRate;

    public MCMCChain(int dim, int niter) {
        this.dim = dim;
        this.niter = niter;
        this.arena = new double[(dim + 1) * niter];
        this.LPOST_OFF = dim * niter;
        this.accRate = 0.0;
    }

    /**
     * Set the parameter vector at iteration i.
     */
    public void setTheta(int i, double[] thetaVec, int off) {
        for (int j = 0; j < dim; j++) {
            arena[j * niter + i] = thetaVec[off + j];
        }
    }

    /**
     * Get the parameter vector at iteration i into out.
     */
    public void getTheta(int i, double[] out, int off) {
        for (int j = 0; j < dim; j++) {
            out[off + j] = arena[j * niter + i];
        }
    }
}
