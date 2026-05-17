package com.curioloop.yum4j.ssm.particle.mcmc;

/**
 * MCMC diagnostic utilities including Mean Squared Jumping Distance (MSJD).
 */
public final class MCMCDiagnostics {

    private MCMCDiagnostics() {}

    /**
     * Compute the Mean Squared Jumping Distance (MSJD) for a chain.
     *
     * <p>MSJD = (1/(niter-1)) * Σ_t Σ_j (θ_j[t] − θ_j[t−1])²
     *
     * @param chain the MCMC chain
     * @return MSJD value
     */
    public static double msjd(MCMCChain chain) {
        return msjd(chain, 0.0);
    }

    /**
     * Compute MSJD after discarding a fraction of initial iterations.
     *
     * @param chain       the MCMC chain
     * @param discardFrac fraction of iterations to discard (burn-in)
     * @return MSJD value
     */
    public static double msjd(MCMCChain chain, double discardFrac) {
        int niter = chain.niter;
        int dim = chain.dim;
        int start = (int) (discardFrac * niter);
        if (start >= niter - 1) return 0.0;

        int nPairs = niter - start - 1;
        if (nPairs <= 0) return 0.0;

        double sumSqDist = 0.0;
        for (int t = start + 1; t < niter; t++) {
            for (int j = 0; j < dim; j++) {
                double diff = chain.arena[j * niter + t] - chain.arena[j * niter + (t - 1)];
                sumSqDist += diff * diff;
            }
        }

        return sumSqDist / nPairs;
    }

    /**
     * Compute the mean acceptance rate from a chain.
     */
    public static double acceptanceRate(MCMCChain chain) {
        return chain.accRate;
    }
}
