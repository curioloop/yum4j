package com.curioloop.yum4j.ssm.particle.mcmc;

import com.curioloop.yum4j.ssm.particle.dist.MultivariateDistribution;

import java.util.random.RandomGenerator;

/**
 * Abstract random-walk Hastings-Metropolis (RWHM) sampler with optional
 * adaptive covariance tracking via {@link VanishCovTracker}.
 *
 * <p>Subclasses implement {@link #computeLogPost(double[])} to define
 * the target log-posterior. The base class handles the proposal loop,
 * acceptance/rejection, and covariance adaptation.
 */
public abstract class GenericRWHM {

    protected final MultivariateDistribution prior;
    protected final int dim;
    protected final int niter;
    protected final boolean adaptive;
    protected final double alpha;

    /** Vanishing covariance tracker for adaptive proposals. */
    protected VanishCovTracker covTracker;

    /** Current Cholesky factor for proposals (dim×dim row-major). */
    protected double[] L;

    /** Scale factor for the proposal: 2.38/√d. */
    protected final double scale;

    /** Reusable per-run scratch (grown lazily in {@link #run}). */
    protected double[] priorOut;
    protected double[] priorScratch;
    protected double[] thetaCurrBuf;
    protected double[] thetaPropBuf;
    protected double[] zBuf;

    protected GenericRWHM(MultivariateDistribution prior, int niter,
                          boolean adaptive, double alpha) {
        this.prior = prior;
        this.dim = prior.dim();
        this.niter = niter;
        this.adaptive = adaptive;
        this.alpha = alpha;
        this.scale = 2.38 / Math.sqrt(dim);

        if (adaptive) {
            this.covTracker = new VanishCovTracker(alpha, dim, null);
        }
        // Initial L = scale * I
        this.L = new double[dim * dim];
        for (int i = 0; i < dim; i++) {
            L[i * dim + i] = scale;
        }
    }

    protected GenericRWHM(MultivariateDistribution prior, int niter) {
        this(prior, niter, true, 0.6);
    }

    /**
     * Compute the log-posterior (or log-target) at the proposed theta.
     * Subclasses define this (e.g. PMMH runs a particle filter here).
     *
     * @param thetaProposed the proposed parameter vector (length dim)
     * @return log-posterior value (may be -∞ for rejection)
     */
    protected abstract double computeLogPost(double[] thetaProposed);

    /**
     * Run the MCMC chain.
     *
     * @param theta0 initial parameter vector (length dim)
     * @param g      random number generator
     * @return the MCMC chain result
     */
    public MCMCChain run(double[] theta0, RandomGenerator g) {
        MCMCChain chain = new MCMCChain(dim, niter);

        if (thetaCurrBuf == null || thetaCurrBuf.length < dim) {
            thetaCurrBuf = new double[dim];
            thetaPropBuf = new double[dim];
            zBuf = new double[dim];
            priorOut = new double[1];
            priorScratch = new double[dim];
        }
        final double[] thetaCurr = thetaCurrBuf;
        final double[] thetaProp = thetaPropBuf;
        final double[] z = zBuf;

        System.arraycopy(theta0, 0, thetaCurr, 0, dim);

        // Compute initial log-posterior
        double lpostCurr = computeLogPost(thetaCurr);

        int accepted = 0;

        for (int iter = 0; iter < niter; iter++) {
            // Propose: thetaProp = thetaCurr + L * z
            for (int j = 0; j < dim; j++) {
                z[j] = g.nextGaussian();
            }
            for (int j = 0; j < dim; j++) {
                double shift = 0.0;
                int rowOff = j * dim;
                for (int k = 0; k <= j; k++) {
                    shift += L[rowOff + k] * z[k];
                }
                thetaProp[j] = thetaCurr[j] + shift;
            }

            // Check prior first (short-circuit if -∞)
            prior.logPdfBatch(thetaProp, 0, 1, priorOut, 0, priorScratch);
            double lpriorProp = priorOut[0];

            double lpostProp;
            if (lpriorProp == Double.NEGATIVE_INFINITY) {
                lpostProp = Double.NEGATIVE_INFINITY;
            } else {
                lpostProp = computeLogPost(thetaProp);
            }

            // Accept/reject
            double logAlpha = lpostProp - lpostCurr;
            if (Math.log(g.nextDouble()) < logAlpha) {
                System.arraycopy(thetaProp, 0, thetaCurr, 0, dim);
                lpostCurr = lpostProp;
                accepted++;
            }

            // Record
            chain.setTheta(iter, thetaCurr, 0);
            chain.arena[chain.LPOST_OFF + iter] = lpostCurr;

            // Adaptive covariance update
            if (adaptive) {
                covTracker.update(thetaCurr, 0);
                // Update L from tracker (scaled by 2.38/√d)
                double[] trackerL = covTracker.choleskyL();
                for (int i = 0; i < dim * dim; i++) {
                    L[i] = scale * trackerL[i];
                }
            }
        }

        chain.accRate = (double) accepted / niter;
        return chain;
    }
}
