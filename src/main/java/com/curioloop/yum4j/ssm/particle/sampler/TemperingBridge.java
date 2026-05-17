package com.curioloop.yum4j.ssm.particle.sampler;

import com.curioloop.yum4j.ssm.particle.dist.MultivariateDistribution;

/**
 * Bridge model for tempering SMC when the user wants to sample from an
 * arbitrary target distribution (not a sequential data model).
 *
 * <p>The "likelihood" is defined as {@code logTarget(θ) − baseDist.logPdf(θ)}
 * so that the log-posterior becomes:
 * <pre>
 *   logpost = baseDist.logPdf(θ) + loglik(θ)
 *           = baseDist.logPdf(θ) + logTarget(θ) − baseDist.logPdf(θ)
 *           = logTarget(θ)
 * </pre>
 *
 * <p>This allows tempering to bridge from the base distribution (prior) to
 * the user-defined target via the exponent schedule γ_t ∈ [0, 1]:
 * <pre>
 *   π_t(θ) ∝ baseDist(θ) · [target(θ) / baseDist(θ)]^{γ_t}
 * </pre>
 *
 * <p>Since there is no sequential data, {@code T = 1} (a single virtual
 * "observation" representing the entire target).
 *
 * @see StaticModel
 */
public abstract class TemperingBridge extends StaticModel {

    /** Scratch buffer for prior logPdf subtraction (length N). */
    private double[] priorBuf;

    /** Scratch buffer for prior logPdfBatch computation (length dim * N). */
    private double[] priorScratchBuf;

    /**
     * Constructs a tempering bridge with the given base distribution.
     *
     * @param baseDist the base distribution (prior) from which tempering starts
     */
    public TemperingBridge(MultivariateDistribution baseDist) {
        super(baseDist, 1);  // T=1: single virtual observation
    }

    /**
     * User-defined log-target density evaluated in batch.
     *
     * <p>Implementations write one log-density value per particle into
     * {@code out[outOff + n]} for {@code n in [0, N)}.
     *
     * @param theta    parameter buffer, column-major (dim, N)
     * @param thetaOff offset into theta
     * @param N        number of particles
     * @param out      output buffer (length ≥ outOff + N)
     * @param outOff   output offset
     */
    public abstract void logTarget(double[] theta, int thetaOff, int N,
                                   double[] out, int outOff);

    /**
     * Computes {@code logTarget(θ) − baseDist.logPdf(θ)} for each particle.
     *
     * <p>The time index {@code t} is ignored since T = 1.
     */
    @Override
    public void logpyt(double[] theta, int thetaOff, int N, int t,
                       double[] out, int outOff) {
        // Compute logTarget into out
        logTarget(theta, thetaOff, N, out, outOff);

        // Ensure scratch buffers are large enough
        int dim = dim();
        if (priorBuf == null || priorBuf.length < N) {
            priorBuf = new double[N];
        }
        if (priorScratchBuf == null || priorScratchBuf.length < dim * N) {
            priorScratchBuf = new double[dim * N];
        }

        // Compute baseDist.logPdf into priorBuf
        prior.logPdfBatch(theta, thetaOff, N, priorBuf, 0, priorScratchBuf);

        // Subtract: out[n] = logTarget[n] - baseDist.logPdf[n]
        for (int n = 0; n < N; n++) {
            out[outOff + n] -= priorBuf[n];
        }
    }

    /**
     * Overrides {@code loglik} to avoid the cumulative-sum loop.
     *
     * <p>Since T = 1, the log-likelihood is simply {@code logpyt(θ, 0)}.
     * This override is cleaner and avoids the scratch-buffer allocation
     * in the base class's loop.
     */
    @Override
    public void loglik(double[] theta, int thetaOff, int N, int t,
                       double[] out, int outOff) {
        logpyt(theta, thetaOff, N, 0, out, outOff);
    }
}
