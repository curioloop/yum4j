package com.curioloop.yum4j.ssm.particle.sampler.nested;
import com.curioloop.yum4j.ssm.particle.sampler.StaticModel;
import com.curioloop.yum4j.ssm.particle.sampler.ThetaParticles;


import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Base class for vanilla nested sampling algorithms.
 *
 * <p>The algorithm maintains N "live" points sampled from the prior.
 * At each iteration the point with the lowest likelihood is removed
 * (becomes a "dead" point), and is replaced by mutating another live
 * point via MCMC, constrained so that the new point has likelihood
 * above the dead point's value.
 *
 * <p>The log-evidence is estimated incrementally using the standard
 * nested sampling weights: {@code w_i = exp(-i/N) - exp(-(i+1)/N)}.
 * The algorithm terminates when
 * {@code |logZ_hat[t] - logZ_hat[t-1]| < eps}.
 *
 * <p>Subclasses must implement {@link #mutate} to define the MCMC
 * strategy for replacing dead points.
 *
 * @see NestedRwMoves
 * @see NestedSamplingResult
 */
public abstract class NestedSampling {

    protected final StaticModel model;
    protected final int N;
    protected final double eps;

    /**
     * @param model static model defining prior and likelihood
     * @param N     number of live points
     * @param eps   stopping criterion on |delta logZ|
     */
    protected NestedSampling(StaticModel model, int N, double eps) {
        this.model = model;
        this.N = N;
        this.eps = eps;
    }

    /**
     * Mutate the dead point slot by starting from another live point
     * and applying constrained MCMC moves.
     *
     * <p>After this call, the particle at {@code deadIdx} in {@code x}
     * must have been replaced with a new point whose likelihood exceeds
     * the dead point's likelihood (the constraint threshold).
     *
     * @param x        live particles container
     * @param deadIdx  index of the dead point (to be overwritten)
     * @param startIdx index of the live point to start MCMC from
     * @param lmin     minimum log-likelihood threshold (new point must exceed this)
     * @param g        random number generator
     */
    public abstract void mutate(ThetaParticles x, int deadIdx, int startIdx,
                                double lmin, RandomGenerator g);

    /**
     * Run the nested sampling algorithm.
     *
     * @param g random number generator
     * @return the nested sampling result containing log-evidence and dead points
     */
    public NestedSamplingResult run(RandomGenerator g) {
        int dim = model.dim();

        // Setup: sample N points from the prior
        ThetaParticles x = ThetaParticles.allocate(dim, N);
        model.prior().sample(g, N, x.arena, 0, null);

        // Evaluate log-prior and log-likelihood for all live points
        int lpriorOff = x.lpriorOff();
        int llikOff = x.llikOff();
        double[] priorScratch = new double[dim * N];
        model.prior().logPdfBatch(x.arena, 0, N, x.arena, lpriorOff, priorScratch);
        model.loglik(x.arena, 0, N, -1, x.arena, llikOff);

        // Iteration state
        List<Double> logWeights = new ArrayList<>();
        List<double[]> deadPoints = new ArrayList<>();
        List<Double> logEvidencePath = new ArrayList<>();

        // First log-weight: log(1 - exp(-1/N))
        double firstLw = Math.log(1.0 - Math.exp(-1.0 / N));
        logWeights.add(firstLw);

        // First step
        int nLowest = argMin(x.arena, llikOff, N);
        deadPoints.add(extractPoint(x, nLowest, dim));
        double deadLlik = x.arena[llikOff + nLowest];

        // Replace dead point
        int startIdx = uniformMinusOne(N, nLowest, g);
        mutate(x, nLowest, startIdx, deadLlik, g);

        // First evidence estimate
        double logZ = firstLw + deadLlik;
        logEvidencePath.add(logZ);

        // Main loop
        int iter = 1;
        while (true) {
            iter++;

            // Find lowest likelihood point
            nLowest = argMin(x.arena, llikOff, N);
            deadPoints.add(extractPoint(x, nLowest, dim));
            deadLlik = x.arena[llikOff + nLowest];

            // Compute next log-weight: previous - 1/N
            double nextLw = logWeights.get(logWeights.size() - 1) - 1.0 / N;
            logWeights.add(nextLw);

            // Update log-evidence: logZ = log(exp(logZ) + exp(nextLw + deadLlik))
            double b = nextLw + deadLlik;
            logZ = logSumExpAB(logZ, b);
            logEvidencePath.add(logZ);

            // Replace dead point
            startIdx = uniformMinusOne(N, nLowest, g);
            mutate(x, nLowest, startIdx, deadLlik, g);

            // Stopping criterion
            if (logEvidencePath.size() >= 2) {
                double prev = logEvidencePath.get(logEvidencePath.size() - 2);
                double curr = logEvidencePath.get(logEvidencePath.size() - 1);
                if (Math.abs(curr - prev) < eps) {
                    break;
                }
            }
        }

        return new NestedSamplingResult(logZ, logEvidencePath, logWeights, deadPoints, iter);
    }

    /** Sample uniformly from {0, ..., N-1} \ {m}. */
    private static int uniformMinusOne(int N, int m, RandomGenerator g) {
        int r = g.nextInt(m + 1, m + N);
        return r % N;
    }

    /** Find the index of the minimum value in arena[off..off+n). */
    private static int argMin(double[] arena, int off, int n) {
        int idx = 0;
        double min = arena[off];
        for (int i = 1; i < n; i++) {
            if (arena[off + i] < min) {
                min = arena[off + i];
                idx = i;
            }
        }
        return idx;
    }

    /** Extract a copy of the theta vector for particle n. */
    private static double[] extractPoint(ThetaParticles x, int n, int dim) {
        double[] point = new double[dim];
        for (int j = 0; j < dim; j++) {
            point[j] = x.arena[j * x.N + n];
        }
        return point;
    }

    /** log(exp(a) + exp(b)) computed stably. */
    private static double logSumExpAB(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) return b;
        if (b == Double.NEGATIVE_INFINITY) return a;
        double m = Math.max(a, b);
        return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
    }
}
