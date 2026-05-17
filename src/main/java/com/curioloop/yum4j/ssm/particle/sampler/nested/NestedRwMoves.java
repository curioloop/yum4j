package com.curioloop.yum4j.ssm.particle.sampler.nested;
import com.curioloop.yum4j.ssm.particle.sampler.StaticModel;
import com.curioloop.yum4j.ssm.particle.sampler.ThetaParticles;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Nested sampling with adaptive random-walk Metropolis moves.
 *
 * <p>At each mutation step, the dead point is replaced by starting from
 * a randomly chosen live point and applying {@code nsteps} constrained
 * random-walk Metropolis-Hastings steps. A step is accepted only if:
 * <ol>
 *   <li>The proposed point has log-likelihood above the threshold
 *       (the dead point's likelihood).</li>
 *   <li>The Metropolis ratio {@code exp(lprior_prop - lprior_curr)}
 *       passes the uniform test.</li>
 * </ol>
 *
 * <p>The proposal covariance is adapted by tracking the mean and
 * covariance of the live points (similar to {@code MeanCovTracker} in
 * the reference). The Cholesky factor of the sample covariance is used
 * to generate proposals: {@code x_prop = x_curr + scale * L * z}
 * where {@code z ~ N(0, I)}.
 *
 * @see NestedSampling
 */
public final class NestedRwMoves extends NestedSampling {

    private final int nsteps;
    private final double scale;

    // Tracker state: mean, covariance, Cholesky factor
    private double[] mean;       // length dim
    private double[] cov;        // dim×dim row-major
    private double[] choleskyL;  // dim×dim row-major lower-triangular
    private int trackerN;        // number of points tracked
    private double[] sumX;       // sum of points, length dim
    private double[] sumXXT;     // sum of outer products, dim×dim row-major

    // Scratch buffers
    private double[] priorScratch;
    private double[] priorOut;
    private double[] llikScratch;

    /**
     * @param model  static model defining prior and likelihood
     * @param N      number of live points
     * @param eps    stopping criterion
     * @param nsteps number of RW-MH steps per mutation
     * @param scale  proposal scale factor (default: 2.38 / sqrt(dim))
     */
    public NestedRwMoves(StaticModel model, int N, double eps, int nsteps, double scale) {
        super(model, N, eps);
        this.nsteps = nsteps;
        this.scale = scale;
    }

    /**
     * Convenience constructor with default scale = 2.38 / sqrt(dim).
     */
    public NestedRwMoves(StaticModel model, int N, double eps, int nsteps) {
        super(model, N, eps);
        this.nsteps = nsteps;
        this.scale = 2.38 / Math.sqrt(model.dim());
    }

    @Override
    public NestedSamplingResult run(RandomGenerator g) {
        // Initialize tracker before running
        initTracker();
        return super.run(g);
    }

    private void initTracker() {
        int dim = model.dim();
        mean = new double[dim];
        cov = new double[dim * dim];
        choleskyL = new double[dim * dim];
        sumX = new double[dim];
        sumXXT = new double[dim * dim];
        trackerN = 0;
        priorScratch = new double[dim];
        priorOut = new double[1];
        llikScratch = new double[1];
    }

    /**
     * Initialize the tracker from the current live points.
     * Called lazily on first mutation.
     */
    private void initTrackerFromPoints(ThetaParticles x) {
        int dim = model.dim();
        int n = x.N;
        trackerN = n;

        // Compute sum and sum of outer products
        Arrays.fill(sumX, 0.0);
        Arrays.fill(sumXXT, 0.0);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                double vj = x.arena[j * n + i];
                sumX[j] += vj;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                double vj = x.arena[j * n + i];
                for (int k = 0; k <= j; k++) {
                    double vk = x.arena[k * n + i];
                    sumXXT[j * dim + k] += vj * vk;
                }
            }
        }
        // Fill upper triangle
        for (int j = 0; j < dim; j++) {
            for (int k = j + 1; k < dim; k++) {
                sumXXT[j * dim + k] = sumXXT[k * dim + j];
            }
        }

        updateMeanCov();
    }

    private void removePoint(ThetaParticles x, int idx) {
        int dim = model.dim();
        int n = x.N;
        trackerN--;
        for (int j = 0; j < dim; j++) {
            double vj = x.arena[j * n + idx];
            sumX[j] -= vj;
            for (int k = 0; k < dim; k++) {
                double vk = x.arena[k * n + idx];
                sumXXT[j * dim + k] -= vj * vk;
            }
        }
        updateMeanCov();
    }

    private void addPoint(ThetaParticles x, int idx) {
        int dim = model.dim();
        int n = x.N;
        trackerN++;
        for (int j = 0; j < dim; j++) {
            double vj = x.arena[j * n + idx];
            sumX[j] += vj;
            for (int k = 0; k < dim; k++) {
                double vk = x.arena[k * n + idx];
                sumXXT[j * dim + k] += vj * vk;
            }
        }
        updateMeanCov();
    }

    private void updateMeanCov() {
        int dim = model.dim();
        if (trackerN <= 0) return;

        double invN = 1.0 / trackerN;
        for (int j = 0; j < dim; j++) {
            mean[j] = sumX[j] * invN;
        }

        // cov = sumXXT/N - mean * mean^T
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                cov[j * dim + k] = sumXXT[j * dim + k] * invN - mean[j] * mean[k];
            }
        }

        // Attempt Cholesky decomposition
        if (!choleskyDecompose(cov, choleskyL, dim)) {
            // Fall back to identity if not PD
            Arrays.fill(choleskyL, 0.0);
            for (int i = 0; i < dim; i++) {
                choleskyL[i * dim + i] = 1.0;
            }
        }
    }

    @Override
    public void mutate(ThetaParticles x, int deadIdx, int startIdx,
                       double lmin, RandomGenerator g) {
        int dim = model.dim();
        int n = x.N;
        int llikOff = x.llikOff();
        int lpriorOff = x.lpriorOff();
        int lpostOff = x.lpostOff();

        // Lazy initialization of tracker
        if (trackerN == 0) {
            initTrackerFromPoints(x);
        }

        // Remove the dead point from the tracker
        removePoint(x, deadIdx);

        // Copy start point into dead slot
        x.copyAt(deadIdx, x, startIdx);

        // Ensure scratch buffers are sized for single-point evaluation
        if (priorScratch == null || priorScratch.length < dim) {
            priorScratch = new double[dim];
        }
        if (priorOut == null || priorOut.length < 1) {
            priorOut = new double[1];
        }
        if (llikScratch == null || llikScratch.length < 1) {
            llikScratch = new double[1];
        }

        // Perform nsteps constrained RW-MH steps
        double[] proposed = new double[dim];
        double[] z = new double[dim];

        for (int step = 0; step < nsteps; step++) {
            // Generate proposal: x_prop = x_curr + scale * L * z
            for (int j = 0; j < dim; j++) {
                z[j] = g.nextGaussian();
            }

            // Compute L * z
            for (int j = 0; j < dim; j++) {
                double sum = 0.0;
                for (int k = 0; k <= j; k++) {
                    sum += choleskyL[j * dim + k] * z[k];
                }
                proposed[j] = x.arena[j * n + deadIdx] + scale * sum;
            }

            // Pack proposed into a single-particle buffer for evaluation
            double[] propBuf = new double[dim]; // single particle, N=1
            for (int j = 0; j < dim; j++) {
                propBuf[j] = proposed[j];
            }

            // Evaluate log-likelihood of proposal
            model.loglik(propBuf, 0, 1, -1, llikScratch, 0);
            double propLlik = llikScratch[0];

            // Constraint check: must exceed lmin
            if (propLlik <= lmin) continue;

            // Evaluate log-prior of proposal
            model.prior().logPdfBatch(propBuf, 0, 1, priorOut, 0, null);
            double propLprior = priorOut[0];

            // Metropolis ratio: exp(lprior_prop - lprior_curr)
            double logAlpha = propLprior - x.arena[lpriorOff + deadIdx];
            if (Math.log(g.nextDouble()) < logAlpha) {
                // Accept: update particle
                for (int j = 0; j < dim; j++) {
                    x.arena[j * n + deadIdx] = proposed[j];
                }
                x.arena[llikOff + deadIdx] = propLlik;
                x.arena[lpriorOff + deadIdx] = propLprior;
                x.arena[lpostOff + deadIdx] = propLprior + propLlik;
            }
        }

        // Add the (possibly updated) point back to the tracker
        addPoint(x, deadIdx);
    }

    /**
     * In-place Cholesky decomposition of a symmetric PD matrix.
     *
     * @return true if successful, false if not positive definite
     */
    private static boolean choleskyDecompose(double[] src, double[] dst, int dim) {
        Arrays.fill(dst, 0.0);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = src[i * dim + j];
                for (int k = 0; k < j; k++) {
                    sum -= dst[i * dim + k] * dst[j * dim + k];
                }
                if (i == j) {
                    if (sum <= 0.0) return false;
                    dst[i * dim + j] = Math.sqrt(sum);
                } else {
                    dst[i * dim + j] = sum / dst[j * dim + j];
                }
            }
        }
        return true;
    }
}
