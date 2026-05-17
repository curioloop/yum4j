package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.ParticleOps;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Multivariate stochastic volatility state-space model with d independent
 * log-volatility processes.
 *
 * <pre>
 *   X_0[k]            ~ N(mu[k], sigma0[k]),   sigma0[k] = sigma[k] / sqrt(1 - rho[k]²)
 *   X_t[k] | X_{t-1}  = mu[k] + rho[k] * (X_{t-1}[k] - mu[k]) + sigma[k] * N(0,1)
 *   Y_t[k] | X_t[k]   ~ N(0, exp(X_t[k] / 2))
 * </pre>
 *
 * <p>The observation log-density for the full d-dimensional vector is:
 * <pre>
 *   log p(Y_t | X_t) = sum_{k=0}^{d-1} [-0.5*log(2π) - 0.5*X_t[k] - 0.5*Y_t[k]²*exp(-X_t[k])]
 * </pre>
 *
 * <p>State layout: particles are stored contiguously per particle:
 * {@code X[n*dim + k]} is dimension k of particle n.
 *
 * <p>All kernel methods delegate to {@link ParticleOps} SIMD primitives
 * operating on each dimension's slice of the state buffer. For each
 * dimension k, the values are gathered from the strided layout into
 * scratch space, processed with SIMD, and scattered back.
 *
 * <p>Mirrors the multivariate stochastic volatility model from the Python
 * reference ({@code reference/particles/particles/state_space_models.py}),
 * simplified to independent dimensions (diagonal F, diagonal covariance).
 */
public final class MVStochVol extends ParticleSSM<double[]> {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    private static final int LANES = SPECIES.length();

    /** -0.5 * log(2π). */
    private static final double HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

    private final int d;
    private final double[] mu;
    private final double[] rho;
    private final double[] sigma;
    private final double[] sigma0;
    private final double[] cTransition; // (1 - rho[k]) * mu[k] for each k

    /**
     * @param mu    long-run log-variance per dimension (length d)
     * @param rho   persistence per dimension; must satisfy {@code |rho[k]| < 1}
     * @param sigma innovation standard deviation per dimension; must be positive
     */
    public MVStochVol(double[] mu, double[] rho, double[] sigma) {
        if (mu == null || rho == null || sigma == null) {
            throw new IllegalArgumentException("mu, rho, sigma must not be null");
        }
        int dim = mu.length;
        if (dim == 0) {
            throw new IllegalArgumentException("dimension must be >= 1");
        }
        if (rho.length != dim || sigma.length != dim) {
            throw new IllegalArgumentException(
                    "mu, rho, sigma must all have the same length; got "
                            + mu.length + ", " + rho.length + ", " + sigma.length);
        }

        this.d = dim;
        this.mu = mu.clone();
        this.rho = rho.clone();
        this.sigma = sigma.clone();
        this.sigma0 = new double[dim];
        this.cTransition = new double[dim];

        for (int k = 0; k < dim; k++) {
            if (!(sigma[k] > 0.0)) {
                throw new IllegalArgumentException("sigma[" + k + "] must be > 0: " + sigma[k]);
            }
            double denom = 1.0 - rho[k] * rho[k];
            if (!(denom > 0.0)) {
                throw new IllegalArgumentException("|rho[" + k + "]| must be < 1: " + rho[k]);
            }
            this.sigma0[k] = sigma[k] / Math.sqrt(denom);
            this.cTransition[k] = (1.0 - rho[k]) * mu[k];
        }
    }

    public double[] mu()    { return mu.clone(); }
    public double[] rho()   { return rho.clone(); }
    public double[] sigma() { return sigma.clone(); }

    @Override public int dim() { return d; }

    // ------------------------------------------------------------------
    // Transition (sampleM0, sampleM) — via ParticleOps per dimension
    // ------------------------------------------------------------------

    @Override
    public void sampleM0(StepContext<double[]> ctx) {
        // X_0[k] ~ N(mu[k], sigma0[k]) for each dimension k independently.
        // State layout: X[n*d + k] for particle n, dimension k.
        // Use scratch to generate N contiguous values per dimension, then scatter.
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        for (int k = 0; k < d; k++) {
            // Generate N draws from N(mu[k], sigma0[k]) into scratch
            ParticleOps.gaussianInto(scratch, sOff, N, mu[k], sigma0[k], ctx.rng());
            // Scatter into strided layout: X[xOff + n*d + k] = scratch[sOff + n]
            scatterDim(X, xOff, k, scratch, sOff, N);
        }
    }

    @Override
    public void sampleM(StepContext<double[]> ctx) {
        // X_t[k] = (1-rho[k])*mu[k] + rho[k]*X_{t-1}[k] + sigma[k]*N(0,1)
        // This is AR(1) per dimension with c[k] = (1-rho[k])*mu[k].
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] Xprev = ctx.Xprev();
        final int xpOff = ctx.xpOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        for (int k = 0; k < d; k++) {
            // Gather previous dimension k into scratch[sOff .. sOff+N)
            gatherDim(scratch, sOff, Xprev, xpOff, k, N);
            // AR(1) into scratch[sOff+N .. sOff+2N): c + rho*Xprev_k + sigma*N(0,1)
            ParticleOps.arOneInto(scratch, sOff + N, N,
                    cTransition[k], rho[k], scratch, sOff, sigma[k], ctx.rng());
            // Scatter result into strided layout
            scatterDim(X, xOff, k, scratch, sOff + N, N);
        }
    }

    // ------------------------------------------------------------------
    // Observation (logG) — sum over d dimensions
    // ------------------------------------------------------------------

    @Override
    public void logG(StepContext<double[]> ctx) {
        // logG[n] = sum_k [-0.5*log(2π) - 0.5*X[n,k] - 0.5*Y[k]²*exp(-X[n,k])]
        final double[] yt = ctx.observation();
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] logW = ctx.logW();
        final int lwOff = ctx.lwOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Initialize logW to zero
        for (int n = 0; n < N; n++) {
            logW[lwOff + n] = 0.0;
        }

        // For each dimension k, accumulate the log-density contribution
        for (int k = 0; k < d; k++) {
            final double yk = yt[k];
            final double yySq = yk * yk;

            // Gather dimension k into scratch[sOff .. sOff+N)
            gatherDim(scratch, sOff, X, xOff, k, N);

            // Negate into scratch[sOff+N .. sOff+2N)
            negateInto(scratch, sOff + N, scratch, sOff, N);

            // exp(-X_k) into scratch[sOff+N .. sOff+2N)
            ParticleOps.expInto(scratch, sOff + N, N, scratch, sOff + N);

            // Accumulate: logW[n] += -HALF_LOG_2PI - 0.5*X[n,k] - 0.5*yk²*exp(-X[n,k])
            final DoubleVector constVec = DoubleVector.broadcast(SPECIES, -HALF_LOG_2PI);
            final DoubleVector negHalfVec = DoubleVector.broadcast(SPECIES, -0.5);
            final DoubleVector negHalfYYVec = DoubleVector.broadcast(SPECIES, -0.5 * yySq);

            int i = 0;
            final int bound = SPECIES.loopBound(N);
            for (; i < bound; i += LANES) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, scratch, sOff + i);
                DoubleVector expNegX = DoubleVector.fromArray(SPECIES, scratch, sOff + N + i);
                DoubleVector lwv = DoubleVector.fromArray(SPECIES, logW, lwOff + i);
                // contribution = -HALF_LOG_2PI + (-0.5)*X_k + (-0.5*yy)*exp(-X_k)
                DoubleVector base = xv.fma(negHalfVec, constVec);
                DoubleVector contrib = expNegX.fma(negHalfYYVec, base);
                lwv.add(contrib).intoArray(logW, lwOff + i);
            }
            // Scalar tail
            for (; i < N; i++) {
                logW[lwOff + i] += -HALF_LOG_2PI - 0.5 * scratch[sOff + i]
                        - 0.5 * yySq * scratch[sOff + N + i];
            }
        }
    }

    // ------------------------------------------------------------------
    // Gather / Scatter helpers for strided state layout
    // ------------------------------------------------------------------

    /**
     * Gathers dimension k from strided layout into contiguous buffer.
     * {@code out[outOff + n] = src[srcOff + n*d + k]} for n in [0, N).
     */
    private void gatherDim(double[] out, int outOff,
                           double[] src, int srcOff, int k, int N) {
        for (int n = 0; n < N; n++) {
            out[outOff + n] = src[srcOff + n * d + k];
        }
    }

    /**
     * Scatters contiguous buffer into dimension k of strided layout.
     * {@code dst[dstOff + n*d + k] = src[srcOff + n]} for n in [0, N).
     */
    private void scatterDim(double[] dst, int dstOff, int k,
                            double[] src, int srcOff, int N) {
        for (int n = 0; n < N; n++) {
            dst[dstOff + n * d + k] = src[srcOff + n];
        }
    }

    /**
     * Writes {@code out[off+n] = -src[srcOff+n]} for n in [0, N).
     * SIMD vectorised with scalar tail.
     */
    private static void negateInto(double[] out, int off,
                                   double[] src, int srcOff, int N) {
        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + i);
            v.neg().intoArray(out, off + i);
        }
        for (; i < N; i++) {
            out[off + i] = -src[srcOff + i];
        }
    }
}
