package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.ParticleOps;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.math.Gamma;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Discrete Cox process (doubly stochastic Poisson) state-space model.
 *
 * <pre>
 *   X_0            ~ N(mu, sigma0²),      sigma0 = sigma / sqrt(1 - rho²)
 *   X_t | X_{t-1}  ~ N(mu + rho * (X_{t-1} - mu), sigma²)
 *   Y_t | X_t      ~ Poisson(exp(X_t))
 * </pre>
 *
 * <p>The state {@code X_t} represents the log-intensity of a Poisson process
 * and follows an AR(1) process around the long-run mean {@code mu}. The
 * observation is a count drawn from a Poisson distribution with rate
 * {@code exp(X_t)}.
 *
 * <p>The Poisson log-pmf used in {@code logG} is:
 * <pre>
 *   log p(y | X) = y * X - exp(X) - log(y!)
 * </pre>
 *
 * <p>All kernel methods delegate to {@link ParticleOps} SIMD primitives —
 * no scalar per-particle loops. The {@code logG} computation uses SIMD
 * (SPECIES_256) for the exp(X) term and the final log-pmf assembly.
 *
 * <p>Mirrors {@code DiscreteCox} in the Python reference
 * ({@code reference/particles/particles/state_space_models.py}).
 */
public final class DiscreteCox extends ParticleSSM<Integer> {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    private static final int LANES = SPECIES.length();

    private final double mu;
    private final double rho;
    private final double sigma;
    private final double sigma0;

    // Cached transition constant: mu * (1 - rho)
    private final double cTransition;

    /**
     * @param mu    long-run log-intensity mean
     * @param rho   persistence (autoregressive coefficient); must satisfy {@code |rho| < 1}
     * @param sigma innovation standard deviation; must be positive
     */
    public DiscreteCox(double mu, double rho, double sigma) {
        if (!(sigma > 0.0)) {
            throw new IllegalArgumentException("sigma must be > 0: " + sigma);
        }
        double denom = 1.0 - rho * rho;
        if (!(denom > 0.0)) {
            throw new IllegalArgumentException("|rho| must be < 1: " + rho);
        }

        this.mu = mu;
        this.rho = rho;
        this.sigma = sigma;
        this.sigma0 = sigma / Math.sqrt(denom);
        this.cTransition = mu * (1.0 - rho);
    }

    /** Reference defaults: mu=0, sigma=1, rho=0.95. */
    public DiscreteCox() {
        this(0.0, 0.95, 1.0);
    }

    public double mu()     { return mu; }
    public double rho()    { return rho; }
    public double sigma()  { return sigma; }
    public double sigma0() { return sigma0; }

    @Override public int dim() { return 1; }

    // ------------------------------------------------------------------
    // Transition (sampleM0, sampleM) — via ParticleOps
    // ------------------------------------------------------------------

    @Override
    public void sampleM0(StepContext<Integer> ctx) {
        // X_0 ~ N(mu, sigma0)
        // sigma0 = sigma / sqrt(1 - rho²) — the stationary distribution
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), ctx.N(), mu, sigma0, ctx.rng());
    }

    @Override
    public void sampleM(StepContext<Integer> ctx) {
        // X_t = mu + rho * (X_{t-1} - mu) + sigma * N(0,1)
        //     = mu*(1-rho) + rho * X_{t-1} + sigma * N(0,1)
        // This is AR(1) with c = mu*(1-rho)
        ParticleOps.arOneInto(ctx.X(), ctx.xOff(), ctx.N(),
                cTransition, rho, ctx.Xprev(), ctx.xpOff(), sigma, ctx.rng());
    }

    // ------------------------------------------------------------------
    // Observation (logG) — SIMD vectorised Poisson log-pmf
    // ------------------------------------------------------------------

    @Override
    public void logG(StepContext<Integer> ctx) {
        // log p(y | X) = y * X - exp(X) - log(y!)
        //
        // Strategy:
        // 1. Compute exp(X) into scratch using ParticleOps.expInto
        // 2. SIMD loop: logW[n] = y * X[n] - scratch[n] - logFactY
        //    where logFactY = lgamma(y + 1) is a constant for all particles
        final int y = ctx.observation();
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] logW = ctx.logW();
        final int lwOff = ctx.lwOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Handle negative observation (outside Poisson support)
        if (y < 0) {
            java.util.Arrays.fill(logW, lwOff, lwOff + N, Double.NEGATIVE_INFINITY);
            return;
        }

        // Precompute the constant log(y!) = lgamma(y + 1)
        final double logFactY = Gamma.lgamma(y + 1.0);
        final double yDouble = (double) y;

        // Step 1: exp(X) into scratch
        ParticleOps.expInto(scratch, sOff, N, X, xOff);

        // Step 2: SIMD computation of Poisson log-pmf
        // logW[n] = y * X[n] - exp(X[n]) - logFactY
        final DoubleVector yVec = DoubleVector.broadcast(SPECIES, yDouble);
        final DoubleVector negLogFactYVec = DoubleVector.broadcast(SPECIES, -logFactY);
        final DoubleVector negOneVec = DoubleVector.broadcast(SPECIES, -1.0);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xv = DoubleVector.fromArray(SPECIES, X, xOff + i);
            DoubleVector expX = DoubleVector.fromArray(SPECIES, scratch, sOff + i);
            // result = y * X[n] - exp(X[n]) - logFactY
            //        = y * X[n] + (-1) * exp(X[n]) + (-logFactY)
            // Use FMA: y * X + (-logFactY), then add (-1) * expX
            DoubleVector base = xv.fma(yVec, negLogFactYVec); // y * X - logFactY
            expX.fma(negOneVec, base).intoArray(logW, lwOff + i); // -exp(X) + base
        }
        // Scalar tail
        for (; i < N; i++) {
            logW[lwOff + i] = yDouble * X[xOff + i] - scratch[sOff + i] - logFactY;
        }
    }
}
