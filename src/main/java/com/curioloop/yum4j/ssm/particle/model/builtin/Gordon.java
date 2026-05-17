package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.ParticleOps;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Gordon, Salmond &amp; Smith (1993) non-linear/non-Gaussian benchmark
 * state-space model.
 *
 * <pre>
 *   X_0            ~ N(0, sigma0²)
 *   X_t | X_{t-1}  = b * X_{t-1}
 *                   + c * X_{t-1} / (1 + X_{t-1}²)
 *                   + d * cos(e * (t - 1))
 *                   + sigmaX * V_t,             V_t ~ N(0, 1)
 *   Y_t | X_t      ~ N(a * X_t², sigmaY²)
 * </pre>
 *
 * <p>Default parameters match the original paper:
 * {@code a = 0.05, b = 0.5, c = 25, d = 8, e = 1.2,
 * sigmaX = sqrt(10) ≈ 3.162278, sigmaY = 1, sigma0 = 2}.
 *
 * <p>All kernel methods use SIMD ({@code SPECIES_256}) for the non-linear
 * computations — no scalar per-particle loops. This model is bootstrap-only:
 * no guided proposal or auxiliary weight is provided.
 *
 * <p>Mirrors {@code Gordon_etal} in the Python reference
 * ({@code reference/particles/particles/state_space_models.py}).
 */
public final class Gordon extends ParticleSSM<Double> {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    private static final int LANES = SPECIES.length();

    // Reference defaults from Gordon et al. (1993).
    public static final double DEFAULT_A = 0.05;
    public static final double DEFAULT_B = 0.5;
    public static final double DEFAULT_C = 25.0;
    public static final double DEFAULT_D = 8.0;
    public static final double DEFAULT_E = 1.2;
    /** {@code sqrt(10)} — innovation standard deviation in the reference. */
    public static final double DEFAULT_SIGMA_X = Math.sqrt(10.0);
    /** Observation noise standard deviation. */
    public static final double DEFAULT_SIGMA_Y = 1.0;
    /** Initial-state standard deviation: N(0, 2²). */
    public static final double DEFAULT_SIGMA_0 = 2.0;

    private final double a;
    private final double b;
    private final double c;
    private final double d;
    private final double e;
    private final double sigmaX;
    private final double sigmaY;
    private final double sigma0;

    /**
     * @param sigmaX transition noise standard deviation (positive)
     * @param sigmaY observation noise standard deviation (positive)
     */
    public Gordon(double sigmaX, double sigmaY) {
        this(DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D, DEFAULT_E,
                sigmaX, sigmaY, DEFAULT_SIGMA_0);
    }

    /**
     * Full-parameter constructor.
     *
     * @param a      observation mean coefficient (Y_t ~ N(a * X_t², sigmaY²))
     * @param b      linear transition coefficient
     * @param c      non-linear transition coefficient
     * @param d      cosine amplitude
     * @param e      cosine frequency
     * @param sigmaX transition noise standard deviation (positive)
     * @param sigmaY observation noise standard deviation (positive)
     * @param sigma0 initial-state standard deviation (positive)
     */
    public Gordon(double a, double b, double c, double d, double e,
                  double sigmaX, double sigmaY, double sigma0) {
        if (!(sigmaX > 0.0)) throw new IllegalArgumentException("sigmaX must be > 0: " + sigmaX);
        if (!(sigmaY > 0.0)) throw new IllegalArgumentException("sigmaY must be > 0: " + sigmaY);
        if (!(sigma0 > 0.0)) throw new IllegalArgumentException("sigma0 must be > 0: " + sigma0);
        this.a = a;
        this.b = b;
        this.c = c;
        this.d = d;
        this.e = e;
        this.sigmaX = sigmaX;
        this.sigmaY = sigmaY;
        this.sigma0 = sigma0;
    }

    /** Reference defaults from Gordon, Salmond &amp; Smith (1993). */
    public Gordon() {
        this(DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D, DEFAULT_E,
                DEFAULT_SIGMA_X, DEFAULT_SIGMA_Y, DEFAULT_SIGMA_0);
    }

    public double a()      { return a; }
    public double b()      { return b; }
    public double c()      { return c; }
    public double d()      { return d; }
    public double e()      { return e; }
    public double sigmaX() { return sigmaX; }
    public double sigmaY() { return sigmaY; }
    public double sigma0() { return sigma0; }

    @Override public int dim() { return 1; }

    // ------------------------------------------------------------------
    // Transition (sampleM0, sampleM) — SIMD vectorised
    // ------------------------------------------------------------------

    @Override
    public void sampleM0(StepContext<Double> ctx) {
        // X_0 ~ N(0, sigma0)
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), ctx.N(), 0.0, sigma0, ctx.rng());
    }

    @Override
    public void sampleM(StepContext<Double> ctx) {
        // X_t = b * Xprev + c * Xprev / (1 + Xprev²) + d * cos(e * (t-1)) + sigmaX * N(0,1)
        //
        // Strategy:
        // 1. Compute non-linear mean into scratch using SIMD
        // 2. Use gaussianInto with vector mean overload to add Gaussian noise
        final int t = ctx.t();
        final int N = ctx.N();
        final double[] Xprev = ctx.Xprev();
        final int xpOff = ctx.xpOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Cosine term is constant across all particles at this time step
        final double cosTerm = d * Math.cos(e * (t - 1));

        // SIMD computation of mean[n] = b*Xprev[n] + c*Xprev[n]/(1+Xprev[n]²) + cosTerm
        final DoubleVector bVec = DoubleVector.broadcast(SPECIES, b);
        final DoubleVector cVec = DoubleVector.broadcast(SPECIES, c);
        final DoubleVector cosTermVec = DoubleVector.broadcast(SPECIES, cosTerm);
        final DoubleVector oneVec = DoubleVector.broadcast(SPECIES, 1.0);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xp = DoubleVector.fromArray(SPECIES, Xprev, xpOff + i);
            // linear part: b * xp
            DoubleVector linear = xp.mul(bVec);
            // non-linear part: c * xp / (1 + xp²)
            DoubleVector xpSq = xp.mul(xp);
            DoubleVector denom = xpSq.add(oneVec);
            DoubleVector nonlinear = xp.mul(cVec).div(denom);
            // mean = linear + nonlinear + cosTerm
            linear.add(nonlinear).add(cosTermVec).intoArray(scratch, sOff + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            double xp = Xprev[xpOff + i];
            scratch[sOff + i] = b * xp + c * xp / (1.0 + xp * xp) + cosTerm;
        }

        // Sample X[n] ~ N(scratch[n], sigmaX)
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), N, scratch, sOff, sigmaX, ctx.rng());
    }

    // ------------------------------------------------------------------
    // Observation (logG) — SIMD vectorised
    // ------------------------------------------------------------------

    @Override
    public void logG(StepContext<Double> ctx) {
        // Y_t | X_t ~ N(a * X_t², sigmaY)
        // log p(Y_t | X_t) = gaussianLogPdf(Y_t; a * X_t², sigmaY)
        //
        // Strategy:
        // 1. Compute per-particle mean: mu[n] = a * X[n]² into scratch
        // 2. Use gaussianLogPdfInto with vector mean to evaluate log-density
        final double y = ctx.observation();
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // SIMD computation of mu[n] = a * X[n]²
        final DoubleVector aVec = DoubleVector.broadcast(SPECIES, a);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xv = DoubleVector.fromArray(SPECIES, X, xOff + i);
            // mu = a * x²
            xv.mul(xv).mul(aVec).intoArray(scratch, sOff + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            double xv = X[xOff + i];
            scratch[sOff + i] = a * xv * xv;
        }

        // Evaluate log N(y; scratch[n], sigmaY) — note: we evaluate the scalar y
        // against the per-particle mean scratch[n], which is equivalent to
        // evaluating log N(scratch[n]; y, sigmaY) by symmetry of the Gaussian.
        // However, gaussianLogPdfInto(out, off, N, x, xOff, mu[], muOff, sigma)
        // computes log N(x[n]; mu[n], sigma). We need log N(y; scratch[n], sigma).
        // Since log N(y; mu, sigma) = -0.5*((y-mu)/sigma)² + logNorm
        //      = log N(mu; y, sigma)
        // We can compute this as gaussianLogPdfInto with x=scratch, mu=y (scalar).
        ParticleOps.gaussianLogPdfInto(ctx.logW(), ctx.lwOff(), N,
                scratch, sOff, y, sigmaY);
    }
}
