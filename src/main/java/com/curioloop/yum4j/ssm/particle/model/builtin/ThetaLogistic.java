package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.ParticleOps;
import com.curioloop.yum4j.ssm.particle.model.Proposal;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Theta-logistic population dynamics state-space model from ecology.
 *
 * <p>State represents log-population on the log scale:
 * <pre>
 *   X_0            ~ N(log(K), sigmaX)
 *   X_t | X_{t-1}  ~ N(X_{t-1} + r * (1 - (exp(X_{t-1}) / K)^theta), sigmaX²)
 *   Y_t | X_t      ~ N(X_t, sigmaY²)
 * </pre>
 *
 * <p>Parameters:
 * <ul>
 *   <li>{@code r} — intrinsic growth rate</li>
 *   <li>{@code K} — carrying capacity</li>
 *   <li>{@code theta} — shape parameter controlling density dependence</li>
 *   <li>{@code sigmaX} — process noise standard deviation</li>
 *   <li>{@code sigmaY} — observation noise standard deviation</li>
 * </ul>
 *
 * <p>All kernel methods delegate to {@link ParticleOps} SIMD primitives —
 * no scalar per-particle loops. Implements {@link Proposal} and
 * {@link TransitionDensity} for guided particle filters using the
 * Normal-Normal conjugate posterior (since the observation is linear Gaussian
 * given the state).
 */
public final class ThetaLogistic extends ParticleSSM<Double>
        implements Proposal<Double>, TransitionDensity<Double> {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    private static final int LANES = SPECIES.length();

    /** -0.5 * log(2π). */
    private static final double HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

    private final double r;
    private final double K;
    private final double theta;
    private final double sigmaX;
    private final double sigmaY;

    // Cached constants
    private final double logK;
    private final double sigmaXSq;
    private final double sigmaYSq;

    // Guided proposal constants (t = 0): posterior of N(log(K), sigmaX) and N(X, sigmaY)
    private final double sigmaStar0;
    private final double sigmaStar0Sq;

    // Guided proposal constants (t >= 1): posterior of N(loc, sigmaX) and N(X, sigmaY)
    private final double sigmaStarT;
    private final double sigmaStarTSq;

    /**
     * @param r      intrinsic growth rate
     * @param K      carrying capacity (must be positive)
     * @param theta  shape parameter (density dependence)
     * @param sigmaX process noise standard deviation (must be positive)
     * @param sigmaY observation noise standard deviation (must be positive)
     */
    public ThetaLogistic(double r, double K, double theta,
                         double sigmaX, double sigmaY) {
        if (!(K > 0.0)) {
            throw new IllegalArgumentException("K must be > 0: " + K);
        }
        if (!(sigmaX > 0.0)) {
            throw new IllegalArgumentException("sigmaX must be > 0: " + sigmaX);
        }
        if (!(sigmaY > 0.0)) {
            throw new IllegalArgumentException("sigmaY must be > 0: " + sigmaY);
        }
        this.r = r;
        this.K = K;
        this.theta = theta;
        this.sigmaX = sigmaX;
        this.sigmaY = sigmaY;

        this.logK = Math.log(K);
        this.sigmaXSq = sigmaX * sigmaX;
        this.sigmaYSq = sigmaY * sigmaY;

        // Optimal proposal variance at t = 0: 1 / (1/sigmaX² + 1/sigmaY²)
        this.sigmaStar0Sq = 1.0 / (1.0 / sigmaXSq + 1.0 / sigmaYSq);
        this.sigmaStar0 = Math.sqrt(sigmaStar0Sq);

        // Optimal proposal variance at t >= 1: 1 / (1/sigmaX² + 1/sigmaY²)
        this.sigmaStarTSq = 1.0 / (1.0 / sigmaXSq + 1.0 / sigmaYSq);
        this.sigmaStarT = Math.sqrt(sigmaStarTSq);
    }

    /** Default parameters. */
    public ThetaLogistic() {
        this(0.3, 1.0, 1.0, 0.5, 0.2);
    }

    public double r()      { return r; }
    public double K()      { return K; }
    public double theta()  { return theta; }
    public double sigmaX() { return sigmaX; }
    public double sigmaY() { return sigmaY; }

    @Override public int dim() { return 1; }

    // ------------------------------------------------------------------
    // Transition (sampleM0, sampleM) — via ParticleOps + SIMD
    // ------------------------------------------------------------------

    @Override
    public void sampleM0(StepContext<Double> ctx) {
        // X_0 ~ N(log(K), sigmaX)
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), ctx.N(), logK, sigmaX, ctx.rng());
    }

    @Override
    public void sampleM(StepContext<Double> ctx) {
        // X_t = X_{t-1} + r*(1 - (exp(X_{t-1})/K)^theta) + sigmaX*N(0,1)
        //
        // Strategy:
        // 1. Compute per-particle non-linear mean into scratch using SIMD
        //    mean[n] = Xprev[n] + r*(1 - (exp(Xprev[n])/K)^theta)
        //            = Xprev[n] + r - r*exp(theta*(Xprev[n] - log(K)))
        // 2. Use gaussianInto with vector mean to add noise
        final int N = ctx.N();
        final double[] Xprev = ctx.Xprev();
        final int xpOff = ctx.xpOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Compute non-linear transition mean into scratch
        computeTransitionMean(scratch, sOff, Xprev, xpOff, N);

        // Sample: X[n] ~ N(scratch[n], sigmaX)
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), N, scratch, sOff, sigmaX, ctx.rng());
    }

    // ------------------------------------------------------------------
    // Observation (logG) — via ParticleOps
    // ------------------------------------------------------------------

    @Override
    public void logG(StepContext<Double> ctx) {
        double y = ctx.observation();
        // log p(Y_t | X_t) = log N(Y_t; X_t, sigmaY)
        ParticleOps.gaussianLogPdfInto(ctx.logW(), ctx.lwOff(), ctx.N(),
                ctx.X(), ctx.xOff(), y, sigmaY);
    }

    // ------------------------------------------------------------------
    // Proposal (guided filter) — Normal-Normal conjugate posterior
    // ------------------------------------------------------------------

    @Override
    public Proposal<Double> proposal() { return this; }

    @Override
    public void sampleQ0(StepContext<Double> ctx) {
        // At t=0: prior is N(log(K), sigmaX), observation is N(X, sigmaY)
        // Posterior mean: sigmaStar0Sq * (log(K)/sigmaX² + y/sigmaY²)
        double y = ctx.observation();
        double mu = sigmaStar0Sq * (logK / sigmaXSq + y / sigmaYSq);
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), ctx.N(), mu, sigmaStar0, ctx.rng());
    }

    @Override
    public void sampleQ(StepContext<Double> ctx) {
        // At t>=1: prior is N(transitionMean(Xprev), sigmaX), observation is N(X, sigmaY)
        // Per-particle posterior mean: sigmaStarTSq * (loc[n]/sigmaX² + y/sigmaY²)
        // where loc[n] = Xprev[n] + r*(1 - (exp(Xprev[n])/K)^theta)
        final double y = ctx.observation();
        final int N = ctx.N();
        final double[] Xprev = ctx.Xprev();
        final int xpOff = ctx.xpOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Compute transition mean into scratch
        computeTransitionMean(scratch, sOff, Xprev, xpOff, N);

        // Compute proposal mean: mu[n] = sigmaStarTSq * (scratch[n]/sigmaX² + y/sigmaY²)
        //                               = sigmaStarTSq/sigmaX² * scratch[n] + sigmaStarTSq * y/sigmaY²
        double scale = sigmaStarTSq / sigmaXSq;
        double intercept = sigmaStarTSq * (y / sigmaYSq);
        computeLinear(scratch, sOff, scratch, sOff, N, scale, intercept);

        // Sample: X[n] ~ N(scratch[n], sigmaStarT)
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), N, scratch, sOff, sigmaStarT, ctx.rng());
    }

    @Override
    public void logQ0(StepContext<Double> ctx, double[] out, int outOff) {
        double y = ctx.observation();
        double mu = sigmaStar0Sq * (logK / sigmaXSq + y / sigmaYSq);
        ParticleOps.gaussianLogPdfInto(out, outOff, ctx.N(),
                ctx.X(), ctx.xOff(), mu, sigmaStar0);
    }

    @Override
    public void logQ(StepContext<Double> ctx, double[] out, int outOff) {
        // Per-particle proposal mean: sigmaStarTSq * (loc[n]/sigmaX² + y/sigmaY²)
        final double y = ctx.observation();
        final int N = ctx.N();
        final double[] Xprev = ctx.Xprev();
        final int xpOff = ctx.xpOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Compute transition mean into scratch
        computeTransitionMean(scratch, sOff, Xprev, xpOff, N);

        // Compute proposal mean
        double scale = sigmaStarTSq / sigmaXSq;
        double intercept = sigmaStarTSq * (y / sigmaYSq);
        computeLinear(scratch, sOff, scratch, sOff, N, scale, intercept);

        // Evaluate log N(X; scratch, sigmaStarT)
        ParticleOps.gaussianLogPdfInto(out, outOff, N,
                ctx.X(), ctx.xOff(), scratch, sOff, sigmaStarT);
    }

    // ------------------------------------------------------------------
    // TransitionDensity (for guided filter and smoothers)
    // ------------------------------------------------------------------

    @Override
    public TransitionDensity<Double> transitionDensity() { return this; }

    @Override
    public void logPt0(StepContext<Double> ctx, double[] out, int outOff) {
        // log p(X_0) = log N(X; log(K), sigmaX)
        ParticleOps.gaussianLogPdfInto(out, outOff, ctx.N(),
                ctx.X(), ctx.xOff(), logK, sigmaX);
    }

    @Override
    public void logPt(StepContext<Double> ctx, double[] out, int outOff) {
        // log p(X_t | X_{t-1}) = log N(X_t; transitionMean(X_{t-1}), sigmaX)
        final int N = ctx.N();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Compute transition mean into scratch
        computeTransitionMean(scratch, sOff, ctx.Xprev(), ctx.xpOff(), N);

        // Evaluate log N(X; scratch, sigmaX)
        ParticleOps.gaussianLogPdfInto(out, outOff, N,
                ctx.X(), ctx.xOff(), scratch, sOff, sigmaX);
    }

    @Override
    public double upperBound(int t) {
        // Normal density maximised at its mean; max log value = -log(sigma) - 0.5*log(2π)
        return -Math.log(sigmaX) - HALF_LOG_2PI;
    }

    // ------------------------------------------------------------------
    // SIMD helpers — no scalar per-particle loops
    // ------------------------------------------------------------------

    /**
     * Computes the non-linear transition mean into {@code out}:
     * <pre>
     *   out[off+n] = Xprev[xpOff+n] + r * (1 - (exp(Xprev[xpOff+n]) / K)^theta)
     *             = Xprev[xpOff+n] + r - r * exp(theta * (Xprev[xpOff+n] - log(K)))
     * </pre>
     *
     * <p>Uses SIMD (SPECIES_256) for the computation. The key insight is that
     * {@code (exp(x)/K)^theta = exp(theta*(x - log(K)))}, avoiding a pow() call.
     */
    private void computeTransitionMean(double[] out, int off,
                                       double[] Xprev, int xpOff, int N) {
        // mean[n] = Xprev[n] + r - r * exp(theta * (Xprev[n] - logK))
        //
        // Let u[n] = theta * (Xprev[n] - logK)
        // Then mean[n] = Xprev[n] + r - r * exp(u[n])
        //              = Xprev[n] + r * (1 - exp(u[n]))

        final DoubleVector rVec = DoubleVector.broadcast(SPECIES, r);
        final DoubleVector thetaVec = DoubleVector.broadcast(SPECIES, theta);
        final DoubleVector logKVec = DoubleVector.broadcast(SPECIES, logK);
        final DoubleVector negRVec = DoubleVector.broadcast(SPECIES, -r);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xp = DoubleVector.fromArray(SPECIES, Xprev, xpOff + i);
            // u = theta * (xp - logK)
            DoubleVector u = xp.sub(logKVec).mul(thetaVec);
            // expU = exp(u)
            DoubleVector expU = u.lanewise(VectorOperators.EXP);
            // mean = xp + r - r * expU = xp + r + (-r) * expU
            //      = expU * (-r) + (xp + r)
            DoubleVector base = xp.add(rVec);
            expU.fma(negRVec, base).intoArray(out, off + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            double xp = Xprev[xpOff + i];
            double u = theta * (xp - logK);
            out[off + i] = xp + r - r * Math.exp(u);
        }
    }

    /**
     * Writes {@code out[off+n] = scale * src[srcOff+n] + intercept} for n in [0, N).
     * SIMD vectorised with scalar tail.
     */
    private static void computeLinear(double[] out, int off,
                                      double[] src, int srcOff,
                                      int N, double scale, double intercept) {
        final DoubleVector scaleVec = DoubleVector.broadcast(SPECIES, scale);
        final DoubleVector interceptVec = DoubleVector.broadcast(SPECIES, intercept);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + i);
            v.fma(scaleVec, interceptVec).intoArray(out, off + i);
        }
        for (; i < N; i++) {
            out[off + i] = scale * src[srcOff + i] + intercept;
        }
    }
}
