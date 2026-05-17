package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.ParticleOps;
import com.curioloop.yum4j.ssm.particle.model.AuxWeight;
import com.curioloop.yum4j.ssm.particle.model.Proposal;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;

/**
 * Univariate linear Gaussian state-space model.
 *
 * <pre>
 *   X_0            ~ N(0, sigma0²)
 *   X_t | X_{t-1}  ~ N(rho * X_{t-1}, sigmaX²)
 *   Y_t | X_t      ~ N(X_t, sigmaY²)
 * </pre>
 *
 * <p>All kernel methods delegate to {@link ParticleOps} SIMD primitives —
 * no scalar per-particle loops. Implements {@link Proposal}, {@link TransitionDensity},
 * and {@link AuxWeight} for guided and auxiliary particle filters.
 *
 * <p>The guided proposal is the optimal Gaussian posterior:
 * <pre>
 *   t = 0:  postVar = 1 / (1/sigma0² + 1/sigmaY²)
 *           postMu  = postVar * (y / sigmaY²)
 *   t ≥ 1:  postVar = 1 / (1/sigmaX² + 1/sigmaY²)
 *           postMu  = postVar * (rho * x_{t-1} / sigmaX² + y / sigmaY²)
 * </pre>
 */
public final class LinearGauss extends ParticleSSM<Double>
        implements Proposal<Double>, TransitionDensity<Double>, AuxWeight<Double> {

    private final double rho;
    private final double sigmaX;
    private final double sigmaY;
    private final double sigma0;

    // Variances
    private final double sigmaXSq;
    private final double sigmaYSq;
    private final double sigma0Sq;

    // Optimal proposal constants for t = 0
    private final double sigmaStar0;
    private final double sigmaStar0Sq;

    // Optimal proposal constants for t >= 1
    private final double sigmaStarT;
    private final double sigmaStarTSq;

    // logEta constants: N(rho * x, sqrt(sigmaX² + sigmaY²))
    private final double etaScale;

    /**
     * @param rho    transition coefficient
     * @param sigmaX transition standard deviation (positive)
     * @param sigmaY observation standard deviation (positive)
     * @param sigma0 initial standard deviation; pass {@link Double#NaN}
     *               to use the stationary value sigmaX / sqrt(1 - rho²)
     */
    public LinearGauss(double rho, double sigmaX, double sigmaY, double sigma0) {
        if (!(sigmaX > 0.0)) throw new IllegalArgumentException("sigmaX must be > 0: " + sigmaX);
        if (!(sigmaY > 0.0)) throw new IllegalArgumentException("sigmaY must be > 0: " + sigmaY);
        if (Double.isNaN(sigma0)) {
            double denom = 1.0 - rho * rho;
            if (!(denom > 0.0)) throw new IllegalArgumentException(
                    "default sigma0 requires |rho| < 1 (got rho = " + rho + ")");
            sigma0 = sigmaX / Math.sqrt(denom);
        } else if (!(sigma0 > 0.0)) {
            throw new IllegalArgumentException("sigma0 must be > 0: " + sigma0);
        }

        this.rho = rho;
        this.sigmaX = sigmaX;
        this.sigmaY = sigmaY;
        this.sigma0 = sigma0;

        this.sigmaXSq = sigmaX * sigmaX;
        this.sigmaYSq = sigmaY * sigmaY;
        this.sigma0Sq = sigma0 * sigma0;

        // Optimal proposal variance at t = 0: 1 / (1/sigma0² + 1/sigmaY²)
        this.sigmaStar0Sq = 1.0 / (1.0 / sigma0Sq + 1.0 / sigmaYSq);
        this.sigmaStar0 = Math.sqrt(sigmaStar0Sq);

        // Optimal proposal variance at t >= 1: 1 / (1/sigmaX² + 1/sigmaY²)
        this.sigmaStarTSq = 1.0 / (1.0 / sigmaXSq + 1.0 / sigmaYSq);
        this.sigmaStarT = Math.sqrt(sigmaStarTSq);

        // logEta scale: sqrt(sigmaX² + sigmaY²)
        this.etaScale = Math.sqrt(sigmaXSq + sigmaYSq);
    }

    public double rho()    { return rho; }
    public double sigmaX() { return sigmaX; }
    public double sigmaY() { return sigmaY; }
    public double sigma0() { return sigma0; }

    @Override public int dim() { return 1; }

    // ------------------------------------------------------------------
    // Transition (sampleM0, sampleM) — via ParticleOps
    // ------------------------------------------------------------------

    @Override
    public void sampleM0(StepContext<Double> ctx) {
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), ctx.N(), 0.0, sigma0, ctx.rng());
    }

    @Override
    public void sampleM(StepContext<Double> ctx) {
        ParticleOps.arOneInto(ctx.X(), ctx.xOff(), ctx.N(),
                0.0, rho, ctx.Xprev(), ctx.xpOff(), sigmaX, ctx.rng());
    }

    // ------------------------------------------------------------------
    // Observation (logG) — via ParticleOps
    // ------------------------------------------------------------------

    @Override
    public void logG(StepContext<Double> ctx) {
        double y = ctx.observation();
        // log p(y | X) = gaussianLogPdf(X; y, sigmaY)
        // i.e. evaluate N(y; X_n, sigmaY) which equals N(X_n; y, sigmaY)
        ParticleOps.gaussianLogPdfInto(ctx.logW(), ctx.lwOff(), ctx.N(),
                ctx.X(), ctx.xOff(), y, sigmaY);
    }

    // ------------------------------------------------------------------
    // Proposal (guided filter) — optimal Gaussian posterior
    // ------------------------------------------------------------------

    @Override
    public Proposal<Double> proposal() { return this; }

    @Override
    public void sampleQ0(StepContext<Double> ctx) {
        double y = ctx.observation();
        double mu = sigmaStar0Sq * (y / sigmaYSq);
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), ctx.N(), mu, sigmaStar0, ctx.rng());
    }

    @Override
    public void sampleQ(StepContext<Double> ctx) {
        double y = ctx.observation();
        double cY = y / sigmaYSq;
        double rInvX = rho / sigmaXSq;
        // Per-particle mean: postVar * (rho * Xprev / sigmaX² + y / sigmaY²)
        // = sigmaStarTSq * (rInvX * Xprev[n] + cY)
        // = sigmaStarTSq * rInvX * Xprev[n] + sigmaStarTSq * cY
        // This is an AR(1) form: out = c + rho' * Xprev + sigma * N(0,1)
        // where c = sigmaStarTSq * cY, rho' = sigmaStarTSq * rInvX
        double c = sigmaStarTSq * cY;
        double rhoEff = sigmaStarTSq * rInvX;
        ParticleOps.arOneInto(ctx.X(), ctx.xOff(), ctx.N(),
                c, rhoEff, ctx.Xprev(), ctx.xpOff(), sigmaStarT, ctx.rng());
    }

    @Override
    public void logQ0(StepContext<Double> ctx, double[] out, int outOff) {
        double y = ctx.observation();
        double mu = sigmaStar0Sq * (y / sigmaYSq);
        ParticleOps.gaussianLogPdfInto(out, outOff, ctx.N(),
                ctx.X(), ctx.xOff(), mu, sigmaStar0);
    }

    @Override
    public void logQ(StepContext<Double> ctx, double[] out, int outOff) {
        double y = ctx.observation();
        double cY = y / sigmaYSq;
        double rInvX = rho / sigmaXSq;
        // Per-particle mean: sigmaStarTSq * (rInvX * Xprev[n] + cY)
        // We need the per-particle mean array in scratch to use gaussianLogPdfInto
        // with vector mean. Compute mu[n] = sigmaStarTSq * rInvX * Xprev[n] + sigmaStarTSq * cY
        // Use scratch buffer for the per-particle means
        double[] scratch = ctx.scratch();
        int sOff = ctx.scratchOff();
        int N = ctx.N();
        double c = sigmaStarTSq * cY;
        double rhoEff = sigmaStarTSq * rInvX;
        // Compute mu[n] = c + rhoEff * Xprev[n] using arOneInto with sigma=0?
        // No — use a simple SIMD approach: write mu into scratch via ParticleOps pattern
        // Actually, we can use gaussianLogPdfInto with vector mean after computing mu
        // Compute mu into scratch: mu[n] = rhoEff * Xprev[n] + c
        computeLinear(scratch, sOff, ctx.Xprev(), ctx.xpOff(), N, rhoEff, c);
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
        // log p(X_0) = gaussianLogPdf(X; 0, sigma0)
        ParticleOps.gaussianLogPdfInto(out, outOff, ctx.N(),
                ctx.X(), ctx.xOff(), 0.0, sigma0);
    }

    @Override
    public void logPt(StepContext<Double> ctx, double[] out, int outOff) {
        // log p(X_t | X_{t-1}) = gaussianLogPdf(X; rho * Xprev, sigmaX)
        // Per-particle mean = rho * Xprev[n]
        // Use scratch for the per-particle means
        double[] scratch = ctx.scratch();
        int sOff = ctx.scratchOff();
        int N = ctx.N();
        computeLinear(scratch, sOff, ctx.Xprev(), ctx.xpOff(), N, rho, 0.0);
        ParticleOps.gaussianLogPdfInto(out, outOff, N,
                ctx.X(), ctx.xOff(), scratch, sOff, sigmaX);
    }

    @Override
    public double upperBound(int t) {
        // Normal density maximised at its mean; max log value = -log(sigma) - 0.5*log(2π)
        double sigma = (t == 0) ? sigma0 : sigmaX;
        return -Math.log(sigma) - 0.5 * Math.log(2.0 * Math.PI);
    }

    // ------------------------------------------------------------------
    // AuxWeight (for auxiliary particle filter)
    // ------------------------------------------------------------------

    @Override
    public AuxWeight<Double> auxWeight() { return this; }

    @Override
    public void logEta(StepContext<Double> ctx, Double yNext, double[] out, int outOff) {
        // logEta = log N(yNext; rho * X, etaScale)
        // Per-particle mean = rho * X[n]
        double[] scratch = ctx.scratch();
        int sOff = ctx.scratchOff();
        int N = ctx.N();
        computeLinear(scratch, sOff, ctx.X(), ctx.xOff(), N, rho, 0.0);
        ParticleOps.gaussianLogPdfInto(out, outOff, N,
                scratch, sOff, yNext, etaScale);
    }

    // ------------------------------------------------------------------
    // Helper: compute out[n] = scale * src[n] + intercept (SIMD via arOneInto trick)
    // ------------------------------------------------------------------

    /**
     * Writes {@code out[off+n] = scale * src[srcOff+n] + intercept} for n in [0, N).
     * Uses ParticleOps-style SIMD without scalar per-particle loops.
     */
    private static void computeLinear(double[] out, int off,
                                      double[] src, int srcOff,
                                      int N, double scale, double intercept) {
        // Inline SIMD linear transform: out = intercept + scale * src
        // This mirrors the structure of arOneInto but without the noise term.
        // We use the Vector API directly for zero-allocation SIMD.
        jdk.incubator.vector.VectorSpecies<Double> SPECIES = jdk.incubator.vector.DoubleVector.SPECIES_256;
        int LANES = SPECIES.length();
        var scaleVec = jdk.incubator.vector.DoubleVector.broadcast(SPECIES, scale);
        var interceptVec = jdk.incubator.vector.DoubleVector.broadcast(SPECIES, intercept);

        int i = 0;
        int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            var v = jdk.incubator.vector.DoubleVector.fromArray(SPECIES, src, srcOff + i);
            v.fma(scaleVec, interceptVec).intoArray(out, off + i);
        }
        for (; i < N; i++) {
            out[off + i] = scale * src[srcOff + i] + intercept;
        }
    }
}
