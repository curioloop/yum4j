package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.ParticleOps;
import com.curioloop.yum4j.ssm.particle.model.AuxWeight;
import com.curioloop.yum4j.ssm.particle.model.Proposal;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Univariate stochastic volatility state-space model.
 *
 * <pre>
 *   X_0            ~ N(mu, sigma0²),      sigma0 = sigma / sqrt(1 - rho²)
 *   X_t | X_{t-1}  ~ N((1 - rho) * mu + rho * X_{t-1}, sigma²)
 *   Y_t | X_t      ~ N(0, exp(X_t))       (i.e. std dev = exp(0.5 * X_t))
 * </pre>
 *
 * <p>The log-density of the observation is:
 * <pre>
 *   log p(Y_t | X_t) = -0.5 * log(2π) - 0.5 * X_t - 0.5 * Y_t² * exp(-X_t)
 * </pre>
 *
 * <p>All kernel methods delegate to {@link ParticleOps} SIMD primitives —
 * no scalar per-particle loops. Implements {@link Proposal}, {@link TransitionDensity},
 * and {@link AuxWeight} for guided and auxiliary particle filters using the
 * Pitt–Shephard proposal.
 *
 * <p>Mirrors {@code StochVol} in the Python reference
 * ({@code reference/particles/particles/state_space_models.py}).
 */
public final class StochVol extends ParticleSSM<Double>
        implements Proposal<Double>, TransitionDensity<Double>, AuxWeight<Double> {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    private static final int LANES = SPECIES.length();

    /** -0.5 * log(2π). */
    private static final double HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

    private final double mu;
    private final double rho;
    private final double sigma;
    private final double sigma0;

    // Cached constants
    private final double sigmaSq;
    private final double sigma0Sq;
    private final double halfInvSigmaSq;     // 0.5 / sigma²
    private final double cTransition;        // (1 - rho) * mu

    /**
     * @param mu    long-run log-variance
     * @param rho   persistence; must satisfy {@code |rho| < 1}
     * @param sigma innovation standard deviation; must be positive
     */
    public StochVol(double mu, double rho, double sigma) {
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

        this.sigmaSq = sigma * sigma;
        this.sigma0Sq = sigma0 * sigma0;
        this.halfInvSigmaSq = 0.5 / sigmaSq;
        this.cTransition = (1.0 - rho) * mu;
    }

    /** Default Pitt &amp; Shephard (1999) parameters. */
    public StochVol() {
        this(-1.02, 0.9702, 0.178);
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
    public void sampleM0(StepContext<Double> ctx) {
        // X_0 ~ N(mu, sigma0)
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), ctx.N(), mu, sigma0, ctx.rng());
    }

    @Override
    public void sampleM(StepContext<Double> ctx) {
        // X_t = (1-rho)*mu + rho*X_{t-1} + sigma*N(0,1)
        // This is AR(1) with c = (1-rho)*mu
        ParticleOps.arOneInto(ctx.X(), ctx.xOff(), ctx.N(),
                cTransition, rho, ctx.Xprev(), ctx.xpOff(), sigma, ctx.rng());
    }

    // ------------------------------------------------------------------
    // Observation (logG) — SIMD vectorised, no scalar per-particle loop
    // ------------------------------------------------------------------

    @Override
    public void logG(StepContext<Double> ctx) {
        // log p(Y_t | X_t) = -0.5*log(2π) - 0.5*X_t - 0.5*Y_t²*exp(-X_t)
        //
        // Strategy:
        // 1. Negate X into scratch: scratch[n] = -X[n]
        // 2. Compute exp(-X) into scratch: expInto(scratch, scratch)
        // 3. SIMD loop: logW[n] = -HALF_LOG_2PI - 0.5*X[n] - 0.5*y²*scratch[n]
        final double y = ctx.observation();
        final double yy = y * y;
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] logW = ctx.logW();
        final int lwOff = ctx.lwOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Step 1: negate X into scratch
        negateInto(scratch, sOff, X, xOff, N);

        // Step 2: exp(-X) into scratch
        ParticleOps.expInto(scratch, sOff, N, scratch, sOff);

        // Step 3: SIMD computation of the full logG formula
        // logW[n] = -HALF_LOG_2PI - 0.5 * X[n] - 0.5 * yy * exp(-X[n])
        final DoubleVector constVec = DoubleVector.broadcast(SPECIES, -HALF_LOG_2PI);
        final DoubleVector negHalfVec = DoubleVector.broadcast(SPECIES, -0.5);
        final DoubleVector negHalfYYVec = DoubleVector.broadcast(SPECIES, -0.5 * yy);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xv = DoubleVector.fromArray(SPECIES, X, xOff + i);
            DoubleVector expNegX = DoubleVector.fromArray(SPECIES, scratch, sOff + i);
            // result = -HALF_LOG_2PI + (-0.5) * X[n] + (-0.5 * yy) * exp(-X[n])
            // = constVec + negHalf * xv + negHalfYY * expNegX
            // Use FMA: negHalfYY * expNegX + (constVec + negHalf * xv)
            DoubleVector base = xv.fma(negHalfVec, constVec); // -0.5 * X + (-HALF_LOG_2PI)
            expNegX.fma(negHalfYYVec, base).intoArray(logW, lwOff + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            logW[lwOff + i] = -HALF_LOG_2PI - 0.5 * X[xOff + i] - 0.5 * yy * scratch[sOff + i];
        }
    }

    // ------------------------------------------------------------------
    // Proposal (Pitt–Shephard guided) — via ParticleOps
    // ------------------------------------------------------------------

    @Override
    public Proposal<Double> proposal() { return this; }

    /**
     * Pitt–Shephard xhat helper (scalar, used for t=0 constant computation).
     * {@code xhat(xst, sigma², y) = xst + 0.5 * sigma² * (y² * exp(-xst) - 1)}
     */
    private static double xhat(double xst, double sigma2, double yt) {
        return xst + 0.5 * sigma2 * (yt * yt * Math.exp(-xst) - 1.0);
    }

    @Override
    public void sampleQ0(StepContext<Double> ctx) {
        // Proposal at t=0: N(xhat(0, sigma0², y_0), sigma0)
        // xhat(0, sigma0², y) is a scalar constant
        double y = ctx.observation();
        double loc = xhat(0.0, sigma0Sq, y);
        ParticleOps.gaussianInto(ctx.X(), ctx.xOff(), ctx.N(), loc, sigma0, ctx.rng());
    }

    @Override
    public void sampleQ(StepContext<Double> ctx) {
        // Proposal at t>=1: N(xhat(EXt(xp), sigma², y_t), sigma)
        // where EXt(xp) = (1-rho)*mu + rho*xp
        // xhat(xst, s², y) = xst + 0.5*s²*(y²*exp(-xst) - 1)
        //
        // Per-particle mean: xhat((1-rho)*mu + rho*Xprev[n], sigma², y_t)
        // This requires exp(-xst) per particle — use scratch + SIMD
        final double y = ctx.observation();
        final double yy = y * y;
        final int N = ctx.N();
        final double[] Xprev = ctx.Xprev();
        final int xpOff = ctx.xpOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Compute xst[n] = cTransition + rho * Xprev[n] into scratch
        computeLinear(scratch, sOff, Xprev, xpOff, N, rho, cTransition);

        // Compute proposal mean: xhat[n] = xst[n] + 0.5*sigma²*(y²*exp(-xst[n]) - 1)
        // We need exp(-xst) — negate xst into X (temporary), then expInto
        // Use the output X buffer as temporary for negated xst
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();

        // Negate xst into X temporarily
        negateInto(X, xOff, scratch, sOff, N);
        // exp(-xst) into X
        ParticleOps.expInto(X, xOff, N, X, xOff);

        // Now compute proposal mean into scratch:
        // mean[n] = xst[n] + 0.5*sigma²*(y²*exp(-xst[n]) - 1)
        //         = xst[n] + 0.5*sigma²*y²*exp(-xst[n]) - 0.5*sigma²
        // scratch currently holds xst, X holds exp(-xst)
        final double halfS2 = 0.5 * sigmaSq;
        final double halfS2YY = halfS2 * yy;
        final double negHalfS2 = -halfS2;

        final DoubleVector halfS2YYVec = DoubleVector.broadcast(SPECIES, halfS2YY);
        final DoubleVector negHalfS2Vec = DoubleVector.broadcast(SPECIES, negHalfS2);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xst = DoubleVector.fromArray(SPECIES, scratch, sOff + i);
            DoubleVector expNeg = DoubleVector.fromArray(SPECIES, X, xOff + i);
            // mean = xst + halfS2YY * expNeg + negHalfS2
            //      = halfS2YY * expNeg + (xst + negHalfS2)
            DoubleVector base = xst.add(negHalfS2Vec);
            expNeg.fma(halfS2YYVec, base).intoArray(scratch, sOff + i);
        }
        for (; i < N; i++) {
            scratch[sOff + i] = scratch[sOff + i] + halfS2YY * X[xOff + i] + negHalfS2;
        }

        // Now scratch holds the per-particle proposal mean
        // Sample: X[n] ~ N(scratch[n], sigma)
        ParticleOps.gaussianInto(X, xOff, N, scratch, sOff, sigma, ctx.rng());
    }

    @Override
    public void logQ0(StepContext<Double> ctx, double[] out, int outOff) {
        // log q_0(X) = log N(X; xhat(0, sigma0², y_0), sigma0)
        double y = ctx.observation();
        double loc = xhat(0.0, sigma0Sq, y);
        ParticleOps.gaussianLogPdfInto(out, outOff, ctx.N(),
                ctx.X(), ctx.xOff(), loc, sigma0);
    }

    @Override
    public void logQ(StepContext<Double> ctx, double[] out, int outOff) {
        // log q_t(X | Xprev) = log N(X; xhat(EXt(Xprev), sigma², y_t), sigma)
        // Per-particle mean = xhat((1-rho)*mu + rho*Xprev[n], sigma², y_t)
        final double y = ctx.observation();
        final double yy = y * y;
        final int N = ctx.N();
        final double[] Xprev = ctx.Xprev();
        final int xpOff = ctx.xpOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Compute xst[n] = cTransition + rho * Xprev[n] into scratch
        computeLinear(scratch, sOff, Xprev, xpOff, N, rho, cTransition);

        // Compute xhat per particle using SIMD
        // Need exp(-xst) — use out buffer as temporary
        negateInto(out, outOff, scratch, sOff, N);
        ParticleOps.expInto(out, outOff, N, out, outOff);

        // Compute proposal mean into scratch:
        // mean[n] = xst[n] + 0.5*sigma²*y²*exp(-xst[n]) - 0.5*sigma²
        final double halfS2 = 0.5 * sigmaSq;
        final double halfS2YY = halfS2 * yy;
        final double negHalfS2 = -halfS2;

        final DoubleVector halfS2YYVec = DoubleVector.broadcast(SPECIES, halfS2YY);
        final DoubleVector negHalfS2Vec = DoubleVector.broadcast(SPECIES, negHalfS2);

        int i = 0;
        int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xst = DoubleVector.fromArray(SPECIES, scratch, sOff + i);
            DoubleVector expNeg = DoubleVector.fromArray(SPECIES, out, outOff + i);
            DoubleVector base = xst.add(negHalfS2Vec);
            expNeg.fma(halfS2YYVec, base).intoArray(scratch, sOff + i);
        }
        for (; i < N; i++) {
            scratch[sOff + i] = scratch[sOff + i] + halfS2YY * out[outOff + i] + negHalfS2;
        }

        // Now scratch holds per-particle proposal mean
        // Evaluate log N(X; scratch, sigma)
        ParticleOps.gaussianLogPdfInto(out, outOff, N,
                ctx.X(), ctx.xOff(), scratch, sOff, sigma);
    }

    // ------------------------------------------------------------------
    // TransitionDensity (for guided filter and smoothers)
    // ------------------------------------------------------------------

    @Override
    public TransitionDensity<Double> transitionDensity() { return this; }

    @Override
    public void logPt0(StepContext<Double> ctx, double[] out, int outOff) {
        // log p(X_0) = log N(X; mu, sigma0)
        ParticleOps.gaussianLogPdfInto(out, outOff, ctx.N(),
                ctx.X(), ctx.xOff(), mu, sigma0);
    }

    @Override
    public void logPt(StepContext<Double> ctx, double[] out, int outOff) {
        // log p(X_t | X_{t-1}) = log N(X_t; (1-rho)*mu + rho*X_{t-1}, sigma)
        // Per-particle mean = cTransition + rho * Xprev[n]
        double[] scratch = ctx.scratch();
        int sOff = ctx.scratchOff();
        int N = ctx.N();
        computeLinear(scratch, sOff, ctx.Xprev(), ctx.xpOff(), N, rho, cTransition);
        ParticleOps.gaussianLogPdfInto(out, outOff, N,
                ctx.X(), ctx.xOff(), scratch, sOff, sigma);
    }

    @Override
    public double upperBound(int t) {
        // Normal density maximised at its mean; max log value = -log(sigma) - 0.5*log(2π)
        double s = (t == 0) ? sigma0 : sigma;
        return -Math.log(s) - HALF_LOG_2PI;
    }

    // ------------------------------------------------------------------
    // AuxWeight (Pitt–Shephard auxiliary logEta)
    // ------------------------------------------------------------------

    @Override
    public AuxWeight<Double> auxWeight() { return this; }

    @Override
    public void logEta(StepContext<Double> ctx, Double yNext, double[] out, int outOff) {
        // xst     = (1-rho)*mu + rho*X[n]
        // xstmmu  = xst - mu = rho*(X[n] - mu)
        // xhat    = xst + 0.5*sigma²*(yNext²*exp(-xst) - 1)
        // xhatmmu = xhat - mu
        // logEta  = 0.5/sigma² * (xhatmmu² - xstmmu²)
        //           - 0.5 * yNext² * exp(-xst) * (1 + xstmmu)
        final double yN = yNext;
        final double yyN = yN * yN;
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // Compute xst[n] = cTransition + rho * X[n] into scratch
        computeLinear(scratch, sOff, X, xOff, N, rho, cTransition);

        // Compute exp(-xst) into out (temporary)
        negateInto(out, outOff, scratch, sOff, N);
        ParticleOps.expInto(out, outOff, N, out, outOff);

        // Now scratch = xst, out = exp(-xst)
        // Compute logEta using SIMD
        final double halfInvS2 = halfInvSigmaSq;
        final double halfS2 = 0.5 * sigmaSq;
        final double m = mu;

        final DoubleVector halfInvS2Vec = DoubleVector.broadcast(SPECIES, halfInvS2);
        final DoubleVector halfS2Vec = DoubleVector.broadcast(SPECIES, halfS2);
        final DoubleVector yyNVec = DoubleVector.broadcast(SPECIES, yyN);
        final DoubleVector muVec = DoubleVector.broadcast(SPECIES, m);
        final DoubleVector oneVec = DoubleVector.broadcast(SPECIES, 1.0);
        final DoubleVector negHalfVec = DoubleVector.broadcast(SPECIES, -0.5);
        final DoubleVector negHalfS2Vec = DoubleVector.broadcast(SPECIES, -halfS2);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xst = DoubleVector.fromArray(SPECIES, scratch, sOff + i);
            DoubleVector expNegXst = DoubleVector.fromArray(SPECIES, out, outOff + i);

            // xstmmu = xst - mu
            DoubleVector xstmmu = xst.sub(muVec);

            // xhat = xst + 0.5*sigma²*(yy*exp(-xst) - 1)
            //      = xst + halfS2 * yy * expNeg - halfS2
            DoubleVector xhat = expNegXst.mul(yyNVec).fma(halfS2Vec, xst.add(negHalfS2Vec));

            // xhatmmu = xhat - mu
            DoubleVector xhatmmu = xhat.sub(muVec);

            // term1 = halfInvS2 * (xhatmmu² - xstmmu²)
            DoubleVector term1 = xhatmmu.mul(xhatmmu).sub(xstmmu.mul(xstmmu)).mul(halfInvS2Vec);

            // term2 = -0.5 * yy * exp(-xst) * (1 + xstmmu)
            DoubleVector term2 = expNegXst.mul(yyNVec).mul(oneVec.add(xstmmu)).mul(negHalfVec);

            // logEta = term1 + term2
            term1.add(term2).intoArray(out, outOff + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            double xst = scratch[sOff + i];
            double expNegXst = out[outOff + i];
            double xstmmu = xst - m;
            double xhatVal = xst + halfS2 * (yyN * expNegXst - 1.0);
            double xhatmmu = xhatVal - m;
            double term1 = halfInvS2 * (xhatmmu * xhatmmu - xstmmu * xstmmu);
            double term2 = -0.5 * yyN * expNegXst * (1.0 + xstmmu);
            out[outOff + i] = term1 + term2;
        }
    }

    // ------------------------------------------------------------------
    // SIMD helpers — no scalar per-particle loops
    // ------------------------------------------------------------------

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
