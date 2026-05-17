package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.ParticleOps;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Univariate stochastic volatility model with leverage effect.
 *
 * <pre>
 *   X_0            ~ N(mu, sigma0²),      sigma0 = sigma / sqrt(1 - rho²)
 *   X_t | X_{t-1}  ~ N(mu + rho*(X_{t-1} - mu) + sigma*rho_lev*eps_y,
 *                      sigma²*(1 - rho_lev²))
 *   Y_t | X_t      ~ N(0, exp(X_t))       (i.e. std dev = exp(0.5 * X_t))
 * </pre>
 *
 * where {@code eps_y = Y_{t-1} / exp(X_{t-1} / 2)} is the standardised return
 * from the previous observation.
 *
 * <p>The leverage correlation {@code rho_lev} (often denoted φ or ρ_lev) induces
 * a dependence between the observation innovation and the volatility innovation.
 * When {@code rho_lev < 0}, negative returns increase future volatility (the
 * classic leverage effect).
 *
 * <p>The observation density {@code logG} is identical to {@link StochVol}'s:
 * <pre>
 *   log p(Y_t | X_t) = -0.5 * log(2π) - 0.5 * X_t - 0.5 * Y_t² * exp(-X_t)
 * </pre>
 *
 * <p>All kernel methods delegate to {@link ParticleOps} SIMD primitives —
 * no scalar per-particle loops. Uses the same SIMD patterns as {@link StochVol}
 * ({@code negateInto}, {@code computeLinear}, {@code expInto}).
 *
 * <p>Mirrors {@code StochVolLeverage} in the Python reference
 * ({@code reference/particles/particles/state_space_models.py}).
 */
public final class StochVolLeverage extends ParticleSSM<Double> {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    private static final int LANES = SPECIES.length();

    /** -0.5 * log(2π). */
    private static final double HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

    private final double mu;
    private final double rho;
    private final double sigma;
    private final double rhoLev;
    private final double sigma0;

    // Cached constants for the transition
    private final double sigmaResid;     // sigma * sqrt(1 - rhoLev²)
    private final double sigmaRhoLev;    // sigma * rhoLev

    /**
     * @param mu     long-run log-variance
     * @param rho    persistence; must satisfy {@code |rho| < 1}
     * @param sigma  innovation standard deviation; must be positive
     * @param rhoLev leverage correlation; must satisfy {@code |rhoLev| < 1}
     */
    public StochVolLeverage(double mu, double rho, double sigma, double rhoLev) {
        if (!(sigma > 0.0)) {
            throw new IllegalArgumentException("sigma must be > 0: " + sigma);
        }
        double denom = 1.0 - rho * rho;
        if (!(denom > 0.0)) {
            throw new IllegalArgumentException("|rho| must be < 1: " + rho);
        }
        if (!(1.0 - rhoLev * rhoLev > 0.0)) {
            throw new IllegalArgumentException("|rhoLev| must be < 1: " + rhoLev);
        }

        this.mu = mu;
        this.rho = rho;
        this.sigma = sigma;
        this.rhoLev = rhoLev;
        this.sigma0 = sigma / Math.sqrt(denom);

        this.sigmaResid = sigma * Math.sqrt(1.0 - rhoLev * rhoLev);
        this.sigmaRhoLev = sigma * rhoLev;
    }

    /** Default parameters. */
    public StochVolLeverage() {
        this(-1.02, 0.9702, 0.178, -0.3);
    }

    public double mu()     { return mu; }
    public double rho()    { return rho; }
    public double sigma()  { return sigma; }
    public double rhoLev() { return rhoLev; }
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

    /**
     * Transition with leverage:
     * <pre>
     *   X_t = mu + rho*(X_{t-1} - mu) + sigma * rhoLev * eps_y
     *         + sigma * sqrt(1 - rhoLev²) * eps_x
     * </pre>
     * where {@code eps_y = Y_{t-1} / exp(X_{t-1}/2)}.
     *
     * <p>The per-particle mean is:
     * {@code mu + rho*(Xprev[n] - mu) + sigmaRhoLev * Y_{t-1} / exp(Xprev[n]/2)}
     * and the residual noise has std dev {@code sigmaResid = sigma * sqrt(1 - rhoLev²)}.
     *
     * <p>SIMD strategy:
     * <ol>
     *   <li>Compute {@code -0.5 * Xprev} into scratch (negate + scale)</li>
     *   <li>Compute {@code exp(-0.5 * Xprev)} into scratch</li>
     *   <li>Compute per-particle mean into scratch:
     *       {@code mean[n] = mu + rho*(Xprev[n] - mu) + sigmaRhoLev * y_{t-1} * exp(-Xprev[n]/2)}</li>
     *   <li>Sample {@code X[n] ~ N(mean[n], sigmaResid)}</li>
     * </ol>
     */
    @Override
    public void sampleM(StepContext<Double> ctx) {
        final int N = ctx.N();
        final double[] Xprev = ctx.Xprev();
        final int xpOff = ctx.xpOff();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] scratch = ctx.scratch();
        final int sOff = ctx.scratchOff();

        // The leverage term needs Y_{t-1}, but ctx.observation() = Y_t.
        // We track Y_{t-1} via the prevObservation field, updated by logG
        // at the end of each step. The BootstrapFk call sequence guarantees:
        //   init: sampleM0 → logG(Y_0)  [sets prevObservation = Y_0]
        //   advance(t=1): sampleM [reads Y_0] → logG(Y_1) [sets prevObservation = Y_1]
        //   advance(t=2): sampleM [reads Y_1] → logG(Y_2) [sets prevObservation = Y_2]
        //   ...
        final double yPrev = prevObservation;
        final double sigRhoLev = sigmaRhoLev;
        final double sigRes = sigmaResid;
        final double m = mu;
        final double r = rho;

        // Step 1: Compute -0.5 * Xprev into scratch for exp(-Xprev/2)
        halfNegateInto(scratch, sOff, Xprev, xpOff, N);

        // Step 2: exp(-0.5 * Xprev) into scratch
        ParticleOps.expInto(scratch, sOff, N, scratch, sOff);

        // Step 3: Compute per-particle mean into scratch:
        //   mean[n] = mu + rho*(Xprev[n] - mu) + sigmaRhoLev * yPrev * exp(-Xprev[n]/2)
        //           = mu*(1-rho) + rho*Xprev[n] + sigmaRhoLev * yPrev * scratch[n]
        //
        // Using SIMD: combine the AR(1) mean with the leverage adjustment
        final double cAR = (1.0 - r) * m;  // intercept of AR(1) part
        final double levCoeff = sigRhoLev * yPrev;  // coefficient for exp(-Xprev/2)

        final DoubleVector cARVec = DoubleVector.broadcast(SPECIES, cAR);
        final DoubleVector rhoVec = DoubleVector.broadcast(SPECIES, r);
        final DoubleVector levCoeffVec = DoubleVector.broadcast(SPECIES, levCoeff);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xp = DoubleVector.fromArray(SPECIES, Xprev, xpOff + i);
            DoubleVector expNegHalfXp = DoubleVector.fromArray(SPECIES, scratch, sOff + i);
            // mean = cAR + rho*Xprev + levCoeff * exp(-Xprev/2)
            //      = rho*Xprev + cAR + levCoeff*expNegHalfXp
            DoubleVector arPart = xp.fma(rhoVec, cARVec); // rho*xp + cAR
            expNegHalfXp.fma(levCoeffVec, arPart).intoArray(scratch, sOff + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            scratch[sOff + i] = cAR + r * Xprev[xpOff + i]
                    + levCoeff * scratch[sOff + i];
        }

        // Step 4: Sample X[n] ~ N(scratch[n], sigmaResid)
        ParticleOps.gaussianInto(X, xOff, N, scratch, sOff, sigRes, ctx.rng());
    }

    // ------------------------------------------------------------------
    // Observation (logG) — identical to StochVol
    // ------------------------------------------------------------------

    @Override
    public void logG(StepContext<Double> ctx) {
        // log p(Y_t | X_t) = -0.5*log(2π) - 0.5*X_t - 0.5*Y_t²*exp(-X_t)
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
        final DoubleVector constVec = DoubleVector.broadcast(SPECIES, -HALF_LOG_2PI);
        final DoubleVector negHalfVec = DoubleVector.broadcast(SPECIES, -0.5);
        final DoubleVector negHalfYYVec = DoubleVector.broadcast(SPECIES, -0.5 * yy);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xv = DoubleVector.fromArray(SPECIES, X, xOff + i);
            DoubleVector expNegX = DoubleVector.fromArray(SPECIES, scratch, sOff + i);
            DoubleVector base = xv.fma(negHalfVec, constVec);
            expNegX.fma(negHalfYYVec, base).intoArray(logW, lwOff + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            logW[lwOff + i] = -HALF_LOG_2PI - 0.5 * X[xOff + i] - 0.5 * yy * scratch[sOff + i];
        }

        // Update prevObservation for the next step's leverage term
        prevObservation = y;
    }

    // ------------------------------------------------------------------
    // Previous observation tracking for leverage term
    // ------------------------------------------------------------------

    /**
     * Stores the previous observation Y_{t-1} needed by the leverage transition.
     * Updated at the end of each logG call (which processes Y_t, making it
     * available as Y_{t-1} for the next step's sampleM).
     *
     * <p>Initialised to 0.0 so that the first sampleM (at t=1) uses Y_0 = 0
     * if sampleM0/logG at t=0 hasn't been called yet. In practice, the bootstrap
     * filter always calls init (sampleM0 + logG) before the first advance (sampleM),
     * so prevObservation will hold Y_0 by the time sampleM is first invoked.
     */
    private double prevObservation = 0.0;

    // ------------------------------------------------------------------
    // SIMD helpers — no scalar per-particle loops
    // ------------------------------------------------------------------

    /**
     * Writes {@code out[off+n] = -src[srcOff+n]} for n in [0, N).
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
     * Writes {@code out[off+n] = -0.5 * src[srcOff+n]} for n in [0, N).
     * Used to compute -X/2 for the exp(-X/2) needed by the leverage term.
     */
    private static void halfNegateInto(double[] out, int off,
                                       double[] src, int srcOff, int N) {
        final DoubleVector negHalfVec = DoubleVector.broadcast(SPECIES, -0.5);
        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + i);
            v.mul(negHalfVec).intoArray(out, off + i);
        }
        for (; i < N; i++) {
            out[off + i] = -0.5 * src[srcOff + i];
        }
    }
}
