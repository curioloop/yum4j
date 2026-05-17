package com.curioloop.yum4j.ssm.particle.model.builtin;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.ParticleOps;
import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Bearings-only tracking state-space model.
 *
 * <p>The state {@code x = (vx, vy, px, py)} stores a 2D velocity and
 * 2D position. Mirrors {@code BearingsOnly} in the Python reference
 * ({@code reference/particles/particles/state_space_models.py}):
 *
 * <pre>
 *   PX0:
 *     vx_0, vy_0 ~ N(x0[0..1], sigmaX)
 *     px_0       = x0[2]                        (Dirac)
 *     py_0       = x0[3]                        (Dirac)
 *
 *   PX (t &gt;= 1):
 *     vx_t ~ N(vx_{t-1}, sigmaX)
 *     vy_t ~ N(vy_{t-1}, sigmaX)
 *     px_t = px_{t-1} + vx_{t-1}                (Dirac)
 *     py_t = py_{t-1} + vy_{t-1}                (Dirac)
 *
 *   PY:
 *     Y_t ~ N(angle(px_t, py_t), sigmaY)
 *     angle(px, py) = atan(py / px) + (px &lt; 0 ? pi : 0)
 * </pre>
 *
 * <p>Buffer layout is column-major {@code (d=4, N)} with leading
 * dimension {@code N}:
 *
 * <pre>
 *   row 0: vx values   at X[xOff + 0*N + n]
 *   row 1: vy values   at X[xOff + 1*N + n]
 *   row 2: px values   at X[xOff + 2*N + n]
 *   row 3: py values   at X[xOff + 3*N + n]
 * </pre>
 *
 * <p>All kernel methods delegate to {@link ParticleOps} SIMD primitives
 * where possible. The observation log-density requires a scalar
 * {@code Math.atan} per particle (no SIMD equivalent), which is
 * permitted under R16.5 since the analytic form has no vector equivalent.
 *
 * <p>Implements {@link TransitionDensity} for FFBS and guided filters.
 * The transition density returns {@link Double#NEGATIVE_INFINITY} when
 * the Dirac position components are inconsistent with the ancestor state.
 */
public final class BearingsOnly extends ParticleSSM<Double>
        implements TransitionDensity<Double> {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    private static final int LANES = SPECIES.length();

    /** -0.5 * log(2π). */
    private static final double LOG_INV_SQRT_2PI = -0.5 * Math.log(2.0 * Math.PI);

    /** Default {@code sigmaX} from the reference. */
    public static final double DEFAULT_SIGMA_X = 2.0e-4;
    /** Default {@code sigmaY} from the reference. */
    public static final double DEFAULT_SIGMA_Y = 1.0e-3;
    /** Default initial state {@code (vx, vy, px, py)} from the reference. */
    public static final double[] DEFAULT_X0 = {3.0e-3, -3.0e-3, 1.0, 1.0};

    private final double sigmaX;
    private final double sigmaY;
    private final double vx0;
    private final double vy0;
    private final double px0;
    private final double py0;

    // Cached constants.
    private final double logNormPX;   // -0.5*log(2π) - log(sigmaX), per Normal component
    private final double logNormPY;   // -0.5*log(2π) - log(sigmaY)
    private final double invSigmaX;
    private final double invSigmaY;

    /**
     * @param sigmaX velocity noise standard deviation (positive)
     * @param sigmaY bearing observation noise standard deviation (positive)
     * @param x0     initial state {@code (vx, vy, px, py)}, length 4
     */
    public BearingsOnly(double sigmaX, double sigmaY, double[] x0) {
        if (!(sigmaX > 0.0)) {
            throw new IllegalArgumentException("sigmaX must be > 0: " + sigmaX);
        }
        if (!(sigmaY > 0.0)) {
            throw new IllegalArgumentException("sigmaY must be > 0: " + sigmaY);
        }
        if (x0 == null || x0.length != 4) {
            throw new IllegalArgumentException(
                "x0 must be length 4 (vx, vy, px, py); got " + (x0 == null ? "null" : x0.length));
        }
        this.sigmaX = sigmaX;
        this.sigmaY = sigmaY;
        this.vx0 = x0[0];
        this.vy0 = x0[1];
        this.px0 = x0[2];
        this.py0 = x0[3];

        this.logNormPX = LOG_INV_SQRT_2PI - Math.log(sigmaX);
        this.logNormPY = LOG_INV_SQRT_2PI - Math.log(sigmaY);
        this.invSigmaX = 1.0 / sigmaX;
        this.invSigmaY = 1.0 / sigmaY;
    }

    /** Reference defaults. */
    public BearingsOnly() {
        this(DEFAULT_SIGMA_X, DEFAULT_SIGMA_Y, DEFAULT_X0);
    }

    public double sigmaX() { return sigmaX; }
    public double sigmaY() { return sigmaY; }
    public double[] x0()   { return new double[] { vx0, vy0, px0, py0 }; }

    @Override public int dim() { return 4; }

    /** Angle formula matching the reference: atan + pi branch for negative x. */
    public static double bearingAngle(double px, double py) {
        double a = Math.atan(py / px);
        if (px < 0.0) a += Math.PI;
        return a;
    }

    // ------------------------------------------------------------------
    // Transition (sampleM0, sampleM) — via ParticleOps
    // ------------------------------------------------------------------

    @Override
    public void sampleM0(StepContext<Double> ctx) {
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final int row0 = xOff;           // vx
        final int row1 = xOff + N;       // vy
        final int row2 = xOff + 2 * N;   // px
        final int row3 = xOff + 3 * N;   // py

        // vx_0 ~ N(vx0, sigmaX) — SIMD via ParticleOps
        ParticleOps.gaussianInto(X, row0, N, vx0, sigmaX, ctx.rng());

        // vy_0 ~ N(vy0, sigmaX) — SIMD via ParticleOps
        ParticleOps.gaussianInto(X, row1, N, vy0, sigmaX, ctx.rng());

        // px_0 = px0 (Dirac) — SIMD fill
        fillConstant(X, row2, N, px0);

        // py_0 = py0 (Dirac) — SIMD fill
        fillConstant(X, row3, N, py0);
    }

    @Override
    public void sampleM(StepContext<Double> ctx) {
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] Xprev = ctx.Xprev();
        final int xpOff = ctx.xpOff();

        final int row0 = xOff;           // vx
        final int row1 = xOff + N;       // vy
        final int row2 = xOff + 2 * N;   // px
        final int row3 = xOff + 3 * N;   // py

        final int xpRow0 = xpOff;           // prev vx
        final int xpRow1 = xpOff + N;       // prev vy
        final int xpRow2 = xpOff + 2 * N;   // prev px
        final int xpRow3 = xpOff + 3 * N;   // prev py

        // vx_t ~ N(vx_{t-1}, sigmaX) — AR(1) with c=0, rho=1
        ParticleOps.arOneInto(X, row0, N, 0.0, 1.0, Xprev, xpRow0, sigmaX, ctx.rng());

        // vy_t ~ N(vy_{t-1}, sigmaX) — AR(1) with c=0, rho=1
        ParticleOps.arOneInto(X, row1, N, 0.0, 1.0, Xprev, xpRow1, sigmaX, ctx.rng());

        // px_t = px_{t-1} + vx_{t-1} (Dirac) — SIMD addition
        addInto(X, row2, Xprev, xpRow2, Xprev, xpRow0, N);

        // py_t = py_{t-1} + vy_{t-1} (Dirac) — SIMD addition
        addInto(X, row3, Xprev, xpRow3, Xprev, xpRow1, N);
    }

    // ------------------------------------------------------------------
    // Observation (logG) — scalar atan per particle (no SIMD equivalent)
    // ------------------------------------------------------------------

    @Override
    public void logG(StepContext<Double> ctx) {
        // log p(Y_t | X_t) = logNormPY - 0.5 * ((y - angle(px, py)) / sigmaY)²
        final double y = ctx.observation();
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] logW = ctx.logW();
        final int lwOff = ctx.lwOff();
        final int rowPx = xOff + 2 * N;
        final int rowPy = xOff + 3 * N;
        final double inv = invSigmaY;
        final double logN = logNormPY;

        for (int n = 0; n < N; n++) {
            double px = X[rowPx + n];
            double py = X[rowPy + n];
            double angle = Math.atan(py / px);
            if (px < 0.0) angle += Math.PI;
            double z = (y - angle) * inv;
            logW[lwOff + n] = logN - 0.5 * z * z;
        }
    }

    // ------------------------------------------------------------------
    // TransitionDensity (for guided filter and smoothers)
    // ------------------------------------------------------------------

    @Override
    public TransitionDensity<Double> transitionDensity() { return this; }

    @Override
    public void logPt0(StepContext<Double> ctx, double[] out, int outOff) {
        // log p(X_0) = log N(vx; vx0, sigmaX) + log N(vy; vy0, sigmaX)
        //              + Dirac(px == px0) + Dirac(py == py0)
        // Returns -inf if Dirac components are violated.
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final int row0 = xOff;
        final int row1 = xOff + N;
        final int row2 = xOff + 2 * N;
        final int row3 = xOff + 3 * N;
        final double s = invSigmaX;
        final double logN = logNormPX;

        for (int n = 0; n < N; n++) {
            if (X[row2 + n] != px0 || X[row3 + n] != py0) {
                out[outOff + n] = Double.NEGATIVE_INFINITY;
                continue;
            }
            double zx = (X[row0 + n] - vx0) * s;
            double zy = (X[row1 + n] - vy0) * s;
            out[outOff + n] = 2.0 * logN - 0.5 * (zx * zx + zy * zy);
        }
    }

    @Override
    public void logPt(StepContext<Double> ctx, double[] out, int outOff) {
        // log p(X_t | X_{t-1}) = log N(vx_t; vx_{t-1}, sigmaX)
        //                       + log N(vy_t; vy_{t-1}, sigmaX)
        //                       + Dirac(px_t == px_{t-1} + vx_{t-1})
        //                       + Dirac(py_t == py_{t-1} + vy_{t-1})
        // Returns -inf if Dirac components are violated.
        final int N = ctx.N();
        final double[] X = ctx.X();
        final int xOff = ctx.xOff();
        final double[] Xprev = ctx.Xprev();
        final int xpOff = ctx.xpOff();
        final double s = invSigmaX;
        final double logN = logNormPX;

        final int row0 = xOff;
        final int row1 = xOff + N;
        final int row2 = xOff + 2 * N;
        final int row3 = xOff + 3 * N;
        final int xpRow0 = xpOff;
        final int xpRow1 = xpOff + N;
        final int xpRow2 = xpOff + 2 * N;
        final int xpRow3 = xpOff + 3 * N;

        for (int n = 0; n < N; n++) {
            double prevVx = Xprev[xpRow0 + n];
            double prevVy = Xprev[xpRow1 + n];
            double prevPx = Xprev[xpRow2 + n];
            double prevPy = Xprev[xpRow3 + n];
            double expectedPx = prevPx + prevVx;
            double expectedPy = prevPy + prevVy;
            if (X[row2 + n] != expectedPx || X[row3 + n] != expectedPy) {
                out[outOff + n] = Double.NEGATIVE_INFINITY;
                continue;
            }
            double zx = (X[row0 + n] - prevVx) * s;
            double zy = (X[row1 + n] - prevVy) * s;
            out[outOff + n] = 2.0 * logN - 0.5 * (zx * zx + zy * zy);
        }
    }

    @Override
    public double upperBound(int t) {
        // Two Normal(σX) components each at their mean contribute
        // 2*logNormPX; two Dirac components contribute 0.
        return 2.0 * logNormPX;
    }

    // ------------------------------------------------------------------
    // SIMD helpers
    // ------------------------------------------------------------------

    /**
     * Fills {@code out[off .. off+N)} with a constant value using SIMD.
     */
    private static void fillConstant(double[] out, int off, int N, double value) {
        final DoubleVector valVec = DoubleVector.broadcast(SPECIES, value);
        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            valVec.intoArray(out, off + i);
        }
        for (; i < N; i++) {
            out[off + i] = value;
        }
    }

    /**
     * Writes {@code out[outOff+n] = a[aOff+n] + b[bOff+n]} for n in [0, N).
     * SIMD vectorised with scalar tail.
     */
    private static void addInto(double[] out, int outOff,
                                double[] a, int aOff,
                                double[] b, int bOff, int N) {
        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector va = DoubleVector.fromArray(SPECIES, a, aOff + i);
            DoubleVector vb = DoubleVector.fromArray(SPECIES, b, bOff + i);
            va.add(vb).intoArray(out, outOff + i);
        }
        for (; i < N; i++) {
            out[outOff + i] = a[aOff + i] + b[bOff + i];
        }
    }
}
