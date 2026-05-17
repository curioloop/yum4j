package com.curioloop.yum4j.ssm.particle.sampler;

import com.curioloop.yum4j.ssm.particle.dist.MultivariateDistribution;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Abstract base class for static (non-state-space) models used by
 * IBIS and tempering SMC samplers.
 *
 * <p>A static model defines a per-observation log-density
 * {@link #logpyt(double[], int, int, int, double[], int)} that the user
 * implements. The base class provides {@link #loglik} (cumulative sum of
 * {@code logpyt} from time 0 to t) and {@link #logpost} (log-prior +
 * log-likelihood).
 *
 * <p>The theta buffer follows the project-wide column-major {@code (dim, N)}
 * layout: element {@code theta[j, n]} lives at {@code theta[thetaOff + j * N + n]}.
 *
 * <h3>Combined evaluators</h3>
 * <p>Samplers that need both {@code loglik} and {@code logprior} (e.g.
 * the IBIS/Tempering MCMC target) should prefer
 * {@link #loglikAndLogprior} over separate calls; it avoids the
 * redundant {@link #logpyt} pass inside {@link #logpost}.
 *
 * <h3>Allocation profile</h3>
 * <p>Per-instance scratch buffers are grown lazily on first use and
 * reused across calls. No allocation occurs on the hot path once
 * warmed up.
 *
 * @see ThetaParticles
 */
public abstract class StaticModel {

    /** Prior distribution over the parameter vector θ. */
    protected final MultivariateDistribution prior;

    /** Number of observations. */
    protected final int T;

    /** Scratch buffer for per-step logpyt accumulation (length N). */
    private double[] scratch;

    /** Scratch buffer for prior logPdfBatch (length dim * N). */
    private double[] priorScratch;

    /** Scratch buffer for prior output (length N). */
    private double[] priorOut;

    /**
     * Constructs a static model with the given prior and number of observations.
     *
     * @param prior prior distribution over θ (dimension d_θ)
     * @param T     number of observations
     */
    protected StaticModel(MultivariateDistribution prior, int T) {
        this.prior = prior;
        this.T = T;
    }

    /**
     * Returns the number of observations.
     */
    public int T() {
        return T;
    }

    /**
     * Returns the prior distribution.
     */
    public MultivariateDistribution prior() {
        return prior;
    }

    /**
     * Returns the parameter dimension d_θ.
     */
    public int dim() {
        return prior.dim();
    }

    /**
     * Log-density of observation Y_t given parameter θ and past data.
     *
     * <p>This method operates on N particles simultaneously (batch).
     * The theta buffer is column-major {@code (dim, N)}: element
     * {@code theta[j, n]} is at {@code theta[thetaOff + j * N + n]}.
     *
     * <p>Implementations write one log-density value per particle into
     * {@code out[outOff + n]} for {@code n in [0, N)}.
     */
    public abstract void logpyt(double[] theta, int thetaOff, int N, int t,
                                double[] out, int outOff);

    /**
     * Cumulative log-likelihood up to time t (inclusive).
     *
     * <p>Computes for each particle n:
     * {@code out[outOff + n] = sum_{s=0}^{t} logpyt(theta_n, s)}.
     *
     * <p>If {@code t < 0}, the full log-likelihood over all T observations
     * is computed (equivalent to {@code t = T - 1}).
     *
     * <p>Any NaN produced by {@code logpyt} is replaced with
     * {@code Double.NEGATIVE_INFINITY} in the cumulative sum.
     */
    public void loglik(double[] theta, int thetaOff, int N, int t,
                       double[] out, int outOff) {
        if (t < 0) {
            t = T - 1;
        }
        if (scratch == null || scratch.length < N) {
            scratch = new double[N];
        }
        // Zero the output
        java.util.Arrays.fill(out, outOff, outOff + N, 0.0);
        for (int s = 0; s <= t; s++) {
            logpyt(theta, thetaOff, N, s, scratch, 0);
            accumulateWithNanGuard(scratch, 0, out, outOff, N);
        }
    }

    /**
     * Log-posterior = log-prior + log-likelihood.
     *
     * <p>Equivalent to but cheaper than
     * {@code loglik(...) + prior.logPdfBatch(...)} when the caller needs
     * only the posterior; if the caller needs both {@code loglik} and
     * {@code lprior} separately, prefer
     * {@link #loglikAndLogprior} which avoids duplicated work.
     */
    public void logpost(double[] theta, int thetaOff, int N, int t,
                        double[] out, int outOff) {
        loglik(theta, thetaOff, N, t, out, outOff);
        ensurePriorScratch(N);
        prior.logPdfBatch(theta, thetaOff, N, priorOut, 0, priorScratch);
        for (int n = 0; n < N; n++) {
            out[outOff + n] += priorOut[n];
        }
    }

    /**
     * Fused {@code loglik} + {@code logprior} evaluator.
     *
     * <p>Computes both the cumulative log-likelihood up to time
     * {@code t} (stored in {@code llikOut[llikOff..llikOff+N)}) and the
     * log-prior (stored in {@code lpriorOut[lpriorOff..lpriorOff+N)})
     * for all {@code N} particles in a single pass, avoiding the
     * duplicate {@code logpyt} accumulation that naive
     * {@code loglik + logpost} performs.
     *
     * <p>Used by {@link IBIS#evaluateTarget} and {@link Tempering#evaluateTarget}.
     *
     * @param theta      parameter buffer, column-major (dim, N)
     * @param thetaOff   offset into {@code theta}
     * @param N          number of particles
     * @param t          time index (inclusive upper bound); negative means full likelihood
     * @param llikOut    output buffer for the log-likelihoods (length ≥ llikOff + N)
     * @param llikOff    offset into {@code llikOut}
     * @param lpriorOut  output buffer for the log-priors (may alias {@code llikOut} at a different offset)
     * @param lpriorOff  offset into {@code lpriorOut}
     */
    public void loglikAndLogprior(double[] theta, int thetaOff, int N, int t,
                                  double[] llikOut, int llikOff,
                                  double[] lpriorOut, int lpriorOff) {
        loglik(theta, thetaOff, N, t, llikOut, llikOff);
        ensurePriorScratch(N);
        prior.logPdfBatch(theta, thetaOff, N, lpriorOut, lpriorOff, priorScratch);
    }

    // ------------------------------------------------------------------
    // Scratch management.
    // ------------------------------------------------------------------

    private void ensurePriorScratch(int N) {
        int dim = prior.dim();
        int needed = dim * N;
        if (priorScratch == null || priorScratch.length < needed) {
            priorScratch = new double[needed];
        }
        if (priorOut == null || priorOut.length < N) {
            priorOut = new double[N];
        }
    }

    // ------------------------------------------------------------------
    // SIMD accumulator with NaN→-Infinity guard.
    //
    // Hot path: called T times per loglik() call. HotSpot will not
    // auto-vectorise the NaN branch, so we hand-SIMD it. A fast path
    // checks for NaN-freedom on each LANES-sized chunk and uses a
    // straight add when clean — typical in practice because logpyt
    // rarely emits NaN once the model is debugged.
    // ------------------------------------------------------------------

    private static final String SIMD_PROPERTY = "yum4j.vector";
    private static final boolean SIMD_ENABLED =
        !"false".equalsIgnoreCase(System.getProperty(SIMD_PROPERTY, "true"));
    private static final VectorSpecies<Double> SPECIES = preferredSpecies256Cap();
    private static final int LANES = SPECIES.length();
    private static final int SIMD_MIN_N = 32;
    private static final boolean SIMD_SUPPORTED = detectSimd();
    private static final DoubleVector NEG_INF_VEC =
        SIMD_SUPPORTED ? DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY) : null;

    private static VectorSpecies<Double> preferredSpecies256Cap() {
        VectorSpecies<Double> preferred = DoubleVector.SPECIES_PREFERRED;
        if (preferred.length() >= DoubleVector.SPECIES_256.length()) {
            return DoubleVector.SPECIES_256;
        }
        return preferred;
    }

    private static boolean detectSimd() {
        if (!SIMD_ENABLED) return false;
        if (LANES <= 1) return false;
        try {
            double[] probe = new double[LANES * 2];
            double[] out = new double[LANES * 2];
            for (int i = 0; i < probe.length; i++) probe[i] = i * 0.25;
            simdAccumulate(probe, 0, out, 0, probe.length);
            return true;
        } catch (LinkageError e) {
            return false;
        }
    }

    /**
     * Accumulate {@code out[i] += (isNaN(incr[i]) ? -Inf : incr[i])}
     * for {@code i in [0, n)}. SIMD-accelerated with a NaN-clean fast
     * path; falls back to scalar when SIMD is unavailable or N is small.
     */
    static void accumulateWithNanGuard(double[] incr, int incrOff,
                                       double[] out, int outOff, int n) {
        if (SIMD_SUPPORTED && n >= SIMD_MIN_N) {
            simdAccumulate(incr, incrOff, out, outOff, n);
            return;
        }
        for (int i = 0; i < n; i++) {
            double v = incr[incrOff + i];
            if (Double.isNaN(v)) v = Double.NEGATIVE_INFINITY;
            out[outOff + i] += v;
        }
    }

    private static void simdAccumulate(double[] incr, int incrOff,
                                       double[] out, int outOff, int n) {
        final int bound = SPECIES.loopBound(n);
        int i = 0;
        for (; i < bound; i += LANES) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, incr, incrOff + i);
            // NaN != NaN → mask selects NaN lanes.
            VectorMask<Double> nanMask = v.compare(VectorOperators.NE, v);
            DoubleVector guarded;
            if (nanMask.anyTrue()) {
                guarded = v.blend(NEG_INF_VEC, nanMask);
            } else {
                guarded = v; // NaN-free fast path — skip the blend
            }
            DoubleVector o = DoubleVector.fromArray(SPECIES, out, outOff + i);
            o.add(guarded).intoArray(out, outOff + i);
        }
        for (; i < n; i++) {
            double v = incr[incrOff + i];
            if (Double.isNaN(v)) v = Double.NEGATIVE_INFINITY;
            out[outOff + i] += v;
        }
    }
}
