package com.curioloop.yum4j.ssm.particle.sampler.nested;
import com.curioloop.yum4j.ssm.particle.sampler.FKSMCSampler;
import com.curioloop.yum4j.ssm.particle.sampler.StaticModel;
import com.curioloop.yum4j.ssm.particle.sampler.ThetaParticles;
import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequence;

import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;

import java.util.Arrays;

/**
 * Nested sampling SMC (Salomone et al 2018).
 *
 * <p>This is a Feynman-Kac SMC sampler where the target at time t is
 * the prior constrained to {@code L(θ) > l_t}. The next threshold
 * {@code l_t} is chosen as the {@code (1 - ESSrmin)} quantile of the
 * current likelihoods, ensuring that approximately {@code ESSrmin}
 * fraction of particles survive.
 *
 * <p>The algorithm terminates when the difference between the current
 * log-evidence estimate and the "final" estimate (taking
 * {@code l_t = +∞}) is below {@code eps}.
 *
 * <p>Successive thresholds are tracked on {@link ThetaParticles#lts} and
 * running log-evidence estimates on {@link ThetaParticles#logEvid}.
 *
 * <p>Operates directly on the engine's {@code Workspace.X} buffer via
 * {@link FKSMCSampler}'s arena aliasing.
 *
 * @see FKSMCSampler
 */
public final class NestedSamplingSMC extends FKSMCSampler {

    private final double ESSrmin;
    private final double eps;

    /** Scratch buffer for percentile computation (grown lazily). */
    private double[] sortScratch;

    /** Scratch buffer for "below threshold" likelihoods (grown lazily). */
    private double[] belowScratch;

    /**
     * @param model    static model defining prior and likelihood
     * @param move     MCMC move sequence
     * @param N        number of particles
     * @param lenChain MCMC chain length (for waste-free)
     * @param ESSrmin  fraction controlling the next threshold quantile
     *                 (corresponds to 1 - rho in Salomone et al)
     * @param eps      stopping criterion for log-evidence convergence
     */
    public NestedSamplingSMC(StaticModel model, MCMCSequence move,
                             int N, int lenChain, double ESSrmin, double eps) {
        super(model, move, N, lenChain, 0.0); // essRmin=0 since we always resample
        this.ESSrmin = ESSrmin;
        this.eps = eps;
    }

    @Override
    public boolean done(RunState rs) {
        if (theta == null || theta.lts == null || theta.ltsLen == 0) return false;
        return theta.lts[theta.ltsLen - 1] == Double.POSITIVE_INFINITY;
    }

    @Override
    protected void initializeParticles(int N) {
        int lpriorOff = theta.lpriorOff();
        int llikOff = theta.llikOff();
        int lpostOff = theta.lpostOff();
        // Fused lprior + full loglik evaluation.
        model.loglikAndLogprior(theta.arena, 0, N, -1,
            theta.arena, llikOff,
            theta.arena, lpriorOff);

        // lpost = lprior (target at t=0 is just the prior, lt = -inf)
        System.arraycopy(theta.arena, lpriorOff, theta.arena, lpostOff, N);

        // Initialise tracking arrays on the theta container.
        theta.appendLt(Double.NEGATIVE_INFINITY);
        theta.appendLogEvid(Double.NEGATIVE_INFINITY);
    }

    @Override
    protected void doLogG(int t, double[] Xprev, double[] X, int N,
                          double[] logW, int lwOff) {
        aliasTheta(X, N);

        int llikOff = theta.llikOff();
        // Re-evaluate likelihoods (the particles after mutation may have new theta values).
        model.loglik(theta.arena, 0, N, -1, theta.arena, llikOff);

        double currEvid = theta.logEvid[theta.logEvidLen - 1];

        // Compute next threshold as (1 - ESSrmin) quantile of likelihoods
        double lt = percentile(theta.arena, llikOff, N);

        // Estimate for non-terminal iteration:
        double lZt = computePartialEvidence(t, lt, N);
        double newEvid = logSumExpAB(currEvid, lZt);

        // Estimate at final time (lt = +inf)
        double lZtFinal = t * Math.log(ESSrmin) - Math.log(N)
            + LogWeight.logSumExp(theta.arena, llikOff, N);
        double newEvidFinal = logSumExpAB(currEvid, lZtFinal);

        if (Math.abs(newEvid - newEvidFinal) < eps) {
            lt = Double.POSITIVE_INFINITY;
            Arrays.fill(logW, lwOff, lwOff + N, 0.0);
            newEvid = newEvidFinal;
        } else {
            for (int n = 0; n < N; n++) {
                logW[lwOff + n] = theta.arena[llikOff + n] > lt ? 0.0 : Double.NEGATIVE_INFINITY;
            }
        }

        theta.appendLt(lt);
        theta.appendLogEvid(newEvid);
    }

    private double computePartialEvidence(int t, double lt, int N) {
        int llikOff = theta.llikOff();
        int count = 0;
        for (int n = 0; n < N; n++) {
            if (theta.arena[llikOff + n] <= lt) count++;
        }
        if (count == 0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (belowScratch == null || belowScratch.length < count) {
            belowScratch = new double[Math.max(count, belowScratch == null ? 16 : belowScratch.length * 2)];
        }
        int idx = 0;
        for (int n = 0; n < N; n++) {
            if (theta.arena[llikOff + n] <= lt) {
                belowScratch[idx++] = theta.arena[llikOff + n];
            }
        }
        return t * Math.log(ESSrmin) - Math.log(N) + LogWeight.logSumExp(belowScratch, 0, count);
    }

    @Override
    protected void evaluateTarget(ThetaParticles x) {
        int N = x.N;
        int d = x.dim;
        int lpriorOff = x.lpriorOff();
        int llikOff = x.llikOff();
        int lpostOff = x.lpostOff();
        ensureEvalScratch(d * N);
        // Fused lprior + full loglik evaluation.
        model.loglikAndLogprior(x.arena, 0, N, -1,
            x.arena, llikOff,
            x.arena, lpriorOff);

        // Current threshold lives on the theta container.
        double lt = (theta != null && theta.lts != null && theta.ltsLen > 0)
            ? theta.lts[theta.ltsLen - 1]
            : Double.NEGATIVE_INFINITY;

        if (lt == Double.NEGATIVE_INFINITY) {
            System.arraycopy(x.arena, lpriorOff, x.arena, lpostOff, N);
        } else {
            for (int n = 0; n < N; n++) {
                x.arena[lpostOff + n] = x.arena[llikOff + n] >= lt ? x.arena[lpriorOff + n] : Double.NEGATIVE_INFINITY;
            }
        }
    }

    /**
     * Compute the (1-ESSrmin)·100 percentile of arena[off..off+n).
     */
    private double percentile(double[] arena, int off, int n) {
        if (sortScratch == null || sortScratch.length < n) {
            sortScratch = new double[Math.max(n, sortScratch == null ? 16 : sortScratch.length * 2)];
        }
        System.arraycopy(arena, off, sortScratch, 0, n);
        Arrays.sort(sortScratch, 0, n);
        double p = 100.0 * (1.0 - ESSrmin);
        double idx = (p / 100.0) * (n - 1);
        int lo = (int) Math.floor(idx);
        int hi = (int) Math.ceil(idx);
        if (lo == hi || hi >= n) return sortScratch[Math.min(lo, n - 1)];
        double frac = idx - lo;
        return sortScratch[lo] * (1.0 - frac) + sortScratch[hi] * frac;
    }

    /** log(exp(a) + exp(b)) computed stably. */
    private static double logSumExpAB(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) return b;
        if (b == Double.NEGATIVE_INFINITY) return a;
        double m = Math.max(a, b);
        return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
    }
}
