package com.curioloop.yum4j.ssm.particle.mcmc;

import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.engine.*;
import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.resample.Resample;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Conditional SMC (CSMC): a particle filter that pins particle 0 to a
 * reference trajectory x* at every time step.
 *
 * <p>Uses the new {@link Engine} infrastructure ({@link Workspace},
 * {@link RunState}, {@link FeynmanKac}) with a manual step loop that
 * overwrites particle 0 with the reference trajectory and forces
 * {@code a[0] = 0} after resampling.
 *
 * <p>The {@code RandomBatch}-accepting overloads (R9.4) are the canonical
 * entry points used by samplers (PMMH, ParticleGibbs) that own the RNG
 * lifecycle and want to share a single batched stream across every CSMC
 * call. The {@link RandomGenerator}-accepting overloads remain for
 * stand-alone callers; they construct a fresh {@code RandomBatch} from
 * one {@code g.nextLong()} draw and delegate to the batched form.
 *
 * @see ParticleGibbs
 */
public final class CSMC {

    private CSMC() {}

    /**
     * Run conditional SMC with a pinned reference trajectory.
     *
     * <p>Allocates a fresh {@link Workspace} and {@link RunState} internally
     * and delegates to the {@link RandomBatch}-accepting overload.
     *
     * @param fk    the Feynman-Kac model
     * @param N     number of particles
     * @param xstar reference trajectory, row-major (T, dim): xstar[t * dim + j]
     * @param data  observation sequence threaded into the engine-style inner loop
     * @param g     random number generator; one {@code nextLong} draw seeds
     *              the underlying {@link RandomBatch}
     */
    public static <Y> CsmcRun run(FeynmanKac<Y> fk, int N,
                                     double[] xstar, List<Y> data,
                                     RandomGenerator g) {
        return run(fk, N, xstar, data, RandomBatch.of(g.nextLong()));
    }

    /**
     * Run conditional SMC with a pinned reference trajectory using a
     * caller-supplied {@link RandomBatch}.
     *
     * <p>Allocates a fresh {@link Workspace} and {@link RunState} internally
     * and reuses {@code rng} as both the workspace RNG and the source of
     * the per-run {@link RunState#reset(long)} seed. This is the entry
     * point used by samplers (PMMH, ParticleGibbs) so the batched stream
     * is shared across every CSMC call without per-call allocation (R9.4).
     *
     * @param fk    the Feynman-Kac model
     * @param N     number of particles
     * @param xstar reference trajectory, row-major (T, dim): xstar[t * dim + j]
     * @param data  observation sequence threaded into the engine-style inner loop
     * @param rng   shared {@link RandomBatch} owned by the caller
     */
    public static <Y> CsmcRun run(FeynmanKac<Y> fk, int N,
                                     double[] xstar, List<Y> data,
                                     RandomBatch rng) {
        return run(fk, N, xstar, data, rng, null, null);
    }

    /**
     * Run conditional SMC with caller-supplied {@code (Workspace, RunState)}
     * when available; falls back to fresh allocation when either is
     * {@code null}. Lets call sites that may or may not have a reusable
     * scratch container thread through a single entry point.
     *
     * @param fk    the Feynman-Kac model
     * @param N     number of particles (used for fallback allocation when
     *              {@code externalWs} is {@code null})
     * @param xstar reference trajectory, row-major (T, dim)
     * @param data  observation sequence
     * @param rng   shared {@link RandomBatch} owned by the caller
     * @param externalWs caller-owned workspace shaped {@code (N, fk.dim(), FULL, T)}, or {@code null}
     * @param externalRs caller-owned run state with capacity at least {@code T}, or {@code null}
     */
    public static <Y> CsmcRun run(FeynmanKac<Y> fk, int N,
                                  double[] xstar, List<Y> data,
                                  RandomBatch rng,
                                  Workspace externalWs, RunState externalRs) {
        int dim = fk.dim();
        int T = data.size();

        Workspace ws = externalWs != null
                ? externalWs
                : Workspace.allocate(N, dim, HistoryMode.FULL, T);
        RunState rs = externalRs != null
                ? externalRs
                : RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, List.of(), 0L);
        ws.rng = rng;

        return run(fk, ws, rs, xstar, data, rng);
    }

    /**
     * Overload that synthesises a placeholder observation list from
     * {@code xstar.length / fk.dim()}. Used by samplers that supplied the
     * observations to their FK factory closure but no longer have a handle
     * on the {@code List<Y> data}; the observations-from-list path is
     * short-circuited because bootstrap/guided read {@code ctx.observation}
     * directly and auxiliary keeps its own list.
     *
     * <p>The placeholder list entries are {@code null}; do not call this
     * variant for FKs that rely on {@code ctx.observation()} being non-null
     * in kernels that reach it (e.g. bootstrap/guided on non-null observation
     * types). For bootstrap/guided parameterised over a model that tolerates
     * a null observation in {@code logG}, it is safe.
     *
     * @deprecated Prefer the overload that takes an explicit {@code List<Y> data}.
     */
    @Deprecated
    public static <Y> CsmcRun run(FeynmanKac<Y> fk, int N,
                                     double[] xstar, RandomGenerator g) {
        int dim = fk.dim();
        int T = xstar.length / dim;
        List<Y> placeholder = new ArrayList<>(Collections.nCopies(T, null));
        return run(fk, N, xstar, placeholder, g);
    }

    /**
     * Run conditional SMC with a pinned reference trajectory, reusing a
     * pre-allocated {@link Workspace} and {@link RunState}. Delegates to
     * the {@link RandomBatch}-accepting overload after wrapping {@code g}
     * in a fresh {@link RandomBatch}.
     *
     * @param fk    the Feynman-Kac model
     * @param ws    pre-allocated workspace (must have FULL history)
     * @param rs    pre-allocated run state
     * @param xstar reference trajectory, row-major (T, dim): xstar[t * dim + j]
     * @param data  observation sequence; {@code data.size()} gives the horizon
     * @param g     random number generator; one {@code nextLong} draw seeds
     *              the underlying {@link RandomBatch}
     * @return the CSMC result containing workspace and run state
     */
    public static <Y> CsmcRun run(FeynmanKac<Y> fk, Workspace ws, RunState rs,
                                     double[] xstar, List<Y> data,
                                     RandomGenerator g) {
        return run(fk, ws, rs, xstar, data, RandomBatch.of(g.nextLong()));
    }

    /**
     * Run conditional SMC with a pinned reference trajectory, reusing a
     * pre-allocated {@link Workspace}, {@link RunState}, and
     * {@link RandomBatch}. The state is reset before use, avoiding
     * allocation on repeated calls (e.g. inside {@link ParticleGibbs}).
     *
     * <p>This is the canonical entry point for samplers that own the RNG
     * lifecycle (R9.4). The {@code rng} reference is installed as
     * {@link Workspace#rng} and the run-state is reset with a long pulled
     * from the shared stream so chain determinism is preserved without
     * re-allocating a {@code RandomBatch} on every call.
     *
     * @param fk    the Feynman-Kac model
     * @param ws    pre-allocated workspace (must have FULL history)
     * @param rs    pre-allocated run state
     * @param xstar reference trajectory, row-major (T, dim): xstar[t * dim + j]
     * @param data  observation sequence; {@code data.size()} gives the horizon
     * @param rng   shared {@link RandomBatch} owned by the caller
     * @return the CSMC result containing workspace and run state
     */
    public static <Y> CsmcRun run(FeynmanKac<Y> fk, Workspace ws, RunState rs,
                                     double[] xstar, List<Y> data,
                                     RandomBatch rng) {
        int N = ws.N;
        int dim = ws.dim;
        int T = data.size();

        // Reset state for a fresh run. The seed comes from the shared stream
        // so successive CSMC calls see distinct run-state seeds while never
        // allocating a new RandomBatch (R9.4).
        rs.reset(rng.asRandomGenerator().nextLong());
        ws.rng = rng;
        ws.resetBuffers();

        // Step 0: init via FeynmanKac, then overwrite particle 0 with xstar[0]
        StepContext<Y> ctx0 = makeCtx(ws, 0, 0, N, data.get(0));
        fk.init(ctx0);
        overwriteParticle0(ws.X, xstar, 0, dim, T);

        // Fused reduction at t=0
        LogWeight.Triple r0 = LogWeight.logSumEssMax(ws.logW, 0, N);
        rs.publishReduction(0, r0);
        rs.logLtSeries[0] = r0.logSum() - Math.log(N);
        rs.resampledFlags[0] = false;
        System.arraycopy(ws.identity, 0, ws.a, 0, N);

        // Save history at t=0
        if (ws.history != null) {
            ws.history.save(0, ws, false);
        }
        rs.stepCount = 1;

        // Steps 1..T-1
        for (int t = 1; t < T; t++) {
            double logSumBefore;
            boolean resampled = rs.essCache < rs.essRmin * N;

            // Resampling decision
            if (resampled) {
                Resample.apply(rs.scheme, ws.logW, N, N, ws.a, ws.rng, ws.scratch);
                // Force a[0] = 0 (pin reference trajectory lineage)
                ws.a[0] = 0;
                gather(ws.X, ws.Xprev, ws.a, dim, N);
                Arrays.fill(ws.logW, 0, N, 0.0);
                logSumBefore = Math.log(N);
                rs.markResampled(t);
            } else {
                System.arraycopy(ws.identity, 0, ws.a, 0, N);
                ws.swapXandXprev();
                logSumBefore = rs.logSumCache;
                rs.markNotResampled(t);
            }

            // Advance via FeynmanKac
            StepContext<Y> ctxT = makeCtx(ws, t, 0, N, data.get(t));
            fk.advance(ctxT);

            // Overwrite particle 0 with xstar[t]
            overwriteParticle0(ws.X, xstar, t, dim, T);

            // Fused reduction
            LogWeight.Triple rt = LogWeight.logSumEssMax(ws.logW, 0, N);
            rs.publishReduction(t, rt);
            rs.logLtSeries[t] = rs.logLtSeries[t - 1] + (rt.logSum() - logSumBefore);

            // Save history
            if (ws.history != null) {
                ws.history.save(t, ws, resampled);
            }
            rs.stepCount = t + 1;
        }

        return new CsmcRun(ws, rs);
    }

    /**
     * Overwrite particle 0 in X with xstar at time t.
     * xstar layout: row-major (T, dim) — xstar[t * dim + j].
     * X layout: row-major (N, dim) — X[0 * dim + j] = X[j].
     *
     * <p>Note: The new engine uses row-major layout for X: X[n*dim + j].
     * Particle 0 occupies X[0..dim-1].
     */
    private static void overwriteParticle0(double[] X, double[] xstar,
                                           int t, int dim, int T) {
        System.arraycopy(xstar, t * dim, X, 0, dim);
    }

    /**
     * Extract the trajectory of particle 0 by tracing ancestry backwards.
     *
     * @param result completed CSMC result with FULL history
     * @return trajectory as row-major (T, dim): out[t * dim + j]
     */
    public static double[] extractTrajectory(CsmcRun result) {
        Workspace ws = result.workspace();
        RunState rs = result.runState();
        int dim = ws.dim;
        int T = rs.stepCount;

        double[] traj = new double[dim * T];

        // Start from particle 0 at the last step
        int idx = 0;

        // Copy particle at T-1 from workspace (current X)
        System.arraycopy(ws.X, idx * dim, traj, (T - 1) * dim, dim);

        // Trace backwards through history using zero-copy view methods.
        if (ws.history != null) {
            int N = ws.N;
            int[] ancBuf = new int[N];
            double[] xBuf = new double[dim * N];
            for (int t = T - 2; t >= 0; t--) {
                ws.history.viewAncestors(t + 1, ancBuf, 0);
                idx = ancBuf[idx];
                ws.history.viewX(t, xBuf, 0);
                System.arraycopy(xBuf, idx * dim, traj, t * dim, dim);
            }
        }

        return traj;
    }

    /**
     * Gathers particles from {@code src} into {@code dst} according to
     * ancestor indices.
     */
    private static void gather(double[] src, double[] dst, int[] a, int dim, int N) {
        if (dim == 1) {
            for (int n = 0; n < N; n++) {
                dst[n] = src[a[n]];
            }
        } else {
            for (int n = 0; n < N; n++) {
                int srcOff = a[n] * dim;
                int dstOff = n * dim;
                System.arraycopy(src, srcOff, dst, dstOff, dim);
            }
        }
    }

    /**
     * Builds a StepContext for the full particle range.
     */
    private static <Y> StepContext<Y> makeCtx(Workspace ws, int t, int off, int count, Y obs) {
        return new StepContext<>(
                t,
                obs,
                ws.Xprev, off * ws.dim,
                ws.X, off * ws.dim,
                ws.logW, off,
                count, ws.dim,
                ws.rng,
                ws.scratch, off
        );
    }

    /**
     * Result container for a CSMC run.
     */
    public record CsmcRun(Workspace workspace, RunState runState) {
        /**
         * Returns the final log marginal likelihood estimate.
         */
        public double logLikelihood() {
            int lastStep = runState.stepCount - 1;
            return lastStep >= 0 ? runState.logLtSeries[lastStep] : 0.0;
        }
    }
}
