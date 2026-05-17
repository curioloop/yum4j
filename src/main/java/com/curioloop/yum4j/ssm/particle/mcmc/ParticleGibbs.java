package com.curioloop.yum4j.ssm.particle.mcmc;

import com.curioloop.yum4j.ssm.particle.filter.FeynmanKac;
import com.curioloop.yum4j.ssm.particle.engine.*;
import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;

import java.util.List;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/**
 * Particle Gibbs sampler: alternates between sampling θ given x
 * (user-defined {@link #updateTheta}) and sampling x given θ via
 * conditional SMC ({@link CSMC}).
 *
 * <p>Iteration 0 runs an unconditional SMC (via {@link Engine}) to
 * initialise the trajectory. Subsequent iterations run CSMC with the
 * previous trajectory pinned, extract a new trajectory, then call
 * {@link #updateTheta}.
 *
 * <p>Uses the new engine infrastructure: {@link Engine}, {@link Workspace},
 * {@link RunState}, {@link FeynmanKac}.
 *
 */
public abstract class ParticleGibbs {

    protected final Function<double[], FeynmanKac<?>> fkFactory;
    protected final List<?> data;
    protected final int Nx;
    protected final int niter;
    protected final int dim;
    protected final int stateDim;
    protected final int T;
    protected final boolean storeX;

    /** Pre-allocated Workspace reused across iterations. */
    private final Workspace reusableWs;

    /** Pre-allocated RunState reused across iterations. */
    private final RunState reusableRs;

    /**
     * Create a Particle Gibbs sampler. Allocates an internal
     * {@link Workspace} sized {@code (Nx, stateDim, FULL, T)} and a
     * {@link RunState} of capacity {@code T}; both are reused across
     * MCMC iterations.
     *
     * @param fkFactory factory that creates a FeynmanKac from a parameter vector
     * @param data      observation sequence threaded into {@code Engine.run} and CSMC
     * @param Nx        number of particles
     * @param niter     number of MCMC iterations
     * @param dim       parameter dimension
     * @param stateDim  state dimension of the model
     * @param T         time horizon (must equal {@code data.size()})
     * @param storeX    whether to store trajectories in the chain
     */
    protected ParticleGibbs(Function<double[], FeynmanKac<?>> fkFactory,
                            List<?> data,
                            int Nx, int niter, int dim, int stateDim, int T,
                            boolean storeX) {
        this(fkFactory, data, Nx, niter, dim, stateDim, T, storeX, null, null);
    }

    /**
     * Create a Particle Gibbs sampler that delegates inner-filter
     * scratch to caller-supplied {@link Workspace} / {@link RunState}
     * instances. The caller (typically a {@code ParticleWorkspace})
     * owns the lifecycle of {@code reusableWs} / {@code reusableRs};
     * the sampler only resets and uses them. Pass {@code null} for
     * either to fall back to internal allocation of that buffer.
     *
     * @param fkFactory  factory that creates a FeynmanKac from a parameter vector
     * @param data       observation sequence threaded into {@code Engine.run} and CSMC
     * @param Nx         number of particles
     * @param niter      number of MCMC iterations
     * @param dim        parameter dimension
     * @param stateDim   state dimension of the model
     * @param T          time horizon (must equal {@code data.size()})
     * @param storeX     whether to store trajectories in the chain
     * @param externalWs workspace shaped {@code (Nx, stateDim, FULL, T)} owned by the caller, or {@code null}
     * @param externalRs run state with capacity at least {@code T} owned by the caller, or {@code null}
     */
    protected ParticleGibbs(Function<double[], FeynmanKac<?>> fkFactory,
                            List<?> data,
                            int Nx, int niter, int dim, int stateDim, int T,
                            boolean storeX,
                            Workspace externalWs, RunState externalRs) {
        this.fkFactory = fkFactory;
        this.data = data;
        this.Nx = Nx;
        this.niter = niter;
        this.dim = dim;
        this.stateDim = stateDim;
        this.T = T;
        this.storeX = storeX;
        this.reusableWs = externalWs != null
                ? externalWs
                : Workspace.allocate(Nx, stateDim, HistoryMode.FULL, T);
        this.reusableRs = externalRs != null
                ? externalRs
                : RunState.allocate(T, 0.5, Scheme.SYSTEMATIC, List.of(), 0L);
    }

    /**
     * User-defined theta update given the current trajectory.
     *
     * @param chain      the chain being built
     * @param iter       current iteration index
     * @param trajectory the current state trajectory, row-major (T, stateDim)
     * @param thetaCurr  current theta (length dim), modified in-place
     */
    protected abstract void updateTheta(MCMCChain chain, int iter,
                                        double[] trajectory, double[] thetaCurr);

    /**
     * Run the Particle Gibbs sampler.
     *
     * @param theta0 initial parameter vector (length dim)
     * @param g      random number generator
     * @return the MCMC chain
     */
    @SuppressWarnings("unchecked")
    public MCMCChain run(double[] theta0, RandomGenerator g) {
        MCMCChain chain = new MCMCChain(dim, niter);
        double[] thetaCurr = new double[dim];
        System.arraycopy(theta0, 0, thetaCurr, 0, dim);

        double[] trajectory = null;

        // Single RandomBatch reused across every iteration (R9.3). The
        // master generator g still advances once per iteration via the
        // run-state seed pulled from sharedRng below; the RandomBatch
        // itself is allocated exactly once per run().
        RandomBatch sharedRng = RandomBatch.of(g.nextLong());
        reusableWs.rng = sharedRng;

        for (int iter = 0; iter < niter; iter++) {
            // Create FK model for current theta
            FeynmanKac<?> fk = fkFactory.apply(thetaCurr);

            if (iter == 0) {
                // Unconditional SMC to initialise trajectory.
                // Re-use sharedRng across iterations; pull a long from the
                // shared stream to seed the RunState reset so iterations
                // remain decorrelated without allocating a new RandomBatch.
                reusableRs.reset(sharedRng.asRandomGenerator().nextLong());
                reusableWs.resetBuffers();

                Engine.run(reusableWs, reusableRs, fk, ParallelStrategy.SERIAL, (List) data);

                // Extract trajectory of particle 0 from the unconditional run
                trajectory = extractTrajectoryFromRun(reusableWs, reusableRs);
            } else {
                // Conditional SMC with previous trajectory pinned. Pass the
                // shared RandomBatch directly so CSMC reuses it instead of
                // allocating a fresh one per call (R9.3, R9.4).
                CSMC.CsmcRun result = CSMC.run(fk, reusableWs, reusableRs,
                    trajectory, (List) data, sharedRng);
                trajectory = CSMC.extractTrajectory(result);
            }

            // Update theta given trajectory
            updateTheta(chain, iter, trajectory, thetaCurr);

            // Record
            chain.setTheta(iter, thetaCurr, 0);
        }

        return chain;
    }

    /**
     * Extract the trajectory of particle 0 from an unconditional SMC run.
     * Uses the same backward-tracing logic as CSMC.
     *
     * @param ws completed workspace with FULL history
     * @param rs completed run state
     * @return trajectory as row-major (T, dim): out[t * dim + j]
     */
    private double[] extractTrajectoryFromRun(Workspace ws, RunState rs) {
        int d = ws.dim;
        int steps = rs.stepCount;

        double[] traj = new double[d * steps];

        // Start from particle 0 at the last step
        int idx = 0;

        // Copy particle at last step from workspace (current X)
        System.arraycopy(ws.X, idx * d, traj, (steps - 1) * d, d);

        // Trace backwards through history using zero-copy view methods
        // with a pair of reusable scratch buffers.
        if (ws.history != null) {
            int N = ws.N;
            int[] ancBuf = new int[N];
            double[] xBuf = new double[d * N];
            for (int t = steps - 2; t >= 0; t--) {
                ws.history.viewAncestors(t + 1, ancBuf, 0);
                idx = ancBuf[idx];
                ws.history.viewX(t, xBuf, 0);
                System.arraycopy(xBuf, idx * d, traj, t * d, d);
            }
        }

        return traj;
    }
}
