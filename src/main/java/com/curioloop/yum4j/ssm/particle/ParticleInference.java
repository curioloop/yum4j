package com.curioloop.yum4j.ssm.particle;

/**
 * Contract for a configured particle-filter inference that can run with
 * an optional reusable workspace.
 *
 * <pre>{@code
 * try (ParticleWorkspace ws = Particle.workspace()) {
 *     FilterResult<Double> r1 = Particle.filter(model)
 *         .particles(10_000).observations(data1).run(ws);
 *     FilterResult<Double> r2 = Particle.filter(model)
 *         .particles(10_000).observations(data2).run(ws);   // workspace reused
 * }
 * }</pre>
 *
 * @param <R> result type produced by {@link #run(Object)}
 * @param <W> workspace type accepted by {@link #run(Object)}
 */
public interface ParticleInference<R, W> {

    /**
     * Runs the configured inference with a fresh, throw-away workspace.
     * Equivalent to {@code run(null)}.
     *
     * @return the inference result
     */
    default R run() {
        return run(null);
    }

    /**
     * Runs the configured inference, optionally reusing an existing
     * workspace.
     *
     * <p>Implementations MAY ignore the workspace argument when their
     * algorithm has no reusable scratch (e.g. PMCMC chains that own
     * dedicated arenas internally), in which case passing {@code null}
     * is equivalent to passing any workspace and no allocation savings
     * apply.
     *
     * @param workspace a reusable workspace, or {@code null} to allocate
     *                  fresh scratch
     * @return the inference result
     */
    R run(W workspace);
}
