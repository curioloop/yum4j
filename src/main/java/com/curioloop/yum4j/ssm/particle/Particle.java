package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.model.ParticleSSM;

/**
 * User-facing facade for sequential Monte Carlo (particle) inference.
 *
 * <pre>{@code
 * // One-shot filter, fresh workspace each call:
 * FilterResult<Double> result = Particle.filter(model)
 *     .particles(10_000)
 *     .observations(data)
 *     .resampling(Scheme.SYSTEMATIC)
 *     .seed(42L)
 *     .run();
 *
 * // Reuse a workspace across many runs:
 * try (ParticleWorkspace ws = Particle.workspace()) {
 *     for (List<Double> dataset : datasets) {
 *         FilterResult<Double> r = Particle.filter(model)
 *             .particles(10_000)
 *             .observations(dataset)
 *             .run(ws);
 *         consume(r);
 *     }
 * }
 *
 * // Step-by-step online filtering on a streaming feed:
 * try (ParticleWorkspace ws = Particle.workspace()) {
 *     ParticleOnline.Run<Double> run = Particle.online(model)
 *         .particles(10_000)
 *         .seed(42L)
 *         .start(ws);
 *     for (Double obs : stream) {
 *         run.step(obs);
 *         consume(run.logLikelihood(), run.ess());
 *     }
 * }
 *
 * // Smooth a completed filter run:
 * try (ParticleWorkspace ws = Particle.workspace()) {
 *     FilterResult<Double> r = Particle.filter(model)
 *         .particles(10_000)
 *         .observations(data)
 *         .history(HistoryMode.FULL)
 *         .run(ws);
 *     TrajectoryBundle<Double> traj = Particle.smooth(r)
 *         .ffbs()
 *         .trajectories(100)
 *         .run(ws);
 * }
 *
 * // Typed particle sample, returning the existing PMCMC result type:
 * PmmhResult chain = Particle.sample(SamplingMethod.PMMH)
 *     .prior(prior)
 *     .model(modelFactory, data)
 *     .particles(1_000)
 *     .iterations(20_000)
 *     .run();
 * }</pre>
 *
 * <p>SMC sampler algorithms (IBIS, SMC², tempering, nested sampling) and
 * particle-MCMC algorithms (PMMH, Particle Gibbs, Conditional SMC) are
 * exposed only through {@link #sample(SamplingMethod)}; the previous
 * {@code api.SMCSampler} and {@code api.PMCMC} façades have been retired.
 */
public final class Particle {

    private Particle() {
    }

    /**
     * Creates a fresh, reusable workspace for the fluent facade.
     * Use try-with-resources to scope reuse:
     *
     * <pre>{@code
     * try (ParticleWorkspace ws = Particle.workspace()) {
     *     // ... inference calls ...
     * }
     * }</pre>
     */
    public static ParticleWorkspace workspace() {
        return new ParticleWorkspace();
    }

    /**
     * Configures a particle-filter run over the given state-space model.
     *
     * @param ssm the state-space model to filter
     * @param <Y> observation type
     * @return a fluent filter configuration object
     * @throws NullPointerException if {@code ssm} is null
     */
    public static <Y> ParticleFiltering<Y> filter(ParticleSSM<Y> ssm) {
        if (ssm == null) {
            throw new NullPointerException("ssm must not be null");
        }
        return new ParticleFiltering<>(ssm);
    }

    /**
     * Configures a step-by-step online particle filter, returning a
     * fluent builder that produces a stateful {@link ParticleOnline.Run}
     * via {@link ParticleOnline#start()} or
     * {@link ParticleOnline#start(ParticleWorkspace)}. The run processes
     * observations incrementally — one {@code step(obs)} call each —
     * while reusing a single workspace and run state.
     *
     * @param ssm the state-space model to filter
     * @param <Y> observation type
     * @return a fluent online-filter configuration object
     * @throws NullPointerException if {@code ssm} is null
     */
    public static <Y> ParticleOnline<Y> online(ParticleSSM<Y> ssm) {
        if (ssm == null) {
            throw new NullPointerException("ssm must not be null");
        }
        return new ParticleOnline<>(ssm);
    }

    /**
     * Configures a smoother that consumes a previously-computed
     * {@link FilterResult}. The result must have been produced with a
     * non-{@code NONE} history mode (R8.4).
     *
     * @param result a completed filter result
     * @param <Y>    observation type
     * @return a fluent smoother configuration object
     * @throws NullPointerException if {@code result} is null
     */
    public static <Y> ParticleSmoothing<Y> smooth(FilterResult<Y> result) {
        if (result == null) {
            throw new NullPointerException("result must not be null");
        }
        return new ParticleSmoothing<>(result);
    }

    /**
     * Configures a smoother that runs the forward filter internally.
     * The smoother selects an appropriate history mode automatically:
     * {@code FULL} for FFBS / TwoFilter / Paris and {@code ROLLING} for
     * FixedLag.
     *
     * @param ssm the state-space model
     * @param <Y> observation type
     * @return a fluent smoother configuration object
     * @throws NullPointerException if {@code ssm} is null
     */
    public static <Y> ParticleSmoothing<Y> smooth(ParticleSSM<Y> ssm) {
        if (ssm == null) {
            throw new NullPointerException("ssm must not be null");
        }
        return new ParticleSmoothing<>(ssm);
    }

    /**
    * Creates a typed particle sample from its method.
     *
     * @param method sampling method token
     * @param <S>    sampling task type
     * @param <R>    sampling result type
    * @return a method-specific sample builder
     * @throws NullPointerException if {@code method} is null
     */
    public static <S extends ParticleSampling<R>, R> S sample(SamplingMethod<S, R> method) {
        if (method == null) {
            throw new NullPointerException("method must not be null");
        }
        return method.create();
    }
}
