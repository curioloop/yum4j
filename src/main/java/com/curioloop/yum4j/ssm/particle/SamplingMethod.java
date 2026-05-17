package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.mcmc.CSMCResult;
import com.curioloop.yum4j.ssm.particle.mcmc.GibbsResult;
import com.curioloop.yum4j.ssm.particle.mcmc.PMMHResult;
import com.curioloop.yum4j.ssm.particle.sampler.IBISResult;
import com.curioloop.yum4j.ssm.particle.sampler.SMC2Result;
import com.curioloop.yum4j.ssm.particle.sampler.TemperingResult;
import com.curioloop.yum4j.ssm.particle.sampler.nested.NestedSamplingResult;

import java.util.Objects;
import java.util.function.Supplier;

/**
 * Typed replacement for a generic enum of particle sampling methods.
 *
 * @param name method name
 * @param factory factory for the method-specific sampling builder
 * @param <S> configured sampling type
 * @param <R> run result type
 */
public record SamplingMethod<S extends ParticleSampling<R>, R>(
        String name,
        Supplier<? extends S> factory) {

    public static final SamplingMethod<ParticleSampling.PMMH<?>, PMMHResult> PMMH =
            new SamplingMethod<>("PMMH", ParticleSampling.PMMH::new);

    public static final SamplingMethod<ParticleSampling.Gibbs, GibbsResult>
            PARTICLE_GIBBS = new SamplingMethod<>(
                    "PARTICLE_GIBBS", ParticleSampling.Gibbs::new);

    public static final SamplingMethod<ParticleSampling.CSMC, CSMCResult>
            CONDITIONAL_SMC = new SamplingMethod<>(
                    "CONDITIONAL_SMC", ParticleSampling.CSMC::new);

    public static final SamplingMethod<ParticleSampling.IBIS, IBISResult> IBIS =
            new SamplingMethod<>("IBIS", ParticleSampling.IBIS::new);

    public static final SamplingMethod<ParticleSampling.SMC2<?>, SMC2Result> SMC2 =
            new SamplingMethod<>("SMC2", ParticleSampling.SMC2::new);

    public static final SamplingMethod<ParticleSampling.Tempering, TemperingResult> TEMPERING =
            new SamplingMethod<>("TEMPERING", ParticleSampling.Tempering::new);

    public static final SamplingMethod<ParticleSampling.AdaptiveTempering, TemperingResult>
            ADAPTIVE_TEMPERING = new SamplingMethod<>(
                    "ADAPTIVE_TEMPERING", ParticleSampling.AdaptiveTempering::new);

    public static final SamplingMethod<ParticleSampling.Nested, NestedSamplingResult> NESTED =
                        new SamplingMethod<>("NESTED", ParticleSampling.Nested::new);

        public SamplingMethod {
                name = Objects.requireNonNull(name, "name");
                factory = Objects.requireNonNull(factory, "factory");
    }

        S create() {
                return factory.get();
        }

    @Override
    public String toString() {
        return name;
    }
}