package com.curioloop.yum4j.ssm.particle.engine;

import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;

/**
 * Mutable context passed to {@code FeynmanKac.init} and {@code FeynmanKac.advance}.
 *
 * <p>Previously an immutable {@code record}; converted to a mutable {@code final class}
 * so the engine can reuse a single instance per slab across every step (R12a).
 * User kernels treat the received {@code StepContext} as read-only — mutation
 * of any field by a user kernel is undefined behaviour.
 *
 * <p>Fields are package-private; the engine (in the same package) writes them
 * in place before each dispatch to {@code fk.init} / {@code fk.advance}.
 * Public accessors return the current field values.
 *
 * <p>The public all-args constructor is retained for external callers that
 * still build a {@code StepContext} by hand (e.g. smoothers, CSMC, samplers).
 * Engine-internal paths use the package-private no-arg constructor and mutate
 * the fields in place via {@code Engine.makeCtx}.
 *
 * @param <Y> observation type
 */
public final class StepContext<Y> {

    // Package-private fields. Written by Engine (and other code in this package)
    // in place before dispatching to a FeynmanKac kernel. User kernels must
    // treat these as read-only via the public accessors below.
    int t;
    Y observation;
    double[] Xprev;
    int xpOff;
    double[] X;
    int xOff;
    double[] logW;
    int lwOff;
    int N;
    int dim;
    RandomBatch rng;
    double[] scratch;
    int scratchOff;

    /**
     * Package-private no-arg constructor for engine-side pooled reuse.
     * Fields are filled in place before dispatch.
     */
    StepContext() {}

    /**
     * Constructs a context with all fields set. Retained for call sites that
     * still build a {@code StepContext} by hand (smoothers, CSMC, samplers);
     * the engine-internal fast path uses the package-private no-arg
     * constructor and mutates in place.
     *
     * @param t           current time index
     * @param observation the observation at time {@code t}
     * @param Xprev       previous-step particle buffer (read-only for advance)
     * @param xpOff       offset into {@code Xprev} for this slab
     * @param X           current-step particle buffer (written by init/advance)
     * @param xOff        offset into {@code X} for this slab
     * @param logW        log-weight buffer (written by init/advance)
     * @param lwOff       offset into {@code logW} for this slab
     * @param N           number of particles in this slab
     * @param dim         state dimension per particle
     * @param rng         batched random number generator for this slab
     * @param scratch     scratch workspace buffer
     * @param scratchOff  offset into {@code scratch} for this slab
     */
    public StepContext(int t, Y observation,
                       double[] Xprev, int xpOff,
                       double[] X, int xOff,
                       double[] logW, int lwOff,
                       int N, int dim,
                       RandomBatch rng,
                       double[] scratch, int scratchOff) {
        this.t = t;
        this.observation = observation;
        this.Xprev = Xprev;
        this.xpOff = xpOff;
        this.X = X;
        this.xOff = xOff;
        this.logW = logW;
        this.lwOff = lwOff;
        this.N = N;
        this.dim = dim;
        this.rng = rng;
        this.scratch = scratch;
        this.scratchOff = scratchOff;
    }

    /** @return current time index */
    public int t() { return t; }

    /** @return observation at time {@code t} */
    public Y observation() { return observation; }

    /** @return previous-step particle buffer */
    public double[] Xprev() { return Xprev; }

    /** @return offset into {@link #Xprev()} for this slab */
    public int xpOff() { return xpOff; }

    /** @return current-step particle buffer */
    public double[] X() { return X; }

    /** @return offset into {@link #X()} for this slab */
    public int xOff() { return xOff; }

    /** @return log-weight buffer */
    public double[] logW() { return logW; }

    /** @return offset into {@link #logW()} for this slab */
    public int lwOff() { return lwOff; }

    /** @return number of particles in this slab */
    public int N() { return N; }

    /** @return state dimension per particle */
    public int dim() { return dim; }

    /** @return batched random number generator for this slab */
    public RandomBatch rng() { return rng; }

    /** @return scratch workspace buffer */
    public double[] scratch() { return scratch; }

    /** @return offset into {@link #scratch()} for this slab */
    public int scratchOff() { return scratchOff; }
}
