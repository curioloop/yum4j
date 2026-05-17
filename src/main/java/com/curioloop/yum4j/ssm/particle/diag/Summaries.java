package com.curioloop.yum4j.ssm.particle.diag;

import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;

/**
 * No-op marker collector that documents the always-on summaries.
 *
 * <p>The engine records {@code ESS}, {@code rsFlag}, and
 * {@code logLt} unconditionally into typed series on {@link RunState}.
 * This collector has no hooks of its own; it exists only so that users
 * who prefer a uniform collector list can add it alongside
 * {@link Moments} or {@link com.curioloop.yum4j.ssm.particle.smooth.online.OnlineSmoothNaive}.
 *
 */
public final class Summaries implements Collector {
    // Intentionally empty; all default-method bodies apply.
}
