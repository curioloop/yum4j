package com.curioloop.yum4j.ssm.particle.diag;

import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;

/**
 * Pluggable online summary collector attached to an SMC run.
 *
 * <p>The engine calls the lifecycle hooks in the following order
 * at every step:
 *
 * <ol>
 *   <li>{@link #afterReweight} — after the log-weight increment from
 *       {@code fk.logG} has been added to {@code ws.logW}. ESS / logLt
 *       summaries on {@code rs} are valid at this point.</li>
 *   <li>{@link #afterResample} — if resampling fired at this step; the
 *       engine evaluates the threshold <i>before</i> the next
 *       propagation, so most online collectors do not need this
 *       callback and may leave it as a no-op.</li>
 *   <li>{@link #afterMutation} — the terminal hook, called after
 *       propagation (and reweighting) have finished and
 *       {@code rs.stepCount} is about to be incremented. This is
 *       the typical hook for moment and online-smoothing collectors.</li>
 * </ol>
 *
 * <p>Implementations must store any per-step output internally; the
 * engine does not persist return values. {@link #attach(Workspace, RunState)}
 * is called once before the first step to let the collector size any
 * caller-owned buffers from workspace/run-state dimensions.
 *
 */
public interface Collector {

    /**
     * Attach to a workspace and run state. Called once per run, before
     * the engine's init step. Default implementation is a no-op.
     *
     * @param ws the workspace (provides N, dim)
     * @param rs the run state (provides T, scheme, etc.)
     */
    default void attach(Workspace ws, RunState rs) {}

    /**
     * Called after the reweight step (i.e. after {@code logW += logG}).
     * {@code rs.essSeries[t]} and {@code rs.logLtSeries[t]} are already
     * populated when this is invoked. Default implementation is a no-op.
     *
     * @param ws the workspace
     * @param rs the run state
     * @param t  current time step
     */
    default void afterReweight(Workspace ws, RunState rs, int t) {}

    /**
     * Called after resampling has fired at step {@code t}. Default
     * implementation is a no-op.
     *
     * @param ws the workspace
     * @param rs the run state
     * @param t  current time step
     */
    default void afterResample(Workspace ws, RunState rs, int t) {}

    /**
     * Called at the end of step {@code t}, after propagation and
     * reweighting. Default implementation is a no-op.
     *
     * @param ws the workspace
     * @param rs the run state
     * @param t  current time step
     */
    default void afterMutation(Workspace ws, RunState rs, int t) {}
}
