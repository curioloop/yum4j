/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.de;

import com.curioloop.yum4j.optim.Bound;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;
import com.curioloop.yum4j.optim.lbfgsb.LBFGSBProblem;
import com.curioloop.yum4j.optim.slsqp.SLSQPProblem;

/**
 * Local minimizer used by differential evolution during final polishing.
 *
 * <p>The global DE phase supplies the best population member and the remaining evaluation
 * budget. A polisher may run any local optimizer, but the DE core accepts the result only
 * if it is finite, inside the supplied bounds, strictly improves the objective, and satisfies
 * configured constraints. Integral coordinates are fixed through exact bounds before mixed
 * integer polishing.</p>
 */
@FunctionalInterface
public interface FinalPolisher {

    /**
     * Runs a local minimization from {@code start}.
     *
    * <p>The differential evolution core validates finiteness, bounds, improvement,
    * hard-error status, and constraint feasibility before accepting the returned
    * point. Feasible limit-status improvements may be accepted; hard-error results
    * are rejected.</p>
     */
    Optimization polish(Univariate.Objective objective,
                        Bound[] bounds,
                        double[] start,
                        int maxEvaluations,
                        Context context);

    /** Constraint context available to custom differential-evolution polishers. */
    record Context(Univariate.Objective[] equalityConstraints,
                   Univariate.Objective[] inequalityConstraints,
                   double equalityTolerance) {

        private static final Context EMPTY = new Context(null, null, 0.0);

        public Context {
            if (equalityTolerance < 0.0 || Double.isNaN(equalityTolerance)) {
                throw new IllegalArgumentException("equalityTolerance must be non-negative, got " + equalityTolerance);
            }
        }

        public static Context empty() {
            return EMPTY;
        }

        public boolean hasConstraints() {
            return (equalityConstraints != null && equalityConstraints.length > 0)
                    || (inequalityConstraints != null && inequalityConstraints.length > 0);
        }
    }

    /** No-op polisher used when final polishing is disabled. */
    static FinalPolisher none() {
        return NoOpHolder.NONE;
    }

    /** Default selector that uses SLSQP for constrained contexts and L-BFGS-B otherwise. */
    static FinalPolisher auto() {
        return new AutoPolisher();
    }

    /** Default L-BFGS-B polisher for bound-constrained problems. */
    static FinalPolisher lbfgsb() {
        LBFGSBProblem problem = new LBFGSBProblem();
        var workspace = LBFGSBProblem.workspace();
        return (objective, bounds, start, maxEvaluations, context) -> problem
                .objective(objective)
                .bounds(bounds)
                .initialPoint(start)
                .maxIterations(Math.max(1, maxEvaluations))
                .maxEvaluations(maxEvaluations)
                .solve(workspace);
    }

    /** Default SLSQP polisher for constrained differential-evolution problems. */
    static FinalPolisher slsqp() {
        SLSQPProblem problem = new SLSQPProblem();
        var workspace = SLSQPProblem.workspace();
        return (objective, bounds, start, maxEvaluations, context) -> {
            Context current = context != null ? context : Context.empty();
            problem.objective(objective)
                    .bounds(bounds)
                    .initialPoint(start)
                    .maxIterations(Math.max(1, maxEvaluations))
                    .maxEvaluations(maxEvaluations);
            problem.equalityConstraints(current.equalityConstraints());
            problem.inequalityConstraints(current.inequalityConstraints());
            return problem.solve(workspace);
        };
    }

    final class NoOpHolder {
        private static final FinalPolisher NONE = (objective, bounds, start, maxEvaluations, context) ->
                new Optimization(Double.NaN, null, Double.POSITIVE_INFINITY,
                        Optimization.Status.MAX_ITERATIONS_REACHED, 0, 0);

        private NoOpHolder() {}
    }

    final class AutoPolisher implements FinalPolisher {
        private FinalPolisher lbfgsb;
        private FinalPolisher slsqp;

        @Override
        public Optimization polish(Univariate.Objective objective,
                                   Bound[] bounds,
                                   double[] start,
                                   int maxEvaluations,
                                   Context context) {
            Context current = context != null ? context : Context.empty();
            if (current.hasConstraints()) {
                if (slsqp == null) slsqp = FinalPolisher.slsqp();
                return slsqp.polish(objective, bounds, start, maxEvaluations, current);
            }
            if (lbfgsb == null) lbfgsb = FinalPolisher.lbfgsb();
            return lbfgsb.polish(objective, bounds, start, maxEvaluations, current);
        }
    }
}