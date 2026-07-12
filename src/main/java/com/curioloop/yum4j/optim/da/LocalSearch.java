/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.da;

import com.curioloop.yum4j.optim.Bound;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;
import com.curioloop.yum4j.optim.lbfgsb.LBFGSBProblem;

/**
 * Local minimizer used by dual annealing after global search steps.
 *
 * <p>The global annealing loop supplies a promising starting point and the remaining budget.
 * A local search implementation may use gradients, finite differences, or any derivative-free
 * method. Matching SciPy's dual-annealing wrapper, the core accepts only finite, in-bounds,
 * strict objective improvements; the returned optimizer status is diagnostic.</p>
 */
@FunctionalInterface
public interface LocalSearch {

    /**
     * Runs a local minimization from {@code start}.
     *
     * <p>The implementation should return an {@link Optimization} whose solution is inside
     * {@code bounds}. The dual annealing core validates finiteness, bounds, and improvement
     * before accepting the returned point.</p>
     */
    Optimization search(Univariate.Objective objective,
                        Bound[] bounds,
                        double[] start,
                        int maxEvaluations);

    /** Default L-BFGS-B local minimizer using a reusable internal workspace. */
    static LocalSearch lbfgsb() {
        LBFGSBProblem problem = new LBFGSBProblem();
        var workspace = LBFGSBProblem.workspace();
        return (objective, bounds, start, maxEvaluations) -> problem
                .objective(objective)
                .bounds(bounds)
                .initialPoint(start)
                .maxIterations(maxEvaluations)
                .maxEvaluations(maxEvaluations)
                .solve(workspace);
    }
}