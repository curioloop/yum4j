package com.curioloop.yum4j.ssm.kalman.mle;

import com.curioloop.yum4j.optim.Minimizer;
import com.curioloop.yum4j.optim.NumericalGradient;
import com.curioloop.yum4j.optim.Optimization;
import com.curioloop.yum4j.optim.Univariate;
import com.curioloop.yum4j.optim.cmaes.CMAESProblem;
import com.curioloop.yum4j.optim.lbfgsb.LBFGSBProblem;
import com.curioloop.yum4j.optim.slsqp.SLSQPProblem;
import com.curioloop.yum4j.optim.subplex.SubplexProblem;

import java.util.Objects;

public final class FixedParameterMLE {

    private FixedParameterMLE() {
    }

    public interface Model {
        double[] transformParams(double[] unconstrainedParams);
        double[] untransformParams(double[] params);
        double logLikelihood(double[] params);
    }

    public record Result(Optimization optimization, double[] params, double[] unconstrainedParams) {}

    public static Result optimize(Model model,
                                  double[] startParams,
                                  FixedParameters fixedParameters,
                                  Minimizer<?, ?, ?> optimizer) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(startParams, "startParams");
        FixedParameters fixed = fixedParameters == null ? FixedParameters.none() : fixedParameters;
        fixed.validate(startParams.length);
        double[] constrainedStart = fixed.apply(startParams);
        if (fixed.isEmpty()) {
            double[] unconstrainedStart = model.untransformParams(constrainedStart);
            Optimization optimization = solve(transformedObjective(model), unconstrainedStart, optimizer);
            double[] unconstrained = optimization.solution().clone();
            double[] params = model.transformParams(unconstrained);
            return new Result(optimization, params, unconstrained);
        }

        double[] fullUnconstrainedStart = model.untransformParams(constrainedStart);
        int[] freeIndices = fixed.freeIndices(startParams.length);
        if (freeIndices.length == 0) {
            double cost = -model.logLikelihood(constrainedStart);
            Optimization optimization = new Optimization(Double.NaN,
                fullUnconstrainedStart.clone(),
                cost,
                Optimization.Status.COEFFICIENT_TOLERANCE_REACHED,
                0,
                1);
            return new Result(optimization, constrainedStart, fullUnconstrainedStart);
        }

        double[] freeStart = extract(fullUnconstrainedStart, freeIndices);
        Univariate.Objective objective = (free, n) -> {
            double[] fullUnconstrained = fullUnconstrainedStart.clone();
            scatter(free, n, freeIndices, fullUnconstrained);
            double[] params = model.transformParams(fullUnconstrained);
            fixed.applyInPlace(params);
            return -model.logLikelihood(params);
        };
        Optimization freeOptimization = solve(objective, freeStart, optimizer);
        double[] fullUnconstrained = fullUnconstrainedStart.clone();
        scatter(freeOptimization.solution(), freeOptimization.solution().length, freeIndices, fullUnconstrained);
        double[] params = model.transformParams(fullUnconstrained);
        fixed.applyInPlace(params);
        double[] unconstrained = model.untransformParams(params);
        Optimization optimization = new Optimization(freeOptimization.root(),
            unconstrained,
            freeOptimization.cost(),
            freeOptimization.status(),
            freeOptimization.iterations(),
            freeOptimization.evaluations());
        return new Result(optimization, params, unconstrained);
    }

    private static Univariate.Objective transformedObjective(Model model) {
        return (x, n) -> -model.logLikelihood(model.transformParams(copy(x, n)));
    }

    private static Optimization solve(Univariate.Objective objective,
                                      double[] initialPoint,
                                      Minimizer<?, ?, ?> optimizer) {
        Minimizer<?, ?, ?> resolved = optimizer == null
            ? Minimizer.lbfgsb().gradientTolerance(1e-8).functionTolerance(1e-12)
            : optimizer;
        double[] start = initialPoint.clone();
        if (resolved instanceof LBFGSBProblem problem) {
            return problem.objective(NumericalGradient.CENTRAL.wrap(objective))
                .initialPoint(start)
                .solve();
        }
        if (resolved instanceof SLSQPProblem problem) {
            return problem.objective(NumericalGradient.CENTRAL.wrap(objective))
                .initialPoint(start)
                .solve();
        }
        if (resolved instanceof SubplexProblem problem) {
            return problem.objective(objective)
                .initialPoint(start)
                .solve();
        }
        if (resolved instanceof CMAESProblem problem) {
            return problem.objective(objective)
                .initialPoint(start)
                .solve();
        }
        throw new IllegalArgumentException(
            "Unsupported MLE optimizer type: " + resolved.getClass().getName()
                + ". Expected LBFGSBProblem, SLSQPProblem, SubplexProblem, or CMAESProblem.");
    }

    private static double[] extract(double[] values, int[] indices) {
        double[] out = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            out[i] = values[indices[i]];
        }
        return out;
    }

    private static void scatter(double[] source, int n, int[] indices, double[] target) {
        if (n != indices.length) {
            throw new IllegalArgumentException("free parameter vector has unexpected length");
        }
        for (int i = 0; i < indices.length; i++) {
            target[indices[i]] = source[i];
        }
    }

    private static double[] copy(double[] source, int n) {
        double[] copy = new double[n];
        System.arraycopy(source, 0, copy, 0, n);
        return copy;
    }
}