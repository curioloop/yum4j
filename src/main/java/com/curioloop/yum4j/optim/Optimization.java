/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim;

/**
 * Immutable result of an optimization or root-finding run.
 *
 * <h2>Solution access</h2>
 * <ul>
 *   <li>When using {@code XxxOptimizer.optimize(x0)}, the solution is written back
 *       into {@code x0} in-place and {@link #solution()} returns {@code null}.</li>
 *   <li>When using {@code XxxProblem.solve()}, the solution is stored in
 *       {@link #solution()} and the original initial-point array is not modified.</li>
 *   <li>For 1-D root-finding ({@code BrentqProblem}), the scalar root is available
 *       via {@link #root()}; {@link #solution()} is {@code null}.</li>
 *   <li>For N-D root-finding ({@code HYBRProblem}, {@code BroydenProblem}),
 *       the solution vector is in {@link #solution()} and {@link #root()}
 *       returns {@code NaN}.</li>
 * </ul>
 *
 * <h2>Cost field semantics</h2>
 * <ul>
 *   <li>Minimizers (L-BFGS-B, SLSQP): value of the objective function F(x).</li>
 *   <li>TRF: residual sum of squares ‖f(x)‖² (or robust cost when a loss is set).</li>
 *   <li>HYBR / Broyden: residual norm ‖F(x)‖.</li>
 *   <li>Brentq: |f(x)| at the returned root.</li>
 * </ul>
 */
public value record Optimization(
    /* Scalar root value for 1-D root-finding results; NaN otherwise. */
    double root,
    /* Solution vector; non-null only when produced by {@code XxxProblem.solve()}. */
    double[] solution,
    /* Objective / residual-norm value at termination. */
    double cost,
    /* Termination status; defaults to {@link Status#ABNORMAL_TERMINATION} if null. */
    Status status,
    /* Number of iterations performed. */
    int iterations,
    /* Number of function evaluations performed. */
    int evaluations) {

    // -----------------------------------------------------------------------
    // Status enum (inner class)
    // -----------------------------------------------------------------------

    /**
     * Termination status of an optimization or root-finding run.
     *
     * <p>Status codes are organized into three categories:</p>
     * <ul>
     *   <li>Convergence (3, 4, 6, 7): satisfied a convergence criterion</li>
     *   <li>Limit reached (1, 2, 5): stopped due to a resource limit</li>
     *   <li>Error (negative): failed due to a numerical or input error</li>
     * </ul>
     */
    public enum Status {

        /** Maximum iterations reached */
        MAX_ITERATIONS_REACHED(1, false,
            "Maximum iterations reached without convergence",
            "Consider increasing maxIterations or relaxing tolerances"),

        /** Maximum function evaluations reached */
        MAX_EVALUATIONS_REACHED(2, false,
            "Maximum function evaluations reached",
            "Consider increasing maxEvaluations"),

        /** Gradient tolerance satisfied */
        GRADIENT_TOLERANCE_REACHED(3, true,
            "Converged: gradient norm below tolerance", null),

        /** Function tolerance satisfied */
        FUNCTION_TOLERANCE_REACHED(4, true,
            "Converged: function value change below tolerance", null),

        /** Maximum CPU time reached */
        MAX_COMPUTATIONS_REACHED(5, false,
            "Maximum computation time reached",
            "Consider increasing maxComputations or simplifying the objective"),

        /** Coefficient/variable tolerance satisfied */
        COEFFICIENT_TOLERANCE_REACHED(6, true,
            "Converged: variable change below tolerance", null),

        /** Chi-squared tolerance satisfied (LM algorithm) */
        CHI_SQUARED_TOLERANCE_REACHED(7, true,
            "Converged: chi-squared reduction below tolerance", null),

        /** Abnormal termination */
        ABNORMAL_TERMINATION(-1, false,
            "Abnormal termination due to internal error",
            "Check objective function for numerical issues (NaN, Infinity)"),

        /** Invalid argument provided */
        INVALID_ARGUMENT(-2, false,
            "Invalid argument provided to optimizer",
            "Verify all parameters satisfy documented constraints"),

        /** Constraints are incompatible */
        CONSTRAINT_INCOMPATIBLE(-3, false,
            "Constraints are incompatible or infeasible",
            "Check that constraints do not contradict each other"),

        /** Line search failed */
        LINE_SEARCH_FAILED(-4, false,
            "Line search failed to find acceptable step",
            "Check objective function continuity and gradient accuracy"),

        /** Callback function error */
        CALLBACK_ERROR(-5, false,
            "Error in callback function evaluation",
            "Check objective/constraint functions for exceptions"),

        /** Bracket condition f(a)*f(b) <= 0 not satisfied */
        INVALID_BRACKET(-6, false,
            "Bracket condition f(a)*f(b) <= 0 not satisfied",
            "Verify the bracket [a,b] contains a root"),

        /** Initial point or function output contains NaN or Infinity */
        INVALID_INPUT(-7, false,
            "Initial point or function output contains NaN or Infinity",
            "Provide finite initial values and check function for numerical issues");

        private final int code;
        private final boolean converged;
        private final String description;
        private final String suggestion;

        Status(int code, boolean converged, String description, String suggestion) {
            this.code = code;
            this.converged = converged;
            this.description = description;
            this.suggestion = suggestion;
        }

        public int code() { return code; }
        public String description() { return description; }
        public String suggestion() { return suggestion; }
        public boolean converged() { return converged; }

        /** Returns true if a resource limit was hit. */
        public boolean limitReached() { return code == 1 || code == 2 || code == 5; }

        /** Returns true if the status represents a hard error ({@code code < 0}). */
        public boolean error() { return code < 0; }

        /**
         * Returns the status whose {@link #code()} matches {@code code},
         * or {@link #ABNORMAL_TERMINATION} if no match is found.
         */
        public static Status fromCode(int code) {
            for (Status s : values()) {
                if (s.code == code) return s;
            }
            return ABNORMAL_TERMINATION;
        }

        @Override
        public String toString() { return name() + "(" + code + ")"; }
    }

    public Optimization {
        status = status != null ? status : Status.ABNORMAL_TERMINATION;
    }

    /**
     * Returns an error description when the status is an error state, otherwise {@code null}.
     */
    public String errorMessage() {
        if (status.error()) {
            return status.description() + ". " + status.suggestion();
        }
        return null;
    }

    /** Returns a formatted single-line summary of this result. */
    public String summary() {
        return String.format("Status: %s | Cost: %.6e | Iterations: %d | Evaluations: %d",
                status.description(), cost, iterations, evaluations);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Optimization {\n");
        sb.append("  status: ").append(status.description()).append('\n');
        sb.append("  cost: ").append(cost).append('\n');
        sb.append("  iterations: ").append(iterations).append('\n');
        sb.append("  evaluations: ").append(evaluations).append('\n');
        if (!status.converged() && status.suggestion() != null) {
            sb.append("  suggestion: ").append(status.suggestion()).append('\n');
        }
        sb.append('}');
        return sb.toString();
    }
}
