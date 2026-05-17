package com.curioloop.yum4j.stats.tool;

/**
 * Information criteria for model selection.
 *
 * <p>Any fitted model that provides a log-likelihood, observation count, and
 * effective parameter count can implement this interface to expose AIC, AICc,
 * BIC, and HQIC.</p>
 *
 * <p>Formulas:</p>
 * <pre>{@code
 * AIC  = -2·logLikelihood + 2·k
 * AICc = -2·logLikelihood + 2·k·n / (n - k - 1)
 * BIC  = -2·logLikelihood + log(n)·k
 * HQIC = -2·logLikelihood + 2·log(log(n))·k
 * }</pre>
 * where {@code k} is the number of estimated parameters and {@code n} is the
 * number of observations.
 */
public interface InformationCriterion {

    /** Log-likelihood of the fitted model. */
    double logLikelihood();

    /** Number of observations used in the fit. */
    int observationCount();

    /** Number of estimated parameters (degrees of freedom consumed by the model). */
    int parameterCount();

    /** Akaike Information Criterion: {@code -2·logLikelihood + 2·k}. */
    default double aic() {
        return -2.0 * logLikelihood() + 2.0 * parameterCount();
    }

    /**
     * Corrected Akaike Information Criterion (AICc) for small samples.
     *
     * <pre>{@code AICc = -2·logLikelihood + 2·k·n / (n - k - 1)}</pre>
     *
     * <p>Returns {@link Double#POSITIVE_INFINITY} when {@code n - k - 1 <= 0}.</p>
     */
    default double aicc() {
        int n = observationCount(), k = parameterCount();
        double dof = n - k - 1.0;
        if (dof <= 0.0) return Double.POSITIVE_INFINITY;
        return -2.0 * logLikelihood() + 2.0 * k * n / dof;
    }

    /** Bayesian Information Criterion: {@code -2·logLikelihood + log(n)·k}. */
    default double bic() {
        int n = observationCount();
        if (n <= 0) return Double.POSITIVE_INFINITY;
        return -2.0 * logLikelihood() + Math.log(n) * parameterCount();
    }

    /**
     * Hannan-Quinn Information Criterion: {@code -2·logLikelihood + 2·log(log(n))·k}.
     *
     * <p>Returns {@link Double#POSITIVE_INFINITY} when {@code n <= 1}.</p>
     */
    default double hqic() {
        int n = observationCount();
        if (n <= 1) return Double.POSITIVE_INFINITY;
        return -2.0 * logLikelihood() + 2.0 * Math.log(Math.log(n)) * parameterCount();
    }
}
