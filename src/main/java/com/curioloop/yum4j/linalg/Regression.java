/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg;

import static java.lang.Math.*;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.reg.OLS;
import com.curioloop.yum4j.linalg.reg.Prediction;
import com.curioloop.yum4j.math.VectorOps;

/**
 * Statistical result of an OLS, WLS, or GLS fit.
 *
 * <p>All quantities are computed lazily and cached on first access.
 * Subclasses extend this class
 * and populate the solver outputs ({@code beta}, {@code unscaledCov}, {@code rank}, {@code cond})
 * during {@code fit()}.
 *
 * <h2>Key statistics</h2>
 * <ul>
 *   <li>{@link #params()} — β̂</li>
 *   <li>{@link #paramCov()}   — Cov(β̂) = σ̂²·(XᵀX)⁻¹</li>
 *   <li>{@link #bse()}        — standard errors √diag(Cov(β̂))</li>
 *   <li>{@link #ssr()}        — sum of squared (whitened) residuals ‖ỹ - X̃β̂‖²</li>
 *   <li>{@link #scale()}      — σ̂² = SSR / (n - rank)</li>
 *   <li>{@link #r2(boolean)}  — R² (optionally adjusted)</li>
 *   <li>{@link #mse()}        — model / residual / total MSE</li>
 *   <li>{@link #logLike()}    — Gaussian log-likelihood</li>
 *   <li>{@link #condNum()}    — condition number √(σ_max²/σ_min²)</li>
 *   <li>{@link #predict(double[], int, double[])} — prediction with confidence intervals</li>
 * </ul>
 */
public abstract class Regression implements com.curioloop.yum4j.stats.tool.InformationCriterion {

    // ---- solver outputs (populated by fit) ----
    protected double[] beta;        // parameter estimates β̂, length k
    protected double[] unscaledCov; // (XᵀX)⁻¹ or (RᵀR)⁻¹, k×k row-major
    protected int      rank;        // numerical rank of X
    protected double   cond;        // condition number √(σ_max²/σ_min²)

    // ---- fitted/residual: set by solver via setFittedResidualCache ----
    private double[] fitted;
    private double[] residual;

    // ---- whitened variants (lazy) ----
    private double[] whitenFitted;
    private double[] whitenResidual;

    // ---- lazy scalar cache ----
    private double ssr    = Double.NaN;
    private double tss    = Double.NaN;
    private double tssCen = Double.NaN;
    private double llf    = Double.NaN;

    // ---- abstract accessors ----

    public abstract int      nObs();
    public abstract int      nParams();
    public abstract int      kConst();
    protected abstract int   dfModel();
    protected abstract int   dfResidual();
    public abstract double[] endog();
    public abstract double[] weights();

    // ---- called by solver after computing fitted/residual into Pool ----

    /**
     * Points the lazy-cache fields at the Pool's fitted/residual buffers.
     * Called once per fit, inside the solver, after those buffers are written.
     */
    protected void setFittedResidualCache(OLS.Pool ws) {
        this.fitted   = ws.fitted;
        this.residual = ws.residual;
        // invalidate whitened and scalar caches
        this.whitenFitted   = null;
        this.whitenResidual = null;
        this.ssr    = Double.NaN;
        this.tss    = Double.NaN;
        this.tssCen = Double.NaN;
        this.llf    = Double.NaN;
    }

    // ---- parameters ----

    /** Parameter estimates β̂ (length k). */
    public double[] params() {
        if (beta == null) throw unavailable("parameter estimates");
        return beta;
    }

    /** Numerical rank of X. */
    public int rank() { return rank; }

    /** Condition number √(σ_max²/σ_min²). */
    public double condNum() { return cond; }

    // ---- fitted values and residuals ----

    /**
     * Fitted values ŷ = Xβ̂.
     *
     * @param whiten if true and weights are present, returns √W·ŷ
     */
    public double[] fitted(boolean whiten) {
        double[] w = weights();
        if (!whiten || w == null) {
            if (fitted == null) throw unavailable("fitted values");
            return fitted;
        } else {
            if (fitted == null) throw unavailable("fitted values");
            if (whitenFitted == null) {
                int n = nObs();
                whitenFitted = new double[n];
                for (int i = 0; i < n; i++) whitenFitted[i] = fitted[i] * sqrt(w[i]);
            }
            return whitenFitted;
        }
    }

    /**
     * Residuals e = y - ŷ.
     *
     * @param whiten if true and weights are present, returns √W·e
     */
    public double[] residual(boolean whiten) {
        double[] w = weights();
        if (!whiten || w == null) {
            if (residual == null) throw unavailable("residuals");
            return residual;
        } else {
            if (residual == null) throw unavailable("residuals");
            if (whitenResidual == null) {
                int n = nObs();
                whitenResidual = new double[n];
                for (int i = 0; i < n; i++) whitenResidual[i] = residual[i] * sqrt(w[i]);
            }
            return whitenResidual;
        }
    }

    // ---- sum of squares ----

    /** Sum of squared (whitened) residuals: SSR = ‖ỹ - X̃β̂‖² = ẽᵀẽ. */
    public double ssr() {
        if (Double.isNaN(ssr)) {
            double[] e = residual(true);
            ssr = BLAS.ddot(e.length, e, 0, 1, e, 0, 1);
        }
        return ssr;
    }

    /**
     * Total sum of squares.
     *
     * @param centered if true, uses centered TSS (subtract mean); otherwise uncentered
     */
    public double tss(boolean centered) {
        if (centered  && !Double.isNaN(tssCen)) return tssCen;
        if (!centered && !Double.isNaN(tss))    return tss;

        double[] y = endog();
        double[] w = weights();
        int n = nObs();

        double mean = 0;
        if (centered) {
            if (w != null) {
                mean = BLAS.ddot(n, w, 0, 1, y, 0, 1) / BLAS.dasum(n, w, 0, 1);
            } else {
                mean = VectorOps.mean(y, 0, n);
            }
        }

        double s = 0;
        if (w != null) {
            for (int i = 0; i < n; i++) { double d = y[i] - mean; s += w[i] * d * d; }
        } else if (centered) {
            s = VectorOps.sumSq(y, 0, n, mean);
        } else {
            s = BLAS.ddot(n, y, 0, 1, y, 0, 1);
        }

        if (centered) tssCen = s; else tss = s;
        return s;
    }

    /** Explained sum of squares (ESS = TSS - SSR). */
    public double ess() {
        return tss(kConst() > 0) - ssr();
    }

    // ---- scale and covariance ----

    /** Scale factor σ̂² = SSR / (n - rank). */
    public double scale() {
        return ssr() / dfResidual();
    }

    /**
     * Parameter covariance matrix Cov(β̂) = σ̂²·(XᵀX)⁻¹ (k×k, row-major).
     * Returns a new array each call.
     */
    public double[] paramCov() {
        if (unscaledCov == null) throw unavailable("parameter covariance");
        double[] cov = unscaledCov.clone();
        BLAS.dscal(cov.length, scale(), cov, 0, 1);
        return cov;
    }

    /** Standard errors of parameter estimates: bse = √diag(Cov(β̂)). */
    public double[] bse() {
        if (unscaledCov == null) throw unavailable("standard errors");
        double s = scale();
        int k = nParams();
        double[] se = new double[k];
        for (int i = 0; i < k; i++) se[i] = sqrt(s * unscaledCov[i * k + i]);
        return se;
    }

    // ---- goodness of fit ----

    /**
     * R² coefficient of determination.
     *
     * @param adjusted if true, returns adjusted R²
     */
    public double r2(boolean adjusted) {
        double r2 = 1.0 - ssr() / tss(kConst() > 0);
        if (adjusted) {
            int n = nObs(), kc = kConst();
            return 1.0 - ((double)(n - kc) / dfResidual()) * (1.0 - r2);
        }
        return r2;
    }

    /**
     * Mean squared errors.
     *
     * @return double[3] = {MSE_model, MSE_residual, MSE_total}
     */
    public double[] mse() {
        double mseModel    = ess() / dfModel();
        double mseResidual = ssr() / dfResidual();
        double mseTotal    = tss(kConst() > 0) / (dfModel() + dfResidual());
        return new double[]{mseModel, mseResidual, mseTotal};
    }

    // ---- information criteria ----

    /**
     * Gaussian log-likelihood:
     * <pre>  llf = -n/2·log(SSR) - n/2·(1 + log(π/n)) + ½·∑log(wᵢ)  (WLS term)</pre>
     */
    public double logLike() {
        if (Double.isNaN(llf)) {
            double n2 = nObs() / 2.0;
            double l  = -n2 * log(ssr()) - n2 * (1.0 + log(PI / n2));
            double[] w = weights();
            if (w != null) {
                double sumLogW = 0;
                for (double wi : w) sumLogW += log(wi);
                l += 0.5 * sumLogW;
            }
            llf = l;
        }
        return llf;
    }

    // ---- InformationCriterion interface ----

    @Override public double logLikelihood() { return logLike(); }
    @Override public int observationCount() { return nObs(); }
    @Override public int parameterCount()   { return dfModel() + kConst(); }

    // ---- prediction ----

    /**
     * Predicts for new observations.
     *
     * @param newX    new exogenous matrix, row-major m×k (not modified)
     * @param m       number of new observations
     * @param weights observation weights for residual variance scaling (null = uniform)
     * @return {@link Prediction} containing mean, paramVar, residualVar
     */
    public Prediction predict(double[] newX, int m, double[] weights) {
        if (unscaledCov == null) throw unavailable("prediction covariance");
        return new Prediction(newX, unscaledCov, beta, scale(), weights, m, nParams(), dfResidual());
    }

    private static IllegalStateException unavailable(String quantity) {
        return new IllegalStateException(quantity + " are not available for this regression output selection");
    }

    // =========================================================================
    // Diagnostics — t/F statistics, confidence intervals
    // =========================================================================

    private double[] tStats;
    private double[] tPVals;
    private double   fStat = Double.NaN;
    private double   fPVal = Double.NaN;
    private double   dw    = Double.NaN;

    /** t-statistics: tᵢ = β̂ᵢ / bseᵢ. */
    public double[] tStatistics() {
        if (tStats == null) {
            double[] b = params(), se = bse();
            tStats = new double[b.length];
            for (int i = 0; i < b.length; i++) tStats[i] = b[i] / se[i];
        }
        return tStats;
    }

    /** Two-tailed p-values from Student's t(df_residual). */
    public double[] tPValues() {
        if (tPVals == null) {
            double[] t = tStatistics();
            int df = dfResidual();
            var dist = new com.curioloop.yum4j.stats.StudentsTDistribution(df);
            tPVals = new double[t.length];
            for (int i = 0; i < t.length; i++) tPVals[i] = 2.0 * dist.ccdf(abs(t[i]));
        }
        return tPVals;
    }

    /** F-statistic for overall model significance: F = MSE_model / MSE_residual. */
    public double fStatistic() {
        if (Double.isNaN(fStat)) {
            if (dfModel() == 0) { fStat = Double.NaN; }
            else { double[] m = mse(); fStat = m[0] / m[1]; }
        }
        return fStat;
    }

    /** p-value of the F-statistic from F(df_model, df_residual). */
    public double fPValue() {
        if (Double.isNaN(fPVal)) {
            double f = fStatistic();
            if (Double.isNaN(f) || dfModel() == 0) { fPVal = Double.NaN; }
            else { fPVal = new com.curioloop.yum4j.stats.FisherFDistribution(dfModel(), dfResidual()).ccdf(f); }
        }
        return fPVal;
    }

    /**
     * Confidence intervals for each parameter.
     *
     * @param alpha significance level (e.g. 0.05 for 95% CI)
     * @return row-major k×2 array: [lower₀, upper₀, lower₁, upper₁, ...]
     */
    public double[] confInt(double alpha) {
        double[] b = params(), se = bse();
        int k = b.length;
        double tCrit = new com.curioloop.yum4j.stats.StudentsTDistribution(dfResidual())
                .quantileUpperTail(alpha / 2.0);
        double[] ci = new double[k * 2];
        for (int i = 0; i < k; i++) {
            double hw = tCrit * se[i];
            ci[i * 2]     = b[i] - hw;
            ci[i * 2 + 1] = b[i] + hw;
        }
        return ci;
    }

    /** 95% confidence intervals (α = 0.05). */
    public double[] confInt() { return confInt(0.05); }

    // =========================================================================
    // Diagnostics — Durbin-Watson, Jarque-Bera, Omnibus
    // =========================================================================


    /** Durbin-Watson statistic on whitened residuals. */
    public double durbinWatson() {
        if (Double.isNaN(dw)) {
            dw = com.curioloop.yum4j.stats.test.DurbinWatson.test(residual(true));
        }
        return dw;
    }

    /** Jarque-Bera normality test on whitened residuals. */
    public com.curioloop.yum4j.stats.test.JarqueBera jarqueBera() {
        return com.curioloop.yum4j.stats.test.JarqueBera.test(residual(true));
    }

    /** D'Agostino-Pearson omnibus normality test on whitened residuals. */
    public com.curioloop.yum4j.stats.test.Omnibus omnibus() {
        return com.curioloop.yum4j.stats.test.Omnibus.test(residual(true));
    }

}