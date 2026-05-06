/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.reg;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.stats.StudentsTDistribution;

import static java.lang.Math.sqrt;

/**
 * Prediction result with lazy-computed variance components.
 *
 * <p>{@link #mean()} is available immediately after construction.
 * {@link #paramVar()} and {@link #residualVar()} are computed on first access and cached.
 *
 * <p>Obtain via {@link com.curioloop.yum4j.linalg.Regression#predict(double[], int, double[])}.
 */
public final class Prediction {

    // ---- inputs held for lazy computation ----
    private final double[] newX;        // m×k row-major
    private final double[] unscaledCov; // k×k (XᵀX)⁻¹, upper triangle
    private final double[] weights;     // observation weights (null = uniform)
    private final int      m;
    private final int      k;
    private final double   scale;       // σ̂²
    private final int dfResidual;       // degrees of freedom for t-quantile

    private final double[] mean;  // ŷ = newX·β̂ (length m)
    private double[] paramVar;    // x·Cov(β̂)·xᵀ
    private double[] residualVar; // σ̂²/wᵢ


    public Prediction(double[] newX, double[] unscaledCov, double[] beta,
                      double scale, double[] weights, int m, int k, int dfResidual) {
        this.newX        = newX;
        this.unscaledCov = unscaledCov;
        this.weights     = weights;
        this.m           = m;
        this.k           = k;
        this.scale       = scale;
        this.dfResidual  = dfResidual;

        // mean is cheap (single dgemv) — compute eagerly
        this.mean = new double[m];
        BLAS.dgemv(BLAS.Trans.NoTrans, m, k, 1.0, newX, 0, k, beta, 0, 1, 0.0, mean, 0, 1);
    }

    /** Predicted mean ŷ = newX·β̂ (length m). */
    public double[] mean() { return mean; }

    /**
     * Parameter variance xᵢᵀ·Cov(β̂)·xᵢ per observation (length m).
     * Computed lazily on first call via dsymv + ddot loop.
     */
    public double[] paramVar() {
        if (paramVar == null) {
            paramVar = new double[m];
            double[] tmp = new double[k];
            for (int i = 0; i < m; i++) {
                int off = i * k;
                BLAS.dsymv(BLAS.Uplo.Upper, k, 1.0, unscaledCov, 0, k, newX, off, 1, 0.0, tmp, 0, 1);
                paramVar[i] = scale * BLAS.ddot(k, tmp, 0, 1, newX, off, 1);
            }
        }
        return paramVar;
    }

    /**
     * Residual variance σ̂²/wᵢ per observation (length m).
     * Computed lazily on first call.
     */
    public double[] residualVar() {
        if (residualVar == null) {
            residualVar = new double[m];
            if (weights != null) {
                for (int i = 0; i < m; i++) residualVar[i] = scale / weights[i];
            } else {
                java.util.Arrays.fill(residualVar, scale);
            }
        }
        return residualVar;
    }

    /**
     * Computes prediction intervals.
     *
     * <p>std[i] = sqrt(residualVar[i] + paramVar[i])
     * <p>lb[i]  = mean[i] - Q·std[i],  ub[i] = mean[i] + Q·std[i]
     * <p>where Q = t_{dfResidual, 1-alpha/2}
     *
     * @param alpha significance level (e.g. 0.05 for 95%)
     * @return double[3][] — [0]=lower bounds, [1]=upper bounds, [2]=std errors
     */
    public double[][] confInt(double alpha) {
        double[] pv  = paramVar();
        double[] rv  = residualVar();
        double   q   = new StudentsTDistribution(dfResidual).quantile(1.0 - alpha / 2.0);
        double[] lb  = new double[m];
        double[] ub  = new double[m];
        double[] std = new double[m];
        for (int i = 0; i < m; i++) {
            std[i] = sqrt(rv[i] + pv[i]);
            lb[i]  = mean[i] - q * std[i];
            ub[i]  = mean[i] + q * std[i];
        }
        return new double[][]{lb, ub, std};
    }
}
