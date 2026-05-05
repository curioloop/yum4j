package com.curioloop.yum4j.tsa.stl;

import java.util.Arrays;

/**
 * Seasonal, trend, residual, and robust-weight components produced by {@link STL}.
 *
 * <p>The decomposition follows {@code observed[t] = seasonal[t] + trend[t] + residual[t]}.
 * Robust weights are all {@code 1} for non-robust STL; in robust mode, values near {@code 0}
 * mark observations treated as outliers in the final outer iteration.</p>
 */
public final class STLResult {

    private final double[] observed;
    private final double[] seasonal;
    private final double[] trend;
    private final double[] residual;
    private final double[] weights;

    STLResult(double[] observed, double[] seasonal, double[] trend, double[] residual, double[] weights) {
        this.observed = observed;
        this.seasonal = seasonal;
        this.trend = trend;
        this.residual = residual;
        this.weights = weights;
    }

    public int length() {
        return observed.length;
    }

    public double[] observed() {
        return Arrays.copyOf(observed, observed.length);
    }

    public double observed(int index) {
        return observed[index];
    }

    public double[] seasonal() {
        return Arrays.copyOf(seasonal, seasonal.length);
    }

    public double seasonal(int index) {
        return seasonal[index];
    }

    public double[] trend() {
        return Arrays.copyOf(trend, trend.length);
    }

    public double trend(int index) {
        return trend[index];
    }

    public double[] residual() {
        return Arrays.copyOf(residual, residual.length);
    }

    public double residual(int index) {
        return residual[index];
    }

    public double[] weights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public double weight(int index) {
        return weights[index];
    }
}
