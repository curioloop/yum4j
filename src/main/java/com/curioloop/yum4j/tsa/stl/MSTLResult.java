package com.curioloop.yum4j.tsa.stl;

import java.util.Arrays;

/**
 * Multi-seasonal STL decomposition result.
 *
 * <p>The reconstruction identity is
 * {@code observed[t] = trend[t] + residual[t] + sum_m seasonal[m][t]}. If MSTL was configured
 * with a finite Box-Cox lambda, {@code observed} is the transformed series.</p>
 */
public final class MSTLResult {

    private final double[] observed;
    private final double[][] seasonal;
    private final double[] trend;
    private final double[] residual;
    private final double[] weights;

    MSTLResult(double[] observed, double[][] seasonal, double[] trend, double[] residual, double[] weights) {
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

    public int seasonalCount() {
        return seasonal.length;
    }

    public double[][] seasonal() {
        double[][] copy = new double[seasonal.length][];
        for (int i = 0; i < seasonal.length; i++) {
            copy[i] = Arrays.copyOf(seasonal[i], seasonal[i].length);
        }
        return copy;
    }

    public double[] seasonal(int index) {
        return Arrays.copyOf(seasonal[index], seasonal[index].length);
    }

    public double seasonal(int component, int index) {
        return seasonal[component][index];
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
