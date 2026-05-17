package com.curioloop.yum4j.tsa.prediction;

import com.curioloop.yum4j.math.Normal;

public final class MultivariatePredictionValues {

    private final int observationDimension;
    private final double[][] mean;
    private final double[][][] variance;

    public MultivariatePredictionValues(double[][] mean, double[][][] variance) {
        validate(mean, variance);
        this.observationDimension = mean.length == 0 ? 0 : mean[0].length;
        this.mean = copyMatrix(mean);
        this.variance = copyCube(variance);
    }

    public int length() {
        return mean.length;
    }

    public int observationDimension() {
        return observationDimension;
    }

    public double[][] mean() {
        return copyMatrix(mean);
    }

    public double[][][] variance() {
        return copyCube(variance);
    }

    public double[][] seMean() {
        double[][] se = new double[mean.length][observationDimension];
        for (int t = 0; t < mean.length; t++) {
            for (int i = 0; i < observationDimension; i++) {
                se[t][i] = UnivariatePredictionValues.standardError(variance[t][i][i]);
            }
        }
        return se;
    }

    public double[][][] confInt(double alpha) {
        UnivariatePredictionValues.validateAlpha(alpha);
        double criticalValue = Normal.inv(1.0 - alpha / 2.0);
        double[][] se = seMean();
        double[][][] intervals = new double[mean.length][observationDimension][2];
        for (int t = 0; t < mean.length; t++) {
            for (int i = 0; i < observationDimension; i++) {
                double margin = criticalValue * se[t][i];
                intervals[t][i][0] = mean[t][i] - margin;
                intervals[t][i][1] = mean[t][i] + margin;
            }
        }
        return intervals;
    }

    public static double[][] copyMatrix(double[][] source) {
        double[][] copy = new double[source.length][];
        for (int i = 0; i < source.length; i++) {
            copy[i] = source[i].clone();
        }
        return copy;
    }

    public static double[][][] copyCube(double[][][] source) {
        double[][][] copy = new double[source.length][][];
        for (int t = 0; t < source.length; t++) {
            copy[t] = copyMatrix(source[t]);
        }
        return copy;
    }

    private static void validate(double[][] mean, double[][][] variance) {
        if (mean == null || variance == null) {
            throw new IllegalArgumentException("mean and variance must not be null");
        }
        if (mean.length != variance.length) {
            throw new IllegalArgumentException("mean and variance lengths must match");
        }
        int dimension = mean.length == 0 ? 0 : validateMeanRow(mean[0], -1);
        for (int t = 0; t < mean.length; t++) {
            int rowDimension = validateMeanRow(mean[t], dimension);
            if (variance[t] == null || variance[t].length != rowDimension) {
                throw new IllegalArgumentException("variance matrices must match mean width");
            }
            for (int row = 0; row < rowDimension; row++) {
                if (variance[t][row] == null || variance[t][row].length != rowDimension) {
                    throw new IllegalArgumentException("variance matrices must be square");
                }
            }
        }
    }

    private static int validateMeanRow(double[] row, int expectedDimension) {
        if (row == null) {
            throw new IllegalArgumentException("mean rows must not be null");
        }
        if (expectedDimension >= 0 && row.length != expectedDimension) {
            throw new IllegalArgumentException("mean rows must all have the same length");
        }
        return row.length;
    }
}