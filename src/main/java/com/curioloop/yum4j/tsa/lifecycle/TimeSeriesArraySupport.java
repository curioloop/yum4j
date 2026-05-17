package com.curioloop.yum4j.tsa.lifecycle;

import java.util.Arrays;

public final class TimeSeriesArraySupport {

    private TimeSeriesArraySupport() {
    }

    public static double[] copyNonEmpty(double[] source, String name) {
        if (source == null || source.length == 0) {
            throw new IllegalArgumentException(name + " must not be empty");
        }
        return source.clone();
    }

    public static double[] concat(double[] left, double[] right) {
        double[] combined = Arrays.copyOf(left, left.length + right.length);
        System.arraycopy(right, 0, combined, left.length, right.length);
        return combined;
    }

    public static double[][] copyRows(double[][] source, String name) {
        if (source == null || source.length == 0) {
            throw new IllegalArgumentException(name + " must not be empty");
        }
        int width = -1;
        double[][] copy = new double[source.length][];
        for (int row = 0; row < source.length; row++) {
            if (source[row] == null) {
                throw new IllegalArgumentException(name + " rows must not be null");
            }
            if (width < 0) {
                width = source[row].length;
                if (width == 0) {
                    throw new IllegalArgumentException(name + " rows must not be empty");
                }
            } else if (source[row].length != width) {
                throw new IllegalArgumentException(name + " rows must all have the same length");
            }
            copy[row] = source[row].clone();
        }
        return copy;
    }

    public static double[][] concatRows(double[][] left, double[][] right) {
        double[][] combined = Arrays.copyOf(left, left.length + right.length);
        for (int row = 0; row < right.length; row++) {
            combined[left.length + row] = right[row].clone();
        }
        return combined;
    }

    public static double[][] sliceRows(double[][] source, int start, int length) {
        if (source == null) {
            return null;
        }
        double[][] slice = new double[length][];
        for (int row = 0; row < length; row++) {
            slice[row] = source[start + row].clone();
        }
        return slice;
    }

    public static double[][] missingRows(int rows, int width) {
        double[][] values = new double[rows][width];
        for (double[] row : values) {
            Arrays.fill(row, Double.NaN);
        }
        return values;
    }

    public static double[] finiteValues(double[] values) {
        int count = 0;
        for (double value : values) {
            if (Double.isFinite(value)) {
                count++;
            }
        }
        double[] clean = new double[count];
        int position = 0;
        for (double value : values) {
            if (Double.isFinite(value)) {
                clean[position++] = value;
            }
        }
        return clean;
    }

    public static double[][] finiteColumns(double[][] values, int columns) {
        int[] counts = new int[columns];
        for (double[] row : values) {
            for (int col = 0; col < columns; col++) {
                if (Double.isFinite(row[col])) {
                    counts[col]++;
                }
            }
        }
        double[][] out = new double[columns][];
        for (int col = 0; col < columns; col++) {
            out[col] = new double[counts[col]];
        }
        int[] positions = new int[columns];
        for (double[] row : values) {
            for (int col = 0; col < columns; col++) {
                if (Double.isFinite(row[col])) {
                    out[col][positions[col]++] = row[col];
                }
            }
        }
        return out;
    }
}