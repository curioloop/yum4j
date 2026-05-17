package com.curioloop.yum4j.tsa.lifecycle;

import java.util.Arrays;

public final class ExogenousDataSupport {

    private ExogenousDataSupport() {
    }

    public static double[][] validateRows(double[][] baseExog,
                                          int rows,
                                          double[][] exog,
                                          String rowCountMessage) {
        if (baseExog == null) {
            if (exog != null && exog.length > 0) {
                throw new IllegalArgumentException("exog provided for a model without exogenous regressors");
            }
            return null;
        }
        if (exog == null || exog.length != rows) {
            throw new IllegalArgumentException(rowCountMessage);
        }
        int width = baseExog[0].length;
        double[][] copy = new double[rows][];
        for (int row = 0; row < rows; row++) {
            if (exog[row] == null || exog[row].length != width) {
                throw new IllegalArgumentException("exog rows must match exogenous dimension");
            }
            copy[row] = exog[row].clone();
        }
        return copy;
    }

    public static double[][] appendRows(double[][] baseExog,
                                        int newRows,
                                        double[][] exog,
                                        String rowCountMessage) {
        double[][] newExog = validateRows(baseExog, newRows, exog, rowCountMessage);
        if (baseExog == null) {
            return null;
        }
        double[][] combined = Arrays.copyOf(baseExog, baseExog.length + newRows);
        for (int row = 0; row < newRows; row++) {
            combined[baseExog.length + row] = newExog[row].clone();
        }
        return combined;
    }

    public static double[][] extendFutureRows(double[][] baseExog,
                                              int outOfSample,
                                              double[][] futureExog) {
        if (baseExog == null) {
            if (futureExog != null && futureExog.length > 0) {
                throw new IllegalArgumentException("futureExog provided for a model without exogenous regressors");
            }
            return null;
        }
        if (outOfSample == 0) {
            return baseExog;
        }
        if (futureExog == null || futureExog.length != outOfSample) {
            throw new IllegalArgumentException("futureExog row count must match out-of-sample horizon");
        }
        int width = baseExog[0].length;
        double[][] extended = Arrays.copyOf(baseExog, baseExog.length + outOfSample);
        for (int row = 0; row < futureExog.length; row++) {
            if (futureExog[row] == null || futureExog[row].length != width) {
                throw new IllegalArgumentException("futureExog rows must match exogenous dimension");
            }
            extended[baseExog.length + row] = futureExog[row].clone();
        }
        return extended;
    }
}