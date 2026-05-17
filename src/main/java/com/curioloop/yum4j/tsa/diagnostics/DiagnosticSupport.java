package com.curioloop.yum4j.tsa.diagnostics;

import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.tsa.lifecycle.TimeSeriesArraySupport;

public final class DiagnosticSupport {

    private DiagnosticSupport() {
    }

    public static double[] finiteValues(double[] values) {
        return TimeSeriesArraySupport.finiteValues(values);
    }

    public static double[][] finiteColumns(double[][] values, int columns) {
        return TimeSeriesArraySupport.finiteColumns(values, columns);
    }

    public static InstantaneousCausality instantaneousCausality(double[][] residuals,
                                                                int caused,
                                                                int causing,
                                                                int dimension) {
        validateEndogIndex(caused, dimension);
        validateEndogIndex(causing, dimension);
        if (caused == causing) {
            throw new IllegalArgumentException("caused and causing indices must differ");
        }
        int count = 0;
        double sumX = 0.0;
        double sumY = 0.0;
        for (double[] row : residuals) {
            if (Double.isFinite(row[caused]) && Double.isFinite(row[causing])) {
                count++;
                sumX += row[caused];
                sumY += row[causing];
            }
        }
        if (count < 3) {
            throw new IllegalArgumentException("insufficient residual observations");
        }
        double meanX = sumX / count;
        double meanY = sumY / count;
        double sxx = 0.0;
        double syy = 0.0;
        double sxy = 0.0;
        for (double[] row : residuals) {
            if (Double.isFinite(row[caused]) && Double.isFinite(row[causing])) {
                double x = row[caused] - meanX;
                double y = row[causing] - meanY;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }
        }
        double corr = sxy / Math.sqrt(sxx * syy);
        double statistic = count * corr * corr;
        double pValue = new ChiSquareDistribution(1.0).ccdf(statistic);
        return new InstantaneousCausality(caused, causing, statistic, pValue, 1);
    }

    public static double[] finitePairMatrix(double[][] residuals, int caused, int causing, int dimension) {
        int count = usablePairCount(residuals, caused, causing, dimension);
        double[] values = new double[count * 2];
        int rowIndex = 0;
        for (double[] residual : residuals) {
            if (Double.isFinite(residual[caused]) && Double.isFinite(residual[causing])) {
                values[rowIndex * 2] = residual[caused];
                values[rowIndex * 2 + 1] = residual[causing];
                rowIndex++;
            }
        }
        return values;
    }

    public static int usablePairCount(double[][] residuals, int caused, int causing, int dimension) {
        validateEndogIndex(caused, dimension);
        validateEndogIndex(causing, dimension);
        if (caused == causing) {
            throw new IllegalArgumentException("caused and causing indices must differ");
        }
        int count = 0;
        for (double[] residual : residuals) {
            if (Double.isFinite(residual[caused]) && Double.isFinite(residual[causing])) {
                count++;
            }
        }
        return count;
    }

    public static void validateEndogIndex(int index, int dimension) {
        if (index < 0 || index >= dimension) {
            throw new IllegalArgumentException("endog index out of range: " + index);
        }
    }
}