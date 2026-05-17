package com.curioloop.yum4j.tsa.sarimax;

import com.curioloop.yum4j.tsa.prediction.PredictionInformationSet;
import com.curioloop.yum4j.tsa.prediction.PredictionKind;
import com.curioloop.yum4j.tsa.prediction.PredictionMetadata;
import com.curioloop.yum4j.tsa.prediction.UnivariatePredictionValues;

import java.util.Arrays;

public final class SARIMAXPrediction {

    private static final double DEFAULT_CONFIDENCE_ALPHA = 0.05;

    private final PredictionMetadata metadata;
    private final UnivariatePredictionValues values;

    SARIMAXPrediction(int start, int end, boolean dynamic, double[] mean, double[] variance) {
        this(start, end, dynamic ? start : -1, mean, variance);
    }

    SARIMAXPrediction(int start, int end, int dynamicStart, double[] mean, double[] variance) {
        this(start, end, dynamicStart,
            dynamicStart >= start && dynamicStart <= end ? PredictionKind.DYNAMIC_IN_SAMPLE : PredictionKind.IN_SAMPLE,
            mean, variance);
    }

    SARIMAXPrediction(int start,
                      int end,
                      int dynamicStart,
                      PredictionKind kind,
                      double[] mean,
                      double[] variance) {
        this(start, end, dynamicStart, kind, PredictionInformationSet.PREDICTED, false, mean, variance);
    }

    SARIMAXPrediction(int start,
                      int end,
                      int dynamicStart,
                      PredictionKind kind,
                      PredictionInformationSet informationSet,
                      boolean signalOnly,
                      double[] mean,
                      double[] variance) {
        this.metadata = new PredictionMetadata(start, end, dynamicStart, kind, informationSet, signalOnly);
        this.values = new UnivariatePredictionValues(mean, variance);
    }

    public int start() {
        return metadata.start();
    }

    public int end() {
        return metadata.end();
    }

    public PredictionKind kind() {
        return metadata.kind();
    }

    public PredictionInformationSet informationSet() {
        return metadata.informationSet();
    }

    public boolean signalOnly() {
        return metadata.signalOnly();
    }

    public boolean dynamic() {
        return metadata.dynamic();
    }

    public int dynamicStart() {
        return metadata.dynamicStart();
    }

    public double[] mean() {
        return values.mean();
    }

    public double[] variance() {
        return values.variance();
    }

    public double[] seMean() {
        return values.seMean();
    }

    public double[][] confInt() {
        return confInt(DEFAULT_CONFIDENCE_ALPHA);
    }

    public double[][] confInt(double alpha) {
        return values.confInt(alpha);
    }

    public double[] meanCiLower() {
        return meanCiLower(DEFAULT_CONFIDENCE_ALPHA);
    }

    public double[] meanCiLower(double alpha) {
        double[][] intervals = confInt(alpha);
        double[] lower = new double[intervals.length];
        for (int i = 0; i < intervals.length; i++) {
            lower[i] = intervals[i][0];
        }
        return lower;
    }

    public double[] meanCiUpper() {
        return meanCiUpper(DEFAULT_CONFIDENCE_ALPHA);
    }

    public double[] meanCiUpper(double alpha) {
        double[][] intervals = confInt(alpha);
        double[] upper = new double[intervals.length];
        for (int i = 0; i < intervals.length; i++) {
            upper[i] = intervals[i][1];
        }
        return upper;
    }

    public SummaryFrame summaryFrame() {
        return summaryFrame(DEFAULT_CONFIDENCE_ALPHA);
    }

    public SummaryFrame summaryFrame(double alpha) {
        double[][] intervals = confInt(alpha);
        double[] lower = new double[intervals.length];
        double[] upper = new double[intervals.length];
        for (int i = 0; i < intervals.length; i++) {
            lower[i] = intervals[i][0];
            upper[i] = intervals[i][1];
        }
        return new SummaryFrame(values.mean(), values.seMean(), lower, upper);
    }

    public static final class SummaryFrame {
        private final double[] mean;
        private final double[] meanSe;
        private final double[] meanCiLower;
        private final double[] meanCiUpper;

        private SummaryFrame(double[] mean, double[] meanSe, double[] meanCiLower, double[] meanCiUpper) {
            this.mean = Arrays.copyOf(mean, mean.length);
            this.meanSe = Arrays.copyOf(meanSe, meanSe.length);
            this.meanCiLower = Arrays.copyOf(meanCiLower, meanCiLower.length);
            this.meanCiUpper = Arrays.copyOf(meanCiUpper, meanCiUpper.length);
        }

        public double[] mean() {
            return Arrays.copyOf(mean, mean.length);
        }

        public double[] meanSe() {
            return Arrays.copyOf(meanSe, meanSe.length);
        }

        public double[] meanCiLower() {
            return Arrays.copyOf(meanCiLower, meanCiLower.length);
        }

        public double[] meanCiUpper() {
            return Arrays.copyOf(meanCiUpper, meanCiUpper.length);
        }
    }
}