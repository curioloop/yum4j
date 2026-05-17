package com.curioloop.yum4j.tsa.varmax;

import com.curioloop.yum4j.tsa.prediction.MultivariatePredictionValues;
import com.curioloop.yum4j.tsa.prediction.PredictionInformationSet;
import com.curioloop.yum4j.tsa.prediction.PredictionKind;
import com.curioloop.yum4j.tsa.prediction.PredictionMetadata;

public final class VARMAXPrediction {

    private static final double DEFAULT_CONFIDENCE_ALPHA = 0.05;

    private final PredictionMetadata metadata;
    private final MultivariatePredictionValues values;

    VARMAXPrediction(int start, int end, int dynamicStart, double[][] mean, double[][][] variance) {
        this(start, end, dynamicStart,
            dynamicStart >= start && dynamicStart <= end ? PredictionKind.DYNAMIC_IN_SAMPLE : PredictionKind.IN_SAMPLE,
            mean, variance);
    }

    VARMAXPrediction(int start,
                     int end,
                     int dynamicStart,
                     PredictionKind kind,
                     double[][] mean,
                     double[][][] variance) {
        this(start, end, dynamicStart, kind, PredictionInformationSet.PREDICTED, false, mean, variance);
    }

    VARMAXPrediction(int start,
                     int end,
                     int dynamicStart,
                     PredictionKind kind,
                     PredictionInformationSet informationSet,
                     boolean signalOnly,
                     double[][] mean,
                     double[][][] variance) {
        this.metadata = new PredictionMetadata(start, end, dynamicStart, kind, informationSet, signalOnly);
        this.values = new MultivariatePredictionValues(mean, variance);
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

    public int observationDimension() {
        return values.observationDimension();
    }

    public double[][] mean() {
        return values.mean();
    }

    public double[][][] variance() {
        return values.variance();
    }

    public double[][] seMean() {
        return values.seMean();
    }

    public double[][][] confInt() {
        return confInt(DEFAULT_CONFIDENCE_ALPHA);
    }

    public double[][][] confInt(double alpha) {
        return values.confInt(alpha);
    }

    public SummaryFrame summaryFrame() {
        return summaryFrame(DEFAULT_CONFIDENCE_ALPHA);
    }

    public SummaryFrame summaryFrame(double alpha) {
        double[][][] intervals = confInt(alpha);
        return new SummaryFrame(values.mean(), values.seMean(), intervals);
    }

    private static double[][] copyMatrix(double[][] source) {
        return MultivariatePredictionValues.copyMatrix(source);
    }

    private static double[][][] copyCube(double[][][] source) {
        return MultivariatePredictionValues.copyCube(source);
    }

    public static final class SummaryFrame {
        private final double[][] mean;
        private final double[][] meanSe;
        private final double[][][] meanCi;

        private SummaryFrame(double[][] mean, double[][] meanSe, double[][][] meanCi) {
            this.mean = copyMatrix(mean);
            this.meanSe = copyMatrix(meanSe);
            this.meanCi = copyCube(meanCi);
        }

        public double[][] mean() {
            return copyMatrix(mean);
        }

        public double[][] meanSe() {
            return copyMatrix(meanSe);
        }

        public double[][][] meanCi() {
            return copyCube(meanCi);
        }
    }
}
