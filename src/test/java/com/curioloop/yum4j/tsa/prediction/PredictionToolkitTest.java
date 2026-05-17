package com.curioloop.yum4j.tsa.prediction;

import com.curioloop.yum4j.stats.StudentsTDistribution;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PredictionToolkitTest {

    private static final double TOL = 1e-12;

    @Test
    void metadataNormalizesInformationSetAndDynamicStart() {
        PredictionMetadata metadata = new PredictionMetadata(5, 8, 5,
            PredictionKind.DYNAMIC_IN_SAMPLE, null, true);

        assertEquals(5, metadata.start());
        assertEquals(8, metadata.end());
        assertEquals(4, metadata.length());
        assertEquals(PredictionInformationSet.PREDICTED, metadata.informationSet());
        assertTrue(metadata.signalOnly());
        assertTrue(metadata.dynamic());
        assertEquals(5, metadata.dynamicStart());

        PredictionMetadata beforeRange = new PredictionMetadata(5, 8, 3,
            PredictionKind.IN_SAMPLE, PredictionInformationSet.FILTERED, false);
        assertFalse(beforeRange.dynamic());
        assertEquals(-1, beforeRange.dynamicStart());
    }

    @Test
    void toolkitClassifiesPredictionRanges() {
        assertEquals(PredictionKind.FORECAST,
            PredictionToolkit.predictionKind(10, 12, -1, 10, true));
        assertEquals(PredictionKind.IN_SAMPLE,
            PredictionToolkit.predictionKind(0, 4, -1, 10, false));
        assertEquals(PredictionKind.DYNAMIC_IN_SAMPLE,
            PredictionToolkit.predictionKind(0, 4, 2, 10, false));
        assertEquals(PredictionKind.MIXED,
            PredictionToolkit.predictionKind(8, 11, -1, 10, false));
        assertEquals(PredictionKind.OUT_OF_SAMPLE,
            PredictionToolkit.predictionKind(10, 12, -1, 10, false));

        assertEquals(5, PredictionToolkit.effectiveDynamicStart(5, 8, 3));
        assertEquals(-1, PredictionToolkit.effectiveDynamicStart(5, 8, 9));
        assertThrows(IllegalArgumentException.class,
            () -> PredictionToolkit.validatePredictionBounds(-1, 0, -1));
        assertThrows(IllegalArgumentException.class,
            () -> PredictionToolkit.validatePredictionBounds(1, 0, -1));
        assertThrows(IllegalArgumentException.class,
            () -> PredictionToolkit.validatePredictionBounds(0, 1, -2));
    }

    @Test
    void univariateValuesCopyInputsAndSupportNormalAndTIntervals() {
        double[] mean = {1.0, 2.0};
        double[] variance = {4.0, -1.0};
        UnivariatePredictionValues values = new UnivariatePredictionValues(mean, variance, 9.0, 10.0);
        mean[0] = 99.0;
        variance[0] = 99.0;

        assertArrayEquals(new double[]{1.0, 2.0}, values.mean(), TOL);
        assertArrayEquals(new double[]{4.0, -1.0}, values.variance(), TOL);
        assertArrayEquals(new double[]{13.0, 8.0}, values.variance(true), TOL);
        assertEquals(2.0, values.seMean()[0], TOL);
        assertTrue(Double.isNaN(values.seMean()[1]));
        assertEquals(Math.sqrt(13.0), values.se(true)[0], TOL);

        double[][] tInterval = values.confInt(0.05, true, true);
        double criticalValue = new StudentsTDistribution(10.0).quantileUpperTail(0.025);
        assertEquals(1.0 - criticalValue * Math.sqrt(13.0), tInterval[0][0], TOL);
        assertEquals(1.0 + criticalValue * Math.sqrt(13.0), tInterval[0][1], TOL);

        double[][] normalInterval = values.confInt(0.05);
        assertTrue(normalInterval[0][0] > tInterval[0][0]);
        assertThrows(IllegalArgumentException.class,
            () -> new UnivariatePredictionValues(new double[]{1.0}, new double[]{1.0}).confInt(0.05, true, false));
        assertThrows(IllegalArgumentException.class,
            () -> new UnivariatePredictionValues(new double[]{1.0}, new double[]{1.0}).confInt(0.0));
    }

    @Test
    void multivariateValuesCopyInputsAndUseDiagonalStandardErrors() {
        double[][] mean = {{1.0, 2.0}, {3.0, 4.0}};
        double[][][] variance = {
            {{4.0, 0.5}, {0.5, 9.0}},
            {{-1.0, 0.0}, {0.0, 16.0}}
        };
        MultivariatePredictionValues values = new MultivariatePredictionValues(mean, variance);
        mean[0][0] = 99.0;
        variance[0][0][0] = 99.0;

        assertEquals(2, values.length());
        assertEquals(2, values.observationDimension());
        assertEquals(1.0, values.mean()[0][0], TOL);
        assertEquals(4.0, values.variance()[0][0][0], TOL);
        assertArrayEquals(new double[]{2.0, 3.0}, values.seMean()[0], TOL);
        assertTrue(Double.isNaN(values.seMean()[1][0]));
        assertEquals(4.0, values.seMean()[1][1], TOL);

        double[][] returnedMean = values.mean();
        returnedMean[0][0] = 123.0;
        assertEquals(1.0, values.mean()[0][0], TOL);

        double[][][] interval = values.confInt(0.05);
        assertEquals(2, interval.length);
        assertEquals(2, interval[0].length);
        assertEquals(2, interval[0][0].length);
        assertThrows(IllegalArgumentException.class,
            () -> new MultivariatePredictionValues(new double[][]{{1.0, 2.0}}, new double[][][]{{{1.0}}}));
    }
}