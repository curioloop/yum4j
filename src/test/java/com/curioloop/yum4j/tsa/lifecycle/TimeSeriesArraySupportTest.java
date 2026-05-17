package com.curioloop.yum4j.tsa.lifecycle;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TimeSeriesArraySupportTest {

    @Test
    void copiesAndCombinesVectorsAndRows() {
        double[] source = {1.0, 2.0};
        double[] copy = TimeSeriesArraySupport.copyNonEmpty(source, "endog");
        source[0] = 99.0;
        assertArrayEquals(new double[]{1.0, 2.0}, copy);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, TimeSeriesArraySupport.concat(copy, new double[]{3.0}));

        double[][] rows = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] copiedRows = TimeSeriesArraySupport.copyRows(rows, "endog");
        rows[0][0] = 99.0;
        assertArrayEquals(new double[]{1.0, 2.0}, copiedRows[0]);
        assertArrayEquals(new double[]{3.0, 4.0}, TimeSeriesArraySupport.sliceRows(copiedRows, 1, 1)[0]);
        assertArrayEquals(new double[]{5.0, 6.0},
            TimeSeriesArraySupport.concatRows(copiedRows, new double[][]{{5.0, 6.0}})[2]);

        double[][] missing = TimeSeriesArraySupport.missingRows(2, 2);
        assertTrue(Double.isNaN(missing[0][0]));
        assertThrows(IllegalArgumentException.class, () -> TimeSeriesArraySupport.copyNonEmpty(new double[0], "endog"));
        assertThrows(IllegalArgumentException.class, () -> TimeSeriesArraySupport.copyRows(new double[][]{{1.0}, {1.0, 2.0}}, "endog"));
    }

    @Test
    void validatesAndExtendsExogenousRows() {
        double[][] baseExog = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] validated = ExogenousDataSupport.validateRows(baseExog, 1, new double[][]{{5.0, 6.0}}, "bad rows");
        assertArrayEquals(new double[]{5.0, 6.0}, validated[0]);

        double[][] appended = ExogenousDataSupport.appendRows(baseExog, 1, new double[][]{{7.0, 8.0}}, "bad rows");
        assertArrayEquals(new double[]{7.0, 8.0}, appended[2]);

        double[][] extended = ExogenousDataSupport.extendFutureRows(baseExog, 1, new double[][]{{9.0, 10.0}});
        assertArrayEquals(new double[]{9.0, 10.0}, extended[2]);
        assertTrue(ExogenousDataSupport.validateRows(null, 0, null, "bad rows") == null);
        assertThrows(IllegalArgumentException.class,
            () -> ExogenousDataSupport.validateRows(baseExog, 2, new double[][]{{1.0, 2.0}}, "bad rows"));
        assertThrows(IllegalArgumentException.class,
            () -> ExogenousDataSupport.extendFutureRows(null, 1, new double[][]{{1.0}}));
    }
}