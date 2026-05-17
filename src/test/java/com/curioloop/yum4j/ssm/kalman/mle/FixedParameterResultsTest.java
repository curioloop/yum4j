package com.curioloop.yum4j.ssm.kalman.mle;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FixedParameterResultsTest {

    @Test
    void masksFixedCovarianceRowsAndColumns() {
        FixedParameters fixed = FixedParameters.of(new int[]{1}, new double[]{2.0});
        double[] covariance = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        double[] adjusted = FixedParameterResults.adjustedCovariance(covariance, fixed);
        assertEquals(2, FixedParameterResults.freeParameterCount(3, fixed));
        assertEquals(4, FixedParameterResults.dfModel(3, fixed, 2));
        assertEquals(1.0, adjusted[0]);
        assertTrue(Double.isNaN(adjusted[1]));
        assertTrue(Double.isNaN(adjusted[3]));
        assertTrue(Double.isNaN(adjusted[4]));
        assertEquals(9.0, adjusted[8]);
        assertSame(covariance, FixedParameterResults.adjustedCovariance(covariance, FixedParameters.none()));
    }
}