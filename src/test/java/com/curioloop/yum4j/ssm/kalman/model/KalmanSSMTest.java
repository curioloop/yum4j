package com.curioloop.yum4j.ssm.kalman.model;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KalmanSSMTest {

    @Test
    void sameLayoutCopyReusesExistingBackingArrays() {
        KalmanSSM template = buildModel(true);
        KalmanSSM working = buildModel(true);
        double[] design = working.designData();
        double[] obsIntercept = working.obsInterceptData();
        double[] obsCov = working.obsCovData();
        double[] transition = working.transitionData();
        double[] stateIntercept = working.stateInterceptData();
        double[] selection = working.selectionData();
        double[] stateCov = working.stateCovarianceData();
        double[] endog = working.endogData();

        Arrays.fill(working.design, -1.0);
        Arrays.fill(working.obsIntercept, -1.0);
        Arrays.fill(working.obsCov, -1.0);
        Arrays.fill(working.transition, -1.0);
        Arrays.fill(working.stateIntercept, -1.0);
        Arrays.fill(working.selection, -1.0);
        Arrays.fill(working.stateCov, -1.0);
        Arrays.fill(working.endog, -1.0);
        Arrays.fill(working.missing, false);
        Arrays.fill(working.nmissing, 0);

        assertTrue(working.sameLayoutAs(template));
        working.copyDataFromSameLayout(template);

        assertSame(design, working.designData());
        assertSame(obsIntercept, working.obsInterceptData());
        assertSame(obsCov, working.obsCovData());
        assertSame(transition, working.transitionData());
        assertSame(stateIntercept, working.stateInterceptData());
        assertSame(selection, working.selectionData());
        assertSame(stateCov, working.stateCovarianceData());
        assertSame(endog, working.endogData());
        assertArrayEquals(template.designData(), working.designData());
        assertArrayEquals(template.obsInterceptData(), working.obsInterceptData());
        assertArrayEquals(template.obsCovData(), working.obsCovData());
        assertArrayEquals(template.transitionData(), working.transitionData());
        assertArrayEquals(template.stateInterceptData(), working.stateInterceptData());
        assertArrayEquals(template.selectionData(), working.selectionData());
        assertArrayEquals(template.stateCovarianceData(), working.stateCovarianceData());
        assertArrayEquals(template.endogData(), working.endogData());
        assertArrayEquals(template.getMissing(), working.getMissing());
        assertArrayEquals(template.getNmissing(), working.getNmissing());
    }

    @Test
    void sameLayoutRejectsDifferentComponentStride() {
        KalmanSSM timeVarying = buildModel(true);
        KalmanSSM invariant = buildModel(false);

        assertFalse(invariant.sameLayoutAs(timeVarying));
        assertThrows(IllegalArgumentException.class,
            () -> invariant.copyDataFromSameLayout(timeVarying));
    }

    private static KalmanSSM buildModel(boolean timeVaryingTransition) {
        KalmanSSM model = KalmanSSM.builder(2, 2, 1, 3)
            .design(new double[]{1.0, 0.0, 0.2, 1.0}, false)
            .obsIntercept(new double[]{0.1, -0.2}, false)
            .obsCov(new double[]{0.5, 0.1, 0.1, 0.8}, false)
            .transition(timeVaryingTransition
                ? new double[]{0.8, 0.1, 0.0, 0.7, 0.7, 0.2, 0.1, 0.6, 0.6, 0.1, 0.2, 0.5}
                : new double[]{0.8, 0.1, 0.0, 0.7}, timeVaryingTransition)
            .stateIntercept(new double[]{0.0, 0.0}, false)
            .selection(new double[]{1.0, 0.0}, false)
            .stateCovariance(new double[]{0.3}, false)
            .endog(new double[]{1.0, 2.0, 1.5, Double.NaN, 2.0, 2.5})
            .missing(new boolean[]{false, false, false, true, false, false})
            .build();
        return model;
    }
}