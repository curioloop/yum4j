package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterMethod;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ExactDiffuseStatsmodelsTest {

    private static final double TOL = 1e-8;
    private static final double[][] DFM_ENDOG = {
        {9.97685232655492, 6.1144429662540745},
        {-0.4771808442672665, 4.154391094858667},
        {1.3978130617488205, 0.43360437850346045},
        {8.876071805717345, 3.8136603506846},
        {-1.8738213130934156, 5.028971219769218},
        {0.653152075567931, -1.5871706073063763},
        {-5.162543978431344, 0.5372132872409452},
        {2.3690365021785453, -0.11185995207014798},
        {7.413813661557356, 5.907936382029888},
        {6.412655664724554, 1.9354521733205843},
        {8.061126418759201, 7.929224802804313},
        {7.110903714734462, 4.236466453014032},
        {4.392206139990407, 4.886493514750612},
        {3.681626885358469, 3.224810681980017},
        {0.9708074856973781, 5.6330208687420225},
        {5.194084181405856, 2.684917722929825},
        {4.981133837251406, 3.8017109127682858},
        {7.4616162956381515, 5.406166649308375},
        {3.029544715731447, 3.3396476673878794},
        {8.878345547537947, 7.821669914095963}
    };

    @Test
    void commonLevelAnalyticMatchesStatsmodelsReference() {
        double y11 = 10.2394;
        double y21 = 8.2304;
        double theta = 0.1111;
        double sigma2Mu = 3.2324;

        FilterResult result = KalmanEngine.filter(
            commonLevelModel(y11, y21, theta, sigma2Mu, false),
            InitialState.diffuse(),
            FilterOptions.defaults());

        assertMatrix(result, 0, new double[][]{{0.0, 0.0}, {0.0, 0.0}}, false);
        assertMatrix(result, 0, new double[][]{{1.0, 0.0}, {0.0, 1.0}}, true);

        assertEquals(y11, result.predictedState(0, 1), TOL);
        assertEquals(y21 - theta * y11, result.predictedState(1, 1), TOL);
        assertMatrix(result, 1, new double[][]{
            {1.0 + sigma2Mu, -theta},
            {-theta, 1.0 + theta * theta}
        }, false);
        assertMatrix(result, 1, new double[][]{{0.0, 0.0}, {0.0, 0.0}}, true);
        assertEquals(1, result.nobsDiffuse);
    }

    @Test
    void restrictedCommonLevelAnalyticMatchesStatsmodelsReference() {
        double y11 = 10.2394;
        double y21 = 8.2304;
        double theta = 0.1111;
        double sigma2Mu = 3.2324;
        double phi = 1.0 / (1.0 + theta * theta);

        FilterResult result = KalmanEngine.filter(
            commonLevelModel(y11, y21, theta, sigma2Mu, true),
            InitialState.diffuse(),
            FilterOptions.defaults());

        assertEquals(0.0, result.predictedStateCov(0, 0, 0), TOL);
        assertEquals(1.0, result.predictedDiffuseStateCov(0, 0, 0), TOL);
        assertEquals(phi * (y11 + theta * y21), result.predictedState(0, 1), TOL);
        assertEquals(phi + sigma2Mu, result.predictedStateCov(0, 0, 1), TOL);
        assertEquals(0.0, result.predictedDiffuseStateCov(0, 0, 1), TOL);
        assertEquals(1, result.nobsDiffuse);
    }

    @Test
    void dynamicFactorExactDiffuseMatchesStatsmodelsReference() {
        FilterResult result = KalmanEngine.filter(
            dynamicFactorModel(2),
            InitialState.diffuse(2),
            FilterOptions.builder().method(FilterMethod.UNIVARIATE).build());

        assertMatrix(result, 0, new double[][]{{0.0, 0.0}, {0.0, 0.0}}, false);
        assertMatrix(result, 0, new double[][]{{1.0, 0.0}, {0.0, 1.0}}, true);
        assertEquals(2, result.nobsDiffuse);

        assertEquals(3.5469084479061324, result.predictedState(0, 2), TOL);
        assertEquals(2.8772028990103675, result.predictedState(1, 2), TOL);
        assertMatrix(result, 2, new double[][]{
            {2.2299999999999995, 1.3499999999999996},
            {1.3499999999999996, 1.4999999999999996}
        }, false);
        assertMatrix(result, 2, new double[][]{{0.0, 0.0}, {0.0, 0.0}}, true);
    }

    @Test
    void collapsedDynamicFactorExactDiffuseMatchesStatsmodelsReference() {
        FilterResult result = KalmanEngine.filter(
            dynamicFactorModel(1),
            InitialState.diffuse(1),
            FilterOptions.builder().method(FilterMethod.COLLAPSED).build());

        assertEquals(0.0, result.predictedStateCov(0, 0, 0), TOL);
        assertEquals(1.0, result.predictedDiffuseStateCov(0, 0, 0), TOL);
        assertEquals(1, result.nobsDiffuse);

        assertEquals(8.616832549171214, result.predictedState(0, 1), TOL);
        assertEquals(2.215, result.predictedStateCov(0, 0, 1), TOL);
        assertEquals(0.0, result.predictedDiffuseStateCov(0, 0, 1), TOL);
    }

    @Test
    void irrelevantDiffuseStateDoesNotDisturbObservedState() {
        double[] endog = {1.2, -0.4, 0.7, 1.5, -0.2, 0.3, 1.0, 0.6};
        FilterResult baseline = KalmanEngine.filter(
            localLevelModel(endog),
            InitialState.diffuse(1),
            FilterOptions.defaults());
        FilterResult withIrrelevantState = KalmanEngine.filter(
            localLevelWithIrrelevantState(endog),
            InitialState.diffuse(2),
            FilterOptions.defaults());

        for (int t = 0; t < endog.length; t++) {
            assertEquals(baseline.filteredState(0, t), withIrrelevantState.filteredState(0, t), TOL, "filtered state t=" + t);
            assertEquals(baseline.filteredStateCov(0, 0, t), withIrrelevantState.filteredStateCov(0, 0, t), TOL,
                "filtered state covariance t=" + t);
            assertEquals(baseline.predictedState(0, t), withIrrelevantState.predictedState(0, t), TOL,
                "predicted state t=" + t);
            assertEquals(baseline.predictedStateCov(0, 0, t), withIrrelevantState.predictedStateCov(0, 0, t), TOL,
                "predicted state covariance t=" + t);
        }
    }

    private static ModelFixture commonLevelModel(double y11,
                                                 double y21,
                                                 double theta,
                                                 double sigma2Mu,
                                                 boolean restricted) {
        int nobs = 10;
        ModelFixture model = restricted
            ? new ModelFixture(2, 1, 1, nobs)
            : new ModelFixture(2, 2, 1, nobs);
        double[] obsCov = {1.0, 0.0, 0.0, 1.0};
        double[] transition = restricted ? new double[]{1.0} : new double[]{1.0, 0.0, 0.0, 1.0};
        double[] selection = restricted ? new double[]{1.0} : new double[]{1.0, 0.0};
        double[] stateCov = {sigma2Mu};
        double[] design = restricted ? new double[]{1.0, theta} : new double[]{1.0, 0.0, theta, 1.0};
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(t == 0 ? new double[]{y11, y21} : new double[]{1.0, 1.0}, t);
        }
        return model;
    }

    private static ModelFixture dynamicFactorModel(int factorOrder) {
        ModelFixture model = new ModelFixture(2, factorOrder, 1, DFM_ENDOG.length);
        double[] design = factorOrder == 1 ? new double[]{0.5, 1.0} : new double[]{0.5, 0.0, 1.0, 0.0};
        double[] transition = factorOrder == 1 ? new double[]{0.9} : new double[]{0.9, 0.1, 1.0, 0.0};
        double[] selection = factorOrder == 1 ? new double[]{1.0} : new double[]{1.0, 0.0};
        double[] obsCov = {1.5, 0.0, 0.0, 2.0};
        double[] stateCov = {1.0};
        for (int t = 0; t < DFM_ENDOG.length; t++) {
            model.setDesign(design, t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(DFM_ENDOG[t], t);
        }
        return model;
    }

    private static ModelFixture localLevelModel(double[] endog) {
        ModelFixture model = new ModelFixture(1, 1, 1, endog.length);
        for (int t = 0; t < endog.length; t++) {
            model.setDesign(new double[]{1.0}, t);
            model.setObsCov(new double[]{0.8}, t);
            model.setTransition(new double[]{1.0}, t);
            model.setSelection(new double[]{1.0}, t);
            model.setStateCov(new double[]{0.4}, t);
            model.setEndog(new double[]{endog[t]}, t);
        }
        return model;
    }

    private static ModelFixture localLevelWithIrrelevantState(double[] endog) {
        ModelFixture model = new ModelFixture(1, 2, 1, endog.length);
        for (int t = 0; t < endog.length; t++) {
            model.setDesign(new double[]{1.0, 0.0}, t);
            model.setObsCov(new double[]{0.8}, t);
            model.setTransition(new double[]{1.0, 0.0, 0.0, 1.0}, t);
            model.setSelection(new double[]{1.0, 0.0}, t);
            model.setStateCov(new double[]{0.4}, t);
            model.setEndog(new double[]{endog[t]}, t);
        }
        return model;
    }

    private static void assertMatrix(FilterResult result, int t, double[][] expected, boolean diffuse) {
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[i].length; j++) {
                double actual = diffuse
                    ? result.predictedDiffuseStateCov(i, j, t)
                    : result.predictedStateCov(i, j, t);
                assertEquals(expected[i][j], actual, TOL, "t=" + t + " i=" + i + " j=" + j);
            }
        }
    }
}