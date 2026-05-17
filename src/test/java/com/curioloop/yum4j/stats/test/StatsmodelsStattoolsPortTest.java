package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.stats.FisherFDistribution;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class StatsmodelsStattoolsPortTest {

    private static final double[] X = {
            -0.1184, -1.3403, 0.0063, -0.612, -0.3869,
            -0.2313, -2.8485, -0.2167, 0.4153, 1.8492,
            -0.3706, 0.9726, -0.1501, -0.0337, -1.4423,
            1.2489, 0.9182, -0.2331, -0.6182, 0.183
    };

    @Test
    void testDurbinWatson() {
        assertEquals(1.95298958377419, DurbinWatson.test(X), 5e-14);
        assertEquals(1.848802400319998, DurbinWatson.test(pow(X, 2.0)), 5e-14);
        assertEquals(1.09897993228779, DurbinWatson.test(lagBlend(0.5)), 5e-14);
        assertEquals(0.937241876707273, DurbinWatson.test(lagBlend(0.8)), 5e-14);
        assertEquals(0.921488912587806, DurbinWatson.test(lagBlend(0.9)), 5e-14);
    }

    @Test
    void testOmniNormtest() {
        Omnibus normal = Omnibus.test(X);
        assertEquals(3.994138321207883, normal.statistic(), 5e-13);
        assertEquals(0.1357325110375005, normal.pValue(), 5e-13);

        Omnibus squared = Omnibus.test(pow(X, 2.0));
        assertEquals(34.523210399523926, squared.statistic(), 5e-12);
        assertEquals(3.186985686465249e-08, squared.pValue(), 5e-18);
    }

    @Test
    void testJarqueBera() {
        assertJarqueBera(X, 1.9662677226861689, 0.3741367669648314, 5e-14);
        assertJarqueBera(pow(X, 2.0), 78.329987305556, 0.0, 5e-13);
        assertJarqueBera(log(pow(X, 2.0)), 5.7135750796706670, 0.0574530296971343, 5e-14);
        assertJarqueBera(expNeg(pow(X, 2.0)), 2.6489315748495761, 0.2659449923067881, 5e-14);
    }

    @Test
    void testBreakvarHeteroskedasticityTest1dInput() {
        double[] residuals = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        double expectedStatistic = 41.0;
        double expectedPValue = twoSidedF(expectedStatistic, 2, 2);

        BreakVar actual = BreakVar.test(residuals, 1.0 / 3.0, BreakVar.Alternative.TWO_SIDED, true);

        assertEquals(expectedStatistic, actual.statistic(), 0.0);
        assertEquals(expectedPValue, actual.pValue(), 5e-15);
    }

    @Test
    void testBreakvarHeteroskedasticityTestSubsetLength() {
        double[] residuals = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};

        BreakVar subsetCount = BreakVar.test(residuals, 2.0, BreakVar.Alternative.TWO_SIDED, true);
        assertEquals(41.0, subsetCount.statistic(), 0.0);
        assertEquals(twoSidedF(41.0, 2, 2), subsetCount.pValue(), 5e-15);

        BreakVar subsetFraction = BreakVar.test(residuals, 0.5, BreakVar.Alternative.TWO_SIDED, true);
        assertEquals(10.0, subsetFraction.statistic(), 0.0);
        assertEquals(twoSidedF(10.0, 3, 3), subsetFraction.pValue(), 5e-15);
    }

    @Test
    void testBreakvarHeteroskedasticityTestAlternative() {
        double[] residuals = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};

        BreakVar twoSided = BreakVar.test(residuals, 1.0 / 3.0, BreakVar.Alternative.TWO_SIDED, true);
        assertEquals(41.0, twoSided.statistic(), 0.0);
        assertEquals(twoSidedF(41.0, 2, 2), twoSided.pValue(), 5e-15);

        BreakVar decreasing = BreakVar.test(residuals, 1.0 / 3.0, BreakVar.Alternative.DECREASING, true);
        assertEquals(1.0 / 41.0, decreasing.statistic(), 0.0);
        assertEquals(new FisherFDistribution(2, 2).ccdf(1.0 / 41.0), decreasing.pValue(), 5e-15);

        BreakVar increasing = BreakVar.test(residuals, 1.0 / 3.0, BreakVar.Alternative.INCREASING, true);
        assertEquals(41.0, increasing.statistic(), 0.0);
        assertEquals(new FisherFDistribution(2, 2).ccdf(41.0), increasing.pValue(), 5e-15);
    }

    @Test
    void testBreakvarHeteroskedasticityTestUseChi2() {
        double[] residuals = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        double expectedStatistic = 41.0;
        double scaled = 2.0 * expectedStatistic;
        ChiSquareDistribution chi2 = new ChiSquareDistribution(2);
        double expectedPValue = 2.0 * Math.min(chi2.cdf(scaled), chi2.ccdf(scaled));

        BreakVar actual = BreakVar.test(residuals, 1.0 / 3.0, BreakVar.Alternative.TWO_SIDED, false);

        assertEquals(expectedStatistic, actual.statistic(), 0.0);
        assertEquals(expectedPValue, actual.pValue(), 5e-15);
    }

    @Test
    void testBreakvarHeteroskedasticityTestMissingColumnWithTooFewValues() {
        double[] residuals = {Double.NaN, 1.0, Double.NaN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        assertThrows(IllegalArgumentException.class, () ->
            BreakVar.test(residuals, 1.0 / 3.0, BreakVar.Alternative.TWO_SIDED, true));
    }

    private static void assertJarqueBera(double[] x, double statistic, double pValue, double tol) {
        JarqueBera actual = JarqueBera.test(x);
        assertEquals(statistic, actual.statistic(), tol);
        assertEquals(pValue, actual.pValue(), tol);
    }

    private static double[] lagBlend(double coefficient) {
        double[] out = new double[X.length - 1];
        for (int i = 0; i < out.length; i++) out[i] = X[i + 1] + coefficient * X[i];
        return out;
    }

    private static double[] pow(double[] x, double power) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) out[i] = Math.pow(x[i], power);
        return out;
    }

    private static double[] log(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) out[i] = Math.log(x[i]);
        return out;
    }

    private static double[] expNeg(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) out[i] = Math.exp(-x[i]);
        return out;
    }

    private static double twoSidedF(double statistic, int dfNum, int dfDen) {
        FisherFDistribution f = new FisherFDistribution(dfNum, dfDen);
        return 2.0 * Math.min(f.cdf(statistic), f.ccdf(statistic));
    }
}
