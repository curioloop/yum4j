package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.math.Double3;
import com.curioloop.yum4j.stats.tool.TrendTerms;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StatsmodelsTsaStattoolsPortTest {

    private static Map<String, double[]> macro;
    private static double[] realgdp;
    private static double[] realcons;
    private static double[] infl;

    @BeforeAll
    static void setupClass() {
        macro = StatsmodelsTestData.macrodata();
        realgdp = macro.get("realgdp");
        realcons = macro.get("realcons");
        infl = macro.get("infl");
    }

    @Test
    void testAdfConstantTeststatPvalueCritvalues() {
        ADF actual = ADF.test(realgdp, 4, TrendTerms.CONSTANT, ADF.AutoLag.NONE);
        assertEquals(0.97505319, actual.statistic(), 5e-6);
        assertEquals(0.99399563, actual.pValue(), 5e-6);
        assertCriticalValues(actual.critValues(), -3.476, -2.883, -2.573);
    }

    @Test
    void testAdfConstantTrendTeststatPvalueCritvalues() {
        ADF actual = ADF.test(realgdp, 4, TrendTerms.CONSTANT_TREND, ADF.AutoLag.NONE);
        assertEquals(-1.8566374, actual.statistic(), 5e-6);
        assertEquals(0.67682968, actual.pValue(), 5e-6);
        assertCriticalValues(actual.critValues(), -4.007, -3.437, -3.137);
    }

    @Test
    void testAdfNoConstantTeststatPvalueCritvalues() {
        ADF actual = ADF.test(realgdp, 4, TrendTerms.NONE, ADF.AutoLag.NONE);
        assertEquals(3.5227498, actual.statistic(), 5e-6);
        assertEquals(0.99999, actual.pValue(), 5e-5);
        assertCriticalValues(actual.critValues(), -2.587, -1.950, -1.617);
    }

    @Test
    void testAdfConstant2TeststatPvalueCritvalues() {
        ADF actual = ADF.test(infl, 1, TrendTerms.CONSTANT, ADF.AutoLag.NONE);
        assertEquals(-4.3346988, actual.statistic(), 5e-6);
        assertEquals(0.00038661, actual.pValue(), 5e-8);
        assertCriticalValues(actual.critValues(), -3.476, -2.883, -2.573);
    }

    @Test
    void testAdfConstantTrend2TeststatPvalueCritvalues() {
        ADF actual = ADF.test(infl, 1, TrendTerms.CONSTANT_TREND, ADF.AutoLag.NONE);
        assertEquals(-4.425093, actual.statistic(), 5e-6);
        assertEquals(0.00199633, actual.pValue(), 5e-8);
        assertCriticalValues(actual.critValues(), -4.006, -3.437, -3.137);
    }

    @Test
    void testAdfNoConstant2TeststatPvalueCritvalues() {
        ADF actual = ADF.test(infl, 1, TrendTerms.NONE, ADF.AutoLag.NONE);
        assertEquals(-2.4511596, actual.statistic(), 5e-6);
        assertEquals(0.013747, actual.pValue(), 5e-6);
        assertCriticalValues(actual.critValues(), -2.587, -1.950, -1.617);
    }

    @Test
    void testAdfullerResidVarianceZero() {
        assertThrows(IllegalArgumentException.class, () ->
                ADF.test(new double[]{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0}));
    }

    @Test
    void testAdfullerShortSeries() {
        ADF.test(new double[]{0.125, -0.25, 0.5, -0.75, 0.875, -0.625, 0.375});
        assertThrows(IllegalArgumentException.class, () -> ADF.test(new double[]{0.1, 0.2}));
        assertThrows(IllegalArgumentException.class, () ->
                ADF.test(new double[]{0.1, 0.2, 0.4}, null, TrendTerms.CONSTANT_TREND, ADF.AutoLag.AIC));
    }

    @Test
    void testAdfullerMaxlagTooLarge() {
        double[] y = new double[100];
        for (int i = 0; i < y.length; i++) y[i] = Math.sin(i * 0.1) + i * 0.01;
        assertThrows(IllegalArgumentException.class, () ->
                ADF.test(y, 51, TrendTerms.CONSTANT, ADF.AutoLag.AIC));
    }

    @Test
    void testCointT() {
        Coint actual = Coint.test(realcons, realgdp, realcons.length, 1,
                TrendTerms.CONSTANT, 0, ADF.AutoLag.NONE);
        assertEquals(-1.830170986148, actual.statistic(), 5e-11);
    }

    @Test
    void testCointIdenticalSeries() {
        Coint actual = Coint.test(realgdp, realgdp, realgdp.length, 1,
                TrendTerms.CONSTANT, 0, ADF.AutoLag.NONE);
        assertEquals(0.0, actual.pValue(), 0.0);
        assertTrue(Double.isInfinite(actual.statistic()) && actual.statistic() < 0.0);
    }

    @Test
    void testCointPerfectCollinearity() {
        int n = 200;
        double[] x = new double[n * 2];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i * 2] = Math.sin(i * 0.13);
            x[i * 2 + 1] = Math.cos(i * 0.07);
            y[i] = 1.0 + x[i * 2] + x[i * 2 + 1];
        }
        Coint actual = Coint.test(y, x, n, 2, TrendTerms.CONSTANT, 0, ADF.AutoLag.NONE);
        assertEquals(0.0, actual.pValue(), 0.0);
        assertTrue(Double.isInfinite(actual.statistic()) && actual.statistic() < 0.0);
    }

    @Test
    void testGrangercausality() {
        double[] data = grangerData();
        GrangerCausality.Lag actual = GrangerCausality.test(data, realgdp.length - 1, 2).lag(2);
        assertEquals(0.243097, actual.ssrFStat(), 5e-7);
        assertEquals(0.7844328, actual.ssrFPValue(), 5e-7);
        assertEquals(195, actual.ssrFDfDenom());
        assertEquals(2, actual.ssrFDfNum());
        assertEquals(actual.ssrFStat(), actual.paramsFStat(), 5e-7);
        assertEquals(actual.ssrFPValue(), actual.paramsFPValue(), 5e-7);
    }

    @Test
    void testGrangercausalitySingle() {
        double[] data = grangerData();
        GrangerCausality all = GrangerCausality.test(data, realgdp.length - 1, 2);
        GrangerCausality onlyLag2 = GrangerCausality.test(data, realgdp.length - 1, new int[]{2});
        assertEquals(1, all.lag(1).lag());
        assertThrows(IllegalArgumentException.class, () -> onlyLag2.lag(1));
        assertEquals(all.lag(2).ssrFStat(), onlyLag2.lag(2).ssrFStat(), 5e-7);
        assertEquals(all.lag(2).paramsFStat(), onlyLag2.lag(2).ssrFStat(), 5e-7);
    }

    @Test
    void testGrangerFailsOnNobsCheck() {
        double[] x = randomTwoColumn(10);
        GrangerCausality.test(x, 10, 2);
        assertThrows(IllegalArgumentException.class, () -> GrangerCausality.test(x, 10, 3));
    }

    @Test
    void testGrangerFailsOnFiniteCheck() {
        double[] x = randomTwoColumn(1000);
        x[500 * 2] = Double.NaN;
        x[750 * 2 + 1] = Double.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, () -> GrangerCausality.test(x, 1000, 2));
    }

    @Test
    void testGrangerFailsOnZeroLag() {
        double[] x = randomTwoColumn(1000);
        assertThrows(IllegalArgumentException.class, () -> GrangerCausality.test(x, 1000, 0, 1, 2));
    }

    @Test
    void testKpssTeststat() {
        KPSS constant = KPSS.test(realgdp, KPSS.Regression.CONSTANT, null, 3);
        assertEquals(5.0169, constant.statistic(), 5e-4);

        KPSS trend = KPSS.test(realgdp, KPSS.Regression.TREND, null, 3);
        assertEquals(1.1828, trend.statistic(), 5e-4);
    }

    @Test
    void testKpssPval() {
        KPSS constant = KPSS.test(realgdp, KPSS.Regression.CONSTANT, null, 3);
        assertEquals(0.01, constant.pValue(), 0.0);

        KPSS trend = KPSS.test(realgdp, KPSS.Regression.TREND, null, 3);
        assertEquals(0.01, trend.pValue(), 0.0);
    }

    @Test
    void testKpssLags() {
        assertEquals(9, KPSS.test(realgdp, KPSS.Regression.CONSTANT, KPSS.LagMethod.AUTO, null).lags());
    }

    @Test
    void testKpssReferenceDatasetLags() {
        double[] sunspots = StatsmodelsTestData.referenceDataset("sunspots/sunspots.csv", ',').get("SUNACTIVITY");
        assertEquals(7, KPSS.test(sunspots, KPSS.Regression.CONSTANT, KPSS.LagMethod.AUTO, null).lags());

        double[] nile = StatsmodelsTestData.referenceDataset("nile/nile.csv", ',').get("volume");
        assertEquals(5, KPSS.test(nile, KPSS.Regression.CONSTANT, KPSS.LagMethod.AUTO, null).lags());

        double[] randhie = StatsmodelsTestData.referenceDataset("randhie/randhie.csv", ',').get("lncoins");
        assertEquals(75, KPSS.test(randhie, KPSS.Regression.TREND, KPSS.LagMethod.AUTO, null).lags());

        double[] modechoice = StatsmodelsTestData.referenceDataset("modechoice/modechoice.csv", ';').get("invt");
        assertEquals(18, KPSS.test(modechoice, KPSS.Regression.TREND, KPSS.LagMethod.AUTO, null).lags());
    }

    @Test
    void testKpssFailsOnNobsCheck() {
        int nobs = realgdp.length;
        assertThrows(IllegalArgumentException.class, () ->
                KPSS.test(realgdp, KPSS.Regression.CONSTANT, null, nobs));
    }

    @Test
    void testKpssAutolagsDoesNotAssignLagsEqualToNobs() {
        double[] base = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0};
        double[] data = new double[297];
        int offset = 0;
        for (int repeat = 0; repeat < 297 / 7; repeat++) {
            System.arraycopy(base, 0, data, offset, base.length);
            offset += base.length;
        }
        data[offset++] = 0.0;
        data[offset++] = 0.0;
        data[offset] = 0.0;
        KPSS actual = KPSS.test(data, KPSS.Regression.CONSTANT, KPSS.LagMethod.AUTO, null);
        assertTrue(actual.lags() < data.length);
    }

    @Test
    void testLegacyLags() {
        assertEquals(15, KPSS.test(realgdp, KPSS.Regression.CONSTANT, KPSS.LagMethod.LEGACY, null).lags());
    }

    private static void assertCriticalValues(Double3 actual, double onePercent, double fivePercent, double tenPercent) {
        assertEquals(onePercent, actual._1(), 1.5e-2);
        assertEquals(fivePercent, actual._2(), 1.5e-2);
        assertEquals(tenPercent, actual._3(), 1.5e-2);
    }

    private static double[] grangerData() {
        double[] gdp = StatsmodelsTestData.logDiff(realgdp);
        double[] cons = StatsmodelsTestData.logDiff(realcons);
        return StatsmodelsTestData.twoColumn(cons, gdp);
    }

    private static double[] randomTwoColumn(int n) {
        Random random = new Random(987654321L);
        double[] x = new double[n * 2];
        for (int i = 0; i < x.length; i++) x[i] = random.nextDouble();
        return x;
    }
}
