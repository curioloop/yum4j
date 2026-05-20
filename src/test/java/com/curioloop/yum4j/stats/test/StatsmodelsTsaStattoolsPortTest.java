package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.math.Double3;
import com.curioloop.yum4j.stats.tool.TrendTerms;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StatsmodelsTsaStattoolsPortTest {

    private static final String JOHANSEN_MATLAB_RESOURCE =
            "statsmodels/tsa/vector_ar/tests/Matlab_results/test_coint.csv";

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

    @ParameterizedTest(name = "{0}")
    @MethodSource("engleGrangerCases")
    void engleGrangerMatchesStatsmodelsMacrodata(EngleGrangerCase testCase) {
        Coint actual = Coint.test(testCase.y0, testCase.y1, testCase.n, testCase.k,
                testCase.trend, testCase.maxLag, ADF.AutoLag.NONE);
        assertEquals(testCase.statistic, actual.statistic(), 6e-12, testCase.name + " statistic");
        assertEquals(testCase.pValue, actual.pValue(), 6e-12, testCase.name + " pValue");
        assertCriticalValues(actual.critValues(), testCase.criticalValues);
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

    @ParameterizedTest(name = "{0}")
    @MethodSource("grangerCases")
    void grangerCausalityMatchesStatsmodelsMacrodata(GrangerCase testCase) {
        GrangerCausality.Lag actual = GrangerCausality.test(
                testCase.dependent, testCase.causing, testCase.maxLag).lag(testCase.lag);
        assertGrangerTest(testCase.ssrFTest, actual.ssrFStat(), actual.ssrFPValue(),
                actual.ssrFDfDenom(), actual.ssrFDfNum(), testCase.name + " ssr_ftest");
        assertGrangerTest(testCase.ssrChi2Test, actual.ssrChi2Stat(), actual.ssrChi2PValue(),
                actual.ssrChi2Df(), testCase.name + " ssr_chi2test");
        assertGrangerTest(testCase.lrTest, actual.lrStat(), actual.lrPValue(),
                actual.lrDf(), testCase.name + " lrtest");
        assertGrangerTest(testCase.paramsFTest, actual.paramsFStat(), actual.paramsFPValue(),
                actual.paramsFDfDenom(), actual.paramsFDfNum(), testCase.name + " params_ftest");
    }

    @Test
    void testGrangercausalitySingle() {
        double[] data = grangerData();
        GrangerCausality all = GrangerCausality.test(data, realgdp.length - 1, 2);
        GrangerCausality onlyLag2 = GrangerCausality.test(data, realgdp.length - 1, new int[]{2});
        assertEquals(1, all.lag(1).lag());
        assertThrows(IllegalArgumentException.class, () -> onlyLag2.lag(1));
        assertEquals(all.lag(2).ssrFStat(), onlyLag2.lag(2).ssrFStat(), 5e-12);
        assertEquals(all.lag(2).paramsFStat(), onlyLag2.lag(2).ssrFStat(), 5e-12);
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
    void testGrangerFailsOnPerfectFit() {
        double[] dependent = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        double[] causing = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};
        assertThrows(IllegalArgumentException.class, () -> GrangerCausality.test(dependent, causing, 1));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("johansenCases")
    void johansenMatchesStatsmodelsMatlabCases(JohansenCase testCase) {
        Johansen actual = Johansen.test(testCase.endog, testCase.observations, testCase.equations,
                testCase.detOrder, testCase.kArDiff);
        assertEquals(testCase.rows, actual.rows(), testCase.name + " rows");
        assertEquals(testCase.rows * testCase.equations, actual.r0t().length, testCase.name + " r0t shape");
        assertArray(testCase.eigenvalues, actual.eigenvalues(), 5e-9);
        assertArray(testCase.traceStatistic, actual.traceStatistic(), 5e-7);
        assertArray(testCase.maxEigenStatistic, actual.maxEigenStatistic(), 5e-7);
        assertArray(testCase.traceCriticalValues, actual.traceCriticalValues(), 0.0);
        assertArray(testCase.maxEigenCriticalValues, actual.maxEigenCriticalValues(), 0.0);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("johansenRankCases")
    void johansenSelectRankMatchesStatsmodelsDecisionRule(JohansenRankCase testCase) {
        Johansen.Rank actual = Johansen.selectRank(testCase.endog, testCase.observations, testCase.equations,
                testCase.detOrder, testCase.kArDiff, testCase.method, testCase.significance);
        assertEquals(testCase.expectedRank, actual.rank(), testCase.name + " rank");
        assertEquals(Math.min(testCase.expectedRank + 1, testCase.equations), actual.testStatistics().length,
                testCase.name + " selected statistic length");
        for (int index = 0; index < testCase.expectedRank; index++) {
            assertTrue(actual.testStatistics()[index] >= actual.criticalValues()[index],
                    testCase.name + " accepted rank " + index);
        }
        if (testCase.expectedRank < testCase.equations) {
            int rank = testCase.expectedRank;
            assertTrue(actual.testStatistics()[rank] < actual.criticalValues()[rank],
                    testCase.name + " rejected rank " + rank);
        }
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

    static Stream<EngleGrangerCase> engleGrangerCases() {
        return Stream.of(
                new EngleGrangerCase("macro realcons~realgdp trend c", realcons, realgdp, realcons.length, 1,
                        TrendTerms.CONSTANT, 0, -1.8301709861482625, 0.6149545601930455,
                        a(-3.951478986373885, -3.3665452338006077, -3.06551269973532)),
                new EngleGrangerCase("macro realcons~realgdp trend ct", realcons, realgdp, realcons.length, 1,
                        TrendTerms.CONSTANT_TREND, 0, -2.4833495908388805, 0.5294919529383552,
                        a(-4.4049236074894615, -3.8279480805803354, -3.53154901146364)),
                new EngleGrangerCase("macro realcons~realgdp trend ctt", realcons, realgdp, realcons.length, 1,
                        TrendTerms.CONSTANT_TREND_QUADRATIC, 0, -2.8903464566823462, 0.5207398335208977,
                        a(-4.7944808062983535, -4.220455253799618, -3.925679074547147)),
                new EngleGrangerCase("macro realcons~realgdp trend n", realcons, realgdp, realcons.length, 1,
                        TrendTerms.NONE, 0, 0.0481520456836473, 0.9450365518706334,
                        a(Double.NaN, Double.NaN, Double.NaN))
        );
    }

    static Stream<GrangerCase> grangerCases() {
        double[] gdp = StatsmodelsTestData.logDiff(realgdp);
        double[] cons = StatsmodelsTestData.logDiff(realcons);
        return Stream.of(
                new GrangerCase("realcons caused by realgdp lag 1", cons, gdp, 2, 1,
                        a(2.880790764760717, 0.09121322302946275, 198.0, 1.0),
                        a(2.924439109681334, 0.08724753958581503, 1.0),
                        a(2.9033687506971546, 0.08839464223795485, 1.0),
                        a(2.8807907647607385, 0.09121322302946114, 198.0, 1.0)),
                new GrangerCase("realcons caused by realgdp lag 2", cons, gdp, 2, 2,
                        a(0.2430969779600214, 0.7844327580357401, 195.0, 2.0),
                        a(0.4986604676103003, 0.7793225722272918, 2.0),
                        a(0.498039843343804, 0.7795644430029454, 2.0),
                        a(0.24309697796001709, 0.7844327580357401, 195.0, 2.0)),
                new GrangerCase("realgdp caused by realcons lag 1", gdp, cons, 2, 1,
                        a(28.996970864156857, 2.0418103098396673e-7, 198.0, 1.0),
                        a(29.436318907553172, 5.778451900023302e-8, 1.0),
                        a(27.470598139901313, 1.595009841393057e-7, 1.0),
                        a(28.996970864156868, 2.0418103098396405e-7, 198.0, 1.0)),
                new GrangerCase("realgdp caused by realcons lag 2", gdp, cons, 2, 2,
                        a(19.03531756939809, 2.8076149148443223e-8, 195.0, 2.0),
                        a(39.04680527056018, 3.3196627673384598e-9, 2.0),
                        a(35.6684008323723, 1.7976501283518487e-8, 2.0),
                        a(19.035317569398167, 2.8076149148441406e-8, 195.0, 2.0))
        );
    }

    static Stream<JohansenCase> johansenCases() {
        StatsmodelsTestData.MatrixData data = StatsmodelsTestData.whitespaceMatrixResource(JOHANSEN_MATLAB_RESOURCE, 8);
        return Stream.of(
                new JohansenCase("Matlab test_coint det=1 k_ar_diff=2", data.data(), data.rows(), data.columns(),
                        1, 2, 170,
                        a(0.3586376068088151, 0.2812806889719111, 0.2074818815675726,
                                0.141259991767926, 0.09880133062878599, 0.08704563854307619,
                                0.048471840356709, 0.01919823444066367),
                        a(241.985452556075, 166.4781461662553, 110.3298006342814,
                                70.79801574443575, 44.90887371527634, 27.22385073668511,
                                11.74205493173769, 3.295435325623445),
                        a(75.50730638981975, 56.14834553197396, 39.5317848898456,
                                25.8891420291594, 17.68502297859124, 15.48179580494741,
                                8.446619606114249, 3.295435325623445),
                        a(169.0618, 175.1584, 187.1891, 133.7852, 139.278, 150.0778,
                                102.4674, 107.3429, 116.9829, 75.1027, 79.3422, 87.7748,
                                51.6492, 55.2459, 62.5202, 32.0645, 35.0116, 41.0815,
                                16.1619, 18.3985, 23.1485, 2.7055, 3.8415, 6.6349),
                        a(52.5858, 55.7302, 62.1741, 46.5583, 49.5875, 55.8171,
                                40.5244, 43.4183, 49.4095, 34.4202, 37.1646, 42.8612,
                                28.2398, 30.8151, 36.193, 21.8731, 24.2522, 29.2631,
                                15.0006, 17.1481, 21.7465, 2.7055, 3.8415, 6.6349),
                        3, 2),
                new JohansenCase("Matlab test_coint det=0 k_ar_diff=9", data.data(), data.rows(), data.columns(),
                        0, 9, 163,
                        a(0.46614865357440605, 0.37361928647101916, 0.24506354877141895,
                                0.17213036500666942, 0.12796709727560032, 0.09548831435967518,
                                0.079088807196285, 0.0025228185812092123),
                        a(307.6888935095814, 205.3839229398245, 129.1330243009336,
                                83.3101865760208, 52.51955460357912, 30.20027050520502,
                                13.84158157562689, 0.4117390188204866),
                        a(102.3049705697569, 76.25089863889085, 45.82283772491284,
                                30.7906319724417, 22.31928409837409, 16.35868892957814,
                                13.4298425568064, 0.4117390188204866),
                        a(153.6341, 159.529, 171.0905, 120.3673, 125.6185, 135.9825,
                                91.109, 95.7542, 104.9637, 65.8202, 69.8189, 77.8202,
                                44.4929, 47.8545, 54.6815, 27.0669, 29.7961, 35.4628,
                                13.4294, 15.4943, 19.9349, 2.7055, 3.8415, 6.6349),
                        a(49.2855, 52.3622, 58.6634, 43.2947, 46.2299, 52.3069,
                                37.2786, 40.0763, 45.8662, 31.2379, 33.8777, 39.3693,
                                25.1236, 27.5858, 32.7172, 18.8928, 21.1314, 25.865,
                                12.2971, 14.2639, 18.52, 2.7055, 3.8415, 6.6349),
                        6, 3),
                new JohansenCase("Matlab test_coint det=-1 k_ar_diff=8", data.data(), data.rows(), data.columns(),
                        -1, 8, 164,
                        a(0.4494521674673037, 0.29346722475993714, 0.19052225612548393,
                                0.13337502818295263, 0.11246534913158243, 0.08784341213925552,
                                0.0636505155496506, 0.013649770664117525),
                        a(260.6786029744658, 162.7966072512681, 105.8253545950566,
                                71.16133060790817, 47.68490211260372, 28.11843682526138,
                                13.03968537077271, 2.25398078597622),
                        a(97.88199572319769, 56.97125265621156, 34.66402398714837,
                                23.47642849530445, 19.56646528734234, 15.07875145448866,
                                10.7857045847965, 2.25398078597622),
                        a(137.9954, 143.6691, 154.7977, 106.7351, 111.7797, 121.7375,
                                79.5329, 83.9383, 92.7136, 56.2839, 60.0627, 67.6367,
                                37.0339, 40.1749, 46.5716, 21.7781, 24.2761, 29.5147,
                                10.4741, 12.3212, 16.364, 2.9762, 4.1296, 6.9406),
                        a(45.893, 48.8795, 55.0335, 39.9085, 42.7679, 48.6606,
                                33.9271, 36.6301, 42.2333, 27.916, 30.4428, 35.7359,
                                21.837, 24.1592, 29.0609, 15.7175, 17.7961, 22.2519,
                                9.4748, 11.2246, 15.0923, 2.9762, 4.1296, 6.9406),
                        7, 2)
        );
    }

    static Stream<JohansenRankCase> johansenRankCases() {
        return johansenCases().flatMap(testCase -> Stream.of(
                new JohansenRankCase(testCase.name + " trace 5%", testCase.endog, testCase.observations,
                        testCase.equations, testCase.detOrder, testCase.kArDiff,
                        Johansen.RankMethod.TRACE, 0.05, testCase.selectTraceRank05),
                new JohansenRankCase(testCase.name + " maxeig 5%", testCase.endog, testCase.observations,
                        testCase.equations, testCase.detOrder, testCase.kArDiff,
                        Johansen.RankMethod.MAX_EIGEN, 0.05, testCase.selectMaxEigenRank05)
        ));
    }

    private static void assertCriticalValues(Double3 actual, double onePercent, double fivePercent, double tenPercent) {
        assertEquals(onePercent, actual._1(), 1.5e-2);
        assertEquals(fivePercent, actual._2(), 1.5e-2);
        assertEquals(tenPercent, actual._3(), 1.5e-2);
    }

    private static void assertCriticalValues(Double3 actual, double[] expected) {
        assertClose(expected[0], actual._1(), 1.5e-10, "critical value 1%");
        assertClose(expected[1], actual._2(), 1.5e-10, "critical value 5%");
        assertClose(expected[2], actual._3(), 1.5e-10, "critical value 10%");
    }

    private static void assertGrangerTest(double[] expected, double statistic, double pValue,
                                          int dfOrDenom, String message) {
        assertEquals(expected[0], statistic, 6e-11, message + " statistic");
        assertEquals(expected[1], pValue, 6e-12, message + " pValue");
        assertEquals((int) expected[2], dfOrDenom, message + " df");
    }

    private static void assertGrangerTest(double[] expected, double statistic, double pValue,
                                          int dfDenom, int dfNum, String message) {
        assertEquals(expected[0], statistic, 6e-11, message + " statistic");
        assertEquals(expected[1], pValue, 6e-12, message + " pValue");
        assertEquals((int) expected[2], dfDenom, message + " dfDenom");
        assertEquals((int) expected[3], dfNum, message + " dfNum");
    }

    private static double[] grangerData() {
        double[] gdp = StatsmodelsTestData.logDiff(realgdp);
        double[] cons = StatsmodelsTestData.logDiff(realcons);
        return StatsmodelsTestData.twoColumn(cons, gdp);
    }

    private static void assertArray(double[] expected, double[] actual, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int index = 0; index < expected.length; index++) {
            assertClose(expected[index], actual[index], tolerance, "index " + index);
        }
    }

    private static void assertClose(double expected, double actual, double tolerance, String message) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), message);
        } else {
            assertEquals(expected, actual, tolerance, message);
        }
    }

    private static double[] a(double... values) {
        return values;
    }

    private static double[] randomTwoColumn(int n) {
        Random random = new Random(987654321L);
        double[] x = new double[n * 2];
        for (int i = 0; i < x.length; i++) x[i] = random.nextDouble();
        return x;
    }

    private record EngleGrangerCase(String name, double[] y0, double[] y1, int n, int k,
                                    TrendTerms trend, int maxLag, double statistic,
                                    double pValue, double[] criticalValues) {
        @Override
        public String toString() {
            return name;
        }
    }

    private record GrangerCase(String name, double[] dependent, double[] causing,
                               int maxLag, int lag, double[] ssrFTest,
                               double[] ssrChi2Test, double[] lrTest, double[] paramsFTest) {
        @Override
        public String toString() {
            return name;
        }
    }

    private record JohansenCase(String name, double[] endog, int observations, int equations,
                                int detOrder, int kArDiff, int rows, double[] eigenvalues,
                                double[] traceStatistic, double[] maxEigenStatistic,
                                double[] traceCriticalValues, double[] maxEigenCriticalValues,
                                int selectTraceRank05, int selectMaxEigenRank05) {
        @Override
        public String toString() {
            return name;
        }
    }

    private record JohansenRankCase(String name, double[] endog, int observations, int equations,
                                    int detOrder, int kArDiff, Johansen.RankMethod method,
                                    double significance, int expectedRank) {
        @Override
        public String toString() {
            return name;
        }
    }
}
