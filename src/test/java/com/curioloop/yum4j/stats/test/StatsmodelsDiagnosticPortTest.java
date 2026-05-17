package com.curioloop.yum4j.stats.test;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StatsmodelsDiagnosticPortTest {

    private static StatsmodelsTestData.DiagnosticFixture fixture;

    @BeforeAll
    static void setupClass() {
        fixture = StatsmodelsTestData.diagnosticFixture();
    }

    @Test
    void testHetBreuschPagan() {
        BreuschPagan actual = BreuschPagan.test(fixture.residual(), fixture.exog(), 202, 3);
        assertEquals(0.709924388395087, actual.lmStat(), 5e-13);
        assertEquals(0.701199952134347, actual.lmPval(), 5e-13);
    }

    @Test
    void testHetBreuschPagan1dErr() {
        double[] lastColumn = StatsmodelsTestData.column(fixture.exog(), 202, 3, 2);
        assertThrows(IllegalArgumentException.class, () ->
                BreuschPagan.test(fixture.residual(), lastColumn, 202, 1));

        double[] ones = new double[202];
        java.util.Arrays.fill(ones, 1.0);
        assertThrows(IllegalArgumentException.class, () ->
                BreuschPagan.test(fixture.residual(), ones, 202, 1));

        double[] noConstant = fixture.exog().clone();
        for (int i = 0; i < 202; i++) noConstant[i * 3] = 0.0;
        assertThrows(IllegalArgumentException.class, () ->
                BreuschPagan.test(fixture.residual(), noConstant, 202, 3));
    }

    @Test
    void testHetBreuschPaganNonrobust() {
        BreuschPagan actual = BreuschPagan.test(fixture.residual(), fixture.exog(), 202, 3, false);
        assertEquals(1.3020140634828294, actual.lmStat(), 5e-13);
        assertEquals(0.5215203247111985, actual.lmPval(), 5e-13);
    }

    @Test
    void testHetWhite() {
        White actual = White.test(fixture.residual(), fixture.exog(), 202, 3);
        assertEquals(33.503722896538441, actual.lmStat(), 5e-7);
        assertEquals(2.9887960597830259e-06, actual.lmPval(), 5e-7);
        assertEquals(7.7945101228430946, actual.fStat(), 5e-7);
        assertEquals(1.0354575277704231e-06, actual.fPval(), 5e-7);
    }

    @Test
    void testHetWhiteNoInteractionTerms() {
        White actual = White.test(fixture.residual(), fixture.exog(), 202, 3, false);
        assertEquals(13.25091965953952, actual.lmStat(), 5e-7);
        assertEquals(0.001326170478134868, actual.lmPval(), 5e-7);
        assertEquals(6.985287047470471, actual.fStat(), 5e-7);
        assertEquals(0.001169716842511783, actual.fPval(), 5e-7);
    }

    @Test
    void testHetWhiteError() {
        double[] constantOnly = StatsmodelsTestData.column(fixture.exog(), 202, 3, 0);
        assertThrows(IllegalArgumentException.class, () ->
                White.test(fixture.residual(), constantOnly, 202, 1));
    }

    @Test
    void testHetArch() {
        ARCH lag4 = ARCH.test(fixture.residual(), 4, 0);
        assertEquals(3.43473400836259, lag4.lmStat(), 5e-12);
        assertEquals(0.487871315392619, lag4.lmPval(), 5e-13);

        ARCH lag12 = ARCH.test(fixture.residual(), 12, 0);
        assertEquals(8.648320999014171, lag12.lmStat(), 5e-12);
        assertEquals(0.732638635007718, lag12.lmPval(), 5e-13);
    }

    @Test
    void testHetArch2() {
        ARCH first = ARCH.test(fixture.residual(), 5, 0);
        ARCH second = ARCH.test(fixture.residual(), 5, 0);
        assertEquals(first.lmStat(), second.lmStat(), 5e-13);
        assertEquals(first.lmPval(), second.lmPval(), 5e-13);
        assertEquals(first.fStat(), second.fStat(), 5e-13);
        assertEquals(first.fPval(), second.fPval(), 5e-13);
    }

    @Test
    void testAcorrBreuschGodfrey() {
        BreuschGodfrey actual = BreuschGodfrey.test(fixture.residual(), fixture.exog(), 202, 3, 4);
        assertEquals(4.771042651230007, actual.lmStat(), 5e-11);
        assertEquals(0.3116067133066697, actual.lmPval(), 5e-11);
        assertEquals(1.179280833676792, actual.fStat(), 5e-11);
        assertEquals(0.321197487261203, actual.fPval(), 5e-11);

        BreuschGodfrey defaultLag = BreuschGodfrey.test(fixture.residual(), fixture.exog(), 202, 3);
        BreuschGodfrey lag10 = BreuschGodfrey.test(fixture.residual(), fixture.exog(), 202, 3, 10);
        assertEquals(defaultLag.lmStat(), lag10.lmStat(), 5e-13);
        assertEquals(defaultLag.lmPval(), lag10.lmPval(), 5e-13);
        assertEquals(defaultLag.fStat(), lag10.fStat(), 5e-13);
        assertEquals(defaultLag.fPval(), lag10.fPval(), 5e-13);
    }

    @Test
    void testAcorrLjungBox() {
        Box actual = Box.test(fixture.residual(), Box.Statistic.BOTH, 0, false, 0, 4);
        assertEquals(5.23587172795227, actual.ljungBoxStat()[3], 5e-13);
        assertEquals(0.263940335284713, actual.ljungBoxPv()[3], 5e-13);
        assertEquals(5.12462932741681, actual.boxPierceStat()[3], 5e-13);
        assertEquals(0.2747471266820692, actual.boxPiercePv()[3], 5e-13);
    }

    @Test
    void testAcorrLjungBoxBigDefault() {
        Box actual = Box.test(fixture.residual(), Box.Statistic.BOTH, 0, false, 0, 40);
        int last = actual.lags().length - 1;
        assertEquals(51.03724531797195, actual.ljungBoxStat()[last], 5e-12);
        assertEquals(0.11334744923390, actual.ljungBoxPv()[last], 5e-13);
        assertEquals(45.12238537034000, actual.boxPierceStat()[last], 5e-12);
        assertEquals(0.26638168491464, actual.boxPiercePv()[last], 5e-13);
    }

    @Test
    void testAcorrLjungBoxSmallDefault() {
        double[] residual = StatsmodelsTestData.copyOfRange(fixture.residual(), 30);
        Box actual = Box.test(residual, Box.Statistic.BOTH, 0, false, 0, 13);
        int last = actual.lags().length - 1;
        assertEquals(9.61503968281915, actual.ljungBoxStat()[last], 5e-13);
        assertEquals(0.72507000996945, actual.ljungBoxPv()[last], 5e-13);
        assertEquals(7.41692150864936, actual.boxPierceStat()[last], 5e-13);
        assertEquals(0.87940785887006, actual.boxPiercePv()[last], 5e-13);
    }

    @Test
    void testLjungboxDofAdj() {
        Box noAdjust = Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, 0, false, 0, 10);
        Box adjusted = Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, 4, false, 0, 10);
        assertArrayEquals(noAdjust.ljungBoxStat(), adjusted.ljungBoxStat(), 0.0);
        for (int i = 0; i < 4; i++) assertTrue(Double.isNaN(adjusted.ljungBoxPv()[i]));
        for (int i = 4; i < adjusted.ljungBoxPv().length; i++) {
            assertTrue(adjusted.ljungBoxPv()[i] <= noAdjust.ljungBoxPv()[i]);
        }
    }

    @Test
    void testLjungboxAutoLagSelection() {
        Box noAdjust = Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, 0, true, 0);
        Box adjusted = Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, 4, true, 0);
        assertTrue(noAdjust.lags().length >= 1);
        assertTrue(adjusted.lags().length >= 1);
        assertArrayEquals(noAdjust.ljungBoxStat(), adjusted.ljungBoxStat(), 0.0);
        for (int i = 0; i < Math.min(4, adjusted.ljungBoxPv().length); i++) {
            assertTrue(Double.isNaN(adjusted.ljungBoxPv()[i]));
        }
    }

    @Test
    void testLjungboxAutoLagWhitenoise() {
        Random random = new Random(1732);
        double[] data = new double[1000];
        for (int i = 0; i < data.length; i++) data[i] = random.nextGaussian();
        Box actual = Box.test(data, Box.Statistic.LJUNG_BOX, 0, true, 0);
        assertTrue(actual.lags().length >= 1);
    }

    @Test
    void testLjungboxErrorsWarnings() {
        assertThrows(IllegalArgumentException.class, () ->
                Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, -1, false, 0));
        assertThrows(IllegalArgumentException.class, () ->
                Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, -1, false, 1));
        assertThrows(IllegalArgumentException.class, () ->
                Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, -1, false, -2));
        assertNotNull(Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, 0, false, 0));
        assertEquals(10, Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, 0, false, 0, 10).lags().length);
    }

    @Test
    void testLjungboxPeriod() {
        Box period = Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, 0, false, 13);
        Box explicit = Box.test(fixture.residual(), Box.Statistic.LJUNG_BOX, 0, false, 0, 26);
        assertArrayEquals(explicit.lags(), period.lags());
        assertArrayEquals(explicit.ljungBoxStat(), period.ljungBoxStat(), 0.0);
        assertArrayEquals(explicit.ljungBoxPv(), period.ljungBoxPv(), 0.0);
    }

    @Test
    void testCusumOls() {
        CUSUM actual = CUSUM.test(fixture.residual(), 3);
        assertEquals(1.055750610401214, actual.statistic(), 5e-13);
        assertEquals(0.2149567397376543, actual.pValue(), 5e-13);
    }

    @Test
    void testResetSmoke() {
        Random random = new Random(20240507L);
        int n = 1000;
        double[] x = new double[n * 7];
        double[] base = new double[n * 4];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i * 7] = 1.0;
            base[i * 4] = 1.0;
            double sum = 1.0;
            for (int j = 1; j < 4; j++) {
                double value = random.nextGaussian();
                x[i * 7 + j] = value;
                base[i * 4 + j] = value;
                sum += value;
            }
            for (int j = 1; j < 4; j++) {
                double square = x[i * 7 + j] * x[i * 7 + j];
                x[i * 7 + 3 + j] = square;
                sum += square;
            }
            y[i] = sum + random.nextGaussian();
        }
        int[][] powers = {{2}, {3}, {4}, {2, 3, 4}};
        for (int[] power : powers) {
            assertDoesNotThrow(() -> RESET.test(y, base, n, 4, false, power));
            assertDoesNotThrow(() -> RESET.test(y, base, n, 4, true, power));
        }
    }
}
