package com.curioloop.yum4j.stats.tool;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TrendTermsTest {

    @Test
    void parsesStatsmodelsTrendAliases() {
        assertSame(TrendTerms.NONE, TrendTerms.fromString(null));
        assertSame(TrendTerms.NONE, TrendTerms.fromString("n"));
        assertSame(TrendTerms.CONSTANT, TrendTerms.fromString("c"));
        assertSame(TrendTerms.LINEAR, TrendTerms.fromString("t"));
        assertSame(TrendTerms.CONSTANT_TREND, TrendTerms.fromString("ct"));
        assertSame(TrendTerms.CONSTANT_TREND_QUADRATIC, TrendTerms.fromString("ctt"));
        assertThrows(IllegalArgumentException.class, () -> TrendTerms.fromString("tc"));
    }

    @Test
    void exposesCodesTermsAndTrendPowers() {
        assertEquals("ct", TrendTerms.CONSTANT_TREND.code());
        assertEquals(0, TrendTerms.NONE.term());
        assertEquals(1, TrendTerms.CONSTANT.term());
        assertEquals(2, TrendTerms.CONSTANT_TREND.term());
        assertEquals(3, TrendTerms.CONSTANT_TREND_QUADRATIC.term());
        assertThrows(IllegalArgumentException.class, TrendTerms.LINEAR::term);
        assertArrayEquals(new int[0], TrendTerms.NONE.powers());
        assertArrayEquals(new int[]{0}, TrendTerms.CONSTANT.powers());
        assertArrayEquals(new int[]{1}, TrendTerms.LINEAR.powers());
        assertArrayEquals(new int[]{0, 1}, TrendTerms.CONSTANT_TREND.powers());
        assertArrayEquals(new int[]{0, 1, 2}, TrendTerms.CONSTANT_TREND_QUADRATIC.powers());
    }

    @Test
    void normalizesTrendPowers() {
        assertArrayEquals(new int[0], TrendTerms.normalizePowers(null));
        assertArrayEquals(new int[]{0, 2, 3}, TrendTerms.normalizePowers(new int[]{3, 0, 2, 2}));
        assertThrows(IllegalArgumentException.class, () -> TrendTerms.normalizePowers(new int[]{-1}));
        assertThrows(IllegalArgumentException.class, () -> TrendTerms.validateOffset(-1));
    }
}