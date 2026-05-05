package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.kalman.arena.SmootherScratchLayout;
import com.curioloop.yum4j.kalman.arena.ZSmootherScratchLayout;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SmootherScratchLayoutTest {

    @Test
    void testRealSmootherScratchLayoutBaseMatchesCurrentRetainedContract() {
        int kEndog = 3;
        int kStates = 4;

        SmootherScratchLayout layout = SmootherScratchLayout.create(
            kEndog,
            kStates,
            false,
            false,
            false,
            false,
            false,
            false,
            false);

        assertEquals(0, layout.rBase());
        assertEquals(kStates, layout.rPrevBase());
        assertEquals(2 * kStates, layout.nBase());
        assertEquals(2L * kStates + 5L * kStates * kStates, layout.totalLength());
        assertEquals(-1, layout.tmpFivBase());
        assertEquals(-1, layout.tmpFcopyBase());
        assertEquals(-1, layout.tmpTkBase());
        assertEquals(-1, layout.tmpKhBase());
        assertEquals(-1, layout.tmpJBase());
        assertEquals(-1, layout.rInfBase());
        assertEquals(-1, layout.scratchSmoothingErrorBase());
    }

    @Test
    void testRealSmootherScratchLayoutOptionalSectionDeltasMatchCurrentFormulas() {
        int kEndog = 3;
        int kStates = 4;

        SmootherScratchLayout base = SmootherScratchLayout.create(
            kEndog, kStates,
            false, false, false, false, false, false, false);
        SmootherScratchLayout observation = SmootherScratchLayout.create(
            kEndog, kStates,
            true, false, false, false, false, false, false);
        SmootherScratchLayout forecast = SmootherScratchLayout.create(
            kEndog, kStates,
            true, true, false, false, false, false, false);
        SmootherScratchLayout transition = SmootherScratchLayout.create(
            kEndog, kStates,
            true, true, true, false, false, false, false);
        SmootherScratchLayout disturbanceCov = SmootherScratchLayout.create(
            kEndog, kStates,
            true, true, true, true, false, false, false);
        SmootherScratchLayout factor = SmootherScratchLayout.create(
            kEndog, kStates,
            true, true, true, true, true, false, false);
        SmootherScratchLayout diffuse = SmootherScratchLayout.create(
            kEndog, kStates,
            true, true, true, true, true, true, false);
        SmootherScratchLayout borrowed = SmootherScratchLayout.create(
            kEndog, kStates,
            true, true, true, true, true, true, true);

        assertEquals(kEndog + 3L * kStates * kEndog + (long) kEndog * kEndog,
            observation.totalLength() - base.totalLength());
        assertEquals((long) kEndog * kEndog,
            forecast.totalLength() - observation.totalLength());
        assertEquals((long) kStates * kEndog,
            transition.totalLength() - forecast.totalLength());
        assertEquals((long) kStates * kEndog + (long) kEndog * kEndog,
            disturbanceCov.totalLength() - transition.totalLength());
        assertEquals((long) kStates * kStates,
            factor.totalLength() - disturbanceCov.totalLength());
        assertEquals(6L * kStates + 11L * kStates * kStates,
            diffuse.totalLength() - factor.totalLength());
        assertEquals(kEndog,
            borrowed.totalLength() - diffuse.totalLength());
    }

    @Test
    void testComplexSmootherScratchLayoutMatchesCurrentRetainedContractAndAlignment() {
        int kEndog = 3;
        int kStates = 4;
        int kPosdef = 2;

        ZSmootherScratchLayout base = ZSmootherScratchLayout.create(
            kEndog, kStates, kPosdef,
            false, false, false, false, false, false, false);
        ZSmootherScratchLayout observation = ZSmootherScratchLayout.create(
            kEndog, kStates, kPosdef,
            true, false, false, false, false, false, false);
        ZSmootherScratchLayout forecast = ZSmootherScratchLayout.create(
            kEndog, kStates, kPosdef,
            true, true, false, false, false, false, false);
        ZSmootherScratchLayout transition = ZSmootherScratchLayout.create(
            kEndog, kStates, kPosdef,
            true, true, true, false, false, false, false);
        ZSmootherScratchLayout disturbanceCov = ZSmootherScratchLayout.create(
            kEndog, kStates, kPosdef,
            true, true, true, true, false, false, false);
        ZSmootherScratchLayout factor = ZSmootherScratchLayout.create(
            kEndog, kStates, kPosdef,
            true, true, true, true, true, false, false);
        ZSmootherScratchLayout diffuse = ZSmootherScratchLayout.create(
            kEndog, kStates, kPosdef,
            true, true, true, true, true, true, false);
        ZSmootherScratchLayout borrowed = ZSmootherScratchLayout.create(
            kEndog, kStates, kPosdef,
            true, true, true, true, true, true, true);

        assertEquals(4L * kStates + 8L * kStates * kStates + 2L * kStates * Math.max(kStates, kPosdef),
            base.totalLength());
        assertEquals(2L * kEndog + 6L * kStates * kEndog + 2L * kEndog * kEndog,
            observation.totalLength() - base.totalLength());
        assertEquals(2L * kEndog * kEndog,
            forecast.totalLength() - observation.totalLength());
        assertEquals(2L * kStates * kEndog,
            transition.totalLength() - forecast.totalLength());
        assertEquals(2L * kStates * kEndog + 2L * kEndog * kEndog,
            disturbanceCov.totalLength() - transition.totalLength());
        assertEquals(2L * kStates * kStates,
            factor.totalLength() - disturbanceCov.totalLength());
        assertEquals(14L * kStates + 22L * kStates * kStates,
            diffuse.totalLength() - factor.totalLength());
        assertEquals(2L * kEndog,
            borrowed.totalLength() - diffuse.totalLength());

        assertTrue(base.rBase() % 2 == 0);
        assertTrue(base.rPrevBase() % 2 == 0);
        assertTrue(base.nBase() % 2 == 0);
        assertTrue(observation.tmpFivBase() % 2 == 0);
        assertTrue(observation.tmpFiZBase() % 2 == 0);
        assertTrue(forecast.tmpFcopyBase() % 2 == 0);
        assertTrue(transition.tmpTkBase() % 2 == 0);
        assertTrue(disturbanceCov.tmpKhBase() % 2 == 0);
        assertTrue(disturbanceCov.tmpFiHBase() % 2 == 0);
        assertTrue(factor.tmpJBase() % 2 == 0);
        assertTrue(diffuse.rInfBase() % 2 == 0);
        assertTrue(diffuse.tmpQuadBase() % 2 == 0);
        assertTrue(borrowed.scratchSmoothingErrorBase() % 2 == 0);
    }
}