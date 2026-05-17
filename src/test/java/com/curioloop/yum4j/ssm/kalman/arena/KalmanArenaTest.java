package com.curioloop.yum4j.ssm.kalman.arena;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KalmanArenaTest {

    @Test
    void doubleArenaGrowsMonotonicallyAndReusesEqualOrSmallerRequests() {
        DoubleArena arena = new DoubleArena();

        double[] first = arena.ensureCapacity(8);
        double[] smaller = arena.ensureCapacity(3);
        double[] equal = arena.ensureCapacity(8);
        double[] larger = arena.ensureCapacity(13);

        assertSame(first, smaller);
        assertSame(first, equal);
        assertNotSame(first, larger);
        assertEquals(13L, arena.retainedDoubleCount());
        assertEquals(13L * Double.BYTES, arena.retainedByteCount());
        assertEquals(13L, arena.peakDoubleCount());
        assertEquals(2L, arena.growthCount());
    }

    @Test
    void intArenaGrowsMonotonicallyAndReusesEqualOrSmallerRequests() {
        IntArena arena = new IntArena();

        int[] first = arena.ensureCapacity(5);
        int[] smaller = arena.ensureCapacity(1);
        int[] larger = arena.ensureCapacity(9);

        assertSame(first, smaller);
        assertNotSame(first, larger);
        assertEquals(9L, arena.retainedIntCount());
        assertEquals(9L * Integer.BYTES, arena.retainedByteCount());
        assertEquals(9L, arena.peakIntCount());
        assertEquals(2L, arena.growthCount());
    }

    @Test
    void arenaPeakCanBeResetIndependentlyFromRetainedBacking() {
        DoubleArena doubleArena = new DoubleArena();
        doubleArena.ensureCapacity(4);
        doubleArena.ensureCapacity(9);
        doubleArena.release();

        assertTrue(doubleArena.isEmpty());
        assertEquals(9L, doubleArena.peakDoubleCount());
        assertEquals(2L, doubleArena.growthCount());

        doubleArena.resetPeak();
        assertEquals(0L, doubleArena.peakDoubleCount());
        assertEquals(0L, doubleArena.growthCount());

        IntArena intArena = new IntArena();
        intArena.ensureCapacity(3);
        intArena.resetPeak();
        intArena.ensureCapacity(2);

        assertEquals(3L, intArena.peakIntCount());
        assertEquals(0L, intArena.growthCount());
    }

    @Test
    void slicesBindCurrentBackingAndValidateBounds() {
        DoubleArena doubleArena = new DoubleArena();
        DoubleArenaSlice doubleSlice = doubleArena.bind(new ArenaSegment(2, 4));
        assertSame(doubleArena.backing(), doubleSlice.backing());
        assertEquals(2, doubleSlice.offset());
        assertEquals(4, doubleSlice.length());
        assertEquals(6, doubleSlice.end());
        assertFalse(doubleSlice.isEmpty());

        IntArena intArena = new IntArena();
        IntArenaSlice intSlice = intArena.bind(1, 0);
        assertSame(intArena.backing(), intSlice.backing());
        assertEquals(1, intSlice.offset());
        assertEquals(0, intSlice.length());
        assertTrue(intSlice.isEmpty());

        assertThrows(IllegalArgumentException.class, () -> new ArenaSegment(-1, 1));
        assertThrows(IllegalArgumentException.class, () -> new DoubleArenaSlice(new double[2], 1, 2));
        assertThrows(IllegalArgumentException.class, () -> new IntArenaSlice(new int[2], 0, 3));
    }

    @Test
    void scratchAndResultArenasReleaseIndependently() {
        DoubleArena scratch = new DoubleArena();
        DoubleArena result = new DoubleArena();

        scratch.ensureCapacity(10);
        result.ensureCapacity(20);
        scratch.release();

        assertTrue(scratch.isEmpty());
        assertEquals(20L, result.retainedDoubleCount());

        result.release();
        assertTrue(result.isEmpty());
    }

    @Test
    void collapsedScratchLayoutPacksEveryRetainedDoubleFamily() {
        CollapsedScratchLayout layout = CollapsedScratchLayout.create(3, 2, 2, 5);

        assertEquals(-1, layout.logLikelihoodAdjustmentBase());
        assertEquals(0, layout.designBase());
        assertEquals(layout.designBase(), layout.forecastMeanBase());
        assertEquals(layout.forecastMeanBase(), layout.selectedErrorBase());
        assertEquals(layout.forecastErrorDiffuseCovBase(), layout.factorBase());
        assertEquals(layout.forecastErrorDiffuseCovBase(), layout.covarianceWorkBase());
        assertEquals(layout.solvedErrorBase(), layout.selectedStandardizedBase());
        assertEquals(layout.designBase() + 3 * 2, layout.obsCovBase());
        assertEquals(-1, layout.transitionBase());
        assertEquals(-1, layout.stateInterceptBase());
        assertEquals(-1, layout.selectionBase());
        assertEquals(-1, layout.stateCovBase());
        assertEquals(61, layout.totalLength());
    }

    @Test
    void collapsedScratchLayoutSkipsInactiveReconstructionFamilies() {
        CollapsedScratchLayout likelihoodOnly = CollapsedScratchLayout.create(8, 2, 2, 5,
            false, false, false, false, false);
        CollapsedScratchLayout forecastOnly = CollapsedScratchLayout.create(8, 2, 2, 5,
            true, true, false, false, false);
        CollapsedScratchLayout full = CollapsedScratchLayout.create(8, 2, 2, 5);

        assertEquals(-1, likelihoodOnly.forecastMeanBase());
        assertEquals(-1, likelihoodOnly.forecastErrorCovBase());
        assertEquals(-1, likelihoodOnly.zPBase());
        assertEquals(-1, forecastOnly.zPBase());
        assertEquals(-1, forecastOnly.selectedCovarianceBase());
        assertTrue(likelihoodOnly.totalLength() < full.totalLength());
        assertTrue(forecastOnly.totalLength() < full.totalLength());
    }

    @Test
    void collapsedIntScratchLayoutPacksSelectedIndexAndPivots() {
        CollapsedIntScratchLayout layout = CollapsedIntScratchLayout.create(4);

        assertEquals(0, layout.selectedIndexBase());
        assertEquals(4, layout.pivotsBase());
        assertEquals(8, layout.totalLength());
    }

    @Test
    void chandrasekharScratchLayoutDocumentsSafeAliases() {
        ChandrasekharScratchLayout layout = ChandrasekharScratchLayout.create(2, 3, true, 4);

        assertEquals(layout.solvedDesignPredictedBase(), layout.factorMWBase());
        assertEquals(layout.nextCovBase(), layout.transitionMinusKZBase());
        assertTrue(layout.luInverseWorkBase() >= 0);
    }

    @Test
    void smootherScratchLayoutAliasesTransitionGainAndLateObservationCovarianceWork() {
        SmootherScratchLayout real = SmootherScratchLayout.create(2, 3,
            true, true, true, true, true, false, false);
        ZSmootherScratchLayout complex = ZSmootherScratchLayout.create(2, 3, 3,
            true, true, true, true, true, false, false);

        assertEquals(real.tmpTkBase(), real.tmpFiHBase());
        assertEquals(complex.tmpTkBase(), complex.tmpFiHBase());
    }
}
