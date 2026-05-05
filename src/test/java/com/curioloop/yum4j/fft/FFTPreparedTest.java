package com.curioloop.yum4j.fft;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FFTPreparedTest {
    @Test
    void preparedOneDimensionalTransformsMatchConvenienceApi() {
        double[] complexInput = complexSample(77);
        double[] expectedComplex = complexInput.clone();
        Transform.c2c(expectedComplex, true, 0.25);
        Transform.PreparedComplex complex = Transform.prepareComplex(77);
        double[] actualComplex = complexInput.clone();
        complex.c2c(actualComplex, true, 0.25, complex.createWorkspace());
        assertArrayEquals(expectedComplex, actualComplex, 1e-10);

        double[] realInput = realSample(77);
        Transform.PreparedReal real = Transform.prepareReal(77);
        FftWorkspace realWorkspace = real.createWorkspace();
        double[] expectedSpectrum = new double[2 * (77 / 2 + 1)];
        double[] actualSpectrum = new double[expectedSpectrum.length];
        double[] realInputCopy1 = realInput.clone();
        Transform.r2c(realInputCopy1, expectedSpectrum, true, 1.0);
        real.r2c(realInput, actualSpectrum, true, 1.0, realWorkspace);
        assertArrayEquals(expectedSpectrum, actualSpectrum, 1e-10);

        double[] expectedRestored = new double[77];
        double[] actualRestored = new double[77];
        Transform.c2r(expectedSpectrum, expectedRestored, false, 1.0 / 77.0);
        real.c2r(actualSpectrum, actualRestored, false, 1.0 / 77.0, realWorkspace);
        assertArrayEquals(expectedRestored, actualRestored, 1e-10);

        double[] expectedFftpack = realInput.clone();
        double[] actualFftpack = realInput.clone();
        Transform.r2rFftpack(expectedFftpack, true, false, 0.5);
        real.r2rFftpack(actualFftpack, true, false, 0.5, realWorkspace);
        assertArrayEquals(expectedFftpack, actualFftpack, 1e-10);

        double[] expectedHartley = realInput.clone();
        Transform.r2rSeparableHartley(expectedHartley, 0.75);
        double[] actualHartley = new double[77];
        real.separableHartley(realInput, actualHartley, false, 0.75, realWorkspace);
        assertArrayEquals(expectedHartley, actualHartley, 1e-10);

        Transform.PreparedDcst dct = Transform.prepareDct(17, 4);
        double[] dctInput = Arrays.copyOf(realInput, 17);
        double[] expectedDct = dctInput.clone();
        Transform.dct(expectedDct, 4, 0.5, false);
        double[] actualDct = dctInput.clone();
        dct.exec(actualDct, 0.5, false, dct.createWorkspace());
        assertArrayEquals(expectedDct, actualDct, 1e-11);

        Transform.PreparedDcst dst = Transform.prepareDst(17, 3);
        double[] dstInput = Arrays.copyOf(realInput, 17);
        double[] expectedDst = dstInput.clone();
        Transform.dst(expectedDst, 3, 0.25, true);
        double[] actualDst = dstInput.clone();
        dst.exec(actualDst, 0.25, true, dst.createWorkspace());
        assertArrayEquals(expectedDst, actualDst, 1e-11);
    }

    @Test
    void preparedRealDirectStagingMatchesConvenienceApiAcrossRoutes() {
        for (int length : new int[] {1, 64, 96, 257}) {
            Transform.PreparedReal real = Transform.prepareReal(length);
            FftWorkspace workspace = real.createWorkspace();
            for (boolean forward : new boolean[] {true, false}) {
                double[] input = realSample(length);
                double[] originalInput = input.clone();
                double[] expectedSpectrum = new double[2 * (length / 2 + 1)];
                double[] actualSpectrum = new double[expectedSpectrum.length];
                double[] convenienceInput = input.clone();

                Transform.r2c(convenienceInput, expectedSpectrum, forward, 0.5);
                real.r2c(input, actualSpectrum, forward, 0.5, workspace);

                assertArrayEquals(originalInput, input, 0.0,
                        "prepared r2c modified input for length " + length);
                assertArrayEquals(expectedSpectrum, actualSpectrum, length * 8e-12,
                        "prepared r2c mismatch for length " + length);

                double[] expectedRestored = new double[length];
                double[] actualRestored = new double[length];
                Transform.c2r(expectedSpectrum.clone(), expectedRestored, !forward, 1.0 / length);
                real.c2r(actualSpectrum.clone(), actualRestored, !forward, 1.0 / length, workspace);

                assertArrayEquals(expectedRestored, actualRestored, length * 8e-12,
                        "prepared c2r mismatch for length " + length);
            }
        }
    }

    @Test
    void preparedNdTransformsMatchConvenienceApi() {
        int[] shape = {3, 4};
        int[] axes = {0, 1};
        long[] realStride = {4, 1};
        long[] complexStride = {8, 2};

        double[] complexInput = complexGrid(shape[0], shape[1]);
        double[] expectedComplex = new double[complexInput.length];
        double[] actualComplex = new double[complexInput.length];
        Transform.c2c(shape, complexStride, complexStride, axes, true, complexInput, expectedComplex, 0.5);
        Transform.PreparedNdComplex complex = Transform.prepareComplex(shape, axes);
        complex.c2c(complexStride, 0L, complexStride, 0L, true, complexInput, actualComplex, 0.5,
                complex.createWorkspace());
        assertArrayEquals(expectedComplex, actualComplex, 1e-11);

        double[] realInput = realGrid(shape[0], shape[1]);
        long[] hermitianStride = {6, 2};
        double[] expectedSpectrum = new double[shape[0] * 2 * (shape[1] / 2 + 1)];
        double[] actualSpectrum = new double[expectedSpectrum.length];
        Transform.r2c(shape, realStride, hermitianStride, axes, true, realInput, expectedSpectrum, 1.0);
        Transform.PreparedNdReal real = Transform.prepareReal(shape, axes);
        FftWorkspace realWorkspace = real.createWorkspace();
        real.r2c(realStride, 0L, hermitianStride, 0L, true, realInput, actualSpectrum, 1.0, realWorkspace);
        assertArrayEquals(expectedSpectrum, actualSpectrum, 1e-11);

        double[] expectedRestored = new double[realInput.length];
        double[] actualRestored = new double[realInput.length];
        Transform.c2r(shape, hermitianStride, realStride, axes, false, expectedSpectrum, expectedRestored,
                1.0 / realInput.length);
        real.c2r(hermitianStride, 0L, realStride, 0L, false, actualSpectrum, actualRestored,
                1.0 / realInput.length, realWorkspace);
        assertArrayEquals(expectedRestored, actualRestored, 1e-11);

        double[] expectedHartley = new double[realInput.length];
        double[] actualHartley = new double[realInput.length];
        Transform.r2rGenuineHartley(shape, realStride, realStride, axes, realInput, expectedHartley, 0.75);
        real.genuineHartley(realStride, 0L, realStride, 0L, false, realInput, actualHartley, 0.75,
                realWorkspace);
        assertArrayEquals(expectedHartley, actualHartley, 1e-11);

        Transform.PreparedNdDcst dct = Transform.prepareDct(shape, new int[] {1}, 2);
        double[] expectedDct = new double[realInput.length];
        double[] actualDct = new double[realInput.length];
        Transform.dct(shape, realStride, realStride, new int[] {1}, 2, realInput, expectedDct, 0.25, false);
        dct.exec(realStride, 0L, realStride, 0L, false, realInput, actualDct, 0.25, dct.createWorkspace());
        assertArrayEquals(expectedDct, actualDct, 1e-12);
    }

    @Test
    void preparedMetadataAccessorsExposeDefensiveCopiesAndWorkspaceRequirements() {
        Transform.PreparedComplex complex = Transform.prepareComplex(8);
        assertEquals(8, complex.length());
        assertTrue(complex.workspaceRequirement().totalDoubles() >= 0);

        Transform.PreparedReal real = Transform.prepareReal(9);
        assertEquals(9, real.length());
        assertEquals(9, real.workspaceRequirement().totalDoubles());
        assertEquals(32, Transform.prepareReal(64).workspaceRequirement().totalDoubles());

        Transform.PreparedDcst dcst = Transform.prepareDct(7, 2);
        assertEquals(7, dcst.length());
        assertTrue(dcst.workspaceRequirement().totalDoubles() >= 0);

        int[] shape = {2, 3, 4};
        int[] axes = {0, 2};
        Transform.PreparedNdComplex ndComplex = Transform.prepareComplex(shape, axes);
        Transform.PreparedNdReal ndReal = Transform.prepareReal(shape, axes);
        Transform.PreparedNdDcst ndDcst = Transform.prepareDst(shape, axes, 4);

        assertArrayEquals(shape, ndComplex.shape());
        assertArrayEquals(axes, ndComplex.axes());
        assertTrue(ndComplex.workspaceRequirement().totalDoubles() > 0);
        assertArrayEquals(shape, ndReal.shape());
        assertArrayEquals(axes, ndReal.axes());
        assertTrue(ndReal.workspaceRequirement().totalDoubles() > 0);
        assertArrayEquals(shape, ndDcst.shape());
        assertArrayEquals(axes, ndDcst.axes());
        assertTrue(ndDcst.workspaceRequirement().totalDoubles() > 0);

        shape[0] = 99;
        axes[0] = 1;
        assertArrayEquals(new int[] {2, 3, 4}, ndComplex.shape());
        assertArrayEquals(new int[] {0, 2}, ndComplex.axes());
    }

    @Test
    void preparedRealHartleyRoutesMatchConvenienceWrappers() {
        double[] realInput = realSample(9);
        Transform.PreparedReal real = Transform.prepareReal(9);
        FftWorkspace workspace = real.createWorkspace();

        double[] expectedGenuineHartley = realInput.clone();
        Transform.r2rGenuineHartley(expectedGenuineHartley, 0.5);
        double[] actualGenuineHartley = new double[realInput.length];
        real.genuineHartley(realInput, actualGenuineHartley, false, 0.5, workspace);
        assertArrayEquals(expectedGenuineHartley, actualGenuineHartley, 1e-12);

        double[] expectedGenuineFht = realInput.clone();
        Transform.r2rGenuineFht(expectedGenuineFht, 0.75);
        double[] actualGenuineFht = new double[realInput.length];
        real.genuineHartley(realInput, actualGenuineFht, true, 0.75, workspace);
        assertArrayEquals(expectedGenuineFht, actualGenuineFht, 1e-12);

        int[] shape = {3, 4};
        int[] axes = {0};
        long[] stride = {4, 1};
        double[] grid = realGrid(shape[0], shape[1]);
        Transform.PreparedNdReal ndReal = Transform.prepareReal(shape, axes);
        double[] expectedSeparableFht = new double[grid.length];
        double[] actualSeparableFht = new double[grid.length];
        Transform.r2rSeparableFht(shape, stride, stride, axes, grid, expectedSeparableFht, 1.25);
        ndReal.separableHartley(stride, 0L, stride, 0L, true, grid, actualSeparableFht, 1.25,
                ndReal.createWorkspace());
        assertArrayEquals(expectedSeparableFht, actualSeparableFht, 1e-12);
    }

    @Test
    void preparedNdComplexBluesteinAxisUsesReservedWorkspaceLane() {
        int[] shape = {2, 257};
        int[] axes = {1};
        long[] stride = {2L * shape[1], 2L};
        double[] input = complexGrid(shape[0], shape[1]);
        double[] expected = new double[input.length];
        double[] actual = new double[input.length];

        for (int row = 0; row < shape[0]; row++) {
            double[] line = Arrays.copyOfRange(input, row * 2 * shape[1], (row + 1) * 2 * shape[1]);
            Transform.c2c(line, true, 0.5);
            System.arraycopy(line, 0, expected, row * 2 * shape[1], line.length);
        }

        Transform.PreparedNdComplex prepared = Transform.prepareComplex(shape, axes);
        prepared.c2c(stride, 0L, stride, 0L, true, input, actual, 0.5, prepared.createWorkspace());

        assertArrayEquals(expected, actual, 1e-10);
    }

    @Test
    void preparedNdComplexCanUseCompactWorkspaceForKnownStandardLayout() {
        int[] shape = {4, 64};
        int[] axes = {0, 1};
        long[] stride = {2L * shape[1], 2L};
        double[] input = complexGrid(shape[0], shape[1]);

        Transform.PreparedNdComplex prepared = Transform.prepareComplex(shape, axes);
        int generalDoubles = prepared.workspaceRequirement().totalDoubles();
        int compactDoubles = prepared.workspaceRequirement(stride, 0L, stride, 0L, input, input).totalDoubles();
        assertTrue(compactDoubles < generalDoubles);

        double[] expected = input.clone();
        double[] actual = input.clone();
        prepared.c2c(stride, 0L, stride, 0L, true, expected, expected, 0.5, prepared.createWorkspace());
        prepared.c2c(stride, 0L, stride, 0L, true, actual, actual, 0.5,
                prepared.createWorkspace(stride, 0L, stride, 0L, actual, actual));

        assertArrayEquals(expected, actual, 1e-10);
    }

    @Test
    void preparedNdComplexUsesBatchedTwoDimensionalFastPathForStandardLayout() {
        int[] shape = {2, 4, 64};
        int[] axes = {1, 2};
        int rows = shape[1];
        int columns = shape[2];
        long[] stride = {2L * rows * columns, 2L * columns, 2L};
        double[] input = complexGrid(shape[0] * rows, columns);
        double[] expected = input.clone();
        double[] actual = input.clone();

        Transform.PreparedNdComplex prepared = Transform.prepareComplex(shape, axes);
        int compactDoubles = prepared.workspaceRequirement(stride, 0L, stride, 0L, actual, actual).totalDoubles();
        assertTrue(compactDoubles < prepared.workspaceRequirement().totalDoubles());

        for (int batch = 0; batch < shape[0]; batch++) {
            int planeOffset = batch * 2 * rows * columns;
            Transform.c2c(new int[] {rows, columns}, new long[] {2L * columns, 2L}, planeOffset,
                    new long[] {2L * columns, 2L}, planeOffset, new int[] {0, 1}, true,
                    expected, expected, 0.5);
        }

        prepared.c2c(stride, 0L, stride, 0L, true, actual, actual, 0.5,
                prepared.createWorkspace(stride, 0L, stride, 0L, actual, actual));

        assertArrayEquals(expected, actual, 1e-10);
    }

    @Test
    void preparedNdRealCanUseCompactWorkspaceForKnownStandardLayout() {
        int[] shape = {4, 64};
        int[] axes = {0, 1};
        long[] realStride = {shape[1], 1L};
        long[] hermitianStride = {2L * (shape[1] / 2 + 1), 2L};
        double[] input = realGrid(shape[0], shape[1]);
        double[] expectedSpectrum = new double[shape[0] * 2 * (shape[1] / 2 + 1)];
        double[] actualSpectrum = new double[expectedSpectrum.length];
        double[] expectedRestored = new double[input.length];
        double[] actualRestored = new double[input.length];

        Transform.PreparedNdReal prepared = Transform.prepareReal(shape, axes);
        int generalDoubles = prepared.workspaceRequirement().totalDoubles();
        int r2cCompactDoubles = prepared.r2cWorkspaceRequirement(realStride, 0L, hermitianStride, 0L, input,
                actualSpectrum).totalDoubles();
        int c2rCompactDoubles = prepared.c2rWorkspaceRequirement(hermitianStride, 0L, realStride, 0L,
                actualSpectrum, actualRestored).totalDoubles();
        assertTrue(r2cCompactDoubles < generalDoubles);
        assertTrue(c2rCompactDoubles < generalDoubles);

        prepared.r2c(realStride, 0L, hermitianStride, 0L, true, input, expectedSpectrum, 1.0,
                prepared.createWorkspace());
        prepared.r2c(realStride, 0L, hermitianStride, 0L, true, input, actualSpectrum, 1.0,
                prepared.createR2cWorkspace(realStride, 0L, hermitianStride, 0L, input, actualSpectrum));
        assertArrayEquals(expectedSpectrum, actualSpectrum, 1e-10);

        prepared.c2r(hermitianStride, 0L, realStride, 0L, false, expectedSpectrum, expectedRestored,
                1.0 / input.length, prepared.createWorkspace());
        prepared.c2r(hermitianStride, 0L, realStride, 0L, false, actualSpectrum, actualRestored,
                1.0 / input.length,
                prepared.createC2rWorkspace(hermitianStride, 0L, realStride, 0L, actualSpectrum, actualRestored));
        assertArrayEquals(expectedRestored, actualRestored, 1e-10);
    }

    @Test
    void preparedNdRealUsesBatchedTwoDimensionalFastPathsForStandardLayout() {
        int[] shape = {2, 4, 64};
        int[] axes = {1, 2};
        int batches = shape[0];
        int rows = shape[1];
        int columns = shape[2];
        int hermitianColumns = columns / 2 + 1;
        long[] realStride = {rows * columns, columns, 1L};
        long[] hermitianStride = {rows * 2L * hermitianColumns, 2L * hermitianColumns, 2L};
        double[] input = realGrid(batches * rows, columns);
        double[] expectedSpectrum = new double[batches * rows * 2 * hermitianColumns];
        double[] actualSpectrum = new double[expectedSpectrum.length];
        double[] expectedRestored = new double[input.length];
        double[] actualRestored = new double[input.length];

        Transform.PreparedNdReal prepared = Transform.prepareReal(shape, axes);
        int r2cCompactDoubles = prepared.r2cWorkspaceRequirement(realStride, 0L, hermitianStride, 0L, input,
                actualSpectrum).totalDoubles();
        int c2rCompactDoubles = prepared.c2rWorkspaceRequirement(hermitianStride, 0L, realStride, 0L,
                actualSpectrum, actualRestored).totalDoubles();
        assertTrue(r2cCompactDoubles < prepared.workspaceRequirement().totalDoubles());
        assertTrue(c2rCompactDoubles < prepared.workspaceRequirement().totalDoubles());

        for (int batch = 0; batch < batches; batch++) {
            int realOffset = batch * rows * columns;
            int spectrumOffset = batch * rows * 2 * hermitianColumns;
            Transform.r2c(new int[] {rows, columns}, new long[] {columns, 1L}, realOffset,
                    new long[] {2L * hermitianColumns, 2L}, spectrumOffset, new int[] {0, 1}, true,
                    input, expectedSpectrum, 1.0);
        }

        prepared.r2c(realStride, 0L, hermitianStride, 0L, true, input, actualSpectrum, 1.0,
                prepared.createR2cWorkspace(realStride, 0L, hermitianStride, 0L, input, actualSpectrum));
        assertArrayEquals(expectedSpectrum, actualSpectrum, 1e-10);

        for (int batch = 0; batch < batches; batch++) {
            int realOffset = batch * rows * columns;
            int spectrumOffset = batch * rows * 2 * hermitianColumns;
            Transform.c2r(new int[] {rows, columns}, new long[] {2L * hermitianColumns, 2L}, spectrumOffset,
                    new long[] {columns, 1L}, realOffset, new int[] {0, 1}, false, expectedSpectrum,
                    expectedRestored, 1.0 / (rows * columns));
        }

        prepared.c2r(hermitianStride, 0L, realStride, 0L, false, actualSpectrum, actualRestored,
                1.0 / (rows * columns),
                prepared.createC2rWorkspace(hermitianStride, 0L, realStride, 0L, actualSpectrum, actualRestored));

        assertArrayEquals(expectedRestored, actualRestored, 1e-10);
    }

    @Test
    void preparedNdRealFftpackBluesteinAxisUsesReservedWorkspaceLane() {
        int[] shape = {2, 257};
        int[] axes = {1};
        long[] stride = {shape[1], 1L};
        double[] input = realGrid(shape[0], shape[1]);
        double[] expected = new double[input.length];
        double[] actual = new double[input.length];

        for (int row = 0; row < shape[0]; row++) {
            double[] line = Arrays.copyOfRange(input, row * shape[1], (row + 1) * shape[1]);
            Transform.r2rFftpack(line, true, true, 0.5);
            System.arraycopy(line, 0, expected, row * shape[1], line.length);
        }

        Transform.PreparedNdReal prepared = Transform.prepareReal(shape, axes);
        prepared.r2rFftpack(stride, 0L, stride, 0L, true, true, input, actual, 0.5,
                prepared.createWorkspace());

        assertArrayEquals(expected, actual, 1e-10);
    }

    @Test
    void genericRealAxisCanUseContiguousOutputAsWorkspace() {
        int[] shape = {3, 64};
        int[] axes = {1};
        long[] stride = {shape[1], 1L};
        double[] input = realGrid(shape[0], shape[1]);
        double[] expected = new double[input.length];
        double[] actual = new double[input.length];

        for (int row = 0; row < shape[0]; row++) {
            double[] line = Arrays.copyOfRange(input, row * shape[1], (row + 1) * shape[1]);
            Transform.r2rFftpack(line, true, true, 0.5);
            System.arraycopy(line, 0, expected, row * shape[1], line.length);
        }

        FftWorkspace.Requirement general = FftNd.realAxisWorkspaceRequirement(shape, axes);
        FftWorkspace.Requirement compact = FftNd.realAxisWorkspaceRequirement(shape, stride, axes);
        assertTrue(compact.totalDoubles() < general.totalDoubles());
        assertEquals(0, compact.totalDoubles());

        Transform.PreparedNdReal prepared = Transform.prepareReal(shape, axes);
        prepared.r2rFftpack(stride, 0L, stride, 0L, true, true, input, actual, 0.5, compact.allocate());

        assertArrayEquals(expected, actual, 1e-10);
    }

    @Test
    void genericHartleyAxisUsesOverlapSafeContiguousOutputWorkspace() {
        int[] shape = {3, 8};
        int[] axes = {1};
        long[] stride = {shape[1], 1L};
        double[] input = realGrid(shape[0], shape[1]);
        Transform.PreparedNdReal prepared = Transform.prepareReal(shape, axes);

        FftWorkspace.Requirement general = FftNd.realAxisWorkspaceRequirement(shape, axes);
        FftWorkspace.Requirement compact = FftNd.separableHartleyWorkspaceRequirement(shape, stride, axes);
        assertTrue(compact.totalDoubles() < general.totalDoubles());
        assertEquals(shape[1] / 2, compact.totalDoubles());

        for (boolean fht : new boolean[] {false, true}) {
            double[] expected = new double[input.length];
            double[] actual = new double[input.length];
            for (int row = 0; row < shape[0]; row++) {
                double[] line = Arrays.copyOfRange(input, row * shape[1], (row + 1) * shape[1]);
                if (fht) {
                    Transform.r2rSeparableFht(line, 0.75);
                } else {
                    Transform.r2rSeparableHartley(line, 0.75);
                }
                System.arraycopy(line, 0, expected, row * shape[1], line.length);
            }

            prepared.separableHartley(stride, 0L, stride, 0L, fht, input, actual, 0.75, compact.allocate());

            assertArrayEquals(expected, actual, 1e-10);
        }
    }

    @Test
    void genericDcstAxisCanUseContiguousOutputAsWorkspace() {
        int[] shape = {3, 64};
        int[] axes = {1};
        long[] stride = {shape[1], 1L};
        double[] input = realGrid(shape[0], shape[1]);
        double[] expected = new double[input.length];
        double[] actual = new double[input.length];

        for (int row = 0; row < shape[0]; row++) {
            double[] line = Arrays.copyOfRange(input, row * shape[1], (row + 1) * shape[1]);
            Transform.dct(line, 2, 0.5, false);
            System.arraycopy(line, 0, expected, row * shape[1], line.length);
        }

        Transform.PreparedNdDcst prepared = Transform.prepareDct(shape, axes, 2);
        FftWorkspace.Requirement compact = FftNd.dcstWorkspaceRequirement(shape, stride, axes, 2, true);
        assertTrue(compact.totalDoubles() < prepared.workspaceRequirement().totalDoubles());
        assertEquals(0, compact.totalDoubles());

        prepared.exec(stride, 0L, stride, 0L, false, input, actual, 0.5, compact.allocate());

        assertArrayEquals(expected, actual, 1e-10);
    }

    @Test
    void preparedNdDcstSupportsOffsetsStridesAndWorkspaceReuse() {
        int[] shape = {4, 3};
        int[] axes = {0};
        long[] strideIn = {5, 1};
        long[] strideOut = {7, 2};
        long offsetIn = 2;
        long offsetOut = 4;
        double[] input = filled(32, Double.NaN);
        double[] expected = filled(40, -9999.0);
        double[] actual = filled(40, -9999.0);

        for (int row = 0; row < shape[0]; row++) {
            for (int column = 0; column < shape[1]; column++) {
                input[(int) (offsetIn + row * strideIn[0] + column * strideIn[1])] =
                        0.25 + 0.5 * row - 0.125 * column;
            }
        }

        Transform.PreparedNdDcst prepared = Transform.prepareDst(shape, axes, 3);
        FftWorkspace workspace = prepared.createWorkspace();
        Transform.dst(shape, strideIn, offsetIn, strideOut, offsetOut, axes, 3, input, expected, 0.5, false);
        prepared.exec(strideIn, offsetIn, strideOut, offsetOut, false, input, actual, 0.5, workspace);
        assertArrayEquals(expected, actual, 1e-12);

        Arrays.fill(expected, -9999.0);
        Arrays.fill(actual, -9999.0);
        Transform.dst(shape, strideIn, offsetIn, strideOut, offsetOut, axes, 3, input, expected, 0.25, false);
        prepared.exec(strideIn, offsetIn, strideOut, offsetOut, false, input, actual, 0.25, workspace);
        assertArrayEquals(expected, actual, 1e-12);
    }

    @Test
    void preparedApiRejectsUndersizedWorkspace() {
        Transform.PreparedReal prepared = Transform.prepareReal(12);
        FftWorkspace tooSmall = FftWorkspace.Requirement.empty().allocate();

        assertThrows(IllegalArgumentException.class,
                () -> prepared.r2c(realSample(12), new double[14], true, 1.0, tooSmall));
    }

    @Test
    void planCacheControlsBoundOnlyImmutablePlans() {
        int previousCapacity = FftPlanCache.maxEntries();
        try {
            FftPlanCache.clear();
            FftPlanCache.setMaxEntries(2);

            Transform.prepareComplex(12);
            Transform.prepareReal(12);
            Transform.prepareDct(12, 2);

            assertTrue(FftPlanCache.size() <= 2);
            assertThrows(IllegalArgumentException.class, () -> FftPlanCache.setMaxEntries(-1));
        } finally {
            FftPlanCache.setMaxEntries(previousCapacity);
            FftPlanCache.clear();
        }
    }

    @Test
    void preparedPlanCanBeSharedAcrossThreadsWithSeparateWorkspaces() throws Exception {
        Transform.PreparedComplex prepared = Transform.prepareComplex(77);
        double[] input = complexSample(77);
        double[] expected = input.clone();
        Transform.c2c(expected, true, 1.0);

        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Callable<double[]> task = () -> {
                double[] data = input.clone();
                prepared.c2c(data, true, 1.0, prepared.createWorkspace());
                return data;
            };
            List<Future<double[]>> results = executor.invokeAll(List.of(task, task));
            for (Future<double[]> result : results) {
                assertArrayEquals(expected, result.get(), 1e-10);
            }
        } finally {
            executor.shutdownNow();
        }
    }

    private static double[] complexSample(int size) {
        double[] data = new double[2 * size];
        for (int index = 0; index < size; index++) {
            data[2 * index] = 0.25 + 0.125 * ((11 * index + 3) % 19);
            data[2 * index + 1] = -0.5 + 0.0625 * ((7 * index + 5) % 23);
        }
        return data;
    }

    private static double[] realSample(int size) {
        double[] data = new double[size];
        for (int index = 0; index < size; index++) {
            data[index] = 0.25 + 0.125 * ((13 * index + 7) % 17) - 0.03125 * (index % 5);
        }
        return data;
    }

    private static double[] filled(int size, double value) {
        double[] data = new double[size];
        Arrays.fill(data, value);
        return data;
    }

    private static double[] complexGrid(int rows, int columns) {
        double[] data = new double[2 * rows * columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                int offset = 2 * (row * columns + column);
                data[offset] = 0.25 + 0.5 * row + 0.125 * ((7 * column + row) % 11);
                data[offset + 1] = -0.375 + 0.0625 * ((5 * row + 3 * column) % 13);
            }
        }
        return data;
    }

    private static double[] realGrid(int rows, int columns) {
        double[] data = new double[rows * columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                data[row * columns + column] = 0.25 + 0.5 * row - 0.125 * column
                        + 0.03125 * ((row + 3 * column) % 5);
            }
        }
        return data;
    }
}
