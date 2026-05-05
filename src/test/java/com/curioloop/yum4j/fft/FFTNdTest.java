package com.curioloop.yum4j.fft;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FFTNdTest {
    @Test
    void c2cTwoDimensionalMatchesDirectOracleAndScalesOnce() {
        int[] shape = {3, 4};
        long[] stride = {8, 2};
        double[] input = complexGrid(shape[0], shape[1]);
        double[] output = new double[input.length];

        Transform.c2c(shape, stride, stride, new int[] {0, 1}, true, input, output, 0.5);

        assertArrayEquals(directComplex2d(input, shape[0], shape[1], true, 0.5), output, 1e-11);
    }

    @Test
    void c2cFastInPlaceTwoDimensionalRouteMatchesDirectOracle() {
        int[] shape = {3, 5};
        long[] stride = {10, 2};
        int[] axes = {0, 1};

        double[] forward = complexGrid(shape[0], shape[1]);
        double[] expectedForward = directComplex2d(forward, shape[0], shape[1], true, 0.25);
        Transform.c2c(shape, stride, 0L, stride, 0L, axes, true, forward, forward, 0.25);
        assertArrayEquals(expectedForward, forward, 1e-11);

        double[] backward = complexGrid(shape[0], shape[1]);
        double[] expectedBackward = directComplex2d(backward, shape[0], shape[1], false, 0.375);
        Transform.c2c(shape, stride, 0L, stride, 0L, axes, false, backward, backward, 0.375);
        assertArrayEquals(expectedBackward, backward, 1e-11);
    }

    @Test
    void c2cFastColumnPassScalesEveryColumnInBatchAndRemainder() {
        int[] shape = {3, 5};
        long[] stride = {10, 2};
        double[] data = complexGrid(shape[0], shape[1]);
        double[] expected = data.clone();

        for (int column = 0; column < shape[1]; column++) {
            double[] line = new double[2 * shape[0]];
            for (int row = 0; row < shape[0]; row++) {
                int source = row * (int) stride[0] + column * (int) stride[1];
                line[2 * row] = data[source];
                line[2 * row + 1] = data[source + 1];
            }
            double[] transformed = directComplex1d(line, true, 0.5);
            for (int row = 0; row < shape[0]; row++) {
                int target = row * (int) stride[0] + column * (int) stride[1];
                expected[target] = transformed[2 * row];
                expected[target + 1] = transformed[2 * row + 1];
            }
        }

        Transform.c2c(shape, stride, 0L, stride, 0L, new int[] {0}, true, data, data, 0.5);

        assertArrayEquals(expected, data, 1e-12);
    }

    @Test
    void c2cSupportsPrimitiveSlotStridesAndOffsets() {
        int[] shape = {2, 3};
        long[] strideIn = {10, 2};
        long[] strideOut = {12, 2};
        double[] input = filled(40, Double.NaN);
        double[] output = filled(40, -9999.0);
        long offsetIn = 4;
        long offsetOut = 5;

        for (int row = 0; row < shape[0]; row++) {
            for (int column = 0; column < shape[1]; column++) {
                int index = (int) (offsetIn + row * strideIn[0] + column * strideIn[1]);
                input[index] = 0.25 + row + 0.125 * column;
                input[index + 1] = -0.5 + 0.25 * row - 0.0625 * column;
            }
        }

        Transform.c2c(shape, strideIn, offsetIn, strideOut, offsetOut, new int[] {1}, true, input, output, 1.0);

        for (int row = 0; row < shape[0]; row++) {
            double[] rowInput = new double[2 * shape[1]];
            for (int column = 0; column < shape[1]; column++) {
                int index = (int) (offsetIn + row * strideIn[0] + column * strideIn[1]);
                rowInput[2 * column] = input[index];
                rowInput[2 * column + 1] = input[index + 1];
            }
            double[] expected = directComplex1d(rowInput, true, 1.0);
            for (int column = 0; column < shape[1]; column++) {
                int index = (int) (offsetOut + row * strideOut[0] + column * strideOut[1]);
                assertEquals(expected[2 * column], output[index], 1e-12);
                assertEquals(expected[2 * column + 1], output[index + 1], 1e-12);
            }
        }
        assertEquals(-9999.0, output[0], 0.0);
        assertEquals(-9999.0, output[39], 0.0);
    }

    @Test
    void c2cContiguousOutputAxisSupportsStridedInputWithOffsets() {
        int[] shape = {2, 3};
        long[] strideIn = {16, 4};
        long[] strideOut = {6, 2};
        double[] input = filled(64, Double.NaN);
        double[] output = filled(40, -9999.0);
        long offsetIn = 5;
        long offsetOut = 4;

        for (int row = 0; row < shape[0]; row++) {
            for (int column = 0; column < shape[1]; column++) {
                int index = (int) (offsetIn + row * strideIn[0] + column * strideIn[1]);
                input[index] = 0.5 + 0.25 * row + 0.125 * column;
                input[index + 1] = -0.25 + 0.0625 * row - 0.03125 * column;
            }
        }

        Transform.c2c(shape, strideIn, offsetIn, strideOut, offsetOut, new int[] {1}, true, input, output, 1.0);

        for (int row = 0; row < shape[0]; row++) {
            double[] rowInput = new double[2 * shape[1]];
            for (int column = 0; column < shape[1]; column++) {
                int index = (int) (offsetIn + row * strideIn[0] + column * strideIn[1]);
                rowInput[2 * column] = input[index];
                rowInput[2 * column + 1] = input[index + 1];
            }
            double[] expected = directComplex1d(rowInput, true, 1.0);
            for (int column = 0; column < shape[1]; column++) {
                int index = (int) (offsetOut + row * strideOut[0] + column * strideOut[1]);
                assertEquals(expected[2 * column], output[index], 1e-12);
                assertEquals(expected[2 * column + 1], output[index + 1], 1e-12);
            }
        }
        assertEquals(-9999.0, output[0], 0.0);
        assertEquals(-9999.0, output[39], 0.0);
    }

    @Test
    void c2cStridedOutputAxisSupportsTemporaryCopyOutputWithOffsets() {
        int[] shape = {3, 2};
        long[] strideIn = {4, 2};
        long[] strideOut = {8, 2};
        double[] input = filled(40, Double.NaN);
        double[] output = filled(60, -9999.0);
        long offsetIn = 6;
        long offsetOut = 5;

        for (int row = 0; row < shape[0]; row++) {
            for (int column = 0; column < shape[1]; column++) {
                int index = (int) (offsetIn + row * strideIn[0] + column * strideIn[1]);
                input[index] = 0.125 + 0.5 * row - 0.25 * column;
                input[index + 1] = -0.375 + 0.125 * row + 0.0625 * column;
            }
        }

        Transform.c2c(shape, strideIn, offsetIn, strideOut, offsetOut, new int[] {0}, true, input, output, 1.0);

        for (int column = 0; column < shape[1]; column++) {
            double[] columnInput = new double[2 * shape[0]];
            for (int row = 0; row < shape[0]; row++) {
                int index = (int) (offsetIn + row * strideIn[0] + column * strideIn[1]);
                columnInput[2 * row] = input[index];
                columnInput[2 * row + 1] = input[index + 1];
            }
            double[] expected = directComplex1d(columnInput, true, 1.0);
            for (int row = 0; row < shape[0]; row++) {
                int index = (int) (offsetOut + row * strideOut[0] + column * strideOut[1]);
                assertEquals(expected[2 * row], output[index], 1e-12);
                assertEquals(expected[2 * row + 1], output[index + 1], 1e-12);
            }
        }
        assertEquals(-9999.0, output[0], 0.0);
        assertEquals(-9999.0, output[59], 0.0);
    }

    @Test
    void r2rFftpackNdMatchesRowWiseFacade() {
        int[] shape = {3, 5};
        long[] stride = {5, 1};
        double[] input = realGrid(shape[0], shape[1]);
        double[] output = new double[input.length];
        double[] expected = new double[input.length];

        for (int row = 0; row < shape[0]; row++) {
            double[] line = Arrays.copyOfRange(input, row * shape[1], (row + 1) * shape[1]);
            Transform.r2rFftpack(line, true, true, 1.0);
            System.arraycopy(line, 0, expected, row * shape[1], shape[1]);
        }

        Transform.r2rFftpack(shape, stride, stride, new int[] {1}, true, true, input, output, 1.0);

        assertArrayEquals(expected, output, 1e-12);
    }

    @Test
    void r2rFftpackNdRoundTripsColumnsWithStrides() {
        int[] shape = {4, 3};
        long[] stride = {3, 1};
        double[] original = realGrid(shape[0], shape[1]);
        double[] data = original.clone();

        Transform.r2rFftpack(shape, stride, stride, new int[] {0}, true, true, data, data, 1.0);
        Transform.r2rFftpack(shape, stride, stride, new int[] {0}, false, false, data, data, 1.0 / shape[0]);

        assertArrayEquals(original, data, 1e-12);
    }

    @Test
    void r2rFftpackNdRoundTripsBackwardConventionColumnsWithStrides() {
        int[] shape = {5, 3};
        long[] stride = {3, 1};
        double[] original = realGrid(shape[0], shape[1]);
        double[] data = original.clone();

        Transform.r2rFftpack(shape, stride, stride, new int[] {0}, true, false, data, data, 1.0);
        Transform.r2rFftpack(shape, stride, stride, new int[] {0}, false, true, data, data, 1.0 / shape[0]);

        assertArrayEquals(original, data, 1e-12);
    }

    @Test
    void r2cNdSingleAxisMatchesRowWiseFacade() {
        int[] shape = {2, 5};
        long[] realStride = {5, 1};
        long[] complexStride = {6, 2};
        double[] input = realGrid(shape[0], shape[1]);
        double[] output = new double[shape[0] * 2 * (shape[1] / 2 + 1)];
        double[] expected = new double[output.length];

        for (int row = 0; row < shape[0]; row++) {
            double[] line = Arrays.copyOfRange(input, row * shape[1], (row + 1) * shape[1]);
            double[] spectrum = new double[2 * (shape[1] / 2 + 1)];
            Transform.r2c(line, spectrum, true, 1.0);
            System.arraycopy(spectrum, 0, expected, row * spectrum.length, spectrum.length);
        }

        Transform.r2c(shape, realStride, complexStride, 1, true, input, output, 1.0);

        assertArrayEquals(expected, output, 1e-12);
    }

    @Test
    void r2cNdSupportsStridesOffsetsOddLengthAndSentinels() {
        int[] shape = {2, 5};
        long[] realStride = {8, 1};
        long[] complexStride = {10, 2};
        long offsetIn = 3;
        long offsetOut = 4;
        double[] input = filled(32, Double.NaN);
        double[] output = filled(40, -9999.0);

        for (int row = 0; row < shape[0]; row++) {
            for (int column = 0; column < shape[1]; column++) {
                input[(int) (offsetIn + row * realStride[0] + column * realStride[1])] =
                        0.25 + 0.5 * row - 0.125 * column;
            }
        }

        Transform.r2c(shape, realStride, offsetIn, complexStride, offsetOut, new int[] {1}, false, input, output, 0.5);

        int complexColumns = shape[1] / 2 + 1;
        for (int row = 0; row < shape[0]; row++) {
            double[] line = new double[shape[1]];
            for (int column = 0; column < shape[1]; column++) {
                line[column] = input[(int) (offsetIn + row * realStride[0] + column * realStride[1])];
            }
            double[] expected = new double[2 * complexColumns];
            Transform.r2c(line, expected, false, 0.5);
            for (int frequency = 0; frequency < complexColumns; frequency++) {
                int actual = (int) (offsetOut + row * complexStride[0] + frequency * complexStride[1]);
                assertEquals(expected[2 * frequency], output[actual], 1e-12);
                assertEquals(expected[2 * frequency + 1], output[actual + 1], 1e-12);
            }
        }
        assertEquals(-9999.0, output[0], 0.0);
        assertEquals(-9999.0, output[39], 0.0);
    }

    @Test
    void r2cNdMultiAxisMatchesDirectOracle() {
        int[] shape = {3, 4};
        long[] realStride = {4, 1};
        long[] complexStride = {6, 2};
        double[] input = realGrid(shape[0], shape[1]);
        double[] output = new double[shape[0] * 2 * (shape[1] / 2 + 1)];

        Transform.r2c(shape, realStride, complexStride, new int[] {0, 1}, true, input, output, 1.0);

        assertArrayEquals(directRealToComplex2d(input, shape[0], shape[1], true, 1.0), output, 1e-11);
    }

    @Test
    void r2cNdBackwardMultiAxisMatchesDirectOracleAndScalesOnce() {
        int[] shape = {3, 5};
        long[] realStride = {5, 1};
        long[] complexStride = {6, 2};
        double[] input = realGrid(shape[0], shape[1]);
        double[] output = new double[shape[0] * 2 * (shape[1] / 2 + 1)];

        Transform.r2c(shape, realStride, complexStride, new int[] {0, 1}, false, input, output, 0.25);

        assertArrayEquals(directRealToComplex2d(input, shape[0], shape[1], false, 0.25), output, 1e-11);
    }

    @Test
    void r2cAndC2rNdRoundTripMultiAxis() {
        int[] shape = {3, 4};
        long[] realStride = {4, 1};
        long[] complexStride = {6, 2};
        double[] input = realGrid(shape[0], shape[1]);
        double[] spectrum = new double[shape[0] * 2 * (shape[1] / 2 + 1)];
        double[] restored = new double[input.length];

        Transform.r2c(shape, realStride, complexStride, new int[] {0, 1}, true, input, spectrum, 1.0);
        Transform.c2r(shape, complexStride, realStride, new int[] {0, 1}, false, spectrum, restored,
                1.0 / (shape[0] * shape[1]));

        assertArrayEquals(input, restored, 1e-12);
    }

    @Test
    void c2rNdSupportsStridesOffsetsEvenNyquistAndSentinels() {
        int[] shape = {2, 4};
        long[] complexStride = {10, 2};
        long[] realStride = {7, 1};
        long offsetIn = 5;
        long offsetOut = 3;
        double[] original = realGrid(shape[0], shape[1]);
        double[] spectrum = filled(40, Double.NaN);
        double[] output = filled(40, -9999.0);
        int complexColumns = shape[1] / 2 + 1;

        for (int row = 0; row < shape[0]; row++) {
            double[] line = Arrays.copyOfRange(original, row * shape[1], (row + 1) * shape[1]);
            double[] packed = new double[2 * complexColumns];
            Transform.r2c(line, packed, true, 1.0);
            for (int frequency = 0; frequency < complexColumns; frequency++) {
                int target = (int) (offsetIn + row * complexStride[0] + frequency * complexStride[1]);
                spectrum[target] = packed[2 * frequency];
                spectrum[target + 1] = packed[2 * frequency + 1];
            }
        }

        Transform.c2r(shape, complexStride, offsetIn, realStride, offsetOut, new int[] {1}, false, spectrum, output,
                1.0 / shape[1]);

        for (int row = 0; row < shape[0]; row++) {
            for (int column = 0; column < shape[1]; column++) {
                int actual = (int) (offsetOut + row * realStride[0] + column * realStride[1]);
                assertEquals(original[row * shape[1] + column], output[actual], 1e-12);
            }
        }
        assertEquals(-9999.0, output[0], 0.0);
        assertEquals(-9999.0, output[39], 0.0);
    }

    @Test
    void c2rNdForwardRoundTripMatchesBackwardR2cConventionAndScalesOnce() {
        int[] shape = {3, 5};
        long[] realStride = {5, 1};
        long[] complexStride = {6, 2};
        double[] input = realGrid(shape[0], shape[1]);
        double[] spectrum = new double[shape[0] * 2 * (shape[1] / 2 + 1)];
        double[] restored = new double[input.length];

        Transform.r2c(shape, realStride, complexStride, new int[] {0, 1}, false, input, spectrum, 1.0);
        Transform.c2r(shape, complexStride, realStride, new int[] {0, 1}, true, spectrum, restored,
                0.5 / (shape[0] * shape[1]));

        double[] expected = input.clone();
        for (int index = 0; index < expected.length; index++) {
            expected[index] *= 0.5;
        }
        assertArrayEquals(expected, restored, 1e-12);
    }

    @Test
    void singleAxisOffsetRealComplexWrappersMatchArrayAxisRoutes() {
        int[] shape = {2, 5};
        long[] realStride = {8, 1};
        long[] complexStride = {10, 2};
        long offsetIn = 3;
        long offsetOut = 4;
        double[] input = filled(32, Double.NaN);
        double[] expected = filled(40, -9999.0);
        double[] actual = filled(40, -9999.0);

        for (int row = 0; row < shape[0]; row++) {
            for (int column = 0; column < shape[1]; column++) {
                input[(int) (offsetIn + row * realStride[0] + column * realStride[1])] =
                        0.125 + 0.25 * row - 0.0625 * column;
            }
        }

        Transform.r2c(shape, realStride, offsetIn, complexStride, offsetOut, new int[] {1}, true, input, expected,
                0.5);
        Transform.r2c(shape, realStride, offsetIn, complexStride, offsetOut, 1, true, input, actual, 0.5);

        assertArrayEquals(expected, actual, 1e-12);
    }

    @Test
    void singleAxisComplexRealWrappersMatchArrayAxisRoutes() {
        int[] shape = {2, 4};
        long[] realStride = {4, 1};
        long[] complexStride = {6, 2};
        double[] realInput = realGrid(shape[0], shape[1]);
        double[] spectrum = new double[shape[0] * 2 * (shape[1] / 2 + 1)];
        double[] expected = new double[realInput.length];
        double[] actual = new double[realInput.length];

        Transform.r2c(shape, realStride, complexStride, 1, true, realInput, spectrum, 1.0);
        Transform.c2r(shape, complexStride, realStride, new int[] {1}, false, spectrum, expected, 0.25);
        Transform.c2r(shape, complexStride, realStride, 1, false, spectrum, actual, 0.25);

        assertArrayEquals(expected, actual, 1e-12);

        long[] offsetComplexStride = {10, 2};
        long[] offsetRealStride = {7, 1};
        long offsetIn = 5;
        long offsetOut = 3;
        double[] stridedSpectrum = filled(40, Double.NaN);
        double[] expectedOffset = filled(40, -9999.0);
        double[] actualOffset = filled(40, -9999.0);
        int complexColumns = shape[1] / 2 + 1;
        for (int row = 0; row < shape[0]; row++) {
            for (int frequency = 0; frequency < complexColumns; frequency++) {
                int source = row * (int) complexStride[0] + frequency * (int) complexStride[1];
                int target = (int) (offsetIn + row * offsetComplexStride[0] + frequency * offsetComplexStride[1]);
                stridedSpectrum[target] = spectrum[source];
                stridedSpectrum[target + 1] = spectrum[source + 1];
            }
        }

        Transform.c2r(shape, offsetComplexStride, offsetIn, offsetRealStride, offsetOut, new int[] {1}, false,
                stridedSpectrum, expectedOffset, 0.25);
        Transform.c2r(shape, offsetComplexStride, offsetIn, offsetRealStride, offsetOut, 1, false, stridedSpectrum,
                actualOffset, 0.25);

        assertArrayEquals(expectedOffset, actualOffset, 1e-12);
    }

    @Test
    void zeroOffsetRealFamilyWrappersMatchExplicitOffsetRoutes() {
        int[] shape = {3, 4};
        long[] stride = {4, 1};
        int[] axes = {1};
        double[] input = realGrid(shape[0], shape[1]);

        double[] expectedDst = new double[input.length];
        double[] actualDst = new double[input.length];
        Transform.dst(shape, stride, 0L, stride, 0L, axes, 2, input, expectedDst, 0.75, false);
        Transform.dst(shape, stride, stride, axes, 2, input, actualDst, 0.75, false);
        assertArrayEquals(expectedDst, actualDst, 1e-12);

        double[] expectedHartley = new double[input.length];
        double[] actualHartley = new double[input.length];
        Transform.r2rSeparableHartley(shape, stride, 0L, stride, 0L, axes, input, expectedHartley, 1.25);
        Transform.r2rSeparableHartley(shape, stride, stride, axes, input, actualHartley, 1.25);
        assertArrayEquals(expectedHartley, actualHartley, 1e-12);
    }

    @Test
    void emptyAxesAndZeroExtentNdRoutesAreNoOps() {
        int[] shape = {2, 3};
        long[] complexStride = {6, 2};
        long[] realStride = {3, 1};
        double[] complexInput = complexGrid(shape[0], shape[1]);
        double[] complexOutput = filled(complexInput.length, -9999.0);
        double[] realInput = realGrid(shape[0], shape[1]);
        double[] realOutput = filled(realInput.length, -9999.0);

        Transform.c2c(shape, complexStride, complexStride, new int[0], true, complexInput, complexOutput, 1.0);
        Transform.r2rFftpack(shape, realStride, realStride, new int[0], true, true, realInput, realOutput, 1.0);
        Transform.dct(shape, realStride, realStride, new int[0], 2, realInput, realOutput, 1.0, false);
        Transform.r2rSeparableHartley(shape, realStride, realStride, new int[0], realInput, realOutput, 1.0);

        assertArrayEquals(filled(complexInput.length, -9999.0), complexOutput, 0.0);
        assertArrayEquals(filled(realInput.length, -9999.0), realOutput, 0.0);

        int[] zeroShape = {0, 3};
        Transform.c2c(zeroShape, complexStride, complexStride, new int[] {1}, true, complexInput, complexOutput,
                1.0);
        Transform.r2rFftpack(zeroShape, realStride, realStride, new int[] {1}, true, true, realInput, realOutput,
                1.0);
        Transform.dst(zeroShape, realStride, realStride, new int[] {1}, 2, realInput, realOutput, 1.0, false);
        Transform.r2rSeparableHartley(zeroShape, realStride, realStride, new int[] {1}, realInput, realOutput,
                1.0);

        assertArrayEquals(filled(complexInput.length, -9999.0), complexOutput, 0.0);
        assertArrayEquals(filled(realInput.length, -9999.0), realOutput, 0.0);
    }

    @Test
    void rejectsInvalidNdArguments() {
        int[] shape = {2, 3};
        long[] stride = {6, 2};
        double[] data = new double[12];

        assertThrows(IllegalArgumentException.class,
                () -> Transform.c2c(shape, stride, stride, new int[] {1, 1}, true, data, new double[12], 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Transform.c2c(shape, stride, stride, new int[] {2}, true, data, new double[12], 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Transform.c2c(shape, stride, new long[] {2, 6}, new int[] {1}, true, data, data, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Transform.c2c(shape, stride, 8L, stride, 0L, new int[] {1}, true, data, new double[12], 1.0));
        assertThrows(IllegalArgumentException.class,
            () -> Transform.r2c(shape, new long[] {3, 1}, new long[] {4, 2}, new int[0], true,
                new double[6], new double[8], 1.0));
        assertThrows(IllegalArgumentException.class,
            () -> Transform.c2r(shape, new long[] {4, 2}, new long[] {3, 1}, new int[] {1}, false,
                data, data, 1.0));
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
                data[row * columns + column] = 0.25 + 0.5 * row - 0.125 * column + 0.03125 * ((row + 3 * column) % 5);
            }
        }
        return data;
    }

    private static double[] filled(int length, double value) {
        double[] data = new double[length];
        Arrays.fill(data, value);
        return data;
    }

    private static double[] directComplex2d(double[] input, int rows, int columns, boolean forward, double factor) {
        double[] output = new double[input.length];
        double sign = forward ? -1.0 : 1.0;
        for (int frequencyRow = 0; frequencyRow < rows; frequencyRow++) {
            for (int frequencyColumn = 0; frequencyColumn < columns; frequencyColumn++) {
                double sumReal = 0.0;
                double sumImaginary = 0.0;
                for (int sampleRow = 0; sampleRow < rows; sampleRow++) {
                    for (int sampleColumn = 0; sampleColumn < columns; sampleColumn++) {
                        double angle = sign * 2.0 * Math.PI
                                * ((double) frequencyRow * sampleRow / rows
                                + (double) frequencyColumn * sampleColumn / columns);
                        int inputOffset = 2 * (sampleRow * columns + sampleColumn);
                        double twiddleReal = Math.cos(angle);
                        double twiddleImaginary = Math.sin(angle);
                        double sampleReal = input[inputOffset];
                        double sampleImaginary = input[inputOffset + 1];
                        sumReal += sampleReal * twiddleReal - sampleImaginary * twiddleImaginary;
                        sumImaginary += sampleReal * twiddleImaginary + sampleImaginary * twiddleReal;
                    }
                }
                int outputOffset = 2 * (frequencyRow * columns + frequencyColumn);
                output[outputOffset] = factor * sumReal;
                output[outputOffset + 1] = factor * sumImaginary;
            }
        }
        return output;
    }

    private static double[] directComplex1d(double[] input, boolean forward, double factor) {
        int size = input.length / 2;
        double[] output = new double[input.length];
        double sign = forward ? -1.0 : 1.0;
        for (int frequency = 0; frequency < size; frequency++) {
            double sumReal = 0.0;
            double sumImaginary = 0.0;
            for (int sample = 0; sample < size; sample++) {
                double angle = sign * 2.0 * Math.PI * frequency * sample / size;
                double twiddleReal = Math.cos(angle);
                double twiddleImaginary = Math.sin(angle);
                double sampleReal = input[2 * sample];
                double sampleImaginary = input[2 * sample + 1];
                sumReal += sampleReal * twiddleReal - sampleImaginary * twiddleImaginary;
                sumImaginary += sampleReal * twiddleImaginary + sampleImaginary * twiddleReal;
            }
            output[2 * frequency] = factor * sumReal;
            output[2 * frequency + 1] = factor * sumImaginary;
        }
        return output;
    }

    private static double[] directRealToComplex2d(double[] input, int rows, int columns, boolean forward,
                                                  double factor) {
        int complexColumns = columns / 2 + 1;
        double[] output = new double[2 * rows * complexColumns];
        double sign = forward ? -1.0 : 1.0;
        for (int frequencyRow = 0; frequencyRow < rows; frequencyRow++) {
            for (int frequencyColumn = 0; frequencyColumn < complexColumns; frequencyColumn++) {
                double sumReal = 0.0;
                double sumImaginary = 0.0;
                for (int sampleRow = 0; sampleRow < rows; sampleRow++) {
                    for (int sampleColumn = 0; sampleColumn < columns; sampleColumn++) {
                        double angle = sign * 2.0 * Math.PI
                                * ((double) frequencyRow * sampleRow / rows
                                + (double) frequencyColumn * sampleColumn / columns);
                        double sample = input[sampleRow * columns + sampleColumn];
                        sumReal += sample * Math.cos(angle);
                        sumImaginary += sample * Math.sin(angle);
                    }
                }
                int outputOffset = 2 * (frequencyRow * complexColumns + frequencyColumn);
                output[outputOffset] = factor * sumReal;
                output[outputOffset + 1] = factor * sumImaginary;
            }
        }
        return output;
    }
}