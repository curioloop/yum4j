package com.curioloop.yum4j.fft;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FFTPocketfftFixtureTest {
    private static final Map<String, Fixture> FIXTURES = loadFixtures();

    @Test
    void c2cMatchesUpstreamFixturesAcrossRadicesAndBluesteinSizes() {
        for (int size : new int[] {1, 2, 3, 4, 5, 8, 13, 77, 257}) {
            double[] input = complexSample(size);

            double[] output = input.clone();
            Transform.c2c(output, true, 1.0);
            assertComplexFixture("c2c_forward_" + size, output, tolerance(size));

            double[] backward = input.clone();
            Transform.c2c(backward, false, 1.0);
            assertComplexFixture("c2c_backward_" + size, backward, tolerance(size));

            Transform.c2c(output, false, 1.0 / size);
            assertComplexFixture("c2c_roundtrip_" + size, output, tolerance(size));
        }
    }

    @Test
    void realComplexTransformsMatchUpstreamFixtures() {
        for (int size : new int[] {1, 2, 3, 4, 5, 8, 17, 77, 257}) {
            double[] input = realSample(size);
            double[] spectrum = new double[2 * (size / 2 + 1)];
            double[] backwardSpectrum = new double[spectrum.length];
            double[] roundtrip = new double[size];

            double[] inputCopy1 = input.clone();
            Transform.r2c(inputCopy1, spectrum, true, 1.0);
            assertComplexFixture("r2c_forward_" + size, spectrum, tolerance(size));

            double[] inputCopy2 = input.clone();
            Transform.r2c(inputCopy2, backwardSpectrum, false, 1.0);
            assertComplexFixture("r2c_backward_" + size, backwardSpectrum, tolerance(size));

            Transform.c2r(spectrum, roundtrip, false, 1.0 / size);
            assertRealFixture("c2r_roundtrip_" + size, roundtrip, tolerance(size));
        }
    }

    @Test
    void realTransformFamiliesMatchUpstreamFixtures() {
        for (int size : new int[] {1, 2, 3, 4, 5, 8, 17}) {
            double[] input = realSample(size);

            double[] output = input.clone();
            Transform.r2rFftpack(output, true, true, 1.0);
            assertRealFixture("r2r_fftpack_forward_" + size, output, tolerance(size));

            output = input.clone();
            Transform.r2rSeparableHartley(output, 1.0);
            assertRealFixture("hartley_" + size, output, tolerance(size));

            output = input.clone();
            Transform.r2rSeparableFht(output, 1.0);
            assertRealFixture("fht_" + size, output, tolerance(size));

            for (int type = 1; type <= 4; type++) {
                if (type == 1 && size < 2) {
                    continue;
                }
                output = input.clone();
                Transform.dct(output, type, 1.0, false);
                assertRealFixture("dct" + type + "_" + size, output, tolerance(size));

                output = input.clone();
                Transform.dst(output, type, 1.0, false);
                assertRealFixture("dst" + type + "_" + size, output, tolerance(size));
            }
        }
    }

    @Test
    void ndTransformsMatchUpstreamFixtures() {
        int[] shape = {3, 4};
        long[] realStride = {4, 1};
        long[] complexStride = {8, 2};
        long[] spectrumStride = {6, 2};
        int[] axes = {0, 1};

        double[] complexInput = complexGrid(3, 4);
        double[] complexOutput = new double[complexInput.length];
        Transform.c2c(shape, complexStride, complexStride, axes, true, complexInput, complexOutput, 0.5);
        assertComplexFixture("c2c_2d_forward_3x4", complexOutput, 1e-11);

        double[] realInput = realGrid(3, 4);
        double[] spectrum = new double[3 * 2 * (4 / 2 + 1)];
        Transform.r2c(shape, realStride, spectrumStride, axes, true, realInput, spectrum, 1.0);
        assertComplexFixture("r2c_2d_forward_3x4", spectrum, 1e-11);

        double[] restored = new double[realInput.length];
        Transform.c2r(shape, spectrumStride, realStride, axes, false, spectrum, restored, 1.0 / 12.0);
        assertRealFixture("c2r_2d_roundtrip_3x4", restored, 1e-11);

        double[] realOutput = new double[realInput.length];
        Transform.r2rGenuineHartley(shape, realStride, realStride, axes, realInput, realOutput, 1.0);
        assertRealFixture("genuine_hartley_2d_3x4", realOutput, 1e-11);

        Transform.r2rGenuineFht(shape, realStride, realStride, axes, realInput, realOutput, 1.0);
        assertRealFixture("genuine_fht_2d_3x4", realOutput, 1e-11);

        Transform.dct(shape, realStride, realStride, new int[] {1}, 2, realInput, realOutput, 0.75, false);
        assertRealFixture("dct2_axis1_3x4", realOutput, 1e-12);
    }

    @Test
    void negativeStrideTransformsMatchUpstreamFixtures() {
        int[] shape = {5};
        int[] axes = {0};

        double[] realInput = realSample(5);
        double[] realOutput = new double[5];
        Transform.r2rFftpack(shape, new long[] {-1}, 4L, new long[] {1}, 0L, axes, true, true, realInput,
                realOutput, 1.0);
        assertRealFixture("r2r_fftpack_negative_stride_5", realOutput, 1e-12);

        double[] complexInput = complexSample(5);
        double[] complexOutput = new double[complexInput.length];
        Transform.c2c(shape, new long[] {-2}, 8L, new long[] {2}, 0L, axes, true, complexInput, complexOutput, 1.0);
        assertComplexFixture("c2c_negative_stride_5", complexOutput, 1e-12);
    }

    @Test
    void stridedRealComplexTransformsMatchUpstreamFixtures() {
        int[] axes = {0};

        double[] c2cInput6 = stridedComplexSignal(6, 2, 2, -901.0, 901.0);
        double[] c2cForward6 = complexFilled(1 + (6 - 1) * 3 + 1, -902.0, 902.0);
        Transform.c2c(new int[] {6}, new long[] {4}, 4L, new long[] {6}, 2L, axes, true, c2cInput6,
                c2cForward6, 0.625);
        assertComplexFixture("c2c_strided_forward_6", c2cForward6, 1e-12);

        double[] c2cInput7 = stridedComplexSignal(7, 1, 3, -903.0, 903.0);
        double[] c2cBackward7 = complexFilled(2 + (7 - 1) * 2 + 1, -904.0, 904.0);
        Transform.c2c(new int[] {7}, new long[] {6}, 2L, new long[] {4}, 4L, axes, false, c2cInput7,
                c2cBackward7, 0.375);
        assertComplexFixture("c2c_strided_backward_7", c2cBackward7, 1e-12);

        double[] r2cInput6 = stridedRealSample(6);
        double[] r2cForward6 = complexFilled(1 + (6 / 2) * 2 + 1, -777.0, 777.0);
        Transform.r2c(new int[] {6}, new long[] {2}, 1L, new long[] {4}, 2L, axes, true, r2cInput6, r2cForward6,
                0.75);
        assertComplexFixture("r2c_strided_forward_6", r2cForward6, 1e-12);

        double[] r2cInput7 = stridedRealSample(7);
        double[] r2cBackward7 = complexFilled(1 + (7 / 2) * 2 + 1, -777.0, 777.0);
        Transform.r2c(new int[] {7}, new long[] {2}, 1L, new long[] {4}, 2L, axes, false, r2cInput7, r2cBackward7,
                1.25);
        assertComplexFixture("r2c_strided_backward_7", r2cBackward7, 1e-12);

        double[] c2rInput6 = stridedComplexSample(6 / 2 + 1);
        double[] c2rForward6 = filled(3 + (6 - 1) * 2 + 1, -333.0);
        Transform.c2r(new int[] {6}, new long[] {4}, 2L, new long[] {2}, 3L, axes, true, c2rInput6, c2rForward6,
                0.5);
        assertRealFixture("c2r_strided_forward_6", c2rForward6, 1e-12);

        double[] c2rInput7 = stridedComplexSample(7 / 2 + 1);
        double[] c2rBackward7 = filled(3 + (7 - 1) * 2 + 1, -333.0);
        Transform.c2r(new int[] {7}, new long[] {4}, 2L, new long[] {2}, 3L, axes, false, c2rInput7, c2rBackward7,
                0.875);
        assertRealFixture("c2r_strided_backward_7", c2rBackward7, 1e-12);
    }

    @Test
    void ndRealComplexAxisOrdersMatchUpstreamFixtures() {
        int[] shape = {2, 3, 4};
        long[] realStride = {12, 4, 1};
        long[] spectrumStride = {18, 6, 2};
        int[] axes = {0, 2};

        double[] input = realCube(2, 3, 4);
        double[] spectrum = new double[2 * 2 * 3 * (4 / 2 + 1)];
        Transform.r2c(shape, realStride, spectrumStride, axes, true, input, spectrum, 0.5);
        assertComplexFixture("r2c_3d_axes_0_2_2x3x4", spectrum, 1e-11);

        double[] restored = new double[input.length];
        Transform.c2r(shape, spectrumStride, realStride, axes, false, spectrum, restored, 0.25);
        assertRealFixture("c2r_3d_axes_0_2_roundtrip_2x3x4", restored, 1e-11);
    }

    @Test
    void orthonormalDcstAndFftpackDirectionFixturesMatchUpstream() {
        int[] axes = {0};

        double[] dct = realSample(8);
        Transform.dct(dct, 2, 0.5, true);
        assertRealFixture("dct2_ortho_factor_8", dct, 1e-12);

        double[] dst = realSample(7);
        Transform.dst(dst, 4, 1.25, true);
        assertRealFixture("dst4_ortho_factor_7", dst, 1e-12);

        double[] h2rInput = stridedRealSample(8);
        double[] h2rOutput = filled(2 + (8 - 1) * 3 + 1, -444.0);
        Transform.r2rFftpack(new int[] {8}, new long[] {2}, 1L, new long[] {3}, 2L, axes, false, false,
                h2rInput, h2rOutput, 0.625);
        assertRealFixture("r2r_fftpack_h2r_strided_backward_8", h2rOutput, 1e-12);
    }

    @Test
    void layoutSensitiveNdFamiliesMatchUpstreamFixtures() {
        int[] shape3 = {2, 3, 5};
        long[] complexStride3 = {30, 10, 2};
        double[] complexInput = complexCube(2, 3, 5);
        double[] complexOutput = new double[complexInput.length];
        Transform.c2c(shape3, complexStride3, complexStride3, new int[] {2, 0}, true, complexInput, complexOutput, 0.25);
        assertComplexFixture("c2c_nd_axes_2_0_factor_2x3x5", complexOutput, 1e-11);

        int[] shape = {3, 4};
        long[] realStride = {4, 1};
        long[] columnMajorStride = {1, 3};
        double[] realInput = realGrid(3, 4);
        double[] realOutput = filled(12, -222.0);
        Transform.r2rSeparableHartley(shape, realStride, 0L, columnMajorStride, 0L, new int[] {0}, realInput,
                realOutput, 1.25);
        assertRealFixture("hartley_axis0_colmajor_3x4", realOutput, 1e-12);

        Arrays.fill(realOutput, -222.0);
        Transform.r2rSeparableFht(shape, realStride, 0L, columnMajorStride, 0L, new int[] {0}, realInput, realOutput,
                1.25);
        assertRealFixture("fht_axis0_colmajor_3x4", realOutput, 1e-12);

        double[] genuineOutput = filled(20, -9999.0);
        Transform.r2rGenuineHartley(shape, realStride, 0L, new long[] {5, 1}, 2L, new int[] {0, 1}, realInput,
                genuineOutput, 0.75);
        assertRealFixture("genuine_hartley_strided_offset_3x4", genuineOutput, 1e-11);

        Arrays.fill(genuineOutput, -9999.0);
        Transform.r2rGenuineFht(shape, realStride, 0L, new long[] {5, 1}, 2L, new int[] {0, 1}, realInput,
                genuineOutput, 0.75);
        assertRealFixture("genuine_fht_strided_offset_3x4", genuineOutput, 1e-11);

        int[] dcstShape = {4, 3};
        long[] dcstStrideIn = {5, 1};
        long[] dcstStrideOut = {7, 2};
        double[] dcstInput = filled(32, -4444.0);
        double[] dcstOutput = filled(40, -9999.0);
        for (int row = 0; row < dcstShape[0]; row++) {
            for (int column = 0; column < dcstShape[1]; column++) {
                dcstInput[2 + row * 5 + column] = 0.25 + 0.5 * row - 0.125 * column;
            }
        }
        Transform.dst(dcstShape, dcstStrideIn, 2L, dcstStrideOut, 4L, new int[] {0}, 3, dcstInput, dcstOutput, 0.5,
                false);
        assertRealFixture("dst3_axis0_strided_offset_4x3", dcstOutput, 1e-12);
    }

    private static Map<String, Fixture> loadFixtures() {
        InputStream stream = FFTPocketfftFixtureTest.class.getResourceAsStream("/fft/pocketfft_oracle.tsv");
        assertNotNull(stream, "pocketfft oracle fixture resource is missing");
        Map<String, Fixture> fixtures = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                String[] parts = line.split("\t");
                assertTrue(parts.length >= 3, "malformed fixture line: " + line);
                boolean complex = switch (parts[0]) {
                    case "real" -> false;
                    case "complex" -> true;
                    default -> throw new IllegalArgumentException("unknown fixture type: " + parts[0]);
                };
                if (complex) {
                    assertFalse(((parts.length - 2) & 1) != 0, "complex fixture has odd value count: " + parts[1]);
                }
                double[] values = new double[parts.length - 2];
                for (int index = 2; index < parts.length; index++) {
                    values[index - 2] = Double.parseDouble(parts[index]);
                }
                fixtures.put(parts[1], new Fixture(complex, values));
            }
        } catch (IOException exception) {
            throw new AssertionError("failed to read pocketfft oracle fixtures", exception);
        }
        return fixtures;
    }

    private static void assertRealFixture(String name, double[] actual, double tolerance) {
        Fixture fixture = fixture(name, false);
        assertArrayEquals(fixture.values, actual, tolerance, name);
    }

    private static void assertComplexFixture(String name, double[] actual, double tolerance) {
        Fixture fixture = fixture(name, true);
        assertArrayEquals(fixture.values, actual, tolerance, name);
    }

    private static Fixture fixture(String name, boolean complex) {
        Fixture fixture = FIXTURES.get(name);
        assertNotNull(fixture, "missing fixture: " + name);
        if (fixture.complex != complex) {
            throw new AssertionError("fixture type mismatch: " + name);
        }
        return fixture;
    }

    private static double tolerance(int size) {
        return Math.max(1e-12, size * 2e-12);
    }

    private static double[] realSample(int size) {
        double[] data = new double[size];
        for (int index = 0; index < size; index++) {
            data[index] = 0.25 + 0.125 * ((13 * index + 7) % 17) - 0.03125 * (index % 5);
        }
        return data;
    }

    private static double[] complexSample(int size) {
        double[] data = new double[2 * size];
        for (int index = 0; index < size; index++) {
            data[2 * index] = 0.25 + 0.125 * ((11 * index + 3) % 19);
            data[2 * index + 1] = -0.5 + 0.0625 * ((7 * index + 5) % 23);
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

    private static double[] complexCube(int planes, int rows, int columns) {
        double[] data = new double[2 * planes * rows * columns];
        for (int plane = 0; plane < planes; plane++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    int offset = 2 * ((plane * rows + row) * columns + column);
                    data[offset] = 0.125 + 0.375 * plane + 0.25 * row - 0.0625 * column
                            + 0.03125 * ((3 * plane + 5 * row + 7 * column) % 11);
                    data[offset + 1] = -0.25 + 0.125 * plane - 0.03125 * row
                            + 0.0625 * ((5 * plane + row + 3 * column) % 13);
                }
            }
        }
        return data;
    }

    private static double[] realCube(int planes, int rows, int columns) {
        double[] data = new double[planes * rows * columns];
        for (int plane = 0; plane < planes; plane++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    int offset = (plane * rows + row) * columns + column;
                    data[offset] = 0.1875 + 0.25 * plane - 0.15625 * row + 0.09375 * column
                            + 0.015625 * ((5 * plane + 7 * row + 11 * column) % 13);
                }
            }
        }
        return data;
    }

    private static double[] stridedRealSample(int size) {
        double[] compact = realSample(size);
        double[] data = filled(1 + (size - 1) * 2 + 1, -1111.0);
        for (int index = 0; index < size; index++) {
            data[1 + 2 * index] = compact[index];
        }
        return data;
    }

    private static double[] stridedComplexSample(int complexLength) {
        double[] compact = complexSample(complexLength);
        double[] data = complexFilled(1 + (complexLength - 1) * 2 + 1, -555.0, 555.0);
        for (int index = 0; index < complexLength; index++) {
            data[2 + 4 * index] = compact[2 * index];
            data[2 + 4 * index + 1] = compact[2 * index + 1];
        }
        return data;
    }

    private static double[] stridedComplexSignal(int size, int offsetSlot, int strideSlots, double real,
                                                 double imaginary) {
        double[] compact = complexSample(size);
        double[] data = complexFilled(offsetSlot + (size - 1) * strideSlots + 1, real, imaginary);
        for (int index = 0; index < size; index++) {
            int slot = offsetSlot + strideSlots * index;
            data[2 * slot] = compact[2 * index];
            data[2 * slot + 1] = compact[2 * index + 1];
        }
        return data;
    }

    private static double[] complexFilled(int complexSlots, double real, double imaginary) {
        double[] data = new double[2 * complexSlots];
        for (int index = 0; index < complexSlots; index++) {
            data[2 * index] = real;
            data[2 * index + 1] = imaginary;
        }
        return data;
    }

    private static double[] filled(int size, double value) {
        double[] data = new double[size];
        Arrays.fill(data, value);
        return data;
    }

    private record Fixture(boolean complex, double[] values) {
    }
}
