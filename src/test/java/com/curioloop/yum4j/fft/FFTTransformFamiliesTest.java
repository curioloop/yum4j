package com.curioloop.yum4j.fft;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FFTTransformFamiliesTest {
    @Test
    void dctTypesMatchDirectDefinitions() {
        for (int size : new int[] {2, 3, 4, 5, 8}) {
            for (int type = 1; type <= 4; type++) {
                double[] input = sample(size);
                double[] output = input.clone();

                Transform.dct(output, type, 1.0, false);

                assertArrayEquals(directDct(input, type, 1.0), output, size * 3e-12,
                        "DCT-" + type + " mismatch for size " + size);
            }
        }
    }

    @Test
    void dstTypesMatchDirectDefinitions() {
        for (int size : new int[] {1, 2, 3, 4, 5, 8}) {
            for (int type = 1; type <= 4; type++) {
                double[] input = sample(size);
                double[] output = input.clone();

                Transform.dst(output, type, 1.0, false);

                assertArrayEquals(directDst(input, type, 1.0), output, size * 3e-12,
                        "DST-" + type + " mismatch for size " + size);
            }
        }
    }

    @Test
    void dcstInversePairingsMatchPocketfftScaling() {
        for (int size : new int[] {2, 3, 4, 5, 8, 17}) {
            double[] original = sample(size);

            double[] dct1 = original.clone();
            Transform.dct(dct1, 1, 1.0, false);
            Transform.dct(dct1, 1, 1.0 / (2.0 * (size - 1)), false);
            assertArrayEquals(original, dct1, size * 3e-13, "DCT-I roundtrip mismatch for size " + size);

            double[] dst1 = original.clone();
            Transform.dst(dst1, 1, 1.0, false);
            Transform.dst(dst1, 1, 1.0 / (2.0 * (size + 1)), false);
            assertArrayEquals(original, dst1, size * 3e-13, "DST-I roundtrip mismatch for size " + size);

            double[] dct23 = original.clone();
            Transform.dct(dct23, 2, 1.0, false);
            Transform.dct(dct23, 3, 1.0 / (2.0 * size), false);
            assertArrayEquals(original, dct23, size * 3e-13, "DCT-II/III roundtrip mismatch for size " + size);

            double[] dst23 = original.clone();
            Transform.dst(dst23, 2, 1.0, false);
            Transform.dst(dst23, 3, 1.0 / (2.0 * size), false);
            assertArrayEquals(original, dst23, size * 3e-13, "DST-II/III roundtrip mismatch for size " + size);

            double[] dct4 = original.clone();
            Transform.dct(dct4, 4, 1.0, false);
            Transform.dct(dct4, 4, 1.0 / (2.0 * size), false);
            assertArrayEquals(original, dct4, size * 3e-13, "DCT-IV roundtrip mismatch for size " + size);

            double[] dst4 = original.clone();
            Transform.dst(dst4, 4, 1.0, false);
            Transform.dst(dst4, 4, 1.0 / (2.0 * size), false);
            assertArrayEquals(original, dst4, size * 3e-13, "DST-IV roundtrip mismatch for size " + size);
        }
    }

    @Test
    void hartleyAndFhtMatchDirectDefinitions() {
        for (int size : new int[] {1, 2, 3, 4, 5, 8, 17}) {
            double[] input = sample(size);

            double[] hartley = input.clone();
            Transform.r2rSeparableHartley(hartley, 1.0);

            double[] fht = input.clone();
            Transform.r2rSeparableFht(fht, 1.0);

            assertArrayEquals(directHartley1d(input, false, 1.0), hartley, size * 3e-12,
                    "Hartley mismatch for size " + size);
            assertArrayEquals(directHartley1d(input, true, 1.0), fht, size * 3e-12,
                    "FHT mismatch for size " + size);
        }
    }

    @Test
    void genuineHartleyAndFhtNdMatchDirectDefinitions() {
        int[] shape = {3, 4};
        long[] stride = {4, 1};
        int[] axes = {0, 1};
        double[] input = realGrid(shape[0], shape[1]);
        double[] hartley = new double[input.length];
        double[] fht = new double[input.length];

        Transform.r2rGenuineHartley(shape, stride, stride, axes, input, hartley, 1.0);
        Transform.r2rGenuineFht(shape, stride, stride, axes, input, fht, 1.0);

        assertArrayEquals(directHartley2d(input, shape[0], shape[1], false, 1.0), hartley, 1e-11);
        assertArrayEquals(directHartley2d(input, shape[0], shape[1], true, 1.0), fht, 1e-11);
    }

    @Test
    void ndDctMatchesAxisWiseDirectDefinition() {
        int[] shape = {3, 5};
        long[] stride = {5, 1};
        double[] input = realGrid(shape[0], shape[1]);
        double[] output = new double[input.length];
        double[] expected = new double[input.length];

        for (int row = 0; row < shape[0]; row++) {
            double[] line = Arrays.copyOfRange(input, row * shape[1], (row + 1) * shape[1]);
            System.arraycopy(directDct(line, 2, 0.75), 0, expected, row * shape[1], shape[1]);
        }

        Transform.dct(shape, stride, stride, new int[] {1}, 2, input, output, 0.75, false);

        assertArrayEquals(expected, output, 1e-12);
    }

    @Test
    void ndDctMultiAxisMatchesSeparableDirectDefinitionAndScalesOnce() {
        int[] shape = {3, 4};
        long[] stride = {4, 1};
        int[] axes = {0, 1};
        double[] input = realGrid(shape[0], shape[1]);
        double[] output = new double[input.length];
        double factor = 0.375;

        Transform.dct(shape, stride, stride, axes, 2, input, output, factor, false);

        assertArrayEquals(directDcstAxes(input, shape[0], shape[1], axes, 2, true, factor), output, 1e-11);
    }

    @Test
    void ndDstSupportsStridesOffsetsAndSentinels() {
        int[] shape = {4, 3};
        long[] strideIn = {5, 1};
        long[] strideOut = {7, 2};
        long offsetIn = 2;
        long offsetOut = 4;
        double[] input = filled(32, Double.NaN);
        double[] output = filled(40, -9999.0);

        for (int row = 0; row < shape[0]; row++) {
            for (int column = 0; column < shape[1]; column++) {
                input[(int) (offsetIn + row * strideIn[0] + column * strideIn[1])] =
                        0.25 + 0.5 * row - 0.125 * column;
            }
        }

        Transform.dst(shape, strideIn, offsetIn, strideOut, offsetOut, new int[] {0}, 3, input, output, 0.5, false);

        for (int column = 0; column < shape[1]; column++) {
            double[] line = new double[shape[0]];
            for (int row = 0; row < shape[0]; row++) {
                line[row] = input[(int) (offsetIn + row * strideIn[0] + column * strideIn[1])];
            }
            double[] expected = directDst(line, 3, 0.5);
            for (int row = 0; row < shape[0]; row++) {
                int actual = (int) (offsetOut + row * strideOut[0] + column * strideOut[1]);
                assertEquals(expected[row], output[actual], 1e-12);
            }
        }
        assertEquals(-9999.0, output[0], 0.0);
        assertEquals(-9999.0, output[39], 0.0);
    }

    @Test
    void rejectsInvalidTransformFamilyArguments() {
        assertThrows(IllegalArgumentException.class, () -> Transform.dct(new double[] {1.0}, 1, 1.0, false));
        assertThrows(IllegalArgumentException.class, () -> Transform.dct(new double[] {1.0, 2.0}, 0, 1.0, false));
        assertThrows(IllegalArgumentException.class, () -> Transform.dst(new double[] {1.0, 2.0}, 5, 1.0, false));

        assertEquals(4, new FftDcstPlan(4, 2, true).length());
        assertThrows(IllegalArgumentException.class, () -> new FftDcstPlan(0, 2, true));
        assertThrows(IllegalArgumentException.class, () -> new FftDcstPlan(4, 0, true));
        assertThrows(IllegalArgumentException.class, () -> new FftDcstPlan(1, 1, true));
        assertThrows(IllegalArgumentException.class,
            () -> new FftDcstPlan(4, 2, true).exec(new double[4], 0, 1.0, false, 3, true,
                FftWorkspace.Requirement.empty().allocate()));
    }

    private static double[] sample(int size) {
        double[] data = new double[size];
        for (int index = 0; index < size; index++) {
            data[index] = 0.25 + 0.125 * ((13 * index + 7) % 17) - 0.03125 * (index % 5);
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

    private static double[] filled(int size, double value) {
        double[] data = new double[size];
        Arrays.fill(data, value);
        return data;
    }

    private static double[] directDcstAxes(double[] input, int rows, int columns, int[] axes, int type,
                                           boolean cosine, double factor) {
        double[] current = input.clone();
        double currentFactor = factor;
        for (int axis : axes) {
            double[] next = new double[current.length];
            if (axis == 0) {
                for (int column = 0; column < columns; column++) {
                    double[] line = new double[rows];
                    for (int row = 0; row < rows; row++) {
                        line[row] = current[row * columns + column];
                    }
                    double[] transformed = cosine ? directDct(line, type, currentFactor)
                            : directDst(line, type, currentFactor);
                    for (int row = 0; row < rows; row++) {
                        next[row * columns + column] = transformed[row];
                    }
                }
            } else {
                for (int row = 0; row < rows; row++) {
                    double[] line = Arrays.copyOfRange(current, row * columns, (row + 1) * columns);
                    double[] transformed = cosine ? directDct(line, type, currentFactor)
                            : directDst(line, type, currentFactor);
                    System.arraycopy(transformed, 0, next, row * columns, columns);
                }
            }
            current = next;
            currentFactor = 1.0;
        }
        return current;
    }

    private static double[] directDct(double[] input, int type, double factor) {
        int size = input.length;
        double[] output = new double[size];
        for (int k = 0; k < size; k++) {
            double sum;
            if (type == 1) {
                sum = input[0] + ((k & 1) == 0 ? input[size - 1] : -input[size - 1]);
                for (int n = 1; n < size - 1; n++) {
                    sum += 2.0 * input[n] * Math.cos(Math.PI * n * k / (size - 1));
                }
            } else if (type == 2) {
                sum = 0.0;
                for (int n = 0; n < size; n++) {
                    sum += 2.0 * input[n] * Math.cos(Math.PI * (n + 0.5) * k / size);
                }
            } else if (type == 3) {
                sum = input[0];
                for (int n = 1; n < size; n++) {
                    sum += 2.0 * input[n] * Math.cos(Math.PI * n * (k + 0.5) / size);
                }
            } else {
                sum = 0.0;
                for (int n = 0; n < size; n++) {
                    sum += 2.0 * input[n] * Math.cos(Math.PI * (n + 0.5) * (k + 0.5) / size);
                }
            }
            output[k] = factor * sum;
        }
        return output;
    }

    private static double[] directDst(double[] input, int type, double factor) {
        int size = input.length;
        double[] output = new double[size];
        for (int k = 0; k < size; k++) {
            double sum = 0.0;
            if (type == 1) {
                for (int n = 0; n < size; n++) {
                    sum += 2.0 * input[n] * Math.sin(Math.PI * (n + 1) * (k + 1) / (size + 1));
                }
            } else if (type == 2) {
                for (int n = 0; n < size; n++) {
                    sum += 2.0 * input[n] * Math.sin(Math.PI * (n + 0.5) * (k + 1) / size);
                }
            } else if (type == 3) {
                sum = ((k & 1) == 0 ? input[size - 1] : -input[size - 1]);
                for (int n = 0; n < size - 1; n++) {
                    sum += 2.0 * input[n] * Math.sin(Math.PI * (n + 1) * (k + 0.5) / size);
                }
            } else {
                for (int n = 0; n < size; n++) {
                    sum += 2.0 * input[n] * Math.sin(Math.PI * (n + 0.5) * (k + 0.5) / size);
                }
            }
            output[k] = factor * sum;
        }
        return output;
    }

    private static double[] directHartley1d(double[] input, boolean fht, double factor) {
        int size = input.length;
        double[] output = new double[size];
        for (int k = 0; k < size; k++) {
            double sum = 0.0;
            for (int n = 0; n < size; n++) {
                double angle = -2.0 * Math.PI * k * n / size;
                sum += input[n] * (fht ? Math.cos(angle) - Math.sin(angle) : Math.cos(angle) + Math.sin(angle));
            }
            output[k] = factor * sum;
        }
        return output;
    }

    private static double[] directHartley2d(double[] input, int rows, int columns, boolean fht, double factor) {
        double[] output = new double[input.length];
        for (int kr = 0; kr < rows; kr++) {
            for (int kc = 0; kc < columns; kc++) {
                double sum = 0.0;
                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < columns; column++) {
                        double angle = -2.0 * Math.PI * ((double) kr * row / rows + (double) kc * column / columns);
                        double kernel = fht ? Math.cos(angle) - Math.sin(angle) : Math.cos(angle) + Math.sin(angle);
                        sum += input[row * columns + column] * kernel;
                    }
                }
                output[kr * columns + kc] = factor * sum;
            }
        }
        return output;
    }
}
