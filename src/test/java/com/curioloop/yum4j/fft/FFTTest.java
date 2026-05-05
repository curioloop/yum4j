package com.curioloop.yum4j.fft;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FFTTest {
    @Test
    void c2cMatchesDirectOracleInPlace() {
        for (int size : new int[] {12, 77, 257}) {
            double[] input = complexSample(size);
            double[] expected = directComplex(input, true);

            double[] inPlace = input.clone();
            Transform.c2c(inPlace, true, 1.0);
            assertArrayEquals(expected, inPlace, Math.max(1e-12, size * 5e-12), "in-place c2c mismatch for size " + size);
        }
    }

    @Test
    void r2cMatchesDirectOracleForForwardAndBackwardConventions() {
        for (int size : new int[] {1, 2, 3, 4, 5, 12, 77, 257}) {
            double[] input = realSample(size);
            double[] forward = new double[2 * (size / 2 + 1)];
            double[] backward = new double[forward.length];

            double[] inputCopy1 = input.clone();
            Transform.r2c(inputCopy1, forward, true, 1.0);
            double[] inputCopy2 = input.clone();
            Transform.r2c(inputCopy2, backward, false, 1.0);

            assertArrayEquals(directRealToComplex(input, true), forward, Math.max(1e-12, size * 6e-12),
                    "forward r2c mismatch for size " + size);
            assertArrayEquals(directRealToComplex(input, false), backward, Math.max(1e-12, size * 6e-12),
                    "backward r2c mismatch for size " + size);
        }
    }

    @Test
    void r2cAndC2rRoundTripBothSignConventions() {
        for (int size : new int[] {1, 2, 3, 4, 5, 12, 77, 257}) {
            double[] input = realSample(size);
            double[] forwardSpectrum = new double[2 * (size / 2 + 1)];
            double[] backwardSpectrum = new double[forwardSpectrum.length];
            double[] restored = new double[size];

            double[] inputCopy1 = input.clone();
            Transform.r2c(inputCopy1, forwardSpectrum, true, 1.0);
            Transform.c2r(forwardSpectrum, restored, false, 1.0 / size);
            assertArrayEquals(input, restored, Math.max(1e-12, size * 4e-13), "forward-sign roundtrip mismatch for size " + size);

            double[] inputCopy2 = input.clone();
            Transform.r2c(inputCopy2, backwardSpectrum, false, 1.0);
            Transform.c2r(backwardSpectrum, restored, true, 1.0 / size);
            assertArrayEquals(input, restored, Math.max(1e-12, size * 4e-13), "backward-sign roundtrip mismatch for size " + size);
        }
    }

    @Test
    void r2rFftpackHandlesDirectionSignLikePocketfft() {
        double[] forward = realSample(15);
        double[] backward = forward.clone();

        Transform.r2rFftpack(forward, true, true, 1.0);
        Transform.r2rFftpack(backward, true, false, 1.0);

        double[] expectedBackward = forward.clone();
        negateHalfComplexImaginary(expectedBackward);
        assertArrayEquals(expectedBackward, backward, 1e-12);
    }

    @Test
    void r2rFftpackRoundTripsBothSignConventions() {
        for (int size : new int[] {2, 3, 4, 5, 12, 77, 257}) {
            double[] input = realSample(size);

            double[] forward = input.clone();
            Transform.r2rFftpack(forward, true, true, 1.0);
            Transform.r2rFftpack(forward, false, false, 1.0 / size);
            assertArrayEquals(input, forward, Math.max(1e-12, size * 4e-13), "forward r2r roundtrip mismatch for size " + size);

            double[] backward = input.clone();
            Transform.r2rFftpack(backward, true, false, 1.0);
            Transform.r2rFftpack(backward, false, true, 1.0 / size);
            assertArrayEquals(input, backward, Math.max(1e-12, size * 4e-13), "backward r2r roundtrip mismatch for size " + size);
        }
    }

    @Test
    void zeroLengthOneDimensionalWrappersAreNoOps() {
        Transform.c2c(new double[0], true, 1.0);
        Transform.r2c(new double[0], new double[0], true, 1.0);
        Transform.c2r(new double[0], new double[0], false, 1.0);
        Transform.r2rFftpack(new double[0], true, true, 1.0);
        Transform.dct(new double[0], 2, 1.0, false);
        Transform.dst(new double[0], 3, 1.0, false);
        Transform.r2rSeparableHartley(new double[0], 1.0);
        Transform.r2rSeparableFht(new double[0], 1.0);
        Transform.r2rGenuineHartley(new double[0], 1.0);
        Transform.r2rGenuineFht(new double[0], 1.0);
    }

    @Test
    void rejectsInvalidPublicApiShapes() {
        assertThrows(IllegalArgumentException.class, () -> Transform.c2c(new double[3], true, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Transform.r2c(new double[4], new double[4], true, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Transform.c2r(new double[4], new double[4], false, 1.0));
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

    private static double[] directComplex(double[] input, boolean forward) {
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
            output[2 * frequency] = sumReal;
            output[2 * frequency + 1] = sumImaginary;
        }
        return output;
    }

    private static double[] directRealToComplex(double[] input, boolean forward) {
        int size = input.length;
        double[] output = new double[2 * (size / 2 + 1)];
        double sign = forward ? -1.0 : 1.0;
        for (int frequency = 0; frequency <= size / 2; frequency++) {
            double sumReal = 0.0;
            double sumImaginary = 0.0;
            for (int sample = 0; sample < size; sample++) {
                double angle = sign * 2.0 * Math.PI * frequency * sample / size;
                sumReal += input[sample] * Math.cos(angle);
                sumImaginary += input[sample] * Math.sin(angle);
            }
            output[2 * frequency] = sumReal;
            output[2 * frequency + 1] = sumImaginary;
        }
        return output;
    }

    private static void negateHalfComplexImaginary(double[] halfComplex) {
        for (int index = 2; index < halfComplex.length; index += 2) {
            halfComplex[index] = -halfComplex[index];
        }
    }
}
