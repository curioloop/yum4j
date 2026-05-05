package com.curioloop.yum4j.fft;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FftRealTransformTest {
    @Test
    void selectionMatchesPocketfftHeuristicForRepresentativeLengths() {
        assertRoute(12, FftRoute.DIRECT, 12, List.of(4, 3));
        assertRoute(64, FftRoute.SPLIT_RADIX, 64, List.of());
        assertRoute(77, FftRoute.DIRECT, 77, List.of(7, 11));
        assertRoute(96, FftRoute.DIRECT, 96, List.of(2, 4, 4, 3));
        assertRoute(120, FftRoute.DIRECT, 120, List.of(2, 4, 3, 5));
        assertRoute(256, FftRoute.SPLIT_RADIX, 256, List.of());
        assertRoute(257, FftRoute.BLUESTEIN, 525, List.of());
        assertRoute(509, FftRoute.BLUESTEIN, 1024, List.of());
        assertRoute(1024, FftRoute.SPLIT_RADIX, 1024, List.of());
        assertRoute(4096, FftRoute.SPLIT_RADIX, 4096, List.of());
    }

    @Test
    void bluesteinRealForwardMatchesDirectOracle() {
        for (int size : new int[] {257, 509}) {
            FftRealTransform transform = new FftRealTransform(size);
            double[] data = sample(size);
            double[] expected = directHalfComplex(data);
            transform.exec(data, 1.0, true);
            assertArrayEquals(expected, data, size * 6e-12, "Real Bluestein mismatch for size " + size);
        }
    }

    @Test
    void flexibleRealTransformRoundTripsDirectAndBluesteinLengths() {
        for (int size : new int[] {12, 77, 143, 257, 509}) {
            FftRealTransform transform = new FftRealTransform(size);
            double[] original = sample(size);
            double[] data = original.clone();
            transform.exec(data, 1.0, true);
            transform.exec(data, 1.0 / size, false);
            assertArrayEquals(original, data, size * 4e-13, "Real roundtrip mismatch for size " + size);
        }
    }

    @Test
    void splitRadixRealForwardCanWriteHermitianOutputDirectly() {
        for (int size : new int[] {4, 8, 64, 256}) {
            FftRealTransform transform = new FftRealTransform(size);
            double[] input = sample(size);
            double[] expectedForward = directHermitian(input, true, 0.5);
            double[] actualForward = new double[2 * (size / 2 + 1)];
            System.arraycopy(input, 0, actualForward, 0, input.length);

            assertTrue(transform.tryRealToHermitianInPlace(actualForward, 0, 0.5, true,
                    transform.workspaceRequirement().allocate()));
            assertArrayEquals(expectedForward, actualForward, size * 6e-12,
                    "direct Hermitian forward mismatch for size " + size);

            double[] expectedBackward = directHermitian(input, false, 0.5);
            double[] actualBackward = new double[2 * (size / 2 + 1)];
            System.arraycopy(input, 0, actualBackward, 0, input.length);

            assertTrue(transform.tryRealToHermitianInPlace(actualBackward, 0, 0.5, false,
                    transform.workspaceRequirement().allocate()));
            assertArrayEquals(expectedBackward, actualBackward, size * 6e-12,
                    "direct Hermitian backward mismatch for size " + size);
        }
    }

    @Test
    void nonSplitRadixRealDeclinesDirectHermitianOutput() {
        FftRealTransform transform = new FftRealTransform(12);
        double[] data = new double[2 * (12 / 2 + 1)];
        double[] original = data.clone();

        assertFalse(transform.tryRealToHermitianInPlace(data, 0, 1.0, true,
                transform.workspaceRequirement().allocate()));
        assertArrayEquals(original, data, 0.0);
    }

    @Test
    void bluesteinRealForwardCanWriteHermitianOutputDirectly() {
        for (int size : new int[] {257, 509}) {
            FftRealTransform transform = new FftRealTransform(size);
            double[] input = sample(size);
            double[] expected = directHermitian(input, true, 0.5);
            double[] actual = new double[2 * (size / 2 + 1)];
            System.arraycopy(input, 0, actual, 0, input.length);

            assertTrue(transform.tryRealToHermitianInPlace(actual, 0, 0.5, true,
                    transform.workspaceRequirement().allocate()));
            assertArrayEquals(expected, actual, size * 6e-12,
                    "direct Bluestein Hermitian mismatch for size " + size);
        }
    }

    @Test
    void bluesteinRealPlanConvenienceOverloadsMatchDirectOracle() {
        FftBluesteinPlan plan = new FftBluesteinPlan(257);
        assertEquals(257, plan.length());
        assertTrue(plan.workspaceRequirement().totalDoubles() > 0);

        double[] forward = sample(257);
        double[] expectedForward = directHalfComplex(forward);
        plan.execReal(forward, 0.5, true);
        for (int index = 0; index < expectedForward.length; index++) {
            expectedForward[index] *= 0.5;
        }
        assertArrayEquals(expectedForward, forward, 257 * 6e-12);

        double[] roundtrip = sample(257);
        double[] original = roundtrip.clone();
        FftWorkspace workspace = plan.workspaceRequirement().allocate();
        plan.execReal(roundtrip, 1.0, true, workspace);
        workspace.reset();
        plan.execReal(roundtrip, 1.0 / 257.0, false, workspace);
        assertArrayEquals(original, roundtrip, 257 * 4e-13);

        FftBluesteinPlan evenPlan = new FftBluesteinPlan(258);
        double[] even = sample(258);
        double[] evenOriginal = even.clone();
        FftWorkspace evenWorkspace = evenPlan.workspaceRequirement().allocate();
        evenPlan.execReal(even, 1.0, true, evenWorkspace);
        evenWorkspace.reset();
        evenPlan.execReal(even, 1.0 / 258.0, false, evenWorkspace);
        assertArrayEquals(evenOriginal, even, 258 * 5e-13);

        assertThrows(IllegalArgumentException.class,
            () -> plan.execReal(new double[plan.length()], -1, 1.0, true, plan.workspaceRequirement().allocate()));
    }

    @Test
    void rejectsInvalidLengthsOffsetsAndNullConvenienceData() {
        assertThrows(IllegalArgumentException.class, () -> new FftRealTransform(0));

        FftRealTransform transform = new FftRealTransform(12);
        FftWorkspace workspace = transform.workspaceRequirement().allocate();
        assertThrows(IllegalArgumentException.class, () -> transform.exec(new double[11], 0, 1.0, true, workspace));
        assertThrows(IllegalArgumentException.class, () -> transform.exec(new double[12], -1, 1.0, true, workspace));

        assertThrows(IllegalArgumentException.class, () -> FftRealTransform.realToHalfComplex(null, 1.0));
        assertThrows(IllegalArgumentException.class, () -> FftRealTransform.halfComplexToReal(null, 1.0));
        assertThrows(IllegalArgumentException.class, () -> FftRealTransform.roundTrip(null));
    }

    private static double[] sample(int size) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = 0.25 + 0.125 * ((13 * i + 7) % 17) - 0.03125 * (i % 5);
        }
        return data;
    }

    private static void assertRoute(int length, FftRoute expectedRoute, int expectedConvolutionLength,
                                    List<Integer> expectedFactors) {
        FftRealTransform transform = new FftRealTransform(length);
        assertEquals(expectedRoute, FftPlanTestUtil.route(transform), "route mismatch for length " + length);
        assertEquals(expectedRoute == FftRoute.BLUESTEIN, FftPlanTestUtil.usesBluestein(transform),
                "usesBluestein mismatch for length " + length);
        assertEquals(expectedConvolutionLength, FftPlanTestUtil.convolutionLength(transform),
                "convolution length mismatch for length " + length);
        assertEquals(expectedFactors, FftPlanTestUtil.factors(transform), "factorization mismatch for length " + length);
    }

    private static double[] directHalfComplex(double[] input) {
        int size = input.length;
        double[] output = new double[size];
        for (int k = 0; k <= size / 2; k++) {
            double sumR = 0.0;
            double sumI = 0.0;
            for (int sample = 0; sample < size; sample++) {
                double angle = -2.0 * Math.PI * k * sample / size;
                sumR += input[sample] * Math.cos(angle);
                sumI += input[sample] * Math.sin(angle);
            }
            if (k == 0) {
                output[0] = sumR;
            } else if (2 * k == size) {
                output[size - 1] = sumR;
            } else {
                output[2 * k - 1] = sumR;
                output[2 * k] = sumI;
            }
        }
        return output;
    }

    private static double[] directHermitian(double[] input, boolean forward, double factor) {
        int size = input.length;
        double[] output = new double[2 * (size / 2 + 1)];
        double sign = forward ? -1.0 : 1.0;
        for (int k = 0; k <= size / 2; k++) {
            double sumR = 0.0;
            double sumI = 0.0;
            for (int sample = 0; sample < size; sample++) {
                double angle = sign * 2.0 * Math.PI * k * sample / size;
                sumR += input[sample] * Math.cos(angle);
                sumI += input[sample] * Math.sin(angle);
            }
            output[2 * k] = sumR * factor;
            output[2 * k + 1] = sumI * factor;
        }
        return output;
    }
}
