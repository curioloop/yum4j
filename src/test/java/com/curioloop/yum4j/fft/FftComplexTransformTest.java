package com.curioloop.yum4j.fft;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FftComplexTransformTest {
    @Test
    void selectionMatchesPocketfftHeuristicForRepresentativeLengths() {
        assertRoute(12, FftRoute.DIRECT, 12, List.of(4, 3));
        assertRoute(47, FftRoute.DIRECT, 47, List.of(47));
        assertRoute(64, FftRoute.SPLIT_RADIX, 64, List.of());
        assertRoute(77, FftRoute.DIRECT, 77, List.of(7, 11));
        assertRoute(96, FftRoute.DIRECT, 96, List.of(8, 4, 3));
        assertRoute(120, FftRoute.DIRECT, 120, List.of(8, 3, 5));
        assertRoute(256, FftRoute.SPLIT_RADIX, 256, List.of());
        assertRoute(257, FftRoute.BLUESTEIN, 525, List.of());
        assertRoute(509, FftRoute.BLUESTEIN, 1024, List.of());
        assertRoute(1024, FftRoute.SPLIT_RADIX, 1024, List.of());
        assertRoute(4096, FftRoute.SPLIT_RADIX, 4096, List.of());
    }

    @Test
    void bluesteinForwardMatchesDirectOracle() {
        for (int size : new int[] { 257, 509 }) {
            FftComplexTransform transform = new FftComplexTransform(size);
            double[] data = sample(size);
            double[] expected = directDft(data, true);
            transform.exec(data, 1.0, true);
            assertArrayEquals(expected, data, size * 5e-12, "Bluestein mismatch for size " + size);
        }
    }

    @Test
    void flexibleTransformRoundTripsDirectAndBluesteinLengths() {
        for (int size : new int[] { 12, 77, 143, 257, 509 }) {
            FftComplexTransform transform = new FftComplexTransform(size);
            double[] original = sample(size);
            double[] data = original.clone();
            transform.exec(data, 1.0, true);
            transform.exec(data, 1.0 / size, false);
            assertArrayEquals(original, data, size * 2e-13, "Roundtrip mismatch for size " + size);
        }
    }

    @Test
    void offsetNoWorkspaceExecCoversAllComplexStrategies() {
        for (int size : new int[] {64, 77, 257}) {
            FftComplexTransform transform = new FftComplexTransform(size);
            double[] original = sample(size);
            double[] expected = directDft(original, true);
            double[] actual = new double[2 * size + 8];
            Arrays.fill(actual, -9999.0);
            System.arraycopy(original, 0, actual, 4, original.length);

            transform.exec(actual, 4, 0.5, true);

            for (int index = 0; index < expected.length; index++) {
                expected[index] *= 0.5;
            }
            assertEquals(-9999.0, actual[0], 0.0);
            assertEquals(-9999.0, actual[actual.length - 1], 0.0);
            assertArrayEquals(expected, Arrays.copyOfRange(actual, 4, 4 + expected.length), size * 5e-12,
                    "offset exec mismatch for size " + size);
        }
    }

    @Test
    void rejectsInvalidLengthsAndOffsetsAcrossComplexStrategies() {
        assertThrows(IllegalArgumentException.class, () -> new FftComplexTransform(0));

        for (int size : new int[] {64, 77, 257}) {
            FftComplexTransform transform = new FftComplexTransform(size);
            FftWorkspace workspace = transform.workspaceRequirement().allocate();
            double[] tooShort = new double[2 * size - 1];
            double[] data = new double[2 * size];

            assertThrows(IllegalArgumentException.class, () -> transform.exec(tooShort, 0, 1.0, true));
            assertThrows(IllegalArgumentException.class, () -> transform.exec(data, -1, 1.0, true, workspace));
        }
    }

    @Test
    void bluesteinPlanConvenienceOverloadsMatchWorkspaceExecution() {
        FftBluesteinPlan plan = new FftBluesteinPlan(257);
        assertEquals(257, plan.length());

        double[] expected = sample(257);
        FftWorkspace workspace = plan.workspaceRequirement().allocate();
        workspace.reset();
        plan.exec(expected, 0, 0.75, true, workspace);

        double[] actual = sample(257);
        plan.exec(actual, 0.75, true);
        assertArrayEquals(expected, actual, 257 * 5e-12);

        double[] expectedBackward = directDft(sample(257), false);
        for (int index = 0; index < expectedBackward.length; index++) {
            expectedBackward[index] *= 0.25;
        }
        double[] backward = sample(257);
        workspace.reset();
        plan.exec(backward, 0.25, false, workspace);
        assertArrayEquals(expectedBackward, backward, 257 * 5e-12);

        FftBluesteinPlan oddConvolutionPlan = new FftBluesteinPlan(5);
        assertEquals(9, oddConvolutionPlan.convolutionLength());
        double[] odd = sample(5);
        double[] oddExpected = directDft(odd, true);
        oddConvolutionPlan.exec(odd, 1.0, true);
        assertArrayEquals(oddExpected, odd, 5e-12);

        assertThrows(IllegalArgumentException.class, () -> new FftBluesteinPlan(0));
        assertThrows(IllegalArgumentException.class,
            () -> plan.exec(new double[2 * plan.length()], -1, 1.0, true, plan.workspaceRequirement().allocate()));
    }

    @Test
    void directBoundaryRoundTripsLength64() {
        FftComplexTransform transform = new FftComplexTransform(64);
        double[] original = sample(64);
        double[] data = original.clone();

        assertEquals(FftRoute.SPLIT_RADIX, FftPlanTestUtil.route(transform));

        transform.exec(data, 1.0, true);
        transform.exec(data, 1.0 / 64, false);

        assertArrayEquals(original, data, 2e-12);
    }

    @Test
    void directBoundaryUsesDirectRouteForLength96() {
        FftComplexTransform transform = new FftComplexTransform(96);

        assertEquals(FftRoute.DIRECT, FftPlanTestUtil.route(transform));
    }

    private static double[] sample(int size) {
        double[] data = new double[2 * size];
        for (int i = 0; i < size; i++) {
            data[2 * i] = 0.25 + 0.125 * ((11 * i + 3) % 19);
            data[2 * i + 1] = -0.5 + 0.0625 * ((7 * i + 5) % 23);
        }
        return data;
    }

    private static void assertRoute(int length, FftRoute expectedRoute, int expectedConvolutionLength,
                                    List<Integer> expectedFactors) {
        FftComplexTransform transform = new FftComplexTransform(length);
        assertEquals(expectedRoute, FftPlanTestUtil.route(transform), "route mismatch for length " + length);
        assertEquals(expectedRoute == FftRoute.BLUESTEIN, FftPlanTestUtil.usesBluestein(transform),
                "usesBluestein mismatch for length " + length);
        assertEquals(expectedConvolutionLength, FftPlanTestUtil.convolutionLength(transform),
                "convolution length mismatch for length " + length);
        assertEquals(expectedFactors, FftPlanTestUtil.factors(transform), "factorization mismatch for length " + length);
    }

    private static double[] directDft(double[] input, boolean forward) {
        int size = input.length / 2;
        double[] output = new double[input.length];
        double sign = forward ? -1.0 : 1.0;
        for (int k = 0; k < size; k++) {
            double sumR = 0.0;
            double sumI = 0.0;
            for (int n = 0; n < size; n++) {
                double angle = sign * 2.0 * Math.PI * k * n / size;
                double wr = Math.cos(angle);
                double wi = Math.sin(angle);
                double xr = input[2 * n];
                double xi = input[2 * n + 1];
                sumR += xr * wr - xi * wi;
                sumI += xr * wi + xi * wr;
            }
            output[2 * k] = sumR;
            output[2 * k + 1] = sumI;
        }
        return output;
    }
}