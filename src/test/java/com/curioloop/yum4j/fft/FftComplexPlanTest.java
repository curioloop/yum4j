package com.curioloop.yum4j.fft;

import java.util.List;
import java.util.Arrays;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for FftComplexPlan - basic validation of the FFTPACK implementation.
 */
class FftComplexPlanTest {

    @Test
    void testFactorization() {
        // Test factorization matches upstream behavior
        FftComplexPlan plan2 = new FftComplexPlan(2);
        assertEquals(List.of(2), FftPlanTestUtil.factors(plan2));
        
        FftComplexPlan plan4 = new FftComplexPlan(4);
        assertEquals(List.of(4), FftPlanTestUtil.factors(plan4));
        
        FftComplexPlan plan6 = new FftComplexPlan(6);
        assertEquals(List.of(2, 3), FftPlanTestUtil.factors(plan6));
        
        FftComplexPlan plan8 = new FftComplexPlan(8);
        assertEquals(List.of(8), FftPlanTestUtil.factors(plan8));
        
        FftComplexPlan plan12 = new FftComplexPlan(12);
        assertEquals(List.of(4, 3), FftPlanTestUtil.factors(plan12));
        
        FftComplexPlan plan15 = new FftComplexPlan(15);
        assertEquals(List.of(3, 5), FftPlanTestUtil.factors(plan15));
        
        FftComplexPlan plan16 = new FftComplexPlan(16);
        assertEquals(List.of(2, 8), FftPlanTestUtil.factors(plan16));
        
        FftComplexPlan plan35 = new FftComplexPlan(35);
        assertEquals(List.of(5, 7), FftPlanTestUtil.factors(plan35));

        assertEquals(List.of(8, 8), FftPlanTestUtil.factors(new FftComplexPlan(64)));
        assertEquals(List.of(8, 8, 4), FftPlanTestUtil.factors(new FftComplexPlan(256)));
        assertEquals(List.of(8, 4, 3), FftPlanTestUtil.factors(new FftComplexPlan(96)));
        assertEquals(13, new FftComplexPlan(13).getLength());
    }

    @Test
    void twiddleLayoutMatchesUpstreamCfftpCompTwiddle() {
        FftComplexPlan plan = new FftComplexPlan(96);

        assertEquals(3, plan.factorCount());
        assertEquals(8, plan.factorAt(0));
        assertEquals(4, plan.factorAt(1));
        assertEquals(3, plan.factorAt(2));

        assertEquals(77, plan.twiddlePairCountAt(0));
        assertEquals(6, plan.twiddlePairCountAt(1));
        assertEquals(0, plan.twiddlePairCountAt(2));
        assertTwiddle(plan, 0, 0, 1.0, 96L);
        assertTwiddle(plan, 0, 10, 11.0, 96L);
        assertTwiddle(plan, 0, 11, 2.0, 96L);
        assertTwiddle(plan, 1, 0, 8.0, 96L);
        assertTwiddle(plan, 1, 5, 48.0, 96L);
    }

    @Test
    void genericTwiddleLayoutMatchesUpstreamCfftpCompTwiddle() {
        FftComplexPlan plan = new FftComplexPlan(13);

        assertEquals(List.of(13), FftPlanTestUtil.factors(plan));
        assertEquals(0, plan.twiddlePairCountAt(0));
        assertEquals(13, plan.genericTwiddlePairCountAt(0));
        for (int index = 0; index < 13; index++) {
            assertGenericTwiddle(plan, 0, index, index, 13L);
        }
    }

    @Test
    void specialMulAndRotationsMatchUpstreamTemplateBranches() {
        double real = 1.25;
        double imag = -0.75;
        double[] wa = {0.5, -0.875};
        double[] out = new double[4];

        FftComplexPlan.specialMulForwardTo(real, imag, wa, 0, out, 0);
        assertEquals(real * wa[0] + imag * wa[1], out[0], 0.0);
        assertEquals(imag * wa[0] - real * wa[1], out[1], 0.0);

        FftComplexPlan.specialMulBackwardTo(real, imag, wa, 0, out, 2);
        assertEquals(real * wa[0] - imag * wa[1], out[2], 0.0);
        assertEquals(real * wa[1] + imag * wa[0], out[3], 0.0);

        double hsqt2 = 0.707106781186547524400844362104849;
        assertEquals(imag, FftComplexPlan.rotx90ForwardReal(real, imag), 0.0);
        assertEquals(-real, FftComplexPlan.rotx90ForwardImag(real, imag), 0.0);
        assertEquals(-imag, FftComplexPlan.rotx90BackwardReal(real, imag), 0.0);
        assertEquals(real, FftComplexPlan.rotx90BackwardImag(real, imag), 0.0);
        assertEquals(hsqt2 * (real + imag), FftComplexPlan.rotx45ForwardReal(real, imag), 0.0);
        assertEquals(hsqt2 * (imag - real), FftComplexPlan.rotx45ForwardImag(real, imag), 0.0);
        assertEquals(hsqt2 * (real - imag), FftComplexPlan.rotx45BackwardReal(real, imag), 0.0);
        assertEquals(hsqt2 * (imag + real), FftComplexPlan.rotx45BackwardImag(real, imag), 0.0);
        assertEquals(hsqt2 * (imag - real), FftComplexPlan.rotx135ForwardReal(real, imag), 0.0);
        assertEquals(hsqt2 * (-real - imag), FftComplexPlan.rotx135ForwardImag(real, imag), 0.0);
        assertEquals(hsqt2 * (-real - imag), FftComplexPlan.rotx135BackwardReal(real, imag), 0.0);
        assertEquals(hsqt2 * (real - imag), FftComplexPlan.rotx135BackwardImag(real, imag), 0.0);
    }
    
    @Test
    void testSize1Transform() {
        FftComplexPlan plan = new FftComplexPlan(1);
        double[] data = {2.0, 3.0}; // [2 + 3i]
        
        plan.exec(data, 1.0, true);
        assertArrayEquals(new double[]{2.0, 3.0}, data, 1e-15);
        
        plan.exec(data, 2.5, false);
        assertArrayEquals(new double[]{5.0, 7.5}, data, 1e-15);
    }
    
    @Test
    void testSize2ForwardTransform() {
        FftComplexPlan plan = new FftComplexPlan(2);
        
        // Test [1, 0] -> [1, 1] (sum and diff)
        double[] data = {1.0, 0.0, 0.0, 0.0};
        plan.exec(data, 1.0, true);
        assertArrayEquals(new double[]{1.0, 0.0, 1.0, 0.0}, data, 1e-15);
        
        // Test [1, 1] -> [2, 0]
        data = new double[]{1.0, 0.0, 1.0, 0.0};
        plan.exec(data, 1.0, true);
        assertArrayEquals(new double[]{2.0, 0.0, 0.0, 0.0}, data, 1e-15);
        
        // Test complex values
        data = new double[]{1.0, 2.0, 3.0, 4.0}; // [1+2i, 3+4i]
        plan.exec(data, 1.0, true);
        assertArrayEquals(new double[]{4.0, 6.0, -2.0, -2.0}, data, 1e-15);
    }
    
    @Test
    void testSize2BackwardTransform() {
        FftComplexPlan plan = new FftComplexPlan(2);
        
        // Test inverse property: backward(forward(x)) = x (with proper scaling)
        double[] original = {1.0, 2.0, 3.0, 4.0};
        double[] data = Arrays.copyOf(original, original.length);
        
        plan.exec(data, 1.0, true);     // forward
        plan.exec(data, 1.0 / 2, false); // backward with normalization
        
        assertArrayEquals(original, data, 1e-14);
    }
    
    @Test
    void testSize3Transform() {
        FftComplexPlan plan = new FftComplexPlan(3);
        
        // Test simple case [1, 0, 0]
        double[] data = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        plan.exec(data, 1.0, true);
        // Result should be [1, 1, 1] for DFT of delta function
        assertArrayEquals(new double[]{1.0, 0.0, 1.0, 0.0, 1.0, 0.0}, data, 1e-15);
    }
    
    @Test
    void testRoundTrip() {
        int[] sizes = {2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 15, 16, 17, 25, 35, 49, 64, 77, 88, 96, 97, 121, 143,
            169, 256};
        
        for (int size : sizes) {
            FftComplexPlan plan = new FftComplexPlan(size);
            
            double[] original = new double[2 * size];
            for (int i = 0; i < 2 * size; i++) {
                original[i] = Math.sin(i * 0.7) + Math.cos(i * 1.3);
            }
            
            double[] data = Arrays.copyOf(original, original.length);
            plan.exec(data, 1.0, true);   // forward
            plan.exec(data, 1.0 / size, false); // inverse with normalization
            
            assertArrayEquals(original, data, 1e-13, "Round trip failed for size " + size);
        }
    }
    
    @Test
    void testParsevalTheorem() {
        int[] sizes = {2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 15, 16, 17, 25, 35, 49, 64, 77, 88, 96, 97, 121, 143,
            169, 256};
        
        for (int size : sizes) {
            FftComplexPlan plan = new FftComplexPlan(size);
            
            double[] data = new double[2 * size];
            double originalEnergy = 0.0;
            for (int i = 0; i < size; i++) {
                data[2 * i] = Math.sin(i * 0.5);
                data[2 * i + 1] = Math.cos(i * 0.3);
                originalEnergy += data[2 * i] * data[2 * i] + data[2 * i + 1] * data[2 * i + 1];
            }
            
            plan.exec(data, 1.0, true);
            
            double transformedEnergy = 0.0;
            for (int i = 0; i < size; i++) {
                transformedEnergy += data[2 * i] * data[2 * i] + data[2 * i + 1] * data[2 * i + 1];
            }
            
            // Transformed energy should be size * original energy
            assertEquals(size * originalEnergy, transformedEnergy, Math.max(1e-12, size * 2e-13), 
                "Parseval's theorem failed for size " + size);
        }
    }

    @Test
    void forwardTransformMatchesDirectOracleForImplementedRadices() {
        int[] sizes = {2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 15, 16, 17, 25, 35, 49, 64, 77, 88, 96, 97, 121, 143,
            169, 256};
        for (int size : sizes) {
            FftComplexPlan plan = new FftComplexPlan(size);
            double[] data = sample(size);
            double[] expected = directDft(data, true);
            plan.exec(data, 1.0, true);
            assertArrayEquals(expected, data, Math.max(1e-12, size * 2e-13), "Forward mismatch for size " + size);
        }
    }
    
    @Test
    void testUnsupportedSizes() {
        // Sizes that require passes not yet implemented should throw exceptions
        int[] unsupportedSizes = {};
        
        for (int size : unsupportedSizes) {
            FftComplexPlan plan = new FftComplexPlan(size);
            double[] data = new double[2 * size];
            Arrays.fill(data, 1.0);
            
            assertThrows(UnsupportedOperationException.class, () -> {
                plan.exec(data, 1.0, true);
            }, "Size " + size + " should throw UnsupportedOperationException");
        }
    }

    private static double[] sample(int size) {
        double[] data = new double[2 * size];
        for (int i = 0; i < size; i++) {
            data[2 * i] = 0.25 + 0.125 * ((11 * i + 3) % 19);
            data[2 * i + 1] = -0.5 + 0.0625 * ((7 * i + 5) % 23);
        }
        return data;
    }

    private static void assertTwiddle(FftComplexPlan plan, int factorIndex, int pairIndex, double numerator,
                                      long denominator) {
        double angle = 2.0 * Math.PI * numerator / denominator;
        assertEquals(Math.cos(angle), plan.twiddleRealAt(factorIndex, pairIndex), 1e-15);
        assertEquals(Math.sin(angle), plan.twiddleImagAt(factorIndex, pairIndex), 1e-15);
    }

    private static void assertGenericTwiddle(FftComplexPlan plan, int factorIndex, int pairIndex, double numerator,
                                             long denominator) {
        double angle = 2.0 * Math.PI * numerator / denominator;
        assertEquals(Math.cos(angle), plan.genericTwiddleRealAt(factorIndex, pairIndex), 1e-15);
        assertEquals(Math.sin(angle), plan.genericTwiddleImagAt(factorIndex, pairIndex), 1e-15);
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