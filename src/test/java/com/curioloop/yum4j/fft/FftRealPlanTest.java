package com.curioloop.yum4j.fft;

import java.util.List;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for FftRealPlan - faithful port of pocketfft rfftp.
 */
class FftRealPlanTest {
    
    private static final double TOLERANCE = 1e-12;
    
    @Test
    void testFactorization() {
        assertEquals(List.of(2), FftPlanTestUtil.factors(new FftRealPlan(2)));
        assertEquals(List.of(4), FftPlanTestUtil.factors(new FftRealPlan(4)));
        assertEquals(List.of(2, 3), FftPlanTestUtil.factors(new FftRealPlan(6)));
        assertEquals(List.of(2, 4), FftPlanTestUtil.factors(new FftRealPlan(8)));
        assertEquals(List.of(4, 3), FftPlanTestUtil.factors(new FftRealPlan(12)));
        assertEquals(List.of(3, 5), FftPlanTestUtil.factors(new FftRealPlan(15)));
        assertEquals(List.of(4, 4), FftPlanTestUtil.factors(new FftRealPlan(16)));
        assertEquals(List.of(2, 3, 5), FftPlanTestUtil.factors(new FftRealPlan(30)));
        assertEquals(List.of(5, 7), FftPlanTestUtil.factors(new FftRealPlan(35)));
        assertEquals(35, new FftRealPlan(35).getLength());
    }
    
    @Test
    void testLength1() {
        double[] data = {5.0};
        FftRealPlan plan = new FftRealPlan(1);
        plan.exec(data, 2.0, true);
        assertEquals(10.0, data[0], TOLERANCE);
        
        plan.exec(data, 0.5, false);
        assertEquals(5.0, data[0], TOLERANCE);
    }
    
    @Test
    void testLength2() {
        // Test [1, 2] -> [3, -1] pattern
        double[] data = {1.0, 2.0};
        FftRealPlan plan = new FftRealPlan(2);
        
        // Forward transform
        plan.exec(data, 1.0, true);
        assertEquals(3.0, data[0], TOLERANCE); // DC component
        assertEquals(-1.0, data[1], TOLERANCE); // Real part of Nyquist
        
        // Inverse transform with scaling
        plan.exec(data, 0.5, false);
        assertEquals(1.0, data[0], TOLERANCE);
        assertEquals(2.0, data[1], TOLERANCE);
    }
    
    @Test
    void testLength4() {
        // Test [1, 2, 3, 4] real input
        double[] data = {1.0, 2.0, 3.0, 4.0};
        double[] original = data.clone();
        FftRealPlan plan = new FftRealPlan(4);
        
        // Forward transform
        plan.exec(data, 1.0, true);
        // Expected FFTPACK half-complex layout: [r0, r1, i1, r2]
        assertNotEquals(0.0, data[0]); // DC should be non-zero
        
        // Inverse transform
        plan.exec(data, 0.25, false);
        for (int i = 0; i < original.length; i++) {
            assertEquals(original[i], data[i], TOLERANCE);
        }
    }
    
    @Test
    void testLength8() {
        // Test length 8 (uses factor 4 and 2)
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        double[] original = data.clone();
        FftRealPlan plan = new FftRealPlan(8);
        
        // Forward transform
        plan.exec(data, 1.0, true);
        
        // Inverse transform
        plan.exec(data, 1.0 / 8.0, false);
        for (int i = 0; i < original.length; i++) {
            assertEquals(original[i], data[i], TOLERANCE);
        }
    }
    
    @Test
    void testRoundTripVariousLengths() {
        // Test various lengths that use different factorizations
        int[] lengths = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 25, 30, 32, 77};
        
        for (int n : lengths) {
            double[] data = new double[n];
            // Fill with some test pattern
            for (int i = 0; i < n; i++) {
                data[i] = Math.sin(2.0 * Math.PI * i / n) + 0.5 * Math.cos(4.0 * Math.PI * i / n);
            }
            double[] original = data.clone();
            
            FftRealPlan plan = new FftRealPlan(n);
            
            // Forward
            plan.exec(data, 1.0, true);
            
            // Inverse
            plan.exec(data, 1.0 / n, false);
            
            // Check roundtrip
            for (int i = 0; i < n; i++) {
                assertEquals(original[i], data[i], TOLERANCE, 
                           "Roundtrip failed for length " + n + " at index " + i);
            }
        }
    }

    @Test
    void forwardMatchesDirectDftHalfComplexLayout() {
        int[] lengths = {2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 16, 17, 25, 30, 32, 77};
        for (int n : lengths) {
            double[] data = new double[n];
            for (int i = 0; i < n; i++) {
                data[i] = 0.125 * ((13 * i + 5) % 17) - 0.0625 * (i % 4);
            }
            double[] expected = directHalfComplex(data);

            FftRealPlan plan = new FftRealPlan(n);
            plan.exec(data, 1.0, true);

            for (int i = 0; i < n; i++) {
                assertEquals(expected[i], data[i], Math.max(TOLERANCE, n * 8e-14),
                        "Forward mismatch for length " + n + " at half-complex index " + i);
            }
        }
    }
    
    @Test
    void testDCComponent() {
        // DC component should be the sum of all real values
        double[] data = {1.0, -1.0, 2.0, -2.0};
        double expectedDC = 0.0; // sum = 0
        
        FftRealPlan plan = new FftRealPlan(4);
        plan.exec(data, 1.0, true);
        
        assertEquals(expectedDC, data[0], TOLERANCE);
    }
    
    @Test
    void testConstantInput() {
        // Constant input should have only DC component
        int n = 8;
        double[] data = new double[n];
        double constant = 5.0;
        for (int i = 0; i < n; i++) {
            data[i] = constant;
        }
        
        FftRealPlan plan = new FftRealPlan(n);
        plan.exec(data, 1.0, true);
        
        assertEquals(constant * n, data[0], TOLERANCE); // DC component
        for (int i = 1; i < n; i++) {
            assertEquals(0.0, data[i], TOLERANCE); // All other components should be zero
        }
    }
    
    @Test
    void testIllegalArguments() {
        assertThrows(IllegalArgumentException.class, () -> new FftRealPlan(0));
        assertThrows(IllegalArgumentException.class, () -> new FftRealPlan(-1));
    }
    
    @Test
    void testGenericFactorsRoundTrip() {
        for (int n : new int[] {7, 11, 13, 17, 19, 29}) {
            FftRealPlan plan = new FftRealPlan(n);
            double[] data = new double[n];
            for (int i = 0; i < n; i++) {
                data[i] = Math.sin(0.31 * i) - 0.25 * Math.cos(0.77 * i);
            }
            double[] original = data.clone();

            plan.exec(data, 1.0, true);
            plan.exec(data, 1.0 / n, false);

            for (int i = 0; i < n; i++) {
                assertEquals(original[i], data[i], Math.max(TOLERANCE, n * 5e-14),
                        "Generic roundtrip failed for length " + n + " at index " + i);
            }
        }
    }
    
    @Test
    void testRealTransformWrapper() {
        // Test the FftRealTransform wrapper class
        double[] data = {1.0, 2.0, 3.0, 4.0};
        double[] original = data.clone();
        
        // Test real to half-complex
        FftRealTransform.realToHalfComplex(data, 1.0);
        
        // Test half-complex to real
        FftRealTransform.halfComplexToReal(data, 0.25);
        
        for (int i = 0; i < original.length; i++) {
            assertEquals(original[i], data[i], TOLERANCE);
        }
    }
    
    @Test
    void testRoundTripWrapper() {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        double[] original = data.clone();
        
        FftRealTransform.roundTrip(data);
        
        for (int i = 0; i < original.length; i++) {
            assertEquals(original[i], data[i], TOLERANCE);
        }
    }

    private static double[] directHalfComplex(double[] input) {
        int n = input.length;
        double[] output = new double[n];
        for (int k = 0; k <= n / 2; k++) {
            double real = 0.0;
            double imag = 0.0;
            for (int sample = 0; sample < n; sample++) {
                double angle = -2.0 * Math.PI * k * sample / n;
                real += input[sample] * Math.cos(angle);
                imag += input[sample] * Math.sin(angle);
            }
            if (k == 0) {
                output[0] = real;
            } else if (2 * k == n) {
                output[n - 1] = real;
            } else {
                output[2 * k - 1] = real;
                output[2 * k] = imag;
            }
        }
        return output;
    }
}