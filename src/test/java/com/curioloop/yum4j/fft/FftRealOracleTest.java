package com.curioloop.yum4j.fft;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Oracle tests for real FFT using exact values from pocketfft reference.
 */
class FftRealOracleTest {
    
    private static final double TOLERANCE = 1e-6;
    
    @Test
    void testLength2Oracle() {
        // Input: [1.0, 2.0] -> Expected: [3.000000, -1.000000]
        double[] data = {1.0, 2.0};
        FftRealPlan plan = new FftRealPlan(2);
        
        plan.exec(data, 1.0, true);
        assertEquals(3.0, data[0], TOLERANCE);
        assertEquals(-1.0, data[1], TOLERANCE);
        
        // Roundtrip test
        plan.exec(data, 0.5, false);
        assertEquals(1.0, data[0], TOLERANCE);
        assertEquals(2.0, data[1], TOLERANCE);
    }
    
    @Test
    void testLength3Oracle() {
        // Input: [1.0, 2.0, 3.0] -> Expected: [6.000000, -1.500000, 0.866025]
        double[] data = {1.0, 2.0, 3.0};
        FftRealPlan plan = new FftRealPlan(3);
        
        plan.exec(data, 1.0, true);
        assertEquals(6.0, data[0], TOLERANCE);
        assertEquals(-1.5, data[1], TOLERANCE);
        assertEquals(0.866025, data[2], TOLERANCE);
        
        // Roundtrip test
        plan.exec(data, 1.0/3.0, false);
        assertEquals(1.0, data[0], TOLERANCE);
        assertEquals(2.0, data[1], TOLERANCE);
        assertEquals(3.0, data[2], TOLERANCE);
    }
    
    @Test 
    void testLength4Oracle() {
        // Input: [1.0, 2.0, 3.0, 4.0] -> Expected: [10.000000, -2.000000, 2.000000, -2.000000]
        double[] data = {1.0, 2.0, 3.0, 4.0};
        FftRealPlan plan = new FftRealPlan(4);
        
        plan.exec(data, 1.0, true);
        assertEquals(10.0, data[0], TOLERANCE);
        assertEquals(-2.0, data[1], TOLERANCE);
        assertEquals(2.0, data[2], TOLERANCE);
        assertEquals(-2.0, data[3], TOLERANCE);
        
        // Roundtrip test
        plan.exec(data, 0.25, false);
        assertEquals(1.0, data[0], TOLERANCE);
        assertEquals(2.0, data[1], TOLERANCE);
        assertEquals(3.0, data[2], TOLERANCE);
        assertEquals(4.0, data[3], TOLERANCE);
    }
    
    @Test 
    void testLength5Oracle() {
        // Input: [1.0, 2.0, 3.0, 4.0, 5.0] -> Expected: [15.000000, -2.500000, 3.440955, -2.500000, 0.812299]
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0};
        FftRealPlan plan = new FftRealPlan(5);
        
        plan.exec(data, 1.0, true);
        assertEquals(15.0, data[0], TOLERANCE);
        assertEquals(-2.5, data[1], TOLERANCE);
        assertEquals(3.440955, data[2], TOLERANCE);
        assertEquals(-2.5, data[3], TOLERANCE);
        assertEquals(0.812299, data[4], TOLERANCE);
        
        // Roundtrip test
        plan.exec(data, 0.2, false);
        assertEquals(1.0, data[0], TOLERANCE);
        assertEquals(2.0, data[1], TOLERANCE);
        assertEquals(3.0, data[2], TOLERANCE);
        assertEquals(4.0, data[3], TOLERANCE);
        assertEquals(5.0, data[4], TOLERANCE);
    }
    
    @Test 
    void testLength8Oracle() {
        // Input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] -> Expected: [36.000000, -4.000000, 9.656854, -4.000000, 4.000000, -4.000000, 1.656854, -4.000000]
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        FftRealPlan plan = new FftRealPlan(8);
        
        plan.exec(data, 1.0, true);
        assertEquals(36.0, data[0], TOLERANCE);
        assertEquals(-4.0, data[1], TOLERANCE);
        assertEquals(9.656854, data[2], TOLERANCE);
        assertEquals(-4.0, data[3], TOLERANCE);
        assertEquals(4.0, data[4], TOLERANCE);
        assertEquals(-4.0, data[5], TOLERANCE);
        assertEquals(1.656854, data[6], TOLERANCE);
        assertEquals(-4.0, data[7], TOLERANCE);
        
        // Roundtrip test
        plan.exec(data, 0.125, false);
        assertEquals(1.0, data[0], TOLERANCE);
        assertEquals(2.0, data[1], TOLERANCE);
        assertEquals(3.0, data[2], TOLERANCE);
        assertEquals(4.0, data[3], TOLERANCE);
        assertEquals(5.0, data[4], TOLERANCE);
        assertEquals(6.0, data[5], TOLERANCE);
        assertEquals(7.0, data[6], TOLERANCE);
        assertEquals(8.0, data[7], TOLERANCE);
    }
}