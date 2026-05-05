/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas;

/**
 * Small compatibility wrapper for fused multiply-add.
 */
public interface FMA {

    static double op(double a, double b, double c) {
        return Math.fma(a, b, c);
    }
}