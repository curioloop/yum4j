/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

final class SIMD {

    private static final String SIMD_PROPERTY = "yum4j.vector";
    private static final boolean SIMD_ENABLED =
        !"false".equalsIgnoreCase(System.getProperty(SIMD_PROPERTY, "true"));

    private static final boolean ZDSCAL_SUPPORTED_AT_INIT = detectZdscal();
    private static final boolean ZSCAL_SUPPORTED_AT_INIT = detectZscal();
    private static final boolean ZAXPY_SUPPORTED_AT_INIT = detectZaxpy();
    private static final boolean ZGEMV_SUPPORTED_AT_INIT = detectZgemv();

    private SIMD() {
    }

    static boolean supportZdscal() {
        return ZDSCAL_SUPPORTED_AT_INIT;
    }

    static boolean supportZscal() {
        return ZSCAL_SUPPORTED_AT_INIT;
    }

    static boolean supportZaxpy() {
        return ZAXPY_SUPPORTED_AT_INIT;
    }

    static boolean supportZgemv() {
        return ZGEMV_SUPPORTED_AT_INIT;
    }

    private static boolean detectZdscal() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return ZdscalSIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }

    private static boolean detectZscal() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return ZscalSIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }

    private static boolean detectZaxpy() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return ZaxpySIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }

    private static boolean detectZgemv() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return ZgemvSIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }
}

final class Gate {

    static final VectorSpecies<Double> SPECIES = preferredSpecies256Cap();
    static final int LANES = SPECIES.length();

    private Gate() {
    }

    private static VectorSpecies<Double> preferredSpecies256Cap() {
        VectorSpecies<Double> preferred = DoubleVector.SPECIES_PREFERRED;
        if (preferred.length() >= DoubleVector.SPECIES_256.length()) {
            return DoubleVector.SPECIES_256;
        }
        return preferred;
    }
}