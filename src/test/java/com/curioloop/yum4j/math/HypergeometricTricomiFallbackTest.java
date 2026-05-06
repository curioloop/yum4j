package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class HypergeometricTricomiFallbackTest {

    private static final Method ONE_F_ONE_TRICOMI = lookupTricomiMethod();

    @TestFactory
    Stream<DynamicTest> dispatcherKeepsStableRouteWhenPrivateTricomiFails() {
        return CASES.stream().map(testCase -> dynamicTest(testCase.name(), () -> assertFallbackCase(testCase)));
    }

    private static void assertFallbackCase(FallbackCase testCase) {
        IllegalStateException thrown = assertThrows(IllegalStateException.class,
            () -> invokePrivateTricomi(testCase.a(), testCase.b(), testCase.z()));
        assertTrue(thrown.getMessage().contains(testCase.failureFragment()),
            "unexpected tricomi failure for a=" + testCase.a() + ", b=" + testCase.b() + ", z=" + testCase.z()
                + ": " + thrown.getMessage());

        assertEquals(testCase.expectedRoute(), Hypergeometric.debugOneFOneRealRoute(testCase.a(), testCase.b(), testCase.z()),
            "dispatcher should preserve the expected public route after private Tricomi failure exposure");

        double actual = Hypergeometric.oneFOne(testCase.a(), testCase.b(), testCase.z());
        double scale = Math.max(1.0, Math.abs(testCase.expected()));
        double error = Math.abs(actual - testCase.expected());
        assertTrue(error <= testCase.tolerance() * scale,
            "fallback value mismatch: expected=" + testCase.expected() + ", actual=" + actual + ", error=" + error);
    }

    private static double invokePrivateTricomi(double a, double b, double z) {
        try {
            return (double) ONE_F_ONE_TRICOMI.invoke(null, a, b, z);
        } catch (InvocationTargetException ex) {
            Throwable cause = ex.getCause();
            if (cause instanceof IllegalStateException illegalStateException) {
                throw illegalStateException;
            }
            throw new AssertionError("unexpected Tricomi invocation failure", cause);
        } catch (IllegalAccessException ex) {
            throw new AssertionError("unable to invoke private Tricomi helper", ex);
        }
    }

    private static Method lookupTricomiMethod() {
        try {
            Method method = Hypergeometric.class.getDeclaredMethod("oneFOneTricomi", double.class, double.class, double.class);
            method.setAccessible(true);
            return method;
        } catch (ReflectiveOperationException ex) {
            throw new AssertionError("unable to access private Tricomi helper", ex);
        }
    }

    private static final List<FallbackCase> CASES = List.of(
        new FallbackCase("moderate-overflow-fallback", -259.5, -182.75, 3.25,
            "divergent-fallback", 102.78438825660471, 5.0e-12, "oneFOne Tricomi series overflowed"),
        new FallbackCase("large-overflow-fallback", -712.5, -173.75, 4.25,
            "divergent-fallback", 7.927320445317183e7, 5.0e-11, "oneFOne Tricomi series overflowed"),
        new FallbackCase("seed-underflow-rational-route", -704.5, 388.5, 0.625,
            "rational", 0.32112126080134773, 5.0e-12, "oneFOne Tricomi Bessel seed underflowed"),
        new FallbackCase("cache-renormalization-fallback", -3557.5, -1361.75, 34.25,
            "divergent-fallback", 5.126714638923236e39, 1.0e-10,
            "oneFOne Tricomi cache renormalization produced a non-finite Bessel value")
    );

    private record FallbackCase(String name,
                                double a,
                                double b,
                                double z,
                                String expectedRoute,
                                double expected,
                                double tolerance,
                                String failureFragment) {
    }
}