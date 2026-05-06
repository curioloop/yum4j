package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class HypergeometricOneFOneRouteAuditTest {

    @TestFactory
    Stream<DynamicTest> oneFOneRealDispatcherKeepsBoostAlignedRoutePrecedence() {
        return CASES.stream().map(testCase -> dynamicTest(testCase.route(), () -> assertCase(testCase)));
    }

    private static void assertCase(RouteCase testCase) {
        String actualRoute = Hypergeometric.debugOneFOneRealRoute(testCase.a(), testCase.b(), testCase.z());
        assertEquals(testCase.route(), actualRoute,
            "unexpected route for a=" + testCase.a() + ", b=" + testCase.b() + ", z=" + testCase.z());

        double actualValue = Hypergeometric.oneFOne(testCase.a(), testCase.b(), testCase.z());
        double scale = Math.max(1.0, Math.abs(testCase.expected()));
        double error = Math.abs(actualValue - testCase.expected());
        assertTrue(error <= testCase.tolerance() * scale,
            testCase.route() + " value mismatch: expected=" + testCase.expected() + ", actual=" + actualValue + ", error=" + error);
    }

    private static final List<RouteCase> CASES = List.of(
        new RouteCase("exact-or-terminating", -80.5, -80.5, -120.0, 7.667648073722e-53, 5.0e-12),
        new RouteCase("pre-13.3.6", 0.009, -120.5, -40.0, 1.0036466863256324, 5.0e-11),
        new RouteCase("small-a-negative-b", 0.009, -120.5, -120.0, 3842063.917670071, 5.0e-11),
        new RouteCase("large-z-asymptotic", -2.5, -10.25, 80.0, 1.5998878829272933e43, 5.0e-11),
        new RouteCase("rational", -80.5, -120.5, -0.25, 0.8461400307044736, 5.0e-12),
        new RouteCase("pade", 1.0, 5.5, -30.0, 0.13388665330787308, 5.0e-13),
        new RouteCase("negative-z-kummer", -80.5, -120.5, -120.0, 5.118687077685334e-40, 5.0e-11),
        new RouteCase("generic-series", -5.0, -10.0, -2.0, 0.34814814814814815, 5.0e-13),
        new RouteCase("tricomi", -80.5, -55.25, 2.5, 39.78121721301848, 5.0e-12),
        new RouteCase("divergent-fallback", -80.5, -55.25, 20.0, -2.0463255243994344e14, 5.0e-12),
        new RouteCase("alt-13.3.6", -40.5, -40.25, 0.5, 1.653882693514888, 5.0e-12),
        new RouteCase("large-abz", 0.75, 40.0, 80.0, 2205834.7641538726, 5.0e-11),
        new RouteCase("checked-series", -80.5, -120.5, 2.5, 5.281964364347567, 5.0e-12),
        new RouteCase("generic-series", -80.5, 4.5, -0.25, 30.328448672632476, 5.0e-12)
    );

    private record RouteCase(String route, double a, double b, double z, double expected, double tolerance) {
    }
}