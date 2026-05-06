package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BesselRouteAuditTest {

    @Test
    void jyRoutesRemainPinnedToRepresentativeRegimes() {
        assertEquals("small-z-series-j-only", Bessel.debugJRoute(0.75, 0.25));
        assertEquals("small-z-series-jy", Bessel.debugYRoute(0.75, 1.0e-12));
        assertEquals("large-x-asymptotic", Bessel.debugYRoute(0.75, 100.0));
        assertEquals("hankel-pq", Bessel.debugYRoute(5.0, 20.0));
        assertEquals("negative-order-reflection/temme-cf1", Bessel.debugJRoute(-0.25, 1.5));
        assertEquals("negative-order-reflection/cf1-cf2", Bessel.debugJRoute(-5.25, 6.0));
    }

    @Test
    void ikRoutesRemainPinnedToRepresentativeRegimes() {
        assertEquals("half-integer-closed-form", Bessel.debugIRoute(0.5, 2.0));
        assertEquals("i0-fast-path", Bessel.debugIRoute(0.0, 2.0));
        assertEquals("i1-fast-path", Bessel.debugIRoute(1.0, 2.0));
        assertEquals("small-z-series", Bessel.debugIRoute(4.0, 0.5));
        assertEquals("temme-ik", Bessel.debugIRoute(0.75, 1.5));
        assertEquals("cf2-ik", Bessel.debugIRoute(0.75, 4.0));
        assertEquals("asymptotic-large-x", Bessel.debugIRoute(0.75, 5000.0));
        assertEquals("integer-order-kn", Bessel.debugKRoute(2.0, 2.0));
        assertEquals("negative-order-reflection/temme-ik", Bessel.debugKRoute(-0.75, 1.5));
        assertEquals("negative-order-reflection/cf2-ik", Bessel.debugKRoute(-0.75, 4.0));
    }
}