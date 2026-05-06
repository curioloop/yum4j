package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class HankelRouteAuditTest {

    @Test
    void representativeRoutesStayPinned() {
        assertEquals("positive-x", Hankel.debugCylHankel1Route(0.75, 1.5));
        assertEquals("negative-x-non-integer-order", Hankel.debugCylHankel1Route(0.5, -2.0));
        assertEquals("negative-x-integer-order", Hankel.debugCylHankel2Route(1.0, -2.0));
        assertEquals("origin", Hankel.debugCylHankel1Route(0.0, 0.0));
        assertEquals("spherical-reduction/positive-x", Hankel.debugSphHankel1Route(0.0, 2.0));
    }
}