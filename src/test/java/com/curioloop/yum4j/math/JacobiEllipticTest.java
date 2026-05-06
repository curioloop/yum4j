package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JacobiEllipticTest {

    @Test
    void specialCasesMatchElementaryLimits() {
        double theta = 0.7;

        Double3 zeroModulus = JacobiElliptic.snCnDn(0.0, theta);
        assertClose(Math.sin(theta), zeroModulus._1(), 5.0e-15);
        assertClose(Math.cos(theta), zeroModulus._2(), 5.0e-15);
        assertEquals(1.0, zeroModulus._3(), 0.0);

        Double3 unitModulus = JacobiElliptic.snCnDn(1.0, theta);
        double sech = 1.0 / Math.cosh(theta);
        assertClose(Math.tanh(theta), unitModulus._1(), 5.0e-15);
        assertClose(sech, unitModulus._2(), 5.0e-15);
        assertClose(sech, unitModulus._3(), 5.0e-15);
    }

    @Test
    void zeroArgumentQuotientSingularitiesMatchBoostBehavior() {
        assertEquals(0.0, JacobiElliptic.sn(0.7, 0.0), 0.0);
        assertTrue(Double.isInfinite(JacobiElliptic.ns(0.7, 0.0)) && JacobiElliptic.ns(0.7, 0.0) > 0.0);
        assertTrue(Double.isInfinite(JacobiElliptic.cs(0.7, 0.0)) && JacobiElliptic.cs(0.7, 0.0) > 0.0);
        assertTrue(Double.isInfinite(JacobiElliptic.ds(0.7, 0.0)) && JacobiElliptic.ds(0.7, 0.0) > 0.0);
        assertEquals(0.0, JacobiElliptic.sc(0.7, 0.0), 0.0);
        assertEquals(0.0, JacobiElliptic.sd(0.7, 0.0), 0.0);
    }

    @Test
    void scalarSurfaceMatchesTripleSurface() {
        double k = 0.6;
        double theta = 0.8;
        Double3 values = JacobiElliptic.snCnDn(k, theta);

        assertClose(values._1(), JacobiElliptic.sn(k, theta), 5.0e-15);
        assertClose(values._2(), JacobiElliptic.cn(k, theta), 5.0e-15);
        assertClose(values._3(), JacobiElliptic.dn(k, theta), 5.0e-15);
    }

    @Test
    void quotientFamiliesReduceToSnCnDnRatios() {
        double k = 0.75;
        double theta = 0.4;
        Double3 values = JacobiElliptic.snCnDn(k, theta);

        assertClose(values._2() / values._3(), JacobiElliptic.cd(k, theta), 5.0e-15);
        assertClose(values._3() / values._2(), JacobiElliptic.dc(k, theta), 5.0e-15);
        assertClose(1.0 / values._1(), JacobiElliptic.ns(k, theta), 5.0e-15);
        assertClose(values._1() / values._3(), JacobiElliptic.sd(k, theta), 5.0e-15);
        assertClose(values._3() / values._1(), JacobiElliptic.ds(k, theta), 5.0e-15);
        assertClose(1.0 / values._2(), JacobiElliptic.nc(k, theta), 5.0e-15);
        assertClose(1.0 / values._3(), JacobiElliptic.nd(k, theta), 5.0e-15);
        assertClose(values._1() / values._2(), JacobiElliptic.sc(k, theta), 5.0e-15);
        assertClose(values._2() / values._1(), JacobiElliptic.cs(k, theta), 5.0e-15);
    }

    @Test
    void modulusAboveOneUsesReciprocalTransform() {
        double k = 1.7;
        double theta = 0.4;
        double reciprocal = 1.0 / k;
        Double3 transformed = JacobiElliptic.snCnDn(reciprocal, theta * k);
        Double3 actual = JacobiElliptic.snCnDn(k, theta);

        assertClose(transformed._1() * reciprocal, actual._1(), 5.0e-14);
        assertClose(transformed._3(), actual._2(), 5.0e-14);
        assertClose(transformed._2(), actual._3(), 5.0e-14);
    }

    @Test
    void poincareIdentityHolds() {
        double k = 0.8;
        double theta = 0.9;
        Double3 values = JacobiElliptic.snCnDn(k, theta);
        double sn = values._1();
        double cn = values._2();
        double dn = values._3();

        assertClose(1.0, sn * sn + cn * cn, 5.0e-13);
        assertClose(1.0, dn * dn + k * k * sn * sn, 5.0e-13);
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> JacobiElliptic.snCnDn(-0.1, 0.3));
        assertThrows(IllegalArgumentException.class, () -> JacobiElliptic.snCnDn(Double.NaN, 0.3));
        assertThrows(IllegalArgumentException.class, () -> JacobiElliptic.snCnDn(0.3, Double.POSITIVE_INFINITY));
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(expected - actual);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}