package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LambertWTest {

    private static final double BRANCH_POINT = -1.0 / Math.E;

    @Test
    void principalBranchInvertsRepresentativePoints() {
        assertBranchIdentity(-1.0, true, 5.0e-14);
        assertBranchIdentity(-0.75, true, 5.0e-14);
        assertBranchIdentity(-0.25, true, 5.0e-14);
        assertBranchIdentity(0.0, true, 0.0);
        assertBranchIdentity(0.25, true, 5.0e-14);
        assertBranchIdentity(1.0, true, 5.0e-14);
        assertBranchIdentity(5.0, true, 5.0e-14);
    }

    @Test
    void lowerBranchInvertsRepresentativePoints() {
        assertBranchIdentity(-1.0, false, 0.0);
        assertBranchIdentity(-1.5, false, 5.0e-14);
        assertBranchIdentity(-2.0, false, 5.0e-14);
        assertBranchIdentity(-5.0, false, 5.0e-14);
        assertBranchIdentity(-20.0, false, 5.0e-13);
    }

    @Test
    void branchesMeetAtSingularityAndSplitInsideDomain() {
        assertEquals(-1.0, LambertW.w0(BRANCH_POINT), 0.0);
        assertEquals(-1.0, LambertW.wm1(BRANCH_POINT), 0.0);

        double z = -0.1;
        double principal = LambertW.w0(z);
        double lower = LambertW.wm1(z);
        assertTrue(lower < -1.0);
        assertTrue(principal > -1.0);
        assertTrue(principal > lower);
    }

    @Test
    void principalBranchPreservesSignedZero() {
        double value = LambertW.w0(-0.0);
        assertEquals(0.0, value, 0.0);
        assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(value));
    }

    @Test
    void derivativesMatchNumericalSlopesAwayFromSingularities() {
        assertDerivativeClose(0.2, true, 5.0e-8);
        assertDerivativeClose(3.0, true, 5.0e-8);
        assertDerivativeClose(-0.1, false, 5.0e-8);
        assertDerivativeClose(-1.0e-6, false, 1.0e-6);
    }

    @Test
    void exactDerivativeSpecialCasesFollowBoostSemantics() {
        assertEquals(1.0, LambertW.w0Prime(0.0), 0.0);
        assertThrows(ArithmeticException.class, () -> LambertW.w0Prime(BRANCH_POINT));
        assertThrows(ArithmeticException.class, () -> LambertW.wm1Prime(BRANCH_POINT));
        assertThrows(ArithmeticException.class, () -> LambertW.wm1Prime(-0.0));
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> LambertW.w0(Double.NaN));
        assertThrows(ArithmeticException.class, () -> LambertW.w0(Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> LambertW.w0(Math.nextDown(BRANCH_POINT)));

        assertThrows(IllegalArgumentException.class, () -> LambertW.wm1(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> LambertW.wm1(Double.NEGATIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> LambertW.wm1(0.1));
        assertThrows(ArithmeticException.class, () -> LambertW.wm1(0.0));
        assertThrows(ArithmeticException.class, () -> LambertW.wm1(-Double.MIN_VALUE));
        assertThrows(IllegalArgumentException.class, () -> LambertW.wm1(Math.nextDown(BRANCH_POINT)));
    }

    private static void assertBranchIdentity(double w, boolean principalBranch, double tolerance) {
        double z = w == 0.0 ? 0.0 : w * Math.exp(w);
        double actual = principalBranch ? LambertW.w0(z) : LambertW.wm1(z);
        assertClose(w, actual, tolerance);
        assertClose(z, actual * Math.exp(actual), Math.max(tolerance, 5.0e-14));
    }

    private static void assertDerivativeClose(double z, boolean principalBranch, double tolerance) {
        double step = Math.cbrt(Math.ulp(1.0)) * Math.max(1.0e-6, Math.abs(z));
        if (!principalBranch && z + step >= 0.0) {
            step = 0.25 * Math.abs(z);
        }

        double upper = principalBranch ? LambertW.w0(z + step) : LambertW.wm1(z + step);
        double lower = principalBranch ? LambertW.w0(z - step) : LambertW.wm1(z - step);
        double numerical = (upper - lower) / (2.0 * step);
        double actual = principalBranch ? LambertW.w0Prime(z) : LambertW.wm1Prime(z);
        assertClose(numerical, actual, tolerance);
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(expected - actual);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}