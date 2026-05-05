package com.curioloop.yum4j.fft;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class FftUtilTest {
    @Test
    void largestPrimeFactorMatchesReferenceCases() {
        assertThat(FftUtil.largestPrimeFactor(1)).isEqualTo(1);
        assertThat(FftUtil.largestPrimeFactor(2)).isEqualTo(2);
        assertThat(FftUtil.largestPrimeFactor(2L * 2L * 3L * 5L * 11L)).isEqualTo(11);
        assertThat(FftUtil.largestPrimeFactor(257)).isEqualTo(257);
        assertThat(FftUtil.largestPrimeFactor(1024)).isEqualTo(2);
    }

    @Test
    void goodSizeComplexMatchesBruteForceForSmallRange() {
        for (long n = 1; n <= 600; n++) {
            assertThat(FftUtil.goodSizeComplex(n)).isEqualTo(bruteGoodSize(n, 2, 3, 5, 7, 11));
        }
    }

    @Test
    void goodSizeRealMatchesBruteForceForSmallRange() {
        for (long n = 1; n <= 600; n++) {
            assertThat(FftUtil.goodSizeReal(n)).isEqualTo(bruteGoodSize(n, 2, 3, 5));
        }
    }

    @Test
    void previousGoodSizesMatchBruteForceForSmallRange() {
        for (long n = 1; n <= 600; n++) {
            assertThat(FftUtil.prevGoodSizeComplex(n)).isEqualTo(brutePrevGoodSize(n, 2, 3, 5, 7, 11));
            assertThat(FftUtil.prevGoodSizeReal(n)).isEqualTo(brutePrevGoodSize(n, 2, 3, 5));
        }
    }

    @Test
    void requiredFactorGoodSizeMatchesReferenceContract() {
        assertThat(FftUtil.goodSizeComplex(77, 8)).isEqualTo(80);
        assertThat(FftUtil.goodSizeComplex(257, 16)).isEqualTo(bruteGoodSize((257 + 15) / 16, 2, 3, 5, 7, 11) * 16);
        assertThat(FftUtil.goodSizeReal(77, 8)).isEqualTo(80);
        assertThat(FftUtil.goodSizeReal(257, 16)).isEqualTo(bruteGoodSize((257 + 15) / 16, 2, 3, 5) * 16);
    }

    @Test
    void sanityCheckRejectsInvalidShapesStridesAndAxes() {
        assertThatThrownBy(() -> FftUtil.sanityCheck(new int[0], new long[0], new long[0], false))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("ndim");
        assertThatThrownBy(() -> FftUtil.sanityCheck(new int[] { 2, 3 }, new long[] { 1 }, new long[] { 1, 2 }, false))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("stride dimension");
        assertThatThrownBy(() -> FftUtil.sanityCheck(new int[] { 2, 3 }, new long[] { 1, 2 }, new long[] { 1, 4 }, true))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("stride mismatch");
        assertThatThrownBy(() -> FftUtil.sanityCheck(new int[] { 2, 3 }, new long[] { 1, 2 }, new long[] { 1, 2 }, false, new int[] { 1, 1 }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("repeatedly");
    }

    @Test
    void sincosUsesPocketfftSymmetryForUnitCircleValues() {
        for (long n : new long[] { 2, 3, 4, 5, 8, 17, 64, 257, 1024 }) {
            FftSincos sincos = new FftSincos(n);
            for (long index = 0; index < n; index++) {
                double angle = 2.0 * Math.PI * (double) index / (double) n;
                assertThat(sincos.real(index)).isCloseTo(Math.cos(angle), withinScaledTolerance(n));
                assertThat(sincos.imag(index)).isCloseTo(Math.sin(angle), withinScaledTolerance(n));
                assertThat(Math.hypot(sincos.real(index), sincos.imag(index))).isCloseTo(1.0, withinScaledTolerance(n));
            }
        }
    }

    private static org.assertj.core.data.Offset<Double> withinScaledTolerance(long n) {
        return org.assertj.core.data.Offset.offset(Math.max(8.0e-15, n * 2.0e-16));
    }

    private static long bruteGoodSize(long n, int... primes) {
        for (long candidate = n; ; candidate++) {
            if (isSmooth(candidate, primes)) {
                return candidate;
            }
        }
    }

    private static long brutePrevGoodSize(long n, int... primes) {
        for (long candidate = n; candidate >= 1; candidate--) {
            if (isSmooth(candidate, primes)) {
                return candidate;
            }
        }
        throw new AssertionError("unreachable");
    }

    private static boolean isSmooth(long value, int... primes) {
        for (int prime : primes) {
            while (value % prime == 0) {
                value /= prime;
            }
        }
        return value == 1;
    }
}