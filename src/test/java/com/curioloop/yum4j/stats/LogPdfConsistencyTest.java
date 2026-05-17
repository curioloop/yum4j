package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Consistency checks for every distribution whose {@code logPdf} was
 * added in the particle-filter follow-up pass.
 *
 * <p>Two invariants per distribution:
 * <ol>
 *   <li>In the safe interior, {@code logPdf(x)} and {@code log(pdf(x))}
 *       agree to a tight relative tolerance (both should be finite and
 *       well within double-precision range).</li>
 *   <li>In the far tail — at points where {@code pdf(x)} underflows to
 *       {@code 0} — {@code logPdf(x)} must stay strictly finite (the
 *       "recovery" property the override was added to provide).</li>
 * </ol>
 *
 * <p>Exact numerical parity against a scipy reference is covered by the
 * dedicated {@code *LogPdfParityTest} fixtures for the high-frequency
 * SMC distributions; this test validates only the algebraic rewrite,
 * which is what can actually be wrong at review time.
 */
class LogPdfConsistencyTest {

    /** Tight comparison for interior consistency. */
    private static final double REL_TOL = 1e-12;

    /** Asserts logPdf matches log(pdf) at a single interior point. */
    private static void assertInteriorMatch(String name, double logPdf, double pdf) {
        assertThat(pdf).as(name + " pdf sanity").isGreaterThan(0.0).isFinite();
        double expected = Math.log(pdf);
        double scale = Math.max(1.0, Math.abs(expected));
        assertThat(logPdf)
            .as(name + " logPdf vs log(pdf)")
            .isCloseTo(expected, within(REL_TOL * scale));
    }

    /** Asserts the logPdf is strictly finite at a tail point. */
    private static void assertTailFinite(String name, double logPdf) {
        assertThat(logPdf)
            .as(name + " tail must stay finite")
            .isNotEqualTo(Double.NEGATIVE_INFINITY)
            .isNotEqualTo(Double.POSITIVE_INFINITY)
            .isNotNaN();
    }

    // ------------------------------------------------------------------
    // Batch 1 — critical / tail-underflow fixes
    // ------------------------------------------------------------------

    @Test
    void weibull_interior_matches_and_tail_is_finite() {
        WeibullDistribution d = new WeibullDistribution(1.5, 2.0);
        // Interior.
        for (double x : new double[]{0.1, 0.5, 1.0, 2.0, 3.5}) {
            assertInteriorMatch("Weibull@" + x, d.logPdf(x), d.pdf(x));
        }
        // Tail where exp(-(x/σ)^k) underflows: (100/2)^1.5 ≈ 353, exp(-353) is ~1e-154.
        // A more extreme probe: shape=4, scale=2, x=60 -> exp(-(30)^4) completely underflows.
        WeibullDistribution d2 = new WeibullDistribution(4.0, 2.0);
        double tail = d2.logPdf(60.0);
        assertTailFinite("Weibull@60", tail);
        assertThat(d2.pdf(60.0)).as("Weibull@60 pdf underflows").isEqualTo(0.0);
    }

    @Test
    void rayleigh_interior_matches_and_tail_is_finite() {
        RayleighDistribution d = new RayleighDistribution(1.5);
        for (double x : new double[]{0.5, 1.0, 2.0, 4.0}) {
            assertInteriorMatch("Rayleigh@" + x, d.logPdf(x), d.pdf(x));
        }
        // x=60, σ=1.5: x²/(2σ²) = 800, exp(-800) completely underflows.
        double tail = d.logPdf(60.0);
        assertTailFinite("Rayleigh@60", tail);
        assertThat(d.pdf(60.0)).as("Rayleigh@60 pdf underflows").isEqualTo(0.0);
    }

    @Test
    void extremeValue_interior_matches_and_tail_is_finite() {
        ExtremeValueDistribution d = new ExtremeValueDistribution(0.0, 1.0);
        // Interior points on both sides of the mode (at location).
        for (double x : new double[]{-2.0, -0.5, 0.0, 1.0, 3.0}) {
            assertInteriorMatch("Gumbel@" + x, d.logPdf(x), d.pdf(x));
        }
        // Left tail: x = -20 → z = 20 → exp(z) ≈ 5e8 → exp(z - exp(z)) completely underflows.
        double leftTail = d.logPdf(-20.0);
        assertTailFinite("Gumbel@-20", leftTail);
        assertThat(d.pdf(-20.0)).as("Gumbel@-20 pdf underflows").isEqualTo(0.0);
        // Right tail is less dramatic but still interesting.
        double rightTail = d.logPdf(800.0);
        assertTailFinite("Gumbel@800", rightTail);
    }

    @Test
    void pareto_interior_matches_and_tail_is_finite() {
        ParetoDistribution d = new ParetoDistribution(1.0, 2.5);
        for (double x : new double[]{1.0, 2.0, 5.0, 20.0}) {
            assertInteriorMatch("Pareto@" + x, d.logPdf(x), d.pdf(x));
        }
        // Large x: pdf shrinks polynomially; default logPdf works here too,
        // but the direct form is both cheaper and ulp-identical. We still
        // sanity-check that extreme x stays finite.
        double tail = d.logPdf(1e300);
        assertTailFinite("Pareto@1e300", tail);
    }

    @Test
    void inverseGaussian_interior_matches_and_tail_is_finite() {
        InverseGaussianDistribution d = new InverseGaussianDistribution(1.0, 1.0);
        for (double x : new double[]{0.2, 0.5, 1.0, 2.0, 5.0}) {
            assertInteriorMatch("IG@" + x, d.logPdf(x), d.pdf(x));
        }
        // Right tail at x=1500: exp(-scale·(x-μ)²/(2·x·μ²)) = exp(-749) underflows.
        double rightTail = d.logPdf(1500.0);
        assertTailFinite("IG@1500", rightTail);
        assertThat(d.pdf(1500.0)).as("IG@1500 pdf underflows").isEqualTo(0.0);
    }

    @Test
    void chiSquare_interior_matches_and_tail_is_finite() {
        ChiSquareDistribution d = new ChiSquareDistribution(5.0);
        for (double x : new double[]{0.5, 2.0, 5.0, 15.0}) {
            assertInteriorMatch("ChiSq@" + x, d.logPdf(x), d.pdf(x));
        }
        // Right tail: x=2000, df=5 → pdf ~ x^{1.5}·exp(-x/2) → exp(-1000).
        double tail = d.logPdf(2000.0);
        assertTailFinite("ChiSq@2000", tail);
        assertThat(d.pdf(2000.0)).as("ChiSq@2000 pdf underflows").isEqualTo(0.0);
    }

    @Test
    void negativeBinomial_interior_matches_and_tail_is_finite() {
        NegativeBinomialDistribution d = new NegativeBinomialDistribution(10.0, 0.3);
        for (double x : new double[]{0.0, 5.0, 20.0, 50.0}) {
            assertInteriorMatch("NB@" + x, d.logPdf(x), d.pdf(x));
        }
        // Large k: pdf = p^r · (1-p)^k · C(k+r-1, k). With p=0.001, k=5000 this
        // underflows through the ibetaDerivative path but stays finite in log.
        NegativeBinomialDistribution d2 = new NegativeBinomialDistribution(2.0, 0.001);
        double tail = d2.logPdf(5000.0);
        assertTailFinite("NB@5000 (p=0.001)", tail);
    }

    // ------------------------------------------------------------------
    // Batch 2 — precision / convenience rewrites
    // ------------------------------------------------------------------

    @Test
    void cauchy_interior_matches() {
        CauchyDistribution d = new CauchyDistribution(0.0, 1.0);
        for (double x : new double[]{-5.0, -0.1, 0.0, 0.5, 10.0}) {
            assertInteriorMatch("Cauchy@" + x, d.logPdf(x), d.pdf(x));
        }
        // Far tail stays finite (Cauchy tail is polynomial; pdf does not underflow
        // as fast as exponential ones, but log1p is more accurate near 0).
        assertTailFinite("Cauchy@1e150", d.logPdf(1e150));
    }

    @Test
    void skewNormal_interior_matches_and_tail_is_finite() {
        SkewNormalDistribution d = new SkewNormalDistribution(0.0, 1.0, 2.0);
        for (double x : new double[]{-1.0, 0.0, 0.5, 2.0}) {
            assertInteriorMatch("SkewN@" + x, d.logPdf(x), d.pdf(x));
        }
        // Deep left tail where Φ(α·z) underflows to zero: α=2, x=-20 → α·z = -40.
        double tail = d.logPdf(-20.0);
        assertTailFinite("SkewN@-20 (α=2)", tail);
    }

    @Test
    void inverseChiSquare_delegates_to_equivalent() {
        InverseChiSquareDistribution d = new InverseChiSquareDistribution(5.0, 0.5);
        InverseGammaDistribution ig = new InverseGammaDistribution(2.5, 1.25);
        for (double x : new double[]{0.1, 0.5, 1.0, 3.0}) {
            assertThat(d.logPdf(x))
                .as("InvChiSq@" + x + " = InvGamma via re-parameterisation")
                .isCloseTo(ig.logPdf(x), within(1e-15));
        }
    }

    @Test
    void hypergeometric_interior_matches() {
        HypergeometricDistribution d = new HypergeometricDistribution(30L, 20L, 100L);
        for (double x : new double[]{0.0, 5.0, 6.0, 10.0, 15.0, 20.0}) {
            double pdf = d.pdf(x);
            if (pdf > 0.0) {
                assertInteriorMatch("HG@" + x, d.logPdf(x), pdf);
            }
        }
        // Out-of-support: logPdf must be -Infinity.
        assertThat(d.logPdf(-1.0)).isEqualTo(Double.NEGATIVE_INFINITY);
        assertThat(d.logPdf(21.0)).isEqualTo(Double.NEGATIVE_INFINITY);
    }

    @Test
    void fisherF_interior_matches_and_tail_is_finite() {
        FisherFDistribution d = new FisherFDistribution(5.0, 10.0);
        for (double x : new double[]{0.1, 0.5, 1.0, 3.0, 10.0}) {
            assertInteriorMatch("FisherF@" + x, d.logPdf(x), d.pdf(x));
        }
        // Large x: pdf ~ x^{-(d1+d2)/2-1} eventually underflows; here d1=5, d2=10,
        // exponent = -8.5, so polynomial decay is slow but ibetaDerivative flips
        // to zero near y→1.
        double tail = d.logPdf(1e50);
        assertTailFinite("FisherF@1e50", tail);
    }

    @Test
    void hyperexponential_interior_matches_and_tail_is_finite() {
        HyperexponentialDistribution d = new HyperexponentialDistribution(
            new double[]{0.3, 0.7}, new double[]{0.5, 2.5}
        );
        for (double x : new double[]{0.1, 1.0, 3.0, 10.0}) {
            assertInteriorMatch("HypExp@" + x, d.logPdf(x), d.pdf(x));
        }
        // Right tail: x=2000 → both exp(-0.5·2000) and exp(-2.5·2000) underflow.
        double tail = d.logPdf(2000.0);
        assertTailFinite("HypExp@2000", tail);
        assertThat(d.pdf(2000.0)).as("HypExp@2000 pdf underflows").isEqualTo(0.0);
    }

    // ------------------------------------------------------------------
    // Normal.logCdf helper used by SkewNormalDistribution
    // ------------------------------------------------------------------

    @Test
    void normalLogCdf_interior_matches_and_deepLeftTail_is_finite() {
        // Interior: log(Φ(x)) agrees with Math.log(Φ(x)) for reasonable x.
        for (double x : new double[]{-1.0, -0.5, 0.0, 0.5, 2.0, 5.0}) {
            double direct = com.curioloop.yum4j.math.Normal.logCdf(x);
            double viaLog = Math.log(com.curioloop.yum4j.math.Normal.cdf(x));
            double scale = Math.max(1.0, Math.abs(viaLog));
            assertThat(direct)
                .as("Normal.logCdf interior @" + x)
                .isCloseTo(viaLog, within(1e-12 * scale));
        }
        // Deep left tail: Normal.cdf(-40) underflows to zero but
        // logCdf(-40) stays finite (-40²/2 = -800 plus log(series) correction).
        double deep = com.curioloop.yum4j.math.Normal.logCdf(-40.0);
        assertThat(deep).as("Normal.logCdf(-40) finite").isFinite();
        assertThat(deep).as("Normal.logCdf(-40) near -x²/2 asymptote")
            .isLessThan(-600.0).isGreaterThan(-900.0);
        assertThat(com.curioloop.yum4j.math.Normal.cdf(-40.0))
            .as("Normal.cdf(-40) underflows")
            .isEqualTo(0.0);
    }
}
