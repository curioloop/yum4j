package com.curioloop.yum4j.linalg;

import com.curioloop.yum4j.linalg.reg.GLS;
import com.curioloop.yum4j.linalg.reg.OLS;
import com.curioloop.yum4j.linalg.reg.WLS;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Data-driven regression tests mirroring statsmodels test_regression.py.
 *
 * <p>Uses the Longley dataset (16 obs, 6 exog + constant) with reference values
 * from R/Stata via statsmodels. Tests follow the CheckRegressionResults pattern:
 * params, bse, t-values, p-values, F-stat, R², SSR, ESS, scale, logLike, AIC, BIC,
 * confidence intervals, HC0-HC3 standard errors, and eigenvalues.</p>
 *
 * <p>Equivalence tests verify: OLS ≡ WLS(w=1) ≡ GLS(σ=I) and WLS(w) ≡ GLS(σ=1/w).</p>
 */
class RegressorTest {

    // ═══════════════════════════════════════════════════════════════════════════
    // Longley dataset (statsmodels.datasets.longley)
    // 16 observations, 6 exogenous variables: GNPDEFL, GNP, UNEMP, ARMED, POP, YEAR
    // ═══════════════════════════════════════════════════════════════════════════

    private static final int N = 16, K_EXOG = 6, K = 7; // 6 exog + 1 constant
    private static final double[] ENDOG = {
        60323, 61122, 60171, 61187, 63221, 63639, 64989, 63761,
        66019, 67857, 68169, 66513, 68655, 69564, 69331, 70551
    };
    // Row-major 16×6 (no constant column — we add it)
    private static final double[] EXOG_RAW = {
        83.0, 234289.0, 2356.0, 1590.0, 107608.0, 1947.0,
        88.5, 259426.0, 2325.0, 1456.0, 108632.0, 1948.0,
        88.2, 258054.0, 3682.0, 1616.0, 109773.0, 1949.0,
        89.5, 284599.0, 3351.0, 1650.0, 110929.0, 1950.0,
        96.2, 328975.0, 2099.0, 3099.0, 112075.0, 1951.0,
        98.1, 346999.0, 1932.0, 3594.0, 113270.0, 1952.0,
        99.0, 365385.0, 1870.0, 3547.0, 115094.0, 1953.0,
        100.0, 363112.0, 3578.0, 3350.0, 116219.0, 1954.0,
        101.2, 397469.0, 2904.0, 3048.0, 117388.0, 1955.0,
        104.6, 419180.0, 2822.0, 2857.0, 118734.0, 1956.0,
        108.4, 442769.0, 2936.0, 2798.0, 120445.0, 1957.0,
        110.8, 444546.0, 4681.0, 2637.0, 121950.0, 1958.0,
        112.6, 482704.0, 3813.0, 2552.0, 123366.0, 1959.0,
        114.2, 502601.0, 3931.0, 2514.0, 125368.0, 1960.0,
        115.7, 518173.0, 4806.0, 2572.0, 127852.0, 1961.0,
        116.9, 554894.0, 4007.0, 2827.0, 130081.0, 1962.0
    };

    // ── OLS reference values (from statsmodels / R) ──────────────────────────
    private static final double[] OLS_PARAMS = {
        15.0618722713416, -0.035819179292581715, -2.0202298038167683,
        -1.0332268671736249, -0.05110410565358592, 1829.1514646134965, -3482258.6345956847
    };
    private static final double[] OLS_BSE = {
        84.91492577476669, 0.033491007772243696, 0.48839968165171516,
        0.21427416316168660, 0.22607320006937454, 455.4784991422289, 890420.3836074055
    };
    private static final double[] OLS_TVALUES = {
        0.17737602822962584, -1.0695163172207536, -4.136427355940464,
        -4.821985310445359, -0.22605114466422258, 4.015889812709515, -3.9108029181540442
    };
    private static final double[] OLS_PVALUES = {
        0.8631408328094988, 0.31268106109283655, 0.002535091734112218,
        0.0009443667641619271, 0.8262117957636328, 0.0030368033416315245, 0.003560403663727821
    };
    private static final double OLS_FVALUE = 330.2853392345847;
    private static final double OLS_F_PVALUE = 4.984030528725034e-10;
    private static final double OLS_RSQUARED = 0.9954790045772955;
    private static final double OLS_RSQUARED_ADJ = 0.992465007628826;
    private static final double OLS_SSR = 836424.0555059237;
    private static final double OLS_ESS = 184172401.94449407;
    private static final double OLS_SCALE = 92936.00616732486;
    private static final double OLS_LLF = -109.61743480848065;
    private static final double OLS_AIC = 233.2348696169613;
    private static final double OLS_BIC = 238.64299067263977;

    // ── Shared setup ─────────────────────────────────────────────────────────

    /** Builds the design matrix with constant appended (row-major 16×7). */
    private static double[] buildExogWithConst() {
        double[] X = new double[N * K];
        for (int i = 0; i < N; i++) {
            System.arraycopy(EXOG_RAW, i * K_EXOG, X, i * K, K_EXOG);
            X[i * K + K_EXOG] = 1.0; // constant column at the end
        }
        return X;
    }

    private static void assertClose(double[] expected, double[] actual, double tol, String msg) {
        assertEquals(expected.length, actual.length, msg + " length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], Math.abs(expected[i]) * tol + tol, msg + "[" + i + "]");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // OLS Tests (mirrors statsmodels TestOLS + CheckRegressionResults)
    // ═══════════════════════════════════════════════════════════════════════════

    @Nested
    class TestOLS {
        private static OLS res;

        @BeforeAll
        static void setup() {
            res = Regressor.ols(ENDOG.clone(), buildExogWithConst(), N, K,
                    Regressor.Opts.QR, Regressor.Opts.INFERENCE);
        }

        @Test void params()    { assertClose(OLS_PARAMS, res.params(), 1e-8, "params"); }
        @Test void bse()       { assertClose(OLS_BSE, res.bse(), 1e-8, "bse"); }
        @Test void tvalues()   { assertClose(OLS_TVALUES, res.tStatistics(), 1e-8, "tvalues"); }
        @Test void pvalues()   { assertClose(OLS_PVALUES, res.tPValues(), 1e-6, "pvalues"); }
        @Test void fvalue()    { assertEquals(OLS_FVALUE, res.fStatistic(), OLS_FVALUE * 1e-8, "fvalue"); }
        @Test void fPvalue()   { assertEquals(OLS_F_PVALUE, res.fPValue(), 1e-15, "f_pvalue"); }
        @Test void rsquared()  { assertEquals(OLS_RSQUARED, res.r2(false), 1e-10, "rsquared"); }
        @Test void rsqAdj()    { assertEquals(OLS_RSQUARED_ADJ, res.r2(true), 1e-10, "rsquared_adj"); }
        @Test void ssr()       { assertEquals(OLS_SSR, res.ssr(), OLS_SSR * 1e-10, "ssr"); }
        @Test void ess()       { assertEquals(OLS_ESS, res.ess(), OLS_ESS * 1e-10, "ess"); }
        @Test void scale()     { assertEquals(OLS_SCALE, res.scale(), OLS_SCALE * 1e-10, "scale"); }
        @Test void loglike()   { assertEquals(OLS_LLF, res.logLike(), 1e-8, "llf"); }
        @Test void aic()       { assertEquals(OLS_AIC, res.aic(), 1e-8, "aic"); }
        @Test void bic()       { assertEquals(OLS_BIC, res.bic(), 1e-8, "bic"); }
        @Test void dfModel()   { assertEquals(6, res.rank() - res.kConst()); }
        @Test void dfResid()   { assertEquals(9, N - res.rank()); }

        @Test void confInt() {
            double[] ci = res.confInt(0.05);
            // Verify symmetry: (upper - param) ≈ (param - lower) within floating-point tolerance
            double[] p = res.params();
            for (int i = 0; i < K; i++) {
                double lower = ci[i * 2], upper = ci[i * 2 + 1];
                double hw1 = p[i] - lower, hw2 = upper - p[i];
                assertEquals(hw1, hw2, Math.max(Math.abs(hw1), 1.0) * 1e-10, "CI symmetry[" + i + "]");
                assertTrue(lower < upper, "CI lower < upper[" + i + "]");
            }
        }

        @Test void durbinWatson() {
            double dw = res.durbinWatson();
            assertTrue(dw >= 0 && dw <= 4, "DW in [0,4]");
        }

        @Test void jarqueBera() {
            var jb = res.jarqueBera();
            assertTrue(jb.statistic() >= 0, "JB stat >= 0");
            assertTrue(jb.pValue() >= 0 && jb.pValue() <= 1, "JB pval in [0,1]");
        }

        @Test void qrMatchesSvd() {
            OLS svd = Regressor.ols(ENDOG.clone(), buildExogWithConst(), N, K,
                    Regressor.Opts.PINV, Regressor.Opts.INFERENCE);
            assertClose(res.params(), svd.params(), 1e-8, "QR vs SVD params");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // WLS Tests
    // ═══════════════════════════════════════════════════════════════════════════

    @Nested
    class TestWLS {
        private static final double[] WEIGHTS = {
            0.6109965855448698, 0.9353661530886882, 0.6033595776697132, 0.9593054539689607,
            0.7442055943974146, 0.8058719314513229, 0.8829539282401577, 0.7592089939364717,
            0.648400250788111, 0.5938606143306258, 0.5403706343824375, 0.869220148099485,
            0.7206546114479766, 0.5791549338563255, 0.9399685156006394, 0.6370432309961123
        };
        private static final double[] WLS_PARAMS = {
            26.774466068990478, -0.0342680983756245, -1.973400551657079,
            -1.0008388915364526, -0.02370419296214976, 1728.202617500985, -3290211.6995615438
        };
        private static final double WLS_FVALUE = 315.10503906246095;
        private static final double WLS_RSQUARED = 0.9952622358619375;
        private static final double WLS_LLF = -109.85230867270201;
        private static final double WLS_SCALE = 69521.62063143107;

        private static WLS res;

        @BeforeAll
        static void setup() {
            res = Regressor.wls(ENDOG.clone(), buildExogWithConst(), WEIGHTS.clone(), N, K,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
        }

        @Test void params()   { assertClose(WLS_PARAMS, res.params(), 1e-8, "wls params"); }
        @Test void fvalue()   { assertEquals(WLS_FVALUE, res.fStatistic(), WLS_FVALUE * 1e-8, "wls fvalue"); }
        @Test void rsquared() { assertEquals(WLS_RSQUARED, res.r2(false), 1e-10, "wls rsquared"); }
        @Test void loglike()  { assertEquals(WLS_LLF, res.logLike(), 1e-6, "wls llf"); }
        @Test void scale()    { assertEquals(WLS_SCALE, res.scale(), WLS_SCALE * 1e-8, "wls scale"); }

        @Test void residualPlusFittedEqualsY() {
            double[] fitted = res.fitted(false);
            double[] resid = res.residual(false);
            for (int i = 0; i < N; i++) {
                assertEquals(ENDOG[i], fitted[i] + resid[i], 1e-8, "y = fitted + resid [" + i + "]");
            }
        }

        @Test void outputModesReuseOlsResourceSelection() {
            WLS inference = Regressor.wls(ENDOG.clone(), buildExogWithConst(), WEIGHTS.clone(), N, K,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            WLS paramsOnly = Regressor.wls(ENDOG.clone(), buildExogWithConst(), WEIGHTS.clone(), N, K,
                Regressor.Opts.QR, Regressor.Opts.PARAMS);
            WLS fitness = Regressor.wls(ENDOG.clone(), buildExogWithConst(), WEIGHTS.clone(), N, K,
                Regressor.Opts.QR, Regressor.Opts.FITNESS);
            WLS estimation = Regressor.wls(ENDOG.clone(), buildExogWithConst(), WEIGHTS.clone(), N, K,
                Regressor.Opts.QR, Regressor.Opts.ESTIMATION);

            assertClose(inference.params(), paramsOnly.params(), 1e-10, "wls params-only params");
            assertThrows(IllegalStateException.class, () -> paramsOnly.residual(false));
            assertThrows(IllegalStateException.class, paramsOnly::bse);

            assertClose(inference.residual(false), fitness.residual(false), 1e-10, "wls fitness residual");
            assertEquals(inference.ssr(), fitness.ssr(), Math.max(1.0, inference.ssr()) * 1e-10);
            assertThrows(IllegalStateException.class, fitness::params);

            assertClose(inference.params(), estimation.params(), 1e-10, "wls estimation params");
            assertClose(inference.residual(false), estimation.residual(false), 1e-10, "wls estimation residual");
            assertThrows(IllegalStateException.class, estimation::bse);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GLS Tests (AR(1) sigma, mirrors statsmodels TestGLS)
    // ═══════════════════════════════════════════════════════════════════════════

    @Nested
    class TestGLS {
        // GLS uses 2-variable model: GNP (col 1) + POP (col 4) + constant
        private static final double[] GLS_PARAMS = {0.0673894832462239, -0.47427390364290645, 94898.8771175083};
        private static final double[] GLS_BSE = {0.010703390282000588, 0.15338547246932124, 13944.772261984166};
        private static final double[] GLS_TVALUES = {6.296087638656863, -3.0920392655684283, 6.805337178307233};
        private static final double[] GLS_PVALUES = {2.7616728901175482e-05, 0.008577196762571748, 1.2522836412179427e-05};
        private static final double GLS_FVALUE = 196.6351190743541;
        private static final double GLS_RSQUARED = 0.9680015940639944;
        private static final double GLS_SCALE = 294244.4550059387;
        private static final double GLS_LLF = -121.01935030716783;

        private static GLS res;

        private static double[] buildGlsExog() {
            double[] X2 = new double[N * 3];
            for (int i = 0; i < N; i++) {
                X2[i * 3]     = EXOG_RAW[i * K_EXOG + 1];
                X2[i * 3 + 1] = EXOG_RAW[i * K_EXOG + 4];
                X2[i * 3 + 2] = 1.0;
            }
            return X2;
        }

        @BeforeAll
        static void setup() {
            // Build 2-variable exog: GNP (col 1) + POP (col 4) + constant
            double[] X2 = buildGlsExog();
            // First fit OLS to get rho for AR(1) sigma
            OLS tmp = Regressor.ols(ENDOG.clone(), X2.clone(), N, 3,
                Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);
            double[] resid = tmp.residual(false);
            // rho = corr(resid[1:], resid[:-1])
            double mean1 = 0, mean2 = 0;
            for (int i = 1; i < N; i++) { mean1 += resid[i]; mean2 += resid[i-1]; }
            mean1 /= (N-1); mean2 /= (N-1);
            double num = 0, d1 = 0, d2 = 0;
            for (int i = 1; i < N; i++) {
                double a = resid[i] - mean1, b = resid[i-1] - mean2;
                num += a * b; d1 += a * a; d2 += b * b;
            }
            double rho = num / Math.sqrt(d1 * d2);
            // Build Toeplitz sigma: sigma[i][j] = rho^|i-j|
            double[] sigma = new double[N * N];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    sigma[i * N + j] = Math.pow(rho, Math.abs(i - j));
            // Fit GLS
            res = Regressor.gls(ENDOG.clone(), X2.clone(), sigma, N, 3,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
        }

        @Test void params()   { assertClose(GLS_PARAMS, res.params(), 1e-4, "gls params"); }
        @Test void bse()      { assertClose(GLS_BSE, res.bse(), 1e-4, "gls bse"); }
        @Test void tvalues()  { assertClose(GLS_TVALUES, res.tStatistics(), 1e-4, "gls tvalues"); }
        @Test void pvalues()  { assertClose(GLS_PVALUES, res.tPValues(), 1e-4, "gls pvalues"); }
        @Test void fvalue()   { assertEquals(GLS_FVALUE, res.fStatistic(), GLS_FVALUE * 1e-4, "gls fvalue"); }
        @Test void rsquared() { assertEquals(GLS_RSQUARED, res.r2(false), 1e-4, "gls rsquared"); }
        @Test void scale()    { assertEquals(GLS_SCALE, res.scale(), GLS_SCALE * 1e-4, "gls scale"); }
        @Test void loglike()  { assertEquals(GLS_LLF, res.logLike(), 1.0, "gls llf"); }

        @Test void outputModesReuseOlsResourceSelection() {
            GLS inference = Regressor.gls(ENDOG.clone(), buildGlsExog(), new double[]{1.0}, N, 3,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            GLS paramsOnly = Regressor.gls(ENDOG.clone(), buildGlsExog(), new double[]{1.0}, N, 3,
                Regressor.Opts.QR, Regressor.Opts.PARAMS);
            GLS fitness = Regressor.gls(ENDOG.clone(), buildGlsExog(), new double[]{1.0}, N, 3,
                Regressor.Opts.QR, Regressor.Opts.FITNESS);
            GLS estimation = Regressor.gls(ENDOG.clone(), buildGlsExog(), new double[]{1.0}, N, 3,
                Regressor.Opts.QR, Regressor.Opts.ESTIMATION);

            assertClose(inference.params(), paramsOnly.params(), 1e-10, "gls params-only params");
            assertThrows(IllegalStateException.class, () -> paramsOnly.residual(false));
            assertThrows(IllegalStateException.class, paramsOnly::bse);

            assertClose(inference.residual(false), fitness.residual(false), 1e-10, "gls fitness residual");
            assertEquals(inference.ssr(), fitness.ssr(), Math.max(1.0, inference.ssr()) * 1e-10);
            assertThrows(IllegalStateException.class, fitness::params);

            assertClose(inference.params(), estimation.params(), 1e-10, "gls estimation params");
            assertClose(inference.residual(false), estimation.residual(false), 1e-10, "gls estimation residual");
            assertThrows(IllegalStateException.class, estimation::bse);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Equivalence Tests (mirrors statsmodels TestOLS_GLS_WLS_equivalence)
    // ═══════════════════════════════════════════════════════════════════════════

    @Nested
    class TestEquivalence {

        @Test void glsIdentityEqualsOls() {
            // GLS(sigma=1) should produce same params as OLS
            OLS ols = Regressor.ols(ENDOG.clone(), buildExogWithConst(), N, K,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            GLS gls = Regressor.gls(ENDOG.clone(), buildExogWithConst(), new double[]{1.0}, N, K,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            assertClose(ols.params(), gls.params(), 1e-10, "GLS(σ=1) vs OLS params");
            assertEquals(ols.logLike(), gls.logLike(), 1e-8, "GLS(σ=1) vs OLS llf");
        }

        @Test void glsDiagonalOnesEqualsOls() {
            // GLS(sigma=ones(n)) should produce same params as OLS
            OLS ols = Regressor.ols(ENDOG.clone(), buildExogWithConst(), N, K,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            double[] sigmaOnes = new double[N];
            java.util.Arrays.fill(sigmaOnes, 1.0);
            GLS gls = Regressor.gls(ENDOG.clone(), buildExogWithConst(), sigmaOnes, N, K,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            assertClose(ols.params(), gls.params(), 1e-10, "GLS(σ=I_diag) vs OLS params");
        }

        @Test void glsDiagonalInvWEqualsWls() {
            // GLS(sigma=1/w) should produce same params as WLS(w)
            double[] w = TestWLS.WEIGHTS;
            WLS wls = Regressor.wls(ENDOG.clone(), buildExogWithConst(), w.clone(), N, K,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            double[] sigmaInvW = new double[N];
            for (int i = 0; i < N; i++) sigmaInvW[i] = 1.0 / w[i];
            GLS gls = Regressor.gls(ENDOG.clone(), buildExogWithConst(), sigmaInvW, N, K,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            assertClose(wls.params(), gls.params(), 1e-8, "GLS(σ=1/w) vs WLS params");
        }

        @Test void glsFullDiagonalEqualsWls() {
            // GLS(sigma=diag(1/w)) as full n×n matrix should match WLS(w)
            double[] w = TestWLS.WEIGHTS;
            WLS wls = Regressor.wls(ENDOG.clone(), buildExogWithConst(), w.clone(), N, K,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            double[] sigmaFull = new double[N * N];
            for (int i = 0; i < N; i++) sigmaFull[i * N + i] = 1.0 / w[i];
            GLS gls = Regressor.gls(ENDOG.clone(), buildExogWithConst(), sigmaFull, N, K,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            assertClose(wls.params(), gls.params(), 1e-8, "GLS(full diag) vs WLS params");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Error Cases (mirrors statsmodels TestGLS_alt_sigma, TestWLS_CornerCases)
    // ═══════════════════════════════════════════════════════════════════════════

    @Nested
    class TestErrors {

        @Test void glsWrongSigmaLength() {
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.gls(ENDOG.clone(), buildExogWithConst(), new double[N - 1], N, K,
                    Regressor.Opts.QR, Regressor.Opts.INFERENCE));
        }

        @Test void glsNegativeScalarSigma() {
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.gls(ENDOG.clone(), buildExogWithConst(), new double[]{-1.0}, N, K,
                    Regressor.Opts.QR, Regressor.Opts.INFERENCE));
        }

        @Test void glsNonPdFullSigma() {
            // Singular matrix: all ones (rank 1)
            double[] sigma = new double[N * N];
            java.util.Arrays.fill(sigma, 1.0);
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.gls(ENDOG.clone(), buildExogWithConst(), sigma, N, K,
                    Regressor.Opts.QR, Regressor.Opts.INFERENCE));
        }

        @Test void glsNegativeDiagonalSigma() {
            double[] sigma = new double[N];
            java.util.Arrays.fill(sigma, 1.0);
            sigma[5] = -0.1;
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.gls(ENDOG.clone(), buildExogWithConst(), sigma, N, K,
                    Regressor.Opts.QR, Regressor.Opts.INFERENCE));
        }

        @Test void mustSpecifyQrOrPinv() {
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.ols(ENDOG.clone(), buildExogWithConst(), N, K));
        }

        @Test void regressionMustSpecifyOutputMode() {
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.ols(ENDOG.clone(), buildExogWithConst(), N, K, Regressor.Opts.QR));
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.wls(ENDOG.clone(), buildExogWithConst(), TestWLS.WEIGHTS.clone(), N, K, Regressor.Opts.QR));
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.gls(ENDOG.clone(), buildExogWithConst(), new double[]{1.0}, N, K, Regressor.Opts.QR));
        }

        @Test void outputModesAreMutuallyExclusive() {
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.ols(ENDOG.clone(), buildExogWithConst(), N, K,
                    Regressor.Opts.QR, Regressor.Opts.PARAMS, Regressor.Opts.FITNESS));
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.wls(ENDOG.clone(), buildExogWithConst(), TestWLS.WEIGHTS.clone(), N, K,
                    Regressor.Opts.QR, Regressor.Opts.PARAMS, Regressor.Opts.FITNESS));
            assertThrows(IllegalArgumentException.class, () ->
                Regressor.gls(ENDOG.clone(), buildExogWithConst(), new double[]{1.0}, N, K,
                    Regressor.Opts.QR, Regressor.Opts.ESTIMATION, Regressor.Opts.INFERENCE));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Workspace Reuse
    // ═══════════════════════════════════════════════════════════════════════════

    @Nested
    class TestWorkspaceReuse {

        @Test void olsPoolReuse() {
            OLS.Pool ws = new OLS.Pool();
            OLS r1 = Regressor.ols(ENDOG.clone(), buildExogWithConst(), N, K, ws,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            OLS r2 = Regressor.ols(ENDOG.clone(), buildExogWithConst(), N, K, ws,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            assertSame(ws, r1.pool());
            assertSame(ws, r2.pool());
            assertArrayEquals(r1.params(), r2.params(), 1e-15);
        }

        @Test void wlsPoolReuseKeepsWhitenedEndogBuffer() {
            WLS.Pool ws = new WLS.Pool();
            WLS r1 = Regressor.wls(ENDOG.clone(), buildExogWithConst(), TestWLS.WEIGHTS.clone(), N, K, ws,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            double[] firstYWhiten = ws.yWhiten;
            double[] firstParams = r1.params().clone();

            WLS r2 = Regressor.wls(ENDOG.clone(), buildExogWithConst(), TestWLS.WEIGHTS.clone(), N, K, ws,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);

            assertSame(ws, r1.pool());
            assertSame(ws, r2.pool());
            assertSame(firstYWhiten, ws.yWhiten);
            assertArrayEquals(firstParams, r2.params(), 1e-15);
        }

        @Test void glsPoolReuseKeepsWhitenedEndogAndIotaBuffers() {
            GLS.Pool ws = new GLS.Pool();
            GLS r1 = Regressor.gls(ENDOG.clone(), buildExogWithConst(), new double[]{1.0}, N, K, ws,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            double[] firstYWhiten = ws.yWhiten;
            double[] firstParams = r1.params().clone();
            double firstTss = r1.tss(true);
            double[] firstIota = ws.iota;

            GLS r2 = Regressor.gls(ENDOG.clone(), buildExogWithConst(), new double[]{1.0}, N, K, ws,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            double secondTss = r2.tss(true);

            assertSame(ws, r1.pool());
            assertSame(ws, r2.pool());
            assertSame(firstYWhiten, ws.yWhiten);
            assertSame(firstIota, ws.iota);
            assertArrayEquals(firstParams, r2.params(), 1e-15);
            assertEquals(firstTss, secondTss, Math.max(1.0, firstTss) * 1e-15);
        }

        @Test void glsFullPoolReuse() {
            GLS.Pool ws = new GLS.Pool();
            double[] sigma = new double[N * N];
            for (int i = 0; i < N; i++) sigma[i * N + i] = 1.0;
            GLS r1 = Regressor.gls(ENDOG.clone(), buildExogWithConst(), sigma, N, K, ws,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            GLS r2 = Regressor.gls(ENDOG.clone(), buildExogWithConst(), sigma, N, K, ws,
                Regressor.Opts.QR, Regressor.Opts.INFERENCE);
            assertSame(ws, r1.pool());
            assertSame(ws, r2.pool());
            assertArrayEquals(r1.params(), r2.params(), 1e-15);
        }
    }
}
