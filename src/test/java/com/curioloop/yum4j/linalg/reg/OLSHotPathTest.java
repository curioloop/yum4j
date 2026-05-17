package com.curioloop.yum4j.linalg.reg;

import com.curioloop.yum4j.linalg.Regressor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class OLSHotPathTest {

    @Test
    void qrAndSvdAgreeForTallFullRankDesign() {
        int n = 128;
        int k = 5;
        double[] y = new double[n];
        double[] x = new double[n * k];
        fillDesign(y, x, n, k, true, 0x515L);

        OLS qr = Regressor.ols(y, x.clone(), n, k,
            Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);
        OLS svd = Regressor.ols(y, x.clone(), n, k,
            Regressor.Opts.PINV, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);

        assertArrayClose(qr.params(), svd.params(), 1e-8);
        assertEquals(qr.ssr(), svd.ssr(), Math.max(1.0, qr.ssr()) * 1e-9);
        assertEquals(qr.rank(), svd.rank());
    }

    @Test
    void reusedPoolMatchesFreshPoolAcrossChangingColumnCounts() {
        int n = 96;
        int maxK = 6;
        double[] y = new double[n];
        double[] x = new double[n * maxK];
        fillDesign(y, x, n, maxK, true, 0x616L);

        OLS.Pool pool = new OLS.Pool();
        for (int k = 2; k <= maxK; k++) {
            double[] xk = firstColumns(x, n, maxK, k);
            OLS fresh = Regressor.ols(y, xk.clone(), n, k,
                Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);
            OLS reused = Regressor.ols(y, xk.clone(), n, k, pool,
                Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);

            assertArrayClose(fresh.params(), reused.params(), 1e-10);
            assertEquals(fresh.ssr(), reused.ssr(), Math.max(1.0, fresh.ssr()) * 1e-10);
        }
    }

    @Test
    void paramsOutputMatchesFullQrParamsAndRejectsUnavailableStats() {
        int n = 128;
        int k = 5;
        double[] y = new double[n];
        double[] x = new double[n * k];
        fillDesign(y, x, n, k, true, 0x717L);

        OLS inference = Regressor.ols(y, x.clone(), n, k,
            Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);
        OLS paramsOnly = Regressor.ols(y, x.clone(), n, k,
            Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.PARAMS);

        assertArrayClose(inference.params(), paramsOnly.params(), 1e-10);
        assertThrows(IllegalStateException.class, () -> paramsOnly.residual(false));
        assertThrows(IllegalStateException.class, paramsOnly::bse);
    }

    @Test
    void paramsOutputMatchesFullSvdParamsForRankDeficientDesign() {
        int n = 96;
        int k = 4;
        double[] y = new double[n];
        double[] x = new double[n * k];
        Random random = new Random(0x818L);
        for (int i = 0; i < n; i++) {
            double a = Math.sin(i * 0.031) + 0.01 * random.nextGaussian();
            double b = Math.cos(i * 0.047) + 0.01 * random.nextGaussian();
            x[i * k] = 1.0;
            x[i * k + 1] = a;
            x[i * k + 2] = b;
            x[i * k + 3] = 2.0 * a;
            y[i] = 0.7 + 0.4 * a - 0.2 * b + 0.01 * random.nextGaussian();
        }

        OLS inference = Regressor.ols(y, x.clone(), n, k,
            Regressor.Opts.PINV, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);
        OLS paramsOnly = Regressor.ols(y, x.clone(), n, k,
            Regressor.Opts.PINV, Regressor.Opts.HAS_CONST, Regressor.Opts.PARAMS);

        assertArrayClose(inference.params(), paramsOnly.params(), 1e-10);
        assertEquals(inference.rank(), paramsOnly.rank());
        assertEquals(inference.condNum(), paramsOnly.condNum(), Math.max(1.0, inference.condNum()) * 1e-12);
        assertThrows(IllegalStateException.class, () -> paramsOnly.residual(false));
        assertThrows(IllegalStateException.class, paramsOnly::bse);
    }

    @Test
    void pinvProjectionIsStableWithDuplicateConstantColumn() {
        int n = 160;
        int reducedK = 3;
        int duplicateK = 4;
        double[] y = new double[n];
        double[] reduced = new double[n * reducedK];
        double[] duplicate = new double[n * duplicateK];
        Random random = new Random(0xB912L);
        for (int i = 0; i < n; i++) {
            double a = Math.sin(i * 0.043) + 0.02 * random.nextGaussian();
            double b = Math.cos(i * 0.071) + 0.02 * random.nextGaussian();
            reduced[i * reducedK] = 1.0;
            reduced[i * reducedK + 1] = a;
            reduced[i * reducedK + 2] = b;
            duplicate[i * duplicateK] = 1.0;
            duplicate[i * duplicateK + 1] = a;
            duplicate[i * duplicateK + 2] = b;
            duplicate[i * duplicateK + 3] = 1.0;
            y[i] = 0.8 + 0.35 * a - 0.18 * b + 0.01 * random.nextGaussian();
        }

        OLS reducedFit = Regressor.ols(y, reduced.clone(), n, reducedK,
            Regressor.Opts.PINV, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);
        OLS duplicateFit = Regressor.ols(y, duplicate.clone(), n, duplicateK,
            Regressor.Opts.PINV, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);

        assertEquals(reducedFit.rank(), duplicateFit.rank());
        assertEquals(reducedFit.r2(false), duplicateFit.r2(false), 1e-13);
        assertEquals(reducedFit.ssr(), duplicateFit.ssr(), Math.max(1.0, reducedFit.ssr()) * 1e-13);
        assertArrayClose(reducedFit.fitted(false), duplicateFit.fitted(false), 1e-12);
        assertArrayClose(reducedFit.residual(false), duplicateFit.residual(false), 1e-12);
    }

    @Test
    void estimationOutputProvidesResidualsButRejectsCovariance() {
        int n = 128;
        int k = 5;
        double[] y = new double[n];
        double[] x = new double[n * k];
        fillDesign(y, x, n, k, true, 0x919L);

        OLS inference = Regressor.ols(y, x.clone(), n, k,
            Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);
        OLS light = Regressor.ols(y, x.clone(), n, k,
            Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.ESTIMATION);

        assertArrayClose(inference.params(), light.params(), 1e-10);
        assertArrayClose(inference.residual(false), light.residual(false), 1e-10);
        assertEquals(inference.ssr(), light.ssr(), Math.max(1.0, inference.ssr()) * 1e-10);
        assertThrows(IllegalStateException.class, light::bse);
    }

    @Test
    void fitnessOutputProvidesResidualsButRejectsParams() {
        int n = 128;
        int k = 5;
        double[] y = new double[n];
        double[] x = new double[n * k];
        fillDesign(y, x, n, k, true, 0xA20L);

        OLS inference = Regressor.ols(y, x.clone(), n, k,
            Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);
        OLS fitness = Regressor.ols(y, x.clone(), n, k,
                Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);

        assertArrayClose(inference.residual(false), fitness.residual(false), 1e-10);
        assertEquals(inference.ssr(), fitness.ssr(), Math.max(1.0, inference.ssr()) * 1e-10);
        assertThrows(IllegalStateException.class, fitness::params);
        assertThrows(IllegalStateException.class, fitness::bse);
    }

    @Test
    void inferenceOutputProvidesEstimationAndCovariance() {
        int n = 128;
        int k = 5;
        double[] y = new double[n];
        double[] x = new double[n * k];
        fillDesign(y, x, n, k, true, 0xB21L);

        OLS inference = Regressor.ols(y, x.clone(), n, k,
            Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);

        assertEquals(k, inference.params().length);
        assertEquals(n, inference.residual(false).length);
        assertEquals(k * k, inference.paramCov().length);
    }

    @Test
    void poolEnsurePreallocatesQrParamsOnlyPath() {
        int n = 128;
        int k = 5;
        double[] y = new double[n];
        double[] x = new double[n * k];
        fillDesign(y, x, n, k, true, 0xC22L);

        OLS.Pool pool = new OLS.Pool().ensure(n, k,
            Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.PARAMS);
        double[] beta = pool.beta;
        double[] work = pool.work;

        OLS fit = Regressor.ols(y, x.clone(), n, k, pool,
            Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.PARAMS);

        assertSame(beta, pool.beta);
        assertSame(work, pool.work);
        assertSame(beta, fit.params());
        assertNull(pool.fitted);
        assertNull(pool.residual);
        assertNull(pool.unscaledCov);
    }

    @Test
    void poolEnsurePreallocatesSvdInferencePath() {
        int n = 128;
        int k = 5;
        double[] y = new double[n];
        double[] x = new double[n * k];
        fillDesign(y, x, n, k, true, 0xD23L);

        OLS.Pool pool = new OLS.Pool().ensure(n, k,
            Regressor.Opts.PINV, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);
        double[] beta = pool.beta;
        double[] work = pool.work;
        double[] fitted = pool.fitted;
        double[] residual = pool.residual;
        double[] unscaledCov = pool.unscaledCov;

        OLS fit = Regressor.ols(y, x.clone(), n, k, pool,
            Regressor.Opts.PINV, Regressor.Opts.HAS_CONST, Regressor.Opts.INFERENCE);

        assertSame(beta, pool.beta);
        assertSame(work, pool.work);
        assertSame(fitted, fit.fitted(false));
        assertSame(residual, fit.residual(false));
        assertSame(unscaledCov, pool.unscaledCov);
    }

    private static void fillDesign(double[] y, double[] x, int n, int k, boolean hasConst, long seed) {
        Random random = new Random(seed);
        double[] beta = new double[k];
        for (int j = 0; j < k; j++) beta[j] = 0.8 - 0.13 * j;
        for (int i = 0; i < n; i++) {
            double signal = 0.0;
            for (int j = 0; j < k; j++) {
                double value = hasConst && j == 0
                        ? 1.0
                        : Math.sin((i + 1) * (j + 1) * 0.021) + 0.03 * random.nextGaussian();
                x[i * k + j] = value;
                signal += value * beta[j];
            }
            y[i] = signal + 0.01 * random.nextGaussian();
        }
    }

    private static double[] firstColumns(double[] x, int n, int sourceK, int targetK) {
        double[] out = new double[n * targetK];
        for (int i = 0; i < n; i++) {
            System.arraycopy(x, i * sourceK, out, i * targetK, targetK);
        }
        return out;
    }

    private static void assertArrayClose(double[] expected, double[] actual, double relativeTol) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            double tol = Math.max(1.0, Math.abs(expected[i])) * relativeTol;
            assertEquals(expected[i], actual[i], tol, "value[" + i + "]");
        }
    }
}