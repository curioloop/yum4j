package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.math.Double3;
import com.curioloop.yum4j.stats.tool.TrendTerms;

import static com.curioloop.yum4j.linalg.Regressor.Opts.FITNESS;
import static com.curioloop.yum4j.linalg.Regressor.Opts.INFERENCE;
import static com.curioloop.yum4j.linalg.Regressor.Opts.PINV;

/**
 * Augmented Dickey-Fuller unit-root test.
 *
 * @param statistic ADF t statistic on the lagged level term
 * @param pValue MacKinnon approximate p-value
 * @param usedLag number of lagged differences included
 * @param nObs observations used in the ADF regression
 * @param critValues MacKinnon critical values at 1%, 5%, and 10%
 * @param icBest selected information criterion, or NaN when autolag is disabled
 */
public record ADF(double statistic, double pValue, int usedLag, int nObs, Double3 critValues, double icBest) {

    public enum AutoLag {
        AIC,
        BIC,
        T_STAT,
        NONE
    }

    public double criticalValue(double alpha) {
        return switch ((int) Math.round(alpha * 100)) {
            case 1 -> critValues._1();
            case 5 -> critValues._2();
            case 10 -> critValues._3();
            default -> throw new IllegalArgumentException("alpha must be 0.01, 0.05, or 0.10");
        };
    }

    public static ADF test(double[] x) {
        return test(x, null, TrendTerms.CONSTANT, AutoLag.AIC);
    }

    public static ADF test(double[] x, Integer maxLag, TrendTerms regression, AutoLag autoLag) {
        TestSupport.checkVector(x, "x");
        if (isConstant(x)) throw new IllegalArgumentException("x is constant");
        TrendTerms reg = regression == null ? TrendTerms.CONSTANT : regression;
        AutoLag lagMethod = autoLag == null ? AutoLag.NONE : autoLag;
        int observations = x.length;
        int trendTerms = reg.term();
        int max = maxLag != null
                ? maxLag
                : (int) Math.ceil(12.0 * Math.pow(observations / 100.0, 0.25));
        max = Math.min(observations / 2 - trendTerms - 1, max);
        if (max < 0) throw new IllegalArgumentException("sample size is too short for selected regression");
        if (maxLag != null && maxLag > observations / 2 - trendTerms - 1) {
            throw new IllegalArgumentException("maxLag is too large for the sample size and deterministic terms");
        }

        LaggedAdf lagged = laggedAdfMatrix(x, max);
        int usedLag = max;
        double icBest = Double.NaN;
        if (lagMethod != AutoLag.NONE) {
            double[] fullRhs = reg == TrendTerms.NONE
                    ? lagged.x
                    : TestSupport.addTrend(lagged.x, lagged.n, max + 1, trendTerms, true);
            int fullK = (reg == TrendTerms.NONE ? 0 : trendTerms) + max + 1;
            int startLag = fullK - (max + 1) + 1;
            AutoLagSelection selection = selectLag(lagged.y, fullRhs, lagged.n, fullK, startLag, max, lagMethod);
            usedLag = selection.bestColumns - startLag;
            icBest = selection.icBest;
            if (usedLag != max) lagged = laggedAdfMatrix(x, usedLag);
        }

        double[] xReg = lagged.x;
        int xRegK = usedLag + 1;
        if (reg != TrendTerms.NONE) {
            xReg = TestSupport.addTrend(xReg, lagged.n, xRegK, trendTerms, false);
            xRegK += trendTerms;
        }
        var ols = Regressor.ols(lagged.y, xReg, lagged.n, xRegK, PINV, INFERENCE);
        double statistic = ols.tStatistics()[0];
        double pValue = MacKinnon.pValue(statistic, reg, 1);
        Double3 crit = MacKinnon.criticalValues(1, reg, lagged.n);
        return new ADF(statistic, pValue, usedLag, lagged.n, crit, icBest);
    }

    private static AutoLagSelection selectLag(double[] y, double[] x, int n, int k, int startLag,
                                              int maxLag, AutoLag method) {
        double best = Double.POSITIVE_INFINITY;
        int bestColumns = startLag;
        int maxColumns = startLag + maxLag;
        double[] candidate = new double[n * maxColumns];
        var pool = new com.curioloop.yum4j.linalg.reg.OLS.Pool();
        if (method == AutoLag.T_STAT) {
            double selected = 0.0;
            for (int columns = startLag + maxLag; columns >= startLag; columns--) {
                TestSupport.copyFirstColumns(x, n, k, columns, candidate);
                var ols = Regressor.ols(y, candidate, n, columns, pool, PINV, INFERENCE);
                selected = Math.abs(ols.tStatistics()[columns - 1]);
                bestColumns = columns;
                if (selected >= 1.6448536269514722) break;
            }
            return new AutoLagSelection(selected, bestColumns);
        }
        for (int columns = startLag; columns <= startLag + maxLag; columns++) {
            TestSupport.copyFirstColumns(x, n, k, columns, candidate);
            var ols = Regressor.ols(y, candidate, n, columns, pool, PINV, FITNESS);
            double ic = method == AutoLag.AIC ? ols.aic() : ols.bic();
            if (ic < best) {
                best = ic;
                bestColumns = columns;
            }
        }
        return new AutoLagSelection(best, bestColumns);
    }

    private static LaggedAdf laggedAdfMatrix(double[] x, int lag) {
        int n = x.length - 1 - lag;
        if (n <= 0) throw new IllegalArgumentException("maxLag leaves no observations");
        int k = lag + 1;
        double[] y = new double[n];
        double[] z = new double[n * k];
        for (int i = 0; i < n; i++) {
            int diffIndex = lag + i;
            y[i] = x[diffIndex + 1] - x[diffIndex];
            z[i * k] = x[diffIndex];
            for (int j = 1; j <= lag; j++) {
                int laggedDiffIndex = diffIndex - j;
                z[i * k + j] = x[laggedDiffIndex + 1] - x[laggedDiffIndex];
            }
        }
        return new LaggedAdf(y, z, n);
    }

    private static boolean isConstant(double[] x) {
        double first = x[0];
        for (int i = 1; i < x.length; i++) if (x[i] != first) return false;
        return true;
    }

    private record LaggedAdf(double[] y, double[] x, int n) {}

    private record AutoLagSelection(double icBest, int bestColumns) {}
}