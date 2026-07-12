package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.linalg.reg.OLS;
import com.curioloop.yum4j.math.Double3;
import com.curioloop.yum4j.stats.tool.TrendTerms;

import java.util.Arrays;

import static com.curioloop.yum4j.linalg.Regressor.Opts.FITNESS;
import static com.curioloop.yum4j.linalg.Regressor.Opts.PINV;

/**
 * Augmented Engle-Granger two-step cointegration test.
 *
 * @param statistic ADF statistic on first-stage residuals
 * @param pValue MacKinnon approximate p-value for no cointegration
 * @param critValues MacKinnon critical values at 1%, 5%, and 10%
 */
public record Coint(double statistic, double pValue, Double3 critValues) {

    private static final double SQRTEPS = 1.4901161193847656e-8;

    public static Coint test(double[] y0, double[] y1) {
        return test(y0, y1, y0.length, 1, TrendTerms.CONSTANT, null, ADF.AutoLag.AIC);
    }

    public static Coint test(double[] y0, double[] y1, int n, int k) {
        return test(y0, y1, n, k, TrendTerms.CONSTANT, null, ADF.AutoLag.AIC);
    }

    public static Coint test(double[] y0, double[] y1, int n, int k, TrendTerms trend,
                             Integer maxLag, ADF.AutoLag autoLag) {
        if (y0.length != n) throw new IllegalArgumentException("y0 length must equal n");
        TestSupport.checkVector(y0, "y0");
        TestSupport.checkMatrix(y1, n, k, "y1");
        TrendTerms tr = trend == null ? TrendTerms.CONSTANT : trend;
        int firstStageK = k + tr.term();
        OLS firstStage;
        if (tr == TrendTerms.NONE) {
            firstStage = Regressor.ols(y0, Arrays.copyOf(y1, n * firstStageK), n, firstStageK, PINV, FITNESS);
        } else {
            double[] firstStageX = TestSupport.addTrend(y1, n, k, tr.term(), false);
            firstStage = Regressor.ols(y0, firstStageX, n, firstStageK, PINV, FITNESS);
        }
        int seriesCount = k + 1;

        double stat;
        if (firstStage.r2(false) < 1.0 - 100.0 * SQRTEPS) {
            stat = ADF.test(firstStage.residual(false), maxLag, TrendTerms.NONE, autoLag).statistic();
        } else {
            stat = Double.NEGATIVE_INFINITY;
        }

        double pValue = MacKinnon.pValue(stat, tr, seriesCount);
        Double3 crit = tr == TrendTerms.NONE
                ? new Double3(Double.NaN, Double.NaN, Double.NaN)
                : MacKinnon.criticalValues(seriesCount, tr, n - 1);
        return new Coint(stat, pValue, crit);
    }
}
