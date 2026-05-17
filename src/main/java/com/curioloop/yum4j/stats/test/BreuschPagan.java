package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.math.VectorOps;

import static com.curioloop.yum4j.linalg.Regressor.Opts.FITNESS;
import static com.curioloop.yum4j.linalg.Regressor.Opts.PINV;

/**
 * Breusch-Pagan and Koenker heteroskedasticity test.
 *
 * @param lmStat LM statistic
 * @param lmPval LM p-value from chi-square(df = number of non-constant regressors)
 * @param fStat auxiliary-regression F statistic
 * @param fPval F-test p-value
 */
public record BreuschPagan(double lmStat, double lmPval, double fStat, double fPval) {

    public static BreuschPagan test(double[] residual, double[] exog, int n, int k) {
        return test(residual, exog, n, k, true);
    }

    public static BreuschPagan test(double[] residual, double[] exog, int n, int k, boolean robust) {
        if (residual.length != n) throw new IllegalArgumentException("residual length must equal n");
        TestSupport.checkHetExog(exog, n, k, "The Breusch-Pagan");
        // Fuse squaring + sum-of-squares into one pass (the squared residuals are
        // needed as the dependent variable y anyway, and Σe² feeds the rescaling).
        double[] y = new double[n];
        double sumSq = 0.0;
        for (int i = 0; i < n; i++) {
            double e2 = residual[i] * residual[i];
            y[i] = e2;
            sumSq += e2;
        }
        if (!robust) {
            VectorOps.scal((double) n / sumSq, y);
        }

        var aux = Regressor.ols(y, exog.clone(), n, k, PINV, FITNESS);
        double lm = robust ? n * aux.r2(false) : aux.ess() / 2.0;
        double lmP = new ChiSquareDistribution(k - 1).ccdf(lm);
        return new BreuschPagan(lm, lmP, aux.fStatistic(), aux.fPValue());
    }
}