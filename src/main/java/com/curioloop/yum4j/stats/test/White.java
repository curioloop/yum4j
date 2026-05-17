package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.stats.ChiSquareDistribution;

import static com.curioloop.yum4j.linalg.Regressor.Opts.FITNESS;
import static com.curioloop.yum4j.linalg.Regressor.Opts.PINV;

/**
 * White's LM test for heteroskedasticity.
 *
 * @param lmStat LM statistic
 * @param lmPval LM p-value
 * @param fStat auxiliary-regression F statistic
 * @param fPval F-test p-value
 */
public record White(double lmStat, double lmPval, double fStat, double fPval) {

    public static White test(double[] residual, double[] exog, int n, int k) {
        return test(residual, exog, n, k, true);
    }

    public static White test(double[] residual, double[] exog, int n, int k, boolean interactionTerms) {
        if (residual.length != n) throw new IllegalArgumentException("residual length must equal n");
        TestSupport.checkHetExog(exog, n, k, "White's heteroskedasticity");
        double[] y = new double[n];
        for (int i = 0; i < n; i++) y[i] = residual[i] * residual[i];

        int auxK = interactionTerms ? k * (k + 1) / 2 : k;
        double[] auxX = new double[n * auxK];
        for (int i = 0; i < n; i++) {
            int out = i * auxK;
            if (interactionTerms) {
                for (int a = 0; a < k; a++) {
                    double xa = exog[i * k + a];
                    for (int b = a; b < k; b++) auxX[out++] = xa * exog[i * k + b];
                }
            } else {
                for (int j = 0; j < k; j++) {
                    double x = exog[i * k + j];
                    auxX[out + j] = x * x;
                }
            }
        }

        var aux = Regressor.ols(y, auxX, n, auxK, PINV, FITNESS);
        double lm = n * aux.r2(false);
        int df = aux.rank() - aux.kConst();
        double lmP = df > 0 ? new ChiSquareDistribution(df).ccdf(lm) : Double.NaN;
        return new White(lm, lmP, aux.fStatistic(), aux.fPValue());
    }
}