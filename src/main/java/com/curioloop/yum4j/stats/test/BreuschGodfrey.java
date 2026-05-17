package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.stats.FisherFDistribution;

import static com.curioloop.yum4j.linalg.Regressor.Opts.INFERENCE;
import static com.curioloop.yum4j.linalg.Regressor.Opts.PINV;

/**
 * Breusch-Godfrey LM test for residual serial correlation.
 *
 * @param lmStat LM statistic
 * @param lmPval LM p-value from chi-square(nLags)
 * @param fStat nested auxiliary-regression F statistic
 * @param fPval F-test p-value
 * @param nLags number of residual lags included
 */
public record BreuschGodfrey(double lmStat, double lmPval, double fStat, double fPval, int nLags) {

    public static BreuschGodfrey test(double[] residual, double[] exog, int n, int k) {
        return test(residual, exog, n, k, null);
    }

    public static BreuschGodfrey test(double[] residual, double[] exog, int n, int k, Integer nLags) {
        if (residual.length != n) throw new IllegalArgumentException("residual length must equal n");
        TestSupport.checkMatrix(exog, n, k, "exog");
        int lags = nLags != null ? nLags : Math.min(10, n / 5);
        if (lags < 1) throw new IllegalArgumentException("nLags must be positive");

        int restrictedK = k + 1;
        int unrestrictedK = restrictedK + lags;
        double[] unrestrictedX = new double[n * unrestrictedK];
        for (int i = 0; i < n; i++) {
            BLAS.dcopy(k, exog, i * k, 1, unrestrictedX, i * unrestrictedK, 1);
            unrestrictedX[i * unrestrictedK + k] = 1.0;
            for (int lag = 1; lag <= lags; lag++) {
                int source = i - lag;
                unrestrictedX[i * unrestrictedK + restrictedK + lag - 1] = source >= 0 ? residual[source] : 0.0;
            }
        }

        var unrestricted = Regressor.ols(residual, unrestrictedX, n, unrestrictedK, PINV, INFERENCE);
        double lm = n * unrestricted.r2(false);
        double lmP = new ChiSquareDistribution(lags).ccdf(lm);
        double f = waldF(unrestricted, restrictedK, lags);
        int dfDenom = n - unrestricted.rank();
        double fP = Double.isFinite(f) && dfDenom > 0 ? new FisherFDistribution(lags, dfDenom).ccdf(f) : Double.NaN;
        return new BreuschGodfrey(lm, lmP, f, fP, lags);
    }

    private static double waldF(com.curioloop.yum4j.linalg.reg.OLS unrestricted, int start, int lags) {
        double[] params = unrestricted.params();
        double[] cov = unrestricted.paramCov();
        int k = unrestricted.nParams();
        double wald = TestSupport.quadraticFormInverse(cov, k, start, params, start, lags);
        return wald / lags;
    }
}
