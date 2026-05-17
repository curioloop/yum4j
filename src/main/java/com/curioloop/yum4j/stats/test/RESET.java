package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.stats.FisherFDistribution;

import java.util.Arrays;

import static com.curioloop.yum4j.linalg.Regressor.Opts.FITNESS;
import static com.curioloop.yum4j.linalg.Regressor.Opts.PINV;

/**
 * Ramsey RESET test using powers of fitted values.
 *
 * @param statistic F statistic when {@code fTest=true}, otherwise chi-square statistic
 * @param pValue p-value of the selected statistic
 * @param dfNum number of tested restrictions
 * @param dfDenom denominator degrees of freedom for the F form
 * @param fTest whether the reported statistic uses the F distribution
 */
public record RESET(double statistic, double pValue, int dfNum, int dfDenom, boolean fTest) {

    public static RESET test(double[] y, double[] exog, int n, int k) {
        return test(y, exog, n, k, 3, false);
    }

    public static RESET test(double[] y, double[] exog, int n, int k, int power, boolean useF) {
        if (power < 2) throw new IllegalArgumentException("power must be >= 2");
        int[] powers = new int[power - 1];
        for (int i = 0; i < powers.length; i++) powers[i] = i + 2;
        return test(y, exog, n, k, useF, powers);
    }

    public static RESET test(double[] y, double[] exog, int n, int k, boolean useF, int... powers) {
        TestSupport.checkMatrix(exog, n, k, "exog");
        if (y.length != n) throw new IllegalArgumentException("y length must equal n");
        if (powers.length == 0) throw new IllegalArgumentException("at least one power is required");
        int[] sorted = powers.length == 1 ? powers : Arrays.copyOf(powers, powers.length);
        if (sorted.length > 1) Arrays.sort(sorted);
        for (int i = 0; i < sorted.length; i++) {
            if (sorted[i] < 2) throw new IllegalArgumentException("powers must be >= 2");
            if (i > 0 && sorted[i] == sorted[i - 1]) throw new IllegalArgumentException("powers must be distinct");
        }

        var restricted = Regressor.ols(y, Arrays.copyOf(exog, n * k), n, k, PINV, FITNESS);
        double[] fitted = restricted.fitted(false);
        int unrestrictedK = k + sorted.length;
        double[] unrestrictedX = new double[n * unrestrictedK];
        for (int i = 0; i < n; i++) {
            BLAS.dcopy(k, exog, i * k, 1, unrestrictedX, i * unrestrictedK, 1);
            for (int j = 0; j < sorted.length; j++) {
                unrestrictedX[i * unrestrictedK + k + j] = Math.pow(fitted[i], sorted[j]);
            }
        }

        var unrestricted = Regressor.ols(y, unrestrictedX, n, unrestrictedK, PINV, FITNESS);
        int restrictions = sorted.length;
        int dfDenom = n - unrestricted.rank();
        double f = TestSupport.restrictedF(restricted.ssr(), unrestricted.ssr(), restrictions, n, unrestricted.rank());
        if (useF) {
            double p = Double.isFinite(f) && dfDenom > 0 ? new FisherFDistribution(restrictions, dfDenom).ccdf(f) : Double.NaN;
            return new RESET(f, p, restrictions, dfDenom, true);
        }
        double chi2 = f * restrictions;
        double p = Double.isFinite(chi2) ? new ChiSquareDistribution(restrictions).ccdf(chi2) : Double.NaN;
        return new RESET(chi2, p, restrictions, dfDenom, false);
    }
}