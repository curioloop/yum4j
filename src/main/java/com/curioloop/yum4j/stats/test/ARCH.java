package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.stats.FisherFDistribution;

/**
 * Engle's ARCH test for autoregressive conditional heteroskedasticity.
 *
 * <p>Tests whether the squared residuals exhibit serial autocorrelation, which
 * is the signature of ARCH effects (time-varying volatility).</p>
 *
 * <h3>Procedure</h3>
 * <ol>
 *   <li>Square the residuals: {@code u_t = e_t²}</li>
 *   <li>Regress {@code u_t} on a constant and {@code u_{t-1}, …, u_{t-p}}</li>
 *   <li>LM statistic: {@code LM = (n - ddof) · R²} ~ χ²(p)</li>
 *   <li>F statistic: {@code F = R²/p / ((1-R²)/(n-p-1))} ~ F(p, n-p-1)</li>
 * </ol>
 *
 * <p>Mirrors {@code statsmodels.stats.diagnostic.het_arch} (which delegates to
 * {@code acorr_lm} on the squared residuals).</p>
 *
 * @param lmStat  Lagrange multiplier test statistic ((n-ddof)·R²)
 * @param lmPval  p-value of the LM statistic (χ²(nLags))
 * @param fStat   F-statistic for the parameter restriction test
 * @param fPval   p-value of the F-statistic
 * @param nLags   number of lags used
 */
public record ARCH(double lmStat, double lmPval, double fStat, double fPval, int nLags) {

    /**
     * Runs the ARCH test on residuals.
     *
     * @param resid  residuals from a fitted model (or any time series)
     * @param nLags  number of lags; {@code null} → {@code min(10, nobs/5)}
     * @param ddof   degrees-of-freedom correction for the LM statistic
     * @return test result
     */
    public static ARCH test(double[] resid, Integer nLags, int ddof) {
        int nobs = resid.length;
        if (nobs < 4) throw new IllegalArgumentException("series too short for ARCH test");
        return acorrLm(resid, nLags, ddof, true);
    }

    /** Convenience overload: no ddof correction, automatic lag selection. */
    public static ARCH test(double[] resid) {
        return test(resid, null, 0);
    }

    /**
     * Generic LM test for autocorrelation.
     *
     * <p>Regresses {@code resid[t]} on a constant and {@code resid[t-1], …, resid[t-nLags]},
     * then computes the LM and F statistics from the R².</p>
     *
     * <p>Mirrors {@code statsmodels.stats.diagnostic.acorr_lm}.</p>
     */
    public static ARCH acorrLm(double[] resid, Integer nLags, int ddof) {
        return acorrLm(resid, nLags, ddof, false);
    }

    private static ARCH acorrLm(double[] resid, Integer nLags, int ddof, boolean square) {
        int nobs = resid.length;
        int maxLag = (nLags != null) ? nLags : Math.min(10, nobs / 5);
        if (maxLag < 1) throw new IllegalArgumentException("nLags must be >= 1");

        int n = nobs - maxLag;
        double[] X = new double[n * (maxLag + 1)];
        double[] y = new double[n];
        for (int t = 0; t < n; t++) {
            y[t] = value(resid, maxLag + t, square);
            X[t * (maxLag + 1)] = 1.0;
            for (int j = 1; j <= maxLag; j++) {
                X[t * (maxLag + 1) + j] = value(resid, maxLag + t - j, square);
            }
        }

        var ols = Regressor.ols(y, X, n, maxLag + 1,
            Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);
        double r2 = ols.r2(false);
        double lm = (n - ddof) * r2;
        double lmPval = new ChiSquareDistribution(maxLag).ccdf(lm);

        int dfNum = maxLag;
        int dfDen = n - maxLag - 1;
        double fStat = (dfDen > 0 && r2 < 1.0)
                ? (r2 / dfNum) / ((1.0 - r2) / dfDen)
                : Double.NaN;
        double fPval = (Double.isNaN(fStat) || dfDen <= 0)
                ? Double.NaN
                : new FisherFDistribution(dfNum, dfDen).ccdf(fStat);

        return new ARCH(lm, lmPval, fStat, fPval, maxLag);
    }

    private static double value(double[] x, int index, boolean square) {
        double value = x[index];
        return square ? value * value : value;
    }
}
