package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.stats.FisherFDistribution;

import static com.curioloop.yum4j.linalg.Regressor.Opts.FITNESS;
import static com.curioloop.yum4j.linalg.Regressor.Opts.PINV;

/**
 * Granger non-causality tests for a two-column time-series matrix.
 *
 * <p>The null hypothesis is that the second series does not Granger-cause the
 * first series. Input is row-major {@code n x 2}: column 0 is the dependent
 * series and column 1 is the candidate causing series.</p>
 *
 * @param lags test results for each requested lag
 */
public record GrangerCausality(Lag[] lags) {

    public record Lag(int lag,
                      double ssrFStat, double ssrFPValue, int ssrFDfDenom, int ssrFDfNum,
                      double ssrChi2Stat, double ssrChi2PValue, int ssrChi2Df,
                      double lrStat, double lrPValue, int lrDf,
                      double paramsFStat, double paramsFPValue, int paramsFDfDenom, int paramsFDfNum) {}

    public static GrangerCausality test(double[] x, int n, int maxLag) {
        if (maxLag <= 0) throw new IllegalArgumentException("maxLag must be positive");
        TestSupport.checkMatrix(x, n, 2, "x");
        if (n <= 3 * maxLag + 1) {
            throw new IllegalArgumentException("insufficient observations for requested maxLag");
        }
        Lag[] out = new Lag[maxLag];
        for (int i = 0; i < maxLag; i++) out[i] = testLag(x, n, i + 1);
        return new GrangerCausality(out);
    }

    public static GrangerCausality test(double[] x, int n, int... lags) {
        TestSupport.checkMatrix(x, n, 2, "x");
        if (lags.length == 0) throw new IllegalArgumentException("at least one lag is required");
        int maxLag = 0;
        for (int lag : lags) {
            if (lag <= 0) throw new IllegalArgumentException("lags must be positive");
            maxLag = Math.max(maxLag, lag);
        }
        if (n <= 3 * maxLag + 1) {
            throw new IllegalArgumentException("insufficient observations for requested maxLag");
        }

        Lag[] out = new Lag[lags.length];
        for (int i = 0; i < lags.length; i++) out[i] = testLag(x, n, lags[i]);
        return new GrangerCausality(out);
    }

    public Lag lag(int lag) {
        for (Lag result : lags) if (result.lag == lag) return result;
        throw new IllegalArgumentException("lag not present: " + lag);
    }

    private static Lag testLag(double[] x, int n, int lag) {
        int rows = n - lag;
        double[] y = new double[rows];
        double[] restrictedX = new double[rows * (lag + 1)];
        double[] unrestrictedX = new double[rows * (2 * lag + 1)];
        for (int i = 0; i < rows; i++) {
            int t = i + lag;
            y[i] = x[t * 2];
            for (int j = 1; j <= lag; j++) {
                double own = x[(t - j) * 2];
                double other = x[(t - j) * 2 + 1];
                restrictedX[i * (lag + 1) + j - 1] = own;
                unrestrictedX[i * (2 * lag + 1) + j - 1] = own;
                unrestrictedX[i * (2 * lag + 1) + lag + j - 1] = other;
            }
            restrictedX[i * (lag + 1) + lag] = 1.0;
            unrestrictedX[i * (2 * lag + 1) + 2 * lag] = 1.0;
        }

        var restricted = Regressor.ols(y, restrictedX, rows, lag + 1, PINV, FITNESS);
        var unrestricted = Regressor.ols(y, unrestrictedX, rows, 2 * lag + 1, PINV, FITNESS);
        int dfDenom = rows - unrestricted.rank();
        double ssrF = TestSupport.restrictedF(restricted.ssr(), unrestricted.ssr(), lag, rows, unrestricted.rank());
        double ssrFP = Double.isFinite(ssrF) && dfDenom > 0 ? new FisherFDistribution(lag, dfDenom).ccdf(ssrF) : Double.NaN;
        double ssrChi2 = restricted.nObs() * (restricted.ssr() - unrestricted.ssr()) / unrestricted.ssr();
        double ssrChi2P = new ChiSquareDistribution(lag).ccdf(ssrChi2);
        double lr = -2.0 * (restricted.logLike() - unrestricted.logLike());
        double lrP = new ChiSquareDistribution(lag).ccdf(lr);
        return new Lag(lag, ssrF, ssrFP, dfDenom, lag,
                ssrChi2, ssrChi2P, lag,
                lr, lrP, lag,
                ssrF, ssrFP, dfDenom, lag);
    }
}