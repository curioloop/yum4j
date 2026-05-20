package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.math.VectorOps;
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

    private static final double INFEASIBLE_EPS = Math.ulp(1.0);

    public record Lag(int lag,
                      double ssrFStat, double ssrFPValue, int ssrFDfDenom, int ssrFDfNum,
                      double ssrChi2Stat, double ssrChi2PValue, int ssrChi2Df,
                      double lrStat, double lrPValue, int lrDf,
                      double paramsFStat, double paramsFPValue, int paramsFDfDenom, int paramsFDfNum) {}

    public static GrangerCausality test(double[] dependent, double[] causing, int maxLag) {
        return test(twoColumn(dependent, causing), dependent.length, maxLag);
    }

    public static GrangerCausality test(double[] dependent, double[] causing, int... lags) {
        return test(twoColumn(dependent, causing), dependent.length, lags);
    }

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
        checkFeasible(y, rows, restrictedX, lag, unrestrictedX, 2 * lag, restricted.ssr(), unrestricted.ssr());
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

    private static double[] twoColumn(double[] dependent, double[] causing) {
        if (dependent.length != causing.length) {
            throw new IllegalArgumentException("dependent and causing lengths must match");
        }
        double[] out = new double[dependent.length * 2];
        BLAS.dcopy(dependent.length, dependent, 0, 1, out, 0, 2);
        BLAS.dcopy(causing.length, causing, 0, 1, out, 1, 2);
        return out;
    }

    private static void checkFeasible(double[] y, int rows,
                                      double[] restrictedX, int restrictedColumnsWithoutConstant,
                                      double[] unrestrictedX, int unrestrictedColumnsWithoutConstant,
                                      double restrictedSsr, double unrestrictedSsr) {
        double totalSumSquares = centeredSumSquares(y, rows);
        if (totalSumSquares == 0.0 || restrictedSsr == 0.0 || unrestrictedSsr == 0.0
                || unrestrictedSsr / totalSumSquares < INFEASIBLE_EPS
                || hasConstantColumn(restrictedX, rows, restrictedColumnsWithoutConstant, restrictedColumnsWithoutConstant + 1)
                || hasConstantColumn(unrestrictedX, rows, unrestrictedColumnsWithoutConstant, unrestrictedColumnsWithoutConstant + 1)) {
            throw new IllegalArgumentException("Granger causality test is infeasible because the regression has a perfect fit or constant column");
        }
    }

    private static double centeredSumSquares(double[] values, int length) {
        return VectorOps.sumSq(values, 0, length, VectorOps.mean(values, 0, length));
    }

    private static boolean hasConstantColumn(double[] matrix, int rows, int columnsToCheck, int totalColumns) {
        for (int column = 0; column < columnsToCheck; column++) {
            double first = matrix[column];
            boolean constant = true;
            for (int row = 1; row < rows; row++) {
                if (matrix[row * totalColumns + column] != first) {
                    constant = false;
                    break;
                }
            }
            if (constant) return true;
        }
        return false;
    }
}