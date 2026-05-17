package com.curioloop.yum4j.stats.tool;

import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;

/**
 * Levinson-Durbin recursion for autoregressive processes.
 *
 * <p>Given the autocovariance sequence {@code s[0], s[1], ..., s[p]}, this algorithm
 * solves the Yule-Walker equations to produce AR coefficients, partial autocorrelations,
 * and the innovation variance at each order from 1 to {@code p}.</p>
 *
 * <h3>Recursion (Durbin 1960)</h3>
 * <pre>{@code
 * φ_{k,k} = (s[k] - Σ_{j=1}^{k-1} φ_{j,k-1} · s[k-j]) / σ²_{k-1}
 * φ_{j,k} = φ_{j,k-1} - φ_{k,k} · φ_{k-j,k-1}   for j = 1..k-1
 * σ²_k    = σ²_{k-1} · (1 - φ_{k,k}²)
 * }</pre>
 *
 * <p>Mirrors {@code statsmodels.tsa.stattools.levinson_durbin} and
 * {@code levinson_durbin_pacf}.</p>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * // From autocovariance
 * LevinsonDurbin ld = LevinsonDurbin.of(acov, 10);
 * double[] ar = ld.arCoefs();
 * double[] pacf = ld.pacf();
 *
 * // From raw data
 * LevinsonDurbin ld = LevinsonDurbin.ofData(x, 10);
 *
 * // Inverse: PACF → AR coefficients + ACF
 * LevinsonDurbin.Inverse inv = LevinsonDurbin.fromPacf(pacf);
 * }</pre>
 *
 * @param sigmaV   innovation variance of the AR({@code p}) model
 * @param arCoefs  AR coefficients φ₁, …, φ_p for the final order (length {@code p})
 * @param pacf     partial autocorrelations for lags 0, 1, …, p (length {@code p+1});
 *                 lag-0 entry is always 1.0
 * @param sigma    innovation variances σ²_0, σ²_1, …, σ²_p (length {@code p+1};
 *                 index 0 is unused / 0.0)
 */
public record LevinsonDurbin(double sigmaV, double[] arCoefs, double[] pacf, double[] sigma) {

    /**
     * Result of the inverse Levinson-Durbin (PACF → AR + ACF).
     *
     * @param arCoefs AR coefficients recovered from the PACF
     * @param acf     autocorrelation function for lags 0, 1, …, p (acf[0] = 1.0)
     */
    public record Inverse(double[] arCoefs, double[] acf) {}

    // (Result type alias removed — use LevinsonDurbin directly)

    // ═══════════════════════════════════════════════════════════════════════════
    // Factory methods
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Runs the Levinson-Durbin recursion on an autocovariance sequence.
     *
     * @param acov   autocovariance array starting at lag 0 (length ≥ {@code nLags+1})
     * @param nLags  AR order / maximum lag
     * @return recursion result
     */
    public static LevinsonDurbin of(double[] acov, int nLags) {
        return compute(acov, nLags, true);
    }

    /**
     * Runs the Levinson-Durbin recursion on raw time-series data.
     *
     * <p>The autocovariance is estimated internally using the biased (non-FFT) estimator.</p>
     *
     * @param data   time-series observations
     * @param nLags  AR order / maximum lag
     * @return recursion result
     */
    public static LevinsonDurbin ofData(double[] data, int nLags) {
        return compute(data, nLags, false);
    }

    /**
     * Runs the Levinson-Durbin recursion.
     *
     * @param s      autocovariance array starting at lag 0, or the raw time series
     *               (see {@code isAcov})
     * @param nLags  AR order / maximum lag
     * @param isAcov if {@code true}, {@code s} is interpreted as the autocovariance
     *               sequence; if {@code false}, the autocovariance is estimated from
     *               the data using the biased (non-FFT) estimator
     * @return recursion result
     */
    public static LevinsonDurbin compute(double[] s, int nLags, boolean isAcov) {
        if (s.length == 0) throw new IllegalArgumentException("s must not be empty");
        if (nLags < 1)     throw new IllegalArgumentException("nLags must be >= 1");

        double[] sxx;
        if (isAcov) {
            if (s.length < nLags + 1)
                throw new IllegalArgumentException("acov array must have at least nLags+1 elements");
            sxx = s;
        } else {
            sxx = AutoCovariance.compute(s, false, true, false,
                    MissingPolicy.NONE, nLags);
        }

        int order = nLags;
        double[] curr = new double[order + 1];
        double[] prev = new double[order + 1];
        double[] sig  = new double[order + 1];
        double[] pacf = new double[order + 1];

        // Initial step (k=1)
        curr[1] = sxx[1] / sxx[0];
        sig[1]  = sxx[0] - curr[1] * sxx[1];
        pacf[1] = curr[1];

        for (int k = 2; k <= order; k++) {
            // Swap buffers: prev ← curr
            double[] tmp = prev; prev = curr; curr = tmp;

            // Compute reflection coefficient φ_{k,k}
            double num = sxx[k] - BLAS.ddot(k - 1, prev, 1, 1, sxx, k - 1, -1);
            curr[k] = num / sig[k - 1];
            pacf[k] = curr[k];

            // Update lower-order coefficients in-place
            for (int j = 1; j < k; j++) {
                curr[j] = prev[j] - curr[k] * prev[k - j];
            }
            sig[k] = sig[k - 1] * (1.0 - curr[k] * curr[k]);
        }

        double sigmaV = sig[order];
        double[] arCoefs = new double[order];
        for (int j = 1; j <= order; j++) arCoefs[j - 1] = curr[j];
        pacf[0] = 1.0;

        return new LevinsonDurbin(sigmaV, arCoefs, pacf, sig);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Inverse Levinson-Durbin
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Inverse Levinson-Durbin: given a PACF array, recover AR coefficients and ACF.
     *
     * <p>Mirrors {@code statsmodels.tsa.stattools.levinson_durbin_pacf}.</p>
     *
     * @param pacf   partial autocorrelations for lags 0, 1, …, p (pacf[0] must be 1.0)
     * @param nLags  number of lags to use; {@code null} uses all lags in {@code pacf}
     * @return inverse result containing AR coefficients and ACF
     */
    public static Inverse fromPacf(double[] pacf, Integer nLags) {
        if (pacf[0] != 1.0)
            throw new IllegalArgumentException("pacf[0] must be 1.0 (lag-0 partial autocorrelation)");

        int n = (nLags != null) ? nLags : pacf.length - 1;
        if (nLags != null && nLags > pacf.length - 1)
            throw new IllegalArgumentException("nLags exceeds available pacf values");

        double[] acf     = new double[n + 1];
        double[] arCoefs = Arrays.copyOfRange(pacf, 1, n + 1);
        double[] nu      = new double[n];

        acf[1] = arCoefs[0];
        nu[0]  = 1.0 - arCoefs[0] * arCoefs[0];
        for (int i = 1; i < n; i++) nu[i] = nu[i - 1] * (1.0 - arCoefs[i] * arCoefs[i]);

        // Reused scratch: avoids O(n) allocation per iteration (O(n²) total garbage).
        double[] prev = new double[n];
        for (int i = 1; i < n; i++) {
            int len = i;
            System.arraycopy(arCoefs, 0, prev, 0, len);
            for (int j = 0; j < len; j++) {
                arCoefs[j] = prev[j] - arCoefs[i] * prev[len - 1 - j];
            }
            double dot = BLAS.ddot(len, prev, 0, 1, acf, i, -1);
            acf[i + 1] = arCoefs[i] * nu[i - 1] + dot;
        }
        acf[0] = 1.0;

        return new Inverse(arCoefs, acf);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Instance methods (convenience accessors on the record)
    // ═══════════════════════════════════════════════════════════════════════════

    /** Returns the AR order (number of coefficients). */
    public int order() {
        return arCoefs.length;
    }

    /** Returns the reflection coefficient (PACF) at the given lag (1-based). */
    public double reflectionCoefficient(int lag) {
        return pacf[lag];
    }
}
