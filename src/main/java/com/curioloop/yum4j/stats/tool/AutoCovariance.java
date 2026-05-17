package com.curioloop.yum4j.stats.tool;

import com.curioloop.yum4j.fft.Transform;
import com.curioloop.yum4j.math.VectorOps;

import java.util.Arrays;

/**
 * Autocovariance function estimator.
 *
 * <p>Mirrors {@code statsmodels.tsa.stattools.acovf} exactly, including all four
 * missing-data strategies and both the direct and FFT computation paths.</p>
 *
 * <h3>Formula</h3>
 * <pre>{@code
 * acovf[k] = (1/d[k]) * Σ (x[t] - x̄)(x[t+k] - x̄)
 * }</pre>
 * where {@code d[k] = n - k} when {@code adjusted=true}, or {@code d[k] = n} otherwise.
 *
 * <h3>Missing-data handling</h3>
 * <ul>
 *   <li>{@link MissingPolicy#NONE} — no NaN checks (fastest)</li>
 *   <li>{@link MissingPolicy#RAISE} — throws {@link IllegalArgumentException} if NaNs present</li>
 *   <li>{@link MissingPolicy#DROP} — removes NaN observations, treats remainder as contiguous</li>
 *   <li>{@link MissingPolicy#CONSERVATIVE} — replaces NaN with 0 before computing, adjusts
 *       denominator by the number of non-missing pairs at each lag</li>
 * </ul>
 */
public final class AutoCovariance {

    private AutoCovariance() {}

    // (MissingPolicy enum has been promoted to a package-level type)

    /**
     * Estimates the autocovariance function of {@code x}.
     *
     * @param x        input time series (not modified)
     * @param adjusted if {@code true}, divide by {@code n-k}; otherwise divide by {@code n}
     * @param demean   if {@code true}, subtract the sample mean before computing
     * @param fft      if {@code true}, use FFT-based computation (preferred for long series)
     * @param missing  strategy for handling NaN values
     * @param nLag     maximum lag to return (inclusive); {@code null} returns all {@code n-1} lags
     * @return array of length {@code nLag+1} (or {@code n}) containing acovf[0..nLag]
     * @throws IllegalArgumentException if {@code x} is empty, {@code nLag >= n}, or NaNs found
     *                                  with {@link MissingPolicy#RAISE}
     */
    public static double[] compute(double[] x, boolean adjusted, boolean demean,
                                   boolean fft, MissingPolicy missing, Integer nLag) {
        if (x.length == 0) {
            throw new IllegalArgumentException("input array x cannot be empty");
        }

        // ── Missing-data pre-processing ───────────────────────────────
        boolean hasMissing = (missing != MissingPolicy.NONE) && hasNaN(x);
        if (hasMissing && missing == MissingPolicy.RAISE) {
            throw new IllegalArgumentException("NaN values found in input (MissingPolicy.RAISE)");
        }

        double[] notMaskInt = null; // 1.0 for non-NaN, 0.0 for NaN
        if (hasMissing) {
            notMaskInt = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                notMaskInt[i] = Double.isNaN(x[i]) ? 0.0 : 1.0;
            }
            if (missing == MissingPolicy.DROP) {
                x = dropNaN(x); // already a fresh array
            } else if (missing == MissingPolicy.CONSERVATIVE) {
                x = Arrays.copyOf(x, x.length);
                for (int i = 0; i < x.length; i++) {
                    if (Double.isNaN(x[i])) x[i] = 0.0;
                }
            } else {
                x = Arrays.copyOf(x, x.length);
            }
        } else if (demean) {
            // demean mutates x in-place via addConst; defensive copy required.
            // When demean=false and no missing, x is only read (VectorOps.dot / FFT pad)
            // so the copy can be skipped — the public API's "not modified" contract
            // is preserved.
            x = Arrays.copyOf(x, x.length);
        }

        // ── Demean ────────────────────────────────────────────────────
        if (demean) {
            double mean;
            if (notMaskInt == null) {
                mean = VectorOps.mean(x);
            } else {
                mean = VectorOps.sum(x) / VectorOps.sum(notMaskInt);
            }
            VectorOps.addConst(-mean, x);
            if (hasMissing && missing == MissingPolicy.CONSERVATIVE) {
                for (int i = 0; i < x.length; i++) x[i] *= notMaskInt[i];
            }
        }

        int n = x.length;
        int lagLen = (nLag == null) ? n - 1 : nLag;
        if (lagLen > n - 1) {
            throw new IllegalArgumentException("nLag must be smaller than nobs - 1");
        }

        // ── Direct path (only when fft=false and nLag is specified) ───
        // When nLag is null (all lags requested), the direct correlate path is O(n²).
        // FFT is always faster for full-length computation, so we upgrade automatically.
        if (!fft && nLag != null) {
            return directAcov(x, n, lagLen, adjusted, notMaskInt, missing);
        }

        // ── FFT path ──────────────────────────────────────────────────
        // fftAcov returns exactly lagLen+1 values — no separate truncation pass.
        double[] acov = fftAcov(x, n, lagLen);

        // Divide by denominator (inlined; no 2n-1 scratch array needed).
        if (notMaskInt != null && missing == MissingPolicy.CONSERVATIVE && adjusted) {
            divideByConservativeDenominator(acov, lagLen, notMaskInt, n);
        } else if (adjusted) {
            for (int i = 0; i <= lagLen; i++) acov[i] /= (n - i);
        } else {
            double s = (notMaskInt != null) ? VectorOps.sum(notMaskInt) : n;
            VectorOps.scal(1.0 / s, acov, 0, lagLen + 1);
        }

        return acov;
    }

    // ── Private helpers ───────────────────────────────────────────────

    private static double[] directAcov(double[] x, int n, int lagLen,
                                        boolean adjusted, double[] notMaskInt,
                                        MissingPolicy missing) {
        double[] acov = new double[lagLen + 1];
        acov[0] = VectorOps.dot(x, 0, x, 0, n);
        for (int i = 1; i <= lagLen; i++) {
            acov[i] = VectorOps.dot(x, i, x, 0, n - i);
        }
        boolean noMask = (notMaskInt == null || missing == MissingPolicy.DROP);
        if (noMask) {
            if (adjusted) {
                for (int i = 0; i <= lagLen; i++) acov[i] /= (n - i);
            } else {
                VectorOps.scal(1.0 / n, acov, 0, lagLen + 1);
            }
        } else {
            if (adjusted) {
                divideByConservativeDenominator(acov, lagLen, notMaskInt, n);
            } else {
                VectorOps.scal(1.0 / VectorOps.sum(notMaskInt), acov, 0, lagLen + 1);
            }
        }
        return acov;
    }

    private static void divideByConservativeDenominator(double[] acov, int lagLen,
                                                        double[] notMaskInt, int n) {
        double div0 = VectorOps.sum(notMaskInt);
        if (div0 != 0.0) acov[0] /= div0;
        for (int i = 1; i <= lagLen; i++) {
            double div = VectorOps.dot(notMaskInt, i, notMaskInt, 0, n - i);
            if (div != 0.0) acov[i] /= div;
        }
    }

    /**
     * FFT-based autocovariance: computes {@code IFFT(|RFFT(x)|²)} on a zero-padded
     * array of length {@code 2n+1}, then returns the first {@code lagLen+1} values.
     *
     * <p>Memory: allocates {@code 2n+1} + {@code 2(n+1)} doubles for FFT scratch,
     * plus the {@code lagLen+1} output. When {@code lagLen << n} (typical ACF call
     * with a bounded lag budget) this saves an {@code n}-double intermediate copy
     * compared with returning the full {@code n}-length buffer.</p>
     */
    private static double[] fftAcov(double[] x, int n, int lagLen) {
        int padLen = 2 * n + 1;
        int specLen = padLen / 2 + 1; // = n + 1

        // Zero-pad x into buffer (r2c uses this as scratch)
        double[] buf = Arrays.copyOf(x, padLen);
        double[] spec = new double[2 * specLen];
        Transform.r2c(buf, spec, true, 1.0);

        // Power spectrum in-place: |X[k]|² = re² + im²
        for (int i = 0; i < specLen; i++) {
            double re = spec[2 * i], im = spec[2 * i + 1];
            spec[2 * i]     = re * re + im * im;
            spec[2 * i + 1] = 0.0;
        }

        // Inverse real FFT — reuse buf as output (same length padLen)
        Transform.c2r(spec, buf, false, 1.0 / padLen);

        // Return only the requested lags in a single copy.
        return Arrays.copyOf(buf, lagLen + 1);
    }

    // ── Array utilities ───────────────────────────────────────────────

    static boolean hasNaN(double[] a) {
        for (double v : a) if (Double.isNaN(v)) return true;
        return false;
    }

    static double[] dropNaN(double[] a) {
        int count = 0;
        for (double v : a) if (!Double.isNaN(v)) count++;
        double[] out = new double[count];
        int j = 0;
        for (double v : a) if (!Double.isNaN(v)) out[j++] = v;
        return out;
    }

}
