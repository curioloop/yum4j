package com.curioloop.yum4j.stats.tool;

import com.curioloop.yum4j.fft.Transform;
import com.curioloop.yum4j.math.VectorOps;

import java.util.Arrays;

// Note: uses real FFT (r2c/c2r) for real-valued inputs to halve memory and compute vs complex FFT.

/**
 * Cross-correlation and convolution utilities.
 *
 * <p>Mirrors the behaviour of {@code numpy.correlate} and
 * {@code scipy.signal.fftconvolve}, providing three output modes:</p>
 * <ul>
 *   <li>{@link Mode#FULL} — full discrete linear cross-correlation, length {@code m+n-1}</li>
 *   <li>{@link Mode#VALID} — only the central part where both sequences fully overlap</li>
 *   <li>{@link Mode#SAME} — same length as the first input, centred</li>
 * </ul>
 *
 * <p>The direct path is O(m·n); the FFT path is O((m+n) log(m+n)) and is
 * preferred for long sequences.</p>
 */
public final class Correlate {

    private Correlate() {}

    /** Output-length mode, matching {@code numpy.correlate} conventions. */
    public enum Mode {
        /** Output length {@code max(m,n) - min(m,n) + 1}. */
        VALID,
        /** Output length {@code m + n - 1}. */
        FULL,
        /** Output length {@code max(m,n)}. */
        SAME
    }

    /**
     * Computes the cross-correlation of {@code a} and {@code b}.
     *
     * <p>Correlation is defined as convolution with {@code b} reversed:
     * {@code corr[k] = Σ a[j] * b[j - k]}.</p>
     *
     * @param a    first input array (not modified)
     * @param b    second input array (not modified)
     * @param mode output length mode
     * @param fft  if {@code true}, use FFT-based convolution (faster for long arrays)
     * @return cross-correlation array
     */
    public static double[] correlate(double[] a, double[] b, Mode mode, boolean fft) {
        if (a.length == 0 || b.length == 0) {
            throw new IllegalArgumentException("input arrays must not be empty");
        }
        if (fft) {
            // correlate(a, b) = convolve(a, reverse(b)); fuse the reverse into
            // the zero-padded kernel buffer so we don't allocate bRev separately.
            return fftConvolveReversed(a, b, mode);
        } else {
            return directCorrelate(a, b, mode);
        }
    }

    // ── Direct O(m·n) implementation ──────────────────────────────────

    private static double[] directCorrelate(double[] a, double[] b, Mode mode) {
        int lenA = a.length, lenB = b.length;
        boolean swap = (mode == Mode.VALID && lenA < lenB);
        if (swap) {
            double[] tmp = a; a = b; b = tmp;
            int t = lenA; lenA = lenB; lenB = t;
        }
        double[] out;
        switch (mode) {
            case VALID -> {
                out = new double[lenA - lenB + 1];
                for (int lag = 0; lag <= lenA - lenB; lag++) {
                    out[lag] = VectorOps.dot(a, lag, b, 0, lenB);
                }
                if (swap) reverse(out);
            }
            case FULL -> {
                out = new double[lenA + lenB - 1];
                for (int lag = -(lenB - 1); lag < lenA; lag++) {
                    int iStart = Math.max(0, -lag);
                    int iEnd = Math.min(lenB, lenA - lag);
                    out[lag + lenB - 1] = VectorOps.dot(a, lag + iStart, b, iStart, iEnd - iStart);
                }
            }
            case SAME -> {
                out = new double[lenA];
                int mid = lenB / 2;
                for (int i = 0; i < lenA; i++) {
                    int jStart = Math.max(0, mid - i);
                    int jEnd = Math.min(lenB, lenA + mid - i);
                    out[i] = VectorOps.dot(a, i + jStart - mid, b, jStart, jEnd - jStart);
                }
            }
            default -> throw new IllegalArgumentException("unsupported mode: " + mode);
        }
        return out;
    }

    // ── FFT-based O((m+n) log(m+n)) implementation ────────────────────

    /**
     * FFT-based {@code correlate(a, b)} that fuses the kernel reversal into the
     * zero-padded work buffer. Saves one O(|b|) allocation + one O(|b|) reverse
     * vs {@code fftConvolve(a, reverse(b))}.
     */
    private static double[] fftConvolveReversed(double[] a, double[] b, Mode mode) {
        int lenA = a.length, lenB = b.length;
        boolean swap = (mode == Mode.VALID && lenA < lenB);
        if (swap) {
            // correlate(a,b) with swap: we correlate b with a instead. The kernel
            // (originally a) is what gets reversed, so rebind before reversal.
            double[] tmp = a; a = b; b = tmp;
            int t = lenA; lenA = lenB; lenB = t;
        }
        double[] conv = convolveWithReversedKernel(a, b);
        double[] out;
        switch (mode) {
            case FULL  -> out = conv;
            case VALID -> out = Arrays.copyOfRange(conv, lenB - 1, lenA);
            case SAME  -> {
                int start = (lenB - 1) / 2;
                out = Arrays.copyOfRange(conv, start, start + lenA);
            }
            default -> throw new IllegalArgumentException("unsupported mode: " + mode);
        }
        return out;
    }

    /**
     * Linear convolution via real FFT, reversing {@code kernel} on the fly while
     * zero-padding. Equivalent to {@code convolve(signal, reverse(kernel))} but
     * avoids the separate {@code reverse} buffer and pass.
     */
    static double[] convolveWithReversedKernel(double[] signal, double[] kernel) {
        int n = signal.length + kernel.length - 1;
        int specLen = n / 2 + 1;

        double[] sigPad = Arrays.copyOf(signal, n);
        double[] spec1  = new double[2 * specLen];
        Transform.r2c(sigPad, spec1, true, 1.0);

        // Zero-pad + reverse the kernel into kerPad in one pass.
        double[] kerPad = new double[n];
        int lenK = kernel.length;
        for (int i = 0; i < lenK; i++) kerPad[i] = kernel[lenK - 1 - i];
        double[] spec2 = new double[2 * specLen];
        Transform.r2c(kerPad, spec2, true, 1.0);

        // Multiply spectra: spec1 *= spec2
        for (int i = 0; i < specLen; i++) {
            double re1 = spec1[2*i], im1 = spec1[2*i+1];
            double re2 = spec2[2*i], im2 = spec2[2*i+1];
            spec1[2*i]   = re1 * re2 - im1 * im2;
            spec1[2*i+1] = re1 * im2 + im1 * re2;
        }

        Transform.c2r(spec1, sigPad, false, 1.0 / n);
        return sigPad;
    }

    private static void reverse(double[] a) {
        for (int i = 0, j = a.length - 1; i < j; i++, j--) {
            double t = a[i]; a[i] = a[j]; a[j] = t;
        }
    }
}
