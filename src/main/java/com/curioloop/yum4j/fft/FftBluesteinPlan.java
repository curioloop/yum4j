package com.curioloop.yum4j.fft;

import java.util.Arrays;
import java.util.Objects;

/**
 * Source-level Java port of PocketFFT {@code fftblue<T0>}.
 *
 * <p>Upstream mapping: {@code reference/pocketfft-upstream/pocketfft_hdronly.h},
 * class {@code fftblue<T0>}: {@code n}, {@code n2}, child {@code cfftp}, single
 * {@code mem} split into {@code bk} and {@code bkf}, chirp recurrence, zero-padded
 * convolution setup, forward/backward {@code special_mul}, final scale, and {@code exec_r}.</p>
 */
final class FftBluesteinPlan {
    private final int n;
    private final int n2;
    private final FftComplexPlan childPlan;
    private final double[] mem;
    private final int bkOffset;
    private final int bkfOffset;
    private final FftWorkspace.Requirement workspaceRequirement;

    FftBluesteinPlan(int length) {
        if (length < 1) {
            throw new IllegalArgumentException("zero-length FFT requested");
        }
        this.n = length;
        this.n2 = Math.toIntExact(FftUtil.goodSizeComplex(Math.addExact(Math.multiplyExact(2L, length), -1L)));
        this.childPlan = new FftComplexPlan(n2);
        this.mem = new double[2 * (n + n2 / 2 + 1)];
        this.bkOffset = 0;
        this.bkfOffset = 2 * n;
        initializeChirp();
        this.workspaceRequirement = FftWorkspace.Requirement.doubles(2 * n2).plus(childPlan.workspaceRequirement());
    }

    void exec(double[] data, double factor, boolean forward) {
        FftWorkspace workspace = workspaceRequirement().allocate();
        workspace.reset();
        exec(data, factor, forward, workspace);
    }

    void exec(double[] data, double factor, boolean forward, FftWorkspace workspace) {
        exec(data, 0, factor, forward, workspace);
    }

    void exec(double[] data, int dataOffset, double factor, boolean forward, FftWorkspace workspace) {
        Objects.requireNonNull(workspace, "workspace");
        if (dataOffset < 0 || data.length - dataOffset < 2 * n) {
            throw new IllegalArgumentException("data length is smaller than transform length");
        }
        if (forward) {
            execForwardWithWorkspace(data, dataOffset, factor, workspace);
        } else {
            execBackwardWithWorkspace(data, dataOffset, factor, workspace);
        }
    }

    private void execForwardWithWorkspace(double[] data, int dataOffset, double factor, FftWorkspace workspace) {
        double[] akf = workspace.doubles();
        int akfOffset = workspace.alloc(2 * n2);
        int childMark = workspace.mark();

        // Chirp modulation: data × conj(bk) → akf  (inline specialMulForward)
        for (int m = 0; m < n; m++) {
            int di = dataOffset + 2 * m;
            int bi = bkOffset + 2 * m;
            int ai = akfOffset + 2 * m;
            double dr = data[di], dim = data[di + 1];
            double br = mem[bi], bim = mem[bi + 1];
            akf[ai]     = dr * br + dim * bim;
            akf[ai + 1] = dim * br - dr * bim;
        }
        Arrays.fill(akf, akfOffset + 2 * n, akfOffset + 2 * n2, 0.0);

        childPlan.exec(akf, akfOffset, 1.0, true, workspace);

        multiplyFrequencyForward(akf, akfOffset);

        workspace.rewind(childMark);
        childPlan.exec(akf, akfOffset, 1.0, false, workspace);
        for (int m = 0; m < n; m++) {
            int ai = akfOffset + 2 * m;
            int bi = bkOffset + 2 * m;
            int di = dataOffset + 2 * m;
            double ar = akf[ai], aimg = akf[ai + 1];
            double br = mem[bi], bimg = mem[bi + 1];
            data[di] = (ar * br + aimg * bimg) * factor;
            data[di + 1] = (aimg * br - ar * bimg) * factor;
        }
    }

    private void execBackwardWithWorkspace(double[] data, int dataOffset, double factor, FftWorkspace workspace) {
        double[] akf = workspace.doubles();
        int akfOffset = workspace.alloc(2 * n2);
        int childMark = workspace.mark();

        // Chirp modulation: data × bk → akf  (inline specialMulBackward)
        for (int m = 0; m < n; m++) {
            int di = dataOffset + 2 * m;
            int bi = bkOffset + 2 * m;
            int ai = akfOffset + 2 * m;
            double dr = data[di], dim = data[di + 1];
            double br = mem[bi], bim = mem[bi + 1];
            akf[ai]     = dr * br - dim * bim;
            akf[ai + 1] = dr * bim + dim * br;
        }
        Arrays.fill(akf, akfOffset + 2 * n, akfOffset + 2 * n2, 0.0);

        childPlan.exec(akf, akfOffset, 1.0, true, workspace);

        multiplyFrequencyBackward(akf, akfOffset);

        workspace.rewind(childMark);
        childPlan.exec(akf, akfOffset, 1.0, false, workspace);
        for (int m = 0; m < n; m++) {
            int ai = akfOffset + 2 * m;
            int bi = bkOffset + 2 * m;
            int di = dataOffset + 2 * m;
            double ar = akf[ai], aimg = akf[ai + 1];
            double br = mem[bi], bimg = mem[bi + 1];
            data[di] = (ar * br - aimg * bimg) * factor;
            data[di + 1] = (ar * bimg + aimg * br) * factor;
        }
    }

    void execReal(double[] data, double factor, boolean forward) {
        FftWorkspace workspace = workspaceRequirement().allocate();
        workspace.reset();
        execReal(data, factor, forward, workspace);
    }

    void execReal(double[] data, double factor, boolean forward, FftWorkspace workspace) {
        execReal(data, 0, factor, forward, workspace);
    }

    void execReal(double[] data, int dataOffset, double factor, boolean forward, FftWorkspace workspace) {
        Objects.requireNonNull(workspace, "workspace");
        if (dataOffset < 0 || data.length - dataOffset < n) {
            throw new IllegalArgumentException("data length is smaller than transform length");
        }
        if (forward) {
            execRealForwardHalfComplexWithWorkspace(data, dataOffset, factor, workspace);
            return;
        }
        execRealBackwardHalfComplexWithWorkspace(data, dataOffset, factor, workspace);
    }

    void execRealToHermitian(double[] data, int dataOffset, double factor, boolean forward, FftWorkspace workspace) {
        Objects.requireNonNull(workspace, "workspace");
        int spectrumLength = 2 * (n / 2 + 1);
        if (dataOffset < 0 || data.length - dataOffset < spectrumLength) {
            throw new IllegalArgumentException("data length is smaller than Hermitian spectrum length");
        }
        execRealForwardHermitianWithWorkspace(data, dataOffset, factor, forward, workspace);
    }

    private void execRealForwardHalfComplexWithWorkspace(double[] data, int dataOffset, double factor,
                                                         FftWorkspace workspace) {
        double[] akf = workspace.doubles();
        int akfOffset = workspace.alloc(2 * n2);
        int childMark = workspace.mark();

        // Real input can be chirp-modulated directly into the zero-padded convolution buffer.
        // This avoids the old 2*n complex expansion temp; only the n..n2 padding is cleared.
        fillForwardRealChirp(data, dataOffset, akf, akfOffset);
        convolveForward(akf, akfOffset, childMark, workspace);

        data[dataOffset] = finalForwardReal(akf, akfOffset, 0, factor);
        for (int m = 1; m <= n / 2; m++) {
            int ai = akfOffset + 2 * m;
            int bi = bkOffset + 2 * m;
            double ar = akf[ai], aimg = akf[ai + 1];
            double br = mem[bi], bimg = mem[bi + 1];
            double real = (ar * br + aimg * bimg) * factor;
            if (2 * m == n) {
                data[dataOffset + n - 1] = real;
            } else {
                data[dataOffset + 2 * m - 1] = real;
                data[dataOffset + 2 * m] = (aimg * br - ar * bimg) * factor;
            }
        }
    }

    private void execRealForwardHermitianWithWorkspace(double[] data, int dataOffset, double factor, boolean forward,
                                                       FftWorkspace workspace) {
        double[] akf = workspace.doubles();
        int akfOffset = workspace.alloc(2 * n2);
        int childMark = workspace.mark();

        fillForwardRealChirp(data, dataOffset, akf, akfOffset);
        convolveForward(akf, akfOffset, childMark, workspace);

        data[dataOffset] = finalForwardReal(akf, akfOffset, 0, factor);
        data[dataOffset + 1] = 0.0;
        double imaginaryFactor = forward ? factor : -factor;
        for (int m = 1; m <= n / 2; m++) {
            int ai = akfOffset + 2 * m;
            int bi = bkOffset + 2 * m;
            double ar = akf[ai], aimg = akf[ai + 1];
            double br = mem[bi], bimg = mem[bi + 1];
            int di = dataOffset + 2 * m;
            data[di] = (ar * br + aimg * bimg) * factor;
            data[di + 1] = 2 * m == n ? 0.0 : (aimg * br - ar * bimg) * imaginaryFactor;
        }
    }

    private void execRealBackwardHalfComplexWithWorkspace(double[] data, int dataOffset, double factor,
                                                          FftWorkspace workspace) {
        double[] akf = workspace.doubles();
        int akfOffset = workspace.alloc(2 * n2);
        int childMark = workspace.mark();

        // Reconstruct Hermitian bins from FFTPACK half-complex layout while applying the chirp.
        // The mirrored upper half is never materialized as a separate complex temp.
        fillBackwardHalfComplexChirp(data, dataOffset, akf, akfOffset);
        convolveBackward(akf, akfOffset, childMark, workspace);

        for (int m = 0; m < n; m++) {
            int ai = akfOffset + 2 * m;
            int bi = bkOffset + 2 * m;
            double ar = akf[ai], aimg = akf[ai + 1];
            double br = mem[bi], bimg = mem[bi + 1];
            data[dataOffset + m] = (ar * br - aimg * bimg) * factor;
        }
    }

    private void fillForwardRealChirp(double[] data, int dataOffset, double[] akf, int akfOffset) {
        for (int m = 0; m < n; m++) {
            int bi = bkOffset + 2 * m;
            int ai = akfOffset + 2 * m;
            double value = data[dataOffset + m];
            akf[ai] = value * mem[bi];
            akf[ai + 1] = -value * mem[bi + 1];
        }
        Arrays.fill(akf, akfOffset + 2 * n, akfOffset + 2 * n2, 0.0);
    }

    private void fillBackwardHalfComplexChirp(double[] data, int dataOffset, double[] akf, int akfOffset) {
        for (int m = 0; m < n; m++) {
            double real;
            double imaginary;
            if (m == 0) {
                real = data[dataOffset];
                imaginary = 0.0;
            } else if (2 * m < n) {
                real = data[dataOffset + 2 * m - 1];
                imaginary = data[dataOffset + 2 * m];
            } else if (2 * m == n) {
                real = data[dataOffset + n - 1];
                imaginary = 0.0;
            } else {
                int mirror = n - m;
                real = data[dataOffset + 2 * mirror - 1];
                imaginary = -data[dataOffset + 2 * mirror];
            }
            specialMulBackwardTo(real, imaginary, mem, bkOffset + 2 * m, akf, akfOffset + 2 * m);
        }
        Arrays.fill(akf, akfOffset + 2 * n, akfOffset + 2 * n2, 0.0);
    }

    private void convolveForward(double[] akf, int akfOffset, int childMark, FftWorkspace workspace) {
        childPlan.exec(akf, akfOffset, 1.0, true, workspace);
        multiplyFrequencyForward(akf, akfOffset);
        workspace.rewind(childMark);
        childPlan.exec(akf, akfOffset, 1.0, false, workspace);
    }

    private void convolveBackward(double[] akf, int akfOffset, int childMark, FftWorkspace workspace) {
        childPlan.exec(akf, akfOffset, 1.0, true, workspace);
        multiplyFrequencyBackward(akf, akfOffset);
        workspace.rewind(childMark);
        childPlan.exec(akf, akfOffset, 1.0, false, workspace);
    }

    private double finalForwardReal(double[] akf, int akfOffset, int m, double factor) {
        int ai = akfOffset + 2 * m;
        int bi = bkOffset + 2 * m;
        return (akf[ai] * mem[bi] + akf[ai + 1] * mem[bi + 1]) * factor;
    }

    private void multiplyFrequencyForward(double[] akf, int akfOffset) {
        {
            double br = mem[bkfOffset], bim = mem[bkfOffset + 1];
            double ar = akf[akfOffset], aim = akf[akfOffset + 1];
            akf[akfOffset]     = ar * br - aim * bim;
            akf[akfOffset + 1] = ar * bim + aim * br;
        }
        int half = (n2 + 1) / 2;
        for (int m = 1; m < half; m++) {
            int bi = bkfOffset + 2 * m;
            double br = mem[bi], bim = mem[bi + 1];
            int ai = akfOffset + 2 * m;
            double ar = akf[ai], aim = akf[ai + 1];
            akf[ai]     = ar * br - aim * bim;
            akf[ai + 1] = ar * bim + aim * br;
            int mi = akfOffset + 2 * (n2 - m);
            ar = akf[mi]; aim = akf[mi + 1];
            akf[mi]     = ar * br - aim * bim;
            akf[mi + 1] = ar * bim + aim * br;
        }
        if ((n2 & 1) == 0) {
            int mid = akfOffset + n2;
            int bi = bkfOffset + n2;
            double br = mem[bi], bim = mem[bi + 1];
            double ar = akf[mid], aim = akf[mid + 1];
            akf[mid]     = ar * br - aim * bim;
            akf[mid + 1] = ar * bim + aim * br;
        }
    }

    private void multiplyFrequencyBackward(double[] akf, int akfOffset) {
        {
            double br = mem[bkfOffset], bim = mem[bkfOffset + 1];
            double ar = akf[akfOffset], aim = akf[akfOffset + 1];
            akf[akfOffset]     = ar * br + aim * bim;
            akf[akfOffset + 1] = aim * br - ar * bim;
        }
        int half = (n2 + 1) / 2;
        for (int m = 1; m < half; m++) {
            int bi = bkfOffset + 2 * m;
            double br = mem[bi], bim = mem[bi + 1];
            int ai = akfOffset + 2 * m;
            double ar = akf[ai], aim = akf[ai + 1];
            akf[ai]     = ar * br + aim * bim;
            akf[ai + 1] = aim * br - ar * bim;
            int mi = akfOffset + 2 * (n2 - m);
            ar = akf[mi]; aim = akf[mi + 1];
            akf[mi]     = ar * br + aim * bim;
            akf[mi + 1] = aim * br - ar * bim;
        }
        if ((n2 & 1) == 0) {
            int mid = akfOffset + n2;
            int bi = bkfOffset + n2;
            double br = mem[bi], bim = mem[bi + 1];
            double ar = akf[mid], aim = akf[mid + 1];
            akf[mid]     = ar * br + aim * bim;
            akf[mid + 1] = aim * br - ar * bim;
        }
    }

    FftWorkspace.Requirement workspaceRequirement() {
        return workspaceRequirement;
    }

    int length() {
        return n;
    }

    int convolutionLength() {
        return n2;
    }

    private void initializeChirp() {
        FftSincos sincos = new FftSincos(2L * n);
        mem[bkOffset] = 1.0;
        mem[bkOffset + 1] = 0.0;
        long coefficient = 0;
        for (int m = 1; m < n; m++) {
            coefficient += 2L * m - 1L;
            if (coefficient >= 2L * n) {
                coefficient -= 2L * n;
            }
            sincos.fill(coefficient, mem, bkOffset + 2 * m);
        }

        double[] tbkf = new double[2 * n2];
        double scale = 1.0 / n2;
        tbkf[0] = mem[bkOffset] * scale;
        tbkf[1] = mem[bkOffset + 1] * scale;
        for (int m = 1; m < n; m++) {
            tbkf[2 * m] = mem[bkOffset + 2 * m] * scale;
            tbkf[2 * m + 1] = mem[bkOffset + 2 * m + 1] * scale;
            int mirror = n2 - m;
            tbkf[2 * mirror] = tbkf[2 * m];
            tbkf[2 * mirror + 1] = tbkf[2 * m + 1];
        }
        childPlan.exec(tbkf, 1.0, true);
        System.arraycopy(tbkf, 0, mem, bkfOffset, 2 * (n2 / 2 + 1));
    }

    private static void specialMulBackwardTo(double ar, double ai, double[] b, int bOffset, double[] out,
                                             int outOffset) {
        double br = b[bOffset];
        double bi = b[bOffset + 1];
        out[outOffset] = ar * br - ai * bi;
        out[outOffset + 1] = ar * bi + ai * br;
    }
}