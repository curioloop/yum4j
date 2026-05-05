package com.curioloop.yum4j.fft;

import java.util.Arrays;
import java.util.Objects;

/**
 * Fast Fourier Transform operations.
 *
 * <p>All 1-D convenience methods operate <b>in-place</b>: the input array is
 * modified and contains the result on return.  For {@link #r2c} and
 * {@link #c2r} the input is also modified (used as scratch space), and the
 * result is written to the separate output array.
 *
 * <p>For repeated transforms of the same length, use the {@code Prepared*}
 * classes ({@link #prepareComplex}, {@link #prepareReal}, etc.) to amortise
 * plan construction and workspace allocation.
 *
 * <p>Multi-dimensional overloads accept shape, stride, and offset parameters
 * following NumPy-style strided array conventions.
 */
public final class Transform {
    private Transform() {
    }

    // ── Factory methods for prepared (reusable) plans ──────────────────

    /**
     * Creates a reusable plan for 1-D complex-to-complex transforms of the
     * given length.
     *
     * @param length number of complex samples (N, not 2N)
     * @return a thread-safe prepared plan
     * @throws IllegalArgumentException if {@code length < 1}
     */
    public static PreparedComplex prepareComplex(int length) {
        return new PreparedComplex(length);
    }

    /**
     * Creates a reusable plan for multi-dimensional complex-to-complex
     * transforms.
     *
     * @param shape dimensions of the array
     * @param axes  axes along which to transform
     * @return a thread-safe prepared plan
     * @throws NullPointerException if {@code shape} or {@code axes} is null
     */
    public static PreparedNdComplex prepareComplex(int[] shape, int[] axes) {
        return new PreparedNdComplex(shape, axes);
    }

    /**
     * Creates a reusable plan for 1-D real transforms (r2c, c2r, and
     * real-to-real families).
     *
     * @param length number of real samples
     * @return a thread-safe prepared plan
     * @throws IllegalArgumentException if {@code length < 1}
     */
    public static PreparedReal prepareReal(int length) {
        return new PreparedReal(length);
    }

    /**
     * Creates a reusable plan for multi-dimensional real transforms.
     *
     * @param shape dimensions of the real array
     * @param axes  axes along which to transform
     * @return a thread-safe prepared plan
     * @throws NullPointerException if {@code shape} or {@code axes} is null
     */
    public static PreparedNdReal prepareReal(int[] shape, int[] axes) {
        return new PreparedNdReal(shape, axes);
    }

    /**
     * Creates a reusable plan for a 1-D Discrete Cosine Transform of the
     * specified type.
     *
     * @param length number of real samples
     * @param type   DCT type (1–4)
     * @return a thread-safe prepared plan
     * @throws IllegalArgumentException if {@code type} is not in [1, 4]
     */
    public static PreparedDcst prepareDct(int length, int type) {
        return new PreparedDcst(length, type, true);
    }

    /**
     * Creates a reusable plan for a multi-dimensional Discrete Cosine
     * Transform of the specified type.
     *
     * @param shape dimensions of the real array
     * @param axes  axes along which to transform
     * @param type  DCT type (1–4)
     * @return a thread-safe prepared plan
     * @throws IllegalArgumentException if {@code type} is not in [1, 4]
     */
    public static PreparedNdDcst prepareDct(int[] shape, int[] axes, int type) {
        return new PreparedNdDcst(shape, axes, type, true);
    }

    /**
     * Creates a reusable plan for a 1-D Discrete Sine Transform of the
     * specified type.
     *
     * @param length number of real samples
     * @param type   DST type (1–4)
     * @return a thread-safe prepared plan
     * @throws IllegalArgumentException if {@code type} is not in [1, 4]
     */
    public static PreparedDcst prepareDst(int length, int type) {
        return new PreparedDcst(length, type, false);
    }

    /**
     * Creates a reusable plan for a multi-dimensional Discrete Sine Transform
     * of the specified type.
     *
     * @param shape dimensions of the real array
     * @param axes  axes along which to transform
     * @param type  DST type (1–4)
     * @return a thread-safe prepared plan
     * @throws IllegalArgumentException if {@code type} is not in [1, 4]
     */
    public static PreparedNdDcst prepareDst(int[] shape, int[] axes, int type) {
        return new PreparedNdDcst(shape, axes, type, false);
    }

    // ── Complex-to-complex ─────────────────────────────────────────────

    /**
     * In-place complex-to-complex FFT.
     *
     * <p>The interleaved complex array {@code data} (length&nbsp;=&nbsp;2&nbsp;×&nbsp;N) is
     * transformed in-place.
     *
     * @param data    interleaved [re₀, im₀, re₁, im₁, …] complex array (modified in-place)
     * @param forward {@code true} for forward (analysis) transform,
     *                {@code false} for backward (synthesis) transform
     * @param factor  scaling factor applied to every output element
     * @throws IllegalArgumentException if {@code data} has odd length
     */
    public static void c2c(double[] data, boolean forward, double factor) {
        Objects.requireNonNull(data, "data");
        int length = complexLength(data, "data");
        if (length == 0) return;
        FftPlanCache.complexTransform(length).exec(data, factor, forward);
    }

    /**
     * Multi-dimensional complex-to-complex FFT with zero offsets.
     *
     * @param shape    dimensions of the complex array
     * @param strideIn  input strides (in complex-element units, i.e. 2 doubles per element)
     * @param strideOut output strides
     * @param axes     axes along which to transform
     * @param forward  {@code true} for forward transform
     * @param input    interleaved complex input array
     * @param output   interleaved complex output array
     * @param factor   scaling factor applied to every output element
     */
    public static void c2c(int[] shape, long[] strideIn, long[] strideOut, int[] axes, boolean forward,
                           double[] input, double[] output, double factor) {
        c2c(shape, strideIn, 0L, strideOut, 0L, axes, forward, input, output, factor);
    }

    /**
     * Multi-dimensional complex-to-complex FFT with explicit offsets.
     *
     * @param shape     dimensions of the complex array
     * @param strideIn  input strides
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides
     * @param offsetOut element offset into the output array
     * @param axes      axes along which to transform
     * @param forward   {@code true} for forward transform
     * @param input     interleaved complex input array
     * @param output    interleaved complex output array
     * @param factor    scaling factor applied to every output element
     */
    public static void c2c(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                           boolean forward, double[] input, double[] output, double factor) {
        FftNd.c2c(shape, strideIn, offsetIn, strideOut, offsetOut, axes, forward, input, output, factor);
    }

    // ── Real-to-complex ────────────────────────────────────────────────

    /**
     * 1-D real-to-complex FFT.
     *
     * <p>Computes the Hermitian-symmetric spectrum of a real signal.
     * The {@code input} array is used as scratch space and its contents are
     * undefined on return.  The result is written to {@code output}.
     *
     * @param input   real input array (contents destroyed)
     * @param output  interleaved complex output of length ≥ 2×(N/2+1)
     * @param forward {@code true} for the exp(−2πi) sign convention,
     *                {@code false} for exp(+2πi)
     * @param factor  scaling factor applied to every output element
     * @throws IllegalArgumentException if {@code output} is too short for the
     *                                  Hermitian spectrum
     */
    public static void r2c(double[] input, double[] output, boolean forward, double factor) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        int length = input.length;
        if (length == 0) {
            return;
        }
        int requiredOutput = complexSpectrumLength(length);
        if (output.length < requiredOutput) {
            throw new IllegalArgumentException("output length is smaller than Hermitian spectrum length");
        }

        FftPlanCache.realTransform(length).exec(input, factor, true);
        halfComplexToComplex(input, output, forward);
    }

    /**
     * Multi-dimensional real-to-complex FFT along a single axis with zero
     * offsets.
     *
     * @param shapeIn  dimensions of the real input array
     * @param strideIn input strides
     * @param strideOut output strides (complex)
     * @param axis     axis along which to transform
     * @param forward  {@code true} for forward transform
     * @param input    real input array
     * @param output   interleaved complex output array
     * @param factor   scaling factor
     */
    public static void r2c(int[] shapeIn, long[] strideIn, long[] strideOut, int axis, boolean forward,
                           double[] input, double[] output, double factor) {
        r2c(shapeIn, strideIn, 0L, strideOut, 0L, new int[] {axis}, forward, input, output, factor);
    }

    /**
     * Multi-dimensional real-to-complex FFT along a single axis with explicit
     * offsets.
     *
     * @param shapeIn   dimensions of the real input array
     * @param strideIn  input strides
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides (complex)
     * @param offsetOut element offset into the output array
     * @param axis      axis along which to transform
     * @param forward   {@code true} for forward transform
     * @param input     real input array
     * @param output    interleaved complex output array
     * @param factor    scaling factor
     */
    public static void r2c(int[] shapeIn, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int axis,
                           boolean forward, double[] input, double[] output, double factor) {
        r2c(shapeIn, strideIn, offsetIn, strideOut, offsetOut, new int[] {axis}, forward, input, output, factor);
    }

    /**
     * Multi-dimensional real-to-complex FFT along multiple axes with zero
     * offsets.
     *
     * @param shapeIn  dimensions of the real input array
     * @param strideIn input strides
     * @param strideOut output strides (complex)
     * @param axes     axes along which to transform
     * @param forward  {@code true} for forward transform
     * @param input    real input array
     * @param output   interleaved complex output array
     * @param factor   scaling factor
     */
    public static void r2c(int[] shapeIn, long[] strideIn, long[] strideOut, int[] axes, boolean forward,
                           double[] input, double[] output, double factor) {
        r2c(shapeIn, strideIn, 0L, strideOut, 0L, axes, forward, input, output, factor);
    }

    /**
     * Multi-dimensional real-to-complex FFT along multiple axes with explicit
     * offsets.
     *
     * @param shapeIn   dimensions of the real input array
     * @param strideIn  input strides
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides (complex)
     * @param offsetOut element offset into the output array
     * @param axes      axes along which to transform
     * @param forward   {@code true} for forward transform
     * @param input     real input array
     * @param output    interleaved complex output array
     * @param factor    scaling factor
     */
    public static void r2c(int[] shapeIn, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                           boolean forward, double[] input, double[] output, double factor) {
        FftNd.r2c(shapeIn, strideIn, offsetIn, strideOut, offsetOut, axes, forward, input, output, factor);
    }

    // ── Complex-to-real ────────────────────────────────────────────────

    /**
     * 1-D complex-to-real inverse FFT.
     *
     * <p>Reconstructs a real signal from its Hermitian-symmetric spectrum.
     * The {@code input} array is used as scratch space and its contents are
     * undefined on return.  The result is written to {@code output}.
     *
     * @param input   interleaved complex spectrum (contents destroyed)
     * @param output  real output array of length N
     * @param forward {@code true} for the exp(−2πi) sign convention,
     *                {@code false} for exp(+2πi)
     * @param factor  scaling factor applied to every output element
     * @throws IllegalArgumentException if {@code input} is too short for the
     *                                  Hermitian spectrum of length N
     */
    public static void c2r(double[] input, double[] output, boolean forward, double factor) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        int length = output.length;
        if (length == 0) {
            return;
        }
        int requiredInput = complexSpectrumLength(length);
        if (input.length < requiredInput) {
            throw new IllegalArgumentException("input length is smaller than Hermitian spectrum length");
        }

        complexToHalfComplex(input, output, forward);
        FftPlanCache.realTransform(length).exec(output, factor, false);
    }

    /**
     * Multi-dimensional complex-to-real FFT along a single axis with zero
     * offsets.
     *
     * @param shapeOut dimensions of the real output array
     * @param strideIn input strides (complex)
     * @param strideOut output strides
     * @param axis     axis along which to transform
     * @param forward  {@code true} for forward sign convention
     * @param input    interleaved complex input array
     * @param output   real output array
     * @param factor   scaling factor
     */
    public static void c2r(int[] shapeOut, long[] strideIn, long[] strideOut, int axis, boolean forward,
                           double[] input, double[] output, double factor) {
        c2r(shapeOut, strideIn, 0L, strideOut, 0L, new int[] {axis}, forward, input, output, factor);
    }

    /**
     * Multi-dimensional complex-to-real FFT along a single axis with explicit
     * offsets.
     *
     * @param shapeOut  dimensions of the real output array
     * @param strideIn  input strides (complex)
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides
     * @param offsetOut element offset into the output array
     * @param axis      axis along which to transform
     * @param forward   {@code true} for forward sign convention
     * @param input     interleaved complex input array
     * @param output    real output array
     * @param factor    scaling factor
     */
    public static void c2r(int[] shapeOut, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int axis,
                           boolean forward, double[] input, double[] output, double factor) {
        c2r(shapeOut, strideIn, offsetIn, strideOut, offsetOut, new int[] {axis}, forward, input, output, factor);
    }

    /**
     * Multi-dimensional complex-to-real FFT along multiple axes with zero
     * offsets.
     *
     * @param shapeOut dimensions of the real output array
     * @param strideIn input strides (complex)
     * @param strideOut output strides
     * @param axes     axes along which to transform
     * @param forward  {@code true} for forward sign convention
     * @param input    interleaved complex input array
     * @param output   real output array
     * @param factor   scaling factor
     */
    public static void c2r(int[] shapeOut, long[] strideIn, long[] strideOut, int[] axes, boolean forward,
                           double[] input, double[] output, double factor) {
        c2r(shapeOut, strideIn, 0L, strideOut, 0L, axes, forward, input, output, factor);
    }

    /**
     * Multi-dimensional complex-to-real FFT along multiple axes with explicit
     * offsets.
     *
     * @param shapeOut  dimensions of the real output array
     * @param strideIn  input strides (complex)
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides
     * @param offsetOut element offset into the output array
     * @param axes      axes along which to transform
     * @param forward   {@code true} for forward sign convention
     * @param input     interleaved complex input array
     * @param output    real output array
     * @param factor    scaling factor
     */
    public static void c2r(int[] shapeOut, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                           boolean forward, double[] input, double[] output, double factor) {
        FftNd.c2r(shapeOut, strideIn, offsetIn, strideOut, offsetOut, axes, forward, input, output, factor);
    }

    // ── Real-to-real FFTPACK ───────────────────────────────────────────

    /**
     * In-place 1-D real transform using the FFTPACK half-complex layout.
     *
     * <p>When {@code realToHermitian} is {@code true} the real input is
     * transformed to half-complex form; when {@code false} the inverse is
     * performed.  The {@code forward} flag controls the sign convention
     * (exp(−2πi) vs exp(+2πi)).
     *
     * @param data            real array (modified in-place)
     * @param realToHermitian {@code true} for real→half-complex,
     *                        {@code false} for half-complex→real
     * @param forward         {@code true} for the forward sign convention
     * @param factor          scaling factor applied to every output element
     */
    public static void r2rFftpack(double[] data, boolean realToHermitian, boolean forward, double factor) {
        Objects.requireNonNull(data, "data");
        if (data.length == 0) {
            return;
        }
        if (!realToHermitian && forward) {
            negateHalfComplexImaginary(data);
        }
        FftPlanCache.realTransform(data.length).exec(data, factor, realToHermitian);
        if (realToHermitian && !forward) {
            negateHalfComplexImaginary(data);
        }
    }

    /**
     * Multi-dimensional real FFTPACK transform with zero offsets.
     *
     * @param shape           dimensions of the real array
     * @param strideIn        input strides
     * @param strideOut       output strides
     * @param axes            axes along which to transform
     * @param realToHermitian {@code true} for real→half-complex
     * @param forward         {@code true} for the forward sign convention
     * @param input           real input array
     * @param output          real output array
     * @param factor          scaling factor
     */
    public static void r2rFftpack(int[] shape, long[] strideIn, long[] strideOut, int[] axes,
                                   boolean realToHermitian, boolean forward, double[] input, double[] output,
                                   double factor) {
        r2rFftpack(shape, strideIn, 0L, strideOut, 0L, axes, realToHermitian, forward, input, output, factor);
    }

    /**
     * Multi-dimensional real FFTPACK transform with explicit offsets.
     *
     * @param shape           dimensions of the real array
     * @param strideIn        input strides
     * @param offsetIn        element offset into the input array
     * @param strideOut       output strides
     * @param offsetOut       element offset into the output array
     * @param axes            axes along which to transform
     * @param realToHermitian {@code true} for real→half-complex
     * @param forward         {@code true} for the forward sign convention
     * @param input           real input array
     * @param output          real output array
     * @param factor          scaling factor
     */
    public static void r2rFftpack(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                                   int[] axes, boolean realToHermitian, boolean forward, double[] input,
                                   double[] output, double factor) {
        FftNd.r2rFftpack(shape, strideIn, offsetIn, strideOut, offsetOut, axes, realToHermitian, forward, input,
                output, factor);
    }

    // ── DCT / DST ──────────────────────────────────────────────────────

    /**
     * In-place 1-D Discrete Cosine Transform.
     *
     * @param data   real array (modified in-place)
     * @param type   DCT type (1–4)
     * @param factor scaling factor applied to every output element
     * @param ortho  {@code true} to apply orthonormal scaling
     * @throws IllegalArgumentException if {@code type} is not in [1, 4], or
     *                                  if type 1 is requested with fewer than 2 samples
     */
    public static void dct(double[] data, int type, double factor, boolean ortho) {
        dcst1d(data, type, true, factor, ortho);
    }

    /**
     * Multi-dimensional DCT with zero offsets.
     *
     * @param shape    dimensions of the real array
     * @param strideIn input strides
     * @param strideOut output strides
     * @param axes     axes along which to transform
     * @param type     DCT type (1–4)
     * @param input    real input array
     * @param output   real output array
     * @param factor   scaling factor
     * @param ortho    {@code true} to apply orthonormal scaling
     */
    public static void dct(int[] shape, long[] strideIn, long[] strideOut, int[] axes, int type, double[] input,
                           double[] output, double factor, boolean ortho) {
        dct(shape, strideIn, 0L, strideOut, 0L, axes, type, input, output, factor, ortho);
    }

    /**
     * Multi-dimensional DCT with explicit offsets.
     *
     * @param shape     dimensions of the real array
     * @param strideIn  input strides
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides
     * @param offsetOut element offset into the output array
     * @param axes      axes along which to transform
     * @param type      DCT type (1–4)
     * @param input     real input array
     * @param output    real output array
     * @param factor    scaling factor
     * @param ortho     {@code true} to apply orthonormal scaling
     */
    public static void dct(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                           int type, double[] input, double[] output, double factor, boolean ortho) {
        FftNd.dcst(shape, strideIn, offsetIn, strideOut, offsetOut, axes, type, true, ortho, input, output, factor);
    }

    /**
     * In-place 1-D Discrete Sine Transform.
     *
     * @param data   real array (modified in-place)
     * @param type   DST type (1–4)
     * @param factor scaling factor applied to every output element
     * @param ortho  {@code true} to apply orthonormal scaling
     * @throws IllegalArgumentException if {@code type} is not in [1, 4]
     */
    public static void dst(double[] data, int type, double factor, boolean ortho) {
        dcst1d(data, type, false, factor, ortho);
    }

    /**
     * Multi-dimensional DST with zero offsets.
     *
     * @param shape    dimensions of the real array
     * @param strideIn input strides
     * @param strideOut output strides
     * @param axes     axes along which to transform
     * @param type     DST type (1–4)
     * @param input    real input array
     * @param output   real output array
     * @param factor   scaling factor
     * @param ortho    {@code true} to apply orthonormal scaling
     */
    public static void dst(int[] shape, long[] strideIn, long[] strideOut, int[] axes, int type, double[] input,
                           double[] output, double factor, boolean ortho) {
        dst(shape, strideIn, 0L, strideOut, 0L, axes, type, input, output, factor, ortho);
    }

    /**
     * Multi-dimensional DST with explicit offsets.
     *
     * @param shape     dimensions of the real array
     * @param strideIn  input strides
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides
     * @param offsetOut element offset into the output array
     * @param axes      axes along which to transform
     * @param type      DST type (1–4)
     * @param input     real input array
     * @param output    real output array
     * @param factor    scaling factor
     * @param ortho     {@code true} to apply orthonormal scaling
     */
    public static void dst(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, int[] axes,
                           int type, double[] input, double[] output, double factor, boolean ortho) {
        FftNd.dcst(shape, strideIn, offsetIn, strideOut, offsetOut, axes, type, false, ortho, input, output, factor);
    }

    // ── Separable Hartley ──────────────────────────────────────────────

    /**
     * In-place 1-D separable Hartley transform.
     *
     * <p>Computes H(k) = Σ x(n)·cas(2πnk/N) where cas(θ) = cos(θ) + sin(θ).
     *
     * @param data   real array (modified in-place)
     * @param factor scaling factor applied to every output element
     */
    public static void r2rSeparableHartley(double[] data, double factor) {
        realFamily1d(data, factor, false, false);
    }

    /**
     * Multi-dimensional separable Hartley transform with zero offsets.
     *
     * @param shape    dimensions of the real array
     * @param strideIn input strides
     * @param strideOut output strides
     * @param axes     axes along which to transform
     * @param input    real input array
     * @param output   real output array
     * @param factor   scaling factor
     */
    public static void r2rSeparableHartley(int[] shape, long[] strideIn, long[] strideOut, int[] axes,
                                             double[] input, double[] output, double factor) {
        r2rSeparableHartley(shape, strideIn, 0L, strideOut, 0L, axes, input, output, factor);
    }

    /**
     * Multi-dimensional separable Hartley transform with explicit offsets.
     *
     * @param shape     dimensions of the real array
     * @param strideIn  input strides
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides
     * @param offsetOut element offset into the output array
     * @param axes      axes along which to transform
     * @param input     real input array
     * @param output    real output array
     * @param factor    scaling factor
     */
    public static void r2rSeparableHartley(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                             long offsetOut, int[] axes, double[] input, double[] output,
                                             double factor) {
        FftNd.separableHartley(shape, strideIn, offsetIn, strideOut, offsetOut, axes, false, input, output, factor);
    }

    // ── Genuine Hartley ────────────────────────────────────────────────

    /**
     * In-place 1-D genuine (non-separable) Hartley transform.
     *
     * <p>Unlike the separable variant, the genuine Hartley transform uses a
     * single multi-dimensional kernel rather than successive 1-D transforms.
     *
     * @param data   real array (modified in-place)
     * @param factor scaling factor applied to every output element
     */
    public static void r2rGenuineHartley(double[] data, double factor) {
        realFamily1d(data, factor, false, true);
    }

    /**
     * Multi-dimensional genuine Hartley transform with zero offsets.
     *
     * @param shape    dimensions of the real array
     * @param strideIn input strides
     * @param strideOut output strides
     * @param axes     axes along which to transform
     * @param input    real input array
     * @param output   real output array
     * @param factor   scaling factor
     */
    public static void r2rGenuineHartley(int[] shape, long[] strideIn, long[] strideOut, int[] axes,
                                           double[] input, double[] output, double factor) {
        r2rGenuineHartley(shape, strideIn, 0L, strideOut, 0L, axes, input, output, factor);
    }

    /**
     * Multi-dimensional genuine Hartley transform with explicit offsets.
     *
     * @param shape     dimensions of the real array
     * @param strideIn  input strides
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides
     * @param offsetOut element offset into the output array
     * @param axes      axes along which to transform
     * @param input     real input array
     * @param output    real output array
     * @param factor    scaling factor
     */
    public static void r2rGenuineHartley(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                           long offsetOut, int[] axes, double[] input, double[] output,
                                           double factor) {
        FftNd.genuineHartley(shape, strideIn, offsetIn, strideOut, offsetOut, axes, false, input, output, factor);
    }

    // ── Separable FHT ──────────────────────────────────────────────────

    /**
     * In-place 1-D separable Fast Hartley Transform (FHT sign convention).
     *
     * <p>Uses the FHT kernel cas(θ) = cos(θ) − sin(θ) instead of the
     * standard Hartley cas(θ) = cos(θ) + sin(θ).
     *
     * @param data   real array (modified in-place)
     * @param factor scaling factor applied to every output element
     */
    public static void r2rSeparableFht(double[] data, double factor) {
        realFamily1d(data, factor, true, false);
    }

    /**
     * Multi-dimensional separable FHT with zero offsets.
     *
     * @param shape    dimensions of the real array
     * @param strideIn input strides
     * @param strideOut output strides
     * @param axes     axes along which to transform
     * @param input    real input array
     * @param output   real output array
     * @param factor   scaling factor
     */
    public static void r2rSeparableFht(int[] shape, long[] strideIn, long[] strideOut, int[] axes, double[] input,
                                         double[] output, double factor) {
        r2rSeparableFht(shape, strideIn, 0L, strideOut, 0L, axes, input, output, factor);
    }

    /**
     * Multi-dimensional separable FHT with explicit offsets.
     *
     * @param shape     dimensions of the real array
     * @param strideIn  input strides
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides
     * @param offsetOut element offset into the output array
     * @param axes      axes along which to transform
     * @param input     real input array
     * @param output    real output array
     * @param factor    scaling factor
     */
    public static void r2rSeparableFht(int[] shape, long[] strideIn, long offsetIn, long[] strideOut,
                                         long offsetOut, int[] axes, double[] input, double[] output,
                                         double factor) {
        FftNd.separableHartley(shape, strideIn, offsetIn, strideOut, offsetOut, axes, true, input, output, factor);
    }

    // ── Genuine FHT ────────────────────────────────────────────────────

    /**
     * In-place 1-D genuine (non-separable) Fast Hartley Transform (FHT sign
     * convention).
     *
     * @param data   real array (modified in-place)
     * @param factor scaling factor applied to every output element
     */
    public static void r2rGenuineFht(double[] data, double factor) {
        realFamily1d(data, factor, true, true);
    }

    /**
     * Multi-dimensional genuine FHT with zero offsets.
     *
     * @param shape    dimensions of the real array
     * @param strideIn input strides
     * @param strideOut output strides
     * @param axes     axes along which to transform
     * @param input    real input array
     * @param output   real output array
     * @param factor   scaling factor
     */
    public static void r2rGenuineFht(int[] shape, long[] strideIn, long[] strideOut, int[] axes, double[] input,
                                       double[] output, double factor) {
        r2rGenuineFht(shape, strideIn, 0L, strideOut, 0L, axes, input, output, factor);
    }

    /**
     * Multi-dimensional genuine FHT with explicit offsets.
     *
     * @param shape     dimensions of the real array
     * @param strideIn  input strides
     * @param offsetIn  element offset into the input array
     * @param strideOut output strides
     * @param offsetOut element offset into the output array
     * @param axes      axes along which to transform
     * @param input     real input array
     * @param output    real output array
     * @param factor    scaling factor
     */
    public static void r2rGenuineFht(int[] shape, long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                                       int[] axes, double[] input, double[] output, double factor) {
        FftNd.genuineHartley(shape, strideIn, offsetIn, strideOut, offsetOut, axes, true, input, output, factor);
    }

    // ── Prepared inner classes ──────────────────────────────────────────

    /**
     * Reusable plan for 1-D complex-to-complex transforms.
     *
     * <p>Instances are thread-safe provided each thread uses its own
     * {@link FftWorkspace} obtained from {@link #createWorkspace()}.
     */
    public static final class PreparedComplex {
        private final int length;
        private final int dataLength;
        private final FftComplexTransform transform;
        private final FftWorkspace.Requirement workspaceRequirement;

        private PreparedComplex(int length) {
            if (length < 1) {
                throw new IllegalArgumentException("zero-length FFT requested");
            }
            this.length = length;
            this.dataLength = 2 * length;
            this.transform = FftPlanCache.complexTransform(length);
            this.workspaceRequirement = transform.workspaceRequirement();
        }

        /**
         * Returns the transform length (number of complex samples).
         *
         * @return transform length
         */
        public int length() {
            return length;
        }

        /**
         * Returns the workspace requirement for this plan.
         *
         * @return workspace requirement
         */
        public FftWorkspace.Requirement workspaceRequirement() {
            return workspaceRequirement;
        }

        /**
         * Allocates a new workspace satisfying this plan's requirements.
         *
         * @return a fresh workspace
         */
        public FftWorkspace createWorkspace() {
            return workspaceRequirement.allocate();
        }

        /**
         * Executes an in-place complex-to-complex transform using the
         * supplied workspace.
         *
         * @param data      interleaved complex array (modified in-place)
         * @param forward   {@code true} for forward transform
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace (not shared across threads)
         * @throws IllegalArgumentException if {@code data} is shorter than
         *                                  2&nbsp;×&nbsp;length
         */
        public void c2c(double[] data, boolean forward, double factor, FftWorkspace workspace) {
            Objects.requireNonNull(data, "data");
            Objects.requireNonNull(workspace, "workspace");
            if (data.length < dataLength) {
                throw new IllegalArgumentException("input/output length is smaller than transform length");
            }
            workspace.reset();
            transform.exec(data, factor, forward, workspace);
        }
    }

    /**
     * Reusable plan for 1-D real transforms (r2c, c2r, FFTPACK, and Hartley
     * families).
     *
     * <p>Instances are thread-safe provided each thread uses its own
     * {@link FftWorkspace}.
     */
    public static final class PreparedReal {
        private final int length;
        private final int spectrumLength;
        private final FftRealTransform transform;
        private final FftWorkspace.Requirement workspaceRequirement;

        private PreparedReal(int length) {
            if (length < 1) {
                throw new IllegalArgumentException("zero-length FFT requested");
            }
            this.length = length;
            this.spectrumLength = complexSpectrumLength(length);
            this.transform = FftPlanCache.realTransform(length);
            this.workspaceRequirement = transform.workspaceRequirement()
                    .max(FftWorkspace.Requirement.doubles(length / 2));
        }

        /**
         * Returns the transform length (number of real samples).
         *
         * @return transform length
         */
        public int length() {
            return length;
        }

        /**
         * Returns the workspace requirement for this plan.
         *
         * @return workspace requirement
         */
        public FftWorkspace.Requirement workspaceRequirement() {
            return workspaceRequirement;
        }

        /**
         * Allocates a new workspace satisfying this plan's requirements.
         *
         * @return a fresh workspace
         */
        public FftWorkspace createWorkspace() {
            return workspaceRequirement.allocate();
        }

        /**
         * Computes the Hermitian-symmetric spectrum of a real signal.
         *
         * <p>The {@code input} array is not modified; an internal scratch
         * buffer from the workspace is used instead.
         *
         * @param input     real input array (not modified)
         * @param output    interleaved complex output
         * @param forward   {@code true} for forward sign convention
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace
         * @throws IllegalArgumentException if arrays are too short
         */
        public void r2c(double[] input, double[] output, boolean forward, double factor, FftWorkspace workspace) {
            Objects.requireNonNull(input, "input");
            Objects.requireNonNull(output, "output");
            Objects.requireNonNull(workspace, "workspace");
            if (input.length < length || output.length < spectrumLength) {
                throw new IllegalArgumentException("input/output length is smaller than transform length");
            }
            workspace.reset();
            System.arraycopy(input, 0, output, 0, length);
            if (!transform.tryRealToHermitianInPlace(output, 0, factor, forward, workspace)) {
                transform.exec(output, 0, factor, true, workspace);
                halfComplexToComplexInPlace(output, length, forward);
            }
        }

        /**
         * Reconstructs a real signal from its Hermitian-symmetric spectrum.
         *
         * @param input     interleaved complex spectrum
         * @param output    real output array
         * @param forward   {@code true} for forward sign convention
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace
         * @throws IllegalArgumentException if arrays are too short
         */
        public void c2r(double[] input, double[] output, boolean forward, double factor, FftWorkspace workspace) {
            Objects.requireNonNull(input, "input");
            Objects.requireNonNull(output, "output");
            Objects.requireNonNull(workspace, "workspace");
            if (input.length < spectrumLength || output.length < length) {
                throw new IllegalArgumentException("input/output length is smaller than transform length");
            }
            workspace.reset();
            complexToHalfComplex(input, output, 0, length, forward);
            transform.exec(output, 0, factor, false, workspace);
        }

        /**
         * In-place real FFTPACK transform using the supplied workspace.
         *
         * @param data            real array (modified in-place)
         * @param realToHermitian {@code true} for real→half-complex
         * @param forward         {@code true} for forward sign convention
         * @param factor          scaling factor
         * @param workspace       pre-allocated workspace
         * @throws IllegalArgumentException if {@code data} is too short
         */
        public void r2rFftpack(double[] data, boolean realToHermitian, boolean forward, double factor,
                              FftWorkspace workspace) {
            Objects.requireNonNull(data, "data");
            Objects.requireNonNull(workspace, "workspace");
            if (data.length < length) {
                throw new IllegalArgumentException("input/output length is smaller than transform length");
            }
            if (!realToHermitian && forward) {
                negateHalfComplexImaginary(data, length);
            }
            workspace.reset();
            transform.exec(data, factor, realToHermitian, workspace);
            if (realToHermitian && !forward) {
                negateHalfComplexImaginary(data, length);
            }
        }

        /**
         * Separable Hartley (or FHT) transform using the supplied workspace.
         *
         * @param input     real input array (not modified)
         * @param output    real output array
         * @param fht       {@code true} for FHT sign convention,
         *                  {@code false} for standard Hartley
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace
         * @throws IllegalArgumentException if arrays are too short
         */
        public void separableHartley(double[] input, double[] output, boolean fht, double factor,
                                     FftWorkspace workspace) {
            Objects.requireNonNull(input, "input");
            Objects.requireNonNull(output, "output");
            Objects.requireNonNull(workspace, "workspace");
            if (input.length < length || output.length < length) {
                throw new IllegalArgumentException("input/output length is smaller than transform length");
            }
            workspace.reset();
            System.arraycopy(input, 0, output, 0, length);
            transform.exec(output, 0, factor, true, workspace);
            workspace.reset();
            halfComplexToHartleyInPlace(output, length, fht, workspace);
        }

        /**
         * Genuine Hartley (or FHT) transform using the supplied workspace.
         *
         * <p>For 1-D data this is identical to
         * {@link #separableHartley(double[], double[], boolean, double, FftWorkspace)}.
         *
         * @param input     real input array (not modified)
         * @param output    real output array
         * @param fht       {@code true} for FHT sign convention
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace
         * @throws IllegalArgumentException if arrays are too short
         */
        public void genuineHartley(double[] input, double[] output, boolean fht, double factor,
                                   FftWorkspace workspace) {
            separableHartley(input, output, fht, factor, workspace);
        }
    }

    /**
     * Reusable plan for 1-D DCT or DST transforms.
     *
     * <p>Instances are thread-safe provided each thread uses its own
     * {@link FftWorkspace}.
     */
    public static final class PreparedDcst {
        private final int length;
        private final FftDcstPlan plan;
        private final FftWorkspace.Requirement workspaceRequirement;

        private PreparedDcst(int length, int type, boolean cosine) {
            this.length = length;
            this.plan = FftPlanCache.dcstPlan(length, type, cosine);
            this.workspaceRequirement = plan.workspaceRequirement();
        }

        /**
         * Returns the transform length.
         *
         * @return transform length
         */
        public int length() {
            return length;
        }

        /**
         * Returns the workspace requirement for this plan.
         *
         * @return workspace requirement
         */
        public FftWorkspace.Requirement workspaceRequirement() {
            return workspaceRequirement;
        }

        /**
         * Allocates a new workspace satisfying this plan's requirements.
         *
         * @return a fresh workspace
         */
        public FftWorkspace createWorkspace() {
            return workspaceRequirement.allocate();
        }

        /**
         * Executes the DCT or DST in-place using the supplied workspace.
         *
         * @param data      real array (modified in-place)
         * @param factor    scaling factor
         * @param ortho     {@code true} to apply orthonormal scaling
         * @param workspace pre-allocated workspace
         * @throws IllegalArgumentException if {@code data} is too short
         */
        public void exec(double[] data, double factor, boolean ortho, FftWorkspace workspace) {
            Objects.requireNonNull(data, "data");
            Objects.requireNonNull(workspace, "workspace");
            if (data.length < length) {
                throw new IllegalArgumentException("input/output length is smaller than transform length");
            }
            workspace.reset();
            plan.exec(data, factor, ortho, workspace);
        }
    }

    /**
     * Reusable plan for multi-dimensional complex-to-complex transforms.
     *
     * <p>Instances are thread-safe provided each thread uses its own
     * {@link FftWorkspace}.
     */
    public static final class PreparedNdComplex {
        private final int[] shape;
        private final int[] axes;
        private final FftComplexTransform[] transforms;
        private final FftWorkspace.Requirement workspaceRequirement;

        private PreparedNdComplex(int[] shape, int[] axes) {
            this.shape = Objects.requireNonNull(shape, "shape").clone();
            this.axes = Objects.requireNonNull(axes, "axes").clone();
            this.workspaceRequirement = FftNd.c2cWorkspaceRequirement(this.shape, this.axes);
            this.transforms = FftNd.complexTransforms(this.shape, this.axes);
        }

        /**
         * Returns a copy of the array shape.
         *
         * @return defensive copy of the shape
         */
        public int[] shape() {
            return shape.clone();
        }

        /**
         * Returns a copy of the transform axes.
         *
         * @return defensive copy of the axes
         */
        public int[] axes() {
            return axes.clone();
        }

        /**
         * Returns the workspace requirement for this plan.
         *
         * @return workspace requirement
         */
        public FftWorkspace.Requirement workspaceRequirement() {
            return workspaceRequirement;
        }

        /**
         * Allocates a new workspace satisfying this plan's requirements.
         *
         * @return a fresh workspace
         */
        public FftWorkspace createWorkspace() {
            return workspaceRequirement.allocate();
        }

        /**
         * Returns the workspace requirement for a concrete complex ND transform call.
         *
         * <p>This can be smaller than {@link #workspaceRequirement()} when the provided
         * strides use a contiguous output axis or the standard in-place 2D fast path.
         * The returned requirement is valid only for calls with the same layout. Use the
         * layout-agnostic {@link #workspaceRequirement()} when one workspace must serve arbitrary
         * stride combinations.</p>
         *
         * @param strideIn  input strides
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides
         * @param offsetOut element offset into the output array
         * @param input     interleaved complex input array
         * @param output    interleaved complex output array
         * @return workspace requirement for this concrete layout
         */
        public FftWorkspace.Requirement workspaceRequirement(long[] strideIn, long offsetIn, long[] strideOut,
                                                             long offsetOut, double[] input, double[] output) {
            return FftNd.c2cWorkspaceRequirement(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input,
                    output);
        }

        /**
         * Allocates a workspace sized for a concrete complex ND transform call.
         *
         * <p>This is the compact companion to {@link #createWorkspace()}. It is intended for hot
         * loops whose strides and input/output aliasing are fixed across invocations.</p>
         *
         * @param strideIn  input strides
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides
         * @param offsetOut element offset into the output array
         * @param input     interleaved complex input array
         * @param output    interleaved complex output array
         * @return a fresh workspace for this concrete layout
         */
        public FftWorkspace createWorkspace(long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                                            double[] input, double[] output) {
            return workspaceRequirement(strideIn, offsetIn, strideOut, offsetOut, input, output).allocate();
        }

        /**
         * Executes a multi-dimensional complex-to-complex transform.
         *
         * @param strideIn  input strides
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides
         * @param offsetOut element offset into the output array
         * @param forward   {@code true} for forward transform
         * @param input     interleaved complex input array
         * @param output    interleaved complex output array
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace
         */
        public void c2c(long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, boolean forward,
                        double[] input, double[] output, double factor, FftWorkspace workspace) {
            Objects.requireNonNull(workspace, "workspace");
            workspace.reset();
            FftNd.validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 2);
            FftNd.c2cPrepared(shape, strideIn, offsetIn, strideOut, offsetOut, axes, transforms, forward, input,
                    output, factor, workspace);
        }
    }

    /**
     * Reusable plan for multi-dimensional real transforms (r2c, c2r, FFTPACK,
     * and Hartley families).
     *
     * <p>Instances are thread-safe provided each thread uses its own
     * {@link FftWorkspace}.
     */
    public static final class PreparedNdReal {
        private final int[] shape;
        private final int[] axes;
        private final FftRealTransform[] realTransforms;
        private final int lastAxis;
        private final int[] hermitianShape;
        private final int[] complexAxes;
        private final long[] tempStride;
        private final int tempSize;
        private final FftRealTransform lastRealTransform;
        private final FftComplexTransform[] complexTransforms;
        private final FftWorkspace.Requirement workspaceRequirement;

        private PreparedNdReal(int[] shape, int[] axes) {
            this.shape = Objects.requireNonNull(shape, "shape").clone();
            this.axes = Objects.requireNonNull(axes, "axes").clone();
            FftWorkspace.Requirement requirement = FftNd.realAxisWorkspaceRequirement(this.shape, this.axes)
                    .max(FftNd.r2cWorkspaceRequirement(this.shape, this.axes))
                    .max(FftNd.c2rWorkspaceRequirement(this.shape, this.axes))
                    .max(FftNd.genuineHartleyWorkspaceRequirement(this.shape, this.axes));
            this.workspaceRequirement = requirement;
                this.realTransforms = FftNd.realTransforms(this.shape, this.axes);
                this.lastAxis = this.axes[this.axes.length - 1];
                this.hermitianShape = FftNd.hermitianShape(this.shape, lastAxis);
                this.complexAxes = Arrays.copyOf(this.axes, this.axes.length - 1);
                this.tempStride = FftNd.contiguousComplexStride(hermitianShape);
                this.tempSize = Math.toIntExact(2L * FftNd.product(hermitianShape));
                this.lastRealTransform = FftPlanCache.realTransform(this.shape[lastAxis]);
                this.complexTransforms = FftNd.complexTransforms(hermitianShape, complexAxes);
        }

        /**
         * Returns a copy of the array shape.
         *
         * @return defensive copy of the shape
         */
        public int[] shape() {
            return shape.clone();
        }

        /**
         * Returns a copy of the transform axes.
         *
         * @return defensive copy of the axes
         */
        public int[] axes() {
            return axes.clone();
        }

        /**
         * Returns the workspace requirement for this plan.
         *
         * @return workspace requirement
         */
        public FftWorkspace.Requirement workspaceRequirement() {
            return workspaceRequirement;
        }

        /**
         * Allocates a new workspace satisfying this plan's requirements.
         *
         * @return a fresh workspace
         */
        public FftWorkspace createWorkspace() {
            return workspaceRequirement.allocate();
        }

        /**
         * Returns the workspace requirement for a concrete real-to-complex ND transform call.
         *
         * <p>This can be smaller than {@link #workspaceRequirement()} for standard 2D layouts that
         * can stage the real-axis transform directly in the complex output rows before the complex
         * column pass. The returned requirement is valid only for calls with the same layout.</p>
         *
         * @param strideIn  input strides
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides (complex)
         * @param offsetOut element offset into the output array
         * @param input     real input array
         * @param output    interleaved complex output array
         * @return workspace requirement for this concrete real-to-complex layout
         */
        public FftWorkspace.Requirement r2cWorkspaceRequirement(long[] strideIn, long offsetIn, long[] strideOut,
                                                                long offsetOut, double[] input, double[] output) {
            return FftNd.r2cWorkspaceRequirement(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input,
                    output);
        }

        /**
         * Allocates a workspace sized for a concrete real-to-complex ND transform call.
         *
         * @param strideIn  input strides
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides (complex)
         * @param offsetOut element offset into the output array
         * @param input     real input array
         * @param output    interleaved complex output array
         * @return a fresh workspace for this concrete real-to-complex layout
         */
        public FftWorkspace createR2cWorkspace(long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                                               double[] input, double[] output) {
            return r2cWorkspaceRequirement(strideIn, offsetIn, strideOut, offsetOut, input, output).allocate();
        }

        /**
         * Returns the workspace requirement for a concrete complex-to-real ND transform call.
         *
         * <p>This can be smaller than {@link #workspaceRequirement()} for standard 2D layouts that
         * can pack transformed Hermitian columns directly into the real output rows before the final
         * real-axis transform. The returned requirement is valid only for calls with the same layout.</p>
         *
         * @param strideIn  input strides (complex)
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides
         * @param offsetOut element offset into the output array
         * @param input     interleaved complex input array
         * @param output    real output array
         * @return workspace requirement for this concrete complex-to-real layout
         */
        public FftWorkspace.Requirement c2rWorkspaceRequirement(long[] strideIn, long offsetIn, long[] strideOut,
                                                                long offsetOut, double[] input, double[] output) {
            return FftNd.c2rWorkspaceRequirement(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input,
                    output);
        }

        /**
         * Allocates a workspace sized for a concrete complex-to-real ND transform call.
         *
         * @param strideIn  input strides (complex)
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides
         * @param offsetOut element offset into the output array
         * @param input     interleaved complex input array
         * @param output    real output array
         * @return a fresh workspace for this concrete complex-to-real layout
         */
        public FftWorkspace createC2rWorkspace(long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                                               double[] input, double[] output) {
            return c2rWorkspaceRequirement(strideIn, offsetIn, strideOut, offsetOut, input, output).allocate();
        }

        /**
         * Multi-dimensional real-to-complex transform.
         *
         * @param strideIn  input strides
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides (complex)
         * @param offsetOut element offset into the output array
         * @param forward   {@code true} for forward transform
         * @param input     real input array
         * @param output    interleaved complex output array
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace
         */
        public void r2c(long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, boolean forward,
                        double[] input, double[] output, double factor, FftWorkspace workspace) {
            Objects.requireNonNull(workspace, "workspace");
            workspace.reset();
            FftNd.validateRealComplexCommon(shape, hermitianShape, strideIn, offsetIn, strideOut, offsetOut, axes,
                    input, output, true);
            FftNd.r2cPrepared(shape, hermitianShape, strideIn, offsetIn, strideOut, offsetOut, axes, complexAxes,
                    lastRealTransform, complexTransforms, forward, input, output, factor, workspace);
        }

        /**
         * Multi-dimensional complex-to-real transform.
         *
         * @param strideIn  input strides (complex)
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides
         * @param offsetOut element offset into the output array
         * @param forward   {@code true} for forward sign convention
         * @param input     interleaved complex input array
         * @param output    real output array
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace
         */
        public void c2r(long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, boolean forward,
                        double[] input, double[] output, double factor, FftWorkspace workspace) {
            Objects.requireNonNull(workspace, "workspace");
            workspace.reset();
            FftNd.validateRealComplexCommon(shape, hermitianShape, strideOut, offsetOut, strideIn, offsetIn, axes,
                    output, input, false);
            FftNd.c2rPrepared(shape, hermitianShape, tempStride, tempSize, strideIn, offsetIn, strideOut, offsetOut,
                    axes, complexAxes, lastRealTransform, complexTransforms, forward, input, output, factor,
                    workspace);
        }

        /**
         * Multi-dimensional real FFTPACK transform.
         *
         * @param strideIn        input strides
         * @param offsetIn        element offset into the input array
         * @param strideOut       output strides
         * @param offsetOut       element offset into the output array
         * @param realToHermitian {@code true} for real→half-complex
         * @param forward         {@code true} for forward sign convention
         * @param input           real input array
         * @param output          real output array
         * @param factor          scaling factor
         * @param workspace       pre-allocated workspace
         */
        public void r2rFftpack(long[] strideIn, long offsetIn, long[] strideOut, long offsetOut,
                               boolean realToHermitian, boolean forward, double[] input, double[] output,
                               double factor, FftWorkspace workspace) {
            Objects.requireNonNull(workspace, "workspace");
            workspace.reset();
            FftNd.validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 1);
            FftNd.r2rFftpackPrepared(shape, strideIn, offsetIn, strideOut, offsetOut, axes, realTransforms,
                    realToHermitian, forward, input, output, factor, workspace);
        }

        /**
         * Multi-dimensional separable Hartley (or FHT) transform.
         *
         * @param strideIn  input strides
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides
         * @param offsetOut element offset into the output array
         * @param fht       {@code true} for FHT sign convention
         * @param input     real input array
         * @param output    real output array
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace
         */
        public void separableHartley(long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, boolean fht,
                                     double[] input, double[] output, double factor, FftWorkspace workspace) {
            Objects.requireNonNull(workspace, "workspace");
            workspace.reset();
            FftNd.validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 1);
            FftNd.separableHartleyPrepared(shape, strideIn, offsetIn, strideOut, offsetOut, axes, realTransforms,
                    fht, input, output, factor, workspace);
        }

        /**
         * Multi-dimensional genuine Hartley (or FHT) transform.
         *
         * @param strideIn  input strides
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides
         * @param offsetOut element offset into the output array
         * @param fht       {@code true} for FHT sign convention
         * @param input     real input array
         * @param output    real output array
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace
         */
        public void genuineHartley(long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, boolean fht,
                                   double[] input, double[] output, double factor, FftWorkspace workspace) {
            Objects.requireNonNull(workspace, "workspace");
            workspace.reset();
            FftNd.validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 1);
            FftNd.genuineHartleyPrepared(shape, hermitianShape, tempStride, tempSize, strideIn, offsetIn, strideOut,
                    offsetOut, axes, complexAxes, realTransforms, lastRealTransform, complexTransforms, fht, input,
                    output, factor, workspace);
        }
    }

    /**
     * Reusable plan for multi-dimensional DCT or DST transforms.
     *
     * <p>Instances are thread-safe provided each thread uses its own
     * {@link FftWorkspace}.
     */
    public static final class PreparedNdDcst {
        private final int[] shape;
        private final int[] axes;
        private final FftDcstPlan[] plans;
        private final int type;
        private final boolean cosine;
        private final FftWorkspace.Requirement workspaceRequirement;

        private PreparedNdDcst(int[] shape, int[] axes, int type, boolean cosine) {
            if (type < 1 || type > 4) {
                throw new IllegalArgumentException("invalid DCT/DST type");
            }
            this.shape = Objects.requireNonNull(shape, "shape").clone();
            this.axes = Objects.requireNonNull(axes, "axes").clone();
            this.type = type;
            this.cosine = cosine;
            this.workspaceRequirement = FftNd.dcstWorkspaceRequirement(this.shape, this.axes, type, cosine);
            this.plans = FftNd.dcstPlans(this.shape, this.axes, type, cosine);
        }

        /**
         * Returns a copy of the array shape.
         *
         * @return defensive copy of the shape
         */
        public int[] shape() {
            return shape.clone();
        }

        /**
         * Returns a copy of the transform axes.
         *
         * @return defensive copy of the axes
         */
        public int[] axes() {
            return axes.clone();
        }

        /**
         * Returns the workspace requirement for this plan.
         *
         * @return workspace requirement
         */
        public FftWorkspace.Requirement workspaceRequirement() {
            return workspaceRequirement;
        }

        /**
         * Allocates a new workspace satisfying this plan's requirements.
         *
         * @return a fresh workspace
         */
        public FftWorkspace createWorkspace() {
            return workspaceRequirement.allocate();
        }

        /**
         * Executes the multi-dimensional DCT or DST.
         *
         * @param strideIn  input strides
         * @param offsetIn  element offset into the input array
         * @param strideOut output strides
         * @param offsetOut element offset into the output array
         * @param ortho     {@code true} to apply orthonormal scaling
         * @param input     real input array
         * @param output    real output array
         * @param factor    scaling factor
         * @param workspace pre-allocated workspace
         */
        public void exec(long[] strideIn, long offsetIn, long[] strideOut, long offsetOut, boolean ortho,
                         double[] input, double[] output, double factor, FftWorkspace workspace) {
            Objects.requireNonNull(workspace, "workspace");
            workspace.reset();
            FftNd.validateCommon(shape, strideIn, offsetIn, strideOut, offsetOut, axes, input, output, 1);
            FftNd.dcstPrepared(shape, strideIn, offsetIn, strideOut, offsetOut, axes, plans, type, cosine, ortho,
                    input, output, factor, workspace);
        }
    }

    // ── Private helpers ─────────────────────────────────────────────────

    private static int complexLength(double[] data, String name) {
        if ((data.length & 1) != 0) {
            throw new IllegalArgumentException(name + " length must be even for interleaved complex data");
        }
        return data.length / 2;
    }

    private static int complexSpectrumLength(int realLength) {
        long required = 2L * (realLength / 2L + 1L);
        if (required > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Hermitian spectrum length is too large");
        }
        return (int) required;
    }

    private static void halfComplexToComplex(double[] halfComplex, double[] complex, boolean forward) {
        halfComplexToComplex(halfComplex, 0, halfComplex.length, complex, forward);
    }

    private static void halfComplexToComplex(double[] halfComplex, int halfComplexOffset, int length,
                                             double[] complex, boolean forward) {
        double sign = forward ? 1.0 : -1.0;
        complex[0] = halfComplex[halfComplexOffset];
        complex[1] = 0.0;
        for (int frequency = 1; frequency <= length / 2; frequency++) {
            int complexOffset = 2 * frequency;
            if (2 * frequency == length) {
                complex[complexOffset] = halfComplex[halfComplexOffset + length - 1];
                complex[complexOffset + 1] = 0.0;
            } else {
                complex[complexOffset] = halfComplex[halfComplexOffset + 2 * frequency - 1];
                complex[complexOffset + 1] = sign * halfComplex[halfComplexOffset + 2 * frequency];
            }
        }
    }

    private static void halfComplexToComplexInPlace(double[] data, int length, boolean forward) {
        double sign = forward ? 1.0 : -1.0;
        for (int frequency = length / 2; frequency >= 1; frequency--) {
            int complexOffset = 2 * frequency;
            if (2 * frequency == length) {
                double real = data[length - 1];
                data[complexOffset] = real;
                data[complexOffset + 1] = 0.0;
            } else {
                double real = data[2 * frequency - 1];
                double imaginary = data[2 * frequency];
                data[complexOffset] = real;
                data[complexOffset + 1] = sign * imaginary;
            }
        }
        data[1] = 0.0;
    }

    private static void dcst1d(double[] data, int type, boolean cosine, double factor, boolean ortho) {
        Objects.requireNonNull(data, "data");
        if (data.length == 0) return;
        FftPlanCache.dcstPlan(data.length, type, cosine).exec(data, factor, ortho);
    }

    private static void realFamily1d(double[] data, double factor, boolean fht, boolean genuine) {
        Objects.requireNonNull(data, "data");
        if (data.length == 0) return;
        int[] shape = {data.length};
        long[] stride = {1L};
        int[] axes = {0};
        if (genuine) {
            FftNd.genuineHartley(shape, stride, 0L, stride, 0L, axes, fht, data, data, factor);
        } else {
            FftNd.separableHartley(shape, stride, 0L, stride, 0L, axes, fht, data, data, factor);
        }
    }

    private static void complexToHalfComplex(double[] complex, double[] halfComplex, boolean forward) {
        complexToHalfComplex(complex, halfComplex, 0, halfComplex.length, forward);
    }

    private static void complexToHalfComplex(double[] complex, double[] halfComplex, int halfComplexOffset,
                                             int length, boolean forward) {
        double sign = forward ? -1.0 : 1.0;
        halfComplex[halfComplexOffset] = complex[0];
        for (int frequency = 1; frequency <= length / 2; frequency++) {
            int complexOffset = 2 * frequency;
            if (2 * frequency == length) {
                halfComplex[halfComplexOffset + length - 1] = complex[complexOffset];
            } else {
                halfComplex[halfComplexOffset + 2 * frequency - 1] = complex[complexOffset];
                halfComplex[halfComplexOffset + 2 * frequency] = sign * complex[complexOffset + 1];
            }
        }
    }

    private static void negateHalfComplexImaginary(double[] halfComplex) {
        negateHalfComplexImaginary(halfComplex, halfComplex.length);
    }

    private static void negateHalfComplexImaginary(double[] halfComplex, int length) {
        for (int index = 2; index < length; index += 2) {
            halfComplex[index] = -halfComplex[index];
        }
    }

    private static void halfComplexToHartleyInPlace(double[] data, int length, boolean fht,
                                                    FftWorkspace workspace) {
        int pairCount = length / 2;
        if (pairCount == 0) {
            return;
        }
        double[] highValues = workspace.doubles();
        int highOffset = workspace.alloc(pairCount);
        int highIndex = 0;
        int frequency = 1;
        for (; 2 * frequency < length; frequency++) {
            double real = data[2 * frequency - 1];
            double imaginary = data[2 * frequency];
            data[frequency] = fht ? real - imaginary : real + imaginary;
            highValues[highOffset + highIndex++] = fht ? real + imaginary : real - imaginary;
        }
        if (2 * frequency == length) {
            data[frequency] = data[length - 1];
        }
        for (int index = 0; index < highIndex; index++) {
            data[length - 1 - index] = highValues[highOffset + index];
        }
    }
}
