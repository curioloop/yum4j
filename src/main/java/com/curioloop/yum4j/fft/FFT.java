package com.curioloop.yum4j.fft;

/**
 * User-friendly entry point for Fast Fourier Transform operations.
 *
 * <p>Method names follow NumPy conventions ({@code fft}, {@code ifft},
 * {@code rfft}, {@code irfft}, {@code dct}, {@code idct}, etc.).  Inverse
 * transforms apply {@code 1/N} normalisation automatically.
 *
 * <p>All 1-D methods operate <b>in-place</b>: the input array is modified
 * and contains the result on return.  For {@link #rfft} and {@link #irfft}
 * the input is also modified (used as scratch space) and the result is
 * written to the separate output array.
 *
 * <h3>Zero-allocation hot path</h3>
 *
 * <p>Each method has a {@code Pool} overload for zero-allocation repeated
 * transforms of the same length:
 * <pre>{@code
 * FFT.CmplxPool pool = FFT.cmplxPool(1024);
 * for (double[] frame : frames) {
 *     FFT.fft(frame, pool);   // zero allocation
 * }
 * }</pre>
 *
 * <p>Pool instances are <b>not thread-safe</b>.  In multi-threaded or
 * virtual-thread scenarios, give each thread its own pool.
 *
 * <p>For full control over scaling factors, sign conventions, and workspace
 * management, use the lower-level {@link Transform} class directly.
 *
 * @see Transform
 * @see FftPlanCache
 */
public final class FFT {
    private FFT() {
    }

    // ── Type enum ──────────────────────────────────────────────────────

    /** DCT/DST type selector. */
    public enum Type {
        /** Type I — self-inverse (up to normalisation). */
        I(1),
        /** Type II — the "standard" DCT/DST; inverse is {@link #III}. */
        II(2),
        /** Type III — inverse of {@link #II}. */
        III(3),
        /** Type IV — self-inverse (up to normalisation). */
        IV(4);

        final int value;

        Type(int value) { this.value = value; }

        /** Returns the inverse type for round-trip transforms. */
        Type inverse() {
            return switch (this) {
                case I  -> I;
                case II -> III;
                case III -> II;
                case IV -> IV;
            };
        }
    }

    // ── Pool types ─────────────────────────────────────────────────────

    /**
     * Reusable workspace for complex FFT of a fixed length.
     * Not thread-safe.  Create via {@link FFT#cmplxPool(int)}.
     */
    public record CmplxPool(Transform.PreparedComplex plan, FftWorkspace workspace) {
        /** In-place forward complex FFT. */
        public void fft(double[] data) { plan.c2c(data, true, 1.0, workspace); }
        /** In-place inverse complex FFT with {@code 1/N} normalisation. */
        public void ifft(double[] data) { plan.c2c(data, false, 2.0 / data.length, workspace); }
    }

    /**
     * Reusable workspace for real FFT of a fixed length.
     * Not thread-safe.  Create via {@link FFT#realPool(int)}.
     */
    public record RealPool(Transform.PreparedReal plan, FftWorkspace workspace, int length) {
        /** Forward real-to-complex FFT. Input is used as scratch by the pool's internal buffer. */
        public void rfft(double[] input, double[] output) { plan.r2c(input, output, true, 1.0, workspace); }
        /** Inverse complex-to-real FFT with {@code 1/N} normalisation. */
        public void irfft(double[] input, double[] output) { plan.c2r(input, output, false, 1.0 / length, workspace); }
    }

    /**
     * Reusable workspace for DCT of a fixed length and type.
     * Not thread-safe.  Create via {@link FFT#dctPool(int, Type)}.
     */
    public record DctPool(Transform.PreparedDcst forwardPlan, Transform.PreparedDcst inversePlan,
                          FftWorkspace forwardWorkspace, FftWorkspace inverseWorkspace, Type type) {
        /** In-place forward DCT (orthonormal). */
        public void forward(double[] data) { forwardPlan.exec(data, 1.0, true, forwardWorkspace); }
        /** In-place inverse DCT with automatic normalisation (orthonormal). */
        public void inverse(double[] data) { inversePlan.exec(data, dctInverseFactor(type, data.length, true), true, inverseWorkspace); }
    }

    /**
     * Reusable workspace for DST of a fixed length and type.
     * Not thread-safe.  Create via {@link FFT#dstPool(int, Type)}.
     */
    public record DstPool(Transform.PreparedDcst forwardPlan, Transform.PreparedDcst inversePlan,
                          FftWorkspace forwardWorkspace, FftWorkspace inverseWorkspace, Type type) {
        /** In-place forward DST (orthonormal). */
        public void forward(double[] data) { forwardPlan.exec(data, 1.0, true, forwardWorkspace); }
        /** In-place inverse DST with automatic normalisation (orthonormal). */
        public void inverse(double[] data) { inversePlan.exec(data, dctInverseFactor(type, data.length, false), true, inverseWorkspace); }
    }

    // ── Pool factories ─────────────────────────────────────────────────

    /** Creates a reusable complex FFT pool for the given length. */
    public static CmplxPool cmplxPool(int length) {
        var plan = Transform.prepareComplex(length);
        return new CmplxPool(plan, plan.createWorkspace());
    }

    /** Creates a reusable real FFT pool for the given length. */
    public static RealPool realPool(int length) {
        var plan = Transform.prepareReal(length);
        return new RealPool(plan, plan.createWorkspace(), length);
    }

    /** Creates a reusable DCT pool for the given length and type. */
    public static DctPool dctPool(int length, Type type) {
        var fwd = Transform.prepareDct(length, type.value);
        var inv = Transform.prepareDct(length, type.inverse().value);
        return new DctPool(fwd, inv, fwd.createWorkspace(), inv.createWorkspace(), type);
    }

    /** Creates a reusable DST pool for the given length and type. */
    public static DstPool dstPool(int length, Type type) {
        var fwd = Transform.prepareDst(length, type.value);
        var inv = Transform.prepareDst(length, type.inverse().value);
        return new DstPool(fwd, inv, fwd.createWorkspace(), inv.createWorkspace(), type);
    }

    // ── Complex FFT ────────────────────────────────────────────────────

    /** In-place forward complex FFT. */
    public static void fft(double[] data) { Transform.c2c(data, true, 1.0); }
    /** In-place forward complex FFT using a pre-allocated pool. */
    public static void fft(double[] data, CmplxPool pool) { pool.fft(data); }
    /** In-place inverse complex FFT with {@code 1/N} normalisation. */
    public static void ifft(double[] data) { Transform.c2c(data, false, 2.0 / data.length); }
    /** In-place inverse complex FFT using a pre-allocated pool. */
    public static void ifft(double[] data, CmplxPool pool) { pool.ifft(data); }

    // ── Real ↔ Complex ─────────────────────────────────────────────────

    /** Forward real-to-complex FFT. Input is used as scratch. */
    public static void rfft(double[] input, double[] output) { Transform.r2c(input, output, true, 1.0); }
    /** Forward real-to-complex FFT using a pre-allocated pool. */
    public static void rfft(double[] input, double[] output, RealPool pool) { pool.rfft(input, output); }
    /** Inverse complex-to-real FFT with {@code 1/N} normalisation. */
    public static void irfft(double[] input, double[] output) { Transform.c2r(input, output, false, 1.0 / output.length); }
    /** Inverse complex-to-real FFT using a pre-allocated pool. */
    public static void irfft(double[] input, double[] output, RealPool pool) { pool.irfft(input, output); }

    // ── DCT ────────────────────────────────────────────────────────────

    /** In-place forward DCT (type-II, orthonormal). */
    public static void dct(double[] data) { dct(data, Type.II); }
    /** In-place forward DCT of the specified type (orthonormal). */
    public static void dct(double[] data, Type type) { Transform.dct(data, type.value, 1.0, true); }
    /** In-place forward DCT using a pre-allocated pool. */
    public static void dct(double[] data, DctPool pool) { pool.forward(data); }
    /** In-place inverse DCT (type-II default, orthonormal, auto-normalised). */
    public static void idct(double[] data) { idct(data, Type.II); }
    /** In-place inverse DCT of the specified type (orthonormal, auto-normalised). */
    public static void idct(double[] data, Type type) { Transform.dct(data, type.inverse().value, dctInverseFactor(type, data.length, true), true); }
    /** In-place inverse DCT using a pre-allocated pool. */
    public static void idct(double[] data, DctPool pool) { pool.inverse(data); }

    // ── DST ────────────────────────────────────────────────────────────

    /** In-place forward DST (type-II, orthonormal). */
    public static void dst(double[] data) { dst(data, Type.II); }
    /** In-place forward DST of the specified type (orthonormal). */
    public static void dst(double[] data, Type type) { Transform.dst(data, type.value, 1.0, true); }
    /** In-place forward DST using a pre-allocated pool. */
    public static void dst(double[] data, DstPool pool) { pool.forward(data); }
    /** In-place inverse DST (type-II default, orthonormal, auto-normalised). */
    public static void idst(double[] data) { idst(data, Type.II); }
    /** In-place inverse DST of the specified type (orthonormal, auto-normalised). */
    public static void idst(double[] data, Type type) { Transform.dst(data, type.inverse().value, dctInverseFactor(type, data.length, false), true); }
    /** In-place inverse DST using a pre-allocated pool. */
    public static void idst(double[] data, DstPool pool) { pool.inverse(data); }

    // ── Private helpers ────────────────────────────────────────────────

    static double dctInverseFactor(Type type, int length, boolean cosine) {
        return switch (type) {
            case I  -> 1.0 / (2.0 * (length + (cosine ? -1 : 1)));
            case II, III, IV -> 1.0 / (2.0 * length);
        };
    }
}
