package com.curioloop.yum4j.math;

import com.curioloop.yum4j.linalg.blas.BLAS;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Common numerical vector operations for dense {@code double[]} arrays.
 *
 * <p>Each operation is available in two forms:</p>
 * <ul>
 *   <li>Full-array: {@code op(x)} — operates on {@code x[0..x.length-1]}</li>
 *   <li>Slice: {@code op(x, off, len)} — operates on {@code x[off..off+len-1]}</li>
 * </ul>
 * <p>The full-array form delegates to the slice form with {@code off=0, len=x.length}.</p>
 */
public final class VectorOps {

    private VectorOps() {
    }

    // ── Reduction — single array ──────────────────────────────────────────────

    /** Returns true if {@code x[off..off+len-1]} contains {@code NaN} or {@code ±Infinity}. */
    public static boolean hasInf(double[] x, int off, int len) {
        if (len >= VectorOpsSIMD.MIN_N && VectorOpsSIMD.SUPPORTED) {
            return VectorOpsSIMD.hasInf(x, off, len);
        }
        for (int i = off, end = off + len; i < end; i++) {
            if (!Double.isFinite(x[i])) return true;
        }
        return false;
    }

    /** Returns true if any element contains {@code NaN} or {@code ±Infinity}. */
    public static boolean hasInf(double[] x) { return hasInf(x, 0, x.length); }

    /** Returns the sum of {@code x[off..off+len-1]}: {@code Σ xᵢ}. */
    public static double sum(double[] x, int off, int len) {
        if (len >= VectorOpsSIMD.MIN_N && VectorOpsSIMD.SUPPORTED) {
            return VectorOpsSIMD.sum(x, off, len);
        }
        double s = 0.0;
        for (int i = off, end = off + len; i < end; i++) s += x[i];
        return s;
    }

    /** Returns the sum of all elements: {@code Σ xᵢ}. */
    public static double sum(double[] x) { return sum(x, 0, x.length); }

    /** Returns the arithmetic mean of {@code x[off..off+len-1]}: {@code Σ xᵢ / len}. */
    public static double mean(double[] x, int off, int len) { return sum(x, off, len) / len; }

    /** Returns the arithmetic mean of all elements: {@code Σ xᵢ / n}. */
    public static double mean(double[] x) { return mean(x, 0, x.length); }

    /** Returns the maximum of {@code x[off..off+len-1]}: {@code max xᵢ}. */
    public static double max(double[] x, int off, int len) {
        if (len >= VectorOpsSIMD.MIN_N && VectorOpsSIMD.SUPPORTED) {
            return VectorOpsSIMD.max(x, off, len);
        }
        if (len == 0) return Double.NEGATIVE_INFINITY;
        double m = x[off];
        for (int i = off + 1, end = off + len; i < end; i++) m = Math.max(m, x[i]);
        return m;
    }

    /** Returns the maximum of all elements: {@code max xᵢ}. */
    public static double max(double[] x) { return max(x, 0, x.length); }

    /** Returns the minimum of {@code x[off..off+len-1]}: {@code min xᵢ}. */
    public static double min(double[] x, int off, int len) {
        if (len >= VectorOpsSIMD.MIN_N && VectorOpsSIMD.SUPPORTED) {
            return VectorOpsSIMD.min(x, off, len);
        }
        if (len == 0) return Double.POSITIVE_INFINITY;
        double m = x[off];
        for (int i = off + 1, end = off + len; i < end; i++) m = Math.min(m, x[i]);
        return m;
    }

    /** Returns the minimum of all elements: {@code min xᵢ}. */
    public static double min(double[] x) { return min(x, 0, x.length); }

    /** Returns the L1 norm of {@code x[off..off+len-1]}: {@code Σ |xᵢ|}. */
    public static double l1Norm(double[] x, int off, int len) { return BLAS.dasum(len, x, off, 1); }

    /** Returns the L1 norm of all elements: {@code Σ |xᵢ|}. */
    public static double l1Norm(double[] x) { return l1Norm(x, 0, x.length); }

    /**
     * Returns the sum of squared differences from a constant of {@code x[off..off+len-1]}:
     * {@code Σ (xᵢ - delta)²}.
     */
    public static double sumSq(double[] x, int off, int len, double delta) {
        if (len >= VectorOpsSIMD.MIN_N && VectorOpsSIMD.SUPPORTED) {
            return VectorOpsSIMD.sumSq(x, off, len, delta);
        }
        double s = 0.0;
        for (int i = off, end = off + len; i < end; i++) { double d = x[i] - delta; s += d * d; }
        return s;
    }

    /** Returns the sum of squared differences from a constant: {@code Σ (xᵢ - delta)²}. */
    public static double sumSq(double[] x, double delta) { return sumSq(x, 0, x.length, delta); }

    /** Returns the biased standard deviation of {@code x[off..off+len-1]}. */
    public static double std(double[] x, int off, int len) {
        return Math.sqrt(sumSq(x, off, len, mean(x, off, len)) / len);
    }

    /** Returns the biased standard deviation of all elements. */
    public static double std(double[] x) { return std(x, 0, x.length); }

    /** Returns the maximum absolute value of {@code x[off..off+len-1]}: {@code max |xᵢ|}. */
    public static double maxAbs(double[] x, int off, int len) { return BLAS.damax(len, x, off, 1); }

    /** Returns the maximum absolute value of all elements: {@code max |xᵢ|}. */
    public static double maxAbs(double[] x) { return maxAbs(x, 0, x.length); }

    /** Returns the L2 norm of {@code x[off..off+len-1]} (Neumaier's scaled algorithm). */
    public static double l2Norm(double[] x, int off, int len) { return BLAS.dnrm2(len, x, off, 1); }

    /** Returns the L2 norm of all elements. */
    public static double l2Norm(double[] x) { return l2Norm(x, 0, x.length); }

    // ── Reduction — two arrays ────────────────────────────────────────────────

    /** Returns the dot product of slices: {@code Σ x[xOff+i]·y[yOff+i]} for {@code i=0..len-1}. */
    public static double dot(double[] x, int xOff, double[] y, int yOff, int len) {
        return BLAS.ddot(len, x, xOff, 1, y, yOff, 1);
    }

    /** Returns the dot product of all elements: {@code Σ xᵢ·yᵢ}. */
    public static double dot(double[] x, double[] y) { return dot(x, 0, y, 0, x.length); }

    /** Returns the L1 distance of slices: {@code Σ |x[xOff+i] - y[yOff+i]|}. */
    public static double l1Dist(double[] x, int xOff, double[] y, int yOff, int len) {
        if (len >= VectorOpsSIMD.MIN_N && VectorOpsSIMD.SUPPORTED) {
            return VectorOpsSIMD.l1Dist(x, xOff, y, yOff, len);
        }
        double s = 0.0;
        for (int i = 0; i < len; i++) s += Math.abs(x[xOff + i] - y[yOff + i]);
        return s;
    }

    /** Returns the L1 distance of all elements: {@code Σ |xᵢ - yᵢ|}. */
    public static double l1Dist(double[] x, double[] y) { return l1Dist(x, 0, y, 0, x.length); }

    /** Returns the L∞ distance of slices: {@code max |x[xOff+i] - y[yOff+i]|}. */
    public static double linfDist(double[] x, int xOff, double[] y, int yOff, int len) {
        if (len >= VectorOpsSIMD.MIN_N && VectorOpsSIMD.SUPPORTED) {
            return VectorOpsSIMD.linfDist(x, xOff, y, yOff, len);
        }
        if (len == 0) return 0.0;
        double norm = Math.abs(x[xOff] - y[yOff]);
        for (int i = 1; i < len; i++) {
            double d = Math.abs(x[xOff + i] - y[yOff + i]);
            if (d > norm || Double.isNaN(norm)) norm = d;
        }
        return norm;
    }

    /** Returns the L∞ distance of all elements: {@code max |xᵢ - yᵢ|}. */
    public static double linfDist(double[] x, double[] y) { return linfDist(x, 0, y, 0, x.length); }

    /** Returns the L2 distance of slices (Neumaier's scaled algorithm). */
    public static double l2Dist(double[] x, int xOff, double[] y, int yOff, int len) {
        if (len >= VectorOpsSIMD.MIN_N && VectorOpsSIMD.SUPPORTED) {
            double fast = VectorOpsSIMD.l2DistFast(x, xOff, y, yOff, len);
            if (!Double.isNaN(fast)) return fast;
            // Fast path overflowed (or hit a non-finite difference); fall through
            // to the scaled Neumaier loop which handles extreme ranges and NaN.
        }
        double scale = 0.0, sumSq = 1.0;
        for (int i = 0; i < len; i++) {
            double v = x[xOff + i] - y[yOff + i];
            if (v == 0.0) continue;
            double abs = Math.abs(v);
            if (Double.isNaN(abs)) return Double.NaN;
            if (scale < abs) { double s = scale / abs; sumSq = 1.0 + sumSq * s * s; scale = abs; }
            else             { double s = abs / scale; sumSq += s * s; }
        }
        if (Double.isInfinite(scale)) return Double.POSITIVE_INFINITY;
        return scale * Math.sqrt(sumSq);
    }

    /** Returns the L2 distance of all elements. */
    public static double l2Dist(double[] x, double[] y) { return l2Dist(x, 0, y, 0, x.length); }

    // ── In-place mutation ─────────────────────────────────────────────────────

    /** Adds a scalar to {@code x[off..off+len-1]} in-place: {@code xᵢ += alpha}. */
    public static void addConst(double alpha, double[] x, int off, int len) {
        for (int i = off, end = off + len; i < end; i++) x[i] += alpha;
    }

    /** Adds a scalar to all elements in-place: {@code xᵢ += alpha}. */
    public static void addConst(double alpha, double[] x) { addConst(alpha, x, 0, x.length); }

    /** Adds {@code s[sOff..]} to {@code dst[dOff..]} in-place: {@code dst[dOff+i] += s[sOff+i]}. */
    public static void add(double[] dst, int dOff, double[] s, int sOff, int len) {
        BLAS.daxpy(len, 1.0, s, sOff, 1, dst, dOff, 1);
    }

    /** Adds {@code s} to {@code dst} element-wise in-place: {@code dstᵢ += sᵢ}. */
    public static void add(double[] dst, double[] s) { add(dst, 0, s, 0, dst.length); }

    /** Divides {@code dst[dOff..]} by {@code s[sOff..]} in-place: {@code dst[dOff+i] /= s[sOff+i]}. */
    public static void div(double[] dst, int dOff, double[] s, int sOff, int len) {
        for (int i = 0; i < len; i++) dst[dOff + i] /= s[sOff + i];
    }

    /** Divides {@code dst} by {@code s} element-wise in-place: {@code dstᵢ /= sᵢ}. */
    public static void div(double[] dst, double[] s) { div(dst, 0, s, 0, dst.length); }

    /** Scales {@code x[off..off+len-1]} in-place: {@code xᵢ *= alpha}. */
    public static void scal(double alpha, double[] x, int off, int len) {
        BLAS.dscal(len, alpha, x, off, 1);
    }

    /** Scales all elements in-place: {@code xᵢ *= alpha}. */
    public static void scal(double alpha, double[] x) { scal(alpha, x, 0, x.length); }

    /** AXPY on slices in-place: {@code y[yOff+i] += alpha·x[xOff+i]}. */
    public static void axpy(double alpha, double[] x, int xOff, double[] y, int yOff, int len) {
        BLAS.daxpy(len, alpha, x, xOff, 1, y, yOff, 1);
    }

    /** AXPY in-place: {@code yᵢ += alpha·xᵢ}. */
    public static void axpy(double alpha, double[] x, double[] y) { axpy(alpha, x, 0, y, 0, x.length); }

    // ── Output to separate array ──────────────────────────────────────────────

    /** Element-wise division to output: {@code dst[dOff+i] = x[xOff+i] / y[yOff+i]}. */
    public static void divTo(double[] dst, int dOff, double[] x, int xOff, double[] y, int yOff, int len) {
        for (int i = 0; i < len; i++) dst[dOff + i] = x[xOff + i] / y[yOff + i];
    }

    /** Element-wise division to output: {@code dstᵢ = xᵢ / yᵢ}. */
    public static double[] divTo(double[] dst, double[] x, double[] y) {
        divTo(dst, 0, x, 0, y, 0, x.length);
        return dst;
    }

    /** Scales to output: {@code dst[dOff+i] = alpha·x[xOff+i]}. Supports overlapping slices. */
    public static void scalTo(double[] dst, int dOff, double alpha, double[] x, int xOff, int len) {
        if (len == 0) return;
        if (dst == x && dOff == xOff) {
            if (alpha != 1.0) scal(alpha, dst, dOff, len);
            return;
        }
        if (alpha == 1.0) {
            System.arraycopy(x, xOff, dst, dOff, len);
            return;
        }
        if (dst == x && dOff > xOff && dOff < xOff + len) {
            for (int i = len - 1; i >= 0; i--) dst[dOff + i] = alpha * x[xOff + i];
            return;
        }
        for (int i = 0; i < len; i++) dst[dOff + i] = alpha * x[xOff + i];
    }

    /** Scales to output: {@code dstᵢ = alpha·xᵢ}. */
    public static double[] scalTo(double[] dst, double alpha, double[] x) {
        scalTo(dst, 0, alpha, x, 0, x.length);
        return dst;
    }

    /** AXPY to output: {@code dst[dOff+i] = alpha·x[xOff+i] + y[yOff+i]}. */
    public static void axpyTo(double[] dst, int dOff, double alpha,
                              double[] x, int xOff, double[] y, int yOff, int len) {
        if (len >= VectorOpsSIMD.MIN_N && VectorOpsSIMD.SUPPORTED) {
            VectorOpsSIMD.axpyTo(dst, dOff, alpha, x, xOff, y, yOff, len);
            return;
        }
        for (int i = 0; i < len; i++) dst[dOff + i] = alpha * x[xOff + i] + y[yOff + i];
    }

    /** AXPY to output: {@code dstᵢ = alpha·xᵢ + yᵢ}. */
    public static double[] axpyTo(double[] dst, double alpha, double[] x, double[] y) {
        axpyTo(dst, 0, alpha, x, 0, y, 0, x.length);
        return dst;
    }

    // ── Prefix operations ─────────────────────────────────────────────────────

    /**
     * Computes the cumulative sum of {@code s[sOff..sOff+len-1]} into
     * {@code dst[dOff..dOff+len-1]}: {@code dst[dOff+i] = Σ_{j≤i} s[sOff+j]}.
     *
     * @param dst  destination array (length ≥ dOff + len)
     * @param dOff offset into dst
     * @param s    source array
     * @param sOff offset into s
     * @param len  number of elements
     * @return {@code dst}
     */
    public static double[] cumSum(double[] dst, int dOff, double[] s, int sOff, int len) {
        if (len == 0) return dst;
        dst[dOff] = s[sOff];
        int i = 1;
        for (; i + 3 < len; i += 4) {
            double s0 = s[sOff+i], s1 = s[sOff+i+1], s2 = s[sOff+i+2], s3 = s[sOff+i+3];
            double d0 = dst[dOff+i-1] + s0;
            double d1 = d0 + s1;
            double d2 = d1 + s2;
            double d3 = d2 + s3;
            dst[dOff+i] = d0; dst[dOff+i+1] = d1; dst[dOff+i+2] = d2; dst[dOff+i+3] = d3;
        }
        for (; i < len; i++) dst[dOff+i] = dst[dOff+i-1] + s[sOff+i];
        return dst;
    }

    /** Full-array form: {@code dst[i] = Σ_{j≤i} s[j]}. */
    public static double[] cumSum(double[] dst, double[] s) {
        return cumSum(dst, 0, s, 0, s.length);
    }

    /**
     * Computes the cumulative product of {@code s[sOff..sOff+len-1]} into
     * {@code dst[dOff..dOff+len-1]}: {@code dst[dOff+i] = Π_{j≤i} s[sOff+j]}.
     *
     * @param dst  destination array (length ≥ dOff + len)
     * @param dOff offset into dst
     * @param s    source array
     * @param sOff offset into s
     * @param len  number of elements
     * @return {@code dst}
     */
    public static double[] cumProd(double[] dst, int dOff, double[] s, int sOff, int len) {
        if (len == 0) return dst;
        dst[dOff] = s[sOff];
        int i = 1;
        for (; i + 3 < len; i += 4) {
            double s0 = s[sOff+i], s1 = s[sOff+i+1], s2 = s[sOff+i+2], s3 = s[sOff+i+3];
            double d0 = dst[dOff+i-1] * s0;
            double d1 = d0 * s1;
            double d2 = d1 * s2;
            double d3 = d2 * s3;
            dst[dOff+i] = d0; dst[dOff+i+1] = d1; dst[dOff+i+2] = d2; dst[dOff+i+3] = d3;
        }
        for (; i < len; i++) dst[dOff+i] = dst[dOff+i-1] * s[sOff+i];
        return dst;
    }

    /** Full-array form: {@code dst[i] = Π_{j≤i} s[j]}. */
    public static double[] cumProd(double[] dst, double[] s) {
        return cumProd(dst, 0, s, 0, s.length);
    }
}

/**
 * SIMD accelerated slice kernels backing selected {@link VectorOps} routines.
 *
 * <p>Only methods without an equivalent BLAS routine live here — {@code sum},
 * {@code sumSq}, {@code l1Dist}, {@code l2Dist}. BLAS-routed reductions
 * (e.g. {@code dot}, {@code l1Norm}, {@code maxAbs}) already have tuned SIMD
 * paths inside {@code com.curioloop.yum4j.linalg.blas}.</p>
 *
 * <p>Species width is capped at 256-bit matching the BLAS gate, which avoids
 * AVX-512 frequency throttling on x86 while preserving NEON / AVX2 throughput.</p>
 *
 * <p>All kernels use two independent accumulators to hide FMA/ADD latency:
 * a single-chain accumulator bottlenecks at the pipeline depth of the floating
 * unit (3–5 cycles on most cores), while two independent chains let the core
 * keep both FP ports busy.</p>
 */
final class VectorOpsSIMD {

    private static final String SIMD_PROPERTY = "yum4j.vector";
    private static final boolean SIMD_ENABLED =
        !"false".equalsIgnoreCase(System.getProperty(SIMD_PROPERTY, "true"));

    static final VectorSpecies<Double> SPECIES = preferredSpecies256Cap();
    static final int LANES = SPECIES.length();

    /** Minimum length below which scalar loops win over SIMD setup cost. */
    static final int MIN_N = 32;

    static final boolean SUPPORTED = detect();

    private VectorOpsSIMD() {
    }

    private static VectorSpecies<Double> preferredSpecies256Cap() {
        VectorSpecies<Double> preferred = DoubleVector.SPECIES_PREFERRED;
        if (preferred.length() >= DoubleVector.SPECIES_256.length()) {
            return DoubleVector.SPECIES_256;
        }
        return preferred;
    }

    private static boolean detect() {
        if (!SIMD_ENABLED) return false;
        if (LANES <= 1) return false;
        try {
            double[] probe = new double[LANES * 2];
            for (int i = 0; i < probe.length; i++) probe[i] = i + 1;
            // Exercise every kernel so LinkageErrors surface at init.
            sum(probe, 0, probe.length);
            hasInf(probe, 0, probe.length);
            max(probe, 0, probe.length);
            min(probe, 0, probe.length);
            sumSq(probe, 0, probe.length, 1.0);
            l1Dist(probe, 0, probe, 0, probe.length);
            linfDist(probe, 0, probe, 0, probe.length);
            l2Dist(probe, 0, probe, 0, probe.length);
            axpyTo(probe, 0, 2.0, probe, 0, probe, 0, probe.length);
            return true;
        } catch (LinkageError e) {
            return false;
        }
    }

    // ── hasInf: any NaN or ±Infinity ─────────────────────────────────────────

    static boolean hasInf(double[] a, int off, int len) {
        DoubleVector finiteMax = DoubleVector.broadcast(SPECIES, Double.MAX_VALUE);
        int step2 = LANES * 2;
        int bound2 = (len / step2) * step2;
        int i = 0;
        for (; i < bound2; i += step2) {
            if (hasInf(DoubleVector.fromArray(SPECIES, a, off + i), finiteMax)
                    || hasInf(DoubleVector.fromArray(SPECIES, a, off + i + LANES), finiteMax)) {
                return true;
            }
        }
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += LANES) {
            if (hasInf(DoubleVector.fromArray(SPECIES, a, off + i), finiteMax)) return true;
        }
        for (; i < len; i++) {
            if (!Double.isFinite(a[off + i])) return true;
        }
        return false;
    }

    private static boolean hasInf(DoubleVector v, DoubleVector finiteMax) {
        return v.compare(VectorOperators.NE, v)
                .or(v.abs().compare(VectorOperators.GT, finiteMax))
                .anyTrue();
    }

    // ── sum: Σ xᵢ ─────────────────────────────────────────────────────────────

    static double sum(double[] a, int off, int len) {
        int step2 = LANES * 2;
        int bound2 = (len / step2) * step2;
        DoubleVector acc0 = DoubleVector.zero(SPECIES);
        DoubleVector acc1 = DoubleVector.zero(SPECIES);
        int i = 0;
        for (; i < bound2; i += step2) {
            acc0 = acc0.add(DoubleVector.fromArray(SPECIES, a, off + i));
            acc1 = acc1.add(DoubleVector.fromArray(SPECIES, a, off + i + LANES));
        }
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += LANES) {
            acc0 = acc0.add(DoubleVector.fromArray(SPECIES, a, off + i));
        }
        double s = acc0.add(acc1).reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) s += a[off + i];
        return s;
    }

    // ── max: max xᵢ ─────────────────────────────────────────────────────────────

    /**
     * SIMD max-reduction with two independent accumulators.
     * Uses {@code VectorOperators.MAX} which propagates NaN correctly.
     */
    static double max(double[] a, int off, int len) {
        int step2 = LANES * 2;
        int bound2 = (len / step2) * step2;
        DoubleVector max0 = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);
        DoubleVector max1 = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < bound2; i += step2) {
            max0 = max0.max(DoubleVector.fromArray(SPECIES, a, off + i));
            max1 = max1.max(DoubleVector.fromArray(SPECIES, a, off + i + LANES));
        }
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += LANES) {
            max0 = max0.max(DoubleVector.fromArray(SPECIES, a, off + i));
        }
        double m = max0.max(max1).reduceLanes(VectorOperators.MAX);
        for (; i < len; i++) m = Math.max(m, a[off + i]);
        return m;
    }

    // ── min: min xᵢ ─────────────────────────────────────────────────────────────

    /**
     * SIMD min-reduction with two independent accumulators.
     * Uses {@code VectorOperators.MIN} which propagates NaN correctly.
     */
    static double min(double[] a, int off, int len) {
        int step2 = LANES * 2;
        int bound2 = (len / step2) * step2;
        DoubleVector min0 = DoubleVector.broadcast(SPECIES, Double.POSITIVE_INFINITY);
        DoubleVector min1 = DoubleVector.broadcast(SPECIES, Double.POSITIVE_INFINITY);
        int i = 0;
        for (; i < bound2; i += step2) {
            min0 = min0.min(DoubleVector.fromArray(SPECIES, a, off + i));
            min1 = min1.min(DoubleVector.fromArray(SPECIES, a, off + i + LANES));
        }
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += LANES) {
            min0 = min0.min(DoubleVector.fromArray(SPECIES, a, off + i));
        }
        double m = min0.min(min1).reduceLanes(VectorOperators.MIN);
        for (; i < len; i++) m = Math.min(m, a[off + i]);
        return m;
    }

    // ── sumSq: Σ (xᵢ − delta)² ───────────────────────────────────────────────

    /**
     * Two-pass algorithm: single-pass {@code Σx² − (Σx)²/n} has catastrophic
     * cancellation when {@code |x̄| >> σ}, which is common for regression
     * residuals shifted to non-zero means. Two-pass keeps the numerical
     * behaviour of the reference loop while still gaining from SIMD on both
     * passes.
     */
    static double sumSq(double[] a, int off, int len, double delta) {
        DoubleVector deltaVec = DoubleVector.broadcast(SPECIES, delta);
        int step2 = LANES * 2;
        int bound2 = (len / step2) * step2;
        DoubleVector acc0 = DoubleVector.zero(SPECIES);
        DoubleVector acc1 = DoubleVector.zero(SPECIES);
        int i = 0;
        for (; i < bound2; i += step2) {
            DoubleVector d0 = DoubleVector.fromArray(SPECIES, a, off + i).sub(deltaVec);
            DoubleVector d1 = DoubleVector.fromArray(SPECIES, a, off + i + LANES).sub(deltaVec);
            acc0 = d0.fma(d0, acc0);
            acc1 = d1.fma(d1, acc1);
        }
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += LANES) {
            DoubleVector d = DoubleVector.fromArray(SPECIES, a, off + i).sub(deltaVec);
            acc0 = d.fma(d, acc0);
        }
        double ss = acc0.add(acc1).reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            double d = a[off + i] - delta;
            ss += d * d;
        }
        return ss;
    }

    // ── l1Dist: Σ |xᵢ − yᵢ| ───────────────────────────────────────────────────

    static double l1Dist(double[] x, int xOff, double[] y, int yOff, int len) {
        int step2 = LANES * 2;
        int bound2 = (len / step2) * step2;
        DoubleVector acc0 = DoubleVector.zero(SPECIES);
        DoubleVector acc1 = DoubleVector.zero(SPECIES);
        int i = 0;
        for (; i < bound2; i += step2) {
            DoubleVector d0 = DoubleVector.fromArray(SPECIES, x, xOff + i)
                    .sub(DoubleVector.fromArray(SPECIES, y, yOff + i)).abs();
            DoubleVector d1 = DoubleVector.fromArray(SPECIES, x, xOff + i + LANES)
                    .sub(DoubleVector.fromArray(SPECIES, y, yOff + i + LANES)).abs();
            acc0 = acc0.add(d0);
            acc1 = acc1.add(d1);
        }
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += LANES) {
            DoubleVector d = DoubleVector.fromArray(SPECIES, x, xOff + i)
                    .sub(DoubleVector.fromArray(SPECIES, y, yOff + i)).abs();
            acc0 = acc0.add(d);
        }
        double s = acc0.add(acc1).reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) s += Math.abs(x[xOff + i] - y[yOff + i]);
        return s;
    }

    // ── linfDist: max |xᵢ − yᵢ| ──────────────────────────────────────────────

    /**
     * SIMD max-reduction with two independent accumulators.
     * The branch {@code if (d > norm || isNaN)} in the scalar loop prevents JIT
     * auto-vectorisation; replacing it with {@code VectorOperators.MAX} (which
     * propagates NaN correctly on all supported ISAs) removes the dependency.
     */
    static double linfDist(double[] x, int xOff, double[] y, int yOff, int len) {
        if (len == 0) return 0.0;
        int step2 = LANES * 2;
        int bound2 = (len / step2) * step2;
        DoubleVector max0 = DoubleVector.zero(SPECIES);
        DoubleVector max1 = DoubleVector.zero(SPECIES);
        int i = 0;
        for (; i < bound2; i += step2) {
            max0 = max0.max(DoubleVector.fromArray(SPECIES, x, xOff + i)
                    .sub(DoubleVector.fromArray(SPECIES, y, yOff + i)).abs());
            max1 = max1.max(DoubleVector.fromArray(SPECIES, x, xOff + i + LANES)
                    .sub(DoubleVector.fromArray(SPECIES, y, yOff + i + LANES)).abs());
        }
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += LANES) {
            max0 = max0.max(DoubleVector.fromArray(SPECIES, x, xOff + i)
                    .sub(DoubleVector.fromArray(SPECIES, y, yOff + i)).abs());
        }
        double max = max0.max(max1).reduceLanes(VectorOperators.MAX);
        for (; i < len; i++) {
            double d = Math.abs(x[xOff + i] - y[yOff + i]);
            if (d > max) max = d;
        }
        return max;
    }

    // ── l2Dist: sqrt(Σ (xᵢ − yᵢ)²) with overflow-safe fallback ──────────────

    /**
     * Fast path: plain {@code sqrt(Σ d²)} computed in SIMD with two
     * accumulators. If the result is non-finite (overflow) or if the inputs
     * contain NaN, the caller falls back to the scaled Neumaier algorithm
     * ({@link VectorOps#l2Dist}).
     *
     * @return Euclidean distance, or {@link Double#NaN} if the fast path
     *         should not be trusted and the caller must fall back
     */
    static double l2DistFast(double[] x, int xOff, double[] y, int yOff, int len) {
        int step2 = LANES * 2;
        int bound2 = (len / step2) * step2;
        DoubleVector acc0 = DoubleVector.zero(SPECIES);
        DoubleVector acc1 = DoubleVector.zero(SPECIES);
        int i = 0;
        for (; i < bound2; i += step2) {
            DoubleVector d0 = DoubleVector.fromArray(SPECIES, x, xOff + i)
                    .sub(DoubleVector.fromArray(SPECIES, y, yOff + i));
            DoubleVector d1 = DoubleVector.fromArray(SPECIES, x, xOff + i + LANES)
                    .sub(DoubleVector.fromArray(SPECIES, y, yOff + i + LANES));
            acc0 = d0.fma(d0, acc0);
            acc1 = d1.fma(d1, acc1);
        }
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += LANES) {
            DoubleVector d = DoubleVector.fromArray(SPECIES, x, xOff + i)
                    .sub(DoubleVector.fromArray(SPECIES, y, yOff + i));
            acc0 = d.fma(d, acc0);
        }
        double ss = acc0.add(acc1).reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            double d = x[xOff + i] - y[yOff + i];
            ss += d * d;
        }
        if (!Double.isFinite(ss)) return Double.NaN; // signal fallback
        return Math.sqrt(ss);
    }

    /** Wrapper used only at init-time probe. */
    private static double l2Dist(double[] x, int xOff, double[] y, int yOff, int len) {
        return l2DistFast(x, xOff, y, yOff, len);
    }

    // ── axpyTo: dst = alpha*x + y ───────────────────────────────────────────

    static void axpyTo(double[] dst, int dOff, double alpha,
                       double[] x, int xOff, double[] y, int yOff, int len) {
        DoubleVector alphaVec = DoubleVector.broadcast(SPECIES, alpha);
        int step2 = LANES * 2;
        int bound2 = (len / step2) * step2;
        int i = 0;
        for (; i < bound2; i += step2) {
            DoubleVector x0 = DoubleVector.fromArray(SPECIES, x, xOff + i);
            DoubleVector y0 = DoubleVector.fromArray(SPECIES, y, yOff + i);
            x0.fma(alphaVec, y0).intoArray(dst, dOff + i);
            DoubleVector x1 = DoubleVector.fromArray(SPECIES, x, xOff + i + LANES);
            DoubleVector y1 = DoubleVector.fromArray(SPECIES, y, yOff + i + LANES);
            x1.fma(alphaVec, y1).intoArray(dst, dOff + i + LANES);
        }
        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += LANES) {
            DoubleVector x0 = DoubleVector.fromArray(SPECIES, x, xOff + i);
            DoubleVector y0 = DoubleVector.fromArray(SPECIES, y, yOff + i);
            x0.fma(alphaVec, y0).intoArray(dst, dOff + i);
        }
        for (; i < len; i++) dst[dOff + i] = alpha * x[xOff + i] + y[yOff + i];
    }
}
