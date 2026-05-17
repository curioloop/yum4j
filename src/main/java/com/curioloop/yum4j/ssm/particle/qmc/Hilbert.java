package com.curioloop.yum4j.ssm.particle.qmc;

import com.curioloop.yum4j.math.VectorOps;

/**
 * Hilbert space-filling-curve foundations for SQMC.
 *
 * <p>Ports Steve Witham's bit-manipulation Hilbert codec (the same one
 * used by {@code reference/particles/particles/hilbert.py}) into
 * allocation-conscious Java.  The public surface is:
 *
 * <ul>
 *   <li>{@link #invLogit(double)} — logistic transform.</li>
 *   <li>{@link #hilbertIndex(long[], int)} — per-particle Hilbert index
 *       for caller-supplied integer coordinates.</li>
 *   <li>{@link #hilbertSort(double[], int, int, int, int[], double[], long[])}
 *       — standardise, invlogit, integerise, then sort particles by
 *       their Hilbert index.</li>
 * </ul>
 *
 * <p>The reference uses {@code maxint = floor(2^(62/d))} so that every
 * Hilbert index fits in a signed 64-bit integer.  We do the same, and
 * all internal arithmetic is {@code long}; for the supported range
 * {@code d ∈ [1, 32]} no intermediate multiplication overflows.  {@link
 * #hilbertIndex(long[], int)} short-circuits to {@code coords[0]} when
 * {@code d == 1}.
 */
public final class Hilbert {

    private Hilbert() {}

    // ------------------------------------------------------------------
    // Scalar helpers
    // ------------------------------------------------------------------

    /** Logistic transform {@code 1 / (1 + exp(-x))}. Reference: {@code hilbert.py::invlogit}. */
    public static double invLogit(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // ------------------------------------------------------------------
    // Internal Gray-code / chunk helpers (all long-valued, non-negative)
    // ------------------------------------------------------------------

    private static long grayEncode(long bn) {
        return bn ^ (bn >>> 1);
    }

    private static long grayDecode(long n) {
        int sh = 1;
        while (true) {
            long div = n >>> sh;
            n ^= div;
            if (div <= 1L) return n;
            sh <<= 1;
        }
    }

    /**
     * Modified Gray encoder with bit-rotation, as per
     * {@code hilbert.py::gray_encode_travel}.
     *
     * <pre>
     *     gray_encode_travel(start, end, mask, 0)    == start
     *     gray_encode_travel(start, end, mask, mask) == end
     * </pre>
     */
    private static long grayEncodeTravel(long start, long end, long mask, long i) {
        long travelBit = start ^ end;
        long modulus = mask + 1L;              // == 2**nBits
        long g = grayEncode(i) * (travelBit * 2L);
        return ((g | (g / modulus)) & mask) ^ start;
    }

    /** Inverse of {@link #grayEncodeTravel}. */
    private static long grayDecodeTravel(long start, long end, long mask, long g) {
        long travelBit = start ^ end;
        long modulus = mask + 1L;
        long rg = (g ^ start) * (modulus / (travelBit * 2L));
        return grayDecode((rg | (rg / modulus)) & mask);
    }

    /**
     * Transpose bits: given {@code nSrcs} source longs (each {@code nDests}
     * bits wide) produce {@code nDests} output longs (each {@code nSrcs}
     * bits wide).  Earlier {@code srcs} become higher bits in dests; earlier
     * dests come from higher bits of srcs.  {@code srcs} is consumed
     * (right-shifted to zero) during the operation.
     */
    private static void transposeBitsInto(long[] srcs, int nSrcs, long[] dests, int nDests) {
        for (int j = nDests - 1; j >= 0; j--) {
            long dest = 0L;
            for (int k = 0; k < nSrcs; k++) {
                dest = (dest << 1) | (srcs[k] & 1L);
                srcs[k] >>>= 1;
            }
            dests[j] = dest;
        }
    }

    // ------------------------------------------------------------------
    // Hilbert index: one particle
    // ------------------------------------------------------------------

    /**
     * Compute the Hilbert curve index of the given {@code d}-dimensional
     * non-negative integer coordinates.
     *
     * <p>For {@code d == 1} this returns {@code coords[0]} (the curve
     * degenerates to the identity). For {@code d ≥ 2} it mirrors
     * {@code hilbert.py::Hilbert_to_int} step-by-step.
     *
     * <p>The returned value fits in a signed {@code long} provided
     * {@code max(coords) < 2^(63/d)}; the companion {@link #hilbertSort}
     * keeps coordinates below {@code 2^(62/d)} so this is always the case
     * in the standard pipeline.
     */
    public static long hilbertIndex(long[] coords, int d) {
        if (d == 1) return coords[0];
        long biggest = 0L;
        for (int i = 0; i < d; i++) {
            long c = coords[i];
            if (c > biggest) biggest = c;
        }
        int nChunks = (biggest == 0L) ? 1 : (64 - Long.numberOfLeadingZeros(biggest));
        long[] srcs = new long[d];
        long[] coordChunks = new long[nChunks];
        return hilbertIndexCore(coords, d, srcs, coordChunks, nChunks);
    }

    /**
     * Core Hilbert-index computation given caller-supplied scratch.
     * {@code srcsScratch.length >= d}, {@code chunksScratch.length >= 63}
     * (the maximum possible {@code nChunks} for a 62-bit Hilbert cube).
     * {@code nChunks} is {@code max(1, ceil(log2(max(coords) + 1)))}.
     */
    private static long hilbertIndexCore(long[] coords, int d,
                                         long[] srcsScratch,
                                         long[] chunksScratch,
                                         int nChunks) {
        // transposeBitsInto consumes srcsScratch; make a copy.
        for (int i = 0; i < d; i++) srcsScratch[i] = coords[i];
        transposeBitsInto(srcsScratch, d, chunksScratch, nChunks);

        long mask = (d >= 64) ? -1L : ((1L << d) - 1L);
        int shift = Math.floorMod(-nChunks - 1, d);
        long start = 0L;
        long end = 1L << shift;

        long z = 0L;
        long p = 1L << d;
        for (int j = 0; j < nChunks; j++) {
            long i = grayDecodeTravel(start, end, mask, chunksScratch[j]);
            z = p * z + i;
            long startI = Math.max(0L, (i - 1L) & ~1L);
            long endI = Math.min(mask, (i + 1L) | 1L);
            long newStart = grayEncodeTravel(start, end, mask, startI);
            long newEnd = grayEncodeTravel(start, end, mask, endI);
            start = newStart;
            end = newEnd;
        }
        return z;
    }

    // ------------------------------------------------------------------
    // Hilbert sort
    // ------------------------------------------------------------------

    /**
     * Sort particles by their Hilbert index.
     *
     * <p>Mirrors {@code hilbert.py::hilbert_sort}:
     * <ol>
     *   <li>{@code d == 1}: plain ascending {@code argsort} of
     *       {@code X[xOff..xOff+N)}.</li>
     *   <li>{@code d ≥ 2}:
     *     <ol type="a">
     *       <li>compute per-dimension population mean and std;</li>
     *       <li>standardise and apply {@link #invLogit}, writing
     *           {@code d·N} doubles into {@code scratchDoubles};</li>
     *       <li>multiply by {@code maxint = floor(2^(62/d))} and floor
     *           to integers, writing {@code d·N} longs into
     *           {@code scratchLongs};</li>
     *       <li>for each particle {@code n}, compute
     *           {@code h[n] = hilbertIndex(xint column n)} and sort;
     *           the resulting permutation is written to
     *           {@code permOut[0..N)}.</li>
     *     </ol>
     *   </li>
     * </ol>
     *
     * <p>Layout: {@code X} is column-major {@code (d, N)} with stride
     * {@code N} — element {@code X[j, n]} lives at {@code X[xOff + j·N + n]}.
     *
     * <p>Scratch:
     * <ul>
     *   <li>{@code permOut.length >= N}</li>
     *   <li>{@code scratchDoubles.length >= d·N}</li>
     *   <li>{@code scratchLongs.length >= d·N}</li>
     * </ul>
     * After return, {@code scratchLongs[0..N)} holds {@code h[n]} (the
     * per-particle Hilbert index keyed by particle index, not sorted);
     * the remaining {@code scratchLongs[N..d·N)} still hold xint rows
     * {@code 1..d-1}.
     */
    public static void hilbertSort(
        double[] X, int xOff,
        int d, int N,
        int[] permOut,
        double[] scratchDoubles,
        long[] scratchLongs
    ) {
        if (N <= 0) return;
        if (d == 1) {
            // Plain ascending argsort of X[xOff..xOff+N).
            // Stage: permOut[i] = i; heap-sort by X[xOff + permOut[i]].
            for (int i = 0; i < N; i++) permOut[i] = i;
            heapSortIndirectDouble(permOut, N, X, xOff);
            return;
        }

        // 1. Per-dimension population mean and std (ddof = 0, matching np.std).
        for (int j = 0; j < d; j++) {
            int base = xOff + j * N;
            double mu = VectorOps.mean(X, base, N);
            double ss = VectorOps.sumSq(X, base, N, mu);
            double sigma = Math.sqrt(ss / N);
            int dst = j * N;
            if (sigma == 0.0) {
                // Degenerate column: invlogit(0) = 0.5.
                for (int n = 0; n < N; n++) scratchDoubles[dst + n] = 0.5;
            } else {
                double invSigma = 1.0 / sigma;
                for (int n = 0; n < N; n++) {
                    double z = (X[base + n] - mu) * invSigma;
                    scratchDoubles[dst + n] = invLogit(z);
                }
            }
        }

        // 2. xint[j, n] = floor(xs[j, n] * maxint).
        long maxint = (long) Math.floor(Math.pow(2.0, 62.0 / d));
        for (int j = 0; j < d; j++) {
            int base = j * N;
            for (int n = 0; n < N; n++) {
                scratchLongs[base + n] = (long) Math.floor(scratchDoubles[base + n] * maxint);
            }
        }

        // 3. Per-particle Hilbert index. We compute h[n] after reading
        //    column n of xint, then overwrite the first-row slot
        //    (scratchLongs[n]) with h[n]. The overwrite is safe because
        //    future iterations read columns m > n, which live in slots
        //    scratchLongs[j*N + m] with m > n; writing slot n (= 0*N + n)
        //    does not touch any of them.
        //
        //    hilbertIndex needs two small scratch buffers (size d and
        //    size nChunks); we allocate them once for the whole sort,
        //    then reuse across particles. nChunks is per-particle
        //    (depends on max(coords)), so we compute it fresh each
        //    iteration, but the chunk buffer itself is bounded by
        //    ceil(log2(maxint + 1)) ≤ 62/d + 1 < 64 and reused.
        long[] coordBuf = new long[d];
        long[] srcsBuf = new long[d];
        long[] chunksBuf = new long[64];
        for (int n = 0; n < N; n++) {
            long biggest = 0L;
            for (int j = 0; j < d; j++) {
                long c = scratchLongs[j * N + n];
                coordBuf[j] = c;
                if (c > biggest) biggest = c;
            }
            int nChunks = (biggest == 0L) ? 1 : (64 - Long.numberOfLeadingZeros(biggest));
            scratchLongs[n] = hilbertIndexCore(coordBuf, d, srcsBuf, chunksBuf, nChunks);
        }

        // 4. Indirect heap-sort: permOut[i] = i; sort keyed by
        //    scratchLongs[permOut[i]]. No allocation.
        for (int i = 0; i < N; i++) permOut[i] = i;
        heapSortIndirectLong(permOut, N, scratchLongs);
    }

    // ------------------------------------------------------------------
    // Allocation-free indirect heap-sort helpers
    // ------------------------------------------------------------------

    private static void heapSortIndirectLong(int[] idx, int n, long[] keys) {
        for (int i = (n >>> 1) - 1; i >= 0; i--) siftDownLong(idx, n, i, keys);
        for (int i = n - 1; i > 0; i--) {
            int t = idx[0]; idx[0] = idx[i]; idx[i] = t;
            siftDownLong(idx, i, 0, keys);
        }
    }

    private static void siftDownLong(int[] idx, int heapSize, int start, long[] keys) {
        int root = start;
        while (true) {
            int left = 2 * root + 1;
            if (left >= heapSize) return;
            int right = left + 1;
            int child = (right < heapSize && keys[idx[right]] > keys[idx[left]]) ? right : left;
            if (keys[idx[child]] > keys[idx[root]]) {
                int t = idx[root]; idx[root] = idx[child]; idx[child] = t;
                root = child;
            } else {
                return;
            }
        }
    }

    private static void heapSortIndirectDouble(int[] idx, int n, double[] keys, int keyOff) {
        for (int i = (n >>> 1) - 1; i >= 0; i--) siftDownDouble(idx, n, i, keys, keyOff);
        for (int i = n - 1; i > 0; i--) {
            int t = idx[0]; idx[0] = idx[i]; idx[i] = t;
            siftDownDouble(idx, i, 0, keys, keyOff);
        }
    }

    private static void siftDownDouble(int[] idx, int heapSize, int start, double[] keys, int keyOff) {
        int root = start;
        while (true) {
            int left = 2 * root + 1;
            if (left >= heapSize) return;
            int right = left + 1;
            int child = (right < heapSize && keys[keyOff + idx[right]] > keys[keyOff + idx[left]]) ? right : left;
            if (keys[keyOff + idx[child]] > keys[keyOff + idx[root]]) {
                int t = idx[root]; idx[root] = idx[child]; idx[child] = t;
                root = child;
            } else {
                return;
            }
        }
    }
}
