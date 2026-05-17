package com.curioloop.yum4j.ssm.particle.resample;

import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

/**
 * Resampling kernels for the particle-filter engine.
 *
 * <p>Every kernel takes a {@code (double[] lw, int N, int M, int[] a,
 * RandomBatch rng, double[] scratch)} signature and writes ancestor
 * indices into {@code a[0..M-1]} in {@code [0, N)}. All kernels are
 * static and allocation-free on the hot path (apart from the
 * caller-supplied {@code scratch}).
 *
 * <h2>Two-pass normaliseCumSum</h2>
 *
 * <p>Replaces the old three-pass version with a two-pass algorithm:
 * <ol>
 *   <li>Pass 1: find {@code max(lw)} using SIMD max kernel.</li>
 *   <li>Pass 2: compute {@code exp(lw[i] - max)} with SIMD EXP,
 *       accumulate running cumsum, write to outCum, then normalise
 *       by dividing by {@code outCum[N-1]}.</li>
 * </ol>
 *
 * <h2>Scratch sizing (R7.4)</h2>
 *
 * <p>Every kernel now draws its uniform stream through a single batched
 * {@link RandomBatch#nextUniforms} call into a dedicated tail region of
 * {@code scratch}. The per-scheme minimum sizes are:
 *
 * <ul>
 *   <li>{@link #systematic}: {@code scratch.length >= N + 1}
 *       — {@code [0..N)} holds the normalised cumulative weights,
 *       {@code [N]} holds the single uniform draw {@code u0}.</li>
 *   <li>{@link #stratified}: {@code scratch.length >= N + M}
 *       — {@code [0..N)} holds the cumulative weights,
 *       {@code [N..N+M)} holds the {@code M} stratum uniforms.</li>
 *   <li>{@link #multinomial}: {@code scratch.length >= N + M + 1}
 *       — {@code [0..N)} holds the cumulative weights,
 *       {@code [N..N+M]} holds the {@code M+1} uniforms (in-place
 *       transformed to a running {@code cumsum(-log U)} for the sorted
 *       uniform trick).</li>
 *   <li>{@link #residual}: {@code scratch.length >= 2 * N}
 *       — {@code [0..N)} holds normalised weights then residuals then
 *       residual-cumulative, {@code [N..N+sres)} is reused for the
 *       residual multinomial's {@code cumsum(-log U)} partial sums
 *       (one uniform is drawn but consumed into a local scalar and not
 *       stored, so the layout fits within {@code 2 * N} for any valid
 *       {@code M <= N}).</li>
 *   <li>{@link #ssp}: {@code scratch.length >= 3 * N - 1} (conservatively
 *       {@code 3 * N}) — {@code [0..N)} holds nr_children,
 *       {@code [N..2N)} holds xi (fractional parts),
 *       {@code [2N..3N-1)} holds the {@code N-1} merge-loop uniforms.</li>
 *   <li>{@link #killing}: {@code scratch.length >= 2 * N}
 *       — {@code [0..N)} holds normalised weights then cumulative,
 *       {@code [N..2N)} first holds the {@code N} killing-decision
 *       uniforms, then the leading {@code nkilled+1} slots are reused
 *       for the residual multinomial's {@code cumsum(-log U)} partial
 *       sums after the decision phase consumes them.</li>
 * </ul>
 *
 * <p>A generic caller that does not know the scheme at build time should
 * allocate {@code scratch.length = 3 * Math.max(N, M)} to cover every
 * scheme (this is what {@link com.curioloop.yum4j.ssm.particle.engine.Workspace#allocate}
 * does).
 *
 */
public final class Resample {

    private Resample() {}

    // ------------------------------------------------------------------
    // SIMD configuration
    // ------------------------------------------------------------------

    private static final VectorSpecies<Double> SPECIES = preferredSpecies256Cap();
    private static final int LANES = SPECIES.length();

    private static VectorSpecies<Double> preferredSpecies256Cap() {
        VectorSpecies<Double> preferred = DoubleVector.SPECIES_PREFERRED;
        if (preferred.length() >= DoubleVector.SPECIES_256.length()) {
            return DoubleVector.SPECIES_256;
        }
        return preferred;
    }

    // ------------------------------------------------------------------
    // Public dispatcher
    // ------------------------------------------------------------------

    /**
     * Dispatches to the per-scheme kernel. The {@code scratch} array must
     * satisfy the per-scheme contract documented on the class Javadoc;
     * passing an under-sized scratch raises {@link IllegalArgumentException}.
     *
     * @param scheme  resampling scheme
     * @param lw      log-weight array (length >= N)
     * @param N       number of source particles
     * @param M       number of target particles (offspring count)
     * @param a       ancestor array to fill (length >= M)
     * @param rng     batched random number generator
     * @param scratch scratch buffer (sizing depends on scheme)
     */
    public static void apply(Scheme scheme, double[] lw, int N, int M,
                             int[] a, RandomBatch rng, double[] scratch) {
        validateScratch(scheme, N, M, scratch);
        switch (scheme) {
            case SYSTEMATIC  -> systematic(lw, N, M, a, rng, scratch);
            case STRATIFIED  -> stratified(lw, N, M, a, rng, scratch);
            case MULTINOMIAL -> multinomial(lw, N, M, a, rng, scratch);
            case RESIDUAL    -> residual(lw, N, M, a, rng, scratch);
            case SSP         -> ssp(lw, N, M, a, rng, scratch);
            case KILLING     -> killing(lw, N, M, a, rng, scratch);
        }
    }

    /**
     * Validates {@code scratch.length} against the per-scheme minimum
     * (R7.4). Called by {@link #apply}; package-private kernels skip
     * this check since every external entry point goes through the
     * dispatcher.
     */
    private static void validateScratch(Scheme scheme, int N, int M, double[] scratch) {
        int required = switch (scheme) {
            case SYSTEMATIC  -> N + 1;
            case STRATIFIED  -> N + M;
            case MULTINOMIAL -> N + M + 1;
            case RESIDUAL    -> 2 * N;
            case SSP         -> 3 * N - 1;
            case KILLING     -> 2 * N;
        };
        if (scratch.length < required) {
            throw new IllegalArgumentException(
                "Resample." + scheme + " requires scratch.length >= " + required
                    + " for N=" + N + ", M=" + M + "; got " + scratch.length);
        }
    }

    // ------------------------------------------------------------------
    // Two-pass normaliseCumSum (R6.1)
    // ------------------------------------------------------------------

    /**
     * Computes normalised cumulative weights from log-weights into
     * {@code outCum[0..N-1]}. On return {@code outCum[N-1] == 1.0}
     * (modulo rounding).
     *
     * <p>Two passes:
     * <ol>
     *   <li>Pass 1: find max using SIMD max kernel.</li>
     *   <li>Pass 2: SIMD exp of blocks, scalar cumsum accumulation,
     *       then normalise by dividing by final cumsum value.</li>
     * </ol>
     *
     * <p>If every weight is {@code -Infinity}, fills {@code outCum}
     * with {@code 1.0} so any subsequent inverse-CDF scan returns
     * index 0 for every target.
     *
     * @param lw     log-weight array (length >= N)
     * @param N      number of particles
     * @param outCum output cumulative array (length >= N)
     */
    static void normaliseCumSum(double[] lw, int N, double[] outCum) {
        if (N <= 0) return;

        // Pass 1: find max
        double m = simdMax(lw, 0, N);

        if (m == Double.NEGATIVE_INFINITY) {
            // Degenerate: all -Inf. Fill so a[m] = 0 for all m.
            Arrays.fill(outCum, 0, N, 1.0);
            return;
        }

        // Pass 2: SIMD exp + scalar cumsum + normalise
        double cum = 0.0;
        int i = 0;
        final int bound = SPECIES.loopBound(N);

        // Reusable lane-local buffer for extracting SIMD exp results
        // into the scalar cumsum. This is stack-allocated by the JIT
        // (escape analysis) since it doesn't escape this method.
        double[] block = new double[LANES];

        for (; i < bound; i += LANES) {
            DoubleVector expVec = DoubleVector.fromArray(SPECIES, lw, i)
                .sub(m)
                .lanewise(VectorOperators.EXP);
            expVec.intoArray(block, 0);
            for (int k = 0; k < LANES; k++) {
                cum += block[k];
                outCum[i + k] = cum;
            }
        }
        // Scalar tail
        for (; i < N; i++) {
            cum += Math.exp(lw[i] - m);
            outCum[i] = cum;
        }

        // Normalise: divide all entries by cum so outCum[N-1] == 1.0
        if (cum > 0.0) {
            double inv = 1.0 / cum;
            for (i = 0; i < N; i++) {
                outCum[i] *= inv;
            }
        }
    }

    /**
     * SIMD max over lw[off..off+n). Uses two-accumulator pattern.
     */
    private static double simdMax(double[] lw, int off, int n) {
        final int step2 = LANES * 2;
        final int bound2 = (n / step2) * step2;

        DoubleVector mx0 = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);
        DoubleVector mx1 = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < bound2; i += step2) {
            mx0 = mx0.max(DoubleVector.fromArray(SPECIES, lw, off + i));
            mx1 = mx1.max(DoubleVector.fromArray(SPECIES, lw, off + i + LANES));
        }
        int bound = SPECIES.loopBound(n);
        for (; i < bound; i += LANES) {
            mx0 = mx0.max(DoubleVector.fromArray(SPECIES, lw, off + i));
        }
        double m = mx0.max(mx1).reduceLanes(VectorOperators.MAX);
        for (; i < n; i++) {
            double v = lw[off + i];
            if (v > m) m = v;
        }
        return m;
    }

    // ------------------------------------------------------------------
    // Systematic resampling
    // ------------------------------------------------------------------

    /**
     * Systematic resampling. Draws a single {@code u ~ U(0, 1/M)} and
     * uses grid points {@code u + m/M} for {@code m in [0, M)}. Total
     * cost {@code O(N + M)}, one uniform draw.
     *
     * <p>Requires {@code scratch.length >= N + 1}. Uses
     * {@code scratch[0..N)} for normalised cumulative weights and
     * {@code scratch[N]} for the single batched uniform draw (R7.1).
     */
    static void systematic(double[] lw, int N, int M, int[] a,
                           RandomBatch rng, double[] scratch) {
        normaliseCumSum(lw, N, scratch);
        // Single uniform draw via batched API (R7.1).
        rng.nextUniforms(scratch, N, 1);
        double u0 = scratch[N] / M;
        double invM = 1.0 / M;
        int n = 0;
        for (int i = 0; i < M; i++) {
            double t = u0 + i * invM;
            while (n < N - 1 && scratch[n] < t) n++;
            a[i] = n;
        }
    }

    // ------------------------------------------------------------------
    // Stratified resampling
    // ------------------------------------------------------------------

    /**
     * Stratified resampling. For each {@code m}, draws an independent
     * {@code u_m ~ U(m/M, (m+1)/M)}. Total cost {@code O(N + M)},
     * {@code M} uniform draws.
     *
     * <p>Requires {@code scratch.length >= N + M}. Uses
     * {@code scratch[0..N)} for normalised cumulative weights and
     * {@code scratch[N..N+M)} for the {@code M} stratum uniforms drawn
     * via a single batched {@link RandomBatch#nextUniforms} call (R7.1).
     */
    static void stratified(double[] lw, int N, int M, int[] a,
                           RandomBatch rng, double[] scratch) {
        normaliseCumSum(lw, N, scratch);
        // Batch the M uniform draws into the reserved tail region (R7.1).
        rng.nextUniforms(scratch, N, M);
        double invM = 1.0 / M;
        int n = 0;
        for (int i = 0; i < M; i++) {
            double t = (i + scratch[N + i]) * invM;
            while (n < N - 1 && scratch[n] < t) n++;
            a[i] = n;
        }
    }

    // ------------------------------------------------------------------
    // Multinomial resampling
    // ------------------------------------------------------------------

    /**
     * Multinomial resampling via the {@code cumsum(-log U) / total}
     * trick for generating sorted uniforms (Devroye, 1986). Draws
     * {@code M + 1} uniforms in a single batch.
     *
     * <p>Requires {@code scratch.length >= N + M + 1}. Uses
     * {@code scratch[0..N)} for cumulative weights and
     * {@code scratch[N..N+M]} to first hold the {@code M+1} raw
     * uniforms (written by {@link RandomBatch#nextUniforms}) and then
     * their in-place {@code cumsum(-log U)} partial sums (R7.2).
     */
    static void multinomial(double[] lw, int N, int M, int[] a,
                            RandomBatch rng, double[] scratch) {
        normaliseCumSum(lw, N, scratch);
        // Single batched uniform draw for all M+1 values (R7.2).
        rng.nextUniforms(scratch, N, M + 1);
        // Transform in-place to running cumsum(-log U) partial sums.
        double cum = 0.0;
        for (int k = 0; k <= M; k++) {
            cum += -Math.log(scratch[N + k]);
            scratch[N + k] = cum;
        }
        double total = scratch[N + M];
        // Scan.
        int n = 0;
        for (int i = 0; i < M; i++) {
            double t = scratch[N + i] / total;
            while (n < N - 1 && scratch[n] < t) n++;
            a[i] = n;
        }
    }

    // ------------------------------------------------------------------
    // Residual resampling
    // ------------------------------------------------------------------

    /**
     * Residual resampling: deterministic {@code floor(M·W[n])}
     * replications, plus the remaining draws by multinomial on the
     * fractional residuals.
     *
     * <p>Requires {@code scratch.length >= 2 * N}. The first {@code N}
     * slots hold normalised weights (then residual-cumulative); the
     * second region holds a single batched {@code sres + 1} uniform
     * draw (where {@code sres <= M - 1} for {@code M = N} and
     * {@code sres <= M} for {@code M < N}, so the writes stay within
     * {@code 2 * N}). The uniforms are transformed in-place into the
     * residual multinomial's {@code cumsum(-log U)} partial sums (R7.3).
     */
    static void residual(double[] lw, int N, int M, int[] a,
                         RandomBatch rng, double[] scratch) {
        // Compute normalised weights W[n] into scratch[0..N-1].
        double m = simdMax(lw, 0, N);
        if (m == Double.NEGATIVE_INFINITY) {
            // All -Inf: degenerate; act as if W[0] = 1.
            scratch[0] = 1.0;
            for (int i = 1; i < N; i++) scratch[i] = 0.0;
        } else {
            double sum = 0.0;
            for (int i = 0; i < N; i++) {
                double w = Math.exp(lw[i] - m);
                scratch[i] = w;
                sum += w;
            }
            double inv = 1.0 / sum;
            for (int i = 0; i < N; i++) scratch[i] *= inv;
        }
        // Fill deterministic intpart: each particle repeated floor(M*W[n]) times.
        int sip = 0;
        for (int i = 0; i < N; i++) {
            double mw = M * scratch[i];
            int k = (int) Math.floor(mw);
            for (int c = 0; c < k && sip < M; c++) {
                a[sip++] = i;
            }
            // Replace scratch[i] with the residual (fractional part).
            scratch[i] = mw - k;
        }
        int sres = M - sip;
        if (sres <= 0) {
            return;
        }
        // Cumulative of residuals in place, using the same scratch[0..N-1].
        double cum = 0.0;
        for (int i = 0; i < N; i++) {
            cum += scratch[i];
            scratch[i] = cum;
        }
        if (!(cum > 0.0)) {
            for (int i = sip; i < M; i++) a[i] = 0;
            return;
        }
        double invCum = 1.0 / cum;
        for (int i = 0; i < N; i++) scratch[i] *= invCum;
        // Multinomial over residuals: one batched draw of sres+1 uniforms,
        // transformed in-place to cumsum(-log U). The last uniform is
        // consumed into a local scalar to keep the layout within 2*N
        // (R7.3).
        rng.nextUniforms(scratch, N, sres + 1);
        double cumLog = 0.0;
        for (int i = 0; i < sres; i++) {
            cumLog += -Math.log(scratch[N + i]);
            scratch[N + i] = cumLog;
        }
        cumLog += -Math.log(scratch[N + sres]);
        double totalLog = cumLog;
        int n = 0;
        for (int i = 0; i < sres; i++) {
            double t = scratch[N + i] / totalLog;
            while (n < N - 1 && scratch[n] < t) n++;
            a[sip + i] = n;
        }
    }

    // ------------------------------------------------------------------
    // SSP (Srinivasan Sampling Process)
    // ------------------------------------------------------------------

    /**
     * SSP resampling (Gerber-Chopin-Whiteley 2019). Number of
     * off-springs of particle {@code n} is either {@code k} or
     * {@code k + 1} where {@code k ≤ M·W[n] < k + 1}. Consistent
     * and negatively associated.
     *
     * <p>Requires {@code scratch.length >= 3 * N - 1} (conservatively
     * {@code 3 * N}) and typically {@code M == N}. Output {@code a[]}
     * contains {@code M} ancestor indices in ascending order.
     *
     * <p>The first {@code N} scratch slots hold {@code nr_children};
     * the next {@code N} slots hold {@code xi} (fractional parts);
     * {@code scratch[2N..3N-1)} holds the {@code N-1} merge-loop
     * uniforms drawn in a single batched {@link RandomBatch#nextUniforms}
     * call. Consumes {@code N - 1} uniforms for {@code N >= 2}
     * (zero when {@code N == 1}).
     */
    static void ssp(double[] lw, int N, int M, int[] a,
                    RandomBatch rng, double[] scratch) {
        if (N <= 0) return;
        // Weight pre-processing: MW = M * W, stored in scratch[0..N-1].
        double mx = simdMax(lw, 0, N);
        if (mx == Double.NEGATIVE_INFINITY) {
            // Degenerate: all -Inf. Collapse to index 0.
            for (int i = 0; i < M; i++) a[i] = 0;
            return;
        }
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            double w = Math.exp(lw[i] - mx);
            scratch[i] = w;
            sum += w;
        }
        double invS = 1.0 / sum;
        // scratch[0..N-1] = MW.
        for (int i = 0; i < N; i++) scratch[i] = scratch[i] * invS * M;
        // Split into nr_children (scratch[0..N-1]) and xi (scratch[N..2N-1]).
        int total = 0;
        for (int i = 0; i < N; i++) {
            double mw = scratch[i];
            double nci = Math.floor(mw);
            scratch[i] = nci;
            scratch[N + i] = mw - nci;
            total += (int) nci;
        }
        if (N == 1) {
            // Single-particle shortcut: MW = M; nr_children[0] = M; xi = 0.
            // No uniforms consumed (matches pre-batching RNG order).
        } else {
            // Batch all N-1 merge-loop uniforms into scratch[2N..3N-1) (R7.x).
            rng.nextUniforms(scratch, 2 * N, N - 1);
            // SSP merge loop.
            int I = 0, J = 1;
            int lastK = N - 2;
            for (int k = 0; k < N - 1; k++) {
                double xii = scratch[N + I];
                double xij = scratch[N + J];
                double deltaI = Math.min(xij, 1.0 - xii);
                double deltaJ = Math.min(xii, 1.0 - xij);
                double sumDelta = deltaI + deltaJ;
                double pj = sumDelta > 0.0 ? deltaI / sumDelta : 0.0;
                double u = scratch[2 * N + k];
                if (u < pj) {
                    int tmp = I; I = J; J = tmp;
                    deltaI = deltaJ;
                    xii = scratch[N + I];
                    xij = scratch[N + J];
                }
                if (xij < 1.0 - xii) {
                    scratch[N + I] = xii + deltaI;
                    J = k + 2;
                } else {
                    scratch[N + J] = xij - deltaI;
                    scratch[I] += 1.0; // nr_children[I]++
                    total++;
                    I = k + 2;
                }
            }
            // Round-off rescue: patch a missing off-spring when
            // accumulated round-off drops the total by exactly one.
            if (total == M - 1) {
                int lastIJ = (J == lastK + 2) ? I : J;
                if (scratch[N + lastIJ] > 0.99) {
                    scratch[lastIJ] += 1.0;
                    total++;
                }
            }
        }
        if (total != M) {
            throw new IllegalStateException(
                "SSP resampling: wrong output size, total=" + total + " M=" + M);
        }
        // Fill a[] with np.arange(N).repeat(nr_children).
        int pos = 0;
        for (int i = 0; i < N; i++) {
            int count = (int) scratch[i];
            for (int c = 0; c < count; c++) {
                a[pos++] = i;
            }
        }
    }

    // ------------------------------------------------------------------
    // Killing resampling
    // ------------------------------------------------------------------

    /**
     * Killing resampling. Each particle {@code i} is kept with
     * probability {@code W[i] / max(W)}; otherwise it is "killed" and
     * replaced by a multinomial draw from the full weight distribution.
     *
     * <p>This scheme is only defined for {@code M == N}; the kernel
     * raises {@link IllegalArgumentException} otherwise.
     *
     * <p>Requires {@code scratch.length >= 2 * N}. Uses
     * {@code scratch[0..N)} for normalised weights (then cumulative)
     * and {@code scratch[N..2N)} first for the {@code N} killing-decision
     * uniforms (a single batched {@link RandomBatch#nextUniforms} call),
     * then — after the decision phase has consumed them — the leading
     * {@code nkilled + 1} slots are reused to hold a second batched draw
     * of uniforms transformed in-place into the residual multinomial's
     * {@code cumsum(-log U)} partial sums (R7.3).
     */
    static void killing(double[] lw, int N, int M, int[] a,
                        RandomBatch rng, double[] scratch) {
        if (M != N) {
            throw new IllegalArgumentException(
                "killing resampling requires M == N; got M=" + M + " N=" + N);
        }
        if (N <= 0) return;
        // Max-shift normalisation; compute W, max(W), sum.
        double mx = simdMax(lw, 0, N);
        if (mx == Double.NEGATIVE_INFINITY) {
            for (int i = 0; i < M; i++) a[i] = 0;
            return;
        }
        double sum = 0.0;
        double rawMax = 0.0;
        for (int i = 0; i < N; i++) {
            double w = Math.exp(lw[i] - mx);
            scratch[i] = w;
            if (w > rawMax) rawMax = w;
            sum += w;
        }
        double invS = 1.0 / sum;
        for (int i = 0; i < N; i++) scratch[i] *= invS;
        double maxProb = rawMax * invS;

        // Batch N uniforms for the killing decisions into scratch[N..2N)
        // (R7.3). These are fully consumed by the decision loop before
        // the residual multinomial overwrites the region.
        rng.nextUniforms(scratch, N, N);
        int nkilled = 0;
        for (int i = 0; i < N; i++) {
            a[i] = i;
            double u = scratch[N + i];
            if (u * maxProb >= scratch[i]) {
                a[i] = -1;
                nkilled++;
            }
        }
        if (nkilled == 0) {
            return;
        }
        // Cumulative W in scratch[0..N-1].
        double cum = 0.0;
        for (int i = 0; i < N; i++) {
            cum += scratch[i];
            scratch[i] = cum;
        }
        // Multinomial over W for nkilled draws via the
        // cumsum(-log U)/total sorted-uniform trick. Overwrites the
        // leading nkilled+1 slots of the killing-uniform region with
        // fresh uniforms; the killing uniforms there have already been
        // consumed.
        rng.nextUniforms(scratch, N, nkilled + 1);
        double cumLog = 0.0;
        for (int j = 0; j <= nkilled; j++) {
            cumLog += -Math.log(scratch[N + j]);
            scratch[N + j] = cumLog;
        }
        double totalLog = scratch[N + nkilled];
        int n = 0;
        int mKilled = 0;
        for (int i = 0; i < N; i++) {
            if (a[i] == -1) {
                double t = scratch[N + mKilled] / totalLog;
                while (n < N - 1 && scratch[n] < t) n++;
                a[i] = n;
                mKilled++;
            }
        }
    }
}
