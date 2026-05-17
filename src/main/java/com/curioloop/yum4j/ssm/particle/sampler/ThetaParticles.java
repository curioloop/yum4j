package com.curioloop.yum4j.ssm.particle.sampler;

/**
 * Parameter-space particle container for SMC samplers.
 *
 * <p>Stores N particles of dimension d_θ in a single contiguous arena:
 * <pre>
 *   arena: [theta: dim*N | lpost: N | llik: N | lprior: N]
 *   total length: (dim + 3) * N
 * </pre>
 *
 * <p>The theta region is column-major {@code (dim, N)}: element
 * {@code theta[j, n]} lives at {@code arena[j * N + n]}.
 * Scalar per-particle regions ({@code lpost}, {@code llik}, {@code lprior})
 * each have length N and are accessed via offset accessors.
 *
 * <h3>Arena aliasing</h3>
 * <p>The {@link #arena} array may either be owned by this container
 * (allocated via {@link #allocate(int, int)}) or aliased to an external
 * buffer (via {@link #attachArena(double[], int)}). Aliasing is used by
 * {@code FKSMCSampler} subclasses to share storage with
 * {@code Workspace.X} so the FK sampler never round-trips data between
 * its side-car and the engine's main particle buffer.
 *
 * <h3>Strongly-typed shared state</h3>
 * <p>Per-sampler metadata (tempering exponents, path-sampling log-Z,
 * nested-sampling thresholds, MCMC acceptance rates) is exposed via
 * explicit fields rather than a string-keyed map. This avoids HashMap
 * rehashing and boxing on the hot path. Fields default to {@code null}
 * or {@code 0} and are populated by the consuming sampler.
 *
 * <p>Supports fancy indexing (resampling via {@link #gather}), waste-free
 * concatenation (via {@link #concatenateInto}), masked copy
 * ({@link #copyFrom}), and single-particle copy ({@link #copyAt}).
 */
public final class ThetaParticles {

    /** Number of particles. */
    public final int N;

    /** Parameter dimension d_θ. */
    public final int dim;

    /**
     * Arena: [theta: dim*N | lpost: N | llik: N | lprior: N].
     * Not final so {@link #attachArena} can retarget it to an external
     * buffer. Once attached, the container does not own the backing
     * array.
     */
    public double[] arena;

    // ------------------------------------------------------------------
    // Strongly-typed shared state (replaces the former shared HashMap).
    // ------------------------------------------------------------------

    /**
     * Tempering path-sampling log-Z estimate.
     * Populated by {@link Tempering}.
     */
    public double pathSamplingLogZ;

    /**
     * Tempering exponent schedule visited so far.
     * Length {@link #exponentsLen} is the number of valid entries; the
     * backing array may be larger. {@code null} when unused.
     */
    public double[] exponents;

    /** Number of valid entries in {@link #exponents}. */
    public int exponentsLen;

    /**
     * Nested-sampling threshold schedule. {@code null} when unused.
     */
    public double[] lts;

    /** Number of valid entries in {@link #lts}. */
    public int ltsLen;

    /**
     * Nested-sampling running log-evidence estimate. {@code null} when
     * unused.
     */
    public double[] logEvid;

    /** Number of valid entries in {@link #logEvid}. */
    public int logEvidLen;

    /**
     * Flat MCMC acceptance-rate log. Organised as a ragged 2D array
     * packed into a single {@code double[]}. Iteration {@code k}
     * occupies indices
     * {@code accRatesOff[k]..accRatesOff[k] + accRatesLen[k])}.
     * {@code null} when unused.
     */
    public double[] accRates;

    /** Offsets into {@link #accRates}, one per recorded iteration. */
    public int[] accRatesOff;

    /** Lengths of each recorded iteration in {@link #accRates}. */
    public int[] accRatesLen;

    /** Number of recorded iterations in {@link #accRates}. */
    public int accRatesCount;

    private ThetaParticles(int dim, int N) {
        this.dim = dim;
        this.N = N;
        this.arena = new double[(dim + 3) * N];
    }

    /** Offset of the theta region (always 0). */
    public int thetaOff() { return 0; }

    /** Offset of the lpost region. */
    public int lpostOff() { return dim * N; }

    /** Offset of the llik region. */
    public int llikOff() { return (dim + 1) * N; }

    /** Offset of the lprior region. */
    public int lpriorOff() { return (dim + 2) * N; }

    /** Total length of the arena. */
    public int arenaLength() { return (dim + 3) * N; }

    /**
     * Allocates a new ThetaParticles container with zeroed arena.
     *
     * @param dim parameter dimension d_θ
     * @param N   number of particles
     * @return a freshly allocated container
     */
    public static ThetaParticles allocate(int dim, int N) {
        return new ThetaParticles(dim, N);
    }

    /**
     * Redirect {@link #arena} to an externally-owned buffer. The backing
     * array must have length at least {@code dim*N + 3*N} and the same
     * layout as this container (column-major theta followed by
     * per-particle scalars). The caller owns the buffer; this container
     * will neither extend it nor free the previously-owned arena (the
     * previously-owned arena becomes garbage if no other reference
     * exists).
     *
     * @param externalArena external buffer to alias
     * @param off           offset within {@code externalArena} at which
     *                      this container's theta region begins
     */
    public void attachArena(double[] externalArena, int off) {
        if (externalArena == null) {
            throw new IllegalArgumentException("externalArena must not be null");
        }
        if (externalArena.length - off < arenaLength()) {
            throw new IllegalArgumentException(
                "externalArena too small: need " + arenaLength() + " from offset "
                    + off + ", got " + (externalArena.length - off));
        }
        if (off != 0) {
            throw new IllegalArgumentException(
                "attachArena currently requires off == 0 (offsets within the "
                    + "backing array are encoded by this container's lpostOff/llikOff/lpriorOff)");
        }
        this.arena = externalArena;
    }

    /**
     * Record one iteration of acceptance rates into the flat
     * {@link #accRates} buffer. Grows the backing arrays as needed.
     */
    public void appendAccRates(double[] rates, int len) {
        int off;
        if (accRates == null) {
            accRates = new double[Math.max(16, len * 2)];
            accRatesOff = new int[8];
            accRatesLen = new int[8];
            off = 0;
        } else {
            off = (accRatesCount == 0)
                ? 0
                : accRatesOff[accRatesCount - 1] + accRatesLen[accRatesCount - 1];
            if (off + len > accRates.length) {
                double[] bigger = new double[Math.max(off + len, accRates.length * 2)];
                System.arraycopy(accRates, 0, bigger, 0, off);
                accRates = bigger;
            }
        }
        if (accRatesCount == accRatesOff.length) {
            int grow = accRatesOff.length * 2;
            int[] biggerOff = new int[grow];
            int[] biggerLen = new int[grow];
            System.arraycopy(accRatesOff, 0, biggerOff, 0, accRatesCount);
            System.arraycopy(accRatesLen, 0, biggerLen, 0, accRatesCount);
            accRatesOff = biggerOff;
            accRatesLen = biggerLen;
        }
        accRatesOff[accRatesCount] = off;
        accRatesLen[accRatesCount] = len;
        System.arraycopy(rates, 0, accRates, off, len);
        accRatesCount++;
    }

    /**
     * Append one value to {@link #exponents}, growing the backing
     * array as needed.
     */
    public void appendExponent(double v) {
        if (exponents == null) {
            exponents = new double[16];
        } else if (exponentsLen == exponents.length) {
            double[] bigger = new double[exponents.length * 2];
            System.arraycopy(exponents, 0, bigger, 0, exponentsLen);
            exponents = bigger;
        }
        exponents[exponentsLen++] = v;
    }

    /** Append one value to {@link #lts}. */
    public void appendLt(double v) {
        if (lts == null) {
            lts = new double[16];
        } else if (ltsLen == lts.length) {
            double[] bigger = new double[lts.length * 2];
            System.arraycopy(lts, 0, bigger, 0, ltsLen);
            lts = bigger;
        }
        lts[ltsLen++] = v;
    }

    /** Append one value to {@link #logEvid}. */
    public void appendLogEvid(double v) {
        if (logEvid == null) {
            logEvid = new double[16];
        } else if (logEvidLen == logEvid.length) {
            double[] bigger = new double[logEvid.length * 2];
            System.arraycopy(logEvid, 0, bigger, 0, logEvidLen);
            logEvid = bigger;
        }
        logEvid[logEvidLen++] = v;
    }

    /**
     * Copy the strongly-typed shared state from {@code src} into this
     * container. The backing arrays are shared (not cloned) because the
     * consumer is the sampler which manages their lifetime.
     */
    public void copySharedFrom(ThetaParticles src) {
        this.pathSamplingLogZ = src.pathSamplingLogZ;
        this.exponents = src.exponents;
        this.exponentsLen = src.exponentsLen;
        this.lts = src.lts;
        this.ltsLen = src.ltsLen;
        this.logEvid = src.logEvid;
        this.logEvidLen = src.logEvidLen;
        this.accRates = src.accRates;
        this.accRatesOff = src.accRatesOff;
        this.accRatesLen = src.accRatesLen;
        this.accRatesCount = src.accRatesCount;
    }

    /**
     * Fancy indexing (resampling): returns a new instance containing the M
     * selected particles with independent storage (no aliasing).
     *
     * <p>Uses {@link Arrays2D#gatherCols} for the theta buffer (column-major
     * gather) and simple indexed copy for scalar arrays (lpost, llik, lprior).
     *
     * <p><b>Note:</b> This method allocates a new ThetaParticles on every call.
     * For zero-allocation resampling in hot paths, use
     * {@link #gatherInto(ThetaParticles, int[], int)} instead.
     *
     * @param ancestors ancestor index vector of length ≥ M; each entry in [0, N)
     * @param M         number of particles to gather
     * @return a new ThetaParticles of size M
     */
    public ThetaParticles gather(int[] ancestors, int M) {
        ThetaParticles out = new ThetaParticles(dim, M);
        gatherInto(out, ancestors, M);
        return out;
    }

    /**
     * Zero-allocation fancy indexing: writes the gathered particles into a
     * pre-allocated target. The target must have {@code target.N >= M} and
     * {@code target.dim == this.dim}.
     *
     * @param target    pre-allocated destination (not null)
     * @param ancestors ancestor index vector of length ≥ M; each entry in [0, N)
     * @param M         number of particles to gather
     */
    public void gatherInto(ThetaParticles target, int[] ancestors, int M) {
        // Column-gather for column-major (dim, N) layout:
        // target.arena[j*M + n] = arena[j*N + ancestors[n]]
        for (int j = 0; j < dim; j++) {
            int rowIn = j * N;
            int rowOut = j * M;
            for (int n = 0; n < M; n++) {
                target.arena[rowOut + n] = arena[rowIn + ancestors[n]];
            }
        }
        int srcLpost = lpostOff();
        int srcLlik = llikOff();
        int srcLprior = lpriorOff();
        int dstLpost = target.lpostOff();
        int dstLlik = target.llikOff();
        int dstLprior = target.lpriorOff();
        for (int n = 0; n < M; n++) {
            target.arena[dstLpost + n] = arena[srcLpost + ancestors[n]];
            target.arena[dstLlik + n] = arena[srcLlik + ancestors[n]];
            target.arena[dstLprior + n] = arena[srcLprior + ancestors[n]];
        }
    }

    /**
     * Waste-free concatenation: writes into a pre-allocated target buffer.
     *
     * <p>Copies theta columns and scalar arrays from each source sequentially
     * into the target. The target must have been allocated with size equal to
     * the sum of all source sizes. No intermediate ThetaParticles objects are
     * created (Req 19.3).
     *
     * @param target  pre-allocated target of size N1 + N2 + ... + NK
     * @param sources source particle containers to concatenate
     */
    public static void concatenateInto(ThetaParticles target, ThetaParticles... sources) {
        int offset = 0;
        int dim = target.dim;
        int targetN = target.N;
        int tgtLpost = target.lpostOff();
        int tgtLlik = target.llikOff();
        int tgtLprior = target.lpriorOff();
        for (ThetaParticles src : sources) {
            int srcN = src.N;
            int srcLpost = src.lpostOff();
            int srcLlik = src.llikOff();
            int srcLprior = src.lpriorOff();
            // Copy theta: column-major (dim, N) layout
            for (int j = 0; j < dim; j++) {
                System.arraycopy(src.arena, j * srcN, target.arena, j * targetN + offset, srcN);
            }
            // Copy scalar arrays
            System.arraycopy(src.arena, srcLpost, target.arena, tgtLpost + offset, srcN);
            System.arraycopy(src.arena, srcLlik, target.arena, tgtLlik + offset, srcN);
            System.arraycopy(src.arena, srcLprior, target.arena, tgtLprior + offset, srcN);
            offset += srcN;
        }
    }

    /**
     * Masked copy: for each n where {@code where[n] == true}, copy particle n
     * from src into self.
     *
     * @param src   source particles (must have same dim and N)
     * @param where boolean mask of length N
     */
    public void copyFrom(ThetaParticles src, boolean[] where) {
        int lpostOff = lpostOff();
        int llikOff = llikOff();
        int lpriorOff = lpriorOff();
        int srcLpost = src.lpostOff();
        int srcLlik = src.llikOff();
        int srcLprior = src.lpriorOff();
        for (int n = 0; n < N; n++) {
            if (where[n]) {
                // Copy theta column for particle n
                for (int j = 0; j < dim; j++) {
                    arena[j * N + n] = src.arena[j * N + n];
                }
                arena[lpostOff + n] = src.arena[srcLpost + n];
                arena[llikOff + n] = src.arena[srcLlik + n];
                arena[lpriorOff + n] = src.arena[srcLprior + n];
            }
        }
    }

    /**
     * Single-particle copy: copy particle m from src into position n of self.
     *
     * @param n   destination particle index in this container
     * @param src source container
     * @param m   source particle index
     */
    public void copyAt(int n, ThetaParticles src, int m) {
        int srcLpost = src.lpostOff();
        int srcLlik = src.llikOff();
        int srcLprior = src.lpriorOff();
        for (int j = 0; j < dim; j++) {
            arena[j * N + n] = src.arena[j * src.N + m];
        }
        arena[lpostOff() + n] = src.arena[srcLpost + m];
        arena[llikOff() + n] = src.arena[srcLlik + m];
        arena[lpriorOff() + n] = src.arena[srcLprior + m];
    }
}
