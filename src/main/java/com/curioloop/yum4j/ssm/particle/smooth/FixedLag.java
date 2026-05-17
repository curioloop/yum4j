package com.curioloop.yum4j.ssm.particle.smooth;

/**
 * Fixed-lag smoother built on top of {@link Rolling} history.
 *
 * <p>Given a rolling-window history of length {@code L ≥ lag + 1} and
 * a test function {@code phi : double[dim·N] → double[N]}, returns
 * {@code E[phi(X_{newest - lag}) | y_{0:newest}]} approximated by the
 * weighted average using the <em>current</em> filter weights at time
 * {@code newest}.
 *
 * <h3>Fixed-lag approximation</h3>
 *
 * <p>The fixed-lag smoother approximates the smoothing distribution at
 * time {@code t} using only particles from {@code [t, t+L]}:
 *
 * <pre>
 *   p(x_t | y_{0:t+L}) ≈ Σ_n w_{t+L}^n · δ(x_t = B_{t:t+L}^n)
 * </pre>
 *
 * where {@code B_{t:t+L}^n} traces the ancestor chain from particle
 * {@code n} at time {@code t+L} back to time {@code t}.
 *
 * <h3>Trajectory reconstruction</h3>
 *
 * <p>Walk the ancestor chain backward:
 * {@code a[newest], a[newest-1], …, a[newest - lag + 1]} to identify
 * the ancestor of each final particle at time {@code newest - lag}.
 * Then read back the particles at that time and apply {@code phi}.
 *
 * <p>Each method offers two overloads: a convenience form that
 * allocates fresh scratch per call, and a
 * {@link SmoothingWorkspace}-accepting form that reuses caller-supplied
 * scratch (R5.2, R5.3). The workspace buffers used are {@code sw.aTp1}
 * (ancestor-read scratch), {@code sw.Xt} / {@code sw.Xtp1Broadcast}
 * (particle gather), and {@code sw.logWt} / {@code sw.logPtBuf}
 * (log-weights and phi values).
 *
 */
public final class FixedLag {

    private FixedLag() {}

    /**
     * Test function on a {@code (dim, N)} particle buffer → one scalar
     * per particle.
     */
    @FunctionalInterface
    public interface Phi {
        /**
         * Applies the test function to the particle buffer.
         *
         * @param particles particle states, length {@code dim * N},
         *                  laid out as {@code [dim0_p0, dim0_p1, ..., dim0_pN-1, dim1_p0, ...]}
         * @param dim       state dimension per particle
         * @param N         number of particles
         * @param out       output buffer of length {@code N}
         */
        void apply(double[] particles, int dim, int N, double[] out);
    }

    /**
     * Identity test function for a {@code dim = 1} history: returns the
     * particle value itself.
     */
    public static Phi identity1D() {
        return (particles, dim, N, out) -> {
            if (dim != 1) {
                throw new IllegalArgumentException(
                    "identity1D only defined for dim=1, got dim=" + dim);
            }
            System.arraycopy(particles, 0, out, 0, N);
        };
    }

    /**
     * Computes the fixed-lag smoothing estimate using a freshly allocated
     * {@link SmoothingWorkspace}. Delegates to
     * {@link #compute(Rolling, int, Phi, SmoothingWorkspace)}.
     *
     * @param history the rolling-window history
     * @param lag     the smoothing lag
     * @param phi     the test function
     * @return the fixed-lag smoothing estimate
     */
    public static double compute(Rolling history, int lag, Phi phi) {
        if (history == null) throw new IllegalArgumentException("history must not be null");
        if (lag < 0) throw new IllegalArgumentException("lag must be >= 0: " + lag);
        SmoothingWorkspace sw = SmoothingWorkspace.allocate(
                1, history.N(), history.dim(), history.capacity());
        return compute(history, lag, phi, sw);
    }

    /**
     * Workspace-accepting overload of {@link #compute(Rolling, int, Phi)}.
     *
     * <p>Uses {@code sw.aTp1} as the ancestor-read scratch buffer,
     * {@code sw.Xt} for the particle gather source, {@code sw.Xtp1Broadcast}
     * for the gathered output, {@code sw.logPtBuf} for phi values, and
     * {@code sw.logWt} for the final-step log-weights. Two {@code int[N]}
     * ping-pong slots for the ancestor trace are allocated locally once per
     * call.
     *
     * @param history the rolling-window history (must have at least
     *                {@code lag + 1} steps stored)
     * @param lag     the smoothing lag (must be ≥ 0 and ≤ stored window)
     * @param phi     the test function to evaluate
     * @param sw      pre-allocated smoothing workspace
     * @return the fixed-lag smoothing estimate
     */
    public static double compute(Rolling history, int lag, Phi phi, SmoothingWorkspace sw) {
        if (history == null) throw new IllegalArgumentException("history must not be null");
        if (lag < 0) throw new IllegalArgumentException("lag must be >= 0: " + lag);
        int newest = history.newest();
        if (newest < 0) {
            throw new IllegalStateException("history is empty");
        }
        int targetStep = newest - lag;
        if (targetStep < 0 || !history.hasStep(targetStep)) {
            throw new IllegalArgumentException(
                "lag " + lag + " exceeds stored window ["
                    + history.oldest() + ", " + newest + "]");
        }
        int N = history.N();
        int dim = history.dim();
        sw.validateShape(1, N, dim, history.capacity());

        // Reconstruct ancestor indices at time targetStep for each final
        // particle by tracing backward through the ancestor chain.
        // Ancestor reads go into sw.aTp1; B and Bnext are local ping-pong.
        int[] ancBuf = sw.aTp1;
        int[] B = new int[N];
        int[] Bnext = new int[N];
        for (int n = 0; n < N; n++) B[n] = n;
        for (int t = newest; t > targetStep; t--) {
            // Identity-skip fast path: when step t was not resampled, the
            // ancestor permutation is the identity so Bnext[n] == B[n] for
            // every n. Both the viewAncestors synthesis and the ping-pong
            // copy are pure no-ops; skip the iteration entirely.
            if (!history.resampledAt(t)) continue;
            history.viewAncestors(t, ancBuf, 0);
            for (int n = 0; n < N; n++) {
                Bnext[n] = ancBuf[B[n]];
            }
            int[] tmp = B;
            B = Bnext;
            Bnext = tmp;
        }

        // Gather particles at targetStep according to B.
        double[] Xt = sw.Xt;
        double[] gathered = sw.Xtp1Broadcast;
        history.viewX(targetStep, Xt, 0);
        for (int j = 0; j < dim; j++) {
            int rowOff = j * N;
            for (int n = 0; n < N; n++) {
                gathered[rowOff + n] = Xt[rowOff + B[n]];
            }
        }

        // Apply phi.
        double[] phiVals = sw.logPtBuf;
        phi.apply(gathered, dim, N, phiVals);

        // Weighted average using current (newest-step) log-weights.
        double[] lw = sw.logWt;
        history.viewLogW(newest, lw, 0);
        double m = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < N; i++) {
            double v = lw[i];
            if (v > m) m = v;
        }
        if (m == Double.NEGATIVE_INFINITY) {
            // All weights are -inf; fall back to uniform average.
            double sum = 0.0;
            for (int n = 0; n < N; n++) sum += phiVals[n];
            return sum / N;
        }
        double sum = 0.0;
        double sumW = 0.0;
        for (int n = 0; n < N; n++) {
            double w = Math.exp(lw[n] - m);
            sumW += w;
            sum += w * phiVals[n];
        }
        return sum / sumW;
    }

    /**
     * Computes the fixed-lag smoothed state estimate for a 1-dimensional
     * model. Convenience method equivalent to
     * {@code compute(history, lag, identity1D())}.
     *
     * @param history the rolling-window history
     * @param lag     the smoothing lag
     * @return the smoothed state estimate at time {@code newest - lag}
     */
    public static double smoothedMean1D(Rolling history, int lag) {
        return compute(history, lag, identity1D());
    }

    /**
     * Computes the fixed-lag smoothed trajectory indices using a freshly
     * allocated {@link SmoothingWorkspace}. Delegates to
     * {@link #traceAncestors(Rolling, int, SmoothingWorkspace)}.
     *
     * @param history the rolling-window history
     * @param lag     the smoothing lag
     * @return ancestor indices at time {@code newest - lag}, length N
     */
    public static int[] traceAncestors(Rolling history, int lag) {
        if (history == null) throw new IllegalArgumentException("history must not be null");
        if (lag < 0) throw new IllegalArgumentException("lag must be >= 0: " + lag);
        SmoothingWorkspace sw = SmoothingWorkspace.allocate(
                1, history.N(), history.dim(), history.capacity());
        return traceAncestors(history, lag, sw);
    }

    /**
     * Workspace-accepting overload of {@link #traceAncestors(Rolling, int)}.
     *
     * <p>Uses {@code sw.aTp1} as the ancestor-read scratch buffer and
     * allocates two {@code int[N]} ping-pong slots locally (one-shot per
     * call; no per-step allocation).
     *
     * @param history the rolling-window history
     * @param lag     the smoothing lag
     * @param sw      pre-allocated smoothing workspace
     * @return ancestor indices at time {@code newest - lag}, length N
     */
    public static int[] traceAncestors(Rolling history, int lag, SmoothingWorkspace sw) {
        if (history == null) throw new IllegalArgumentException("history must not be null");
        if (lag < 0) throw new IllegalArgumentException("lag must be >= 0: " + lag);
        int newest = history.newest();
        if (newest < 0) {
            throw new IllegalStateException("history is empty");
        }
        int targetStep = newest - lag;
        if (targetStep < 0 || !history.hasStep(targetStep)) {
            throw new IllegalArgumentException(
                "lag " + lag + " exceeds stored window ["
                    + history.oldest() + ", " + newest + "]");
        }
        int N = history.N();
        sw.validateShape(1, N, history.dim(), history.capacity());

        // Trace backward from newest to targetStep. Ancestor reads go into
        // sw.aTp1; B and Bnext are local ping-pong slots so the returned B
        // array is caller-owned and safe to hand off.
        int[] ancBuf = sw.aTp1;
        int[] B = new int[N];
        int[] Bnext = new int[N];
        for (int n = 0; n < N; n++) B[n] = n;

        for (int t = newest; t > targetStep; t--) {
            // Identity-skip fast path: when step t was not resampled, the
            // ancestor permutation is the identity so Bnext[n] == B[n] for
            // every n. Both the viewAncestors synthesis and the ping-pong
            // copy are pure no-ops; skip the iteration entirely.
            if (!history.resampledAt(t)) continue;
            history.viewAncestors(t, ancBuf, 0);
            for (int n = 0; n < N; n++) {
                Bnext[n] = ancBuf[B[n]];
            }
            int[] tmp = B;
            B = Bnext;
            Bnext = tmp;
        }
        return B;
    }
}
