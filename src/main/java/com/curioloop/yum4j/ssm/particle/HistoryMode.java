package com.curioloop.yum4j.ssm.particle;

/**
 * Controls how much particle history the {@code Workspace} retains across
 * time steps. Smoothers require at least {@link #FULL} or {@link #ROLLING};
 * pure filtering with no smoothing should use {@link #NONE} to minimise
 * memory footprint.
 *
 * <h2>Lossless equivalence with the pre-spec implementation</h2>
 *
 * <p>The {@code FULL}, {@code ROLLING}, and {@code PARTIAL} variants
 * were rewritten in place by the {@code particle-v2-mem} spec
 * (ancestor compression — Stage B). The enum values are unchanged,
 * the {@code Workspace.allocate(...)} dispatch is unchanged, and the
 * smoother surfaces are unchanged. Output equivalence with the
 * pre-spec implementation is governed by R9 of the spec:
 *
 * <ul>
 *   <li><b>Bitwise on integer outputs.</b> Ancestor indices returned
 *       by {@code viewAncestors(t, ...)} and every smoother output
 *       (FFBS, TwoFilter, FixedLag, Paris) are bit-for-bit identical
 *       to the pre-spec baseline for every seed in the parity grid
 *       (R9.2 ancestors clause; R9.3 smoother clause).</li>
 *   <li><b>≤ 10⁻¹⁰ relative on continuous-valued outputs.</b> The
 *       continuous-valued buffers returned by {@code viewX(t, ...)}
 *       and {@code viewLogW(t, ...)} match the pre-spec baseline to
 *       within a 10⁻¹⁰ relative tolerance — the empirical IEEE 754
 *       / JIT noise floor on the reference machine class. The
 *       1-ULP-level drift never propagates into smoother output
 *       because the smoother quantises X / logW into ancestor
 *       indices and ESS-threshold decisions, both of which absorb
 *       1-ULP input perturbations.</li>
 * </ul>
 *
 * <p>Achieved Stage B reduction at the gate workload
 * ({@code T = 10⁴, N = 10⁵, d = 1, rate = 0.4}): 12% total on
 * {@link #FULL}, 60% on {@link #PARTIAL}; see
 * {@code bench/mem-results.md} for the full breakdown.
 *
 * @see com.curioloop.yum4j.ssm.particle.engine.Workspace
 */
public enum HistoryMode {

    /**
     * No history is retained. The engine does not allocate or write to any
     * history buffer. Smoothing is unavailable in this mode.
     */
    NONE,

    /**
     * Full history: the {@code Workspace} retains the {@code X}, {@code logW},
     * and ancestor {@code a} trajectories for all {@code T} steps in
     * contiguous buffers without per-step heap allocation.
     *
     * <p>Implemented in place by
     * {@link com.curioloop.yum4j.ssm.particle.smooth.Full}; ancestor
     * storage is compressed via identity-skip + char-narrowing
     * ({@code particle-v2-mem} R2 + R3). Output is bitwise on ancestors
     * and smoother outputs, ≤ 10⁻¹⁰ relative on continuous-valued
     * X / logW outputs (R9.2 noise floor). See {@code bench/mem-results.md}
     * for the achieved 12% reduction at the gate workload.
     */
    FULL,

    /**
     * Partial history: retains only the information required by online
     * additive smoothers (e.g. Paris-style). The exact subset is
     * implementation-defined but always includes ancestor indices.
     *
     * <p>Implemented in place by
     * {@link com.curioloop.yum4j.ssm.particle.smooth.Partial}; ancestor
     * storage is compressed via identity-skip + char-narrowing
     * ({@code particle-v2-mem} R2 + R3). Output is bitwise on ancestors
     * and smoother outputs (R9.3); {@code viewX} / {@code viewLogW} are
     * not supported by this variant. See {@code bench/mem-results.md} for
     * the achieved 60% reduction at the gate workload.
     */
    PARTIAL,

    /**
     * Rolling history: retains only the last {@code L} steps where {@code L}
     * is the configured lag. Suitable for fixed-lag smoothers that do not
     * need the full trajectory.
     *
     * <p>Implemented in place by
     * {@link com.curioloop.yum4j.ssm.particle.smooth.Rolling}; ancestor
     * storage is compressed via identity-skip + char-narrowing
     * ({@code particle-v2-mem} R2 + R3) on a circular ancestor pool sized
     * to the configured {@code maxResampleRate}. Output is bitwise on
     * ancestors and smoother outputs, ≤ 10⁻¹⁰ relative on continuous-valued
     * X / logW outputs (R9.2 noise floor). See {@code bench/mem-results.md}
     * for the wall-clock and footprint comparison at the gate workload.
     */
    ROLLING
}
