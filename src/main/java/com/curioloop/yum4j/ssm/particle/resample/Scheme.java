package com.curioloop.yum4j.ssm.particle.resample;

/**
 * Resampling-scheme selector consumed by {@code Resample.apply}.
 *
 * <p>All six schemes are implemented: {@link #SYSTEMATIC},
 * {@link #STRATIFIED}, {@link #MULTINOMIAL}, {@link #RESIDUAL},
 * {@link #SSP} (Gerber-Chopin-Whiteley 2019), and {@link #KILLING}.
 * {@link #SSP} and {@link #KILLING} require {@code M == N}.
 *
 */
public enum Scheme {
    SYSTEMATIC,
    STRATIFIED,
    MULTINOMIAL,
    RESIDUAL,
    SSP,
    KILLING,
}
