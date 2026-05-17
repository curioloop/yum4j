package com.curioloop.yum4j.test;

import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * {@link RandomGenerator} that replays a fixed sequence of {@code double}
 * uniforms in {@code [0, 1)} previously captured by the Python-side
 * {@code RecordedStream}.
 *
 * <p>The one primitive this class emits is {@link #nextDouble()}. All
 * other {@code RandomGenerator} methods are rebuilt on top of it so that
 * the Java parity path consumes the exact same uniform stream as the
 * reference run — including the inverse-CDF Gaussian path used by
 * {@code RecordedStream} on the Python side.
 *
 * <p>When the recorded stream is exhausted, any draw throws
 * {@link IllegalStateException}. Fixtures MUST record a superset of the
 * draws consumed by the Java path.
 *
 * <p>Not thread-safe.
 */
public final class RecordedRandomGenerator implements RandomGenerator {

    private final double[] uniforms;
    private int cursor;

    public RecordedRandomGenerator(double[] uniforms) {
        this.uniforms = Objects.requireNonNull(uniforms, "uniforms");
        this.cursor = 0;
    }

    /** Number of draws remaining in the recorded stream. */
    public int remaining() {
        return uniforms.length - cursor;
    }

    /** Number of draws consumed so far. */
    public int consumed() {
        return cursor;
    }

    @Override
    public double nextDouble() {
        if (cursor >= uniforms.length) {
            throw new IllegalStateException(
                "RecordedRandomGenerator exhausted: consumed " + cursor
                    + " uniforms; fixture provided only " + uniforms.length);
        }
        return uniforms[cursor++];
    }

    @Override
    public double nextDouble(double bound) {
        if (!(bound > 0.0) || Double.isInfinite(bound) || Double.isNaN(bound)) {
            throw new IllegalArgumentException("bound must be finite and positive: " + bound);
        }
        return nextDouble() * bound;
    }

    @Override
    public double nextDouble(double origin, double bound) {
        if (!(bound > origin) || Double.isInfinite(bound - origin) || Double.isNaN(bound - origin)) {
            throw new IllegalArgumentException("require origin < bound with finite span: origin=" + origin + " bound=" + bound);
        }
        return origin + nextDouble() * (bound - origin);
    }

    @Override
    public long nextLong() {
        // 53 bits are recoverable from a [0,1) double; promote to 64 via two
        // successive draws. This path is rarely consumed on the hot
        // reference paths; fixtures that exercise nextLong must record
        // twice as many uniforms as the Java path consumes here.
        long hi = (long) (nextDouble() * (1L << 26));
        long lo = (long) (nextDouble() * (1L << 38));
        return (hi << 38) | lo;
    }

    @Override
    public int nextInt() {
        return (int) (nextDouble() * (1L << 32)) ^ Integer.MIN_VALUE;
    }

    @Override
    public int nextInt(int bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive: " + bound);
        return (int) (nextDouble() * bound);
    }

    @Override
    public int nextInt(int origin, int bound) {
        if (bound <= origin) throw new IllegalArgumentException("require origin < bound: origin=" + origin + " bound=" + bound);
        return origin + nextInt(bound - origin);
    }

    @Override
    public boolean nextBoolean() {
        return nextDouble() < 0.5;
    }

    @Override
    public float nextFloat() {
        return (float) nextDouble();
    }

    @Override
    public float nextFloat(float bound) {
        if (!(bound > 0.0f)) throw new IllegalArgumentException("bound must be positive: " + bound);
        return (float) (nextDouble() * bound);
    }

    @Override
    public float nextFloat(float origin, float bound) {
        if (bound <= origin) throw new IllegalArgumentException("require origin < bound: origin=" + origin + " bound=" + bound);
        return origin + (float) (nextDouble() * (bound - origin));
    }

    @Override
    public long nextLong(long bound) {
        if (bound <= 0L) throw new IllegalArgumentException("bound must be positive: " + bound);
        return (long) (nextDouble() * bound);
    }

    @Override
    public long nextLong(long origin, long bound) {
        if (bound <= origin) throw new IllegalArgumentException("require origin < bound: origin=" + origin + " bound=" + bound);
        return origin + nextLong(bound - origin);
    }

    @Override
    public double nextGaussian() {
        // Beasley-Springer-Moro inverse standard-normal CDF, matching
        // the Python-side RecordedStream. Accuracy ~1e-9 which is more
        // than sufficient for parity fixtures.
        return invStdNormal(nextDouble());
    }

    @Override
    public double nextGaussian(double mean, double stddev) {
        if (!(stddev > 0.0)) throw new IllegalArgumentException("stddev must be positive: " + stddev);
        return mean + stddev * nextGaussian();
    }

    @Override
    public double nextExponential() {
        // inverse-CDF: -log(U)
        return -Math.log(1.0 - nextDouble());
    }

    // ---- inverse standard normal (Beasley-Springer-Moro) ------------

    private static final double[] A = {
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    };

    private static final double[] B = {
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    };

    private static final double[] C = {
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    };

    private static final double[] D = {
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    };

    private static final double PLOW = 0.02425;
    private static final double PHIGH = 1.0 - PLOW;

    static double invStdNormal(double u) {
        if (u < PLOW) {
            double q = Math.sqrt(-2.0 * Math.log(u));
            return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
        }
        if (u <= PHIGH) {
            double q = u - 0.5;
            double r = q * q;
            return (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
                / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
        }
        double q = Math.sqrt(-2.0 * Math.log(1.0 - u));
        return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
}
