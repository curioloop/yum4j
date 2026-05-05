package com.curioloop.yum4j.fft;

final class FftSincos {
    private static final double PI = 3.141592653589793238462643383279502884197;

    private final long n;
    private final int mask;
    private final int shift;
    private final double[] v1; // interleaved [re, im, re, im, ...] size = 2*(mask+1)
    private final double[] v2; // interleaved [re, im, re, im, ...] size = 2*v2Count

    FftSincos(long n) {
        if (n < 1) {
            throw new IllegalArgumentException("n must be >= 1");
        }
        this.n = n;
        long nval = (n + 2) / 2;
        int localShift = 1;
        while ((1L << localShift) <= Long.MAX_VALUE / (1L << localShift)
                && (1L << localShift) * (1L << localShift) < nval) {
            localShift++;
        }
        long localMask = (1L << localShift) - 1L;
        if (localMask + 1L > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("twiddle table is too large");
        }
        long v2Size = (nval + localMask) / (localMask + 1L);
        if (v2Size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("twiddle table is too large");
        }
        this.shift = localShift;
        this.mask = (int) localMask;
        int v1Count = mask + 1;
        int v2Count = (int) v2Size;
        this.v1 = new double[2 * v1Count];
        this.v2 = new double[2 * v2Count];
        double angle = 0.25 * PI / (double) n;
        v1[0] = 1.0;
        v1[1] = 0.0;
        for (int index = 1; index < v1Count; index++) {
            calc(index, n, angle, v1, 2 * index);
        }
        v2[0] = 1.0;
        v2[1] = 0.0;
        long stride = localMask + 1L;
        for (int index = 1; index < v2Count; index++) {
            calc(index * stride, n, angle, v2, 2 * index);
        }
    }

    double real(long index) {
        return value(index)[0];
    }

    double imag(long index) {
        return value(index)[1];
    }

    void fill(long index, double[] out, int offset) {
        if (index < 0 || index >= n) {
            index = Math.floorMod(index, n);
        }
        if (2L * index <= n) {
            int lo = (int) (index & mask) << 1;
            int hi = (int) (index >> shift) << 1;
            double v1r = v1[lo], v1i = v1[lo + 1];
            double v2r = v2[hi], v2i = v2[hi + 1];
            out[offset] = v1r * v2r - v1i * v2i;
            out[offset + 1] = v1r * v2i + v1i * v2r;
            return;
        }
        long reflected = n - index;
        int lo = (int) (reflected & mask) << 1;
        int hi = (int) (reflected >> shift) << 1;
        double v1r = v1[lo], v1i = v1[lo + 1];
        double v2r = v2[hi], v2i = v2[hi + 1];
        out[offset] = v1r * v2r - v1i * v2i;
        out[offset + 1] = -(v1r * v2i + v1i * v2r);
    }

    double[] value(long index) {
        double[] value = new double[2];
        fill(index, value, 0);
        return value;
    }

    private static void calc(long x, long n, double angle, double[] out, int offset) {
        long y = Math.multiplyExact(x, 8L);
        if (y < 4L * n) {
            if (y < 2L * n) {
                if (y < n) {
                    out[offset] = Math.cos((double) y * angle);
                    out[offset + 1] = Math.sin((double) y * angle);
                } else {
                    out[offset] = Math.sin((double) (2L * n - y) * angle);
                    out[offset + 1] = Math.cos((double) (2L * n - y) * angle);
                }
            } else {
                y -= 2L * n;
                if (y < n) {
                    out[offset] = -Math.sin((double) y * angle);
                    out[offset + 1] = Math.cos((double) y * angle);
                } else {
                    out[offset] = -Math.cos((double) (2L * n - y) * angle);
                    out[offset + 1] = Math.sin((double) (2L * n - y) * angle);
                }
            }
        } else {
            y = 8L * n - y;
            if (y < 2L * n) {
                if (y < n) {
                    out[offset] = Math.cos((double) y * angle);
                    out[offset + 1] = -Math.sin((double) y * angle);
                } else {
                    out[offset] = Math.sin((double) (2L * n - y) * angle);
                    out[offset + 1] = -Math.cos((double) (2L * n - y) * angle);
                }
            } else {
                y -= 2L * n;
                if (y < n) {
                    out[offset] = -Math.sin((double) y * angle);
                    out[offset + 1] = -Math.cos((double) y * angle);
                } else {
                    out[offset] = -Math.cos((double) (2L * n - y) * angle);
                    out[offset + 1] = -Math.sin((double) (2L * n - y) * angle);
                }
            }
        }
    }
}
