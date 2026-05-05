package com.curioloop.yum4j.tsa.stl;

import java.util.Arrays;
import java.util.Objects;

/**
 * Seasonal-trend decomposition using LOESS.
 *
 * <p>For a single seasonal period {@code p}, STL models an observed series as</p>
 *
 * <pre>{@code
 * y[t] = T[t] + S[t] + R[t]
 * }</pre>
 *
 * <p>where {@code T} is a slowly varying trend, {@code S} is a seasonal component with period
 * {@code p}, and {@code R} is the remainder. Each inner iteration alternates between smoothing
 * the cycle-subseries {@code y[t] - T[t]} to estimate {@code S}, and smoothing the deseasonalized
 * series {@code y[t] - S[t]} to estimate {@code T}. Robust mode adds outer iterations with
 * bisquare weights computed from {@code |R[t]|}.</p>
 */
public final class STL {

    private static final STLSmooth DEFAULT_SEASONAL = STLSmooth.of(7, 1, 1);

    private final int period;
    private final boolean robust;
    private final STLSmooth seasonal;
    private final STLSmooth trend;
    private final STLSmooth lowPass;

    public STL(int period) {
        this(builder(period));
    }

    public STL(int period, boolean robust) {
        this(builder(period).robust(robust));
    }

    private STL(Builder builder) {
        if (builder.period < 2) {
            throw new IllegalArgumentException("period must be at least 2");
        }
        this.period = builder.period;
        this.robust = builder.robust;
        this.seasonal = builder.seasonal == null ? DEFAULT_SEASONAL : builder.seasonal;
        this.trend = builder.trend == null ? defaultTrend(period, seasonal.length()) : builder.trend;
        this.lowPass = builder.lowPass == null ? defaultLowPass(period) : builder.lowPass;
        validateWindow("trend", trend, period);
        validateWindow("lowPass", lowPass, period);
    }

    public static STL of(int period) {
        return builder(period).build();
    }

    public static Builder builder(int period) {
        return new Builder(period);
    }

    public int period() {
        return period;
    }

    public boolean robust() {
        return robust;
    }

    public STLSmooth seasonal() {
        return seasonal;
    }

    public STLSmooth trend() {
        return trend;
    }

    public STLSmooth lowPass() {
        return lowPass;
    }

    /**
     * Decomposes {@code values} using the default STL iteration counts.
     *
     * <p>The returned arrays satisfy {@code values[t] = seasonal[t] + trend[t] + residual[t]}
     * up to floating-point roundoff.</p>
     */
    public STLResult fit(double[] values) {
        return fit(values, robust ? 2 : 5, robust ? 15 : 0);
    }

    /**
     * Decomposes {@code values} with explicit inner and outer iteration counts.
     *
     * <p>This mirrors statsmodels' {@code fit(inner_iter, outer_iter)} controls: the inner loop
     * updates seasonal and trend estimates, while each outer loop recomputes robust weights.</p>
     */
    public STLResult fit(double[] values, int innerIter, int outerIter) {
        Objects.requireNonNull(values, "values");
        if (values.length < period) {
            throw new IllegalArgumentException("values length must be at least the seasonal period");
        }
        if (innerIter < 1) {
            throw new IllegalArgumentException("innerIter must be positive");
        }
        if (outerIter < 0) {
            throw new IllegalArgumentException("outerIter must be non-negative");
        }

        int n = values.length;
        double[][] work = new double[5][n + 2 * period];
        Context ctx = new Context(values, new double[n], new double[n], new double[n], work);
        Arrays.fill(ctx.robust, 1.0);

        for (int k = 0; ; k++) {
            decompose(ctx, innerIter);
            if (k + 1 > outerIter) {
                break;
            }
            robustWeight(ctx);
            ctx.useRobustWeights = true;
        }

        double[] residual = new double[n];
        for (int i = 0; i < n; i++) {
            residual[i] = ctx.value[i] - ctx.seasonal[i] - ctx.trend[i];
        }
        return new STLResult(Arrays.copyOf(values, n), ctx.seasonal, ctx.trend, residual, ctx.robust);
    }

    /**
     * Runs the STL inner loop:
     *
     * <pre>{@code
     * D[t] = y[t] - T[t]
     * C   = smooth each period-p cycle-subseries of D
     * L   = low-pass(C)
     * S[t] = C[t] - L[t]
     * Q[t] = y[t] - S[t]
     * T[t] = loess(Q)
     * }</pre>
     */
    private void decompose(Context ctx, int innerIter) {
        double[] values = ctx.value;
        double[] trend = ctx.trend;
        double[] season = ctx.seasonal;
        double[][] work = ctx.work;
        int n = values.length;
        for (int iter = 0; iter < innerIter; iter++) {
            for (int i = 0; i < n; i++) {
                work[0][i] = values[i] - trend[i];
            }
            cycleSubSeries(ctx);
            lowPassFilter(ctx);
            for (int i = 0; i < n; i++) {
                season[i] = work[1][period + i] - work[0][i];
                work[0][i] = values[i] - season[i];
            }
            smooth(trend(), work[0], n, ctx.useRobustWeights, ctx.robust, trend, 0, work[2]);
        }
    }

    /**
     * Smooths each cycle-subseries {@code D[j + k * period]} independently.
     *
     * <p>The smoothed values are stored with a {@code period} offset so that two extrapolated
     * boundary cycles can live at indices {@code -period..-1} and {@code n..n+period-1} in the
     * original STL indexing.</p>
     */
    private void cycleSubSeries(Context ctx) {
        int n = ctx.value.length;
        double[][] work = ctx.work;
        double[] detrended = work[0];
        double[] cycle = work[1];
        double[] subSeries = work[2];
        double[] smoothValues = work[3];
        double[] robustSubSeries = work[4];
        double[] loessWeights = ctx.seasonal;

        for (int j = 0; j < period; j++) {
            int k = (n - (j + 1)) / period + 1;
            for (int i = 0; i < k; i++) {
                subSeries[i] = detrended[i * period + j];
            }
            if (ctx.useRobustWeights) {
                for (int i = 0; i < k; i++) {
                    robustSubSeries[i] = ctx.robust[i * period + j];
                }
            }
            smooth(seasonal, subSeries, k, ctx.useRobustWeights, robustSubSeries, smoothValues, 1, loessWeights);

            int right = Math.min(seasonal.length(), k);
            smoothValues[0] = loess(seasonal, subSeries, k, 0, 1, right, loessWeights, ctx.useRobustWeights, robustSubSeries);
            if (Double.isNaN(smoothValues[0])) {
                smoothValues[0] = smoothValues[1];
            }

            int left = Math.max(1, k - seasonal.length() + 1);
            smoothValues[k + 1] = loess(seasonal, subSeries, k, k + 1, left, k, loessWeights, ctx.useRobustWeights, robustSubSeries);
            if (Double.isNaN(smoothValues[k + 1])) {
                smoothValues[k + 1] = smoothValues[k];
            }

            for (int m = 0; m < k + 2; m++) {
                cycle[m * period + j] = smoothValues[m];
            }
        }
    }

    /**
     * Removes the low-frequency drift from the cycle-subseries estimate.
     *
     * <p>STL applies two moving averages of span {@code period}, then one moving average of span
     * {@code 3}, followed by a LOESS pass:</p>
     *
     * <pre>{@code
     * L = loess_low_pass(MA_3(MA_p(MA_p(C))))
     * }</pre>
     */
    private void lowPassFilter(Context ctx) {
        int workLength = ctx.value.length + 2 * period;
        double[][] work = ctx.work;
        movingAverage(work[1], workLength, period, work[2]);
        movingAverage(work[2], workLength - period + 1, period, work[0]);
        movingAverage(work[0], workLength - 2 * period + 2, 3, work[2]);
        smooth(lowPass, work[2], ctx.value.length, false, work[3], work[0], 0, work[4]);
    }

    /**
     * Updates robust bisquare weights from the current residual magnitudes.
     *
     * <pre>{@code
     * e[t] = |y[t] - T[t] - S[t]|
     * c    = 6 * median(e)        // implemented as 3 * (lowerMedian + upperMedian)
     * w[t] = 1                    if e[t] <= 0.001 * c
     *      = (1 - (e[t] / c)^2)^2 if e[t] <= 0.999 * c
     *      = 0                    otherwise
     * }</pre>
     */
    private void robustWeight(Context ctx) {
        int n = ctx.value.length;
        for (int i = 0; i < n; i++) {
            ctx.robust[i] = Math.abs(ctx.value[i] - ctx.trend[i] - ctx.seasonal[i]);
        }

        double[] sorted = ctx.work[0];
        System.arraycopy(ctx.robust, 0, sorted, 0, n);
        Arrays.sort(sorted, 0, n);
        double a = sorted[n / 2];
        double b = sorted[n - (n / 2) - 1];
        double c = 3.0 * (a + b);

        if (c == 0.0) {
            Arrays.fill(ctx.robust, 0, n, 1.0);
            return;
        }
        double c1 = 0.001 * c;
        double c9 = 0.999 * c;
        for (int i = 0; i < n; i++) {
            double weight = ctx.robust[i];
            if (weight <= c1) {
                ctx.robust[i] = 1.0;
            } else if (weight <= c9) {
                double u = weight / c;
                double oneMinusSquared = 1.0 - u * u;
                ctx.robust[i] = oneMinusSquared * oneMinusSquared;
            } else {
                ctx.robust[i] = 0.0;
            }
        }
    }

    /** Computes {@code average[i] = mean(values[i], ..., values[i + step - 1])}. */
    private static void movingAverage(double[] values, int n, int step, double[] average) {
        double sum = 0.0;
        for (int i = 0; i < step; i++) {
            sum += values[i];
        }
        average[0] = sum / step;
        for (int out = 1, add = step, remove = 0; out < n - step + 1; out++, add++, remove++) {
            sum += values[add] - values[remove];
            average[out] = sum / step;
        }
    }

    /**
     * Evaluates LOESS at positions {@code 1, 1 + jump, 1 + 2 * jump, ...} and linearly
     * interpolates skipped points. This matches the STL {@code *_jump} acceleration knobs.
     */
    private static void smooth(STLSmooth config,
                               double[] values,
                               int n,
                               boolean useRobustWeights,
                               double[] robust,
                               double[] smoothed,
                               int smoothedOffset,
                               double[] weights) {
        if (n < 2) {
            smoothed[smoothedOffset] = values[0];
            return;
        }

        int jump = Math.min(config.jump(), n - 1);
        int left = 0;
        int right = 0;
        if (config.length() >= n) {
            left = 1;
            right = n;
            for (int i = 0; i < n; i += jump) {
                smoothed[smoothedOffset + i] = loess(config, values, n, i + 1, left, right, weights, useRobustWeights, robust);
                if (Double.isNaN(smoothed[smoothedOffset + i])) {
                    smoothed[smoothedOffset + i] = values[i];
                }
            }
        } else if (jump == 1) {
            int halfWindow = (config.length() + 2) / 2;
            left = 1;
            right = config.length();
            for (int i = 0; i < n; i++) {
                if (i + 1 > halfWindow && right != n) {
                    left++;
                    right++;
                }
                smoothed[smoothedOffset + i] = loess(config, values, n, i + 1, left, right, weights, useRobustWeights, robust);
                if (Double.isNaN(smoothed[smoothedOffset + i])) {
                    smoothed[smoothedOffset + i] = values[i];
                }
            }
        } else {
            int halfWindow = (config.length() + 1) / 2;
            for (int i = 0; i < n; i += jump) {
                if (i + 1 < halfWindow) {
                    left = 1;
                    right = config.length();
                } else if (i + 1 >= n - halfWindow + 1) {
                    left = n - config.length() + 1;
                    right = n;
                } else {
                    left = i + 1 - halfWindow + 1;
                    right = config.length() + i + 1 - halfWindow;
                }
                smoothed[smoothedOffset + i] = loess(config, values, n, i + 1, left, right, weights, useRobustWeights, robust);
                if (Double.isNaN(smoothed[smoothedOffset + i])) {
                    smoothed[smoothedOffset + i] = values[i];
                }
            }
        }

        if (jump == 1) {
            return;
        }

        for (int i = 0; i < n - jump; i += jump) {
            double delta = (smoothed[smoothedOffset + i + jump] - smoothed[smoothedOffset + i]) / jump;
            for (int j = i; j < i + jump; j++) {
                smoothed[smoothedOffset + j] = smoothed[smoothedOffset + i] + delta * (j - i);
            }
        }

        int lastSmoothed = ((n - 1) / jump) * jump + 1;
        if (lastSmoothed != n) {
            smoothed[smoothedOffset + n - 1] = loess(config, values, n, n, left, right, weights, useRobustWeights, robust);
            if (Double.isNaN(smoothed[smoothedOffset + n - 1])) {
                smoothed[smoothedOffset + n - 1] = values[n - 1];
            }
            if (lastSmoothed != n - 1) {
                double delta = (smoothed[smoothedOffset + n - 1] - smoothed[smoothedOffset + lastSmoothed - 1]) / (n - lastSmoothed);
                for (int j = lastSmoothed; j < n; j++) {
                    smoothed[smoothedOffset + j] = smoothed[smoothedOffset + lastSmoothed - 1] + delta * (j + 1 - lastSmoothed);
                }
            }
        }
    }

    /**
     * Local regression at one 1-based target position {@code x0}.
     *
     * <p>For neighbors {@code x_j in [left, right]}, STL uses tricube distance weights</p>
     *
     * <pre>{@code
     * u_j = |x_j - x0| / h
     * K(u_j) = (1 - u_j^3)^3, 0 <= u_j < 1
     * w_j = K(u_j) * robust_j
     * }</pre>
     *
     * <p>Weights are normalized to sum to one. With {@code degree == 1}, they are then adjusted
     * to the equivalent local-linear estimate at {@code x0}; with {@code degree == 0}, the result
     * is the local weighted mean.</p>
     */
    private static double loess(STLSmooth config,
                                double[] values,
                                int n,
                                int x0,
                                int left,
                                int right,
                                double[] weights,
                                boolean useRobustWeights,
                                double[] robust) {
        int h = Math.max(x0 - left, right - x0);
        if (config.length() > n) {
            h += (config.length() - n) / 2;
        }

        double weightSum = 0.0;
        double h1 = 0.001 * h;
        double h9 = 0.999 * h;
        for (int j = left - 1; j < right; j++) {
            weights[j] = 0.0;
            double distance = Math.abs(j + 1.0 - x0);
            if (distance <= h9) {
                if (distance <= h1) {
                    weights[j] = 1.0;
                } else {
                    double u = distance / h;
                    double cubic = 1.0 - u * u * u;
                    weights[j] = cubic * cubic * cubic;
                }
                if (useRobustWeights) {
                    weights[j] *= robust[j];
                }
                weightSum += weights[j];
            }
        }
        if (weightSum <= 0.0) {
            return Double.NaN;
        }
        for (int j = left - 1; j < right; j++) {
            weights[j] /= weightSum;
        }

        if (h > 0 && config.degree() > 0) {
            double weightedPosition = 0.0;
            for (int j = left - 1; j < right; j++) {
                weightedPosition += weights[j] * (j + 1.0);
            }
            double center = x0 - weightedPosition;
            double spread = 0.0;
            for (int j = left - 1; j < right; j++) {
                double distance = j + 1.0 - weightedPosition;
                spread += weights[j] * distance * distance;
            }
            double range = 0.001 * (n - 1.0);
            if (Math.sqrt(spread) > range) {
                center /= spread;
                for (int j = left - 1; j < right; j++) {
                    weights[j] *= center * (j + 1.0 - weightedPosition) + 1.0;
                }
            }
        }

        double smoothed = 0.0;
        for (int j = left - 1; j < right; j++) {
            smoothed += weights[j] * values[j];
        }
        return smoothed;
    }

    /** Default STL trend window: next odd integer above {@code 1.5p / (1 - 1.5 / seasonal)}. */
    private static STLSmooth defaultTrend(int period, int seasonalLength) {
        int length = (int) Math.ceil(1.5 * period / (1.0 - 1.5 / seasonalLength));
        if ((length & 1) == 0) {
            length++;
        }
        return STLSmooth.of(length, 1, 1);
    }

    /** Default low-pass window: the next odd integer greater than {@code period}. */
    private static STLSmooth defaultLowPass(int period) {
        int length = period + 1;
        if ((length & 1) == 0) {
            length++;
        }
        return STLSmooth.of(length, 1, 1);
    }

    private static void validateWindow(String name, STLSmooth smooth, int period) {
        if (smooth.length() <= period) {
            throw new IllegalArgumentException(name + ".length must be greater than period");
        }
    }

    private static final class Context {
        private boolean useRobustWeights;
        private final double[] value;
        private final double[] seasonal;
        private final double[] trend;
        private final double[] robust;
        private final double[][] work;

        private Context(double[] value, double[] seasonal, double[] trend, double[] robust, double[][] work) {
            this.value = value;
            this.seasonal = seasonal;
            this.trend = trend;
            this.robust = robust;
            this.work = work;
        }
    }

    public static final class Builder {
        private final int period;
        private boolean robust;
        private STLSmooth seasonal;
        private STLSmooth trend;
        private STLSmooth lowPass;

        private Builder(int period) {
            this.period = period;
        }

        public Builder robust(boolean robust) {
            this.robust = robust;
            return this;
        }

        public Builder seasonal(STLSmooth seasonal) {
            this.seasonal = Objects.requireNonNull(seasonal, "seasonal");
            return this;
        }

        public Builder seasonal(int length) {
            return seasonal(STLSmooth.of(length, 1, 1));
        }

        public Builder seasonal(int length, int degree, int jump) {
            return seasonal(STLSmooth.of(length, degree, jump));
        }

        public Builder trend(STLSmooth trend) {
            this.trend = Objects.requireNonNull(trend, "trend");
            return this;
        }

        public Builder trend(int length) {
            return trend(STLSmooth.of(length, 1, 1));
        }

        public Builder trend(int length, int degree, int jump) {
            return trend(STLSmooth.of(length, degree, jump));
        }

        public Builder lowPass(STLSmooth lowPass) {
            this.lowPass = Objects.requireNonNull(lowPass, "lowPass");
            return this;
        }

        public Builder lowPass(int length) {
            return lowPass(STLSmooth.of(length, 1, 1));
        }

        public Builder lowPass(int length, int degree, int jump) {
            return lowPass(STLSmooth.of(length, degree, jump));
        }

        public STL build() {
            return new STL(this);
        }
    }
}
