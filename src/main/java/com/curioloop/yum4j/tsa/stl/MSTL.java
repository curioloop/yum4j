package com.curioloop.yum4j.tsa.stl;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Objects;

/**
 * Multi-seasonal STL decomposition.
 *
 * <p>MSTL extends STL to a collection of seasonal periods {@code p_m} by fitting one seasonal
 * component at a time:</p>
 *
 * <pre>{@code
 * y[t] = T[t] + sum_m S_m[t] + R[t]
 * }</pre>
 *
 * <p>During each iteration, the working series starts as the original series with the currently
 * known seasonal components removed. Before refitting component {@code m}, its previous estimate
 * is added back; after STL refits that period, the new estimate is subtracted again:</p>
 *
 * <pre>{@code
 * Z[t]       = y[t] - sum_m S_m[t]
 * Z[t]      += old S_m[t]
 * S_m, T, R  = STL(period = p_m, seasonalWindow = w_m).fit(Z)
 * Z[t]      -= new S_m[t]
 * }</pre>
 *
 * <p>If {@link Builder#lambda(double)} is finite, decomposition is performed on the Box-Cox
 * transformed series</p>
 *
 * <pre>{@code
 * z[t] = log(y[t])                    if lambda == 0
 *      = (y[t]^lambda - 1) / lambda   otherwise
 * }</pre>
 *
 * <p>The result's {@code observed} component is this transformed series, matching statsmodels'
 * MSTL convention.</p>
 */
public final class MSTL {

    private final STL base;
    private final int[][] seasons;
    private final int iterations;
    private final double lambda;
    private final Integer innerIterations;
    private final Integer outerIterations;

    public MSTL(int... periods) {
        this(builder(periods));
    }

    private MSTL(Builder builder) {
        int[] periods = Objects.requireNonNull(builder.periods, "periods");
        if (periods.length == 0) {
            throw new IllegalArgumentException("periods must not be empty");
        }
        int[] windows = builder.windows;
        if (windows == null) {
            windows = new int[periods.length];
            for (int i = 0; i < windows.length; i++) {
                windows[i] = 7 + 4 * (i + 1);
            }
        }
        if (periods.length != windows.length) {
            throw new IllegalArgumentException("periods and windows must have same length");
        }

        this.base = builder.base == null ? STL.builder(2).build() : builder.base;
        this.iterations = builder.iterations <= 0 ? 2 : builder.iterations;
        this.lambda = builder.lambda;
        this.innerIterations = builder.innerIterations;
        this.outerIterations = builder.outerIterations;
        this.seasons = new int[periods.length][2];
        for (int i = 0; i < periods.length; i++) {
            if (periods[i] < 2) {
                throw new IllegalArgumentException("periods must be at least 2");
            }
            STLSmooth.of(windows[i], base.seasonal().degree(), base.seasonal().jump());
            this.seasons[i][0] = periods[i];
            this.seasons[i][1] = windows[i];
        }
        Arrays.sort(this.seasons, Comparator.<int[]>comparingInt(pair -> pair[0]).thenComparingInt(pair -> pair[1]));
    }

    public static Builder builder(int... periods) {
        return new Builder(periods);
    }

    public int seasonalCount() {
        return seasons.length;
    }

    public int[] periods() {
        int[] periods = new int[seasons.length];
        for (int i = 0; i < seasons.length; i++) {
            periods[i] = seasons[i][0];
        }
        return periods;
    }

    public int[] windows() {
        int[] windows = new int[seasons.length];
        for (int i = 0; i < seasons.length; i++) {
            windows[i] = seasons[i][1];
        }
        return windows;
    }

    public int iterations() {
        return iterations;
    }

    public double lambda() {
        return lambda;
    }

    public Integer innerIterations() {
        return innerIterations;
    }

    public Integer outerIterations() {
        return outerIterations;
    }

    /** Runs the base single-season STL instance configured for this MSTL object. */
    public STLResult fitSingle(double[] values) {
        return base.fit(values);
    }

    /**
     * Decomposes {@code values} into multiple seasonal components, one trend, and one residual.
     *
      * <p>The returned arrays satisfy
      * {@code observed[t] = trend[t] + residual[t] + sum_m seasonal[m][t]} up to roundoff.</p>
      *
      * <p>Seasonal periods greater than or equal to half the series length are omitted, as in
      * statsmodels. If all configured periods are too long, decomposition is undefined and an
      * exception is thrown.</p>
     */
    public MSTLResult fit(double[] values) {
        Objects.requireNonNull(values, "values");
        int n = values.length;
        if (n == 0) {
            throw new IllegalArgumentException("values must not be empty");
        }
        int[][] activeSeasons = activeSeasons(n);
        int seasonalCount = activeSeasons.length;
        int effectiveIterations = seasonalCount == 1 ? 1 : iterations;

        double[] observed = transform(values);
        double[] deseasonalized = Arrays.copyOf(observed, n);
        double[][] seasonal = new double[seasonalCount][];
        double[] trend = new double[n];
        double[] residual = new double[n];
        double[] weights = new double[n];

        for (int iter = 0; iter < effectiveIterations; iter++) {
            for (int i = 0; i < activeSeasons.length; i++) {
                if (seasonal[i] != null) {
                    for (int j = 0; j < n; j++) {
                        deseasonalized[j] += seasonal[i][j];
                    }
                }

                STLResult result = fitComponent(activeSeasons[i][0], activeSeasons[i][1], deseasonalized);
                seasonal[i] = result.seasonal();
                trend = result.trend();
                weights = result.weights();

                for (int j = 0; j < n; j++) {
                    deseasonalized[j] -= seasonal[i][j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            residual[i] = deseasonalized[i] - trend[i];
        }
        return new MSTLResult(observed, seasonal, trend, residual, weights);
    }

    private int[][] activeSeasons(int n) {
        int count = 0;
        for (int[] season : seasons) {
            if (season[0] < n / 2.0) {
                count++;
            }
        }
        if (count == 0) {
            throw new IllegalArgumentException(
                "all periods are greater than or equal to half the length of the time series");
        }
        return Arrays.copyOf(seasons, count);
    }

    private double[] transform(double[] values) {
        double[] transformed = Arrays.copyOf(values, values.length);
        if (Double.isNaN(lambda)) {
            return transformed;
        }
        for (int i = 0; i < transformed.length; i++) {
            if (!(transformed[i] > 0.0)) {
                throw new IllegalArgumentException("Box-Cox transformation requires positive values");
            }
            transformed[i] = lambda == 0.0
                ? Math.log(transformed[i])
                : (Math.pow(transformed[i], lambda) - 1.0) / lambda;
        }
        return transformed;
    }

    private STLResult fitComponent(int period, int window, double[] values) {
        STL stl = componentSTL(period, window);
        if (innerIterations == null && outerIterations == null) {
            return stl.fit(values);
        }
        int inner = innerIterations == null ? (stl.robust() ? 2 : 5) : innerIterations;
        int outer = outerIterations == null ? (stl.robust() ? 15 : 0) : outerIterations;
        return stl.fit(values, inner, outer);
    }

    /** Creates the per-period STL problem while reusing base robust/trend/low-pass controls. */
    private STL componentSTL(int period, int window) {
        STL.Builder builder = STL.builder(period)
            .robust(base.robust())
            .seasonal(STLSmooth.of(window, base.seasonal().degree(), base.seasonal().jump()));
        if (base.trend().length() > period) {
            builder.trend(base.trend());
        }
        if (base.lowPass().length() > period) {
            builder.lowPass(base.lowPass());
        }
        return builder.build();
    }

    public static final class Builder {
        private final int[] periods;
        private int[] windows;
        private STL base;
        private int iterations = 2;
        private double lambda = Double.NaN;
        private Integer innerIterations;
        private Integer outerIterations;

        private Builder(int... periods) {
            this.periods = Objects.requireNonNull(periods, "periods").clone();
        }

        public Builder windows(int... windows) {
            this.windows = Objects.requireNonNull(windows, "windows").clone();
            return this;
        }

        public Builder base(STL base) {
            this.base = Objects.requireNonNull(base, "base");
            return this;
        }

        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        /**
         * Sets the Box-Cox lambda. Use {@link Double#NaN} to disable the transform.
         */
        public Builder lambda(double lambda) {
            if (!Double.isNaN(lambda) && !Double.isFinite(lambda)) {
                throw new IllegalArgumentException("lambda must be finite or NaN");
            }
            this.lambda = lambda;
            return this;
        }

        public Builder innerIterations(int innerIterations) {
            if (innerIterations < 1) {
                throw new IllegalArgumentException("innerIterations must be positive");
            }
            this.innerIterations = innerIterations;
            return this;
        }

        public Builder outerIterations(int outerIterations) {
            if (outerIterations < 0) {
                throw new IllegalArgumentException("outerIterations must be non-negative");
            }
            this.outerIterations = outerIterations;
            return this;
        }

        public MSTL build() {
            return new MSTL(this);
        }
    }
}
