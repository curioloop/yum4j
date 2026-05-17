package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.random.RandomGenerator;

/**
 * Transformed distributions for parameter estimation.
 *
 * <p>Given a base distribution on X and a bijective transform Y = f(X),
 * this class computes the induced distribution on Y using the change-of-variables
 * formula: {@code p_Y(y) = p_X(f^{-1}(y)) · |d(f^{-1})/dy|}.
 *
 * <p>Concrete subclasses implement the forward transform {@link #f(double)},
 * its inverse {@link #finv(double)}, and the log absolute Jacobian
 * {@link #logJac(double)} of the inverse.
 */
public abstract class TransformedDist implements ContinuousDistribution {

    protected final ContinuousDistribution baseDist;

    protected TransformedDist(ContinuousDistribution baseDist) {
        if (baseDist == null) throw new IllegalArgumentException("baseDist must not be null");
        this.baseDist = baseDist;
    }

    /** Forward transform: Y = f(X). */
    public abstract double f(double x);

    /** Inverse transform: X = f^{-1}(Y). */
    public abstract double finv(double y);

    /** Log absolute Jacobian: log |d(f^{-1})/dy|. */
    public abstract double logJac(double y);

    @Override
    public double logPdf(double y) {
        return baseDist.logPdf(finv(y)) + logJac(y);
    }

    @Override
    public double pdf(double y) {
        return Math.exp(logPdf(y));
    }

    @Override
    public double sample(RandomGenerator g) {
        return f(baseDist.sample(g));
    }

    @Override
    public double quantile(double p) {
        return f(baseDist.quantile(p));
    }

    @Override
    public double quantileUpperTail(double p) {
        return f(baseDist.quantileUpperTail(p));
    }

    @Override
    public double cdf(double y) {
        return baseDist.cdf(finv(y));
    }

    @Override
    public double ccdf(double y) {
        return baseDist.ccdf(finv(y));
    }

    @Override
    public double mean() {
        // No closed-form in general; delegate to base if linear.
        throw new UnsupportedOperationException(
            "mean() not available for general transformed distributions");
    }

    @Override
    public double variance() {
        throw new UnsupportedOperationException(
            "variance() not available for general transformed distributions");
    }

    @Override
    public Double2 range() {
        Double2 baseRange = baseDist.range();
        double lo = f(baseRange.lower());
        double hi = f(baseRange.upper());
        return lo <= hi ? Double2.bound(lo, hi) : Double2.bound(hi, lo);
    }

    @Override
    public Double2 support() {
        Double2 baseSupport = baseDist.support();
        double lo = f(baseSupport.lower());
        double hi = f(baseSupport.upper());
        return lo <= hi ? Double2.bound(lo, hi) : Double2.bound(hi, lo);
    }

    // ------------------------------------------------------------------
    // Concrete subclasses
    // ------------------------------------------------------------------

    /**
     * Linear transform: Y = a*X + b.
     * Inverse: X = (Y - b) / a.
     * logJac = log(1/|a|) = -log(|a|).
     */
    public static final class LinearDist extends TransformedDist {
        private final double a;
        private final double b;
        private final double logAbsA;

        public LinearDist(ContinuousDistribution baseDist, double a, double b) {
            super(baseDist);
            if (a == 0.0) throw new IllegalArgumentException("a must be non-zero");
            this.a = a;
            this.b = b;
            this.logAbsA = Math.log(Math.abs(a));
        }

        @Override public double f(double x) { return a * x + b; }
        @Override public double finv(double y) { return (y - b) / a; }
        @Override public double logJac(double y) { return -logAbsA; }

        @Override
        public double mean() { return a * baseDist.mean() + b; }

        @Override
        public double variance() { return a * a * baseDist.variance(); }

        @Override
        public double cdf(double y) {
            return a > 0 ? baseDist.cdf(finv(y)) : baseDist.ccdf(finv(y));
        }

        @Override
        public double ccdf(double y) {
            return a > 0 ? baseDist.ccdf(finv(y)) : baseDist.cdf(finv(y));
        }
    }

    /**
     * Log transform: Y = log(X).
     * Inverse: X = exp(Y).
     * logJac = log|d(exp(y))/dy| = y.
     */
    public static final class LogDist extends TransformedDist {

        public LogDist(ContinuousDistribution baseDist) {
            super(baseDist);
        }

        @Override public double f(double x) { return Math.log(x); }
        @Override public double finv(double y) { return Math.exp(y); }
        @Override public double logJac(double y) { return y; }
    }

    /**
     * Logit transform: Y = logit((X - a) / (b - a)).
     * Maps X ∈ (a, b) to Y ∈ (-∞, +∞).
     * Inverse: X = a + (b - a) * sigmoid(Y).
     * logJac = log|(b-a) * sigmoid(y) * (1 - sigmoid(y))|.
     */
    public static final class LogitDist extends TransformedDist {
        private final double a;
        private final double b;
        private final double range; // b - a
        private final double logRange;

        public LogitDist(ContinuousDistribution baseDist, double a, double b) {
            super(baseDist);
            if (!(b > a)) throw new IllegalArgumentException("b must be > a: a=" + a + " b=" + b);
            this.a = a;
            this.b = b;
            this.range = b - a;
            this.logRange = Math.log(range);
        }

        @Override
        public double f(double x) {
            double u = (x - a) / range;
            return Math.log(u / (1.0 - u));
        }

        @Override
        public double finv(double y) {
            double sig = 1.0 / (1.0 + Math.exp(-y));
            return a + range * sig;
        }

        @Override
        public double logJac(double y) {
            // |d(finv)/dy| = range * sigmoid(y) * (1 - sigmoid(y))
            // log of that = logRange + log(sigmoid(y)) + log(1 - sigmoid(y))
            // = logRange - softplus(y) - softplus(-y)
            // = logRange + y - 2*softplus(y)  [since softplus(-y) = softplus(y) - y]
            // Numerically stable: log(sig) = -softplus(-y), log(1-sig) = -softplus(y)
            double sp = softplus(y);
            double spNeg = softplus(-y);
            return logRange - sp - spNeg;
        }

        private static double softplus(double x) {
            if (x > 20.0) return x;
            if (x < -20.0) return 0.0;
            return Math.log1p(Math.exp(x));
        }
    }
}
