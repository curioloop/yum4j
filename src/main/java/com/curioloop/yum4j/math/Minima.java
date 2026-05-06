package com.curioloop.yum4j.math;

import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

/**
 * Boost-style bounded minimization tools.
 *
 * <p>This utility mirrors the public surface of Boost.Math
 * {@code boost::math::tools::brent_find_minima}: a Brent search over a bounded interval
 * that returns the minimizer abscissa and the objective value as a pair.
 */
public value record Minima(double point, double value, int iterations, boolean converged) {

    private static final double GOLDEN_SECTION = 0.3819660112501051;

    public static Minima brentFindMinima(DoubleUnaryOperator function,
                                         double lower,
                                         double upper,
                                         int bits,
                                         int maxIterations) {
        
        Objects.requireNonNull(function, "function cannot be null");
        if (bits <= 0) {
            throw new IllegalArgumentException("bits must be positive: " + bits);
        }
        if (maxIterations <= 0) {
            throw new IllegalArgumentException("maxIterations must be positive: " + maxIterations);
        }
        if (!Double.isFinite(lower) || !Double.isFinite(upper)) {
            throw new IllegalArgumentException("brentFindMinima requires finite bounds");
        }
        if (lower == upper) {
            throw new IllegalArgumentException("brentFindMinima requires distinct bounds");
        }

        double min = Math.min(lower, upper);
        double max = Math.max(lower, upper);
        double tolerance = Math.scalb(1.0, 1 - bits);

        double x = max;
        double w = x;
        double v = x;
        double fx = evaluate(function, x);
        double fw = fx;
        double fv = fx;
        double delta = 0.0;
        double delta2 = 0.0;

        int remainingIterations = maxIterations;
        boolean converged = false;

        while (true) {
            double midpoint = (min + max) / 2.0;
            double fract1 = tolerance * Math.abs(x) + tolerance / 4.0;
            double fract2 = 2.0 * fract1;
            if (Math.abs(x - midpoint) <= fract2 - (max - min) / 2.0) {
                converged = true;
                break;
            }

            if (Math.abs(delta2) > fract1) {
                double r = (x - w) * (fx - fv);
                double q = (x - v) * (fx - fw);
                double p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                if (q > 0.0) {
                    p = -p;
                }
                q = Math.abs(q);
                double previousDelta2 = delta2;
                delta2 = delta;

                if ((Math.abs(p) >= Math.abs(q * previousDelta2 / 2.0))
                    || (p <= q * (min - x))
                    || (p >= q * (max - x))) {
                    delta2 = x >= midpoint ? min - x : max - x;
                    delta = GOLDEN_SECTION * delta2;
                } else {
                    delta = p / q;
                    double candidate = x + delta;
                    if ((candidate - min < fract2) || (max - candidate < fract2)) {
                        delta = midpoint - x < 0.0 ? -Math.abs(fract1) : Math.abs(fract1);
                    }
                }
            } else {
                delta2 = x >= midpoint ? min - x : max - x;
                delta = GOLDEN_SECTION * delta2;
            }

            double candidate = Math.abs(delta) >= fract1
                ? x + delta
                : (delta > 0.0 ? x + Math.abs(fract1) : x - Math.abs(fract1));
            double fCandidate = evaluate(function, candidate);

            if (fCandidate <= fx) {
                if (candidate >= x) {
                    min = x;
                } else {
                    max = x;
                }
                v = w;
                w = x;
                x = candidate;
                fv = fw;
                fw = fx;
                fx = fCandidate;
            } else {
                if (candidate < x) {
                    min = candidate;
                } else {
                    max = candidate;
                }
                if ((fCandidate <= fw) || (w == x)) {
                    v = w;
                    w = candidate;
                    fv = fw;
                    fw = fCandidate;
                } else if ((fCandidate <= fv) || (v == x) || (v == w)) {
                    v = candidate;
                    fv = fCandidate;
                }
            }

            if (--remainingIterations == 0) {
                break;
            }
        }

        return new Minima(x, fx, maxIterations - remainingIterations, converged);
    }

    private static double evaluate(DoubleUnaryOperator function, double x) {
        double value = function.applyAsDouble(x);
        if (!Double.isFinite(value)) {
            throw new ArithmeticException("Objective function returned a non-finite value during brentFindMinima");
        }
        return value;
    }

}