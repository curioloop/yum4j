/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.da;

import com.curioloop.yum4j.math.Gamma;
import com.curioloop.yum4j.optim.Bound;

import java.util.Random;

/**
 * Distorted Cauchy-Lorentz visiting distribution used by dual annealing.
 *
 * <p>The visiting parameter {@code qv} controls tail heaviness. Values near 1 produce
 * shorter jumps; values near 3 produce very heavy tails. The implementation follows the
 * SciPy formula and clamps extreme sampled jumps to keep bounded wrapping numerically stable.</p>
 */
final class VisitingDistribution {

    static final double TAIL_LIMIT = 1.0e8;
    static final double MIN_VISIT_BOUND = 1.0e-10;

    private final double visitingParam;
    private final double factor4p;
    private final double factor6;

    /** Precomputes constants that depend only on the visiting parameter. */
    VisitingDistribution(double visitingParam) {
        this.visitingParam = effectiveVisitingParam(visitingParam);
        double q = this.visitingParam;
        double factor2 = Math.exp((4.0 - q) * Math.log(q - 1.0));
        double factor3 = Math.exp((2.0 - q) * Math.log(2.0) / (q - 1.0));
        this.factor4p = Math.sqrt(Math.PI) * factor2 / (factor3 * (3.0 - q));
        double factor5 = 1.0 / (q - 1.0) - 0.5;
        double d1 = 2.0 - factor5;
        this.factor6 = Math.PI * (1.0 - factor5)
                / Math.sin(Math.PI * (1.0 - factor5))
                / Math.exp(Gamma.lgamma(d1));
    }

    /** Maps SciPy's accepted endpoint {@code qv=3} to the closest finite non-singular value. */
    static double effectiveVisitingParam(double visitingParam) {
        return visitingParam == 3.0 ? Math.nextDown(3.0) : visitingParam;
    }

    double visitingParam() {
        return visitingParam;
    }

    /**
     * Generates one proposal using SciPy's two-phase coordinate schedule.
     *
     * <p>For the first {@code n} inner steps every coordinate receives a jump. For the next
     * {@code n} steps only one coordinate is changed. Values are wrapped back into their
     * finite bounds instead of clipped, preserving the annealing proposal's displacement.</p>
     */
    void visiting(double[] x,
                  int step,
                  double temperatureScale,
                  Bound[] bounds,
                  double[] out,
                  Random rng) {
        int n = x.length;
        if (step < n) {
            for (int i = 0; i < n; i++) {
                double visit = boundedVisit(temperatureScale, rng);
                double lower = bounds[i].lower();
                out[i] = wrap(x[i] + visit, lower, bounds[i].upper() - lower);
                if (Math.abs(out[i] - lower) < MIN_VISIT_BOUND) out[i] += MIN_VISIT_BOUND;
            }
        } else {
            System.arraycopy(x, 0, out, 0, n);
            int index = step - n;
            double visit = boundedVisit(temperatureScale, rng);
            double lower = bounds[index].lower();
            out[index] = wrap(x[index] + visit, lower, bounds[index].upper() - lower);
            if (Math.abs(out[index] - lower) < MIN_VISIT_BOUND) out[index] += MIN_VISIT_BOUND;
        }
    }

    /** Converts the annealing temperature into the scale parameter used by {@link #visit}. */
    double temperatureScale(double temperature) {
        double q = visitingParam;
        double factor1 = Math.exp(Math.log(temperature) / (q - 1.0));
        double factor4 = factor4p * factor1;
        return Math.exp(-(q - 1.0) * Math.log(factor6 / factor4) / (3.0 - q));
    }

    /** Samples a jump and limits pathological tails before bound wrapping. */
    private double boundedVisit(double temperatureScale, Random rng) {
        double visit = visit(temperatureScale, rng);
        if (visit > TAIL_LIMIT) return TAIL_LIMIT * rng.nextDouble();
        if (visit < -TAIL_LIMIT) return -TAIL_LIMIT * rng.nextDouble();
        return visit;
    }

    /** Draws a raw distorted Cauchy-Lorentz jump from two Gaussian variates. */
    private double visit(double temperatureScale, Random rng) {
        double q = visitingParam;
        double x = rng.nextGaussian();
        double y;
        do {
            y = rng.nextGaussian();
        } while (y == 0.0);

        double denominator = Math.exp((q - 1.0) * Math.log(Math.abs(y)) / (3.0 - q));
        return x * temperatureScale / denominator;
    }

    /** Wraps a coordinate periodically into {@code [lower, lower + range)}. */
    private static double wrap(double value, double lower, double range) {
        double shifted = value - lower;
        double wrapped = ((shifted % range) + range) % range;
        return lower + wrapped;
    }
}