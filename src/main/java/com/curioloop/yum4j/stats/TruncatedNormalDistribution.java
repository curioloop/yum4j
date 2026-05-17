package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;
import com.curioloop.yum4j.math.Normal;

import java.util.random.RandomGenerator;

/**
 * {@code Normal(mu, sigma^2)} truncated to the interval {@code [a, b]}.
 * Either bound may be infinite: pass {@link Double#NEGATIVE_INFINITY}
 * for {@code a} to indicate a one-sided {@code (-infinity, b]} interval,
 * and {@link Double#POSITIVE_INFINITY} for {@code b} for {@code [a, +infinity)}.
 *
 * <p>Sampling uses the Robert (1995) rejection algorithm:
 * <ul>
 *   <li>When both bounds are infinite, falls back to a plain
 *       {@link RandomGenerator#nextGaussian()} draw (trivial case).</li>
 *   <li>When {@code alpha <= 0 <= beta} (standardised bounds), uses
 *       standard Normal rejection.</li>
 *   <li>When {@code alpha > 0} (left-truncated far from the mode), uses
 *       an exponential envelope with optimal rate
 *       {@code lambda* = (alpha + sqrt(alpha^2 + 4)) / 2}.</li>
 *   <li>When {@code beta < 0} (right-truncated far below the mode),
 *       reflects the {@code alpha > 0} branch: draws on
 *       {@code [-beta, -alpha]} then negates.</li>
 * </ul>
 *
 * <p>This is the branch-for-branch form used by the reference
 * {@code particles.distributions.TruncNormal} fixtures.
 */
public value record TruncatedNormalDistribution(double mu, double sigma, double a, double b)
    implements ContinuousDistribution {

    public TruncatedNormalDistribution {
        if (Double.isNaN(mu) || !Double.isFinite(mu)) {
            throw new IllegalArgumentException("mu must be finite: " + mu);
        }
        if (Double.isNaN(sigma) || !(sigma > 0.0) || !Double.isFinite(sigma)) {
            throw new IllegalArgumentException("sigma must be finite and positive: " + sigma);
        }
        if (Double.isNaN(a) || Double.isNaN(b)) {
            throw new IllegalArgumentException(
                "bounds must not be NaN: a=" + a + ", b=" + b
            );
        }
        if (!(a < b)) {
            throw new IllegalArgumentException(
                "require a < b: a=" + a + ", b=" + b
            );
        }
    }

    /** Lower standardised bound {@code (a - mu) / sigma}. */
    private double alpha() {
        return (a - mu) / sigma;
    }

    /** Upper standardised bound {@code (b - mu) / sigma}. */
    private double beta() {
        return (b - mu) / sigma;
    }

    /** Log of the normalising mass {@code Phi(beta) - Phi(alpha)}. */
    private double logZ() {
        double alpha = alpha();
        double beta = beta();
        double phiA = Double.isInfinite(alpha) && alpha < 0.0 ? 0.0 : Normal.cdf(alpha);
        double phiB = Double.isInfinite(beta) && beta > 0.0 ? 1.0 : Normal.cdf(beta);
        double z = phiB - phiA;
        return Math.log(z);
    }

    private double zMass() {
        double alpha = alpha();
        double beta = beta();
        double phiA = Double.isInfinite(alpha) && alpha < 0.0 ? 0.0 : Normal.cdf(alpha);
        double phiB = Double.isInfinite(beta) && beta > 0.0 ? 1.0 : Normal.cdf(beta);
        return phiB - phiA;
    }

    @Override
    public Double2 range() {
        return Double2.bound(a, b);
    }

    @Override
    public Double2 support() {
        return Double2.bound(a, b);
    }

    @Override
    public double pdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        if (x < a || x > b) {
            return 0.0;
        }
        double z = (x - mu) / sigma;
        return Normal.pdf(z) / (sigma * zMass());
    }

    @Override
    public double cdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        if (x <= a) {
            return 0.0;
        }
        if (x >= b) {
            return 1.0;
        }
        double alpha = alpha();
        double z = zMass();
        double phiA = Double.isInfinite(alpha) && alpha < 0.0 ? 0.0 : Normal.cdf(alpha);
        return (Normal.cdf((x - mu) / sigma) - phiA) / z;
    }

    @Override
    public double ccdf(double x) {
        return 1.0 - cdf(x);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) return a;
        if (probability == 1.0) return b;
        double alpha = alpha();
        double phiA = Double.isInfinite(alpha) && alpha < 0.0 ? 0.0 : Normal.cdf(alpha);
        double z = zMass();
        double target = phiA + probability * z;
        return mu + sigma * Normal.inv(target);
    }

    @Override
    public double quantileUpperTail(double probability) {
        return quantile(1.0 - probability);
    }

    @Override
    public double mean() {
        double alpha = alpha();
        double beta = beta();
        double z = zMass();
        double phiAlpha = Double.isInfinite(alpha) ? 0.0 : Normal.pdf(alpha);
        double phiBeta = Double.isInfinite(beta) ? 0.0 : Normal.pdf(beta);
        return mu + sigma * (phiAlpha - phiBeta) / z;
    }

    @Override
    public double variance() {
        double alpha = alpha();
        double beta = beta();
        double z = zMass();
        double phiAlpha = Double.isInfinite(alpha) ? 0.0 : Normal.pdf(alpha);
        double phiBeta = Double.isInfinite(beta) ? 0.0 : Normal.pdf(beta);
        // Handle infinite endpoints: if alpha -> -inf, alpha * phi(alpha) -> 0.
        double aPhi = Double.isInfinite(alpha) ? 0.0 : alpha * phiAlpha;
        double bPhi = Double.isInfinite(beta) ? 0.0 : beta * phiBeta;
        double ratio = (phiAlpha - phiBeta) / z;
        return sigma * sigma * (1.0 + (aPhi - bPhi) / z - ratio * ratio);
    }

    // ---------------------------------------------------------------
    // Batch / Sampling specialisations
    // ---------------------------------------------------------------

    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x) || x < a || x > b) {
            return Double.NEGATIVE_INFINITY;
        }
        double z = (x - mu) / sigma;
        return Normal.logPdf(z) - Math.log(sigma) - logZ();
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n,
                            double[] out, int outOff) {
        if (metric == Metric.LOG_PDF) {
            if (n == 0) return;
            final double lo = a;
            final double hi = b;
            final double m = mu;
            final double s = sigma;
            final double invS = 1.0 / s;
            final double shift = -Math.log(s) - logZ();
            for (int i = 0; i < n; i++) {
                double v = x[xOff + i * xStride];
                double r;
                if (Double.isNaN(v) || v < lo || v > hi) {
                    r = Double.NEGATIVE_INFINITY;
                } else {
                    double z = (v - m) * invS;
                    r = Normal.logPdf(z) + shift;
                }
                out[outOff + i] = r;
            }
        } else {
            ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    @Override
    public double sample(RandomGenerator g) {
        double alpha = alpha();
        double beta = beta();
        double standard = drawStandardised(g, alpha, beta);
        return mu + sigma * standard;
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        final double alpha = alpha();
        final double beta = beta();
        final double m = mu;
        final double s = sigma;
        final int branch = selectBranch(alpha, beta);
        double lambdaStar = 0.0;
        double negAlpha = 0.0;
        double negBeta = 0.0;
        if (branch == BRANCH_LEFT_TAIL) {
            lambdaStar = (alpha + Math.sqrt(alpha * alpha + 4.0)) / 2.0;
        } else if (branch == BRANCH_RIGHT_TAIL) {
            negAlpha = -beta;
            negBeta = -alpha;
            lambdaStar = (negAlpha + Math.sqrt(negAlpha * negAlpha + 4.0)) / 2.0;
        }
        for (int i = 0; i < n; i++) {
            double standard;
            switch (branch) {
                case BRANCH_BOTH_INFINITE:
                    standard = g.nextGaussian();
                    break;
                case BRANCH_CENTRAL:
                    standard = drawCentralRejection(g, alpha, beta);
                    break;
                case BRANCH_LEFT_TAIL:
                    standard = drawLeftTail(g, alpha, beta, lambdaStar);
                    break;
                case BRANCH_RIGHT_TAIL:
                    standard = -drawLeftTail(g, negAlpha, negBeta, lambdaStar);
                    break;
                default:
                    throw new AssertionError("unreachable branch: " + branch);
            }
            out[off + i * stride] = m + s * standard;
        }
    }

    // ---------------------------------------------------------------
    // Robert (1995) standardised draw
    // ---------------------------------------------------------------

    private static final int BRANCH_BOTH_INFINITE = 0;
    private static final int BRANCH_CENTRAL = 1;
    private static final int BRANCH_LEFT_TAIL = 2;
    private static final int BRANCH_RIGHT_TAIL = 3;

    private static int selectBranch(double alpha, double beta) {
        boolean aNegInf = Double.isInfinite(alpha) && alpha < 0.0;
        boolean bPosInf = Double.isInfinite(beta) && beta > 0.0;
        if (aNegInf && bPosInf) {
            return BRANCH_BOTH_INFINITE;
        }
        if (alpha > 0.0) {
            return BRANCH_LEFT_TAIL;
        }
        if (beta < 0.0) {
            return BRANCH_RIGHT_TAIL;
        }
        // alpha <= 0 <= beta (one or both may be infinite on an unbounded side)
        return BRANCH_CENTRAL;
    }

    private static double drawStandardised(RandomGenerator g, double alpha, double beta) {
        int branch = selectBranch(alpha, beta);
        switch (branch) {
            case BRANCH_BOTH_INFINITE:
                return g.nextGaussian();
            case BRANCH_CENTRAL:
                return drawCentralRejection(g, alpha, beta);
            case BRANCH_LEFT_TAIL: {
                double lambdaStar = (alpha + Math.sqrt(alpha * alpha + 4.0)) / 2.0;
                return drawLeftTail(g, alpha, beta, lambdaStar);
            }
            case BRANCH_RIGHT_TAIL: {
                double negAlpha = -beta;
                double negBeta = -alpha;
                double lambdaStar = (negAlpha + Math.sqrt(negAlpha * negAlpha + 4.0)) / 2.0;
                return -drawLeftTail(g, negAlpha, negBeta, lambdaStar);
            }
            default:
                throw new AssertionError("unreachable branch: " + branch);
        }
    }

    private static double drawCentralRejection(RandomGenerator g, double alpha, double beta) {
        while (true) {
            double z = g.nextGaussian();
            if (z >= alpha && z <= beta) {
                return z;
            }
        }
    }

    private static double drawLeftTail(RandomGenerator g, double alpha, double beta, double lambdaStar) {
        while (true) {
            double u1 = g.nextDouble();
            double x = alpha - Math.log(u1) / lambdaStar;
            double u2 = g.nextDouble();
            double diff = x - lambdaStar;
            double rho = Math.exp(-0.5 * diff * diff);
            if (u2 <= rho) {
                if (x <= beta) {
                    return x;
                }
            }
        }
    }

    // ---------------------------------------------------------------

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}
