package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Beta;
import com.curioloop.yum4j.math.Gamma;
import com.curioloop.yum4j.math.Hypergeometric;
import com.curioloop.yum4j.math.RootIterators;

import java.util.function.DoubleUnaryOperator;

/** Boost-style non-central beta distribution object that owns the shared non-central beta kernel. */
public value record NonCentralBetaDistribution(double alpha, double beta, double nonCentrality)
    implements ContinuousDistribution {

    private static final double EPSILON = Math.ulp(1.0);
    private static final double MIN_PARAMETER = Double.MIN_NORMAL;
    private static final int DOUBLE_PRECISION_BITS = 53;
    private static final int MAX_SERIES_ITERATIONS = 100000;
    private static final int MAX_ROOT_ITERATIONS = 256;

    public NonCentralBetaDistribution {
        validateShape(alpha, "alpha");
        validateShape(beta, "beta");
        validateNonCentrality(nonCentrality);
    }

    public static double cdf(double a, double b, double nonCentrality, double x) {
        validateParameters(a, b, nonCentrality);
        validateX(x, "cdf");

        if (x == 0.0) {
            return 0.0;
        }
        if (x == 1.0) {
            return 1.0;
        }
        if (nonCentrality == 0.0) {
            return Beta.ibeta(a, b, x);
        }

        double y = 1.0 - x;
        double cross = crossover(a, b, nonCentrality);
        double value = x > cross
            ? 1.0 - nonCentralBetaQ(a, b, nonCentrality, x, y)
            : nonCentralBetaP(a, b, nonCentrality, x, y);
        return clampProbability(value);
    }

    public static double ccdf(double a, double b, double nonCentrality, double x) {
        validateParameters(a, b, nonCentrality);
        validateX(x, "ccdf");

        if (x == 0.0) {
            return 1.0;
        }
        if (x == 1.0) {
            return 0.0;
        }
        if (nonCentrality == 0.0) {
            return Beta.ibetac(a, b, x);
        }

        double y = 1.0 - x;
        double cross = crossover(a, b, nonCentrality);
        double value = x > cross
            ? nonCentralBetaQ(a, b, nonCentrality, x, y)
            : 1.0 - nonCentralBetaP(a, b, nonCentrality, x, y);
        return clampProbability(value);
    }

    public static double pdf(double a, double b, double nonCentrality, double x) {
        validateParameters(a, b, nonCentrality);
        validateX(x, "pdf");

        if (nonCentrality == 0.0) {
            return Beta.ibetaDerivative(a, b, x);
        }
        if (x == 0.0 || x == 1.0) {
            return 0.0;
        }

        double y = 1.0 - x;
        double halfLambda = 0.5 * nonCentrality;
        long k = (long) Math.floor(halfLambda);
        double poisson = Gamma.gammaPDerivative(k + 1.0, halfLambda);
        double beta = betaDerivative(a, b, k, x, y);

        while (Math.abs(beta * poisson) < Double.MIN_NORMAL) {
            if (k == 0 || poisson == 0.0) {
                return 0.0;
            }
            k /= 2;
            poisson = Gamma.gammaPDerivative(k + 1.0, halfLambda);
            beta = betaDerivative(a, b, k, x, y);
        }

        double sum = 0.0;
        double forwardPoisson = poisson;
        double forwardBeta = beta;

        long count = k;
        double ratio = 0.0;
        double previousRatio = 0.0;
        for (long i = k; i >= 0; i--) {
            double term = beta * poisson;
            sum += term;
            ratio = Math.abs(term / sum);
            if (((ratio < EPSILON) && (ratio < previousRatio)) || term == 0.0) {
                count = k - i;
                break;
            }
            previousRatio = ratio;
            poisson *= i / halfLambda;
            if (a + b + i != 1.0) {
                beta *= (a + i - 1.0) / (x * (a + b + i - 1.0));
            }
        }

        previousRatio = 0.0;
        for (long i = k + 1; ; i++) {
            forwardPoisson *= halfLambda / i;
            forwardBeta *= x * (a + b + i - 1.0) / (a + i - 1.0);
            double term = forwardPoisson * forwardBeta;
            sum += term;
            ratio = Math.abs(term / sum);
            if (((ratio < EPSILON) && (ratio < previousRatio)) || term == 0.0) {
                return sum;
            }
            previousRatio = ratio;
            if (count + i - k > MAX_SERIES_ITERATIONS) {
                throw new ArithmeticException("pdf(nonCentralBeta) series did not converge");
            }
        }
    }

    public static double quantile(double a, double b, double nonCentrality, double probability) {
        validateParameters(a, b, nonCentrality);
        validateProbability(probability, "quantile");

        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            return 1.0;
        }
        if (nonCentrality == 0.0) {
            return Beta.ibetaInv(a, b, probability);
        }

        return quantileImpl(a, b, nonCentrality, probability, false);
    }

    public static double quantileUpperTail(double a, double b, double nonCentrality, double probability) {
        validateParameters(a, b, nonCentrality);
        validateProbability(probability, "quantileUpperTail");

        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            return 1.0;
        }
        if (nonCentrality == 0.0) {
            return Beta.ibetacInv(a, b, probability);
        }

        return quantileImpl(a, b, nonCentrality, probability, true);
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, 1.0);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, 1.0);
    }

    @Override
    public double pdf(double x) {
        return pdf(alpha, beta, nonCentrality, x);
    }

    @Override
    public double cdf(double x) {
        return cdf(alpha, beta, nonCentrality, x);
    }

    @Override
    public double ccdf(double x) {
        return ccdf(alpha, beta, nonCentrality, x);
    }

    @Override
    public double quantile(double probability) {
        return quantile(alpha, beta, nonCentrality, probability);
    }

    @Override
    public double quantileUpperTail(double probability) {
        return quantileUpperTail(alpha, beta, nonCentrality, probability);
    }

    public double mode() {
        return StatisticsModeSupport.unitIntervalMode(this, crossover(alpha, beta, nonCentrality));
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        if (nonCentrality == 0.0) {
            return alpha / (alpha + beta);
        }
        double alphaPlusBeta = alpha + beta;
        return Math.exp(-nonCentrality / 2.0)
            * alpha
            * Hypergeometric.pFq(
                new double[]{1.0 + alpha, alphaPlusBeta},
                new double[]{alpha, 1.0 + alphaPlusBeta},
                nonCentrality / 2.0
            )
            / alphaPlusBeta;
    }

    @Override
    public double variance() {
        if (nonCentrality == 0.0) {
            double alphaPlusBeta = alpha + beta;
            return alpha * beta / (alphaPlusBeta * alphaPlusBeta * (alphaPlusBeta + 1.0));
        }
        double alphaPlusBeta = alpha + beta;
        double firstMomentSeries = Hypergeometric.pFq(
            new double[]{1.0 + alpha, alphaPlusBeta},
            new double[]{alpha, 1.0 + alphaPlusBeta},
            nonCentrality / 2.0
        );
        double result = firstMomentSeries * firstMomentSeries * -Math.exp(-nonCentrality) * alpha * alpha
            / (alphaPlusBeta * alphaPlusBeta);
        result += Math.exp(-nonCentrality / 2.0)
            * alpha
            * (1.0 + alpha)
            * Hypergeometric.pFq(
                new double[]{2.0 + alpha, alphaPlusBeta},
                new double[]{alpha, 2.0 + alphaPlusBeta},
                nonCentrality / 2.0
            )
            / (alphaPlusBeta * (1.0 + alphaPlusBeta));
        return result;
    }

    public double skewness() {
        throw new ArithmeticException("Skewness is undefined for NonCentralBetaDistribution");
    }

    public double kurtosisExcess() {
        throw new ArithmeticException("Kurtosis excess is undefined for NonCentralBetaDistribution");
    }

    public double kurtosis() {
        return kurtosisExcess() + 3.0;
    }

    private static double quantileImpl(double a,
                                       double b,
                                       double nonCentrality,
                                       double probability,
                                       boolean complement) {
        double mean = crossover(a, b, nonCentrality);
        double guess = Math.min(1.0 - MIN_PARAMETER, Math.max(MIN_PARAMETER, mean));
        DoubleUnaryOperator objective = complement
            ? x -> probability - ccdf(a, b, nonCentrality, x)
            : x -> cdf(a, b, nonCentrality, x) - probability;

        return RootIterators.toms748Solve01(
            objective,
            guess,
            2.5,
            true,
            DOUBLE_PRECISION_BITS,
            MAX_ROOT_ITERATIONS,
            complement ? "quantileUpperTail(nonCentralBeta)" : "quantile(nonCentralBeta)"
        );
    }

    private static double nonCentralBetaP(double a, double b, double nonCentrality, double x, double y) {
        double halfLambda = 0.5 * nonCentrality;
        long k = (long) Math.floor(halfLambda);
        if (k == 0) {
            k = 1;
        }
        double poisson = Gamma.gammaPDerivative(k + 1.0, halfLambda);
        if (poisson == 0.0) {
            return 0.0;
        }

        double beta = lowerBeta(a, b, k, x, y);
        while (Math.abs(beta * poisson) < Double.MIN_NORMAL) {
            if (k == 0 || poisson == 0.0) {
                return 0.0;
            }
            k /= 2;
            poisson = Gamma.gammaPDerivative(k + 1.0, halfLambda);
            beta = lowerBeta(a, b, k, x, y);
        }

        double xTerm = betaIncrement(a, b, k, x, y);
        double forwardPoisson = poisson;
        double forwardBeta = beta;
        double forwardXTerm = xTerm;
        double sum = 0.0;

        double lastTerm = 0.0;
        long count = k;
        for (long i = k; i >= 0; i--) {
            double term = beta * poisson;
            sum += term;
            if (converged(term, sum, lastTerm)) {
                count = k - i;
                break;
            }
            poisson *= i / halfLambda;
            beta += xTerm;
            if (a + b + i != 2.0) {
                xTerm *= (a + i - 1.0) / (x * (a + b + i - 2.0));
            }
            lastTerm = term;
        }

        lastTerm = 0.0;
        double forwardBetaLimit = forwardBeta * EPSILON * 4.0;
        for (long i = k + 1; ; i++) {
            forwardPoisson *= halfLambda / i;
            forwardXTerm *= x * (a + b + i - 2.0) / (a + i - 1.0);
            forwardBeta -= forwardXTerm;
            if (forwardBeta < forwardBetaLimit) {
                return sum;
            }

            double term = forwardPoisson * forwardBeta;
            sum += term;
            if (converged(term, sum, lastTerm)) {
                return sum;
            }
            lastTerm = term;
            if (count + i - k > MAX_SERIES_ITERATIONS) {
                throw new ArithmeticException("cdf(nonCentralBeta) lower series did not converge");
            }
        }
    }

    private static double nonCentralBetaQ(double a, double b, double nonCentrality, double x, double y) {
        double halfLambda = 0.5 * nonCentrality;
        long k = (long) Math.floor(halfLambda);
        if (k <= 30) {
            if (a + b > 1.0) {
                k = 0;
            } else if (k == 0) {
                k = 1;
            }
        }

        double poisson = k == 0 ? Math.exp(-halfLambda) : Gamma.gammaPDerivative(k + 1.0, halfLambda);
        if (poisson == 0.0) {
            return 0.0;
        }

        double beta = upperBeta(a, b, k, x, y);
        double xTerm = betaIncrement(a, b, k, x, y);
        double forwardPoisson = poisson;
        double forwardBeta = beta;
        double forwardXTerm = xTerm;
        double sum = 0.0;

        if (beta == 0.0 && xTerm == 0.0) {
            return 0.0;
        }

        double lastTerm = 0.0;
        long count = 0;
        for (long i = k + 1; ; i++) {
            forwardPoisson *= halfLambda / i;
            forwardXTerm *= x * (a + b + i - 2.0) / (a + i - 1.0);
            forwardBeta += forwardXTerm;
            double term = forwardPoisson * forwardBeta;
            sum += term;
            if ((relativeTerm(term, sum) < EPSILON) && (lastTerm >= term)) {
                count = i - k;
                break;
            }
            lastTerm = term;
            if (i - k > MAX_SERIES_ITERATIONS) {
                throw new ArithmeticException("ccdf(nonCentralBeta) upper series did not converge");
            }
        }

        for (long i = k; i >= 0; i--) {
            double term = beta * poisson;
            sum += term;
            if (relativeTerm(term, sum) < EPSILON) {
                return sum;
            }
            if (count + k - i > MAX_SERIES_ITERATIONS) {
                throw new ArithmeticException("ccdf(nonCentralBeta) backward series did not converge");
            }
            poisson *= i / halfLambda;
            beta -= xTerm;
            if (a + b + i - 2.0 != 0.0) {
                xTerm *= (a + i - 1.0) / (x * (a + b + i - 2.0));
            }
        }
        return sum;
    }

    private static double lowerBeta(double a, double b, long k, double x, double y) {
        return x < y ? Beta.ibeta(a + k, b, x) : Beta.ibetac(b, a + k, y);
    }

    private static double upperBeta(double a, double b, long k, double x, double y) {
        return x < y ? Beta.ibetac(a + k, b, x) : Beta.ibeta(b, a + k, y);
    }

    private static double betaDerivative(double a, double b, long k, double x, double y) {
        return x < y ? Beta.ibetaDerivative(a + k, b, x) : Beta.ibetaDerivative(b, a + k, y);
    }

    private static double betaIncrement(double a, double b, long k, double x, double y) {
        return betaDerivative(a, b, k, x, y) * y / (a + b + k - 1.0);
    }

    private static double crossover(double a, double b, double nonCentrality) {
        double c = a + b + 0.5 * nonCentrality;
        return 1.0 - (b / c) * (1.0 + nonCentrality / (2.0 * c * c));
    }

    private static boolean converged(double term, double sum, double lastTerm) {
        return (relativeTerm(term, sum) < EPSILON && Math.abs(lastTerm) >= Math.abs(term)) || term == 0.0;
    }

    private static double relativeTerm(double term, double sum) {
        if (sum == 0.0) {
            return Math.abs(term);
        }
        return Math.abs(term / sum);
    }

    private static double clampProbability(double value) {
        if (value <= 0.0) {
            return 0.0;
        }
        if (value >= 1.0) {
            return 1.0;
        }
        return value;
    }

    private static void validateParameters(double a, double b, double nonCentrality) {
        validateShape(a, "a");
        validateShape(b, "b");
        if (!Double.isFinite(nonCentrality) || nonCentrality < 0.0) {
            throw new IllegalArgumentException("nonCentrality must be finite and non-negative: " + nonCentrality);
        }
    }

    private static void validateShape(double value, String label) {
        if (!Double.isFinite(value) || !(value > 0.0)) {
            throw new IllegalArgumentException(label + " must be finite and positive: " + value);
        }
    }

    private static void validateNonCentrality(double nonCentrality) {
        if (Double.isNaN(nonCentrality) || nonCentrality < 0.0 || Double.isInfinite(nonCentrality)) {
            throw new IllegalArgumentException(
                "nonCentrality must be finite and in [0, +infinity): " + nonCentrality
            );
        }
    }

    private static void validateX(double x, String functionName) {
        if (!Double.isFinite(x) || x < 0.0 || x > 1.0) {
            throw new IllegalArgumentException(functionName + ": x must be in [0, 1]: " + x);
        }
    }

    private static void validateProbability(double probability, String functionName) {
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException(functionName + ": probability must be in [0, 1]: " + probability);
        }
    }
}