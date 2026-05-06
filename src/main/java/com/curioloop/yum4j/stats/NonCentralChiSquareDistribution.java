package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;
import com.curioloop.yum4j.math.Bessel;
import com.curioloop.yum4j.math.Gamma;
import com.curioloop.yum4j.math.Minima;
import com.curioloop.yum4j.math.RootIterators;

import java.util.function.DoubleUnaryOperator;

/**
 * Non-central chi-square distribution with Boost-aligned evaluation branches.
 */
public value record NonCentralChiSquareDistribution(double degreesOfFreedom, double nonCentrality)
    implements ContinuousDistribution {

    private static final double SMALL_NON_CENTRALITY_CUTOFF = 200.0;
    private static final double PDF_BESSEL_CUTOFF = 50.0;
    private static final double RELATIVE_TOLERANCE = Math.ulp(1.0);
    private static final double THOMAS_LUU_GUESS_CUTOFF = 0.005;
    private static final double MIN_PARAMETER = 1.0e-12;
    private static final double MIN_POSITIVE_GUESS = Double.MIN_NORMAL;
    private static final double LOG_MAX_VALUE = Math.log(Double.MAX_VALUE);
    private static final int MAX_ITERATIONS = 1000000;
    private static final int MAX_ROOT_ITERATIONS = 256;
    private static final int ROOT_BITS = 50;
    private static final double MAX_NON_CENTRALITY = (double) Long.MAX_VALUE - MAX_ROOT_ITERATIONS;
    private static final int MODE_MINIMIZER_BITS = 53;

    public NonCentralChiSquareDistribution {
        validateParameters(degreesOfFreedom, nonCentrality);
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, Double.MAX_VALUE);
    }

    public double cdf(double x) {
        return cdf(degreesOfFreedom, nonCentrality, x);
    }

    public double ccdf(double x) {
        return ccdf(degreesOfFreedom, nonCentrality, x);
    }

    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return 0.0;
        }
        if (probability == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        return solveQuantile(probability);
    }

    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 1.0) {
            return 0.0;
        }
        if (probability == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return solveUpperTailQuantile(probability);
    }

    public double pdf(double x) {
        return pdf(degreesOfFreedom, nonCentrality, x);
    }

    public double median() {
        return quantile(0.5);
    }

    public double mode() {
        return mode(degreesOfFreedom, nonCentrality);
    }

    public double mean() {
        return degreesOfFreedom + nonCentrality;
    }

    public double variance() {
        return 2.0 * (degreesOfFreedom + 2.0 * nonCentrality);
    }

    public double skewness() {
        double variance = variance();
        return 8.0 * (degreesOfFreedom + 3.0 * nonCentrality)
            / Math.pow(variance, 1.5);
    }

    public double kurtosis() {
        return 3.0 + kurtosisExcess();
    }

    public double kurtosisExcess() {
        double base = degreesOfFreedom + 2.0 * nonCentrality;
        return 12.0 * (degreesOfFreedom + 4.0 * nonCentrality) / (base * base);
    }

    public static double cdf(double degreesOfFreedom, double nonCentrality, double x) {
        validateParameters(degreesOfFreedom, nonCentrality);
        validateX(x);
        return clampProbability(nonCentralChiSquaredCdf(x, degreesOfFreedom, nonCentrality, false));
    }

    public static double ccdf(double degreesOfFreedom, double nonCentrality, double x) {
        validateParameters(degreesOfFreedom, nonCentrality);
        validateX(x);
        return clampProbability(nonCentralChiSquaredCdf(x, degreesOfFreedom, nonCentrality, true));
    }

    public static double pdf(double degreesOfFreedom, double nonCentrality, double x) {
        validateParameters(degreesOfFreedom, nonCentrality);
        validateX(x);

        if (nonCentrality == 0.0) {
            return 0.5 * Gamma.gammaPDerivative(0.5 * degreesOfFreedom, 0.5 * x);
        }
        if (x == 0.0) {
            return 0.0;
        }
        if (nonCentrality > PDF_BESSEL_CUTOFF) {
            return nonCentralChiSquarePdfSeries(x, degreesOfFreedom, nonCentrality);
        }

        double logPrefix = Math.log(x / nonCentrality) * (degreesOfFreedom / 4.0 - 0.5)
            - (x + nonCentrality) / 2.0;
        if (Math.abs(logPrefix) >= LOG_MAX_VALUE / 4.0) {
            return nonCentralChiSquarePdfSeries(x, degreesOfFreedom, nonCentrality);
        }

        double argument = Math.sqrt(nonCentrality * x);
        double bessel = Bessel.i(degreesOfFreedom / 2.0 - 1.0, argument);
        return 0.5 * Math.exp(logPrefix) * bessel;
    }

    public static double quantile(double degreesOfFreedom, double nonCentrality, double probability) {
        return new NonCentralChiSquareDistribution(degreesOfFreedom, nonCentrality).quantile(probability);
    }

    public static double quantileUpperTail(double degreesOfFreedom, double nonCentrality, double probability) {
        return new NonCentralChiSquareDistribution(degreesOfFreedom, nonCentrality).quantileUpperTail(probability);
    }

    private double solveQuantile(double probability) {
        double guess = initialQuantileGuess(degreesOfFreedom, nonCentrality, probability, false);
        return RootIterators.toms748SolveBounded(
            x -> cdf(x) - probability,
            guess,
            2.0,
            true,
            0.0,
            ROOT_BITS,
            MAX_ROOT_ITERATIONS,
            "quantile(nonCentralChiSquareDistribution)"
        );
    }

    private double solveUpperTailQuantile(double probability) {
        double guess = initialQuantileGuess(degreesOfFreedom, nonCentrality, probability, true);
        return RootIterators.toms748SolveBounded(
            x -> probability - ccdf(x),
            guess,
            2.0,
            true,
            0.0,
            ROOT_BITS,
            MAX_ROOT_ITERATIONS,
            "quantile(complement(nonCentralChiSquareDistribution))"
        );
    }

    public static double findDegreesOfFreedom(double nonCentrality, double x, double probability) {
        validateNonCentrality(nonCentrality);
        validateX(x);
        validateOpenProbability(probability);
        return findDegreesOfFreedom(nonCentrality, x, probability, false);
    }

    public static double findDegreesOfFreedomUpperTail(double nonCentrality, double x, double probability) {
        validateNonCentrality(nonCentrality);
        validateX(x);
        validateOpenProbability(probability);
        return findDegreesOfFreedom(nonCentrality, x, probability, true);
    }

    public static double findNonCentrality(double degreesOfFreedom, double x, double probability) {
        validateParameters(degreesOfFreedom, 0.0);
        validateX(x);
        validateOpenProbability(probability);
        return findNonCentrality(degreesOfFreedom, x, probability, false);
    }

    public static double findNonCentralityUpperTail(double degreesOfFreedom, double x, double probability) {
        validateParameters(degreesOfFreedom, 0.0);
        validateX(x);
        validateOpenProbability(probability);
        return findNonCentrality(degreesOfFreedom, x, probability, true);
    }

    public static double mode(double degreesOfFreedom, double nonCentrality) {
        validateParameters(degreesOfFreedom, nonCentrality);

        double pdfAtZero = pdf(degreesOfFreedom, nonCentrality, 0.0);
        if (Double.isInfinite(pdfAtZero)) {
            return 0.0;
        }

        double startingPoint = degreesOfFreedom < nonCentrality / 4.0
            ? degreesOfFreedom + nonCentrality - 3.0
            : 1.0 + degreesOfFreedom;
        return genericFindMode(degreesOfFreedom, nonCentrality, startingPoint);
    }

    private static double nonCentralChiSquaredCdf(double x,
                                                  double degreesOfFreedom,
                                                  double nonCentrality,
                                                  boolean invert) {
        if (x == 0.0) {
            return invert ? 1.0 : 0.0;
        }
        if (nonCentrality == 0.0) {
            return invert
                ? Gamma.gammaQ(0.5 * degreesOfFreedom, 0.5 * x)
                : Gamma.gammaP(0.5 * degreesOfFreedom, 0.5 * x);
        }

        double result;
        if (x > degreesOfFreedom + nonCentrality) {
            result = nonCentralChiSquareQ(x, degreesOfFreedom, nonCentrality, invert ? 0.0 : -1.0);
            invert = !invert;
        } else if (nonCentrality < SMALL_NON_CENTRALITY_CUTOFF) {
            result = nonCentralChiSquarePDing(x, degreesOfFreedom, nonCentrality, invert ? -1.0 : 0.0);
        } else {
            result = nonCentralChiSquareP(x, degreesOfFreedom, nonCentrality, invert ? -1.0 : 0.0);
        }

        if (invert) {
            result = -result;
        }
        return result;
    }

    private static double nonCentralChiSquareQ(double x,
                                               double degreesOfFreedom,
                                               double nonCentrality,
                                               double initSum) {
        if (x == 0.0) {
            return 1.0;
        }

        double lambda = nonCentrality / 2.0;
        double delta = degreesOfFreedom / 2.0;
        double y = x / 2.0;

        long k = Math.round(lambda);
        double poissonForward = Gamma.gammaPDerivative(k + 1.0, lambda);
        double poissonBackward = poissonForward * k / lambda;

        double gammaForward = Gamma.gammaQ(delta + k, y);
        double xtermForward = Gamma.gammaPDerivative(delta + 1.0 + k, y);
        double xtermBackward = xtermForward * (delta + k) / y;
        double gammaBackward = gammaForward - xtermBackward;

        double sum = initSum;

        long i;
        for (i = k; i - k < MAX_ITERATIONS; i++) {
            double term = poissonForward * gammaForward;
            sum += term;

            poissonForward *= lambda / (i + 1.0);
            gammaForward += xtermForward;
            xtermForward *= y / (delta + i + 1.0);

            if (seriesConverged(term, sum) && term >= poissonForward * gammaForward) {
                break;
            }
        }

        if (i - k >= MAX_ITERATIONS) {
            throw new IllegalStateException("Non-central chi-square Q forward series did not converge");
        }

        for (long j = k - 1; j >= 0; j--) {
            double term = poissonBackward * gammaBackward;
            sum += term;

            poissonBackward *= j / lambda;
            xtermBackward *= (delta + j) / y;
            gammaBackward -= xtermBackward;

            if (seriesConverged(term, sum)) {
                break;
            }
            if (j == 0) {
                break;
            }
        }

        return sum;
    }

    private static double nonCentralChiSquarePDing(double x,
                                                   double degreesOfFreedom,
                                                   double nonCentrality,
                                                   double initSum) {
        if (x == 0.0) {
            return 0.0;
        }

        double tk = Gamma.gammaPDerivative(degreesOfFreedom / 2.0 + 1.0, x / 2.0);
        double lambda = nonCentrality / 2.0;
        double vk = Math.exp(-lambda);
        double uk = vk;
        double sum = initSum + tk * vk;

        if (sum == 0.0) {
            return sum;
        }

        double previousTerm = 0.0;
        for (int i = 1; i < MAX_ITERATIONS; i++) {
            tk = tk * x / (degreesOfFreedom + 2.0 * i);
            uk = uk * lambda / i;
            vk += uk;

            double term = vk * tk;
            sum += term;

            if (seriesConverged(term, sum) && term <= previousTerm) {
                return sum;
            }
            previousTerm = term;
        }

        throw new IllegalStateException("Non-central chi-square Ding series did not converge");
    }

    private static double nonCentralChiSquareP(double x,
                                               double degreesOfFreedom,
                                               double nonCentrality,
                                               double initSum) {
        if (x == 0.0) {
            return 0.0;
        }

        double y = x / 2.0;
        double lambda = nonCentrality / 2.0;

        long k = Math.round(lambda);
        double a = degreesOfFreedom / 2.0 + k;
        double gammaForward = Gamma.gammaP(a, y);

        if (nonCentrality == 0.0) {
            return gammaForward;
        }

        double gammaBackward = gammaForward;
        double poissonForward = Gamma.gammaPDerivative(k + 1.0, lambda);
        double poissonBackward = poissonForward;
        double xtermForward = Gamma.gammaPDerivative(a, y);
        double xtermBackward = xtermForward * y / a;

        double sum = initSum + poissonForward * gammaForward;
        if (sum == 0.0) {
            return 0.0;
        }

        double backwardError = 0.0;
        double forwardError = 0.0;

        int i = 1;
        while (i <= k) {
            xtermBackward *= (a - i + 1.0) / y;
            gammaBackward += xtermBackward;
            poissonBackward = poissonBackward * (k - i + 1.0) / lambda;

            forwardError = backwardError;
            backwardError = gammaBackward * poissonBackward;
            sum += backwardError;

            if (seriesConverged(backwardError, sum) && backwardError <= forwardError) {
                break;
            }
            i++;
        }

        i = 1;
        do {
            xtermForward = xtermForward * y / (a + i - 1.0);
            gammaForward -= xtermForward;
            poissonForward = poissonForward * lambda / (k + i);

            forwardError = poissonForward * gammaForward;
            sum += forwardError;
            i++;
        } while (!seriesConverged(forwardError, sum) && i < MAX_ITERATIONS);

        if (i >= MAX_ITERATIONS) {
            throw new IllegalStateException("Non-central chi-square large-lambda series did not converge");
        }

        return sum;
    }

    private static double nonCentralChiSquarePdfSeries(double x,
                                                       double degreesOfFreedom,
                                                       double nonCentrality) {
        double xHalf = x / 2.0;
        double degreesOfFreedomHalf = degreesOfFreedom / 2.0;
        double nonCentralityHalf = nonCentrality / 2.0;

        long k = (long) Math.floor(nonCentralityHalf);
        double term = Gamma.gammaPDerivative(k + 1.0, nonCentralityHalf)
            * Gamma.gammaPDerivative(degreesOfFreedomHalf + k, xHalf);
        if (term == 0.0) {
            return 0.0;
        }

        double sum = 0.0;
        double backwardTerm = term;
        for (long i = k; ; i++) {
            sum += term;
            if (seriesConverged(term, sum)) {
                break;
            }
            if (i - k >= MAX_ITERATIONS) {
                throw new IllegalStateException("Non-central chi-square PDF series did not converge");
            }
            term *= nonCentralityHalf * xHalf / ((i + 1.0) * (degreesOfFreedomHalf + i));
        }

        for (long i = k - 1; i >= 0; i--) {
            backwardTerm *= (i + 1.0) * (degreesOfFreedomHalf + i) / (nonCentralityHalf * xHalf);
            sum += backwardTerm;
            if (seriesConverged(backwardTerm, sum)) {
                break;
            }
            if (i == 0) {
                break;
            }
        }
        return sum / 2.0;
    }

    private static double initialQuantileGuess(double degreesOfFreedom,
                                               double nonCentrality,
                                               double probability,
                                               boolean complement) {
        double b = -(nonCentrality * nonCentrality) / (degreesOfFreedom + 3.0 * nonCentrality);
        double c = (degreesOfFreedom + 3.0 * nonCentrality) / (degreesOfFreedom + 2.0 * nonCentrality);
        double ff = (degreesOfFreedom + 2.0 * nonCentrality) / (c * c);
        ChiSquareDistribution chiSquare = new ChiSquareDistribution(ff);
        double guess = b + c * (complement
            ? chiSquare.quantileUpperTail(probability)
            : chiSquare.quantile(probability));

        if (guess < THOMAS_LUU_GUESS_CUTOFF) {
            double pp = complement ? 1.0 - probability : probability;
            double logGuess = (degreesOfFreedom / 2.0 - 1.0) * Math.log(2.0)
                + nonCentrality / 2.0
                + Math.log(pp)
                + Math.log(degreesOfFreedom)
                + Gamma.lgamma(degreesOfFreedom / 2.0);
            guess = Math.exp((2.0 / degreesOfFreedom) * logGuess);
            if (guess == 0.0) {
                guess = MIN_POSITIVE_GUESS;
            }
        }

        if (!Double.isFinite(guess)) {
            guess = Double.MAX_VALUE / 4.0;
        }
        return Math.max(guess, 0.0);
    }

    private static double genericFindMode(double degreesOfFreedom,
                                          double nonCentrality,
                                          double guess) {
        double upperBound = guess;
        double value = pdf(degreesOfFreedom, nonCentrality, guess);
        if (value == 0.0) {
            throw new ArithmeticException(
                "Could not locate a starting location for the search for the mode, original guess was " + guess
            );
        }

        double maxValue;
        do {
            maxValue = value;
            upperBound *= 2.0;
            value = pdf(degreesOfFreedom, nonCentrality, upperBound);
        } while (maxValue < value);

        double lowerBound = upperBound;
        do {
            maxValue = value;
            lowerBound /= 2.0;
            if (lowerBound < MIN_POSITIVE_GUESS) {
                return 0.0;
            }
            value = pdf(degreesOfFreedom, nonCentrality, lowerBound);
        } while (maxValue < value);

        Minima result = Minima.brentFindMinima(
            x -> -pdf(degreesOfFreedom, nonCentrality, x),
            lowerBound,
            upperBound,
            MODE_MINIMIZER_BITS,
            MAX_ROOT_ITERATIONS
        );
        if (!result.converged()) {
            throw new ArithmeticException(
                "Unable to locate solution in a reasonable time: either there is no answer to the mode of the distribution or the answer is infinite"
            );
        }
        return result.point();
    }

    private static double findDegreesOfFreedom(double nonCentrality,
                                               double x,
                                               double probability,
                                               boolean complementProbability) {
        double lowerTailProbability = complementProbability ? 1.0 - probability : probability;
        double upperTailProbability = 1.0 - lowerTailProbability;
        double target = lowerTailProbability < upperTailProbability ? lowerTailProbability : upperTailProbability;
        boolean complement = lowerTailProbability >= upperTailProbability;
        DoubleUnaryOperator objective = degreesOfFreedom -> complement
            ? target - ccdf(degreesOfFreedom, nonCentrality, x)
            : cdf(degreesOfFreedom, nonCentrality, x) - target;
        double guess = Math.max(1.0, x - nonCentrality);
        return RootIterators.toms748SolveBounded(
            objective,
            guess,
            2.0,
            false,
            MIN_PARAMETER,
            ROOT_BITS,
            MAX_ROOT_ITERATIONS,
            complementProbability ? "findDegreesOfFreedomUpperTail" : "findDegreesOfFreedom"
        );
    }

    private static double findNonCentrality(double degreesOfFreedom,
                                            double x,
                                            double probability,
                                            boolean complementProbability) {
        double lowerTailProbability = complementProbability ? 1.0 - probability : probability;
        double upperTailProbability = 1.0 - lowerTailProbability;
        double target = lowerTailProbability < upperTailProbability ? lowerTailProbability : upperTailProbability;
        boolean complement = lowerTailProbability >= upperTailProbability;
        DoubleUnaryOperator objective = nonCentrality -> complement
            ? target - ccdf(degreesOfFreedom, nonCentrality, x)
            : cdf(degreesOfFreedom, nonCentrality, x) - target;
        double guess = Math.max(1.0, x - degreesOfFreedom);
        return RootIterators.toms748SolveBounded(
            objective,
            guess,
            2.0,
            false,
            0.0,
            ROOT_BITS,
            MAX_ROOT_ITERATIONS,
            complementProbability ? "findNonCentralityUpperTail" : "findNonCentrality"
        );
    }

    private static boolean seriesConverged(double term, double sum) {
        return sum == 0.0 || Math.abs(term / sum) < RELATIVE_TOLERANCE;
    }

    private static void validateParameters(double degreesOfFreedom, double nonCentrality) {
        if (!(degreesOfFreedom > 0.0) || !Double.isFinite(degreesOfFreedom)) {
            throw new IllegalArgumentException("degreesOfFreedom must be positive: " + degreesOfFreedom);
        }
        validateNonCentrality(nonCentrality);
    }

    private static void validateX(double x) {
        if (!Double.isFinite(x) || x < 0.0) {
            throw new IllegalArgumentException("x must be finite and non-negative: " + x);
        }
    }

    private static void validateProbability(double probability) {
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }

    private static void validateOpenProbability(double probability) {
        if (!Double.isFinite(probability) || probability <= 0.0 || probability >= 1.0) {
            throw new IllegalArgumentException("probability must be in (0, 1): " + probability);
        }
    }

    private static void validateNonCentrality(double nonCentrality) {
        if (!Double.isFinite(nonCentrality) || nonCentrality < 0.0 || nonCentrality > MAX_NON_CENTRALITY) {
            throw new IllegalArgumentException("nonCentrality must be non-negative: " + nonCentrality);
        }
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
}