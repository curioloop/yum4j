package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Beta;
import com.curioloop.yum4j.math.Gamma;
import com.curioloop.yum4j.math.Hypergeometric;
import com.curioloop.yum4j.math.Minima;
import com.curioloop.yum4j.math.Normal;
import com.curioloop.yum4j.quad.de.DEPool;
import com.curioloop.yum4j.quad.de.DoubleExponentialCore;

/** Boost-style non-central Student's t distribution with aligned PDF/CDF/quantile branches. */
public value record NonCentralTDistribution(double degreesOfFreedom, double nonCentrality)
    implements ContinuousDistribution {

    private static final double EPSILON = Math.ulp(1.0);
    private static final double LARGE_DEGREES_OF_FREEDOM = 1.0 / Math.ulp(1.0);
    private static final double ROOT_EPSILON = Math.sqrt(Math.ulp(1.0));
    private static final double SQRT_TWO = Math.sqrt(2.0);
    private static final double SQRT_PI = Math.sqrt(Math.PI);
    private static final double LOG_MAX_VALUE = Math.log(Double.MAX_VALUE);
    private static final int MAX_SERIES_ITERATIONS = 100000;
    private static final int MODE_MINIMIZER_BITS = 48;
    private static final int MODE_MINIMIZER_ITERATIONS = 256;
    private static final int PDF_INTEGRAL_MAX_REFINEMENTS = 9;
    private static final DEPool PDF_INTEGRAL_POOL = new DEPool();

    public NonCentralTDistribution {
        validateDegreesOfFreedom(degreesOfFreedom);
        validateNonCentrality(nonCentrality);
    }

    @Override
    public Double2 range() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (nonCentrality == 0.0) {
            return StudentsTDistribution.pdfKernel(degreesOfFreedom, x);
        }
        return Math.max(0.0, nonCentralTPdf(degreesOfFreedom, nonCentrality, x));
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        if (nonCentrality == 0.0) {
            return StudentsTDistribution.cdfKernel(degreesOfFreedom, x);
        }
        return clampProbability(nonCentralTCdf(degreesOfFreedom, nonCentrality, x, false));
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        if (nonCentrality == 0.0) {
            return StudentsTDistribution.ccdfKernel(degreesOfFreedom, x);
        }
        return clampProbability(nonCentralTCdf(degreesOfFreedom, nonCentrality, x, true));
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (probability == 1.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (nonCentrality == 0.0) {
            return StudentsTDistribution.quantileKernel(degreesOfFreedom, probability);
        }
        return StatisticsQuantileSupport.continuousQuantile(this, probability, initialQuantileGuess(probability));
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        if (probability == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        if (probability == 1.0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (nonCentrality == 0.0) {
            return StudentsTDistribution.quantileUpperTailKernel(degreesOfFreedom, probability);
        }
        return StatisticsQuantileSupport.continuousUpperTailQuantile(this, probability, initialUpperTailQuantileGuess(probability));
    }

    public double mode() {
        if (useNormalApproximation()) {
            return nonCentrality;
        }
        double guess = degreesOfFreedom < 3.0 ? 0.0 : meanUnchecked(degreesOfFreedom, nonCentrality);
        double variance = degreesOfFreedom < 4.0 ? 1.0 : varianceUnchecked(degreesOfFreedom, nonCentrality);
        double step = Math.sqrt(Math.max(variance, Double.MIN_NORMAL));
        return genericMode(guess, step);
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        if (degreesOfFreedom <= 1.0) {
            throw new ArithmeticException(
                "The non-central t distribution has no defined mean for degrees of freedom <= 1: got v="
                    + degreesOfFreedom
            );
        }
        return meanUnchecked(degreesOfFreedom, nonCentrality);
    }

    @Override
    public double variance() {
        if (degreesOfFreedom <= 2.0) {
            throw new ArithmeticException(
                "The non-central t distribution has no defined variance for degrees of freedom <= 2: got v="
                    + degreesOfFreedom
            );
        }
        return varianceUnchecked(degreesOfFreedom, nonCentrality);
    }

    public double skewness() {
        if (degreesOfFreedom <= 3.0) {
            throw new ArithmeticException(
                "The non-central t distribution has no defined skewness for degrees of freedom <= 3: got v="
                    + degreesOfFreedom
            );
        }
        return skewnessUnchecked(degreesOfFreedom, nonCentrality);
    }

    public double kurtosisExcess() {
        if (degreesOfFreedom <= 4.0) {
            throw new ArithmeticException(
                "The non-central t distribution has no defined kurtosis for degrees of freedom <= 4: got v="
                    + degreesOfFreedom
            );
        }
        return kurtosisExcessUnchecked(degreesOfFreedom, nonCentrality);
    }

    public double kurtosis() {
        return kurtosisExcess() + 3.0;
    }

    private double initialQuantileGuess(double probability) {
        if (useNormalApproximation()) {
            return shiftedNormalQuantile(nonCentrality, probability);
        }
        return resolveInitialQuantileGuess(probability, approximateQuantileGuess(probability), false);
    }

    private double initialUpperTailQuantileGuess(double probability) {
        if (useNormalApproximation()) {
            return shiftedNormalQuantileUpperTail(nonCentrality, probability);
        }
        return resolveInitialQuantileGuess(probability, approximateUpperTailQuantileGuess(probability), true);
    }

    private double approximateQuantileGuess(double probability) {
        if (degreesOfFreedom <= 3.0) {
            return 0.0;
        }
        double mean = meanUnchecked(degreesOfFreedom, nonCentrality);
        double variance = varianceUnchecked(degreesOfFreedom, nonCentrality);
        double standardDeviation = Math.sqrt(Math.max(variance, Double.MIN_NORMAL));
        return scaledNormalQuantile(mean, standardDeviation, probability);
    }

    private double approximateUpperTailQuantileGuess(double probability) {
        if (degreesOfFreedom <= 3.0) {
            return 0.0;
        }
        double mean = meanUnchecked(degreesOfFreedom, nonCentrality);
        double variance = varianceUnchecked(degreesOfFreedom, nonCentrality);
        double standardDeviation = Math.sqrt(Math.max(variance, Double.MIN_NORMAL));
        return scaledNormalQuantileUpperTail(mean, standardDeviation, probability);
    }

    private double resolveInitialQuantileGuess(double probability, double guess, boolean complement) {
        double probabilityAtZero = cdf(0.0);
        int sign = complement ? sign(probabilityAtZero - probability) : sign(probability - probabilityAtZero);
        return resolveInitialQuantileGuess(guess, sign);
    }

    private static double resolveInitialQuantileGuess(double guess, int sign) {
        if (sign == 0) {
            return 0.0;
        }
        if (guess == 0.0 || sign(guess) != sign) {
            return sign;
        }
        return guess;
    }

    private double genericMode(double guess, double step) {
        if (!Double.isFinite(guess)) {
            guess = 0.0;
        }
        if (!(step > 0.0) || !Double.isFinite(step)) {
            step = 1.0;
        }
        double upperBound = guess;
        double value = pdf(guess);
        double maxValue;
        do {
            maxValue = value;
            upperBound += step;
            value = pdf(upperBound);
        } while (maxValue < value);

        double lowerBound = upperBound;
        do {
            maxValue = value;
            lowerBound -= step;
            value = pdf(lowerBound);
        } while (maxValue < value);

        Minima result = Minima.brentFindMinima(
            x -> -pdf(x),
            lowerBound,
            upperBound,
            MODE_MINIMIZER_BITS,
            MODE_MINIMIZER_ITERATIONS
        );
        if (!result.converged()) {
            throw new ArithmeticException("Unable to bracket non-central t mode");
        }
        return result.point();
    }

    private boolean useNormalApproximation() {
        return Double.isInfinite(degreesOfFreedom) || degreesOfFreedom > LARGE_DEGREES_OF_FREEDOM;
    }

    private static double nonCentralTCdf(double v, double delta, double t, boolean invert) {
        if (Double.isInfinite(v)) {
            double result = shiftedNormalCdf(delta, t);
            return invert ? 1.0 - result : result;
        }
        if (t < 0.0) {
            t = -t;
            delta = -delta;
            invert = !invert;
        }
        if (Math.abs(delta / (4.0 * v)) < EPSILON) {
            double result = StudentsTDistribution.cdfKernel(v, t - delta);
            return invert ? 1.0 - result : result;
        }

        double x = t * t / (v + t * t);
        double y = v / (v + t * t);
        double deltaSquared = delta * delta;
        double b = v / 2.0;
        double c = 0.5 + b + deltaSquared / 2.0;
        double cross = 1.0 - (b / c) * (1.0 + deltaSquared / (2.0 * c * c));

        double result;
        if (x < cross) {
            if (x != 0.0) {
                result = NonCentralBetaDistribution.cdf(0.5, b, deltaSquared, x);
                result = nonCentralT2P(v, delta, x, y, result) / 2.0;
            } else {
                result = 0.0;
            }
            if (invert) {
                result = Normal.ccdf(-delta) - result;
                invert = false;
            } else {
                result += Normal.cdf(-delta);
            }
        } else {
            invert = !invert;
            if (x != 0.0) {
                result = NonCentralBetaDistribution.ccdf(0.5, b, deltaSquared, x);
                result = nonCentralT2Q(v, delta, x, y, result) / 2.0;
            } else {
                result = Normal.ccdf(-delta);
            }
        }
        if (invert) {
            result = 1.0 - result;
        }
        return clampProbability(result);
    }

    private static double nonCentralT2P(double v, double delta, double x, double y, double initValue) {
        double deltaSquaredHalf = delta * delta / 2.0;
        long k = Math.max(1L, (long) Math.floor(deltaSquaredHalf));
        double poisson = poissonWeight(k, deltaSquaredHalf, delta);
        if (poisson == 0.0) {
            return initValue;
        }

        double beta = lowerBetaT(k, v, x, y);
        double xTerm = betaIncrementT(k, v, x, y);
        while (Math.abs(beta * poisson) < Double.MIN_NORMAL) {
            if (k == 0L || poisson == 0.0) {
                return initValue;
            }
            k /= 2L;
            poisson = poissonWeight(k, deltaSquaredHalf, delta);
            beta = lowerBetaT(k, v, x, y);
            xTerm = betaIncrementT(k, v, x, y);
        }

        double forwardPoisson = poisson;
        double forwardBeta = beta;
        double forwardXTerm = xTerm;
        double sum = initValue;
        if (beta == 0.0 && xTerm == 0.0) {
            return initValue;
        }

        double lastTerm = 0.0;
        long count = 0L;
        for (long i = k; i >= 0L; i--) {
            double term = beta * poisson;
            sum += term;
            if (((Math.abs(lastTerm) > Math.abs(term)) && relativeTerm(term, sum) < EPSILON) || (v == 2.0 && i == 0L)) {
                break;
            }
            lastTerm = term;
            poisson *= (i + 0.5) / deltaSquaredHalf;
            beta += xTerm;
            xTerm *= i / (x * (v / 2.0 + i - 1.0));
            count++;
            if (count > MAX_SERIES_ITERATIONS) {
                throw new ArithmeticException("cdf(nonCentralT) lower series did not converge");
            }
        }

        lastTerm = 0.0;
        double forwardLimit = forwardBeta * EPSILON * 4.0;
        for (long i = k + 1L; ; i++) {
            forwardPoisson *= deltaSquaredHalf / (i + 0.5);
            forwardXTerm *= (x * (v / 2.0 + i - 1.0)) / i;
            forwardBeta -= forwardXTerm;
            if (forwardBeta < forwardLimit) {
                break;
            }

            double term = forwardPoisson * forwardBeta;
            sum += term;
            if ((Math.abs(lastTerm) >= Math.abs(term)) && relativeTerm(term, sum) < EPSILON) {
                break;
            }
            lastTerm = term;
            count++;
            if (count > MAX_SERIES_ITERATIONS) {
                throw new ArithmeticException("cdf(nonCentralT) lower forward series did not converge");
            }
        }
        return sum;
    }

    private static double nonCentralT2Q(double v, double delta, double x, double y, double initValue) {
        double deltaSquaredHalf = delta * delta / 2.0;
        long k = Math.max(1L, (long) Math.floor(deltaSquaredHalf));
        double poisson = poissonWeight(k, deltaSquaredHalf, delta);
        if (poisson == 0.0) {
            return initValue;
        }

        double beta = upperBetaT(k, v, x, y);
        double xTerm = betaIncrementT(k, v, x, y);
        double forwardPoisson = poisson;
        double forwardBeta = beta;
        double forwardXTerm = xTerm;
        double sum = initValue;
        if (beta == 0.0 && xTerm == 0.0) {
            return initValue;
        }

        long count = 0L;
        double lastTerm = 0.0;
        for (long i = k + 1L, j = k; ; i++, j--) {
            forwardPoisson *= deltaSquaredHalf / (i + 0.5);
            forwardXTerm *= (x * (v / 2.0 + i - 1.0)) / i;
            forwardBeta += forwardXTerm;
            double term = forwardPoisson * forwardBeta;

            if (j >= 0L) {
                term += beta * poisson;
                poisson *= (j + 0.5) / deltaSquaredHalf;
                beta -= xTerm;
                if (!(v == 2.0 && j == 0L)) {
                    xTerm *= j / (x * (v / 2.0 + j - 1.0));
                }
            }

            sum += term;
            if ((Math.abs(lastTerm) > Math.abs(term)) && relativeTerm(term, sum) < EPSILON) {
                break;
            }
            lastTerm = term;
            count++;
            if (count > MAX_SERIES_ITERATIONS) {
                throw new ArithmeticException("cdf(nonCentralT) upper series did not converge");
            }
        }
        return sum;
    }

    private static double nonCentralTPdf(double v, double delta, double t) {
        if (Double.isInfinite(v)) {
            return shiftedNormalPdf(delta, t);
        }
        if (t * t < EPSILON) {
            return Gamma.tgammaDeltaRatio(v / 2.0 + 0.5, 0.5)
                * Math.sqrt(v / Math.PI)
                * Math.exp(-delta * delta / 2.0)
                / 2.0;
        }
        if (Math.abs(delta / (4.0 * v)) < EPSILON) {
            return StudentsTDistribution.pdfKernel(v, t - delta);
        }

        double a = (v + 1.0) / 2.0;
        double hypergeometricArgument = delta * delta * t * t / (2.0 * (t * t + v));
        double summit = (Math.sqrt(hypergeometricArgument * (4.0 * a + hypergeometricArgument)) + hypergeometricArgument) / 2.0;
        if (summit < 40.0) {
            return nonCentralTPdfHypergeometric(t, v, delta);
        }

        boolean reflected = false;
        if (t < 0.0) {
            t = -t;
            delta = -delta;
            reflected = true;
        }

        double x = t * t / (v + t * t);
        double y = v / (v + t * t);
        double result = NonCentralBetaDistribution.pdf(0.5, v / 2.0, delta * delta, x);
        double tolerance = ROOT_EPSILON * result;
        result = nonCentralT2Pdf(v, delta, x, y, result);
        result *= v * t / (v * v + 2.0 * v * t * t + t * t * t * t);
        if (result <= tolerance) {
            if (reflected) {
                delta = -delta;
                t = -t;
            }
            result = nonCentralTPdfIntegral(t, v, delta);
        }
        return result;
    }

    private static double nonCentralTPdfHypergeometric(double x, double v, double mu) {
        double argument = mu * mu * x * x / (2.0 * (x * x + v));
        double a = Hypergeometric.oneFOne((v + 1.0) / 2.0, 0.5, argument);
        double b = Hypergeometric.oneFOne(v / 2.0 + 1.0, 1.5, argument);
        b *= Gamma.tgammaDeltaRatio(v / 2.0 + 1.0, -0.5);
        b *= SQRT_TWO * mu * x / Math.sqrt(x * x + v);
        double tolerance = ROOT_EPSILON * Math.abs(a) * 4.0;
        double sum = a + b;
        if (sum < tolerance) {
            return nonCentralTPdfIntegral(x, v, mu);
        }
        sum *= Math.exp(-mu * mu / 2.0)
            * Math.pow(1.0 + x * x / v, -(v + 1.0) / 2.0)
            * Gamma.tgammaDeltaRatio(v / 2.0 + 0.5, -0.5);
        sum /= Math.sqrt(v * Math.PI);
        return sum;
    }

    private static double nonCentralTPdfIntegral(double x, double v, double mu) {
        double xSquaredPlusV = x * x + v;
        double leading = Math.pow(v, v / 2.0) * Math.exp(-v * mu * mu / (2.0 * xSquaredPlusV));
        if (leading == 0.0) {
            return 0.0;
        }
        double shift = mu * x / Math.sqrt(xSquaredPlusV);
        double integral = DoubleExponentialCore.expSinh(
            y -> {
                if (y == 0.0) {
                    return 0.0;
                }
                double gaussianExponent = -0.5 * (y - shift) * (y - shift);
                double logPower = v * Math.log(y);
                if (logPower < LOG_MAX_VALUE) {
                    return Math.pow(y, v) * Math.exp(gaussianExponent);
                }
                return Math.exp(logPower + gaussianExponent);
            },
            0.0,
            Double.POSITIVE_INFINITY,
            ROOT_EPSILON,
            PDF_INTEGRAL_MAX_REFINEMENTS,
            PDF_INTEGRAL_POOL
        ).value();
        return leading * integral
            / (SQRT_PI * Gamma.tgamma(v / 2.0) * Math.pow(2.0, (v - 1.0) / 2.0) * Math.pow(xSquaredPlusV, (v + 1.0) / 2.0));
    }

    private static double nonCentralT2Pdf(double v, double delta, double x, double y, double initValue) {
        double deltaSquaredHalf = delta * delta / 2.0;
        long k = Math.max(1L, (long) Math.floor(deltaSquaredHalf));
        double poisson = poissonWeight(k, deltaSquaredHalf, delta);
        double xTerm = pdfBetaTermT(k, v, x, y);

        while (Math.abs(xTerm * poisson) < Double.MIN_NORMAL) {
            if (k == 0L) {
                return initValue;
            }
            k /= 2L;
            poisson = poissonWeight(k, deltaSquaredHalf, delta);
            xTerm = pdfBetaTermT(k, v, x, y);
        }

        double forwardPoisson = poisson;
        double forwardXTerm = xTerm;
        double sum = initValue;
        long count = 0L;
        double previousRatio = 1.0;
        for (long i = k; i >= 0L; i--) {
            double term = xTerm * poisson;
            sum += term;
            double ratio = relativeTerm(term, sum);
            if (((ratio < EPSILON) && (i != k) && (ratio < previousRatio)) || term == 0.0) {
                break;
            }
            previousRatio = ratio;
            poisson *= (i + 0.5) / deltaSquaredHalf;
            xTerm *= i / (x * (v / 2.0 + i));
            count++;
            if (count > MAX_SERIES_ITERATIONS) {
                throw new ArithmeticException("pdf(nonCentralT) backward series did not converge");
            }
        }

        previousRatio = 0.0;
        for (long i = k + 1L; ; i++) {
            forwardPoisson *= deltaSquaredHalf / (i + 0.5);
            forwardXTerm *= (x * (v / 2.0 + i)) / i;
            double term = forwardPoisson * forwardXTerm;
            sum += term;
            double ratio = relativeTerm(term, sum);
            if (((ratio < EPSILON) && (ratio < previousRatio)) || term == 0.0) {
                break;
            }
            previousRatio = ratio;
            count++;
            if (count > MAX_SERIES_ITERATIONS) {
                throw new ArithmeticException("pdf(nonCentralT) forward series did not converge");
            }
        }
        return sum;
    }

    private static double lowerBetaT(long k, double v, double x, double y) {
        return x < y ? Beta.ibeta(k + 1.0, v / 2.0, x) : Beta.ibetac(v / 2.0, k + 1.0, y);
    }

    private static double upperBetaT(long k, double v, double x, double y) {
        return x < y ? Beta.ibetac(k + 1.0, v / 2.0, x) : Beta.ibeta(v / 2.0, k + 1.0, y);
    }

    private static double betaIncrementT(long k, double v, double x, double y) {
        double derivative = x < y
            ? Beta.ibetaDerivative(k + 1.0, v / 2.0, x)
            : Beta.ibetaDerivative(v / 2.0, k + 1.0, y);
        return derivative * y / (v / 2.0 + k);
    }

    private static double pdfBetaTermT(long k, double v, double x, double y) {
        return x < y
            ? Beta.ibetaDerivative(k + 1.0, v / 2.0, x)
            : Beta.ibetaDerivative(v / 2.0, k + 1.0, y);
    }

    private static double poissonWeight(long k, double deltaSquaredHalf, double delta) {
        if (deltaSquaredHalf == 0.0) {
            return 0.0;
        }
        return Gamma.gammaPDerivative(k + 1.0, deltaSquaredHalf)
            * Gamma.tgammaDeltaRatio(k + 1.0, 0.5)
            * delta
            / SQRT_TWO;
    }

    private static double meanUnchecked(double v, double delta) {
        if (Double.isInfinite(v) || v > LARGE_DEGREES_OF_FREEDOM) {
            return delta;
        }
        return delta * Math.sqrt(v / 2.0) * Gamma.tgammaDeltaRatio((v - 1.0) / 2.0, 0.5);
    }

    private static double varianceUnchecked(double v, double delta) {
        if (Double.isInfinite(v)) {
            return 1.0;
        }
        if (delta == 0.0) {
            return v / (v - 2.0);
        }
        double result = ((delta * delta + 1.0) * v) / (v - 2.0);
        double mean = meanUnchecked(v, delta);
        return result - mean * mean;
    }

    private static double skewnessUnchecked(double v, double delta) {
        if (Double.isInfinite(v) || delta == 0.0) {
            return 0.0;
        }
        double mean = meanUnchecked(v, delta);
        double deltaSquared = delta * delta;
        double variance = ((deltaSquared + 1.0) * v) / (v - 2.0) - mean * mean;
        double result = -2.0 * variance;
        result += v * (deltaSquared + 2.0 * v - 3.0) / ((v - 3.0) * (v - 2.0));
        result *= mean;
        result /= Math.pow(variance, 1.5);
        return result;
    }

    private static double kurtosisExcessUnchecked(double v, double delta) {
        if (Double.isInfinite(v) || delta == 0.0) {
            return 1.0;
        }
        double mean = meanUnchecked(v, delta);
        double deltaSquared = delta * delta;
        double variance = ((deltaSquared + 1.0) * v) / (v - 2.0) - mean * mean;
        double result = -3.0 * variance;
        result += v * (deltaSquared * (v + 1.0) + 3.0 * (3.0 * v - 5.0)) / ((v - 3.0) * (v - 2.0));
        result *= -mean * mean;
        result += v * v * (deltaSquared * deltaSquared + 6.0 * deltaSquared + 3.0) / ((v - 4.0) * (v - 2.0));
        result /= variance * variance;
        return result - 3.0;
    }

    private static double relativeTerm(double term, double sum) {
        if (sum == 0.0) {
            return Math.abs(term);
        }
        return Math.abs(term / sum);
    }

    private static int sign(double value) {
        return value > 0.0 ? 1 : value < 0.0 ? -1 : 0;
    }

    private static double shiftedNormalPdf(double mean, double x) {
        return Normal.pdf(x - mean);
    }

    private static double shiftedNormalCdf(double mean, double x) {
        return Normal.cdf(x - mean);
    }

    private static double shiftedNormalQuantile(double mean, double probability) {
        return mean + Normal.inv(probability);
    }

    private static double shiftedNormalQuantileUpperTail(double mean, double probability) {
        return mean + Normal.invUpperTail(probability);
    }

    private static double scaledNormalQuantile(double mean, double standardDeviation, double probability) {
        return mean + standardDeviation * Normal.inv(probability);
    }

    private static double scaledNormalQuantileUpperTail(double mean, double standardDeviation, double probability) {
        return mean + standardDeviation * Normal.invUpperTail(probability);
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

    private static void validateDegreesOfFreedom(double degreesOfFreedom) {
        if (Double.isNaN(degreesOfFreedom) || !(degreesOfFreedom > 0.0 || Double.isInfinite(degreesOfFreedom))) {
            throw new IllegalArgumentException(
                "degreesOfFreedom must be positive or +infinity: " + degreesOfFreedom
            );
        }
    }

    private static void validateNonCentrality(double nonCentrality) {
        if (Double.isNaN(nonCentrality) || Double.isInfinite(nonCentrality)) {
            throw new IllegalArgumentException("nonCentrality must be finite: " + nonCentrality);
        }
    }

    private static void validateX(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}