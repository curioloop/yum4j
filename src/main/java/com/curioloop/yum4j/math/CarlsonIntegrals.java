package com.curioloop.yum4j.math;

import static com.curioloop.yum4j.math.Commons.validateFinite;

import java.util.Arrays;

/**
 * Real scalar Carlson symmetric elliptic integrals.
 *
 * <p>This class exposes the five standard symmetric forms on the real domain:
 *
 * <pre>
 * R_C(x, y)
 * R_F(x, y, z)
 * R_D(x, y, z)
 * R_J(x, y, z, p)
 * R_G(x, y, z)
 * </pre>
 *
 * <p>The implementation follows Carlson-style duplication with low-order correction
 * polynomials for the converged neighborhood of the symmetric mean. The current surface
 * stays in the real-valued domain and matches the real principal-value conventions used by
 * Boost for {@code R_C(x, y < 0)} and {@code R_J(x, y, z, p < 0)}.
 */
final class CarlsonIntegrals {

    private static final double MACHINE_EPSILON = Math.ulp(1.0);
    private static final double ROOT_EPSILON = Math.sqrt(MACHINE_EPSILON);
    private static final double AGM_CONVERGENCE_FACTOR = 2.7;
    private static final int MAX_DUPLICATION_ITERATIONS = 10000;

    private CarlsonIntegrals() {
    }

    /**
     * Returns Carlson's degenerate symmetric integral {@code R_C(x, y)}.
     *
     * <p>The real principal value is returned when {@code y < 0}.
     *
     * @param x non-negative argument
     * @param y non-zero argument
     * @return {@code R_C(x, y)}
     */
    public static double rc(double x, double y) {
        validateFinite(x, "rc: x");
        validateFinite(y, "rc: y");
        if (x < 0.0) {
            throw new IllegalArgumentException("rc: x must be non-negative: " + x);
        }
        if (y == 0.0) {
            throw new IllegalArgumentException("rc: y must be non-zero: " + y);
        }

        double prefix = 1.0;
        if (y < 0.0) {
            prefix = Math.sqrt(x / (x - y));
            x -= y;
            y = -y;
        }

        double result;
        if (x == 0.0) {
            result = 0.5 * Math.PI / Math.sqrt(y);
        } else if (x == y) {
            result = 1.0 / Math.sqrt(x);
        } else if (y > x) {
            result = Math.atan(Math.sqrt((y - x) / x)) / Math.sqrt(y - x);
        } else if (y / x > 0.5) {
            double argument = Math.sqrt((x - y) / x);
            result = InverseHyperbolic.atanh(argument) / Math.sqrt(x - y);
        } else {
            result = Math.log((Math.sqrt(x) + Math.sqrt(x - y)) / Math.sqrt(y)) / Math.sqrt(x - y);
        }
        return prefix * result;
    }

    /**
     * Returns Carlson's symmetric integral of the first kind {@code R_F(x, y, z)}.
     *
     * @param x non-negative argument
     * @param y non-negative argument
     * @param z non-negative argument
     * @return {@code R_F(x, y, z)}
     */
    public static double rf(double x, double y, double z) {
        validateFinite(x, "rf: x");
        validateFinite(y, "rf: y");
        validateFinite(z, "rf: z");
        if (x < 0.0 || y < 0.0 || z < 0.0) {
            throw new IllegalArgumentException("rf: all arguments must be non-negative");
        }
        if (x + y == 0.0 || y + z == 0.0 || z + x == 0.0) {
            throw new IllegalArgumentException("rf: at most one argument may be zero");
        }

        if (x == y) {
            if (x == z) {
                return 1.0 / Math.sqrt(x);
            }
            return z == 0.0 ? 0.5 * Math.PI / Math.sqrt(x) : rc(z, x);
        }
        if (x == z) {
            return y == 0.0 ? 0.5 * Math.PI / Math.sqrt(x) : rc(y, x);
        }
        if (y == z) {
            return x == 0.0 ? 0.5 * Math.PI / Math.sqrt(y) : rc(x, y);
        }

        if (x == 0.0) {
            double swapped = z;
            z = x;
            x = swapped;
        } else if (y == 0.0) {
            double swapped = z;
            z = y;
            y = swapped;
        }
        if (z == 0.0) {
            return rfWithOneZero(x, y);
        }

        double mean = (x + y + z) / 3.0;
        double initialMean = mean;
        double currentX = x;
        double currentY = y;
        double currentZ = z;
        double threshold = Math.pow(3.0 * MACHINE_EPSILON, -0.125)
            * max3(Math.abs(mean - currentX), Math.abs(mean - currentY), Math.abs(mean - currentZ));
        double scale = 1.0;

        for (int iteration = 0; iteration < MAX_DUPLICATION_ITERATIONS; iteration++) {
            double rootX = Math.sqrt(currentX);
            double rootY = Math.sqrt(currentY);
            double rootZ = Math.sqrt(currentZ);
            double lambda = rootX * rootY + rootX * rootZ + rootY * rootZ;
            mean = 0.25 * (mean + lambda);
            currentX = 0.25 * (currentX + lambda);
            currentY = 0.25 * (currentY + lambda);
            currentZ = 0.25 * (currentZ + lambda);
            threshold *= 0.25;
            scale *= 4.0;
            if (threshold < Math.abs(mean)) {
                double xDeviation = (initialMean - x) / (mean * scale);
                double yDeviation = (initialMean - y) / (mean * scale);
                double zDeviation = -xDeviation - yDeviation;
                double secondOrder = xDeviation * yDeviation - zDeviation * zDeviation;
                double thirdOrder = xDeviation * yDeviation * zDeviation;
                double correction = 1.0
                    + thirdOrder * (1.0 / 14.0 + 3.0 * thirdOrder / 104.0)
                    + secondOrder * (-1.0 / 10.0
                    + secondOrder / 24.0
                    - 3.0 * thirdOrder / 44.0
                    - 5.0 * secondOrder * secondOrder / 208.0
                    + secondOrder * thirdOrder / 16.0);
                return correction / Math.sqrt(mean);
            }
        }

        throw new IllegalStateException("rf duplication did not converge for x=" + x + ", y=" + y + ", z=" + z);
    }

    /**
     * Returns Carlson's symmetric integral of the second kind {@code R_D(x, y, z)}.
     *
     * @param x non-negative argument
     * @param y non-negative argument
     * @param z positive argument
     * @return {@code R_D(x, y, z)}
     */
    public static double rd(double x, double y, double z) {
        validateFinite(x, "rd: x");
        validateFinite(y, "rd: y");
        validateFinite(z, "rd: z");
        if (x < 0.0 || y < 0.0 || z <= 0.0) {
            throw new IllegalArgumentException("rd: requires x >= 0, y >= 0, z > 0");
        }
        if (x + y == 0.0) {
            throw new IllegalArgumentException("rd: at most one argument may be zero");
        }

        if (x == z) {
            double swapped = y;
            y = x;
            x = swapped;
        }
        if (y == z) {
            if (x == y) {
                return 1.0 / (x * Math.sqrt(x));
            }
            if (x == 0.0) {
                return 0.75 * Math.PI / (y * Math.sqrt(y));
            }
            if (max2(x, y) / min2(x, y) > 1.3) {
                return 1.5 * (rc(x, y) - Math.sqrt(x) / y) / (y - x);
            }
        }
        if (x == y && max2(x, z) / min2(x, z) > 1.3) {
            return 3.0 * (rc(z, x) - 1.0 / Math.sqrt(z)) / (z - x);
        }
        if (y == 0.0) {
            double swapped = x;
            x = y;
            y = swapped;
        }
        if (x == 0.0) {
            return rdWithZeroX(y, z);
        }

        double mean = (x + y + 3.0 * z) / 5.0;
        double initialMean = mean;
        double currentX = x;
        double currentY = y;
        double currentZ = z;
        double threshold = 1.2 * Math.pow(MACHINE_EPSILON / 4.0, -0.125)
            * max3(mean - x, mean - y, mean - z);
        double scale = 1.0;
        double sum = 0.0;

        for (int iteration = 0; iteration < MAX_DUPLICATION_ITERATIONS; iteration++) {
            double rootX = Math.sqrt(currentX);
            double rootY = Math.sqrt(currentY);
            double rootZ = Math.sqrt(currentZ);
            double lambda = rootX * rootY + rootX * rootZ + rootY * rootZ;
            sum += scale / (rootZ * (currentZ + lambda));
            mean = 0.25 * (mean + lambda);
            currentX = 0.25 * (currentX + lambda);
            currentY = 0.25 * (currentY + lambda);
            currentZ = 0.25 * (currentZ + lambda);
            scale *= 0.25;
            threshold *= 0.25;
            if (threshold < mean) {
                double xDeviation = scale * (initialMean - x) / mean;
                double yDeviation = scale * (initialMean - y) / mean;
                double zDeviation = -(xDeviation + yDeviation) / 3.0;
                double secondOrder = xDeviation * yDeviation - 6.0 * zDeviation * zDeviation;
                double thirdOrder = (3.0 * xDeviation * yDeviation - 8.0 * zDeviation * zDeviation) * zDeviation;
                double fourthOrder = 3.0 * (xDeviation * yDeviation - zDeviation * zDeviation) * zDeviation * zDeviation;
                double fifthOrder = xDeviation * yDeviation * zDeviation * zDeviation * zDeviation;
                double inversePower = scale / (mean * Math.sqrt(mean));
                double correction = 1.0
                    - 3.0 * secondOrder / 14.0
                    + thirdOrder / 6.0
                    + 9.0 * secondOrder * secondOrder / 88.0
                    - 3.0 * fourthOrder / 22.0
                    - 9.0 * secondOrder * thirdOrder / 52.0
                    + 3.0 * fifthOrder / 26.0
                    - secondOrder * secondOrder * secondOrder / 16.0
                    + 3.0 * thirdOrder * thirdOrder / 40.0
                    + 3.0 * secondOrder * fourthOrder / 20.0
                    + 45.0 * secondOrder * secondOrder * thirdOrder / 272.0
                    - 9.0 * (thirdOrder * fourthOrder + secondOrder * fifthOrder) / 68.0;
                return inversePower * correction + 3.0 * sum;
            }
        }

        throw new IllegalStateException("rd duplication did not converge for x=" + x + ", y=" + y + ", z=" + z);
    }

    /**
     * Returns Carlson's symmetric integral of the third kind {@code R_J(x, y, z, p)}.
     *
     * <p>The real principal value is returned when {@code p < 0}.
     *
     * @param x non-negative argument
     * @param y non-negative argument
     * @param z non-negative argument
     * @param p non-zero argument
     * @return {@code R_J(x, y, z, p)}
     */
    public static double rj(double x, double y, double z, double p) {
        validateFinite(x, "rj: x");
        validateFinite(y, "rj: y");
        validateFinite(z, "rj: z");
        validateFinite(p, "rj: p");
        if (x < 0.0 || y < 0.0 || z < 0.0) {
            throw new IllegalArgumentException("rj: requires x >= 0, y >= 0, z >= 0");
        }
        if (p == 0.0) {
            throw new IllegalArgumentException("rj: p must be non-zero");
        }
        if (x + y == 0.0 || y + z == 0.0 || z + x == 0.0) {
            throw new IllegalArgumentException("rj: at most one of x, y, z may be zero");
        }

        if (p < 0.0) {
            double[] ordered = ascending(x, y, z);
            double lower = ordered[0];
            double middle = ordered[1];
            double upper = ordered[2];
            double reflectedP = -p;
            double transformedP = (upper * (lower + middle + reflectedP) - lower * middle) / (upper + reflectedP);
            double value = (transformedP - upper) * rjPositive(lower, middle, upper, transformedP);
            value -= 3.0 * rf(lower, middle, upper);
            value += 3.0 * Math.sqrt((lower * middle * upper) / (lower * middle + transformedP * reflectedP))
                * rc(lower * middle + transformedP * reflectedP, transformedP * reflectedP);
            return value / (upper + reflectedP);
        }
        return rjPositive(x, y, z, p);
    }

    /**
     * Returns Carlson's symmetric integral {@code R_G(x, y, z)}.
     *
     * @param x non-negative argument
     * @param y non-negative argument
     * @param z non-negative argument
     * @return {@code R_G(x, y, z)}
     */
    public static double rg(double x, double y, double z) {
        validateFinite(x, "rg: x");
        validateFinite(y, "rg: y");
        validateFinite(z, "rg: z");
        if (x < 0.0 || y < 0.0 || z < 0.0) {
            throw new IllegalArgumentException("rg: all arguments must be non-negative");
        }

        double[] ordered = ascending(x, y, z);
        double lowest = ordered[0];
        double middle = ordered[1];
        double highest = ordered[2];
        x = highest;
        y = lowest;
        z = middle;

        if (x == z) {
            if (y == z) {
                return Math.sqrt(x);
            }
            if (y == 0.0) {
                return 0.25 * Math.PI * Math.sqrt(x);
            }
            double swapped = x;
            x = y;
            y = swapped;
            return x == 0.0 ? 0.5 * Math.sqrt(z) : 0.5 * (z * rc(x, z) + Math.sqrt(x));
        }
        if (y == z) {
            return y == 0.0 ? 0.5 * Math.sqrt(x) : 0.5 * (y * rc(x, y) + Math.sqrt(x));
        }
        if (y == 0.0) {
            double swapped = z;
            z = y;
            y = swapped;
            return rgWithZeroArgument(x, y);
        }
        return 0.5 * (z * rf(x, y, z) - (x - z) * (y - z) * rd(x, y, z) / 3.0 + Math.sqrt(x * y / z));
    }

    private static double rfWithOneZero(double x, double y) {
        double arithmetic = Math.sqrt(x);
        double geometric = Math.sqrt(y);

        while (!agmConverged(arithmetic, geometric)) {
            double nextGeometric = Math.sqrt(arithmetic * geometric);
            arithmetic = 0.5 * (arithmetic + geometric);
            geometric = nextGeometric;
        }
        return Math.PI / (arithmetic + geometric);
    }

    private static double rdWithZeroX(double y, double z) {
        double arithmetic = Math.sqrt(y);
        double geometric = Math.sqrt(z);
        double initialArithmetic = arithmetic;
        double initialGeometric = geometric;
        double sum = 0.0;
        double weight = 0.25;

        while (!agmConverged(arithmetic, geometric)) {
            double nextGeometric = Math.sqrt(arithmetic * geometric);
            arithmetic = 0.5 * (arithmetic + geometric);
            geometric = nextGeometric;
            weight *= 2.0;
            double difference = arithmetic - geometric;
            sum += weight * difference * difference;
        }

        double reducedRf = Math.PI / (arithmetic + geometric);
        double term = (initialArithmetic + 3.0 * initialGeometric) / (4.0 * z * (initialArithmetic + initialGeometric));
        term -= sum / (z * (y - z));
        return 3.0 * term * reducedRf;
    }

    private static double rjPositive(double x, double y, double z, double p) {
        if (x == y) {
            if (x == z) {
                return x == p ? 1.0 / (x * Math.sqrt(x)) : 3.0 * (rc(x, p) - 1.0 / Math.sqrt(x)) / (x - p);
            }
            double swapped = z;
            z = x;
            x = swapped;
            if (y == p) {
                return rd(x, y, y);
            }
            if (max2(y, p) / min2(y, p) > 1.2) {
                return 3.0 * (rc(x, y) - rc(x, p)) / (p - y);
            }
        }
        if (y == z) {
            if (y == p) {
                return rd(x, y, y);
            }
            if (max2(y, p) / min2(y, p) > 1.2) {
                return 3.0 * (rc(x, y) - rc(x, p)) / (p - y);
            }
        }
        if (z == p) {
            return rd(x, y, z);
        }

        double mean = (x + y + z + 2.0 * p) / 5.0;
        double initialMean = mean;
        double currentX = x;
        double currentY = y;
        double currentZ = z;
        double currentP = p;
        double delta = (p - x) * (p - y) * (p - z);
        double threshold = Math.pow(MACHINE_EPSILON / 5.0, -0.125)
            * max4(Math.abs(mean - x), Math.abs(mean - y), Math.abs(mean - z), Math.abs(mean - p));
        double scale = 1.0;
        double sum = 0.0;

        for (int iteration = 0; iteration < MAX_DUPLICATION_ITERATIONS; iteration++) {
            double rootX = Math.sqrt(currentX);
            double rootY = Math.sqrt(currentY);
            double rootZ = Math.sqrt(currentZ);
            double rootP = Math.sqrt(currentP);
            double denominator = (rootP + rootX) * (rootP + rootY) * (rootP + rootZ);
            double epsilon = delta / (denominator * denominator);
            if (epsilon < -0.5 && epsilon > -1.5) {
                double directArgument = 2.0 * rootP * (currentP + rootX * (rootY + rootZ) + rootY * rootZ) / denominator;
                sum += scale / denominator * rc(1.0, directArgument);
            } else {
                sum += scale / denominator * rcOnePlus(epsilon);
            }

            double lambda = rootX * rootY + rootX * rootZ + rootY * rootZ;
            mean = 0.25 * (mean + lambda);
            scale *= 0.25;
            if (scale * threshold < mean) {
                double xDeviation = scale * (initialMean - x) / mean;
                double yDeviation = scale * (initialMean - y) / mean;
                double zDeviation = scale * (initialMean - z) / mean;
                double pDeviation = -0.5 * (xDeviation + yDeviation + zDeviation);
                double secondOrder = xDeviation * yDeviation + xDeviation * zDeviation + yDeviation * zDeviation
                    - 3.0 * pDeviation * pDeviation;
                double thirdOrder = xDeviation * yDeviation * zDeviation
                    + 2.0 * secondOrder * pDeviation
                    + 4.0 * pDeviation * pDeviation * pDeviation;
                double fourthOrder = (2.0 * xDeviation * yDeviation * zDeviation
                    + secondOrder * pDeviation
                    + 3.0 * pDeviation * pDeviation * pDeviation) * pDeviation;
                double fifthOrder = xDeviation * yDeviation * zDeviation * pDeviation * pDeviation;
                double inversePower = scale / (mean * Math.sqrt(mean));
                double correction = 1.0
                    - 3.0 * secondOrder / 14.0
                    + thirdOrder / 6.0
                    + 9.0 * secondOrder * secondOrder / 88.0
                    - 3.0 * fourthOrder / 22.0
                    - 9.0 * secondOrder * thirdOrder / 52.0
                    + 3.0 * fifthOrder / 26.0
                    - secondOrder * secondOrder * secondOrder / 16.0
                    + 3.0 * thirdOrder * thirdOrder / 40.0
                    + 3.0 * secondOrder * fourthOrder / 20.0
                    + 45.0 * secondOrder * secondOrder * thirdOrder / 272.0
                    - 9.0 * (thirdOrder * fourthOrder + secondOrder * fifthOrder) / 68.0;
                return inversePower * correction + 6.0 * sum;
            }

            currentX = 0.25 * (currentX + lambda);
            currentY = 0.25 * (currentY + lambda);
            currentZ = 0.25 * (currentZ + lambda);
            currentP = 0.25 * (currentP + lambda);
            delta /= 64.0;
        }

        throw new IllegalStateException("rj duplication did not converge for x=" + x + ", y=" + y + ", z=" + z + ", p=" + p);
    }

    private static double rcOnePlus(double y) {
        if (y < -1.0) {
            return Math.sqrt(1.0 / -y) * rc(-y, -1.0 - y);
        }
        if (y == 0.0) {
            return 1.0;
        }
        if (y > 0.0) {
            return Math.atan(Math.sqrt(y)) / Math.sqrt(y);
        }
        if (y > -0.5) {
            double argument = Math.sqrt(-y);
            return InverseHyperbolic.atanh(argument) / Math.sqrt(-y);
        }
        return Math.log((1.0 + Math.sqrt(-y)) / Math.sqrt(1.0 + y)) / Math.sqrt(-y);
    }

    private static double rgWithZeroArgument(double x, double y) {
        double arithmetic = Math.sqrt(x);
        double geometric = Math.sqrt(y);
        double initialArithmetic = arithmetic;
        double initialGeometric = geometric;
        double sum = 0.0;
        double weight = 0.25;

        while (!agmConverged(arithmetic, geometric)) {
            double nextGeometric = Math.sqrt(arithmetic * geometric);
            arithmetic = 0.5 * (arithmetic + geometric);
            geometric = nextGeometric;
            weight *= 2.0;
            double difference = arithmetic - geometric;
            sum += weight * difference * difference;
        }

        double reducedRf = Math.PI / (arithmetic + geometric);
        double midpoint = 0.5 * (initialArithmetic + initialGeometric);
        return 0.5 * ((midpoint * midpoint - sum) * reducedRf);
    }

    private static boolean agmConverged(double arithmetic, double geometric) {
        return Math.abs(arithmetic - geometric) < AGM_CONVERGENCE_FACTOR * ROOT_EPSILON * Math.abs(arithmetic);
    }

    private static double min2(double first, double second) {
        return Math.min(first, second);
    }

    private static double max2(double first, double second) {
        return Math.max(first, second);
    }

    private static double max3(double first, double second, double third) {
        return Math.max(first, Math.max(second, third));
    }

    private static double max4(double first, double second, double third, double fourth) {
        return Math.max(max3(first, second, third), fourth);
    }

    private static double[] ascending(double x, double y, double z) {
        double[] values = {x, y, z};
        Arrays.sort(values);
        return values;
    }

}