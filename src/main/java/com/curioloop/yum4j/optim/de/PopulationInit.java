/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.de;

import com.curioloop.yum4j.optim.Bound;

import java.util.random.RandomGenerator;

/**
 * Initial population construction methods for differential evolution.
 *
 * <p>All methods write a flat row-major population and immediately apply the same projection
 * used for trial repair: finite bounds are respected and integral coordinates are rounded to
 * available integer values. The quasi-random methods intentionally cover only small fixed
 * dimensions because their direction/permutation tables are embedded in this enum.</p>
 */
public enum PopulationInit {
    /** Uniform independent samples within each bound. */
    RANDOM {
        @Override
        void initialize(double[] population,
                        int populationSize,
                        int dimension,
                        Bound[] bounds,
                        DEProblem config,
                        RandomGenerator rng) {
            for (int member = 0; member < populationSize; member++) {
                int offset = member * dimension;
                for (int coordinate = 0; coordinate < dimension; coordinate++) {
                    double value = uniform(bounds[coordinate].lower(), bounds[coordinate].upper(), rng);
                    population[offset + coordinate] = DECore.projectValue(value, coordinate, bounds, config);
                }
            }
        }
    },
    /** Latin hypercube samples, shuffled independently per coordinate. */
    LATIN_HYPERCUBE {
        @Override
        void initialize(double[] population,
                        int populationSize,
                        int dimension,
                        Bound[] bounds,
                        DEProblem config,
                        RandomGenerator rng) {
            for (int coordinate = 0; coordinate < dimension; coordinate++) {
                double lower = bounds[coordinate].lower();
                double range = bounds[coordinate].upper() - lower;
                for (int member = 0; member < populationSize; member++) {
                    int offset = member * dimension + coordinate;
                    double value = lower + range * ((member + rng.nextDouble()) / populationSize);
                    population[offset] = DECore.projectValue(value, coordinate, bounds, config);
                }
                for (int member = populationSize - 1; member > 0; member--) {
                    int other = rng.nextInt(member + 1);
                    int currentOffset = member * dimension + coordinate;
                    int otherOffset = other * dimension + coordinate;
                    double tmp = population[currentOffset];
                    population[currentOffset] = population[otherOffset];
                    population[otherOffset] = tmp;
                }
            }
        }
    },
    /** Sobol quasi-random samples; population size is rounded up to a power of two. */
    SOBOL {
        @Override
        int populationSize(int requestedSize) {
            return nextPowerOfTwo(requestedSize);
        }

        @Override
        void validateDimension(int dimension) {
            if (dimension > Sobol.MAX_DIMENSION) {
                throw new IllegalArgumentException("Sobol initialization supports at most "
                        + Sobol.MAX_DIMENSION + " dimensions, got " + dimension);
            }
        }

        @Override
        void initialize(double[] population,
                        int populationSize,
                        int dimension,
                        Bound[] bounds,
                        DEProblem config,
                        RandomGenerator rng) {
            Sobol.fill(population, populationSize, dimension, bounds, config);
        }
    },
    /** Halton quasi-random samples. */
    HALTON {
        @Override
        void validateDimension(int dimension) {
            if (dimension > Halton.MAX_DIMENSION) {
                throw new IllegalArgumentException("Halton initialization supports at most "
                        + Halton.MAX_DIMENSION + " dimensions, got " + dimension);
            }
        }

        @Override
        void initialize(double[] population,
                        int populationSize,
                        int dimension,
                        Bound[] bounds,
                        DEProblem config,
                        RandomGenerator rng) {
            Halton.fill(population, populationSize, dimension, bounds, config);
        }
    };

    int populationSize(int requestedSize) {
        return requestedSize;
    }

    void validateDimension(int dimension) {}

    abstract void initialize(double[] population,
                             int populationSize,
                             int dimension,
                             Bound[] bounds,
                             DEProblem config,
                             RandomGenerator rng);

    private static int nextPowerOfTwo(int value) {
        int power = 1;
        while (power < value) power <<= 1;
        return power;
    }

    private static double uniform(double lower, double upper, RandomGenerator rng) {
        return lower + rng.nextDouble() * (upper - lower);
    }

    private static final class Sobol {

        private static final int MAX_DIMENSION = 10;
        private static final double EPS = 1.0e-10;
        private static final int MAX_BITS = 30;

        private static final int[][] DIRECTION_NUMBERS = {
                {1, 0, 1},
                {2, 1, 1, 1},
                {3, 1, 1, 3, 1},
                {3, 2, 1, 1, 1},
                {4, 1, 1, 1, 3, 3},
                {4, 4, 1, 3, 5, 13},
                {5, 2, 1, 1, 5, 5, 17},
                {5, 4, 1, 1, 5, 5, 5},
                {5, 7, 1, 1, 7, 11, 19},
        };

            private static final long[][] DIRECTIONS = computeDirectionNumbers();

        private Sobol() {}

        private static void fill(double[] population,
                                 int populationSize,
                                 int dimension,
                                 Bound[] bounds,
                                 DEProblem config) {
            if (populationSize <= 0) return;

            long[] point = new long[dimension];
            double scale = 1.0 / (1L << MAX_BITS);

            for (int coordinate = 0; coordinate < dimension; coordinate++) {
                write(population, 0, coordinate, dimension, clamp(0.0), bounds, config);
            }

            for (int member = 1; member < populationSize; member++) {
                int bit = rightmostZeroBit(member - 1);
                for (int coordinate = 0; coordinate < dimension; coordinate++) {
                    point[coordinate] ^= DIRECTIONS[coordinate][bit];
                    write(population, member, coordinate, dimension,
                            clamp(point[coordinate] * scale), bounds, config);
                }
            }
        }

        private static long[][] computeDirectionNumbers() {
            long[][] directions = new long[MAX_DIMENSION][MAX_BITS];
            for (int bit = 0; bit < MAX_BITS; bit++) {
                directions[0][bit] = 1L << (MAX_BITS - 1 - bit);
            }

            for (int dim = 1; dim < MAX_DIMENSION; dim++) {
                int[] params = DIRECTION_NUMBERS[dim - 1];
                int degree = params[0];
                int coefficient = params[1];

                for (int bit = 0; bit < degree && bit < MAX_BITS; bit++) {
                    directions[dim][bit] = ((long) params[2 + bit]) << (MAX_BITS - 1 - bit);
                }

                for (int bit = degree; bit < MAX_BITS; bit++) {
                    long value = directions[dim][bit - degree] ^ (directions[dim][bit - degree] >> degree);
                    for (int k = 1; k < degree; k++) {
                        value ^= ((long) ((coefficient >> (degree - 1 - k)) & 1)) * directions[dim][bit - k];
                    }
                    directions[dim][bit] = value;
                }
            }
            return directions;
        }

        private static int rightmostZeroBit(int value) {
            int bit = 0;
            int current = value;
            while ((current & 1) == 1) {
                current >>= 1;
                bit++;
            }
            return bit;
        }
    }

    private static final class Halton {

        private static final int MAX_DIMENSION = 20;
        private static final double EPS = 1.0e-10;

        private static final int[] PRIMES = {
                2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                31, 37, 41, 43, 47, 53, 59, 61, 67, 71
        };

        private static final int[][] PERMUTATIONS = computePermutations();

        private Halton() {}

        private static void fill(double[] population,
                                 int populationSize,
                                 int dimension,
                                 Bound[] bounds,
                                 DEProblem config) {
            for (int coordinate = 0; coordinate < dimension; coordinate++) {
                int base = PRIMES[coordinate];
                int[] permutation = PERMUTATIONS[coordinate];
                for (int member = 0; member < populationSize; member++) {
                    write(population, member, coordinate, dimension,
                            clamp(vanDerCorput(member + 1, base, permutation)), bounds, config);
                }
            }
        }

        private static double vanDerCorput(int index, int base, int[] permutation) {
            double result = 0.0;
            double denominator = 1.0;
            int value = index;

            while (value > 0) {
                denominator *= base;
                int digit = value % base;
                if (permutation != null) digit = permutation[digit];
                result += digit / denominator;
                value /= base;
            }
            return result;
        }

        private static int[][] computePermutations() {
            int[][] permutations = new int[MAX_DIMENSION][];
            for (int coordinate = 5; coordinate < MAX_DIMENSION; coordinate++) {
                permutations[coordinate] = faurePermutation(PRIMES[coordinate]);
            }
            return permutations;
        }

        private static int[] faurePermutation(int base) {
            int[] permutation = new int[base];
            if (base <= 2) {
                for (int i = 0; i < base; i++) permutation[i] = i;
                return permutation;
            }
            permutation[0] = 0;
            if (base == 3) {
                permutation[1] = 2;
                permutation[2] = 1;
            } else {
                for (int i = 0; i < base; i++) {
                    permutation[i] = base - 1 - i;
                }
                permutation[0] = 0;
            }
            return permutation;
        }
    }

    private static void write(double[] population,
                              int member,
                              int coordinate,
                              int dimension,
                              double unit,
                              Bound[] bounds,
                              DEProblem config) {
        Bound bound = bounds[coordinate];
        double value = bound.lower() + (bound.upper() - bound.lower()) * unit;
        population[member * dimension + coordinate] = DECore.projectValue(value, coordinate, bounds, config);
    }

    private static double clamp(double value) {
        if (value < Sobol.EPS) return Sobol.EPS;
        if (value > 1.0 - Sobol.EPS) return 1.0 - Sobol.EPS;
        return value;
    }
}