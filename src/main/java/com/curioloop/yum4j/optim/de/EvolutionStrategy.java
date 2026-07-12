/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.de;

/**
 * Built-in mutation and crossover strategies for differential evolution.
 *
 * <p>Each enum constant represents one SciPy-compatible strategy name. The suffix encodes
 * crossover type: {@code BIN} for binomial crossover and {@code EXP} for exponential
 * crossover. The numeric middle part records how many differential pairs are used.</p>
 *
 * <h2>Mutation formulas</h2>
 * <p>Let {@code x0} be the current best member, {@code xi} the candidate being evolved,
 * and {@code xr0..xr4} distinct random population members excluding {@code xi}. The built-in
 * mutant vector {@code b} is computed as:</p>
 * <ul>
 *   <li>{@code best1}: {@code b = x0 + F * (xr0 - xr1)}</li>
 *   <li>{@code rand1}: {@code b = xr0 + F * (xr1 - xr2)}</li>
 *   <li>{@code rand2}: {@code b = xr0 + F * (xr1 + xr2 - xr3 - xr4)}</li>
 *   <li>{@code best2}: {@code b = x0 + F * (xr0 + xr1 - xr2 - xr3)}</li>
 *   <li>{@code randtobest1}: {@code b = xr0 + F * (x0 - xr0 + xr1 - xr2)}</li>
 *   <li>{@code currenttobest1}: {@code b = xi + F * (x0 - xi + xr0 - xr1)}</li>
 * </ul>
 *
 * <p>The core owns sample selection and crossover. The enum owns only the formula that
 * writes the mutant vector for the selected samples.</p>
 */
public enum EvolutionStrategy {

    BEST1BIN(true, 2) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            best1(population, dimension, samples, scale, mutant);
        }
    },
    BEST1EXP(false, 2) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            best1(population, dimension, samples, scale, mutant);
        }
    },
    RAND1BIN(true, 3) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            rand1(population, dimension, samples, scale, mutant);
        }
    },
    RAND1EXP(false, 3) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            rand1(population, dimension, samples, scale, mutant);
        }
    },
    RAND2BIN(true, 5) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            rand2(population, dimension, samples, scale, mutant);
        }
    },
    RAND2EXP(false, 5) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            rand2(population, dimension, samples, scale, mutant);
        }
    },
    BEST2BIN(true, 4) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            best2(population, dimension, samples, scale, mutant);
        }
    },
    BEST2EXP(false, 4) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            best2(population, dimension, samples, scale, mutant);
        }
    },
    RAND_TO_BEST1BIN(true, 3) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            randToBest1(population, dimension, samples, scale, mutant);
        }
    },
    RAND_TO_BEST1EXP(false, 3) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            randToBest1(population, dimension, samples, scale, mutant);
        }
    },
    CURRENT_TO_BEST1BIN(true, 2) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            currentToBest1(population, dimension, candidate, samples, scale, mutant);
        }
    },
    CURRENT_TO_BEST1EXP(false, 2) {
        @Override
        void mutate(double[] population, int dimension, int candidate, int[] samples, double scale, double[] mutant) {
            currentToBest1(population, dimension, candidate, samples, scale, mutant);
        }
    };

    final boolean binomial;
    final int sampleCount;

    EvolutionStrategy(boolean binomial, int sampleCount) {
        this.binomial = binomial;
        this.sampleCount = sampleCount;
    }

    /** Returns true for binomial crossover strategies and false for exponential ones. */
    public boolean binomial() {
        return binomial;
    }

    /** Number of distinct population members sampled by this strategy. */
    public int sampleCount() {
        return sampleCount;
    }

    /** Writes this strategy's mutant vector into {@code mutant}. */
    abstract void mutate(double[] population,
                         int dimension,
                         int candidate,
                         int[] samples,
                         double scale,
                         double[] mutant);

    /** {@code best1}: mutate the current best member using one random difference. */
    private static void best1(double[] population, int dimension, int[] samples, double scale, double[] mutant) {
        int sample0Offset = samples[0] * dimension;
        int sample1Offset = samples[1] * dimension;
        for (int coordinate = 0; coordinate < dimension; coordinate++) {
            mutant[coordinate] = population[coordinate]
                    + scale * (population[sample0Offset + coordinate] - population[sample1Offset + coordinate]);
        }
    }

    /** {@code rand1}: mutate a random base member using one random difference. */
    private static void rand1(double[] population, int dimension, int[] samples, double scale, double[] mutant) {
        int sample0Offset = samples[0] * dimension;
        int sample1Offset = samples[1] * dimension;
        int sample2Offset = samples[2] * dimension;
        for (int coordinate = 0; coordinate < dimension; coordinate++) {
            mutant[coordinate] = population[sample0Offset + coordinate]
                    + scale * (population[sample1Offset + coordinate] - population[sample2Offset + coordinate]);
        }
    }

    /** {@code rand2}: mutate a random base member using two random differences. */
    private static void rand2(double[] population, int dimension, int[] samples, double scale, double[] mutant) {
        int sample0Offset = samples[0] * dimension;
        int sample1Offset = samples[1] * dimension;
        int sample2Offset = samples[2] * dimension;
        int sample3Offset = samples[3] * dimension;
        int sample4Offset = samples[4] * dimension;
        for (int coordinate = 0; coordinate < dimension; coordinate++) {
            mutant[coordinate] = population[sample0Offset + coordinate]
                    + scale * (population[sample1Offset + coordinate] + population[sample2Offset + coordinate]
                    - population[sample3Offset + coordinate] - population[sample4Offset + coordinate]);
        }
    }

    /** {@code best2}: mutate the current best member using two random differences. */
    private static void best2(double[] population, int dimension, int[] samples, double scale, double[] mutant) {
        int sample0Offset = samples[0] * dimension;
        int sample1Offset = samples[1] * dimension;
        int sample2Offset = samples[2] * dimension;
        int sample3Offset = samples[3] * dimension;
        for (int coordinate = 0; coordinate < dimension; coordinate++) {
            mutant[coordinate] = population[coordinate]
                    + scale * (population[sample0Offset + coordinate] + population[sample1Offset + coordinate]
                    - population[sample2Offset + coordinate] - population[sample3Offset + coordinate]);
        }
    }

    /** {@code randtobest1}: move a random base toward the best while adding one difference. */
    private static void randToBest1(double[] population, int dimension, int[] samples, double scale, double[] mutant) {
        int sample0Offset = samples[0] * dimension;
        int sample1Offset = samples[1] * dimension;
        int sample2Offset = samples[2] * dimension;
        for (int coordinate = 0; coordinate < dimension; coordinate++) {
            mutant[coordinate] = population[sample0Offset + coordinate]
                    + scale * (population[coordinate] - population[sample0Offset + coordinate]
                    + population[sample1Offset + coordinate] - population[sample2Offset + coordinate]);
        }
    }

    /** {@code currenttobest1}: move the current candidate toward the best plus one difference. */
    private static void currentToBest1(double[] population,
                                       int dimension,
                                       int candidate,
                                       int[] samples,
                                       double scale,
                                       double[] mutant) {
        int candidateOffset = candidate * dimension;
        int sample0Offset = samples[0] * dimension;
        int sample1Offset = samples[1] * dimension;
        for (int coordinate = 0; coordinate < dimension; coordinate++) {
            mutant[coordinate] = population[candidateOffset + coordinate]
                    + scale * (population[coordinate] - population[candidateOffset + coordinate]
                    + population[sample0Offset + coordinate] - population[sample1Offset + coordinate]);
        }
    }

}