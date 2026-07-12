/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.optim.de;

import java.util.Arrays;

/**
 * Reusable storage for the differential evolution optimizer.
 *
 * <p>The workspace owns all mutable arrays used by the solver. Population-like arrays use
 * flat row-major layout: member {@code i}, coordinate {@code j} is at
 * {@code array[i * dimension + j]}. Hook/evaluator APIs may expose some of these arrays
 * directly, so callers that mutate them are mutating live solver state.</p>
 */
public final class DEWorkspace {

    /** Number of decision variables. */
    int dimension;

    /** Number of population members currently allocated. */
    int populationSize;

    /** Number of scalar constraint components in constrained solves. */
    int constraintCount;

    /** Current population, flat row-major layout {@code populationSize x dimension}. */
    double[] population;

    /** Objective value for each current population member; member 0 is kept as the best. */
    double[] populationEnergies;

    /** Whether each current population member satisfies all DE constraints. */
    boolean[] feasible;

    /** Scalar total constraint violation for each current population member; 0 means feasible. */
    double[] constraintViolations;

    /** Component-wise constraint violations for the current population, flat row-major layout. */
    double[] constraintViolationComponents;

    /** Candidate trial population used by deferred/batch evaluation, flat row-major layout. */
    double[] trialPopulation;

    /** Objective values for {@link #trialPopulation}; only evaluated trial slots are meaningful. */
    double[] trialEnergies;

    /** Scalar total constraint violation for each trial member in deferred mode. */
    double[] trialConstraintViolations;

    /** Component-wise constraint violations for each trial member in deferred mode. */
    double[] trialConstraintViolationComponents;

    /** Whether each deferred constrained trial is eligible for selection. */
    boolean[] trialEligible;

    /** Compacted feasible deferred trials evaluated through the configured evaluator. */
    double[] feasibleTrialPopulation;

    /** Objective values for {@link #feasibleTrialPopulation}. */
    double[] feasibleTrialEnergies;

    /** Original trial indices for compacted feasible deferred trials. */
    int[] feasibleTrialIndices;

    /** Single candidate component-wise violation scratch used in immediate constrained mode. */
    double[] constraintViolationScratch;

    /**
     * Per-candidate scratch arrays passed to {@link DEEvaluator}.
     * Row 0 aliases {@link #trial} to save one {@code double[dimension]} allocation.
     */
    double[][] evaluationScratch;

    /** Single candidate scratch used for immediate-mode trial construction and serial evaluation. */
    double[] trial;

    /** Mutant vector produced by the built-in mutation strategies before crossover. */
    double[] mutant;

    /** Copy of the current best solution, refreshed before hooks and final result creation. */
    double[] best;

    /** Selected distinct sample indices for the current mutation; length 5 covers rand2 strategies. */
    int[] samples;

    /**
     * Ensures that all buffers required by the requested shape and mode are allocated.
     *
     * <p>Immediate unconstrained solves allocate only the main population, energies, trial,
     * mutant, best, and sample buffers. Constrained solves add population-level violation
     * metadata; deferred solves add trial-population staging; constrained deferred solves also
     * add compacted feasible-trial buffers for batch objective evaluation.</p>
     */
    public void ensure(int dimension, int populationSize, int constraintCount, boolean deferred) {
        if (dimension <= 0) throw new IllegalArgumentException("dimension must be positive");
        if (populationSize <= 1) throw new IllegalArgumentException("populationSize must be > 1");
        if (this.dimension != dimension || this.populationSize != populationSize || population == null) {
            this.dimension = dimension;
            this.populationSize = populationSize;
            this.population = new double[dimension * populationSize];
            this.populationEnergies = new double[populationSize];
            this.trial = new double[dimension];
            this.mutant = new double[dimension];
            this.best = new double[dimension];
            this.samples = new int[5];
        }
        boolean constrained = constraintCount > 0;
        this.constraintCount = constrained ? constraintCount : 0;
        if (constrained) {
            if (feasible == null || feasible.length < populationSize) feasible = new boolean[populationSize];
            if (constraintViolations == null || constraintViolations.length < populationSize) constraintViolations = new double[populationSize];
            int componentStorage = populationSize * constraintCount;
            if (constraintViolationComponents == null || constraintViolationComponents.length < componentStorage) {
                constraintViolationComponents = new double[componentStorage];
            }
            if (constraintViolationScratch == null || constraintViolationScratch.length < constraintCount) {
                constraintViolationScratch = new double[constraintCount];
            }
        } else {
            feasible = null;
            constraintViolations = null;
            constraintViolationComponents = null;
            constraintViolationScratch = null;
        }

        if (deferred) {
            int populationStorage = dimension * populationSize;
            if (trialPopulation == null || trialPopulation.length < populationStorage) trialPopulation = new double[populationStorage];
            if (trialEnergies == null || trialEnergies.length < populationSize) trialEnergies = new double[populationSize];
        } else {
            trialPopulation = null;
            trialEnergies = null;
            evaluationScratch = null;
        }

        if (constrained && deferred) {
            if (trialConstraintViolations == null || trialConstraintViolations.length < populationSize) {
                trialConstraintViolations = new double[populationSize];
            }
            int componentStorage = populationSize * constraintCount;
            if (trialConstraintViolationComponents == null || trialConstraintViolationComponents.length < componentStorage) {
                trialConstraintViolationComponents = new double[componentStorage];
            }
            if (trialEligible == null || trialEligible.length < populationSize) trialEligible = new boolean[populationSize];
            int populationStorage = dimension * populationSize;
            if (feasibleTrialPopulation == null || feasibleTrialPopulation.length < populationStorage) {
                feasibleTrialPopulation = new double[populationStorage];
            }
            if (feasibleTrialEnergies == null || feasibleTrialEnergies.length < populationSize) {
                feasibleTrialEnergies = new double[populationSize];
            }
            if (feasibleTrialIndices == null || feasibleTrialIndices.length < populationSize) {
                feasibleTrialIndices = new int[populationSize];
            }
        } else {
            trialConstraintViolations = null;
            trialConstraintViolationComponents = null;
            trialEligible = null;
            feasibleTrialPopulation = null;
            feasibleTrialEnergies = null;
            feasibleTrialIndices = null;
        }
        reset();
    }

    /**
     * Clears numeric state that may be read before it is overwritten.
     * Per-candidate vector buffers are intentionally left untouched and reused.
     */
    public void reset() {
        if (populationEnergies != null) Arrays.fill(populationEnergies, Double.POSITIVE_INFINITY);
        if (feasible != null) Arrays.fill(feasible, true);
        if (constraintViolations != null) Arrays.fill(constraintViolations, 0.0);
        if (constraintViolationComponents != null) Arrays.fill(constraintViolationComponents, 0.0);
        if (trialEnergies != null) Arrays.fill(trialEnergies, Double.POSITIVE_INFINITY);
        if (trialConstraintViolations != null) Arrays.fill(trialConstraintViolations, 0.0);
        if (trialConstraintViolationComponents != null) Arrays.fill(trialConstraintViolationComponents, 0.0);
        if (trialEligible != null) Arrays.fill(trialEligible, false);
        if (feasibleTrialEnergies != null) Arrays.fill(feasibleTrialEnergies, Double.POSITIVE_INFINITY);
    }

    /** Ensures per-row objective scratch arrays for custom or parallel batch evaluators. */
    void ensureEvaluationScratch(int count, int dimension) {
        if (count <= 0) return;
        if (evaluationScratch != null
                && evaluationScratch.length >= count
                && evaluationScratch[0] == trial
                && evaluationScratch[0].length >= dimension) {
            return;
        }
        evaluationScratch = new double[count][];
        evaluationScratch[0] = trial;
        for (int i = 1; i < count; i++) {
            evaluationScratch[i] = new double[dimension];
        }
    }
}